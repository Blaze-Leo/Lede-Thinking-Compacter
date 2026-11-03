import { createHash, randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import {
  mkdir,
  readFile,
  rename,
  unlink,
  writeFile,
} from "node:fs/promises";
import { join } from "node:path";
import { spawn } from "node:child_process";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Intended location:
//   ~/.pi/agent/extensions/thinking-lede.ts
//
// Deterministic compressor:
//   ~/.pi/lede-isolate/plain_compress.py
//
// This extension never invokes llm_compress.py, pi -p, or an LLM.

// ============================================================
// TYPES
// ============================================================

interface Settings {
  pythonExecutable: string;
  maxThinkingTokens: number;
  charsPerToken: number;
  target: number;
  dedup: number;
  headFrac: number;
  tailFrac: number;
  timeoutMs: number;
}

type CompressionResult =
  | { status: "compressed"; text: string }
  | { status: "unchanged" }
  | { status: "failed"; error?: string };

interface CacheFile {
  version: 1;
  sessionId: string;
  forceCompactAll: boolean;

  // SHA-256(original thinking text) -> compressed text.
  summaries: Record<string, string>;

  // Fingerprints of blocks excluded by /remove-all-think.
  removedThinking: string[];
}

interface SessionCache extends CacheFile {
  filePath: string;
  writeQueue: Promise<void>;
}

// ============================================================
// PATHS AND DEFAULT SETTINGS
// ============================================================

const AGENT_DIR = join(homedir(), ".pi", "agent");

const SETTINGS_PATH = join(
  AGENT_DIR,
  "thinking-lede.json",
);

const CACHE_DIR = join(
  AGENT_DIR,
  "thinking-lede",
);

// Deliberately fixed to the deterministic implementation.
// The settings file cannot redirect this extension to llm_compress.py.
const COMPRESSOR_SCRIPT = join(
  homedir(),
  ".pi",
  "lede-isolate",
  "plain_compress.py",
);

const DEFAULT_SETTINGS: Settings = {
  pythonExecutable: "python3",

  // Automatic compression threshold.
  maxThinkingTokens: 500,
  charsPerToken: 4,

  // plain_compress.py arguments.
  target: 0.05,
  dedup: 0.85,
  headFrac: 0.15,
  tailFrac: 0.25,

  // Maximum time allowed for one compression operation.
  timeoutMs: 120000,
};

const HEADING = "[Compressed Thinking]";

// ============================================================
// SETTINGS
// ============================================================

function isRecord(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function ensureSettingsFile(): Record<string, unknown> {
  mkdirSync(AGENT_DIR, {
    recursive: true,
    mode: 0o700,
  });

  try {
    const parsed: unknown = JSON.parse(
      readFileSync(SETTINGS_PATH, "utf8"),
    );

    if (!isRecord(parsed)) {
      throw new Error(
        `Settings file must contain a JSON object: ${SETTINGS_PATH}`,
      );
    }

    return parsed;
  } catch (error) {
    if (getErrorCode(error) !== "ENOENT") {
      throw error;
    }
  }

  // Create the default settings file on first load.
  // wx prevents overwriting a file created concurrently.
  try {
    writeFileSync(
      SETTINGS_PATH,
      `${JSON.stringify(DEFAULT_SETTINGS, null, 2)}\n`,
      {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600,
      },
    );
  } catch (error) {
    if (getErrorCode(error) !== "EEXIST") {
      throw error;
    }
  }

  const parsed: unknown = JSON.parse(
    readFileSync(SETTINGS_PATH, "utf8"),
  );

  if (!isRecord(parsed)) {
    throw new Error(
      `Settings file must contain a JSON object: ${SETTINGS_PATH}`,
    );
  }

  return parsed;
}

function loadSettings(): Settings {
  const parsed = ensureSettingsFile();

  const settings: Settings = {
    pythonExecutable:
      typeof parsed.pythonExecutable === "string" &&
      parsed.pythonExecutable.trim()
        ? parsed.pythonExecutable.trim()
        : DEFAULT_SETTINGS.pythonExecutable,

    maxThinkingTokens:
      typeof parsed.maxThinkingTokens === "number"
        ? parsed.maxThinkingTokens
        : DEFAULT_SETTINGS.maxThinkingTokens,

    charsPerToken:
      typeof parsed.charsPerToken === "number"
        ? parsed.charsPerToken
        : DEFAULT_SETTINGS.charsPerToken,

    target:
      typeof parsed.target === "number"
        ? parsed.target
        : DEFAULT_SETTINGS.target,

    dedup:
      typeof parsed.dedup === "number"
        ? parsed.dedup
        : DEFAULT_SETTINGS.dedup,

    headFrac:
      typeof parsed.headFrac === "number"
        ? parsed.headFrac
        : DEFAULT_SETTINGS.headFrac,

    tailFrac:
      typeof parsed.tailFrac === "number"
        ? parsed.tailFrac
        : DEFAULT_SETTINGS.tailFrac,

    timeoutMs:
      typeof parsed.timeoutMs === "number"
        ? parsed.timeoutMs
        : DEFAULT_SETTINGS.timeoutMs,
  };

  if (
    !Number.isFinite(settings.maxThinkingTokens) ||
    settings.maxThinkingTokens < 1
  ) {
    throw new Error(
      "maxThinkingTokens must be a positive number.",
    );
  }

  if (
    !Number.isFinite(settings.charsPerToken) ||
    settings.charsPerToken <= 0
  ) {
    throw new Error(
      "charsPerToken must be a positive number.",
    );
  }

  if (
    !Number.isFinite(settings.target) ||
    settings.target <= 0 ||
    settings.target > 1
  ) {
    throw new Error(
      "target must be greater than 0 and at most 1.",
    );
  }

  if (
    !Number.isFinite(settings.dedup) ||
    settings.dedup < 0 ||
    settings.dedup > 1
  ) {
    throw new Error(
      "dedup must be between 0 and 1.",
    );
  }

  if (
    !Number.isFinite(settings.headFrac) ||
    settings.headFrac < 0 ||
    settings.headFrac > 1
  ) {
    throw new Error(
      "headFrac must be between 0 and 1.",
    );
  }

  if (
    !Number.isFinite(settings.tailFrac) ||
    settings.tailFrac < 0 ||
    settings.tailFrac > 1
  ) {
    throw new Error(
      "tailFrac must be between 0 and 1.",
    );
  }

  if (settings.headFrac + settings.tailFrac > 1) {
    throw new Error(
      "headFrac + tailFrac must not exceed 1.",
    );
  }

  if (
    !Number.isInteger(settings.timeoutMs) ||
    settings.timeoutMs < 1
  ) {
    throw new Error(
      "timeoutMs must be a positive integer.",
    );
  }

  return settings;
}

// ============================================================
// UTILITIES
// ============================================================

function getErrorCode(
  error: unknown,
): string | undefined {
  if (
    error &&
    typeof error === "object" &&
    "code" in error &&
    typeof error.code === "string"
  ) {
    return error.code;
  }

  return undefined;
}

function countChars(value: string): number {
  return Array.from(value).length;
}

function estimateTokens(
  value: string,
  settings: Settings,
): number {
  return Math.ceil(
    countChars(value) / settings.charsPerToken,
  );
}

function hashThinking(value: string): string {
  return createHash("sha256")
    .update(value, "utf8")
    .digest("hex");
}

function safeSessionId(sessionId: string): string {
  return sessionId.replace(
    /[^a-zA-Z0-9_-]/g,
    "_",
  );
}

/**
 * Stable JSON serialization for ordinary Pi message data.
 */
function stableSerialize(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? String(value);
  }

  if (Array.isArray(value)) {
    return `[${value
      .map((item) => stableSerialize(item))
      .join(",")}]`;
  }

  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();

  return `{${keys
    .map(
      (key) =>
        `${JSON.stringify(key)}:${stableSerialize(record[key])}`,
    )
    .join(",")}}`;
}

/**
 * Produce possible keys for one thinking block.
 *
 * The message-ID key is used when Pi provides a stable message ID.
 * The fallback key uses timestamp and non-thinking content.
 *
 * Both are stored so that the context hook can match the block
 * even if one representation does not contain the message ID.
 */
function getThinkingBlockKeys(
  message: any,
  partIndex: number,
  thinking: string,
): string[] {
  const thinkingHash = hashThinking(thinking);
  const keys: string[] = [];

  const messageId =
    typeof message.id === "string" ||
    typeof message.id === "number"
      ? String(message.id)
      : typeof message.messageId === "string"
        ? message.messageId
        : undefined;

  if (messageId) {
    keys.push(
      hashThinking(
        `id:${messageId}:${partIndex}:${thinkingHash}`,
      ),
    );
  }

  // Exclude thinking text from this fallback identity.
  // This allows it to remain stable if the thinking content is
  // replaced with compressed output.
  const nonThinkingContent = Array.isArray(message.content)
    ? message.content.map((part: any, index: number) =>
        part?.type === "thinking"
          ? {
              type: "thinking",
              index,
            }
          : part,
      )
    : message.content;

  const fallbackIdentity = hashThinking(
    stableSerialize({
      timestamp:
        message.timestamp ??
        message.createdAt ??
        null,

      role: message.role ?? null,

      content: nonThinkingContent ?? null,
    }),
  );

  keys.push(
    hashThinking(
      `fallback:${fallbackIdentity}:${partIndex}:${thinkingHash}`,
    ),
  );

  return keys;
}

// ============================================================
// DETERMINISTIC LEDE COMPRESSOR ADAPTER
// ============================================================

/**
 * Run only plain_compress.py.
 *
 * Input is passed through stdin, avoiding shell quoting problems
 * and ARG_MAX limitations with large thinking blocks.
 *
 * A failed, timed-out, empty, or non-shrinking result is rejected.
 * In those cases, original thinking remains available.
 */
function compressThinking(
  originalThinking: string,
  settings: Settings,
): Promise<CompressionResult> {
  return new Promise((resolve) => {
    const args = [
      COMPRESSOR_SCRIPT,
      "--target",
      String(settings.target),
      "--dedup",
      String(settings.dedup),
      "--head-frac",
      String(settings.headFrac),
      "--tail-frac",
      String(settings.tailFrac),
    ];

    let child: ReturnType<typeof spawn>;

    try {
      child = spawn(settings.pythonExecutable, args, {
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
      });
    } catch (error) {
      resolve({
        status: "failed",
        error: error instanceof Error ? error.message : String(error),
      });
      return;
    }

    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (result: CompressionResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish({
        status: "failed",
        error: `Compressor timed out after ${settings.timeoutMs} ms.`,
      });
    }, settings.timeoutMs);

    if (!child.stdin || !child.stdout || !child.stderr) {
      child.kill("SIGKILL");
      finish({
        status: "failed",
        error: "Could not initialize compressor streams.",
      });
      return;
    }

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");

    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });

    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      if (stderr.length > 8000) {
        stderr = stderr.slice(-8000);
      }
    });

    child.stdin.on("error", () => {
      // The child's exit status determines whether the operation failed.
    });

    child.on("error", (error) => {
      finish({
        status: "failed",
        error: error.message,
      });
    });

    child.on("close", (code, signal) => {
      if (settled) return;

      if (code !== 0) {
        finish({
          status: "failed",
          error: `plain_compress.py exited with code ${String(code)}${
            signal ? ` (signal ${signal})` : ""
          }${stderr.trim() ? `: ${stderr.trim()}` : ""}`,
        });
        return;
      }

      const compressed = stdout.trim();

      // A successful process that produces no usable/smaller output
      // is an unchanged result, not an execution failure.
      if (!compressed) {
        finish({ status: "unchanged" });
        return;
      }

      if (
        countChars(compressed) >=
        countChars(originalThinking.trim())
      ) {
        finish({ status: "unchanged" });
        return;
      }

      finish({
        status: "compressed",
        text: compressed,
      });
    });

    child.stdin.end(originalThinking, "utf8");
  });
}

// ============================================================
// SESSION CACHE
// ============================================================

const sessionCaches = new Map<
  string,
  Promise<SessionCache>
>();

const inFlightCompressions = new Map<
  string,
  Promise<CompressionResult>
>();

async function loadSessionCache(
  sessionId: string,
): Promise<SessionCache> {
  const filePath = join(
    CACHE_DIR,
    `${safeSessionId(sessionId)}.json`,
  );

  const state: SessionCache = {
    version: 1,
    sessionId,
    forceCompactAll: false,
    summaries: {},
    removedThinking: [],
    filePath,
    writeQueue: Promise.resolve(),
  };

  let raw: string;

  try {
    raw = await readFile(filePath, "utf8");
  } catch (error) {
    if (getErrorCode(error) === "ENOENT") {
      return state;
    }

    throw error;
  }

  const parsed: unknown = JSON.parse(raw);

  if (!isRecord(parsed)) {
    throw new Error(
      "Invalid LEDE thinking cache: expected a JSON object.",
    );
  }

  if (
    parsed.version !== 1 ||
    parsed.sessionId !== sessionId ||
    !isRecord(parsed.summaries)
  ) {
    throw new Error(
      "Invalid or incompatible LEDE thinking cache.",
    );
  }

  state.forceCompactAll =
    parsed.forceCompactAll === true;

  for (const [hash, compressed] of Object.entries(
    parsed.summaries,
  )) {
    if (
      /^[a-f0-9]{64}$/.test(hash) &&
      typeof compressed === "string" &&
      compressed.trim()
    ) {
      state.summaries[hash] = compressed;
    }
  }

  if (Array.isArray(parsed.removedThinking)) {
    state.removedThinking = parsed.removedThinking.filter(
      (key): key is string =>
        typeof key === "string" &&
        /^[a-f0-9]{64}$/.test(key),
    );
  }

  return state;
}

async function getSessionCache(
  sessionId: string,
): Promise<SessionCache> {
  const existing = sessionCaches.get(sessionId);

  if (existing) {
    return existing;
  }

  const pending = loadSessionCache(sessionId);

  sessionCaches.set(sessionId, pending);

  try {
    return await pending;
  } catch (error) {
    if (sessionCaches.get(sessionId) === pending) {
      sessionCaches.delete(sessionId);
    }

    throw error;
  }
}

async function saveSessionCache(
  state: SessionCache,
): Promise<void> {
  state.writeQueue = state.writeQueue
    .catch(() => undefined)
    .then(async () => {
      await mkdir(CACHE_DIR, {
        recursive: true,
        mode: 0o700,
      });

      const payload: CacheFile = {
        version: 1,
        sessionId: state.sessionId,
        forceCompactAll: state.forceCompactAll,
        summaries: { ...state.summaries },
        removedThinking: [...state.removedThinking],
      };

      const temporaryPath =
        `${state.filePath}.${process.pid}.${randomUUID()}.tmp`;

      try {
        await writeFile(
          temporaryPath,
          `${JSON.stringify(payload, null, 2)}\n`,
          {
            encoding: "utf8",
            flag: "wx",
            mode: 0o600,
          },
        );

        await rename(
          temporaryPath,
          state.filePath,
        );
      } finally {
        await unlink(temporaryPath).catch(
          () => undefined,
        );
      }
    });

  await state.writeQueue;
}

// ============================================================
// HASH CACHE AND LEDE COMPRESSION
// ============================================================

async function getOrCreateCompression(
  originalThinking: string,
  state: SessionCache,
  settings: Settings,
): Promise<CompressionResult> {
  const hash = hashThinking(originalThinking);

  const cached = state.summaries[hash];
  if (cached) {
    return {
      status: "compressed",
      text: cached,
    };
  }

  const inFlightKey = `${state.sessionId}:${hash}`;
  const existing = inFlightCompressions.get(inFlightKey);
  if (existing) {
    return existing;
  }

  const pending = (async (): Promise<CompressionResult> => {
    const result = await compressThinking(
      originalThinking,
      settings,
    );

    if (result.status !== "compressed") {
      return result;
    }

    state.summaries[hash] = result.text;

    try {
      // Persist before allowing compressed output to replace original thinking.
      await saveSessionCache(state);
    } catch (error) {
      delete state.summaries[hash];
      return {
        status: "failed",
        error: `Could not persist compressed thinking: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
    }

    return result;
  })();

  inFlightCompressions.set(inFlightKey, pending);

  try {
    return await pending;
  } finally {
    if (inFlightCompressions.get(inFlightKey) === pending) {
      inFlightCompressions.delete(inFlightKey);
    }
  }
}

// ============================================================
// COMPACTION PROGRESS UI
// ============================================================

const PROGRESS_WIDGET_KEY =
  "thinking-lede-progress";

function updateCompactionProgress(
  ctx: any,
  completed: number,
  total: number,
  compressed: number,
  unchanged: number,
  failed: number,
): void {
  if (!ctx.hasUI) {
    return;
  }

  const width = 24;

  const percent =
    total > 0
      ? Math.round((completed / total) * 100)
      : 100;

  const filled = Math.round(
    (completed / Math.max(total, 1)) * width,
  );

  const bar =
    "█".repeat(filled) +
    "░".repeat(width - filled);

  const current = Math.min(
    completed + 1,
    total,
  );

  ctx.ui.setWidget(PROGRESS_WIDGET_KEY, [
    "┌─ LEDE Thinking Compression ────────────────────┐",
    `│ [${bar}] ${String(percent).padStart(3)}% (${completed}/${total})`,
    `│ Compressed: ${compressed} | Unchanged: ${unchanged} | Failed: ${failed}`,
    `│ ${
      completed < total
        ? `Currently processing block ${current} of ${total}`
        : "Compression complete"
    }`,
    "└───────────────────────────────────────────────┘",
  ]);
}

function clearCompactionProgress(
  ctx: any,
): void {
  if (ctx.hasUI) {
    ctx.ui.setWidget(
      PROGRESS_WIDGET_KEY,
      undefined,
    );
  }
}

// ============================================================
// THINKING BLOCK COLLECTION AND COMPRESSION
// ============================================================

async function compressCompletedContext(
  messages: any[],
  ctx: any,
  settings: Settings,
): Promise<void> {
  const sessionId = ctx.sessionManager.getSessionId();

  let state: SessionCache;
  try {
    state = await getSessionCache(sessionId);
  } catch {
    // Cache errors must never affect the conversation.
    return;
  }

  const forceCompact = state.forceCompactAll;
  const work = new Map<string, string>();

  for (const message of messages) {
    if (
      message.role !== "assistant" ||
      !Array.isArray(message.content)
    ) {
      continue;
    }

    for (const part of message.content) {
      if (part.type !== "thinking") continue;

      const originalThinking =
        typeof part.thinking === "string" ? part.thinking : "";

      if (part.redacted || !originalThinking.trim()) {
        continue;
      }

      const hash = hashThinking(originalThinking);
      if (state.summaries[hash]) {
        continue;
      }

      const shouldCompress =
        forceCompact ||
        estimateTokens(originalThinking, settings) >
          settings.maxThinkingTokens;

      if (shouldCompress) {
        // Identical thinking text is compressed only once.
        work.set(hash, originalThinking);
      }
    }
  }

  const total = work.size;
  let completed = 0;
  let compressed = 0;
  let unchanged = 0;
  let failed = 0;

  try {
    if (total > 0) {
      updateCompactionProgress(
        ctx,
        completed,
        total,
        compressed,
        unchanged,
        failed,
      );
    }

    for (const originalThinking of work.values()) {
      try {
        const result = await getOrCreateCompression(
          originalThinking,
          state,
          settings,
        );

        switch (result.status) {
          case "compressed":
            compressed++;
            break;
          case "unchanged":
            unchanged++;
            break;
          case "failed":
            failed++;
            console.error(
              `[thinking-lede] Compression failed: ${result.error ?? "Unknown error"}`,
            );
            break;
        }
      } catch (error) {
        failed++;
        console.error(
          `[thinking-lede] Unexpected compression error: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      }

      completed++;
      updateCompactionProgress(
        ctx,
        completed,
        total,
        compressed,
        unchanged,
        failed,
      );
    }

    // Unchanged is a completed outcome, not an execution failure.
    // Keep force mode active only when an operation actually failed.
    if (forceCompact && failed === 0) {
      state.forceCompactAll = false;
      try {
        await saveSessionCache(state);
      } catch {
        state.forceCompactAll = true;
      }
    }
  } finally {
    clearCompactionProgress(ctx);
  }
}

// ============================================================
// CONTEXT PROCESSING
// ============================================================

async function applyContext(
  event: any,
  ctx: any,
): Promise<any> {
  const sessionId =
    ctx.sessionManager.getSessionId();

  let state: SessionCache;

  try {
    state = await getSessionCache(sessionId);
  } catch {
    // Cache errors must never destroy original context.
    return;
  }

  const removedThinking = new Set(
    state.removedThinking,
  );

  let contextChanged = false;

  const messages: typeof event.messages = [];

  for (const message of event.messages) {
    if (
      message.role !== "assistant" ||
      !Array.isArray(message.content)
    ) {
      messages.push(message);
      continue;
    }

    const content: typeof message.content = [];

    for (
      let partIndex = 0;
      partIndex < message.content.length;
      partIndex++
    ) {
      const part = message.content[partIndex];

      if (part.type !== "thinking") {
        // Preserve normal text and tool calls.
        content.push(part);
        continue;
      }

      const originalThinking =
        typeof part.thinking === "string"
          ? part.thinking
          : "";

      if (
        part.redacted ||
        !originalThinking.trim()
      ) {
        content.push(part);
        continue;
      }

      // Apply explicit removal before considering a cached replacement.
      const blockKeys = getThinkingBlockKeys(
        message,
        partIndex,
        originalThinking,
      );

      if (
        blockKeys.some((key) =>
          removedThinking.has(key),
        )
      ) {
        contextChanged = true;
        continue;
      }

      // Cached compressed text takes precedence over original thinking.
      const hash = hashThinking(originalThinking);

      const compressed = state.summaries[hash];

      if (compressed) {
        content.push({
          type: "thinking",
          thinking: `${HEADING}\n${compressed}`,
        });

        contextChanged = true;
        continue;
      }

      // New and unmarked thinking remains included normally.
      content.push(part);
    }

    const unchanged =
      content.length === message.content.length &&
      content.every(
        (part: any, index: number) =>
          part === message.content[index],
      );

    messages.push(
      unchanged
        ? message
        : {
            ...message,
            content,
          },
    );
  }

  if (!contextChanged) {
    return;
  }

  return { messages };
}

// ============================================================
// COMMANDS AND EXTENSION REGISTRATION
// ============================================================

export default function (pi: ExtensionAPI) {
  const settings = loadSettings();

  // Create the storage directory when the extension loads.
  // Per-session cache files are written there when required.
  mkdirSync(CACHE_DIR, {
    recursive: true,
    mode: 0o700,
  });

  async function getCurrentState(
    ctx: any,
  ): Promise<SessionCache> {
    return getSessionCache(
      ctx.sessionManager.getSessionId(),
    );
  }

  // ----------------------------------------------------------
  // /compact-all-thinking
  // ----------------------------------------------------------

  pi.registerCommand("compact-all-thinking", {
    description:
      "Immediately compress all historical thinking blocks with deterministic LEDE, regardless of size.",

    handler: async (_args: string, ctx: any) => {
      await ctx.waitForIdle();

      const state = await getCurrentState(ctx);

      state.forceCompactAll = true;

      await saveSessionCache(state);

      // Read the current session branch and compress immediately.
      const messages = ctx.sessionManager
        .getBranch()
        .filter(
          (entry: any) => entry.type === "message",
        )
        .map(
          (entry: any) => entry.message,
        );

      await compressCompletedContext(
        messages,
        ctx,
        settings,
      );

      ctx.ui.notify(
        state.forceCompactAll
          ? "Some thinking blocks failed to compress. Run the command again to retry. Unchanged blocks were left as-is."
          : "All processable historical thinking blocks have been compressed and cached with LEDE.",
        "info",
      );
    },
  });

  // ----------------------------------------------------------
  // /remove-all-think
  // ----------------------------------------------------------

  // Mark every thinking block currently present in the session branch.
  // Thinking generated after this command remains included.

  pi.registerCommand("remove-all-think", {
    description:
      "Exclude existing thinking blocks from future prompts. New thinking remains included.",

    handler: async (_args: string, ctx: any) => {
      await ctx.waitForIdle();

      const state = await getCurrentState(ctx);

      const messages = ctx.sessionManager
        .getBranch()
        .filter(
          (entry: any) => entry.type === "message",
        )
        .map(
          (entry: any) => entry.message,
        );

      const removed = new Set(
        state.removedThinking,
      );

      let newlyMarkedBlocks = 0;

      for (const message of messages) {
        if (
          message.role !== "assistant" ||
          !Array.isArray(message.content)
        ) {
          continue;
        }

        for (
          let partIndex = 0;
          partIndex < message.content.length;
          partIndex++
        ) {
          const part = message.content[partIndex];

          if (part.type !== "thinking") {
            continue;
          }

          const thinking =
            typeof part.thinking === "string"
              ? part.thinking
              : "";

          if (
            part.redacted ||
            !thinking.trim()
          ) {
            continue;
          }

          const keys = getThinkingBlockKeys(
            message,
            partIndex,
            thinking,
          );

          const wasAlreadyMarked = keys.some(
            (key) => removed.has(key),
          );

          for (const key of keys) {
            removed.add(key);
          }

          if (!wasAlreadyMarked) {
            newlyMarkedBlocks++;
          }
        }
      }

      state.removedThinking = [...removed];

      await saveSessionCache(state);

      ctx.ui.notify(
        `Marked ${newlyMarkedBlocks} existing thinking blocks for removal. New thinking blocks will remain included.`,
        "info",
      );
    },
  });

  // ----------------------------------------------------------
  // /add-all-think and /add-all-thinking
  // ----------------------------------------------------------

  // Clear the removal list.
  // Previously excluded thinking becomes eligible for inclusion again.

  const addAllThinkingHandler = async (
    _args: string,
    ctx: any,
  ) => {
    await ctx.waitForIdle();

    const state = await getCurrentState(ctx);

    const restoredCount =
      state.removedThinking.length;

    state.removedThinking = [];

    await saveSessionCache(state);

    ctx.ui.notify(
      `Cleared ${restoredCount} removal identifiers. Previously excluded thinking is eligible for inclusion again.`,
      "info",
    );
  };

  pi.registerCommand("add-all-think", {
    description:
      "Restore thinking blocks previously excluded by /remove-all-think.",

    handler: addAllThinkingHandler,
  });

  // Longer alias.
  pi.registerCommand("add-all-thinking", {
    description:
      "Restore thinking blocks previously excluded by /remove-all-think.",

    handler: addAllThinkingHandler,
  });

  // ----------------------------------------------------------
  // AUTOMATIC END-OF-RUN COMPRESSION
  // ----------------------------------------------------------

  pi.on("agent_before_settle", async (event, ctx) => {
    try {
      await compressCompletedContext(
        event.context.contextMessages,
        ctx,
        settings,
      );
    } catch {
      // Never let compressor/cache errors interfere with agent completion.
    }
  });

  // ----------------------------------------------------------
  // CONTEXT HOOK
  // ----------------------------------------------------------

  pi.on("context", async (event, ctx) => {
    return applyContext(event, ctx);
  });
}

