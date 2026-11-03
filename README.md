# Lede-Thinker — compress a model's thinking

A self-contained blackbox for compressing an AI agent's chain-of-thought (CoT).
Two CLIs take raw thinking text in (`--text` or `stdin`) and write compressed text
to stdout. A ready-to-install **pi extension** wires the deterministic compressor
into a live coding session and automatically shrinks old thinking blocks.

The bundled `lede/` package is a fork of [yonk-labs/lede](https://github.com/yonk-labs/lede)
(deterministic extractive summarization). This project reuses its sentence
segmenter and extends the idea into CoT compression.

```
Lede-Thinker/
├ isolate/                  # the self-contained blackbox (no preset data ships here)
   plain_compress.py        # deterministic CoT compressor (text in, text out, no LLM)
   llm_compress.py          # compress + ONE pi -p call to resume (needs pi)
   lede/                    # bundled summarizer package (fork of yonk-labs/lede)
├ Structure/                # install layout — how this repo becomes a pi extension
   .pi/agent/extensions/thinking-lede.ts   # the pi extension
   .pi/agent/thinking-lede.json            # settings (installed to ~/.pi/agent/)
   .pi/lede-isolate/                       # copy of isolate/ (installed to ~/.pi/lede-isolate/)
├ test/                     # test harness + sample session data
├ Analysis.md               # file-by-file analysis of the isolate/ code
└ README.md                 # this file
```

---

## 1. `isolate/` — the blackbox

### `plain_compress.py` — deterministic, no LLM

Shrinks a thinking trace to a target size using a **reasoning-event taxonomy**
(13 labels: GAP, REJECTION, OBJECTION, REVISION, RECONSIDERATION, ACCEPTANCE,
PROPOSAL, OBSERVATION, ACTION, SUPPORT, CONCLUSION, UNCERTAINTY, CONTRAST).
Selection reserves the **head** (task setup) and **tail** (current state /
what's done / what's blocked), guarantees at least one kept sentence per event
category present in the input, then fills the remaining budget with
novelty-penalized (MMR) selection. Agent scaffolding and pure filler are
stripped first; deduplication only merges near-identical sentences with
compatible polarity (negation-aware) and modality, so "X works" is never
collapsed into "X doesn't work".

No LLM or network is used. Its a pure Python stdlib + the bundled `lede.sentences.split_sentences`
segmenter. Same bytes every run.

```bash
# stdin -> stdout (the realistic path for large thinking)
cat thinking.md | python3 isolate/plain_compress.py --target 0.05 > compressed.md

# or pass text directly
python3 isolate/plain_compress.py --text "$(cat thinking.md)" --target 0.10

# optional: dump the sentence-level event ledger (labels/polarity/modality per sentence)
cat thinking.md | python3 isolate/plain_compress.py --target 0.05 --ledger-json events.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--text TEXT` | *(stdin if absent)* | raw thinking text |
| `--target F` | `0.05` | output size as a fraction of input (0.05 = 5%) |
| `--dedup F` | `0.85` | Jaccard threshold for conservative deduplication |
| `--head-frac F` | `0.15` | fraction of the target reserved for beginning context |
| `--tail-frac F` | `0.25` | fraction of the target reserved for ending context |
| `--debug` | off | print compression stats to stderr (sents kept / meta stripped / dups removed / label counts / output %) |
| `--ledger-json FILE` | *(none)* | also write a sentence-level event ledger (position, text, labels, polarity, modality, score) as JSON |

Use standalone when you only need compressed output to re-inject into a prompt.

### `llm_compress.py` — compress + ONE LLM call

Same compression as above, then a single `pi -p` call that hands the compressed
thinking to a model and asks it to resume: plan its next actions and report the
state. One LLM call in all cases, with or without ground truth.

```bash
# stdin -> stdout; default model "Summary" on provider llama.relay
cat thinking.md | python3 isolate/llm_compress.py --target 0.05

# with a completed-work file -> the SAME single call also self-scores (RAR)
cat thinking.md | python3 isolate/llm_compress.py --ground-truth done.md --target 0.05

# choose a different model / provider
cat thinking.md | python3 isolate/llm_compress.py --model "Qwen3.8 27B" --provider llama.relay
```

| Flag | Default | Meaning |
|---|---|---|
| `--text TEXT` | *(stdin if absent)* | raw thinking text |
| `--target F` | `0.05` | compression target as a fraction of input |
| `--dedup F` | `0.85` | Jaccard dedup threshold |
| `--head-frac F` | `0.15` | fraction of the target reserved for beginning context |
| `--tail-frac F` | `0.25` | fraction of the target reserved for ending context |
| `--model M` | `"Summary"` | pi model to use |
| `--provider P` | `llama.relay` | pi provider (ignored if `--model` already has a `provider/` prefix) |
| `--ground-truth FILE` | *(none)* | optional file describing work already completed; enables self-scoring in the same call |
| `--debug` | off | print compression stats + the exact prompt sent to pi (to stderr) |

**Output** — the LLM's reply on stdout:

```
ACTIONS: a numbered list of what it would do next, in order.
REPORT: <=3 sentences — what is done, what is blocked or open.
```

If `--ground-truth` is given, a third section is added in the same call:

```
SCORE: per-action label (redundant / necessary / neutral) + RAR <redundant>/<total>.
```

Without `--ground-truth`, the output is the continuation only. Either way it is
one `pi -p` call (60 s timeout); the compressed text is intermediate and shown
only under `--debug`.

---

## 2. `Structure/` — install the repo as a pi extension

The `Structure/` folder mirrors the **install layout**: every file under it maps
1:1 to a location in `~/.pi/`. After copying, the repo functions as a pi
extension that compresses old thinking in place.

| From this repo | Installed to |
|---|---|
| `Structure/.pi/agent/extensions/thinking-lede.ts` | `~/.pi/agent/extensions/thinking-lede.ts` |
| `Structure/.pi/agent/thinking-lede.json` | `~/.pi/agent/thinking-lede.json` |
| `Structure/.pi/lede-isolate/` | `~/.pi/lede-isolate/` |

```bash
# install (from the repo root)
mkdir -p ~/.pi/agent/extensions
cp Structure/.pi/agent/extensions/thinking-lede.ts ~/.pi/agent/extensions/
cp Structure/.pi/agent/thinking-lede.json ~/.pi/agent/   # optional — created with defaults on first load if absent
cp -r Structure/.pi/lede-isolate ~/.pi/
```

Then restart pi (or reload extensions). The extension loads, validates the
settings, and creates `~/.pi/agent/thinking-lede/` for its per-session caches.

### What the extension does

`thinking-lede.ts` is a pi agent extension. It **only ever runs
`~/.pi/lede-isolate/plain_compress.py`** — the deterministic compressor. It never
invokes `llm_compress.py`, `pi -p`, or any LLM; the path is hardcoded and the
settings file cannot redirect it elsewhere.

- **Automatic end-of-run compression.** On every `agent_before_settle`, each
  completed assistant thinking block whose estimated size exceeds
  `maxThinkingTokens` (default 500, estimated as `chars / charsPerToken` with
  `charsPerToken = 4`) is compressed via the CLI (stdin in, stdout out) and
  cached. Identical blocks are compressed only once.
- **Context replacement.** On every `context` event, thinking blocks that have a
  cached compression are swapped for `[Compressed Thinking]\n<compressed>` before
  the prompt is built; blocks marked for removal are dropped entirely. Everything
  else passes through untouched.
- **Per-session cache.** Compressions are keyed by SHA-256 of the original text
  and stored per session in `~/.pi/agent/thinking-lede/<session>.json`, written
  atomically (temp file + rename) and only *after* persistence succeeds is the
  compressed output allowed to replace the original. Cache errors never affect
  the conversation.
- **Safe failure.** A failed, timed-out, empty, or non-shrinking compression
  result is rejected — the original thinking simply stays in context.

### Commands

| Command | Effect |
|---|---|
| `/compact-all-thinking` | Immediately compress **all** historical thinking blocks, regardless of size (force mode). Retries on re-run if some failed. |
| `/remove-all-think` | Exclude all *existing* thinking blocks from future prompts. New thinking remains included. |
| `/add-all-think` (alias `/add-all-thinking`) | Clear the removal list; previously excluded thinking becomes eligible for inclusion again. |

A progress widget is shown while bulk compression runs (per-block bar,
compressed/unchanged/failed counts).

### Settings — `~/.pi/agent/thinking-lede.json`

Created with defaults on first load if absent. The shipped file:

```json
{
  "pythonExecutable": "python3",
  "maxThinkingTokens": 500,
  "charsPerToken": 4,
  "target": 0.20,
  "dedup": 0.85,
  "headFrac": 0.15,
  "tailFrac": 0.25,
  "timeoutMs": 120000
}
```

| Key | Default (code) | Meaning |
|---|---|---|
| `pythonExecutable` | `"python3"` | interpreter used to run the compressor |
| `maxThinkingTokens` | `500` | blocks above this estimated size are auto-compressed at end of run |
| `charsPerToken` | `4` | divisor for the token estimate (`ceil(chars / charsPerToken)`) |
| `target` | `0.05` (shipped file: `0.20`) | `--target` passed to plain_compress.py |
| `dedup` | `0.85` | `--dedup` passed to plain_compress.py |
| `headFrac` | `0.15` | `--head-frac` passed to plain_compress.py |
| `tailFrac` | `0.25` | `--tail-frac` passed to plain_compress.py |
| `timeoutMs` | `120000` | max time per compression; the child is SIGKILLed on timeout |

`headFrac + tailFrac` must not exceed 1. All values are validated at load; an
invalid settings file raises instead of silently falling back.

---

## 3. `test/` — test harness and sample data

| Path | What it does |
|---|---|
| `test/test-files/session-html/` | A real pi session export: `pi-session.html` (4.6 MB) and its decoded `session.json`. |
| `test/test-files/extract_thinking.py` | Extracts the raw thinking blocks from `session.json` ➜ `thinking_blocks.json` (682 blocks). |
| `test/test-files/thinking_blocks.json` | The 682 raw thinking strings, in session order. |
| `test/remove_code_blocks.py` | Strips fenced code (``` ... ```) from each block ➜ `thinking_nocode.json`. Fences are paired in order of appearance; an unpaired trailing fence keeps itself and everything after it (preserves blocks that merely *mention* a fence in prose). |
| `test/test-files/thinking_nocode.json` | The 682 blocks with code removed. |
| `test/run_plain_compress.py` | Drives **only the CLI** of plain_compress.py (`subprocess`, stdin ➜ stdout, nothing imported) over each block in `thinking_nocode.json` at `--target 0.20`, and writes the collected outputs to `results/plain_thinking.json` (JSON array of strings, same order as input). |
| `test/results/plain_thinking.json` | Per-block compression output from the run above (~630k chars in ➜ ~129k out). |

```bash
# reproduce the per-block run
python3 test/run_plain_compress.py
```

---

## Requirements

- **Python 3.10+** (stdlib only for compression; no third-party packages needed).
- **`pi` CLI on PATH with a working model** — needed only for `llm_compress.py`.
  Default is `--model "Summary"` on `--provider llama.relay`; override with
  `--model` / `--provider` if your setup uses different names. The prompt is
  piped via stdin (it can exceed ARG_MAX).
- **A pi agent** — needed only for the `Structure/` extension, which runs inside
  pi and needs Python on PATH for the compressor.

## Notes

- **Deterministic compression, non-deterministic resume.** `plain_compress.py`
  and the compression step of `llm_compress.py` share one engine and are
  byte-identical across runs; the LLM continuation is not (it's a live model call).
  The pi extension only ever uses the deterministic path.
- **Coverage guarantee.** Every reasoning-event category present in the input
  appears in the output (at least one kept sentence per category).
- **Auditability.** `--ledger-json` writes a position-tied event ledger — useful
  for debugging selections or running deterministic fact-coverage checks without
  any LLM.
- **Fail-safe extension.** Compressor failures, timeouts, and non-shrinking
  outputs leave the original thinking untouched.

## Credit

The `lede/` package — deterministic extractive summarization (TF-IDF + position + length scoring, sentence segmentation, fact extraction) — is written by
[yonk-labs](https://github.com/yonk-labs/lede) and ships here as a fork. This
project (Lede-Thinker) builds on it: the compressor reuses `lede.sentences.split_sentences`
for segmentation, and the reasoning-event taxonomy, coverage-first MMR selection,
the two CLIs, and the pi extension are this project's additions.
