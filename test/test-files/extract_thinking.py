#!/usr/bin/env python3
"""Extract thinking blocks from a pi harness session export (decoded session JSON).

Input : the decoded pi session v3 JSON (as embedded, base64-encoded, in the
        exported HTML under <script id="session-data">).
Output: a JSON array containing only the thinking texts, in entry order.
"""

import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SESSION_JSON_PATH = "/mnt/Personal/Repository/Lede/test/test-files/session-html/session.json"
OUTPUT_JSON_PATH = "/mnt/Personal/Repository/Lede/test/test-files/thinking_blocks.json"


def main() -> None:
    with open(SESSION_JSON_PATH, encoding="utf-8") as f:
        session = json.load(f)

    thinking_texts: list[str] = []
    for entry in session.get("entries", []):
        if entry.get("type") != "message":
            continue
        content = (entry.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text = (block.get("thinking") or "").strip()
                if text:
                    thinking_texts.append(text)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(thinking_texts, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(thinking_texts)} thinking blocks -> {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
