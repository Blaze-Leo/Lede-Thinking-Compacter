#!/usr/bin/env python3
"""Remove fenced code blocks (``` ... ```) from each thinking block.

The input is a JSON array of raw thinking strings (one per model
thinking block). For every block we remove the content of its fenced
code blocks and write the cleaned strings back out as a new JSON array.

Fence pairing
-------------
``` markers are paired in order of appearance: the 1st opens, the 2nd
closes, the 3rd opens, the 4th closes, and so on. Each complete
open/close span (both fence markers plus everything between them) is
removed.

Odd number of fences
--------------------
If a block has an odd number of ``` markers, the final marker has no
closing fence. Removal then STOPS at that point: the unpaired marker and
everything after it are kept verbatim. This is what preserves blocks that
merely *mention* "```" in prose (a single literal fence) instead of
deleting half the block.

Only stdlib is used.
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# Global input / output paths (both live next to this script, in test-files/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "test-files", "thinking_blocks.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "test-files", "thinking_nocode.json")

# Exactly three backticks, not part of a longer run (no 4+ runs exist in the data).
_FENCE_RE = re.compile(r"(?<!`)```(?!`)")


def remove_code_blocks(text: str) -> tuple[str, int]:
    """Return (cleaned_text, number_of_code_spans_removed)."""
    matches = list(_FENCE_RE.finditer(text))

    # Fewer than a full pair -> nothing to remove. This also covers the
    # odd single-fence case (a literal ``` quoted in prose): left untouched.
    if len(matches) < 2:
        return text, 0

    # Complete pairs only; an unpaired trailing fence is intentionally kept.
    n_pairs = len(matches) // 2
    spans = [
        (matches[2 * k].start(), matches[2 * k + 1].end()) for k in range(n_pairs)
    ]

    # Reconstruct the string, dropping each fenced span.
    parts: list[str] = []
    prev = 0
    for start, end in spans:
        parts.append(text[prev:start])
        prev = end
    parts.append(text[prev:])
    cleaned = "".join(parts)

    # Collapse the blank-line artifacts left behind by removing a block that
    # sat on its own lines (e.g. "\n\n```\n...\n```\n\n" -> one blank line).
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned, len(spans)


def main() -> None:
    with open(INPUT_FILE, encoding="utf-8") as f:
        blocks = json.load(f)

    total_spans = 0
    blocks_touched = 0
    out_blocks: list[str] = []
    for block in blocks:
        cleaned, n = remove_code_blocks(block)
        out_blocks.append(cleaned)
        total_spans += n
        if n:
            blocks_touched += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_blocks, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(blocks)} blocks")
    print(f"Blocks with code removed : {blocks_touched}")
    print(f"Code spans removed       : {total_spans}")
    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
