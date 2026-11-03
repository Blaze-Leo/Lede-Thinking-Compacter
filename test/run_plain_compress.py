#!/usr/bin/env python3
"""Run plain_compress (CLI) over each thinking block in thinking_nocode.json,
one at a time, and collect the compressed outputs into results/plain_thinking.json.

The compressor is driven **only via its CLI** (subprocess: stdin in, stdout
out). Nothing from plain_compress is imported as Python.

Output is a JSON array of strings, one per input block, in the same order.
"""

import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Global paths + settings
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # .../test
REPO_ROOT = os.path.dirname(BASE_DIR)                            # .../Lede
PLAIN_COMPRESS = os.path.join(REPO_ROOT, "isolate", "plain_compress.py")
INPUT_FILE = os.path.join(BASE_DIR, "test-files", "thinking_nocode.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "results", "plain_thinking.json")

TARGET = 0.20            # compress each block to 20% of its own length
PER_BLOCK_TIMEOUT = 60   # seconds; a single block should finish well under this


def compress_block(text: str) -> str:
    """Compress one block by invoking the plain_compress CLI (stdin -> stdout)."""
    proc = subprocess.run(
        [sys.executable, PLAIN_COMPRESS, "--target", f"{TARGET}"],
        input=text,
        capture_output=True,
        text=True,
        timeout=PER_BLOCK_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"plain_compress failed (rc={proc.returncode}):\n{proc.stderr}"
        )
    return proc.stdout


def main() -> None:
    with open(INPUT_FILE, encoding="utf-8") as f:
        blocks = json.load(f)

    out: list[str] = []
    for i, block in enumerate(blocks):
        out.append(compress_block(block))
        if (i + 1) % 50 == 0 or i + 1 == len(blocks):
            print(f"  {i + 1}/{len(blocks)} done")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    total_in = sum(len(b) for b in blocks)
    total_out = sum(len(b) for b in out)
    print(f"Blocks : {len(blocks)}")
    print(f"Chars  : {total_in} -> {total_out}")
    print(f"Output : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
