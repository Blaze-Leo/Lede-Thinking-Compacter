#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

diff -qr \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    "$ROOT/isolate/lede" \
    "$ROOT/Structure/.pi/lede-isolate/lede"

diff -q "$ROOT/isolate/llm_compress.py" \
    "$ROOT/Structure/.pi/lede-isolate/llm_compress.py"

diff -q "$ROOT/isolate/plain_compress.py" \
    "$ROOT/Structure/.pi/lede-isolate/plain_compress.py"

echo "Everything is synchronized."
