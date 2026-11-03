#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE="$ROOT/isolate"
TARGET="$ROOT/Structure/.pi/lede-isolate"

mkdir -p "$TARGET"

rsync -av --delete \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    "$SOURCE/lede/" \
    "$TARGET/lede/"

cp "$SOURCE/llm_compress.py" "$TARGET/"
cp "$SOURCE/plain_compress.py" "$TARGET/"

echo "Runtime synchronization complete."
