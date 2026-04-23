#!/usr/bin/env bash
# Download latest kernel output and submit to the competition.
# Usage: submit_from_kernel.sh <message>
set -euo pipefail

MESSAGE="${1:-aug_v2 submission}"
KERNEL="georgymamarin/asr-spoken-numbers-full-test-infer"
COMPETITION="asr-2026-spoken-numbers-recognition-challenge"
OUT_DIR="$(cd "$(dirname "$0")"/.. && pwd)/kaggle_assets/full_test_kernel/output_latest"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
kaggle kernels output "$KERNEL" -p "$OUT_DIR"

SUB="$OUT_DIR/submission.csv"
if [ ! -f "$SUB" ]; then
    echo "submission.csv not found in $OUT_DIR" >&2
    ls -la "$OUT_DIR" >&2
    exit 1
fi
head -3 "$SUB"
wc -l "$SUB"
kaggle competitions submit -c "$COMPETITION" -f "$SUB" -m "$MESSAGE"
echo "submitted"
