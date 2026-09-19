#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "config/PRI_inference.env"

command=("$PYTHON" -u src/predict.py
  --task pri --saprot-dir "$SAPROT_DIR"
  --checkpoint "$CHECKPOINT" --input "$INPUT"
  --device "$DEVICE" --output "$OUTPUT")
command+=(--ernie-checkpoint "$ERNIE_CHECKPOINT" --ernie-code "$ERNIE_CODE")

# Full benchmark mode uses frozen masks and the configuration beside the weights.
if [[ "${1:-}" == "--benchmark" ]]; then
  shift
  command=("$PYTHON" -u src/benchmark.py --task pri
    --saprot-dir "$SAPROT_DIR" --checkpoint "$CHECKPOINT"
    --device "$DEVICE" --output "${BENCHMARK_OUTPUT:-data/benchmarks/pri}")
  command+=(--ernie-checkpoint "$ERNIE_CHECKPOINT" --ernie-code "$ERNIE_CODE")
fi

# Preview the command without loading a model or starting training.
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  printf '%q ' "${command[@]}" "$@"
  printf '\n'
  exit 0
fi
exec "${command[@]}" "$@"
