#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
source "config/PRI_train.env"

command=("$PYTHON" -u src/train.py
  --task pri --saprot-dir "$SAPROT_DIR"
  --data-dir "$DATA_DIR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE" --seed "$SEED"
  --device "$DEVICE" --output "$OUTPUT")
command+=(--ernie-checkpoint "$ERNIE_CHECKPOINT" --ernie-code "$ERNIE_CODE")
if [[ -n "$PROTEIN_INIT" ]]; then
  command+=(--protein-init "$PROTEIN_INIT")
fi

# Preview the command without loading a model or starting training.
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  printf '%q ' "${command[@]}" "$@"
  printf '\n'
  exit 0
fi
exec "${command[@]}" "$@"
