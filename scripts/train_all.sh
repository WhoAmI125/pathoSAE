#!/usr/bin/env bash
set -euo pipefail

MODELS=(vanilla gated topk jumprelu)

for model in "${MODELS[@]}"; do
  echo "[train] model=${model}"
  python tasks/train.py --model "${model}" "$@"
done
