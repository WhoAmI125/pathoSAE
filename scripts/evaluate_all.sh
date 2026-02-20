#!/usr/bin/env bash
set -euo pipefail

MODELS=(vanilla gated topk jumprelu)
CHECKPOINTS=()

for model in "${MODELS[@]}"; do
  run_name="${model}_e32_ep10"
  ckpt="models/checkpoints/${run_name}/best.pt"

  if [[ -f "${ckpt}" ]]; then
    echo "[evaluate] checkpoint=${ckpt}"
    python tasks/evaluate.py --checkpoint "${ckpt}" "$@"
    CHECKPOINTS+=("${ckpt}")
  else
    echo "[skip] missing checkpoint: ${ckpt}" >&2
  fi
done

if [[ ${#CHECKPOINTS[@]} -ge 2 ]]; then
  echo "[compare] checkpoints=${#CHECKPOINTS[@]}"
  python tasks/compare.py --checkpoints "${CHECKPOINTS[@]}" "$@"
else
  echo "[info] need at least two checkpoints to run compare" >&2
fi
