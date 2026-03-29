#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/evaluate_individual_top_features.sh <checkpoint> [gpu] [split] [top_n] [top_k] [max_files]
#
# Example:
#   bash scripts/evaluate_individual_top_features.sh \
#     models/checkpoints/vanilla_e32_ep10/best.pt 0 val 10 5

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-}"
GPU="${2:-auto}"
SPLIT="${3:-val}"
TOP_N="${4:-10}"
TOP_K="${5:-5}"
MAX_FILES="${6:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint path is required."
  echo "Usage: bash scripts/evaluate_individual_top_features.sh <checkpoint> [gpu] [split] [top_n] [top_k] [max_files]"
  exit 1
fi

cd "${PROJECT_ROOT}"

source /home/kimhj/miniconda3/etc/profile.d/conda.sh
conda activate pathosae3

CMD=(
  python -u tasks/evaluate.py
  --checkpoint "${CHECKPOINT}"
  --gpu "${GPU}"
  --feature_figures
  --split "${SPLIT}"
  --top_n "${TOP_N}"
  --top_k "${TOP_K}"
  --save_individual_top_features
)

if [[ -f "data/meta/group_meta_auto_vstatus_full.csv" ]]; then
  CMD+=(--meta_csv "data/meta/group_meta_auto_vstatus_full.csv")
fi

if [[ -n "${MAX_FILES}" ]]; then
  CMD+=(--max_files "${MAX_FILES}")
fi

PYTHONPATH="${PROJECT_ROOT}" "${CMD[@]}"
