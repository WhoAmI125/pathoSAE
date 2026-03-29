#!/usr/bin/env bash
set -euo pipefail

# DriftPrior matched-L0 sweep (sequential, single-process).
# Default priors are chosen to roughly match L0 bands with canonical runs:
#   p=0.96 -> expected L0~983, p=0.97 -> ~737, p=0.98 -> ~492 (for hidden_dim=24576)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-auto}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EXPANSION="${EXPANSION:-32}"
DATA_DIR="${DATA_DIR:-data/activations}"
TAU="${TAU:-0.1}"
BETA_START="${BETA_START:-0.01}"
BETA_END="${BETA_END:-0.1}"
N_POS="${N_POS:-512}"
SUB_BATCH="${SUB_BATCH:-512}"
PHI_DIM="${PHI_DIM:-256}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${PRIORS:-}" ]]; then
  # shellcheck disable=SC2206
  PRIOR_LIST=(${PRIORS})
else
  PRIOR_LIST=(0.96 0.97 0.98)
fi

export PYTHONPATH="${PYTHONPATH:-./}"

for prior in "${PRIOR_LIST[@]}"; do
  tag="$(echo "${prior}" | tr '.' 'p')"
  run_name="driftprior_e${EXPANSION}_ep${EPOCHS}_ps${tag}"

  echo "============================================================"
  echo "Run: ${run_name}"
  echo "  prior_sparsity=${prior} gpu=${GPU} epochs=${EPOCHS}"
  echo "============================================================"

  "${PYTHON_BIN}" tasks/train.py \
    --model driftprior \
    --gpu "${GPU}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --expansion_factor "${EXPANSION}" \
    --data_dir "${DATA_DIR}" \
    --drift_phi_dim "${PHI_DIM}" \
    --drift_sub_batch "${SUB_BATCH}" \
    --drift_n_pos "${N_POS}" \
    --drift_tau "${TAU}" \
    --drift_beta_start "${BETA_START}" \
    --drift_beta_end "${BETA_END}" \
    --drift_prior_sparsity "${prior}" \
    --run_name "${run_name}" \
    --no_wandb

  "${PYTHON_BIN}" tasks/evaluate.py \
    --checkpoint "models/checkpoints/${run_name}/best.pt" \
    --gpu "${GPU}" \
    --split val
done

echo "DriftPrior matched-L0 sweep completed."
