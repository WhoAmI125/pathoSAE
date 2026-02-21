#!/usr/bin/env bash
# train_mean.pt 한 번 계산 후 캐시 → 이후 학습은 즉시 시작
set -euo pipefail
PROJECT_DIR="/home/kimhj/projects/pathoSAEv3"
CONDA_ENV="pathosae3"
LOG="${PROJECT_DIR}/logs/precompute_mean.log"

mkdir -p "${PROJECT_DIR}/logs"
echo "[$(date)] Computing train mean (one-time, ~10-40 min)..." | tee "${LOG}"

source /home/kimhj/miniconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV}"
PYTHONPATH="${PROJECT_DIR}" python - <<'EOF' 2>&1 | tee -a "${LOG}"
import torch
from src.config import SAEConfig
from src.extract.activation_store import ActivationStore

cfg = SAEConfig()
store = ActivationStore(cfg)
print(f"Train files: {len(store.train_files)}")
mean = store.compute_train_mean()
print(f"Mean computed: shape={mean.shape}, norm={mean.norm().item():.4f}")
print(f"Cached to: {store.data_dir}/train_mean.pt")
EOF

echo "[$(date)] Done. Now run: bash scripts/launch_training.sh" | tee -a "${LOG}"
