#!/usr/bin/env bash
# ============================================================
#  pathoSAEv3 원클릭 런처
#
#  사용법:
#    bash run.sh                      # 모든 variant (5개)
#    bash run.sh vanilla jumprelu     # 지정 variant만
#    bash run.sh msae                 # 1개만
#
#  처음 실행 (train_mean.pt 없을 때):
#    → sae_prep tmux에서 mean 계산 (10~40분)
#    → 완료 후 다시 bash run.sh [...models] 실행하면 즉시 학습 시작
#
#  이후 실행 (train_mean.pt 있을 때):
#    → 즉시 sae_train tmux N개 window 생성 (free GPU 자동 할당)
#
#  접속:   tmux attach -t sae_train
#  detach: Ctrl-b, d  /  window: Ctrl-b, 0~N
#  종료:   tmux kill-session -t sae_train
#  로그:   tail -f logs/<variant>_<timestamp>.log
# ============================================================
set -euo pipefail

PROJECT_DIR="/home/kimhj/projects/pathoSAEv3"
CONDA_ENV="pathosae3"
SESSION="sae_train"
LOG_DIR="${PROJECT_DIR}/logs"
MEAN_CACHE="${PROJECT_DIR}/data/activations/train_mean.pt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NUM_WORKERS=2

VALID_MODELS=(vanilla gated topk jumprelu msae)

# MSAE 전용 추가 인수
declare -A EXTRA_ARGS=(
  [vanilla]=""
  [gated]=""
  [topk]=""
  [jumprelu]=""
  [msae]="--msae_nesting 64,128,256,512 --msae_importance uniform"
)

# ── 실행할 모델 결정 ($@ 없으면 전체) ─────────────────────────
if [ $# -eq 0 ]; then
  MODELS=("${VALID_MODELS[@]}")
else
  MODELS=("$@")
  for m in "${MODELS[@]}"; do
    valid=false
    for v in "${VALID_MODELS[@]}"; do
      [ "$m" = "$v" ] && valid=true && break
    done
    if ! $valid; then
      echo "[error] Unknown model: '$m'"
      echo "        Valid: ${VALID_MODELS[*]}"
      exit 1
    fi
  done
fi

echo "  학습 대상: ${MODELS[*]}"

# ── 비어있는 GPU 목록 수집 (free ≥ 20000 MB) ──────────────────
FREE_GPUS=()
while IFS=, read -r idx free_mem; do
  idx="${idx// /}"
  free_mem="${free_mem// /}"
  if [ "$free_mem" -ge 20000 ]; then
    FREE_GPUS+=("$idx")
  fi
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

echo "  사용 가능 GPU: ${FREE_GPUS[*]:-없음}"

if [ ${#FREE_GPUS[@]} -eq 0 ]; then
  echo "[error] 사용 가능한 GPU(free ≥ 20GB)가 없습니다."
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
  exit 1
fi

if [ ${#FREE_GPUS[@]} -lt ${#MODELS[@]} ]; then
  echo "[warn] 요청 모델(${#MODELS[@]}개) > free GPU(${#FREE_GPUS[@]}개) → GPU 공유 발생"
fi

mkdir -p "${LOG_DIR}"

# ── Case 1: train_mean.pt 없으면 먼저 계산 ──────────────────────
if [ ! -f "${MEAN_CACHE}" ]; then
  PREP_SESSION="sae_prep"
  MEAN_LOG="${LOG_DIR}/precompute_mean_${TIMESTAMP}.log"

  if tmux has-session -t "${PREP_SESSION}" 2>/dev/null; then
    echo "[info] Mean computation already running."
    echo "       확인: tmux attach -t ${PREP_SESSION}"
    echo "       로그: tail -f ${LOG_DIR}/precompute_mean_*.log"
    exit 0
  fi

  echo "=========================================="
  echo "  [Step 1] train_mean.pt 사전 계산"
  echo "  tmux session: '${PREP_SESSION}'"
  echo "  완료 예상: 10~40분"
  echo "=========================================="

  tmux new-session -d -s "${PREP_SESSION}" -n "mean" \
    "cd ${PROJECT_DIR} && \
     source /home/kimhj/miniconda3/etc/profile.d/conda.sh && \
     conda activate ${CONDA_ENV} && \
     echo '[$(date)] Computing train_mean.pt ...' | tee '${MEAN_LOG}' && \
     PYTHONPATH=${PROJECT_DIR} python -c \"\
import torch; \
from src.config import SAEConfig; \
from src.extract.activation_store import ActivationStore; \
cfg = SAEConfig(); \
store = ActivationStore(cfg); \
print(f'train files: {len(store.train_files)}'); \
mean = store.compute_train_mean(); \
print(f'mean norm: {mean.norm().item():.4f}'); \
print('Done: ' + str(store.data_dir) + '/train_mean.pt'); \
\" 2>&1 | tee -a '${MEAN_LOG}' && \
     echo '[$(date)] DONE. Run: bash run.sh ${MODELS[*]}' | tee -a '${MEAN_LOG}' ; \
     echo 'Press Enter to close...' && read"

  echo ""
  echo "  진행 확인: tmux attach -t ${PREP_SESSION}"
  echo "  로그 확인: tail -f ${MEAN_LOG}"
  echo ""
  echo "  완료 후 다시 실행하세요:"
  echo "    bash run.sh ${MODELS[*]}"
  echo "=========================================="
  exit 0
fi

# ── Case 2: train_mean.pt 있으면 즉시 학습 시작 ──────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[error] tmux session '${SESSION}' already exists."
  echo "        attach:  tmux attach -t ${SESSION}"
  echo "        kill:    tmux kill-session -t ${SESSION}"
  exit 1
fi

echo "=========================================="
echo "  pathoSAEv3 학습 시작"
echo "  (train_mean.pt cached ✓)"
echo "  num_workers=${NUM_WORKERS}"
echo "=========================================="

first_model="${MODELS[0]}"
first_gpu="${FREE_GPUS[0]}"
first_log="${LOG_DIR}/${first_model}_${TIMESTAMP}.log"

tmux new-session -d -s "${SESSION}" -n "${first_model}" \
  "cd ${PROJECT_DIR} && \
   source /home/kimhj/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && \
   echo '[$(date)] START ${first_model} GPU=${first_gpu}' | tee '${first_log}' && \
   PYTHONPATH=${PROJECT_DIR} python tasks/train.py \
     --model ${first_model} --gpu ${first_gpu} --no_wandb --num_workers ${NUM_WORKERS} ${EXTRA_ARGS[${first_model}]} \
     2>&1 | tee -a '${first_log}' ; \
   echo '[$(date)] DONE ${first_model}' | tee -a '${first_log}' ; \
   echo 'Press Enter to close...' && read"

echo "  [window 0] ${first_model} → GPU ${first_gpu}"
echo "             log: ${first_log}"

for i in $(seq 1 $((${#MODELS[@]} - 1))); do
  model="${MODELS[$i]}"
  gpu_idx=$(( i < ${#FREE_GPUS[@]} ? i : ${#FREE_GPUS[@]} - 1 ))
  gpu="${FREE_GPUS[${gpu_idx}]}"
  logfile="${LOG_DIR}/${model}_${TIMESTAMP}.log"

  tmux new-window -t "${SESSION}" -n "${model}" \
    "cd ${PROJECT_DIR} && \
     source /home/kimhj/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && \
     echo '[$(date)] START ${model} GPU=${gpu}' | tee '${logfile}' && \
     PYTHONPATH=${PROJECT_DIR} python tasks/train.py \
       --model ${model} --gpu ${gpu} --no_wandb --num_workers ${NUM_WORKERS} ${EXTRA_ARGS[${model}]} \
       2>&1 | tee -a '${logfile}' ; \
     echo '[$(date)] DONE ${model}' | tee -a '${logfile}' ; \
     echo 'Press Enter to close...' && read"

  echo "  [window $i] ${model} → GPU ${gpu}"
  echo "             log: ${logfile}"
done

echo ""
echo "=========================================="
echo "  tmux session '${SESSION}' launched!"
echo ""
echo "  접속:    tmux attach -t ${SESSION}"
echo "  detach:  Ctrl-b, d  /  window: Ctrl-b, 0~$((${#MODELS[@]} - 1))"
echo "  종료:    tmux kill-session -t ${SESSION}"
echo "  로그:    tail -f ${LOG_DIR}/<variant>_${TIMESTAMP}.log"
echo "=========================================="
