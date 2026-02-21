#!/usr/bin/env bash
# ============================================================
#  pathoSAEv3 — 4 SAE variants 동시 학습 런처
#  tmux 세션 안에서 돌려서 로그아웃해도 학습 유지
#
#  사용법:
#    bash scripts/launch_training.sh
#
#  GPU 할당: vanilla=1, gated=2, topk=3, jumprelu=5
#  로그 경로: logs/<variant>_<timestamp>.log
#
#  학습 중단: tmux kill-session -t sae_train
#  로그 확인: tail -f logs/vanilla_YYYYMMDD_HHMMSS.log
#  tmux 접속: tmux attach -t sae_train  →  Ctrl-b + 0~3 window 전환
# ============================================================
set -euo pipefail

PROJECT_DIR="/home/kimhj/projects/pathoSAEv3"
CONDA_ENV="pathosae3"
SESSION="sae_train"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# GPU 할당 (0,4는 사용중이므로 제외)
declare -A GPU_MAP=(
  [vanilla]=1
  [gated]=2
  [topk]=3
  [jumprelu]=5
)

MODELS=(vanilla gated topk jumprelu)

# 기존 세션 있으면 안내
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[error] tmux session '${SESSION}' already exists."
  echo "        attach:  tmux attach -t ${SESSION}"
  echo "        kill:    tmux kill-session -t ${SESSION}"
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "  pathoSAEv3 Training Launcher"
echo "=========================================="
echo "  conda env  : ${CONDA_ENV}"
echo "  project    : ${PROJECT_DIR}"
echo "  session    : ${SESSION}"
echo "  log dir    : ${LOG_DIR}"
echo ""

# 첫 번째 모델로 세션 생성
first_model="${MODELS[0]}"
first_gpu="${GPU_MAP[${first_model}]}"
first_log="${LOG_DIR}/${first_model}_${TIMESTAMP}.log"

tmux new-session -d -s "${SESSION}" -n "${first_model}" \
  "cd ${PROJECT_DIR} && \
   source /home/kimhj/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && \
   echo '[$(date)] START ${first_model} on GPU ${first_gpu}' | tee '${first_log}' && \
   PYTHONPATH=${PROJECT_DIR} python tasks/train.py \
     --model ${first_model} \
     --gpu ${first_gpu} \
     --no_wandb \
     2>&1 | tee -a '${first_log}' ; \
   echo '[$(date)] DONE ${first_model}' | tee -a '${first_log}' ; \
   echo 'Press Enter to close...' && read"

echo "  [window 0] ${first_model} → GPU ${first_gpu}  log: ${first_log}"

# 나머지 모델은 새 window로 추가
for i in $(seq 1 $((${#MODELS[@]} - 1))); do
  model="${MODELS[$i]}"
  gpu="${GPU_MAP[${model}]}"
  logfile="${LOG_DIR}/${model}_${TIMESTAMP}.log"

  tmux new-window -t "${SESSION}" -n "${model}" \
    "cd ${PROJECT_DIR} && \
     source /home/kimhj/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && \
     echo '[$(date)] START ${model} on GPU ${gpu}' | tee '${logfile}' && \
     PYTHONPATH=${PROJECT_DIR} python tasks/train.py \
       --model ${model} \
       --gpu ${gpu} \
       --no_wandb \
       2>&1 | tee -a '${logfile}' ; \
     echo '[$(date)] DONE ${model}' | tee -a '${logfile}' ; \
     echo 'Press Enter to close...' && read"

  echo "  [window $i] ${model} → GPU ${gpu}  log: ${logfile}"
done

echo ""
echo "=========================================="
echo "  tmux session '${SESSION}' launched!"
echo ""
echo "  접속:      tmux attach -t ${SESSION}"
echo "  detach:    Ctrl-b, d"
echo "  window:    Ctrl-b, 0~3"
echo "  로그 확인: tail -f ${LOG_DIR}/<variant>_${TIMESTAMP}.log"
echo "  전체 종료: tmux kill-session -t ${SESSION}"
echo "=========================================="
