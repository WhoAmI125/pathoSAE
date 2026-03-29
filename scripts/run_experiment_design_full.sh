#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/kimhj/projects/pathoSAEv3}"
ENV_NAME="${ENV_NAME:-pathosae3}"
CLINICAL_CSV="${CLINICAL_CSV:-/home/kimhj/projects/past_try/images/data/tcga_brca_survival/clinical_data(labels).csv}"
META_SMOKE="${META_SMOKE:-data/meta/group_meta_auto_vstatus_smoke.csv}"
META_FULL="${META_FULL:-data/meta/group_meta_auto_vstatus_full.csv}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT}/logs/exp_full_${TS}"
MAIN_LOG="${LOG_DIR}/main.log"

mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT}/data/meta"

run_py() {
  local name="$1"
  shift
  echo "[$(date)] START ${name}" | tee -a "${MAIN_LOG}"
  (
    cd "${PROJECT}" && \
    conda run --no-capture-output -n "${ENV_NAME}" \
      env PYTHONPATH="${PROJECT}" \
      python -u "$@"
  ) 2>&1 | tee -a "${MAIN_LOG}" "${LOG_DIR}/${name}.log"
  echo "[$(date)] DONE ${name}" | tee -a "${MAIN_LOG}"
}

run_pair() {
  local name1="$1"
  shift
  local cmd1="$1"
  shift
  local name2="$1"
  shift
  local cmd2="$1"
  shift

  echo "[$(date)] START PAIR ${name1} + ${name2}" | tee -a "${MAIN_LOG}"
  (
    cd "${PROJECT}" && \
    eval "${cmd1}"
  ) 2>&1 | tee -a "${MAIN_LOG}" "${LOG_DIR}/${name1}.log" &
  local p1=$!

  (
    cd "${PROJECT}" && \
    eval "${cmd2}"
  ) 2>&1 | tee -a "${MAIN_LOG}" "${LOG_DIR}/${name2}.log" &
  local p2=$!

  wait "${p1}"
  wait "${p2}"
  echo "[$(date)] DONE PAIR ${name1} + ${name2}" | tee -a "${MAIN_LOG}"
}

echo "[$(date)] EXPERIMENT DESIGN FULL RUN START" | tee -a "${MAIN_LOG}"
echo "PROJECT=${PROJECT}" | tee -a "${MAIN_LOG}"
echo "LOG_DIR=${LOG_DIR}" | tee -a "${MAIN_LOG}"

# 1) Gate: compile
(
  cd "${PROJECT}" && python -m compileall -q src tasks
) 2>&1 | tee -a "${MAIN_LOG}" "${LOG_DIR}/compileall.log"
echo "[$(date)] compileall OK" | tee -a "${MAIN_LOG}"

# 2) Gate: build meta smoke
run_py "meta_smoke" tasks/build_group_meta.py \
  --split full --max_files 1 \
  --clinical_csv "${CLINICAL_CSV}" \
  --patient_col bcr_patient_barcode \
  --class_col vital_status \
  --output_csv "${META_SMOKE}"

# 3) Gate: evaluate smoke
run_py "eval_smoke_vanilla" tasks/evaluate.py \
  --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
  --feature_figures --split full --max_files 1 --top_n 10 --top_k 5 \
  --meta_csv "${META_SMOKE}" --gpu 0

# 4) Gate: group smoke
run_py "group_smoke_vanilla" tasks/group_analysis.py \
  --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
  --meta_csv "${META_SMOKE}" \
  --split full --max_files 1 --top_n 50 --min_group_samples 20 --gpu 1

# 5) Full meta
run_py "meta_full" tasks/build_group_meta.py \
  --split full \
  --clinical_csv "${CLINICAL_CSV}" \
  --patient_col bcr_patient_barcode \
  --class_col vital_status \
  --output_csv "${META_FULL}"

# 6) Full evaluate (wave by 2)
run_pair \
  "eval_full_vanilla" \
  "conda run --no-capture-output -n ${ENV_NAME} env PYTHONPATH=${PROJECT} python -u tasks/evaluate.py --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt --feature_figures --split full --max_files 96 --top_n 10 --top_k 5 --meta_csv ${META_FULL} --gpu 0" \
  "eval_full_jumprelu" \
  "conda run --no-capture-output -n ${ENV_NAME} env PYTHONPATH=${PROJECT} python -u tasks/evaluate.py --checkpoint models/checkpoints/jumprelu_e32_ep10/best.pt --feature_figures --split full --max_files 96 --top_n 10 --top_k 5 --meta_csv ${META_FULL} --gpu 1"

run_pair \
  "eval_full_gated" \
  "conda run --no-capture-output -n ${ENV_NAME} env PYTHONPATH=${PROJECT} python -u tasks/evaluate.py --checkpoint models/checkpoints/gated_e32_ep10/best.pt --feature_figures --split full --max_files 96 --top_n 10 --top_k 5 --meta_csv ${META_FULL} --gpu 0" \
  "eval_full_topk" \
  "conda run --no-capture-output -n ${ENV_NAME} env PYTHONPATH=${PROJECT} python -u tasks/evaluate.py --checkpoint models/checkpoints/topk_e32_ep10/best.pt --feature_figures --split full --max_files 96 --top_n 10 --top_k 5 --meta_csv ${META_FULL} --gpu 1"

run_py "eval_full_msae" tasks/evaluate.py \
  --checkpoint models/checkpoints/msae_e32_ep10/best.pt \
  --feature_figures --split full --max_files 96 --top_n 10 --top_k 5 \
  --meta_csv "${META_FULL}" --gpu 0

# 7) Full group analysis (wave by 2)
run_pair \
  "group_full_vanilla" \
  "conda run --no-capture-output -n ${ENV_NAME} env PYTHONPATH=${PROJECT} python -u tasks/group_analysis.py --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt --meta_csv ${META_FULL} --split full --max_files 96 --top_n 50 --min_group_samples 20 --gpu 2" \
  "group_full_jumprelu" \
  "conda run --no-capture-output -n ${ENV_NAME} env PYTHONPATH=${PROJECT} python -u tasks/group_analysis.py --checkpoint models/checkpoints/jumprelu_e32_ep10/best.pt --meta_csv ${META_FULL} --split full --max_files 96 --top_n 50 --min_group_samples 20 --gpu 3"

run_pair \
  "group_full_gated" \
  "conda run --no-capture-output -n ${ENV_NAME} env PYTHONPATH=${PROJECT} python -u tasks/group_analysis.py --checkpoint models/checkpoints/gated_e32_ep10/best.pt --meta_csv ${META_FULL} --split full --max_files 96 --top_n 50 --min_group_samples 20 --gpu 2" \
  "group_full_topk" \
  "conda run --no-capture-output -n ${ENV_NAME} env PYTHONPATH=${PROJECT} python -u tasks/group_analysis.py --checkpoint models/checkpoints/topk_e32_ep10/best.pt --meta_csv ${META_FULL} --split full --max_files 96 --top_n 50 --min_group_samples 20 --gpu 3"

run_py "group_full_msae" tasks/group_analysis.py \
  --checkpoint models/checkpoints/msae_e32_ep10/best.pt \
  --meta_csv "${META_FULL}" \
  --split full --max_files 96 --top_n 50 --min_group_samples 20 --gpu 2

# 8) 5-model comparison
run_py "compare_full_5models" tasks/compare.py \
  --checkpoints \
  models/checkpoints/vanilla_e32_ep10/best.pt \
  models/checkpoints/gated_e32_ep10/best.pt \
  models/checkpoints/topk_e32_ep10/best.pt \
  models/checkpoints/jumprelu_e32_ep10/best.pt \
  models/checkpoints/msae_e32_ep10/best.pt \
  --gpu 0

# 9) Final report
run_py "report_full" tasks/generate_experiment_report.py \
  --output results/summary/experiment_design_full_report.md

echo "[$(date)] EXPERIMENT DESIGN FULL RUN DONE" | tee -a "${MAIN_LOG}"
echo "MAIN_LOG=${MAIN_LOG}"
