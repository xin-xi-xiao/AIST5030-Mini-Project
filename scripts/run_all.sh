#!/usr/bin/env bash
set -euo pipefail

# Full 8-experiment matrix for one subject.
# Usage:
#   bash scripts/run_all.sh dog ./data/dreambooth/dog "a photo of sks dog" 800

SUBJECT="${1:-dog}"
DATA_DIR="${2:-./data/dreambooth/dog}"
PROMPT="${3:-a photo of sks dog}"
STEPS="${4:-800}"
MODEL="${MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
PROMPT_FILE="${PROMPT_FILE:-prompts/val_prompts_25.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${SUBJECT}}"
LOG_ROOT="${LOG_ROOT:-logs/${SUBJECT}}"
FIGURES_ROOT="${FIGURES_ROOT:-figures}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] data directory not found: ${DATA_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" logs "${FIGURES_ROOT}"
mkdir -p "${LOG_ROOT}" "${FIGURES_ROOT}"

echo "=========================================="
echo "OFT DreamBooth Suite"
echo "subject=${SUBJECT}"
echo "data=${DATA_DIR}"
echo "steps=${STEPS}"
echo "model=${MODEL}"
echo "=========================================="

run_pair() {
  local name_a="$1"
  local args_a="$2"
  local name_b="$3"
  local args_b="$4"

  local run_a=1
  local run_b=1
  if [[ -f "${OUTPUT_ROOT}/${name_a}/loss_history.json" ]]; then
    run_a=0
  fi
  if [[ -f "${OUTPUT_ROOT}/${name_b}/loss_history.json" ]]; then
    run_b=0
  fi

  echo "[RUN] ${name_a} on GPU0 | ${name_b} on GPU1"

  if [[ "${run_a}" -eq 0 ]]; then
    echo "[SKIP] ${name_a} already trained"
  fi
  if [[ "${run_b}" -eq 0 ]]; then
    echo "[SKIP] ${name_b} already trained"
  fi

  if [[ "${run_a}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} scripts/train.py \
      --pretrained_model "${MODEL}" \
      --instance_data_dir "${DATA_DIR}" \
      --instance_prompt "${PROMPT}" \
      --subject_class "${SUBJECT}" \
      --max_train_steps "${STEPS}" \
      --output_dir "${OUTPUT_ROOT}/${name_a}" \
      --val_prompt_file "${PROMPT_FILE}" \
      --measure_energy \
      ${args_a} > "${LOG_ROOT}/${name_a}.log" 2>&1 &
  fi

  if [[ "${run_b}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=1 ${PYTHON_BIN} scripts/train.py \
      --pretrained_model "${MODEL}" \
      --instance_data_dir "${DATA_DIR}" \
      --instance_prompt "${PROMPT}" \
      --subject_class "${SUBJECT}" \
      --max_train_steps "${STEPS}" \
      --output_dir "${OUTPUT_ROOT}/${name_b}" \
      --val_prompt_file "${PROMPT_FILE}" \
      --measure_energy \
      ${args_b} > "${LOG_ROOT}/${name_b}.log" 2>&1 &
  fi

  if [[ "${run_a}" -eq 1 || "${run_b}" -eq 1 ]]; then
    wait
  fi
  echo "[DONE] ${name_a}, ${name_b}"
}

run_pair "E1_oft_r4" "--method oft --r 4" "E2_oft_r8" "--method oft --r 8"
run_pair "E3_oft_r2" "--method oft --r 2" "E4_coft_r4" "--method coft --r 4 --eps 1e-3"
run_pair "E5_oft_cayley" "--method oft --r 4 --use_cayley_neumann" "E6_boft_b4" "--method boft --boft_block_size 4 --boft_n_butterfly_factor 2"
run_pair "E7_lora_r4" "--method lora --r 4" "E8_lora_r16" "--method lora --r 16"

echo "[EVAL] running evaluate.py for all experiments"
for exp_dir in "${OUTPUT_ROOT}"/E*; do
  [[ -d "${exp_dir}" ]] || continue
  if [[ -f "${exp_dir}/eval_results.json" ]]; then
    echo "[SKIP] eval exists for $(basename "${exp_dir}")"
    continue
  fi
  ${PYTHON_BIN} scripts/evaluate.py \
    --output_dir "${exp_dir}" \
    --instance_data_dir "${DATA_DIR}" \
    --subject_class "${SUBJECT}" \
    --prompt_file "${PROMPT_FILE}" > "${LOG_ROOT}/$(basename "${exp_dir}")_eval.log" 2>&1 || true
done

${PYTHON_BIN} scripts/collect_results.py --outputs_root "${OUTPUT_ROOT}" --save_csv "${FIGURES_ROOT}/summary_${SUBJECT}.csv"
${PYTHON_BIN} scripts/visualize.py --outputs_root "${OUTPUT_ROOT}" --figures_root "${FIGURES_ROOT}/${SUBJECT}"

echo "All done. See logs/ outputs/ and figures/."
