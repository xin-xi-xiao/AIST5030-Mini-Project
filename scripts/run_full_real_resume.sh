#!/usr/bin/env bash
set -euo pipefail

# Resumable full real-data run for dog/backpack/cat.
# Behavior:
# - If a subject is fully complete (loss + eval for E1..E8), skip.
# - If training outputs exist but eval is missing, only run missing eval and redraw figures.
# - Otherwise, run full training matrix for the subject with two GPUs.

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
MODEL="${MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
STEPS="${STEPS:-800}"
PROMPT_FILE="${PROMPT_FILE:-prompts/val_prompts_25.txt}"
OUTPUTS_BASE="${OUTPUTS_BASE:-outputs_real_full}"

EXPS=(
  E1_oft_r4
  E2_oft_r8
  E3_oft_r2
  E4_coft_r4
  E5_oft_cayley
  E6_boft_b4
  E7_lora_r4
  E8_lora_r16
)

is_subject_complete() {
  local subject="$1"
  local root="${OUTPUTS_BASE}/${subject}"
  for exp in "${EXPS[@]}"; do
    [[ -f "${root}/${exp}/loss_history.json" ]] || return 1
    [[ -f "${root}/${exp}/eval_results.json" ]] || return 1
  done
  return 0
}

needs_training() {
  local subject="$1"
  local root="${OUTPUTS_BASE}/${subject}"
  for exp in "${EXPS[@]}"; do
    [[ -f "${root}/${exp}/loss_history.json" ]] || return 0
    [[ -f "${root}/${exp}/config.json" ]] || return 0
  done
  return 1
}

run_missing_eval() {
  local subject="$1"
  local data_dir="data/dreambooth/${subject}"
  local root="${OUTPUTS_BASE}/${subject}"

  echo "[EVAL] subject=${subject}"
  for exp in "${EXPS[@]}"; do
    local d="${root}/${exp}"
    if [[ -d "${d}" && -f "${d}/loss_history.json" && ! -f "${d}/eval_results.json" ]]; then
      echo "  - evaluating ${subject}/${exp}"
      "${PYTHON_BIN}" scripts/evaluate.py \
        --output_dir "${d}" \
        --instance_data_dir "${data_dir}" \
        --subject_class "${subject}" \
        --prompt_file "${PROMPT_FILE}" || true
    fi
  done

  "${PYTHON_BIN}" scripts/collect_results.py \
    --outputs_root "${root}" \
    --save_csv "figures/summary_${subject}.csv"

  "${PYTHON_BIN}" scripts/visualize.py \
    --outputs_root "${root}" \
    --figures_root "figures/${subject}"
}

run_subject() {
  local subject="$1"
  local data_dir="data/dreambooth/${subject}"
  local prompt="a photo of sks ${subject}"

  if [[ ! -d "${data_dir}" ]]; then
    echo "[WARN] skip ${subject}: data not found at ${data_dir}"
    return
  fi

  if is_subject_complete "${subject}"; then
    echo "[SKIP] ${subject} already complete."
    return
  fi

  if needs_training "${subject}"; then
    echo "[TRAIN] ${subject}: running full E1..E8 matrix"
    MODEL="${MODEL}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    PROMPT_FILE="${PROMPT_FILE}" \
    OUTPUT_ROOT="${OUTPUTS_BASE}/${subject}" \
    bash scripts/run_all.sh "${subject}" "${data_dir}" "${prompt}" "${STEPS}"
  else
    echo "[RESUME] ${subject}: training exists, completing missing eval/figures"
    run_missing_eval "${subject}"
  fi
}

echo "=================================================="
echo "Resumable Full Real Run"
echo "MODEL=${MODEL}"
echo "STEPS=${STEPS}"
echo "OUTPUTS_BASE=${OUTPUTS_BASE}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "=================================================="

run_subject dog
run_subject backpack
run_subject cat

"${PYTHON_BIN}" scripts/collect_results.py \
  --outputs_root "${OUTPUTS_BASE}" \
  --save_csv "figures/summary_all_subjects.csv" \
  --recursive

echo "[DONE] summary file: figures/summary_all_subjects.csv"