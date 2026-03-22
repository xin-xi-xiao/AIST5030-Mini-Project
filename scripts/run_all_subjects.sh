#!/usr/bin/env bash
set -euo pipefail

# Run full 8-experiment matrix for dog/backpack/cat sequentially.
# Each subject run internally uses both GPUs in parallel.

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
MODEL="${MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
STEPS="${STEPS:-800}"
PROMPT_FILE="${PROMPT_FILE:-prompts/val_prompts_25.txt}"
OUTPUTS_BASE="${OUTPUTS_BASE:-outputs}"
LOGS_BASE="${LOGS_BASE:-logs}"
FIGURES_ROOT="${FIGURES_ROOT:-figures}"

run_subject() {
  local subject="$1"
  local data_dir="data/dreambooth/${subject}"
  local prompt="a photo of sks ${subject}"

  if [[ ! -d "${data_dir}" ]]; then
    echo "[WARN] skip ${subject}: ${data_dir} not found"
    return
  fi

  echo "=================================================="
  echo "Running subject=${subject} | steps=${STEPS} | model=${MODEL}"
  echo "=================================================="

  MODEL="${MODEL}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  PROMPT_FILE="${PROMPT_FILE}" \
  LOG_ROOT="${LOGS_BASE}/${subject}" \
  FIGURES_ROOT="${FIGURES_ROOT}" \
  OUTPUT_ROOT="${OUTPUTS_BASE}/${subject}" \
  bash scripts/run_all.sh "${subject}" "${data_dir}" "${prompt}" "${STEPS}"
}

run_subject dog
run_subject backpack
run_subject cat

# Global merge table across all subjects
mkdir -p "${FIGURES_ROOT}"
${PYTHON_BIN} scripts/collect_results.py --outputs_root "${OUTPUTS_BASE}" --save_csv "${FIGURES_ROOT}/summary_all_subjects.csv" --recursive

echo "Done all subjects. See ${FIGURES_ROOT}/summary_all_subjects.csv"
