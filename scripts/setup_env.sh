#!/usr/bin/env bash
set -euo pipefail

# Create project-local virtual environment and install dependencies.
# Usage:
#   bash scripts/setup_env.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -d .venv ]]; then
  echo "[INFO] .venv already exists, reusing it."
else
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first.
# Try nightly CUDA build for latest GPU architectures (e.g., RTX 5090 / sm_120).
if ! pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128; then
  echo "[WARN] Nightly CUDA install failed, trying stable cu124."
  if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124; then
    echo "[WARN] Stable CUDA install failed, trying default index."
    pip install torch torchvision torchaudio
  fi
fi

pip install -r requirements.txt

echo "[OK] Environment ready. Activate with: source .venv/bin/activate"
