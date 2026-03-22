#!/usr/bin/env bash
set -euo pipefail

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic not found. Install with:"
  echo "  conda install -y -n base -c conda-forge tectonic"
  exit 1
fi

tectonic main.tex
tectonic submission_checklist.tex

echo "Built: $(pwd)/main.pdf"
echo "Built: $(pwd)/submission_checklist.pdf"
