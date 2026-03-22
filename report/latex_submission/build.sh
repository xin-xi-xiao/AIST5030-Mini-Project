#!/usr/bin/env bash
set -euo pipefail

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Install TeX first, e.g.:"
  echo "  sudo apt install texlive-latex-base texlive-latex-recommended texlive-latex-extra"
  exit 1
fi

pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

echo "Built: $(pwd)/main.pdf"
