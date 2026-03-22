# LaTeX Submission Package (3-page compact report)

This folder is a standalone LaTeX package for the final report.

## Files

- `main.tex`: compact 2-column report in English.
- `assets/plots/`: selected metric/loss/energy figures.
- `assets/qual/`: curated before-vs-after qualitative mosaics.
- `assets/tables/`: CSV result tables used by the report.

## Build

```bash
cd report/latex_submission
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Output PDF:

- `main.pdf`

## Notes

- This package is self-contained and does not depend on external figure paths.
- Student info is embedded in the title section.
