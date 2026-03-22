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
tectonic main.tex
tectonic submission_checklist.tex
```

Output PDF:

- `main.pdf`
- `submission_checklist.pdf`

## Notes

- This package is self-contained and does not depend on external figure paths.
- Student info is embedded in the title section.
- If `tectonic` is missing, run: `conda install -y -n base -c conda-forge tectonic`
