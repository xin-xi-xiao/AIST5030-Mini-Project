# Course Submission Checklist

Student: Mingjun Wang
Student ID: 1155251294
Repository: https://github.com/xin-xi-xiao/AIST5030-Mini-Project.git

## 1) Deliverables

- Code repository with clean README: done.
- Final Chinese report: report/report_final.md
- Final English report: report/report_final_en.md
- Compact LaTeX report package: report/latex_submission/main.tex

## 2) Reproducibility Commands

- Setup: bash scripts/setup_env.sh
- Full matrix run: bash scripts/run_all_subjects.sh
- Collect results: python scripts/collect_results.py --outputs_root outputs --save_csv figures/summary.csv
- Make plots: python scripts/visualize.py --outputs_root outputs --figures_root figures

## 3) Evidence Paths

- Full run outputs: outputs_realfull_20260323_011116
- Figures and summaries: figures/realfull_20260323_011116
- Submission tables: report/tables/
- Integrity recheck note: report/latex_submission/assets/tables/integrity_audit.txt

## 4) Integrity Audit Snapshot

- Expected groups: 24
- Observed groups: 24
- Issue count: 0
- ALL_OK=True

## 5) Risk Boundary

Conclusions are valid for the fixed assignment setup: 3 subjects, fixed 25 prompts, 800 steps, and SD-v1.5 base model.
