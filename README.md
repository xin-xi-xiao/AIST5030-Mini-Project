# OFT DreamBooth Mini-Project

Parameter-efficient finetuning for pretrained foundation models using OFT, COFT, BOFT, and LoRA on Stable Diffusion v1.5.

## 1. What This Project Delivers

1. Full 8-run PEFT matrix (E1..E8) on DreamBooth-style subject generation.
2. Three subjects: dog, backpack, cat.
3. Quantitative evaluation: CLIP-I, CLIP-T, DINO, trainable params, energy change.
4. Qualitative before/after outputs.
5. Final reports:
   - Chinese: report/report_final.md
   - English: report/report_final_en.md

## 2. Repository Layout

```text
oft-dreambooth-project/
├── README.md
├── requirements.txt
├── configs/
├── data/
├── prompts/
├── scripts/
├── figures/
├── outputs/
└── report/
```

Important scripts:
1. scripts/train.py: unified trainer for OFT/COFT/BOFT/LoRA.
2. scripts/evaluate.py: CLIP-I/CLIP-T/DINO evaluation.
3. scripts/collect_results.py: summary CSV generation.
4. scripts/visualize.py: report figures.
5. scripts/run_all.sh: 8 runs for one subject.
6. scripts/run_all_subjects.sh: all subjects.

## 3. Environment Setup

```bash
cd /data0/3DIC_workspace/miniproject/oft-dreambooth-project
bash scripts/setup_env.sh
source .venv/bin/activate
```

Manual install:

```bash
pip install -r requirements.txt
```

## 4. Data Preparation

Put subject images under:

```text
data/dreambooth/dog/
data/dreambooth/backpack/
data/dreambooth/cat/
```

Use 4-6 clean reference images per subject.

## 5. Run Experiments

Single subject full matrix:

```bash
bash scripts/run_all.sh dog data/dreambooth/dog "a photo of sks dog" 800
```

All subjects:

```bash
bash scripts/run_all_subjects.sh
```

Validation prompt file:

```text
prompts/val_prompts_25.txt
```

## 6. Key Output Artifacts

1. figures/realfull_20260323_011116/summary_all_subjects.csv
2. figures/realfull_20260323_011116/table_method_means_efficiency.csv
3. figures/realfull_20260323_011116/table_subject_best_detailed.csv
4. figures/realfull_20260323_011116/table_all_experiments_compact.csv
5. report/tables/summary_all_subjects.csv
6. report/tables/table_method_means_efficiency.csv
7. report/tables/table_subject_best_detailed.csv
8. report/tables/table_all_experiments_compact.csv
9. report/report_final.md
10. report/report_final_en.md

## 7. Integrity Checklist (Expected)

For each subject and each experiment E1..E8:
1. config.json
2. loss_history.json
3. trainable_stats.json
4. hyperspherical_energy.json
5. eval_results.json
6. baseline_images (25)
7. generated_images (25)

## 8. GitHub Submission Policy

Do upload:
1. Source code and configs.
2. Prompt files.
3. Report markdown files and final PDF.
4. Lightweight summary CSV files.

Do not upload:
1. Large raw training outputs.
2. Local virtual environments.
3. Local caches/checkpoints/log dumps.
4. Raw subject images if licensing/privacy is unclear.

The .gitignore is configured to prevent accidental upload of common large/local artifacts.

## 9. Minimal Git Commands

If this folder is not yet a git repo:

```bash
cd /data0/3DIC_workspace/miniproject/oft-dreambooth-project
git init
git add .
git commit -m "Finalize OFT mini-project reports and reproducible pipeline"
git branch -M main
git remote add origin <your_github_repo_url>
git push -u origin main
```
