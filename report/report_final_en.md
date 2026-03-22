# Mini-Project Report (Final, English)

## Parameter-efficient Finetuning for Pretrained Foundation Models with OFT

### 1. Requirement Alignment

The assignment requires:
1. Finetune a pretrained model using OFT on a downstream task.
2. Submit a GitHub repository with a clear README.
3. Submit a 3-page report including training loss curves and final performance and/or qualitative before-vs-after results.

This project fully aligns with these requirements and with the proposed experiment plan:
1. Base model: Stable Diffusion v1.5.
2. Task: subject-driven generation (DreamBooth style).
3. Methods: OFT/COFT/BOFT/LoRA (8-run matrix).
4. Subjects: dog/backpack/cat.
5. Metrics: CLIP-I, CLIP-T, DINO, trainable parameters, hyperspherical energy change.

### 2. Experimental Setup

1. Hardware: 2 x RTX 5090 (parallel execution inside each subject run matrix).
2. Training budget: 800 steps per experiment.
3. Validation prompts: 25 prompts, shared across all methods.
4. Experiment IDs:
   - E1 OFT r=4
   - E2 OFT r=8
   - E3 OFT r=2
   - E4 COFT r=4 eps=1e-3
   - E5 OFT + Cayley-Neumann
   - E6 BOFT block=4 butterfly=2
   - E7 LoRA r=4
   - E8 LoRA r=16

Official run artifacts:
1. outputs_realfull_20260323_011116
2. figures/realfull_20260323_011116
3. logs/realfull_20260323_011116

### 3. Full Integrity Verification (3 Subjects x 8 Runs)

Verification scope:
1. 3 subjects x 8 experiments = 24 experiment directories.
2. Required files per run:
   - config.json
   - loss_history.json
   - trainable_stats.json
   - hyperspherical_energy.json
   - eval_results.json
3. Required image counts per run:
   - generated_images: 25
   - baseline_images: 25

Machine-check results:
1. TOTAL_GROUPS_EXPECTED = 24
2. TOTAL_GROUPS_FOUND = 24
3. ISSUE_COUNT = 0
4. summary_all_subjects.csv rows = 24

Conclusion: all 24 runs are complete and submission-ready.

### 4. Final Tables and Consolidated Results

Main summary files:
1. figures/realfull_20260323_011116/summary_all_subjects.csv
2. figures/realfull_20260323_011116/summary_dog.csv
3. figures/realfull_20260323_011116/summary_backpack.csv
4. figures/realfull_20260323_011116/summary_cat.csv

Added report-ready key tables:
1. figures/realfull_20260323_011116/table_method_means_efficiency.csv
2. figures/realfull_20260323_011116/table_subject_best_detailed.csv
3. figures/realfull_20260323_011116/table_all_experiments_compact.csv

GitHub-friendly lightweight mirrors:
1. report/tables/summary_all_subjects.csv
2. report/tables/table_method_means_efficiency.csv
3. report/tables/table_subject_best_detailed.csv
4. report/tables/table_all_experiments_compact.csv

Method-level mean results (across all 3 subjects):

1. OFT
   - clip_i: 0.735508
   - clip_t: 0.314046
   - dino: 0.255829
   - trainable_params: 13058880
   - trainable_ratio: 0.014914
   - energy_change_pct: 1.183572e-06

2. COFT
   - clip_i: 0.713505
   - clip_t: 0.316171
   - dino: 0.196210
   - trainable_params: 11602368
   - trainable_ratio: 0.013319
   - energy_change_pct: 1.186924e-06

3. BOFT
   - clip_i: 0.712165
   - clip_t: 0.317216
   - dino: 0.197124
   - trainable_params: 522624
   - trainable_ratio: 0.000608
   - clip_i_per_million_params: 1.362671 (best parameter-efficiency)

4. LoRA
   - clip_i: 0.711544
   - clip_t: 0.317414
   - dino: 0.207301
   - trainable_params: 1992960
   - trainable_ratio: 0.002311
   - energy_change_pct: 1.695606e-06

Subject-wise best runs:

1. backpack
   - best clip_i: E3_oft_r2, clip_i=0.774560, dino=0.408406
   - best dino: E3_oft_r2, dino=0.408406

2. dog
   - best clip_i: E3_oft_r2, clip_i=0.741112, dino=0.201619
   - best dino: E3_oft_r2, dino=0.201619

3. cat
   - best clip_i: E2_oft_r8, clip_i=0.730467, dino=0.191642
   - best dino: E3_oft_r2, dino=0.221627

### 5. Key Figures for the 3-page Report

Required figures to include:
1. Training loss curves (one per subject):
   - figures/realfull_20260323_011116/dog/loss_curves.png
   - figures/realfull_20260323_011116/backpack/loss_curves.png
   - figures/realfull_20260323_011116/cat/loss_curves.png
2. Metrics comparison (one per subject):
   - figures/realfull_20260323_011116/dog/metrics_comparison.png
   - figures/realfull_20260323_011116/backpack/metrics_comparison.png
   - figures/realfull_20260323_011116/cat/metrics_comparison.png
3. Qualitative before/after samples:
   - outputs_realfull_20260323_011116/<subject>/<experiment>/baseline_images
   - outputs_realfull_20260323_011116/<subject>/<experiment>/generated_images

Optional but recommended:
1. trainable_params.png for parameter-efficiency discussion.
2. energy_comparison.png for geometric stability discussion.

### 6. Discussion

1. Task quality:
   OFT-family methods outperform LoRA/COFT/BOFT on mean CLIP-I and mean DINO under this setup. In particular, E3_oft_r2 dominates dog and backpack on both identity and semantic metrics.

2. Parameter efficiency:
   BOFT is the most compact method by trainable parameters (~0.52M), only around 4% of OFT mean trainable parameters, while preserving competitive CLIP-T.

3. Geometric behavior:
   Energy change remains at ~1e-06 for all methods; OFT/COFT/BOFT are generally lower than LoRA, consistent with stronger orthogonality-preserving behavior.

4. Multi-metric trade-off:
   For cat, best CLIP-I (E2) and best DINO (E3) do not coincide, showing that identity fidelity and semantic consistency are separable objectives. Conclusions should therefore be multi-metric, not single-metric.

5. Validity boundary:
   Claims are validated on 3 subjects, fixed 25 prompts, and 800-step training. Expanding subject diversity or step budget may alter rankings.

### 7. Final Compliance Statement

All assignment and plan requirements are satisfied:
1. OFT-based downstream finetuning completed.
2. Full matrix completed with verified completeness (24/24).
3. Loss curves and before/after qualitative outputs generated.
4. Quantitative tables and key visualizations consolidated.
5. Report content is ready for 3-page final PDF formatting.
