# Mini-Project Report (Final, Chinese)

## Parameter-efficient Finetuning for Pretrained Foundation Models with OFT

### 1. 任务要求与方案对齐

项目 PDF 的硬性要求为：
1. 使用 OFT 微调一个预训练基础模型并完成下游任务。
2. 提交 GitHub 仓库（含 README）。
3. 提交 3 页报告，包含训练 loss 曲线，以及微调前后性能或定性结果。

本项目方案（Stable Diffusion v1.5 + DreamBooth + OFT 系列对比）与上述要求完全一致，并增加了系统性消融：
1. 8 组方法矩阵（OFT/COFT/BOFT/LoRA）。
2. 3 个主体（dog/backpack/cat）。
3. 统一 25 条验证 prompt。
4. 定量指标（CLIP-I/CLIP-T/DINO）、参数效率、超球面能量分析。

### 2. 实验设置

1. Base model: Stable Diffusion v1.5。
2. 硬件：2 x RTX 5090，主体内双卡并行。
3. 训练预算：每组 800 steps。
4. 实验矩阵：
  - E1 OFT r=4
  - E2 OFT r=8
  - E3 OFT r=2
  - E4 COFT r=4 eps=1e-3
  - E5 OFT + Cayley-Neumann
  - E6 BOFT block=4 butterfly=2
  - E7 LoRA r=4
  - E8 LoRA r=16

正式批次路径：
1. outputs_realfull_20260323_011116
2. figures/realfull_20260323_011116
3. logs/realfull_20260323_011116

### 3. 全量完整性核验（3 主体 x 8 组）

核验范围：dog/backpack/cat x E1..E8，共 24 组。

每组核验项：
1. config.json
2. loss_history.json
3. trainable_stats.json
4. hyperspherical_energy.json
5. eval_results.json
6. generated_images（应为 25）
7. baseline_images（应为 25）

核验结果：
1. TOTAL_GROUPS_EXPECTED = 24
2. TOTAL_GROUPS_FOUND = 24
3. ISSUE_COUNT = 0
4. summary_all_subjects.csv 行数 = 24

结论：全量产物完整，满足可提交标准。

### 4. 总表与关键表格收口

主汇总表：
1. figures/realfull_20260323_011116/summary_all_subjects.csv

主体汇总表：
1. figures/realfull_20260323_011116/summary_dog.csv
2. figures/realfull_20260323_011116/summary_backpack.csv
3. figures/realfull_20260323_011116/summary_cat.csv

新增关键表（用于报告插表）：
1. figures/realfull_20260323_011116/table_method_means_efficiency.csv
2. figures/realfull_20260323_011116/table_subject_best_detailed.csv
3. figures/realfull_20260323_011116/table_all_experiments_compact.csv

为 GitHub 提交准备的轻量副本：
1. report/tables/summary_all_subjects.csv
2. report/tables/table_method_means_efficiency.csv
3. report/tables/table_subject_best_detailed.csv
4. report/tables/table_all_experiments_compact.csv

方法级均值（跨 3 主体）：

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
  - energy_change_pct: 1.173516e-06
  - clip_i_per_million_params: 1.362671（参数效率最高）

4. LoRA
  - clip_i: 0.711544
  - clip_t: 0.317414
  - dino: 0.207301
  - trainable_params: 1992960
  - trainable_ratio: 0.002311
  - energy_change_pct: 1.695606e-06

主体级最优：

1. backpack
  - best clip_i: E3_oft_r2, clip_i=0.774560, dino=0.408406
  - best dino: E3_oft_r2, dino=0.408406

2. dog
  - best clip_i: E3_oft_r2, clip_i=0.741112, dino=0.201619
  - best dino: E3_oft_r2, dino=0.201619

3. cat
  - best clip_i: E2_oft_r8, clip_i=0.730467, dino=0.191642
  - best dino: E3_oft_r2, dino=0.221627

### 5. 关键图表（报告必选）

建议主文插图（满足“loss + before/after + final performance”）：

1. loss 曲线（每主体 1 张）
  - figures/realfull_20260323_011116/dog/loss_curves.png
  - figures/realfull_20260323_011116/backpack/loss_curves.png
  - figures/realfull_20260323_011116/cat/loss_curves.png

2. 指标对比（每主体 1 张）
  - figures/realfull_20260323_011116/dog/metrics_comparison.png
  - figures/realfull_20260323_011116/backpack/metrics_comparison.png
  - figures/realfull_20260323_011116/cat/metrics_comparison.png

3. 参数与能量（可二选一或合并）
  - figures/realfull_20260323_011116/*/trainable_params.png
  - figures/realfull_20260323_011116/*/energy_comparison.png

4. 定性前后对比（每主体 1 组图网格）
  - outputs_realfull_20260323_011116/<subject>/<experiment>/baseline_images
  - outputs_realfull_20260323_011116/<subject>/<experiment>/generated_images

### 6. 分析与结论（严谨版）

1. 任务表现：
  OFT 方法族在 clip_i 与 dino 的方法均值上领先；其中 E3_oft_r2 在 dog/backpack 上同时达到保真与语义最优。

2. 参数效率：
  BOFT 的参数量最低（约 52 万），仅为 OFT 均值参数量的约 4%，且保持了接近的 clip_t，体现了高压缩优势。

3. 几何保持：
  各方法 energy_change_pct 均在 1e-06 量级；OFT/COFT/BOFT 相比 LoRA 更低，支持“正交约束降低几何扰动”的观察。

4. 指标分离现象：
  cat 主体出现 clip_i 最优（E2）与 dino 最优（E3）不一致，说明主体保真与语义一致性并非完全同一目标。报告结论必须联合多指标，不能单看 clip_i。

5. 风险边界：
  结果来自 3 个主体、固定 25 prompts、固定 800 steps；结论在该设置下成立。若扩展到更多主体/更长训练预算，可能出现排序变化。

### 7. 最终提交合规性结论

对照项目要求与个人方案：
1. OFT 微调任务：完成。
2. 全矩阵实验（3 主体 x 8 组）：完成且核验无缺失。
3. 训练 loss 曲线与前后定性结果：已生成。
4. 定量结果与讨论：已完成并给出关键表格。
5. 报告可提交：满足条件（后续仅需版式压缩至 3 页 PDF）。

## Appendix A: 后台稳定复跑命令

```bash
cd /data0/3DIC_workspace/miniproject/oft-dreambooth-project
source .venv/bin/activate

nohup env \
PYTHON_BIN=./.venv/bin/python \
MODEL=stable-diffusion-v1-5/stable-diffusion-v1-5 \
STEPS=800 \
PROMPT_FILE=prompts/val_prompts_25.txt \
OUTPUTS_BASE=outputs_realfull_20260323_011116 \
FIGURES_ROOT=figures/realfull_20260323_011116 \
LOGS_BASE=logs/realfull_20260323_011116 \
bash scripts/run_all_subjects.sh > logs/realfull_20260323_011116/launcher.log 2>&1 < /dev/null &
```
