# 项目要求-方案-实现逐条审计（最终版）

最后更新：2026-03-23 01:26

## A. 作业硬性要求

1. 使用 OFT 对预训练基础模型做参数高效微调并汇总结果。
2. 提交代码仓库（含 README）。
3. 提交 3 页报告，至少包含：
- 训练 loss 曲线
- 微调前后性能/定性结果

## B. 方案关键要求（你的实验计划）

1. 任务：Stable Diffusion v1.5 DreamBooth。
2. 方法矩阵：OFT/COFT/BOFT/LoRA 共 8 组。
3. 主体：dog/backpack/cat。
4. 指标：CLIP-I、CLIP-T、DINO、loss、可训参数、超球面能量。
5. 双 RTX 5090 并行。
6. 正式设置：真实数据 + 800 steps。

## C. 当前实现覆盖（代码与环境）

### C1 代码能力
- 统一训练：scripts/train.py（OFT/COFT/BOFT/LoRA）
- 评估：scripts/evaluate.py（CLIP-I/CLIP-T/DINO）
- 可视化：scripts/visualize.py（loss/metrics/energy/params）
- 汇总：scripts/collect_results.py（支持递归）
- 单主体双卡矩阵：scripts/run_all.sh
- 三主体编排：scripts/run_all_subjects.sh

### C2 环境能力
- 项目独立环境：.venv
- torch CUDA 构建可用，RTX 5090 双卡可见并可训练。

## D. 数据真实性核查（dog/backpack/cat）

核查结论：三类数据均为真实图片并带来源清单，非占位图。

- data/dreambooth/dog/sources.csv
- data/dreambooth/backpack/sources.csv
- data/dreambooth/cat/sources.csv

本次正式运行的数据指纹（含 sha256）已写入：

- logs/realfull_20260323_011116/run_metadata.txt

## E. 旧结果完整性核查

旧目录 outputs_real_full 不完整：
- dog：完整
- backpack：完整
- cat：缺 E5-E8，且 E1-E4 缺 eval_results.json

因此旧结果不满足“3 主体 x 8 组全量完成”的最终提交要求。

## F. 已执行的修复动作（关键）

1. 已将脚本增强为可分离日志/图表根目录，避免覆盖旧结果。
2. 已使用 nohup 在后台启动全量正式重跑（终端断开不影响）。

本次正式运行：
- RUN_ID: realfull_20260323_011116
- 输出目录：outputs_realfull_20260323_011116
- 图表目录：figures/realfull_20260323_011116
- 日志目录：logs/realfull_20260323_011116
- 总日志：logs/realfull_20260323_011116/launcher.log
- 状态文件：logs/realfull_20260323_011116/RUN_STATUS.txt

## G. 与项目要求的差距（最终时刻）

当前已无硬性缺口；剩余工作仅为版式层面的 PDF 排版与导出。

## H. 完成判据（核验结果）

1. outputs_realfull_20260323_011116/{dog,backpack,cat}/E1..E8：已全部存在。
2. 每组必需产物：已齐全（含 25 generated_images + 25 baseline_images）。
3. figures/realfull_20260323_011116：已生成主体图和 summary_all_subjects.csv。
4. report/report_final.md：已更新为本次正式结果数值版。

完整性机检结论：ALL_OK=True。

关键证据路径：
- logs/realfull_20260323_011116/launcher.log
- logs/realfull_20260323_011116/run_metadata.txt
- figures/realfull_20260323_011116/summary_all_subjects.csv
- figures/realfull_20260323_011116/summary_method_means.csv
- figures/realfull_20260323_011116/summary_subject_best.csv
