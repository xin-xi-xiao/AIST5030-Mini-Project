# Mini-Project Report Draft

## Parameter-efficient Finetuning for Pretrained Foundation Models via OFT

### 1. Introduction
This project applies Orthogonal Finetuning (OFT) to a pretrained diffusion model for a subject-driven generation downstream task (DreamBooth style). The goal is to compare OFT-family methods against LoRA under parameter efficiency, convergence behavior, and representation-preservation analysis.

### 2. Method
Base model: Stable Diffusion v1.5 (main target), with smoke validation on hf-internal-testing/tiny-stable-diffusion-pipe.

Target modules: UNet attention projections (to_q, to_k, to_v, to_out.0).

Methods implemented in one training framework:
- OFT
- COFT
- BOFT
- LoRA baseline

The project also includes hyperspherical energy analysis before/after finetuning to test geometry-preservation claims of orthogonal updates.

### 3. Experimental Setup
Data plan:
- Subjects: dog, backpack, cat
- 4-6 reference images per subject

Main matrix (8 runs):
- E1 OFT r=4
- E2 OFT r=8
- E3 OFT r=2
- E4 COFT r=4 eps=1e-3
- E5 OFT r=4 + Cayley-Neumann
- E6 BOFT block=4 butterfly=2
- E7 LoRA r=4
- E8 LoRA r=16

Metrics:
- CLIP-I
- CLIP-T
- DINO
- Trainable parameter count
- Training loss
- Hyperspherical energy change

### 4. Current Executed Results (Smoke Validation)
To ensure code correctness and runtime stability, we executed smoke runs and full evaluation on generated images.

| Experiment | Method | r | Trainable Params | CLIP-I | CLIP-T | DINO |
|---|---:|---:|---:|---:|---:|---:|
| smoke_oft | oft | 2 | 38,592 | 0.7894 | 0.1915 | 0.1211 |
| smoke_lora | lora | 4 | 23,040 | 0.7894 | 0.1915 | 0.1211 |
| smoke_oft_gpu | oft | 2 | 38,592 | 0.7770 | 0.1805 | 0.1248 |

Notes:
- These two smoke runs use a tiny diffusion checkpoint and 2 training steps for pipeline validation.
- Full 8-run SD v1.5 experiments are fully configured and can be launched with the provided scripts.

### 5. Qualitative Outputs
Generated before/after images are available under:
- outputs/smoke_oft/baseline_images
- outputs/smoke_oft/generated_images
- outputs/smoke_lora/baseline_images
- outputs/smoke_lora/generated_images

### 6. Figures
Auto-generated figures:
- figures/loss_curves.png
- figures/metrics_comparison.png
- figures/summary.csv

### 7. Engineering Notes
The environment was upgraded to nightly PyTorch CUDA wheels and validated to support RTX 5090 architecture (sm_120), enabling direct GPU experiments for the full matrix.

### 8. Next Step for Final Submission
Run the full 8-run matrix for each target subject and update this report with:
- full metrics table,
- energy comparison figure,
- OFT/COFT/BOFT vs LoRA qualitative grid,
- final conclusions.
