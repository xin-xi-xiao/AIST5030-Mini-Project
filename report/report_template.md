# Mini-Project Report (3 Pages)

## Title
Parameter-efficient Finetuning for Pretrained Foundation Models via OFT on Stable Diffusion v1.5

## 1. Introduction (Page 1)

### 1.1 Task
We finetune a pretrained Stable Diffusion v1.5 model for subject-driven image generation using Orthogonal Finetuning (OFT).

### 1.2 Motivation
Compared with additive low-rank updates (LoRA), OFT applies orthogonal transformations to preserve geometric relations of pretrained neuron weights, potentially reducing forgetting.

### 1.3 Research Questions
- How does OFT rank `r` influence quality and parameter efficiency?
- Does COFT (`eps` constraint) improve stability?
- Can BOFT improve efficiency further?
- How do OFT-family methods compare to LoRA in quality/speed/efficiency?
- Is hyperspherical energy better preserved by OFT-family methods?

## 2. Method (Page 1)

### 2.1 Model and PEFT Methods
- Base model: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Finetuned module: UNet attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`)
- Methods: OFT, COFT, BOFT, LoRA

### 2.2 Experiment Matrix
| ID | Method | Key Params |
|---|---|---|
| E1 | OFT | r=4 |
| E2 | OFT | r=8 |
| E3 | OFT | r=2 |
| E4 | COFT | r=4, eps=1e-3 |
| E5 | OFT+Cayley | r=4, use_cayley_neumann |
| E6 | BOFT | block=4, butterfly=2 |
| E7 | LoRA | r=4 |
| E8 | LoRA | r=16 |

## 3. Experimental Setup (Page 2)

### 3.1 Data
- Subjects: dog, backpack, cat
- 4-6 reference images per subject
- Shared validation prompt set (8 prompts)

### 3.2 Hyperparameters
- Resolution: 512
- Max steps: 800
- Batch size: 1
- Grad accumulation: 4
- LR: 1e-5
- Mixed precision: bf16

### 3.3 Metrics
- CLIP-I (subject fidelity)
- CLIP-T (text alignment)
- DINO score (semantic similarity)
- Trainable parameter count
- Training loss curve
- Hyperspherical energy change (%)

## 4. Results (Page 2-3)

### 4.1 Quantitative Table
Insert `figures/summary.csv` summary as a formatted table.

### 4.2 Training Curves
Insert `figures/loss_curves.png`.

### 4.3 Qualitative Before/After
Insert paired images from:
- `outputs/E*_*/baseline_images`
- `outputs/E*_*/generated_images`

### 4.4 Hyperspherical Energy Analysis
Insert `figures/energy_comparison.png` and discuss:
- OFT-family expected lower energy drift than LoRA.
- Relation between energy preservation and fidelity.

## 5. Discussion and Conclusion (Page 3)

### 5.1 Key Findings
- [Fill from your actual results]

### 5.2 Ablation Insights
- OFT rank effect (`r=2/4/8`)
- COFT stability (`eps` constraint)
- BOFT efficiency-performance tradeoff

### 5.3 Limitations
- Small subject dataset
- Prompt coverage limitations
- Image quality metric bias

### 5.4 Conclusion
Summarize best method under quality-efficiency tradeoff and practical recommendations.

## Appendix (Optional)
- Runtime and GPU memory usage
- Additional qualitative samples
