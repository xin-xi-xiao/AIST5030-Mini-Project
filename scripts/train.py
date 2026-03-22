#!/usr/bin/env python3
"""Unified OFT/COFT/BOFT/LoRA DreamBooth training script.

This script trains Stable Diffusion v1.5 UNet adapters using PEFT and writes:
- adapter weights
- loss history
- run config
- trainable parameter stats
- optional hyperspherical energy measurements
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import BOFTConfig, LoraConfig, OFTConfig, PeftModel, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger("oft_train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@dataclass
class RunStats:
    total_params: int
    trainable_params: int
    trainable_ratio: float


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_dir: str,
        instance_prompt: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 512,
        repeats: int = 100,
    ) -> None:
        self.instance_data_dir = Path(instance_data_dir)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.repeats = repeats

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.instance_images = [
            p for p in sorted(self.instance_data_dir.iterdir()) if p.suffix.lower() in exts
        ]
        if not self.instance_images:
            raise ValueError(f"No images found in {self.instance_data_dir}")

        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.instance_images) * self.repeats

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.instance_images[idx % len(self.instance_images)]
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        tokens = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
        }


def build_peft_config(args: argparse.Namespace):
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    if args.method == "oft":
        return OFTConfig(
            r=args.r,
            oft_block_size=args.oft_block_size,
            target_modules=target_modules,
            bias="none",
            module_dropout=args.module_dropout,
            use_cayley_neumann=args.use_cayley_neumann,
        )
    if args.method == "coft":
        return OFTConfig(
            r=args.r,
            oft_block_size=args.oft_block_size,
            target_modules=target_modules,
            bias="none",
            module_dropout=args.module_dropout,
            use_cayley_neumann=args.use_cayley_neumann,
            coft=True,
            eps=args.eps,
        )
    if args.method == "boft":
        return BOFTConfig(
            boft_block_size=args.boft_block_size,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=target_modules,
            boft_dropout=args.boft_dropout,
            bias="boft_only",
        )
    if args.method == "lora":
        return LoraConfig(
            r=args.r,
            lora_alpha=max(8, args.r * 2),
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )

    raise ValueError(f"Unsupported method: {args.method}")


def count_trainable_params(model: torch.nn.Module) -> RunStats:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / max(1, total_params)
    return RunStats(
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_ratio=ratio,
    )


def compute_hyperspherical_energy(
    weight_matrix: torch.Tensor,
    s: float = 2.0,
    max_rows: int = 256,
) -> float:
    w = weight_matrix.float()
    # Cap neuron count for stable runtime; keeps trend comparisons while avoiding O(n^2) blowups.
    if max_rows > 0 and w.size(0) > max_rows:
        idx = torch.linspace(0, w.size(0) - 1, steps=max_rows, device=w.device).long()
        w = w.index_select(0, idx)
    norms = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w_hat = w / norms
    cos = w_hat @ w_hat.T
    dist_sq = (2.0 - 2.0 * cos).clamp(min=1e-12)
    n = w_hat.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=w_hat.device)
    dist_sq_off = dist_sq[mask]
    energy = (dist_sq_off ** (-s / 2.0)).sum().item()
    return energy


def measure_all_hyperspherical_energy(
    model: torch.nn.Module,
    s: float = 2.0,
    max_layers: int = 20,
    max_rows: int = 256,
) -> Dict[str, float]:
    energies: Dict[str, float] = {}
    target_keys = ("to_q", "to_k", "to_v", "to_out.0")
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        if not any(k in name for k in target_keys):
            continue
        if param.dim() != 2 or param.shape[0] < 4:
            continue
        try:
            energies[name] = compute_hyperspherical_energy(
                param.detach(),
                s=s,
                max_rows=max_rows,
            )
            if max_layers > 0 and len(energies) >= max_layers:
                break
        except Exception:
            continue
    return energies


def default_val_prompts(subject_class: str) -> List[str]:
    return [
        f"a photo of sks {subject_class}",
        f"a photo of sks {subject_class} on the beach",
        f"a photo of sks {subject_class} in the snow",
        f"a painting of sks {subject_class} in the style of Van Gogh",
        f"a photo of sks {subject_class} wearing sunglasses",
        f"a photo of sks {subject_class} in a garden",
        f"a photo of sks {subject_class} with the Eiffel Tower",
        f"a watercolor painting of sks {subject_class}",
    ]


def load_val_prompts(subject_class: str, prompt_file: str = "") -> List[str]:
    if not prompt_file:
        return default_val_prompts(subject_class)

    p = Path(prompt_file)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    prompts: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        prompts.append(text.replace("{subject}", subject_class))

    if not prompts:
        raise ValueError(f"No valid prompts loaded from {prompt_file}")
    return prompts


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train(args: argparse.Namespace) -> Tuple[List[Dict[str, float]], RunStats]:
    set_seed(args.seed)

    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    energy_before: Dict[str, float] = {}
    if args.measure_energy:
        logger.info("Measuring pre-finetuning hyperspherical energy.")
        energy_before = measure_all_hyperspherical_energy(
            unet,
            max_layers=args.max_energy_layers,
            max_rows=args.max_energy_neurons,
        )

    peft_cfg = build_peft_config(args)
    unet = get_peft_model(unet, peft_cfg)
    stats = count_trainable_params(unet)

    dataset = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        resolution=args.resolution,
        repeats=args.dataset_repeats,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    if args.cpu:
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    loss_history: List[Dict[str, float]] = []
    global_step = 0

    logger.info(
        "Start training: method=%s, steps=%d, trainable=%d (%.4f%%)",
        args.method,
        args.max_train_steps,
        stats.trainable_params,
        stats.trainable_ratio * 100,
    )

    unet.train()
    for _epoch in range(math.ceil(args.max_train_steps / max(1, len(dataloader)))):
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0].to(weight_dtype)

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.detach().item())
            loss_history.append({"step": global_step, "loss": loss_value})

            if global_step % args.log_every == 0:
                logger.info("step %d/%d | loss=%.6f", global_step, args.max_train_steps, loss_value)

            global_step += 1

    accelerator.wait_for_everyone()

    if args.measure_energy and accelerator.is_main_process:
        logger.info("Measuring post-finetuning hyperspherical energy.")
        # unwrap so we can read parameters reliably
        unwrapped = accelerator.unwrap_model(unet)
        energy_after = measure_all_hyperspherical_energy(
            unwrapped,
            max_layers=args.max_energy_layers,
            max_rows=args.max_energy_neurons,
        )
        save_json(
            Path(args.output_dir) / "hyperspherical_energy.json",
            {
                "before": energy_before,
                "after": energy_after,
            },
        )

    if accelerator.is_main_process:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(out_dir / "unet_peft")
        save_json(out_dir / "loss_history.json", loss_history)
        save_json(out_dir / "trainable_stats.json", asdict(stats))

    return loss_history, stats


def generate_images(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated_images"
    base_dir = out_dir / "baseline_images"
    gen_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_val_prompts(args.subject_class, args.val_prompt_file)

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    pipe.unet = PeftModel.from_pretrained(pipe.unet, out_dir / "unet_peft")
    pipe.unet = pipe.unet.merge_and_unload()

    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device).manual_seed(args.seed)
        img = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        ).images[0]
        img.save(gen_dir / f"gen_{i:03d}.png")

    baseline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device).manual_seed(args.seed)
        img = baseline(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        ).images[0]
        img.save(base_dir / f"baseline_{i:03d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type=str, default="")
    parser.add_argument("--instance_prompt", type=str, default="")
    parser.add_argument("--subject_class", type=str, default="dog")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dataset_repeats", type=int, default=100)

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training/inference")

    parser.add_argument("--method", type=str, default="oft", choices=["oft", "coft", "boft", "lora"])
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--oft_block_size", type=int, default=0)
    parser.add_argument("--use_cayley_neumann", action="store_true")
    parser.add_argument("--module_dropout", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-3)

    parser.add_argument("--boft_block_size", type=int, default=4)
    parser.add_argument("--boft_n_butterfly_factor", type=int, default=2)
    parser.add_argument("--boft_dropout", type=float, default=0.05)

    parser.add_argument("--measure_energy", action="store_true")
    parser.add_argument("--max_energy_layers", type=int, default=20)
    parser.add_argument("--max_energy_neurons", type=int, default=256)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--skip_generate", action="store_true")

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--val_prompt_file", type=str, default="", help="Optional prompt file; supports {subject} placeholder")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (args.generate_only and (out_dir / "config.json").exists()):
        save_json(out_dir / "config.json", vars(args))

    if args.generate_only:
        generate_images(args)
        return

    if not args.instance_data_dir or not args.instance_prompt:
        raise ValueError("--instance_data_dir and --instance_prompt are required for training.")

    _, stats = train(args)

    if not args.skip_generate:
        generate_images(args)

    logger.info(
        "Finished. trainable=%d / total=%d (%.4f%%)",
        stats.trainable_params,
        stats.total_params,
        stats.trainable_ratio * 100,
    )


if __name__ == "__main__":
    main()
