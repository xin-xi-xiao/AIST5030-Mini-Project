#!/usr/bin/env python3
"""Quantitative evaluation for DreamBooth outputs.

Metrics:
- CLIP-I (image-image similarity)
- CLIP-T (text-image alignment)
- DINO score (semantic similarity)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms


def load_images(image_dir: Path, patterns: List[str] | None = None) -> List[Image.Image]:
    if patterns is None:
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    paths = []
    for pat in patterns:
        paths.extend(sorted(image_dir.glob(pat)))
    return [Image.open(p).convert("RGB") for p in paths]


class CLIPEvaluator:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        feats = self.model.encode_image(x)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        t = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(t)
        return feats / feats.norm(dim=-1, keepdim=True)

    def clip_i(self, generated: List[Image.Image], refs: List[Image.Image]) -> float:
        gen = self.encode_images(generated)
        ref = self.encode_images(refs)
        return float((gen @ ref.T).mean().item())

    def clip_t(self, generated: List[Image.Image], prompts: List[str]) -> float:
        gen = self.encode_images(generated)
        txt = self.encode_text(prompts)
        n = min(gen.shape[0], txt.shape[0])
        return float((gen[:n] * txt[:n]).sum(dim=-1).mean().item())


class DINOEvaluator:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.to(device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)
        feats = self.model(x)
        return feats / feats.norm(dim=-1, keepdim=True)

    def score(self, generated: List[Image.Image], refs: List[Image.Image]) -> float:
        gen = self.encode_images(generated)
        ref = self.encode_images(refs)
        return float((gen @ ref.T).mean().item())


def default_prompts(subject: str) -> List[str]:
    return [
        f"a photo of sks {subject}",
        f"a photo of sks {subject} on the beach",
        f"a photo of sks {subject} in the snow",
        f"a painting of sks {subject} in the style of Van Gogh",
        f"a photo of sks {subject} wearing sunglasses",
        f"a photo of sks {subject} in a garden",
        f"a photo of sks {subject} with the Eiffel Tower",
        f"a watercolor painting of sks {subject}",
    ]


def load_prompts(subject: str, prompt_file: str = "") -> List[str]:
    if not prompt_file:
        return default_prompts(subject)

    p = Path(prompt_file)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    prompts: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        prompts.append(text.replace("{subject}", subject))

    if not prompts:
        raise ValueError(f"No valid prompts loaded from {prompt_file}")
    return prompts


def evaluate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    gen_images = load_images(output_dir / "generated_images")
    ref_images = load_images(Path(args.instance_data_dir))

    if not gen_images:
        raise ValueError(f"No generated images found in {output_dir / 'generated_images'}")
    if not ref_images:
        raise ValueError(f"No reference images found in {args.instance_data_dir}")

    prompts = load_prompts(args.subject_class, args.prompt_file)

    clip_eval = CLIPEvaluator(device=args.device)
    dino_eval = DINOEvaluator(device=args.device)

    results = {
        "clip_i": clip_eval.clip_i(gen_images, ref_images),
        "clip_t": clip_eval.clip_t(gen_images, prompts[: len(gen_images)]),
        "dino": dino_eval.score(gen_images, ref_images),
    }

    cfg_path = output_dir / "config.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        results["method"] = cfg.get("method", "unknown")
        results["r"] = cfg.get("r", "N/A")

    stats_path = output_dir / "trainable_stats.json"
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        results["trainable_params"] = stats.get("trainable_params")

    with (output_dir / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--subject_class", type=str, default="dog")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--prompt_file", type=str, default="", help="Optional prompt file matching generation prompts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate(args)
    print("=" * 60)
    print(f"Evaluation Results: {args.output_dir}")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:>20}: {v:.4f}")
        else:
            print(f"{k:>20}: {v}")


if __name__ == "__main__":
    main()
