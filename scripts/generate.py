#!/usr/bin/env python3
"""Generate images from a trained adapter without retraining."""

from __future__ import annotations

import argparse

from train import generate_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--subject_class", type=str, default="dog")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
