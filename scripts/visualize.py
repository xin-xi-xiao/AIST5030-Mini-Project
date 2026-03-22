#!/usr/bin/env python3
"""Visualization utilities for report figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["figure.figsize"] = (10, 6)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def smooth_curve(y: List[float], window: int = 20):
    if len(y) < window:
        return np.array(y), np.arange(len(y))
    kernel = np.ones(window) / window
    ys = np.convolve(np.array(y), kernel, mode="valid")
    xs = np.arange(window - 1, window - 1 + len(ys))
    return ys, xs


def plot_loss_curves(exp_dirs: List[Path], labels: List[str], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for d, label in zip(exp_dirs, labels):
        p = d / "loss_history.json"
        if not p.exists():
            continue
        data = load_json(p)
        losses = [row["loss"] for row in data]
        ys, xs = smooth_curve(losses, window=20)
        plt.plot(xs, ys, label=label, linewidth=2)

    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curves: OFT/BOFT/COFT vs LoRA")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics(exp_dirs: List[Path], labels: List[str], out_path: Path) -> None:
    keys = ["clip_i", "clip_t", "dino"]
    vals: Dict[str, List[float]] = {k: [] for k in keys}
    valid_labels: List[str] = []

    for d, label in zip(exp_dirs, labels):
        p = d / "eval_results.json"
        if not p.exists():
            continue
        data = load_json(p)
        for k in keys:
            vals[k].append(float(data.get(k, 0.0)))
        valid_labels.append(label)

    if not valid_labels:
        return

    x = np.arange(len(valid_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, vals["clip_i"], width, label="CLIP-I", color="#1f77b4")
    ax.bar(x, vals["clip_t"], width, label="CLIP-T", color="#2ca02c")
    ax.bar(x + width, vals["dino"], width, label="DINO", color="#ff7f0e")

    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_title("Quantitative Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_energy(exp_dirs: List[Path], labels: List[str], out_path: Path) -> None:
    def match_after_value(before_key: str, after: Dict[str, float]):
        candidates = [
            before_key,
            f"base_model.model.{before_key}",
        ]
        if before_key.endswith(".weight"):
            candidates.append(f"base_model.model.{before_key[:-7]}.base_layer.weight")

        for c in candidates:
            if c in after:
                return after[c]

        # Fallback: suffix matching for wrapped names.
        for k, v in after.items():
            if k.endswith(before_key) or k.endswith(before_key.replace(".weight", ".base_layer.weight")):
                return v
        return None

    changes = []
    valid = []

    for d, label in zip(exp_dirs, labels):
        p = d / "hyperspherical_energy.json"
        if not p.exists():
            continue
        data = load_json(p)
        before = data.get("before", {})
        after = data.get("after", {})
        per_layer = []
        for k, b in before.items():
            a = match_after_value(k, after)
            if a is None:
                continue
            if b <= 1e-3:
                continue
            change = abs(a - b) / b * 100.0
            if np.isfinite(change):
                # Robust clip to avoid pathological layers dominating visualization.
                per_layer.append(float(min(change, 500.0)))
        if per_layer:
            changes.append(float(np.mean(per_layer)))
            valid.append(label)

    if not valid:
        return

    colors = ["#1f77b4" if any(tag in l for tag in ["OFT", "COFT", "BOFT"]) else "#d62728" for l in valid]
    plt.figure(figsize=(10, 6))
    plt.bar(valid, changes, color=colors)
    plt.xlabel("Method")
    plt.ylabel("Avg. Energy Change (%)")
    plt.title("Hyperspherical Energy Change")
    plt.grid(axis="y", alpha=0.3)
    plt.subplots_adjust(bottom=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_trainable_params(exp_dirs: List[Path], labels: List[str], out_path: Path) -> None:
    vals = []
    valid = []
    for d, label in zip(exp_dirs, labels):
        p = d / "trainable_stats.json"
        if not p.exists():
            continue
        data = load_json(p)
        vals.append(int(data.get("trainable_params", 0)))
        valid.append(label)

    if not valid:
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid, vals, color="#4c72b0")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Trainable Parameters")
    plt.xlabel("Method")
    plt.title("Trainable Parameter Comparison")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--figures_root", type=str, default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.outputs_root)

    exp_dirs = [
        root / "E1_oft_r4",
        root / "E2_oft_r8",
        root / "E3_oft_r2",
        root / "E4_coft_r4",
        root / "E5_oft_cayley",
        root / "E6_boft_b4",
        root / "E7_lora_r4",
        root / "E8_lora_r16",
    ]
    labels = [
        "OFT r=4",
        "OFT r=8",
        "OFT r=2",
        "COFT r=4",
        "OFT+Cayley",
        "BOFT b=4",
        "LoRA r=4",
        "LoRA r=16",
    ]

    # Fallback for smoke/custom runs if standard E1-E8 matrix is not present.
    if not any((d / "loss_history.json").exists() or (d / "eval_results.json").exists() for d in exp_dirs):
        exp_dirs = [d for d in sorted(root.glob("*")) if d.is_dir()]
        exp_dirs = [d for d in exp_dirs if (d / "loss_history.json").exists() or (d / "eval_results.json").exists()]
        labels = [d.name for d in exp_dirs]

    if not exp_dirs:
        print("No experiment directories with results found.")
        return

    fig_root = Path(args.figures_root)
    plot_loss_curves(exp_dirs, labels, fig_root / "loss_curves.png")
    plot_metrics(exp_dirs, labels, fig_root / "metrics_comparison.png")
    plot_energy(exp_dirs, labels, fig_root / "energy_comparison.png")
    plot_trainable_params(exp_dirs, labels, fig_root / "trainable_params.png")


if __name__ == "__main__":
    main()
