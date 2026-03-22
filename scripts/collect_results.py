#!/usr/bin/env python3
"""Collect all experiment metrics into a single table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _match_after_value(before_key: str, after: Dict[str, float]):
    candidates = [
        before_key,
        f"base_model.model.{before_key}",
    ]
    if before_key.endswith(".weight"):
        candidates.append(f"base_model.model.{before_key[:-7]}.base_layer.weight")

    for c in candidates:
        if c in after:
            return after[c]

    for k, v in after.items():
        if k.endswith(before_key) or k.endswith(before_key.replace(".weight", ".base_layer.weight")):
            return v
    return None


def _energy_change_pct(before: Dict[str, float], after: Dict[str, float]):
    changes = []
    for k, b in before.items():
        a = _match_after_value(k, after)
        if a is None or not b or b <= 1e-3:
            continue
        change = abs(a - b) / b * 100
        if change == change:
            changes.append(min(change, 500.0))
    if not changes:
        return None
    return sum(changes) / len(changes)


def summarize(outputs_root: Path) -> List[Dict]:
    rows = []
    exp_dirs = [d for d in sorted(outputs_root.glob("*")) if d.is_dir()]
    exp_dirs = [d for d in exp_dirs if (d / "config.json").exists() or (d / "eval_results.json").exists()]

    for d in exp_dirs:
        row = {"experiment": d.name}

        cfg = d / "config.json"
        if cfg.exists():
            c = load_json(cfg)
            row["method"] = c.get("method")
            row["r"] = c.get("r")

        stats = d / "trainable_stats.json"
        if stats.exists():
            s = load_json(stats)
            row["trainable_params"] = s.get("trainable_params")
            row["trainable_ratio"] = s.get("trainable_ratio")

        evalp = d / "eval_results.json"
        if evalp.exists():
            e = load_json(evalp)
            row["clip_i"] = e.get("clip_i")
            row["clip_t"] = e.get("clip_t")
            row["dino"] = e.get("dino")

        energy = d / "hyperspherical_energy.json"
        if energy.exists():
            data = load_json(energy)
            before = data.get("before", {})
            after = data.get("after", {})
            e = _energy_change_pct(before, after)
            if e is not None:
                row["energy_change_pct"] = e

        rows.append(row)

    return rows


def summarize_recursive(outputs_root: Path) -> List[Dict]:
    rows = []
    for cfg in sorted(outputs_root.glob("**/config.json")):
        d = cfg.parent
        row = {"experiment": d.name, "relative_dir": str(d.relative_to(outputs_root))}

        c = load_json(cfg)
        row["method"] = c.get("method")
        row["r"] = c.get("r")

        stats = d / "trainable_stats.json"
        if stats.exists():
            s = load_json(stats)
            row["trainable_params"] = s.get("trainable_params")
            row["trainable_ratio"] = s.get("trainable_ratio")

        evalp = d / "eval_results.json"
        if evalp.exists():
            e = load_json(evalp)
            row["clip_i"] = e.get("clip_i")
            row["clip_t"] = e.get("clip_t")
            row["dino"] = e.get("dino")

        energy = d / "hyperspherical_energy.json"
        if energy.exists():
            data = load_json(energy)
            before = data.get("before", {})
            after = data.get("after", {})
            e = _energy_change_pct(before, after)
            if e is not None:
                row["energy_change_pct"] = e

        rows.append(row)

    return rows


def save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_root", type=str, default="outputs")
    p.add_argument("--save_csv", type=str, default="figures/summary.csv")
    p.add_argument("--recursive", action="store_true", help="Collect from nested output directories")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.recursive:
        rows = summarize_recursive(Path(args.outputs_root))
    else:
        rows = summarize(Path(args.outputs_root))
    if not rows:
        print("No experiment directories found.")
        return
    save_csv(rows, Path(args.save_csv))
    print(f"Saved summary CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
