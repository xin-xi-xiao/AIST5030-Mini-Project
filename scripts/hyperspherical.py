#!/usr/bin/env python3
"""Standalone hyperspherical energy analysis for one experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    p = Path(args.output_dir) / "hyperspherical_energy.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    before = data.get("before", {})
    after = data.get("after", {})

    changes = []
    for k, b in before.items():
        a = after.get(k)
        if a is None or not b:
            continue
        changes.append(abs(a - b) / b * 100)

    if not changes:
        print("No comparable layers.")
        return

    print(f"Avg energy change: {sum(changes)/len(changes):.4f}%")
    print(f"Min energy change: {min(changes):.4f}%")
    print(f"Max energy change: {max(changes):.4f}%")


if __name__ == "__main__":
    main()
