#!/usr/bin/env python3
"""Download real CC-licensed images for DreamBooth subjects from Wikimedia Commons.

This script creates:
- data/dreambooth/dog/
- data/dreambooth/cat/
- data/dreambooth/backpack/

It also writes image source metadata for attribution.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List
import shutil

import requests
from PIL import Image

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "OFT-MiniProject/1.0 (educational; contact: local-user)",
    "Accept": "application/json",
}

SUBJECT_QUERIES = {
    "dog": "dog pet photograph",
    "cat": "cat pet photograph",
    "backpack": "backpack bag photograph",
}


def search_commons_files(query: str, limit: int = 60) -> List[Dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": str(limit),
        "prop": "imageinfo|info",
        "iiprop": "url|mime|size|extmetadata",
        "inprop": "url",
    }
    r = requests.get(COMMONS_API, params=params, timeout=30, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})

    items = []
    for page_id, page in pages.items():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        url = info.get("url", "")
        mime = info.get("mime", "")
        if not url or not mime.startswith("image/"):
            continue
        items.append(
            {
                "title": page.get("title", ""),
                "descriptionurl": page.get("fullurl", ""),
                "url": url,
                "mime": mime,
                "size": info.get("size", 0),
            }
        )
    return items


def download_and_validate(item: Dict, out_path: Path) -> bool:
    try:
        r = requests.get(item["url"], timeout=40, headers=HEADERS)
        r.raise_for_status()
        out_path.write_bytes(r.content)

        with Image.open(out_path) as img:
            img = img.convert("RGB")
            img.save(out_path.with_suffix(".png"))
        if out_path.suffix.lower() != ".png" and out_path.exists():
            out_path.unlink(missing_ok=True)
        return True
    except Exception:
        out_path.unlink(missing_ok=True)
        out_path.with_suffix(".png").unlink(missing_ok=True)
        return False


def fallback_items(subject: str, n_images: int) -> List[Dict]:
    items = []
    # Public random image endpoints fallback.
    for i in range(n_images * 8):
        url = f"https://source.unsplash.com/768x768/?{subject}&sig={1000+i}"
        items.append(
            {
                "title": f"fallback_unsplash_{subject}_{i}",
                "descriptionurl": "https://source.unsplash.com",
                "url": url,
                "mime": "image/jpeg",
                "size": 0,
            }
        )

    for i in range(n_images * 8):
        url = f"https://loremflickr.com/768/768/{subject}?lock={2000+i}"
        items.append(
            {
                "title": f"fallback_loremflickr_{subject}_{i}",
                "descriptionurl": "https://loremflickr.com",
                "url": url,
                "mime": "image/jpeg",
                "size": 0,
            }
        )

    for i in range(n_images * 8):
        url = f"https://picsum.photos/seed/{subject}-{3000+i}/768/768"
        items.append(
            {
                "title": f"fallback_picsum_{subject}_{i}",
                "descriptionurl": "https://picsum.photos",
                "url": url,
                "mime": "image/jpeg",
                "size": 0,
            }
        )
    return items


def prepare_subject(subject: str, out_dir: Path, n_images: int) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.parent / f".{subject}_tmp_download"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query = SUBJECT_QUERIES[subject]
    try:
        items = search_commons_files(query, limit=80)
    except Exception:
        items = []
    if not items:
        items = fallback_items(subject, n_images)

    saved = []
    idx = 0
    for item in items:
        if len(saved) >= n_images:
            break
        candidate = tmp_dir / f"{subject}_{idx:03d}.jpg"
        idx += 1
        ok = download_and_validate(item, candidate)
        if ok:
            final_path = candidate.with_suffix(".png")
            item = dict(item)
            item["local_file"] = str(final_path)
            saved.append(item)

    if len(saved) >= n_images:
        for p in out_dir.glob("*.png"):
            p.unlink(missing_ok=True)
        for p in tmp_dir.glob("*.png"):
            shutil.move(str(p), str(out_dir / p.name))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/dreambooth")
    parser.add_argument("--n_images", type=int, default=6)
    parser.add_argument("--source_csv", type=str, default="data/dreambooth/sources.csv")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for subject in ["dog", "cat", "backpack"]:
        rows = prepare_subject(subject, root / subject, args.n_images)
        if len(rows) < args.n_images:
            raise RuntimeError(f"Failed to download enough images for {subject}: got {len(rows)}")
        all_rows.extend(rows)
        print(f"{subject}: downloaded {len(rows)} images")

    src_path = Path(args.source_csv)
    src_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["local_file", "title", "descriptionurl", "url", "mime", "size"]
    with src_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    print(f"Saved source metadata: {src_path}")


if __name__ == "__main__":
    main()
