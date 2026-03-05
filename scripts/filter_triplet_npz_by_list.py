#!/usr/bin/env python3
"""
Build a training NPZ that contains only the samples you list.

List file format: one key per line (SN_mission_filter), e.g.:
  2008cj_GALEX_nd
  2009hd_SWIFT_uuu

Keys are matched against triplet metadata (sn_name, mission, filter). Any real or
bogus sample whose key matches a line is kept. Alphabetize and dedupe your list;
lines starting with # are ignored.

Usage:
    python scripts/filter_triplet_npz_by_list.py \
        --input-dir output/datasets/best_yield/training_triplets_full \
        --list curated.txt \
        --output-dir output/datasets/best_yield/training_triplets_curated \
        --copy-visualizations
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

# Same pattern as parse_list_file: key is everything before _real_ or _bogus_
KEY_FROM_FILENAME_RE = re.compile(r"^(.+)_(?:real|bogus)_\d+", re.IGNORECASE)


def parse_list_file(path: Path) -> set[str]:
    """Parse list file: one key (SN_mission_filter) or viz filename per line. Returns set of keys."""
    keys: set[str] = set()
    # If line looks like a viz filename, extract key (part before _real_ or _bogus_)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = Path(line).name
        m = KEY_FROM_FILENAME_RE.search(name)
        if m:
            keys.add(m.group(1))
        else:
            # Plain key line
            keys.add(line)
    return keys


def meta_key(entry: dict) -> str:
    """Build key from metadata entry: sn_name_mission_filter."""
    sn = entry.get("sn_name", "")
    mission = entry.get("mission", "")
    filt = entry.get("filter", entry.get("filter_name", ""))
    return f"{sn}_{mission}_{filt}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter triplet NPZ to keep only samples whose key (SN_mission_filter) is in the list"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True, help="Text file: one key per line (SN_mission_filter)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-visualizations",
        action="store_true",
        help="Copy PNGs from input-dir/visualizations that match curated keys so you can review before training",
    )
    args = parser.parse_args()

    keep_keys = parse_list_file(args.list)
    print(f"List: {len(keep_keys)} keys")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_path = args.input_dir / "train_real.npz"
    bogus_path = args.input_dir / "train_bogus.npz"
    if not real_path.exists() or not bogus_path.exists():
        raise FileNotFoundError(f"Need {real_path} and {bogus_path}")

    def filter_by_keys(
        npz_path: Path,
        meta_path: Path,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
        data = np.load(npz_path)
        images = np.asarray(data["images"])
        labels = np.asarray(data["labels"])
        if not meta_path.exists():
            print(f"  {name}: no metadata, cannot filter by key; skipping")
            return np.zeros((0,) + images.shape[1:], dtype=images.dtype), np.array([], dtype=labels.dtype), None
        with open(meta_path) as f:
            meta = json.load(f)
        indices = [i for i in range(len(meta)) if meta_key(meta[i]) in keep_keys]
        if not indices:
            return np.zeros((0,) + images.shape[1:], dtype=images.dtype), np.array([], dtype=labels.dtype), None
        new_images = images[indices]
        new_labels = labels[indices]
        return new_images, new_labels, indices

    real_images, real_labels, real_idx = filter_by_keys(
        real_path, args.input_dir / "train_real_metadata.json", "real"
    )
    bogus_images, bogus_labels, bogus_idx = filter_by_keys(
        bogus_path, args.input_dir / "train_bogus_metadata.json", "bogus"
    )

    np.savez_compressed(
        args.output_dir / "train_real.npz",
        images=real_images.astype(np.float32),
        labels=real_labels.astype(np.int64),
    )
    np.savez_compressed(
        args.output_dir / "train_bogus.npz",
        images=bogus_images.astype(np.float32),
        labels=bogus_labels.astype(np.int64),
    )
    print(f"  real:  {len(real_images)} samples")
    print(f"  bogus: {len(bogus_images)} samples")

    # Filter metadata if present
    for split, idx in (("real", real_idx), ("bogus", bogus_idx)):
        meta_path = args.input_dir / f"train_{split}_metadata.json"
        if meta_path.exists() and idx is not None:
            with open(meta_path) as f:
                meta = json.load(f)
            new_meta = [meta[i] for i in idx]
            with open(args.output_dir / f"train_{split}_metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)

    # Summary
    summary = {
        "real_samples": int(len(real_images)),
        "bogus_samples": int(len(bogus_images)),
        "total_samples": int(len(real_images) + len(bogus_images)),
        "curated_from_list": str(args.list),
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.copy_visualizations:
        viz_src_real = args.input_dir / "visualizations" / "real"
        viz_src_bogus = args.input_dir / "visualizations" / "bogus"
        viz_dst_real = args.output_dir / "visualizations" / "real"
        viz_dst_bogus = args.output_dir / "visualizations" / "bogus"
        copied = 0
        for src_dir, dst_dir in ((viz_src_real, viz_dst_real), (viz_src_bogus, viz_dst_bogus)):
            if not src_dir.exists():
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            for png in src_dir.glob("*.png"):
                m = KEY_FROM_FILENAME_RE.search(png.name)
                if m and m.group(1) in keep_keys:
                    shutil.copy2(png, dst_dir / png.name)
                    copied += 1
        print(f"  Copied {copied} visualizations to {args.output_dir / 'visualizations'}")

    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
