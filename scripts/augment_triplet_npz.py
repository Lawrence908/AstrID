#!/usr/bin/env python3
"""
Augment an existing triplet dataset (train_real.npz, train_bogus.npz) by adding
rotations and flips. Saves to a new directory so the original is unchanged.

Each sample becomes 6 versions: original + rot90, rot180, rot270 + fliplr + flipud.
Use the output dir as --triplet-dir for training to get ~6x more data.

Usage:
    python scripts/augment_triplet_npz.py \
        --input-dir output/datasets/best_yield/training_triplets_curated \
        --output-dir output/datasets/best_yield/training_triplets_curated_aug \
        --visualize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_visualizations(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    split: str,
    max_to_plot: int = 500,
) -> None:
    """Write 3-panel PNGs (ref, sci, diff) for up to max_to_plot samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "visualizations" / split
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(images), max_to_plot)
    for i in range(n):
        arr = images[i]  # (3, H, W)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["Reference (pre-SN)", "Science (with SN)", "Difference (sci - ref)"]
        for ax, ch, title in zip(axes, range(3), titles):
            im = ax.imshow(arr[ch], cmap="gray" if ch < 2 else "RdBu_r", origin="lower", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(viz_dir / f"{split}_{i:04d}.png", dpi=100, bbox_inches="tight")
        plt.close()
    print(f"  Wrote {n} visualizations to {viz_dir}")


def augment_images(images: np.ndarray) -> np.ndarray:
    """Return stacked array of shape (N*6, C, H, W): original + 3 rots + 2 flips per image."""
    # images: (N, C, H, W)
    out = [images]  # original
    for k in (1, 2, 3):
        out.append(np.rot90(images, k, axes=(-2, -1)).copy())
    # (N, C, H, W): flip width (last dim) and height (second-to-last)
    out.append(images[..., ::-1].copy())      # fliplr
    out.append(images[..., ::-1, :].copy())  # flipud
    return np.concatenate(out, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment triplet NPZ (rotations, flips)")
    parser.add_argument("--input-dir", type=Path, required=True, help="Dir with train_real.npz, train_bogus.npz")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write augmented NPZs")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write 3-panel PNGs for augmented samples (up to 500 per split) for review",
    )
    args = parser.parse_args()

    real_path = args.input_dir / "train_real.npz"
    bogus_path = args.input_dir / "train_bogus.npz"
    for p in (real_path, bogus_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    factor = 6  # original + 3 rots + 2 flips

    for name, path in (("real", real_path), ("bogus", bogus_path)):
        data = np.load(path)
        images = data["images"]
        labels = data["labels"]
        aug_images = augment_images(images)
        aug_labels = np.repeat(labels, factor)
        out_path = args.output_dir / f"train_{name}.npz"
        np.savez_compressed(out_path, images=aug_images.astype(np.float32), labels=aug_labels.astype(np.int64))
        print(f"  {name}: {len(images)} -> {len(aug_images)} samples")

    # Copy summary and adjust
    summary_path = args.input_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary["augmentation"] = True
        summary["real_samples"] = summary["real_samples"] * factor
        summary["bogus_samples"] = summary["bogus_samples"] * factor
        summary["total_samples"] = summary["total_samples"] * factor
        with open(args.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    if args.visualize:
        for name, path in (("real", args.output_dir / "train_real.npz"), ("bogus", args.output_dir / "train_bogus.npz")):
            data = np.load(path)
            write_visualizations(
                np.asarray(data["images"]),
                np.asarray(data["labels"]),
                args.output_dir,
                name,
            )

    print(f"Wrote augmented data to {args.output_dir}")


if __name__ == "__main__":
    main()
