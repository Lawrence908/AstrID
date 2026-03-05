#!/usr/bin/env python3
"""
Filter existing triplet NPZ datasets to remove samples where reference or science
channel has no usable variation (all white, all black, or constant).

Uses the same logic as create_training_triplets.is_usable_cutout: reject if
std(channel) < min_std. NPZ channels: 0=reference, 1=science, 2=difference.

Usage:
    python scripts/filter_triplet_npz_quality.py \
        --input-dir output/datasets/best_yield/training_triplets \
        --output-dir output/datasets/best_yield/training_triplets_quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def is_usable_cutout(
    arr: np.ndarray,
    min_std: float = 0.02,
    black_threshold: float = 0.05,
    max_black_frac: float = 0.4,
    center_window: int = 11,
    max_center_black_frac: float = 0.5,
    max_linear_gradient_ratio: float = 2.0,
) -> bool:
    """Return False if the cutout has no usable variation or bad data (black void, center in void, shade bands)."""
    if arr is None or arr.size == 0:
        return False
    valid = np.isfinite(arr)
    if not np.any(valid):
        return False
    std = np.nanstd(arr)
    if std < min_std:
        return False

    h, w = arr.shape
    black_mask = arr < black_threshold
    black_frac = np.sum(black_mask & valid) / max(np.sum(valid), 1)
    if black_frac > max_black_frac:
        return False

    half = center_window // 2
    cy, cx = h // 2, w // 2
    r0, r1 = max(0, cy - half), min(h, cy + half + 1)
    c0, c1 = max(0, cx - half), min(w, cx + half + 1)
    center_region = arr[r0:r1, c0:c1]
    center_valid = np.isfinite(center_region)
    if np.any(center_valid):
        center_black_frac = np.sum(center_region < black_threshold) / max(np.sum(center_valid), 1)
        if center_black_frac > max_center_black_frac:
            return False

    if max_linear_gradient_ratio > 0 and std > 1e-9:
        rows = np.arange(h, dtype=float)
        cols = np.arange(w, dtype=float)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        r_flat = rr.ravel()
        c_flat = cc.ravel()
        v_flat = arr.ravel()
        fin = np.isfinite(v_flat)
        if np.sum(fin) >= 10:
            X = np.column_stack([r_flat[fin], c_flat[fin], np.ones(np.sum(fin))])
            y = v_flat[fin]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                a, b, c0 = coeffs
                plane = a * rr + b * cc + c0
                plane_range = float(np.nanmax(plane) - np.nanmin(plane))
                if plane_range > max_linear_gradient_ratio * std:
                    return False
            except Exception:
                pass

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter triplet NPZ by reference/science quality (reject constant images)"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-std", type=float, default=0.02, help="Min std (reject constant images)")
    parser.add_argument(
        "--max-black-frac",
        type=float,
        default=0.4,
        help="Reject if fraction of black (no-data) pixels > this (default 0.4)",
    )
    parser.add_argument(
        "--max-center-black-frac",
        type=float,
        default=0.5,
        help="Reject if center window has more than this fraction black (default 0.5)",
    )
    parser.add_argument(
        "--max-linear-gradient-ratio",
        type=float,
        default=2.0,
        help="Reject if plane-fit range > this * std (shade bands; default 2.0). Set 0 to disable.",
    )
    args = parser.parse_args()

    real_path = args.input_dir / "train_real.npz"
    bogus_path = args.input_dir / "train_bogus.npz"
    for p in (real_path, bogus_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def filter_npz(path: Path, name: str) -> tuple[np.ndarray, np.ndarray, int]:
        data = np.load(path)
        images = np.asarray(data["images"])  # (N, 3, H, W); 0=ref, 1=sci, 2=diff
        labels = np.asarray(data["labels"])
        keep = []
        for i in range(len(images)):
            ref_ok = is_usable_cutout(
                images[i, 0],
                min_std=args.min_std,
                max_black_frac=args.max_black_frac,
                max_center_black_frac=args.max_center_black_frac,
                max_linear_gradient_ratio=args.max_linear_gradient_ratio,
            )
            sci_ok = is_usable_cutout(
                images[i, 1],
                min_std=args.min_std,
                max_black_frac=args.max_black_frac,
                max_center_black_frac=args.max_center_black_frac,
                max_linear_gradient_ratio=args.max_linear_gradient_ratio,
            )
            if ref_ok and sci_ok:
                keep.append(i)
        keep = np.array(keep)
        if len(keep) == 0:
            return np.zeros((0,) + images.shape[1:], dtype=images.dtype), np.array([], dtype=labels.dtype), len(images)
        new_images = images[keep]
        new_labels = labels[keep]
        removed = len(images) - len(keep)
        return new_images, new_labels, removed

    total_removed = 0
    for name, path in (("real", real_path), ("bogus", bogus_path)):
        new_images, new_labels, removed = filter_npz(path, name)
        total_removed += removed
        out_path = args.output_dir / f"train_{name}.npz"
        np.savez_compressed(out_path, images=new_images.astype(np.float32), labels=new_labels.astype(np.int64))
        print(f"  {name}: kept {len(new_images)} / {len(new_images) + removed}, removed {removed}")

    # Copy and update summary
    summary_path = args.input_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        # Recompute counts from saved NPZ
        r = np.load(args.output_dir / "train_real.npz")
        b = np.load(args.output_dir / "train_bogus.npz")
        summary["real_samples"] = int(len(r["images"]))
        summary["bogus_samples"] = int(len(b["images"]))
        summary["total_samples"] = summary["real_samples"] + summary["bogus_samples"]
        summary["quality_filtered"] = True
        summary["quality_removed"] = total_removed
        with open(args.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Total removed: {total_removed}")
    print(f"Wrote filtered data to {args.output_dir}")


if __name__ == "__main__":
    main()
