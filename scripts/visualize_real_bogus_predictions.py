#!/usr/bin/env python3
"""
Visualize real/bogus CNN predictions: run inference on a triplet dataset and write
3-panel PNGs (reference, science, difference) with true label, P(real), and correct/wrong.

Use for reviewing training results or for "live" batch predictions. Output can be
symlinked to frontend/public/training-data/validation for the dashboard gallery.

Usage:
    python scripts/visualize_real_bogus_predictions.py \\
        --checkpoint output/models/real_bogus_cnn_full_aug/best.pt \\
        --triplet-dir output/datasets/sn2014j/training_triplets_aug \\
        --output-dir output/predictions_viz/sn2014j_aug \\
        --max-samples 100
    # Val-only (same split as evaluator):
    python scripts/visualize_real_bogus_predictions.py \\
        --checkpoint output/models/real_bogus_cnn_full_aug/best.pt \\
        --triplet-dir output/datasets/full_catalog/training_triplets_quality_aug \\
        --output-dir output/predictions_viz/full_catalog_val --val-only --max-samples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domains.detection.architectures.real_bogus_cnn import RealBogusCNN


def load_triplets(triplet_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load real and bogus NPZ arrays. Same logic as train/evaluate scripts."""
    real_path = triplet_dir / "train_real.npz"
    bogus_path = triplet_dir / "train_bogus.npz"
    if not real_path.exists() or not bogus_path.exists():
        raise FileNotFoundError(
            f"Need {real_path.name} and {bogus_path.name} in {triplet_dir}"
        )
    real = np.load(real_path)
    bogus = np.load(bogus_path)
    images = np.concatenate([real["images"], bogus["images"]], axis=0)
    labels = np.concatenate([real["labels"], bogus["labels"]], axis=0)
    images = images.astype(np.float32)
    labels = labels.astype(np.int64)
    images = np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
    return images, labels


class TripletDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def run_inference(
    model: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return P(real) for each image, shape (N,)."""
    ds = TripletDataset(images, np.zeros(len(images), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            p = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.append(p)
    return np.concatenate(probs, axis=0)


def draw_panel(ax: Any, arr: np.ndarray, title: str, cmap: str) -> None:
    im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Visualize real/bogus CNN predictions as 3-panel PNGs"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt")
    parser.add_argument(
        "--triplet-dir",
        type=Path,
        required=True,
        help="Directory with train_real.npz and train_bogus.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs and predictions.json (default: output/predictions_viz/<triplet_dir name>)",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Only visualize the validation split (same seed/val_frac as evaluator)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max number of samples to render (default 500)",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    images, labels = load_triplets(args.triplet_dir)

    # Optional val-only split (same as evaluate_real_bogus_cnn)
    if args.val_only:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(images))
        rng.shuffle(idx)
        images = images[idx]
        labels = labels[idx]
        n_val = max(1, int(len(images) * args.val_frac))
        images = images[-n_val:]
        labels = labels[-n_val:]

    # Load model and run inference
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = RealBogusCNN(in_channels=3).to(device)
    model.load_state_dict(state, strict=True)
    probs = run_inference(model, images, device, args.batch_size)

    n = min(len(images), args.max_samples)
    out_dir = args.output_dir or (Path("output/predictions_viz") / args.triplet_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for i in range(n):
        arr = images[i]  # (3, H, W)
        true_label = int(labels[i])
        pred_prob = float(probs[i])
        pred_label = 1 if pred_prob >= 0.5 else 0
        correct = pred_label == true_label
        true_str = "real" if true_label == 1 else "bogus"
        filename = f"pred_{true_str}_{i:04d}_p{pred_prob:.2f}_{'ok' if correct else 'wrong'}.png"
        out_path = out_dir / filename

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["Reference (pre-SN)", "Science (with SN)", "Difference (sci - ref)"]
        cmaps = ["gray", "gray", "RdBu_r"]
        for ax, ch, title, cmap in zip(axes, range(3), titles, cmaps):
            draw_panel(ax, arr[ch], title, cmap)

        status = "✓" if correct else "✗"
        color = "green" if correct else "red"
        fig.suptitle(
            f"True: {true_str}  |  P(real) = {pred_prob:.3f}  |  {status}",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        index.append({
            "path": filename,
            "true_label": true_label,
            "pred_prob": round(pred_prob, 4),
            "correct": correct,
        })

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "triplet_dir": str(args.triplet_dir),
                "val_only": args.val_only,
                "n_rendered": n,
                "n_total": len(images),
                "samples": index,
            },
            f,
            indent=2,
        )

    n_correct = sum(1 for s in index if s["correct"])
    print(f"Wrote {n} PNGs and predictions.json to {out_dir}")
    print(f"  Correct: {n_correct}/{n} ({100 * n_correct / n:.1f}%)")
    print(f"  To show in dashboard: symlink to frontend/public/training-data/validation")
    print(f"    ln -snf $(realpath {out_dir}) frontend/public/training-data/validation")


if __name__ == "__main__":
    main()
