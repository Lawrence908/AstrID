#!/usr/bin/env python3
"""
Evaluate a trained real/bogus CNN checkpoint on the validation split of a triplet dataset.

Uses the same data load and train/val split as training (same seed/val_frac) so
metrics are comparable. Reports P/R/F1, AUCPR, and confusion matrix.

Usage:
    python scripts/evaluate_real_bogus_cnn.py \
        --checkpoint output/models/real_bogus_cnn/best.pt \
        --triplet-dir output/datasets/best_yield/training_triplets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domains.detection.architectures.real_bogus_cnn import RealBogusCNN


def load_triplets(triplet_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load real and bogus NPZ arrays. Same logic as train script."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real/bogus CNN checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt (or last.pt)")
    parser.add_argument(
        "--triplet-dir",
        type=Path,
        required=True,
        help="Same triplet dir used for training (train_real.npz, train_bogus.npz)",
    )
    parser.add_argument("--val-frac", type=float, default=0.2, help="Must match training")
    parser.add_argument("--seed", type=int, default=42, help="Must match training")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Same split as training
    images, labels = load_triplets(args.triplet_dir)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(images))
    rng.shuffle(idx)
    images, labels = images[idx], labels[idx]
    n_val = max(1, int(len(images) * args.val_frac))
    val_images, val_labels = images[-n_val:], labels[-n_val:]

    val_ds = TripletDataset(val_images, val_labels)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load checkpoint (best.pt has model_state_dict; last.pt is state_dict only)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        print(f"Checkpoint from epoch {ckpt.get('epoch', '?')} (val AUCPR {ckpt.get('val_aucpr', '?'):.3f})")
    else:
        state = ckpt
        print("Loaded state_dict only (e.g. last.pt)")

    model = RealBogusCNN(in_channels=3).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    aucpr = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0
    # Explicit labels: 0 = bogus, 1 = real (match training)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    n_real = int(y_true.sum())
    n_bogus = len(y_true) - n_real
    print(f"\nValidation set: n = {len(y_true)} ({n_real} real, {n_bogus} bogus)")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  AUCPR:     {aucpr:.3f}")
    print("\nConfusion matrix (label 0=bogus, 1=real; rows=true, cols=pred)")
    print("              pred_bogus  pred_real  (row total)")
    print(f"  true_bogus     {cm[0,0]:6d}     {cm[0,1]:6d}     {cm[0].sum()}")
    print(f"  true_real     {cm[1,0]:6d}     {cm[1,1]:6d}     {cm[1].sum()}")


if __name__ == "__main__":
    main()
