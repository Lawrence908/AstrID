#!/usr/bin/env python3
"""
Train the real/bogus CNN on image triplets (science, reference, difference).

Uses Braai-style architecture; reports precision, recall, F1, AUCPR (not accuracy).
Saves best checkpoint by validation AUCPR.

Usage:
    python scripts/train_real_bogus_cnn.py \
        --triplet-dir output/datasets/sn2014j/training_triplets \
        --output-dir output/models/real_bogus_cnn \
        --epochs 100 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domains.detection.architectures.real_bogus_cnn import RealBogusCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_triplets(triplet_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load real and bogus NPZ arrays. Returns (images, labels)."""
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
    return images.astype(np.float32), labels.astype(np.int64)


class TripletDataset(Dataset):
    """PyTorch Dataset for (images, labels) triplets."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor | None,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images).unsqueeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    n_pos = max(int(y_true.sum()), 1)
    n_neg = max(int((1 - y_true).sum()), 1)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    aucpr = average_precision_score(y_true, y_prob) if n_pos > 0 else 0.0

    return prec, rec, f1, aucpr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train real/bogus CNN on triplets")
    parser.add_argument(
        "--triplet-dir",
        type=Path,
        default=Path("output/datasets/sn2014j/training_triplets"),
        help="Directory with train_real.npz and train_bogus.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/models/real_bogus_cnn"),
        help="Where to save checkpoints and metrics",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of data for validation (0–1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading triplets from %s", args.triplet_dir)
    images, labels = load_triplets(args.triplet_dir)
    n_real = int((labels == 1).sum())
    n_bogus = int((labels == 0).sum())
    logger.info("Samples: %d real, %d bogus, total %d", n_real, n_bogus, len(images))

    # Train/val split (stratified by label)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(images))
    rng.shuffle(idx)
    images, labels = images[idx], labels[idx]

    n_val = max(1, int(len(images) * args.val_frac))
    n_train = len(images) - n_val
    train_images, val_images = images[:n_train], images[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]

    train_ds = TripletDataset(train_images, train_labels)
    val_ds = TripletDataset(val_images, val_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Class weight for imbalance (emphasize real = positive)
    pos_weight = torch.tensor(
        [n_bogus / max(n_real, 1)], dtype=torch.float32, device=device
    )

    # Model and optimizer
    model = RealBogusCNN(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_aucpr = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, pos_weight
        )
        prec, rec, f1, aucpr = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_precision": float(prec),
            "val_recall": float(rec),
            "val_f1": float(f1),
            "val_aucpr": float(aucpr),
        })

        logger.info(
            "Epoch %3d  loss=%.4f  val P=%.3f R=%.3f F1=%.3f AUCPR=%.3f",
            epoch, train_loss, prec, rec, f1, aucpr,
        )

        if aucpr > best_aucpr:
            best_aucpr = aucpr
            ckpt_path = args.output_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_aucpr": aucpr,
                "val_f1": f1,
            }, ckpt_path)
            logger.info("  -> saved best checkpoint to %s", ckpt_path)

    # Save final model and history
    torch.save(model.state_dict(), args.output_dir / "last.pt")
    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Done. Best val AUCPR: %.3f", best_aucpr)
    logger.info("Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
