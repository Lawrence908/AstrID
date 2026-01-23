#!/usr/bin/env python3
"""
Example script showing how to load and visualize training triplets.

This demonstrates how to:
1. Load the compressed numpy arrays
2. Create a PyTorch dataset
3. Visualize sample triplets

Usage:
    python scripts/example_load_triplets.py \
        --triplet-dir output/datasets/sn2014j/training_triplets_augmented
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_triplets(triplet_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all training triplets from a directory.
    
    Returns:
        images: (N, 3, 63, 63) array of triplets
        labels: (N,) array of labels (1=real, 0=bogus)
    """
    # Load real samples
    real_data = np.load(triplet_dir / "train_real.npz")
    real_images = real_data["images"]
    real_labels = real_data["labels"]
    
    # Load bogus samples
    bogus_data = np.load(triplet_dir / "train_bogus.npz")
    bogus_images = bogus_data["images"]
    bogus_labels = bogus_data["labels"]
    
    # Combine
    images = np.concatenate([real_images, bogus_images], axis=0)
    labels = np.concatenate([real_labels, bogus_labels], axis=0)
    
    return images, labels


def visualize_triplet(triplet: np.ndarray, label: int, idx: int = 0) -> None:
    """Visualize a single triplet (reference, science, difference).
    
    Args:
        triplet: (3, H, W) array
        label: 1=real, 0=bogus
        idx: Sample index for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    titles = ["Reference", "Science", "Difference"]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.imshow(triplet[i], cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")
    
    label_str = "REAL" if label == 1 else "BOGUS"
    fig.suptitle(f"Sample {idx}: {label_str}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Load and visualize training triplets")
    parser.add_argument(
        "--triplet-dir",
        type=Path,
        required=True,
        help="Directory containing training triplets",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5)",
    )
    
    args = parser.parse_args()
    
    if not args.triplet_dir.exists():
        print(f"Error: Directory does not exist: {args.triplet_dir}")
        return
    
    # Load data
    print(f"Loading triplets from: {args.triplet_dir}")
    images, labels = load_triplets(args.triplet_dir)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(images)}")
    print(f"  Real samples:  {np.sum(labels == 1)}")
    print(f"  Bogus samples: {np.sum(labels == 0)}")
    print(f"  Image shape:   {images.shape}")
    print(f"  Value range:   [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mean:          {images.mean():.3f}")
    print(f"  Std:           {images.std():.3f}")
    
    # Visualize random samples
    print(f"\nVisualizing {args.n_samples} random samples...")
    indices = np.random.choice(len(images), size=min(args.n_samples, len(images)), replace=False)
    
    for idx in indices:
        visualize_triplet(images[idx], labels[idx], idx)
    
    # Show class distribution
    print("\nClass Distribution:")
    real_count = np.sum(labels == 1)
    bogus_count = np.sum(labels == 0)
    print(f"  Real:  {real_count} ({100*real_count/len(labels):.1f}%)")
    print(f"  Bogus: {bogus_count} ({100*bogus_count/len(labels):.1f}%)")
    
    # Example: Create PyTorch dataset
    print("\n" + "="*60)
    print("Example PyTorch Dataset:")
    print("="*60)
    print("""
import torch
from torch.utils.data import Dataset, DataLoader

class SupernovaDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create dataset and dataloader
dataset = SupernovaDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
for batch_images, batch_labels in dataloader:
    # batch_images: (8, 3, 63, 63)
    # batch_labels: (8,)
    # ... train your model ...
    pass
    """)


if __name__ == "__main__":
    main()
