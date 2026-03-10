# Training Triplet Generation Pipeline

## Overview

This document describes the complete pipeline for generating training triplets (reference, science, difference) from supernova FITS data for machine learning classifier training.

## Pipeline Stages

### 1. Download FITS Files
**Script**: `scripts/download_sn_fits.py` or `scripts/run_pipeline_from_config.py`

Downloads reference and science FITS images for supernovae from MAST archive.

```bash
# Using the full pipeline (recommended)
python scripts/run_pipeline_from_config.py \
    --config configs/datasets/sn2014j.yaml \
    --stages download organize differencing
```

### 2. Organize Training Pairs
**Script**: `scripts/organize_training_pairs.py`

Organizes downloaded FITS files into a clean directory structure and creates a training manifest.

```bash
python scripts/organize_training_pairs.py \
    --input-dir output/datasets/sn2014j/fits_downloads \
    --output-dir output/datasets/sn2014j/fits_training \
    --clean
```

**Output Structure**:
```
fits_training/
├── training_manifest.json
├── 2014J/
│   ├── reference/
│   │   ├── SWIFT_sw00032503001uuu_sk.fits
│   │   └── ...
│   └── science/
│       ├── SWIFT_sw00032503094uuu_sk.fits
│       └── ...
└── ...
```

### 3. Generate Difference Images
**Script**: `scripts/generate_difference_images.py`

Creates proper astronomical difference images using WCS alignment, PSF matching, and ZOGY differencing.

```bash
python scripts/generate_difference_images.py \
    --input-dir output/datasets/sn2014j/fits_training \
    --output-dir output/datasets/sn2014j/difference_images
```

**Output Structure**:
```
difference_images/
├── processing_summary.json
├── 2014J/
│   ├── 2014J_SWIFT_uuu_diff.fits
│   ├── 2014J_SWIFT_uuu_sig.fits
│   └── 2014J_SWIFT_uuu_mask.fits
└── ...
```

### 4. Create Training Triplets ⭐ NEW
**Script**: `scripts/create_training_triplets.py`

Generates normalized 63×63 pixel triplets centered on supernova positions, with both real and bogus samples.

```bash
# Basic usage (test dataset)
python scripts/create_training_triplets.py \
    --fits-dir output/datasets/sn2014j/fits_training \
    --diff-dir output/datasets/sn2014j/difference_images \
    --output-dir output/datasets/sn2014j/training_triplets \
    --cutout-size 63 \
    --bogus-ratio 1.0

# With data augmentation (recommended for training)
python scripts/create_training_triplets.py \
    --fits-dir output/datasets/sn2014j/fits_training \
    --diff-dir output/datasets/sn2014j/difference_images \
    --output-dir output/datasets/sn2014j/training_triplets_augmented \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --augment

# With visualization (creates PNG images for inspection)
python scripts/create_training_triplets.py \
    --fits-dir output/datasets/sn2014j/fits_training \
    --diff-dir output/datasets/sn2014j/difference_images \
    --output-dir output/datasets/sn2014j/training_triplets_viz \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --augment \
    --visualize

# Full dataset
python scripts/create_training_triplets.py \
    --fits-dir output/fits_training \
    --diff-dir output/difference_images \
    --output-dir output/training_triplets \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --augment
```

**Output Structure**:
```
training_triplets/
├── summary.json
├── train_real.npz              # Real samples (N, 3, 63, 63)
├── train_real_metadata.json
├── train_bogus.npz             # Bogus samples (N, 3, 63, 63)
├── train_bogus_metadata.json
└── visualizations/             # Created with --visualize flag
    ├── real/
    │   ├── 2014J_SWIFT_uuu_real_000.png
    │   ├── 2014J_SWIFT_uuu_real_001_rot90.png
    │   └── ...
    └── bogus/
        ├── 2014J_SWIFT_uuu_bogus_000.png
        └── ...
```

## Data Format

### Triplet Structure

Each training sample is a **3-channel image** of shape `(3, 63, 63)`:
- **Channel 0**: Reference image (pre-supernova)
- **Channel 1**: Science image (with supernova)
- **Channel 2**: Difference image (science - reference)

All channels are:
- ✅ Normalized to [0, 1] range
- ✅ 32-bit float precision
- ✅ NaN/Inf values replaced with 0
- ✅ Robust percentile-based normalization (1st-99.5th percentile)

### Labels

- **1** = Real transient (at known supernova position)
- **0** = Bogus (random position, artifact, or noise)

### Data Augmentation

When `--augment` is enabled, each real sample generates 6 versions:
1. Original
2. Rotated 90°
3. Rotated 180°
4. Rotated 270°
5. Horizontal flip
6. Vertical flip

This increases training data by 6× while maintaining physical validity.

## Loading Training Data

### Python Example

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Load triplets
real_data = np.load("training_triplets/train_real.npz")
bogus_data = np.load("training_triplets/train_bogus.npz")

images = np.concatenate([real_data["images"], bogus_data["images"]], axis=0)
labels = np.concatenate([real_data["labels"], bogus_data["labels"]], axis=0)

print(f"Images shape: {images.shape}")  # (N, 3, 63, 63)
print(f"Labels shape: {labels.shape}")  # (N,)

# Create PyTorch dataset
class SupernovaDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create dataloader
dataset = SupernovaDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch_images, batch_labels in dataloader:
    # batch_images: (32, 3, 63, 63)
    # batch_labels: (32,)
    # ... train your model ...
    pass
```

### Visualization

**Option 1: Use `--visualize` flag** (recommended for inspection):

```bash
python scripts/create_training_triplets.py \
    --fits-dir output/datasets/sn2014j/fits_training \
    --diff-dir output/datasets/sn2014j/difference_images \
    --output-dir output/datasets/sn2014j/training_triplets \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --visualize
```

This creates PNG images in `visualizations/real/` and `visualizations/bogus/` showing:
- Side-by-side triplets (reference, science, difference)
- Red crosshair marking the center position
- Metadata (SN name, mission, filter, significance, overlap)
- Color-coded labels (green=REAL, red=BOGUS)
- Augmentation type in filename (e.g., `_rot90`, `_flip_lr`)

**Option 2: Programmatic visualization**:

```python
import matplotlib.pyplot as plt

def visualize_triplet(triplet, label, idx=0):
    """Visualize a single triplet."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    titles = ["Reference", "Science", "Difference"]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.imshow(triplet[i], cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")
    
    label_str = "REAL" if label == 1 else "BOGUS"
    fig.suptitle(f"Sample {idx}: {label_str}", fontsize=14)
    plt.tight_layout()
    plt.show()

# Visualize first sample
visualize_triplet(images[0], labels[0], idx=0)
```

## Complete Workflow Examples

### Test on 2014j Dataset

```bash
# 1. Run pipeline to generate difference images
python scripts/run_pipeline_from_config.py \
    --config configs/datasets/sn2014j.yaml \
    --stages download organize differencing

# 2. Create training triplets with augmentation
python scripts/create_training_triplets.py \
    --fits-dir output/datasets/sn2014j/fits_training \
    --diff-dir output/datasets/sn2014j/difference_images \
    --output-dir output/datasets/sn2014j/training_triplets \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --augment

# 3. Visualize samples
python scripts/example_load_triplets.py \
    --triplet-dir output/datasets/sn2014j/training_triplets \
    --n-samples 5
```

### Process Full Dataset

```bash
# 1. Generate difference images for all SNe
python scripts/generate_difference_images.py \
    --input-dir output/fits_training \
    --output-dir output/difference_images

# 2. Create training triplets
python scripts/create_training_triplets.py \
    --fits-dir output/fits_training \
    --diff-dir output/difference_images \
    --output-dir output/training_triplets \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --augment

# 3. Check summary
cat output/training_triplets/summary.json
```

## Parameters

### `create_training_triplets.py` Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fits-dir` | Required | Directory with organized FITS files |
| `--diff-dir` | Required | Directory with difference images |
| `--output-dir` | Required | Output directory for triplets |
| `--cutout-size` | 63 | Size of cutouts (63×63 pixels) |
| `--bogus-ratio` | 1.0 | Ratio of bogus to real samples (1.0 = balanced) |
| `--augment` | False | Apply data augmentation (6× increase) |
| `--visualize` | False | Create PNG visualizations of all triplets |
| `--sn` | None | Only process specific supernovae |

### Recommended Settings

**For Testing** (small dataset like 2014j):
```bash
--cutout-size 63 --bogus-ratio 1.0 --augment
```

**For Production** (full dataset):
```bash
--cutout-size 63 --bogus-ratio 1.0 --augment
```

**For Debugging** (no augmentation, with visualization):
```bash
--cutout-size 63 --bogus-ratio 1.0 --visualize
```

**For Quality Check** (visualize a few SNe):
```bash
--cutout-size 63 --bogus-ratio 1.0 --augment --visualize --sn 2014J 2014ai
```

## Dataset Statistics

### Example: 2014j Dataset (5 SNe)

**Without Augmentation**:
- Real samples: 5
- Bogus samples: 4
- Total: 9 samples
- Class balance: 0.80:1

**With Augmentation**:
- Real samples: 30 (5 × 6)
- Bogus samples: 23
- Total: 53 samples
- Class balance: 0.77:1

### Expected: Full Dataset (~100 SNe)

**With Augmentation**:
- Real samples: ~600 (100 × 6)
- Bogus samples: ~600
- Total: ~1,200 samples
- Class balance: 1.0:1

## Next Steps

1. **Train CNN Classifier**
   - Use triplets as input to a 3-channel CNN
   - Binary classification: real (1) vs bogus (0)
   - Recommended architecture: ResNet-18 or custom CNN

2. **Evaluation Metrics**
   - Precision, Recall, F1 Score
   - AUCPR (Area Under Precision-Recall Curve)
   - Confusion matrix
   - **Goal**: High recall (don't miss real SNe) with reasonable precision

3. **Cross-Validation**
   - Split by supernova (not by sample) to avoid data leakage
   - Temporal or spatial separation for validation
   - Test on held-out SNe

4. **Deployment**
   - Integrate classifier into detection pipeline
   - Real-time inference on new detections
   - Automated alert filtering

## Troubleshooting

### No triplets created

**Problem**: `No triplets were created!`

**Solutions**:
1. Check that difference images exist: `ls output/difference_images/*/`
2. Verify FITS files are organized: `ls output/fits_training/*/reference/`
3. Ensure filter names match between difference and FITS files
4. Check logs for specific errors

### FITS loading errors

**Problem**: `Failed to load *.fits: 'NoneType' object has no attribute 'astype'`

**Solution**: The script now handles SWIFT `.img` files (data in HDU 1) automatically.

### No SN position in header

**Problem**: `No SN position in header for {sn_name}, using image center`

**Impact**: Bogus samples may be less diverse (all centered around image center)

**Solution**: This is expected for some SNe. The script uses image center as fallback.

### Imbalanced classes

**Problem**: More real samples than bogus (or vice versa)

**Solution**: Adjust `--bogus-ratio` parameter:
- `--bogus-ratio 2.0` → 2× more bogus than real
- `--bogus-ratio 0.5` → 0.5× bogus (more real samples)

## File Formats

### `.npz` Files (Compressed NumPy)

Efficient storage format for large arrays:
- Compressed with gzip
- Fast loading with `np.load()`
- Contains multiple arrays in one file

### Metadata JSON

Human-readable metadata for each sample:
```json
{
  "sn_name": "2014J",
  "mission": "SWIFT",
  "filter": "uuu",
  "center_x": 696.5,
  "center_y": 681.2,
  "ref_date": "2014-01-15",
  "sci_date": "2014-02-20",
  "overlap": 0.662,
  "sig_max": 17472326537.9
}
```

## References

- **ZTF Real/Bogus Classifier**: [Duev et al. 2019](https://arxiv.org/abs/1902.01936)
- **ZOGY Difference Imaging**: [Zackay et al. 2016](https://arxiv.org/abs/1601.02655)
- **CNN Architecture**: ResNet, EfficientNet, or custom designs
- **Data Augmentation**: Standard practice in astronomical ML

## Summary

The training triplet pipeline provides a complete, automated workflow for generating ML-ready data from raw FITS files. Key features:

✅ Proper astronomical differencing (WCS, PSF, ZOGY)  
✅ Normalized, ready-to-train format  
✅ Balanced real/bogus samples  
✅ Data augmentation for robustness  
✅ Efficient compressed storage  
✅ Metadata tracking for reproducibility  
✅ Flexible configuration for different datasets  

**Next**: Train a CNN classifier using these triplets! 🚀
