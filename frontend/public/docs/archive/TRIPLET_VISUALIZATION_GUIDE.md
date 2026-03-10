# Training Triplet Visualization Guide

## Overview

The `--visualize` flag in `create_training_triplets.py` creates PNG images showing the reference, science, and difference images side-by-side for visual inspection and quality control.

## Usage

```bash
python scripts/create_training_triplets.py \
    --fits-dir output/datasets/sn2014j/fits_training \
    --diff-dir output/datasets/sn2014j/difference_images \
    --output-dir output/datasets/sn2014j/training_triplets \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --visualize
```

## Output Structure

```
training_triplets/
├── train_real.npz
├── train_bogus.npz
└── visualizations/
    ├── real/
    │   ├── 2014J_SWIFT_uuu_real_000.png
    │   ├── 2014J_SWIFT_uuu_real_001_rot90.png      # Augmented
    │   ├── 2014J_SWIFT_uuu_real_002_rot180.png     # Augmented
    │   ├── 2014J_SWIFT_uuu_real_003_rot270.png     # Augmented
    │   ├── 2014J_SWIFT_uuu_real_004_flip_lr.png    # Augmented
    │   ├── 2014J_SWIFT_uuu_real_005_flip_ud.png    # Augmented
    │   └── ...
    └── bogus/
        ├── 2014J_SWIFT_uuu_bogus_000.png
        ├── 2014J_SWIFT_uuu_bogus_001.png
        └── ...
```

## Visualization Features

Each PNG image shows:

### Layout
- **3 panels side-by-side**: Reference | Science | Difference
- **Size**: 15" × 5" at 150 DPI (~90-100 KB per image)
- **Colormap**: Grayscale with colorbars
- **Origin**: Lower-left (astronomical convention)

### Annotations
- **Red crosshair** (+): Marks the center position (SN location or random point)
- **Title bar** (top): Contains metadata
  - SN name, mission, filter
  - Pixel position (x, y)
  - Maximum significance (σ_max)
  - Overlap fraction
- **Label** (subtitle): "REAL" (green) or "BOGUS" (red)

### Filename Convention

**Real samples**:
```
{sn_name}_{mission}_{filter}_real_{index:03d}[_{augmentation}].png
```

Examples:
- `2014J_SWIFT_uuu_real_000.png` - Original
- `2014J_SWIFT_uuu_real_001_rot90.png` - 90° rotation
- `2014J_SWIFT_uuu_real_002_rot180.png` - 180° rotation
- `2014J_SWIFT_uuu_real_003_rot270.png` - 270° rotation
- `2014J_SWIFT_uuu_real_004_flip_lr.png` - Horizontal flip
- `2014J_SWIFT_uuu_real_005_flip_ud.png` - Vertical flip

**Bogus samples**:
```
{sn_name}_{mission}_{filter}_bogus_{index:03d}.png
```

Example:
- `2014J_SWIFT_uuu_bogus_000.png`

## Use Cases

### 1. Quality Control
Quickly inspect samples to verify:
- ✅ Images are properly aligned
- ✅ Difference images show clear signals
- ✅ Cutouts are centered correctly
- ✅ No obvious artifacts or errors
- ✅ Augmentations are applied correctly

### 2. Dataset Inspection
Browse samples to understand:
- Distribution of real vs bogus examples
- Variety in image quality and appearance
- Range of supernova brightnesses
- Different mission/filter characteristics

### 3. Debugging
Identify issues:
- Misaligned images (dipole artifacts)
- Poor normalization (too bright/dark)
- Incorrect center positions
- Missing or corrupted data

### 4. Presentation
Use visualizations for:
- Documentation
- Reports and papers
- Team discussions
- Training data review meetings

## Interpretation Guide

### What to Look For in REAL Samples

**Reference (left panel)**:
- Should show pre-supernova state
- May contain host galaxy
- Should be relatively clean

**Science (middle panel)**:
- Should show supernova as bright point source
- May show host galaxy + supernova
- Taken after supernova explosion

**Difference (right panel)**:
- Should show **clear point source** at center
- Minimal residuals elsewhere
- High significance (σ_max >> 5)
- Clean subtraction indicates good alignment

### What to Look For in BOGUS Samples

**Difference (right panel)**:
- Should show **no significant source** at center
- May show noise, artifacts, or residuals
- Lower significance (σ_max < 5 typically)
- Random position away from actual SN

## Performance Notes

### File Sizes
- Each PNG: ~90-100 KB
- 1,000 samples: ~100 MB
- Compressed storage in `.npz`: ~5-10 MB

### Generation Time
- ~1 second per visualization
- 100 samples: ~2 minutes
- Recommended: Use `--visualize` only for inspection, not production training

## Tips

### Quick Inspection (Few SNe)
```bash
python scripts/create_training_triplets.py \
    --fits-dir output/fits_training \
    --diff-dir output/difference_images \
    --output-dir output/triplets_check \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --visualize \
    --sn 2014J 2014ai 2014bh
```

### Full Dataset (No Visualization)
```bash
# Skip --visualize for production to save time
python scripts/create_training_triplets.py \
    --fits-dir output/fits_training \
    --diff-dir output/difference_images \
    --output-dir output/training_triplets \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --augment
```

### Selective Visualization
```bash
# Generate all data, but only visualize specific SNe
python scripts/create_training_triplets.py \
    --fits-dir output/fits_training \
    --diff-dir output/difference_images \
    --output-dir output/training_triplets \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --augment

# Then create visualizations for specific SNe
python scripts/create_training_triplets.py \
    --fits-dir output/fits_training \
    --diff-dir output/difference_images \
    --output-dir output/triplets_viz_sample \
    --cutout-size 63 \
    --bogus-ratio 1.0 \
    --visualize \
    --sn 2014J 2014ai
```

## Viewing Visualizations

### Command Line
```bash
# List all real samples
ls output/training_triplets/visualizations/real/

# List all bogus samples
ls output/training_triplets/visualizations/bogus/

# View with image viewer (Linux)
eog output/training_triplets/visualizations/real/2014J_SWIFT_uuu_real_000.png

# View all in directory
eog output/training_triplets/visualizations/real/*.png
```

### File Manager
Simply navigate to `output/training_triplets/visualizations/` and browse with your system's image viewer.

### Jupyter Notebook
```python
from IPython.display import Image, display
from pathlib import Path

viz_dir = Path("output/training_triplets/visualizations/real")
for png_file in sorted(viz_dir.glob("*.png"))[:5]:
    display(Image(filename=png_file))
```

## Comparison with Difference Image Visualizations

The triplet visualizations differ from `generate_difference_images.py --visualize`:

| Feature | Difference Images | Training Triplets |
|---------|------------------|-------------------|
| **Purpose** | Full-image QA | Training data inspection |
| **Size** | Full image (~1000×1000) | Cutouts (63×63) |
| **Layout** | 4-panel (ref, sci, diff, sig) | 3-panel (ref, sci, diff) |
| **Annotations** | SN marker, histogram | Center crosshair, metadata |
| **Output** | One per SN | Multiple per SN (real + bogus) |
| **Augmentation** | No | Yes (6 versions) |

**Use both**:
1. `generate_difference_images.py --visualize` - Verify full differencing pipeline
2. `create_training_triplets.py --visualize` - Inspect training cutouts

## Summary

The `--visualize` flag provides essential quality control for your training data:

✅ **Visual verification** of cutouts before training  
✅ **Quick debugging** of alignment or normalization issues  
✅ **Dataset exploration** to understand sample diversity  
✅ **Augmentation validation** to verify transformations  
✅ **Documentation** for reports and presentations  

**Recommendation**: Always use `--visualize` when testing on small datasets (like 2014j), then disable for production runs to save time.
