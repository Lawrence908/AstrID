# AstrID: Supernova Detection with Deep Learning Pipelines

> **Project Goal**: Build an automated machine learning pipeline to detect supernovae in astronomical images using image differencing and deep learning classification.

---

## Executive Summary

Modern astronomical surveys generate millions of transient alerts per night (e.g., Vera Rubin/LSST: ~10 million alerts/night). Manual inspection is impossible at this scale—automation is the only path forward.

This document outlines the technical pipeline for detecting transient events like supernovae, tracing the complete chain from raw FITS images to classified candidates. The approach combines **classical astronomical algorithms** (decades-old, robust) with **modern deep learning** (CNNs for classification).

**Key Insight**: This is not about choosing between classical and ML approaches—it's about combining them. You need robust algorithms to prepare data perfectly, then feed clean inputs to neural networks for classification.

---

## Table of Contents

1. [Pipeline Architecture Overview](#1-pipeline-architecture-overview)
2. [Phase 1: Data Acquisition & Catalog Building](#phase-1-data-acquisition--catalog-building)
3. [Phase 2: Image Calibration & Alignment](#phase-2-image-calibration--alignment)
4. [Phase 3: Difference Image Generation (ZOGY)](#phase-3-difference-image-generation-zogy)
5. [Phase 4: Source Extraction & Candidate Generation](#phase-4-source-extraction--candidate-generation)
6. [Phase 5: Real/Bogus Classification (CNN)](#phase-5-realbogus-classification-cnn)
7. [Phase 6: Training Data Strategy](#phase-6-training-data-strategy)
8. [Phase 7: Model Deployment & Inference](#phase-7-model-deployment--inference)
9. [Phase 8: Evaluation & Metrics](#phase-8-evaluation--metrics)
10. [Current Implementation Status](#current-implementation-status)
11. [Next Steps & Roadmap](#next-steps--roadmap)
12. [References & Background](#references--background)

---

## 1. Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SUPERNOVA DETECTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Supernova  │───▶│    MAST      │───▶│  Reference   │                   │
│  │   Catalog    │    │  Archive     │    │  + Science   │                   │
│  │  (6500+ SNe) │    │  Query       │    │    Images    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                                       │                            │
│         ▼                                       ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    CLASSICAL ALGORITHMS                           │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │       │
│  │  │ Calibration │─▶│    WCS      │─▶│  ZOGY       │               │       │
│  │  │ (Bias/Flat) │  │ Alignment   │  │ Subtraction │               │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                       │                                      │
│                                       ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    SOURCE EXTRACTION                              │       │
│  │  Difference Image ──▶ 5σ Detections ──▶ Candidate Triplets       │       │
│  │                       (SEP/Photutils)    (Sci, Ref, Diff)        │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                       │                                      │
│                                       ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    MACHINE LEARNING                               │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │       │
│  │  │   Image     │─▶│    CNN      │─▶│  Real/Bogus │               │       │
│  │  │  Triplets   │  │ Classifier  │  │  Score 0-1  │               │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                       │                                      │
│                                       ▼                                      │
│                          TRANSIENT CANDIDATES                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Acquisition & Catalog Building

### Objective
Build a comprehensive catalog of known supernovae with coordinates and metadata, then acquire corresponding astronomical images (reference pre-SN + science post-SN).

### Technical Requirements

| Component | Description | Status |
|-----------|-------------|--------|
| SN Catalog | 6500+ supernovae with RA/Dec, discovery date, type, host galaxy | ✅ Complete |
| Archive Query | MAST API to find observations near SN coordinates | ✅ Complete |
| Reference Images | Pre-discovery observations (no SN present) | ✅ Working |
| Science Images | Post-discovery observations (SN visible) | ✅ Working |
| Mission Matching | Same-mission pairs for valid differencing | ✅ Implemented |

### Implementation Files

```
resources/sncat_compiled.txt          # 6500+ SN catalog from Open Supernova Catalog
scripts/compile_supernova_catalog.py  # Catalog compilation
scripts/query_sn_fits_from_catalog.py # MAST archive queries
scripts/download_sn_fits.py           # FITS file downloads
scripts/organize_training_pairs.py    # Organize ref/sci pairs
```

### Key Considerations

1. **Same-Mission Requirement**: For valid differencing, reference and science images must come from the same telescope/instrument (e.g., SWIFT-SWIFT, PS1-PS1). Cross-mission differencing fails due to different PSFs, filters, and noise characteristics.

2. **Filter Matching**: Within the same mission, images should use the same filter band (e.g., both UVW1, both UUU) for proper flux comparison.

3. **Temporal Separation**: Reference images should be taken well before SN discovery; science images during/after peak brightness.

### Current Data Inventory

- **Total SNe in catalog**: 6,542
- **SNe with downloaded data**: 19 (pilot set)
- **Same-mission pairs (usable)**: 8 (SWIFT-SWIFT)
- **Cross-mission pairs**: 11 (not suitable for differencing)

---

## Phase 2: Image Calibration & Alignment

### Objective
Prepare raw FITS images for differencing by correcting instrument effects and precisely aligning images to a common coordinate system.

### Technical Steps

#### 2.1 Instrument Calibration
Standard corrections applied to raw astronomical images:

| Correction | Purpose | Notes |
|------------|---------|-------|
| Bias Subtraction | Remove electronic offset | Usually pre-applied in archive data |
| Flat Field | Correct pixel sensitivity variations | Usually pre-applied |
| Bad Pixel Mask | Flag dead/hot pixels | Important for artifact rejection |
| Cosmic Ray Rejection | Remove particle hits | Critical for single exposures |

> **Note**: Most archive data (SWIFT, PS1) is already calibrated. Verify by checking FITS headers for `CALIBRATED` or `LEVEL` keywords.

#### 2.2 Astrometric Alignment (WCS)

**Why It's Critical**: The entire method relies on pixel-by-pixel subtraction. Even sub-pixel misalignment creates massive residuals around every bright source—noise that looks exactly like the faint transients you're hunting.

**Implementation**:

```python
from reproject import reproject_interp
from astropy.wcs import WCS

# Reproject science image to match reference WCS grid
sci_aligned, footprint = reproject_interp(
    (sci_data, sci_wcs),
    ref_wcs,
    shape_out=ref_data.shape
)
```

**Key Metrics**:
- Overlap fraction: Percentage of pixels with valid data after reprojection
- Alignment residuals: Should be <0.1 pixel RMS

### Implementation Files

```
notebooks/differencing.ipynb           # WCS alignment implementation
scripts/generate_difference_images.py  # Production differencing pipeline
```

---

## Phase 3: Difference Image Generation (ZOGY)

### Objective
Compute an optimal difference image that reveals new sources (transients) while suppressing noise and artifacts from static sources.

### The Problem with Simple Subtraction

Simple `Difference = Science - Reference` fails when:
- Images have different PSF (point spread function) / blur levels
- Different exposure times / flux levels
- Different noise characteristics

Result: Massive residuals around every bright object, drowning out faint transients.

### The ZOGY Algorithm (Zackay, Ofek, Gal-Yam 2016)

ZOGY is the optimal image subtraction method for transient detection. Key features:

1. **PSF Matching**: Mathematically convolves images to match PSFs before subtraction
2. **Fourier-Space Operations**: Performs convolution and subtraction in frequency space for efficiency
3. **Whitened Noise**: Output difference image has uniform, predictable noise properties
4. **Significance Map**: Outputs SNR (signal-to-noise ratio) per pixel, not just flux difference

#### Why "Whitened Noise" Matters

If noise is uniform and predictable, you can apply a simple threshold (e.g., 5σ) to find significant detections. This enables detecting a faint supernova sitting on the bright core of a massive galaxy.

### Current Implementation (Simplified ZOGY-style)

```python
class SNDifferencingPipeline:
    """
    Pipeline stages:
    1. WCS alignment via reprojection
    2. Background estimation and subtraction
    3. PSF estimation (FWHM from star finding)
    4. PSF matching (convolve sharper image to match broader)
    5. Flux normalization (match median levels)
    6. Difference computation
    7. Significance map (diff / combined_noise)
    8. Source detection (>5σ peaks)
    """
```

### Pipeline Output

| Output | Description | Use |
|--------|-------------|-----|
| Difference Image | Science - Reference (flux units) | Visual inspection |
| Significance Map | SNR per pixel | Detection thresholding |
| Footprint Mask | Valid overlap region | Exclude edge artifacts |
| SN Mask | Known SN position mask | Training labels |

### Implementation Files

```
notebooks/differencing.ipynb           # Interactive pipeline development
scripts/generate_difference_images.py  # Production batch processing
output/difference_images/              # Generated training data
```

### Demonstrated Results

- **SN 2014J** (UUU filter): Max significance 2120σ, 73 candidate detections
- 8 same-mission SNe successfully differenced

---

## Phase 4: Source Extraction & Candidate Generation

### Objective
Detect significant sources in the difference image and package them as candidate alerts with all necessary context for classification.

### Source Detection

```python
from photutils.detection import DAOStarFinder

# Find peaks > 5σ above background
daofind = DAOStarFinder(
    fwhm=4.0,           # Expected PSF FWHM
    threshold=5.0*std,  # 5σ detection threshold
    sharplo=0.2,        # Reject extended/cosmic rays
    sharphi=1.0
)
sources = daofind(significance_map)
```

### The "Image Triplet" Package

Each detection is packaged with three cutout images:

| Image | Content | Purpose |
|-------|---------|---------|
| **Science** | New observation | Shows current state |
| **Reference** | Archival observation | Baseline (no transient) |
| **Difference** | Science - Reference | Highlights what changed |

**Typical cutout size**: 63×63 pixels centered on detection

### Candidate Features

Beyond images, compute features for each candidate:

| Feature | Description |
|---------|-------------|
| SNR | Peak significance |
| Sharpness | PSF-like vs. cosmic ray |
| Roundness | Symmetry measure |
| FWHM | Source size |
| Flux | Integrated brightness |
| Position (RA, Dec) | Celestial coordinates |
| Host offset | Distance to nearest galaxy |

### The "Bogus" Problem

**Challenge**: Not every 5σ peak is a real transient. False positives include:
- Cosmic ray hits
- Subtraction artifacts (dipoles)
- Bad pixels / hot pixels
- Satellite trails
- Image edges / vignetting
- Moving objects (asteroids)

**Reality**: False positives vastly outnumber true transients. A model that always predicts "bogus" might be 99.99% accurate but scientifically useless.

---

## Phase 5: Real/Bogus Classification (CNN)

### Objective
Train a convolutional neural network to classify candidate alerts as "real" (genuine transient) or "bogus" (artifact/noise).

### Why Deep Learning?

Earlier systems used Random Forest classifiers on engineered features (sharpness, roundness, etc.). Deep learning (CNNs) directly learns from pixel patterns in the image triplets:

| Approach | Pros | Cons |
|----------|------|------|
| Random Forest | Interpretable, fast training | Limited by human-designed features |
| CNN | Learns complex patterns, better accuracy | Needs more data, less interpretable |

### Model Architecture (Reference: Braai/ZTF)

```
Input: 3-channel image (63×63×3)
       [Science, Reference, Difference]

Conv2D(32, 3×3) → ReLU → MaxPool
Conv2D(64, 3×3) → ReLU → MaxPool
Conv2D(128, 3×3) → ReLU → MaxPool
Flatten
Dense(256) → ReLU → Dropout(0.5)
Dense(1) → Sigmoid

Output: Probability [0.0 = bogus, 1.0 = real]
```

### What the CNN Learns

| Real Transient | Bogus Artifact |
|----------------|----------------|
| Consistent PSF-like point source | Messy, asymmetric residuals |
| Present in science, absent in reference | Dipole pattern (subtraction error) |
| Round, centered in difference | Elongated, edge effects |
| Coherent across channels | Inconsistent appearance |

### Framework Options

| Framework | Notes |
|-----------|-------|
| TensorFlow/Keras | Used by ZTF/Braai, good Edge TPU support |
| PyTorch | More Pythonic, easier debugging |
| PyTorch Lightning | Structured training, good for experiments |

### Training Strategy

1. **Balanced sampling**: Upsample rare positives or use class weights
2. **Data augmentation**: Rotation, flipping (astronomical images are orientation-invariant)
3. **Multi-task learning**: Optionally predict transient type simultaneously
4. **Transfer learning**: Pre-train on large dataset, fine-tune on specific survey

---

## Phase 6: Training Data Strategy

### The Domain Shift Problem

**Critical Challenge**: If you train a model on images from one observing season (nice, dry, cold), it may fail on images from a different season (humid, different instrument state).

The model learns **instrument quirks**, not **transient physics**.

### Ensuring Generalization

#### 6.1 Representative Training Set

The training data must span the entire domain:

| Dimension | Variation to Include |
|-----------|---------------------|
| Time | All seasons, years |
| Instrument | All CCD chips, filters |
| Conditions | Good and bad seeing |
| Brightness | Faint and bright transients |
| Host | On/off galaxy, different galaxy types |

#### 6.2 Validation Strategies

| Strategy | Description |
|----------|-------------|
| **Temporal Split** | Train on past, test on future (mimics deployment) |
| **Cross-CCD** | Train on some CCDs, test on others |
| **Cross-Survey** | Ultimate test: train on Survey A, test on Survey B |

If your model works on a completely different telescope, you've learned real physics, not instrument artifacts.

#### 6.3 Active Learning

Modern frameworks (e.g., Astronomaly) incorporate human-in-the-loop feedback:

1. Model predicts on new data
2. Uncertain cases flagged for human review
3. Human labels fed back into training
4. Model improves iteratively

**Why this matters**: Even 99% automated systems benefit from occasional human correction to catch novel transient types and evolving instrument behavior.

### Training Data Sources

| Source | Type | Use |
|--------|------|-----|
| Confirmed spectroscopic SNe | Positive (real) | Ground truth transients |
| Human-vetted bogus | Negative | Common artifacts |
| Citizen science labels | Both | Large volume, some noise |
| Synthetic injection | Positive | Controlled properties |

---

## Phase 7: Model Deployment & Inference

### Objective
Deploy the trained model to classify alerts in real-time as new observations arrive.

### Inference Requirements

| Constraint | ZTF Example | Notes |
|------------|-------------|-------|
| Volume | ~1M alerts/night | Must scale |
| Latency | Minutes | Enable rapid follow-up |
| Cost | Limited budget | Efficient hardware |

### Hardware Options

| Hardware | Use Case | Notes |
|----------|----------|-------|
| CPU | Prototyping, small volume | Slowest but simplest |
| GPU | Training, batch inference | High throughput |
| Edge TPU | Real-time inference | Google's specialized chip, cost-efficient |

**ZTF uses Edge TPUs** for production inference—specialized for the matrix operations neural networks need.

### Deployment Considerations

1. **Model versioning**: Track which model produced which predictions
2. **Monitoring**: Alert if prediction distribution shifts
3. **Fallback**: Human review queue for edge cases
4. **Retraining**: Periodic updates as new labeled data accumulates

---

## Phase 8: Evaluation & Metrics

### The Class Imbalance Problem

**Searching for rare events**: If 1 in 10,000 candidates is a real supernova:
- A model predicting "bogus" always is 99.99% accurate
- But it finds **zero** supernovae—scientifically useless

**Accuracy is the wrong metric.**

### Correct Metrics

| Metric | Definition | Goal |
|--------|------------|------|
| **Precision** | TP / (TP + FP) | Of things I called real, how many are? |
| **Recall** | TP / (TP + FN) | Of real things, how many did I find? |
| **F1 Score** | Harmonic mean of precision/recall | Balance |
| **AUCPR** | Area Under Precision-Recall Curve | Overall performance |

### The False Negative vs. False Positive Trade-off

| Priority | Setting | Implication |
|----------|---------|-------------|
| **Minimize FN** | Permissive threshold | Find all transients, but more junk to review |
| **Minimize FP** | Strict threshold | Clean alerts, but risk missing events |

**ZTF's choice**: Minimize false negatives (set permissive threshold). Rationale:
- Missing a unique, once-in-a-lifetime event = **scientifically catastrophic**
- Reviewing extra false alarms = **efficiency cost** (acceptable)

### Retrospective Testing

To verify the model can find genuinely novel events:

1. Take archival data containing a known, unusual transient
2. **Remove** that event from training data entirely
3. Run the model on the test data
4. Verify the model still flags the event as anomalous

This proves the system can discover things it wasn't explicitly taught.

---

## Current Implementation Status

### ✅ Completed

| Component | Location | Notes |
|-----------|----------|-------|
| Supernova catalog | `resources/sncat_compiled.txt` | 6,542 SNe |
| Archive query scripts | `scripts/query_sn_fits_*.py` | MAST API integration |
| Download pipeline | `scripts/download_sn_fits.py` | With validation |
| Data organization | `scripts/organize_training_pairs.py` | Ref/Sci pairs |
| Differencing pipeline | `notebooks/differencing.ipynb` | Full implementation |
| Batch processing | `scripts/generate_difference_images.py` | Production ready |
| Training data structure | `output/fits_training/` | 19 SNe organized |
| Difference images | `output/difference_images/` | Generated outputs |

### 🚧 In Progress

| Component | Status | Next Steps |
|-----------|--------|------------|
| Scale data acquisition | 19 → 500+ SNe | Run batch downloads |
| Source extraction | Basic DAOStarFinder | Add cutout generation |
| Training triplets | Framework exists | Generate standardized cutouts |

### ❌ Not Yet Implemented

| Component | Priority | Approach |
|-----------|----------|----------|
| CNN classifier | High | TensorFlow/PyTorch model |
| Training pipeline | High | With augmentation, validation |
| Model evaluation | High | AUCPR, confusion matrix |
| Inference service | Medium | API endpoint for predictions |
| Active learning loop | Low | Human feedback integration |

---

## Next Steps & Roadmap

### Short-term (Phase 6A: Training Data)

1. **Scale data acquisition**
   - Download 500+ same-mission SN pairs
   - Focus on SWIFT data (best same-mission coverage)
   - Validate filter matching

2. **Generate training triplets**
   - Run differencing on all pairs
   - Extract 63×63 cutouts at SN positions
   - Generate corresponding labels (real=1 at known SN)

3. **Create bogus examples**
   - Random positions in difference images (no known SN)
   - Include edge artifacts, cosmic rays
   - Balance dataset (~50% real, ~50% bogus)

### Medium-term (Phase 5: Classification)

4. **Build CNN model**
   - Implement architecture (Braai-style)
   - Set up training pipeline with augmentation
   - Implement proper train/val/test splits

5. **Train and evaluate**
   - Monitor loss, precision, recall
   - Generate AUCPR curves
   - Test cross-survey generalization

### Long-term (Phase 7-8: Deployment)

6. **Production inference**
   - API endpoint for predictions
   - Batch processing capabilities
   - Monitoring and logging

7. **Active learning**
   - Human review interface
   - Feedback loop to model retraining
   - Novelty detection for unusual events

---

## References & Background

### Key Papers

1. **ZOGY Algorithm**: Zackay, Ofek, & Gal-Yam (2016). "Proper Image Subtraction—Optimal Transient Detection, Photometry, and Hypothesis Testing." [arXiv:1601.02655](https://arxiv.org/abs/1601.02655)

2. **Braai Classifier**: Duev et al. (2019). "Real-bogus classification for the Zwicky Transient Facility using deep learning." [MNRAS, 489, 3582](https://academic.oup.com/mnras/article/489/3/3582/5554758)

3. **ZTF Alert System**: Masci et al. (2019). "The Zwicky Transient Facility: Data Processing, Products, and Archive." [PASP, 131, 018003](https://iopscience.iop.org/article/10.1088/1538-3873/aae8ac)

### Software Dependencies

| Package | Purpose |
|---------|---------|
| `astropy` | FITS handling, WCS, coordinates |
| `photutils` | Source detection, background estimation |
| `reproject` | Image alignment/reprojection |
| `sep` | Fast source extraction (SExtractor in Python) |
| `tensorflow`/`pytorch` | Deep learning framework |

### Useful Resources

- [ZTF Data System](https://www.ztf.caltech.edu/ztf-public-releases.html)
- [Open Supernova Catalog](https://sne.space/)
- [MAST Portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)
- [Astronomaly Active Learning](https://github.com/MichelleLochner/astronomaly)

---

## Glossary

| Term | Definition |
|------|------------|
| **FITS** | Flexible Image Transport System—standard astronomical data format |
| **WCS** | World Coordinate System—maps pixels to sky coordinates |
| **PSF** | Point Spread Function—how a point source appears on detector |
| **FWHM** | Full Width at Half Maximum—measure of PSF size |
| **ZOGY** | Zackay-Ofek-Gal-Yam optimal image subtraction algorithm |
| **SNR** | Signal-to-Noise Ratio |
| **Transient** | Astronomical source that varies in brightness |
| **Real/Bogus** | Classification: genuine transient vs. artifact |
| **CNN** | Convolutional Neural Network |
| **AUCPR** | Area Under Precision-Recall Curve |
| **Edge TPU** | Google's tensor processing unit for edge inference |
| **MAST** | Mikulski Archive for Space Telescopes |
| **SWIFT** | NASA UV/X-ray telescope with UVOT imager |

---

*Document generated for the AstrID project. Based on deep learning pipelines for transient detection as used by ZTF, LSST, and other modern astronomical surveys.*

