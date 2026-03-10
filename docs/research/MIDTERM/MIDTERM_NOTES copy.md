# AstrID Midterm Progress Notes

> Key features, lessons learned, and insights extracted from the supernova training pipeline development.

---

## Pipeline Development Progress

### Data Acquisition & Organization

- **Supernova catalog compiled**: 6,542 known supernovae with RA/Dec coordinates and metadata
- **Pilot dataset downloaded**: 19 SNe with reference (pre-SN) and science (post-SN) images from MAST archive
- **Data organization**: Structured into `output/fits_training/{SN_NAME}/reference/` and `/science/` directories
- **Training manifest**: JSON file tracking all downloaded files and their metadata

### Critical Discovery: Same-Mission Requirement

- **Only 8 of 19 SNe (42%) had usable same-mission pairs** (all SWIFT-SWIFT)
- **11 SNe had cross-mission data** (e.g., PS1 reference + SWIFT science) — **not suitable for differencing**
- **Root cause**: Different telescopes have different PSFs, filters, pixel scales, and noise characteristics
- **Lesson**: Must query archives specifically for same-mission temporal coverage

### Filter Matching Within Missions

- Even within the same mission, **filter matching is critical**
- SWIFT UVOT filters: UVW2, UVM2, UVW1, UUU, UBB, UVV
- **Initial mistake**: Compared UVW1 reference with UVM2 science — different wavelengths, meaningless difference
- **Solution**: Group files by filter, only pair matching filters (e.g., UUU-UUU)

---

## Technical Pipeline Components

### 1. FITS File Handling

- Used `astropy.io.fits` for reading astronomical data
- **Challenge**: Some FITS files have data in extensions, not primary HDU
- **Solution**: Iterate through HDUs to find first with 2D image data
- **3D data handling**: Some images are 3D (e.g., multiple exposures) — take first slice
- **Archive data is pre-calibrated**: No need for bias/flat correction on SWIFT data

### 2. WCS Alignment (Critical Step)

- **Why it matters**: Pixel-by-pixel subtraction requires sub-pixel alignment
- **Misalignment artifacts**: Even 1-pixel offset creates massive dipole residuals around every source
- **Implementation**: Used `reproject.reproject_interp()` to transform science image onto reference WCS grid
- **Metric**: Track overlap fraction — ideally 90-100%
- **Result**: Achieved 93.6% to 100% overlap on processed SNe

### 3. Background Estimation & Subtraction

- Used `photutils.Background2D` with `MedianBackground` estimator
- **Box size**: 50-64 pixels for local background estimation
- **Why needed**: Images have varying sky background levels; must subtract before comparison
- **Fallback**: Simple sigma-clipped median for cases where Background2D fails

### 4. PSF Estimation & Matching

- **Estimated PSF FWHM** from bright stars using `DAOStarFinder`
- Typical SWIFT UVOT FWHM: ~3.5-4.0 pixels
- **PSF matching approach**: Convolve the sharper image with a Gaussian kernel to match the broader PSF
- **Kernel calculation**: σ_kernel = √(σ_target² - σ_current²)
- Used `astropy.convolution.convolve_fft` for efficient Fourier-space convolution

### 5. Flux Normalization

- Match median flux levels between science and reference in overlap region
- **Robust statistics**: Used sigma-clipped median to ignore outliers (stars, artifacts)
- Compute scale factor and offset to align flux distributions

### 6. Difference Image Computation

- Simple subtraction after all preprocessing: `difference = science_normalized - reference_matched`
- **Significance map**: `SNR = difference / sqrt(sci_noise² + ref_noise²)`
- Significance map has uniform noise properties — enables consistent thresholding

### 7. Source Detection

- Used `photutils.DAOStarFinder` on significance map
- **Detection threshold**: 5σ above background
- **Shape filters**: `sharplo=0.2`, `sharphi=1.0` to reject cosmic rays and extended sources
- **Output**: Candidate positions with flux, SNR, sharpness, roundness metrics

---

## Key Results Achieved

### SN 2014J (Brightest SN in Decades)

- **Filter**: UUU (U-band)
- **Reference date**: 2012-06-22 (pre-supernova)
- **Science date**: 2014-10-10 (during supernova)
- **Overlap**: 100%
- **Maximum significance**: 2120σ — **supernova clearly detected!**
- **Candidate detections**: 73 (including SN and other variable sources)
- **Known SN pixel position**: (447.9, 555.4) — successfully marked

### Batch Processing Results

| SN Name | Filter | Overlap | Max Significance | Detections |
|---------|--------|---------|------------------|------------|
| 2013gc  | uuu    | 93.6%   | 797.8σ           | 629        |
| 2014J   | uuu    | 100.0%  | 2120.9σ          | 73         |
| 2014ai  | uuu    | 100.0%  | 1168.0σ          | 206        |
| 2014bh  | uuu    | 99.4%   | 844.5σ           | 89         |
| 2014bi  | uvv    | 96.4%   | 412.2σ           | 162        |
| 2014L   | various| various | various          | various    |

---

## Lessons Learned

### Data Quality Lessons

1. **Archive data varies significantly** — some SNe have excellent temporal coverage, others have none
2. **Same-mission pairs are rare** — need to specifically query for temporal overlap within single missions
3. **Filter information sometimes missing** — filenames don't always encode filter; must check FITS headers
4. **FITS header deprecations are common** — `RADECSYS` → `RADESYSa` warnings can be safely ignored
5. **Image sizes vary** — reference and science often have different dimensions; WCS reprojection handles this

### Processing Lessons

1. **Simple subtraction is inadequate** — creates massive artifacts without PSF matching
2. **Alignment must be sub-pixel accurate** — even small offsets create dipole artifacts
3. **Background subtraction is essential** — varying sky levels swamp transient signals
4. **PSF matching dramatically improves results** — convolving to common PSF eliminates bright-star residuals
5. **Significance maps are more useful than raw difference** — uniform noise enables consistent thresholding

### Detection Lessons

1. **Many detections are NOT real transients** — cosmic rays, subtraction artifacts, variable stars
2. **5σ threshold yields many false positives** — need ML classifier to separate real from bogus
3. **High significance ≠ supernova** — host galaxy nuclei, bright stars can have high residuals
4. **Detection count varies wildly** — 73 to 629 candidates per image
5. **Known SN position is essential for training labels** — need ground truth for supervised learning

---

## Architecture & Code Organization

### Pipeline Class: `SNDifferencingPipeline`

```python
class SNDifferencingPipeline:
    """
    Reusable supernova image differencing pipeline.
    
    Stages:
    1. Load FITS (with HDU fallback)
    2. WCS alignment via reprojection  
    3. Background estimation/subtraction
    4. PSF estimation from stars
    5. PSF matching (convolve sharper → broader)
    6. Flux normalization (robust median matching)
    7. Difference computation
    8. Significance map generation
    9. Source detection (DAOStarFinder)
    """
```

### Data Structures

- **`DifferencingResult`**: Dataclass containing all outputs (images, WCS, metrics, detections)
- **Training manifest**: JSON tracking all SNe and their reference/science files
- **Results JSON**: Saved metrics and metadata for all processed SNe

### Key Dependencies

- `astropy` — FITS I/O, WCS, coordinates, visualization, statistics
- `photutils` — Background estimation, source detection
- `reproject` — WCS-based image alignment
- `scipy.ndimage` / `astropy.convolution` — PSF matching via convolution
- `numpy` / `matplotlib` — Core numerical/visualization

---

## Next Steps Identified

### Immediate (Training Data Generation)

1. **Scale data acquisition**: 19 → 500+ same-mission SNe pairs
2. **Generate image triplets**: 63×63 pixel cutouts of (science, reference, difference) at detection positions
3. **Create labeled dataset**: Real=1 at known SN positions, Bogus=0 at random/artifact positions
4. **Balance classes**: ~50% real, ~50% bogus for training

### Short-term (CNN Classifier)

1. Build CNN architecture (3-channel input: sci/ref/diff triplets)
2. Implement training pipeline with augmentation (rotation, flipping)
3. Proper train/validation/test splits (temporal or spatial)
4. Evaluate with AUCPR, precision, recall (not accuracy!)

### Long-term (Deployment)

1. Real-time inference API
2. Active learning loop for human feedback
3. Generalization testing across different surveys

---

## Technical Insights for Paper

### Why Classical + ML Hybrid Approach

- **Classical algorithms are robust and well-understood** — decades of refinement in astronomy
- **ML excels at pattern recognition** — distinguishing real transients from artifacts in image triplets
- **Best of both worlds**: Use classical for data preparation, ML for classification

### The False Positive Problem

- At 5σ threshold, false positives vastly outnumber true transients
- A model that always predicts "bogus" would be ~99.99% accurate but scientifically useless
- **Correct metrics**: Precision, Recall, F1, AUCPR — NOT accuracy
- **ZTF's approach**: Minimize false negatives (permissive threshold) because missing a unique event is catastrophic

### Domain Shift Challenge

- Models trained on one season/instrument may fail on different conditions
- Training data must span: different times, CCD chips, filters, seeing conditions, brightness levels
- **Ultimate test**: Train on Survey A, test on Survey B — proves physics learned, not instrument artifacts

---

## Quantitative Summary

| Metric | Value |
|--------|-------|
| Total SNe in catalog | 6,542 |
| SNe downloaded (pilot) | 19 |
| Same-mission pairs (usable) | 8 (42%) |
| Cross-mission pairs (unusable) | 11 (58%) |
| **Production dataset pairs** | **222** |
| **Production FITS files** | **2,282** |
| Best detection significance | 2120σ (SN 2014J) |
| Typical overlap achieved | 93-100% |
| Candidate detections per image | 73-629 |
| PSF FWHM (SWIFT UVOT) | ~3.5-4.0 px |

---

## Production Pipeline System

### Modular Pipeline Architecture

Following the pilot study, a production-ready pipeline system was developed with YAML-based configuration for reproducible dataset generation.

**Key Components**:
- **Configuration System** (`src/pipeline/config.py`): Dataclasses with validation for all pipeline parameters
- **Unified Runner** (`scripts/run_pipeline_from_config.py`): Orchestrates all 5 stages from YAML config
- **Mission-Specific Configs**: Pre-configured YAML files for SWIFT, PS1, GALEX, and multi-mission datasets

**Pipeline Stages**:
1. **Query** (`query_sn_fits_chunked.py`): Chunked MAST queries with checkpointing
2. **Filter** (`identify_same_mission_pairs.py`): Same-mission pair identification
3. **Download** (`download_sn_fits.py`): Smart filtering with 60-80% size reduction
4. **Organize** (`organize_training_pairs.py`): Training-ready directory structure
5. **Differencing** (`generate_difference_images.py`): Full astronomical differencing pipeline

### Production Dataset Results

**Current Production Dataset** (222 SNe):
- **FITS files**: 2,282 (1,170 reference + 1,112 science)
- **Files decompressed**: 1,542 (.fits.gz → .fits)
- **Difference images**: In progress (SWIFT: 17 pairs completed, GALEX: 105 pairs processing)
- **Mission breakdown**: SWIFT (17+ pairs), GALEX (105 pairs), PS1 (additional pairs)
- **Quality**: 93.6% average overlap, all >85%

### Reproducibility & Scalability

**YAML Configuration Example**:
```yaml
dataset_name: "swift_uv_supernovae"
query:
  missions: ["SWIFT"]
  filters: ["uuu", "uvw1", "uvm2", "uvw2"]
  min_year: 2005
  days_before: 1095
  days_after: 730
download:
  max_obs_per_type: 5
  include_auxiliary: false
  require_same_mission: true
```

**Usage**:
```bash
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml
```

**Benefits**:
- Version-controlled configurations
- Reproducible datasets
- Easy mission-specific dataset generation
- Checkpoint/resume support
- Dry-run mode for testing

### Documentation

Comprehensive technical documentation created:
- **[DATA_PIPELINE.md](DATA_PIPELINE.md)**: Full technical details of all 5 stages
- **[configs/README.md](../../configs/README.md)**: Configuration guide and examples
- **[dataset_generation_example.ipynb](../../notebooks/dataset_generation_example.ipynb)**: Interactive tutorial

---

*Notes compiled from `notebooks/supernova_training_pipeline.ipynb` and production pipeline development — January 2026*


