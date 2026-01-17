# Pipeline Configuration Files

This directory contains YAML configuration files for generating mission-specific supernova training datasets.

## Available Configurations

### Recommended Starting Points

#### `best_yield_combined.yaml` ⭐ RECOMMENDED
**Maximum yield training dataset**
- Missions: GALEX, SWIFT, HST
- Time range: 2005-2011 (GALEX golden era)
- Use case: **Best config for building a large training set**
- Why: All three missions operational, maximizes usable pairs

#### `galex_golden_era.yaml` ⭐ BEST FOR UV PAIRS
**GALEX when it was fully operational**
- Mission: GALEX only
- Time range: 2005-2011
- Use case: Maximum UV difference imaging pairs
- Why: GALEX active for BOTH reference and science windows

### Mission-Specific Configurations

#### `swift_uv_dataset.yaml`
**SWIFT UVOT UV-band dataset**
- Mission: SWIFT only
- Filters: UV filters (uuu, uvw1, uvm2, uvw2)
- Time range: 2005-2020
- Use case: UV-bright transients, early-time supernova detection

#### `swift_continuous.yaml`
**SWIFT with 20 years of continuous coverage**
- Mission: SWIFT only
- Time range: 2005-2024
- Use case: Any-era supernovae with UV follow-up
- Why: SWIFT is the ONLY UV mission with continuous 20-year coverage

#### `galex_uv_dataset.yaml`
**GALEX UV dataset**
- Mission: GALEX only
- Filters: FUV and NUV
- Time range: 2005-2020
- Use case: Far-UV transients, wide-field surveys
- Note: GALEX ended 2013; post-2013 SNe have no science images

#### `ps1_optical_dataset.yaml`
**Pan-STARRS1 optical dataset**
- Mission: PS1 only
- Filters: All optical bands (g, r, i, z, y)
- Time range: 2010-2020
- Use case: Optical transients, multi-band photometry
- Note: PS1 stacked data often stored as HLSP in MAST

#### `hst_high_resolution.yaml`
**HST high-resolution imaging**
- Mission: HST only
- Time range: 2000-2024
- Use case: Nearby, well-studied supernovae
- Why: Highest resolution; best for famous SNe (2014J, 2011fe, etc.)

#### `tess_allsky_modern.yaml`
**TESS all-sky for modern SNe**
- Mission: TESS only
- Time range: 2018-2025
- Use case: Recent supernovae with all-sky coverage
- Note: Large pixels (21"/pixel); best for bright, nearby SNe

#### `sdss_optical_references.yaml`
**SDSS for ground-based reference images**
- Missions: SDSS + SWIFT + HST
- Time range: 2010-2020 (SNe), with 10-year lookback for SDSS refs
- Use case: Cross-mission pairs using SDSS pre-explosion imaging
- Note: Allow cross-mission pairs (SDSS ref + space-based science)

### IRSA Archive Configurations (ZTF, PTF, etc.)

#### `ztf_modern_transients.yaml` ⭐ NEW
**ZTF optical transients for modern supernovae**
- Mission: ZTF only
- Archive: IRSA (not MAST)
- Time range: 2018-2025
- Use case: Modern optical transients with excellent cadence
- Why: ZTF observes ~3750 deg²/night with 30s exposures in g, r, i bands

#### `ztf_ptf_optical.yaml` ⭐ NEW
**ZTF + PTF combined optical coverage**
- Missions: ZTF, PTF
- Archive: IRSA (not MAST)
- Time range: 2009-2025
- Use case: Continuous optical coverage from 2009 to present
- Why: PTF (2009-2017) + ZTF (2018+) provides 16+ years of data

### Combined/Legacy Configurations

#### `multi_mission_dataset.yaml`
**Multi-mission comprehensive dataset**
- Missions: SWIFT, PS1, GALEX
- Filters: All available
- Time range: 2005-2020
- Use case: Transfer learning, ensemble models, mission comparison

#### `extended_sn_dataset.yaml`
**Extended 2011-2020 dataset**
- Missions: GALEX, SWIFT, PS1
- Time range: 2011-2020
- Use case: Adding recent SNe to existing datasets
- Note: Limited GALEX pairs (mission ended 2013)

#### `sn2014j_only.yaml`
**SN 2014J focused (test config)**
- Missions: GALEX, SWIFT, PS1, HST
- Time range: 2014 only
- Use case: Testing with famous nearby SN Ia
- Note: GALEX won't work (ended 2013); use SWIFT/HST

## Archive Support

The pipeline now supports **both MAST and IRSA archives**:

- **MAST**: HST, JWST, TESS, GALEX, SWIFT, PS1, SDSS, HLA, HLSP
- **IRSA**: ZTF, PTF, WISE, NEOWISE, 2MASS, Spitzer

Missions are automatically routed to the correct archive. You can also explicitly specify archives in config:

```yaml
query:
  missions: ["ZTF", "GALEX", "HST"]  # Auto-routed: ZTF→IRSA, GALEX/HST→MAST
  archives: ["MAST", "IRSA"]        # Optional: explicit archive selection
```

## Mission Availability Timeline

```
Mission   2000  2005  2010  2015  2020  2025
GALEX     [----=========-------]              (2003-2013) [MAST]
SWIFT           [===========================> (2004-present) [MAST]
PS1                  [=====----]              (2010-2014 main survey) [MAST]
HST       [===============================>   (1990-present) [MAST]
TESS                           [============> (2018-present) [MAST]
SDSS      [========]                          (2000-2009 imaging) [MAST]
JWST                                [======>  (2022-present) [MAST]
PTF                  [========]                (2009-2017) [IRSA]
ZTF                           [============>  (2018-present) [IRSA]

= = Active with good coverage
- = Limited/degraded coverage
```

**Key Insights**: 
- For UV difference imaging, target 2005-2011 supernovae where GALEX was operational for both pre- and post-SN observations.
- For modern optical transients, use ZTF (2018+) or ZTF+PTF (2009+) for continuous coverage.

## Usage

### Basic Usage

```bash
# Generate SWIFT UV dataset
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml

# Generate PS1 optical dataset
python scripts/run_pipeline_from_config.py --config configs/ps1_optical_dataset.yaml
```

### Advanced Options

```bash
# Dry run (preview without executing)
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --dry-run

# Resume from checkpoint
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --resume

# Run specific stage only
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --stage download

# Use symlinks (saves disk space)
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --symlink

# Decompress FITS files
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --decompress

# Generate visualizations
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --visualize
```

## Configuration Structure

Each YAML file contains the following sections:

### 1. Dataset Metadata
```yaml
dataset_name: "swift_uv_supernovae"
description: "SWIFT UVOT UV-band supernova training set"
```

### 2. Query Parameters
```yaml
query:
  missions: ["SWIFT"]           # Space telescope missions
  filters: ["uuu", "uvw1"]      # Specific filters (optional)
  min_year: 2005                # Discovery year range
  max_year: 2020
  days_before: 1095             # Reference image window
  days_after: 730               # Science image window
  radius_deg: 0.1               # Spatial search radius
  chunk_size: 250               # Batch size for processing
  start_index: 0                # Skip first N entries
  limit: null                   # Limit total SNe (null = all)
```

### 3. Download Parameters
```yaml
download:
  max_obs_per_type: 5           # Max observations per type (ref/sci)
  max_products_per_obs: 3       # Max FITS files per observation
  include_auxiliary: false      # Include masks/weights (increases size)
  require_same_mission: true    # Only same-mission pairs (critical!)
  verify_fits: true             # Validate WCS after download
```

### 4. Quality Filters
```yaml
quality:
  min_overlap_fraction: 0.85    # Minimum WCS overlap for differencing
  max_file_size_mb: 500         # Skip files larger than this
  verify_wcs: true              # Check for valid coordinates
```

### 5. Output Paths
```yaml
output:
  query_results: "output/datasets/swift_uv/queries.json"
  fits_downloads: "output/datasets/swift_uv/fits_downloads"
  fits_training: "output/datasets/swift_uv/fits_training"
  difference_images: "output/datasets/swift_uv/difference_images"
  checkpoint: "output/datasets/swift_uv/checkpoint.json"
  chunk_dir: "output/datasets/swift_uv/chunks"
```

## Creating Custom Configurations

To create a custom configuration:

1. Copy an existing config file:
   ```bash
   cp configs/swift_uv_dataset.yaml configs/my_custom_dataset.yaml
   ```

2. Edit the parameters:
   - Change `dataset_name` and `description`
   - Adjust `missions` and `filters` for your use case
   - Modify temporal parameters (`days_before`, `days_after`)
   - Update output paths

3. Validate the configuration:
   ```bash
   python scripts/run_pipeline_from_config.py --config configs/my_custom_dataset.yaml --dry-run
   ```

4. Run the pipeline:
   ```bash
   python scripts/run_pipeline_from_config.py --config configs/my_custom_dataset.yaml
   ```

## Important Notes

### Same-Mission Requirement
The `require_same_mission: true` parameter is **critical**. Cross-mission image pairs (e.g., SWIFT reference + PS1 science) are unusable for differencing due to:
- Different PSFs (point spread functions)
- Different pixel scales
- Different filter systems
- Different noise characteristics

Only same-mission pairs (SWIFT-SWIFT, PS1-PS1, etc.) produce meaningful difference images.

### Disk Space Considerations

Typical dataset sizes:
- **With auxiliary files**: ~1-2 GB per SN
- **Without auxiliary files** (default): ~300-500 MB per SN
- **With symlinks**: Minimal additional space
- **With decompression**: ~3x larger than compressed

For 50 SNe:
- Compressed, no auxiliary: ~15-25 GB
- Decompressed, no auxiliary: ~45-75 GB
- With auxiliary: ~50-100 GB

### Processing Time

Approximate times (varies by network and MAST load):
- **Query stage**: 1-2 SNe per second (~5 min for 500 SNe)
- **Download stage**: 2-5 min per SN (~2-4 hours for 50 SNe)
- **Organize stage**: <1 min for 50 SNe
- **Differencing stage**: 1-2 min per SN (~1-2 hours for 50 SNe)

**Total for 50 SNe**: 3-6 hours

## Troubleshooting

### Configuration Validation Errors
```bash
# Check configuration validity
python -c "from src.pipeline.config import PipelineConfig; config = PipelineConfig.from_yaml('configs/swift_uv_dataset.yaml'); print(config.validate())"
```

### Resume from Checkpoint
If pipeline fails mid-execution:
```bash
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --resume
```

### Run Specific Stage
To re-run a single stage:
```bash
python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --stage download
```

## See Also

- [Data Pipeline Documentation](../docs/research/DATA_PIPELINE.md) - Technical details
- [Pipeline Scripts](../scripts/) - Individual stage scripts
- [Example Notebook](../notebooks/dataset_generation_example.ipynb) - Interactive examples
