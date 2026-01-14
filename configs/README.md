# Pipeline Configuration Files

This directory contains YAML configuration files for generating mission-specific supernova training datasets.

## Available Configurations

### 1. `swift_uv_dataset.yaml`
**SWIFT UVOT UV-band dataset**
- Mission: SWIFT only
- Filters: UV filters (uuu, uvw1, uvm2, uvw2)
- Time range: 2005-2020
- Use case: UV-bright transients, early-time supernova detection

### 2. `ps1_optical_dataset.yaml`
**Pan-STARRS1 optical dataset**
- Mission: PS1 only
- Filters: All optical bands (g, r, i, z, y)
- Time range: 2010-2020
- Use case: Optical transients, multi-band photometry

### 3. `galex_uv_dataset.yaml`
**GALEX UV dataset**
- Mission: GALEX only
- Filters: FUV and NUV
- Time range: 2005-2020
- Use case: Far-UV transients, wide-field surveys

### 4. `multi_mission_dataset.yaml`
**Multi-mission comprehensive dataset**
- Missions: SWIFT, PS1, GALEX
- Filters: All available
- Time range: 2005-2020
- Use case: Transfer learning, ensemble models, mission comparison

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
