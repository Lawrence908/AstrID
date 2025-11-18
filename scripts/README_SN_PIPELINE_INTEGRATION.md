# Supernova Catalog Pipeline Integration

This guide explains how to integrate the supernova catalog workflow into the main AstrID data pipeline for ML training.

## Complete Workflow

```
1. Query SN Catalog → 2. Download FITS → 3. Ingest to Pipeline → 4. Process → 5. ML Training
```

## Step-by-Step Process

### Step 1: Query MAST for Supernova Observations

Query MAST to find reference and science images for supernovae:

```bash
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --min-year 2010 \
    --limit 10 \
    --missions HST JWST TESS \
    --days-before 1095 \
    --output output/sn_queries.json
```

**Output**: JSON file with observation metadata

### Step 2: Download FITS Files

Download the actual FITS files from MAST:

```bash
python scripts/download_sn_fits.py \
    --query-results output/sn_queries.json \
    --output-dir data/fits \
    --filter-has-both \
    --max-obs 3
```

**Output**: FITS files in `data/fits/{sn_name}/reference/` and `data/fits/{sn_name}/science/`

### Step 3: Ingest into AstrID Pipeline

Ingest the downloaded FITS files into the main pipeline:

```bash
python scripts/ingest_sn_fits_to_pipeline.py \
    --query-results output/sn_queries.json \
    --fits-dir data/fits \
    --survey-id <your-survey-uuid> \
    --filter-has-both
```

**What this does**:
- Uploads FITS files to R2 storage
- Creates observation records in the database
- Marks observations as "ingested"
- **Triggers preprocessing workflow** (which cascades to differencing → ML)

### Step 4: Pipeline Processing (Automatic)

Once ingested, the pipeline automatically:

1. **Preprocessing** (Dramatiq worker)
   - WCS alignment
   - Image registration
   - Calibration (if available)

2. **Differencing** (Dramatiq worker)
   - Finds reference image (from SN catalog or historical)
   - Applies ZOGY algorithm
   - Creates difference image

3. **ML Detection** (Dramatiq worker)
   - Runs U-Net inference on difference image
   - Scores detections
   - Stores results

4. **Validation** (Human review)
   - Review detections
   - Assign labels (valid/invalid/type)

## Integration Points

### Database Schema

Observations are stored with:
- `observation_type`: "reference" or "science"
- `sn_name`: Supernova name (custom metadata)
- `fits_url`: R2 path to FITS file
- Standard observation fields (RA, Dec, filter, exposure time, etc.)

### R2 Storage Structure

```
observations/
  {sn_name}/
    reference/
      {obs_id}.fits
    science/
      {obs_id}.fits
```

### Pipeline Triggers

The ingestion script triggers:
1. `preprocess_observation` → Preprocessing worker
2. Preprocessing triggers → `create_difference_image` → Differencing worker
3. Differencing triggers → ML inference → Detection worker

## Creating Training Data

After processing, you can extract training data:

1. **Difference images** are stored in R2
2. **Detection results** are in the database
3. **Human labels** (from validation) provide ground truth

Extract for ML training:
```python
# Query difference images with labels
# Filter by:
# - Has reference AND science images
# - Has human validation labels
# - Quality metrics above threshold
```

## Example: Complete Workflow

```bash
# 1. Query
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --min-year 2010 \
    --limit 5 \
    --missions HST \
    --output output/sn_queries.json

# 2. Download
python scripts/download_sn_fits.py \
    --query-results output/sn_queries.json \
    --output-dir data/fits \
    --filter-has-both \
    --max-obs 2

# 3. Ingest
python scripts/ingest_sn_fits_to_pipeline.py \
    --query-results output/sn_queries.json \
    --fits-dir data/fits \
    --survey-id $(uuidgen) \
    --filter-has-both

# 4. Monitor processing
# Check Dramatiq dashboard or logs for:
# - Preprocessing status
# - Differencing status
# - ML inference results
```

## Pipeline Diagram

See `docs/diagrams/data-flow-pipeline.puml` for the complete flow diagram, which now includes:
- Supernova catalog querying
- Reference/science image pairing
- Integration with existing pipeline

## Troubleshooting

### No Observations Ingested

- Check that FITS files were downloaded: `ls data/fits/{sn_name}/`
- Verify survey UUID exists in database
- Check R2 credentials and permissions

### Preprocessing Not Triggered

- Check Dramatiq worker is running
- Verify observation was marked as "ingested"
- Check worker logs for errors

### Differencing Fails

- Ensure reference image exists
- Check WCS alignment in both images
- Verify image sizes are compatible

### ML Inference Issues

- Check difference image quality
- Verify model is loaded
- Check inference worker logs

## Next Steps

1. **Scale up**: Process more supernovae from catalog
2. **Quality filtering**: Filter by image quality metrics
3. **Label extraction**: Extract validated detections for training
4. **Model training**: Use difference images + labels to train/retrain model

