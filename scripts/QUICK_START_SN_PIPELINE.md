# Quick Start: Supernova Pipeline Integration

## Step-by-Step Guide

### Step 1: Download FITS Files for Supernovae with Both Reference and Science

First, download the FITS files. Only 2010O and 2010P have both reference and science observations:

```bash
python scripts/download_sn_fits.py \
    --query-results output/sn_queries_2010s.json \
    --output-dir data/fits \
    --filter-has-both \
    --max-obs 3
```

This will download:
- **2010O**: 31 reference + 22 science observations
- **2010P**: 31 reference + 22 science observations

### Step 2: Verify Files Were Downloaded

Check that files exist:

```bash
ls -la data/fits/2010O/reference/
ls -la data/fits/2010O/science/
ls -la data/fits/2010P/reference/
ls -la data/fits/2010P/science/
```

### Step 3: Ingest into Pipeline

Now ingest the downloaded files. Use the TESS survey UUID from your database:

```bash
python scripts/ingest_sn_fits_to_pipeline.py \
    --query-results output/sn_queries_2010s.json \
    --fits-dir data/fits \
    --survey-id 49e8d057-184a-4239-9bff-9be72fbcfd02 \
    --filter-has-both
```

Or use HST survey UUID:

```bash
python scripts/ingest_sn_fits_to_pipeline.py \
    --query-results output/sn_queries_2010s.json \
    --fits-dir data/fits \
    --survey-id 05e6090c-bac5-4b78-8d7d-ae15a7dde50 \
    --filter-has-both
```

### Step 4: Monitor Pipeline Processing

The ingestion script automatically triggers:
1. **Preprocessing** → WCS alignment, image registration
2. **Differencing** → ZOGY algorithm on reference + science pairs
3. **ML Detection** → U-Net inference on difference images

Check your Dramatiq workers or database to see processing status.

## Troubleshooting

### "No FITS files found"

**Solution**: Run `download_sn_fits.py` first (Step 1)

### "No observations - skipping"

**Solution**: That supernova doesn't have observations. Use `--filter-has-both` to only process supernovae with both reference and science images.

### Wrong Survey UUID

Check your database for the correct survey UUID:
- HST: `05e6090c-bac5-4b78-8d7d-ae15a7dde50`
- JWST: `3ae172d0-c51a-4dad-8033-9813792ce503`
- TESS: `49e8d057-184a-4239-9bff-9be72fbcfd02`

## Expected Results

After successful ingestion:
- FITS files uploaded to R2 storage
- Observation records created in database
- Preprocessing tasks queued
- Difference images generated
- ML inference results stored

## Next Steps

1. **Extract Training Data**: Query difference images with labels
2. **Train Model**: Use difference images + ground truth labels
3. **Validate**: Test on held-out supernovae

