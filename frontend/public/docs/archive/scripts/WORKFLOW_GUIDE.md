# Supernova FITS Download & Ingestion Workflow

## Complete Workflow from Catalog to ML Pipeline

### Step 1: Query MAST for Observations

Query the supernova catalog to find available observations in MAST:

```bash
python3 scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --missions HST JWST \
    --min-year 2010 \
    --days-before 730 \
    --days-after 365 \
    --limit 100 \
    --output output/sn_queries.json
```

**What this does:**
- Reads supernova catalog with coordinates and discovery dates
- Queries MAST for observations BEFORE discovery (reference images)
- Queries MAST for observations AFTER discovery (science images)
- Saves results to JSON file

### Step 2: Preview Available Data (Dry Run)

Before downloading, check what's actually available:

```bash
python3 scripts/download_sn_fits.py \
    --query-results output/sn_queries.json \
    --output-dir data/fits \
    --dry-run \
    --limit 20
```

**Output example:**
```
============================================================
PRE-FILTERING QUERY RESULTS
============================================================
Mode: Require both reference AND science

Filtering results:
  Total supernovae in query: 100
  ✅ With both ref & sci (viable): 25
  ⚠️  Reference only: 15
  ⚠️  Science only: 10
  ❌ Neither: 50

  Supernovae to process: 25
============================================================
```

**Decision point:** If you have 0 viable SN with both ref & sci, try:
- Different missions: `--missions HST JWST` or `--no-mission-filter`
- Wider time windows: `--days-before 1095 --days-after 730`
- More recent SNe: `--min-year 2015`

### Step 3: Download FITS Files

Download only viable before/after pairs:

```bash
python3 scripts/download_sn_fits.py \
    --query-results output/sn_queries.json \
    --output-dir data/fits \
    --max-obs 5 \
    --limit 10
```

**What this does:**
- Filters to only SN with downloadable reference AND science observations
- Downloads up to 5 observations of each type per SN
- Validates downloaded FITS files have WCS and required metadata
- Saves detailed results to `data/fits/download_results.json`

**Output example:**
```
============================================================
DOWNLOAD SUMMARY
============================================================
Supernovae processed: 10
  ✅ Complete before/after pairs: 8

Files downloaded:
  Reference files: 20
  Science files: 18
  Total: 38

FITS validation:
  ✅ Valid files: 36
  ❌ Invalid files: 2
  Success rate: 94.7%

✅ Ready for differencing: 8 supernovae with complete before/after pairs
============================================================
```

### Step 4: Visualize FITS Files (Optional)

Before ingesting, visualize some files to confirm quality:

```python
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt

# Example: Visualize a reference/science pair
ref_file = Path("data/fits/2010SN/reference/some_ref.fits")
sci_file = Path("data/fits/2010SN/science/some_sci.fits")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

with fits.open(ref_file) as hdul:
    axes[0].imshow(hdul[0].data, cmap='gray', vmin=0, vmax=100)
    axes[0].set_title(f'Reference: {ref_file.name}')

with fits.open(sci_file) as hdul:
    axes[1].imshow(hdul[0].data, cmap='gray', vmin=0, vmax=100)
    axes[1].set_title(f'Science: {sci_file.name}')

plt.tight_layout()
plt.show()
```

### Step 5: Ingest into AstrID Pipeline

Upload FITS files to R2 and create observation records:

```bash
python3 scripts/ingest_sn_fits_to_pipeline.py \
    --query-results output/sn_queries.json \
    --fits-dir data/fits \
    --survey-id <your-survey-uuid> \
    --limit 10
```

**What this does:**
- Reads FITS metadata (RA, DEC, time, filter, etc.)
- Uploads FITS files to R2 storage
- Creates observation records in database
- Triggers preprocessing → differencing → ML pipeline

**Output example:**
```
============================================================
INGESTION SUMMARY
============================================================
Supernovae processed: 8
Reference observations ingested: 20
Science observations ingested: 18
Total errors: 0
============================================================

✅ Observations have been queued for preprocessing → differencing → ML pipeline
```

## Quick Reference Commands

### Query + Download in One Go

```bash
# Query MAST
python3 scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --missions HST JWST \
    --min-year 2010 \
    --limit 100 \
    --output output/sn_queries.json

# Dry run to preview
python3 scripts/download_sn_fits.py \
    --query-results output/sn_queries.json \
    --output-dir data/fits \
    --dry-run

# Download actual files
python3 scripts/download_sn_fits.py \
    --query-results output/sn_queries.json \
    --output-dir data/fits \
    --limit 10

# Ingest to pipeline
python3 scripts/ingest_sn_fits_to_pipeline.py \
    --query-results output/sn_queries.json \
    --fits-dir data/fits \
    --survey-id <uuid> \
    --limit 10
```

## Troubleshooting

### No viable supernovae found

```
⚠️  No supernovae to process after filtering!
```

**Solutions:**
1. Query different missions: `--missions HST JWST` or `--no-mission-filter`
2. Use wider time windows: `--days-before 1095 --days-after 730`
3. Query more recent SNe: `--min-year 2015`
4. Try more SNe: `--limit 500`

### All observations have filesize: 0

This means MAST catalogs the observations but doesn't have downloadable files.

**Solutions:**
1. Try HST/JWST specifically (better data availability)
2. Query more recent supernovae (2010+)
3. Consider alternative data sources (Pan-STARRS, ZTF)

### Invalid FITS files

```
⚠️  Invalid FITS: file.fits - Missing WCS
```

**This is OK** - The script filters these out automatically. Valid files are still processed.

### Not enough reference images

```
⚠️  SN has no reference observations
```

**Solutions:**
1. Increase `--days-before 1095` (3 years)
2. Use `--allow-partial` to process anyway (though won't have before/after pairs)

## File Structure

After running the pipeline:

```
data/fits/
├── download_results.json          # Download status for each SN
├── ingestion_results.json         # Ingestion status
├── 2010O/
│   ├── reference/
│   │   ├── obs_12345.fits
│   │   └── obs_12346.fits
│   └── science/
│       ├── obs_12347.fits
│       └── obs_12348.fits
├── 2010P/
│   ├── reference/
│   │   └── obs_12349.fits
│   └── science/
│       └── obs_12350.fits
...
```

## Performance Tips

1. **Use --dry-run first** - Avoid wasting time on unavailable data
2. **Start with --limit 10** - Test pipeline before large batches
3. **Filter by year** - Recent SNe (2010+) have better coverage
4. **Choose missions wisely** - HST/JWST have better data availability than SDSS
5. **Verify before ingesting** - Check `download_results.json` for complete pairs

## Next Steps After Ingestion

Once observations are ingested, the AstrID pipeline automatically:

1. **Preprocessing** - Calibration, background subtraction
2. **Differencing** - ZOGY algorithm to create difference images
3. **ML Detection** - CNN model identifies transients
4. **Classification** - Classifies detected objects

Monitor the pipeline through the API or web interface.





