# Quick Start: Using Compiled Supernova Catalog

This guide helps you query and download FITS files using the compiled supernova catalog.

## Overview

The compiled catalog (`resources/sncat_compiled.txt`) is now the default catalog used by the query script. It contains the same entries as the original catalog, sorted by discovery date (newest first).

## Step-by-Step Workflow

### Option 1: Automated Script (Recommended)

Use the automated workflow script:

```bash
# Basic usage - queries and downloads up to 1000 supernovae from 2010 onwards
./scripts/query_and_download_all.sh

# Custom parameters
./scripts/query_and_download_all.sh \
    resources/sncat_compiled.txt \
    data/fits \
    output/sn_queries_compiled.json \
    2000 \  # limit
    2015    # min year
```

### Option 2: Manual Steps

#### Step 1: Query MAST for Observations

Query the compiled catalog to find available FITS files:

```bash
python3 scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_compiled.txt \
    --output output/sn_queries_compiled.json \
    --min-year 2010 \
    --limit 1000 \
    --missions HST JWST TESS GALEX PS1 SWIFT \
    --days-before 1095 \
    --days-after 730
```

**What this does:**
- Reads the compiled catalog
- Queries MAST for observations before discovery (reference images)
- Queries MAST for observations after discovery (science images)
- Saves results to JSON file

**Expected time:** 10-30 minutes depending on catalog size

#### Step 2: Preview Available Data (Dry Run)

Before downloading, check what's available:

```bash
python3 scripts/download_sn_fits.py \
    --query-results output/sn_queries_compiled.json \
    --output-dir data/fits \
    --dry-run \
    --require-both \
    --limit 20
```

**Look for:**
```
✅ With both ref & sci (viable): XX
```

If you see 0 viable, try:
- Different missions: `--missions HST JWST`
- Wider time windows: `--days-before 1825 --days-after 1095`
- More recent SNe: `--min-year 2015`

#### Step 3: Download FITS Files

Download only viable before/after pairs:

```bash
python3 scripts/download_sn_fits.py \
    --query-results output/sn_queries_compiled.json \
    --output-dir data/fits \
    --require-both \
    --max-obs 3 \
    --verify-fits
```

**What this does:**
- Filters to only SN with downloadable reference AND science observations
- Downloads up to 3 observations per type per supernova
- Verifies FITS files are valid
- Organizes files in `data/fits/{sn_name}/reference/` and `data/fits/{sn_name}/science/`

**Expected time:** 1-4 hours depending on number of files and network speed

#### Step 4: Verify Downloads

Check what was downloaded:

```bash
# Count total FITS files
find data/fits -name "*.fits*" | wc -l

# Check file sizes
du -sh data/fits/*/

# List some example files
find data/fits -name "*.fits*" | head -10
```

#### Step 5: Ingest into Pipeline

Ingest the downloaded FITS files into your pipeline:

```bash
python3 scripts/ingest_sn_fits_to_pipeline.py \
    --query-results output/sn_queries_compiled.json \
    --fits-dir data/fits \
    --survey-id <your-survey-uuid> \
    --filter-has-both \
    --verify-wcs
```

**What this does:**
- Uploads FITS files to R2 storage
- Creates observation records in the database
- Marks observations as "ingested"
- Triggers preprocessing workflow

## Expected Results

With the compiled catalog, you should see:
- **More recent supernovae** (sorted newest first)
- **Same total entries** (~6,500)
- **Better coverage** for recent discoveries (2010+)

From a query of 1000 supernovae, expect:
- **~500-700 viable** (with both ref & sci)
- **~100-200 reference only**
- **~50-100 science only**
- **~100-200 neither**

## Troubleshooting

### Query Takes Too Long

- Reduce `--limit` to test with fewer supernovae first
- Use `--min-year` to focus on recent discoveries
- Try fewer missions: `--missions HST JWST`

### No Viable Supernovae

- Check that missions have data for your time range
- Widen time windows: `--days-before 1825 --days-after 1095`
- Try different missions
- Remove `--min-year` to include older supernovae

### Download Fails

- Check network connection
- Verify MAST API is accessible
- Try downloading fewer files: `--limit 10`
- Check disk space: `df -h`

### Files Are Too Large

- Reduce `--max-obs` to download fewer observations per supernova
- Some missions (like SDSS) have very large files - consider filtering them out

## Next Steps

After downloading and ingesting:

1. **Monitor preprocessing**: Check that observations are being processed
2. **Check differencing**: Verify ZOGY differencing is working
3. **Review ML pipeline**: Ensure training data is being generated
4. **Scale up**: Once working, increase `--limit` to process more supernovae

## Notes

- The compiled catalog uses the same format as the original
- Default catalog path in `query_sn_fits_from_catalog.py` is now `sncat_compiled.txt`
- All existing scripts work with the compiled catalog
- The catalog is sorted by discovery date (newest first) for easier processing of recent discoveries


