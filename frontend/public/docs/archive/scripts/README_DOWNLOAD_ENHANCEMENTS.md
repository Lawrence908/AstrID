# Supernova FITS Download Script Enhancements

## Overview

The `download_sn_fits.py` script has been enhanced to ensure you get viable before/after supernova event FITS files for differencing and ML analysis.

## New Features

### 1. **Pre-filtering of Query Results**

The script now automatically filters out observations that don't have downloadable data:
- Removes observations with `filesize: 0` (no actual data)
- Removes observations without valid `dataURL`
- Shows detailed statistics about what's viable

### 2. **Require Both Reference AND Science Images**

By default, only processes supernovae that have BOTH:
- At least one viable reference image (before the SN event)
- At least one viable science image (after the SN event)

This ensures you get complete before/after pairs for differencing.

### 3. **FITS File Verification**

Downloaded FITS files are automatically validated to ensure they have:
- Valid WCS (World Coordinate System) - RA/DEC coordinates
- Proper headers and data structure

Invalid files are logged and reported separately.

### 4. **Dry Run Mode**

Test what would be downloaded without actually downloading:
```bash
python scripts/download_sn_fits.py \
    --query-results output/sn_queries_all_missions.json \
    --output-dir data/fits \
    --dry-run
```

### 5. **Enhanced Reporting**

The script now provides detailed summaries:
- ✅ Complete before/after pairs count
- 📊 File validation statistics
- ⚠️ Warnings for partial data
- 💡 Next steps for pipeline ingestion

## Usage Examples

### Basic Usage (Recommended)
Process only SN with complete before/after pairs:
```bash
python scripts/download_sn_fits.py \
    --query-results output/sn_queries_all_missions.json \
    --output-dir data/fits \
    --limit 10
```

### Dry Run First
Check what would be downloaded:
```bash
python scripts/download_sn_fits.py \
    --query-results output/sn_queries_all_missions.json \
    --output-dir data/fits \
    --dry-run
```

### Allow Partial Data
Process SN even if they only have reference OR science (not both):
```bash
python scripts/download_sn_fits.py \
    --query-results output/sn_queries_all_missions.json \
    --output-dir data/fits \
    --allow-partial
```

### Skip Verification (Faster)
Skip FITS file verification (not recommended):
```bash
python scripts/download_sn_fits.py \
    --query-results output/sn_queries_all_missions.json \
    --output-dir data/fits \
    --no-verify
```

## Command-Line Arguments

### New Arguments

- `--require-both` (default: True) - Only process SN with both reference AND science observations
- `--allow-partial` - Allow processing SN with only reference OR science
- `--verify-fits` (default: True) - Verify downloaded FITS files are valid
- `--no-verify` - Skip FITS file verification
- `--dry-run` - Show what would be downloaded without downloading

### Existing Arguments

- `--query-results` - Path to JSON file from query_sn_fits_from_catalog.py (required)
- `--output-dir` - Base directory for downloaded FITS files (default: data/fits)
- `--max-obs` - Maximum observations per type per SN (default: 5)
- `--skip-reference` - Skip downloading reference images
- `--skip-science` - Skip downloading science images
- `--limit` - Limit number of supernovae to process

## Output

### Download Results JSON

The script saves a `download_results.json` file with:
```json
{
  "sn_name": "2000C",
  "reference_files": ["path/to/ref1.fits", "path/to/ref2.fits"],
  "science_files": ["path/to/sci1.fits"],
  "verified_files": ["path/to/ref1.fits", "path/to/sci1.fits"],
  "invalid_files": ["path/to/ref2.fits"],
  "errors": []
}
```

### Summary Output

```
============================================================
DOWNLOAD SUMMARY
============================================================
Supernovae processed: 10
  ✅ Complete before/after pairs: 8

Files downloaded:
  Reference files: 15
  Science files: 12
  Total: 27

FITS validation:
  ✅ Valid files: 25
  ❌ Invalid files: 2
  Success rate: 92.6%

Output directory: data/fits
============================================================

✅ Ready for differencing: 8 supernovae with complete before/after pairs
   Next step: Run ingest_sn_fits_to_pipeline.py to process these files
```

## Pre-filtering Output

Before downloading, you'll see statistics:

```
============================================================
PRE-FILTERING QUERY RESULTS
============================================================
Mode: Require both reference AND science

Filtering results:
  Total supernovae in query: 100
  ✅ With both ref & sci (viable): 35
  ⚠️  Reference only: 20
  ⚠️  Science only: 15
  ❌ Neither: 30

  Total viable reference observations: 150
  Total viable science observations: 120

  Supernovae to process: 35
============================================================
```

## Workflow

1. **Query MAST for observations**
   ```bash
   python scripts/query_sn_fits_from_catalog.py \
       --catalog resources/sncat_latest_view.txt \
       --limit 100 \
       --output output/sn_queries_all_missions.json
   ```

2. **Check what's available (dry run)**
   ```bash
   python scripts/download_sn_fits.py \
       --query-results output/sn_queries_all_missions.json \
       --output-dir data/fits \
       --dry-run
   ```

3. **Download viable SN with complete pairs**
   ```bash
   python scripts/download_sn_fits.py \
       --query-results output/sn_queries_all_missions.json \
       --output-dir data/fits \
       --limit 10
   ```

4. **Ingest into pipeline**
   ```bash
   python scripts/ingest_sn_fits_to_pipeline.py \
       --query-results output/sn_queries_all_missions.json \
       --fits-dir data/fits \
       --survey-id <uuid>
   ```

## Tips

- Start with `--dry-run` to see what's available
- Use `--limit 5` for initial testing
- The default settings (`--require-both --verify-fits`) are recommended for production
- Check `download_results.json` for detailed download status
- Look for "Complete before/after pairs" count in the summary





