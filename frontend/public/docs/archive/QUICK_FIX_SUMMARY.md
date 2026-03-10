# Quick Fix Summary - Data Pipeline Bottleneck

## The Problem

You're getting only **56 complete pairs** from **225 SNe** (25% success rate).

## Root Cause

The `identify_same_mission_pairs.py` script identifies SNe with overlapping missions but **doesn't filter the observation lists**. This causes downloads of incompatible data:

- Example: SN 2005at has GALEX ref + SWIFT sci → Can't be paired
- Result: Wasted downloads, incomplete pairs

## The Solution

**New script created:** `scripts/filter_same_mission_observations.py`

This script filters observations to keep only matching missions BEFORE download.

## Immediate Action

### 1. Filter your existing query results (DONE ✅)

```bash
python3 scripts/filter_same_mission_observations.py \
    --input output/sn_queries_same_mission.json \
    --output output/sn_queries_filtered.json \
    --require-both-missions
```

**Result:** Created `output/sn_queries_filtered.json` with:
- 225 entries (same as input)
- 88% of ref obs kept (282 filtered out)
- 62% of sci obs kept (1700 filtered out)
- **Expected 168-225 complete pairs (75-100% success rate)**

### 2. Download with filtered results

```bash
python3 scripts/download_sn_fits.py \
  --query-results output/sn_queries_filtered.json \
  --output-dir output/fits_downloads_filtered \
  --filter-has-both \
  --max-obs 5 \
  --max-products-per-obs 3
```

**Optional: Increase max-obs for even better success rate**
```bash
python3 scripts/download_sn_fits.py \
  --query-results output/sn_queries_filtered.json \
  --output-dir output/fits_downloads_filtered \
  --filter-has-both \
  --max-obs 10 \
  --max-products-per-obs 2
```

### 3. Organize and generate difference images

```bash
# Organize
python3 scripts/organize_training_pairs.py \
  --input-dir output/fits_downloads_filtered \
  --output-dir output/fits_training_filtered \
  --clean \
  --decompress

# Generate difference images
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training_filtered \
  --output-dir output/difference_images_filtered \
  --visualize
```

## Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Complete pairs | 56 (25%) | 168-225 (75-100%) |
| Wasted downloads | 169 (75%) | 0-57 (0-25%) |
| Disk space efficiency | 25% | 75-100% |

## Bonus: Split-by-Mission Option

For maximum training data, use the split version:

```bash
python3 scripts/filter_same_mission_observations.py \
    --input output/sn_queries_same_mission.json \
    --output output/sn_queries_filtered_split.json \
    --split-by-mission \
    --require-both-missions
```

This creates **240 entries** (15 extra from multi-mission SNe like 2009nz).

Example:
- 2009nz → 2009nz_PS1 (PS1 ref + PS1 sci) + 2009nz_SWIFT (SWIFT ref + SWIFT sci)

## Files Created

1. ✅ `scripts/filter_same_mission_observations.py` - New filter script
2. ✅ `output/sn_queries_filtered.json` - Filtered query results (225 entries)
3. ✅ `output/sn_queries_filtered_split.json` - Split by mission (240 entries)
4. ✅ `PIPELINE_ANALYSIS_AND_FIXES.md` - Detailed analysis
5. ✅ `IMPROVED_PIPELINE_GUIDE.md` - Complete updated pipeline guide
6. ✅ `QUICK_FIX_SUMMARY.md` - This file

## What Changed in the Data

### Example: SN 2005at

**Before filtering:**
- Reference: 1 GALEX obs
- Science: 12 SWIFT obs + 2 GALEX obs
- Result: Downloaded GALEX ref + SWIFT sci → No pairs

**After filtering:**
- Reference: 1 GALEX obs
- Science: 2 GALEX obs (SWIFT removed)
- Result: Will download GALEX ref + GALEX sci → Usable pair

### Statistics

**Observations filtered out:**
- Reference: 282 obs (12%) - removed non-matching missions
- Science: 1700 obs (38%) - removed non-matching missions

**Observations kept:**
- Reference: 2100 obs (88%)
- Science: 2758 obs (62%)

**Mission breakdown:**
- GALEX: 182 entries (most common)
- PS1: 32 entries
- SWIFT: 26 entries

## Next Steps

1. **Test on small subset first** (recommended):
   ```bash
   jq '.[0:10]' output/sn_queries_filtered.json > output/sn_queries_test.json
   # Then run download/organize on test set
   ```

2. **Run full pipeline** with filtered results

3. **Compare results** (expect 3-4x more complete pairs)

4. **Adjust parameters** if needed:
   - Increase `--max-obs` for higher success rate
   - Decrease `--max-products-per-obs` to save disk space

## Questions?

- See `IMPROVED_PIPELINE_GUIDE.md` for complete documentation
- See `PIPELINE_ANALYSIS_AND_FIXES.md` for detailed analysis
- The filter script has `--help` for all options
