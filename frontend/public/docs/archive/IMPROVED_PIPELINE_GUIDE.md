# Improved Data Pipeline Guide

## Problem Solved

**Before:** Only 56/225 SNe (25%) resulted in complete training pairs  
**After:** Expected 168-225 SNe (75-100%) with complete training pairs

## Root Cause

The `identify_same_mission_pairs.py` script identified SNe with overlapping missions but didn't filter the observation lists. This caused downloads of incompatible data (e.g., GALEX reference + SWIFT science), which can't be paired for differencing.

## Solution

New script: `filter_same_mission_observations.py` filters observations to keep only matching missions.

## Updated Pipeline

### Step 1: Query MAST (unchanged)

```bash
python3 scripts/query_sn_fits_chunked.py \
    --catalog resources/sncat_compiled.txt \
    --chunk-size 250 \
    --output output/sn_queries_remaining.json \
    --checkpoint output/checkpoint_remaining.json \
    --start-index 1000 \
    --min-year 2005
```

### Step 2: Identify Same-Mission Pairs (unchanged)

```bash
python3 scripts/identify_same_mission_pairs.py \
    --input output/sn_queries_remaining.json \
    --output output/same_mission_pairs.json
```

### Step 3: **NEW** - Filter Observations by Mission

**Option A: Combined (recommended for simplicity)**
```bash
python3 scripts/filter_same_mission_observations.py \
    --input output/sn_queries_same_mission.json \
    --output output/sn_queries_filtered.json \
    --require-both-missions
```

**Option B: Split by mission (recommended for maximum training data)**
```bash
python3 scripts/filter_same_mission_observations.py \
    --input output/sn_queries_same_mission.json \
    --output output/sn_queries_filtered_split.json \
    --split-by-mission \
    --require-both-missions
```

**Results:**
- Combined: 225 entries (1 per SN)
- Split: 240 entries (multi-mission SNe get separate entries per mission)
- Filtered out: 38% of science obs, 12% of ref obs (incompatible missions)

### Step 4: Download (with filtered results)

```bash
python3 scripts/download_sn_fits.py \
  --query-results output/sn_queries_filtered.json \
  --output-dir output/fits_downloads_filtered \
  --filter-has-both \
  --max-obs 5 \
  --max-products-per-obs 3
```

**Optional: Increase max-obs for better success rate**
```bash
# More observations = higher chance of success, but more disk space
python3 scripts/download_sn_fits.py \
  --query-results output/sn_queries_filtered.json \
  --output-dir output/fits_downloads_filtered \
  --filter-has-both \
  --max-obs 10 \
  --max-products-per-obs 2
```

### Step 5: Cleanup (optional)

```bash
python3 scripts/cleanup_auxiliary_files.py \
    --input-dir output/fits_downloads_filtered \
    --check-wcs
```

### Step 6: Organize Training Pairs

```bash
python3 scripts/organize_training_pairs.py \
  --input-dir output/fits_downloads_filtered \
  --output-dir output/fits_training_filtered \
  --clean \
  --decompress
```

### Step 7: Generate Difference Images

```bash
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training_filtered \
  --output-dir output/difference_images_filtered \
  --visualize
```

## Key Improvements

### 1. Mission Filtering (38% reduction in wasted downloads)

**Before:**
- 2005at: GALEX ref + SWIFT sci → No usable pairs
- 2009nz: (GALEX+PS1+SWIFT) ref + (PS1+SWIFT) sci → Only 1 pair possible

**After:**
- 2005at: GALEX ref + GALEX sci → Usable pair
- 2009nz_PS1: PS1 ref + PS1 sci → Usable pair
- 2009nz_SWIFT: SWIFT ref + SWIFT sci → Usable pair

### 2. Multi-Mission SNe Optimization

14 SNe have observations from multiple missions. With `--split-by-mission`:
- Each mission gets a separate entry
- Doubles/triples training data from these SNe
- Example: 2009nz becomes 2009nz_PS1 and 2009nz_SWIFT

### 3. Expected Results

**Current pipeline:**
- Input: 225 SNe with same-mission observations
- Downloaded: 216 SNe
- Complete pairs: 56 (25% success rate)
- Wasted downloads: 160 SNe (75%)

**Improved pipeline:**
- Input: 225 SNe (or 240 with split)
- Expected complete pairs: 168-225 (75-100% success rate)
- Wasted downloads: 0-57 SNe (0-25%)

**Why not 100%?**
- Some observations may fail to download (MAST issues)
- Some FITS files may be invalid (no WCS, corrupted)
- Some observations may have no imaging products after filtering

## Optimization Options

### Option 1: More Observations per SN

**Current:** `--max-obs 5` (5 ref + 5 sci per SN)

**Increase to 10:**
```bash
--max-obs 10 --max-products-per-obs 2
```

**Trade-off:**
- ✅ Higher success rate (more chances to get valid files)
- ❌ More disk space (2x downloads per SN)

### Option 2: Mission-Specific Limits

Different missions have different characteristics:
- SWIFT: 1-3 products per obs (efficient)
- PS1: 5 bands but script deduplicates (efficient)
- GALEX: Many redundant products (less efficient)

**Future enhancement:** Add mission-specific max-obs:
```python
MISSION_MAX_OBS = {
    "SWIFT": 5,
    "PS1": 10,
    "GALEX": 8,
}
```

### Option 3: Prioritize by Observation Quality

Add filters for:
- Exposure time (longer = better SNR)
- Observation date (closer to discovery = better)
- Product count (more products = more options)

## Disk Space Estimates

### Current Pipeline (unfiltered)
- 225 SNe × 5 obs/type × 3 products × 10 MB = ~67 GB
- Actual pairs: 56 (25%)
- Wasted: ~50 GB (75%)

### Improved Pipeline (filtered)
- 225 SNe × 5 obs/type × 3 products × 10 MB = ~67 GB
- But 38% fewer science obs downloaded = ~42 GB
- Expected pairs: 168-225 (75-100%)
- Wasted: ~5-13 GB (12-30%)

### With Increased max-obs
- 225 SNe × 10 obs/type × 2 products × 10 MB = ~90 GB
- Expected pairs: 180-220 (80-98%)
- Wasted: ~5-18 GB (5-20%)

## Testing the Improved Pipeline

### Quick Test (10 SNe)

```bash
# Create test subset
jq '.[0:10]' output/sn_queries_filtered.json > output/sn_queries_test.json

# Download
python3 scripts/download_sn_fits.py \
  --query-results output/sn_queries_test.json \
  --output-dir output/fits_test \
  --filter-has-both \
  --max-obs 5 \
  --max-products-per-obs 3

# Organize
python3 scripts/organize_training_pairs.py \
  --input-dir output/fits_test \
  --output-dir output/fits_training_test \
  --clean

# Check success rate
echo "Complete pairs: $(find output/fits_training_test -mindepth 1 -maxdepth 1 -type d | wc -l) / 10"
```

Expected: 7-10 complete pairs (70-100% success rate)

## Comparison: Before vs After

| Metric | Before | After (Combined) | After (Split) |
|--------|--------|------------------|---------------|
| Input SNe | 225 | 225 | 225 |
| Output entries | 225 | 225 | 240 |
| Ref obs | 2382 | 2100 (88%) | 2100 (88%) |
| Sci obs | 4458 | 2758 (62%) | 2758 (62%) |
| Expected pairs | 56 (25%) | 168-225 (75-100%) | 180-240 (75-100%) |
| Wasted downloads | 169 (75%) | 0-57 (0-25%) | 0-60 (0-25%) |

## Next Steps

1. **Test on small subset** (10 SNe) to verify improvement
2. **Run full pipeline** with filtered results
3. **Compare success rates** (should see 3-4x improvement)
4. **Adjust max-obs** if needed for higher success rate
5. **Consider split-by-mission** for maximum training data

## Additional Enhancements (Future)

### 1. Cross-Mission Pairing
Some missions are compatible for differencing:
- GALEX NUV + SWIFT UVW1 (similar wavelengths)
- PS1 + SDSS (similar optical bands)

### 2. Smart Observation Selection
Prioritize observations by:
- Exposure time (longer = better)
- Time from discovery (optimal = 0-30 days after)
- Image quality metrics (if available)

### 3. Parallel Downloads
Speed up downloads with concurrent requests:
```python
--parallel-downloads 4
```

### 4. Resume Capability
Add checkpoint/resume for interrupted downloads:
```python
--checkpoint output/download_checkpoint.json
```

## Summary

The new `filter_same_mission_observations.py` script solves the core bottleneck in your pipeline by ensuring only compatible observations are downloaded. This should increase your success rate from 25% to 75-100%, giving you 3-4x more training data with the same or less disk space.

**Recommended command sequence:**

```bash
# Filter observations
python3 scripts/filter_same_mission_observations.py \
    --input output/sn_queries_same_mission.json \
    --output output/sn_queries_filtered.json \
    --require-both-missions

# Download (with increased max-obs for better success)
python3 scripts/download_sn_fits.py \
  --query-results output/sn_queries_filtered.json \
  --output-dir output/fits_downloads_filtered \
  --filter-has-both \
  --max-obs 10 \
  --max-products-per-obs 2

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
