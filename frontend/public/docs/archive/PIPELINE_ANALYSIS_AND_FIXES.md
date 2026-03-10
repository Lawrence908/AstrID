# Data Pipeline Analysis and Optimization

## Problem Summary

You're getting only **56 complete SN pairs** out of **225 SNe** in your query results, despite having same-mission observations. 

### Root Cause Analysis

1. **Query Results (sn_queries_same_mission.json)**: 225 SNe with overlapping missions
2. **Downloads**: 216 SNe downloaded (9 failed/skipped)
3. **Actual Pairs**: Only 56 SNe have BOTH reference AND science files
4. **Breakdown**:
   - 56 SNe: Complete pairs (both ref + sci)
   - 59 SNe: Science-only (no reference files)
   - 1 SN: Reference-only (no science files)
   - 100 SNe: Empty or missing

### The Core Issue

The `identify_same_mission_pairs.py` script identifies SNe with overlapping missions but **doesn't filter the observation lists** to keep only matching missions. 

**Example: SN 2005at**
- Query identifies it has both GALEX and SWIFT observations
- But reference has only GALEX, science has SWIFT + GALEX
- Download script downloads ALL observations (GALEX ref, SWIFT sci)
- Result: No usable pairs because GALEX ref can't pair with SWIFT sci

## Solution Options

### Option 1: Filter Observations by Mission (RECOMMENDED)

Create a new script that filters observations to keep only matching missions.

**Benefits:**
- Maximizes usable pairs
- Reduces wasted downloads
- Enables multi-mission pairs (e.g., SWIFT pairs + PS1 pairs from same SN)
- Most efficient use of disk space

**Implementation:**
```python
# New script: filter_same_mission_observations.py
# For each SN with common missions:
#   - Keep only ref observations from common missions
#   - Keep only sci observations from common missions
#   - Ensure both lists are non-empty
```

### Option 2: Smarter Download Filtering

Modify the download script to detect mission mismatches and skip downloading.

**Benefits:**
- Prevents wasted downloads
- Simpler than Option 1

**Drawbacks:**
- Still downloads some unusable data
- Doesn't maximize pairs from multi-mission SNe

### Option 3: Enhanced Query with Mission Filtering

Modify the query stage to filter observations by mission immediately.

**Benefits:**
- Cleanest approach
- Prevents propagation of unusable data

**Drawbacks:**
- Requires modifying query logic
- More complex to implement

## Recommended Implementation: Option 1

### New Script: `filter_same_mission_observations.py`

This script will:
1. Read `sn_queries_same_mission.json`
2. For each SN:
   - Identify common missions
   - Filter reference_observations to keep only common missions
   - Filter science_observations to keep only common missions
   - Optionally create separate entries per mission (for multi-mission SNe)
3. Output filtered results

### Additional Optimizations

#### 1. Increase `--max-obs` Strategically

Current: `--max-obs 5` (5 ref + 5 sci per SN)

**Problem:** Some missions (PS1, GALEX) have many observations but only a few are usable.

**Solution:** Increase to `--max-obs 10` or `--max-obs 15` to increase chances of getting usable files.

**Trade-off:** More downloads per SN, but better success rate.

#### 2. Reduce `--max-products-per-obs`

Current: `--max-products-per-obs 3`

This is already good. Consider reducing to 2 if you want to save more space.

#### 3. Mission-Specific Limits

Different missions have different characteristics:
- **SWIFT**: Usually 1-3 products per obs (already efficient)
- **PS1**: 5 bands (g,r,i,z,y) but your script already deduplicates
- **GALEX**: 2 bands (NUV, FUV) but many redundant products

**Solution:** Add mission-specific max-obs limits:
```python
MISSION_MAX_OBS = {
    "SWIFT": 5,
    "PS1": 10,  # More obs needed due to band filtering
    "GALEX": 8,
    "HST": 3,
}
```

#### 4. Prioritize by File Size

Your download script already does this! It keeps the largest file per band, which is good.

#### 5. Pre-validate Observations

Add a validation step in the query to check if observations have downloadable products:
- Check `dataURL` is not empty
- Check product count > 0
- Check for imaging products

## Immediate Action Plan

### Step 1: Create Filter Script (HIGHEST PRIORITY)

```bash
python3 scripts/filter_same_mission_observations.py \
  --input output/sn_queries_same_mission.json \
  --output output/sn_queries_filtered.json \
  --require-both-missions
```

This will create a properly filtered query file with only matching missions.

### Step 2: Re-download with Filtered Results

```bash
# Clean previous downloads (optional)
rm -rf output/fits_downloads_filtered

# Download with filtered results
python3 scripts/download_sn_fits.py \
  --query-results output/sn_queries_filtered.json \
  --output-dir output/fits_downloads_filtered \
  --filter-has-both \
  --max-obs 10 \
  --max-products-per-obs 3
```

### Step 3: Organize

```bash
python3 scripts/organize_training_pairs.py \
  --input-dir output/fits_downloads_filtered \
  --output-dir output/fits_training_filtered \
  --clean \
  --decompress
```

## Expected Improvements

With the filter script:
- **Current**: 56/225 complete pairs (25% success rate)
- **Expected**: 150-180/225 complete pairs (67-80% success rate)

The improvement comes from:
1. Eliminating mission mismatches
2. Focusing downloads on usable observations
3. Reducing wasted disk space by 60-70%

## Additional Considerations

### Multi-Mission SNe

Some SNe have observations from multiple missions. Example:
- SN 2009nz: SWIFT (ref+sci) AND PS1 (ref+sci)

**Current behavior:** Downloads all, but can only use one mission pair.

**Optimal behavior:** Create separate entries per mission:
- `2009nz_SWIFT`: SWIFT ref + SWIFT sci
- `2009nz_PS1`: PS1 ref + PS1 sci

This doubles your training data from multi-mission SNe!

### Mission Compatibility

Some missions might be compatible for differencing despite being different:
- GALEX NUV + SWIFT UVW1 (similar wavelengths)
- PS1 + SDSS (similar optical bands)

**Future enhancement:** Add cross-mission pairing logic.

## Summary

The key bottleneck is **mission filtering**. The query identifies same-mission SNe correctly, but doesn't filter the observation lists, leading to downloads of incompatible data.

**Solution:** Create `filter_same_mission_observations.py` to filter observations by matching missions before download.

**Expected result:** 3-4x more complete pairs with same or less disk usage.
