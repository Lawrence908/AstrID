# Analysis of Query Results

## What Happened

The initial query found:
- **5 supernovae processed** (2000B, 2000C, 2000D, 2000E, 2000G)
- **0 reference observations** (images before discovery)
- **30 science observations** (only for 2000E, all from HST)
- **0 downloads** (because `--filter-has-both` requires both reference AND science images)

## Key Issues Identified

### 1. Mission Filtering Too Restrictive
- MAST returned **raw observations** (71, 159, 66, 540, 41 respectively)
- But **all were filtered out** except for 2000E's science images
- This suggests the observations exist but from missions not in the filter list

### 2. No Reference Images
- Even 2000E (which had 30 science images) had **0 reference images**
- This could mean:
  - The field wasn't observed before the supernova appeared
  - Observations exist but from missions not in the filter
  - Time window (1 year) was too narrow

### 3. Limited Mission Coverage for 2000 Era
- Year 2000 was early for space telescope observations
- HST was active but had limited coverage
- JWST didn't exist yet (launched 2022)
- TESS didn't exist yet (launched 2018)

## Improvements Made

### 1. Better Mission Filtering
- Made mission filtering **case-insensitive**
- Added **diagnostic logging** to show which missions are in raw results
- Added **warning** when requested missions aren't found

### 2. Expanded Time Windows
- Changed default `--days-before` from 365 to **730 days (2 years)**
- This gives more chance to find reference images

### 3. New Options
- `--no-mission-filter`: See ALL available observations (useful for discovery)
- Better logging to show what missions are actually available

## Recommended Next Steps

### Step 1: Discover What Missions Are Available

```bash
# See what missions are actually in the data (no filtering)
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --limit 5 \
    --min-year 2000 \
    --no-mission-filter \
    --output output/sn_queries_all_missions.json
```

This will show you:
- Which missions actually have data for these supernovae
- Whether reference images exist from other missions
- What the actual mission names are in MAST

### Step 2: Try More Recent Supernovae

Year 2000 is quite early. Try more recent supernovae with better coverage:

```bash
# Try supernovae from 2010+ (better HST coverage, more missions)
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --limit 10 \
    --min-year 2010 \
    --missions HST JWST TESS \
    --days-before 1095 \
    --output output/sn_queries_2010s.json
```

### Step 3: Include Ground-Based Surveys

For year 2000 supernovae, consider ground-based surveys:
- **Pan-STARRS**: Use `MASTClient.fetch_ps1_cutout()` for reference images
- **SDSS**: Sloan Digital Sky Survey (good for reference images)
- **ZTF**: Zwicky Transient Facility (2018+, excellent for recent supernovae)

### Step 4: Process 2000E (Has Science Images)

Even though 2000E has no reference images in the query, you could:
1. Use Pan-STARRS or SDSS for reference images
2. Download the 30 HST science images
3. Create reference from deep stack of earlier observations

```bash
# Download science images for 2000E (skip the filter)
python scripts/download_sn_fits.py \
    --query-results output/sn_queries.json \
    --output-dir data/fits \
    --skip-reference \
    --max-obs 5
```

## Expected Results After Improvements

With the improvements, you should see:
- **Mission names** in the logs showing what's actually available
- **Warnings** when requested missions aren't found
- **More reference observations** with expanded time window (730 days)
- **Better diagnostics** to understand what's happening

## Alternative Approach: Use Pan-STARRS for Reference

Since space telescope reference images are rare, consider:

1. **Query MAST for science images** (after discovery)
2. **Use Pan-STARRS for reference images** (deep survey, good coverage)
3. **Combine for ZOGY differencing**

This is a common approach in transient astronomy!

