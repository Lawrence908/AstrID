# Querying FITS Files from Supernova Catalog

This guide explains how to use the supernova catalog to query and download FITS files for ZOGY differencing and machine learning training.

## Overview

The workflow for getting FITS files for supernova training data:

1. **Parse Catalog**: Extract supernova coordinates and discovery dates from `sncat_latest_view.txt`
2. **Query MAST**: Search for reference images (before discovery) and science images (after discovery)
3. **Download FITS**: Download the FITS files for processing
4. **ZOGY Differencing**: Use the reference and science images for image differencing

## Catalog Format

The catalog (`resources/sncat_latest_view.txt`) is a pipe-delimited text file with columns:

- `sn_name`: Supernova name (e.g., "1885A", "1895B")
- `sn_ra`: Right ascension in HH MM SS.SS format
- `sn_dec`: Declination in +/-DD MM SS.S format
- `disc_date`: Discovery date in YYYY-MM-DD format
- `max_date`: Maximum brightness date in YYYY-MM-DD format
- `sn_type`: Supernova type (e.g., "Ia", "II", "IPec")
- `gal_name`: Host galaxy name (e.g., "NGC0224")

## Usage

### Step 1: Query MAST for Observations

Query MAST to find available observations for supernovae in the catalog:

```bash
# Basic usage - process first 10 supernovae
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --limit 10

# Process supernovae from year 2000 onwards
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --min-year 2000 \
    --missions HST JWST TESS

# Custom time windows and search radius
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --limit 5 \
    --days-before 730 \
    --days-after 730 \
    --radius 0.2 \
    --missions HST JWST
```

### Step 2: Review Results

The script outputs a JSON file (`output/sn_mast_queries.json` by default) with:

- For each supernova:
  - `reference_observations`: Observations before discovery date
  - `science_observations`: Observations after discovery date
  - `errors`: Any errors encountered

### Step 3: Download FITS Files

Once you have the query results, you can download the FITS files. You can use the existing MAST client or create a download script.

## Important Considerations

### 1. **Data Availability**

- **Historical Supernovae**: Very old supernovae (pre-1990) may have limited or no space telescope observations
- **Modern Supernovae**: Recent supernovae (2000+) are more likely to have HST/JWST data
- **Ground-based Surveys**: Consider querying ground-based survey archives (Pan-STARRS, SDSS, etc.) for older supernovae

### 2. **Time Windows**

- **Reference Images**: Need images taken BEFORE the supernova appeared
  - Default: 365 days before discovery
  - Adjust based on how often the field was observed
- **Science Images**: Need images taken AFTER discovery
  - Default: 365 days after discovery
  - Supernovae are visible for weeks to months

### 3. **Coordinate Accuracy**

- The catalog provides coordinates in sexagesimal format
- Some entries may have missing or uncertain coordinates
- The script skips entries with invalid coordinates

### 4. **Mission Selection**

Available missions in MAST:
- `HST`: Hubble Space Telescope (1990-present)
- `JWST`: James Webb Space Telescope (2022-present)
- `TESS`: Transiting Exoplanet Survey Satellite (2018-present)
- `Kepler`: Kepler Mission (2009-2018)
- `GALEX`: Galaxy Evolution Explorer (2003-2013)

### 5. **ZOGY Differencing Requirements**

For proper ZOGY differencing, you need:
- **Reference Image**: Deep, high-quality image before the supernova
- **Science Image**: Image after discovery with the supernova visible
- **Same Filter**: Ideally same filter/band for both images
- **WCS Alignment**: Both images need proper WCS headers for alignment

## Example Workflow

```python
# 1. Query MAST
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_latest_view.txt \
    --min-year 2000 \
    --limit 20 \
    --output output/sn_queries_2000s.json

# 2. Review results
cat output/sn_queries_2000s.json | jq '.[] | select(.reference_observations | length > 0) | select(.science_observations | length > 0) | .sn_name'

# 3. For each supernova with both reference and science images:
#    - Download reference FITS
#    - Download science FITS
#    - Run ZOGY differencing
#    - Extract difference image for ML training
```

## Alternative Data Sources

If MAST doesn't have sufficient data for your needs:

1. **Pan-STARRS**: Public survey with extensive coverage
   - Use `MASTClient.fetch_ps1_cutout()` for reference images
   - Query PS1 archive for science images

2. **SDSS**: Sloan Digital Sky Survey
   - Good for reference images
   - Limited time-domain coverage

3. **ZTF**: Zwicky Transient Facility
   - Excellent for recent supernovae (2018+)
   - High cadence observations

4. **LSST**: Legacy Survey of Space and Time
   - Future data source (2025+)
   - Will provide extensive supernova data

## Troubleshooting

### No Observations Found

- **Check date range**: Older supernovae may not have space telescope observations
- **Increase search radius**: Try `--radius 0.2` or `0.5` degrees
- **Try different missions**: Some missions have better coverage in certain regions
- **Check coordinates**: Verify the supernova coordinates are correct

### Parsing Errors

- Some catalog entries may have missing or malformed data
- The script skips entries with invalid coordinates or dates
- Check the logs for specific parsing errors

### Download Issues

- MAST may have rate limiting
- Large files may take time to download
- Some observations may be proprietary (require authentication)

## Next Steps

1. **Filter Results**: Identify supernovae with both reference and science images
2. **Download FITS**: Use MAST client to download the actual FITS files
3. **Process Images**: Run ZOGY differencing on the image pairs
4. **Create Training Set**: Extract difference images and labels for ML training

