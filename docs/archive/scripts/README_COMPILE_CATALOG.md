# Compile Supernova Catalog from Multiple Sources

This script compiles a comprehensive supernova catalog by downloading data from multiple sources and merging them with your existing catalog.

## Sources

1. **Open Supernova Catalog (OSC)** - https://sne.space
   - Over 36,000 supernovae
   - Updated daily
   - Includes coordinates, discovery dates, types, and host galaxies

2. **TNS (Transient Name Server)** - https://www.wis-tns.org
   - Requires API key (optional)
   - Professional-grade transient catalog

3. **Existing Catalog** - Your current `sncat_latest_view.txt`
   - Preserved and merged with new sources

## Usage

### Basic Usage (OSC only)

```bash
python scripts/compile_supernova_catalog.py \
    --existing-catalog resources/sncat_latest_view.txt \
    --output resources/sncat_compiled.txt
```

This will:
- Load your existing catalog
- Download OSC catalog (cached for future runs)
- Merge and deduplicate entries
- Output compiled catalog

### With TNS (requires API key)

```bash
python scripts/compile_supernova_catalog.py \
    --existing-catalog resources/sncat_latest_view.txt \
    --output resources/sncat_compiled.txt \
    --tns-api-key YOUR_TNS_API_KEY
```

### Options

- `--existing-catalog PATH`: Path to existing catalog (default: `resources/sncat_latest_view.txt`)
- `--output PATH`: Output catalog path (default: `resources/sncat_compiled.txt`)
- `--cache-dir PATH`: Directory to cache downloaded catalogs (default: `data/catalog_cache`)
- `--tns-api-key KEY`: TNS API key (optional)
- `--skip-osc`: Skip Open Supernova Catalog download
- `--skip-tns`: Skip TNS download
- `--coord-tolerance FLOAT`: Coordinate matching tolerance in degrees for deduplication (default: 0.001)

## Output Format

The output catalog uses the same pipe-delimited format as your existing catalog, with columns:
- `sn_name`: Supernova name
- `sn_ra`: Right ascension in HH MM SS.SS format
- `sn_dec`: Declination in +/-DD MM SS.S format
- `disc_date`: Discovery date in YYYY-MM-DD format
- `max_date`: Maximum brightness date in YYYY-MM-DD format
- `sn_type`: Supernova type
- `gal_name`: Host galaxy name

## Deduplication

The script automatically deduplicates entries by:
1. **Name matching**: Entries with the same name (case-insensitive) are merged
2. **Coordinate matching**: Entries within the coordinate tolerance are merged

When merging duplicates, the script prioritizes:
1. Existing catalog entries
2. OSC entries
3. TNS entries

Missing fields are filled from duplicate entries.

## Caching

Downloaded catalogs are cached in `data/catalog_cache/` to avoid re-downloading on subsequent runs. To force a fresh download, delete the cache files.

## Expected Results

With OSC integration, you should see:
- **Current catalog**: ~6,500 entries
- **OSC additions**: ~30,000+ additional entries
- **After deduplication**: ~30,000-35,000 unique entries

This significantly expands your training dataset!

## Next Steps

After compiling the catalog:

1. **Query MAST for FITS files**:
```bash
python scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_compiled.txt \
    --min-year 2010 \
    --limit 1000 \
    --missions HST JWST TESS
```

2. **Download FITS files**:
```bash
python scripts/download_sn_fits.py \
    --query-results output/sn_queries.json \
    --output-dir data/fits
```

3. **Ingest into pipeline**:
```bash
python scripts/ingest_sn_fits_to_pipeline.py \
    --query-results output/sn_queries.json \
    --fits-dir data/fits \
    --survey-id <your-survey-uuid>
```

## Troubleshooting

### OSC Download Fails

If OSC download fails, the script will try to use a cached version. If no cache exists, check your internet connection and try again. The OSC catalog is large (~100MB+), so the first download may take several minutes.

### TNS API Errors

TNS API requires authentication. If you don't have an API key, simply omit the `--tns-api-key` option. The script will work fine with just OSC and your existing catalog.

### Memory Issues

If you encounter memory issues with very large catalogs, you can:
- Process sources separately using `--skip-osc` or `--skip-tns`
- Increase coordinate tolerance to reduce deduplication overhead
- Process in batches by filtering the output

## Notes

- The script preserves all fields from your existing catalog
- New entries from OSC/TNS may have fewer fields filled in
- Coordinates are automatically converted to the required format
- Dates are normalized to YYYY-MM-DD format


