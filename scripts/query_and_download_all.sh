#!/bin/bash
# Query and download supernova FITS files from compiled catalog
# This script automates the full workflow

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CATALOG="${1:-resources/sncat_compiled.txt}"
OUTPUT_DIR="${2:-data/fits}"
QUERY_OUTPUT="${3:-output/sn_queries_compiled.json}"
LIMIT="${4:-1000}"  # Process up to 1000 supernovae
MIN_YEAR="${5:-2010}"  # Only process supernovae from 2010 onwards

echo "============================================================"
echo "Supernova FITS Query and Download Workflow"
echo "============================================================"
echo "Catalog: $CATALOG"
echo "Output directory: $OUTPUT_DIR"
echo "Query results: $QUERY_OUTPUT"
echo "Limit: $LIMIT supernovae"
echo "Min year: $MIN_YEAR"
echo "============================================================"
echo ""

# Step 1: Query MAST for observations
echo "Step 1: Querying MAST for observations..."
python3 scripts/query_sn_fits_from_catalog.py \
    --catalog "$CATALOG" \
    --output "$QUERY_OUTPUT" \
    --min-year "$MIN_YEAR" \
    --limit "$LIMIT" \
    --missions TESS GALEX PS1 SWIFT HST JWST \
    --days-before 1095 \
    --days-after 730 \
    --radius 0.1

if [ ! -f "$QUERY_OUTPUT" ]; then
    echo "ERROR: Query failed - output file not found"
    exit 1
fi

echo ""
echo "Step 1 complete. Checking results..."

# Check how many viable supernovae we have
python3 << 'PYTHON_SCRIPT'
import json
import sys

with open('$QUERY_OUTPUT', 'r') as f:
    data = json.load(f)

total = len(data)
viable = sum(1 for sn in data if sn.get('reference_observations') and sn.get('science_observations'))
ref_only = sum(1 for sn in data if sn.get('reference_observations') and not sn.get('science_observations'))
sci_only = sum(1 for sn in data if not sn.get('reference_observations') and sn.get('science_observations'))
neither = total - viable - ref_only - sci_only

print(f"Query Results Summary:")
print(f"  Total supernovae: {total}")
print(f"  ✅ With both ref & sci (viable): {viable}")
print(f"  ⚠️  Reference only: {ref_only}")
print(f"  ⚠️  Science only: {sci_only}")
print(f"  ❌ Neither: {neither}")
print("")

if viable == 0:
    print("WARNING: No viable supernovae found with both reference and science observations!")
    print("You may want to:")
    print("  - Try different missions: --missions HST JWST")
    print("  - Widen time windows: --days-before 1825 --days-after 1095")
    print("  - Remove year filter: remove --min-year")
    sys.exit(1)
else:
    print(f"Proceeding with {viable} viable supernovae...")
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
read -p "Continue with download? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Step 2: Dry run first to preview
echo ""
echo "Step 2: Preview download (dry run)..."
python3 scripts/download_sn_fits.py \
    --query-results "$QUERY_OUTPUT" \
    --output-dir "$OUTPUT_DIR" \
    --dry-run \
    --require-both \
    --max-obs 3 \
    --limit 10

echo ""
read -p "Dry run complete. Proceed with actual download? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Step 3: Download FITS files
echo ""
echo "Step 3: Downloading FITS files..."
echo "This may take a while depending on file sizes and network speed..."
python3 scripts/download_sn_fits.py \
    --query-results "$QUERY_OUTPUT" \
    --output-dir "$OUTPUT_DIR" \
    --require-both \
    --max-obs 3 \
    --verify-fits

echo ""
echo "============================================================"
echo "Download Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Verify downloaded files:"
echo "   find $OUTPUT_DIR -name '*.fits*' | wc -l"
echo ""
echo "2. Ingest into pipeline:"
echo "   python3 scripts/ingest_sn_fits_to_pipeline.py \\"
echo "       --query-results $QUERY_OUTPUT \\"
echo "       --fits-dir $OUTPUT_DIR \\"
echo "       --survey-id <your-survey-uuid> \\"
echo "       --filter-has-both"
echo ""
