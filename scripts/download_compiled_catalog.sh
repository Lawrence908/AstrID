#!/bin/bash
# Download FITS files from compiled catalog query results

set -e

QUERY_RESULTS="${1:-output/sn_queries_compiled.json}"
OUTPUT_DIR="${2:-data/fits}"
MAX_OBS="${3:-3}"  # Max observations per type per SN

echo "============================================================"
echo "Downloading Supernova FITS Files"
echo "============================================================"
echo "Query results: $QUERY_RESULTS"
echo "Output directory: $OUTPUT_DIR"
echo "Max observations per type: $MAX_OBS"
echo "============================================================"
echo ""

# Step 1: Dry run to preview
echo "Step 1: Preview download (dry run)..."
python3 scripts/download_sn_fits.py \
    --query-results "$QUERY_RESULTS" \
    --output-dir "$OUTPUT_DIR" \
    --dry-run \
    --require-both \
    --max-obs "$MAX_OBS" \
    --limit 10

echo ""
read -p "Dry run complete. Proceed with actual download? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Step 2: Download FITS files
echo ""
echo "Step 2: Downloading FITS files..."
echo "This may take a while depending on file sizes and network speed..."
python3 scripts/download_sn_fits.py \
    --query-results "$QUERY_RESULTS" \
    --output-dir "$OUTPUT_DIR" \
    --require-both \
    --max-obs "$MAX_OBS" \
    --verify-fits

echo ""
echo "============================================================"
echo "Download Complete!"
echo "============================================================"
echo ""
echo "Downloaded files:"
echo "  find $OUTPUT_DIR -name '*.fits*' | wc -l"
echo ""
echo "Next step: Ingest into pipeline"
echo "  python3 scripts/ingest_sn_fits_to_pipeline.py \\"
echo "      --query-results $QUERY_RESULTS \\"
echo "      --fits-dir $OUTPUT_DIR \\"
echo "      --survey-id <your-survey-uuid> \\"
echo "      --filter-has-both"
echo ""
