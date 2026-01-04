#!/bin/bash
# Query 1000 supernovae in chunks to avoid memory issues

CHUNK_SIZE=250  # Process 250 at a time
OUTPUT="output/sn_queries_compiled_1000.json"
CHUNK_DIR="output/chunks_1000"
CHECKPOINT="output/checkpoint_1000.json"

echo "============================================================"
echo "Chunked Query: 1000 Supernovae"
echo "============================================================"
echo "Chunk size: $CHUNK_SIZE"
echo "Output: $OUTPUT"
echo "Chunk directory: $CHUNK_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "============================================================"
echo ""

# Reset checkpoint if starting fresh (uncomment to clear checkpoint)
# Use --reset-checkpoint flag to clear checkpoint and empty chunks

# Run chunked query
# Add --reset-checkpoint to clear checkpoint and start fresh
# Set PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 scripts/query_sn_fits_chunked.py \
    --catalog resources/sncat_compiled.txt \
    --output "$OUTPUT" \
    --chunk-size "$CHUNK_SIZE" \
    --chunk-dir "$CHUNK_DIR" \
    --checkpoint "$CHECKPOINT" \
    --limit 1000 \
    --min-year 2010 \
    --missions TESS GALEX PS1 SWIFT \
    --days-before 1095 \
    --days-after 730
    # Removed --reset-checkpoint to allow resuming from checkpoint

echo ""
echo "Query complete! Results saved to: $OUTPUT"
echo ""
echo "To check results:"
echo "  python3 -c \"import json; data = json.load(open('$OUTPUT')); viable = sum(1 for sn in data if sn.get('reference_observations') and sn.get('science_observations')); print(f'Viable: {viable}/{len(data)} ({viable/len(data)*100:.1f}%)')\""
