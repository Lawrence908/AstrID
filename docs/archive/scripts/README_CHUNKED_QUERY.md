# Chunked Query Script

## Problem Solved

The original query script loads all results into memory and only saves at the end. For 1000+ supernovae, this can cause:
- Memory issues (hanging/crashing)
- Loss of progress if it crashes
- Very large JSON files

## Solution

The chunked version (`query_sn_fits_chunked.py`) processes in batches and saves incrementally:
- **Processes in chunks** (default: 250 supernovae per chunk)
- **Saves each chunk** to a separate file immediately
- **Checkpoint tracking** - can resume if interrupted
- **Combines chunks** at the end into final output

## Usage

### Basic Usage (1000 supernovae in 4 chunks of 250)

```bash
./scripts/query_chunked_1000.sh
```

Or manually:

```bash
python3 scripts/query_sn_fits_chunked.py \
    --catalog resources/sncat_compiled.txt \
    --output output/sn_queries_compiled_1000.json \
    --chunk-size 250 \
    --chunk-dir output/chunks_1000 \
    --checkpoint output/checkpoint_1000.json \
    --limit 1000 \
    --min-year 2010 \
    --missions TESS GALEX PS1 SWIFT \
    --days-before 1095 \
    --days-after 730
```

### Resume After Interruption

If the script crashes or is interrupted, simply run it again with the same parameters. It will:
1. Load the checkpoint to see what's already processed
2. Skip already-processed supernovae
3. Continue from where it left off

### Combine Existing Chunks Only

If chunks are already saved but you need to recombine:

```bash
python3 scripts/query_sn_fits_chunked.py \
    --chunk-dir output/chunks_1000 \
    --output output/sn_queries_compiled_1000.json \
    --combine-only
```

## File Structure

```
output/
├── chunks_1000/           # Individual chunk files
│   ├── chunk_000.json     # First 250 supernovae
│   ├── chunk_001.json     # Next 250 supernovae
│   ├── chunk_002.json     # Next 250 supernovae
│   └── chunk_003.json     # Last 250 supernovae
├── checkpoint_1000.json   # Tracks which SN have been processed
└── sn_queries_compiled_1000.json  # Final combined output
```

## Benefits

1. **Memory efficient** - Only processes 250 at a time
2. **Progress saved** - Each chunk saved immediately
3. **Resumable** - Can continue after interruption
4. **Safer** - Less risk of losing all progress
5. **Faster recovery** - Only need to redo failed chunks

## Expected Performance

- **300 supernovae**: ~30 minutes (original script)
- **1000 supernovae in chunks**: ~2 hours (chunked script)
  - Each chunk of 250: ~30 minutes
  - 4 chunks = ~2 hours total
  - Plus time to combine at end

## Troubleshooting

### Script hangs on a specific supernova

The checkpoint will save progress up to that point. You can:
1. Manually edit checkpoint to skip problematic SN
2. Or let it timeout and resume (it will skip that SN)

### Want to change chunk size

Use `--chunk-size` parameter:
- Smaller chunks (100): More frequent saves, slower overall
- Larger chunks (500): Fewer files, but more memory risk

### Chunks are saved but final file missing

Use `--combine-only` to regenerate the final combined file.






