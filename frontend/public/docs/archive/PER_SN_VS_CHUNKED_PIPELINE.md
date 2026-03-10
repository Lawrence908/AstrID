# Per-SN vs Chunked Pipeline Approaches

## Overview

There are two main approaches to prevent memory issues when processing large supernova datasets:

1. **Per-SN Pipeline**: Process one SN at a time through all stages
2. **Chunked Pipeline**: Process chunks of SNe through all stages

## Comparison

### Per-SN Pipeline (`run_pipeline_per_sn.py`)

**How it works:**
- For each SN: query → filter → download → organize → differencing
- Only one SN's data in memory at a time
- Checkpoint after each SN

**Pros:**
- ✅ **Lowest memory footprint** - only one SN at a time
- ✅ **Immediate results** - see output for each SN right away
- ✅ **Easy to resume** - just skip completed SNe
- ✅ **Better for debugging** - can see exactly which SN failed
- ✅ **Incremental progress** - can stop/start anytime
- ✅ **No intermediate files** - don't need to store all query results

**Cons:**
- ❌ Less efficient for batch operations
- ❌ More file I/O overhead (checkpointing each SN)
- ❌ Can't easily parallelize downloads
- ❌ Harder to do cross-SN analysis

**Best for:**
- Large datasets with memory constraints
- When you want to see results immediately
- When you need to resume frequently
- Debugging and development

### Chunked Pipeline (Current `run_pipeline_from_config.py`)

**How it works:**
- Stage 1: Query all SNe → save to JSON
- Stage 2: Filter all results → save to JSON
- Stage 3: Download all → save files
- Stage 4: Organize all → save files
- Stage 5: Differencing all → save files

**Pros:**
- ✅ More efficient for batch operations
- ✅ Can parallelize downloads
- ✅ Easier to do cross-SN analysis
- ✅ Less file I/O overhead

**Cons:**
- ❌ Higher memory footprint (all query results in memory)
- ❌ Can't see results until all SNe processed
- ❌ Harder to resume (need to re-run entire stages)
- ❌ If one stage fails, lose all progress

**Best for:**
- Smaller datasets with sufficient memory
- When you need all data before processing
- When you want to analyze all SNe together

## Memory Analysis

### Per-SN Approach
```
Memory per SN:
- Query result: ~1-10 MB (depending on observations)
- Downloaded FITS: ~50-500 MB (depends on mission)
- Organized files: ~50-500 MB
- Difference image: ~10-50 MB
Total: ~100-1000 MB per SN (cleaned up after each SN)
```

### Chunked Approach
```
Memory accumulation:
- Query results: N × 1-10 MB (all SNe)
- Downloaded FITS: N × 50-500 MB (all SNe)
- Organized files: N × 50-500 MB (all SNe)
Total: N × 100-1000 MB (all in memory at once)
```

For 1000 SNe:
- Per-SN: ~100-1000 MB peak
- Chunked: ~100-1000 GB peak (1000× more!)

## Recommendation

**Use Per-SN Pipeline** for:
- Large datasets (>100 SNe)
- Memory-constrained systems
- When you want incremental progress
- When you need to resume frequently

**Use Chunked Pipeline** for:
- Small datasets (<50 SNe)
- Systems with plenty of RAM (>32GB)
- When you need all data before processing
- When you want to analyze all SNe together

## Implementation

### Per-SN Pipeline Usage

```bash
# Process all SNe one at a time
python scripts/run_pipeline_per_sn.py \
    --config configs/galex_golden_era.yaml

# Resume from checkpoint
python scripts/run_pipeline_per_sn.py \
    --config configs/galex_golden_era.yaml \
    --resume

# Process first 10 SNe
python scripts/run_pipeline_per_sn.py \
    --config configs/galex_golden_era.yaml \
    --limit 10

# With options
python scripts/run_pipeline_per_sn.py \
    --config configs/galex_golden_era.yaml \
    --decompress \
    --visualize
```

### Chunked Pipeline Usage (Current)

```bash
# Process all stages for all SNe
python scripts/run_pipeline_from_config.py \
    --config configs/galex_golden_era.yaml

# Process one stage at a time
python scripts/run_pipeline_from_config.py \
    --config configs/galex_golden_era.yaml \
    --stage query
```

## Hybrid Approach

You could also combine both:
1. Use chunked query stage (processes in chunks, saves to files)
2. Use per-SN for download/organize/differencing (processes one SN at a time)

This gives you:
- Efficient querying (chunked)
- Low memory for processing (per-SN)

## Conclusion

**For your use case (large datasets, memory constraints), the Per-SN approach is recommended.**

The new `run_pipeline_per_sn.py` script implements this approach and should prevent OOM kills while still processing all your data efficiently.
