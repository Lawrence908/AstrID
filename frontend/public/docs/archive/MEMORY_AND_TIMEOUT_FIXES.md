# Memory and Timeout Fixes for Query Pipeline

## Problem Summary

The supernova dataset gathering pipeline was failing with:
1. **SIGKILL (signal 9)**: Process killed by OOM (Out-Of-Memory) killer
2. **MAST query timeouts**: Many queries timing out after 30 seconds
3. **Memory accumulation**: Large result sets (46,890+ observations) consuming too much memory

## Root Causes

1. **Short timeout**: Default 30s timeout too short for large MAST queries
2. **No memory cleanup**: Large astropy `obs_table` objects and Time objects not being freed
3. **Memory accumulation**: Processing many large result sets without garbage collection
4. **Large result sets**: Some positions return 46,890+ observations, overwhelming memory

## Fixes Applied

### 1. Increased Timeout (120s)
- **File**: `src/adapters/external/archive_router.py`
- **Change**: Increased default timeout from 30s to 120s
- **Impact**: Allows large queries to complete without timing out

### 2. Memory Cleanup in MAST Adapter
- **File**: `src/adapters/external/mast.py`
- **Changes**:
  - Added explicit deletion of `obs_table` after processing
  - Added garbage collection after processing large result sets (>1000 observations)
  - Reduced max processing multiplier from 10x to 5x to limit memory usage
  - Added dynamic max_results reduction for very large queries (>20,000 observations)

### 3. Memory Cleanup in Chunked Script
- **File**: `scripts/query_sn_fits_chunked.py`
- **Changes**:
  - Added `gc` import
  - Added garbage collection every 10 supernovae to prevent memory accumulation

### 4. Reduced Memory Limits
- **File**: `src/adapters/external/mast.py`
- **Changes**:
  - Reduced processing limit from `max_results * 10` to `max_results * 5`
  - Added cap of 5000 observations for very large queries (>20,000 raw results)

## Recommendations

### Running One Pipeline at a Time
**Yes, you should run one pipeline at a time** to avoid:
- Memory contention between processes
- Network rate limiting from MAST
- OOM killer killing processes

### Configuration Adjustments

For very large datasets, consider:

1. **Smaller chunk sizes**:
   ```yaml
   query:
     chunk_size: 50  # Instead of 150
   ```

2. **Smaller radius**:
   ```yaml
   query:
     radius_deg: 0.1  # Instead of 0.15 or 0.2
   ```

3. **Mission filtering**:
   ```yaml
   query:
     missions: ["GALEX"]  # Single mission instead of multiple
   ```

4. **Year range limiting**:
   ```yaml
   query:
     min_year: 2005
     max_year: 2011  # Narrower range
   ```

### Monitoring Memory Usage

To monitor memory usage while running:
```bash
# In another terminal
watch -n 1 'free -h && ps aux | grep query_sn_fits | head -5'
```

### If Still Running Out of Memory

1. **Reduce chunk size** in config (e.g., 50 instead of 150)
2. **Process in smaller batches** using `--start-index` and `--limit`
3. **Increase system swap** if possible
4. **Use a machine with more RAM** for very large datasets

## Testing

After these fixes, the pipeline should:
- Handle large result sets without OOM kills
- Complete queries that previously timed out
- Use memory more efficiently with periodic cleanup
- Be more resilient to memory pressure

## Next Steps

1. Test with `sn2014j_only.yaml` (should still work)
2. Test with `galex_golden_era.yaml` (should now complete)
3. Test with `best_yield_combined.yaml` (may need smaller chunk size)

If issues persist, consider:
- Further reducing chunk sizes
- Processing missions separately
- Using a machine with more RAM
