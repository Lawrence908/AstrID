# Difference Image Generation - Memory Fixes

## Problem
The `generate_difference_images.py` script was being killed by the OS due to memory exhaustion when processing large FITS images, particularly when processing PS1 optical data which has larger image dimensions.

## Changes Made

### 1. Memory-Efficient Data Loading
- Changed FITS loading to use `memmap=True` for memory-mapped file access
- Switched from `float64` to `float32` arrays (50% memory reduction)
- Added explicit cleanup of intermediate arrays

### 2. Improved Convolution
- Changed `nan_treatment="fill"` to `preserve_nan=True` in PSF matching (more memory efficient)
- Added cleanup after convolution operations

### 3. Background Estimation
- Added explicit copy and deletion of Background2D objects
- Ensured background arrays use float32

### 4. Visualization Memory Management
- Set matplotlib to use non-interactive 'Agg' backend
- Explicitly close and delete figure objects after saving
- Added cleanup after visualization

### 5. Batch Processing
- Added `--batch-size` parameter (default: 10) to force garbage collection between batches
- Added explicit `gc.collect()` calls after each SN and on errors
- Track progress with batch counters

### 6. Explicit Cleanup
- Delete large intermediate arrays immediately after use:
  - Reference and science images after alignment
  - Background-subtracted images after PSF matching
  - Matched images after normalization
  - Difference, significance, and mask arrays after saving

## Usage

**IMPORTANT**: Activate the virtual environment first:
```bash
source /home/chris/AstrID/.venv/bin/activate
```

### Basic Usage (Memory Optimized)
```bash
# Process all SNe with default batch size (10)
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training \
  --output-dir output/difference_images

# Process with smaller batches for very low memory systems
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training \
  --output-dir output/difference_images \
  --batch-size 5

# Process specific mission only (reduces memory by limiting scope)
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training \
  --output-dir output/difference_images \
  --mission SWIFT

# Process with visualization (uses more memory)
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training \
  --output-dir output/difference_images \
  --visualize \
  --batch-size 5  # Use smaller batches with visualization
```

### Processing Specific SNe
```bash
# Process just a few SNe for testing
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training \
  --output-dir output/difference_images \
  --sn 2009hk 2009hl 2009hz

# Resume from where it stopped (2009hk onwards)
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training \
  --output-dir output/difference_images \
  --sn 2009hk 2009hl 2009hz 2009ih 2009in 2009io 2009jf 2009jp 2009jq 2009ka 2009kc 2009kr 2009kw 2009ky 2009ld 2009le 2009lg 2009lo 2009lw 2009lz 2009mg 2009ml 2009nn 2009no 2009nr 2009nx 2009nz 2010G 2010N 2010R 2010bb 2010bc
```

## Memory Requirements

### Estimated Memory Usage per SN
- **SWIFT UV**: ~500-800 MB per SN
- **GALEX UV**: ~800-1200 MB per SN  
- **PS1 Optical**: ~1500-2500 MB per SN (larger images)

### Recommendations
- **System with 8GB RAM**: Use `--batch-size 3` and process one mission at a time
- **System with 16GB RAM**: Use `--batch-size 10` (default)
- **System with 32GB+ RAM**: Can use `--batch-size 20` or larger

### Current System Status
Your system has:
- Total RAM: 14GB
- Available: ~6GB
- Swap: Nearly full (7.9GB/8GB used)

**Recommendation**: Use `--batch-size 5` and avoid visualization initially, or process one mission at a time.

## Monitoring Memory

### Check memory during processing
```bash
# In another terminal, watch memory usage
watch -n 2 'free -h && echo "---" && ps aux | grep generate_difference | grep -v grep'
```

### If the script gets killed again
1. Reduce batch size further: `--batch-size 3`
2. Process missions separately:
   ```bash
   # Process SWIFT only (smaller images)
   python3 scripts/generate_difference_images.py \
     --input-dir output/fits_training \
     --output-dir output/difference_images \
     --mission SWIFT
   
   # Then GALEX
   python3 scripts/generate_difference_images.py \
     --input-dir output/fits_training \
     --output-dir output/difference_images \
     --mission GALEX
   
   # Finally PS1 (largest images)
   python3 scripts/generate_difference_images.py \
     --input-dir output/fits_training \
     --output-dir output/difference_images \
     --mission PS1 \
     --batch-size 3
   ```
3. Close other applications to free up RAM
4. Clear swap: `sudo swapoff -a && sudo swapon -a` (requires sudo)

## Next Steps

1. **Resume Processing**: Run the script with the remaining SNe starting from 2009hk
2. **Monitor Progress**: Check the output directory for completed files
3. **Verify Results**: Check `processing_summary.json` for completion status
4. **Generate Visualizations**: Once all processing is complete, you can optionally re-run with `--visualize` on specific SNe

## Expected Output

After successful completion, you should have:
- `output/difference_images/processing_summary.json` - Overall summary
- `output/difference_images/{sn_name}/` - Directory per SN containing:
  - `{sn_name}_{mission}_{filter}_diff.fits` - Difference image
  - `{sn_name}_{mission}_{filter}_sig.fits` - Significance map
  - `{sn_name}_{mission}_{filter}_mask.fits` - SN position mask
  - `{sn_name}_{mission}_{filter}_viz.png` - Visualization (if --visualize used)
