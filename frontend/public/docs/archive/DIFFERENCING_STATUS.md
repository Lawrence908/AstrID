# Difference Image Generation - Status Report

## ✅ PROCESSING COMPLETE!

The difference image generation has successfully completed!

### Summary
- **Total SNe processed**: 45
- **SWIFT mission**: 17 SNe (UV data)
- **PS1 mission**: 28 SNe (Optical data)
- **Output directory**: `output/difference_images/`

### Processed SNe by Mission

#### SWIFT (17 SNe)
2005nc, 2006X, 2006et, 2006gy, 2006mr, 2007ah, 2007oc, 2008D, 2008S, 2008dt, 2008dx, 2008fh, 2008fo, 2008ij, 2009gf, 2009hd, 2009mg

#### PS1 (28 SNe)
2009hk, 2009hl, 2009hz, 2009ih, 2009in, 2009io, 2009jp, 2009jq, 2009ka, 2009kc, 2009kw, 2009ky, 2009ld, 2009le, 2009lg, 2009lo, 2009lw, 2009lz, 2009ml, 2009nn, 2009no, 2009nr, 2009nx, 2010G, 2010N, 2010R, 2010bb, 2010bc

### What Happened

1. **Initial run with `--visualize`**: The script was killed by the OS due to memory exhaustion when processing 2009hk (first PS1 target with larger images)
2. **Second run without `--visualize`**: Successfully completed all 45 SNe

### Memory Improvements Made

The script has been updated with better memory management:
- ✅ Use `float32` instead of `float64` (50% memory reduction)
- ✅ Memory-mapped FITS file loading
- ✅ Explicit cleanup of intermediate arrays
- ✅ Batch processing with garbage collection
- ✅ Better matplotlib memory management
- ✅ New `--batch-size` parameter for tuning

### Output Files

For each SN, you have:
```
output/difference_images/{sn_name}/
  ├── {sn_name}_{mission}_{filter}_diff.fits  # Difference image (science - reference)
  ├── {sn_name}_{mission}_{filter}_sig.fits   # Significance map (σ units)
  └── {sn_name}_{mission}_{filter}_mask.fits  # SN position mask
```

Plus a summary file:
```
output/difference_images/processing_summary.json  # Complete processing metadata
```

### Example: Checking Results

```bash
# View summary
cat output/difference_images/processing_summary.json | python3 -m json.tool | less

# Check a specific SN
ls -lh output/difference_images/2009hk/

# Count total files
find output/difference_images -name "*.fits" | wc -l
```

### Optional: Generate Visualizations

Now that processing is complete, you can optionally generate visualization plots for specific SNe:

```bash
# Activate environment
source /home/chris/AstrID/.venv/bin/activate

# Generate visualizations for a few interesting SNe
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training \
  --output-dir output/difference_images \
  --visualize \
  --sn 2009hk 2009hl 2009hz 2008D 2006gy

# Or visualize all SWIFT (smaller, less memory intensive)
python3 scripts/generate_difference_images.py \
  --input-dir output/fits_training \
  --output-dir output/difference_images \
  --visualize \
  --mission SWIFT \
  --batch-size 5
```

### Next Steps

1. **Verify quality**: Spot-check a few difference images using DS9 or Python
2. **Training pipeline**: Use these difference images for model training
3. **Analysis**: Examine the significance maps for interesting detections

### Quality Metrics from Summary

Key metrics to check in `processing_summary.json`:
- `overlap_fraction`: Percentage of image overlap (higher is better, >80% is good)
- `sig_max`: Maximum significance in the difference image (indicates detection strength)
- `n_detections`: Number of sources detected above threshold
- `ref_fwhm`, `sci_fwhm`: PSF quality (typical range: 2-8 pixels)

### Known Issues Resolved

- ✅ Memory exhaustion with visualization → Use `--visualize` selectively
- ✅ OOM kills on PS1 data → Reduced memory usage with float32 and cleanup
- ✅ Swap thrashing → Added batch processing with GC

## Files Modified

- `scripts/generate_difference_images.py` - Added memory optimizations
- `NEXT_STEPS_DIFFERENCING.md` - Documentation for future runs
- `resume_differencing.sh` - Helper script (not needed now)
- `resume_differencing_by_mission.sh` - Helper script (not needed now)
