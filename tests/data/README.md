# Test Data

This directory holds small, deterministic samples used by tests.

Contents:
- npy/  - small NumPy arrays derived from notebooks
- fits/ - tiny FITS files generated or copied from notebook runs (sanitized)
- golden/*.json - expected metrics and values for assertions

Refreshing from notebooks:
1) Export minimal arrays (float32, downsampled) from notebooks to `tests/data/npy/`.
2) Export expected metrics to `tests/data/golden/*.json`.
3) For FITS, prefer generating in tests via `tests/utils.FileTestUtils.create_temp_fits_file` unless a specific header is needed.

Guidelines:
- Keep single file < 100KB where possible
- Use deterministic seeds
- Avoid sensitive tokens/URLs in headers
- Document source notebook and cell where sample came from
