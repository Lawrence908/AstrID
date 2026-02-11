"""
Shared FITS loading with WCS extraction and validation.

Provides a single strategy for loading FITS image data and WCS across the pipeline,
preserving astropy fits.Header for reliable WCS reconstruction (not plain dict).
Follows the pattern from OLD/imageProcessing.py (extractWCSFromFits) and
OLD/dataGathering.py (getPixelCoordsFromStar).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


def _log_wcs_params(filepath: Path, hdu_index: int, wcs: WCS) -> None:
    """Log key WCS parameters for debugging."""
    try:
        if wcs.has_celestial:
            crval = wcs.wcs.crval
            crpix = wcs.wcs.crpix
            ctype = getattr(wcs.wcs, "ctype", ["?", "?"])
            logger.debug(
                f"WCS {filepath.name} HDU {hdu_index}: "
                f"CRVAL=({crval[0]:.6f}, {crval[1]:.6f}), "
                f"CRPIX=({crpix[0]:.2f}, {crpix[1]:.2f}), "
                f"CTYPE=({ctype[0]!s}, {ctype[1]!s})"
            )
    except Exception as e:
        logger.debug(f"Could not log WCS params for {filepath.name}: {e}")


def load_fits_with_wcs(
    filepath: Path,
    *,
    verify_wcs: bool = True,
    memmap: bool = False,
    return_dict_header: bool = False,
) -> tuple[np.ndarray, fits.Header | dict[str, Any], WCS | None]:
    """
    Load FITS image data, header, and WCS using a consistent HDU strategy.

    Tries primary HDU first; if it has no 2D data, iterates extensions until
    the first HDU with 2D image data is found. WCS is built from that HDU's
    header (preserved as fits.Header for reliable WCS reconstruction).

    Args:
        filepath: Path to the FITS file.
        verify_wcs: If True, require valid celestial WCS; raise if missing/invalid.
        memmap: If True, open with memmap=True (for large files).
        return_dict_header: If True, also return header as dict for backward compat;
            primary return is always fits.Header.

    Returns:
        (data, header, wcs) where:
        - data: float32 2D array (first slice if 3D).
        - header: astropy.io.fits.Header from the selected HDU (or dict if requested).
        - wcs: WCS object or None if WCS could not be built.

    Raises:
        ValueError: No image data found, or verify_wcs=True and WCS invalid/missing.
    """
    open_kw: dict[str, Any] = {}
    if memmap:
        open_kw["memmap"] = True

    with fits.open(filepath, **open_kw) as hdul:
        data: np.ndarray | None = None
        header: fits.Header | None = None
        hdu_index = -1

        # Try primary first, then extensions (consistent strategy everywhere)
        for i, hdu in enumerate(hdul):
            if hdu.data is None or len(hdu.data.shape) < 2:
                continue
            d = hdu.data
            if len(d.shape) == 3:
                d = d[0]
            elif len(d.shape) > 3:
                d = d[0, 0]
            data = d.astype(np.float32)
            header = hdu.header.copy()
            hdu_index = i
            break

        if data is None or header is None:
            raise ValueError(f"No image data found in {filepath}")

        wcs = None
        try:
            wcs = WCS(header, naxis=2)
        except Exception as e:
            logger.warning(f"Could not build WCS from {filepath.name} HDU {hdu_index}: {e}")

        if verify_wcs:
            if wcs is None:
                raise ValueError(f"No valid WCS in {filepath} (verify_wcs=True)")
            if not wcs.has_celestial:
                raise ValueError(
                    f"WCS in {filepath} is not celestial (has_celestial=False, verify_wcs=True)"
                )

        if wcs is not None:
            _log_wcs_params(filepath, hdu_index, wcs)
            logger.debug(f"Loaded {filepath.name} from HDU {hdu_index}, shape={data.shape}")

        out_header: fits.Header | dict[str, Any] = header
        if return_dict_header:
            out_header = dict(header)

        return data, out_header, wcs
