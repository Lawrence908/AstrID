"""Utilities for creating synthetic FITS files with valid WCS for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def create_test_fits_file(
    temp_dir: str, size: tuple[int, int] = (100, 100), with_wcs: bool = True
) -> str:
    """Create a test FITS file with optional, valid WCS information.

    This helper constructs a robust WCS using a CD matrix (preferred over CDELT)
    to avoid common validation issues and ensures headers are standards-compliant.

    Args:
        temp_dir: Directory to write the FITS file into.
        size: Image shape (ny, nx).
        with_wcs: Whether to include a valid WCS in the header.

    Returns:
        Absolute path to the created FITS file.
    """
    ny, nx = size
    data = np.random.random((ny, nx)).astype(np.float32) * 1000.0 + 100.0

    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["DATE-OBS"] = "2023-01-01T12:00:00"

    if with_wcs:
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cunit = ["deg", "deg"]
        w.wcs.crval = [180.0, 0.0]
        w.wcs.crpix = [nx / 2.0, ny / 2.0]
        # ~0.001 deg/pixel scale with no rotation
        w.wcs.cd = [[-0.001, 0.0], [0.0, 0.001]]
        w.wcs.radesys = "ICRS"
        w.wcs.equinox = 2000.0
        # Merge WCS header (includes CTYPE, CRVAL, CRPIX, and CD keywords)
        wcs_hdr = w.to_header()
        header.update(wcs_hdr)
        # Explicitly assert presence of core WCS keywords for downstream validators
        header["CTYPE1"] = header.get("CTYPE1", "RA---TAN")
        header["CTYPE2"] = header.get("CTYPE2", "DEC--TAN")
        header["CRVAL1"] = header.get("CRVAL1", 180.0)
        header["CRVAL2"] = header.get("CRVAL2", 0.0)
        header["CRPIX1"] = header.get("CRPIX1", nx / 2.0)
        header["CRPIX2"] = header.get("CRPIX2", ny / 2.0)
        # Ensure CD matrix keywords exist explicitly (some tools expect these forms)
        header["CD1_1"] = float(w.wcs.cd[0][0])
        header["CD1_2"] = float(w.wcs.cd[0][1])
        header["CD2_1"] = float(w.wcs.cd[1][0])
        header["CD2_2"] = float(w.wcs.cd[1][1])

    # Add a few common instrument/obs keys (optional)
    header["EXPTIME"] = 60.0
    header["FILTER"] = "V"
    header["TELESCOP"] = "Test Telescope"
    header["INSTRUME"] = "Test Camera"

    out_path = Path(temp_dir) / f"test_{ny}x{nx}.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(out_path, overwrite=True)
    return str(out_path)
