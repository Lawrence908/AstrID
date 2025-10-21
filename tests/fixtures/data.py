"""Data fixtures for testing.

Provides:
- Sample astronomical data
- Test FITS file data
- Mock image data
- Coordinate data
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture
def sample_fits_header() -> dict[str, Any]:
    """Sample FITS header data for testing."""
    return {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": 1024,
        "NAXIS2": 1024,
        "EXTEND": True,
        "OBJECT": "Test Object",
        "TELESCOP": "Test Telescope",
        "INSTRUME": "Test Instrument",
        "FILTER": "g",
        "EXPTIME": 300.0,
        "DATE-OBS": "2024-09-19T12:00:00.000",
        "RA": 180.0,
        "DEC": 45.0,
        "AIRMASS": 1.2,
        "SEEING": 1.5,
        "SKYBKG": 20.5,
        "GAIN": 1.0,
        "RDNOISE": 5.0,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 512.0,
        "CRPIX2": 512.0,
        "CRVAL1": 180.0,
        "CRVAL2": 45.0,
        "CD1_1": -5.5556e-05,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 5.5556e-05,
    }


@pytest.fixture
def sample_image_data() -> np.ndarray:
    """Sample 2D image data for testing."""
    # Create a 1024x1024 image with some structure
    np.random.seed(42)  # For reproducible tests

    # Base noise level
    image = np.random.normal(1000, 50, (1024, 1024)).astype(np.float32)

    # Add some "stars"
    star_positions = [
        (100, 100),
        (300, 200),
        (500, 400),
        (700, 600),
        (900, 800),
    ]

    for x, y in star_positions:
        # Create a Gaussian profile for each star
        xx, yy = np.meshgrid(
            np.arange(max(0, x - 20), min(1024, x + 21)),
            np.arange(max(0, y - 20), min(1024, y + 21)),
        )
        star_profile = 5000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3**2))

        x_slice = slice(max(0, x - 20), min(1024, x + 21))
        y_slice = slice(max(0, y - 20), min(1024, y + 21))
        image[y_slice, x_slice] += star_profile

    return image


@pytest.fixture
def sample_difference_image() -> np.ndarray:
    """Sample difference image data for testing."""
    np.random.seed(123)

    # Create a difference image with mostly noise and a few transients
    diff_image = np.random.normal(0, 10, (1024, 1024)).astype(np.float32)

    # Add some transient detections
    transient_positions = [
        (250, 300),
        (600, 450),
        (800, 200),
    ]

    for x, y in transient_positions:
        # Create positive and negative sources
        sign = 1 if np.random.random() > 0.5 else -1
        magnitude = sign * 500 * np.random.uniform(0.5, 2.0)

        xx, yy = np.meshgrid(
            np.arange(max(0, x - 10), min(1024, x + 11)),
            np.arange(max(0, y - 10), min(1024, y + 11)),
        )
        transient_profile = magnitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2**2)
        )

        x_slice = slice(max(0, x - 10), min(1024, x + 11))
        y_slice = slice(max(0, y - 10), min(1024, y + 11))
        diff_image[y_slice, x_slice] += transient_profile

    return diff_image


@pytest.fixture
def sample_calibration_frames() -> dict[str, np.ndarray]:
    """Sample calibration frames for testing."""
    np.random.seed(456)

    return {
        "bias": np.random.normal(100, 5, (1024, 1024)).astype(np.float32),
        "dark": np.random.normal(10, 2, (1024, 1024)).astype(np.float32),
        "flat": np.random.normal(1.0, 0.05, (1024, 1024)).astype(np.float32),
    }


@pytest.fixture
def sample_wcs_data() -> dict[str, Any]:
    """Sample WCS (World Coordinate System) data for testing."""
    return {
        "ctype": ["RA---TAN", "DEC--TAN"],
        "crpix": [512.0, 512.0],
        "crval": [180.0, 45.0],
        "cdelt": [-5.5556e-05, 5.5556e-05],
        "cd": [[-5.5556e-05, 0.0], [0.0, 5.5556e-05]],
        "pc": [[1.0, 0.0], [0.0, 1.0]],
    }


@pytest.fixture
def sample_source_catalog() -> list[dict[str, Any]]:
    """Sample source catalog for testing."""
    return [
        {
            "id": 1,
            "x": 100.5,
            "y": 100.7,
            "ra": 179.99,
            "dec": 44.99,
            "flux": 10000,
            "flux_err": 100,
            "mag": 16.5,
            "mag_err": 0.01,
            "fwhm": 2.1,
            "ellipticity": 0.05,
            "theta": 45.0,
            "class_star": 0.95,
        },
        {
            "id": 2,
            "x": 300.2,
            "y": 200.8,
            "ra": 180.01,
            "dec": 45.01,
            "flux": 8000,
            "flux_err": 120,
            "mag": 16.8,
            "mag_err": 0.015,
            "fwhm": 2.3,
            "ellipticity": 0.08,
            "theta": 30.0,
            "class_star": 0.98,
        },
        {
            "id": 3,
            "x": 500.1,
            "y": 400.6,
            "ra": 180.02,
            "dec": 45.02,
            "flux": 15000,
            "flux_err": 80,
            "mag": 16.1,
            "mag_err": 0.008,
            "fwhm": 1.9,
            "ellipticity": 0.03,
            "theta": 60.0,
            "class_star": 0.99,
        },
    ]


@pytest.fixture
def sample_detection_candidates() -> list[dict[str, Any]]:
    """Sample detection candidates for testing."""
    return [
        {
            "x": 250.3,
            "y": 300.7,
            "ra": 180.001,
            "dec": 45.001,
            "flux": 500,
            "flux_err": 25,
            "snr": 20.0,
            "significance": 5.2,
            "class_prob": 0.85,
            "is_real": True,
        },
        {
            "x": 600.8,
            "y": 450.2,
            "ra": 180.002,
            "dec": 45.002,
            "flux": -300,
            "flux_err": 30,
            "snr": -10.0,
            "significance": 3.8,
            "class_prob": 0.72,
            "is_real": True,
        },
        {
            "x": 800.1,
            "y": 200.9,
            "ra": 180.003,
            "dec": 45.003,
            "flux": 150,
            "flux_err": 50,
            "snr": 3.0,
            "significance": 2.1,
            "class_prob": 0.45,
            "is_real": False,
        },
    ]


@pytest.fixture
def sample_filter_data() -> dict[str, dict[str, Any]]:
    """Sample filter information for testing."""
    return {
        "g": {
            "name": "g",
            "central_wavelength": 4770,  # Angstroms
            "bandwidth": 1370,
            "zero_point": 25.0,
            "extinction_coeff": 0.15,
        },
        "r": {
            "name": "r",
            "central_wavelength": 6230,
            "bandwidth": 1380,
            "zero_point": 25.1,
            "extinction_coeff": 0.10,
        },
        "i": {
            "name": "i",
            "central_wavelength": 7630,
            "bandwidth": 1520,
            "zero_point": 25.0,
            "extinction_coeff": 0.08,
        },
    }


@pytest.fixture
def sample_psf_data() -> dict[str, Any]:
    """Sample PSF (Point Spread Function) data for testing."""
    np.random.seed(789)

    # Create a simple Gaussian PSF
    size = 25
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    center = size // 2

    # PSF parameters
    fwhm = 2.5
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    psf = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    psf = psf / np.sum(psf)  # Normalize

    return {
        "psf_image": psf.astype(np.float32),
        "fwhm": fwhm,
        "sigma": sigma,
        "size": size,
        "center": [center, center],
        "ellipticity": 0.1,
        "position_angle": 30.0,
    }


@pytest.fixture
def sample_time_series_data() -> dict[str, Any]:
    """Sample time series data for testing."""
    np.random.seed(101)

    # Generate a time series with some variability
    n_points = 50
    times = np.linspace(0, 100, n_points)  # Days

    # Base magnitude with some trend and noise
    base_mag = 18.5
    trend = 0.01 * times  # Slow brightening
    noise = np.random.normal(0, 0.05, n_points)

    # Add some periodic variability
    period = 15.0  # days
    amplitude = 0.2
    periodic = amplitude * np.sin(2 * np.pi * times / period)

    magnitudes = base_mag + trend + periodic + noise
    mag_errors = np.random.uniform(0.03, 0.08, n_points)

    return {
        "times": times,
        "magnitudes": magnitudes,
        "magnitude_errors": mag_errors,
        "period": period,
        "amplitude": amplitude,
        "base_magnitude": base_mag,
    }


@pytest.fixture
def sample_spectral_data() -> dict[str, Any]:
    """Sample spectral data for testing."""
    np.random.seed(202)

    # Simple emission line spectrum
    wavelengths = np.linspace(4000, 7000, 1000)  # Angstroms

    # Continuum
    continuum = 1e-16 * (wavelengths / 5000) ** (-1.5)

    # Add emission lines
    line_centers = [4861, 5007, 6563]  # H-beta, [OIII], H-alpha
    line_fluxes = [5e-16, 3e-16, 8e-16]
    line_widths = [50, 30, 60]  # Angstroms

    spectrum = continuum.copy()
    for center, flux, width in zip(
        line_centers, line_fluxes, line_widths, strict=False
    ):
        line = flux * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        spectrum += line

    # Add noise
    noise = np.random.normal(0, 1e-17, len(wavelengths))
    spectrum += noise

    return {
        "wavelengths": wavelengths,
        "flux": spectrum,
        "flux_errors": np.full_like(spectrum, 1e-17),
        "continuum": continuum,
        "line_centers": line_centers,
        "line_fluxes": line_fluxes,
    }
