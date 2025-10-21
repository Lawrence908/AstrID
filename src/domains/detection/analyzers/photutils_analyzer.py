"""Photutils-based analysis utilities for ASTR-79."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
from photutils.detection import find_peaks
from photutils.segmentation import make_2dgaussian_kernel

from src.domains.detection.extractors.sep_extractor import (
    Source,
)


@dataclass
class PhotometryConfig:
    aperture_radius: float = 3.0
    annulus_inner: float = 6.0
    annulus_outer: float = 8.0


class PhotutilsAnalyzer:
    """Analyzer for photometric and morphological measurements using photutils."""

    def __init__(self, phot_config: PhotometryConfig | None = None) -> None:
        self.phot_config = phot_config or PhotometryConfig()

    def analyze_source_properties(
        self, sources: list[Source], image: np.ndarray
    ) -> list[dict]:
        results: list[dict] = []
        for s in sources:
            x, y = s.coordinates
            r = self.phot_config.aperture_radius
            aperture = CircularAperture([(x, y)], r=r)
            annulus = CircularAnnulus(
                [(x, y)],
                r_in=self.phot_config.annulus_inner,
                r_out=self.phot_config.annulus_outer,
            )
            aper_stats = ApertureStats(image, aperture)
            ann_stats = ApertureStats(image, annulus)
            results.append(
                {
                    "source_id": str(s.source_id),
                    "x": x,
                    "y": y,
                    "flux_sum": float(aper_stats.sum)
                    if np.isfinite(aper_stats.sum)
                    else None,
                    "local_bkg_mean": float(ann_stats.mean)
                    if np.isfinite(ann_stats.mean)
                    else None,
                    "max": float(aper_stats.max)
                    if np.isfinite(aper_stats.max)
                    else None,
                    "min": float(aper_stats.min)
                    if np.isfinite(aper_stats.min)
                    else None,
                    "std": float(aper_stats.std)
                    if np.isfinite(aper_stats.std)
                    else None,
                }
            )
        return results

    def calculate_aperture_photometry(
        self, sources: list[Source], image: np.ndarray
    ) -> list[dict]:
        results: list[dict] = []
        for s in sources:
            x, y = s.coordinates
            r = self.phot_config.aperture_radius
            aperture = CircularAperture([(x, y)], r=r)
            stats = ApertureStats(image, aperture)
            results.append(
                {
                    "source_id": str(s.source_id),
                    "aperture_flux": float(stats.sum)
                    if np.isfinite(stats.sum)
                    else None,
                    "aperture_mean": float(stats.mean)
                    if np.isfinite(stats.mean)
                    else None,
                    "aperture_median": float(stats.median)
                    if np.isfinite(stats.median)
                    else None,
                }
            )
        return results

    def measure_source_morphology(
        self, sources: list[Source], image: np.ndarray
    ) -> list[dict]:
        results: list[dict] = []
        for s in sources:
            x, y = s.coordinates
            r = max(self.phot_config.aperture_radius, 3.0)
            aperture = CircularAperture([(x, y)], r=r)
            stats = ApertureStats(image, aperture, wcs=None)
            results.append(
                {
                    "source_id": str(s.source_id),
                    "xcentroid": float(stats.xcentroid)
                    if stats.xcentroid is not None
                    else None,
                    "ycentroid": float(stats.ycentroid)
                    if stats.ycentroid is not None
                    else None,
                    "semimajor_sigma": float(stats.semimajor_sigma)
                    if stats.semimajor_sigma is not None
                    else None,
                    "semiminor_sigma": float(stats.semiminor_sigma)
                    if stats.semiminor_sigma is not None
                    else None,
                    "orientation": float(stats.orientation)
                    if stats.orientation is not None
                    else None,
                    "eccentricity": float(stats.eccentricity)
                    if stats.eccentricity is not None
                    else None,
                }
            )
        return results

    def detect_source_peaks(
        self, sources: list[Source], image: np.ndarray
    ) -> list[dict]:
        kernel = make_2dgaussian_kernel(1.5, size=5)
        peaks_tbl = find_peaks(
            image, threshold=np.mean(image) + 3 * np.std(image), footprint=kernel.array
        )
        peaks: list[dict] = []
        if peaks_tbl is not None:
            for row in peaks_tbl:
                peaks.append(
                    {
                        "x_peak": float(row["x_peak"]),
                        "y_peak": float(row["y_peak"]),
                        "peak_value": float(row["peak_value"]),
                    }
                )
        return peaks

    def calculate_source_statistics(
        self, sources: list[Source], image: np.ndarray
    ) -> dict:
        snrs = [s.snr for s in sources if s.snr is not None]
        sizes = [s.size for s in sources if s.size is not None]
        return {
            "num_sources": len(sources),
            "snr_mean": float(np.mean(snrs)) if snrs else None,
            "snr_median": float(np.median(snrs)) if snrs else None,
            "size_mean": float(np.mean(sizes)) if sizes else None,
            "size_median": float(np.median(sizes)) if sizes else None,
        }
