"""SEP-based source extraction utilities for difference images (ASTR-79)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import sep  # type: ignore


@dataclass
class Source:
    source_id: UUID
    coordinates: tuple[float, float]
    world_coordinates: tuple[float, float] | None
    flux: float | None
    flux_error: float | None
    snr: float | None
    size: float | None
    ellipticity: float | None
    position_angle: float | None
    peak_value: float | None
    background: float | None
    noise: float | None
    quality_flags: list[str] = field(default_factory=list)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceExtractionConfig:
    detection_threshold: float = 1.5
    min_pixels: int = 5
    deblend_threshold: float = 32.0
    deblend_cont: float = 0.005
    clean_param: float = 1.0
    filter_kernel: str | None = None
    background_estimation: str = "global"
    noise_estimation: str = "local"
    aperture_radius: float = 3.0
    annulus_inner: float = 6.0
    annulus_outer: float = 8.0


class SEPExtractor:
    """Wrapper around the `sep` library for source extraction."""

    def __init__(self, config: SourceExtractionConfig | None = None) -> None:
        self.config = config or SourceExtractionConfig()

    def _detect(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        threshold = kwargs.get("threshold", self.config.detection_threshold)
        minarea = kwargs.get("minarea", self.config.min_pixels)
        deblend_nthresh = kwargs.get(
            "deblend_nthresh", int(self.config.deblend_threshold)
        )
        deblend_cont = kwargs.get("deblend_cont", self.config.deblend_cont)
        clean_param = kwargs.get("clean_param", self.config.clean_param)

        objects = sep.extract(
            image,
            thresh=threshold,
            minarea=minarea,
            deblend_nthresh=deblend_nthresh,
            deblend_cont=deblend_cont,
            clean=True,
            clean_param=clean_param,
        )
        return objects

    def _to_sources(
        self,
        objects: np.ndarray,
        image: np.ndarray,
        background: np.ndarray | None = None,
        noise: np.ndarray | None = None,
    ) -> list[Source]:
        sources: list[Source] = []
        for obj in objects:
            x = float(obj["x"])  # centroid x
            y = float(obj["y"])  # centroid y

            names = obj.dtype.names or ()
            a = float(obj["a"]) if "a" in names else np.nan
            b = float(obj["b"]) if "b" in names else np.nan
            theta = float(obj["theta"]) if "theta" in names else np.nan
            peak = float(obj["peak"]) if "peak" in names else np.nan
            area = float(obj["area"]) if "area" in names else np.nan
            ellip = None
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                ellip = 1.0 - (b / a)

            # Simple local noise estimate if not provided
            noise_value: float | None
            if noise is not None:
                noise_value = float(noise[int(round(y)), int(round(x))])
            else:
                # Use local RMS in small window
                y0 = max(int(y) - 3, 0)
                y1 = min(int(y) + 4, image.shape[0])
                x0 = max(int(x) - 3, 0)
                x1 = min(int(x) + 4, image.shape[1])
                patch = image[y0:y1, x0:x1]
                noise_value = float(np.std(patch)) if patch.size > 0 else None

            bkg_value: float | None
            if background is not None:
                bkg_value = float(background[int(round(y)), int(round(x))])
            else:
                bkg_value = None

            # SEP flux/fluxerr if present
            flux = float(obj["flux"]) if "flux" in names else np.nan
            fluxerr = float(obj["fluxerr"]) if "fluxerr" in names else np.nan

            snr = None
            if np.isfinite(flux) and np.isfinite(fluxerr) and fluxerr > 0:
                snr = float(flux / fluxerr)
            elif (
                noise_value is not None and np.isfinite(noise_value) and noise_value > 0
            ):
                snr = float(peak / noise_value) if np.isfinite(peak) else None

            sources.append(
                Source(
                    source_id=uuid4(),
                    coordinates=(x, y),
                    world_coordinates=None,
                    flux=float(flux) if np.isfinite(flux) else None,
                    flux_error=float(fluxerr) if np.isfinite(fluxerr) else None,
                    snr=snr,
                    size=float(area) if np.isfinite(area) else None,
                    ellipticity=ellip,
                    position_angle=float(theta) if np.isfinite(theta) else None,
                    peak_value=float(peak) if np.isfinite(peak) else None,
                    background=bkg_value,
                    noise=noise_value,
                    quality_flags=[],
                    extraction_metadata={
                        "a": a if np.isfinite(a) else None,
                        "b": b if np.isfinite(b) else None,
                    },
                )
            )
        return sources

    def extract_sources(
        self, image: np.ndarray, threshold: float | None = None
    ) -> list[Source]:
        objects = self._detect(
            image, threshold=threshold or self.config.detection_threshold
        )
        return self._to_sources(objects, image)

    def extract_sources_with_background(
        self, image: np.ndarray, background: np.ndarray
    ) -> list[Source]:
        objects = self._detect(image)
        return self._to_sources(objects, image, background=background)

    def extract_sources_with_noise(
        self, image: np.ndarray, noise: np.ndarray
    ) -> list[Source]:
        objects = self._detect(image)
        return self._to_sources(objects, image, noise=noise)

    def extract_sources_with_mask(
        self, image: np.ndarray, mask: np.ndarray
    ) -> list[Source]:
        masked = np.array(image, copy=True)
        masked = np.where(mask.astype(bool), 0.0, masked)
        objects = self._detect(masked)
        return self._to_sources(objects, masked)

    def calculate_source_flux(
        self, sources: list[Source], image: np.ndarray
    ) -> list[float]:
        radii = self.config.aperture_radius
        fluxes: list[float] = []
        for s in sources:
            x, y = s.coordinates
            rr, cc = np.ogrid[: image.shape[0], : image.shape[1]]
            mask = (rr - y) ** 2 + (cc - x) ** 2 <= radii**2
            aperture_flux = float(np.sum(image[mask]))
            fluxes.append(aperture_flux)
        return fluxes
