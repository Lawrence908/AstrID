"""Candidate filtering utilities for ASTR-79."""

from __future__ import annotations

from collections.abc import Iterable

from src.domains.detection.extractors.sep_extractor import Source


class CandidateFilter:
    """Collection of static filtering helpers over `Source` objects."""

    @staticmethod
    def filter_by_flux(
        sources: Iterable[Source], min_flux: float, max_flux: float
    ) -> list[Source]:
        return [
            s for s in sources if s.flux is not None and min_flux <= s.flux <= max_flux
        ]

    @staticmethod
    def filter_by_snr(sources: Iterable[Source], min_snr: float) -> list[Source]:
        return [s for s in sources if s.snr is not None and s.snr >= min_snr]

    @staticmethod
    def filter_by_size(
        sources: Iterable[Source], min_size: float, max_size: float
    ) -> list[Source]:
        return [
            s for s in sources if s.size is not None and min_size <= s.size <= max_size
        ]

    @staticmethod
    def filter_by_shape(
        sources: Iterable[Source], max_ellipticity: float
    ) -> list[Source]:
        return [
            s
            for s in sources
            if s.ellipticity is None or s.ellipticity <= max_ellipticity
        ]

    @staticmethod
    def filter_by_position(
        sources: Iterable[Source], region_bounds: dict
    ) -> list[Source]:
        x_min = float(region_bounds.get("x_min", 0))
        y_min = float(region_bounds.get("y_min", 0))
        x_max = float(region_bounds.get("x_max", float("inf")))
        y_max = float(region_bounds.get("y_max", float("inf")))
        return [
            s
            for s in sources
            if x_min <= s.coordinates[0] <= x_max and y_min <= s.coordinates[1] <= y_max
        ]

    @staticmethod
    def filter_duplicates(
        sources: Iterable[Source], min_distance: float
    ) -> list[Source]:
        # Simple greedy deduplication by distance
        remaining: list[Source] = []
        for s in sources:
            x, y = s.coordinates
            too_close = False
            for t in remaining:
                xt, yt = t.coordinates
                if (x - xt) ** 2 + (y - yt) ** 2 < (min_distance**2):
                    too_close = True
                    break
            if not too_close:
                remaining.append(s)
        return remaining
