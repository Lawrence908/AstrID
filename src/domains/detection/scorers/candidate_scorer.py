"""Candidate scoring utilities for ASTR-79."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from src.domains.detection.extractors.sep_extractor import Source


@dataclass
class ScoringWeights:
    detection_weight: float = 0.4
    quality_weight: float = 0.3
    anomaly_weight: float = 0.2
    confidence_weight: float = 0.1


class CandidateScorer:
    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def calculate_detection_score(self, source: Source, image: np.ndarray) -> float:
        snr = source.snr or 0.0
        peak = source.peak_value or 0.0
        return float(
            np.tanh(0.1 * snr) * 0.7 + np.tanh(peak / (np.std(image) + 1e-6)) * 0.3
        )

    def calculate_quality_score(self, source: Source, image: np.ndarray) -> float:
        if source.size is None:
            return 0.0
        size_term = 1.0 - abs((source.size - 50.0) / 50.0)
        size_term = float(np.clip(size_term, 0.0, 1.0))
        shape_term = (
            1.0
            if (source.ellipticity is None)
            else float(np.clip(1.0 - source.ellipticity, 0.0, 1.0))
        )
        return 0.6 * size_term + 0.4 * shape_term

    def calculate_anomaly_score(self, source: Source, reference: np.ndarray) -> float:
        # Placeholder: without reference features, use local contrast proxy
        return float(
            np.clip((source.peak_value or 0.0) / (np.std(reference) + 1e-6), 0.0, 5.0)
            / 5.0
        )

    def calculate_confidence_score(self, source: Source, context: dict) -> float:
        # Placeholder heuristic: penalize edge proximity if provided
        width = int(context.get("width", 0))
        height = int(context.get("height", 0))
        margin = float(context.get("edge_margin", 10))
        x, y = source.coordinates
        if width and height:
            away_from_edge = (
                (x > margin)
                and (y > margin)
                and (x < width - margin)
                and (y < height - margin)
            )
            return 1.0 if away_from_edge else 0.7
        return 0.9

    def rank_candidates(
        self,
        sources: Iterable[Source],
        image: np.ndarray,
        reference: np.ndarray | None = None,
        context: dict | None = None,
    ) -> list[tuple[Source, float]]:
        context = context or {}
        ranked: list[tuple[Source, float]] = []
        for s in sources:
            d = self.calculate_detection_score(s, image)
            q = self.calculate_quality_score(s, image)
            a = self.calculate_anomaly_score(
                s, reference if reference is not None else image
            )
            c = self.calculate_confidence_score(s, context)
            score = (
                self.weights.detection_weight * d
                + self.weights.quality_weight * q
                + self.weights.anomaly_weight * a
                + self.weights.confidence_weight * c
            )
            ranked.append((s, float(score)))
        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked
