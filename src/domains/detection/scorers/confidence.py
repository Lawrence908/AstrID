"""Confidence scoring utilities for detection outputs.

These are lightweight, pure-domain functions that combine model probabilities
with quality metrics from upstream stages (image quality, source extraction).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_confidence(
    probability_map: np.ndarray,
    *,
    image_quality: float | None = None,
    extraction_quality: float | None = None,
    calibration: dict[str, Any] | None = None,
) -> float:
    """Compute a single confidence score for a detection candidate.

    This baseline implementation uses the max probability in the map and blends
    in optional quality signals using simple weighting. A calibration mapping can
    be provided to remap raw confidence into calibrated space.
    """

    # Base score from model
    base = float(np.clip(np.nanmax(probability_map), 0.0, 1.0))

    # Blend quality metrics if provided
    weights = {"base": 0.7, "image": 0.15, "extract": 0.15}
    score = base * weights["base"]
    if image_quality is not None:
        score += float(np.clip(image_quality, 0.0, 1.0)) * weights["image"]
    if extraction_quality is not None:
        score += float(np.clip(extraction_quality, 0.0, 1.0)) * weights["extract"]

    # Optional calibration: simple affine transform y = a*x + b within [0,1]
    if calibration and {"a", "b"} <= calibration.keys():
        a = float(calibration["a"])  # slope
        b = float(calibration["b"])  # intercept
        score = a * score + b

    return float(np.clip(score, 0.0, 1.0))
