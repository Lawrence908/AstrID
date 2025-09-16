"""Image quality assessment utilities for preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats


@dataclass
class QualityMetrics:
    background_level: float
    noise_level: float
    cosmic_ray_count: int
    saturation_percentage: float
    flatness_score: float
    alignment_accuracy: float | None
    overall_quality_score: float
    quality_flags: list[str]


class QualityAssessor:
    """Compute image quality metrics and scoring."""

    def assess_image_quality(self, image: np.ndarray) -> dict:
        return {
            "background_level": self.calculate_background_level(image),
            "noise_level": self.calculate_noise_level(image),
            "cosmic_ray_count": int(self.detect_cosmic_rays(image).sum()),
            "saturation_level": self.calculate_saturation_level(image),
        }

    def calculate_background_level(self, image: np.ndarray) -> float:
        _, median, _ = sigma_clipped_stats(image, sigma=3.0)
        return float(median)

    def calculate_noise_level(self, image: np.ndarray) -> float:
        _, _, std = sigma_clipped_stats(image, sigma=3.0)
        return float(std)

    def detect_cosmic_rays(
        self, image: np.ndarray, threshold_sigma: float = 5.0
    ) -> np.ndarray:
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        return (image - median) > (threshold_sigma * std)

    def assess_flatness(self, image: np.ndarray) -> dict:
        smoothed = image - self._large_scale_background(image)
        flatness = 1.0 / (1.0 + float(np.std(smoothed) / (np.mean(image) + 1e-9)))
        return {"flatness_score": flatness}

    def calculate_saturation_level(
        self, image: np.ndarray, max_value: float | None = None
    ) -> float:
        if max_value is None:
            max_value = float(np.max(image)) if np.max(image) > 0 else 1.0
        saturated = image >= 0.98 * max_value
        return float(100.0 * saturated.mean())

    def score_quality(self, metrics: dict) -> QualityMetrics:
        flags: list[str] = []
        # Simple heuristic scoring
        snr = float(
            np.mean(
                metrics.get("signal", np.mean(metrics.get("background_level", 1.0)))
            )
            / (metrics.get("noise_level", 1.0) + 1e-9)
        )
        cosmic_penalty = min(1.0, metrics.get("cosmic_ray_count", 0) / 100.0)
        sat_penalty = min(1.0, metrics.get("saturation_level", 0.0) / 50.0)
        flat_score = metrics.get("flatness_score", 0.5)
        align_acc = metrics.get("alignment_accuracy", None)

        score = np.clip(
            0.6 * (snr / 10.0)
            + 0.3 * flat_score
            - 0.1 * (cosmic_penalty + sat_penalty),
            0.0,
            1.0,
        )

        if metrics.get("noise_level", 0) > 100:
            flags.append("high_noise")
        if metrics.get("saturation_level", 0) > 10:
            flags.append("saturation")

        return QualityMetrics(
            background_level=float(metrics.get("background_level", 0.0)),
            noise_level=float(metrics.get("noise_level", 0.0)),
            cosmic_ray_count=int(metrics.get("cosmic_ray_count", 0)),
            saturation_percentage=float(metrics.get("saturation_level", 0.0)),
            flatness_score=float(flat_score),
            alignment_accuracy=None if align_acc is None else float(align_acc),
            overall_quality_score=float(score),
            quality_flags=flags,
        )

    def _large_scale_background(self, image: np.ndarray, box: int = 64) -> np.ndarray:
        # Fast block-wise mean approximation for large-scale background
        h, w = image.shape
        bh = max(1, h // box)
        bw = max(1, w // box)
        bg_small = (
            image[: bh * box, : bw * box].reshape(bh, box, bw, box).mean(axis=(1, 3))
        )
        # Nearest-neighbor upsample back to image size
        bg = np.repeat(np.repeat(bg_small, box, axis=0), box, axis=1)
        return bg[:h, :w]
