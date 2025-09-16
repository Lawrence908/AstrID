"""Preprocessing pipeline orchestration for observations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from astropy.wcs import WCS

from src.domains.preprocessing.alignment.wcs_aligner import WCSAligner
from src.domains.preprocessing.calibration.calibration_processor import (
    CalibrationProcessor,
)
from src.domains.preprocessing.quality.quality_assessor import (
    QualityAssessor,
    QualityMetrics,
)


@dataclass
class PreprocessingResult:
    observation_id: UUID
    calibrated_image: np.ndarray
    aligned_image: np.ndarray
    quality_metrics: dict
    calibration_metadata: dict
    processing_time: float
    processing_errors: list[str]
    output_file_path: str | None


class PreprocessingPipeline:
    """Coordinate calibration, alignment, and QA for an observation."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.calibrator = CalibrationProcessor()
        self.aligner = WCSAligner()
        self.quality = QualityAssessor()

    def validate_calibration_frames(self, observation: Any) -> bool:
        # Placeholder validation; integrate with storage/DB as needed
        return True

    def select_optimal_calibration_frames(self, observation: Any) -> dict:
        # Placeholder selection logic; return provided frames
        return {
            "master_bias": observation.get("master_bias"),
            "master_dark": observation.get("master_dark"),
            "master_flat": observation.get("master_flat"),
            "exposure_time": observation.get("exposure_time", 1.0),
        }

    def monitor_processing_progress(self, observation_id: UUID) -> dict:
        return {"observation_id": str(observation_id), "status": "running"}

    def handle_processing_failure(self, observation_id: UUID, error: Exception) -> None:
        # Hook for events/logging
        pass

    def process_observation(self, observation: Any) -> PreprocessingResult:
        start = time.time()
        errors: list[str] = []
        obs_id: UUID = observation["id"]
        image: np.ndarray = observation["image"]
        wcs: WCS = observation.get("wcs", WCS())

        # Select calibration
        calib = self.select_optimal_calibration_frames(observation)

        # Calibration
        calibrated = image.astype(float)
        cal_meta: dict[str, Any] = {}
        try:
            if calib.get("master_bias") is not None:
                calibrated = self.calibrator.apply_bias_correction(
                    calibrated, calib["master_bias"]
                )
                cal_meta["bias_applied"] = True
            if calib.get("master_dark") is not None:
                calibrated = self.calibrator.apply_dark_correction(
                    calibrated,
                    calib["master_dark"],
                    float(calib.get("exposure_time", 1.0)),
                )
                cal_meta["dark_applied"] = True
            if calib.get("master_flat") is not None:
                calibrated = self.calibrator.apply_flat_correction(
                    calibrated, calib["master_flat"]
                )
                cal_meta["flat_applied"] = True
        except Exception as e:
            errors.append(f"calibration_error: {e}")

        # Alignment
        try:
            reference = observation.get("reference_image")
            if reference is not None:
                aligned, _aligned_wcs = self.aligner.align_to_reference_image(
                    calibrated, reference, wcs
                )
            else:
                aligned, _aligned_wcs = calibrated, wcs
        except Exception as e:
            errors.append(f"alignment_error: {e}")
            aligned, _aligned_wcs = calibrated, wcs

        # Quality assessment
        qm_basic = self.quality.assess_image_quality(aligned)
        qm_flat = self.quality.assess_flatness(aligned)
        qm = {**qm_basic, **qm_flat}
        scored: QualityMetrics = self.quality.score_quality(qm)

        duration = time.time() - start
        return PreprocessingResult(
            observation_id=obs_id,
            calibrated_image=calibrated,
            aligned_image=aligned,
            quality_metrics={
                "background_level": scored.background_level,
                "noise_level": scored.noise_level,
                "cosmic_ray_count": scored.cosmic_ray_count,
                "saturation_percentage": scored.saturation_percentage,
                "flatness_score": scored.flatness_score,
                "alignment_accuracy": scored.alignment_accuracy or 0.0,
                "overall_quality_score": scored.overall_quality_score,
                "quality_flags": scored.quality_flags,
            },
            calibration_metadata=cal_meta,
            processing_time=duration,
            processing_errors=errors,
            output_file_path=None,
        )
