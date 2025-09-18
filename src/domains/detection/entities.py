"""Detection domain entities and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np


@dataclass
class Anomaly:
    """Represents a detected astronomical anomaly."""

    anomaly_id: UUID
    coordinates: tuple[float, float]  # (x, y) pixel coordinates
    world_coordinates: tuple[float, float]  # (ra, dec) world coordinates
    confidence: float
    size: float  # pixel area
    magnitude: float
    classification: str
    metadata: dict[str, Any]
    validation_flags: list[str]


@dataclass
class DetectionResult:
    """Complete detection result for an observation."""

    detection_id: UUID
    observation_id: UUID
    anomalies: list[Anomaly]
    confidence_scores: list[float]
    processing_time: float
    model_version: str
    validation_status: str
    quality_metrics: dict[str, Any]
    created_at: datetime


@dataclass
class DetectionConfig:
    """Configuration for detection pipeline."""

    model_path: str
    confidence_threshold: float
    batch_size: int
    max_detections_per_image: int
    validation_enabled: bool
    false_positive_filtering: bool
    quality_assessment: bool
    caching_enabled: bool


@dataclass
class Observation:
    """Observation data for processing."""

    id: UUID
    survey_id: UUID
    observation_id: str
    ra: float
    dec: float
    observation_time: datetime
    filter_band: str
    exposure_time: float
    fits_url: str
    image_data: np.ndarray | None = None
    status: str = "ingested"
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Model:
    """ML model for anomaly detection."""

    id: UUID
    name: str
    version: str
    model_type: str
    model_path: str
    confidence_threshold: float
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None
