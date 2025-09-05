"""Pydantic schemas for detection data validation."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, validator

from src.domains.detection.models import (
    DetectionStatus,
    DetectionType,
    ModelRunStatus,
    ModelType,
)


class ModelBase(BaseModel):
    """Base schema with shared model attributes."""

    name: str
    version: str
    model_type: ModelType
    architecture: dict[str, Any] | None = None
    hyperparameters: dict[str, Any] | None = None
    training_dataset: str | None = None
    training_metrics: dict[str, Any] | None = None
    is_active: bool = False

    class Config:
        from_attributes = True


class ModelCreate(ModelBase):
    """Schema for creating a new model."""

    model_path: str | None = None
    mlflow_run_id: str | None = None

    class Config:
        from_attributes = True


class ModelUpdate(BaseModel):
    """Schema for updating an existing model."""

    name: str | None = None
    version: str | None = None
    model_type: ModelType | None = None
    architecture: dict[str, Any] | None = None
    hyperparameters: dict[str, Any] | None = None
    training_dataset: str | None = None
    training_metrics: dict[str, Any] | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    accuracy: float | None = None
    is_active: bool | None = None
    model_path: str | None = None
    mlflow_run_id: str | None = None

    class Config:
        from_attributes = True


class ModelRead(ModelBase):
    """Schema for reading model data."""

    id: UUID
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    accuracy: float | None = None
    deployment_date: datetime | None = None
    model_path: str | None = None
    mlflow_run_id: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ModelListParams(BaseModel):
    """Parameters for model queries."""

    name: str | None = None
    model_type: ModelType | None = None
    is_active: bool | None = None
    limit: int = 100
    offset: int = 0

    class Config:
        from_attributes = True


class ModelRunBase(BaseModel):
    """Base schema with shared model run attributes."""

    model_id: UUID
    observation_id: UUID | None = None
    input_image_path: str | None = None
    output_mask_path: str | None = None
    confidence_map_path: str | None = None

    class Config:
        from_attributes = True


class ModelRunCreate(ModelRunBase):
    """Schema for creating a new model run."""

    pass


class ModelRunUpdate(BaseModel):
    """Schema for updating an existing model run."""

    status: ModelRunStatus | None = None
    input_image_path: str | None = None
    output_mask_path: str | None = None
    confidence_map_path: str | None = None
    inference_time_ms: int | None = None
    memory_usage_mb: int | None = None
    total_predictions: int | None = None
    high_confidence_predictions: int | None = None
    error_message: str | None = None

    class Config:
        from_attributes = True


class ModelRunRead(ModelRunBase):
    """Schema for reading model run data."""

    id: UUID
    status: ModelRunStatus
    inference_time_ms: int | None = None
    memory_usage_mb: int | None = None
    total_predictions: int | None = None
    high_confidence_predictions: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ModelRunListParams(BaseModel):
    """Parameters for model run queries."""

    model_id: UUID | None = None
    observation_id: UUID | None = None
    status: ModelRunStatus | None = None
    limit: int = 100
    offset: int = 0

    class Config:
        from_attributes = True


class DetectionBase(BaseModel):
    """Base schema with shared detection attributes."""

    observation_id: UUID
    ra: float
    dec: float
    pixel_x: int
    pixel_y: int
    confidence_score: float
    detection_type: DetectionType
    model_version: str
    inference_time_ms: int | None = None
    prediction_metadata: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class DetectionCreate(DetectionBase):
    """Schema for creating a new detection."""

    model_run_id: UUID

    class Config:
        from_attributes = True


class DetectionUpdate(BaseModel):
    """Schema for updating an existing detection."""

    ra: float | None = None
    dec: float | None = None
    pixel_x: int | None = None
    pixel_y: int | None = None
    confidence_score: float | None = None
    detection_type: DetectionType | None = None
    status: DetectionStatus | None = None
    is_validated: bool | None = None
    validation_confidence: float | None = None
    human_label: str | None = None
    prediction_metadata: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class DetectionRead(DetectionBase):
    """Schema for reading detection data."""

    id: UUID
    model_run_id: UUID
    status: DetectionStatus
    is_validated: bool
    validation_confidence: float | None = None
    human_label: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DetectionWithModelRunRead(DetectionRead):
    """Schema for reading detection data with model run information."""

    model_run: ModelRunRead

    class Config:
        from_attributes = True


class DetectionListParams(BaseModel):
    """Parameters for detection queries."""

    observation_id: UUID | None = None
    model_run_id: UUID | None = None
    status: DetectionStatus | None = None
    detection_type: DetectionType | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    model_version: str | None = None
    is_validated: bool | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    limit: int = 100
    offset: int = 0

    class Config:
        from_attributes = True

    @validator("max_confidence")  # type: ignore[misc]
    def max_confidence_must_be_greater_than_min_confidence(
        cls, v: float | None, values: dict
    ) -> float | None:
        if (
            v is not None
            and "min_confidence" in values
            and values["min_confidence"] is not None
        ):
            if v <= values["min_confidence"]:
                raise ValueError("max_confidence must be greater than min_confidence")
        return v
