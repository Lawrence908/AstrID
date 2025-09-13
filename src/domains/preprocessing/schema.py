"""Pydantic schemas for preprocessing data validation."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PreprocessRunStatus(str, Enum):
    """Preprocessing run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PreprocessRunBase(BaseModel):
    """Base schema with shared preprocessing run attributes."""

    observation_id: UUID = Field(..., description="ID of the observation to preprocess")

    class Config:
        from_attributes = True


class PreprocessRunCreate(PreprocessRunBase):
    """Schema for creating a new preprocessing run."""

    pass


class PreprocessRunUpdate(BaseModel):
    """Schema for updating an existing preprocessing run."""

    status: PreprocessRunStatus | None = None
    calibration_applied: bool | None = None
    wcs_aligned: bool | None = None
    registration_quality: float | None = Field(
        None, ge=0, le=1, description="Registration quality score (0-1)"
    )
    processed_fits_path: str | None = Field(
        None, max_length=500, description="Path to processed FITS file"
    )
    wcs_info: dict[str, Any] | None = Field(
        None, description="World Coordinate System information"
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processing_time_seconds: int | None = Field(
        None, gt=0, description="Processing time in seconds"
    )
    error_message: str | None = None
    retry_count: int | None = Field(None, ge=0, description="Number of retry attempts")

    class Config:
        from_attributes = True


class PreprocessRunRead(PreprocessRunBase):
    """Schema for reading preprocessing run data."""

    id: UUID
    status: PreprocessRunStatus
    calibration_applied: bool
    wcs_aligned: bool
    registration_quality: float | None = None
    processed_fits_path: str | None = None
    wcs_info: dict[str, Any] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processing_time_seconds: int | None = None
    error_message: str | None = None
    retry_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PreprocessRunListParams(BaseModel):
    """Parameters for preprocessing run queries."""

    observation_id: UUID | None = Field(None, description="Filter by observation ID")
    status: PreprocessRunStatus | None = Field(None, description="Filter by status")
    calibration_applied: bool | None = Field(
        None, description="Filter by calibration status"
    )
    wcs_aligned: bool | None = Field(None, description="Filter by WCS alignment status")
    date_from: datetime | None = Field(None, description="Start date")
    date_to: datetime | None = Field(None, description="End date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    class Config:
        from_attributes = True
