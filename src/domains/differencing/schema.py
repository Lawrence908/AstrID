"""Pydantic schemas for differencing data validation."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class DifferenceRunStatus(str, Enum):
    """Differencing run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DifferenceAlgorithm(str, Enum):
    """Differencing algorithms."""

    ZOGY = "zogy"
    CLASSIC = "classic"
    HOTPANTS = "hotpants"


class CandidateStatus(str, Enum):
    """Candidate status."""

    PENDING = "pending"
    PROCESSING = "processing"
    DETECTED = "detected"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class DifferenceRunBase(BaseModel):
    """Base schema with shared differencing run attributes."""

    observation_id: UUID = Field(..., description="ID of the observation to difference")
    reference_observation_id: UUID | None = Field(
        None, description="ID of the reference observation"
    )
    algorithm: DifferenceAlgorithm = Field(
        DifferenceAlgorithm.ZOGY, description="Differencing algorithm to use"
    )
    parameters: dict[str, Any] | None = Field(
        None, description="Algorithm-specific parameters"
    )

    class Config:
        from_attributes = True


class DifferenceRunCreate(DifferenceRunBase):
    """Schema for creating a new differencing run."""

    pass


class DifferenceRunUpdate(BaseModel):
    """Schema for updating an existing differencing run."""

    status: DifferenceRunStatus | None = None
    algorithm: DifferenceAlgorithm | None = None
    parameters: dict[str, Any] | None = None
    difference_image_path: str | None = Field(
        None, max_length=500, description="Path to difference image"
    )
    significance_map_path: str | None = Field(
        None, max_length=500, description="Path to significance map"
    )
    noise_level: float | None = Field(
        None, gt=0, description="Noise level in the difference image"
    )
    detection_threshold: float | None = Field(
        None, gt=0, description="Detection threshold used"
    )
    candidates_found: int | None = Field(
        None, ge=0, description="Number of candidates found"
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


class DifferenceRunRead(DifferenceRunBase):
    """Schema for reading differencing run data."""

    id: UUID
    status: DifferenceRunStatus
    difference_image_path: str | None = None
    significance_map_path: str | None = None
    noise_level: float | None = None
    detection_threshold: float | None = None
    candidates_found: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processing_time_seconds: int | None = None
    error_message: str | None = None
    retry_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DifferenceRunListParams(BaseModel):
    """Parameters for differencing run queries."""

    observation_id: UUID | None = Field(None, description="Filter by observation ID")
    reference_observation_id: UUID | None = Field(
        None, description="Filter by reference observation ID"
    )
    status: DifferenceRunStatus | None = Field(None, description="Filter by status")
    algorithm: DifferenceAlgorithm | None = Field(
        None, description="Filter by algorithm"
    )
    date_from: datetime | None = Field(None, description="Start date")
    date_to: datetime | None = Field(None, description="End date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    class Config:
        from_attributes = True


class CandidateBase(BaseModel):
    """Base schema with shared candidate attributes."""

    difference_run_id: UUID = Field(..., description="ID of the differencing run")
    observation_id: UUID = Field(..., description="ID of the observation")
    ra: float = Field(..., ge=0, le=360, description="Right ascension in degrees")
    dec: float = Field(..., ge=-90, le=90, description="Declination in degrees")
    pixel_x: int = Field(..., description="X pixel coordinate")
    pixel_y: int = Field(..., description="Y pixel coordinate")
    flux: float | None = Field(None, description="Flux value")
    flux_error: float | None = Field(None, description="Flux error")
    significance: float | None = Field(None, description="Detection significance")
    snr: float | None = Field(None, description="Signal-to-noise ratio")
    fwhm: float | None = Field(None, gt=0, description="Full width at half maximum")
    ellipticity: float | None = Field(None, ge=0, le=1, description="Ellipticity")
    position_angle: float | None = Field(
        None, ge=0, le=360, description="Position angle in degrees"
    )

    class Config:
        from_attributes = True


class CandidateCreate(CandidateBase):
    """Schema for creating a new candidate."""

    is_saturated: bool = Field(False, description="Whether the candidate is saturated")
    is_blended: bool = Field(False, description="Whether the candidate is blended")
    is_edge: bool = Field(False, description="Whether the candidate is near image edge")

    class Config:
        from_attributes = True


class CandidateUpdate(BaseModel):
    """Schema for updating an existing candidate."""

    ra: float | None = Field(None, ge=0, le=360)
    dec: float | None = Field(None, ge=-90, le=90)
    pixel_x: int | None = None
    pixel_y: int | None = None
    flux: float | None = None
    flux_error: float | None = None
    significance: float | None = None
    snr: float | None = None
    fwhm: float | None = Field(None, gt=0)
    ellipticity: float | None = Field(None, ge=0, le=1)
    position_angle: float | None = Field(None, ge=0, le=360)
    is_saturated: bool | None = None
    is_blended: bool | None = None
    is_edge: bool | None = None
    status: CandidateStatus | None = None

    class Config:
        from_attributes = True


class CandidateRead(CandidateBase):
    """Schema for reading candidate data."""

    id: UUID
    is_saturated: bool
    is_blended: bool
    is_edge: bool
    status: CandidateStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CandidateListParams(BaseModel):
    """Parameters for candidate queries."""

    difference_run_id: UUID | None = Field(
        None, description="Filter by differencing run ID"
    )
    observation_id: UUID | None = Field(None, description="Filter by observation ID")
    status: CandidateStatus | None = Field(None, description="Filter by status")
    min_significance: float | None = Field(None, description="Minimum significance")
    max_significance: float | None = Field(None, description="Maximum significance")
    min_snr: float | None = Field(None, description="Minimum signal-to-noise ratio")
    max_snr: float | None = Field(None, description="Maximum signal-to-noise ratio")
    is_saturated: bool | None = Field(None, description="Filter by saturation status")
    is_blended: bool | None = Field(None, description="Filter by blending status")
    is_edge: bool | None = Field(None, description="Filter by edge status")
    date_from: datetime | None = Field(None, description="Start date")
    date_to: datetime | None = Field(None, description="End date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    class Config:
        from_attributes = True

    @field_validator("max_significance")
    @classmethod
    def max_significance_must_be_greater_than_min_significance(
        cls, v: float | None, info
    ) -> float | None:
        if (
            v is not None
            and hasattr(info, "data")
            and "min_significance" in info.data
            and info.data["min_significance"] is not None
        ):
            if v <= info.data["min_significance"]:
                raise ValueError(
                    "max_significance must be greater than min_significance"
                )
        return v

    @field_validator("max_snr")
    @classmethod
    def max_snr_must_be_greater_than_min_snr(
        cls, v: float | None, info
    ) -> float | None:
        if (
            v is not None
            and hasattr(info, "data")
            and "min_snr" in info.data
            and info.data["min_snr"] is not None
        ):
            if v <= info.data["min_snr"]:
                raise ValueError("max_snr must be greater than min_snr")
        return v
