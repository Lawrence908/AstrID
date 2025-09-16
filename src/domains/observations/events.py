"""Domain events for observations domain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from src.domains.observations.models import ObservationStatus


@dataclass
class ObservationIngested:
    """Event raised when an observation is successfully ingested into the system."""

    observation_id: UUID
    survey_id: UUID
    observation_time: datetime
    coordinates: tuple[float, float]  # (ra, dec)
    filter_band: str
    exposure_time: float
    fits_url: str
    ingested_at: datetime

    @property
    def ra(self) -> float:
        """Right Ascension in degrees."""
        return self.coordinates[0]

    @property
    def dec(self) -> float:
        """Declination in degrees."""
        return self.coordinates[1]


@dataclass
class ObservationStatusChanged:
    """Event raised when an observation's processing status changes."""

    observation_id: UUID
    survey_id: UUID
    old_status: ObservationStatus
    new_status: ObservationStatus
    changed_at: datetime
    changed_by: str | None = None  # User or system component that triggered the change
    reason: str | None = None  # Reason for the status change
    metadata: dict | None = None  # Additional metadata about the change


@dataclass
class ObservationFailed:
    """Event raised when observation processing fails."""

    observation_id: UUID
    survey_id: UUID
    failed_stage: str  # e.g., "preprocessing", "differencing", "detection"
    error_message: str
    failed_at: datetime
    previous_status: ObservationStatus
    error_details: dict | None = None
    retry_count: int = 0
    is_retryable: bool = True


@dataclass
class ObservationProcessingStarted:
    """Event raised when observation processing begins at any stage."""

    observation_id: UUID
    survey_id: UUID
    processing_stage: str  # e.g., "preprocessing", "differencing", "detection"
    started_at: datetime
    previous_status: ObservationStatus
    estimated_duration_minutes: int | None = None


@dataclass
class ObservationProcessingCompleted:
    """Event raised when observation processing completes successfully at any stage."""

    observation_id: UUID
    survey_id: UUID
    processing_stage: str
    completed_at: datetime
    previous_status: ObservationStatus
    new_status: ObservationStatus
    processing_duration_seconds: float | None = None
    output_files: list[str] | None = None  # List of generated file paths


@dataclass
class ObservationValidationFailed:
    """Event raised when observation data validation fails."""

    observation_id: UUID
    survey_id: UUID
    validation_errors: list[str]
    failed_at: datetime
    validation_stage: str  # e.g., "coordinate_validation", "metadata_validation"


@dataclass
class ObservationArchived:
    """Event raised when an observation is archived."""

    observation_id: UUID
    survey_id: UUID
    archived_at: datetime
    archive_reason: str
    archived_by: str | None = None
    retention_until: datetime | None = None  # When the archived data can be deleted
