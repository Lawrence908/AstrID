"""SQLAlchemy database models for observations domain."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.db.session import Base


class ObservationStatus(str, Enum):
    """Observation processing status."""

    INGESTED = "ingested"
    PREPROCESSING = "preprocessing"
    PREPROCESSED = "preprocessed"
    DIFFERENCING = "differencing"
    DIFFERENCED = "differenced"
    FAILED = "failed"
    ARCHIVED = "archived"


class Survey(Base):
    """Survey information table."""

    __tablename__ = "surveys"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    base_url: Mapped[str | None] = mapped_column(String(500))
    api_endpoint: Mapped[str | None] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    observations: Mapped[list[Observation]] = relationship(
        "Observation", back_populates="survey"
    )

    def __repr__(self) -> str:
        return f"<Survey(id={self.id}, name='{self.name}')>"

    def get_survey_stats(self) -> dict[str, Any]:
        """Get survey statistics.

        Returns:
            dict: Survey statistics including observation counts by status
        """
        # Note: This would typically involve a database query to count observations
        # For now, return basic survey info structure
        return {
            "survey_id": str(self.id),
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "base_url": self.base_url,
            "api_endpoint": self.api_endpoint,
            # Observation counts would be populated by the service layer
            "observation_counts": {
                "total": 0,
                "by_status": {},
                "by_filter_band": {},
            },
        }

    def is_configured_for_ingestion(self) -> bool:
        """Check if survey is properly configured for data ingestion.

        Returns:
            bool: True if survey can be used for data ingestion
        """
        return (
            self.is_active
            and self.name is not None
            and len(self.name.strip()) > 0
            and (self.base_url is not None or self.api_endpoint is not None)
        )

    def get_capabilities(self) -> dict[str, Any]:
        """Get survey capabilities and configuration.

        Returns:
            dict: Survey capabilities information
        """
        capabilities = {
            "can_ingest": self.is_configured_for_ingestion(),
            "has_api": bool(self.api_endpoint),
            "has_base_url": bool(self.base_url),
            "is_active": self.is_active,
            "configuration_score": self._calculate_configuration_score(),
        }

        # Add filter bands and other capabilities that would be stored in metadata
        # This could be extended to include supported filter bands, coordinate systems, etc.

        return capabilities

    def _calculate_configuration_score(self) -> float:
        """Calculate a configuration completeness score (0.0 to 1.0).

        Returns:
            float: Configuration score
        """
        score = 0.0
        max_score = 5.0

        if self.name and len(self.name.strip()) > 0:
            score += 1.0
        if self.description and len(self.description.strip()) > 0:
            score += 1.0
        if self.base_url:
            score += 1.0
        if self.api_endpoint:
            score += 1.0
        if self.is_active:
            score += 1.0

        return score / max_score


class Observation(Base):
    """Astronomical observations table."""

    __tablename__ = "observations"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    survey_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("surveys.id"), nullable=False
    )
    observation_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Astronomical coordinates
    ra: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    dec: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    observation_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Observation properties
    filter_band: Mapped[str] = mapped_column(String(50), nullable=False)
    exposure_time: Mapped[float] = mapped_column(Numeric(10, 3), nullable=False)
    fits_url: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Image metadata
    pixel_scale: Mapped[float | None] = mapped_column(Numeric(8, 4))
    image_width: Mapped[int | None] = mapped_column(Integer)
    image_height: Mapped[int | None] = mapped_column(Integer)
    airmass: Mapped[float | None] = mapped_column(Numeric(5, 2))
    seeing: Mapped[float | None] = mapped_column(Numeric(6, 3))

    # Processing status
    status: Mapped[ObservationStatus] = mapped_column(
        String(50), default=ObservationStatus.INGESTED
    )

    # Storage references
    fits_file_path: Mapped[str | None] = mapped_column(String(500))
    thumbnail_path: Mapped[str | None] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    survey: Mapped[Survey] = relationship("Survey", back_populates="observations")

    # Constraints and indexes
    __table_args__ = (
        Index("idx_observations_ra_dec", "ra", "dec"),
        Index("idx_observations_observation_time", "observation_time"),
        Index("idx_observations_status", "status"),
        Index("idx_observations_survey_id", "survey_id"),
        CheckConstraint("ra >= 0 AND ra <= 360"),
        CheckConstraint("dec >= -90 AND dec <= 90"),
        CheckConstraint("exposure_time > 0"),
        CheckConstraint("airmass > 0"),
        CheckConstraint("seeing > 0"),
    )

    def __repr__(self) -> str:
        return f"<Observation(id={self.id}, survey='{self.survey.name if self.survey else 'Unknown'}', ra={self.ra}, dec={self.dec})>"

    def validate_coordinates(self) -> bool:
        """Validate astronomical coordinates.

        Returns:
            bool: True if coordinates are valid, False otherwise
        """
        # RA should be between 0 and 360 degrees
        if not (0 <= self.ra <= 360):
            return False

        # Dec should be between -90 and 90 degrees
        if not (-90 <= self.dec <= 90):
            return False

        return True

    def calculate_airmass(self, observatory_lat: float = 0.0) -> float | None:
        """Calculate airmass for the observation.

        Args:
            observatory_lat: Observatory latitude in degrees (default 0.0 for generic)

        Returns:
            float: Calculated airmass, or None if airmass already stored
        """
        if self.airmass is not None:
            return self.airmass

        # Convert to radians
        dec_rad = math.radians(self.dec)
        lat_rad = math.radians(observatory_lat)

        # Simple zenith distance calculation (assumes object at transit)
        # More complex calculation would require LST and RA
        zenith_distance = abs(dec_rad - lat_rad)

        # Airmass approximation: sec(z) for small zenith distances
        if zenith_distance < math.radians(80):  # Avoid extreme airmass values
            airmass = 1.0 / math.cos(zenith_distance)
            return round(airmass, 2)
        else:
            return 10.0  # Very high airmass for low altitude observations

    def get_processing_status(self) -> dict[str, Any]:
        """Get detailed processing status information.

        Returns:
            dict: Processing status details
        """
        status_info = {
            "status": self.status,
            "status_description": self._get_status_description(),
            "can_process": self._can_process(),
            "next_stage": self._get_next_stage(),
            "processing_metadata": {
                "has_fits_file": bool(self.fits_file_path),
                "has_thumbnail": bool(self.thumbnail_path),
                "coordinate_validation": self.validate_coordinates(),
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            },
        }

        return status_info

    def _get_status_description(self) -> str:
        """Get human-readable status description."""
        status_descriptions = {
            ObservationStatus.INGESTED: "Observation has been ingested and is ready for processing",
            ObservationStatus.PREPROCESSING: "Observation is currently being preprocessed",
            ObservationStatus.PREPROCESSED: "Observation has been preprocessed and is ready for differencing",
            ObservationStatus.DIFFERENCING: "Observation is currently being differenced",
            ObservationStatus.DIFFERENCED: "Observation has been differenced and is ready for detection",
            ObservationStatus.FAILED: "Observation processing has failed",
            ObservationStatus.ARCHIVED: "Observation has been archived",
        }
        return status_descriptions.get(self.status, "Unknown status")

    def _can_process(self) -> bool:
        """Check if observation can be processed."""
        return (
            self.status in [ObservationStatus.INGESTED, ObservationStatus.PREPROCESSED]
            and self.validate_coordinates()
            and self.exposure_time > 0
        )

    def _get_next_stage(self) -> str | None:
        """Get the next processing stage."""
        next_stages = {
            ObservationStatus.INGESTED: "preprocessing",
            ObservationStatus.PREPROCESSED: "differencing",
            ObservationStatus.DIFFERENCED: "detection",
            ObservationStatus.PREPROCESSING: None,  # Currently processing
            ObservationStatus.DIFFERENCING: None,  # Currently processing
            ObservationStatus.FAILED: None,
            ObservationStatus.ARCHIVED: None,
        }
        return next_stages.get(self.status)

    def get_sky_region_bounds(self, radius_degrees: float = 0.1) -> dict[str, float]:
        """Get sky region bounds around this observation.

        Args:
            radius_degrees: Radius around the observation position in degrees

        Returns:
            dict: Sky region bounds with ra_min, ra_max, dec_min, dec_max
        """
        # Simple rectangular bounds (doesn't account for coordinate wrap-around)
        ra_min = max(0, self.ra - radius_degrees)
        ra_max = min(360, self.ra + radius_degrees)
        dec_min = max(-90, self.dec - radius_degrees)
        dec_max = min(90, self.dec + radius_degrees)

        return {
            "ra_min": ra_min,
            "ra_max": ra_max,
            "dec_min": dec_min,
            "dec_max": dec_max,
            "center_ra": self.ra,
            "center_dec": self.dec,
            "radius_degrees": radius_degrees,
        }
