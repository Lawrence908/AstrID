"""SQLAlchemy database models for observations domain."""

from datetime import datetime
from enum import Enum
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
    observations: Mapped[list["Observation"]] = relationship(
        "Observation", back_populates="survey"
    )

    def __repr__(self) -> str:
        return f"<Survey(id={self.id}, name='{self.name}')>"


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
    survey: Mapped["Survey"] = relationship("Survey", back_populates="observations")

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
