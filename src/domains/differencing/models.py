"""SQLAlchemy database models for differencing domain."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.db.session import Base
from src.domains.observations.models import Observation


class DifferenceRunStatus(str, Enum):
    """Difference run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CandidateStatus(str, Enum):
    """Candidate status."""

    PENDING = "pending"
    PROCESSING = "processing"
    DETECTED = "detected"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class DifferenceRun(Base):
    """Differencing runs table."""

    __tablename__ = "difference_runs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    observation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observations.id"), nullable=False
    )
    reference_observation_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observations.id")
    )

    status: Mapped[DifferenceRunStatus] = mapped_column(
        SQLAlchemyEnum(DifferenceRunStatus, name="difference_run_status"),
        nullable=False,
        default=DifferenceRunStatus.PENDING,
    )

    # Algorithm details
    algorithm: Mapped[str] = mapped_column(String(50), default="zogy")
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Output files
    difference_image_path: Mapped[str | None] = mapped_column(String(500))
    significance_map_path: Mapped[str | None] = mapped_column(String(500))

    # Quality metrics
    noise_level: Mapped[float | None] = mapped_column(Numeric(10, 6))
    detection_threshold: Mapped[float | None] = mapped_column(Numeric(10, 6))
    candidates_found: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_time_seconds: Mapped[int | None] = mapped_column(Integer)

    # Error handling
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    observation: Mapped["Observation"] = relationship(
        "Observation", foreign_keys=[observation_id]
    )
    reference_observation: Mapped[Optional["Observation"]] = relationship(
        "Observation", foreign_keys=[reference_observation_id]
    )
    candidates: Mapped[list["Candidate"]] = relationship(
        "Candidate", back_populates="difference_run"
    )

    def __repr__(self) -> str:
        return f"<DifferenceRun(id={self.id}, observation_id={self.observation_id}, status='{self.status}')>"


class Candidate(Base):
    """Candidates table - potential transients from differencing."""

    __tablename__ = "candidates"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    difference_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("difference_runs.id"), nullable=False
    )
    observation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observations.id"), nullable=False
    )

    # Position
    ra: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    dec: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    pixel_x: Mapped[int] = mapped_column(Integer, nullable=False)
    pixel_y: Mapped[int] = mapped_column(Integer, nullable=False)

    # Detection properties
    flux: Mapped[float | None] = mapped_column(Numeric(15, 6))
    flux_error: Mapped[float | None] = mapped_column(Numeric(15, 6))
    significance: Mapped[float | None] = mapped_column(Numeric(10, 3))
    snr: Mapped[float | None] = mapped_column(Numeric(8, 3))

    # Shape parameters
    fwhm: Mapped[float | None] = mapped_column(Numeric(8, 3))
    ellipticity: Mapped[float | None] = mapped_column(Numeric(5, 3))
    position_angle: Mapped[float | None] = mapped_column(Numeric(6, 2))

    # Quality flags
    is_saturated: Mapped[bool] = mapped_column(Boolean, default=False)
    is_blended: Mapped[bool] = mapped_column(Boolean, default=False)
    is_edge: Mapped[bool] = mapped_column(Boolean, default=False)

    # Status
    status: Mapped[CandidateStatus] = mapped_column(
        SQLAlchemyEnum(CandidateStatus, name="candidate_status"),
        nullable=False,
        default=CandidateStatus.PENDING,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    difference_run: Mapped["DifferenceRun"] = relationship(
        "DifferenceRun", back_populates="candidates"
    )
    observation: Mapped["Observation"] = relationship("Observation")

    def __repr__(self) -> str:
        return f"<Candidate(id={self.id}, ra={self.ra}, dec={self.dec}, significance={self.significance})>"
