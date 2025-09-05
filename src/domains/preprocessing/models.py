"""SQLAlchemy database models for preprocessing domain."""

from datetime import datetime
from enum import Enum

# Import Observation for type hints
from typing import TYPE_CHECKING, Any
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
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.db.session import Base

if TYPE_CHECKING:
    from src.domains.observations.models import Observation


class PreprocessRunStatus(str, Enum):
    """Preprocessing run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PreprocessRun(Base):
    """Preprocessing runs table."""

    __tablename__ = "preprocess_runs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    observation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observations.id"), nullable=False
    )

    status: Mapped[PreprocessRunStatus] = mapped_column(
        String(50), default=PreprocessRunStatus.PENDING
    )

    # Processing details
    calibration_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    wcs_aligned: Mapped[bool] = mapped_column(Boolean, default=False)
    registration_quality: Mapped[float | None] = mapped_column(Numeric(5, 3))

    # Output files
    processed_fits_path: Mapped[str | None] = mapped_column(String(500))
    wcs_info: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

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
    observation: Mapped["Observation"] = relationship("Observation")

    def __repr__(self) -> str:
        return f"<PreprocessRun(id={self.id}, observation_id={self.observation_id}, status='{self.status}')>"
