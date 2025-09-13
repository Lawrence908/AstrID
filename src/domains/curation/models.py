"""SQLAlchemy database models for curation domain."""

from datetime import datetime
from enum import Enum

# Import Detection for type hints
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.db.session import Base

if TYPE_CHECKING:
    from src.domains.detection.models import Detection


class ConfidenceLevel(str, Enum):
    """Confidence level for validation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERT = "expert"


class QualityLevel(str, Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class AlertPriority(str, Enum):
    """Alert priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert delivery status."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationEvent(Base):
    """Validation events table - human review."""

    __tablename__ = "validation_events"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    detection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("detections.id"), nullable=False
    )
    validator_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Validation results
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    label: Mapped[str | None] = mapped_column(String(50))
    confidence_level: Mapped[ConfidenceLevel | None] = mapped_column(
        SQLAlchemyEnum(ConfidenceLevel, name="confidence_level"), nullable=True
    )

    # Additional information
    notes: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str] | None] = mapped_column(JSONB)

    # Quality assessment
    image_quality: Mapped[QualityLevel | None] = mapped_column(
        SQLAlchemyEnum(QualityLevel, name="image_quality"), nullable=True
    )
    detection_quality: Mapped[QualityLevel | None] = mapped_column(
        SQLAlchemyEnum(QualityLevel, name="detection_quality"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    detection: Mapped["Detection"] = relationship("Detection")

    def __repr__(self) -> str:
        return f"<ValidationEvent(id={self.id}, detection_id={self.detection_id}, is_valid={self.is_valid})>"


class Alert(Base):
    """Alerts and notifications table."""

    __tablename__ = "alerts"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    detection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("detections.id"), nullable=False
    )

    # Alert configuration
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    priority: Mapped[AlertPriority] = mapped_column(
        SQLAlchemyEnum(AlertPriority, name="alert_priority"),
        nullable=False,
        default=AlertPriority.MEDIUM,
    )

    # Content
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    alert_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Delivery
    status: Mapped[AlertStatus] = mapped_column(
        SQLAlchemyEnum(AlertStatus, name="alert_status"),
        nullable=False,
        default=AlertStatus.PENDING,
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_attempts: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    detection: Mapped["Detection"] = relationship("Detection")

    def __repr__(self) -> str:
        return (
            f"<Alert(id={self.id}, type='{self.alert_type}', status='{self.status}')>"
        )
