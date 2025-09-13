"""Pydantic schemas for curation data validation."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Validation confidence levels."""

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


class AlertType(str, Enum):
    """Alert types."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    SMS = "sms"


class AlertPriority(str, Enum):
    """Alert priorities."""

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


class ValidationEventBase(BaseModel):
    """Base schema with shared validation event attributes."""

    detection_id: UUID = Field(..., description="ID of the detection to validate")
    validator_id: str = Field(
        ..., min_length=1, max_length=100, description="ID of the validator"
    )
    is_valid: bool = Field(..., description="Whether the detection is valid")
    label: str | None = Field(None, max_length=50, description="Human-assigned label")
    confidence_level: ConfidenceLevel | None = Field(
        None, description="Confidence level of validation"
    )
    notes: str | None = Field(None, description="Validation notes")
    tags: list[str] | None = Field(None, description="Validation tags")
    image_quality: QualityLevel | None = Field(
        None, description="Image quality assessment"
    )
    detection_quality: QualityLevel | None = Field(
        None, description="Detection quality assessment"
    )

    class Config:
        from_attributes = True


class ValidationEventCreate(ValidationEventBase):
    """Schema for creating a new validation event."""

    pass


class ValidationEventUpdate(BaseModel):
    """Schema for updating an existing validation event."""

    is_valid: bool | None = None
    label: str | None = Field(None, max_length=50)
    confidence_level: ConfidenceLevel | None = None
    notes: str | None = None
    tags: list[str] | None = None
    image_quality: QualityLevel | None = None
    detection_quality: QualityLevel | None = None

    class Config:
        from_attributes = True


class ValidationEventRead(ValidationEventBase):
    """Schema for reading validation event data."""

    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ValidationEventListParams(BaseModel):
    """Parameters for validation event queries."""

    detection_id: UUID | None = Field(None, description="Filter by detection ID")
    validator_id: str | None = Field(None, description="Filter by validator ID")
    is_valid: bool | None = Field(None, description="Filter by validation result")
    confidence_level: ConfidenceLevel | None = Field(
        None, description="Filter by confidence level"
    )
    image_quality: QualityLevel | None = Field(
        None, description="Filter by image quality"
    )
    detection_quality: QualityLevel | None = Field(
        None, description="Filter by detection quality"
    )
    date_from: datetime | None = Field(None, description="Start date")
    date_to: datetime | None = Field(None, description="End date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    class Config:
        from_attributes = True


class AlertBase(BaseModel):
    """Base schema with shared alert attributes."""

    detection_id: UUID = Field(..., description="ID of the detection to alert about")
    alert_type: AlertType = Field(..., description="Type of alert")
    priority: AlertPriority = Field(AlertPriority.MEDIUM, description="Alert priority")
    title: str = Field(..., min_length=1, max_length=200, description="Alert title")
    message: str = Field(..., min_length=1, description="Alert message")
    metadata: dict[str, Any] | None = Field(
        None, description="Additional alert metadata"
    )

    class Config:
        from_attributes = True


class AlertCreate(AlertBase):
    """Schema for creating a new alert."""

    pass


class AlertUpdate(BaseModel):
    """Schema for updating an existing alert."""

    alert_type: AlertType | None = None
    priority: AlertPriority | None = None
    title: str | None = Field(None, min_length=1, max_length=200)
    message: str | None = Field(None, min_length=1)
    metadata: dict[str, Any] | None = None
    status: AlertStatus | None = None
    sent_at: datetime | None = None
    delivery_attempts: int | None = Field(
        None, ge=0, description="Number of delivery attempts"
    )
    error_message: str | None = None

    class Config:
        from_attributes = True


class AlertRead(AlertBase):
    """Schema for reading alert data."""

    id: UUID
    status: AlertStatus
    sent_at: datetime | None = None
    delivery_attempts: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AlertListParams(BaseModel):
    """Parameters for alert queries."""

    detection_id: UUID | None = Field(None, description="Filter by detection ID")
    alert_type: AlertType | None = Field(None, description="Filter by alert type")
    priority: AlertPriority | None = Field(None, description="Filter by priority")
    status: AlertStatus | None = Field(None, description="Filter by status")
    date_from: datetime | None = Field(None, description="Start date")
    date_to: datetime | None = Field(None, description="End date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    class Config:
        from_attributes = True
