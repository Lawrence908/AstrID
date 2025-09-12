"""
AstrID - Consolidated Database Models

This file contains all SQLAlchemy database models from the AstrID astronomical
identification system, consolidated from the domain-driven design structure.

Original structure:
- src/domains/observations/models.py
- src/domains/preprocessing/models.py
- src/domains/differencing/models.py
- src/domains/detection/models.py
- src/domains/curation/models.py
- src/domains/catalog/models.py

This consolidated version is for reference and discussion purposes only.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
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
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship


# Base class would be imported from: from ...core.db.session import Base
# For this consolidated version, we'll define it as a placeholder
class Base:
    """Placeholder for SQLAlchemy Base class"""

    pass


# ============================================================================
# OBSERVATIONS DOMAIN
# ============================================================================


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


# ============================================================================
# PREPROCESSING DOMAIN
# ============================================================================


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


# ============================================================================
# DIFFERENCING DOMAIN
# ============================================================================


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


# ============================================================================
# DETECTION DOMAIN
# ============================================================================


class ModelType(str, Enum):
    """ML model types."""

    UNET = "unet"
    ANOMALY_DETECTOR = "anomaly_detector"
    CLASSIFIER = "classifier"
    SEGMENTATION = "segmentation"


class ModelRunStatus(str, Enum):
    """Model run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DetectionStatus(str, Enum):
    """Detection status."""

    PENDING = "pending"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"
    ARCHIVED = "archived"


class DetectionType(str, Enum):
    """Detection types."""

    SUPERNOVA = "supernova"
    VARIABLE = "variable"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"
    ARTIFACT = "artifact"


class Model(Base):
    """ML Models registry table."""

    __tablename__ = "models"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[ModelType] = mapped_column(
        SQLAlchemyEnum(ModelType, name="model_type"), nullable=False
    )

    # Model metadata
    architecture: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    training_dataset: Mapped[str | None] = mapped_column(String(200))
    training_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Performance metrics
    precision: Mapped[float | None] = mapped_column(Numeric(5, 4))
    recall: Mapped[float | None] = mapped_column(Numeric(5, 4))
    f1_score: Mapped[float | None] = mapped_column(Numeric(5, 4))
    accuracy: Mapped[float | None] = mapped_column(Numeric(5, 4))

    # Deployment
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    deployment_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Storage
    model_path: Mapped[str | None] = mapped_column(String(500))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    model_runs: Mapped[list["ModelRun"]] = relationship(
        "ModelRun", back_populates="model"
    )

    # Constraints
    __table_args__ = (
        Index("idx_models_name_version", "name", "version", unique=True),
        Index("idx_models_model_type", "model_type"),
        Index("idx_models_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name='{self.name}', version='{self.version}')>"


class ModelRun(Base):
    """Model inference runs table."""

    __tablename__ = "model_runs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    model_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("models.id"), nullable=False
    )
    observation_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observations.id")
    )

    # Inference details
    input_image_path: Mapped[str | None] = mapped_column(String(500))
    output_mask_path: Mapped[str | None] = mapped_column(String(500))
    confidence_map_path: Mapped[str | None] = mapped_column(String(500))

    # Performance
    inference_time_ms: Mapped[int | None] = mapped_column(Integer)
    memory_usage_mb: Mapped[int | None] = mapped_column(Integer)

    # Results
    total_predictions: Mapped[int | None] = mapped_column(Integer)
    high_confidence_predictions: Mapped[int | None] = mapped_column(Integer)

    status: Mapped[ModelRunStatus] = mapped_column(
        SQLAlchemyEnum(ModelRunStatus, name="model_run_status"),
        nullable=False,
        default=ModelRunStatus.PENDING,
    )

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    model: Mapped["Model"] = relationship("Model", back_populates="model_runs")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="model_run"
    )

    def __repr__(self) -> str:
        return f"<ModelRun(id={self.id}, model_id={self.model_id}, status='{self.status}')>"


class Detection(Base):
    """Detections table - ML-identified anomalies."""

    __tablename__ = "detections"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    model_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_runs.id"), nullable=False
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
    confidence_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    detection_type: Mapped[DetectionType] = mapped_column(
        SQLAlchemyEnum(DetectionType, name="detection_type"), nullable=False
    )

    # ML metadata
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    inference_time_ms: Mapped[int | None] = mapped_column(Integer)
    prediction_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Status and workflow
    status: Mapped[DetectionStatus] = mapped_column(
        SQLAlchemyEnum(DetectionStatus, name="detection_status"),
        nullable=False,
        default=DetectionStatus.PENDING,
    )

    # Validation
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    human_label: Mapped[str | None] = mapped_column(String(50))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    model_run: Mapped["ModelRun"] = relationship(
        "ModelRun", back_populates="detections"
    )

    # Constraints
    __table_args__ = (
        Index("idx_detections_ra_dec", "ra", "dec"),
        Index("idx_detections_confidence_score", "confidence_score"),
        Index("idx_detections_status", "status"),
        Index("idx_detections_detection_type", "detection_type"),
        Index("idx_detections_observation_id", "observation_id"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1"),
    )

    def __repr__(self) -> str:
        return f"<Detection(id={self.id}, type='{self.detection_type}', confidence={self.confidence_score})>"


# ============================================================================
# CURATION DOMAIN
# ============================================================================


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


# ============================================================================
# CATALOG DOMAIN (System Tables)
# ============================================================================


class ProcessingJobStatus(str, Enum):
    """Processing job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SystemConfig(Base):
    """System configuration table."""

    __tablename__ = "system_config"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<SystemConfig(id={self.id}, key='{self.key}')>"


class ProcessingJob(Base):
    """Processing jobs table - for workflow tracking."""

    __tablename__ = "processing_jobs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Job details
    status: Mapped[ProcessingJobStatus] = mapped_column(
        SQLAlchemyEnum(ProcessingJobStatus, name="processing_job_status"),
        nullable=False,
        default=ProcessingJobStatus.PENDING,
    )
    priority: Mapped[int] = mapped_column(Integer, default=0)

    # Workflow tracking
    workflow_id: Mapped[str | None] = mapped_column(String(100))
    task_id: Mapped[str | None] = mapped_column(String(100))
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    # Timing
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Error handling
    error_message: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<ProcessingJob(id={self.id}, type='{self.job_type}', status='{self.status}')>"


class AuditLog(Base):
    """Audit log table for important operations."""

    __tablename__ = "audit_log"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(100))

    # Change details
    old_values: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    new_values: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    change_summary: Mapped[str | None] = mapped_column(Text)

    # Context
    ip_address: Mapped[str | None] = mapped_column(String(45))  # IPv6 max length
    user_agent: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, entity_type='{self.entity_type}', action='{self.action}')>"


# ============================================================================
# SUMMARY
# ============================================================================

# AstrID Database Models Summary:
#
# DOMAIN STRUCTURE:
# - Observations: Survey, Observation (2 models)
# - Preprocessing: PreprocessRun (1 model)
# - Differencing: DifferenceRun, Candidate (2 models)
# - Detection: Model, ModelRun, Detection (3 models)
# - Curation: ValidationEvent, Alert (2 models)
# - Catalog: SystemConfig, ProcessingJob, AuditLog (3 models)
#
# TOTAL: 13 models across 6 domains

# KEY FEATURES:
# - UUID primary keys for all models
# - SQLAlchemy 2.0+ with Mapped type annotations
# - Python enums with SQLAlchemy Enum integration
# - JSONB fields for flexible metadata storage
# - Proper relationships and foreign key constraints
# - Spatial indexing on RA/Dec coordinates
# - Comprehensive audit and workflow tracking
# - ML model registry and inference tracking
# - Human validation and curation workflows

# DATA FLOW:
# Survey → Observation → PreprocessRun → DifferenceRun → Candidate → ModelRun → Detection → ValidationEvent/Alert

# This represents a complete astronomical transient detection pipeline from raw observations
# through ML-based detection to human validation and alerting.
