"""SQLAlchemy database models for detection domain."""

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
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.db.session import Base


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
