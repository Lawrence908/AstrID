"""SQLAlchemy models for ML training data management (ASTR-113)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.db.session import Base


class TrainingDataset(Base):
    """Represents a versioned dataset assembled for ML training."""

    __tablename__ = "training_datasets"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    collection_params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    total_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    anomaly_ratio: Mapped[float] = mapped_column(
        Numeric(5, 4), nullable=False, default=0
    )
    quality_score: Mapped[float] = mapped_column(
        Numeric(5, 4), nullable=False, default=0
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")

    # Relationships
    samples: Mapped[list[TrainingSample]] = relationship(
        "TrainingSample", back_populates="dataset", cascade="all, delete-orphan"
    )
    runs: Mapped[list[TrainingRun]] = relationship(
        "TrainingRun", back_populates="dataset", cascade="all, delete-orphan"
    )


class TrainingSample(Base):
    """Links observations/detections to dataset sample artifacts."""

    __tablename__ = "training_samples"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    dataset_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("training_datasets.id"), nullable=False
    )
    observation_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observations.id")
    )
    detection_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("detections.id")
    )

    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    mask_path: Mapped[str | None] = mapped_column(String(500))
    labels: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    # 'metadata' is reserved in SQLAlchemy Declarative API; map attribute to column name 'metadata'
    sample_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    dataset: Mapped[TrainingDataset] = relationship(
        "TrainingDataset", back_populates="samples"
    )


class TrainingRun(Base):
    """Tracks training executions tied to a dataset and model."""

    __tablename__ = "training_runs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    dataset_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("training_datasets.id"), nullable=False
    )
    model_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("models.id")
    )

    mlflow_run_id: Mapped[str | None] = mapped_column(String(255))
    training_params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    performance_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    dataset: Mapped[TrainingDataset] = relationship(
        "TrainingDataset", back_populates="runs"
    )
