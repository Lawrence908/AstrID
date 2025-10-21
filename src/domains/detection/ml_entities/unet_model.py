"""Domain-level UNet model entity and metadata.

This layer holds model metadata and health checks, and is independent of
any specific ML framework. The adapter-side TensorFlow/Keras model lives in
`src/adapters/ml/unet.py` and is loaded via services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.domains.detection.config import ModelConfig


@dataclass
class UNetMetadata:
    """Metadata about a UNet model version."""

    version: str
    training_date: datetime | None = None
    performance_metrics: dict[str, float] = field(default_factory=dict)
    input_channels: int = 1
    output_channels: int = 1
    input_size: tuple[int, int] = (512, 512)


class UNetModelEntity:
    """Domain entity describing a UNet model instance."""

    def __init__(self, config: ModelConfig, metadata: UNetMetadata | None = None):
        self.config = config
        self.metadata = metadata or UNetMetadata(
            version=config.model_version, input_size=config.input_size
        )

    @property
    def confidence_threshold(self) -> float:
        return self.config.confidence_threshold

    def describe(self) -> dict[str, Any]:
        """Return a serializable description of the model entity."""
        return {
            "model_name": self.config.model_name,
            "version": self.metadata.version,
            "input_size": self.metadata.input_size,
            "input_channels": self.metadata.input_channels,
            "output_channels": self.metadata.output_channels,
            "performance_metrics": self.metadata.performance_metrics,
            "training_date": self.metadata.training_date.isoformat()
            if self.metadata.training_date
            else None,
            "confidence_threshold": self.config.confidence_threshold,
        }

    def validate(self) -> list[str]:
        """Validate configuration/metadata; return list of issues (empty if ok)."""
        issues: list[str] = []
        h, w = self.config.input_size
        if h <= 0 or w <= 0:
            issues.append("input_size must be positive")
        if not (0.0 <= self.config.confidence_threshold <= 1.0):
            issues.append("confidence_threshold must be in [0,1]")
        return issues

    def health_check(self) -> dict[str, Any]:
        """Simple health check summary for operational monitoring."""
        issues = self.validate()
        return {
            "model": self.config.model_name,
            "version": self.metadata.version,
            "status": "healthy" if not issues else "degraded",
            "issues": issues,
        }
