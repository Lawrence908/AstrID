"""Detection domain configuration.

This module holds configuration objects used by the detection domain.
It intentionally avoids importing heavy adapter/framework libs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for ML models used by detection.

    Placed in the domain layer so business logic can depend on it
    without importing adapter-layer frameworks.
    """

    model_name: str = Field(default="unet_astronomical")
    model_version: str = Field(default="latest")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=16, ge=1, le=256)
    input_size: tuple[int, int] = Field(default=(512, 512))
    enable_gpu: bool = Field(default=True)

    # Pre/Post-processing
    normalize: bool = Field(default=True)
    resize: bool = Field(default=True)
    apply_nms: bool = Field(default=False)

    class Config:
        from_attributes = True
