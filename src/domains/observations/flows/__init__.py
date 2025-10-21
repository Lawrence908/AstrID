"""Observation processing workflows using Prefect."""

from .observation_flows import (
    observation_detection_flow,
    observation_differencing_flow,
    observation_ingestion_flow,
    observation_preprocessing_flow,
    observation_validation_flow,
)

__all__ = [
    "observation_ingestion_flow",
    "observation_preprocessing_flow",
    "observation_differencing_flow",
    "observation_detection_flow",
    "observation_validation_flow",
]
