"""Prefect-based workflow orchestration for AstrID."""

from .flows.model_training import scheduled_retraining_flow, train_model_flow
from .flows.monitoring import daily_report_flow, system_monitoring_flow
from .flows.process_new import run as process_new_observations_flow

__all__ = [
    "train_model_flow",
    "scheduled_retraining_flow",
    "system_monitoring_flow",
    "daily_report_flow",
    "process_new_observations_flow",
]
