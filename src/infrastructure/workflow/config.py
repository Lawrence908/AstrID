"""Workflow configuration and data models."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ..storage.config import StorageConfig


class FlowType(Enum):
    """Types of workflows in the system."""

    OBSERVATION_INGESTION = "observation_ingestion"
    OBSERVATION_PREPROCESSING = "observation_preprocessing"
    OBSERVATION_DIFFERENCING = "observation_differencing"
    OBSERVATION_DETECTION = "observation_detection"
    OBSERVATION_VALIDATION = "observation_validation"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_RETRAINING = "model_retraining"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    SYSTEM_MAINTENANCE = "system_maintenance"


class FlowStatus(Enum):
    """Status of workflow execution."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"


@dataclass
class FlowStatusInfo:
    """Information about workflow execution status."""

    flow_id: str
    flow_type: FlowType
    status: FlowStatus
    start_time: datetime
    end_time: datetime | None = None
    progress: float = 0.0  # 0.0 to 1.0
    current_step: str = ""
    error_message: str | None = None
    metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = {}


@dataclass
class WorkflowConfig:
    """Configuration for workflow orchestration."""

    prefect_server_url: str
    database_url: str
    storage_config: StorageConfig
    authentication_enabled: bool = True
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    max_concurrent_flows: int = 10
    flow_timeout: int = 3600  # 1 hour in seconds
    retry_attempts: int = 3
    retry_delay: int = 60  # 1 minute in seconds

    # Prefect specific settings
    prefect_api_url: str | None = None
    prefect_ui_url: str | None = None
    prefect_work_pool: str = "default"
    prefect_work_queue: str = "default"

    # Monitoring settings
    metrics_retention_days: int = 30
    alert_webhook_url: str | None = None
    alert_email_recipients: list[str] | None = None

    def __post_init__(self) -> None:
        if self.prefect_api_url is None:
            self.prefect_api_url = f"{self.prefect_server_url}/api"
        if self.prefect_ui_url is None:
            self.prefect_ui_url = f"{self.prefect_server_url}/ui"
        if self.alert_email_recipients is None:
            self.alert_email_recipients = []


@dataclass
class FlowExecutionResult:
    """Result of workflow execution."""

    flow_id: str
    success: bool
    result_data: dict[str, Any] | None = None
    error_message: str | None = None
    execution_time: float = 0.0
    metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = {}


@dataclass
class AlertConfig:
    """Configuration for workflow alerts."""

    flow_id: str
    alert_type: str  # "failure", "timeout", "performance"
    threshold: float | None = None
    webhook_url: str | None = None
    email_recipients: list[str] | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.email_recipients is None:
            self.email_recipients = []
