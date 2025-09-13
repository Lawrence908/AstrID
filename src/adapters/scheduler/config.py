"""Configuration for Prefect flows and deployments."""

import os
from typing import Any

from pydantic import BaseModel, Field


class PrefectConfig(BaseModel):
    """Configuration for Prefect server and flows."""

    # Server configuration
    server_url: str = Field(
        default="http://localhost:4200/api",
        description="Prefect server API URL",
    )

    database_url: str | None = Field(
        default=None,
        description="Database URL for Prefect server",
    )

    # Flow execution configuration
    work_pool_name: str = Field(
        default="astrid-pool",
        description="Work pool name for flow execution",
    )

    # Observation processing configuration
    observation_processing: dict[str, Any] = Field(
        default_factory=lambda: {
            "schedule_cron": "0 2 * * *",  # Daily at 2 AM
            "max_observations_per_run": 50,
            "processing_timeout_minutes": 120,
            "retry_failed_observations": True,
        }
    )

    # Model training configuration
    model_training: dict[str, Any] = Field(
        default_factory=lambda: {
            "schedule_cron": "0 4 * * 0",  # Weekly on Sunday at 4 AM
            "min_new_samples_for_retraining": 1000,
            "min_f1_score_threshold": 0.85,
            "training_timeout_hours": 6,
        }
    )

    # Monitoring configuration
    monitoring: dict[str, Any] = Field(
        default_factory=lambda: {
            "health_check_schedule_cron": "*/15 * * * *",  # Every 15 minutes
            "daily_report_cron": "0 9 * * *",  # Daily at 9 AM
            "alert_thresholds": {
                "system_response_time_ms": 1000,
                "error_rate_threshold": 0.05,
                "queue_backlog_threshold": 100,
                "model_accuracy_degradation_threshold": 0.05,
            },
        }
    )

    @classmethod
    def from_env(cls) -> "PrefectConfig":
        """Create configuration from environment variables."""
        config_dict = {}

        # Basic Prefect configuration
        if prefect_url := os.getenv("PREFECT_API_URL"):
            config_dict["server_url"] = prefect_url

        if db_url := os.getenv("PREFECT_SERVER_DATABASE_CONNECTION_URL"):
            config_dict["database_url"] = db_url

        if work_pool := os.getenv("PREFECT_WORK_POOL_NAME"):
            config_dict["work_pool_name"] = work_pool

        # Flow-specific configuration from environment
        if obs_schedule := os.getenv("ASTRID_OBSERVATION_SCHEDULE"):
            config_dict.setdefault("observation_processing", {})["schedule_cron"] = (
                obs_schedule
            )

        if training_schedule := os.getenv("ASTRID_TRAINING_SCHEDULE"):
            config_dict.setdefault("model_training", {})["schedule_cron"] = (
                training_schedule
            )

        if monitoring_schedule := os.getenv("ASTRID_MONITORING_SCHEDULE"):
            config_dict.setdefault("monitoring", {})["health_check_schedule_cron"] = (
                monitoring_schedule
            )

        return cls(**config_dict)


class DeploymentConfig(BaseModel):
    """Configuration for Prefect deployments."""

    # Common deployment settings
    work_queue_name: str = Field(
        default="astrid-queue",
        description="Work queue for flow execution",
    )

    tags: list[str] = Field(
        default_factory=lambda: ["astrid", "astronomy", "ml"],
        description="Tags for deployment organization",
    )

    # Resource limits
    cpu_limit: str | None = Field(
        default="2000m",
        description="CPU limit for flow execution",
    )

    memory_limit: str | None = Field(
        default="4Gi",
        description="Memory limit for flow execution",
    )

    # Infrastructure configuration
    infrastructure_type: str = Field(
        default="process",
        description="Infrastructure type (process, docker, kubernetes)",
    )

    docker_image: str | None = Field(
        default="astrid:latest",
        description="Docker image for containerized execution",
    )


def get_prefect_config() -> PrefectConfig:
    """Get Prefect configuration from environment."""
    return PrefectConfig.from_env()


def get_deployment_config() -> DeploymentConfig:
    """Get deployment configuration."""
    return DeploymentConfig()
