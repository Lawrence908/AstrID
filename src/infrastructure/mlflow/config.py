"""
MLflow configuration management for AstrID.

This module provides configuration classes and utilities for MLflow integration
with proper environment management and validation.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from ..storage.config import StorageConfig


@dataclass
class MLflowConfig:
    """Configuration for MLflow integration."""

    # Core MLflow settings
    tracking_uri: str = "http://localhost:9003"
    artifact_root: str = "s3://astrid-models"
    database_url: str = ""

    # Feature flags
    authentication_enabled: bool = False
    model_registry_enabled: bool = True
    experiment_auto_logging: bool = True
    artifact_compression: bool = True
    max_artifact_size: int = 100 * 1024 * 1024  # 100MB

    # Server configuration
    server_host: str = "0.0.0.0"
    server_port: int = 5000
    server_workers: int = 4
    server_timeout: int = 120

    # Authentication settings
    auth_config: dict[str, Any] = field(default_factory=dict)

    # Storage configuration
    storage_config: StorageConfig | None = None

    @classmethod
    def from_env(cls) -> "MLflowConfig":
        """Create MLflow configuration from environment variables.

        Returns:
            MLflow configuration instance
        """
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:9003"),
            artifact_root=os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://astrid-models"),
            database_url=os.getenv("MLFLOW_DATABASE_URL", ""),
            authentication_enabled=os.getenv("MLFLOW_AUTH_ENABLED", "false").lower()
            == "true",
            model_registry_enabled=os.getenv(
                "MLFLOW_MODEL_REGISTRY_ENABLED", "true"
            ).lower()
            == "true",
            experiment_auto_logging=os.getenv("MLFLOW_AUTO_LOGGING", "true").lower()
            == "true",
            artifact_compression=os.getenv(
                "MLFLOW_ARTIFACT_COMPRESSION", "true"
            ).lower()
            == "true",
            max_artifact_size=int(
                os.getenv("MLFLOW_MAX_ARTIFACT_SIZE", "104857600")
            ),  # 100MB
            server_host=os.getenv("MLFLOW_SERVER_HOST", "0.0.0.0"),
            server_port=int(os.getenv("MLFLOW_SERVER_PORT", "5000")),
            server_workers=int(os.getenv("MLFLOW_SERVER_WORKERS", "4")),
            server_timeout=int(os.getenv("MLFLOW_SERVER_TIMEOUT", "120")),
        )

    @classmethod
    def from_storage_config(cls, storage_config: StorageConfig) -> "MLflowConfig":
        """Create MLflow configuration from storage configuration.

        Args:
            storage_config: Storage configuration instance

        Returns:
            MLflow configuration instance
        """
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:9003"),
            artifact_root=storage_config.mlflow_artifact_root,
            database_url=os.getenv("MLFLOW_DATABASE_URL", ""),
            storage_config=storage_config,
        )

    def get_server_command(self) -> str:
        """Get MLflow server command with configuration.

        Returns:
            MLflow server command string
        """
        cmd_parts = [
            "mlflow",
            "server",
            f"--host={self.server_host}",
            f"--port={self.server_port}",
            f"--workers={self.server_workers}",
            f"--timeout={self.server_timeout}",
        ]

        if self.database_url:
            cmd_parts.append(f"--backend-store-uri={self.database_url}")

        if self.artifact_root:
            cmd_parts.append(f"--default-artifact-root={self.artifact_root}")

        if self.authentication_enabled and self.auth_config:
            # Add authentication configuration
            for key, value in self.auth_config.items():
                cmd_parts.append(f"--{key}={value}")

        return " ".join(cmd_parts)

    def get_environment_variables(self) -> dict[str, str]:
        """Get environment variables for MLflow configuration.

        Returns:
            Dictionary of environment variables
        """
        env_vars = {
            "MLFLOW_TRACKING_URI": self.tracking_uri,
            "MLFLOW_DEFAULT_ARTIFACT_ROOT": self.artifact_root,
        }

        if self.database_url:
            env_vars["MLFLOW_DATABASE_URL"] = self.database_url

        if self.storage_config:
            # Add R2/S3 configuration
            env_vars.update(
                {
                    "MLFLOW_S3_ENDPOINT_URL": self.storage_config.r2_endpoint_url,
                    "AWS_ACCESS_KEY_ID": self.storage_config.r2_access_key_id,
                    "AWS_SECRET_ACCESS_KEY": self.storage_config.r2_secret_access_key,
                    "AWS_DEFAULT_REGION": self.storage_config.r2_region,
                }
            )

        return env_vars

    def validate(self) -> None:
        """Validate MLflow configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.tracking_uri:
            raise ValueError("MLflow tracking URI is required")

        if not self.artifact_root:
            raise ValueError("MLflow artifact root is required")

        if self.max_artifact_size <= 0:
            raise ValueError("Max artifact size must be positive")

        if self.server_port <= 0 or self.server_port > 65535:
            raise ValueError("Server port must be between 1 and 65535")

        if self.server_workers <= 0:
            raise ValueError("Server workers must be positive")

        if self.server_timeout <= 0:
            raise ValueError("Server timeout must be positive")
