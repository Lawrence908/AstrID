"""
MLflow model versioning system for AstrID.

This module provides comprehensive model versioning functionality including
semantic versioning, model deployment tracking, and performance monitoring.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import mlflow
    from mlflow.entities.model_registry import ModelVersion
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MlflowException = None
    ModelVersion = None
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Install with: uv add mlflow")

from .config import MLflowConfig

logger = logging.getLogger(__name__)


class VersionType(Enum):
    """Semantic version types."""

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class ModelVersionData:
    """Model version data structure."""

    name: str
    version: str
    model_data: bytes
    metadata: dict[str, Any]
    created_at: float
    size_bytes: int
    checksum: str


@dataclass
class DeploymentInfo:
    """Model deployment information."""

    model_name: str
    version: str
    deployment_id: str
    environment: str
    deployed_at: float
    status: str
    endpoint_url: str | None = None
    replicas: int = 1
    resources: dict[str, Any] | None = None


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""

    model_name: str
    version: str
    timestamp: float
    metrics: dict[str, float]
    environment: str
    deployment_id: str | None = None


class ModelVersioning:
    """MLflow model versioning manager."""

    def __init__(self, config: MLflowConfig):
        """Initialize model versioning.

        Args:
            config: MLflow configuration
        """
        if not MLFLOW_AVAILABLE or mlflow is None:
            raise ImportError("MLflow not installed. Install with: uv add mlflow")

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set tracking URI
        mlflow.set_tracking_uri(config.tracking_uri)

        # Initialize MLflow client
        self.client = (
            MlflowClient(tracking_uri=config.tracking_uri) if MlflowClient else None
        )

        # In-memory storage for deployment tracking
        self._deployments: dict[str, DeploymentInfo] = {}
        self._performance_metrics: list[PerformanceMetrics] = []

    def create_model_version(
        self,
        model_name: str,
        model_data: bytes,
        metadata: dict[str, Any],
        version_type: VersionType = VersionType.PATCH,
    ) -> str:
        """Create a new model version with semantic versioning.

        Args:
            model_name: Model name
            model_data: Model data as bytes
            metadata: Model metadata
            version_type: Type of version increment

        Returns:
            Model version string

        Raises:
            MlflowException: If model version creation fails
        """
        try:
            # Get current version
            current_version = self._get_current_version(model_name)

            # Calculate next version
            next_version = self._calculate_next_version(current_version, version_type)

            # Create temporary file for model data
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                temp_file.write(model_data)
                temp_file_path = temp_file.name

            try:
                # Register model version
                self.client.create_model_version(
                    name=model_name,
                    source=temp_file_path,
                    run_id=metadata.get("run_id", ""),
                    description=metadata.get("description", ""),
                    tags={
                        **metadata.get("tags", {}),
                        "version_type": version_type.value,
                        "size_bytes": str(len(model_data)),
                        "created_at": str(time.time()),
                    },
                )

                # Store model data (for future use if needed)
                # model_version_data = ModelVersionData(
                #     name=model_name,
                #     version=next_version,
                #     model_data=model_data,
                #     metadata=metadata,
                #     created_at=time.time(),
                #     size_bytes=len(model_data),
                #     checksum=self._calculate_checksum(model_data),
                # )

                self.logger.info(f"Created model version '{model_name}' {next_version}")
                return next_version

            finally:
                # Clean up temporary file
                Path(temp_file_path).unlink(missing_ok=True)

        except Exception as e:
            error_msg = f"Failed to create model version '{model_name}': {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def get_model_version(self, model_name: str, version: str) -> bytes:
        """Get model version data.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Model data as bytes

        Raises:
            MlflowException: If model version retrieval fails
        """
        try:
            # Get model version
            model_version = self.client.get_model_version(model_name, version)

            # Download model artifact
            model_uri = f"models:/{model_name}/{version}"
            try:
                import mlflow.artifacts

                model_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=model_uri, dst_path=None
                )
            except AttributeError:
                # Fallback for older MLflow versions
                model_path = self.client.download_artifacts(
                    run_id=model_version.run_id, path="model", dst_path=None
                )

            # Read model data
            with open(model_path, "rb") as f:
                model_data = f.read()

            self.logger.debug(f"Retrieved model '{model_name}' version {version}")
            return model_data

        except Exception as e:
            error_msg = f"Failed to get model version '{model_name}' {version}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def compare_model_versions(
        self, model_name: str, version1: str, version2: str
    ) -> dict[str, Any]:
        """Compare two model versions.

        Args:
            model_name: Model name
            version1: First version
            version2: Second version

        Returns:
            Comparison results

        Raises:
            MlflowException: If model comparison fails
        """
        try:
            # Get both model versions
            model1 = self.client.get_model_version(model_name, version1)
            model2 = self.client.get_model_version(model_name, version2)

            # Get run information
            run1 = self.client.get_run(model1.run_id)
            run2 = self.client.get_run(model2.run_id)

            comparison = {
                "model_name": model_name,
                "version1": version1,
                "version2": version2,
                "version1_info": {
                    "stage": model1.current_stage,
                    "creation_time": model1.creation_timestamp,
                    "run_id": model1.run_id,
                    "parameters": dict(run1.data.params) if run1.data.params else {},
                    "metrics": dict(run1.data.metrics) if run1.data.metrics else {},
                    "tags": dict(model1.tags) if model1.tags else {},
                },
                "version2_info": {
                    "stage": model2.current_stage,
                    "creation_time": model2.creation_timestamp,
                    "run_id": model2.run_id,
                    "parameters": dict(run2.data.params) if run2.data.params else {},
                    "metrics": dict(run2.data.metrics) if run2.data.metrics else {},
                    "tags": dict(model2.tags) if model2.tags else {},
                },
                "differences": {"parameters": {}, "metrics": {}, "tags": {}},
            }

            # Compare parameters
            all_params = set(comparison["version1_info"]["parameters"].keys())
            all_params.update(comparison["version2_info"]["parameters"].keys())

            for param in all_params:
                val1 = comparison["version1_info"]["parameters"].get(param)
                val2 = comparison["version2_info"]["parameters"].get(param)
                if val1 != val2:
                    comparison["differences"]["parameters"][param] = {
                        "version1": val1,
                        "version2": val2,
                    }

            # Compare metrics
            all_metrics = set(comparison["version1_info"]["metrics"].keys())
            all_metrics.update(comparison["version2_info"]["metrics"].keys())

            for metric in all_metrics:
                val1 = comparison["version1_info"]["metrics"].get(metric)
                val2 = comparison["version2_info"]["metrics"].get(metric)
                if val1 != val2:
                    diff = None
                    if isinstance(val1, int | float) and isinstance(val2, int | float):
                        diff = val2 - val1

                    comparison["differences"]["metrics"][metric] = {
                        "version1": val1,
                        "version2": val2,
                        "difference": diff,
                    }

            # Compare tags
            all_tags = set(comparison["version1_info"]["tags"].keys())
            all_tags.update(comparison["version2_info"]["tags"].keys())

            for tag in all_tags:
                val1 = comparison["version1_info"]["tags"].get(tag)
                val2 = comparison["version2_info"]["tags"].get(tag)
                if val1 != val2:
                    comparison["differences"]["tags"][tag] = {
                        "version1": val1,
                        "version2": val2,
                    }

            self.logger.info(
                f"Compared model '{model_name}' versions {version1} and {version2}"
            )
            return comparison

        except Exception as e:
            error_msg = f"Failed to compare model versions '{model_name}' {version1} vs {version2}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def rollback_model_version(self, model_name: str, target_version: str) -> None:
        """Rollback model to a specific version.

        Args:
            model_name: Model name
            target_version: Target version to rollback to

        Raises:
            MlflowException: If rollback fails
        """
        try:
            # Get current production version
            try:
                current_prod = self.client.get_latest_versions(
                    model_name, stages=["Production"]
                )[0]
                current_version = current_prod.version
            except IndexError:
                current_version = None

            # Transition target version to Production
            self.client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production",
                archive_existing_versions=True,
            )

            # Archive previous production version if exists
            if current_version and current_version != target_version:
                self.client.transition_model_version_stage(
                    name=model_name, version=current_version, stage="Archived"
                )

            self.logger.info(
                f"Rolled back model '{model_name}' to version {target_version}"
            )

        except Exception as e:
            error_msg = f"Failed to rollback model '{model_name}' to version {target_version}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def archive_model_version(self, model_name: str, version: str) -> None:
        """Archive a model version.

        Args:
            model_name: Model name
            version: Model version

        Raises:
            MlflowException: If archiving fails
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage="Archived"
            )

            self.logger.info(f"Archived model '{model_name}' version {version}")

        except Exception as e:
            error_msg = f"Failed to archive model '{model_name}' version {version}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def track_deployment(
        self,
        model_name: str,
        version: str,
        environment: str,
        endpoint_url: str | None = None,
        replicas: int = 1,
        resources: dict[str, Any] | None = None,
    ) -> str:
        """Track model deployment.

        Args:
            model_name: Model name
            version: Model version
            environment: Deployment environment
            endpoint_url: Optional endpoint URL
            replicas: Number of replicas
            resources: Optional resource requirements

        Returns:
            Deployment ID

        Raises:
            MlflowException: If deployment tracking fails
        """
        try:
            deployment_id = f"{model_name}-{version}-{environment}-{int(time.time())}"

            deployment_info = DeploymentInfo(
                model_name=model_name,
                version=version,
                deployment_id=deployment_id,
                environment=environment,
                deployed_at=time.time(),
                status="deployed",
                endpoint_url=endpoint_url,
                replicas=replicas,
                resources=resources,
            )

            # Store deployment info
            self._deployments[deployment_id] = deployment_info

            # Add deployment tag to model version
            self.client.set_model_version_tag(
                model_name, version, "deployment_id", deployment_id
            )

            self.logger.info(
                f"Tracked deployment '{deployment_id}' for model '{model_name}' version {version}"
            )
            return deployment_id

        except Exception as e:
            error_msg = f"Failed to track deployment for model '{model_name}' version {version}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def log_performance_metrics(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, float],
        environment: str,
        deployment_id: str | None = None,
    ) -> None:
        """Log model performance metrics.

        Args:
            model_name: Model name
            version: Model version
            metrics: Performance metrics
            environment: Environment where metrics were collected
            deployment_id: Optional deployment ID

        Raises:
            MlflowException: If metric logging fails
        """
        try:
            performance_metrics = PerformanceMetrics(
                model_name=model_name,
                version=version,
                timestamp=time.time(),
                metrics=metrics,
                environment=environment,
                deployment_id=deployment_id,
            )

            # Store metrics
            self._performance_metrics.append(performance_metrics)

            # Log to MLflow run if available
            try:
                model_version = self.client.get_model_version(model_name, version)
                with mlflow.start_run(run_id=model_version.run_id):
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"deployment_{metric_name}", metric_value)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to MLflow run: {e}")

            self.logger.debug(
                f"Logged performance metrics for model '{model_name}' version {version}"
            )

        except Exception as e:
            error_msg = f"Failed to log performance metrics for model '{model_name}' version {version}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def get_deployment_status(self, deployment_id: str) -> DeploymentInfo | None:
        """Get deployment status.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment information or None if not found
        """
        return self._deployments.get(deployment_id)

    def list_deployments(self, model_name: str | None = None) -> list[DeploymentInfo]:
        """List deployments.

        Args:
            model_name: Optional model name filter

        Returns:
            List of deployment information
        """
        deployments = list(self._deployments.values())

        if model_name:
            deployments = [d for d in deployments if d.model_name == model_name]

        return deployments

    def get_performance_history(
        self,
        model_name: str,
        version: str | None = None,
        environment: str | None = None,
    ) -> list[PerformanceMetrics]:
        """Get performance metrics history.

        Args:
            model_name: Model name
            version: Optional version filter
            environment: Optional environment filter

        Returns:
            List of performance metrics
        """
        metrics = [m for m in self._performance_metrics if m.model_name == model_name]

        if version:
            metrics = [m for m in metrics if m.version == version]

        if environment:
            metrics = [m for m in metrics if m.environment == environment]

        return sorted(metrics, key=lambda x: x.timestamp)

    def _get_current_version(self, model_name: str) -> str | None:
        """Get current model version.

        Args:
            model_name: Model name

        Returns:
            Current version or None if no versions exist
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return None

            # Sort by creation timestamp and get latest
            latest_version = max(versions, key=lambda v: v.creation_timestamp)
            return latest_version.version

        except Exception:
            return None

    def _calculate_next_version(
        self, current_version: str | None, version_type: VersionType
    ) -> str:
        """Calculate next semantic version.

        Args:
            current_version: Current version string
            version_type: Type of version increment

        Returns:
            Next version string
        """
        if not current_version:
            return "1.0.0"

        try:
            # Parse current version
            parts = current_version.split(".")
            if len(parts) != 3:
                raise ValueError(f"Invalid version format: {current_version}")

            major, minor, patch = map(int, parts)

            # Calculate next version
            if version_type == VersionType.MAJOR:
                return f"{major + 1}.0.0"
            elif version_type == VersionType.MINOR:
                return f"{major}.{minor + 1}.0"
            else:  # PATCH
                return f"{major}.{minor}.{patch + 1}"

        except (ValueError, IndexError):
            # If version parsing fails, start from 1.0.0
            return "1.0.0"

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for model data.

        Args:
            data: Model data as bytes

        Returns:
            Checksum string
        """
        import hashlib

        return hashlib.sha256(data).hexdigest()
