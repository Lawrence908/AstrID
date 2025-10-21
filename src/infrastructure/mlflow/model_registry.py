"""
MLflow model registry for AstrID.

This module provides comprehensive model registry functionality including
model registration, versioning, stage management, and model lineage tracking.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import mlflow
    from mlflow.entities.model_registry import (
        ModelVersion,
        ModelVersionTag,
        RegisteredModel,
        RegisteredModelTag,
    )
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MlflowException = None
    RegisteredModel = None
    ModelVersion = None
    RegisteredModelTag = None
    ModelVersionTag = None
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Install with: uv add mlflow")

from .config import MLflowConfig

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model registry stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ModelVersionInfo:
    """Model version information."""

    name: str
    version: str
    stage: str
    description: str | None = None
    user_id: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None
    run_id: str | None = None
    status: str | None = None
    tags: dict[str, str] | None = None


@dataclass
class RegisteredModelInfo:
    """Registered model information."""

    name: str
    description: str | None = None
    latest_versions: list[ModelVersionInfo] | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None
    tags: dict[str, str] | None = None


class ModelRegistry:
    """MLflow model registry manager."""

    def __init__(self, config: MLflowConfig):
        """Initialize model registry.

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

    def register_model(
        self,
        model_path: str,
        model_name: str,
        run_id: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Register a model in the model registry.

        Args:
            model_path: Path to the model artifact
            model_name: Name of the model
            run_id: MLflow run ID
            description: Optional model description
            tags: Optional model tags

        Returns:
            Model version

        Raises:
            MlflowException: If model registration fails
        """
        try:
            # Register model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_path,
                run_id=run_id,
                description=description,
                tags=tags,
            )

            # Add tags to registered model if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_registered_model_tag(model_name, key, value)

            self.logger.info(
                f"Registered model '{model_name}' version {model_version.version}"
            )
            return model_version.version

        except Exception as e:
            error_msg = f"Failed to register model '{model_name}': {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def get_model_version(self, model_name: str, version: str) -> ModelVersionInfo:
        """Get model version information.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Model version information

        Raises:
            MlflowException: If model version retrieval fails
        """
        try:
            model_version = self.client.get_model_version(model_name, version)

            return ModelVersionInfo(
                name=model_name,
                version=model_version.version,
                stage=model_version.current_stage,
                description=model_version.description,
                user_id=model_version.user_id,
                creation_timestamp=model_version.creation_timestamp,
                last_updated_timestamp=model_version.last_updated_timestamp,
                run_id=model_version.run_id,
                status=model_version.status,
                tags=dict(model_version.tags) if model_version.tags else None,
            )

        except Exception as e:
            error_msg = f"Failed to get model version '{model_name}/{version}': {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def list_model_versions(self, model_name: str) -> list[ModelVersionInfo]:
        """List all versions of a model.

        Args:
            model_name: Model name

        Returns:
            List of model version information

        Raises:
            MlflowException: If model version listing fails
        """
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")

            version_list = []
            for version in model_versions:
                version_list.append(
                    ModelVersionInfo(
                        name=version.name,
                        version=version.version,
                        stage=version.current_stage,
                        description=version.description,
                        user_id=version.user_id,
                        creation_timestamp=version.creation_timestamp,
                        last_updated_timestamp=version.last_updated_timestamp,
                        run_id=version.run_id,
                        status=version.status,
                        tags=dict(version.tags) if version.tags else None,
                    )
                )

            self.logger.debug(
                f"Listed {len(version_list)} versions for model '{model_name}'"
            )
            return version_list

        except Exception as e:
            error_msg = f"Failed to list versions for model '{model_name}': {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = True,
    ) -> None:
        """Transition model version to a new stage.

        Args:
            model_name: Model name
            version: Model version
            stage: Target stage
            archive_existing_versions: Whether to archive existing versions in the target stage

        Raises:
            MlflowException: If stage transition fails
        """
        try:
            # Validate stage
            valid_stages = [s.value for s in ModelStage]
            if stage not in valid_stages:
                raise ValueError(
                    f"Invalid stage '{stage}'. Valid stages: {valid_stages}"
                )

            # Transition model version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions,
            )

            self.logger.info(
                f"Transitioned model '{model_name}' version {version} to stage '{stage}'"
            )

        except Exception as e:
            error_msg = f"Failed to transition model '{model_name}' version {version} to stage '{stage}': {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def get_latest_model(
        self, model_name: str, stage: str = "None"
    ) -> ModelVersionInfo:
        """Get the latest model version in a specific stage.

        Args:
            model_name: Model name
            stage: Model stage (default: "None")

        Returns:
            Latest model version information

        Raises:
            MlflowException: If latest model retrieval fails
        """
        try:
            if stage == "None":
                # Get latest version regardless of stage
                model_version = self.client.get_latest_versions(model_name)[0]
            else:
                # Get latest version in specific stage
                model_versions = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                if not model_versions:
                    raise ValueError(f"No model versions found in stage '{stage}'")
                model_version = model_versions[0]

            return ModelVersionInfo(
                name=model_version.name,
                version=model_version.version,
                stage=model_version.current_stage,
                description=model_version.description,
                user_id=model_version.user_id,
                creation_timestamp=model_version.creation_timestamp,
                last_updated_timestamp=model_version.last_updated_timestamp,
                run_id=model_version.run_id,
                status=model_version.status,
                tags=dict(model_version.tags) if model_version.tags else None,
            )

        except Exception as e:
            error_msg = (
                f"Failed to get latest model '{model_name}' in stage '{stage}': {e}"
            )
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a model version.

        Args:
            model_name: Model name
            version: Model version

        Raises:
            MlflowException: If model version deletion fails
        """
        try:
            self.client.delete_model_version(model_name, version)

            self.logger.info(f"Deleted model '{model_name}' version {version}")

        except Exception as e:
            error_msg = f"Failed to delete model '{model_name}' version {version}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def list_registered_models(self) -> list[RegisteredModelInfo]:
        """List all registered models.

        Returns:
            List of registered model information

        Raises:
            MlflowException: If model listing fails
        """
        try:
            registered_models = self.client.search_registered_models()

            model_list = []
            for model in registered_models:
                # Get latest versions
                latest_versions = []
                for version in model.latest_versions:
                    latest_versions.append(
                        ModelVersionInfo(
                            name=version.name,
                            version=version.version,
                            stage=version.current_stage,
                            description=version.description,
                            user_id=version.user_id,
                            creation_timestamp=version.creation_timestamp,
                            last_updated_timestamp=version.last_updated_timestamp,
                            run_id=version.run_id,
                            status=version.status,
                            tags=dict(version.tags) if version.tags else None,
                        )
                    )

                model_list.append(
                    RegisteredModelInfo(
                        name=model.name,
                        description=model.description,
                        latest_versions=latest_versions,
                        creation_timestamp=model.creation_timestamp,
                        last_updated_timestamp=model.last_updated_timestamp,
                        tags=dict(model.tags) if model.tags else None,
                    )
                )

            self.logger.debug(f"Listed {len(model_list)} registered models")
            return model_list

        except Exception as e:
            error_msg = f"Failed to list registered models: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def get_model_lineage(self, model_name: str, version: str) -> dict[str, Any]:
        """Get model lineage and provenance information.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Model lineage information

        Raises:
            MlflowException: If lineage retrieval fails
        """
        try:
            # Get model version
            model_version = self.client.get_model_version(model_name, version)

            # Get run information
            run = self.client.get_run(model_version.run_id)

            # Get experiment information
            experiment = self.client.get_experiment(run.info.experiment_id)

            lineage_info = {
                "model_name": model_name,
                "version": version,
                "run_id": model_version.run_id,
                "experiment_id": run.info.experiment_id,
                "experiment_name": experiment.name,
                "creation_timestamp": model_version.creation_timestamp,
                "run_start_time": run.info.start_time,
                "run_end_time": run.info.end_time,
                "run_status": run.info.status,
                "parameters": dict(run.data.params) if run.data.params else {},
                "metrics": dict(run.data.metrics) if run.data.metrics else {},
                "tags": dict(run.data.tags) if run.data.tags else {},
                "artifacts": [
                    artifact.path
                    for artifact in self.client.list_artifacts(model_version.run_id)
                ],
            }

            self.logger.debug(
                f"Retrieved lineage for model '{model_name}' version {version}"
            )
            return lineage_info

        except Exception as e:
            error_msg = (
                f"Failed to get model lineage for '{model_name}' version {version}: {e}"
            )
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
            model1 = self.get_model_version(model_name, version1)
            model2 = self.get_model_version(model_name, version2)

            # Get run information for both versions
            run1 = self.client.get_run(model1.run_id)
            run2 = self.client.get_run(model2.run_id)

            comparison = {
                "model_name": model_name,
                "version1": version1,
                "version2": version2,
                "version1_info": {
                    "stage": model1.stage,
                    "creation_time": model1.creation_timestamp,
                    "run_id": model1.run_id,
                    "parameters": dict(run1.data.params) if run1.data.params else {},
                    "metrics": dict(run1.data.metrics) if run1.data.metrics else {},
                },
                "version2_info": {
                    "stage": model2.stage,
                    "creation_time": model2.creation_timestamp,
                    "run_id": model2.run_id,
                    "parameters": dict(run2.data.params) if run2.data.params else {},
                    "metrics": dict(run2.data.metrics) if run2.data.metrics else {},
                },
                "parameter_differences": {},
                "metric_differences": {},
            }

            # Compare parameters
            all_params = set(comparison["version1_info"]["parameters"].keys())
            all_params.update(comparison["version2_info"]["parameters"].keys())

            for param in all_params:
                val1 = comparison["version1_info"]["parameters"].get(param)
                val2 = comparison["version2_info"]["parameters"].get(param)
                if val1 != val2:
                    comparison["parameter_differences"][param] = {
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
                    comparison["metric_differences"][metric] = {
                        "version1": val1,
                        "version2": val2,
                        "difference": val2 - val1
                        if isinstance(val1, int | float)
                        and isinstance(val2, int | float)
                        else None,
                    }

            self.logger.info(
                f"Compared model '{model_name}' versions {version1} and {version2}"
            )
            return comparison

        except Exception as e:
            error_msg = f"Failed to compare model versions '{model_name}' {version1} vs {version2}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e
