"""
MLflow artifact storage integration with Cloudflare R2.

This module provides configuration and management for MLflow artifacts
using R2 as the backend storage with proper integration into the AstrID workflow.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import mlflow
    from mlflow.entities import FileInfo
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None
    FileInfo = None
    logging.warning("MLflow not installed. Install with: uv add mlflow")

from .config import StorageConfig

logger = logging.getLogger(__name__)


@dataclass
class MLflowStorageConfig:
    """Configuration for MLflow artifact storage."""

    # R2 backend configuration
    artifact_root: str
    access_key_id: str
    secret_access_key: str
    endpoint_url: str
    region: str = "auto"

    # Artifact path structure
    experiment_path_template: str = "experiments/{experiment_id}"
    run_path_template: str = "experiments/{experiment_id}/runs/{run_id}"
    model_path_template: str = "models/{model_name}/versions/{version}"

    # Access control settings
    default_permissions: str = "private"
    public_read_experiments: list[str] | None = None

    @classmethod
    def from_storage_config(
        cls, storage_config: StorageConfig
    ) -> "MLflowStorageConfig":
        """Create MLflow config from storage config.

        Args:
            storage_config: Base storage configuration

        Returns:
            MLflow storage configuration
        """
        return cls(
            artifact_root=storage_config.mlflow_artifact_root,
            access_key_id=storage_config.r2_access_key_id,
            secret_access_key=storage_config.r2_secret_access_key,
            endpoint_url=storage_config.r2_endpoint_url,
            region=storage_config.r2_region,
            public_read_experiments=None,
        )

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables for MLflow S3 configuration.

        Returns:
            Dictionary of environment variables
        """
        return {
            "MLFLOW_S3_ENDPOINT_URL": self.endpoint_url,
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
            "AWS_DEFAULT_REGION": self.region,
            "MLFLOW_S3_IGNORE_TLS": "false",
        }


class MLflowArtifactStorage:
    """MLflow artifact storage client with R2 backend."""

    def __init__(
        self,
        config: MLflowStorageConfig,
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow artifact storage.

        Args:
            config: MLflow storage configuration
            tracking_uri: MLflow tracking server URI
        """
        if mlflow is None:
            raise ImportError("MLflow not installed. Install with: uv add mlflow")

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set environment variables for S3 configuration
        os.environ.update(self.config.get_env_vars())

        # Configure MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set artifact root
        if self.config.artifact_root:
            os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = self.config.artifact_root

        # Initialize MLflow client
        self.client = MlflowClient() if mlflow else None

    def store_model_artifact(
        self,
        model_path: str | Path,
        run_id: str,
        artifact_path: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store model artifact in MLflow with R2 backend.

        Args:
            model_path: Local path to model file or directory
            run_id: MLflow run ID
            artifact_path: Artifact path within run
            metadata: Additional metadata

        Returns:
            Artifact URI
        """
        try:
            if not mlflow:
                raise RuntimeError("MLflow not available")

            model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            # Log artifact to MLflow
            if model_path.is_dir():
                mlflow.log_artifacts(str(model_path), artifact_path)
            else:
                mlflow.log_artifact(str(model_path), artifact_path)

            # Get artifact URI
            artifact_uri = mlflow.get_artifact_uri(artifact_path)

            # Add metadata as tags if provided
            if metadata:
                for key, value in metadata.items():
                    mlflow.set_tag(f"artifact.{key}", str(value))

            self.logger.info(f"Stored model artifact: {artifact_uri}")
            return artifact_uri

        except Exception as e:
            self.logger.error(f"Error storing model artifact: {e}")
            raise

    def retrieve_model_artifact(
        self,
        artifact_uri: str,
        local_path: str | Path | None = None,
    ) -> bytes:
        """Retrieve model artifact from MLflow storage.

        Args:
            artifact_uri: MLflow artifact URI
            local_path: Local download path (optional)

        Returns:
            Artifact data as bytes
        """
        try:
            if not self.client:
                raise RuntimeError("MLflow client not available")

            if local_path:
                # Download to specific path
                local_path = Path(local_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract artifact path from URI
                artifact_path = artifact_uri.split("/")[-1]
                download_path = self.client.download_artifacts(
                    run_id=self._extract_run_id_from_uri(artifact_uri),
                    path=artifact_path,
                    dst_path=str(local_path.parent),
                )

                # Read downloaded file
                with open(download_path, "rb") as f:
                    data = f.read()
            else:
                # Download to temporary location
                with tempfile.TemporaryDirectory() as temp_dir:
                    artifact_path = artifact_uri.split("/")[-1]
                    download_path = self.client.download_artifacts(
                        run_id=self._extract_run_id_from_uri(artifact_uri),
                        path=artifact_path,
                        dst_path=temp_dir,
                    )

                    with open(download_path, "rb") as f:
                        data = f.read()

            self.logger.info(f"Retrieved model artifact: {artifact_uri}")
            return data

        except Exception as e:
            self.logger.error(f"Error retrieving model artifact: {e}")
            raise

    def list_model_artifacts(
        self,
        experiment_id: str,
        artifact_path: str = "",
    ) -> list[dict[str, Any]]:
        """List model artifacts in an experiment.

        Args:
            experiment_id: MLflow experiment ID
            artifact_path: Artifact path filter

        Returns:
            List of artifact information dictionaries
        """
        try:
            if not self.client:
                raise RuntimeError("MLflow client not available")

            # Get all runs in experiment
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                max_results=1000,
            )

            artifacts = []
            for run in runs:
                try:
                    # List artifacts for each run
                    run_artifacts = self.client.list_artifacts(
                        run_id=run.info.run_id,
                        path=artifact_path,
                    )

                    for artifact in run_artifacts:
                        artifact_info = {
                            "run_id": run.info.run_id,
                            "experiment_id": experiment_id,
                            "artifact_path": artifact.path,
                            "file_size": artifact.file_size,
                            "is_dir": artifact.is_dir,
                            "artifact_uri": f"{run.info.artifact_uri}/{artifact.path}",
                        }
                        artifacts.append(artifact_info)

                except Exception as e:
                    self.logger.warning(
                        f"Error listing artifacts for run {run.info.run_id}: {e}"
                    )

            self.logger.info(
                f"Listed {len(artifacts)} artifacts for experiment {experiment_id}"
            )
            return artifacts

        except Exception as e:
            self.logger.error(f"Error listing model artifacts: {e}")
            raise

    def delete_model_artifact(
        self,
        run_id: str,
        artifact_path: str,
    ) -> bool:
        """Delete model artifact.

        Note: MLflow doesn't provide direct artifact deletion.
        This would require direct R2 client operations.

        Args:
            run_id: MLflow run ID
            artifact_path: Artifact path to delete

        Returns:
            True if deletion successful
        """
        try:
            # MLflow doesn't support artifact deletion directly
            # Would need to use R2 client directly
            self.logger.warning(
                "MLflow artifact deletion not supported directly. "
                "Use R2 client for direct deletion if needed."
            )
            return False

        except Exception as e:
            self.logger.error(f"Error deleting model artifact: {e}")
            return False

    def get_artifact_metadata(
        self,
        run_id: str,
        artifact_path: str = "",
    ) -> dict[str, Any]:
        """Get metadata for artifacts in a run.

        Args:
            run_id: MLflow run ID
            artifact_path: Specific artifact path

        Returns:
            Artifact metadata dictionary
        """
        try:
            # Get run information
            run = self.client.get_run(run_id)

            # List artifacts
            artifacts = self.client.list_artifacts(run_id, artifact_path)

            metadata = {
                "run_id": run_id,
                "experiment_id": run.info.experiment_id,
                "artifact_uri": run.info.artifact_uri,
                "artifacts": [],
            }

            for artifact in artifacts:
                artifact_meta = {
                    "path": artifact.path,
                    "file_size": artifact.file_size,
                    "is_dir": artifact.is_dir,
                }
                metadata["artifacts"].append(artifact_meta)

            return metadata

        except Exception as e:
            self.logger.error(f"Error getting artifact metadata: {e}")
            raise

    def _extract_run_id_from_uri(self, artifact_uri: str) -> str:
        """Extract run ID from artifact URI.

        Args:
            artifact_uri: MLflow artifact URI

        Returns:
            Run ID string
        """
        # Artifact URI format: s3://bucket/experiments/exp_id/runs/run_id/artifacts/path
        try:
            parts = artifact_uri.split("/")
            run_idx = parts.index("runs") + 1
            return parts[run_idx]
        except (ValueError, IndexError) as err:
            raise ValueError(f"Cannot extract run ID from URI: {artifact_uri}") from err

    def configure_experiment_tracking(
        self,
        experiment_name: str,
        artifact_location: str | None = None,
    ) -> str:
        """Configure MLflow experiment with R2 artifact storage.

        Args:
            experiment_name: Name of the experiment
            artifact_location: Custom artifact location

        Returns:
            Experiment ID
        """
        try:
            # Set artifact location if provided
            if artifact_location:
                full_location = f"{self.config.artifact_root}/{artifact_location}"
            else:
                full_location = None

            # Create or get experiment
            try:
                experiment_id = self.client.create_experiment(
                    name=experiment_name,
                    artifact_location=full_location,
                )
            except Exception:
                # Experiment might already exist
                experiment = self.client.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id

            self.logger.info(
                f"Configured experiment: {experiment_name} (ID: {experiment_id})"
            )
            return experiment_id

        except Exception as e:
            self.logger.error(f"Error configuring experiment: {e}")
            raise
