"""
MLflow experiment tracking for AstrID.

This module provides comprehensive experiment tracking functionality including
run management, parameter logging, metrics tracking, and artifact management.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

try:
    import mlflow
    from mlflow.entities import Experiment, Run, RunStatus
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    Experiment = None
    Run = None
    RunStatus = None
    MlflowException = None
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Install with: uv add mlflow")

from .config import MLflowConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentInfo:
    """Experiment information."""

    experiment_id: str
    name: str
    description: str | None = None
    artifact_location: str | None = None
    lifecycle_stage: str | None = None
    creation_time: int | None = None
    last_update_time: int | None = None


@dataclass
class RunInfo:
    """Run information."""

    run_id: str
    experiment_id: str
    name: str | None = None
    status: str | None = None
    start_time: int | None = None
    end_time: int | None = None
    user_id: str | None = None
    tags: dict[str, str] | None = None


class ExperimentTracker:
    """MLflow experiment tracking manager."""

    def __init__(self, config: MLflowConfig):
        """Initialize experiment tracker.

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

        # Current run context
        self._current_run_id: str | None = None
        self._current_experiment_id: str | None = None

    def create_experiment(self, name: str, description: str = "") -> str:
        """Create a new experiment.

        Args:
            name: Experiment name
            description: Experiment description

        Returns:
            Experiment ID

        Raises:
            MlflowException: If experiment creation fails
        """
        try:
            # Check if experiment already exists
            try:
                existing_experiment = self.client.get_experiment_by_name(name)
                if existing_experiment is not None:
                    self.logger.info(
                        f"Experiment '{name}' already exists with ID: {existing_experiment.experiment_id}"
                    )
                    return existing_experiment.experiment_id
                else:
                    self.logger.info(f"Experiment '{name}' does not exist, will create it")
            except MlflowException:
                # Experiment doesn't exist, create it
                self.logger.info(f"Experiment '{name}' does not exist, will create it")

            # Create new experiment
            experiment_id = self.client.create_experiment(
                name=name,
                artifact_location=self.config.artifact_root,
                tags={"description": description} if description else None,
            )

            self.logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
            return experiment_id

        except Exception as e:
            error_msg = f"Failed to create experiment '{name}': {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def start_run(
        self,
        experiment_id: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start a new MLflow run.

        Args:
            experiment_id: Experiment ID
            run_name: Optional run name
            tags: Optional run tags

        Returns:
            Run ID

        Raises:
            MlflowException: If run creation fails
        """
        try:
            # Set experiment context
            mlflow.set_experiment(experiment_id)

            # Start run
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                run_id = run.info.run_id
                self._current_run_id = run_id
                self._current_experiment_id = experiment_id

                self.logger.info(
                    f"Started run '{run_name or 'unnamed'}' with ID: {run_id}"
                )
                return run_id

        except Exception as e:
            error_msg = f"Failed to start run in experiment {experiment_id}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def log_parameters(self, params: dict[str, Any], run_id: str | None = None) -> None:
        """Log parameters to MLflow run.

        Args:
            params: Parameters to log
            run_id: Optional run ID (uses current run if not provided)

        Raises:
            MlflowException: If parameter logging fails
        """
        try:
            target_run_id = run_id or self._current_run_id
            if not target_run_id:
                raise ValueError("No active run and no run_id provided")

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            self.logger.debug(f"Logged {len(params)} parameters to run {target_run_id}")

        except Exception as e:
            error_msg = f"Failed to log parameters: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log metrics to MLflow run.

        Args:
            metrics: Metrics to log
            step: Optional step number
            run_id: Optional run ID (uses current run if not provided)

        Raises:
            MlflowException: If metric logging fails
        """
        try:
            target_run_id = run_id or self._current_run_id
            if not target_run_id:
                raise ValueError("No active run and no run_id provided")

            # Log metrics
            for key, value in metrics.items():
                if step is not None:
                    mlflow.log_metric(key, value, step=step)
                else:
                    mlflow.log_metric(key, value)

            self.logger.debug(f"Logged {len(metrics)} metrics to run {target_run_id}")

        except Exception as e:
            error_msg = f"Failed to log metrics: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def log_artifacts(
        self, artifacts: str | dict[str, str], run_id: str | None = None
    ) -> None:
        """Log artifacts to MLflow run.

        Args:
            artifacts: Artifact path(s) to log
            run_id: Optional run ID (uses current run if not provided)

        Raises:
            MlflowException: If artifact logging fails
        """
        try:
            target_run_id = run_id or self._current_run_id
            if not target_run_id:
                raise ValueError("No active run and no run_id provided")

            if isinstance(artifacts, str):
                # Single artifact path
                mlflow.log_artifacts(artifacts)
            elif isinstance(artifacts, dict):
                # Multiple artifacts with custom paths
                for artifact_path, artifact_name in artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_name)
            else:
                raise ValueError(
                    "Artifacts must be a string path or dict of path->name mappings"
                )

            self.logger.debug(f"Logged artifacts to run {target_run_id}")

        except Exception as e:
            error_msg = f"Failed to log artifacts: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def end_run(self, run_id: str | None = None, status: str = "FINISHED") -> None:
        """End MLflow run.

        Args:
            run_id: Optional run ID (uses current run if not provided)
            status: Run status (FINISHED, FAILED, KILLED, RUNNING)

        Raises:
            MlflowException: If run ending fails
        """
        try:
            target_run_id = run_id or self._current_run_id
            if not target_run_id:
                raise ValueError("No active run and no run_id provided")

            # End run
            mlflow.end_run(status=status)

            # Clear current run context
            if target_run_id == self._current_run_id:
                self._current_run_id = None
                self._current_experiment_id = None

            self.logger.info(f"Ended run {target_run_id} with status: {status}")

        except Exception as e:
            error_msg = f"Failed to end run: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def get_experiment(self, experiment_id: str) -> ExperimentInfo:
        """Get experiment information.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment information

        Raises:
            MlflowException: If experiment retrieval fails
        """
        try:
            experiment = self.client.get_experiment(experiment_id)

            return ExperimentInfo(
                experiment_id=experiment.experiment_id,
                name=experiment.name,
                description=experiment.tags.get("description")
                if experiment.tags
                else None,
                artifact_location=experiment.artifact_location,
                lifecycle_stage=experiment.lifecycle_stage,
                creation_time=experiment.creation_time,
                last_update_time=experiment.last_update_time,
            )

        except Exception as e:
            error_msg = f"Failed to get experiment {experiment_id}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def list_experiments(self) -> list[ExperimentInfo]:
        """List all experiments.

        Returns:
            List of experiment information

        Raises:
            MlflowException: If experiment listing fails
        """
        try:
            experiments = self.client.search_experiments()

            experiment_list = []
            for exp in experiments:
                experiment_list.append(
                    ExperimentInfo(
                        experiment_id=exp.experiment_id,
                        name=exp.name,
                        description=exp.tags.get("description") if exp.tags else None,
                        artifact_location=exp.artifact_location,
                        lifecycle_stage=exp.lifecycle_stage,
                        creation_time=exp.creation_time,
                        last_update_time=exp.last_update_time,
                    )
                )

            self.logger.debug(f"Listed {len(experiment_list)} experiments")
            return experiment_list

        except Exception as e:
            error_msg = f"Failed to list experiments: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def get_run(self, run_id: str) -> RunInfo:
        """Get run information.

        Args:
            run_id: Run ID

        Returns:
            Run information

        Raises:
            MlflowException: If run retrieval fails
        """
        try:
            run = self.client.get_run(run_id)

            return RunInfo(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                name=run.data.tags.get("mlflow.runName") if run.data.tags else None,
                status=run.info.status,
                start_time=run.info.start_time,
                end_time=run.info.end_time,
                user_id=run.data.tags.get("mlflow.user") if run.data.tags else None,
                tags=dict(run.data.tags) if run.data.tags else None,
            )

        except Exception as e:
            error_msg = f"Failed to get run {run_id}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def search_runs(
        self, experiment_id: str, filter_string: str = "", max_results: int = 1000
    ) -> list[RunInfo]:
        """Search runs in an experiment.

        Args:
            experiment_id: Experiment ID
            filter_string: MLflow search filter string
            max_results: Maximum number of results

        Returns:
            List of run information

        Raises:
            MlflowException: If run search fails
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                max_results=max_results,
            )

            run_list = []
            for run in runs:
                run_list.append(
                    RunInfo(
                        run_id=run.info.run_id,
                        experiment_id=run.info.experiment_id,
                        name=run.data.tags.get("mlflow.runName")
                        if run.data.tags
                        else None,
                        status=run.info.status,
                        start_time=run.info.start_time,
                        end_time=run.info.end_time,
                        user_id=run.data.tags.get("mlflow.user")
                        if run.data.tags
                        else None,
                        tags=dict(run.data.tags) if run.data.tags else None,
                    )
                )

            self.logger.debug(
                f"Found {len(run_list)} runs in experiment {experiment_id}"
            )
            return run_list

        except Exception as e:
            error_msg = f"Failed to search runs in experiment {experiment_id}: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Comparison results

        Raises:
            MlflowException: If run comparison fails
        """
        try:
            if len(run_ids) < 2:
                raise ValueError("At least 2 runs required for comparison")

            # Get run data
            runs_data = []
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                runs_data.append(
                    {
                        "run_id": run_id,
                        "params": dict(run.data.params) if run.data.params else {},
                        "metrics": dict(run.data.metrics) if run.data.metrics else {},
                        "tags": dict(run.data.tags) if run.data.tags else {},
                    }
                )

            # Compare parameters
            param_comparison = {}
            all_params = set()
            for run_data in runs_data:
                all_params.update(run_data["params"].keys())

            for param in all_params:
                param_comparison[param] = {}
                for run_data in runs_data:
                    param_comparison[param][run_data["run_id"]] = run_data[
                        "params"
                    ].get(param, "N/A")

            # Compare metrics
            metric_comparison = {}
            all_metrics = set()
            for run_data in runs_data:
                all_metrics.update(run_data["metrics"].keys())

            for metric in all_metrics:
                metric_comparison[metric] = {}
                for run_data in runs_data:
                    metric_comparison[metric][run_data["run_id"]] = run_data[
                        "metrics"
                    ].get(metric, "N/A")

            comparison_result = {
                "run_ids": run_ids,
                "parameters": param_comparison,
                "metrics": metric_comparison,
                "timestamp": time.time(),
            }

            self.logger.info(f"Compared {len(run_ids)} runs")
            return comparison_result

        except Exception as e:
            error_msg = f"Failed to compare runs: {e}"
            self.logger.error(error_msg)
            raise MlflowException(error_msg) from e
