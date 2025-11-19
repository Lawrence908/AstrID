"""MLflow logging helpers for training datasets (ASTR-113)."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterable
from datetime import datetime
from typing import Any

try:
    import mlflow
    import mlflow.data
except Exception:  # pragma: no cover - optional at runtime
    mlflow = None  # type: ignore

logger = logging.getLogger(__name__)


def _suppress_mlflow_prints():
    """Suppress MLflow's automatic print statements for run URLs."""
    # Redirect stdout temporarily to suppress MLflow's print statements
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    return original_stdout


def _restore_stdout(original_stdout):
    """Restore stdout after suppressing MLflow prints."""
    sys.stdout.close()
    sys.stdout = original_stdout


def _print_accessible_urls(run_id: str, experiment_id: str):
    """Print accessible URLs using localhost instead of mlflow:5000."""
    if mlflow is None:
        return

    # Get the current tracking URI and replace mlflow:5000 with localhost:5000
    try:
        tracking_uri = mlflow.get_tracking_uri()
        if "mlflow:5000" in tracking_uri:
            display_uri = tracking_uri.replace("mlflow:5000", "localhost:5000")
        else:
            display_uri = tracking_uri

        print(
            f"🏃 View run training_data_{run_id} at: {display_uri}/#/experiments/{experiment_id}/runs/{run_id}"
        )
        print(f"🧪 View experiment at: {display_uri}/#/experiments/{experiment_id}")
    except Exception:
        # Fallback to localhost if there's any issue
        print(
            f"🏃 View run training_data_{run_id} at: http://localhost:9003/#/experiments/{experiment_id}/runs/{run_id}"
        )
        print(
            f"🧪 View experiment at: http://localhost:9003/#/experiments/{experiment_id}"
        )


def log_training_dataset_info(
    dataset_id: str,
    stats: dict[str, Any],
    sample_images: Iterable[str] | None = None,
    collection_params: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
) -> str | None:
    """Enhanced MLflow logging for training datasets with comprehensive metadata.

    Args:
        dataset_id: Unique identifier for the dataset
        stats: Basic dataset statistics
        sample_images: Optional paths to sample images for logging
        collection_params: Parameters used for data collection
        quality_report: Detailed quality assessment report

    Returns:
        MLflow run ID if successful, None otherwise
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping dataset logging")
        return None

    # Suppress MLflow's automatic print statements
    original_stdout = _suppress_mlflow_prints()

    try:
        with mlflow.start_run(run_name=f"training_data_{dataset_id}") as run:
            # Log basic metrics
            for key in ("total_samples", "anomaly_ratio", "quality_score"):
                if key in stats:
                    mlflow.log_metric(key, float(stats[key]))

            # Log quality metrics if available
            if quality_report:
                for key, value in quality_report.items():
                    if key in ("image_quality_score", "label_consistency"):
                        mlflow.log_metric(f"quality_{key}", float(value))
                    elif key == "issues" and isinstance(value, list):
                        mlflow.log_metric("quality_issues_count", len(value))
                        # Log issues as a text artifact
                        if value:
                            issues_text = "\n".join(f"- {issue}" for issue in value)
                            mlflow.log_text(issues_text, "quality_issues.txt")
                    elif key in ("survey_coverage", "temporal_distribution"):
                        # Log coverage/distribution as JSON artifacts
                        if isinstance(value, dict):
                            mlflow.log_text(json.dumps(value, indent=2), f"{key}.json")

            # Log collection parameters
            if collection_params:
                for key, value in collection_params.items():
                    if isinstance(value, int | float | bool):
                        mlflow.log_param(key, value)
                    elif isinstance(value, str | list):
                        mlflow.log_param(key, str(value))
                    elif key == "date_range" and isinstance(value, tuple):
                        mlflow.log_param("start_date", str(value[0]))
                        mlflow.log_param("end_date", str(value[1]))

            # Log dataset metadata as tags
            mlflow.set_tags(
                {
                    "dataset_id": dataset_id,
                    "dataset_type": "real_astronomical_data",
                    "data_source": "astrid_validated_detections",
                    "created_at": datetime.now().isoformat(),
                    "stage": "data_collection",
                }
            )

            # Log sample images if provided
            if sample_images:
                logger.info(
                    f"Logging {len(list(sample_images))} sample images to MLflow"
                )
                for i, path in enumerate(sample_images):
                    try:
                        mlflow.log_artifact(path, artifact_path=f"samples/{i}")
                    except Exception as e:
                        logger.warning(f"Failed to log sample image {path}: {e}")
                        continue

            # Log dataset summary as artifact
            summary = {
                "dataset_id": dataset_id,
                "created_at": datetime.now().isoformat(),
                "statistics": stats,
                "collection_parameters": collection_params,
                "quality_assessment": quality_report,
            }
            mlflow.log_text(
                json.dumps(summary, indent=2, default=str), "dataset_summary.json"
            )

            logger.info(
                f"Successfully logged dataset {dataset_id} to MLflow run {run.info.run_id}"
            )

            # Print accessible URLs using localhost
            _print_accessible_urls(run.info.run_id, run.info.experiment_id)

            return run.info.run_id

    except Exception as e:
        logger.error(f"Failed to log dataset to MLflow: {e}")
        return None
    finally:
        # Restore stdout
        _restore_stdout(original_stdout)


def log_training_run_with_real_data(
    run_name: str,
    dataset_id: str,
    model_params: dict[str, Any],
    training_metrics: dict[str, Any],
    energy_metrics: dict[str, Any] | None = None,
    model_artifacts: dict[str, str] | None = None,
) -> str | None:
    """Log a training run that uses real data with comprehensive tracking.

    Args:
        run_name: Name for the training run
        dataset_id: ID of the dataset used for training
        model_params: Model architecture and training parameters
        training_metrics: Training performance metrics
        energy_metrics: GPU energy consumption metrics
        model_artifacts: Paths to model artifacts to log

    Returns:
        MLflow run ID if successful, None otherwise
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping training run logging")
        return None

    original_stdout = _suppress_mlflow_prints()

    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Log model parameters
            for key, value in model_params.items():
                if isinstance(value, int | float | bool | str):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))

            # Log training metrics
            for key, value in training_metrics.items():
                if isinstance(value, int | float):
                    mlflow.log_metric(key, value)

            # Log energy metrics if available
            if energy_metrics:
                for key, value in energy_metrics.items():
                    if isinstance(value, int | float):
                        mlflow.log_metric(f"energy_{key}", value)

            # Set tags for real data training
            mlflow.set_tags(
                {
                    "dataset_id": dataset_id,
                    "data_type": "real_astronomical_data",
                    "training_type": "real_data_training",
                    "energy_tracking": "enabled" if energy_metrics else "disabled",
                    "framework": "pytorch",
                    "stage": "training",
                }
            )

            # Log model artifacts if provided
            if model_artifacts:
                for artifact_name, artifact_path in model_artifacts.items():
                    try:
                        mlflow.log_artifact(
                            artifact_path, artifact_path=f"models/{artifact_name}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to log model artifact {artifact_name}: {e}"
                        )

            # Create training summary
            training_summary = {
                "run_name": run_name,
                "dataset_id": dataset_id,
                "model_parameters": model_params,
                "training_metrics": training_metrics,
                "energy_metrics": energy_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            mlflow.log_text(
                json.dumps(training_summary, indent=2, default=str),
                "training_summary.json",
            )

            logger.info(f"Successfully logged training run {run_name} to MLflow")
            _print_accessible_urls(run.info.run_id, run.info.experiment_id)

            return run.info.run_id

    except Exception as e:
        logger.error(f"Failed to log training run to MLflow: {e}")
        return None
    finally:
        _restore_stdout(original_stdout)


def create_mlflow_dataset_from_real_data(
    dataset_id: str, name: str, data_source_info: dict[str, Any]
) -> Any | None:
    """Create MLflow dataset from real astronomical data.

    Args:
        dataset_id: Unique identifier for the dataset
        name: Human-readable name for the dataset
        data_source_info: Information about the data source

    Returns:
        MLflow dataset object if successful, None otherwise
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping dataset creation")
        return None

    try:
        # Create MLflow dataset (this would use mlflow.data in newer versions)
        dataset_info = {
            "name": name,
            "dataset_id": dataset_id,
            "source": "astrid_real_astronomical_data",
            "created_at": datetime.now().isoformat(),
            **data_source_info,
        }

        logger.info(f"Created MLflow dataset reference for {dataset_id}")
        return dataset_info

    except Exception as e:
        logger.error(f"Failed to create MLflow dataset: {e}")
        return None
