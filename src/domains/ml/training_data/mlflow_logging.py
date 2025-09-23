"""MLflow logging helpers for training datasets (ASTR-113)."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import Any

try:
    import mlflow
except Exception:  # pragma: no cover - optional at runtime
    mlflow = None  # type: ignore


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
            f"🏃 View run training_data_{run_id} at: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}"
        )
        print(
            f"🧪 View experiment at: http://localhost:5000/#/experiments/{experiment_id}"
        )


def log_training_dataset_info(
    dataset_id: str, stats: dict[str, Any], sample_images: Iterable[str] | None = None
) -> None:
    if mlflow is None:
        return

    # Suppress MLflow's automatic print statements
    original_stdout = _suppress_mlflow_prints()

    try:
        with mlflow.start_run(run_name=f"training_data_{dataset_id}") as run:
            for key in ("total_samples", "anomaly_ratio", "quality_score"):
                if key in stats:
                    mlflow.log_metric(key, float(stats[key]))
            if sample_images:
                for i, path in enumerate(sample_images):
                    try:
                        mlflow.log_artifact(path, artifact_path=f"samples/{i}")
                    except Exception:
                        continue

            # Print accessible URLs using localhost
            _print_accessible_urls(run.info.run_id, run.info.experiment_id)

    finally:
        # Restore stdout
        _restore_stdout(original_stdout)
