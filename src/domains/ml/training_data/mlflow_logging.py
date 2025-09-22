"""MLflow logging helpers for training datasets (ASTR-113)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

try:
    import mlflow
except Exception:  # pragma: no cover - optional at runtime
    mlflow = None  # type: ignore


def log_training_dataset_info(
    dataset_id: str, stats: dict[str, Any], sample_images: Iterable[str] | None = None
) -> None:
    if mlflow is None:
        return
    with mlflow.start_run(run_name=f"training_data_{dataset_id}"):
        for key in ("total_samples", "anomaly_ratio", "quality_score"):
            if key in stats:
                mlflow.log_metric(key, float(stats[key]))
        if sample_images:
            for i, path in enumerate(sample_images):
                try:
                    mlflow.log_artifact(path, artifact_path=f"samples/{i}")
                except Exception:
                    continue
