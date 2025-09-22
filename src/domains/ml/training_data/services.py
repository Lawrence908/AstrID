"""Services for collecting and loading real training data (ASTR-113)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.adapters.imaging.fits_io import FITSProcessor
from src.domains.ml.training_data.mlflow_logging import log_training_dataset_info
from src.domains.ml.training_data.models import TrainingDataset, TrainingSample
from src.domains.preprocessing.processors import AstronomicalImageProcessor
from src.infrastructure.storage.r2_client import R2StorageClient


@dataclass
class TrainingDataCollectionParams:
    survey_ids: list[str]
    date_range: tuple[datetime, datetime]
    confidence_threshold: float = 0.7
    anomaly_types: list[str] | None = None
    quality_score_threshold: float = 0.8
    max_samples: int = 10000
    validation_status: str = "validated"


class TrainingDataCollector:
    """Service for collecting training data from validated detections."""

    def __init__(self, db_session: Session, r2_client: R2StorageClient):
        self.db = db_session
        self.r2 = r2_client
        self.fits = FITSProcessor()
        self.preprocessor = AstronomicalImageProcessor()

    def collect_training_data(
        self, params: TrainingDataCollectionParams
    ) -> list[TrainingSample]:
        """Collect training samples from validated detections (minimal implementation)."""
        # NOTE: This is a thin MVP that demonstrates wiring. We will enrich queries/logic next.
        from src.domains.detection.models import Detection
        from src.domains.observations.models import Observation

        q = (
            self.db.query(Detection, Observation)
            .join(Observation, Detection.observation_id == Observation.id)
            .filter(Detection.is_validated.is_(True))
            .filter(Detection.confidence_score >= params.confidence_threshold)
            .order_by(Detection.created_at.desc())
        )

        if params.anomaly_types:
            q = q.filter(Detection.human_label.in_(params.anomaly_types))

        q = q.limit(params.max_samples)

        samples: list[TrainingSample] = []
        for det, obs in q.all():
            # Use stored R2 paths; if not present, skip for now
            image_path = obs.fits_file_path or obs.fits_url
            if not image_path:
                continue

            sample = TrainingSample(
                dataset_id=None,  # assigned when persisted via manager
                observation_id=obs.id,
                detection_id=det.id,
                image_path=image_path,
                mask_path=None,
                labels={
                    "anomaly_type": det.human_label,
                    "confidence": float(det.confidence_score),
                    "validated": True,
                },
                sample_metadata={
                    "survey": str(obs.survey_id),
                    "filter_band": obs.filter_band,
                    "exposure_time": float(obs.exposure_time),
                },
            )
            samples.append(sample)

        return samples

    def validate_data_quality(self, samples: list[TrainingSample]) -> dict[str, Any]:
        """Simple quality validation stub; returns basic metrics."""
        total = len(samples)
        anomaly_ratio = (
            sum(
                1
                for s in samples
                if s.labels.get("validated") and s.labels.get("confidence", 0) >= 0.7
            )
            / total
            if total
            else 0.0
        )
        return {
            "total_samples": total,
            "anomaly_ratio": anomaly_ratio,
            "quality_score": anomaly_ratio,  # placeholder heuristic
        }


class RealDataLoader:
    """Loader for assembling datasets into train/val/test splits."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def load_training_dataset(
        self, dataset_id: str
    ) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
        ds = self.db.get(TrainingDataset, dataset_id)
        if not ds:
            raise ValueError(f"TrainingDataset not found: {dataset_id}")
        samples = (
            self.db.query(TrainingSample)
            .filter(TrainingSample.dataset_id == ds.id)
            .all()
        )
        return self.create_data_splits(samples)

    def create_data_splits(
        self, samples: list[TrainingSample]
    ) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
        n = len(samples)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train = samples[:n_train]
        val = samples[n_train : n_train + n_val]
        test = samples[n_train + n_val :]
        return train, val, test


class TrainingDatasetManager:
    """Manager for creating and managing TrainingDataset records."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def create_dataset(
        self,
        name: str,
        created_by: str,
        samples: list[TrainingSample],
        quality: dict[str, Any],
    ) -> TrainingDataset:
        ds = TrainingDataset(
            name=name,
            description=None,
            collection_params={},
            total_samples=quality.get("total_samples", len(samples)),
            anomaly_ratio=quality.get("anomaly_ratio", 0.0),
            quality_score=quality.get("quality_score", 0.0),
            created_by=created_by,
            status="active",
        )
        self.db.add(ds)
        self.db.flush()  # get ds.id

        # Attach samples to dataset
        for s in samples:
            s.dataset_id = ds.id
            self.db.add(s)

        self.db.commit()

        # Log to MLflow (best-effort)
        try:
            log_training_dataset_info(
                str(ds.id),
                {
                    "total_samples": ds.total_samples,
                    "anomaly_ratio": float(ds.anomaly_ratio),
                    "quality_score": float(ds.quality_score),
                },
            )
        except Exception:
            pass

        return ds
