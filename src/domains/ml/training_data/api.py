"""FastAPI routes for training data management (ASTR-113)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.core.api.response_wrapper import create_response
from src.core.db.session import get_db
from src.domains.ml.training_data.models import TrainingDataset, TrainingSample
from src.domains.ml.training_data.services import (
    TrainingDataCollectionParams,
    TrainingDataCollector,
    TrainingDatasetManager,
)
from src.infrastructure.storage.r2_client import R2StorageClient

router = APIRouter(prefix="/training", tags=["training"])


class CollectionParams(BaseModel):
    survey_ids: list[str]
    start: str
    end: str
    confidence_threshold: float = 0.7
    anomaly_types: list[str] | None = None
    quality_score_threshold: float = 0.8
    max_samples: int = 10000
    name: str = "dataset"


@router.post("/datasets/collect")
async def collect_training_data(
    params: CollectionParams, db: Session = Depends(get_db)
):
    r2 = R2StorageClient()
    collector = TrainingDataCollector(db, r2)
    parsed = TrainingDataCollectionParams(
        survey_ids=params.survey_ids,
        date_range=(
            __import__("datetime").datetime.fromisoformat(params.start),
            __import__("datetime").datetime.fromisoformat(params.end),
        ),
        confidence_threshold=params.confidence_threshold,
        anomaly_types=params.anomaly_types,
        quality_score_threshold=params.quality_score_threshold,
        max_samples=params.max_samples,
    )
    samples = collector.collect_training_data(parsed)
    quality = collector.validate_data_quality(samples)

    manager = TrainingDatasetManager(db)
    ds = manager.create_dataset(
        name=params.name,
        created_by="api",  # TODO: map from auth user
        samples=samples,
        quality=quality,
    )

    return create_response(
        {
            "dataset_id": str(ds.id),
            "name": ds.name,
            "total": ds.total_samples,
            "quality": {
                "anomaly_ratio": float(ds.anomaly_ratio),
                "quality_score": float(ds.quality_score),
            },
        }
    )


@router.get("/datasets")
async def list_training_datasets(db: Session = Depends(get_db)):
    datasets = (
        db.query(TrainingDataset).order_by(TrainingDataset.created_at.desc()).all()
    )
    return create_response(
        [
            {
                "id": str(ds.id),
                "name": ds.name,
                "total_samples": ds.total_samples,
                "quality_score": float(ds.quality_score),
                "status": ds.status,
                "created_at": ds.created_at.isoformat() if ds.created_at else None,
            }
            for ds in datasets
        ]
    )


@router.get("/datasets/{dataset_id}")
async def get_training_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(TrainingDataset, dataset_id)
    if not ds:
        return create_response({"error": {"message": "Not found"}}, status_code=404)
    samples = (
        db.query(TrainingSample).filter(TrainingSample.dataset_id == ds.id).count()
    )
    return create_response(
        {
            "id": str(ds.id),
            "name": ds.name,
            "total_samples": ds.total_samples,
            "quality_score": float(ds.quality_score),
            "status": ds.status,
            "samples": samples,
        }
    )
