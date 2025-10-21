"""FastAPI routes for training data management (ASTR-113)."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.auth.api_key_auth import require_permission_or_api_key
from src.adapters.auth.rbac import Permission
from src.core.api.response_wrapper import create_response
from src.core.db.session import get_db
from src.domains.ml.training_data.models import TrainingDataset, TrainingSample
from src.domains.ml.training_data.services import (
    RealDataLoader,
    TrainingDataCollectionParams,
    TrainingDataCollector,
    TrainingDatasetManager,
)
from src.infrastructure.storage.r2_client import R2StorageClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


class CollectionParams(BaseModel):
    """Parameters for training data collection."""

    survey_ids: list[str] = Field(description="List of survey IDs to collect from")
    start: str = Field(description="Start date (ISO format)")
    end: str = Field(description="End date (ISO format)")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    anomaly_types: list[str] | None = Field(
        default=None, description="Specific anomaly types to filter"
    )
    quality_score_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum quality score"
    )
    max_samples: int = Field(
        default=10000, gt=0, description="Maximum number of samples"
    )
    name: str = Field(default="dataset", description="Dataset name")
    description: str | None = Field(default=None, description="Dataset description")


class DatasetFilters(BaseModel):
    """Filters for listing datasets."""

    status: str | None = Field(default=None, description="Dataset status filter")
    created_by: str | None = Field(default=None, description="Creator filter")
    min_quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum quality score"
    )


class DatasetResponse(BaseModel):
    """Response model for dataset information."""

    id: str
    name: str
    description: str | None
    total_samples: int
    anomaly_ratio: float
    quality_score: float
    status: str
    created_by: str
    created_at: str
    sample_count: int | None = None


@router.post("/datasets/collect")
async def collect_training_data(
    params: CollectionParams,
    db: AsyncSession = Depends(get_db),
    auth=Depends(require_permission_or_api_key(Permission.MANAGE_OPERATIONS)),
):
    """Collect training data from validated detections and create a new dataset."""
    try:
        logger.info(f"Starting training data collection with params: {params.name}")

        r2 = R2StorageClient()
        collector = TrainingDataCollector(db, r2)

        # Parse date parameters
        try:
            start_date = datetime.fromisoformat(params.start)
            end_date = datetime.fromisoformat(params.end)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid date format: {e}"
            ) from e

        # Create collection parameters
        parsed = TrainingDataCollectionParams(
            survey_ids=params.survey_ids,
            date_range=(start_date, end_date),
            confidence_threshold=params.confidence_threshold,
            anomaly_types=params.anomaly_types,
            quality_score_threshold=params.quality_score_threshold,
            max_samples=params.max_samples,
        )

        # Collect training samples
        samples = await collector.collect_training_data(parsed)
        if not samples:
            raise HTTPException(
                status_code=404,
                detail="No training samples found with specified criteria",
            )

        # Validate data quality
        quality_report = collector.validate_data_quality(samples)

        # Create dataset
        manager = TrainingDatasetManager(db)
        ds = await manager.create_dataset(
            name=params.name,
            created_by="api",  # TODO: map from auth user
            samples=samples,
            quality_report=quality_report,
            collection_params=parsed.__dict__,
            description=params.description,
        )

        logger.info(f"Successfully created dataset {ds.id} with {len(samples)} samples")

        return create_response(
            {
                "dataset_id": str(ds.id),
                "name": ds.name,
                "description": ds.description,
                "total_samples": ds.total_samples,
                "quality_report": {
                    "anomaly_ratio": quality_report.anomaly_ratio,
                    "quality_score": quality_report.quality_score,
                    "image_quality_score": quality_report.image_quality_score,
                    "label_consistency": quality_report.label_consistency,
                    "survey_coverage": quality_report.survey_coverage,
                    "temporal_distribution": quality_report.temporal_distribution,
                    "issues": quality_report.issues,
                },
                "collection_params": parsed.__dict__,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error collecting training data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/datasets")
async def list_training_datasets(
    filters: DatasetFilters = Depends(),
    db: AsyncSession = Depends(get_db),
    auth=Depends(require_permission_or_api_key(Permission.READ_DATA)),
):
    """List available training datasets with optional filtering."""
    try:
        query = select(TrainingDataset)

        # Apply filters
        if filters.status:
            query = query.where(TrainingDataset.status == filters.status)
        if filters.created_by:
            query = query.where(TrainingDataset.created_by == filters.created_by)
        if filters.min_quality_score is not None:
            query = query.where(
                TrainingDataset.quality_score >= filters.min_quality_score
            )

        query = query.order_by(TrainingDataset.created_at.desc())

        result = await db.execute(query)
        datasets = result.scalars().all()

        return create_response(
            [
                DatasetResponse(
                    id=str(ds.id),
                    name=ds.name,
                    description=ds.description,
                    total_samples=ds.total_samples,
                    anomaly_ratio=float(ds.anomaly_ratio),
                    quality_score=float(ds.quality_score),
                    status=ds.status,
                    created_by=ds.created_by,
                    created_at=ds.created_at.isoformat() if ds.created_at else "",
                ).dict()
                for ds in datasets
            ]
        )

    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/datasets/{dataset_id}")
async def get_training_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    auth=Depends(require_permission_or_api_key(Permission.READ_DATA)),
):
    """Get detailed information about a specific training dataset."""
    try:
        ds = await db.get(TrainingDataset, dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Training dataset not found")

        # Count actual samples
        result = await db.execute(
            select(func.count()).select_from(
                select(TrainingSample.id)
                .where(TrainingSample.dataset_id == ds.id)
                .subquery()
            )
        )
        sample_count = result.scalar_one()

        return create_response(
            {
                "id": str(ds.id),
                "name": ds.name,
                "description": ds.description,
                "total_samples": ds.total_samples,
                "anomaly_ratio": float(ds.anomaly_ratio),
                "quality_score": float(ds.quality_score),
                "status": ds.status,
                "created_by": ds.created_by,
                "created_at": ds.created_at.isoformat() if ds.created_at else None,
                "sample_count": sample_count,
                "collection_params": ds.collection_params,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/datasets/{dataset_id}/quality")
async def get_dataset_quality(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    auth=Depends(require_permission_or_api_key(Permission.READ_DATA)),
):
    """Get detailed quality report for a training dataset."""
    try:
        ds = await db.get(TrainingDataset, dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Training dataset not found")

        # Get samples to generate fresh quality report
        result = await db.execute(
            select(TrainingSample).where(TrainingSample.dataset_id == ds.id)
        )
        samples = list(result.scalars().all())

        # Generate quality report
        r2 = R2StorageClient()
        collector = TrainingDataCollector(db, r2)
        quality_report = collector.validate_data_quality(list(samples))

        return create_response(
            {
                "dataset_id": dataset_id,
                "quality_report": {
                    "total_samples": quality_report.total_samples,
                    "anomaly_ratio": quality_report.anomaly_ratio,
                    "image_quality_score": quality_report.image_quality_score,
                    "label_consistency": quality_report.label_consistency,
                    "temporal_distribution": quality_report.temporal_distribution,
                    "survey_coverage": quality_report.survey_coverage,
                    "quality_score": quality_report.quality_score,
                    "issues": quality_report.issues,
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset quality {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/datasets/{dataset_id}/load")
async def load_dataset_for_training(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    auth=Depends(require_permission_or_api_key(Permission.READ_DATA)),
):
    """Load dataset for training and return data splits information."""
    try:
        ds = await db.get(TrainingDataset, dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Training dataset not found")

        # Get samples
        result = await db.execute(
            select(TrainingSample).where(TrainingSample.dataset_id == ds.id)
        )
        samples = list(result.scalars().all())

        if not samples:
            raise HTTPException(status_code=404, detail="No samples found in dataset")

        # Create data loader and splits
        r2 = R2StorageClient()
        data_loader = RealDataLoader(db, r2)
        train_samples, val_samples, test_samples = data_loader.create_data_splits(
            list(samples)
        )

        return create_response(
            {
                "dataset_id": dataset_id,
                "splits": {
                    "train": len(train_samples),
                    "validation": len(val_samples),
                    "test": len(test_samples),
                },
                "total_samples": len(samples),
                "ready_for_training": True,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e
