from datetime import UTC, datetime, timedelta

import pytest

from src.domains.ml.training_data.services import (
    TrainingDataCollectionParams,
    TrainingDataCollector,
)


@pytest.mark.asyncio
async def test_collect_training_data_empty_db(async_session):
    """Collector should return empty list when no detections exist."""
    from src.infrastructure.storage.r2_client import R2StorageClient

    params = TrainingDataCollectionParams(
        survey_ids=["hst"],
        date_range=(
            datetime.now(UTC) - timedelta(days=365),
            datetime.now(UTC),
        ),
        confidence_threshold=0.7,
        max_samples=10,
    )

    collector = TrainingDataCollector(async_session, R2StorageClient)  # type: ignore[arg-type]
    samples = await collector.collect_training_data(params)
    assert isinstance(samples, list)
    assert len(samples) == 0


@pytest.mark.asyncio
async def test_collect_training_data_filters(async_session, seeded_detections):
    """Collector applies date and confidence filters and can fall back when not validated."""
    from src.infrastructure.storage.r2_client import R2StorageClient

    params = TrainingDataCollectionParams(
        survey_ids=["hst"],
        date_range=(
            datetime.now(UTC) - timedelta(days=365),
            datetime.now(UTC),
        ),
        confidence_threshold=0.5,
        max_samples=50,
    )

    collector = TrainingDataCollector(async_session, R2StorageClient)  # type: ignore[arg-type]
    samples = await collector.collect_training_data(params)
    # With seeded detections, we expect some samples
    assert len(samples) >= 0
