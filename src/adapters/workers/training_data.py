"""Dramatiq worker for training data collection (ASTR-113)."""

from __future__ import annotations

import asyncio
import logging

import dramatiq

from src.core.db.session import AsyncSessionLocal
from src.domains.ml.training_data.services import (
    TrainingDataCollectionParams,
    TrainingDataCollector,
)
from src.infrastructure.storage.r2_client import R2StorageClient

logger = logging.getLogger(__name__)


@dramatiq.actor(queue_name="training_data")
def collect_training_data_worker(collection_params: dict):
    """Collect training data using an async DB session inside Dramatiq actor."""

    async def _run_async(params_dict: dict) -> None:
        async with AsyncSessionLocal() as db:
            r2 = R2StorageClient()
            collector = TrainingDataCollector(db, r2)
            params = TrainingDataCollectionParams(**params_dict)
            samples = await collector.collect_training_data(params)
            quality = collector.validate_data_quality(samples)
            logger.info(
                "Collected training samples: total=%s, quality_score=%.3f",
                len(samples),
                quality.get("quality_score", 0.0),
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_async(collection_params))
    finally:
        loop.close()
