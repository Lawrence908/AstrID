"""Dramatiq worker for training data collection (ASTR-113)."""

from __future__ import annotations

import logging

import dramatiq
from sqlalchemy.orm import Session

from src.core.db.session import SessionLocal
from src.domains.ml.training_data.services import (
    TrainingDataCollectionParams,
    TrainingDataCollector,
)
from src.infrastructure.storage.r2_client import R2StorageClient

logger = logging.getLogger(__name__)


def _get_db() -> Session:
    return SessionLocal()


@dramatiq.actor(queue_name="training_data")
def collect_training_data_worker(collection_params: dict):
    db = _get_db()
    try:
        r2 = R2StorageClient()
        collector = TrainingDataCollector(db, r2)
        params = TrainingDataCollectionParams(**collection_params)
        samples = collector.collect_training_data(params)
        quality = collector.validate_data_quality(samples)
        logger.info(
            "Collected training samples: total=%s, quality_score=%.3f",
            len(samples),
            quality.get("quality_score", 0.0),
        )
    finally:
        db.close()
