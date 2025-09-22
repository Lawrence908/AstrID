"""
Dramatiq tasks for AstrID background processing.

This module defines the background tasks that will be processed by dramatiq workers.
"""

import logging

import dramatiq

from src.adapters.workers.config import get_worker_config, worker_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure worker manager
config = get_worker_config()
worker_manager.setup_broker(config)


# Import all worker modules to register their actors
from src.adapters.workers.curation.curation_workers import (  # noqa: E402
    send_notifications,
)
from src.adapters.workers.detection.detection_workers import (  # noqa: E402
    detect_anomalies,
)
from src.adapters.workers.differencing.differencing_workers import (  # noqa: E402
    difference_observation,
)
from src.adapters.workers.ingestion.observation_workers import (  # noqa: E402
    ingest_observation,
)
from src.adapters.workers.preprocessing.preprocessing_workers import (  # noqa: E402
    preprocess_observation,
)
from src.adapters.workers.training_data import (  # noqa: E402
    collect_training_data_worker,
)


# Legacy tasks for backward compatibility
@dramatiq.actor(queue_name="observation_ingestion")
def legacy_ingest_observation(observation_id: str):
    """Legacy ingest observation task for backward compatibility."""
    logger.info(f"Starting legacy ingestion for observation {observation_id}")
    # Redirect to new worker
    return ingest_observation.send({"observation_id": observation_id})


@dramatiq.actor(queue_name="preprocessing")
def legacy_preprocess_observation(observation_id: str):
    """Legacy preprocess observation task for backward compatibility."""
    logger.info(f"Starting legacy preprocessing for observation {observation_id}")
    # Redirect to new worker
    return preprocess_observation.send(observation_id)


@dramatiq.actor(queue_name="differencing")
def legacy_difference_observation(observation_id: str):
    """Legacy difference observation task for backward compatibility."""
    logger.info(f"Starting legacy differencing for observation {observation_id}")
    # Redirect to new worker
    return difference_observation.send(observation_id)


@dramatiq.actor(queue_name="detection")
def legacy_infer_detections(observation_id: str):
    """Legacy infer detections task for backward compatibility."""
    logger.info(f"Starting legacy inference for observation {observation_id}")
    # Redirect to new worker
    return detect_anomalies.send(f"diff_{observation_id}", "unet_v1")


@dramatiq.actor(queue_name="notification")
def legacy_notify_detection(detection_data: dict):
    """Legacy notify detection task for backward compatibility."""
    logger.info(f"Starting legacy notification for detection {detection_data}")
    # Redirect to new worker
    return send_notifications.send(detection_data.get("detection_id", "unknown"))


@dramatiq.actor(queue_name="training_data")
def legacy_collect_training_data(collection_params: dict):
    """Legacy wrapper to trigger training data collection actor."""
    logger.info("Starting training data collection via legacy wrapper")
    return collect_training_data_worker.send(collection_params)


# Health check task
@dramatiq.actor(queue_name="default")
def health_check():
    """Health check task to verify worker is functioning."""
    logger.info("Health check task executed")
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}
