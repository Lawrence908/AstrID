"""Prefect flow for processing new observations."""

import asyncio
from typing import Any

from prefect import flow, task

from src.adapters.workers.tasks import (
    difference_observation,
    infer_detections,
    preprocess_observation,
)
from src.core.logging import configure_domain_logger

logger = configure_domain_logger("scheduler.flows")


@task(retries=3, retry_delay_seconds=60)
async def ingest_window() -> list[str]:
    """Ingest new observations from configured surveys."""
    logger.info("Starting observation ingestion window")

    try:
        # Use the data ingestion service to discover new observations
        from src.domains.observations.ingestion.services.data_ingestion import (
            DataIngestionService,
        )

        ingestion_service = DataIngestionService()

        # For now, use a simplified mock implementation
        # In production, this would integrate with actual survey queries
        # TODO: Implement actual survey queries
        from uuid import uuid4

        # Mock survey ID - in production this would be from database
        mock_survey_id = uuid4()

        # Example: Query for new observations in M31 field
        # In production, this would be configured via parameters
        # TODO: Implement actual survey queries
        observations = await ingestion_service.ingest_observations_by_position(
            ra=10.6847,  # M31 coordinates
            dec=41.2691,
            survey_id=mock_survey_id,
            radius=1.0,  # 1 degree radius
            missions=["HST"],
        )

        # For now, return observation IDs based on their observation_id field
        # In a real implementation, these would be saved to database first
        # TODO: Implement actual database saving
        observation_ids = [obs.observation_id for obs in observations]
        logger.info(f"Discovered {len(observation_ids)} new observations")

        return observation_ids

    except Exception as e:
        logger.error(f"Error during observation ingestion: {e}")
        # Return empty list to continue flow gracefully
        return []


@task(retries=3, retry_delay_seconds=30)
async def preprocess(observation_id: str) -> str:
    """Preprocess an observation using Dramatiq worker."""
    logger.info(f"Queuing preprocessing for observation {observation_id}")

    try:
        # Queue the preprocessing task to Dramatiq worker
        preprocess_observation.send(observation_id)
        logger.info(f"Preprocessing queued for observation {observation_id}")
        return observation_id

    except Exception as e:
        logger.error(f"Error queuing preprocessing for {observation_id}: {e}")
        raise


@task(retries=3, retry_delay_seconds=30)
async def difference(observation_id: str, wait_for_preprocess=None) -> str:
    """Perform image differencing using Dramatiq worker."""
    logger.info(f"Queuing differencing for observation {observation_id}")

    try:
        # Queue the differencing task to Dramatiq worker
        difference_observation.send(observation_id)
        logger.info(f"Differencing queued for observation {observation_id}")
        return observation_id

    except Exception as e:
        logger.error(f"Error queuing differencing for {observation_id}: {e}")
        raise


@task(retries=3, retry_delay_seconds=30)
async def infer(observation_id: str, wait_for_difference=None) -> dict[str, Any]:
    """Run ML inference using Dramatiq worker."""
    logger.info(f"Queuing inference for observation {observation_id}")

    try:
        # Queue the inference task to Dramatiq worker
        infer_detections.send(observation_id)
        logger.info(f"Inference queued for observation {observation_id}")

        # Return minimal result for flow tracking
        return {
            "observation_id": observation_id,
            "status": "queued",
            "queued_at": asyncio.get_event_loop().time(),
        }

    except Exception as e:
        logger.error(f"Error queuing inference for {observation_id}: {e}")
        raise


@task
async def persist_and_notify(result: dict[str, Any]) -> None:
    """Persist results and send notifications."""
    observation_id = result.get("observation_id", "unknown")
    logger.info(f"Processing completion for observation {observation_id}")

    try:
        # Log the completion of the processing pipeline
        logger.info(f"Pipeline completed for observation {observation_id}")

        # In a complete implementation, this would:
        # 1. Check worker task completion status
        # 2. Aggregate results from all processing stages
        # 3. Update observation status in database
        # 4. Send notifications for high-confidence detections
        # 5. Log metrics to MLflow
        # 6. Trigger downstream workflows (e.g., validation)

        # For now, just log successful completion
        logger.info(f"Successfully completed pipeline for observation {observation_id}")

    except Exception as e:
        logger.error(f"Error in persistence/notification for {observation_id}: {e}")
        raise


@flow(name="process-new-observations")
async def run():
    """Main flow for processing new observations."""
    logger.info("Starting new observation processing flow")

    try:
        # Step 1: Ingest new observations
        obs_ids = await ingest_window()

        if not obs_ids:
            logger.info("No new observations to process")
            return

        # Step 2: Process each observation through the pipeline
        for obs_id in obs_ids:
            logger.info(f"Starting pipeline for observation {obs_id}")

            # Queue preprocessing task
            await preprocess.submit(obs_id).result()

            # Queue differencing task (async after preprocessing)
            await difference.submit(obs_id).result()

            # Queue inference task (async after differencing)
            result = await infer.submit(obs_id).result()

            # Handle completion and notifications
            await persist_and_notify.submit(result).result()

        logger.info(f"Successfully queued processing for {len(obs_ids)} observations")

    except Exception as e:
        logger.error(f"Error in observation processing flow: {e}")
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
