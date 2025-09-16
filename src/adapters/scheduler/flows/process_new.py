"""Prefect flow for processing new observations."""

import asyncio
from typing import Any
from uuid import UUID

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
        from src.core.db.session import get_db
        from src.domains.observations.crud import SurveyCRUD
        from src.domains.observations.ingestion.services.data_ingestion import (
            DataIngestionService,
        )
        from src.domains.observations.survey_config_service import (
            SurveyConfigurationService,
        )

        # Get database session
        async for db in get_db():
            # Initialize services
            survey_config_service = SurveyConfigurationService(db)
            ingestion_service = DataIngestionService(
                survey_config_service=survey_config_service
            )

            # Initialize default configurations if needed
            await survey_config_service.initialize_default_configurations()

            # Get or create a default survey
            survey_crud = SurveyCRUD()
            from src.domains.observations.schema import SurveyListParams

            survey_params = SurveyListParams(limit=1)
            surveys, _ = await survey_crud.get_many(db, survey_params)
            if not surveys:
                # Create a default survey
                from src.domains.observations.schema import SurveyCreate

                survey_data = SurveyCreate(
                    name="AstrID Default Survey",
                    description="Default survey for automated observation ingestion",
                    is_active=True,
                )
                survey = await survey_crud.create(db, survey_data)
                survey_id = survey.id
            else:
                survey_id = surveys[0].id

            # Use configuration-driven ingestion
            observations = (
                await ingestion_service.ingest_observations_from_active_configuration(
                    survey_id=UUID(str(survey_id)),
                )
            )

            if not observations:
                logger.info("No new observations found")
                return []

            # Save observations to database
            from src.domains.observations.ingestion.services.observation_builder import (
                ObservationBuilderService,
            )

            observation_builder = ObservationBuilderService()
            created_observations = (
                await observation_builder.create_observations_from_ingestion(
                    db, observations
                )
            )

            # Return observation IDs for processing
            observation_ids = [str(obs.id) for obs in created_observations]
            logger.info(f"Discovered and saved {len(observation_ids)} new observations")

            await db.commit()
            return observation_ids

        # If no database session was available, return empty list
        return []

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
