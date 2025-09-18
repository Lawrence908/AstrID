"""Observation processing workflows using Prefect."""

import logging
from typing import Any
from uuid import UUID

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact

from ...observations.service import ObservationService

logger = logging.getLogger(__name__)


@task(name="validate_observation_data")
async def validate_observation_data_task(
    observation_data: dict[str, Any],
) -> dict[str, Any]:
    """Validate observation data before processing.

    Args:
        observation_data: Raw observation data

    Returns:
        Validated observation data
    """
    log = get_run_logger()
    log.info("Validating observation data")

    # Basic validation
    required_fields = ["survey_id", "coordinates", "exposure_time", "filter_name"]
    for field in required_fields:
        if field not in observation_data:
            raise ValueError(f"Missing required field: {field}")

    # Validate coordinates
    coords = observation_data["coordinates"]
    if not isinstance(coords, dict) or "ra" not in coords or "dec" not in coords:
        raise ValueError("Invalid coordinates format")

    # Validate exposure time
    if observation_data["exposure_time"] <= 0:
        raise ValueError("Exposure time must be positive")

    log.info("Observation data validation completed")
    return observation_data


@task(name="create_observation_record")
async def create_observation_record_task(
    observation_data: dict[str, Any], observation_service: ObservationService
) -> dict[str, Any]:
    """Create observation record in database.

    Args:
        observation_data: Validated observation data
        observation_service: Observation service

    Returns:
        Created observation record
    """
    log = get_run_logger()
    log.info("Creating observation record")

    # Create observation using the service
    observation = await observation_service.create_observation(observation_data)

    # Emit event (simplified for now)
    log.info(f"Created observation record: {observation.id}")
    return {
        "id": str(observation.id),
        "survey_id": str(observation.survey_id),
        "status": observation.status,
        "coordinates": observation.coordinates,
    }


@task(name="preprocess_observation")
async def preprocess_observation_task(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Preprocess observation image.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Preprocessing results
    """
    log = get_run_logger()
    log.info(f"Preprocessing observation: {observation_id}")

    # Get observation (for validation purposes)
    await observation_service.get_observation(observation_id)

    # Simulate preprocessing (in production, this would call the preprocessing service)
    log.info(f"Preprocessing completed for observation: {observation_id}")
    return {
        "observation_id": str(observation_id),
        "status": "preprocessed",
        "processing_time": 120.5,
    }


@task(name="create_difference_image")
async def create_difference_image_task(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Create difference image for anomaly detection.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Differencing results
    """
    log = get_run_logger()
    log.info(f"Creating difference image for observation: {observation_id}")

    # Simulate differencing (in production, this would call the differencing service)
    log.info(f"Difference image created for observation: {observation_id}")
    return {
        "observation_id": str(observation_id),
        "status": "differenced",
        "processing_time": 95.2,
    }


@task(name="detect_anomalies")
async def detect_anomalies_task(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Detect anomalies using ML model.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Detection results
    """
    log = get_run_logger()
    log.info(f"Detecting anomalies for observation: {observation_id}")

    # Simulate anomaly detection (in production, this would call the detection service)
    log.info(f"Anomaly detection completed for observation: {observation_id}")
    return {
        "observation_id": str(observation_id),
        "status": "detected",
        "anomalies_found": 3,
        "processing_time": 45.8,
    }


@task(name="validate_detections")
async def validate_detections_task(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Validate detected anomalies.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Validation results
    """
    log = get_run_logger()
    log.info(f"Validating detections for observation: {observation_id}")

    # Simulate validation (in production, this would call the validation service)
    log.info(f"Detection validation completed for observation: {observation_id}")
    return {
        "observation_id": str(observation_id),
        "status": "validated",
        "valid_detections": 2,
        "processing_time": 15.3,
    }


@task(name="update_observation_status")
async def update_observation_status_task(
    observation_id: UUID, status: str, observation_service: ObservationService
) -> None:
    """Update observation processing status.

    Args:
        observation_id: Observation identifier
        status: New status
        observation_service: Observation service
    """
    log = get_run_logger()
    log.info(f"Updating observation status: {observation_id} -> {status}")

    # Update status using the service
    await observation_service.update_observation_status(observation_id, status)
    log.info(f"Observation status updated: {observation_id} -> {status}")


@flow(name="observation_ingestion_flow")
async def observation_ingestion_flow(
    observation_data: dict[str, Any], observation_service: ObservationService
) -> dict[str, Any]:
    """Ingest new observation into the system.

    Args:
        observation_data: Raw observation data
        observation_service: Observation service

    Returns:
        Ingestion results
    """
    log = get_run_logger()
    log.info("Starting observation ingestion flow")

    try:
        # Validate data
        validated_data = await validate_observation_data_task(observation_data)

        # Create observation record
        observation = await create_observation_record_task(
            validated_data, observation_service
        )

        # Update status
        await update_observation_status_task(
            UUID(observation["id"]), "ingested", observation_service
        )

        result = {
            "observation_id": observation["id"],
            "status": "ingested",
            "survey_id": observation["survey_id"],
            "coordinates": observation["coordinates"],
        }

        # Create artifact
        create_table_artifact(
            key="observation-ingestion-result",
            table=result,
            description="Observation ingestion results",
        )

        log.info(f"Observation ingestion completed: {observation['id']}")
        return result

    except Exception as e:
        log.error(f"Observation ingestion failed: {e}")
        raise


@flow(name="observation_preprocessing_flow")
async def observation_preprocessing_flow(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Preprocess observation image.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Preprocessing results
    """
    log = get_run_logger()
    log.info(f"Starting preprocessing flow for observation: {observation_id}")

    try:
        # Preprocess observation
        result = await preprocess_observation_task(observation_id, observation_service)

        # Update status
        await update_observation_status_task(
            observation_id, "preprocessed", observation_service
        )

        log.info(f"Preprocessing completed for observation: {observation_id}")
        return result

    except Exception as e:
        log.error(f"Preprocessing failed for observation {observation_id}: {e}")
        raise


@flow(name="observation_differencing_flow")
async def observation_differencing_flow(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Create difference image for observation.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Differencing results
    """
    log = get_run_logger()
    log.info(f"Starting differencing flow for observation: {observation_id}")

    try:
        # Create difference image
        result = await create_difference_image_task(observation_id, observation_service)

        # Update status
        await update_observation_status_task(
            observation_id, "differenced", observation_service
        )

        log.info(f"Differencing completed for observation: {observation_id}")
        return result

    except Exception as e:
        log.error(f"Differencing failed for observation {observation_id}: {e}")
        raise


@flow(name="observation_detection_flow")
async def observation_detection_flow(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Detect anomalies in observation.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Detection results
    """
    log = get_run_logger()
    log.info(f"Starting detection flow for observation: {observation_id}")

    try:
        # Detect anomalies
        detection_result = await detect_anomalies_task(
            observation_id, observation_service
        )

        # Validate detections
        validation_result = await validate_detections_task(
            observation_id, observation_service
        )

        # Update status
        await update_observation_status_task(
            observation_id, "detected", observation_service
        )

        result = {
            "observation_id": str(observation_id),
            "detection_result": detection_result,
            "validation_result": validation_result,
        }

        log.info(f"Detection completed for observation: {observation_id}")
        return result

    except Exception as e:
        log.error(f"Detection failed for observation {observation_id}: {e}")
        raise


@flow(name="observation_validation_flow")
async def observation_validation_flow(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Validate observation processing results.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Validation results
    """
    log = get_run_logger()
    log.info(f"Starting validation flow for observation: {observation_id}")

    try:
        # Get observation (for validation purposes)
        await observation_service.get_observation(observation_id)

        # Simulate validation
        validation_result = {
            "observation_id": str(observation_id),
            "status": "validated",
            "validation_score": 0.95,
        }

        # Update status
        await update_observation_status_task(
            observation_id, "validated", observation_service
        )

        log.info(f"Validation completed for observation: {observation_id}")
        return validation_result

    except Exception as e:
        log.error(f"Validation failed for observation {observation_id}: {e}")
        raise


@flow(name="complete_observation_processing_flow")
async def complete_observation_processing_flow(
    observation_id: UUID, observation_service: ObservationService
) -> dict[str, Any]:
    """Complete end-to-end observation processing.

    Args:
        observation_id: Observation identifier
        observation_service: Observation service

    Returns:
        Complete processing results
    """
    log = get_run_logger()
    log.info(f"Starting complete processing flow for observation: {observation_id}")

    try:
        # Run all processing steps
        preprocessing_result = await observation_preprocessing_flow(
            observation_id, observation_service
        )

        differencing_result = await observation_differencing_flow(
            observation_id, observation_service
        )

        detection_result = await observation_detection_flow(
            observation_id, observation_service
        )

        validation_result = await observation_validation_flow(
            observation_id, observation_service
        )

        # Update final status
        await update_observation_status_task(
            observation_id, "completed", observation_service
        )

        result = {
            "observation_id": str(observation_id),
            "status": "completed",
            "preprocessing": preprocessing_result,
            "differencing": differencing_result,
            "detection": detection_result,
            "validation": validation_result,
        }

        # Create artifact
        create_table_artifact(
            key="observation-processing-result",
            table=result,
            description="Complete observation processing results",
        )

        log.info(f"Complete processing finished for observation: {observation_id}")
        return result

    except Exception as e:
        log.error(f"Complete processing failed for observation {observation_id}: {e}")
        raise
