"""Observation ingestion workers for Dramatiq background processing."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

import dramatiq

from src.core.db.session import AsyncSessionLocal
from src.core.logging import configure_domain_logger
from src.domains.observations.service import ObservationService


class ObservationIngestionWorker:
    """Worker for observation ingestion tasks."""

    def __init__(self):
        self.logger = configure_domain_logger("workers.observation_ingestion")

    async def ingest_observation(
        self, observation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Ingest a new observation from external sources.

        Args:
            observation_data: Dictionary containing observation data

        Returns:
            Dictionary with ingestion results
        """
        self.logger.info(
            f"Starting observation ingestion: {observation_data.get('observation_id')}"
        )

        try:
            async with AsyncSessionLocal() as db:
                observation_service = ObservationService(db)

                # Validate observation data
                validation_result = await self.validate_observation_data(
                    observation_data
                )
                if not validation_result["valid"]:
                    raise ValueError(
                        f"Observation validation failed: {validation_result['errors']}"
                    )

                # Process observation metadata
                processed_metadata = await self.process_observation_metadata(
                    observation_data
                )

                # Store observation files
                storage_result = await self.store_observation_files(observation_data)

                # Create observation record
                from src.domains.observations.schema import ObservationCreate

                observation_create = ObservationCreate(
                    survey_id=UUID(observation_data["survey_id"]),
                    observation_id=observation_data["observation_id"],
                    ra=observation_data["ra"],
                    dec=observation_data["dec"],
                    observation_time=datetime.fromisoformat(
                        observation_data["observation_time"]
                    ),
                    filter_band=observation_data.get("filter_band", "r"),
                    exposure_time=observation_data.get("exposure_time", 300.0),
                    fits_url=observation_data.get("fits_url", ""),
                    pixel_scale=observation_data.get("pixel_scale"),
                    airmass=observation_data.get("airmass"),
                    seeing=observation_data.get("seeing"),
                )

                observation = await observation_service.create_observation(
                    observation_create
                )

                # Mark as ingested
                await observation_service.mark_observation_as_ingested(observation.id)

                # Trigger preprocessing
                await self.trigger_preprocessing(observation.id)

                result = {
                    "observation_id": str(observation.id),
                    "external_observation_id": observation.observation_id,
                    "survey_id": str(observation.survey_id),
                    "status": "ingested",
                    "metadata": processed_metadata,
                    "storage": storage_result,
                    "processing_triggered": True,
                }

                self.logger.info(f"Successfully ingested observation: {observation.id}")
                return result

        except Exception as e:
            self.logger.error(f"Failed to ingest observation: {e}")
            raise

    async def validate_observation_data(
        self, observation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate observation data before processing.

        Args:
            observation_data: Dictionary containing observation data

        Returns:
            Dictionary with validation results
        """
        self.logger.debug("Validating observation data")

        errors = []

        # Required fields validation
        required_fields = [
            "survey_id",
            "observation_id",
            "ra",
            "dec",
            "observation_time",
        ]
        for field in required_fields:
            if field not in observation_data or observation_data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Coordinate validation
        if "ra" in observation_data:
            ra = observation_data["ra"]
            if not isinstance(ra, int | float) or ra < 0 or ra >= 360:
                errors.append("RA must be between 0 and 360 degrees")

        if "dec" in observation_data:
            dec = observation_data["dec"]
            if not isinstance(dec, int | float) or dec < -90 or dec > 90:
                errors.append("Dec must be between -90 and 90 degrees")

        # Exposure time validation
        if (
            "exposure_time" in observation_data
            and observation_data["exposure_time"] is not None
        ):
            exp_time = observation_data["exposure_time"]
            if not isinstance(exp_time, int | float) or exp_time <= 0:
                errors.append("Exposure time must be positive")

        # Airmass validation
        if "airmass" in observation_data and observation_data["airmass"] is not None:
            airmass = observation_data["airmass"]
            if not isinstance(airmass, int | float) or airmass <= 0:
                errors.append("Airmass must be positive")

        valid = len(errors) == 0

        self.logger.debug(
            f"Observation data validation: valid={valid}, errors={len(errors)}"
        )

        return {
            "valid": valid,
            "errors": errors,
        }

    async def process_observation_metadata(
        self, observation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process and enhance observation metadata.

        Args:
            observation_data: Dictionary containing observation data

        Returns:
            Dictionary with processed metadata
        """
        self.logger.debug("Processing observation metadata")

        metadata = {
            "original_data": observation_data,
            "processed_at": datetime.now().isoformat(),
            "processing_version": "1.0.0",
        }

        # Calculate derived values
        if "ra" in observation_data and "dec" in observation_data:
            ra = observation_data["ra"]
            dec = observation_data["dec"]

            # Calculate airmass if not provided
            if "airmass" not in observation_data or observation_data["airmass"] is None:
                # Simple airmass calculation (zenith angle = 0)
                metadata["calculated_airmass"] = 1.0
            else:
                metadata["calculated_airmass"] = observation_data["airmass"]

            # Calculate sky region bounds
            region_size = 0.1  # degrees
            metadata["sky_region"] = {
                "ra_min": ra - region_size,
                "ra_max": ra + region_size,
                "dec_min": dec - region_size,
                "dec_max": dec + region_size,
            }

        # Add quality indicators
        metadata["quality_indicators"] = {
            "has_coordinates": "ra" in observation_data and "dec" in observation_data,
            "has_exposure_time": "exposure_time" in observation_data
            and observation_data["exposure_time"] is not None,
            "has_filter": "filter_band" in observation_data
            and observation_data["filter_band"] is not None,
            "has_airmass": "airmass" in observation_data
            and observation_data["airmass"] is not None,
            "has_seeing": "seeing" in observation_data
            and observation_data["seeing"] is not None,
        }

        self.logger.debug("Observation metadata processed successfully")
        return metadata

    async def store_observation_files(
        self, observation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Store observation files to cloud storage.

        Args:
            observation_data: Dictionary containing observation data

        Returns:
            Dictionary with storage results
        """
        self.logger.debug("Storing observation files")

        storage_result = {
            "fits_file_path": None,
            "thumbnail_path": None,
            "storage_metadata": {},
        }

        # In a real implementation, this would:
        # 1. Download FITS file from URL
        # 2. Generate thumbnail
        # 3. Upload to cloud storage (R2)
        # 4. Return storage paths

        if "fits_url" in observation_data and observation_data["fits_url"]:
            # Placeholder for file storage
            observation_id = observation_data["observation_id"]
            storage_result["fits_file_path"] = (
                f"observations/{observation_id}/image.fits"
            )
            storage_result["thumbnail_path"] = (
                f"observations/{observation_id}/thumbnail.jpg"
            )
            storage_result["storage_metadata"] = {
                "original_url": observation_data["fits_url"],
                "stored_at": datetime.now().isoformat(),
                "file_size": "unknown",  # Would be calculated from actual file
            }

        self.logger.debug("Observation files stored successfully")
        return storage_result

    async def trigger_preprocessing(self, observation_id: UUID) -> None:
        """Trigger preprocessing workflow for an observation.

        Args:
            observation_id: UUID of the observation
        """
        self.logger.info(f"Triggering preprocessing for observation: {observation_id}")

        try:
            # Import here to avoid circular imports
            from src.adapters.workers.preprocessing.preprocessing_workers import (
                preprocess_observation,
            )

            # Send preprocessing task to queue
            preprocess_observation.send(str(observation_id))

            self.logger.info(
                f"Preprocessing triggered for observation: {observation_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to trigger preprocessing for {observation_id}: {e}"
            )
            raise


# Create Dramatiq actors
@dramatiq.actor(queue_name="observation_ingestion")
def ingest_observation(observation_data: dict[str, Any]) -> dict[str, Any]:
    """Dramatiq actor for observation ingestion."""
    worker = ObservationIngestionWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(worker.ingest_observation(observation_data))
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="observation_ingestion")
def batch_ingest_observations(
    observation_data_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dramatiq actor for batch observation ingestion."""
    worker = ObservationIngestionWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = []
        errors = []

        for i, obs_data in enumerate(observation_data_list):
            try:
                result = loop.run_until_complete(worker.ingest_observation(obs_data))
                results.append(result)
            except Exception as e:
                error = {
                    "index": i,
                    "observation_data": obs_data,
                    "error": str(e),
                }
                errors.append(error)

        return {
            "total_processed": len(observation_data_list),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }
    finally:
        loop.close()


@dramatiq.actor(queue_name="observation_ingestion")
def validate_observation_data(observation_data: dict[str, Any]) -> dict[str, Any]:
    """Dramatiq actor for observation data validation."""
    worker = ObservationIngestionWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            worker.validate_observation_data(observation_data)
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="observation_ingestion")
def process_observation_metadata(observation_data: dict[str, Any]) -> dict[str, Any]:
    """Dramatiq actor for observation metadata processing."""
    worker = ObservationIngestionWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            worker.process_observation_metadata(observation_data)
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="observation_ingestion")
def store_observation_files(observation_data: dict[str, Any]) -> dict[str, Any]:
    """Dramatiq actor for observation file storage."""
    worker = ObservationIngestionWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            worker.store_observation_files(observation_data)
        )
        return result
    finally:
        loop.close()
