"""Preprocessing workers for Dramatiq background processing."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

import dramatiq

from src.core.db.session import AsyncSessionLocal
from src.core.logging import configure_domain_logger
from src.domains.observations.service import ObservationService
from src.domains.preprocessing.service import PreprocessRunService


class PreprocessingWorker:
    """Worker for preprocessing tasks."""

    def __init__(self):
        self.logger = configure_domain_logger("workers.preprocessing")

    async def preprocess_observation(self, observation_id: UUID) -> dict[str, Any]:
        """Preprocess an observation (calibration, registration, etc.).

        Args:
            observation_id: UUID of the observation to preprocess

        Returns:
            Dictionary with preprocessing results
        """
        self.logger.info(f"Starting preprocessing for observation: {observation_id}")

        try:
            async with AsyncSessionLocal() as db:
                observation_service = ObservationService(db)
                preprocessing_service = PreprocessRunService(db)

                # Get observation
                observation = await observation_service.get_observation_by_id(
                    observation_id
                )
                if not observation:
                    raise ValueError(f"Observation {observation_id} not found")

                # Mark as preprocessing
                await observation_service.mark_observation_as_preprocessing(
                    observation_id
                )

                # Apply calibration
                calibration_result = await self.apply_calibration(observation_id, {})

                # Align observation
                alignment_result = await self.align_observation(observation_id, None)

                # Assess quality
                quality_result = await self.assess_quality(observation_id)

                # Create preprocessing run record
                from src.domains.preprocessing.schema import PreprocessRunCreate

                preprocess_create = PreprocessRunCreate(
                    observation_id=observation_id,
                )

                preprocess_run = await preprocessing_service.create_preprocess_run(
                    preprocess_create
                )

                # Mark observation as preprocessed
                await observation_service.mark_observation_as_preprocessed(
                    observation_id
                )

                # Trigger differencing
                await self.trigger_differencing(observation_id)

                result = {
                    "observation_id": str(observation_id),
                    "preprocess_run_id": str(preprocess_run.id),
                    "status": "preprocessed",
                    "calibration": calibration_result,
                    "alignment": alignment_result,
                    "quality": quality_result,
                    "processing_triggered": True,
                }

                self.logger.info(
                    f"Successfully preprocessed observation: {observation_id}"
                )
                return result

        except Exception as e:
            self.logger.error(f"Failed to preprocess observation {observation_id}: {e}")
            # Mark observation as failed
            try:
                async with AsyncSessionLocal() as db:
                    observation_service = ObservationService(db)
                    await observation_service.mark_observation_as_failed(observation_id)
            except Exception as mark_error:
                self.logger.error(f"Failed to mark observation as failed: {mark_error}")
            raise

    async def apply_calibration(
        self, observation_id: UUID, calibration_frames: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply calibration corrections to an observation.

        Args:
            observation_id: UUID of the observation
            calibration_frames: Dictionary containing calibration frame data

        Returns:
            Dictionary with calibration results
        """
        self.logger.debug(f"Applying calibration for observation: {observation_id}")

        calibration_result = {
            "observation_id": str(observation_id),
            "calibration_applied": True,
            "bias_correction": {
                "applied": True,
                "bias_frame": calibration_frames.get("bias_frame"),
                "correction_factor": 1.0,
            },
            "dark_correction": {
                "applied": True,
                "dark_frame": calibration_frames.get("dark_frame"),
                "exposure_time": 300.0,  # seconds
                "correction_factor": 1.0,
            },
            "flat_correction": {
                "applied": True,
                "flat_frame": calibration_frames.get("flat_frame"),
                "normalization_factor": 1.0,
            },
            "processed_at": datetime.now().isoformat(),
        }

        # In a real implementation, this would:
        # 1. Load the observation FITS file
        # 2. Apply bias correction
        # 3. Apply dark correction
        # 4. Apply flat field correction
        # 5. Calculate uncertainty propagation
        # 6. Store calibrated image

        self.logger.debug(f"Calibration applied for observation: {observation_id}")
        return calibration_result

    async def align_observation(
        self, observation_id: UUID, reference_id: UUID | None
    ) -> dict[str, Any]:
        """Align observation with reference frame.

        Args:
            observation_id: UUID of the observation to align
            reference_id: UUID of the reference observation (optional)

        Returns:
            Dictionary with alignment results
        """
        self.logger.debug(f"Aligning observation: {observation_id}")

        alignment_result = {
            "observation_id": str(observation_id),
            "reference_id": str(reference_id) if reference_id else None,
            "alignment_applied": True,
            "wcs_correction": {
                "applied": True,
                "ra_offset": 0.0,  # arcseconds
                "dec_offset": 0.0,  # arcseconds
                "rotation": 0.0,  # degrees
                "scale_factor": 1.0,
            },
            "registration_quality": {
                "rms_error": 0.1,  # arcseconds
                "n_matched_stars": 50,
                "correlation_coefficient": 0.95,
            },
            "processed_at": datetime.now().isoformat(),
        }

        # In a real implementation, this would:
        # 1. Load observation and reference images
        # 2. Extract star catalogs
        # 3. Match stars between images
        # 4. Calculate WCS transformation
        # 5. Apply alignment correction
        # 6. Validate alignment quality

        self.logger.debug(f"Alignment completed for observation: {observation_id}")
        return alignment_result

    async def assess_quality(self, observation_id: UUID) -> dict[str, Any]:
        """Assess image quality metrics.

        Args:
            observation_id: UUID of the observation

        Returns:
            Dictionary with quality assessment results
        """
        self.logger.debug(f"Assessing quality for observation: {observation_id}")

        quality_result = {
            "observation_id": str(observation_id),
            "overall_quality_score": 0.85,
            "background_analysis": {
                "mean_background": 1000.0,
                "background_rms": 50.0,
                "background_uniformity": 0.95,
            },
            "noise_analysis": {
                "read_noise": 5.0,
                "photon_noise": 10.0,
                "total_noise": 11.2,
                "snr_estimate": 89.3,
            },
            "cosmic_ray_analysis": {
                "n_cosmic_rays": 12,
                "cosmic_ray_fraction": 0.001,
                "cleaned": True,
            },
            "flatness_analysis": {
                "flatness_rms": 0.02,
                "flatness_quality": "good",
            },
            "saturation_analysis": {
                "saturated_pixels": 0,
                "saturation_fraction": 0.0,
            },
            "processed_at": datetime.now().isoformat(),
        }

        # In a real implementation, this would:
        # 1. Load the preprocessed image
        # 2. Calculate background statistics
        # 3. Measure noise characteristics
        # 4. Detect and count cosmic rays
        # 5. Assess flat field quality
        # 6. Check for saturation
        # 7. Calculate overall quality score

        self.logger.debug(
            f"Quality assessment completed for observation: {observation_id}"
        )
        return quality_result

    async def trigger_differencing(self, observation_id: UUID) -> None:
        """Trigger differencing workflow for an observation.

        Args:
            observation_id: UUID of the observation
        """
        self.logger.info(f"Triggering differencing for observation: {observation_id}")

        try:
            # Import here to avoid circular imports
            from src.adapters.workers.differencing.differencing_workers import (
                difference_observation,
            )

            # Send differencing task to queue
            difference_observation.send(str(observation_id))

            self.logger.info(
                f"Differencing triggered for observation: {observation_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to trigger differencing for {observation_id}: {e}"
            )
            raise


# Create Dramatiq actors
@dramatiq.actor(queue_name="preprocessing")
def preprocess_observation(observation_id: str) -> dict[str, Any]:
    """Dramatiq actor for observation preprocessing."""
    worker = PreprocessingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            worker.preprocess_observation(UUID(observation_id))
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="preprocessing")
def apply_calibration(
    observation_id: str, calibration_frames: dict[str, Any]
) -> dict[str, Any]:
    """Dramatiq actor for calibration application."""
    worker = PreprocessingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            worker.apply_calibration(UUID(observation_id), calibration_frames)
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="preprocessing")
def align_observation(
    observation_id: str, reference_id: str | None = None
) -> dict[str, Any]:
    """Dramatiq actor for observation alignment."""
    worker = PreprocessingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ref_uuid = UUID(reference_id) if reference_id else None
        result = loop.run_until_complete(
            worker.align_observation(UUID(observation_id), ref_uuid)
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="preprocessing")
def assess_quality(observation_id: str) -> dict[str, Any]:
    """Dramatiq actor for quality assessment."""
    worker = PreprocessingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(worker.assess_quality(UUID(observation_id)))
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="preprocessing")
def batch_preprocess_observations(observation_ids: list[str]) -> dict[str, Any]:
    """Dramatiq actor for batch preprocessing."""
    worker = PreprocessingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = []
        errors = []

        for obs_id in observation_ids:
            try:
                result = loop.run_until_complete(
                    worker.preprocess_observation(UUID(obs_id))
                )
                results.append(result)
            except Exception as e:
                error = {
                    "observation_id": obs_id,
                    "error": str(e),
                }
                errors.append(error)

        return {
            "total_processed": len(observation_ids),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }
    finally:
        loop.close()
