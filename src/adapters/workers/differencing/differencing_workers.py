"""Differencing workers for Dramatiq background processing."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

import dramatiq

from src.core.db.session import AsyncSessionLocal
from src.core.logging import configure_domain_logger
from src.domains.differencing.service import DifferenceRunService
from src.domains.observations.service import ObservationService


class DifferencingWorker:
    """Worker for differencing tasks."""

    def __init__(self):
        self.logger = configure_domain_logger("workers.differencing")

    async def create_difference_image(
        self, observation_id: UUID, reference_id: UUID
    ) -> dict[str, Any]:
        """Create difference image between observation and reference.

        Args:
            observation_id: UUID of the observation
            reference_id: UUID of the reference observation

        Returns:
            Dictionary with differencing results
        """
        self.logger.info(
            f"Creating difference image: obs={observation_id}, ref={reference_id}"
        )

        try:
            async with AsyncSessionLocal() as db:
                observation_service = ObservationService(db)
                differencing_service = DifferenceRunService(db)

                # Get observations
                observation = await observation_service.get_observation_by_id(
                    observation_id
                )
                reference = await observation_service.get_observation_by_id(
                    reference_id
                )

                if not observation:
                    raise ValueError(f"Observation {observation_id} not found")
                if not reference:
                    raise ValueError(f"Reference {reference_id} not found")

                # Mark observation as differencing
                await observation_service.mark_observation_as_differencing(
                    observation_id
                )

                # Apply differencing algorithm
                algorithm_result = await self.apply_differencing_algorithm(
                    observation_id, "zogy"
                )

                # Validate difference image
                validation_result = await self.validate_difference_image(
                    algorithm_result["difference_id"]
                )

                # Extract sources
                extraction_result = await self.extract_sources(
                    algorithm_result["difference_id"]
                )

                # Create difference run record
                from src.domains.differencing.schema import (
                    DifferenceAlgorithm,
                    DifferenceRunCreate,
                )

                difference_create = DifferenceRunCreate(
                    observation_id=observation_id,
                    reference_observation_id=reference_id,
                    algorithm=DifferenceAlgorithm.ZOGY,
                    parameters={},
                )

                difference_run = await differencing_service.create_difference_run(
                    difference_create
                )

                # Mark observation as differenced
                await observation_service.mark_observation_as_differenced(
                    observation_id
                )

                # Trigger detection
                await self.trigger_detection(algorithm_result["difference_id"])

                result = {
                    "observation_id": str(observation_id),
                    "reference_id": str(reference_id),
                    "difference_run_id": str(difference_run.id),
                    "difference_id": algorithm_result["difference_id"],
                    "status": "differenced",
                    "algorithm": algorithm_result,
                    "validation": validation_result,
                    "extraction": extraction_result,
                    "processing_triggered": True,
                }

                self.logger.info(
                    f"Successfully created difference image: {observation_id}"
                )
                return result

        except Exception as e:
            self.logger.error(
                f"Failed to create difference image for {observation_id}: {e}"
            )
            # Mark observation as failed
            try:
                async with AsyncSessionLocal() as db:
                    observation_service = ObservationService(db)
                    await observation_service.mark_observation_as_failed(observation_id)
            except Exception as mark_error:
                self.logger.error(f"Failed to mark observation as failed: {mark_error}")
            raise

    async def apply_differencing_algorithm(
        self, observation_id: UUID, algorithm: str
    ) -> dict[str, Any]:
        """Apply differencing algorithm to create difference image.

        Args:
            observation_id: UUID of the observation
            algorithm: Name of the differencing algorithm to use

        Returns:
            Dictionary with algorithm results
        """
        self.logger.debug(
            f"Applying {algorithm} algorithm for observation: {observation_id}"
        )

        algorithm_result = {
            "observation_id": str(observation_id),
            "algorithm": algorithm,
            "difference_id": f"diff_{observation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "difference_image_path": f"differences/{observation_id}/difference.fits",
            "algorithm_parameters": {
                "kernel_size": 3,
                "noise_threshold": 3.0,
                "correlation_threshold": 0.8,
            },
            "processing_metrics": {
                "processing_time": 45.2,  # seconds
                "memory_usage": 256.0,  # MB
                "cpu_usage": 75.0,  # percentage
            },
            "processed_at": datetime.now().isoformat(),
        }

        # In a real implementation, this would:
        # 1. Load observation and reference images
        # 2. Apply ZOGY algorithm or other differencing method
        # 3. Generate difference image
        # 4. Calculate noise properties
        # 5. Store difference image to cloud storage
        # 6. Return processing metrics

        if algorithm == "zogy":
            algorithm_result["zogy_parameters"] = {
                "psf_observation": "estimated",
                "psf_reference": "estimated",
                "noise_observation": 10.5,
                "noise_reference": 9.8,
                "correlation_matrix": "calculated",
            }
        elif algorithm == "classic":
            algorithm_result["classic_parameters"] = {
                "scaling_factor": 1.0,
                "offset": 0.0,
                "normalization": "mean",
            }

        self.logger.debug(
            f"Algorithm {algorithm} applied for observation: {observation_id}"
        )
        return algorithm_result

    async def validate_difference_image(self, difference_id: str) -> dict[str, Any]:
        """Validate quality of difference image.

        Args:
            difference_id: ID of the difference image

        Returns:
            Dictionary with validation results
        """
        self.logger.debug(f"Validating difference image: {difference_id}")

        validation_result = {
            "difference_id": difference_id,
            "validation_passed": True,
            "quality_metrics": {
                "noise_level": 5.2,
                "dynamic_range": 1000.0,
                "saturation_fraction": 0.0,
                "artifact_count": 0,
            },
            "statistical_metrics": {
                "mean": 0.0,
                "std": 5.2,
                "min": -25.0,
                "max": 30.0,
                "skewness": 0.1,
                "kurtosis": 3.2,
            },
            "spatial_metrics": {
                "spatial_correlation": 0.95,
                "edge_preservation": 0.92,
                "detail_retention": 0.88,
            },
            "processed_at": datetime.now().isoformat(),
        }

        # In a real implementation, this would:
        # 1. Load the difference image
        # 2. Calculate noise statistics
        # 3. Check for artifacts
        # 4. Validate spatial properties
        # 5. Assess overall quality
        # 6. Flag any issues

        self.logger.debug(f"Difference image validation completed: {difference_id}")
        return validation_result

    async def extract_sources(self, difference_id: str) -> dict[str, Any]:
        """Extract sources from difference image.

        Args:
            difference_id: ID of the difference image

        Returns:
            Dictionary with source extraction results
        """
        self.logger.debug(f"Extracting sources from difference image: {difference_id}")

        # Generate mock candidates
        candidates = []
        for i in range(5):  # Mock 5 candidates
            candidate = {
                "candidate_id": f"cand_{difference_id}_{i}",
                "ra": 180.0 + i * 0.1,
                "dec": 30.0 + i * 0.1,
                "x": 100 + i * 50,
                "y": 100 + i * 50,
                "flux": 100.0 + i * 10.0,
                "snr": 5.0 + i * 0.5,
                "candidate_type": "transient" if i % 2 == 0 else "variable",
                "score": 0.8 + i * 0.05,
                "detection_threshold": 3.0,
            }
            candidates.append(candidate)

        extraction_result = {
            "difference_id": difference_id,
            "source_count": len(candidates),
            "candidates": candidates,
            "extraction_parameters": {
                "detection_threshold": 3.0,
                "min_area": 5,
                "max_area": 1000,
                "deblend_threshold": 0.005,
            },
            "quality_metrics": {
                "completeness": 0.95,
                "reliability": 0.90,
                "false_positive_rate": 0.05,
            },
            "processed_at": datetime.now().isoformat(),
        }

        # In a real implementation, this would:
        # 1. Load the difference image
        # 2. Run source detection (SEP/photutils)
        # 3. Apply quality cuts
        # 4. Calculate source properties
        # 5. Classify candidate types
        # 6. Store candidates in database

        self.logger.debug(
            f"Source extraction completed for difference image: {difference_id}"
        )
        return extraction_result

    async def trigger_detection(self, difference_id: str) -> None:
        """Trigger detection workflow for difference image.

        Args:
            difference_id: ID of the difference image
        """
        self.logger.info(f"Triggering detection for difference image: {difference_id}")

        try:
            # Import here to avoid circular imports
            from src.adapters.workers.detection.detection_workers import (
                detect_anomalies,
            )

            # Send detection task to queue
            detect_anomalies.send(difference_id, "unet_v1")

            self.logger.info(
                f"Detection triggered for difference image: {difference_id}"
            )

        except Exception as e:
            self.logger.error(f"Failed to trigger detection for {difference_id}: {e}")
            raise


# Create Dramatiq actors
@dramatiq.actor(queue_name="differencing")
def difference_observation(observation_id: str) -> dict[str, Any]:
    """Dramatiq actor for observation differencing."""
    worker = DifferencingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # For now, use a mock reference ID
        # In real implementation, this would find the best reference
        reference_id = UUID("00000000-0000-0000-0000-000000000001")
        result = loop.run_until_complete(
            worker.create_difference_image(UUID(observation_id), reference_id)
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="differencing")
def create_difference_image(observation_id: str, reference_id: str) -> dict[str, Any]:
    """Dramatiq actor for creating difference images."""
    worker = DifferencingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            worker.create_difference_image(UUID(observation_id), UUID(reference_id))
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="differencing")
def apply_differencing_algorithm(observation_id: str, algorithm: str) -> dict[str, Any]:
    """Dramatiq actor for applying differencing algorithms."""
    worker = DifferencingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            worker.apply_differencing_algorithm(UUID(observation_id), algorithm)
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="differencing")
def validate_difference_image(difference_id: str) -> dict[str, Any]:
    """Dramatiq actor for validating difference images."""
    worker = DifferencingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            worker.validate_difference_image(difference_id)
        )
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="differencing")
def extract_sources(difference_id: str) -> dict[str, Any]:
    """Dramatiq actor for source extraction."""
    worker = DifferencingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(worker.extract_sources(difference_id))
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="differencing")
def batch_difference_observations(observation_ids: list[str]) -> dict[str, Any]:
    """Dramatiq actor for batch differencing."""
    worker = DifferencingWorker()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = []
        errors = []

        for obs_id in observation_ids:
            try:
                # Use mock reference for batch processing
                reference_id = UUID("00000000-0000-0000-0000-000000000001")
                result = loop.run_until_complete(
                    worker.create_difference_image(UUID(obs_id), reference_id)
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
