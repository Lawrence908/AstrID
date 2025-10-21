"""Comprehensive detection service layer for anomaly detection pipeline."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.detection.entities import (
    Anomaly,
    DetectionConfig,
    DetectionResult,
    Model,
    Observation,
)
from src.domains.detection.metrics.detection_metrics import DetectionMetrics
from src.domains.detection.repository import DetectionRepository
from src.domains.detection.services.model_inference import ModelInferenceService
from src.domains.detection.storage.detection_storage import DetectionStorage
from src.domains.detection.validators.detection_validator import DetectionValidator


class DetectionService:
    """Comprehensive detection service for anomaly detection pipeline."""

    def __init__(
        self,
        db: AsyncSession,
        config: DetectionConfig | None = None,
        validator: DetectionValidator | None = None,
        storage: DetectionStorage | None = None,
        metrics: DetectionMetrics | None = None,
    ) -> None:
        self.db = db
        self.repository = DetectionRepository(db)
        self.config = config or DetectionConfig(
            model_path="models/unet_astronomical",
            confidence_threshold=0.5,
            batch_size=16,
            max_detections_per_image=100,
            validation_enabled=True,
            false_positive_filtering=True,
            quality_assessment=True,
            caching_enabled=True,
        )
        self.validator = validator or DetectionValidator()
        self.storage = storage or DetectionStorage()
        self.metrics = metrics or DetectionMetrics()
        self.model_inference = ModelInferenceService()
        self.logger = configure_domain_logger("detection.detection_service")

    async def process_observation(self, observation: Observation) -> DetectionResult:
        """Process a single observation and return detection results."""
        self.logger.info(f"Processing observation: {observation.observation_id}")

        start_time = time.time()
        detection_id = uuid4()

        try:
            # Load model for inference
            model = await self._load_model()

            # Run anomaly detection
            anomalies = await self.detect_anomalies(observation.image_data, model)

            # Validate detections if enabled
            if self.config.validation_enabled:
                anomalies = await self.validate_detections(
                    anomalies, observation.image_data
                )

            # Calculate confidence scores
            confidence_scores = await self.calculate_detection_confidence(anomalies)

            # Filter by confidence threshold
            filtered_anomalies = await self.filter_detections_by_confidence(
                anomalies, self.config.confidence_threshold
            )

            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                filtered_anomalies, observation
            )

            processing_time = time.time() - start_time

            # Create detection result
            result = DetectionResult(
                detection_id=detection_id,
                observation_id=observation.id,
                anomalies=filtered_anomalies,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                model_version=model.version,
                validation_status="completed"
                if self.config.validation_enabled
                else "skipped",
                quality_metrics=quality_metrics,
                created_at=datetime.now(),
            )

            # Store result if caching enabled
            if self.config.caching_enabled:
                await self.storage.store_detection_result(result)

            self.logger.info(
                f"Successfully processed observation {observation.observation_id}: "
                f"{len(filtered_anomalies)} anomalies detected in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Failed to process observation {observation.observation_id}: {e}"
            )
            raise

    async def detect_anomalies(
        self, image_data: np.ndarray | None, model: Model
    ) -> list[Anomaly]:
        """Detect anomalies in image data using the provided model."""
        self.logger.debug("Running anomaly detection on image data")

        if image_data is None:
            self.logger.warning("No image data provided for anomaly detection")
            return []

        try:
            # Run model inference
            inference_result = self.model_inference.infer_batch([image_data])

            # Extract anomalies from inference results
            anomalies = []
            result = inference_result["results"][0]
            probability_map = result["probability"]
            mask = result["mask"]

            # Find connected components in the mask
            from scipy import ndimage

            labeled_array, num_features = ndimage.label(mask)

            for i in range(1, int(num_features) + 1):
                # Get coordinates of this component
                coords = np.where(labeled_array == i)
                if len(coords[0]) == 0:
                    continue

                # Calculate center coordinates
                y_center = int(np.mean(coords[0]))
                x_center = int(np.mean(coords[1]))

                # Calculate confidence as max probability in this region
                region_mask = labeled_array == i
                confidence = float(np.max(probability_map[region_mask]))

                # Calculate size (pixel area)
                size = float(np.sum(region_mask))

                # Convert pixel coordinates to world coordinates (simplified)
                ra, dec = self._pixel_to_world_coordinates(
                    x_center, y_center, image_data.shape
                )

                # Create anomaly
                anomaly = Anomaly(
                    anomaly_id=uuid4(),
                    coordinates=(float(x_center), float(y_center)),
                    world_coordinates=(ra, dec),
                    confidence=confidence,
                    size=size,
                    magnitude=self._calculate_magnitude(confidence, size),
                    classification=self._classify_anomaly(confidence, size),
                    metadata={
                        "region_mask": region_mask.tolist(),
                        "probability_map": probability_map[region_mask].tolist(),
                        "inference_time_ms": inference_result["metrics"]["latency_ms"],
                    },
                    validation_flags=[],
                )

                anomalies.append(anomaly)

            self.logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies

        except Exception as e:
            self.logger.error(f"Failed to detect anomalies: {e}")
            raise

    async def validate_detections(
        self, detections: list[Anomaly], image: np.ndarray | None
    ) -> list[Anomaly]:
        """Validate detected anomalies using advanced validation logic."""
        self.logger.debug(f"Validating {len(detections)} detections")

        if not self.config.validation_enabled:
            return detections

        validated_detections = []

        for detection in detections:
            try:
                # Validate detection quality
                if not await self.validator.validate_detection_quality(
                    detection, image
                ):
                    detection.validation_flags.append("low_quality")
                    continue

                # Validate coordinate bounds
                if not await self.validator.validate_coordinate_bounds(
                    detection, image.shape
                ):
                    detection.validation_flags.append("out_of_bounds")
                    continue

                # Assess reliability
                reliability = await self.validator.assess_detection_reliability(
                    detection
                )
                if reliability < 0.3:  # Low reliability threshold
                    detection.validation_flags.append("low_reliability")
                    continue

                validated_detections.append(detection)

            except Exception as e:
                self.logger.warning(
                    f"Validation failed for detection {detection.anomaly_id}: {e}"
                )
                detection.validation_flags.append("validation_error")
                continue

        # Check for duplicates
        validated_detections = await self.validator.check_detection_duplicates(
            validated_detections
        )

        # Filter false positives
        if self.config.false_positive_filtering:
            validated_detections = await self.validator.filter_false_positives(
                validated_detections
            )

        self.logger.info(
            f"Validated {len(validated_detections)} out of {len(detections)} detections"
        )
        return validated_detections

    async def calculate_detection_confidence(
        self, detections: list[Anomaly]
    ) -> list[float]:
        """Calculate confidence scores for detections."""
        self.logger.debug(f"Calculating confidence for {len(detections)} detections")

        confidence_scores = []
        for detection in detections:
            # Base confidence from model
            base_confidence = detection.confidence

            # Adjust based on size (larger detections might be more reliable)
            size_factor = min(
                1.0, detection.size / 100.0
            )  # Normalize to reasonable range

            # Adjust based on validation flags
            validation_factor = 1.0
            if "low_quality" in detection.validation_flags:
                validation_factor *= 0.7
            if "low_reliability" in detection.validation_flags:
                validation_factor *= 0.5

            # Final confidence score
            final_confidence = base_confidence * size_factor * validation_factor
            confidence_scores.append(min(1.0, max(0.0, final_confidence)))

        return confidence_scores

    async def filter_detections_by_confidence(
        self, detections: list[Anomaly], threshold: float
    ) -> list[Anomaly]:
        """Filter detections by confidence threshold."""
        self.logger.debug(f"Filtering detections with threshold {threshold}")

        filtered = [d for d in detections if d.confidence >= threshold]

        # Limit to max detections per image
        if len(filtered) > self.config.max_detections_per_image:
            # Sort by confidence and take top N
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[: self.config.max_detections_per_image]

        self.logger.info(
            f"Filtered to {len(filtered)} detections (threshold: {threshold})"
        )
        return filtered

    async def process_batch_observations(
        self, observations: list[Observation]
    ) -> list[DetectionResult]:
        """Process multiple observations in batch."""
        self.logger.info(f"Processing batch of {len(observations)} observations")

        results = []
        for i, observation in enumerate(observations):
            try:
                result = await self.process_observation(observation)
                results.append(result)

                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(
                        f"Processed {i + 1}/{len(observations)} observations"
                    )

            except Exception as e:
                self.logger.error(
                    f"Failed to process observation {observation.observation_id}: {e}"
                )
                # Continue with next observation
                continue

        self.logger.info(
            f"Batch processing completed: {len(results)}/{len(observations)} successful"
        )
        return results

    async def get_detection_result(self, detection_id: UUID) -> DetectionResult | None:
        """Retrieve a detection result by ID."""
        if not self.config.caching_enabled:
            return None

        return await self.storage.retrieve_detection_result(str(detection_id))

    async def query_detections_by_observation(
        self, observation_id: UUID
    ) -> list[DetectionResult]:
        """Query detection results by observation ID."""
        if not self.config.caching_enabled:
            return []

        return await self.storage.query_detections_by_observation(observation_id)

    async def query_detections_by_confidence(
        self, confidence_min: float, confidence_max: float
    ) -> list[DetectionResult]:
        """Query detection results by confidence range."""
        if not self.config.caching_enabled:
            return []

        return await self.storage.query_detections_by_confidence(
            confidence_min, confidence_max
        )

    async def archive_detection_result(self, detection_id: UUID) -> None:
        """Archive a detection result."""
        if not self.config.caching_enabled:
            return

        await self.storage.archive_detection_result(str(detection_id))

    async def get_detection_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive detection metrics summary."""
        return await self.metrics.get_detection_metrics_summary()

    async def _load_model(self) -> Model:
        """Load the ML model for inference."""
        # This would typically load from model registry or file system
        # For now, return a mock model
        return Model(
            id=uuid4(),
            name="unet_astronomical",
            version="1.0.0",
            model_type="unet",
            model_path=self.config.model_path,
            confidence_threshold=self.config.confidence_threshold,
        )

    async def _calculate_quality_metrics(
        self, anomalies: list[Anomaly], observation: Observation
    ) -> dict[str, Any]:
        """Calculate quality metrics for the detection result."""
        return {
            "total_anomalies": len(anomalies),
            "avg_confidence": np.mean([a.confidence for a in anomalies])
            if anomalies
            else 0.0,
            "max_confidence": max([a.confidence for a in anomalies])
            if anomalies
            else 0.0,
            "total_area": sum([a.size for a in anomalies]),
            "observation_ra": observation.ra,
            "observation_dec": observation.dec,
            "filter_band": observation.filter_band,
            "exposure_time": observation.exposure_time,
        }

    def _pixel_to_world_coordinates(
        self, x: int, y: int, image_shape: tuple[int, ...]
    ) -> tuple[float, float]:
        """Convert pixel coordinates to world coordinates (simplified)."""
        # This is a simplified conversion - in practice, you'd use WCS
        height, width = image_shape[:2]

        # Simple linear mapping (this should use proper WCS in production)
        ra = (x / width) * 360.0
        dec = (y / height) * 180.0 - 90.0

        return (ra, dec)

    def _calculate_magnitude(self, confidence: float, size: float) -> float:
        """Calculate apparent magnitude from confidence and size."""
        # Simplified magnitude calculation
        # In practice, this would use proper photometry
        base_magnitude = 20.0  # Base magnitude
        confidence_factor = (1.0 - confidence) * 5.0  # Lower confidence = brighter
        size_factor = np.log10(max(1.0, size)) * 2.0  # Larger size = brighter

        return base_magnitude - confidence_factor - size_factor

    def _classify_anomaly(self, confidence: float, size: float) -> str:
        """Classify anomaly type based on confidence and size."""
        if confidence > 0.8 and size > 50:
            return "supernova"
        elif confidence > 0.6 and size > 20:
            return "variable"
        elif confidence > 0.4:
            return "transient"
        else:
            return "unknown"
