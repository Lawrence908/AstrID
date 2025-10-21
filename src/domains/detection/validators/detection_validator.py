"""Advanced detection validation logic for anomaly detection pipeline."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from src.core.logging import configure_domain_logger
from src.domains.detection.entities import Anomaly


class DetectionValidator:
    """Advanced validation logic for detected anomalies."""

    def __init__(self) -> None:
        self.logger = configure_domain_logger("detection.validator")

    async def validate_detection_quality(
        self, detection: Anomaly, image: np.ndarray
    ) -> bool:
        """Validate the quality of a single detection."""
        try:
            # Check if detection has reasonable confidence
            if detection.confidence < 0.1:
                return False

            # Check if detection size is reasonable
            if detection.size < 1.0 or detection.size > 10000.0:
                return False

            # Check if detection is within image bounds
            x, y = detection.coordinates
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                return False

            # Check if detection has reasonable magnitude
            if detection.magnitude < 10.0 or detection.magnitude > 30.0:
                return False

            # Additional quality checks based on image data
            if image is not None:
                # Check if detection is in a reasonable region of the image
                # (not too close to edges, not in saturated regions, etc.)
                if not self._check_detection_region_quality(detection, image):
                    return False

            return True

        except Exception as e:
            self.logger.warning(
                f"Quality validation failed for detection {detection.anomaly_id}: {e}"
            )
            return False

    async def check_detection_duplicates(
        self, detections: list[Anomaly]
    ) -> list[Anomaly]:
        """Check for and remove duplicate detections."""
        if len(detections) <= 1:
            return detections

        self.logger.debug(f"Checking {len(detections)} detections for duplicates")

        # Calculate pairwise distances between detections
        coordinates = np.array([d.coordinates for d in detections])
        distances = cdist(coordinates, coordinates)

        # Find duplicates (detections within 5 pixels of each other)
        duplicate_threshold = 5.0
        duplicates = set()

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if distances[i, j] < duplicate_threshold:
                    # Keep the detection with higher confidence
                    if detections[i].confidence >= detections[j].confidence:
                        duplicates.add(j)
                    else:
                        duplicates.add(i)

        # Remove duplicates
        filtered_detections = [
            d for i, d in enumerate(detections) if i not in duplicates
        ]

        self.logger.info(f"Removed {len(duplicates)} duplicate detections")
        return filtered_detections

    async def validate_coordinate_bounds(
        self, detection: Anomaly, image_shape: tuple[int, ...]
    ) -> bool:
        """Validate that detection coordinates are within image bounds."""
        x, y = detection.coordinates

        # Check pixel coordinates
        if x < 0 or y < 0 or x >= image_shape[1] or y >= image_shape[0]:
            return False

        # Check world coordinates (RA should be 0-360, Dec should be -90 to +90)
        ra, dec = detection.world_coordinates
        if not (0 <= ra <= 360) or not (-90 <= dec <= 90):
            return False

        return True

    async def assess_detection_reliability(self, detection: Anomaly) -> float:
        """Assess the reliability of a detection based on multiple factors."""
        reliability_factors = []

        # Confidence factor (0-1)
        confidence_factor = min(1.0, detection.confidence)
        reliability_factors.append(confidence_factor)

        # Size factor (optimal size range)
        size_factor = self._calculate_size_factor(detection.size)
        reliability_factors.append(size_factor)

        # Magnitude factor (reasonable magnitude range)
        magnitude_factor = self._calculate_magnitude_factor(detection.magnitude)
        reliability_factors.append(magnitude_factor)

        # Classification factor (known types are more reliable)
        classification_factor = self._calculate_classification_factor(
            detection.classification
        )
        reliability_factors.append(classification_factor)

        # Calculate weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Confidence is most important
        reliability = sum(
            w * f for w, f in zip(weights, reliability_factors, strict=False)
        )

        return min(1.0, max(0.0, reliability))

    async def filter_false_positives(self, detections: list[Anomaly]) -> list[Anomaly]:
        """Filter out likely false positive detections."""
        if not detections:
            return detections

        self.logger.debug(
            f"Filtering false positives from {len(detections)} detections"
        )

        filtered_detections = []

        for detection in detections:
            # Apply false positive filters
            if await self._is_likely_false_positive(detection):
                detection.validation_flags.append("false_positive")
                continue

            filtered_detections.append(detection)

        self.logger.info(
            f"Filtered to {len(filtered_detections)} detections after false positive removal"
        )
        return filtered_detections

    def _check_detection_region_quality(
        self, detection: Anomaly, image: np.ndarray
    ) -> bool:
        """Check if the detection region has good quality characteristics."""
        try:
            x, y = detection.coordinates
            x, y = int(x), int(y)

            # Define region around detection
            region_size = 10
            x_min = max(0, x - region_size)
            x_max = min(image.shape[1], x + region_size)
            y_min = max(0, y - region_size)
            y_max = min(image.shape[0], y + region_size)

            region = image[y_min:y_max, x_min:x_max]

            if region.size == 0:
                return False

            # Check for saturation (too bright)
            if np.max(region) > 0.95:  # Assuming normalized image
                return False

            # Check for too much noise (high standard deviation)
            if np.std(region) > 0.5:  # Assuming normalized image
                return False

            # Check for edge effects (detection too close to image edge)
            edge_threshold = 5
            if (
                x < edge_threshold
                or y < edge_threshold
                or x > image.shape[1] - edge_threshold
                or y > image.shape[0] - edge_threshold
            ):
                return False

            return True

        except Exception:
            return False

    def _calculate_size_factor(self, size: float) -> float:
        """Calculate size factor for reliability assessment."""
        # Optimal size range is 5-100 pixels
        if 5 <= size <= 100:
            return 1.0
        elif size < 5:
            # Too small, linearly decrease
            return size / 5.0
        else:
            # Too large, decrease but not as severely
            return max(0.5, 100.0 / size)

    def _calculate_magnitude_factor(self, magnitude: float) -> float:
        """Calculate magnitude factor for reliability assessment."""
        # Reasonable magnitude range is 15-25
        if 15 <= magnitude <= 25:
            return 1.0
        elif magnitude < 15:
            # Too bright, might be artifact
            return max(0.3, magnitude / 15.0)
        else:
            # Too faint, might be noise
            return max(0.3, (30.0 - magnitude) / 5.0)

    def _calculate_classification_factor(self, classification: str) -> float:
        """Calculate classification factor for reliability assessment."""
        classification_scores = {
            "supernova": 1.0,
            "variable": 0.8,
            "transient": 0.6,
            "unknown": 0.4,
            "artifact": 0.1,
        }
        return classification_scores.get(classification, 0.5)

    async def _is_likely_false_positive(self, detection: Anomaly) -> bool:
        """Check if a detection is likely a false positive."""
        # Very low confidence
        if detection.confidence < 0.2:
            return True

        # Very small size (likely noise)
        if detection.size < 2.0:
            return True

        # Very large size (likely artifact)
        if detection.size > 1000.0:
            return True

        # Unreasonable magnitude
        if detection.magnitude < 10.0 or detection.magnitude > 30.0:
            return True

        # Classification as artifact
        if detection.classification == "artifact":
            return True

        return False
