"""Comprehensive detection metrics calculation and monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.core.logging import configure_domain_logger
from src.domains.detection.entities import Anomaly, DetectionResult


class DetectionMetrics:
    """Comprehensive metrics calculation for detection pipeline."""

    def __init__(self) -> None:
        self.logger = configure_domain_logger("detection.metrics")
        self.metrics_history = []

    async def calculate_detection_precision(
        self, detections: list[Anomaly], ground_truth: list[Anomaly]
    ) -> float:
        """Calculate precision for detections against ground truth."""
        if not detections:
            return 0.0

        try:
            # Match detections to ground truth based on proximity
            matched_detections = await self._match_detections_to_ground_truth(
                detections, ground_truth
            )

            # Calculate precision: true positives / (true positives + false positives)
            true_positives = len(matched_detections)
            false_positives = len(detections) - true_positives

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )

            self.logger.debug(
                f"Calculated precision: {precision:.3f} ({true_positives} TP, {false_positives} FP)"
            )
            return precision

        except Exception as e:
            self.logger.error(f"Failed to calculate precision: {e}")
            return 0.0

    async def calculate_detection_recall(
        self, detections: list[Anomaly], ground_truth: list[Anomaly]
    ) -> float:
        """Calculate recall for detections against ground truth."""
        if not ground_truth:
            return 1.0 if not detections else 0.0

        try:
            # Match ground truth to detections based on proximity
            matched_ground_truth = await self._match_ground_truth_to_detections(
                ground_truth, detections
            )

            # Calculate recall: true positives / (true positives + false negatives)
            true_positives = len(matched_ground_truth)
            false_negatives = len(ground_truth) - true_positives

            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0.0
            )

            self.logger.debug(
                f"Calculated recall: {recall:.3f} ({true_positives} TP, {false_negatives} FN)"
            )
            return recall

        except Exception as e:
            self.logger.error(f"Failed to calculate recall: {e}")
            return 0.0

    async def calculate_detection_f1_score(
        self, detections: list[Anomaly], ground_truth: list[Anomaly]
    ) -> float:
        """Calculate F1 score for detections against ground truth."""
        try:
            precision = await self.calculate_detection_precision(
                detections, ground_truth
            )
            recall = await self.calculate_detection_recall(detections, ground_truth)

            if precision + recall == 0:
                return 0.0

            f1_score = 2 * (precision * recall) / (precision + recall)

            self.logger.debug(f"Calculated F1 score: {f1_score:.3f}")
            return f1_score

        except Exception as e:
            self.logger.error(f"Failed to calculate F1 score: {e}")
            return 0.0

    async def calculate_detection_auc(
        self, detections: list[Anomaly], ground_truth: list[Anomaly]
    ) -> float:
        """Calculate Area Under Curve (AUC) for detection confidence scores."""
        if not detections:
            return 0.0

        try:
            # Create binary labels (1 for true positive, 0 for false positive)
            labels = []
            scores = []

            # Match detections to ground truth
            matched_detections = await self._match_detections_to_ground_truth(
                detections, ground_truth
            )
            matched_detection_ids = {d.anomaly_id for d in matched_detections}

            for detection in detections:
                labels.append(1 if detection.anomaly_id in matched_detection_ids else 0)
                scores.append(detection.confidence)

            if len(set(labels)) < 2:  # Need both positive and negative examples
                return 0.5  # Random performance

            # Calculate AUC using sklearn
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(labels, scores)

            self.logger.debug(f"Calculated AUC: {auc:.3f}")
            return auc

        except Exception as e:
            self.logger.error(f"Failed to calculate AUC: {e}")
            return 0.0

    async def calculate_detection_latency(
        self, start_time: datetime, end_time: datetime
    ) -> float:
        """Calculate detection processing latency in seconds."""
        try:
            latency = (end_time - start_time).total_seconds()
            self.logger.debug(f"Calculated latency: {latency:.3f}s")
            return latency

        except Exception as e:
            self.logger.error(f"Failed to calculate latency: {e}")
            return 0.0

    async def calculate_throughput_metrics(
        self, results: list[DetectionResult], time_window_hours: int = 24
    ) -> dict[str, Any]:
        """Calculate throughput metrics for a time window."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

            # Filter results within time window
            recent_results = [r for r in results if r.created_at >= cutoff_time]

            if not recent_results:
                return {
                    "observations_per_hour": 0.0,
                    "detections_per_hour": 0.0,
                    "avg_processing_time": 0.0,
                    "total_observations": 0,
                    "total_detections": 0,
                }

            # Calculate metrics
            total_observations = len(recent_results)
            total_detections = sum(len(r.anomalies) for r in recent_results)
            total_processing_time = sum(r.processing_time for r in recent_results)

            observations_per_hour = total_observations / time_window_hours
            detections_per_hour = total_detections / time_window_hours
            avg_processing_time = (
                total_processing_time / total_observations
                if total_observations > 0
                else 0.0
            )

            metrics = {
                "observations_per_hour": observations_per_hour,
                "detections_per_hour": detections_per_hour,
                "avg_processing_time": avg_processing_time,
                "total_observations": total_observations,
                "total_detections": total_detections,
            }

            self.logger.debug(f"Calculated throughput metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate throughput metrics: {e}")
            return {}

    async def calculate_quality_metrics(
        self, results: list[DetectionResult]
    ) -> dict[str, Any]:
        """Calculate quality metrics for detection results."""
        try:
            if not results:
                return {}

            all_anomalies = []
            processing_times = []
            confidence_scores = []

            for result in results:
                all_anomalies.extend(result.anomalies)
                processing_times.append(result.processing_time)
                confidence_scores.extend(result.confidence_scores)

            if not all_anomalies:
                return {}

            # Calculate quality metrics
            quality_metrics = {
                "total_detections": len(all_anomalies),
                "avg_confidence": np.mean(confidence_scores)
                if confidence_scores
                else 0.0,
                "std_confidence": np.std(confidence_scores)
                if confidence_scores
                else 0.0,
                "min_confidence": np.min(confidence_scores)
                if confidence_scores
                else 0.0,
                "max_confidence": np.max(confidence_scores)
                if confidence_scores
                else 0.0,
                "avg_processing_time": np.mean(processing_times)
                if processing_times
                else 0.0,
                "std_processing_time": np.std(processing_times)
                if processing_times
                else 0.0,
                "detections_per_observation": len(all_anomalies) / len(results),
                "confidence_distribution": self._calculate_confidence_distribution(
                    confidence_scores
                ),
                "size_distribution": self._calculate_size_distribution(all_anomalies),
                "classification_distribution": self._calculate_classification_distribution(
                    all_anomalies
                ),
            }

            self.logger.debug(f"Calculated quality metrics for {len(results)} results")
            return quality_metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate quality metrics: {e}")
            return {}

    async def get_detection_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive detection metrics summary."""
        try:
            # This would typically query from a database or storage system
            # For now, return a mock summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_detections": 0,
                "total_observations": 0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1_score": 0.0,
                "avg_auc": 0.0,
                "avg_processing_time": 0.0,
                "throughput_24h": {
                    "observations_per_hour": 0.0,
                    "detections_per_hour": 0.0,
                },
                "quality_metrics": {
                    "avg_confidence": 0.0,
                    "confidence_std": 0.0,
                    "detections_per_observation": 0.0,
                },
                "model_performance": {
                    "active_models": [],
                    "best_performing_model": None,
                },
                "recent_trends": {
                    "detection_rate_trend": "stable",
                    "confidence_trend": "stable",
                    "processing_time_trend": "stable",
                },
            }

            self.logger.info("Generated detection metrics summary")
            return summary

        except Exception as e:
            self.logger.error(f"Failed to generate metrics summary: {e}")
            return {}

    async def _match_detections_to_ground_truth(
        self, detections: list[Anomaly], ground_truth: list[Anomaly]
    ) -> list[Anomaly]:
        """Match detections to ground truth based on proximity."""
        if not detections or not ground_truth:
            return []

        matched_detections = []
        used_ground_truth = set()

        # Calculate distance matrix
        detection_coords = np.array([d.coordinates for d in detections])
        ground_truth_coords = np.array([gt.coordinates for gt in ground_truth])

        from scipy.spatial.distance import cdist

        distances = cdist(detection_coords, ground_truth_coords)

        # Match each detection to closest ground truth within threshold
        threshold = 10.0  # pixels
        for i, detection in enumerate(detections):
            min_distance_idx = np.argmin(distances[i])
            min_distance = distances[i, min_distance_idx]

            if min_distance <= threshold and min_distance_idx not in used_ground_truth:
                matched_detections.append(detection)
                used_ground_truth.add(min_distance_idx)

        return matched_detections

    async def _match_ground_truth_to_detections(
        self, ground_truth: list[Anomaly], detections: list[Anomaly]
    ) -> list[Anomaly]:
        """Match ground truth to detections based on proximity."""
        if not ground_truth or not detections:
            return []

        matched_ground_truth = []
        used_detections = set()

        # Calculate distance matrix
        ground_truth_coords = np.array([gt.coordinates for gt in ground_truth])
        detection_coords = np.array([d.coordinates for d in detections])

        from scipy.spatial.distance import cdist

        distances = cdist(ground_truth_coords, detection_coords)

        # Match each ground truth to closest detection within threshold
        threshold = 10.0  # pixels
        for i, gt in enumerate(ground_truth):
            min_distance_idx = np.argmin(distances[i])
            min_distance = distances[i, min_distance_idx]

            if min_distance <= threshold and min_distance_idx not in used_detections:
                matched_ground_truth.append(gt)
                used_detections.add(min_distance_idx)

        return matched_ground_truth

    def _calculate_confidence_distribution(
        self, confidence_scores: list[float]
    ) -> dict[str, int]:
        """Calculate confidence score distribution."""
        if not confidence_scores:
            return {}

        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        distribution = {}

        for i in range(len(bins) - 1):
            bin_name = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
            count = sum(
                1 for score in confidence_scores if bins[i] <= score < bins[i + 1]
            )
            distribution[bin_name] = count

        return distribution

    def _calculate_size_distribution(self, anomalies: list[Anomaly]) -> dict[str, int]:
        """Calculate anomaly size distribution."""
        if not anomalies:
            return {}

        sizes = [a.size for a in anomalies]
        bins = [0, 5, 10, 25, 50, 100, float("inf")]
        distribution = {}

        for i in range(len(bins) - 1):
            bin_name = f"{bins[i]}-{bins[i+1] if bins[i+1] != float('inf') else '∞'}"
            count = sum(1 for size in sizes if bins[i] <= size < bins[i + 1])
            distribution[bin_name] = count

        return distribution

    def _calculate_classification_distribution(
        self, anomalies: list[Anomaly]
    ) -> dict[str, int]:
        """Calculate anomaly classification distribution."""
        if not anomalies:
            return {}

        classifications = [a.classification for a in anomalies]
        distribution = {}

        for classification in classifications:
            distribution[classification] = distribution.get(classification, 0) + 1

        return distribution
