"""Advanced detection result storage and retrieval system."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
from uuid import UUID

from src.core.logging import configure_domain_logger
from src.domains.detection.entities import DetectionResult


class DetectionStorage:
    """Advanced storage system for detection results."""

    def __init__(self, storage_path: str = "storage/detections") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = configure_domain_logger("detection.storage")

        # Create subdirectories
        (self.storage_path / "results").mkdir(exist_ok=True)
        (self.storage_path / "index").mkdir(exist_ok=True)
        (self.storage_path / "archived").mkdir(exist_ok=True)

    async def store_detection_result(self, result: DetectionResult) -> str:
        """Store a detection result with indexing and versioning."""
        detection_id = str(result.detection_id)

        try:
            # Store the full result as pickle for fast loading
            result_path = self.storage_path / "results" / f"{detection_id}.pkl"
            with open(result_path, "wb") as f:
                pickle.dump(result, f)

            # Store metadata as JSON for easy querying
            metadata = {
                "detection_id": detection_id,
                "observation_id": str(result.observation_id),
                "model_version": result.model_version,
                "validation_status": result.validation_status,
                "processing_time": result.processing_time,
                "num_anomalies": len(result.anomalies),
                "avg_confidence": sum(result.confidence_scores)
                / len(result.confidence_scores)
                if result.confidence_scores
                else 0.0,
                "max_confidence": max(result.confidence_scores)
                if result.confidence_scores
                else 0.0,
                "created_at": result.created_at.isoformat(),
                "quality_metrics": result.quality_metrics,
            }

            metadata_path = self.storage_path / "index" / f"{detection_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Update observation index
            await self._update_observation_index(result.observation_id, detection_id)

            # Update confidence index
            await self._update_confidence_index(
                metadata["avg_confidence"], detection_id
            )

            self.logger.info(f"Stored detection result: {detection_id}")
            return detection_id

        except Exception as e:
            self.logger.error(f"Failed to store detection result {detection_id}: {e}")
            raise

    async def retrieve_detection_result(
        self, detection_id: str
    ) -> DetectionResult | None:
        """Retrieve a detection result by ID."""
        try:
            result_path = self.storage_path / "results" / f"{detection_id}.pkl"

            if not result_path.exists():
                self.logger.warning(f"Detection result not found: {detection_id}")
                return None

            with open(result_path, "rb") as f:
                result = pickle.load(f)

            self.logger.debug(f"Retrieved detection result: {detection_id}")
            return result

        except Exception as e:
            self.logger.error(
                f"Failed to retrieve detection result {detection_id}: {e}"
            )
            return None

    async def query_detections_by_observation(
        self, observation_id: UUID
    ) -> list[DetectionResult]:
        """Query detection results by observation ID."""
        try:
            observation_id_str = str(observation_id)
            index_path = self.storage_path / "index" / "observations.json"

            if not index_path.exists():
                return []

            with open(index_path) as f:
                observation_index = json.load(f)

            detection_ids = observation_index.get(observation_id_str, [])
            results = []

            for detection_id in detection_ids:
                result = await self.retrieve_detection_result(detection_id)
                if result:
                    results.append(result)

            self.logger.debug(
                f"Found {len(results)} detections for observation {observation_id}"
            )
            return results

        except Exception as e:
            self.logger.error(
                f"Failed to query detections by observation {observation_id}: {e}"
            )
            return []

    async def query_detections_by_confidence(
        self, confidence_min: float, confidence_max: float
    ) -> list[DetectionResult]:
        """Query detection results by confidence range."""
        try:
            index_path = self.storage_path / "index" / "confidence.json"

            if not index_path.exists():
                return []

            with open(index_path) as f:
                confidence_index = json.load(f)

            results = []

            # Find all detections in confidence range
            for confidence_str, detection_ids in confidence_index.items():
                confidence = float(confidence_str)
                if confidence_min <= confidence <= confidence_max:
                    for detection_id in detection_ids:
                        result = await self.retrieve_detection_result(detection_id)
                        if result:
                            results.append(result)

            self.logger.debug(
                f"Found {len(results)} detections in confidence range [{confidence_min}, {confidence_max}]"
            )
            return results

        except Exception as e:
            self.logger.error(f"Failed to query detections by confidence: {e}")
            return []

    async def archive_detection_result(self, detection_id: str) -> None:
        """Archive a detection result."""
        try:
            # Move result file to archived directory
            result_path = self.storage_path / "results" / f"{detection_id}.pkl"
            archived_path = self.storage_path / "archived" / f"{detection_id}.pkl"

            if result_path.exists():
                result_path.rename(archived_path)

            # Move metadata file
            metadata_path = self.storage_path / "index" / f"{detection_id}.json"
            archived_metadata_path = (
                self.storage_path / "archived" / f"{detection_id}.json"
            )

            if metadata_path.exists():
                metadata_path.rename(archived_metadata_path)

            # Remove from indexes
            await self._remove_from_indexes(detection_id)

            self.logger.info(f"Archived detection result: {detection_id}")

        except Exception as e:
            self.logger.error(f"Failed to archive detection result {detection_id}: {e}")
            raise

    async def get_detection_analytics(self) -> dict[str, Any]:
        """Get analytics and reporting data for detection results."""
        try:
            analytics = {
                "total_detections": 0,
                "total_observations": 0,
                "avg_processing_time": 0.0,
                "confidence_distribution": {},
                "model_versions": {},
                "validation_status_distribution": {},
                "recent_activity": [],
            }

            # Scan all metadata files
            index_dir = self.storage_path / "index"
            metadata_files = list(index_dir.glob("*.json"))

            processing_times = []
            confidences = []
            model_versions = {}
            validation_statuses = {}

            for metadata_file in metadata_files:
                if metadata_file.name in ["observations.json", "confidence.json"]:
                    continue

                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    analytics["total_detections"] += 1
                    processing_times.append(metadata.get("processing_time", 0.0))
                    confidences.append(metadata.get("avg_confidence", 0.0))

                    # Model version distribution
                    model_version = metadata.get("model_version", "unknown")
                    model_versions[model_version] = (
                        model_versions.get(model_version, 0) + 1
                    )

                    # Validation status distribution
                    validation_status = metadata.get("validation_status", "unknown")
                    validation_statuses[validation_status] = (
                        validation_statuses.get(validation_status, 0) + 1
                    )

                    # Recent activity
                    created_at = metadata.get("created_at", "")
                    if created_at:
                        analytics["recent_activity"].append(
                            {
                                "detection_id": metadata["detection_id"],
                                "created_at": created_at,
                                "num_anomalies": metadata.get("num_anomalies", 0),
                            }
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process metadata file {metadata_file}: {e}"
                    )
                    continue

            # Calculate statistics
            if processing_times:
                analytics["avg_processing_time"] = sum(processing_times) / len(
                    processing_times
                )

            # Confidence distribution (binned)
            if confidences:
                confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                for i in range(len(confidence_bins) - 1):
                    bin_name = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
                    count = sum(
                        1
                        for c in confidences
                        if confidence_bins[i] <= c < confidence_bins[i + 1]
                    )
                    analytics["confidence_distribution"][bin_name] = count

            analytics["model_versions"] = model_versions
            analytics["validation_status_distribution"] = validation_statuses

            # Sort recent activity by creation time
            analytics["recent_activity"].sort(
                key=lambda x: x["created_at"], reverse=True
            )
            analytics["recent_activity"] = analytics["recent_activity"][:10]  # Last 10

            # Count unique observations
            observation_index_path = self.storage_path / "index" / "observations.json"
            if observation_index_path.exists():
                with open(observation_index_path) as f:
                    observation_index = json.load(f)
                analytics["total_observations"] = len(observation_index)

            self.logger.info(
                f"Generated analytics for {analytics['total_detections']} detections"
            )
            return analytics

        except Exception as e:
            self.logger.error(f"Failed to generate detection analytics: {e}")
            return {}

    async def _update_observation_index(
        self, observation_id: UUID, detection_id: str
    ) -> None:
        """Update the observation index."""
        try:
            index_path = self.storage_path / "index" / "observations.json"
            observation_id_str = str(observation_id)

            if index_path.exists():
                with open(index_path) as f:
                    observation_index = json.load(f)
            else:
                observation_index = {}

            if observation_id_str not in observation_index:
                observation_index[observation_id_str] = []

            if detection_id not in observation_index[observation_id_str]:
                observation_index[observation_id_str].append(detection_id)

            with open(index_path, "w") as f:
                json.dump(observation_index, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to update observation index: {e}")

    async def _update_confidence_index(
        self, confidence: float, detection_id: str
    ) -> None:
        """Update the confidence index."""
        try:
            index_path = self.storage_path / "index" / "confidence.json"

            # Round confidence to 1 decimal place for binning
            confidence_key = f"{confidence:.1f}"

            if index_path.exists():
                with open(index_path) as f:
                    confidence_index = json.load(f)
            else:
                confidence_index = {}

            if confidence_key not in confidence_index:
                confidence_index[confidence_key] = []

            if detection_id not in confidence_index[confidence_key]:
                confidence_index[confidence_key].append(detection_id)

            with open(index_path, "w") as f:
                json.dump(confidence_index, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to update confidence index: {e}")

    async def _remove_from_indexes(self, detection_id: str) -> None:
        """Remove detection from all indexes."""
        try:
            # Remove from observation index
            observation_index_path = self.storage_path / "index" / "observations.json"
            if observation_index_path.exists():
                with open(observation_index_path) as f:
                    observation_index = json.load(f)

                for _obs_id, detection_ids in observation_index.items():
                    if detection_id in detection_ids:
                        detection_ids.remove(detection_id)

                with open(observation_index_path, "w") as f:
                    json.dump(observation_index, f, indent=2)

            # Remove from confidence index
            confidence_index_path = self.storage_path / "index" / "confidence.json"
            if confidence_index_path.exists():
                with open(confidence_index_path) as f:
                    confidence_index = json.load(f)

                for _conf_key, detection_ids in confidence_index.items():
                    if detection_id in detection_ids:
                        detection_ids.remove(detection_id)

                with open(confidence_index_path, "w") as f:
                    json.dump(confidence_index, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to remove detection from indexes: {e}")
