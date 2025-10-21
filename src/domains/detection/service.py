"""Detection service layer for business logic."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.detection.repository import DetectionRepository
from src.domains.detection.schema import DetectionCreate
from src.domains.detection.services.detection_service import (
    DetectionService as ComprehensiveDetectionService,
)


class DetectionService:
    """Service for Detection business logic."""

    def __init__(self, db: AsyncSession) -> None:
        self.repository = DetectionRepository(db)
        self.logger = configure_domain_logger("detection.detection")
        # Initialize comprehensive detection service
        self.comprehensive_service = ComprehensiveDetectionService(db)

    async def create_detection(self, detection_data: DetectionCreate) -> Any:
        """Create a new detection with business logic."""
        self.logger.info(
            f"Creating detection: observation_id={detection_data.observation_id}, score={detection_data.score}"
        )
        try:
            # TODO: Add validation logic, confidence scoring, etc.
            result = await self.repository.create(detection_data)
            self.logger.info(
                f"Successfully created detection: id={result.id}, score={result.score}, status={result.status}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create detection: observation_id={detection_data.observation_id}, error={str(e)}"
            )
            raise

    async def get_detection(self, detection_id: str) -> Any:
        """Get detection by ID."""
        self.logger.debug(f"Retrieving detection by ID: {detection_id}")
        result = await self.repository.get_by_id(detection_id)
        if result:
            self.logger.debug(
                f"Found detection: id={result.id}, score={result.score}, status={result.status}"
            )
        else:
            self.logger.warning(f"Detection not found: id={detection_id}")
        return result

    async def list_detections(
        self,
        observation_id: str | None = None,
        status: str | None = None,
        min_score: float | None = None,
        since: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        """List detections."""
        self.logger.debug(
            f"Listing detections: observation_id={observation_id}, status={status}, min_score={min_score}, limit={limit}"
        )
        result = await self.repository.list(
            observation_id=observation_id,
            status=status,
            min_score=min_score,
            since=since,
            limit=limit,
            offset=offset,
        )
        self.logger.debug(f"Retrieved {len(result)} detections")
        return result

    async def run_inference(
        self, observation_id: str, model_version: str | None = None
    ) -> dict[str, Any]:
        """Run inference on an observation to detect anomalies."""
        self.logger.info(
            f"Starting inference: observation_id={observation_id}, model_version={model_version}"
        )
        try:
            # TODO: Implement actual inference logic
            # This would typically:
            # 1. Load the observation data
            # 2. Preprocess the data
            # 3. Run the ML model
            # 4. Create detection records
            # 5. Return results
            result = {
                "message": f"Inference initiated for observation {observation_id}",
                "observation_id": observation_id,
                "model_version": model_version,
                "status": "queued",
            }
            self.logger.info(
                f"Inference queued successfully: observation_id={observation_id}, model_version={model_version}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to start inference: observation_id={observation_id}, error={str(e)}"
            )
            raise

    async def validate_detection(
        self,
        detection_id: str,
        is_valid: bool,
        label: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Validate a detection (human review)."""
        self.logger.info(
            f"Validating detection: detection_id={detection_id}, is_valid={is_valid}, label={label}"
        )
        try:
            # TODO: Implement validation logic
            # This would typically:
            # 1. Update the detection status
            # 2. Create a validation event
            # 3. Update confidence scores
            result = {
                "message": f"Detection {detection_id} validation recorded",
                "detection_id": detection_id,
                "is_valid": is_valid,
                "label": label,
                "status": "validated",
            }
            self.logger.info(
                f"Detection validation recorded: detection_id={detection_id}, is_valid={is_valid}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to validate detection: detection_id={detection_id}, error={str(e)}"
            )
            raise
