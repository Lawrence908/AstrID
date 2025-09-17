"""Detection repository for API routes."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.detection.crud import DetectionCRUD
from src.domains.detection.schema import DetectionCreate, DetectionListParams


class DetectionRepository:
    """Repository for Detection operations."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.crud = DetectionCRUD()

    async def create(self, detection_data: DetectionCreate) -> Any:
        """Create a new detection."""
        return await self.crud.create(self.db, detection_data)  # type: ignore[call-arg]

    async def get_by_id(self, detection_id: str) -> Any:
        """Get detection by ID."""
        from uuid import UUID

        return await DetectionCRUD.get_by_id(self.db, UUID(detection_id))  # type: ignore[call-arg]

    async def list(
        self,
        observation_id: str | None = None,
        status: str | None = None,
        min_score: float | None = None,
        since: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        """List detections with filtering."""
        from uuid import UUID

        from src.domains.detection.models import DetectionStatus

        parsed_obs = UUID(observation_id) if observation_id else None
        parsed_status = DetectionStatus(status) if status else None

        params = DetectionListParams(
            observation_id=parsed_obs,
            detection_type=None,  # TODO: Add detection_type parameter if needed
            status=parsed_status,
            min_confidence=min_score,
            limit=limit,
            offset=offset,
        )
        detections, _ = await DetectionCRUD.get_many(self.db, params)
        return detections
