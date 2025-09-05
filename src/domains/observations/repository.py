"""Observations repository for API routes."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.observations.crud import ObservationCRUD
from src.domains.observations.models import Observation
from src.domains.observations.schema import ObservationCreate, ObservationListParams


class ObservationRepository:
    """Repository for Observation operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, observation_data: ObservationCreate) -> Observation:
        """Create a new observation."""
        crud = ObservationCRUD()
        return await crud.create(self.db, observation_data)

    async def get_by_id(self, observation_id: str) -> Observation | None:
        """Get observation by ID."""
        from uuid import UUID

        return await ObservationCRUD.get_by_id(self.db, UUID(observation_id))

    async def list(
        self,
        survey: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Observation]:
        """List observations with filtering."""
        params = ObservationListParams(
            survey_id=None,  # TODO: Convert survey name to ID
            status=status,
            limit=limit,
            offset=offset,
        )
        observations, _ = await ObservationCRUD.get_many(self.db, params)
        return observations
