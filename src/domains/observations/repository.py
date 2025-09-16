"""Observations repository for API routes."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.observations.crud import ObservationCRUD
from src.domains.observations.models import Observation, ObservationStatus
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
        # Convert string status to ObservationStatus enum if provided
        status_enum = None
        if status:
            try:
                status_enum = ObservationStatus(status)
            except ValueError:
                pass  # Invalid status, will be ignored

        params = ObservationListParams(
            survey_id=None,  # TODO: Convert survey name to ID
            status=status_enum,
            limit=limit,
            offset=offset,
        )
        observations, _ = await ObservationCRUD.get_many(self.db, params)
        return observations

    async def get_by_survey(self, survey_id: UUID) -> list[Observation]:
        """Get observations by survey ID.

        Args:
            survey_id: Survey UUID

        Returns:
            list[Observation]: List of observations for the survey
        """
        params = ObservationListParams(survey_id=survey_id, limit=1000)
        observations, _ = await ObservationCRUD.get_many(self.db, params)
        return observations

    async def get_by_status(self, status: ObservationStatus) -> list[Observation]:
        """Get observations by processing status.

        Args:
            status: Observation status

        Returns:
            list[Observation]: List of observations with the given status
        """
        params = ObservationListParams(status=status, limit=1000)
        observations, _ = await ObservationCRUD.get_many(self.db, params)
        return observations

    async def update_status(
        self, observation_id: UUID, status: ObservationStatus
    ) -> None:
        """Update observation status.

        Args:
            observation_id: Observation UUID
            status: New observation status
        """
        await ObservationCRUD.update_status(self.db, observation_id, status.value)

    async def get_by_coordinates(
        self, ra: float, dec: float, radius: float
    ) -> list[Observation]:
        """Get observations within a circular region.

        Args:
            ra: Right Ascension center in degrees
            dec: Declination center in degrees
            radius: Search radius in degrees

        Returns:
            list[Observation]: List of observations within the region
        """
        # Calculate bounding box for the search region
        ra_min = max(0, ra - radius)
        ra_max = min(360, ra + radius)
        dec_min = max(-90, dec - radius)
        dec_max = min(90, dec + radius)

        return await ObservationCRUD.get_by_spatial_region(
            self.db, ra_min, ra_max, dec_min, dec_max, limit=1000
        )

    async def get_observations_for_processing(
        self, status: ObservationStatus = ObservationStatus.INGESTED, limit: int = 100
    ) -> list[Observation]:
        """Get observations ready for processing.

        Args:
            status: Status of observations to retrieve
            limit: Maximum number of observations to return

        Returns:
            list[Observation]: List of observations ready for processing
        """
        params = ObservationListParams(status=status, limit=limit)
        observations, _ = await ObservationCRUD.get_many(self.db, params)
        return observations

    async def count_by_survey(self, survey_id: UUID) -> int:
        """Count observations for a survey.

        Args:
            survey_id: Survey UUID

        Returns:
            int: Number of observations for the survey
        """
        params = ObservationListParams(survey_id=survey_id, limit=1)
        _, total = await ObservationCRUD.get_many(self.db, params)
        return total

    async def count_by_status(self, status: ObservationStatus) -> int:
        """Count observations by status.

        Args:
            status: Observation status

        Returns:
            int: Number of observations with the given status
        """
        params = ObservationListParams(status=status, limit=1)
        _, total = await ObservationCRUD.get_many(self.db, params)
        return total

    async def get_recent_observations(
        self, hours: int = 24, limit: int = 100
    ) -> list[Observation]:
        """Get recent observations within specified hours.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of observations to return

        Returns:
            list[Observation]: List of recent observations
        """
        from datetime import datetime, timedelta

        date_from = datetime.now() - timedelta(hours=hours)
        params = ObservationListParams(date_from=date_from, limit=limit)
        observations, _ = await ObservationCRUD.get_many(self.db, params)
        return observations
