"""Differencing repository for API routes."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.differencing.crud import CandidateCRUD, DifferenceRunCRUD
from src.domains.differencing.schema import (
    CandidateCreate,
    CandidateListParams,
    DifferenceRunCreate,
    DifferenceRunListParams,
)


class DifferenceRunRepository:
    """Repository for DifferenceRun operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, run_data: DifferenceRunCreate):
        """Create a new difference run."""
        return await DifferenceRunCRUD.create(self.db, run_data)

    async def get_by_id(self, run_id: str):
        """Get difference run by ID."""
        from uuid import UUID

        return await DifferenceRunCRUD.get_by_id(self.db, UUID(run_id))

    async def list(
        self,
        observation_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List difference runs with filtering."""
        params = DifferenceRunListParams(
            observation_id=observation_id, status=status, limit=limit, offset=offset
        )
        runs, _ = await DifferenceRunCRUD.get_many(self.db, params)
        return runs


class CandidateRepository:
    """Repository for Candidate operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, candidate_data: CandidateCreate):
        """Create a new candidate."""
        return await CandidateCRUD.create(self.db, candidate_data)

    async def get_by_id(self, candidate_id: str):
        """Get candidate by ID."""
        from uuid import UUID

        return await CandidateCRUD.get_by_id(self.db, UUID(candidate_id))

    async def list(
        self,
        difference_run_id: str | None = None,
        candidate_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List candidates with filtering."""
        params = CandidateListParams(
            difference_run_id=difference_run_id,
            candidate_type=candidate_type,
            status=status,
            limit=limit,
            offset=offset,
        )
        candidates, _ = await CandidateCRUD.get_many(self.db, params)
        return candidates
