"""Preprocessing repository for API routes."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.preprocessing.crud import PreprocessRunCRUD
from src.domains.preprocessing.schema import (
    PreprocessRunCreate,
    PreprocessRunListParams,
)


class PreprocessRunRepository:
    """Repository for PreprocessRun operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, run_data: PreprocessRunCreate):
        """Create a new preprocessing run."""
        return await PreprocessRunCRUD.create(self.db, run_data)

    async def get_by_id(self, run_id: str):
        """Get preprocessing run by ID."""
        from uuid import UUID

        return await PreprocessRunCRUD.get_by_id(self.db, UUID(run_id))

    async def list(
        self,
        observation_id: str | None = None,
        status: str | None = None,
        algorithm: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List preprocessing runs with filtering."""
        params = PreprocessRunListParams(
            observation_id=observation_id,
            status=status,
            algorithm=algorithm,
            limit=limit,
            offset=offset,
        )
        runs, _ = await PreprocessRunCRUD.get_many(self.db, params)
        return runs
