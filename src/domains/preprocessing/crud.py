"""CRUD operations for preprocessing domain."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db.exceptions import create_db_error
from src.domains.preprocessing.models import PreprocessRun
from src.domains.preprocessing.schema import (
    PreprocessRunCreate,
    PreprocessRunListParams,
    PreprocessRunUpdate,
)


class PreprocessRunCRUD:
    """CRUD operations for PreprocessRun model."""

    def __init__(self):
        """Initialize the CRUD class."""
        self.model = PreprocessRun
        self.logger = logging.getLogger(__name__)

    async def create(
        self, db: AsyncSession, run_data: PreprocessRunCreate
    ) -> PreprocessRun:
        """Create a new preprocessing run."""
        try:
            run = PreprocessRun(**run_data.model_dump())
            db.add(run)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(run)
            self.logger.info(
                f"Successfully created preprocessing run with ID: {run.id}"
            )
            return run
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating preprocessing run: {str(e)}")
            raise create_db_error(
                f"Failed to create preprocessing run: {str(e)}", e
            ) from e

    async def get_by_id(self, db: AsyncSession, run_id: UUID) -> PreprocessRun | None:
        """Get preprocessing run by ID."""
        try:
            result = await db.execute(
                select(PreprocessRun).where(PreprocessRun.id == run_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting preprocessing run with ID {run_id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to get preprocessing run with ID {run_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: PreprocessRunListParams
    ) -> tuple[list[PreprocessRun], int]:
        """Get multiple preprocessing runs with pagination."""
        # Build query
        query = select(PreprocessRun)
        count_query = select(func.count(PreprocessRun.id))

        # Apply filters
        conditions = []
        if params.observation_id:
            conditions.append(PreprocessRun.observation_id == params.observation_id)
        if params.status:
            conditions.append(PreprocessRun.status == params.status)
        if params.algorithm:
            conditions.append(PreprocessRun.algorithm == params.algorithm)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(PreprocessRun.created_at.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        runs = result.scalars().all()

        return list(runs), total

    @staticmethod
    async def update(
        db: AsyncSession, run_id: UUID, run_data: PreprocessRunUpdate
    ) -> PreprocessRun | None:
        """Update a preprocessing run."""
        run = await PreprocessRunCRUD.get_by_id(db, run_id)
        if not run:
            return None

        update_data = run_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(run, field, value)

        await db.commit()
        await db.refresh(run)
        return run

    @staticmethod
    async def delete(db: AsyncSession, run_id: UUID) -> bool:
        """Delete a preprocessing run."""
        run = await PreprocessRunCRUD.get_by_id(db, run_id)
        if not run:
            return False

        await db.delete(run)
        await db.commit()
        return True
