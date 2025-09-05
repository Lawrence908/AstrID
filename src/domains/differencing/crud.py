"""CRUD operations for differencing domain."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db.exceptions import create_db_error
from src.domains.differencing.models import Candidate, DifferenceRun
from src.domains.differencing.schema import (
    CandidateCreate,
    CandidateListParams,
    CandidateUpdate,
    DifferenceRunCreate,
    DifferenceRunListParams,
    DifferenceRunUpdate,
)


class DifferenceRunCRUD:
    """CRUD operations for DifferenceRun model."""

    def __init__(self):
        """Initialize the CRUD class."""
        self.model = DifferenceRun
        self.logger = logging.getLogger(__name__)

    async def create(
        self, db: AsyncSession, run_data: DifferenceRunCreate
    ) -> DifferenceRun:
        """Create a new difference run."""
        try:
            run = DifferenceRun(**run_data.model_dump())
            db.add(run)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(run)
            self.logger.info(f"Successfully created difference run with ID: {run.id}")
            return run
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating difference run: {str(e)}")
            raise create_db_error(
                f"Failed to create difference run: {str(e)}", e
            ) from e

    async def get_by_id(self, db: AsyncSession, run_id: UUID) -> DifferenceRun | None:
        """Get difference run by ID."""
        try:
            result = await db.execute(
                select(DifferenceRun).where(DifferenceRun.id == run_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting difference run with ID {run_id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to get difference run with ID {run_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: DifferenceRunListParams
    ) -> tuple[list[DifferenceRun], int]:
        """Get multiple difference runs with pagination."""
        # Build query
        query = select(DifferenceRun)
        count_query = select(func.count(DifferenceRun.id))

        # Apply filters
        conditions = []
        if params.observation_id:
            conditions.append(DifferenceRun.observation_id == params.observation_id)
        if params.status:
            conditions.append(DifferenceRun.status == params.status)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(DifferenceRun.created_at.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        runs = result.scalars().all()

        return list(runs), total

    @staticmethod
    async def update(
        db: AsyncSession, run_id: UUID, run_data: DifferenceRunUpdate
    ) -> DifferenceRun | None:
        """Update a difference run."""
        run = await DifferenceRunCRUD.get_by_id(db, run_id)
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
        """Delete a difference run."""
        run = await DifferenceRunCRUD.get_by_id(db, run_id)
        if not run:
            return False

        await db.delete(run)
        await db.commit()
        return True


class CandidateCRUD:
    """CRUD operations for Candidate model."""

    def __init__(self):
        """Initialize the CRUD class."""
        self.model = Candidate
        self.logger = logging.getLogger(__name__)

    async def create(
        self, db: AsyncSession, candidate_data: CandidateCreate
    ) -> Candidate:
        """Create a new candidate."""
        try:
            candidate = Candidate(**candidate_data.model_dump())
            db.add(candidate)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(candidate)
            self.logger.info(f"Successfully created candidate with ID: {candidate.id}")
            return candidate
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating candidate: {str(e)}")
            raise create_db_error(f"Failed to create candidate: {str(e)}", e) from e

    async def get_by_id(self, db: AsyncSession, candidate_id: UUID) -> Candidate | None:
        """Get candidate by ID."""
        try:
            result = await db.execute(
                select(Candidate).where(Candidate.id == candidate_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting candidate with ID {candidate_id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to get candidate with ID {candidate_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: CandidateListParams
    ) -> tuple[list[Candidate], int]:
        """Get multiple candidates with pagination."""
        # Build query
        query = select(Candidate)
        count_query = select(func.count(Candidate.id))

        # Apply filters
        conditions = []
        if params.difference_run_id:
            conditions.append(Candidate.difference_run_id == params.difference_run_id)
        if params.candidate_type:
            conditions.append(Candidate.candidate_type == params.candidate_type)
        if params.status:
            conditions.append(Candidate.status == params.status)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(Candidate.created_at.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        candidates = result.scalars().all()

        return list(candidates), total

    @staticmethod
    async def update(
        db: AsyncSession, candidate_id: UUID, candidate_data: CandidateUpdate
    ) -> Candidate | None:
        """Update a candidate."""
        candidate = await CandidateCRUD.get_by_id(db, candidate_id)
        if not candidate:
            return None

        update_data = candidate_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(candidate, field, value)

        await db.commit()
        await db.refresh(candidate)
        return candidate

    @staticmethod
    async def delete(db: AsyncSession, candidate_id: UUID) -> bool:
        """Delete a candidate."""
        candidate = await CandidateCRUD.get_by_id(db, candidate_id)
        if not candidate:
            return False

        await db.delete(candidate)
        await db.commit()
        return True
