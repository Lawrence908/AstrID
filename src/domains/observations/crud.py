"""CRUD operations for observations domain."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.db.exceptions import create_db_error
from src.domains.observations.models import Observation, Survey
from src.domains.observations.schema import (
    ObservationCreate,
    ObservationListParams,
    ObservationUpdate,
    SurveyCreate,
    SurveyListParams,
    SurveyUpdate,
)


class SurveyCRUD:
    """CRUD operations for Survey model."""

    def __init__(self) -> None:
        """Initialize the CRUD class."""
        self.model = Survey
        self.logger = logging.getLogger(__name__)

    async def create(self, db: AsyncSession, survey_data: SurveyCreate) -> Survey:
        """Create a new survey."""
        try:
            survey = Survey(**survey_data.model_dump())
            db.add(survey)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(survey)
            self.logger.info(f"Successfully created survey with ID: {survey.id}")
            return survey
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating survey: {str(e)}")
            raise create_db_error(f"Failed to create survey: {str(e)}", e) from e

    async def get_by_id(self, db: AsyncSession, survey_id: UUID) -> Survey | None:
        """Get survey by ID."""
        try:
            result = await db.execute(select(Survey).where(Survey.id == survey_id))
            survey = result.scalar_one_or_none()
            return survey  # type: ignore[no-any-return]
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting survey with ID {survey_id}: {str(e)}")
            raise create_db_error(
                f"Failed to get survey with ID {survey_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_by_name(db: AsyncSession, name: str) -> Survey | None:
        """Get survey by name."""
        result = await db.execute(select(Survey).where(Survey.name == name))
        survey = result.scalar_one_or_none()
        return survey  # type: ignore[no-any-return]

    @staticmethod
    async def get_many(
        db: AsyncSession, params: SurveyListParams
    ) -> tuple[list[Survey], int]:
        """Get multiple surveys with pagination."""
        # Build query
        query = select(Survey)
        count_query = select(func.count(Survey.id))

        # Apply filters
        conditions = []
        if params.name:
            conditions.append(Survey.name.ilike(f"%{params.name}%"))
        if params.is_active is not None:
            conditions.append(Survey.is_active == params.is_active)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = query.order_by(Survey.name).offset(params.offset).limit(params.limit)

        # Execute query
        result = await db.execute(query)
        surveys = result.scalars().all()

        return list(surveys), total

    async def update(
        self, db: AsyncSession, survey_id: UUID, survey_data: SurveyUpdate
    ) -> Survey | None:
        """Update a survey."""
        survey = await self.get_by_id(db, survey_id)
        if not survey:
            return None

        update_data = survey_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(survey, field, value)

        await db.commit()
        await db.refresh(survey)
        return survey

    async def delete(self, db: AsyncSession, survey_id: UUID) -> bool:
        """Delete a survey."""
        survey = await self.get_by_id(db, survey_id)
        if not survey:
            return False

        await db.delete(survey)
        await db.commit()
        return True


class ObservationCRUD:
    """CRUD operations for Observation model."""

    def __init__(self) -> None:
        """Initialize the CRUD class."""
        self.model = Observation
        self.logger = logging.getLogger(__name__)

    async def create(
        self, db: AsyncSession, observation_data: ObservationCreate
    ) -> Observation:
        """Create a new observation."""
        try:
            observation = Observation(**observation_data.model_dump())
            db.add(observation)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(observation)
            self.logger.info(
                f"Successfully created observation with ID: {observation.id}"
            )
            return observation
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating observation: {str(e)}")
            raise create_db_error(f"Failed to create observation: {str(e)}", e) from e

    @staticmethod
    async def get_by_id(db: AsyncSession, observation_id: UUID) -> Observation | None:
        """Get observation by ID with survey information."""
        result = await db.execute(
            select(Observation)
            .options(selectinload(Observation.survey))
            .where(Observation.id == observation_id)
        )
        observation = result.scalar_one_or_none()
        return observation  # type: ignore[no-any-return]

    @staticmethod
    async def get_by_survey_and_observation_id(
        db: AsyncSession, survey_id: UUID, observation_id: str
    ) -> Observation | None:
        """Get observation by survey ID and observation ID."""
        result = await db.execute(
            select(Observation)
            .options(selectinload(Observation.survey))
            .where(
                and_(
                    Observation.survey_id == survey_id,
                    Observation.observation_id == observation_id,
                )
            )
        )
        observation = result.scalar_one_or_none()
        return observation  # type: ignore[no-any-return]

    @staticmethod
    async def get_many(
        db: AsyncSession, params: ObservationListParams
    ) -> tuple[list[Observation], int]:
        """Get multiple observations with pagination and filtering."""
        # Build query
        query = select(Observation).options(selectinload(Observation.survey))
        count_query = select(func.count(Observation.id))

        # Apply filters
        conditions = []

        if params.survey_id:
            conditions.append(Observation.survey_id == params.survey_id)
        if params.status:
            conditions.append(Observation.status == params.status.value)
        if params.filter_band:
            conditions.append(Observation.filter_band == params.filter_band)

        # Spatial filters
        if params.ra_min is not None:
            conditions.append(Observation.ra >= params.ra_min)
        if params.ra_max is not None:
            conditions.append(Observation.ra <= params.ra_max)
        if params.dec_min is not None:
            conditions.append(Observation.dec >= params.dec_min)
        if params.dec_max is not None:
            conditions.append(Observation.dec <= params.dec_max)

        # Temporal filters
        if params.date_from:
            conditions.append(Observation.observation_time >= params.date_from)
        if params.date_to:
            conditions.append(Observation.observation_time <= params.date_to)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(Observation.observation_time.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        observations = result.scalars().all()

        return list(observations), total

    @staticmethod
    async def get_by_spatial_region(
        db: AsyncSession,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        limit: int = 100,
    ) -> list[Observation]:
        """Get observations within a spatial region."""
        result = await db.execute(
            select(Observation)
            .options(selectinload(Observation.survey))
            .where(
                and_(
                    Observation.ra >= ra_min,
                    Observation.ra <= ra_max,
                    Observation.dec >= dec_min,
                    Observation.dec <= dec_max,
                )
            )
            .order_by(Observation.observation_time.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update(
        db: AsyncSession, observation_id: UUID, observation_data: ObservationUpdate
    ) -> Observation | None:
        """Update an observation."""
        observation = await ObservationCRUD.get_by_id(db, observation_id)
        if not observation:
            return None

        update_data = observation_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(observation, field, value)

        await db.commit()
        await db.refresh(observation)
        return observation

    @staticmethod
    async def delete(db: AsyncSession, observation_id: UUID) -> bool:
        """Delete an observation."""
        observation = await ObservationCRUD.get_by_id(db, observation_id)
        if not observation:
            return False

        await db.delete(observation)
        await db.commit()
        return True

    @staticmethod
    async def update_status(
        db: AsyncSession, observation_id: UUID, status: str
    ) -> Observation | None:
        """Update observation status."""
        observation = await ObservationCRUD.get_by_id(db, observation_id)
        if not observation:
            return None

        observation.status = status
        await db.commit()
        await db.refresh(observation)
        return observation
