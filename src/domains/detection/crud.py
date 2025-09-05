"""CRUD operations for detection domain."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db.exceptions import create_db_error
from src.domains.detection.models import Detection, Model, ModelRun
from src.domains.detection.schema import (
    DetectionCreate,
    DetectionListParams,
    DetectionUpdate,
    ModelCreate,
    ModelListParams,
    ModelRunCreate,
    ModelRunListParams,
    ModelUpdate,
)


class ModelCRUD:
    """CRUD operations for Model model."""

    def __init__(self) -> None:
        """Initialize the CRUD class."""
        self.model = Model
        self.logger = logging.getLogger(__name__)

    async def create(self, db: AsyncSession, model_data: ModelCreate) -> Model:
        """Create a new model."""
        try:
            model = Model(**model_data.model_dump())
            db.add(model)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(model)
            self.logger.info(f"Successfully created model with ID: {model.id}")
            return model
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating model: {str(e)}")
            raise create_db_error(f"Failed to create model: {str(e)}", e) from e

    async def get_by_id(self, db: AsyncSession, model_id: UUID) -> Model | None:
        """Get model by ID."""
        try:
            result = await db.execute(select(Model).where(Model.id == model_id))
            return result.scalar_one_or_none()  # type: ignore[no-any-return]
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting model with ID {model_id}: {str(e)}")
            raise create_db_error(
                f"Failed to get model with ID {model_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: ModelListParams
    ) -> tuple[list[Model], int]:
        """Get multiple models with pagination."""
        # Build query
        query = select(Model)
        count_query = select(func.count(Model.id))

        # Apply filters
        conditions = []
        if params.model_type:
            conditions.append(Model.model_type == params.model_type)
        if params.is_active is not None:
            conditions.append(Model.is_active == params.is_active)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = query.order_by(Model.name).offset(params.offset).limit(params.limit)

        # Execute query
        result = await db.execute(query)
        models = result.scalars().all()

        return list(models), total

    @staticmethod
    async def update(
        db: AsyncSession, model_id: UUID, model_data: ModelUpdate
    ) -> Model | None:
        """Update a model."""
        model = await ModelCRUD.get_by_id(db, model_id)  # type: ignore[call-arg]  # type: ignore[call-arg]
        if not model:
            return None

        update_data = model_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(model, field, value)

        await db.commit()
        await db.refresh(model)
        return model

    @staticmethod
    async def delete(db: AsyncSession, model_id: UUID) -> bool:
        """Delete a model."""
        model = await ModelCRUD.get_by_id(db, model_id)  # type: ignore[call-arg]  # type: ignore[call-arg]
        if not model:
            return False

        await db.delete(model)
        await db.commit()
        return True


class ModelRunCRUD:
    """CRUD operations for ModelRun model."""

    def __init__(self) -> None:
        """Initialize the CRUD class."""
        self.model = ModelRun
        self.logger = logging.getLogger(__name__)

    async def create(self, db: AsyncSession, run_data: ModelRunCreate) -> ModelRun:
        """Create a new model run."""
        try:
            run = ModelRun(**run_data.model_dump())
            db.add(run)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(run)
            self.logger.info(f"Successfully created model run with ID: {run.id}")
            return run
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating model run: {str(e)}")
            raise create_db_error(f"Failed to create model run: {str(e)}", e) from e

    async def get_by_id(self, db: AsyncSession, run_id: UUID) -> ModelRun | None:
        """Get model run by ID."""
        try:
            result = await db.execute(select(ModelRun).where(ModelRun.id == run_id))
            return result.scalar_one_or_none()  # type: ignore[no-any-return]
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting model run with ID {run_id}: {str(e)}")
            raise create_db_error(
                f"Failed to get model run with ID {run_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: ModelRunListParams
    ) -> tuple[list[ModelRun], int]:
        """Get multiple model runs with pagination."""
        # Build query
        query = select(ModelRun)
        count_query = select(func.count(ModelRun.id))

        # Apply filters
        conditions = []
        if params.model_id:
            conditions.append(ModelRun.model_id == params.model_id)
        if params.status:
            conditions.append(ModelRun.status == params.status)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(ModelRun.created_at.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        runs = result.scalars().all()

        return list(runs), total


class DetectionCRUD:
    """CRUD operations for Detection model."""

    def __init__(self) -> None:
        """Initialize the CRUD class."""
        self.model = Detection
        self.logger = logging.getLogger(__name__)

    async def create(
        self, db: AsyncSession, detection_data: DetectionCreate
    ) -> Detection:
        """Create a new detection."""
        try:
            detection = Detection(**detection_data.model_dump())
            db.add(detection)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(detection)
            self.logger.info(f"Successfully created detection with ID: {detection.id}")
            return detection
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating detection: {str(e)}")
            raise create_db_error(f"Failed to create detection: {str(e)}", e) from e

    async def get_by_id(self, db: AsyncSession, detection_id: UUID) -> Detection | None:
        """Get detection by ID."""
        try:
            result = await db.execute(
                select(Detection).where(Detection.id == detection_id)
            )
            return result.scalar_one_or_none()  # type: ignore[no-any-return]
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting detection with ID {detection_id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to get detection with ID {detection_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: DetectionListParams
    ) -> tuple[list[Detection], int]:
        """Get multiple detections with pagination."""
        # Build query
        query = select(Detection)
        count_query = select(func.count(Detection.id))

        # Apply filters
        conditions = []
        if params.observation_id:
            conditions.append(Detection.observation_id == params.observation_id)
        if params.detection_type:
            conditions.append(Detection.detection_type == params.detection_type)
        if params.status:
            conditions.append(Detection.status == params.status)
        if params.min_confidence_score is not None:
            conditions.append(Detection.confidence_score >= params.min_confidence_score)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(Detection.created_at.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        detections = result.scalars().all()

        return list(detections), total

    @staticmethod
    async def update(
        db: AsyncSession, detection_id: UUID, detection_data: DetectionUpdate
    ) -> Detection | None:
        """Update a detection."""
        detection = await DetectionCRUD.get_by_id(db, detection_id)  # type: ignore[call-arg]  # type: ignore[call-arg]
        if not detection:
            return None

        update_data = detection_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(detection, field, value)

        await db.commit()
        await db.refresh(detection)
        return detection

    @staticmethod
    async def delete(db: AsyncSession, detection_id: UUID) -> bool:
        """Delete a detection."""
        detection = await DetectionCRUD.get_by_id(db, detection_id)  # type: ignore[call-arg]  # type: ignore[call-arg]
        if not detection:
            return False

        await db.delete(detection)
        await db.commit()
        return True
