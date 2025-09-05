"""Base CRUD class with common database operations and error handling."""

import logging
from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from src.core.db.exceptions import create_db_error

# Type variables for generic CRUD operations
ModelType = TypeVar("ModelType", bound=DeclarativeBase)
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")
ReadSchemaType = TypeVar("ReadSchemaType")


class BaseCRUD(Generic[ModelType, CreateSchemaType, UpdateSchemaType, ReadSchemaType]):
    """Base CRUD class with common database operations and error handling."""

    def __init__(self, model: type[ModelType], logger_name: str | None = None):
        """
        Initialize the CRUD class.

        Args:
            model: SQLAlchemy model class
            logger_name: Optional logger name (defaults to model class name)
        """
        self.model = model
        self.logger = logging.getLogger(
            logger_name or f"{model.__module__}.{model.__name__}"
        )

    async def create(
        self, db: AsyncSession, obj_in: CreateSchemaType, **kwargs: Any
    ) -> ModelType:
        """
        Create a new record.

        Args:
            db: Database session
            obj_in: Data to create
            **kwargs: Additional fields to set

        Returns:
            Created model instance

        Raises:
            DBError: If creation fails
        """
        try:
            # Convert Pydantic model to dict if needed
            if hasattr(obj_in, "model_dump"):
                obj_data = obj_in.model_dump()
            else:
                obj_data = obj_in

            # Add any additional fields
            obj_data.update(kwargs)

            # Create model instance
            db_obj = self.model(**obj_data)
            db.add(db_obj)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(db_obj)

            self.logger.info(
                f"Successfully created {self.model.__name__} with ID: {db_obj.id}"
            )
            return db_obj  # type: ignore[no-any-return]

        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating {self.model.__name__}: {str(e)}")
            raise create_db_error(
                f"Failed to create {self.model.__name__}: {str(e)}", e
            ) from e

    async def get(self, db: AsyncSession, id: UUID) -> ModelType | None:
        """
        Get a record by ID.

        Args:
            db: Database session
            id: Record ID

        Returns:
            Model instance or None if not found

        Raises:
            DBError: If query fails
        """
        try:
            result = await db.execute(select(self.model).where(self.model.id == id))
            return result.scalar_one_or_none()  # type: ignore[no-any-return]

        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting {self.model.__name__} with ID {id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to get {self.model.__name__} with ID {id}: {str(e)}", e
            ) from e

    async def get_multi(
        self, db: AsyncSession, skip: int = 0, limit: int = 100, **filters: Any
    ) -> tuple[list[ModelType], int]:
        """
        Get multiple records with pagination and filtering.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            **filters: Field filters (e.g., name="test", is_active=True)

        Returns:
            Tuple of (records, total_count)

        Raises:
            DBError: If query fails
        """
        try:
            # Build base query
            query = select(self.model)
            count_query = select(func.count(self.model.id))

            # Apply filters
            conditions = []
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    conditions.append(getattr(self.model, field) == value)

            if conditions:
                from sqlalchemy import and_

                query = query.where(and_(*conditions))
                count_query = count_query.where(and_(*conditions))

            # Get total count
            total_result = await db.execute(count_query)
            total = total_result.scalar()

            # Apply pagination
            query = query.offset(skip).limit(limit)

            # Execute query
            result = await db.execute(query)
            items = result.scalars().all()

            return list(items), total

        except SQLAlchemyError as e:
            self.logger.error(f"Error getting multiple {self.model.__name__}: {str(e)}")
            raise create_db_error(
                f"Failed to get multiple {self.model.__name__}: {str(e)}", e
            ) from e

    async def update(
        self, db: AsyncSession, db_obj: ModelType, obj_in: UpdateSchemaType
    ) -> ModelType:
        """
        Update a record.

        Args:
            db: Database session
            db_obj: Existing model instance
            obj_in: Update data

        Returns:
            Updated model instance

        Raises:
            DBError: If update fails
        """
        try:
            # Convert Pydantic model to dict if needed
            if hasattr(obj_in, "model_dump"):
                update_data = obj_in.model_dump(exclude_unset=True)
            else:
                update_data = obj_in

            # Update fields
            for field, value in update_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)

            await db.flush()
            await db.refresh(db_obj)

            self.logger.info(
                f"Successfully updated {self.model.__name__} with ID: {db_obj.id}"
            )
            return db_obj

        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(
                f"Error updating {self.model.__name__} with ID {db_obj.id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to update {self.model.__name__} with ID {db_obj.id}: {str(e)}",
                e,
            ) from e

    async def delete(self, db: AsyncSession, id: UUID) -> bool:
        """
        Delete a record.

        Args:
            db: Database session
            id: Record ID

        Returns:
            True if deleted, False if not found

        Raises:
            DBError: If deletion fails
        """
        try:
            result = await db.execute(select(self.model).where(self.model.id == id))
            obj = result.scalar_one_or_none()

            if not obj:
                return False

            await db.delete(obj)
            await db.flush()

            self.logger.info(
                f"Successfully deleted {self.model.__name__} with ID: {id}"
            )
            return True

        except SQLAlchemyError as e:
            await db.rollback()
            self.logger.error(
                f"Error deleting {self.model.__name__} with ID {id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to delete {self.model.__name__} with ID {id}: {str(e)}", e
            ) from e

    async def exists(self, db: AsyncSession, id: UUID) -> bool:
        """
        Check if a record exists.

        Args:
            db: Database session
            id: Record ID

        Returns:
            True if exists, False otherwise

        Raises:
            DBError: If query fails
        """
        try:
            result = await db.execute(
                select(func.count(self.model.id)).where(self.model.id == id)
            )
            return bool(result.scalar() > 0)

        except SQLAlchemyError as e:
            self.logger.error(
                f"Error checking existence of {self.model.__name__} with ID {id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to check existence of {self.model.__name__} with ID {id}: {str(e)}",
                e,
            ) from e
