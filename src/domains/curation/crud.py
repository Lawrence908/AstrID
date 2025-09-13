"""CRUD operations for curation domain."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db.exceptions import create_db_error
from src.domains.curation.models import Alert, ValidationEvent
from src.domains.curation.schema import (
    AlertCreate,
    AlertListParams,
    AlertUpdate,
    ValidationEventCreate,
    ValidationEventListParams,
    ValidationEventUpdate,
)


class ValidationEventCRUD:
    """CRUD operations for ValidationEvent model."""

    def __init__(self):
        """Initialize the CRUD class."""
        self.model = ValidationEvent
        self.logger = logging.getLogger(__name__)

    async def create(
        self, db: AsyncSession, event_data: ValidationEventCreate
    ) -> ValidationEvent:
        """Create a new validation event."""
        try:
            event = ValidationEvent(**event_data.model_dump())
            db.add(event)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(event)
            self.logger.info(
                f"Successfully created validation event with ID: {event.id}"
            )
            return event
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating validation event: {str(e)}")
            raise create_db_error(
                f"Failed to create validation event: {str(e)}", e
            ) from e

    async def get_by_id(
        self, db: AsyncSession, event_id: UUID
    ) -> ValidationEvent | None:
        """Get validation event by ID."""
        try:
            result = await db.execute(
                select(ValidationEvent).where(ValidationEvent.id == event_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting validation event with ID {event_id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to get validation event with ID {event_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: ValidationEventListParams
    ) -> tuple[list[ValidationEvent], int]:
        """Get multiple validation events with pagination."""
        # Build query
        query = select(ValidationEvent)
        count_query = select(func.count(ValidationEvent.id))

        # Apply filters
        conditions = []
        if params.detection_id:
            conditions.append(ValidationEvent.detection_id == params.detection_id)
        if params.validator_id:
            conditions.append(ValidationEvent.validator_id == params.validator_id)
        if params.is_valid is not None:
            conditions.append(ValidationEvent.is_valid == params.is_valid)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(ValidationEvent.created_at.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        events = result.scalars().all()

        return list(events), total

    @staticmethod
    async def update(
        db: AsyncSession, event_id: UUID, event_data: ValidationEventUpdate
    ) -> ValidationEvent | None:
        """Update a validation event."""
        event = await ValidationEventCRUD.get_by_id(db, event_id)
        if not event:
            return None

        update_data = event_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(event, field, value)

        await db.commit()
        await db.refresh(event)
        return event

    @staticmethod
    async def delete(db: AsyncSession, event_id: UUID) -> bool:
        """Delete a validation event."""
        event = await ValidationEventCRUD.get_by_id(db, event_id)
        if not event:
            return False

        await db.delete(event)
        await db.commit()
        return True


class AlertCRUD:
    """CRUD operations for Alert model."""

    def __init__(self):
        """Initialize the CRUD class."""
        self.model = Alert
        self.logger = logging.getLogger(__name__)

    async def create(self, db: AsyncSession, alert_data: AlertCreate) -> Alert:
        """Create a new alert."""
        try:
            alert = Alert(**alert_data.model_dump())
            db.add(alert)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(alert)
            self.logger.info(f"Successfully created alert with ID: {alert.id}")
            return alert
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating alert: {str(e)}")
            raise create_db_error(f"Failed to create alert: {str(e)}", e) from e

    async def get_by_id(self, db: AsyncSession, alert_id: UUID) -> Alert | None:
        """Get alert by ID."""
        try:
            result = await db.execute(select(Alert).where(Alert.id == alert_id))
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting alert with ID {alert_id}: {str(e)}")
            raise create_db_error(
                f"Failed to get alert with ID {alert_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: AlertListParams
    ) -> tuple[list[Alert], int]:
        """Get multiple alerts with pagination."""
        # Build query
        query = select(Alert)
        count_query = select(func.count(Alert.id))

        # Apply filters
        conditions = []
        if params.alert_type:
            conditions.append(Alert.alert_type == params.alert_type)
        if params.severity:
            conditions.append(Alert.severity == params.severity)
        if params.is_resolved is not None:
            conditions.append(Alert.is_resolved == params.is_resolved)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(Alert.created_at.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        alerts = result.scalars().all()

        return list(alerts), total

    @staticmethod
    async def update(
        db: AsyncSession, alert_id: UUID, alert_data: AlertUpdate
    ) -> Alert | None:
        """Update an alert."""
        alert = await AlertCRUD.get_by_id(db, alert_id)
        if not alert:
            return None

        update_data = alert_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(alert, field, value)

        await db.commit()
        await db.refresh(alert)
        return alert

    @staticmethod
    async def delete(db: AsyncSession, alert_id: UUID) -> bool:
        """Delete an alert."""
        alert = await AlertCRUD.get_by_id(db, alert_id)
        if not alert:
            return False

        await db.delete(alert)
        await db.commit()
        return True
