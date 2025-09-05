"""Curation repository for API routes."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.curation.crud import AlertCRUD, ValidationEventCRUD
from src.domains.curation.schema import (
    AlertCreate,
    AlertListParams,
    ValidationEventCreate,
    ValidationEventListParams,
)


class ValidationEventRepository:
    """Repository for ValidationEvent operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, event_data: ValidationEventCreate):
        """Create a new validation event."""
        return await ValidationEventCRUD.create(self.db, event_data)

    async def get_by_id(self, event_id: str):
        """Get validation event by ID."""
        from uuid import UUID

        return await ValidationEventCRUD.get_by_id(self.db, UUID(event_id))

    async def list(
        self,
        detection_id: str | None = None,
        validator_id: str | None = None,
        is_valid: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List validation events with filtering."""
        params = ValidationEventListParams(
            detection_id=detection_id,
            validator_id=validator_id,
            is_valid=is_valid,
            limit=limit,
            offset=offset,
        )
        events, _ = await ValidationEventCRUD.get_many(self.db, params)
        return events


class AlertRepository:
    """Repository for Alert operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, alert_data: AlertCreate):
        """Create a new alert."""
        return await AlertCRUD.create(self.db, alert_data)

    async def get_by_id(self, alert_id: str):
        """Get alert by ID."""
        from uuid import UUID

        return await AlertCRUD.get_by_id(self.db, UUID(alert_id))

    async def list(
        self,
        alert_type: str | None = None,
        severity: str | None = None,
        is_resolved: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List alerts with filtering."""
        params = AlertListParams(
            alert_type=alert_type,
            severity=severity,
            is_resolved=is_resolved,
            limit=limit,
            offset=offset,
        )
        alerts, _ = await AlertCRUD.get_many(self.db, params)
        return alerts
