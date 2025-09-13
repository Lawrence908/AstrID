"""Curation service layer for business logic."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.curation.repository import AlertRepository, ValidationEventRepository
from src.domains.curation.schema import AlertCreate, ValidationEventCreate


class ValidationEventService:
    """Service for ValidationEvent business logic."""

    def __init__(self, db: AsyncSession):
        self.repository = ValidationEventRepository(db)
        self.logger = configure_domain_logger("curation.validation_event")

    async def create_validation_event(self, event_data: ValidationEventCreate):
        """Create a new validation event with business logic."""
        self.logger.info(
            f"Creating validation event: detection_id={event_data.detection_id}, validator_id={event_data.validator_id}"
        )
        try:
            # TODO: Add validation logic, automatic timestamps, etc.
            result = await self.repository.create(event_data)
            self.logger.info(
                f"Successfully created validation event: id={result.id}, is_valid={result.is_valid}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create validation event: detection_id={event_data.detection_id}, error={str(e)}"
            )
            raise

    async def get_validation_event(self, event_id: str):
        """Get validation event by ID."""
        self.logger.debug(f"Retrieving validation event by ID: {event_id}")
        result = await self.repository.get_by_id(event_id)
        if result:
            self.logger.debug(
                f"Found validation event: id={result.id}, is_valid={result.is_valid}"
            )
        else:
            self.logger.warning(f"Validation event not found: id={event_id}")
        return result

    async def list_validation_events(
        self,
        detection_id: str | None = None,
        validator_id: str | None = None,
        is_valid: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List validation events."""
        self.logger.debug(
            f"Listing validation events: detection_id={detection_id}, validator_id={validator_id}, is_valid={is_valid}, limit={limit}"
        )
        result = await self.repository.list(
            detection_id=detection_id,
            validator_id=validator_id,
            is_valid=is_valid,
            limit=limit,
            offset=offset,
        )
        self.logger.debug(f"Retrieved {len(result)} validation events")
        return result


class AlertService:
    """Service for Alert business logic."""

    def __init__(self, db: AsyncSession):
        self.repository = AlertRepository(db)
        self.logger = configure_domain_logger("curation.alert")

    async def create_alert(self, alert_data: AlertCreate):
        """Create a new alert with business logic."""
        self.logger.warning(
            f"Creating alert: type={alert_data.alert_type}, severity={alert_data.severity}, message={alert_data.message}"
        )
        try:
            # TODO: Add alert escalation, notification logic, etc.
            result = await self.repository.create(alert_data)
            self.logger.warning(
                f"Successfully created alert: id={result.id}, type={result.alert_type}, severity={result.severity}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create alert: type={alert_data.alert_type}, error={str(e)}"
            )
            raise

    async def get_alert(self, alert_id: str):
        """Get alert by ID."""
        self.logger.debug(f"Retrieving alert by ID: {alert_id}")
        result = await self.repository.get_by_id(alert_id)
        if result:
            self.logger.debug(
                f"Found alert: id={result.id}, type={result.alert_type}, severity={result.severity}"
            )
        else:
            self.logger.warning(f"Alert not found: id={alert_id}")
        return result

    async def list_alerts(
        self,
        alert_type: str | None = None,
        severity: str | None = None,
        is_resolved: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List alerts."""
        self.logger.debug(
            f"Listing alerts: type={alert_type}, severity={severity}, is_resolved={is_resolved}, limit={limit}"
        )
        result = await self.repository.list(
            alert_type=alert_type,
            severity=severity,
            is_resolved=is_resolved,
            limit=limit,
            offset=offset,
        )
        self.logger.debug(f"Retrieved {len(result)} alerts")
        return result
