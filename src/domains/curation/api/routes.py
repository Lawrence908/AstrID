"""Curation API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.adapters.auth.rbac import (
    Permission,
    UserWithRole,
    require_permission,
)
from src.core.api.response_wrapper import ResponseEnvelope, create_response
from src.core.db.session import get_db
from src.domains.curation.service import AlertService, ValidationEventService

router = APIRouter()


# ValidationEvent Models
class ValidationEventCreate(BaseModel):
    """Create validation event request model."""

    detection_id: str
    validator_id: str
    is_valid: bool
    label: str | None = None
    notes: str | None = None


class ValidationEventResponse(BaseModel):
    """Validation event response model."""

    id: str
    detection_id: str
    validator_id: str
    is_valid: bool
    label: str | None
    notes: str | None
    created_at: str
    updated_at: str


# Alert Models
class AlertCreate(BaseModel):
    """Create alert request model."""

    alert_type: str
    severity: str
    message: str
    resource_type: str
    resource_id: str
    metadata: dict


class AlertResponse(BaseModel):
    """Alert response model."""

    id: str
    alert_type: str
    severity: str
    message: str
    resource_type: str
    resource_id: str
    metadata: dict
    is_resolved: bool
    created_at: str
    updated_at: str


# ValidationEvent Routes
@router.post(
    "/validation-events",
    response_model=ResponseEnvelope[ValidationEventResponse],
    responses={
        200: {"description": "Validation event created successfully"},
        400: {"description": "Invalid validation event data"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def create_validation_event(
    event: ValidationEventCreate,
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.WRITE_DATA)),
) -> ResponseEnvelope[ValidationEventResponse]:
    """Create a new validation event."""
    service = ValidationEventService(db)
    result = await service.create_validation_event(event)
    return create_response(result)


@router.get(
    "/validation-events",
    response_model=ResponseEnvelope[list[ValidationEventResponse]],
    responses={
        200: {"description": "Validation events retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def list_validation_events(
    detection_id: str | None = Query(None, description="Filter by detection ID"),
    validator_id: str | None = Query(None, description="Filter by validator ID"),
    is_valid: bool | None = Query(None, description="Filter by validation result"),
    limit: int = Query(100, le=1000, description="Maximum number of events"),
    offset: int = Query(0, ge=0, description="Number of events to skip"),
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> ResponseEnvelope[list[ValidationEventResponse]]:
    """List validation events with optional filtering."""
    service = ValidationEventService(db)
    result = await service.list_validation_events(
        detection_id=detection_id,
        validator_id=validator_id,
        is_valid=is_valid,
        limit=limit,
        offset=offset,
    )
    return create_response(result)


@router.get(
    "/validation-events/{event_id}",
    response_model=ResponseEnvelope[ValidationEventResponse],
    responses={
        200: {"description": "Validation event retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Validation event not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_validation_event(
    event_id: str,
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> ResponseEnvelope[ValidationEventResponse]:
    """Get a specific validation event by ID."""
    service = ValidationEventService(db)
    event = await service.get_validation_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Validation event not found")
    return create_response(event)


# Alert Routes
@router.post(
    "/alerts",
    response_model=ResponseEnvelope[AlertResponse],
    responses={
        200: {"description": "Alert created successfully"},
        400: {"description": "Invalid alert data"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def create_alert(
    alert: AlertCreate,
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.WRITE_DATA)),
) -> ResponseEnvelope[AlertResponse]:
    """Create a new alert."""
    service = AlertService(db)
    result = await service.create_alert(alert)
    return create_response(result)


@router.get(
    "/alerts",
    response_model=ResponseEnvelope[list[AlertResponse]],
    responses={
        200: {"description": "Alerts retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def list_alerts(
    alert_type: str | None = Query(None, description="Filter by alert type"),
    severity: str | None = Query(None, description="Filter by severity"),
    is_resolved: bool | None = Query(None, description="Filter by resolution status"),
    limit: int = Query(100, le=1000, description="Maximum number of alerts"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip"),
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> ResponseEnvelope[list[AlertResponse]]:
    """List alerts with optional filtering."""
    service = AlertService(db)
    result = await service.list_alerts(
        alert_type=alert_type,
        severity=severity,
        is_resolved=is_resolved,
        limit=limit,
        offset=offset,
    )
    return create_response(result)


@router.get(
    "/alerts/{alert_id}",
    response_model=ResponseEnvelope[AlertResponse],
    responses={
        200: {"description": "Alert retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Alert not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_alert(
    alert_id: str,
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> ResponseEnvelope[AlertResponse]:
    """Get a specific alert by ID."""
    service = AlertService(db)
    alert = await service.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return create_response(alert)
