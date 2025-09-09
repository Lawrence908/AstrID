"""Observations API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.auth.rbac import (
    Permission,
    UserWithRole,
    require_permission,
)
from src.core.api.response_wrapper import ResponseEnvelope, create_response
from src.core.db.session import get_db
from src.core.exceptions import (
    AstrIDException,
    InvalidCoordinatesError,
    InvalidExposureTimeError,
    InvalidFilterBandError,
    ObservationAlreadyExistsError,
    ResourceNotFoundError,
    ValidationError,
)
from src.domains.observations.repository import ObservationRepository

router = APIRouter()


class ObservationCreate(BaseModel):
    """Create observation request model."""

    survey: str
    observation_id: str
    ra: float
    dec: float
    observation_time: str
    filter_band: str
    exposure_time: float
    fits_url: str


class ObservationResponse(BaseModel):
    """Observation response model."""

    id: str
    survey: str
    observation_id: str
    ra: float
    dec: float
    observation_time: str
    filter_band: str
    exposure_time: float
    fits_url: str
    status: str
    created_at: str
    updated_at: str


@router.post(
    "/",
    response_model=ResponseEnvelope[ObservationResponse],
    responses={
        200: {"description": "Observation created successfully"},
        400: {"description": "Invalid observation data"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        409: {"description": "Observation already exists"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def create_observation(
    observation: ObservationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> ResponseEnvelope[ObservationResponse]:
    """Create a new observation."""
    try:
        repo = ObservationRepository(db)
        result = await repo.create(observation)
        return create_response(result)
    except AstrIDException as e:
        # Convert to HTTPException for FastAPI
        status_code = (
            400
            if isinstance(
                e,
                ValidationError
                | InvalidCoordinatesError
                | InvalidExposureTimeError
                | InvalidFilterBandError,
            )
            else 409
            if isinstance(e, ObservationAlreadyExistsError)
            else 500
        )
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.get(
    "/",
    response_model=ResponseEnvelope[list[ObservationResponse]],
    responses={
        200: {"description": "Observations retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def list_observations(
    survey: str | None = Query(None, description="Filter by survey"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, le=1000, description="Maximum number of observations"),
    offset: int = Query(0, ge=0, description="Number of observations to skip"),
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> ResponseEnvelope[list[ObservationResponse]]:
    """List observations with optional filtering."""
    repo = ObservationRepository(db)
    result = await repo.list(survey=survey, status=status, limit=limit, offset=offset)
    return create_response(result)


@router.get(
    "/{observation_id}",
    response_model=ResponseEnvelope[ObservationResponse],
    responses={
        200: {"description": "Observation retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Observation not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_observation(
    observation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> ResponseEnvelope[ObservationResponse]:
    """Get a specific observation by ID."""
    try:
        repo = ObservationRepository(db)
        observation = await repo.get_by_id(observation_id)
        if not observation:
            raise ResourceNotFoundError(
                message=f"Observation with ID {observation_id} not found",
                error_code="OBSERVATION_NOT_FOUND",
                details={"observation_id": observation_id},
            )
        return create_response(observation)
    except AstrIDException as e:
        status_code = 404 if isinstance(e, ResourceNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.post(
    "/sync",
    responses={
        200: {"description": "Observation sync initiated successfully"},
        400: {"description": "Invalid sync parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Admin access required"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def sync_observations(
    survey: str = Query(..., description="Survey to sync"),
    from_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.ADMIN_ACCESS)),
):
    """Trigger observation synchronization for a survey."""
    # This would typically enqueue a background job
    # For now, return a placeholder response
    result = {
        "message": f"Observation sync initiated for survey {survey}",
        "survey": survey,
        "from_date": from_date,
        "to_date": to_date,
        "status": "queued",
    }
    return create_response(result)
