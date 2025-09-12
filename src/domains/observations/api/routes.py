"""Observations API routes."""

from datetime import datetime

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


# Ingestion routes


class IngestionRequest(BaseModel):
    """Base ingestion request model."""

    survey_id: str


class MASTIngestionRequest(IngestionRequest):
    """MAST ingestion request model."""

    ra: float
    dec: float
    radius: float = 0.1
    missions: list[str] | None = None
    start_time: str | None = None
    end_time: str | None = None


class ReferenceDatasetRequest(IngestionRequest):
    """Reference dataset creation request model."""

    ra: float
    dec: float
    size: float = 0.25
    pixels: int = 512
    surveys: list[str] | None = None


class BatchIngestionRequest(IngestionRequest):
    """Batch ingestion request model."""

    count: int = 10
    missions: list[str] | None = None
    avoid_galactic_plane: bool = True


class DirectoryIngestionRequest(IngestionRequest):
    """Directory ingestion request model."""

    directory_path: str
    file_pattern: str = "*.fits"


@router.post(
    "/ingest/mast",
    response_model=ResponseEnvelope[list[ObservationResponse]],
    responses={
        200: {"description": "MAST observations ingested successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def ingest_mast_observations(
    request: MASTIngestionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> ResponseEnvelope[list[ObservationResponse]]:
    """Ingest observations from MAST for a specific sky position."""
    try:
        from uuid import UUID

        from src.domains.observations.service import ObservationService

        service = ObservationService(db)

        # Parse optional datetime strings
        start_time = None
        end_time = None
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)

        observations = await service.ingest_observations_from_mast(
            survey_id=UUID(request.survey_id),
            ra=request.ra,
            dec=request.dec,
            radius=request.radius,
            missions=request.missions,
            start_time=start_time,
            end_time=end_time,
        )

        # Convert to response format (simplified for now)
        response_observations = []
        for obs in observations:
            response_obs = ObservationResponse(
                id=str(obs.id),
                survey=str(obs.survey_id),  # In real implementation, get survey name
                observation_id=obs.observation_id,
                ra=obs.ra,
                dec=obs.dec,
                observation_time=obs.observation_time.isoformat(),
                filter_band=obs.filter_band,
                exposure_time=obs.exposure_time,
                fits_url=obs.fits_url,
                status=obs.status.value,
                created_at=obs.created_at.isoformat(),
                updated_at=obs.updated_at.isoformat(),
            )
            response_observations.append(response_obs)

        return create_response(response_observations)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ingestion failed: {str(e)}"
        ) from e


@router.post(
    "/ingest/reference-dataset",
    responses={
        200: {"description": "Reference dataset created successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def create_reference_dataset(
    request: ReferenceDatasetRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
):
    """Create a complete reference dataset with image, catalog, and mask."""
    try:
        from uuid import UUID

        from src.domains.observations.service import ObservationService

        service = ObservationService(db)

        result = await service.create_reference_dataset(
            survey_id=UUID(request.survey_id),
            ra=request.ra,
            dec=request.dec,
            size=request.size,
            pixels=request.pixels,
            surveys=request.surveys,
        )

        return create_response(result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Dataset creation failed: {str(e)}"
        ) from e


@router.post(
    "/ingest/batch-random",
    response_model=ResponseEnvelope[list[ObservationResponse]],
    responses={
        200: {"description": "Batch ingestion completed successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def batch_ingest_random_observations(
    request: BatchIngestionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> ResponseEnvelope[list[ObservationResponse]]:
    """Batch ingest observations from random sky positions."""
    try:
        from uuid import UUID

        from src.domains.observations.service import ObservationService

        service = ObservationService(db)

        observations = await service.batch_ingest_random_observations(
            survey_id=UUID(request.survey_id),
            count=request.count,
            missions=request.missions,
            avoid_galactic_plane=request.avoid_galactic_plane,
        )

        # Convert to response format
        response_observations = []
        for obs in observations:
            response_obs = ObservationResponse(
                id=str(obs.id),
                survey=str(obs.survey_id),
                observation_id=obs.observation_id,
                ra=obs.ra,
                dec=obs.dec,
                observation_time=obs.observation_time.isoformat(),
                filter_band=obs.filter_band,
                exposure_time=obs.exposure_time,
                fits_url=obs.fits_url,
                status=obs.status.value,
                created_at=obs.created_at.isoformat(),
                updated_at=obs.updated_at.isoformat(),
            )
            response_observations.append(response_obs)

        return create_response(response_observations)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch ingestion failed: {str(e)}"
        ) from e


@router.post(
    "/ingest/directory",
    response_model=ResponseEnvelope[list[ObservationResponse]],
    responses={
        200: {"description": "Directory ingestion completed successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def ingest_from_directory(
    request: DirectoryIngestionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> ResponseEnvelope[list[ObservationResponse]]:
    """Ingest observations from FITS files in a directory."""
    try:
        from uuid import UUID

        from src.domains.observations.service import ObservationService

        service = ObservationService(db)

        observations = await service.ingest_from_fits_directory(
            survey_id=UUID(request.survey_id),
            directory_path=request.directory_path,
            file_pattern=request.file_pattern,
        )

        # Convert to response format
        response_observations = []
        for obs in observations:
            response_obs = ObservationResponse(
                id=str(obs.id),
                survey=str(obs.survey_id),
                observation_id=obs.observation_id,
                ra=obs.ra,
                dec=obs.dec,
                observation_time=obs.observation_time.isoformat(),
                filter_band=obs.filter_band,
                exposure_time=obs.exposure_time,
                fits_url=obs.fits_url,
                status=obs.status.value,
                created_at=obs.created_at.isoformat(),
                updated_at=obs.updated_at.isoformat(),
            )
            response_observations.append(response_obs)

        return create_response(response_observations)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Directory ingestion failed: {str(e)}"
        ) from e
