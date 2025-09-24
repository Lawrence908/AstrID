"""Observations API routes."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.auth.api_key_auth import require_permission_or_api_key
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
from src.domains.observations.schema import (
    ObservationCreate as DomainObservationCreate,
)
from src.domains.observations.schema import (
    ObservationRead,
    ObservationStatus,
)
from src.domains.observations.service import ObservationService

router = APIRouter()


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
    response_model=ResponseEnvelope[ObservationRead],
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
    observation: DomainObservationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Create a new observation."""
    try:
        repo = ObservationRepository(db)
        result = await repo.create(observation)
        # Convert DB model to domain read schema for consistent serialization
        response_obj = ObservationRead.model_validate(result)
        return create_response(response_obj)
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
) -> JSONResponse:
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
) -> JSONResponse:
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


@router.put(
    "/{observation_id}/status",
    response_model=ResponseEnvelope[ObservationRead],
    responses={
        200: {"description": "Observation status updated successfully"},
        400: {"description": "Invalid status transition"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Observation not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def update_observation_status(
    observation_id: str,
    status: ObservationStatus,
    reason: str | None = Query(None, description="Reason for status change"),
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Update observation processing status."""
    try:
        from uuid import UUID

        service = ObservationService(db)

        updated_obs = await service.process_observation_status_change(
            observation_id=UUID(observation_id),
            new_status=status,
            reason=reason,
            changed_by=getattr(current_user, "username", "system"),
        )

        if not updated_obs:
            raise ResourceNotFoundError(
                message=f"Observation with ID {observation_id} not found",
                error_code="OBSERVATION_NOT_FOUND",
                details={"observation_id": observation_id},
            )

        return create_response(updated_obs)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except AstrIDException as e:
        status_code = 404 if isinstance(e, ResourceNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.get(
    "/search",
    response_model=ResponseEnvelope[list[ObservationResponse]],
    responses={
        200: {"description": "Observations found successfully"},
        400: {"description": "Invalid search parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def search_observations_by_coordinates(
    ra: float = Query(..., description="Right Ascension in degrees (0-360)"),
    dec: float = Query(..., description="Declination in degrees (-90 to 90)"),
    radius: float = Query(0.1, description="Search radius in degrees"),
    limit: int = Query(100, le=1000, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Search for observations by coordinates within a circular region."""
    try:
        # Validate coordinates
        if not (0 <= ra <= 360):
            raise HTTPException(
                status_code=400, detail="RA must be between 0 and 360 degrees"
            )
        if not (-90 <= dec <= 90):
            raise HTTPException(
                status_code=400, detail="Dec must be between -90 and 90 degrees"
            )
        if radius <= 0:
            raise HTTPException(status_code=400, detail="Radius must be positive")

        repo = ObservationRepository(db)
        observations = await repo.get_by_coordinates(ra, dec, radius)

        # Limit results
        observations = observations[:limit]

        # Convert to response format
        response_observations = []
        for obs in observations:
            # Get survey name (simplified for now)
            survey_name = obs.survey.name if obs.survey else "Unknown"

            response_obs = ObservationResponse(
                id=str(obs.id),
                survey=survey_name,
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.get(
    "/metrics/{observation_id}",
    responses={
        200: {"description": "Observation metrics calculated successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Observation not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_observation_metrics(
    observation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get calculated metrics and derived values for an observation."""
    try:
        from uuid import UUID

        service = ObservationService(db)

        # Get observation first
        observation = await service.get_observation_by_id(UUID(observation_id))
        if not observation:
            raise ResourceNotFoundError(
                message=f"Observation with ID {observation_id} not found",
                error_code="OBSERVATION_NOT_FOUND",
                details={"observation_id": observation_id},
            )

        # Calculate metrics
        metrics = await service.calculate_observation_metrics(observation)

        return create_response(metrics)

    except AstrIDException as e:
        status_code = 404 if isinstance(e, ResourceNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Metrics calculation failed: {str(e)}"
        ) from e


@router.get(
    "/survey/{survey_id}/summary",
    responses={
        200: {"description": "Survey observation summary retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Survey not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_survey_observation_summary(
    survey_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get comprehensive summary of observations for a survey."""
    try:
        from uuid import UUID

        service = ObservationService(db)
        summary = await service.get_survey_observation_summary(UUID(survey_id))

        return create_response(summary)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Summary generation failed: {str(e)}"
        ) from e


@router.post(
    "/validate",
    responses={
        200: {"description": "Observation data is valid"},
        400: {"description": "Validation failed"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def validate_observation_data(
    observation: DomainObservationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Validate observation data without creating the observation."""
    try:
        service = ObservationService(db)
        is_valid = await service.validate_observation_data(observation)

        result = {
            "is_valid": is_valid,
            "observation_id": observation.observation_id,
            "survey_id": str(observation.survey_id),
            "validation_passed": True,
            "message": "Observation data is valid",
        }

        return create_response(result)

    except Exception as e:
        result = {
            "is_valid": False,
            "observation_id": observation.observation_id,
            "survey_id": str(observation.survey_id),
            "validation_passed": False,
            "message": str(e),
            "error_type": type(e).__name__,
        }
        return create_response(result, status_code=400)


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
) -> JSONResponse:
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
    auth=Depends(require_permission_or_api_key(Permission.MANAGE_OPERATIONS)),
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
    auth=Depends(require_permission_or_api_key(Permission.MANAGE_OPERATIONS)),
) -> JSONResponse:
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
    auth=Depends(require_permission_or_api_key(Permission.MANAGE_OPERATIONS)),
) -> JSONResponse:
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


# Survey Integration Endpoints


class SurveySearchRequest(BaseModel):
    """Survey search request model."""

    coordinates: tuple[float, float]
    radius: float = 0.1
    missions: list[str] | None = None
    start_time: str | None = None
    end_time: str | None = None
    limit: int = 1000


class SurveyDownloadRequest(BaseModel):
    """Survey download request model."""

    observation_id: str


class MetadataRequest(BaseModel):
    """Metadata extraction request model."""

    fits_data_url: str | None = None
    observation_id: str | None = None


class IngestionResult(BaseModel):
    """Ingestion result payload for survey ingestion endpoints."""

    survey_id: str
    search_parameters: dict
    total_found: int
    converted: int
    skipped: int
    observations: list[ObservationResponse]
    errors: list[dict]


@router.post(
    "/surveys/{survey_id}/search",
    responses={
        200: {"description": "Survey search completed successfully"},
        400: {"description": "Invalid search parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def search_survey_observations(
    survey_id: str,
    request: SurveySearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Search for observations in external survey APIs."""
    try:
        from src.domains.observations.integrations.mast_client import MASTClient

        # Validate coordinates
        ra, dec = request.coordinates
        if not (0 <= ra <= 360):
            raise HTTPException(
                status_code=400, detail="RA must be between 0 and 360 degrees"
            )
        if not (-90 <= dec <= 90):
            raise HTTPException(
                status_code=400, detail="Dec must be between -90 and 90 degrees"
            )
        if request.radius <= 0:
            raise HTTPException(status_code=400, detail="Radius must be positive")

        # Parse optional datetime strings
        start_time = None
        end_time = None
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)

        # Search using MAST client
        async with MASTClient() as client:
            results = await client.search_observations(
                coordinates=request.coordinates,
                radius=request.radius,
                missions=request.missions,
                start_time=start_time,
                end_time=end_time,
                limit=request.limit,
            )

        # Convert results to response format
        response_data = {
            "survey_id": survey_id,
            "search_parameters": request.model_dump(),
            "results_count": len(results),
            "observations": [result.model_dump() for result in results],
        }

        return create_response(response_data)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Survey search failed: {str(e)}"
        ) from e


@router.post(
    "/surveys/{survey_id}/ingest",
    response_model=ResponseEnvelope[IngestionResult],
    responses={
        200: {"description": "Survey observations ingested successfully"},
        400: {"description": "Invalid ingest parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def ingest_survey_observations(
    survey_id: str,
    request: SurveySearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole | None = Depends(
        require_permission_or_api_key(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Ingest observations from external survey APIs into our database."""
    try:
        from uuid import UUID

        from src.domains.observations.adapters import HSTAdapter, JWSTAdapter
        from src.domains.observations.integrations.mast_client import MASTClient
        from src.domains.observations.service import ObservationService

        # Validate coordinates
        ra, dec = request.coordinates
        if not (0 <= ra <= 360):
            raise HTTPException(
                status_code=400, detail="RA must be between 0 and 360 degrees"
            )
        if not (-90 <= dec <= 90):
            raise HTTPException(
                status_code=400, detail="Dec must be between -90 and 90 degrees"
            )

        service = ObservationService(db)
        survey_uuid = UUID(survey_id)

        # Parse optional datetime strings
        start_time = None
        end_time = None
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)

        # Search and normalize observations
        async with MASTClient() as client:
            mast_results = await client.search_observations(
                coordinates=request.coordinates,
                radius=request.radius,
                missions=request.missions,
                start_time=start_time,
                end_time=end_time,
                limit=request.limit,
            )

        # Normalize and create observations using adapters
        created_observations = []
        conversion_errors: list[dict] = []
        hst_adapter = HSTAdapter()
        jwst_adapter = JWSTAdapter()

        for mast_result in mast_results:
            try:
                # Convert to dict for adapter processing
                raw_data = mast_result.model_dump()

                # Choose appropriate adapter based on mission
                adapter = None
                if mast_result.mission == "HST":
                    adapter = hst_adapter
                elif mast_result.mission == "JWST":
                    adapter = jwst_adapter

                if adapter:
                    # Normalize data using adapter
                    normalized_obs = await adapter.normalize_observation_data(
                        raw_data, survey_uuid
                    )

                    # Create observation in database
                    created_obs = await service.create_observation(normalized_obs)
                    created_observations.append(created_obs)

            except Exception as e:
                # Record error but continue processing other observations
                conversion_errors.append(
                    {
                        "observation_id": mast_result.observation_id,
                        "mission": mast_result.mission,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                # Structured log for troubleshooting
                from src.core.logging import configure_domain_logger

                log = configure_domain_logger("observations.ingest")
                log.warning(
                    "conversion_failed",
                    extra={
                        "observation_id": mast_result.observation_id,
                        "mission": mast_result.mission,
                        "ra": mast_result.ra,
                        "dec": mast_result.dec,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                continue

        # Convert to response format
        response_observations: list[ObservationResponse] = []
        for obs in created_observations:
            response_observations.append(
                ObservationResponse(
                    id=str(obs.id),
                    survey=survey_id,
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
            )

        result = IngestionResult(
            survey_id=survey_id,
            search_parameters=request.model_dump(),
            total_found=len(mast_results),
            converted=len(response_observations),
            skipped=max(len(mast_results) - len(response_observations), 0),
            observations=response_observations,
            errors=conversion_errors,
        )

        return create_response(result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Survey ingestion failed: {str(e)}"
        ) from e


@router.get(
    "/surveys/{survey_id}/observations",
    response_model=ResponseEnvelope[list[ObservationResponse]],
    responses={
        200: {"description": "Survey observations retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Survey not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_survey_observations(
    survey_id: str,
    limit: int = Query(100, le=1000, description="Maximum number of observations"),
    offset: int = Query(0, ge=0, description="Number of observations to skip"),
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole | None = Depends(
        require_permission_or_api_key(Permission.READ_DATA)
    ),
) -> JSONResponse:
    """Get observations for a specific survey."""
    try:
        from uuid import UUID

        from src.domains.observations.service import ObservationService

        service = ObservationService(db)
        survey_uuid = UUID(survey_id)

        # Get survey observations
        from src.domains.observations.schema import ObservationListParams

        params = ObservationListParams(
            survey_id=survey_uuid, limit=limit, offset=offset
        )
        observations, total = await service.list_observations(params)

        # Convert to response format
        response_observations = []
        for obs in observations:
            response_obs = ObservationResponse(
                id=str(obs.id),
                survey=survey_id,
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

        response_data = {
            "observations": response_observations,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

        return create_response(response_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get survey observations: {str(e)}"
        ) from e


@router.get(
    "/surveys/{survey_id}/metadata/{observation_id}",
    responses={
        200: {"description": "Observation metadata retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Observation not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_observation_metadata(
    survey_id: str,
    observation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get detailed metadata for a specific observation."""
    try:
        from src.domains.observations.integrations.mast_client import MASTClient

        # Get metadata from MAST
        async with MASTClient() as client:
            metadata = await client.get_observation_metadata(observation_id)

        response_data = {
            "survey_id": survey_id,
            "observation_id": observation_id,
            "metadata": metadata,
        }

        return create_response(response_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Metadata retrieval failed: {str(e)}"
        ) from e


@router.post(
    "/surveys/{survey_id}/download/{observation_id}",
    responses={
        200: {"description": "Observation data download initiated"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Observation not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def download_observation_data(
    survey_id: str,
    observation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Download observation data from external survey."""
    try:
        from src.domains.observations.extractors import MetadataExtractor
        from src.domains.observations.integrations.mast_client import MASTClient

        # Download data from MAST
        async with MASTClient() as client:
            data = await client.download_observation_data(observation_id)

        # Extract metadata from FITS data
        extractor = MetadataExtractor()

        # Validate FITS file
        if not extractor.validate_fits_file(data):
            raise ValueError("Downloaded data is not a valid FITS file")

        # Extract basic metadata
        metadata = await extractor.extract_all_metadata(data)

        response_data = {
            "survey_id": survey_id,
            "observation_id": observation_id,
            "download_size_bytes": len(data),
            "file_type": "FITS",
            "metadata_extracted": True,
            "metadata": metadata,
            "message": "Observation data downloaded and processed successfully",
        }

        return create_response(response_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}") from e


@router.post(
    "/extract-metadata",
    responses={
        200: {"description": "Metadata extracted successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def extract_fits_metadata(
    request: MetadataRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Extract metadata from FITS data."""
    try:
        from src.domains.observations.extractors import MetadataExtractor
        from src.domains.observations.integrations.mast_client import MASTClient

        extractor = MetadataExtractor()

        if request.fits_data_url:
            # Download FITS data from URL
            async with MASTClient() as client:
                import httpx

                async with httpx.AsyncClient() as http_client:
                    response = await http_client.get(request.fits_data_url)
                    response.raise_for_status()
                    fits_data = response.content

        elif request.observation_id:
            # Download from MAST using observation ID
            async with MASTClient() as client:
                fits_data = await client.download_observation_data(
                    request.observation_id
                )
        else:
            raise ValueError("Either fits_data_url or observation_id must be provided")

        # Validate and extract metadata
        if not extractor.validate_fits_file(fits_data):
            raise ValueError("Data is not a valid FITS file")

        metadata = await extractor.extract_all_metadata(fits_data)

        response_data = {
            "file_size_bytes": len(fits_data),
            "extraction_successful": True,
            "metadata": metadata,
        }

        return create_response(response_data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Metadata extraction failed: {str(e)}"
        ) from e
