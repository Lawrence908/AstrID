"""API routes for survey configuration management."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.auth.rbac import Permission, UserWithRole, require_permission
from src.core.api.response_wrapper import (
    JSONResponse,
    ResponseEnvelope,
    create_response,
)
from src.core.db.session import get_db
from src.domains.observations.survey_config import (
    SurveyConfigurationCreate,
    SurveyConfigurationRead,
    SurveyConfigurationUpdate,
)
from src.domains.observations.survey_config_service import SurveyConfigurationService

router = APIRouter(prefix="/survey-configurations", tags=["survey-configurations"])


@router.post(
    "/",
    response_model=ResponseEnvelope[SurveyConfigurationRead],
    responses={
        200: {"description": "Survey configuration created successfully"},
        400: {"description": "Invalid configuration data"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        409: {"description": "Configuration already exists"},
        500: {"description": "Internal server error"},
    },
)
async def create_survey_configuration(
    config_data: SurveyConfigurationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Create a new survey configuration."""
    try:
        service = SurveyConfigurationService(db)
        result = await service.create_configuration(config_data)
        return create_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create configuration: {str(e)}"
        ) from e


@router.get(
    "/",
    response_model=ResponseEnvelope[list[SurveyConfigurationRead]],
    responses={
        200: {"description": "Survey configurations retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def list_survey_configurations(
    is_active: bool | None = Query(None, description="Filter by active status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """List survey configurations."""
    try:
        service = SurveyConfigurationService(db)
        results = await service.list_configurations(
            is_active=is_active,
            limit=limit,
            offset=offset,
        )
        return create_response(results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list configurations: {str(e)}"
        ) from e


@router.get(
    "/{config_id}",
    response_model=ResponseEnvelope[SurveyConfigurationRead],
    responses={
        200: {"description": "Survey configuration retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Configuration not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_survey_configuration(
    config_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get a survey configuration by ID."""
    try:
        service = SurveyConfigurationService(db)
        result = await service.get_configuration(config_id)
        if not result:
            raise HTTPException(status_code=404, detail="Configuration not found")
        return create_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get configuration: {str(e)}"
        ) from e


@router.get(
    "/by-name/{name}",
    response_model=ResponseEnvelope[SurveyConfigurationRead],
    responses={
        200: {"description": "Survey configuration retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Configuration not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_survey_configuration_by_name(
    name: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get a survey configuration by name."""
    try:
        service = SurveyConfigurationService(db)
        result = await service.get_configuration_by_name(name)
        if not result:
            raise HTTPException(status_code=404, detail="Configuration not found")
        return create_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get configuration: {str(e)}"
        ) from e


@router.put(
    "/{config_id}",
    response_model=ResponseEnvelope[SurveyConfigurationRead],
    responses={
        200: {"description": "Survey configuration updated successfully"},
        400: {"description": "Invalid configuration data"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Configuration not found"},
        500: {"description": "Internal server error"},
    },
)
async def update_survey_configuration(
    config_id: UUID,
    update_data: SurveyConfigurationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Update an existing survey configuration."""
    try:
        service = SurveyConfigurationService(db)
        result = await service.update_configuration(config_id, update_data)
        if not result:
            raise HTTPException(status_code=404, detail="Configuration not found")
        return create_response(result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update configuration: {str(e)}"
        ) from e


@router.delete(
    "/{config_id}",
    response_model=ResponseEnvelope[dict[str, str]],
    responses={
        200: {"description": "Survey configuration deleted successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Configuration not found"},
        500: {"description": "Internal server error"},
    },
)
async def delete_survey_configuration(
    config_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Delete a survey configuration."""
    try:
        service = SurveyConfigurationService(db)
        success = await service.delete_configuration(config_id)
        if not success:
            raise HTTPException(status_code=404, detail="Configuration not found")
        return create_response({"message": "Configuration deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete configuration: {str(e)}"
        ) from e


@router.get(
    "/active/list",
    response_model=ResponseEnvelope[list[SurveyConfigurationRead]],
    responses={
        200: {"description": "Active survey configurations retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def get_active_survey_configurations(
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get all active survey configurations."""
    try:
        service = SurveyConfigurationService(db)
        results = await service.get_active_configurations()
        return create_response(results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get active configurations: {str(e)}"
        ) from e


@router.post(
    "/initialize-defaults",
    response_model=ResponseEnvelope[list[SurveyConfigurationRead]],
    responses={
        200: {"description": "Default configurations initialized successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def initialize_default_configurations(
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Initialize default survey configurations."""
    try:
        service = SurveyConfigurationService(db)
        results = await service.initialize_default_configurations()
        return create_response(results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize defaults: {str(e)}"
        ) from e
