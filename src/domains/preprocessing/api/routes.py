"""Preprocessing API routes."""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.adapters.auth.rbac import (
    Permission,
    UserWithRole,
    require_permission,
)
from src.core.api.response_wrapper import create_response
from src.core.db.session import get_db
from src.domains.preprocessing import schema as preprocessing_schema
from src.domains.preprocessing.calibration.calibration_processor import (
    CalibrationProcessor,
)
from src.domains.preprocessing.service import PreprocessRunService

router = APIRouter()


# PreprocessRun Routes
@router.post(
    "/preprocess-runs",
    responses={
        200: {"description": "Preprocessing run created successfully"},
        400: {"description": "Invalid preprocessing run data"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def create_preprocess_run(
    run: preprocessing_schema.PreprocessRunCreate,
    db=Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Create a new preprocessing run."""
    service = PreprocessRunService(db)
    result = await service.create_preprocess_run(run)
    return create_response(result)


@router.get(
    "/preprocess-runs",
    responses={
        200: {"description": "Preprocessing runs retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def list_preprocess_runs(
    observation_id: str | None = Query(None, description="Filter by observation ID"),
    status: str | None = Query(None, description="Filter by status"),
    algorithm: str | None = Query(None, description="Filter by algorithm"),
    limit: int = Query(100, le=1000, description="Maximum number of runs"),
    offset: int = Query(0, ge=0, description="Number of runs to skip"),
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """List preprocessing runs with optional filtering."""
    service = PreprocessRunService(db)
    result = await service.list_preprocess_runs(
        observation_id=observation_id,
        status=status,
        algorithm=algorithm,
        limit=limit,
        offset=offset,
    )
    return create_response(result)


@router.get(
    "/preprocess-runs/{run_id}",
    responses={
        200: {"description": "Preprocessing run retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Preprocessing run not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_preprocess_run(
    run_id: str,
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get a specific preprocessing run by ID."""
    service = PreprocessRunService(db)
    run = await service.get_preprocess_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Preprocessing run not found")
    return create_response(run)


@router.post(
    "/preprocess-runs/run",
    responses={
        200: {"description": "Preprocessing process initiated successfully"},
        400: {"description": "Invalid parameters or algorithm"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)
async def run_preprocessing(
    observation_id: str = Query(..., description="Observation ID to process"),
    algorithm: str = Query("standard", description="Preprocessing algorithm to use"),
    db=Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
):
    """Run preprocessing algorithm on an observation."""
    service = PreprocessRunService(db)
    result = await service.run_preprocessing(observation_id, algorithm)
    return create_response(result)


# Additional endpoints per ASTR-76


@router.post(
    "/observations/{id}/preprocess",
    responses={200: {"description": "Preprocessing initiated"}},
)
async def preprocess_observation(
    id: str,
    db=Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
):
    service = PreprocessRunService(db)
    result = await service.run_preprocessing(id, algorithm="standard")
    return create_response(result)


@router.get(
    "/observations/{id}/preprocessing-status",
    responses={200: {"description": "Preprocessing status"}},
)
async def preprocessing_status(
    id: str,
    db=Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
):
    service = PreprocessRunService(db)
    runs = await service.list_preprocess_runs(observation_id=id, limit=1)
    return create_response(runs[0] if runs else {"status": "not_found"})


class CalibrationUpload(BaseModel):
    bias_frames: list[list[float]] | None = None
    dark_frames: list[list[float]] | None = None
    flat_frames: list[list[float]] | None = None
    exposure_times: list[float] | None = None


@router.post(
    "/calibration-frames/upload",
    responses={200: {"description": "Calibration frames processed"}},
)
async def upload_calibration_frames(
    payload: CalibrationUpload,
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
):
    cp = CalibrationProcessor()
    result: dict = {}
    if payload.bias_frames:
        bias_list = [np.array(f, dtype=float) for f in payload.bias_frames]  # type: ignore[name-defined]
        master_bias = cp.create_master_bias(bias_list)
        result["master_bias_stats"] = cp.validate_calibration_quality(master_bias)
    if payload.dark_frames and payload.exposure_times:
        dark_list = [np.array(f, dtype=float) for f in payload.dark_frames]  # type: ignore[name-defined]
        master_dark = cp.create_master_dark(dark_list, payload.exposure_times)
        result["master_dark_stats"] = cp.validate_calibration_quality(master_dark)
    if payload.flat_frames and payload.bias_frames:
        flat_list = [np.array(f, dtype=float) for f in payload.flat_frames]  # type: ignore[name-defined]
        bias_list = [np.array(f, dtype=float) for f in payload.bias_frames]  # type: ignore[name-defined]
        master_bias = cp.create_master_bias(bias_list)
        master_flat = cp.create_master_flat(flat_list, master_bias)
        result["master_flat_stats"] = cp.validate_calibration_quality(master_flat)
    return create_response(result)


@router.get(
    "/calibration-frames/{type}/latest",
    responses={200: {"description": "Latest master calibration frame metadata"}},
)
async def latest_calibration_frame(
    type: str,
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
):
    # Placeholder implementation; would fetch from storage/DB
    return create_response({"type": type, "status": "unavailable"})


@router.get(
    "/observations/{id}/quality-metrics",
    responses={200: {"description": "Quality metrics for latest preprocessing"}},
)
async def get_quality_metrics(
    id: str,
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
):
    # Placeholder; would read latest processed result
    return create_response({"observation_id": id, "metrics": {}})


@router.post(
    "/preprocessing/pipeline/configure",
    responses={200: {"description": "Pipeline configured"}},
)
async def configure_pipeline(
    config: dict,
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
):
    # Placeholder acknowledgement
    return create_response({"configured": True, "config_keys": list(config.keys())})
