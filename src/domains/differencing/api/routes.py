"""Differencing API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.core.db.session import get_db
from src.core.exceptions import (
    AstrIDException,
    CandidateNotFoundError,
    DifferenceRunNotFoundError,
    DifferencingFailedError,
    InvalidAlgorithmError,
    ValidationError,
)
from src.domains.differencing.service import CandidateService, DifferenceRunService

router = APIRouter()


# DifferenceRun Models
class DifferenceRunCreate(BaseModel):
    """Create difference run request model."""

    observation_id: str
    algorithm: str
    parameters: dict


class DifferenceRunResponse(BaseModel):
    """Difference run response model."""

    id: str
    observation_id: str
    algorithm: str
    parameters: dict
    status: str
    created_at: str
    updated_at: str


# Candidate Models
class CandidateCreate(BaseModel):
    """Create candidate request model."""

    difference_run_id: str
    ra: float
    dec: float
    candidate_type: str
    metadata: dict


class CandidateResponse(BaseModel):
    """Candidate response model."""

    id: str
    difference_run_id: str
    ra: float
    dec: float
    candidate_type: str
    metadata: dict
    status: str
    created_at: str
    updated_at: str


# DifferenceRun Routes
@router.post("/difference-runs", response_model=DifferenceRunResponse)
async def create_difference_run(run: DifferenceRunCreate, db=Depends(get_db)):
    """Create a new difference run."""
    try:
        service = DifferenceRunService(db)
        return await service.create_difference_run(run)
    except AstrIDException as e:
        status_code = (
            400 if isinstance(e, ValidationError | InvalidAlgorithmError) else 500
        )
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.get("/difference-runs", response_model=list[DifferenceRunResponse])
async def list_difference_runs(
    observation_id: str | None = Query(None, description="Filter by observation ID"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, le=1000, description="Maximum number of runs"),
    offset: int = Query(0, ge=0, description="Number of runs to skip"),
    db=Depends(get_db),
):
    """List difference runs with optional filtering."""
    service = DifferenceRunService(db)
    return await service.list_difference_runs(
        observation_id=observation_id, status=status, limit=limit, offset=offset
    )


@router.get("/difference-runs/{run_id}", response_model=DifferenceRunResponse)
async def get_difference_run(run_id: str, db=Depends(get_db)):
    """Get a specific difference run by ID."""
    try:
        service = DifferenceRunService(db)
        run = await service.get_difference_run(run_id)
        if not run:
            raise DifferenceRunNotFoundError(
                message=f"Difference run with ID {run_id} not found",
                error_code="DIFFERENCE_RUN_NOT_FOUND",
                details={"run_id": run_id},
            )
        return run
    except AstrIDException as e:
        status_code = 404 if isinstance(e, DifferenceRunNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.post("/difference-runs/run")
async def run_differencing(
    observation_id: str = Query(..., description="Observation ID to process"),
    algorithm: str = Query("zogy", description="Differencing algorithm to use"),
    db=Depends(get_db),
):
    """Run differencing algorithm on an observation."""
    try:
        service = DifferenceRunService(db)
        return await service.run_differencing(observation_id, algorithm)
    except AstrIDException as e:
        status_code = (
            400
            if isinstance(e, ValidationError | InvalidAlgorithmError)
            else 500
            if isinstance(e, DifferencingFailedError)
            else 500
        )
        raise HTTPException(status_code=status_code, detail=str(e)) from e


# Candidate Routes
@router.post("/candidates", response_model=CandidateResponse)
async def create_candidate(candidate: CandidateCreate, db=Depends(get_db)):
    """Create a new candidate."""
    service = CandidateService(db)
    return await service.create_candidate(candidate)


@router.get("/candidates", response_model=list[CandidateResponse])
async def list_candidates(
    difference_run_id: str | None = Query(
        None, description="Filter by difference run ID"
    ),
    candidate_type: str | None = Query(None, description="Filter by candidate type"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, le=1000, description="Maximum number of candidates"),
    offset: int = Query(0, ge=0, description="Number of candidates to skip"),
    db=Depends(get_db),
):
    """List candidates with optional filtering."""
    service = CandidateService(db)
    return await service.list_candidates(
        difference_run_id=difference_run_id,
        candidate_type=candidate_type,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get("/candidates/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(candidate_id: str, db=Depends(get_db)):
    """Get a specific candidate by ID."""
    try:
        service = CandidateService(db)
        candidate = await service.get_candidate(candidate_id)
        if not candidate:
            raise CandidateNotFoundError(
                message=f"Candidate with ID {candidate_id} not found",
                error_code="CANDIDATE_NOT_FOUND",
                details={"candidate_id": candidate_id},
            )
        return candidate
    except AstrIDException as e:
        status_code = 404 if isinstance(e, CandidateNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e
