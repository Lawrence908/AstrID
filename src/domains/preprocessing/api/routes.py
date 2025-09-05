"""Preprocessing API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.core.db.session import get_db
from src.domains.preprocessing.service import PreprocessRunService

router = APIRouter()


# PreprocessRun Models
class PreprocessRunCreate(BaseModel):
    """Create preprocessing run request model."""

    observation_id: str
    algorithm: str
    parameters: dict


class PreprocessRunResponse(BaseModel):
    """Preprocessing run response model."""

    id: str
    observation_id: str
    algorithm: str
    parameters: dict
    status: str
    created_at: str
    updated_at: str


# PreprocessRun Routes
@router.post("/preprocess-runs", response_model=PreprocessRunResponse)
async def create_preprocess_run(run: PreprocessRunCreate, db=Depends(get_db)):
    """Create a new preprocessing run."""
    service = PreprocessRunService(db)
    return await service.create_preprocess_run(run)


@router.get("/preprocess-runs", response_model=list[PreprocessRunResponse])
async def list_preprocess_runs(
    observation_id: str | None = Query(None, description="Filter by observation ID"),
    status: str | None = Query(None, description="Filter by status"),
    algorithm: str | None = Query(None, description="Filter by algorithm"),
    limit: int = Query(100, le=1000, description="Maximum number of runs"),
    offset: int = Query(0, ge=0, description="Number of runs to skip"),
    db=Depends(get_db),
):
    """List preprocessing runs with optional filtering."""
    service = PreprocessRunService(db)
    return await service.list_preprocess_runs(
        observation_id=observation_id,
        status=status,
        algorithm=algorithm,
        limit=limit,
        offset=offset,
    )


@router.get("/preprocess-runs/{run_id}", response_model=PreprocessRunResponse)
async def get_preprocess_run(run_id: str, db=Depends(get_db)):
    """Get a specific preprocessing run by ID."""
    service = PreprocessRunService(db)
    run = await service.get_preprocess_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Preprocessing run not found")
    return run


@router.post("/preprocess-runs/run")
async def run_preprocessing(
    observation_id: str = Query(..., description="Observation ID to process"),
    algorithm: str = Query("standard", description="Preprocessing algorithm to use"),
    db=Depends(get_db),
):
    """Run preprocessing algorithm on an observation."""
    service = PreprocessRunService(db)
    return await service.run_preprocessing(observation_id, algorithm)
