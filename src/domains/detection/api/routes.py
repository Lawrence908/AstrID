"""Detections API routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db.session import get_db
from src.domains.detection.repository import DetectionRepository

router = APIRouter()


class DetectionCreate(BaseModel):
    """Create detection request model."""

    observation_id: str
    ra: float
    dec: float
    confidence_score: float
    detection_type: str
    metadata: dict


class DetectionResponse(BaseModel):
    """Detection response model."""

    id: str
    observation_id: str
    ra: float
    dec: float
    confidence_score: float
    detection_type: str
    status: str
    metadata: dict
    created_at: str
    updated_at: str


class DetectionValidation(BaseModel):
    """Detection validation request model."""

    is_valid: bool
    label: str | None = None
    notes: str | None = None


@router.post("/infer")  # type: ignore[misc]
async def run_inference(
    observation_id: str = Query(..., description="Observation ID to analyze"),
    model_version: str | None = Query(None, description="Model version to use"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Run inference on an observation to detect anomalies."""
    # This would typically enqueue a background job
    # For now, return a placeholder response
    return {
        "message": f"Inference initiated for observation {observation_id}",
        "observation_id": observation_id,
        "model_version": model_version,
        "status": "queued",
    }


@router.get("/", response_model=list[DetectionResponse])  # type: ignore[misc]
async def list_detections(
    observation_id: str | None = Query(None, description="Filter by observation ID"),
    status: str | None = Query(None, description="Filter by status"),
    min_score: float | None = Query(
        None, ge=0.0, le=1.0, description="Minimum confidence score"
    ),
    since: str | None = Query(None, description="Filter by creation date (ISO format)"),
    limit: int = Query(100, le=1000, description="Maximum number of detections"),
    offset: int = Query(0, ge=0, description="Number of detections to skip"),
    db: AsyncSession = Depends(get_db),
) -> list[DetectionResponse]:
    """List detections with optional filtering."""
    repo = DetectionRepository(db)
    return await repo.list(  # type: ignore[no-any-return]
        observation_id=observation_id,
        status=status,
        min_score=min_score,
        since=since,
        limit=limit,
        offset=offset,
    )


@router.get("/{detection_id}", response_model=DetectionResponse)  # type: ignore[misc]
async def get_detection(
    detection_id: str, db: AsyncSession = Depends(get_db)
) -> DetectionResponse:
    """Get a specific detection by ID."""
    repo = DetectionRepository(db)
    detection = await repo.get_by_id(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection  # type: ignore[no-any-return]


@router.post("/{detection_id}/validate")  # type: ignore[misc]
async def validate_detection(
    detection_id: str,
    validation: DetectionValidation,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Validate a detection (human review)."""
    # This would update the detection status and create a validation event
    # repo = DetectionRepository(db)  # TODO: Implement validation logic
    return {
        "message": f"Detection {detection_id} validation recorded",
        "detection_id": detection_id,
        "is_valid": validation.is_valid,
        "label": validation.label,
        "status": "validated",
    }
