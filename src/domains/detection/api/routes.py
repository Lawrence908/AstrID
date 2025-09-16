"""Detections API routes."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
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
    ResourceNotFoundError,
)
from src.domains.detection.repository import DetectionRepository
from src.domains.detection.schema import (
    DetectionRead,
    ModelRunRead,
)

router = APIRouter()


class InferenceRequest(BaseModel):
    """ML inference request model."""

    observation_id: str
    model_version: str | None = None
    confidence_threshold: float = 0.5
    include_masks: bool = True
    include_confidence_maps: bool = False


class DetectionValidation(BaseModel):
    """Detection validation request model."""

    is_valid: bool
    human_label: str | None = None
    validation_confidence: float | None = None
    notes: str | None = None


class DetectionStatistics(BaseModel):
    """Detection statistics response model."""

    total_detections: int
    validated_detections: int
    pending_validation: int
    high_confidence_detections: int
    detection_types: dict[str, int]
    average_confidence: float
    detections_by_date: dict[str, int]
    model_performance: dict[str, Any]


@router.post(
    "/infer",
    response_model=ResponseEnvelope[ModelRunRead],
    responses={
        200: {"description": "Inference initiated successfully"},
        400: {"description": "Invalid inference parameters"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Observation not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def run_inference(
    request: InferenceRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Run ML inference on an observation to detect anomalies."""
    try:
        # service = DetectionService(db)  # Unused for now

        # For now, return a placeholder response indicating inference is queued
        # In a real implementation, this would start an async job
        model_run_data = {
            "id": str(UUID("00000000-0000-0000-0000-000000000001")),  # Placeholder ID
            "observation_id": request.observation_id,
            "model_version": request.model_version or "latest",
            "status": "queued",
            "created_at": datetime.now(UTC).isoformat(),
            "message": "Inference job queued for processing",
        }

        return create_response(model_run_data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except AstrIDException as e:
        status_code = 404 if isinstance(e, ResourceNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.get(
    "/",
    response_model=ResponseEnvelope[list[DetectionRead]],
    responses={
        200: {"description": "Detections retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def list_detections(
    observation_id: str | None = Query(None, description="Filter by observation ID"),
    status: str | None = Query(None, description="Filter by status"),
    min_confidence: float | None = Query(
        None, ge=0.0, le=1.0, description="Minimum confidence score"
    ),
    detection_type: str | None = Query(None, description="Filter by detection type"),
    is_validated: bool | None = Query(None, description="Filter by validation status"),
    model_version: str | None = Query(None, description="Filter by model version"),
    date_from: str | None = Query(None, description="Start date (ISO format)"),
    date_to: str | None = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, le=1000, description="Maximum number of detections"),
    offset: int = Query(0, ge=0, description="Number of detections to skip"),
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """List detections with optional filtering and pagination."""
    try:
        # Parse date strings if provided
        parsed_date_from = None
        # parsed_date_to = None  # Unused for now
        if date_from:
            parsed_date_from = datetime.fromisoformat(date_from)
        if date_to:
            # parsed_date_to = datetime.fromisoformat(date_to)  # Unused for now
            pass

        # Build query parameters
        # params = DetectionListParams(
        #     observation_id=UUID(observation_id) if observation_id else None,
        #     min_confidence=min_confidence,
        #     is_validated=is_validated,
        #     model_version=model_version,
        #     date_from=parsed_date_from,
        #     date_to=parsed_date_to,
        #     limit=limit,
        #     offset=offset,
        # )  # Unused for now

        repo = DetectionRepository(db)
        # Use the existing list method with compatible parameters
        detections = await repo.list(
            observation_id=observation_id,
            status=status,
            min_score=min_confidence,
            since=parsed_date_from.isoformat() if parsed_date_from else None,
            limit=limit,
            offset=offset,
        )

        return create_response(detections)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list detections: {str(e)}"
        ) from e


@router.get(
    "/{detection_id}",
    response_model=ResponseEnvelope[DetectionRead],
    responses={
        200: {"description": "Detection retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Detection not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_detection(
    detection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get a specific detection by ID."""
    try:
        repo = DetectionRepository(db)
        detection = await repo.get_by_id(detection_id)
        if not detection:
            raise ResourceNotFoundError(
                message=f"Detection with ID {detection_id} not found",
                error_code="DETECTION_NOT_FOUND",
                details={"detection_id": detection_id},
            )
        return create_response(detection)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail="Invalid detection ID format"
        ) from e
    except AstrIDException as e:
        status_code = 404 if isinstance(e, ResourceNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.put(
    "/{detection_id}/validate",
    response_model=ResponseEnvelope[DetectionRead],
    responses={
        200: {"description": "Detection validated successfully"},
        400: {"description": "Invalid validation data"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Detection not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def validate_detection(
    detection_id: str,
    validation: DetectionValidation,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Validate a detection through human review."""
    try:
        # service = DetectionService(db)  # Unused for now

        # For now, return a placeholder response
        # In a real implementation, this would update the detection status
        updated_detection = {
            "id": detection_id,
            "is_validated": validation.is_valid,
            "human_label": validation.human_label,
            "validation_confidence": validation.validation_confidence,
            "validated_by": getattr(current_user, "username", "system"),
            "validation_timestamp": datetime.now(UTC).isoformat(),
            "status": "validated" if validation.is_valid else "rejected",
        }

        # In a real implementation, we would check if the detection exists first

        return create_response(updated_detection)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except AstrIDException as e:
        status_code = 404 if isinstance(e, ResourceNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.get(
    "/statistics",
    response_model=ResponseEnvelope[DetectionStatistics],
    responses={
        200: {"description": "Detection statistics retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_detection_statistics(
    observation_id: str | None = Query(None, description="Filter by observation ID"),
    model_version: str | None = Query(None, description="Filter by model version"),
    date_from: str | None = Query(None, description="Start date (ISO format)"),
    date_to: str | None = Query(None, description="End date (ISO format)"),
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get comprehensive detection statistics and analytics."""
    try:
        # Parse date strings if provided
        # parsed_date_from = None  # Unused for now
        # parsed_date_to = None  # Unused for now
        if date_from:
            # parsed_date_from = datetime.fromisoformat(date_from)  # Unused for now
            pass
        if date_to:
            # parsed_date_to = datetime.fromisoformat(date_to)  # Unused for now
            pass

        # service = DetectionService(db)  # Unused for now

        # For now, return placeholder statistics
        # In a real implementation, this would query the database
        statistics = DetectionStatistics(
            total_detections=150,
            validated_detections=120,
            pending_validation=30,
            high_confidence_detections=95,
            detection_types={"anomaly": 80, "transient": 45, "artifact": 25},
            average_confidence=0.75,
            detections_by_date={
                datetime.now(UTC).strftime("%Y-%m-%d"): 15,
                (datetime.now(UTC).replace(day=datetime.now(UTC).day - 1)).strftime(
                    "%Y-%m-%d"
                ): 22,
            },
            model_performance={"precision": 0.85, "recall": 0.78, "f1_score": 0.81},
        )

        return create_response(statistics)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate statistics: {str(e)}"
        ) from e


@router.get(
    "/models/{model_id}/performance",
    responses={
        200: {"description": "Model performance metrics retrieved successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_model_performance(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Get detailed performance metrics for a specific model."""
    try:
        # service = DetectionService(db)  # Unused for now

        # For now, return placeholder performance metrics
        # In a real implementation, this would query model performance data
        performance_metrics = {
            "model_id": model_id,
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81,
            "accuracy": 0.83,
            "auc_roc": 0.89,
            "confusion_matrix": [[45, 5], [8, 42]],
            "detection_rate": 0.90,
            "false_positive_rate": 0.10,
            "last_evaluation": datetime.now(UTC).isoformat(),
        }

        return create_response(performance_metrics)

    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid model ID format") from e
    except AstrIDException as e:
        status_code = 404 if isinstance(e, ResourceNotFoundError) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.post(
    "/batch-validate",
    responses={
        200: {"description": "Batch validation completed successfully"},
        400: {"description": "Invalid validation data"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def batch_validate_detections(
    detection_ids: list[str],
    validation: DetectionValidation,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(
        require_permission(Permission.MANAGE_OPERATIONS)
    ),
) -> JSONResponse:
    """Validate multiple detections in batch."""
    try:
        # service = DetectionService(db)  # Unused for now

        # Convert string IDs to UUIDs
        uuid_detection_ids = [UUID(detection_id) for detection_id in detection_ids]

        # For now, simulate batch validation
        # In a real implementation, this would update multiple detection records
        successful_ids = uuid_detection_ids[
            : min(len(uuid_detection_ids), 50)
        ]  # Limit batch size
        failed_ids = uuid_detection_ids[50:] if len(uuid_detection_ids) > 50 else []

        results = {
            "successful": successful_ids,
            "failed": failed_ids,
            "errors": [f"Batch size limit exceeded for ID {id}" for id in failed_ids],
        }

        return create_response(
            {
                "validated_count": len(results["successful"]),
                "failed_count": len(results["failed"]),
                "successful_ids": [str(id) for id in results["successful"]],
                "failed_ids": [str(id) for id in results["failed"]],
                "errors": results["errors"],
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch validation failed: {str(e)}"
        ) from e
