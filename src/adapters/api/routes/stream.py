"""Stream API routes for Server-Sent Events."""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.auth.rbac import (
    Permission,
    UserWithRole,
    require_permission,
)
from src.core.db.session import get_db

router = APIRouter()


@router.get("/detections")  # type: ignore[misc]
async def stream_detections(
    request: Request,
    observation_id: str | None = Query(None, description="Filter by observation ID"),
    min_confidence: float | None = Query(
        None, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    model_version: str | None = Query(None, description="Filter by model version"),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream live detection events using Server-Sent Events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for detections."""
        try:
            # last_check = datetime.now(UTC)  # Unused for now

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    # For now, simulate new detections
                    # In a real implementation, this would query for new detections
                    import random

                    new_detections = []

                    # Occasionally generate a simulated detection
                    if random.random() < 0.1:  # 10% chance of new detection
                        simulated_detection_data = {
                            "id": "12345678-1234-5678-9012-123456789012",
                            "observation_id": observation_id
                            if observation_id
                            else "87654321-4321-8765-2109-876543210987",
                            "ra": random.uniform(0, 360),
                            "dec": random.uniform(-90, 90),
                            "confidence_score": random.uniform(0.5, 1.0),
                            "detection_type": "anomaly",
                            "status": "pending",
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                        new_detections = [simulated_detection_data]

                    # Send new detections as SSE events
                    for detection in new_detections:
                        event_data = {
                            "type": "detection",
                            "data": detection,  # detection is already a dict with the right structure
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"

                    # Update last check time
                    # last_check = datetime.now(UTC)  # Unused for now

                    # Send heartbeat if no new detections
                    if not new_detections:
                        heartbeat = {
                            "type": "heartbeat",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "status": "alive",
                        }
                        yield f"data: {json.dumps(heartbeat)}\n\n"

                except Exception as e:
                    # Send error event but continue streaming
                    error_event = {
                        "type": "error",
                        "message": f"Error fetching detections: {str(e)}",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"

                await asyncio.sleep(5)  # Check for new detections every 5 seconds

        except asyncio.CancelledError:
            # Client disconnected
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@router.get("/status")  # type: ignore[misc]
async def stream_status(
    request: Request,
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream system status updates including processing and health metrics."""

    async def status_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for system status."""
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    # For now, simulate system status
                    # In a real implementation, this would query actual system metrics
                    import random

                    processing_stats = {
                        "pending_count": random.randint(0, 10),
                        "processing_count": random.randint(0, 5),
                        "completed_today": random.randint(50, 200),
                        "failed_today": random.randint(0, 5),
                    }

                    detection_stats = {
                        "total_today": random.randint(10, 50),
                        "high_confidence_today": random.randint(5, 30),
                        "pending_validation": random.randint(0, 15),
                        "average_confidence": round(random.uniform(0.6, 0.9), 2),
                        "active_models": 3,
                        "inference_queue": random.randint(0, 8),
                    }

                    # Build status update
                    status_update = {
                        "type": "status",
                        "data": {
                            "system_health": "healthy",  # This would check actual health
                            "processing": {
                                "pending_observations": processing_stats.get(
                                    "pending_count", 0
                                ),
                                "processing_observations": processing_stats.get(
                                    "processing_count", 0
                                ),
                                "completed_today": processing_stats.get(
                                    "completed_today", 0
                                ),
                                "failed_today": processing_stats.get("failed_today", 0),
                            },
                            "detections": {
                                "total_today": detection_stats.get("total_today", 0),
                                "high_confidence_today": detection_stats.get(
                                    "high_confidence_today", 0
                                ),
                                "pending_validation": detection_stats.get(
                                    "pending_validation", 0
                                ),
                                "average_confidence": detection_stats.get(
                                    "average_confidence", 0.0
                                ),
                            },
                            "models": {
                                "active_models": detection_stats.get(
                                    "active_models", 0
                                ),
                                "inference_queue": detection_stats.get(
                                    "inference_queue", 0
                                ),
                            },
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

                    yield f"data: {json.dumps(status_update)}\n\n"

                except Exception as e:
                    # Send error event but continue streaming
                    error_event = {
                        "type": "error",
                        "message": f"Error fetching system status: {str(e)}",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"

                await asyncio.sleep(30)  # Status updates every 30 seconds

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        status_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@router.get("/health")  # type: ignore[misc]
async def stream_health_monitoring(
    request: Request,
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream real-time health monitoring updates."""

    async def health_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for system health monitoring."""
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    # In a real implementation, this would check:
                    # - Database connectivity
                    # - External service availability
                    # - Memory/CPU usage
                    # - Queue depths
                    # - Error rates

                    health_data = {
                        "type": "health",
                        "data": {
                            "overall_status": "healthy",
                            "services": {
                                "database": "healthy",
                                "redis": "healthy",
                                "storage": "healthy",
                                "ml_models": "healthy",
                            },
                            "metrics": {
                                "cpu_usage": 45.2,
                                "memory_usage": 67.8,
                                "disk_usage": 34.1,
                                "active_connections": 12,
                                "queue_depth": 3,
                            },
                            "uptime_seconds": 86400,  # Placeholder
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

                    yield f"data: {json.dumps(health_data)}\n\n"

                except Exception as e:
                    # Send error event but continue streaming
                    error_event = {
                        "type": "error",
                        "message": f"Error fetching health data: {str(e)}",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"

                await asyncio.sleep(10)  # Health updates every 10 seconds

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        health_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@router.get("/processing")  # type: ignore[misc]
async def stream_processing_updates(
    request: Request,
    observation_id: str | None = Query(
        None, description="Filter by specific observation"
    ),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream processing pipeline status updates."""

    async def processing_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for processing updates."""
        try:
            # last_check = datetime.now(UTC)  # Unused for now

            while True:
                if await request.is_disconnected():
                    break

                try:
                    # For now, simulate processing updates
                    # In a real implementation, this would query for actual updates
                    import random

                    updates = []

                    # Occasionally generate a simulated processing update
                    if random.random() < 0.15:  # 15% chance of update
                        simulated_update_data = {
                            "observation_id": observation_id
                            if observation_id
                            else "12345678-1234-5678-9012-123456789012",
                            "status": random.choice(
                                ["processing", "completed", "failed"]
                            ),
                            "processing_stage": random.choice(
                                [
                                    "preprocessing",
                                    "differencing",
                                    "detection",
                                    "validation",
                                ]
                            ),
                            "progress_percentage": random.randint(0, 100),
                            "message": "Processing update",
                            "updated_at": datetime.now(UTC).isoformat(),
                        }
                        updates = [simulated_update_data]

                    # Send processing updates as SSE events
                    for update in updates:
                        event_data = {
                            "type": "processing_update",
                            "data": {
                                "observation_id": update["observation_id"],
                                "status": update["status"],
                                "stage": update["processing_stage"],
                                "progress": update["progress_percentage"],
                                "message": update["message"],
                                "updated_at": update["updated_at"],
                            },
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"

                    # Update last check time
                    # last_check = datetime.now(UTC)  # Unused for now

                    # Send heartbeat if no updates
                    if not updates:
                        heartbeat = {
                            "type": "heartbeat",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "status": "monitoring",
                        }
                        yield f"data: {json.dumps(heartbeat)}\n\n"

                except Exception as e:
                    error_event = {
                        "type": "error",
                        "message": f"Error fetching processing updates: {str(e)}",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"

                await asyncio.sleep(3)  # Check every 3 seconds for processing updates

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        processing_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )
