"""Stream API routes for Server-Sent Events."""

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.adapters.auth.rbac import (
    Permission,
    UserWithRole,
    require_permission,
)

router = APIRouter()


@router.get("/detections")  # type: ignore[misc]
async def stream_detections(
    request: Request,
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> StreamingResponse:
    """Stream live detection events using Server-Sent Events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for detections."""
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                # In a real implementation, this would:
                # 1. Subscribe to Redis pub/sub channel for detections
                # 2. Stream new detection events as they occur
                # 3. Handle client disconnection gracefully

                # For now, send a heartbeat every 30 seconds
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': '2024-01-01T00:00:00Z'})}\n\n"

                await asyncio.sleep(30)

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
) -> StreamingResponse:
    """Stream system status updates."""

    async def status_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for system status."""
        try:
            while True:
                if await request.is_disconnected():
                    break

                # In a real implementation, this would stream:
                # - Pipeline status updates
                # - Model training progress
                # - System health metrics

                yield f"data: {json.dumps({'type': 'status', 'status': 'healthy', 'timestamp': '2024-01-01T00:00:00Z'})}\n\n"

                await asyncio.sleep(60)  # Status updates every minute

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
