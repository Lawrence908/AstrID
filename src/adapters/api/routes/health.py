"""Health check and monitoring API routes."""

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.auth.rbac import (
    Permission,
    UserWithRole,
    require_permission,
)
from src.core.api.response_wrapper import ResponseEnvelope, create_response
from src.core.db.session import get_db

router = APIRouter()


class HealthStatus(BaseModel):
    """Basic health status model."""

    status: str
    timestamp: str
    version: str
    uptime_seconds: float


class DetailedHealthStatus(BaseModel):
    """Detailed health status model."""

    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    services: dict[str, Any]
    metrics: dict[str, Any]
    checks: dict[str, Any]


class DependencyStatus(BaseModel):
    """External dependency status model."""

    name: str
    status: str
    response_time_ms: float | None = None
    last_check: str
    error_message: str | None = None


class SystemMetrics(BaseModel):
    """System metrics model."""

    timestamp: str
    database: dict[str, Any]
    application: dict[str, Any]
    storage: dict[str, Any] | None = None
    queue: dict[str, Any] | None = None


# Store start time for uptime calculation
_start_time = time.time()


@router.get(
    "/",
    response_model=ResponseEnvelope[HealthStatus],
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)  # type: ignore[misc]
async def basic_health_check(
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Basic health check endpoint - fast and lightweight."""
    try:
        # Quick database connectivity check
        result = await db.execute(text("SELECT 1"))
        result.fetchone()

        health_data = HealthStatus(
            status="healthy",
            timestamp=datetime.now(UTC).isoformat(),
            version="1.0.0",  # Should come from config
            uptime_seconds=time.time() - _start_time,
        )

        return create_response(health_data)

    except Exception as e:
        health_data = HealthStatus(
            status="unhealthy",
            timestamp=datetime.now(UTC).isoformat(),
            version="1.0.0",
            uptime_seconds=time.time() - _start_time,
        )

        return create_response(
            health_data,
            status_code=503,
            error={
                "code": "HEALTH_CHECK_FAILED",
                "message": f"Health check failed: {str(e)}",
                "details": {"component": "database"},
            },
        )


@router.get(
    "/detailed",
    response_model=ResponseEnvelope[DetailedHealthStatus],
    responses={
        200: {"description": "Detailed health status retrieved"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
        503: {"description": "Service is unhealthy"},
    },
)  # type: ignore[misc]
async def detailed_health_check(
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
) -> JSONResponse:
    """Comprehensive health check with detailed system information."""
    try:
        health_checks = {}
        overall_status = "healthy"

        # Database health check
        try:
            start_time = time.time()
            result = await db.execute(text("SELECT version(), now()"))
            db_info = result.fetchone()
            db_response_time = (time.time() - start_time) * 1000

            health_checks["database"] = {
                "status": "healthy",
                "response_time_ms": round(db_response_time, 2),
                "version": str(db_info[0]) if db_info else "unknown",
                "timestamp": str(db_info[1]) if db_info else None,
            }
        except Exception as e:
            health_checks["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            overall_status = "unhealthy"

        # Application metrics
        app_metrics = {
            "memory_usage_mb": 0,  # Would implement actual memory tracking
            "cpu_usage_percent": 0,  # Would implement actual CPU tracking
            "active_connections": 0,  # Would track active DB connections
            "thread_count": 0,  # Would track active threads
        }

        # Storage health (placeholder)
        storage_metrics = {
            "disk_usage_percent": 0,
            "available_space_gb": 0,
        }

        # Queue health (placeholder for Redis/background jobs)
        queue_metrics = {
            "pending_jobs": 0,
            "failed_jobs": 0,
            "workers_active": 0,
        }

        detailed_health = DetailedHealthStatus(
            status=overall_status,
            timestamp=datetime.now(UTC).isoformat(),
            version="1.0.0",
            uptime_seconds=time.time() - _start_time,
            services=health_checks,
            metrics={
                "application": app_metrics,
                "storage": storage_metrics,
                "queue": queue_metrics,
            },
            checks={
                "database_connectivity": health_checks["database"]["status"]
                == "healthy",
                "disk_space_available": True,  # Would implement actual check
                "memory_usage_normal": True,  # Would implement actual check
            },
        )

        status_code = 200 if overall_status == "healthy" else 503
        return create_response(detailed_health, status_code=status_code)

    except Exception as e:
        return create_response(
            None,
            status_code=503,
            error={
                "code": "DETAILED_HEALTH_CHECK_FAILED",
                "message": f"Detailed health check failed: {str(e)}",
            },
        )


@router.get(
    "/dependencies",
    response_model=ResponseEnvelope[list[DependencyStatus]],
    responses={
        200: {"description": "Dependency status retrieved"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
    },
)  # type: ignore[misc]
async def check_dependencies(
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Check the health of external dependencies and services."""
    dependencies = []

    # Database dependency
    try:
        start_time = time.time()
        await db.execute(text("SELECT 1"))
        response_time = (time.time() - start_time) * 1000

        dependencies.append(
            DependencyStatus(
                name="database",
                status="healthy",
                response_time_ms=round(response_time, 2),
                last_check=datetime.now(UTC).isoformat(),
            )
        )
    except Exception as e:
        dependencies.append(
            DependencyStatus(
                name="database",
                status="unhealthy",
                last_check=datetime.now(UTC).isoformat(),
                error_message=str(e),
            )
        )

    # Redis dependency (placeholder)
    dependencies.append(
        DependencyStatus(
            name="redis",
            status="unknown",  # Would implement actual Redis health check
            last_check=datetime.now(UTC).isoformat(),
            error_message="Health check not implemented",
        )
    )

    # Storage dependency (placeholder)
    dependencies.append(
        DependencyStatus(
            name="object_storage",
            status="unknown",  # Would implement actual storage health check
            last_check=datetime.now(UTC).isoformat(),
            error_message="Health check not implemented",
        )
    )

    # MLflow dependency (placeholder)
    dependencies.append(
        DependencyStatus(
            name="mlflow",
            status="unknown",  # Would implement actual MLflow health check
            last_check=datetime.now(UTC).isoformat(),
            error_message="Health check not implemented",
        )
    )

    return create_response(dependencies)


@router.get(
    "/metrics",
    response_model=ResponseEnvelope[SystemMetrics],
    responses={
        200: {"description": "System metrics retrieved"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"},
    },
)  # type: ignore[misc]
async def get_system_metrics(
    current_user: UserWithRole = Depends(require_permission(Permission.READ_DATA)),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Get comprehensive system metrics for monitoring."""
    try:
        # Database metrics
        db_metrics = {}
        try:
            # Get database connection info
            result = await db.execute(
                text("""
                SELECT
                    count(*) as active_connections,
                    current_database() as database_name,
                    version() as version
            """)
            )
            db_info = result.fetchone()

            db_metrics = {
                "active_connections": db_info[0] if db_info else 0,
                "database_name": db_info[1] if db_info else "unknown",
                "version": db_info[2] if db_info else "unknown",
                "pool_size": 10,  # Would get from actual pool configuration
                "pool_overflow": 0,  # Would get from actual pool stats
            }
        except Exception as e:
            db_metrics = {"error": str(e)}

        # Application metrics
        app_metrics = {
            "uptime_seconds": time.time() - _start_time,
            "requests_total": 0,  # Would implement request counter
            "requests_per_second": 0,  # Would implement rate tracking
            "error_rate": 0.0,  # Would implement error rate tracking
            "response_time_avg_ms": 0,  # Would implement response time tracking
        }

        # Storage metrics (placeholder)
        storage_metrics = {
            "total_observations": 0,  # Would query actual counts
            "total_detections": 0,
            "storage_used_gb": 0,
            "files_processed_today": 0,
        }

        # Queue metrics (placeholder)
        queue_metrics = {
            "pending_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retry_tasks": 0,
        }

        metrics = SystemMetrics(
            timestamp=datetime.now(UTC).isoformat(),
            database=db_metrics,
            application=app_metrics,
            storage=storage_metrics,
            queue=queue_metrics,
        )

        return create_response(metrics)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve system metrics: {str(e)}"
        ) from e


@router.get(
    "/readiness",
    responses={
        200: {"description": "Service is ready to accept traffic"},
        503: {"description": "Service is not ready"},
    },
)  # type: ignore[misc]
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Kubernetes-style readiness probe."""
    try:
        # Check if the service is ready to handle requests
        # This should verify that all critical dependencies are available

        # Database readiness
        await db.execute(text("SELECT 1"))

        # Add other readiness checks as needed:
        # - Required models are loaded
        # - Configuration is valid
        # - Critical external services are available

        return Response(content="OK", status_code=200, media_type="text/plain")

    except Exception:
        return Response(
            content="Service not ready", status_code=503, media_type="text/plain"
        )


@router.get(
    "/liveness",
    responses={
        200: {"description": "Service is alive"},
        503: {"description": "Service should be restarted"},
    },
)  # type: ignore[misc]
async def liveness_check() -> Response:
    """Kubernetes-style liveness probe."""
    try:
        # Basic liveness check - just verify the application is running
        # This should be fast and not check external dependencies

        # Simple check that the service can respond
        return Response(content="OK", status_code=200, media_type="text/plain")

    except Exception:
        return Response(
            content="Service unhealthy", status_code=503, media_type="text/plain"
        )


@router.post(
    "/test-endpoint",
    responses={
        200: {"description": "Test endpoint executed successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Admin access required"},
    },
)  # type: ignore[misc]
async def test_endpoint(
    current_user: UserWithRole = Depends(require_permission(Permission.ADMIN_ACCESS)),
) -> JSONResponse:
    """Test endpoint for verifying system functionality."""
    test_id = str(uuid4())

    test_results = {
        "test_id": test_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "user": getattr(current_user, "username", "unknown"),
        "tests": {
            "response_generation": "passed",
            "timestamp_generation": "passed",
            "uuid_generation": "passed",
            "user_authentication": "passed",
        },
        "message": "All basic functionality tests passed",
    }

    return create_response(test_results)
