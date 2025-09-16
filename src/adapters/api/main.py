"""Main FastAPI application for AstrID API."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.api.routes import auth, health, storage, stream
from src.core.api.response_wrapper import create_response
from src.core.constants import (
    API_DESCRIPTION,
    API_TITLE,
    APP_VERSION,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ORIGINS,
    ENVIRONMENT,
)
from src.core.exceptions import AstrIDException
from src.domains.catalog.api.routes import router as catalog_router
from src.domains.curation.api.routes import router as curation_router
from src.domains.detection.api.routes import router as detections_router
from src.domains.differencing.api.routes import router as differencing_router
from src.domains.observations.api.routes import router as observations_router
from src.domains.preprocessing.api.routes import router as preprocessing_router

# Setup logging
logger = logging.getLogger(__name__)


# Lifespan context manager for proper startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan with proper startup/shutdown."""
    # Startup
    logger.info("Starting AstrID API...")
    try:
        # Add any startup logic here (database connections, background services, etc.)
        logger.info("AstrID API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start AstrID API: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down AstrID API...")
    try:
        # Add any cleanup logic here (close connections, stop services, etc.)
        logger.info("AstrID API shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during AstrID API shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    swagger_ui_parameters={
        "persistAuthorization": True,
        "displayRequestDuration": True,
        "filter": True,
        "syntaxHighlight": {"theme": "monokai"},
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)


# Exception handlers
@app.exception_handler(AstrIDException)  # type: ignore[misc]
async def astrid_exception_handler(
    request: Request, exc: AstrIDException
) -> JSONResponse:
    """Handle AstrID custom exceptions."""
    from src.core.db.exceptions import (
        CheckConstraintViolationError,
        ColumnNotFoundError,
        DatabaseConnectionError,
        DatabasePermissionError,
        DatabaseQueryError,
        DatabaseTimeoutError,
        DatabaseTransactionError,
        DBError,
        ForeignKeyViolationError,
        InvalidDataFormatError,
        NotNullViolationError,
        TableNotFoundError,
        UniqueConstraintViolationError,
    )
    from src.core.exceptions import (
        AuthenticationError,
        AuthorizationError,
        DatabaseError,
        PermissionDeniedError,
        ResourceConflictError,
        ResourceNotFoundError,
        ResourceValidationError,
        ValidationError,
    )

    # Map exception types to HTTP status codes
    status_code = 500  # Default

    if isinstance(exc, ResourceNotFoundError):
        status_code = 404
    elif isinstance(exc, ValidationError | ResourceValidationError):
        status_code = 400
    elif isinstance(exc, AuthenticationError | AuthorizationError):
        status_code = 401
    elif isinstance(exc, PermissionDeniedError):
        status_code = 403
    elif isinstance(exc, ResourceConflictError):
        status_code = 409
    # Database-specific exceptions
    elif isinstance(
        exc,
        ForeignKeyViolationError
        | UniqueConstraintViolationError
        | NotNullViolationError
        | CheckConstraintViolationError,
    ):
        status_code = 400  # Bad Request - constraint violations
    elif isinstance(exc, InvalidDataFormatError | DatabaseQueryError):
        status_code = 400  # Bad Request - invalid data/query
    elif isinstance(exc, TableNotFoundError | ColumnNotFoundError):
        status_code = 500  # Internal Server Error - schema issues
    elif isinstance(exc, DatabasePermissionError):
        status_code = 403  # Forbidden - insufficient privileges
    elif isinstance(
        exc, DatabaseConnectionError | DatabaseTimeoutError | DatabaseTransactionError
    ):
        status_code = 503  # Service Unavailable - database issues
    elif isinstance(exc, DBError | DatabaseError):
        status_code = 500  # Internal Server Error - general database errors

    return create_response(
        error={
            "message": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "type": exc.__class__.__name__,
        },
        status_code=status_code,
    )


@app.exception_handler(RequestValidationError)  # type: ignore[misc]
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI validation errors."""
    return create_response(
        error={
            "message": "Request validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": exc.errors(),
            "type": "RequestValidationError",
        },
        status_code=422,
    )


# Include routers
app.include_router(auth.router)
app.include_router(observations_router, prefix="/observations", tags=["observations"])
app.include_router(detections_router, prefix="/detections", tags=["detections"])
app.include_router(curation_router, prefix="/curation", tags=["curation"])
app.include_router(differencing_router, prefix="/differencing", tags=["differencing"])
app.include_router(catalog_router, prefix="/catalog", tags=["catalog"])
app.include_router(
    preprocessing_router, prefix="/preprocessing", tags=["preprocessing"]
)
app.include_router(stream.router, prefix="/stream", tags=["stream"])
app.include_router(storage.router, prefix="/storage", tags=["storage"])
app.include_router(health.router, prefix="/health", tags=["health"])


@app.get("/")  # type: ignore[misc]
async def root():
    """Root endpoint."""
    return create_response(
        {
            "message": "AstrID API",
            "version": APP_VERSION,
            "docs": "/docs",
            "redoc": "/redoc",
            "frontend": "http://localhost:3000",
            "planning_dashboard": "http://localhost:3000",
        }
    )


@app.get("/health")  # type: ignore[misc]
async def health_check():
    """Simple health check endpoint - redirects to comprehensive health check."""
    # For backward compatibility, provide a simple health response
    # Full health checks are available at /health/ endpoints
    import time
    from datetime import datetime

    start_time = time.time()
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "services": {},
    }

    # Check database connectivity (this was working in the logs)
    try:
        from src.core.db.session import check_pool_health, test_connection

        db_connected = await test_connection()
        if db_connected:
            pool_stats = await check_pool_health()
            health_status["services"]["database"] = {
                "status": "healthy",
                "connection": "connected",
                "pool_stats": pool_stats,
            }
        else:
            health_status["services"]["database"] = {
                "status": "unhealthy",
                "connection": "failed",
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy",
            "connection": "error",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check Redis connectivity (simple ping test)
    try:
        import redis.asyncio as redis

        from src.core.constants import REDIS_URL

        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        await redis_client.close()

        health_status["services"]["redis"] = {
            "status": "healthy",
            "connection": "connected",
        }
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": "unhealthy",
            "connection": "error",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check MLflow connectivity (if configured)
    try:
        from src.core.constants import MLFLOW_SUPABASE_URL

        if MLFLOW_SUPABASE_URL:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try both localhost (for external access) and mlflow (for container network)
                mlflow_urls = [
                    "http://localhost:5000/health",
                    "http://mlflow:5000/health",
                ]
                mlflow_healthy = False

                for url in mlflow_urls:
                    try:
                        response = await client.get(url)
                        if response.status_code == 200:
                            mlflow_healthy = True
                            break
                    except Exception:
                        continue

                if mlflow_healthy:
                    health_status["services"]["mlflow"] = {
                        "status": "healthy",
                        "connection": "connected",
                    }
                else:
                    health_status["services"]["mlflow"] = {
                        "status": "unhealthy",
                        "connection": "all_connection_attempts_failed",
                    }
                    health_status["status"] = "degraded"
        else:
            health_status["services"]["mlflow"] = {
                "status": "not_configured",
                "connection": "disabled",
            }
    except Exception as e:
        health_status["services"]["mlflow"] = {
            "status": "unhealthy",
            "connection": "error",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check Prefect connectivity (if configured)
    try:
        from src.core.constants import PREFECT_API_URL

        if PREFECT_API_URL:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try both localhost (for external access) and prefect (for container network)
                prefect_urls = [
                    "http://localhost:4200/health",
                    "http://prefect:4200/health",
                ]
                prefect_healthy = False

                for url in prefect_urls:
                    try:
                        response = await client.get(url)
                        if response.status_code == 200:
                            prefect_healthy = True
                            break
                    except Exception:
                        continue

                if prefect_healthy:
                    health_status["services"]["prefect"] = {
                        "status": "healthy",
                        "connection": "connected",
                    }
                else:
                    health_status["services"]["prefect"] = {
                        "status": "unhealthy",
                        "connection": "all_connection_attempts_failed",
                    }
                    health_status["status"] = "degraded"
        else:
            health_status["services"]["prefect"] = {
                "status": "not_configured",
                "connection": "disabled",
            }
    except Exception as e:
        health_status["services"]["prefect"] = {
            "status": "unhealthy",
            "connection": "error",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Calculate response time
    response_time = time.time() - start_time
    health_status["response_time_ms"] = round(response_time * 1000, 2)

    # Determine overall status
    if health_status["status"] == "healthy":
        status_code = 200
    else:
        status_code = 503  # Service Unavailable for degraded/unhealthy status

    return create_response(health_status, status_code=status_code)
