"""Main FastAPI application for AstrID API."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.api.routes import stream
from src.core.constants import (
    API_DESCRIPTION,
    API_TITLE,
    APP_VERSION,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ORIGINS,
)
from src.core.exceptions import AstrIDException
from src.domains.detection.api.routes import router as detections_router
from src.domains.observations.api.routes import router as observations_router

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

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": exc.message,
                "error_code": exc.error_code,
                "details": exc.details,
                "type": exc.__class__.__name__,
            }
        },
    )


@app.exception_handler(RequestValidationError)  # type: ignore[misc]
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "error_code": "VALIDATION_ERROR",
                "details": exc.errors(),
                "type": "RequestValidationError",
            }
        },
    )


# Include routers
app.include_router(observations_router, prefix="/observations", tags=["observations"])
app.include_router(detections_router, prefix="/detections", tags=["detections"])
app.include_router(stream.router, prefix="/stream", tags=["stream"])


@app.get("/")  # type: ignore[misc]
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "AstrID API",
        "version": APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")  # type: ignore[misc]
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
