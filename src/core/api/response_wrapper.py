"""Standard API response envelope and utilities."""

from enum import Enum
from typing import Any, Generic, TypeVar

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class ResponseEnvelope(BaseModel, Generic[T]):
    """Standard API response envelope."""

    status: ResponseStatus = Field(..., description="Response status (success/error)")
    data: T | None = Field(None, description="Response data payload")
    meta: dict[str, Any] | None = Field(
        None, description="Metadata about the response (pagination, etc.)"
    )
    error: dict[str, Any] | None = Field(
        None, description="Error details if status is error"
    )

    def to_response(self, status_code: int = 200) -> JSONResponse:
        """Convert the envelope to a proper FastAPI response object.

        This ensures that middleware and decorators that need to modify
        response headers have a proper Response object to work with.

        Args:
            status_code: HTTP status code for the response

        Returns:
            JSONResponse with the envelope content
        """
        return JSONResponse(content=jsonable_encoder(self), status_code=status_code)


def create_response(
    data: Any | None = None,
    error: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    status_code: int = 200,
    request_id: str | None = None,
) -> JSONResponse:
    """Create a standard API response envelope.

    Args:
        data: Response data payload
        error: Error details if any
        meta: Response metadata
        status_code: HTTP status code
        request_id: Request ID for tracing

    Returns:
        Response envelope with standardized format
    """
    if error:
        status = ResponseStatus.ERROR
    else:
        status = ResponseStatus.SUCCESS

    if meta is None:
        meta = {}

    if request_id:
        meta["request_id"] = request_id

    envelope = ResponseEnvelope(status=status, data=data, meta=meta, error=error)
    return envelope.to_response(status_code)
