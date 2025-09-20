"""API documentation utilities and OpenAPI configuration."""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def create_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Create comprehensive OpenAPI schema for AstrID API."""
    
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AstrID API",
        version="1.0.0",
        description="""
# AstrID API Documentation

The AstrID (Astronomical Identification) API provides comprehensive endpoints for:

- **Observation Management**: Ingest, process, and manage astronomical observations
- **Image Processing**: Calibration, alignment, and preprocessing of astronomical images  
- **Anomaly Detection**: Machine learning-powered detection of astronomical anomalies
- **Workflow Orchestration**: Automated processing pipelines and workflows
- **Data Cataloging**: Search, filter, and export astronomical data
- **Human Validation**: Curator interfaces for anomaly validation

## Authentication

The API uses JWT-based authentication with role-based access control (RBAC). Include the JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Rate Limiting

API requests are rate-limited per user and endpoint. Rate limit information is included in response headers:

- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in current window  
- `X-RateLimit-Reset`: Window reset timestamp

## Error Handling

All errors follow a consistent format:

```json
{
  "success": false,
  "error": {
    "message": "Human-readable error message",
    "error_code": "ERROR_CODE", 
    "details": "Additional error details",
    "type": "ExceptionType"
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## Versioning

The API supports multiple versioning strategies:

- **URL-based**: `/api/v1/observations`
- **Header-based**: `Accept: application/vnd.astrid.v1+json`
- **Query parameter**: `?version=1`
        """,
        routes=app.routes,
    )

    # Add custom OpenAPI extensions
    openapi_schema["info"]["contact"] = {
        "name": "AstrID Team",
        "email": "astronomical.identification@gmail.com",
        "url": "https://github.com/astrid-project"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    openapi_schema["info"]["termsOfService"] = "https://astrid.chrislawrence.ca/terms"

    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "https://api.astrid.chrislawrence.ca/v1",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.astrid.chrislawrence.ca/v1", 
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000/v1",
            "description": "Development server"
        }
    ]

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for authentication"
        }
    }

    # Add global security
    openapi_schema["security"] = [{"BearerAuth": []}]

    # Add comprehensive tags
    openapi_schema["tags"] = [
        {
            "name": "observations",
            "description": "Astronomical observation management"
        },
        {
            "name": "detections", 
            "description": "Anomaly detection and ML inference"
        },
        {
            "name": "preprocessing",
            "description": "Image preprocessing and calibration"
        },
        {
            "name": "differencing",
            "description": "Image differencing algorithms"
        },
        {
            "name": "curation",
            "description": "Human validation and curation"
        },
        {
            "name": "catalog",
            "description": "Data cataloging and search"
        },
        {
            "name": "workflows",
            "description": "Workflow orchestration and automation"
        },
        {
            "name": "health",
            "description": "System health and monitoring"
        },
        {
            "name": "auth",
            "description": "Authentication and authorization"
        },
        {
            "name": "storage",
            "description": "Cloud storage management"
        },
        {
            "name": "mlflow",
            "description": "MLflow experiment tracking"
        },
        {
            "name": "workers",
            "description": "Background worker management"
        },
        {
            "name": "stream",
            "description": "Real-time streaming endpoints"
        }
    ]

    # Add common response schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}

    # Add error response schema
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "example": False
            },
            "error": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "example": "Resource not found"
                    },
                    "error_code": {
                        "type": "string", 
                        "example": "RESOURCE_NOT_FOUND"
                    },
                    "details": {
                        "type": "string",
                        "example": "Observation with ID 123 not found"
                    },
                    "type": {
                        "type": "string",
                        "example": "ResourceNotFoundError"
                    }
                }
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "example": "2025-01-01T00:00:00Z"
            }
        }
    }

    # Add pagination schemas
    openapi_schema["components"]["schemas"]["PaginationParams"] = {
        "type": "object",
        "properties": {
            "page": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
                "description": "Page number (1-based)"
            },
            "size": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 20,
                "description": "Number of items per page"
            },
            "sort": {
                "type": "string",
                "description": "Sort field and direction (e.g., 'created_at:desc')"
            },
            "search": {
                "type": "string",
                "description": "Search query"
            }
        }
    }

    openapi_schema["components"]["schemas"]["PaginatedResponse"] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {}
            },
            "pagination": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "example": 1},
                    "size": {"type": "integer", "example": 20},
                    "total": {"type": "integer", "example": 100},
                    "pages": {"type": "integer", "example": 5},
                    "has_next": {"type": "boolean", "example": True},
                    "has_prev": {"type": "boolean", "example": False}
                }
            }
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def add_endpoint_documentation(
    app: FastAPI,
    path: str,
    method: str,
    summary: str,
    description: str,
    tags: List[str],
    responses: Optional[Dict[int, Dict[str, Any]]] = None,
    request_body: Optional[Dict[str, Any]] = None,
    parameters: Optional[List[Dict[str, Any]]] = None,
    security: Optional[List[Dict[str, List[str]]]] = None
) -> None:
    """Add comprehensive documentation to an endpoint."""
    
    # This would be used to enhance existing endpoints with better documentation
    # Implementation would depend on how we want to structure the documentation
    pass


def generate_api_examples() -> Dict[str, Any]:
    """Generate comprehensive API examples for documentation."""
    
    return {
        "observation_creation": {
            "request": {
                "survey": "ZTF",
                "observation_id": "ZTF_20230101_000000",
                "ra": 180.5,
                "dec": 45.2,
                "observation_time": "2025-01-01T00:00:00Z",
                "filter_band": "r",
                "exposure_time": 30.0
            },
            "response": {
                "success": True,
                "data": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "survey": "ZTF",
                    "observation_id": "ZTF_20230101_000000",
                    "ra": 180.5,
                    "dec": 45.2,
                    "observation_time": "2025-01-01T00:00:00Z",
                    "filter_band": "r",
                    "exposure_time": 30.0,
                    "status": "ingested",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z"
                },
                "timestamp": "2025-01-01T00:00:00Z"
            }
        },
        "detection_inference": {
            "request": {
                "observation_id": "123e4567-e89b-12d3-a456-426614174000",
                "model_version": "latest",
                "confidence_threshold": 0.8
            },
            "response": {
                "success": True,
                "data": {
                    "detection_id": "456e7890-e89b-12d3-a456-426614174000",
                    "observation_id": "123e4567-e89b-12d3-a456-426614174000",
                    "ra": 180.5,
                    "dec": 45.2,
                    "confidence": 0.95,
                    "magnitude": 18.5,
                    "status": "detected",
                    "created_at": "2025-01-01T00:00:00Z"
                },
                "timestamp": "2025-01-01T00:00:00Z"
            }
        }
    }
