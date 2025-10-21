"""API test configuration and fixtures."""

import pytest
import asyncio
from typing import AsyncGenerator, Dict, Any
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.api.main import app
# Using db_session from main conftest.py
from tests.mocks.supabase import MockSupabaseClient


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def api_client() -> AsyncGenerator[AsyncClient, None]:
    """Create API client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def authenticated_client(api_client: AsyncClient) -> AsyncGenerator[AsyncClient, None]:
    """Create authenticated API client."""
    # Mock authentication by setting headers
    api_client.headers.update({"Authorization": "Bearer test-token"})
    yield api_client


@pytest.fixture
def test_observation_data() -> Dict[str, Any]:
    """Sample observation data for testing."""
    return {
        "survey": "ZTF",
        "observation_id": "ZTF_20230101_000000",
        "ra": 180.5,
        "dec": 45.2,
        "observation_time": "2025-01-01T00:00:00Z",
        "filter_band": "r",
        "exposure_time": 30.0
    }


@pytest.fixture
def test_detection_data() -> Dict[str, Any]:
    """Sample detection data for testing."""
    return {
        "observation_id": "123e4567-e89b-12d3-a456-426614174000",
        "ra": 180.5,
        "dec": 45.2,
        "confidence": 0.95,
        "magnitude": 18.5,
        "model_version": "v1.0.0"
    }


@pytest.fixture
def test_workflow_data() -> Dict[str, Any]:
    """Sample workflow data for testing."""
    return {
        "flow_type": "observation_processing",
        "parameters": {
            "observation_id": "123e4567-e89b-12d3-a456-426614174000",
            "preprocessing_enabled": True,
            "differencing_enabled": True,
            "detection_enabled": True
        }
    }


@pytest.fixture
def test_auth_data() -> Dict[str, Any]:
    """Sample authentication data for testing."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "role": "user"
    }


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    return MockSupabaseClient()


@pytest.fixture
def api_test_config() -> Dict[str, Any]:
    """API test configuration."""
    return {
        "base_url": "http://test",
        "timeout": 30.0,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "rate_limit_delay": 0.1
    }


@pytest.fixture
def test_user_roles() -> Dict[str, str]:
    """Test user roles for different permission levels."""
    return {
        "admin": "admin",
        "user": "user",
        "curator": "curator",
        "viewer": "viewer"
    }


@pytest.fixture
def test_rate_limits() -> Dict[str, Dict[str, int]]:
    """Test rate limit configurations."""
    return {
        "default": {"requests": 1000, "window_seconds": 3600},
        "observations": {"requests": 100, "window_seconds": 3600},
        "detections": {"requests": 200, "window_seconds": 3600},
        "workflows": {"requests": 50, "window_seconds": 3600},
        "admin": {"requests": 10000, "window_seconds": 3600}
    }


@pytest.fixture
def test_error_scenarios() -> Dict[str, Dict[str, Any]]:
    """Test error scenarios for comprehensive testing."""
    return {
        "invalid_uuid": {
            "endpoint": "/observations/invalid-uuid",
            "expected_status": 422,
            "expected_error_code": "VALIDATION_ERROR"
        },
        "resource_not_found": {
            "endpoint": "/observations/00000000-0000-0000-0000-000000000000",
            "expected_status": 404,
            "expected_error_code": "RESOURCE_NOT_FOUND"
        },
        "unauthorized_access": {
            "endpoint": "/observations",
            "method": "GET",
            "expected_status": 401,
            "expected_error_code": "AUTHENTICATION_ERROR"
        },
        "validation_error": {
            "endpoint": "/observations",
            "method": "POST",
            "data": {"invalid_field": "invalid_value"},
            "expected_status": 400,
            "expected_error_code": "VALIDATION_ERROR"
        }
    }


@pytest.fixture
def test_pagination_params() -> Dict[str, Any]:
    """Test pagination parameters."""
    return {
        "page": 1,
        "size": 10,
        "sort": "created_at:desc",
        "search": "test"
    }


@pytest.fixture
def test_filter_params() -> Dict[str, Any]:
    """Test filter parameters for different endpoints."""
    return {
        "observations": {
            "survey": "ZTF",
            "status": "completed",
            "ra_min": 180.0,
            "ra_max": 181.0,
            "dec_min": 45.0,
            "dec_max": 46.0
        },
        "detections": {
            "min_confidence": 0.8,
            "max_confidence": 1.0,
            "status": "detected",
            "model_version": "v1.0.0"
        },
        "workflows": {
            "flow_type": "observation_processing",
            "status": "running",
            "created_after": "2025-01-01T00:00:00Z"
        }
    }


@pytest.fixture
def test_batch_data() -> Dict[str, Any]:
    """Test batch operation data."""
    return {
        "observations": [
            {
                "survey": "ZTF",
                "observation_id": "ZTF_20230101_000000",
                "ra": 180.5,
                "dec": 45.2,
                "observation_time": "2025-01-01T00:00:00Z",
                "filter_band": "r",
                "exposure_time": 30.0
            },
            {
                "survey": "ZTF",
                "observation_id": "ZTF_20230101_000001",
                "ra": 181.0,
                "dec": 46.0,
                "observation_time": "2025-01-01T00:01:00Z",
                "filter_band": "g",
                "exposure_time": 30.0
            }
        ],
        "detections": [
            {
                "observation_id": "123e4567-e89b-12d3-a456-426614174000",
                "ra": 180.5,
                "dec": 45.2,
                "confidence": 0.95,
                "magnitude": 18.5,
                "model_version": "v1.0.0"
            },
            {
                "observation_id": "123e4567-e89b-12d3-a456-426614174001",
                "ra": 181.0,
                "dec": 46.0,
                "confidence": 0.85,
                "magnitude": 19.0,
                "model_version": "v1.0.0"
            }
        ]
    }


@pytest.fixture
def test_performance_metrics() -> Dict[str, Any]:
    """Test performance metrics configuration."""
    return {
        "max_response_time_ms": 1000,
        "max_memory_usage_mb": 100,
        "max_cpu_usage_percent": 80,
        "concurrent_requests": 10,
        "test_duration_seconds": 60
    }


@pytest.fixture
def test_security_config() -> Dict[str, Any]:
    """Test security configuration."""
    return {
        "require_https": True,
        "max_request_size_mb": 10,
        "allowed_origins": ["https://astrid.chrislawrence.ca", "https://app.astrid.chrislawrence.ca"],
        "cors_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "cors_headers": ["Authorization", "Content-Type", "X-API-Version"]
    }


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "auth: Authentication tests")
    config.addinivalue_line("markers", "rate_limit: Rate limiting tests")
    config.addinivalue_line("markers", "error_handling: Error handling tests")
    config.addinivalue_line("markers", "validation: Validation tests")
    config.addinivalue_line("markers", "pagination: Pagination tests")
    config.addinivalue_line("markers", "filtering: Filtering tests")
    config.addinivalue_line("markers", "batch: Batch operation tests")
    config.addinivalue_line("markers", "workflow: Workflow tests")
    config.addinivalue_line("markers", "observations: Observation tests")
    config.addinivalue_line("markers", "detections: Detection tests")
    config.addinivalue_line("markers", "health: Health check tests")
