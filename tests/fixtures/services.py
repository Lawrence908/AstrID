"""Service fixtures for testing.

Provides:
- Mock FastAPI app
- Test client fixtures
- Service instance mocks
- Configuration fixtures
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def test_app() -> FastAPI:
    """Create a test FastAPI application."""
    app = FastAPI(
        title="AstrID Test API",
        description="Test version of AstrID API",
        version="0.1.0-test",
    )

    # Add minimal test endpoints
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "environment": "test"}

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test endpoint working"}

    return app


@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)


@pytest.fixture
def mock_observation_service() -> AsyncMock:
    """Mock observation service for testing."""
    service = AsyncMock()
    service.create_observation.return_value = {
        "id": "test-obs-001",
        "status": "created",
    }
    service.get_observation.return_value = {
        "id": "test-obs-001",
        "survey_id": "test-survey-001",
        "processing_status": "pending",
    }
    service.update_observation_status.return_value = {
        "id": "test-obs-001",
        "processing_status": "processing",
    }
    service.validate_observation_data.return_value = True
    service.calculate_observation_metrics.return_value = {
        "airmass": 1.2,
        "moon_separation": 45.0,
        "galactic_latitude": 30.0,
    }
    return service


@pytest.fixture
def mock_detection_service() -> AsyncMock:
    """Mock detection service for testing."""
    service = AsyncMock()
    service.process_detections.return_value = [
        {"id": "test-detection-001", "confidence": 0.95, "classification": "transient"}
    ]
    service.validate_detection.return_value = True
    service.calculate_detection_metrics.return_value = {
        "snr": 15.2,
        "significance": 5.8,
        "quality_score": 0.92,
    }
    return service


@pytest.fixture
def mock_preprocessing_service() -> AsyncMock:
    """Mock preprocessing service for testing."""
    service = AsyncMock()
    service.calibrate_image.return_value = {
        "calibrated_path": "test://storage/calibrated/image.fits",
        "quality_metrics": {"background_std": 10.5, "fwhm": 2.1, "ellipticity": 0.05},
    }
    service.align_image.return_value = {
        "aligned_path": "test://storage/aligned/image.fits",
        "wcs_solution": {"rms": 0.3, "n_matches": 150},
    }
    return service


@pytest.fixture
def mock_differencing_service() -> AsyncMock:
    """Mock differencing service for testing."""
    service = AsyncMock()
    service.create_difference_image.return_value = {
        "difference_path": "test://storage/diff/image.fits",
        "metrics": {"psf_match_quality": 0.95, "background_rms": 8.2, "n_sources": 25},
    }
    service.extract_sources.return_value = [
        {"x": 100.5, "y": 200.3, "flux": 1500, "snr": 12.5}
    ]
    return service


@pytest.fixture
def mock_ml_service() -> AsyncMock:
    """Mock ML service for testing."""
    service = AsyncMock()
    service.predict.return_value = {
        "predictions": [0.95, 0.12, 0.88],
        "model_version": "1.0.0",
        "inference_time": 0.045,
    }
    service.load_model.return_value = True
    service.get_model_info.return_value = {
        "name": "unet_model",
        "version": "1.0.0",
        "architecture": "U-Net",
        "training_date": "2024-09-01",
    }
    return service


@pytest.fixture
def mock_workflow_service() -> AsyncMock:
    """Mock workflow service for testing."""
    service = AsyncMock()
    service.create_flow_run.return_value = {
        "flow_run_id": "test-flow-run-001",
        "status": "pending",
    }
    service.get_flow_run_status.return_value = {
        "flow_run_id": "test-flow-run-001",
        "status": "completed",
        "duration": 125.6,
    }
    service.schedule_flow.return_value = {
        "schedule_id": "test-schedule-001",
        "next_run": "2024-09-20T00:00:00Z",
    }
    return service


@pytest.fixture
def mock_storage_client() -> AsyncMock:
    """Mock storage client for testing."""
    client = AsyncMock()
    client.upload_file.return_value = "test://storage/path/file.fits"
    client.download_file.return_value = b"fake_file_content"
    client.delete_file.return_value = True
    client.file_exists.return_value = True
    client.list_files.return_value = ["file1.fits", "file2.fits", "file3.fits"]
    client.get_file_metadata.return_value = {
        "size": 1024000,
        "modified": "2024-09-19T12:00:00Z",
        "content_type": "application/octet-stream",
    }
    return client


@pytest.fixture
def mock_auth_service() -> MagicMock:
    """Mock authentication service for testing."""
    service = MagicMock()
    service.authenticate_user.return_value = {
        "user_id": "test-user-001",
        "email": "test@example.com",
        "roles": ["user"],
    }
    service.authorize_action.return_value = True
    service.get_user_permissions.return_value = [
        "read:observations",
        "write:observations",
        "read:detections",
    ]
    service.create_api_key.return_value = {
        "api_key": "test-api-key-123",
        "expires_at": "2024-12-31T23:59:59Z",
    }
    return service


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Test configuration settings."""
    return {
        "database": {
            "url": "sqlite+aiosqlite:///:memory:",
            "pool_size": 1,
            "max_overflow": 0,
        },
        "storage": {"type": "memory", "bucket": "test-bucket"},
        "mlflow": {"tracking_uri": "memory://", "experiment_name": "test_experiment"},
        "prefect": {
            "api_url": "http://localhost:4200/api",
            "workspace": "test-workspace",
        },
        "auth": {
            "secret_key": "test-secret-key",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30,
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }


@pytest.fixture
def mock_external_apis() -> dict[str, MagicMock]:
    """Mock external API clients."""
    mast_mock = MagicMock()
    mast_mock.query_observations.return_value = [
        {
            "obs_id": "ext-obs-001",
            "target_name": "External Target",
            "ra": 180.0,
            "dec": 45.0,
            "filters": "g,r,i",
        }
    ]

    skyview_mock = MagicMock()
    skyview_mock.get_images.return_value = ["http://skyview.example.com/image1.fits"]

    simbad_mock = MagicMock()
    simbad_mock.query_object.return_value = {
        "object_name": "HD 123456",
        "object_type": "Star",
        "coordinates": {"ra": 180.0, "dec": 45.0},
    }

    return {"mast": mast_mock, "skyview": skyview_mock, "simbad": simbad_mock}


@pytest.fixture
def performance_config() -> dict[str, Any]:
    """Performance testing configuration."""
    return {
        "max_response_time": 1.0,  # seconds
        "max_memory_usage": 100 * 1024 * 1024,  # 100MB
        "max_cpu_usage": 80.0,  # percent
        "concurrent_requests": 10,
        "test_duration": 30,  # seconds
        "warmup_time": 5,  # seconds
    }


@pytest.fixture
def error_simulation() -> dict[str, callable]:
    """Error simulation utilities for testing."""

    def simulate_database_error():
        from sqlalchemy.exc import DatabaseError

        raise DatabaseError("Simulated database error", None, None)

    def simulate_network_error():
        import httpx

        raise httpx.ConnectError("Simulated network error")

    def simulate_timeout_error():
        raise TimeoutError("Simulated timeout error")

    def simulate_validation_error():
        from pydantic import ValidationError

        raise ValidationError("Simulated validation error", model=dict)

    return {
        "database": simulate_database_error,
        "network": simulate_network_error,
        "timeout": simulate_timeout_error,
        "validation": simulate_validation_error,
    }
