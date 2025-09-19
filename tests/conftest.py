"""Shared pytest fixtures and configuration for AstrID.

Provides:
- Asyncio event loop for pytest
- Async SQLite test database and `AsyncSession` fixture
- FastAPI app and test client when needed
- Lightweight mocks for external adapters (MAST, SkyView, R2)
- Mock services for MLflow, Prefect, and other infrastructure
- Sample data fixtures for testing
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (  # type: ignore
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

# Local lightweight Base for tests to avoid importing production engine
TestBase = declarative_base()


@pytest_asyncio.fixture(scope="session")
async def async_engine():
    """Create an in-memory SQLite async engine for tests."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def create_all(async_engine) -> None:
    """Create tables for models that opt-in during tests.

    Test modules can define models inheriting from `TestBase` and import this
    fixture to create all mapped tables.
    """

    async with async_engine.begin() as conn:  # type: ignore[attr-defined]
        await conn.run_sync(TestBase.metadata.create_all)  # type: ignore[arg-type]


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine, create_all) -> AsyncGenerator[AsyncSession, None]:
    """Provide a new `AsyncSession` per test function with transaction rollback."""

    SessionLocal = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with SessionLocal() as session:
        # Start a transaction
        transaction = await session.connection()
        await transaction.begin()

        try:
            yield session
        finally:
            # Always rollback to ensure test isolation
            await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def db_transaction(
    db_session: AsyncSession,
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session with explicit transaction management."""
    async with db_session.begin():
        yield db_session


@pytest.fixture
def test_db_url() -> str:
    """Provide test database URL."""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# --- External adapter mocks -------------------------------------------------


class DummyMASTClient:
    async def query_observations(
        self, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:  # noqa: D401
        return []

    async def query_observations_by_position(
        self, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return []


class DummySkyViewClient:
    async def download_reference_images(self, *args: Any, **kwargs: Any) -> list[str]:  # noqa: D401
        return []


class DummyR2Client:
    async def upload_file(self, *args: Any, **kwargs: Any) -> str:  # noqa: D401
        return "r2://dummy/path"


@pytest.fixture()
def mast_client() -> DummyMASTClient:
    return DummyMASTClient()


@pytest.fixture()
def skyview_client() -> DummySkyViewClient:
    return DummySkyViewClient()


@pytest.fixture()
def r2_client() -> DummyR2Client:
    return DummyR2Client()


# --- Sample Data Fixtures ---------------------------------------------------


@pytest.fixture
def sample_observation_data() -> dict[str, Any]:
    """Sample observation data for testing."""
    return {
        "id": "test-obs-001",
        "survey_id": "test-survey-001",
        "target_name": "Test Target",
        "ra": 180.0,
        "dec": 45.0,
        "observation_date": datetime.now(UTC),
        "exposure_time": 300.0,
        "filter_name": "g",
        "instrument": "test_instrument",
        "telescope": "test_telescope",
        "airmass": 1.2,
        "seeing": 1.5,
        "sky_background": 20.5,
        "processing_status": "pending",
        "metadata": {
            "observer": "test_observer",
            "weather": "clear",
            "moon_phase": 0.5,
        },
    }


@pytest.fixture
def sample_survey_data() -> dict[str, Any]:
    """Sample survey data for testing."""
    return {
        "id": "test-survey-001",
        "name": "Test Survey",
        "description": "A test astronomical survey",
        "survey_type": "photometric",
        "cadence": "nightly",
        "filters": ["g", "r", "i"],
        "field_of_view": 1.0,
        "limiting_magnitude": 24.0,
        "active": True,
        "configuration": {
            "exposure_time": 300,
            "readout_time": 30,
            "filter_change_time": 60,
        },
    }


@pytest.fixture
def sample_detection_data() -> dict[str, Any]:
    """Sample detection data for testing."""
    return {
        "id": "test-detection-001",
        "observation_id": "test-obs-001",
        "x": 512.5,
        "y": 1024.7,
        "ra": 180.001,
        "dec": 45.001,
        "magnitude": 18.5,
        "magnitude_error": 0.1,
        "flux": 1000.0,
        "flux_error": 50.0,
        "fwhm": 2.1,
        "ellipticity": 0.1,
        "confidence": 0.95,
        "classification": "transient",
        "detection_type": "ml_detection",
    }


@pytest.fixture
def sample_model_data() -> dict[str, Any]:
    """Sample ML model data for testing."""
    return {
        "id": "test-model-001",
        "name": "Test U-Net Model",
        "model_type": "unet",
        "architecture": "unet_resnet34",
        "framework": "tensorflow",
        "version": "1.0.0",
        "training_dataset": "test_dataset_v1",
        "hyperparameters": {"learning_rate": 0.001, "batch_size": 32, "epochs": 100},
        "performance_metrics": {
            "accuracy": 0.92,
            "precision": 0.88,
            "recall": 0.85,
            "f1_score": 0.865,
        },
    }


# --- Mock Services -----------------------------------------------------------


@pytest.fixture
def mock_mlflow_client() -> MagicMock:
    """Mock MLflow client for testing."""
    mock_client = MagicMock()
    mock_client.create_experiment.return_value = "test-experiment-id"
    mock_client.create_run.return_value = MagicMock(
        info=MagicMock(run_id="test-run-id")
    )
    mock_client.log_param.return_value = None
    mock_client.log_metric.return_value = None
    mock_client.log_artifact.return_value = None
    mock_client.set_tag.return_value = None
    mock_client.end_run.return_value = None
    return mock_client


@pytest.fixture
def mock_prefect_client() -> AsyncMock:
    """Mock Prefect client for testing."""
    mock_client = AsyncMock()
    mock_client.create_flow_run.return_value = MagicMock(id="test-flow-run-id")
    mock_client.read_flow_run.return_value = MagicMock(
        id="test-flow-run-id", state=MagicMock(type="COMPLETED")
    )
    mock_client.set_flow_run_state.return_value = None
    return mock_client


@pytest.fixture
def mock_dramatiq_broker() -> MagicMock:
    """Mock Dramatiq broker for testing."""
    mock_broker = MagicMock()
    mock_broker.enqueue.return_value = MagicMock(message_id="test-message-id")
    mock_broker.flush_all.return_value = None
    return mock_broker


@pytest.fixture
def mock_supabase_client() -> MagicMock:
    """Mock Supabase client for testing."""
    mock_client = MagicMock()
    mock_client.auth.sign_in_with_password.return_value = MagicMock(
        user=MagicMock(id="test-user-id", email="test@example.com"),
        session=MagicMock(access_token="test-token"),
    )
    mock_client.auth.get_user.return_value = MagicMock(
        user=MagicMock(id="test-user-id", email="test@example.com")
    )
    return mock_client


@pytest.fixture
def mock_storage_service() -> AsyncMock:
    """Mock storage service for testing."""
    mock_service = AsyncMock()
    mock_service.upload.return_value = "test://storage/path/file.fits"
    mock_service.download.return_value = b"fake_fits_data"
    mock_service.delete.return_value = True
    mock_service.exists.return_value = True
    mock_service.list_files.return_value = ["file1.fits", "file2.fits"]
    return mock_service


# --- Performance and Testing Utilities --------------------------------------


@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started/stopped")
            return self.end_time - self.start_time

    return Timer()


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
