"""Database fixtures for testing.

Provides:
- Test database setup and teardown
- Sample data creation and cleanup
- Database migration utilities
- Performance testing utilities
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

# Test-specific database base
TestBase = declarative_base()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
        pool_pre_ping=True,
    )
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def setup_test_db(test_engine):
    """Set up test database schema."""
    async with test_engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.create_all)


@pytest_asyncio.fixture(scope="function")
async def test_session(
    test_engine, setup_test_db
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a test database session with automatic rollback."""
    TestSessionLocal = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with TestSessionLocal() as session:
        # Begin a nested transaction
        async with session.begin():
            try:
                yield session
            finally:
                # Always rollback to ensure test isolation
                await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def test_transaction(test_session: AsyncSession):
    """Provide a test session within an explicit transaction."""
    async with test_session.begin_nested():
        yield test_session


@pytest.fixture
def db_cleanup():
    """Database cleanup utilities."""

    class DatabaseCleaner:
        def __init__(self):
            self.cleanup_tasks: list[Callable] = []

        def add_cleanup_task(self, task: Callable) -> None:
            """Add a cleanup task to be executed after test."""
            self.cleanup_tasks.append(task)

        async def cleanup(self, session: AsyncSession) -> None:
            """Execute all cleanup tasks."""
            for task in self.cleanup_tasks:
                try:
                    if asyncio.iscoroutinefunction(task):
                        await task(session)  # type: ignore
                    else:
                        task(session)  # type: ignore
                except Exception:
                    # Log error but continue cleanup
                    pass
            self.cleanup_tasks.clear()

    return DatabaseCleaner()


# --- Sample Data Creation Utilities -----------------------------------------


@pytest.fixture
def sample_data_factory():
    """Factory for creating sample test data."""

    class SampleDataFactory:
        @staticmethod
        def create_observation_data(**overrides: Any) -> dict[str, Any]:
            """Create sample observation data."""
            base_data = {
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
            base_data.update(overrides)
            return base_data

        @staticmethod
        def create_survey_data(**overrides: Any) -> dict[str, Any]:
            """Create sample survey data."""
            base_data = {
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
            base_data.update(overrides)
            return base_data

        @staticmethod
        def create_detection_data(**overrides: Any) -> dict[str, Any]:
            """Create sample detection data."""
            base_data = {
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
            base_data.update(overrides)
            return base_data

        @staticmethod
        def create_model_data(**overrides: Any) -> dict[str, Any]:
            """Create sample ML model data."""
            base_data = {
                "id": "test-model-001",
                "name": "Test U-Net Model",
                "model_type": "unet",
                "architecture": "unet_resnet34",
                "framework": "tensorflow",
                "version": "1.0.0",
                "training_dataset": "test_dataset_v1",
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 100,
                },
                "performance_metrics": {
                    "accuracy": 0.92,
                    "precision": 0.88,
                    "recall": 0.85,
                    "f1_score": 0.865,
                },
            }
            base_data.update(overrides)
            return base_data

        @staticmethod
        def create_bulk_observations(
            count: int, survey_id: str = "test-survey-001"
        ) -> list[dict[str, Any]]:
            """Create multiple observation records for testing."""
            observations = []
            for i in range(count):
                obs = SampleDataFactory.create_observation_data(
                    id=f"test-obs-{i:03d}",
                    survey_id=survey_id,
                    ra=180.0 + (i * 0.1),
                    dec=45.0 + (i * 0.1),
                    target_name=f"Test Target {i+1}",
                )
                observations.append(obs)
            return observations

    return SampleDataFactory()


# --- Database Performance Testing -------------------------------------------


@pytest.fixture
def db_performance_monitor():
    """Monitor database performance during tests."""

    class DatabasePerformanceMonitor:
        def __init__(self):
            self.query_times: list[float] = []
            self.query_counts: dict[str, int] = {}

        async def time_query(self, query_name: str, coro):
            """Time a database query."""
            import time

            start_time = time.perf_counter()
            try:
                result = await coro
                return result
            finally:
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                self.query_times.append(elapsed)
                self.query_counts[query_name] = self.query_counts.get(query_name, 0) + 1

        def get_average_query_time(self) -> float:
            """Get average query time."""
            return (
                sum(self.query_times) / len(self.query_times)
                if self.query_times
                else 0.0
            )

        def get_query_stats(self) -> dict[str, Any]:
            """Get comprehensive query statistics."""
            return {
                "total_queries": len(self.query_times),
                "average_time": self.get_average_query_time(),
                "max_time": max(self.query_times) if self.query_times else 0.0,
                "min_time": min(self.query_times) if self.query_times else 0.0,
                "query_counts": self.query_counts.copy(),
            }

    return DatabasePerformanceMonitor()


# --- Migration Testing Utilities --------------------------------------------


@pytest.fixture
def migration_tester():
    """Utilities for testing database migrations."""

    class MigrationTester:
        def __init__(self):
            self.applied_migrations: list[str] = []

        async def apply_migration(self, session: AsyncSession, migration_name: str):
            """Apply a test migration."""
            # This would integrate with Alembic for real migration testing
            self.applied_migrations.append(migration_name)

        async def rollback_migration(self, session: AsyncSession, migration_name: str):
            """Rollback a test migration."""
            if migration_name in self.applied_migrations:
                self.applied_migrations.remove(migration_name)

        def get_applied_migrations(self) -> list[str]:
            """Get list of applied migrations."""
            return self.applied_migrations.copy()

    return MigrationTester()
