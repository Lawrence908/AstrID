"""Shared pytest fixtures and configuration for AstrID.

Provides:
- Asyncio event loop for pytest
- Async SQLite test database and `AsyncSession` fixture
- FastAPI app and test client when needed
- Lightweight mocks for external adapters (MAST, SkyView, R2)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

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
    """Provide a new `AsyncSession` per test function."""

    SessionLocal = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with SessionLocal() as session:
        yield session


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
