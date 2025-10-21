"""Test utilities and helper functions."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTimer:
    """Timer utility for performance testing."""

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer."""
        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer not properly started/stopped")
        return self.end_time - self.start_time

    def __enter__(self) -> TestTimer:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


class MemoryTracker:
    """Memory usage tracking utility."""

    def __init__(self):
        self.initial_memory: int | None = None
        self.peak_memory: int | None = None

    def start(self) -> None:
        """Start memory tracking."""
        import psutil

        process = psutil.Process()
        self.initial_memory = process.memory_info().rss
        self.peak_memory = self.initial_memory

    def update(self) -> None:
        """Update peak memory usage."""
        if self.initial_memory is None:
            raise ValueError("Memory tracking not started")

        import psutil

        process = psutil.Process()
        current_memory = process.memory_info().rss
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    @property
    def memory_used(self) -> int:
        """Get memory used since start (in bytes)."""
        if self.initial_memory is None or self.peak_memory is None:
            raise ValueError("Memory tracking not properly initialized")
        return self.peak_memory - self.initial_memory


@contextlib.asynccontextmanager
async def temporary_environment(**env_vars: str) -> AsyncGenerator[None, None]:
    """Temporarily set environment variables."""
    import os

    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


class DatabaseTestUtils:
    """Database testing utilities."""

    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool], timeout: float = 5.0, interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
        return False

    @staticmethod
    async def count_table_rows(session: Any, table_name: str) -> int:
        """Count rows in a table."""
        result = await session.execute(f"SELECT COUNT(*) FROM {table_name}")
        return result.scalar()

    @staticmethod
    async def truncate_tables(session: Any, table_names: list[str]) -> None:
        """Truncate multiple tables."""
        for table_name in table_names:
            await session.execute(f"DELETE FROM {table_name}")
        await session.commit()


class APITestUtils:
    """API testing utilities."""

    @staticmethod
    def assert_response_structure(
        response_data: dict[str, Any], expected_fields: list[str]
    ) -> None:
        """Assert that response has expected structure."""
        for field in expected_fields:
            assert field in response_data, f"Missing field: {field}"

    @staticmethod
    def assert_pagination_response(
        response_data: dict[str, Any], expected_fields: list[str] | None = None
    ) -> None:
        """Assert pagination response structure."""
        if expected_fields is None:
            expected_fields = ["data", "total", "page", "per_page", "pages"]

        APITestUtils.assert_response_structure(response_data, expected_fields)
        assert isinstance(response_data["data"], list)
        assert isinstance(response_data["total"], int)
        assert response_data["total"] >= 0

    @staticmethod
    async def poll_for_status(
        get_status_func: Callable[[], Any],
        expected_status: str,
        timeout: float = 30.0,
        interval: float = 1.0,
    ) -> bool:
        """Poll for a specific status."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                current_status = await get_status_func()
                if current_status == expected_status:
                    return True
            except Exception:
                pass
            await asyncio.sleep(interval)
        return False


class FileTestUtils:
    """File testing utilities."""

    @staticmethod
    def create_temp_fits_file(
        temp_dir: Path, filename: str = "test.fits", width: int = 100, height: int = 100
    ) -> Path:
        """Create a temporary FITS file."""
        import numpy as np
        from astropy.io import fits

        # Create simple test data
        data = np.random.random((height, width)).astype(np.float32)

        # Create FITS HDU
        hdu = fits.PrimaryHDU(data)
        hdu.header["OBJECT"] = "Test Object"
        hdu.header["TELESCOP"] = "Test Telescope"
        hdu.header["INSTRUME"] = "Test Instrument"
        hdu.header["FILTER"] = "g"
        hdu.header["EXPTIME"] = 300.0
        hdu.header["RA"] = 180.0
        hdu.header["DEC"] = 45.0

        # Write to file
        file_path = temp_dir / filename
        hdu.writeto(file_path, overwrite=True)
        return file_path

    @staticmethod
    def create_temp_text_file(temp_dir: Path, filename: str, content: str) -> Path:
        """Create a temporary text file."""
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path

    @staticmethod
    def assert_file_exists(file_path: Path) -> None:
        """Assert that a file exists."""
        assert file_path.exists(), f"File does not exist: {file_path}"

    @staticmethod
    def assert_file_not_exists(file_path: Path) -> None:
        """Assert that a file does not exist."""
        assert not file_path.exists(), f"File should not exist: {file_path}"


class MockTestUtils:
    """Mock testing utilities."""

    @staticmethod
    def create_async_mock(**kwargs: Any) -> AsyncMock:
        """Create an AsyncMock with default return values."""
        mock = AsyncMock(**kwargs)
        return mock

    @staticmethod
    def create_mock_with_methods(methods: dict[str, Any]) -> MagicMock:
        """Create a mock with specific method return values."""
        mock = MagicMock()
        for method_name, return_value in methods.items():
            getattr(mock, method_name).return_value = return_value
        return mock

    @staticmethod
    @contextlib.contextmanager
    def patch_multiple(
        target_dict: dict[str, Any],
    ) -> Generator[dict[str, MagicMock], None, None]:
        """Patch multiple targets at once."""
        with contextlib.ExitStack() as stack:
            mocks = {}
            for target, mock_value in target_dict.items():
                if isinstance(mock_value, MagicMock | AsyncMock):
                    mocks[target] = stack.enter_context(patch(target, mock_value))
                else:
                    mocks[target] = stack.enter_context(
                        patch(target, return_value=mock_value)
                    )
            yield mocks


class AsyncTestUtils:
    """Async testing utilities."""

    @staticmethod
    async def run_with_timeout(coro: Any, timeout: float = 10.0) -> Any:
        """Run coroutine with timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)

    @staticmethod
    async def run_concurrently(
        coros: list[Any], return_when: str = "ALL_COMPLETED"
    ) -> list[Any]:
        """Run multiple coroutines concurrently."""
        if return_when == "ALL_COMPLETED":
            return await asyncio.gather(*coros)
        else:
            done, pending = await asyncio.wait(coros, return_when=return_when)

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            return [task.result() for task in done if not task.cancelled()]

    @staticmethod
    async def retry_async(
        func: Callable[[], Any],
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 1.0,
    ) -> Any:
        """Retry an async function with exponential backoff."""
        last_exception = None
        current_delay = delay

        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        assert last_exception is not None
        raise last_exception


class ValidationTestUtils:
    """Validation testing utilities."""

    @staticmethod
    def assert_valid_uuid(uuid_string: str) -> None:
        """Assert that a string is a valid UUID."""
        import uuid

        try:
            uuid.UUID(uuid_string)
        except ValueError:
            pytest.fail(f"Invalid UUID: {uuid_string}")

    @staticmethod
    def assert_valid_timestamp(timestamp: str | float) -> None:
        """Assert that a timestamp is valid."""
        if isinstance(timestamp, str):
            from datetime import datetime

            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pytest.fail(f"Invalid timestamp: {timestamp}")
        elif isinstance(timestamp, int | float):
            assert timestamp > 0, f"Invalid timestamp: {timestamp}"
        else:
            pytest.fail(f"Invalid timestamp type: {type(timestamp)}")

    @staticmethod
    def assert_valid_coordinates(ra: float, dec: float) -> None:
        """Assert that coordinates are valid."""
        assert 0 <= ra <= 360, f"Invalid RA: {ra}"
        assert -90 <= dec <= 90, f"Invalid Dec: {dec}"

    @staticmethod
    def assert_positive_number(value: float, name: str = "value") -> None:
        """Assert that a number is positive."""
        assert value > 0, f"{name} must be positive: {value}"

    @staticmethod
    def assert_in_range(
        value: float, min_val: float, max_val: float, name: str = "value"
    ) -> None:
        """Assert that a value is in range."""
        assert (
            min_val <= value <= max_val
        ), f"{name} must be in range [{min_val}, {max_val}]: {value}"


class PerformanceTestUtils:
    """Performance testing utilities."""

    @staticmethod
    async def measure_throughput(
        func: Callable[[], Any], duration: float = 10.0, warmup: float = 1.0
    ) -> dict[str, float]:
        """Measure function throughput."""
        # Warmup
        warmup_end = time.time() + warmup
        while time.time() < warmup_end:
            await func()

        # Measurement
        start_time = time.time()
        end_time = start_time + duration
        count = 0

        while time.time() < end_time:
            await func()
            count += 1

        actual_duration = time.time() - start_time
        throughput = count / actual_duration

        return {
            "operations": count,
            "duration": actual_duration,
            "throughput": throughput,
            "avg_time": actual_duration / count if count > 0 else 0,
        }

    @staticmethod
    async def measure_latency(
        func: Callable[[], Any], iterations: int = 100
    ) -> dict[str, float]:
        """Measure function latency statistics."""
        latencies = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            await func()
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

        latencies.sort()
        n = len(latencies)

        return {
            "min": latencies[0],
            "max": latencies[-1],
            "mean": sum(latencies) / n,
            "median": latencies[n // 2],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[int(n * 0.99)],
        }


# Test decorators
def slow_test(func: Callable) -> Callable:
    """Mark a test as slow."""
    return pytest.mark.slow(func)


def requires_gpu(func: Callable) -> Callable:
    """Mark a test as requiring GPU."""
    return pytest.mark.requires_gpu(func)


def requires_external(func: Callable) -> Callable:
    """Mark a test as requiring external services."""
    return pytest.mark.requires_external(func)


def performance_test(func: Callable) -> Callable:
    """Mark a test as a performance test."""
    return pytest.mark.performance(func)
