import pathlib

import pytest

try:
    import numpy as np  # type: ignore

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMPY_AVAILABLE = False

from src.adapters.workers.differencing.differencing_workers import DifferencingWorker

DATA_DIR = pathlib.Path("tests/data")
NPY_DIR = DATA_DIR / "npy"


def _load_or_generate_diff(shape=(64, 64)):
    if NUMPY_AVAILABLE and (NPY_DIR / "sample_diff_64.npy").exists():
        return np.load(NPY_DIR / "sample_diff_64.npy")
    if not NUMPY_AVAILABLE:
        pytest.skip("numpy not available for generating synthetic diff image")
    np.random.seed(7)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    diff = np.random.normal(0, 5, shape).astype(np.float32)
    diff += 300 * np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / (2 * 2**2))
    return diff


@pytest.fixture()
def differencing_worker():
    return DifferencingWorker()


@pytest.mark.asyncio
async def test_apply_differencing_algorithm_small(
    differencing_worker: DifferencingWorker,
):
    _ = _load_or_generate_diff()
    result = await differencing_worker.apply_differencing_algorithm(
        observation_id="test_obs_64",
        algorithm="zogy",
    )
    assert result["observation_id"] == "test_obs_64"
    assert result["algorithm"] == "zogy"
    assert "difference_id" in result
    assert "difference_image_path" in result
    assert "algorithm_parameters" in result


@pytest.mark.asyncio
async def test_extract_sources_small(differencing_worker: DifferencingWorker):
    result = await differencing_worker.extract_sources("test_diff_64")
    assert result["difference_id"] == "test_diff_64"
    assert "source_count" in result
    assert "candidates" in result
