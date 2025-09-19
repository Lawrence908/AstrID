import pathlib
import uuid

import pytest

try:
    import numpy as np  # type: ignore

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMPY_AVAILABLE = False

from src.adapters.workers.preprocessing.preprocessing_workers import PreprocessingWorker

DATA_DIR = pathlib.Path("tests/data")
NPY_DIR = DATA_DIR / "npy"


def _load_or_generate_image(shape=(64, 64)):
    if NUMPY_AVAILABLE and (NPY_DIR / "sample_image_64.npy").exists():
        return np.load(NPY_DIR / "sample_image_64.npy")
    if not NUMPY_AVAILABLE:
        pytest.skip("numpy not available for generating synthetic image")
    np.random.seed(42)
    img = np.random.normal(1000, 30, shape).astype(np.float32)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    img += 5000 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 3**2))
    return img


@pytest.fixture()
def preprocessing_worker():
    return PreprocessingWorker()


@pytest.mark.asyncio
async def test_preprocessing_quality_metrics(preprocessing_worker: PreprocessingWorker):
    _load_or_generate_image()

    # Exercise quality path that returns metrics
    test_obs_id = uuid.uuid4()
    result = await preprocessing_worker.assess_quality(observation_id=test_obs_id)

    assert "observation_id" in result
    assert result["observation_id"] == test_obs_id
    assert "overall_quality_score" in result
    # Ensure key metric groups are present
    for key in ("background_analysis", "noise_analysis"):
        assert key in result


@pytest.mark.asyncio
async def test_preprocessing_calibration_minimal(
    preprocessing_worker: PreprocessingWorker,
):
    # Minimal validation — primarily that structure is correct on small input
    test_obs_id = uuid.uuid4()
    result = await preprocessing_worker.apply_calibration(
        observation_id=test_obs_id, calibration_frames={}
    )
    assert result["observation_id"] == test_obs_id
    assert result["calibration_applied"] is True
    for key in ("bias_correction", "dark_correction", "flat_correction"):
        assert key in result
