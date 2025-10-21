import json
import pathlib

import pytest

from src.adapters.workers.ingestion.observation_workers import (
    ObservationIngestionWorker,
)

DATA_FILE = pathlib.Path("tests/data/astr92_observation_cases.json")


@pytest.fixture(scope="module")
def ingestion_cases():
    data = json.loads(DATA_FILE.read_text())
    return data


@pytest.fixture()
def ingestion_worker():
    return ObservationIngestionWorker()


@pytest.mark.asyncio
async def test_valid_ingestion_cases(
    ingestion_worker: ObservationIngestionWorker, ingestion_cases
):
    for case in ingestion_cases.get("valid_cases", []):
        result = await ingestion_worker.validate_observation_data(case["input"])
        assert result["valid"] is True
        assert not result.get("errors")


@pytest.mark.asyncio
async def test_invalid_ingestion_cases(
    ingestion_worker: ObservationIngestionWorker, ingestion_cases
):
    for case in ingestion_cases.get("invalid_cases", []):
        result = await ingestion_worker.validate_observation_data(case["input"])
        assert result["valid"] is False
        expected_subs = case.get("expected_error_substrings", [])
        if expected_subs:
            error_blob = " ".join(result.get("errors", []))
            assert any(sub in error_blob for sub in expected_subs)
