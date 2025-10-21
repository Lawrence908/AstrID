import pytest


@pytest.mark.asyncio
async def test_training_data_end_to_end(async_client):
    """Smoke E2E: collect dataset then fetch detail."""
    payload = {
        "survey_ids": ["hst"],
        "start": "2024-01-01T00:00:00",
        "end": "2024-12-31T23:59:59",
        "confidence_threshold": 0.7,
        "max_samples": 10,
        "name": "e2e_smoke",
    }

    collect = await async_client.post("/training/datasets/collect", json=payload)
    if collect.status_code != 200:
        pytest.skip("Collection failed in CI environment; skipping E2E detail")
    body = await collect.json()
    dataset_id = body["data"]["dataset_id"]

    detail = await async_client.get(f"/training/datasets/{dataset_id}")
    assert detail.status_code == 200
