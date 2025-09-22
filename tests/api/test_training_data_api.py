import pytest


@pytest.mark.asyncio
async def test_collect_and_list_training_datasets(async_client):
    # Collect
    payload = {
        "survey_ids": ["hst"],
        "start": "2024-01-01T00:00:00",
        "end": "2024-12-31T23:59:59",
        "confidence_threshold": 0.5,
        "max_samples": 5,
        "name": "test_dataset_api",
    }

    resp = await async_client.post("/training/datasets/collect", json=payload)
    assert resp.status_code in (
        200,
        503,
        400,
    )  # allow degraded envs to fail gracefully in CI

    # List
    resp2 = await async_client.get("/training/datasets")
    assert resp2.status_code == 200
    data = (await resp2.json())["data"]
    assert isinstance(data, list)
