"""Example test showing how mocked external adapters are used."""

from __future__ import annotations

import pytest

from src.domains.observations.ingestion.services.data_ingestion import (
    DataIngestionService,
)


@pytest.mark.asyncio
async def test_ingestion_service_uses_injected_mocks(
    mast_client, skyview_client, r2_client
):
    service = DataIngestionService(
        mast_client=mast_client, skyview_client=skyview_client, r2_client=r2_client
    )

    # Call a method that doesn't require real network access in our test
    # Using a tiny radius just to confirm the method can be invoked with mocks
    result = await service.ingest_observations_by_position(
        ra=180.0,
        dec=45.0,
        survey_id="00000000-0000-0000-0000-000000000000",  # type: ignore[arg-type]
    )

    assert isinstance(result, list)
