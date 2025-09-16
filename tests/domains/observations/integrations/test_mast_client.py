"""Tests for MAST API client."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.domains.observations.integrations.mast_client import (
    MASTClient,
    MASTSearchResult,
)


class TestMASTClient:
    """Test suite for MAST API client."""

    @pytest.fixture
    async def client(self):
        """Create MAST client for testing."""
        client = MASTClient(timeout=10.0, max_retries=1)
        yield client
        await client.close()

    @pytest.fixture
    def mock_mast_response(self):
        """Mock MAST API response data."""
        return {
            "data": [
                {
                    "obs_id": "hst_12345_01_acs_wfc_f606w",
                    "s_ra": 210.8023,
                    "s_dec": 54.3489,
                    "t_min": 58849.5,  # MJD
                    "filters": "F606W",
                    "t_exptime": 507.0,
                    "instrument_name": "ACS",
                    "obs_collection": "HST",
                    "target_name": "NGC5194",
                    "dataURL": "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/hst_12345_01_acs_wfc_f606w_drz.fits",
                    "s_resolution": 0.05,
                },
                {
                    "obs_id": "jwst_02345_01_nircam_f200w",
                    "s_ra": 210.8025,
                    "s_dec": 54.3491,
                    "t_min": 59500.2,  # MJD
                    "filters": "F200W",
                    "t_exptime": 1065.0,
                    "instrument_name": "NIRCAM",
                    "obs_collection": "JWST",
                    "target_name": "NGC5194",
                    "dataURL": "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/jwst_02345_01_nircam_f200w_i2d.fits",
                },
            ]
        }

    @pytest.mark.asyncio
    async def test_search_observations_success(self, client, mock_mast_response):
        """Test successful observation search."""
        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_mast_response

            results = await client.search_observations(
                coordinates=(210.8, 54.35),
                radius=0.1,
                missions=["HST", "JWST"],
                limit=100,
            )

            assert len(results) == 2
            assert isinstance(results[0], MASTSearchResult)
            assert results[0].observation_id == "hst_12345_01_acs_wfc_f606w"
            assert results[0].mission == "HST"
            assert results[0].ra == 210.8023
            assert results[0].dec == 54.3489

            # Verify request was made correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "invoke"

    @pytest.mark.asyncio
    async def test_search_observations_invalid_coordinates(self, client):
        """Test search with invalid coordinates."""
        with pytest.raises(ValueError, match="Invalid RA"):
            await client.search_observations(coordinates=(400.0, 54.35))

        with pytest.raises(ValueError, match="Invalid Dec"):
            await client.search_observations(coordinates=(210.8, 100.0))

        with pytest.raises(ValueError, match="Invalid radius"):
            await client.search_observations(coordinates=(210.8, 54.35), radius=-1.0)

    @pytest.mark.asyncio
    async def test_search_observations_with_time_range(
        self, client, mock_mast_response
    ):
        """Test search with time range filters."""
        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_mast_response

            start_time = datetime(2023, 1, 1)
            end_time = datetime(2023, 12, 31)

            results = await client.search_observations(
                coordinates=(210.8, 54.35), start_time=start_time, end_time=end_time
            )

            assert len(results) == 2

            # Check that time parameters were included in request
            call_args = mock_request.call_args
            request_data = call_args[1]["json"]
            params = request_data["params"]
            assert "t_min" in params
            assert "t_max" in params

    @pytest.mark.asyncio
    async def test_get_observation_metadata_success(self, client):
        """Test successful metadata retrieval."""
        mock_metadata = {
            "data": [
                {
                    "obs_id": "hst_12345_01_acs_wfc_f606w",
                    "instrument_name": "ACS",
                    "detector": "WFC",
                    "proposal_id": "12345",
                    "target_name": "NGC5194",
                    "s_ra": 210.8023,
                    "s_dec": 54.3489,
                    "t_exptime": 507.0,
                    "filters": "F606W",
                }
            ]
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_metadata

            metadata = await client.get_observation_metadata(
                "hst_12345_01_acs_wfc_f606w"
            )

            assert metadata["obs_id"] == "hst_12345_01_acs_wfc_f606w"
            assert metadata["instrument_name"] == "ACS"
            assert metadata["proposal_id"] == "12345"

    @pytest.mark.asyncio
    async def test_get_observation_metadata_not_found(self, client):
        """Test metadata retrieval for non-existent observation."""
        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {"data": []}

            with pytest.raises(ValueError, match="not found in MAST"):
                await client.get_observation_metadata("non_existent_obs")

    @pytest.mark.asyncio
    async def test_download_observation_data_success(self, client):
        """Test successful data download."""
        mock_metadata = {
            "data": [
                {
                    "obs_id": "hst_12345_01_acs_wfc_f606w",
                    "dataURL": "https://example.com/test.fits",
                }
            ]
        }

        mock_fits_data = (
            b"SIMPLE  =                    T / file does conform to FITS standard"
        )

        with patch.object(
            client, "get_observation_metadata", new_callable=AsyncMock
        ) as mock_metadata_call:
            mock_metadata_call.return_value = mock_metadata["data"][0]

            with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get:
                mock_response = AsyncMock()
                mock_response.content = mock_fits_data
                mock_response.raise_for_status = AsyncMock()
                mock_get.return_value = mock_response

                data = await client.download_observation_data(
                    "hst_12345_01_acs_wfc_f606w"
                )

                assert data == mock_fits_data
                mock_get.assert_called_once_with("https://example.com/test.fits")

    @pytest.mark.asyncio
    async def test_download_observation_data_no_url(self, client):
        """Test download when no data URL is available."""
        mock_metadata = {
            "obs_id": "hst_12345_01_acs_wfc_f606w",
            # No dataURL field
        }

        with patch.object(
            client, "get_observation_metadata", new_callable=AsyncMock
        ) as mock_metadata_call:
            mock_metadata_call.return_value = mock_metadata

            with pytest.raises(ValueError, match="No data URL found"):
                await client.download_observation_data("hst_12345_01_acs_wfc_f606w")

    @pytest.mark.asyncio
    async def test_get_available_missions_success(self, client):
        """Test getting available missions."""
        mock_response = {
            "data": [
                {"obs_collection": "HST"},
                {"obs_collection": "JWST"},
                {"obs_collection": "KEPLER"},
                {"obs_collection": "HST"},  # Duplicate
            ]
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            missions = await client.get_available_missions()

            assert len(missions) == 3  # Duplicates removed
            assert "HST" in missions
            assert "JWST" in missions
            assert "KEPLER" in missions
            assert sorted(missions) == missions  # Should be sorted

    @pytest.mark.asyncio
    async def test_get_mission_statistics_success(self, client):
        """Test getting mission statistics."""
        mock_response = {
            "data": [
                {
                    "count": 12345,
                    "earliest": 50000.0,
                    "latest": 60000.0,
                }
            ]
        }

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            stats = await client.get_mission_statistics("HST")

            assert stats["mission"] == "HST"
            assert stats["observation_count"] == 12345
            assert stats["earliest_observation"] == 50000.0
            assert stats["latest_observation"] == 60000.0

    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting between requests."""
        import time

        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {"data": []}

            start_time = time.time()

            # Make two requests
            await client.search_observations(coordinates=(210.8, 54.35))
            await client.search_observations(coordinates=(210.8, 54.35))

            end_time = time.time()

            # Should have waited at least the rate limit delay
            assert end_time - start_time >= client.rate_limit_delay

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, client):
        """Test retry logic on server errors."""
        with patch.object(
            client.client, "request", new_callable=AsyncMock
        ) as mock_request:
            # First call fails with 500, second succeeds
            mock_response_error = AsyncMock()
            mock_response_error.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error", request=AsyncMock(), response=AsyncMock(status_code=500)
            )

            mock_response_success = AsyncMock()
            mock_response_success.raise_for_status = AsyncMock()
            mock_response_success.json.return_value = {"data": []}

            mock_request.side_effect = [mock_response_error, mock_response_success]

            result = await client._make_request("GET", "test")

            assert result == {"data": []}
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, client):
        """Test retry logic on rate limit errors."""
        with patch.object(
            client.client, "request", new_callable=AsyncMock
        ) as mock_request:
            # First call gets rate limited, second succeeds
            mock_response_error = AsyncMock()
            mock_response_error.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Rate Limited", request=AsyncMock(), response=AsyncMock(status_code=429)
            )

            mock_response_success = AsyncMock()
            mock_response_success.raise_for_status = AsyncMock()
            mock_response_success.json.return_value = {"data": []}

            mock_request.side_effect = [mock_response_error, mock_response_success]

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await client._make_request("GET", "test")

                assert result == {"data": []}
                assert mock_request.call_count == 2
                mock_sleep.assert_called()  # Should have slept before retry

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, client):
        """Test behavior when max retries are exceeded."""
        with patch.object(
            client.client, "request", new_callable=AsyncMock
        ) as mock_request:
            # All calls fail with 500
            mock_response_error = AsyncMock()
            mock_response_error.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error", request=AsyncMock(), response=AsyncMock(status_code=500)
            )

            mock_request.return_value = mock_response_error

            with pytest.raises(httpx.HTTPStatusError):
                await client._make_request("GET", "test")

            # Should have tried max_retries + 1 times
            assert mock_request.call_count == client.max_retries + 1

    @pytest.mark.asyncio
    async def test_convert_mast_observation_invalid_data(self, client):
        """Test conversion with invalid observation data."""
        invalid_data = {
            # Missing required fields
            "obs_id": "test_obs",
            # Missing s_ra, s_dec, t_min, etc.
        }

        result = client._convert_mast_observation(invalid_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_convert_mast_observation_minimal_data(self, client):
        """Test conversion with minimal valid data."""
        minimal_data = {
            "obs_id": "test_obs",
            "s_ra": 210.8,
            "s_dec": 54.35,
            "t_min": 58849.5,
            "filters": "V",
            "t_exptime": 300.0,
        }

        result = client._convert_mast_observation(minimal_data)

        assert result is not None
        assert result.observation_id == "test_obs"
        assert result.ra == 210.8
        assert result.dec == 54.35
        assert result.filter_band == "V"
        assert result.exposure_time == 300.0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test MASTClient as context manager."""
        async with MASTClient() as client:
            assert client.client is not None

        # Client should be closed after context exit
        # Note: We can't easily test this without implementation details

    @pytest.mark.asyncio
    async def test_search_with_empty_response(self, client):
        """Test search when API returns empty results."""
        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {"data": []}

            results = await client.search_observations(coordinates=(210.8, 54.35))

            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_with_malformed_response(self, client):
        """Test search when API returns malformed data."""
        with patch.object(
            client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {}  # Missing 'data' key

            results = await client.search_observations(coordinates=(210.8, 54.35))

            assert len(results) == 0
