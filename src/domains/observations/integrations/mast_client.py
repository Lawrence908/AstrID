"""MAST (Mikulski Archive for Space Telescopes) API client for astronomical observations."""

import asyncio
import time
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

from src.core.logging import configure_domain_logger


class MASTSearchResult(BaseModel):
    """MAST search result response model."""

    observation_id: str
    target_name: str | None = None
    ra: float
    dec: float
    observation_time: datetime
    filter_band: str
    exposure_time: float
    instrument: str | None = None
    mission: str | None = None
    fits_url: str | None = None
    preview_url: str | None = None
    airmass: float | None = None
    pixel_scale: float | None = None
    image_width: int | None = None
    image_height: int | None = None


class MASTClient:
    """Client for accessing MAST (Mikulski Archive for Space Telescopes) API."""

    def __init__(
        self,
        base_url: str = "https://mast.stsci.edu/api/v0.1",
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
    ):
        """Initialize MAST client.

        Args:
            base_url: MAST API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_delay: Delay between requests to respect rate limits
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.logger = configure_domain_logger("observations.integrations.mast")

        # HTTP client configuration
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": "AstrID/1.0 (https://github.com/user/AstrID)",
                "Accept": "application/json",
            },
        )
        self._last_request_time = 0.0

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self._last_request_time = time.time()

    async def _make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for httpx request

        Returns:
            Response JSON data

        Raises:
            httpx.HTTPError: For HTTP errors after all retries
            ValueError: For invalid response data
        """
        await self._rate_limit()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(
                    f"Making {method} request to {url} (attempt {attempt + 1})"
                )
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()

                # Handle empty responses
                if not response.content:
                    return {}

                data = response.json()
                self.logger.debug(f"Request successful: {method} {url}")
                return data

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries:
                        wait_time = (2**attempt) * self.rate_limit_delay
                        self.logger.warning(
                            f"Rate limited, waiting {wait_time:.1f}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        self.logger.error(
                            f"Rate limit exceeded after {self.max_retries} retries"
                        )
                        raise

                elif 500 <= e.response.status_code < 600:  # Server errors
                    if attempt < self.max_retries:
                        wait_time = (2**attempt) * 2.0
                        self.logger.warning(
                            f"Server error {e.response.status_code}, retrying in {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        self.logger.error(
                            f"Server errors persisted after {self.max_retries} retries"
                        )
                        raise
                else:
                    self.logger.error(f"HTTP error {e.response.status_code}: {e}")
                    raise

            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    wait_time = (2**attempt) * 2.0
                    self.logger.warning(
                        f"Request error, retrying in {wait_time:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self.logger.error(
                        f"Request failed after {self.max_retries} retries: {e}"
                    )
                    raise

        raise RuntimeError("Should not reach here")

    async def search_observations(
        self,
        coordinates: tuple[float, float],
        radius: float = 0.1,
        missions: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[MASTSearchResult]:
        """Search for observations at given coordinates.

        Args:
            coordinates: (RA, Dec) in degrees
            radius: Search radius in degrees
            missions: List of missions to search (e.g., ['HST', 'JWST'])
            start_time: Start of observation time range
            end_time: End of observation time range
            limit: Maximum number of results

        Returns:
            List of MAST search results

        Raises:
            ValueError: For invalid input parameters
            httpx.HTTPError: For API errors
        """
        ra, dec = coordinates

        # Validate coordinates
        if not (0 <= ra <= 360):
            raise ValueError(f"Invalid RA: {ra}. Must be between 0 and 360 degrees")
        if not (-90 <= dec <= 90):
            raise ValueError(f"Invalid Dec: {dec}. Must be between -90 and 90 degrees")
        if radius <= 0:
            raise ValueError(f"Invalid radius: {radius}. Must be positive")

        self.logger.info(
            f"Searching MAST observations at RA={ra:.4f}°, Dec={dec:.4f}°, radius={radius:.4f}°"
        )

        # Build search parameters
        params = {
            "service": "Mast.Caom.Cone",
            "params": {
                "ra": ra,
                "dec": dec,
                "radius": radius,
            },
            "format": "json",
            "pagesize": min(limit, 50000),  # MAST limit
            "page": 1,
        }

        # Add mission filter if specified
        if missions:
            # Convert missions to collection names
            mission_mapping = {
                "HST": "HST",
                "JWST": "JWST",
                "KEPLER": "Kepler",
                "TESS": "TESS",
                "GALEX": "GALEX",
            }
            collections = [mission_mapping.get(m, m) for m in missions]
            params["params"]["collection"] = collections

        # Add time range filters
        if start_time:
            params["params"]["t_min"] = (
                start_time.timestamp() / 86400.0 + 2440587.5
            )  # Convert to MJD
        if end_time:
            params["params"]["t_max"] = (
                end_time.timestamp() / 86400.0 + 2440587.5
            )  # Convert to MJD

        try:
            response_data = await self._make_request("POST", "invoke", json=params)

            if "data" not in response_data:
                self.logger.warning("No data field in MAST response")
                return []

            observations = response_data["data"]
            self.logger.info(f"Found {len(observations)} observations from MAST")

            # Convert to our result format
            results = []
            for obs in observations:
                try:
                    result = self._convert_mast_observation(obs)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to convert observation: {e}")
                    continue

            self.logger.info(f"Successfully converted {len(results)} observations")
            return results

        except Exception as e:
            self.logger.error(f"Failed to search MAST observations: {e}")
            raise

    def _convert_mast_observation(
        self, obs_data: dict[str, Any]
    ) -> MASTSearchResult | None:
        """Convert MAST observation data to our format.

        Args:
            obs_data: Raw observation data from MAST

        Returns:
            Converted observation or None if conversion fails
        """
        try:
            # Extract required fields with fallbacks
            observation_id = obs_data.get("obs_id") or obs_data.get("obsid", "unknown")

            # Coordinates
            ra = float(obs_data.get("s_ra", 0))
            dec = float(obs_data.get("s_dec", 0))

            # Observation time - handle various formats
            obs_time_mjd = obs_data.get("t_min")
            if obs_time_mjd:
                # Convert MJD to datetime
                obs_time = datetime.fromtimestamp(
                    (float(obs_time_mjd) - 2440587.5) * 86400.0
                )
            else:
                obs_time = datetime.now()  # Fallback

            # Filter and exposure
            filter_band = obs_data.get("filters") or obs_data.get(
                "filter_name", "unknown"
            )
            exposure_time = float(obs_data.get("t_exptime", 0) or 0)

            # URLs
            fits_url = obs_data.get("dataURL") or obs_data.get("data_uri")
            preview_url = obs_data.get("jpegURL") or obs_data.get("preview_uri")

            # Optional metadata
            instrument = obs_data.get("instrument_name")
            mission = obs_data.get("obs_collection")
            target_name = obs_data.get("target_name")

            # Image metadata
            pixel_scale = obs_data.get("s_resolution")
            if pixel_scale:
                pixel_scale = float(pixel_scale)

            # Create result
            result = MASTSearchResult(
                observation_id=observation_id,
                target_name=target_name,
                ra=ra,
                dec=dec,
                observation_time=obs_time,
                filter_band=filter_band,
                exposure_time=exposure_time,
                instrument=instrument,
                mission=mission,
                fits_url=fits_url,
                preview_url=preview_url,
                pixel_scale=pixel_scale,
            )

            return result

        except (KeyError, ValueError, TypeError) as e:
            self.logger.warning(
                f"Failed to convert observation {obs_data.get('obs_id', 'unknown')}: {e}"
            )
            return None

    async def get_observation_metadata(self, observation_id: str) -> dict[str, Any]:
        """Get detailed metadata for a specific observation.

        Args:
            observation_id: MAST observation ID

        Returns:
            Detailed observation metadata

        Raises:
            ValueError: If observation not found
            httpx.HTTPError: For API errors
        """
        self.logger.info(f"Getting MAST observation metadata for {observation_id}")

        params = {
            "service": "Mast.Caom.Filtered",
            "params": {
                "columns": "*",
                "filters": [
                    {
                        "paramName": "obs_id",
                        "values": [observation_id],
                    }
                ],
            },
            "format": "json",
        }

        try:
            response_data = await self._make_request("POST", "invoke", json=params)

            if "data" not in response_data or not response_data["data"]:
                raise ValueError(f"Observation {observation_id} not found in MAST")

            metadata = response_data["data"][0]
            self.logger.info(f"Retrieved metadata for observation {observation_id}")
            return metadata

        except Exception as e:
            self.logger.error(
                f"Failed to get observation metadata for {observation_id}: {e}"
            )
            raise

    async def download_observation_data(self, observation_id: str) -> bytes:
        """Download observation data file.

        Args:
            observation_id: MAST observation ID

        Returns:
            Raw observation data

        Raises:
            ValueError: If observation or data URL not found
            httpx.HTTPError: For download errors
        """
        self.logger.info(f"Downloading observation data for {observation_id}")

        try:
            # First get the metadata to find the download URL
            metadata = await self.get_observation_metadata(observation_id)

            data_url = metadata.get("dataURL") or metadata.get("data_uri")
            if not data_url:
                raise ValueError(f"No data URL found for observation {observation_id}")

            # Download the data
            self.logger.info(f"Downloading from URL: {data_url}")
            response = await self.client.get(data_url)
            response.raise_for_status()

            data = response.content
            self.logger.info(
                f"Downloaded {len(data)} bytes for observation {observation_id}"
            )
            return data

        except Exception as e:
            self.logger.error(
                f"Failed to download observation data for {observation_id}: {e}"
            )
            raise

    async def get_available_missions(self) -> list[str]:
        """Get list of available missions/collections.

        Returns:
            List of available mission names
        """
        self.logger.info("Getting available MAST missions")

        params = {
            "service": "Mast.Caom.All",
            "params": {
                "columns": "DISTINCT obs_collection",
            },
            "format": "json",
            "pagesize": 1000,
        }

        try:
            response_data = await self._make_request("POST", "invoke", json=params)

            if "data" not in response_data:
                return []

            missions = [
                item["obs_collection"]
                for item in response_data["data"]
                if item.get("obs_collection")
            ]
            missions = sorted(set(missions))  # Remove duplicates and sort

            self.logger.info(f"Found {len(missions)} available missions")
            return missions

        except Exception as e:
            self.logger.error(f"Failed to get available missions: {e}")
            return []

    async def get_mission_statistics(self, mission: str) -> dict[str, Any]:
        """Get statistics for a specific mission.

        Args:
            mission: Mission name

        Returns:
            Mission statistics including observation counts
        """
        self.logger.info(f"Getting statistics for mission {mission}")

        params = {
            "service": "Mast.Caom.Filtered.Position",
            "params": {
                "columns": "COUNT(*) as count, MIN(t_min) as earliest, MAX(t_max) as latest",
                "filters": [
                    {
                        "paramName": "obs_collection",
                        "values": [mission],
                    }
                ],
            },
            "format": "json",
        }

        try:
            response_data = await self._make_request("POST", "invoke", json=params)

            if "data" not in response_data or not response_data["data"]:
                return {"mission": mission, "count": 0}

            stats = response_data["data"][0]

            result = {
                "mission": mission,
                "observation_count": stats.get("count", 0),
                "earliest_observation": stats.get("earliest"),
                "latest_observation": stats.get("latest"),
            }

            self.logger.info(f"Retrieved statistics for mission {mission}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to get mission statistics for {mission}: {e}")
            return {"mission": mission, "count": 0, "error": str(e)}
