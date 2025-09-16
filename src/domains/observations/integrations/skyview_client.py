"""SkyView client for retrieving astronomical survey images."""

import asyncio
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from astropy.io import fits
from pydantic import BaseModel

from src.core.logging import configure_domain_logger


class SkyViewImageMetadata(BaseModel):
    """Metadata extracted from SkyView images."""

    survey: str
    coordinates: tuple[float, float]
    size_degrees: float
    pixels: int
    coordinate_system: str
    projection: str
    pixel_scale: float | None = None
    created_at: datetime
    file_size_bytes: int
    fits_headers: dict[str, Any] | None = None


class SkyViewClient:
    """Client for accessing SkyView astronomical image service."""

    def __init__(
        self,
        base_url: str = "https://skyview.gsfc.nasa.gov/current/cgi",
        timeout: float = 60.0,
        max_retries: int = 3,
        rate_limit_delay: float = 2.0,
    ):
        """Initialize SkyView client.

        Args:
            base_url: SkyView base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_delay: Delay between requests to respect rate limits
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.logger = configure_domain_logger("observations.integrations.skyview")

        # HTTP client configuration
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": "AstrID/1.0 (https://github.com/user/AstrID)",
            },
        )
        self._last_request_time = 0.0

        # Available surveys mapping
        self.available_surveys = {
            # Optical
            "DSS": "Digitized Sky Survey",
            "DSS1 Blue": "Digitized Sky Survey 1 Blue",
            "DSS1 Red": "Digitized Sky Survey 1 Red",
            "DSS2 Blue": "Digitized Sky Survey 2 Blue",
            "DSS2 Red": "Digitized Sky Survey 2 Red",
            "DSS2 IR": "Digitized Sky Survey 2 Infrared",
            # Near-infrared
            "2MASS-J": "2MASS J-band",
            "2MASS-H": "2MASS H-band",
            "2MASS-K": "2MASS K-band",
            # Mid/Far-infrared
            "WISE 3.4": "WISE 3.4 micron",
            "WISE 4.6": "WISE 4.6 micron",
            "WISE 12": "WISE 12 micron",
            "WISE 22": "WISE 22 micron",
            "IRAS 12": "IRAS 12 micron",
            "IRAS 25": "IRAS 25 micron",
            "IRAS 60": "IRAS 60 micron",
            "IRAS 100": "IRAS 100 micron",
            # Radio
            "1420 MHz": "HI 21cm",
            "NVSS": "NRAO VLA Sky Survey",
            "FIRST": "Faint Images of the Radio Sky at Twenty Centimeters",
            # X-ray
            "RASS": "ROSAT All-Sky Survey",
            "PSPC": "ROSAT PSPC",
            # UV
            "GALEX Near UV": "GALEX Near UV",
            "GALEX Far UV": "GALEX Far UV",
        }

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
    ) -> httpx.Response:
        """Make an HTTP request with retry logic and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for httpx request

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: For HTTP errors after all retries
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

                self.logger.debug(f"Request successful: {method} {url}")
                return response

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
                        wait_time = (2**attempt) * 3.0
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
                    wait_time = (2**attempt) * 3.0
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

    async def get_survey_image(
        self,
        coordinates: tuple[float, float],
        survey: str = "DSS",
        size: float = 0.25,
        pixels: int = 512,
        coordinate_system: str = "J2000",
        projection: str = "Tan",
        return_format: str = "FITS",
    ) -> bytes:
        """Get survey image for given coordinates.

        Args:
            coordinates: (RA, Dec) in degrees
            survey: Survey name (e.g., 'DSS', '2MASS-J')
            size: Image size in degrees
            pixels: Image size in pixels (square)
            coordinate_system: Coordinate system ('J2000', 'B1950', 'Galactic', 'Ecliptic')
            projection: Map projection ('Tan', 'Sin', 'Arc')
            return_format: Return format ('FITS', 'JPEG', 'GIF')

        Returns:
            Image data as bytes

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
        if size <= 0:
            raise ValueError(f"Invalid size: {size}. Must be positive")
        if pixels <= 0:
            raise ValueError(f"Invalid pixels: {pixels}. Must be positive")

        self.logger.info(
            f"Getting {survey} image at RA={ra:.4f}°, Dec={dec:.4f}°, size={size:.4f}°, {pixels}px"
        )

        # Build request parameters
        params = {
            "Position": f"{ra:.6f},{dec:.6f}",
            "Survey": survey,
            "Size": f"{size:.6f}",
            "Pixels": str(pixels),
            "Coordinates": coordinate_system,
            "Projection": projection,
            "Return": return_format,
            "EBINS": "",  # Empty for most surveys
        }

        try:
            response = await self._make_request("GET", "runquery.pl", params=params)

            # Check if we got an error page
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                # SkyView returns HTML error pages
                error_text = response.text
                if "error" in error_text.lower() or "not found" in error_text.lower():
                    raise ValueError(
                        f"SkyView error: Image not available for {survey} at these coordinates"
                    )

            data = response.content
            self.logger.info(f"Retrieved {len(data)} bytes for {survey} image")

            if len(data) == 0:
                raise ValueError(f"Empty response for {survey} image")

            return data

        except Exception as e:
            self.logger.error(f"Failed to get {survey} image: {e}")
            raise

    async def get_multiple_surveys(
        self,
        coordinates: tuple[float, float],
        surveys: list[str],
        size: float = 0.25,
        pixels: int = 512,
        coordinate_system: str = "J2000",
        projection: str = "Tan",
        return_format: str = "FITS",
    ) -> dict[str, bytes]:
        """Get images from multiple surveys for the same coordinates.

        Args:
            coordinates: (RA, Dec) in degrees
            surveys: List of survey names
            size: Image size in degrees
            pixels: Image size in pixels
            coordinate_system: Coordinate system
            projection: Map projection
            return_format: Return format

        Returns:
            Dictionary mapping survey names to image data

        Raises:
            ValueError: For invalid input parameters
        """
        if not surveys:
            raise ValueError("Survey list cannot be empty")

        self.logger.info(f"Getting images from {len(surveys)} surveys: {surveys}")

        # Create tasks for concurrent requests
        tasks = {}
        for survey in surveys:
            task = asyncio.create_task(
                self.get_survey_image(
                    coordinates=coordinates,
                    survey=survey,
                    size=size,
                    pixels=pixels,
                    coordinate_system=coordinate_system,
                    projection=projection,
                    return_format=return_format,
                )
            )
            tasks[task] = survey

        # Wait for all requests to complete
        results = {}
        for task in asyncio.as_completed(tasks.keys()):
            survey_name = tasks[task]
            try:
                image_data = await task
                results[survey_name] = image_data
                self.logger.info(f"Successfully retrieved {survey_name} image")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve {survey_name} image: {e}")
                # Continue with other surveys even if one fails

        self.logger.info(
            f"Successfully retrieved {len(results)}/{len(surveys)} survey images"
        )
        return results

    async def get_image_metadata(
        self, image_data: bytes, survey: str | None = None
    ) -> SkyViewImageMetadata:
        """Extract metadata from image data.

        Args:
            image_data: FITS image data
            survey: Survey name for metadata

        Returns:
            Image metadata

        Raises:
            ValueError: For invalid FITS data
        """
        self.logger.debug("Extracting metadata from image data")

        try:
            # Write to temporary file for astropy processing
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name

            try:
                # Open FITS file and extract metadata
                with fits.open(temp_path) as hdul:
                    primary_hdu = hdul[0]
                    header = primary_hdu.header

                    # Extract coordinate information
                    ra = header.get("CRVAL1", 0.0)
                    dec = header.get("CRVAL2", 0.0)

                    # Extract image size information
                    naxis1 = header.get("NAXIS1", 0)
                    naxis2 = header.get("NAXIS2", 0)
                    pixels = max(naxis1, naxis2)  # Assume square for simplicity

                    # Calculate pixel scale and image size
                    cdelt1 = abs(header.get("CDELT1", 0.0))
                    cdelt2 = abs(header.get("CDELT2", 0.0))
                    pixel_scale = max(cdelt1, cdelt2) if cdelt1 and cdelt2 else None

                    size_degrees = (
                        pixel_scale * pixels if pixel_scale else 0.25
                    )  # Default fallback

                    # Extract coordinate system and projection
                    ctype1 = header.get("CTYPE1", "RA---TAN")
                    coordinate_system = "J2000"  # Default
                    projection = "Tan"  # Default

                    if "GLON" in ctype1:
                        coordinate_system = "Galactic"
                    elif "ELON" in ctype1:
                        coordinate_system = "Ecliptic"

                    if "SIN" in ctype1:
                        projection = "Sin"
                    elif "ARC" in ctype1:
                        projection = "Arc"

                    # Convert header to dictionary
                    fits_headers = dict(header)

                metadata = SkyViewImageMetadata(
                    survey=survey or "Unknown",
                    coordinates=(ra, dec),
                    size_degrees=size_degrees,
                    pixels=pixels,
                    coordinate_system=coordinate_system,
                    projection=projection,
                    pixel_scale=pixel_scale,
                    created_at=datetime.now(),
                    file_size_bytes=len(image_data),
                    fits_headers=fits_headers,
                )

                self.logger.debug(
                    f"Extracted metadata: {metadata.survey} at ({ra:.4f}, {dec:.4f})"
                )
                return metadata

            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.error(f"Failed to extract image metadata: {e}")
            raise ValueError(f"Invalid FITS data: {e}") from e

    async def get_available_surveys(self) -> dict[str, str]:
        """Get available surveys and their descriptions.

        Returns:
            Dictionary mapping survey names to descriptions
        """
        self.logger.info("Returning available SkyView surveys")
        return self.available_surveys.copy()

    async def validate_survey(self, survey: str) -> bool:
        """Validate if a survey is available.

        Args:
            survey: Survey name to validate

        Returns:
            True if survey is available
        """
        return survey in self.available_surveys

    async def get_survey_coverage(
        self,
        survey: str,
        coordinates: tuple[float, float],
        size: float = 0.1,
    ) -> bool:
        """Check if a survey has coverage at given coordinates.

        Args:
            survey: Survey name
            coordinates: (RA, Dec) in degrees
            size: Test image size in degrees

        Returns:
            True if survey has coverage at coordinates
        """
        ra, dec = coordinates
        self.logger.debug(f"Checking {survey} coverage at RA={ra:.4f}°, Dec={dec:.4f}°")

        try:
            # Try to get a small test image
            test_image = await self.get_survey_image(
                coordinates=coordinates,
                survey=survey,
                size=size,
                pixels=64,  # Small test image
            )

            # If we got data, there's coverage
            has_coverage = len(test_image) > 1000  # Minimum reasonable FITS file size

            self.logger.debug(
                f"{survey} coverage at ({ra:.4f}, {dec:.4f}): {has_coverage}"
            )
            return has_coverage

        except Exception as e:
            self.logger.debug(f"No {survey} coverage at ({ra:.4f}, {dec:.4f}): {e}")
            return False

    async def batch_download_surveys(
        self,
        coordinates_list: list[tuple[float, float]],
        surveys: list[str],
        size: float = 0.25,
        pixels: int = 512,
        max_concurrent: int = 5,
    ) -> dict[tuple[float, float], dict[str, bytes]]:
        """Batch download survey images for multiple coordinates.

        Args:
            coordinates_list: List of (RA, Dec) coordinate pairs
            surveys: List of survey names
            size: Image size in degrees
            pixels: Image size in pixels
            max_concurrent: Maximum concurrent downloads

        Returns:
            Dictionary mapping coordinates to survey image data
        """
        self.logger.info(
            f"Batch downloading {len(surveys)} surveys for {len(coordinates_list)} coordinates"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_for_coordinates(
            coords: tuple[float, float],
        ) -> tuple[tuple[float, float], dict[str, bytes]]:
            async with semaphore:
                try:
                    survey_data = await self.get_multiple_surveys(
                        coordinates=coords,
                        surveys=surveys,
                        size=size,
                        pixels=pixels,
                    )
                    return coords, survey_data
                except Exception as e:
                    self.logger.warning(f"Failed to download surveys for {coords}: {e}")
                    return coords, {}

        # Create tasks for all coordinate pairs
        tasks = [download_for_coordinates(coords) for coords in coordinates_list]

        # Wait for all downloads to complete
        results = {}
        for task in asyncio.as_completed(tasks):
            coords, survey_data = await task
            results[coords] = survey_data

        successful_coords = len([r for r in results.values() if r])
        self.logger.info(
            f"Batch download complete: {successful_coords}/{len(coordinates_list)} coordinates successful"
        )

        return results
