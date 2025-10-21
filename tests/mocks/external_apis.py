"""Mock external API clients for testing."""

from __future__ import annotations

from typing import Any


class MockMASTClient:
    """Mock MAST (Mikulski Archive for Space Telescopes) client."""

    def __init__(self):
        self.observations: list[dict[str, Any]] = []
        self.error_on_operation: dict[str, bool] = {}
        self.delay_seconds = 0.0

        # Add some default observations
        self._add_default_observations()

    def _add_default_observations(self) -> None:
        """Add default test observations."""
        default_observations = [
            {
                "obs_id": "mast-obs-001",
                "target_name": "NGC 1234",
                "ra": 180.0,
                "dec": 45.0,
                "observation_start_time": "2024-01-01T00:00:00",
                "exposure_time": 300.0,
                "filters": "g,r,i",
                "instrument": "ACS/WFC",
                "telescope": "HST",
                "data_product_type": "image",
                "obs_collection": "HST",
                "project": "GO-12345",
            },
            {
                "obs_id": "mast-obs-002",
                "target_name": "M31",
                "ra": 10.684,
                "dec": 41.269,
                "observation_start_time": "2024-01-02T00:00:00",
                "exposure_time": 600.0,
                "filters": "F814W",
                "instrument": "WFC3/UVIS",
                "telescope": "HST",
                "data_product_type": "image",
                "obs_collection": "HST",
                "project": "GO-14234",
            },
        ]
        self.observations.extend(default_observations)

    async def query_observations(
        self,
        target_name: str | None = None,
        coordinates: tuple[float, float] | None = None,
        radius: float = 0.1,
        filters: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Query observations from MAST."""
        if self.error_on_operation.get("query_observations", False):
            raise Exception("Simulated MAST query error")

        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        # Filter observations based on criteria
        results = self.observations.copy()

        if target_name:
            results = [
                obs
                for obs in results
                if target_name.lower() in obs["target_name"].lower()
            ]

        if coordinates:
            ra, dec = coordinates
            # Simple distance check (not great circle distance)
            results = [
                obs
                for obs in results
                if abs(obs["ra"] - ra) <= radius and abs(obs["dec"] - dec) <= radius
            ]

        if filters:
            results = [
                obs
                for obs in results
                if any(f in obs.get("filters", "") for f in filters)
            ]

        if start_time:
            results = [
                obs for obs in results if obs["observation_start_time"] >= start_time
            ]

        if end_time:
            results = [
                obs for obs in results if obs["observation_start_time"] <= end_time
            ]

        return results

    async def query_observations_by_position(
        self, ra: float, dec: float, radius: float = 0.1, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Query observations by sky position."""
        return await self.query_observations(
            coordinates=(ra, dec), radius=radius, **kwargs
        )

    async def get_observation_metadata(self, obs_id: str) -> dict[str, Any] | None:
        """Get detailed metadata for a specific observation."""
        for obs in self.observations:
            if obs["obs_id"] == obs_id:
                # Return extended metadata
                return {
                    **obs,
                    "obs_collection": obs.get("obs_collection", "HST"),
                    "dataproduct_type": obs.get("data_product_type", "image"),
                    "obstype": "science",
                    "proposal_id": obs.get("project", "GO-00000"),
                    "sequence_number": 1,
                    "obsnum": 1,
                    "visit": 1,
                    "processing_level": "3",
                    "calibration_level": "3",
                }
        return None

    async def download_observation_data(
        self, obs_id: str, product_type: str = "science"
    ) -> bytes:
        """Download observation data."""
        if self.error_on_operation.get("download_data", False):
            raise Exception("Simulated MAST download error")

        # Return mock FITS data
        return b"SIMPLE  =                    T / file does conform to FITS standard"

    def add_observation(self, observation: dict[str, Any]) -> None:
        """Add a test observation."""
        self.observations.append(observation)

    def clear_observations(self) -> None:
        """Clear all observations."""
        self.observations.clear()
        self._add_default_observations()

    def simulate_error(self, operation: str, should_error: bool = True) -> None:
        """Enable/disable error simulation."""
        self.error_on_operation[operation] = should_error

    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for all operations."""
        self.delay_seconds = seconds


class MockSkyViewClient:
    """Mock SkyView client for downloading reference images."""

    def __init__(self):
        self.surveys: list[str] = ["DSS", "2MASS-K", "WISE 3.4", "GALEX FUV"]
        self.images: dict[str, bytes] = {}
        self.error_on_operation: dict[str, bool] = {}
        self.delay_seconds = 0.0

    async def get_images(
        self,
        position: tuple[float, float] | str,
        survey: str = "DSS",
        width: float = 0.5,
        height: float = 0.5,
        pixels: tuple[int, int] = (1024, 1024),
        **kwargs: Any,
    ) -> list[str]:
        """Get images from SkyView."""
        if self.error_on_operation.get("get_images", False):
            raise Exception("Simulated SkyView error")

        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        # Generate mock image URLs
        if isinstance(position, tuple):
            ra, dec = position
            position_str = f"{ra:.3f}_{dec:.3f}"
        else:
            position_str = position.replace(" ", "_")

        image_key = f"{survey}_{position_str}_{width}x{height}"

        # Store mock image data
        self.images[image_key] = self._generate_mock_fits_data(pixels)

        # Return mock URL
        return [f"http://skyview.example.com/images/{image_key}.fits"]

    async def download_reference_images(
        self,
        ra: float,
        dec: float,
        surveys: list[str] | None = None,
        size: float = 0.5,
        **kwargs: Any,
    ) -> list[str]:
        """Download reference images for a sky position."""
        if surveys is None:
            surveys = ["DSS"]

        image_urls = []
        for survey in surveys:
            urls = await self.get_images((ra, dec), survey, size, size, **kwargs)
            image_urls.extend(urls)

        return image_urls

    def _generate_mock_fits_data(self, pixels: tuple[int, int]) -> bytes:
        """Generate mock FITS file data."""
        width, height = pixels

        # Simple mock FITS header
        header = f"""SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                  -32 / number of bits per data pixel
NAXIS   =                    2 / number of data axes
NAXIS1  =                {width:4d} / length of data axis 1
NAXIS2  =                {height:4d} / length of data axis 2
EXTEND  =                    T / FITS dataset may contain extensions
OBJECT  = 'Mock Sky Image'     / object name
TELESCOP= 'Mock Telescope'     / telescope name
INSTRUME= 'Mock Instrument'    / instrument name
END""".ljust(2880, " ")

        # Mock image data (all zeros for simplicity)
        data_size = width * height * 4  # 4 bytes per float32 pixel
        data = b"\x00" * data_size

        return header.encode("ascii") + data

    async def get_survey_info(self, survey: str) -> dict[str, Any]:
        """Get information about a survey."""
        survey_info = {
            "DSS": {
                "name": "Digitized Sky Survey",
                "description": "Optical sky survey",
                "wavelength": "optical",
                "resolution": 1.7,  # arcsec/pixel
                "coverage": "all-sky",
            },
            "2MASS-K": {
                "name": "2MASS K-band",
                "description": "Near-infrared sky survey",
                "wavelength": "near-infrared",
                "resolution": 1.0,
                "coverage": "all-sky",
            },
            "WISE 3.4": {
                "name": "WISE 3.4 micron",
                "description": "Mid-infrared sky survey",
                "wavelength": "mid-infrared",
                "resolution": 2.75,
                "coverage": "all-sky",
            },
        }

        return survey_info.get(
            survey,
            {
                "name": survey,
                "description": "Unknown survey",
                "wavelength": "unknown",
                "resolution": 1.0,
                "coverage": "unknown",
            },
        )

    def get_available_surveys(self) -> list[str]:
        """Get list of available surveys."""
        return self.surveys.copy()

    def simulate_error(self, operation: str, should_error: bool = True) -> None:
        """Enable/disable error simulation."""
        self.error_on_operation[operation] = should_error

    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for all operations."""
        self.delay_seconds = seconds

    def clear_images(self) -> None:
        """Clear cached images."""
        self.images.clear()


class MockSimbadClient:
    """Mock SIMBAD client for object identification."""

    def __init__(self):
        self.objects: dict[str, dict[str, Any]] = {}
        self.error_on_operation: dict[str, bool] = {}
        self.delay_seconds = 0.0

        # Add some default objects
        self._add_default_objects()

    def _add_default_objects(self) -> None:
        """Add default test objects."""
        default_objects = [
            {
                "main_id": "M31",
                "ra": 10.684,
                "dec": 41.269,
                "object_type": "Galaxy",
                "spectral_type": "SA(s)b",
                "magnitude_v": 3.44,
                "distance": 778000,  # parsecs
                "identifiers": ["M31", "NGC 224", "Andromeda Galaxy"],
            },
            {
                "main_id": "HD 123456",
                "ra": 180.0,
                "dec": 45.0,
                "object_type": "Star",
                "spectral_type": "G2V",
                "magnitude_v": 8.5,
                "distance": 50.2,
                "identifiers": ["HD 123456", "HIP 67890"],
            },
        ]

        for obj in default_objects:
            for identifier in obj["identifiers"]:
                self.objects[identifier.upper()] = obj

    async def query_object(
        self, object_name: str, wildcard: bool = False
    ) -> dict[str, Any] | None:
        """Query object by name."""
        if self.error_on_operation.get("query_object", False):
            raise Exception("Simulated SIMBAD query error")

        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        search_name = object_name.upper()

        if wildcard:
            # Simple wildcard search
            for name, obj in self.objects.items():
                if search_name in name:
                    return obj
        else:
            # Exact match
            return self.objects.get(search_name)

        return None

    async def query_region(
        self, ra: float, dec: float, radius: float = 0.1
    ) -> list[dict[str, Any]]:
        """Query objects in a region."""
        if self.error_on_operation.get("query_region", False):
            raise Exception("Simulated SIMBAD region query error")

        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        results = []
        for obj in self.objects.values():
            # Simple distance check
            distance = ((obj["ra"] - ra) ** 2 + (obj["dec"] - dec) ** 2) ** 0.5
            if distance <= radius:
                results.append(obj)

        # Remove duplicates (since objects can have multiple identifiers)
        unique_results = []
        seen_ids = set()
        for obj in results:
            if obj["main_id"] not in seen_ids:
                unique_results.append(obj)
                seen_ids.add(obj["main_id"])

        return unique_results

    async def get_object_identifiers(self, object_name: str) -> list[str]:
        """Get all identifiers for an object."""
        obj = await self.query_object(object_name)
        if obj:
            return obj.get("identifiers", [])
        return []

    async def get_object_coordinates(
        self, object_name: str
    ) -> tuple[float, float] | None:
        """Get coordinates for an object."""
        obj = await self.query_object(object_name)
        if obj:
            return (obj["ra"], obj["dec"])
        return None

    def add_object(self, object_data: dict[str, Any]) -> None:
        """Add a test object."""
        for identifier in object_data.get("identifiers", [object_data["main_id"]]):
            self.objects[identifier.upper()] = object_data

    def clear_objects(self) -> None:
        """Clear all objects."""
        self.objects.clear()
        self._add_default_objects()

    def simulate_error(self, operation: str, should_error: bool = True) -> None:
        """Enable/disable error simulation."""
        self.error_on_operation[operation] = should_error

    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for all operations."""
        self.delay_seconds = seconds

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        unique_objects = {obj["main_id"] for obj in self.objects.values()}
        return {
            "total_identifiers": len(self.objects),
            "unique_objects": len(unique_objects),
            "object_types": list({obj["object_type"] for obj in self.objects.values()}),
        }
