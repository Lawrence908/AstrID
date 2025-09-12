"""Data ingestion service that coordinates external adapters for observation processing."""

import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS

from src.adapters.external import MASTClient, R2StorageClient, SkyViewClient
from src.adapters.imaging.fits_io import FITSProcessor
from src.core.logging import configure_domain_logger
from src.domains.observations.ingestion.processors import CoordinateProcessor
from src.domains.observations.schema import ObservationCreate


class DataIngestionService:
    """Service for ingesting astronomical observation data."""

    def __init__(
        self,
        mast_client: MASTClient | None = None,
        skyview_client: SkyViewClient | None = None,
        r2_client: R2StorageClient | None = None,
    ):
        """Initialize the data ingestion service.

        Args:
            mast_client: MAST client for querying observations
            skyview_client: SkyView client for reference images
            r2_client: R2 storage client for file storage
        """
        self.logger = configure_domain_logger("observations.ingestion")

        # Initialize external adapters (dependency injection)
        self.mast_client = mast_client or MASTClient()
        self.skyview_client = skyview_client or SkyViewClient()
        self.r2_client = r2_client or R2StorageClient()

        # Initialize processors
        self.coord_processor = CoordinateProcessor()
        self.fits_processor = FITSProcessor()

    async def ingest_observations_by_position(
        self,
        ra: float,
        dec: float,
        survey_id: UUID,
        radius: float = 0.1,
        missions: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[ObservationCreate]:
        """Ingest observations for a specific sky position.

        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            survey_id: UUID of the survey to associate observations with
            missions: List of mission names to query
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of ObservationCreate objects ready for database insertion
        """
        self.logger.info(
            f"Ingesting observations for position RA={ra:.4f}°, Dec={dec:.4f}°, radius={radius:.2f}°"
        )

        try:
            # Query MAST for observations
            mast_observations = await self.mast_client.query_observations_by_position(
                ra=ra,
                dec=dec,
                radius=radius,
                missions=missions,
                start_time=start_time,
                end_time=end_time,
            )

            self.logger.info(f"Found {len(mast_observations)} observations from MAST")

            # Convert MAST observations to ObservationCreate objects
            observation_creates = []
            for mast_obs in mast_observations:
                try:
                    obs_create = await self._convert_mast_observation(
                        mast_obs, survey_id
                    )
                    observation_creates.append(obs_create)
                except Exception as e:
                    self.logger.error(
                        f"Failed to convert MAST observation {mast_obs.get('obs_id')}: {e}"
                    )
                    continue

            self.logger.info(
                f"Successfully converted {len(observation_creates)} observations"
            )
            return observation_creates

        except Exception as e:
            self.logger.error(
                f"Failed to ingest observations for position ({ra}, {dec}): {e}"
            )
            raise

    async def create_reference_dataset(
        self,
        ra: float,
        dec: float,
        size: float = 0.25,
        pixels: int = 512,
        surveys: list[str] | None = None,
        catalog: str = "II/246",  # 2MASS catalog
        output_dir: str | None = None,
    ) -> str:
        """Create a complete reference dataset with image, catalog, and mask.

        This recreates the functionality from dataGathering.py createStarDataset function
        using the new architecture with external adapters.

        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            size: Image size in degrees
            pixels: Image size in pixels
            surveys: List of surveys to use (defaults to DSS)
            catalog: Catalog identifier for star data
            output_dir: Output directory (creates temp if None)

        Returns:
            Path to the created FITS file
        """
        self.logger.info(f"Creating reference dataset for RA={ra:.4f}°, Dec={dec:.4f}°")

        if surveys is None:
            surveys = ["DSS"]

        try:
            # Create temporary directory if needed
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="astrid_reference_")
            else:
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Get image from SkyView
            cutouts = await self.skyview_client.get_image_cutouts(
                ra=ra, dec=dec, size=size, surveys=surveys, pixels=pixels
            )

            if not cutouts:
                raise ValueError(f"No image data available for position ({ra}, {dec})")

            # Use the first available survey
            # survey_name = list(cutouts.keys())[0]  # Will be used when implementing proper WCS
            # image_data = cutouts[survey_name]  # Will be used for WCS creation

            # For now, create mock WCS - in real implementation this would come from SkyView
            # TODO: Extract actual WCS from SkyView response
            wcs = self._create_mock_wcs(ra, dec, size, pixels)

            # Get star catalog from Vizier (this would use VizierClient in full implementation)
            # For now, create a mock catalog
            # TODO: Implement VizierClient integration
            # star_catalog = self._create_mock_star_catalog(ra, dec, size)  # Will be used for complete FITS

            # Create complete FITS file
            output_filename = f"reference_{ra:.4f}_{dec:.4f}.fits"
            output_path = str(Path(output_dir) / output_filename)

            # Create mock image data for now
            mock_image = np.random.normal(1000, 100, (pixels, pixels)).astype(
                np.float32
            )

            # Note: create_complete_fits_file moved to preprocessing domain
            # For now, create a basic FITS file using the lightweight processor
            # TODO: Integrate with preprocessing domain for complete FITS creation

            # Save basic FITS file with image and WCS
            self.fits_processor.save_fits(
                data=mock_image, header=wcs.to_header(), file_path=output_path
            )

            self.logger.info(f"Created reference dataset: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to create reference dataset: {e}")
            raise

    async def download_and_store_observation(
        self, obs_id: str, observation_data: dict[str, Any]
    ) -> dict[str, str]:
        """Download observation files from MAST and store in R2.

        Args:
            obs_id: MAST observation ID
            observation_data: Metadata about the observation

        Returns:
            Dictionary with storage paths and metadata
        """
        self.logger.info(f"Downloading and storing observation: {obs_id}")

        try:
            # Get data products from MAST
            products = await self.mast_client.get_data_products(obs_id)

            if not products:
                raise ValueError(f"No data products found for observation {obs_id}")

            # Download files to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                downloaded_files = await self.mast_client.download_files(
                    products, temp_dir
                )

                # Store each file in R2
                storage_results = {}
                for file_path in downloaded_files:
                    file_path_obj = Path(file_path)

                    # Create R2 object key
                    object_key = f"observations/{obs_id}/{file_path_obj.name}"

                    # Upload to R2
                    upload_result = await self.r2_client.upload_file(
                        local_path=file_path,
                        object_key=object_key,
                        metadata={
                            "obs_id": obs_id,
                            "mission": observation_data.get("mission", "unknown"),
                            "instrument": observation_data.get("instrument", "unknown"),
                            "filter": observation_data.get("filters", "unknown"),
                        },
                    )

                    storage_results[file_path_obj.name] = {
                        "r2_path": object_key,
                        "url": upload_result["url"],
                        "size": upload_result["size_bytes"],
                        "etag": upload_result["etag"],
                    }

            self.logger.info(
                f"Stored {len(storage_results)} files for observation {obs_id}"
            )
            return storage_results

        except Exception as e:
            self.logger.error(f"Failed to download and store observation {obs_id}: {e}")
            raise

    async def batch_ingest_random_observations(
        self,
        survey_id: UUID,
        count: int = 10,
        avoid_galactic_plane: bool = True,
        missions: list[str] | None = None,
    ) -> list[ObservationCreate]:
        """Batch ingest observations from random sky positions.

        Useful for testing and building initial datasets.

        Args:
            count: Number of random positions to query
            survey_id: Survey ID to associate with observations
            avoid_galactic_plane: Whether to avoid galactic plane regions
            missions: List of missions to query

        Returns:
            List of ObservationCreate objects
        """
        self.logger.info(f"Batch ingesting {count} random observations")

        all_observations = []

        for i in range(count):
            try:
                # Generate random coordinates
                if avoid_galactic_plane:
                    # Use logic adapted from dataGathering.py
                    ra, dec = self._generate_random_coordinates(
                        avoid_galactic_plane=True
                    )
                else:
                    ra, dec = self._generate_random_coordinates(
                        avoid_galactic_plane=False
                    )

                self.logger.debug(
                    f"Random position {i+1}/{count}: RA={ra:.4f}°, Dec={dec:.4f}°"
                )

                # Ingest observations for this position
                observations = await self.ingest_observations_by_position(
                    ra=ra, dec=dec, survey_id=survey_id, missions=missions
                )

                all_observations.extend(observations)

            except Exception as e:
                self.logger.warning(f"Failed to ingest random position {i+1}: {e}")
                continue

        self.logger.info(
            f"Batch ingestion completed: {len(all_observations)} total observations"
        )
        return all_observations

    async def _convert_mast_observation(
        self, mast_obs: dict[str, Any], survey_id: UUID
    ) -> ObservationCreate:
        """Convert MAST observation data to ObservationCreate object.

        Args:
            mast_obs: MAST observation dictionary
            survey_id: Survey UUID to associate with

        Returns:
            ObservationCreate object
        """
        # Parse observation time
        obs_time_str = mast_obs.get("obs_date")
        if obs_time_str:
            obs_time = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))
        else:
            obs_time = datetime.now(UTC)

        return ObservationCreate(
            survey_id=survey_id,
            observation_id=mast_obs["obs_id"],
            ra=float(mast_obs["ra"]),
            dec=float(mast_obs["dec"]),
            observation_time=obs_time,
            filter_band=mast_obs.get("filters", "unknown"),
            exposure_time=float(mast_obs.get("exposure_time", 0.0)),
            fits_url=mast_obs.get("dataURL", ""),
            # Optional fields
            image_width=None,  # Will be filled when FITS is processed
            image_height=None,
            pixel_scale=None,
            airmass=None,
            seeing=None,
        )

    def _generate_random_coordinates(
        self, avoid_galactic_plane: bool = True
    ) -> tuple[float, float]:
        """Generate random sky coordinates.

        Adapted from dataGathering.py getRandomCoordinates function.

        Args:
            avoid_galactic_plane: Whether to avoid the galactic plane

        Returns:
            Tuple of (RA, Dec) coordinates in degrees
        """

        if avoid_galactic_plane:
            max_attempts = 100
            for _ in range(max_attempts):
                ra = random.uniform(0, 360)
                dec = random.uniform(-60, 60)  # Limit dec to avoid galactic plane

                coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                galactic_coords = coords.galactic

                # Avoid ±10 degrees around the galactic plane
                if abs(galactic_coords.b.deg) > 10:
                    return ra, dec

            self.logger.warning("Could not find coordinates avoiding galactic plane")

        # Fallback
        ra = random.uniform(0, 360)
        dec = random.uniform(-90, 90)
        return ra, dec

    def _create_mock_wcs(self, ra: float, dec: float, size: float, pixels: int) -> WCS:
        """Create a mock WCS for testing purposes.

        Args:
            ra: Center RA in degrees
            dec: Center Dec in degrees
            size: Image size in degrees
            pixels: Image size in pixels

        Returns:
            Mock WCS object
        """
        wcs = WCS(naxis=2)

        # Set center coordinates
        wcs.wcs.crpix = [pixels / 2, pixels / 2]
        wcs.wcs.crval = [ra, dec]

        # Set pixel scale (degrees per pixel)
        pixel_scale = size / pixels
        wcs.wcs.cdelt = [-pixel_scale, pixel_scale]  # RA decreases with increasing X

        # Set coordinate system
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]

        return wcs

    def _create_mock_star_catalog(self, ra: float, dec: float, size: float) -> Table:
        """Create a mock star catalog for testing.

        Args:
            ra: Center RA in degrees
            dec: Center Dec in degrees
            size: Field size in degrees

        Returns:
            Mock star catalog table
        """

        # Generate random stars within the field
        num_stars = random.randint(50, 200)

        star_data = {
            "RAJ2000": [],
            "DEJ2000": [],
            "_2MASS": [],
            "Jmag": [],
        }

        for _i in range(num_stars):
            # Random position within field
            star_ra = ra + random.uniform(-size / 2, size / 2)
            star_dec = dec + random.uniform(-size / 2, size / 2)

            # Mock 2MASS identifier (simplified format)
            ra_str = f"{int(star_ra*10000):08d}"
            dec_str = f"{int(abs(star_dec)*10000):07d}"
            sign = "+" if star_dec >= 0 else "-"
            mass_id = f"{ra_str}{sign}{dec_str}"

            # Random magnitude
            jmag = random.uniform(10.0, 18.0)

            star_data["RAJ2000"].append(star_ra)
            star_data["DEJ2000"].append(star_dec)
            star_data["_2MASS"].append(mass_id)
            star_data["Jmag"].append(jmag)

        return Table(star_data)
