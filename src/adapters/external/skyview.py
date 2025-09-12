"""
SkyView Virtual Observatory client for astronomical survey images.

RESEARCH DOCS:
- SkyView Homepage: https://skyview.gsfc.nasa.gov/
- SkyView Web API: https://skyview.gsfc.nasa.gov/current/help/batch.html
- Astroquery SkyView: https://astroquery.readthedocs.io/en/latest/skyview/skyview.html
- Available Surveys: https://skyview.gsfc.nasa.gov/current/cgi/titlepage.pl?survey=

PYTHON PACKAGES:
- astroquery.skyview: Primary interface for SkyView
- astropy: For coordinate handling and FITS processing
- requests: For direct HTTP queries

USE CASES FOR ASTRID:
1. Get reference images for any sky position from multiple surveys
2. Download images in various wavelengths (optical, IR, radio, X-ray)
3. Generate difference image templates from historical surveys
4. Obtain survey metadata and coverage information
5. Create multi-wavelength cutouts for analysis
6. Get deep reference images for transient detection
"""

import logging
from pathlib import Path
from typing import Any

from astropy.coordinates import SkyCoord

# TODO: Install these packages in your environment
# uv add astroquery astropy
try:
    from astroquery.skyview import SkyView
except ImportError:
    # Graceful fallback for development
    SkyView = None
    logging.warning("astroquery not installed. Install with: uv add astroquery")

logger = logging.getLogger(__name__)


class SkyViewClient:
    """Client for querying SkyView Virtual Observatory."""

    def __init__(self, timeout: int = 300):
        """Initialize SkyView client.

        Args:
            timeout: Query timeout in seconds
        """
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        if SkyView is None:
            raise ImportError(
                "astroquery package required. Install with: uv add astroquery"
            )

    async def get_available_surveys(self) -> dict[str, list[str]]:
        """Get list of available surveys organized by wavelength.

        RESEARCH:
        - SkyView.list_surveys()
        - Survey wavelength categorizations
        - Data quality and resolution differences

        Returns:
            Dictionary mapping wavelength bands to survey names

        USE CASE: System configuration - know which surveys are available
        """
        try:
            # TODO: Research SkyView.list_surveys()
            # Key considerations:
            # - Which surveys have the best coverage for transient detection?
            # - Resolution and depth differences between surveys
            # - Update frequency of different surveys

            # PLACEHOLDER: Common surveys organized by wavelength
            surveys = {
                "optical": [
                    "DSS",  # Digitized Sky Survey (red, blue)
                    "DSS2 Red",  # DSS2 red plates
                    "DSS2 Blue",  # DSS2 blue plates
                    "SDSS DR12",  # Sloan Digital Sky Survey
                    "POSS2/UKSTU Red",
                    "POSS2/UKSTU Blue",
                ],
                "infrared": [
                    "2MASS-J",  # 2MASS J-band
                    "2MASS-H",  # 2MASS H-band
                    "2MASS-K",  # 2MASS K-band
                    "WISE 3.4",  # WISE W1 band
                    "WISE 4.6",  # WISE W2 band
                    "WISE 12",  # WISE W3 band
                    "WISE 22",  # WISE W4 band
                ],
                "radio": [
                    "VLA FIRST (1.4 GHz)",  # VLA FIRST survey
                    "NVSS",  # NRAO VLA Sky Survey
                    "WENSS",  # Westerbork Northern Sky Survey
                    "GB6 (4850 MHz)",  # Green Bank 6cm survey
                ],
                "xray": [
                    "ROSAT WFC F1",  # ROSAT Wide Field Camera
                    "ROSAT WFC F2",
                    "ROSAT PSPC 3",  # ROSAT PSPC hard band
                    "HEAO 1 A-2",  # HEAO-1 A2 experiment
                ],
            }

            self.logger.info(f"Available surveys in {len(surveys)} wavelength bands")
            return surveys

        except Exception as e:
            self.logger.error(f"Error getting available surveys: {e}")
            raise

    async def get_image_cutouts(
        self,
        ra: float,
        dec: float,
        size: float | tuple[float, float] = 0.1,
        surveys: list[str] | None = None,
        pixels: int | None = 512,
        coordinate_system: str = "J2000",
    ) -> dict[str, dict[str, Any]]:
        """Get image cutouts from multiple surveys.

        RESEARCH:
        - SkyView.get_images()
        - Coordinate system handling
        - Image size and pixel scaling
        - Survey-specific parameters

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            size: Image size in degrees (single value or (width, height))
            surveys: List of survey names to query
            pixels: Image size in pixels (will determine resolution)
            coordinate_system: Coordinate system ('J2000', 'B1950', 'Galactic')

        Returns:
            Dictionary mapping survey names to image data and metadata

        USE CASE: Get reference images for difference image analysis
        """
        try:
            if surveys is None:
                # Default to commonly used surveys for transient detection
                surveys = ["DSS2 Red", "2MASS-J", "WISE 3.4"]

            # Create coordinate object
            SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

            # TODO: Implement actual SkyView query
            # Research areas:
            # - How to handle different pixel scales across surveys
            # - Error handling for surveys with no coverage
            # - Optimal image sizes for different use cases
            # - How to preserve WCS information

            # Example structure:
            # images = SkyView.get_images(
            #     position=coord,
            #     survey=surveys,
            #     pixels=pixels,
            #     coordinate_system=coordinate_system
            # )

            results = {}
            for survey in surveys:
                # PLACEHOLDER: Mock image data
                results[survey] = {
                    "survey_name": survey,
                    "ra": ra,
                    "dec": dec,
                    "size_deg": size,
                    "pixels": pixels,
                    "image_data": None,  # Would contain actual FITS HDU
                    "wcs": None,  # World coordinate system
                    "metadata": {
                        "exposure_time": "unknown",
                        "filter": "unknown",
                        "epoch": "unknown",
                        "resolution_arcsec_per_pixel": "unknown",
                    },
                    "download_url": f"https://skyview.gsfc.nasa.gov/tempspace/fits/{survey}_mock.fits",
                    "coverage_available": True,
                }

            self.logger.info(
                f"Retrieved cutouts from {len(results)} surveys for position ({ra}, {dec})"
            )
            return results

        except Exception as e:
            self.logger.error(f"Error getting image cutouts: {e}")
            raise

    async def get_survey_coverage(
        self, ra: float, dec: float, surveys: list[str] | None = None
    ) -> dict[str, bool]:
        """Check which surveys have coverage for a given position.

        RESEARCH:
        - SkyView coverage checking methods
        - Survey footprints and boundaries
        - How to efficiently batch coverage queries

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            surveys: List of survey names to check

        Returns:
            Dictionary mapping survey names to coverage boolean

        USE CASE: Pre-filter surveys before requesting images
        """
        try:
            if surveys is None:
                survey_info = await self.get_available_surveys()
                surveys = []
                for band_surveys in survey_info.values():
                    surveys.extend(band_surveys)

            # TODO: Research efficient coverage checking
            # Questions:
            # - Does SkyView provide a coverage-only API?
            # - How to batch check multiple positions?
            # - Cache coverage information for regions?

            coverage = {}
            for survey in surveys:
                # PLACEHOLDER: Mock coverage (most optical surveys cover most sky)
                if "DSS" in survey or "SDSS" in survey:
                    coverage[survey] = abs(dec) < 80  # Rough declination limit
                elif "2MASS" in survey:
                    coverage[survey] = True  # All-sky coverage
                elif "WISE" in survey:
                    coverage[survey] = True  # All-sky coverage
                else:
                    coverage[survey] = False  # Conservative default

            available_count = sum(coverage.values())
            self.logger.info(
                f"Position ({ra}, {dec}) has coverage in {available_count}/{len(surveys)} surveys"
            )
            return coverage

        except Exception as e:
            self.logger.error(f"Error checking survey coverage: {e}")
            raise

    async def download_reference_images(
        self,
        ra: float,
        dec: float,
        output_dir: str,
        surveys: list[str] | None = None,
        size: float = 0.1,
        pixels: int = 512,
    ) -> list[str]:
        """Download reference images for a sky position.

        RESEARCH:
        - File naming conventions for SkyView downloads
        - FITS header preservation
        - Batch download optimization

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            output_dir: Directory to save downloaded images
            surveys: List of surveys to download from
            size: Image size in degrees
            pixels: Image size in pixels

        Returns:
            List of downloaded file paths

        USE CASE: Build reference image library for difference imaging
        """
        try:
            # First check coverage
            coverage = await self.get_survey_coverage(ra, dec, surveys)
            available_surveys = [
                s for s, has_coverage in coverage.items() if has_coverage
            ]

            if not available_surveys:
                self.logger.warning(f"No survey coverage for position ({ra}, {dec})")
                return []

            # Get image cutouts
            cutouts = await self.get_image_cutouts(
                ra=ra, dec=dec, size=size, surveys=available_surveys, pixels=pixels
            )

            # TODO: Implement actual file download and saving
            # Research areas:
            # - FITS file naming conventions
            # - Metadata preservation in downloaded files
            # - Progress tracking for multiple downloads
            # - Error handling for partial failures

            downloaded_files = []
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for survey, _data in cutouts.items():
                # PLACEHOLDER: Mock file creation
                filename = (
                    f"{survey.replace(' ', '_')}_{ra:.4f}_{dec:.4f}_{size:.3f}deg.fits"
                )
                filepath = output_path / filename

                # In real implementation, save the FITS data here
                # fits.writeto(filepath, data['image_data'], header=data['wcs'].to_header())

                downloaded_files.append(str(filepath))
                self.logger.debug(f"Downloaded {survey} image to {filename}")

            self.logger.info(
                f"Downloaded {len(downloaded_files)} reference images to {output_dir}"
            )
            return downloaded_files

        except Exception as e:
            self.logger.error(f"Error downloading reference images: {e}")
            raise

    async def get_multi_wavelength_data(
        self, ra: float, dec: float, size: float = 0.1, pixels: int = 512
    ) -> dict[str, dict[str, Any]]:
        """Get multi-wavelength images for comprehensive analysis.

        RESEARCH:
        - Optimal survey combinations for different science cases
        - Wavelength-dependent resolution matching
        - Color correction between surveys

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            size: Image size in degrees
            pixels: Image size in pixels

        Returns:
            Dictionary with wavelength bands and corresponding images

        USE CASE: Multi-wavelength transient characterization
        """
        try:
            # Define optimal survey combinations
            wavelength_surveys = {
                "optical_red": ["DSS2 Red", "POSS2/UKSTU Red"],
                "optical_blue": ["DSS2 Blue", "POSS2/UKSTU Blue"],
                "near_infrared": ["2MASS-J", "2MASS-H", "2MASS-K"],
                "mid_infrared": ["WISE 3.4", "WISE 4.6", "WISE 12"],
                "radio": ["VLA FIRST (1.4 GHz)", "NVSS"],
            }

            multi_wavelength_data = {}

            for wavelength, surveys in wavelength_surveys.items():
                # Check coverage and get best available survey
                coverage = await self.get_survey_coverage(ra, dec, surveys)
                available = [s for s in surveys if coverage.get(s, False)]

                if available:
                    # Use first available survey (could implement priority logic)
                    best_survey = available[0]
                    cutouts = await self.get_image_cutouts(
                        ra=ra, dec=dec, size=size, surveys=[best_survey], pixels=pixels
                    )

                    if best_survey in cutouts:
                        multi_wavelength_data[wavelength] = cutouts[best_survey]
                        multi_wavelength_data[wavelength]["wavelength_band"] = (
                            wavelength
                        )

            self.logger.info(
                f"Retrieved multi-wavelength data in {len(multi_wavelength_data)} bands"
            )
            return multi_wavelength_data

        except Exception as e:
            self.logger.error(f"Error getting multi-wavelength data: {e}")
            raise


class ReferenceImageManager:
    """Manager for building and maintaining reference image libraries."""

    def __init__(self, skyview_client: SkyViewClient, storage_dir: str):
        """Initialize reference image manager.

        Args:
            skyview_client: SkyView client instance
            storage_dir: Directory for storing reference images
        """
        self.client = skyview_client
        self.storage_dir = Path(storage_dir)
        self.logger = logging.getLogger(__name__)

    async def build_reference_library(
        self,
        target_positions: list[tuple[float, float]],
        surveys: list[str] | None = None,
        size: float = 0.2,
        pixels: int = 1024,
    ) -> dict[str, list[str]]:
        """Build reference image library for multiple positions.

        Args:
            target_positions: List of (ra, dec) tuples in degrees
            surveys: Surveys to include in library
            size: Image size in degrees
            pixels: Image resolution in pixels

        Returns:
            Dictionary mapping positions to downloaded file lists

        USE CASE: Pre-populate reference images for survey fields
        """
        try:
            if surveys is None:
                surveys = ["DSS2 Red", "2MASS-J", "WISE 3.4"]  # Good defaults

            library = {}

            for ra, dec in target_positions:
                position_key = f"{ra:.4f}_{dec:.4f}"
                position_dir = self.storage_dir / position_key

                downloaded = await self.client.download_reference_images(
                    ra=ra,
                    dec=dec,
                    output_dir=str(position_dir),
                    surveys=surveys,
                    size=size,
                    pixels=pixels,
                )

                library[position_key] = downloaded
                self.logger.info(
                    f"Built reference library for position ({ra}, {dec}): {len(downloaded)} images"
                )

            return library

        except Exception as e:
            self.logger.error(f"Error building reference library: {e}")
            raise


# INTEGRATION EXAMPLE:
# This shows how the SkyView client would be used in AstrID


async def example_usage():
    """Example usage of SkyView client for AstrID."""

    client = SkyViewClient()

    # Use case 1: Get reference images for a transient candidate
    await client.download_reference_images(
        ra=180.0,
        dec=45.0,
        output_dir="/tmp/references",
        surveys=["DSS2 Red", "2MASS-J"],
        size=0.1,  # 0.1 degree cutout
        pixels=512,
    )

    # Use case 2: Multi-wavelength analysis
    await client.get_multi_wavelength_data(ra=180.0, dec=45.0, size=0.05, pixels=256)

    # Use case 3: Build reference library for survey area
    manager = ReferenceImageManager(client, "/data/references")
    survey_positions = [
        (180.0 + i * 0.1, 45.0 + j * 0.1) for i in range(10) for j in range(10)
    ]
    library = await manager.build_reference_library(survey_positions)

    print(
        f"Downloaded {sum(len(files) for files in library.values())} reference images"
    )


# TODO: Integration points with AstrID:
# 1. Connect to difference imaging pipeline for reference templates
# 2. Integrate with R2 storage for reference image caching
# 3. Add reference image metadata to database
# 4. Create workflows for updating reference libraries
# 5. Add configuration for preferred surveys and image parameters
