"""
MAST (Mikulski Archive for Space Telescopes) API client for astronomical observation data.

RESEARCH DOCS:
- Main MAST Portal: https://mast.stsci.edu/
- MAST API Documentation: https://mast.stsci.edu/api/v0/_c_a_o_mfields.html
- Astroquery MAST: https://astroquery.readthedocs.io/en/latest/mast/mast.html
- MAST Services Guide: https://mast.stsci.edu/api/v0/_services.html

PYTHON PACKAGES:
- astroquery.mast: Primary interface for MAST queries
- astropy: For coordinate handling and time conversions
- requests: For direct HTTP API calls if needed

USE CASES FOR ASTRID:
1. Query observations by sky position (RA/Dec) and time range
2. Download FITS files from HST, JWST, TESS, Kepler missions
3. Get observation metadata (exposure time, filter, instrument)
4. Batch download observations for survey areas
5. Query calibrated vs raw data availability
6. Check for new observations since last ingestion run
"""

import logging
from datetime import datetime, timedelta
from typing import Any

# Installation help for optional dependencies
_INSTALL_HINT = (
    "Required packages missing: astroquery, astropy.\n"
    "Install with one of:\n"
    "  - uv:   uv add astroquery astropy\n"
    "  - pip:  pip install astroquery astropy\n"
    "  - conda: conda install -c conda-forge astroquery astropy"
)
try:
    from astroquery.mast import Catalogs, Observations
except ImportError:
    # Graceful fallback for development
    Observations = None
    Catalogs = None
    logging.warning(_INSTALL_HINT)

logger = logging.getLogger(__name__)


class MASTClient:
    """Client for querying MAST astronomical archive."""

    def __init__(self, timeout: int = 300):
        """Initialize MAST client.

        Args:
            timeout: Query timeout in seconds
        """
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        if Observations is None:
            raise ImportError(_INSTALL_HINT)

    # --- Lightweight sync helper used by notebooks/services ---
    @staticmethod
    def fetch_ps1_cutout(
        ra_deg: float,
        dec_deg: float,
        *,
        size_pixels: int = 240,
        filt: str = "g",
        timeout_sec: int = 30,
        to_display_image_fn=None,
    ) -> tuple[Any | None, dict[str, Any]]:
        """Fetch a small PS1 cutout as a fallback reference image.

        Returns (display_image, info). If unavailable, returns (None, info with error).
        """
        from io import BytesIO

        import numpy as np
        import requests
        from astropy.io import fits

        if to_display_image_fn is None:
            from ..imaging.utils import to_display_image as _to_disp

            to_display_image_fn = _to_disp

        info: dict[str, Any] = {"source": "ps1", "filter": filt, "error": None}
        url = (
            "https://ps1images.stsci.edu/cgi-bin/ps1cutouts?"
            f"pos={ra_deg}+{dec_deg}&filter={filt}&format=fits&size={size_pixels}"
        )
        headers = {"Accept": "application/fits, application/octet-stream"}
        try:
            r = requests.get(url, timeout=timeout_sec, headers=headers)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").lower()
            if "fits" not in ctype and not r.content.startswith(b"SIMPLE"):
                raise ValueError(f"Unexpected PS1 content type: {ctype or 'unknown'}")
            with fits.open(BytesIO(r.content), ignore_missing_simple=True) as hdul:
                data = None
                for h in hdul:
                    hdata = getattr(h, "data", None)
                    if hdata is None:
                        continue
                    arr = np.asarray(hdata)
                    if arr.size == 0:
                        continue
                    data = arr
                    break
            if data is None:
                info["error"] = "PS1 response had no image data"
                return None, info
            return to_display_image_fn(data), info
        except Exception as e:
            info["error"] = str(e)
            return None, info

    async def query_observations_by_position(
        self,
        ra: float,
        dec: float,
        radius: float = 0.1,
        missions: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        data_rights: str = "PUBLIC",
    ) -> list[dict[str, Any]]:
        """Query observations by sky position.

        RESEARCH:
        - astroquery.mast.Observations.query_region()
        - MAST coordinate systems and units
        - Available mission names in MAST

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            missions: List of mission names (e.g., ['HST', 'JWST', 'TESS'])
            start_time: Start of observation time range
            end_time: End of observation time range
            data_rights: 'PUBLIC' or 'EXCLUSIVE'

        Returns:
            List of observation metadata dictionaries

        USE CASE: Find all observations covering a specific sky region
        """
        try:
            # Create coordinate object (will be used in actual implementation)

            # TODO: Implement actual MAST query
            # Research: How to use Observations.query_region()
            # Example query structure:
            # obs_table = Observations.query_region(
            #     coordinates=coord,
            #     radius=radius*u.deg
            # )

            # PLACEHOLDER: Return mock data for now
            mock_observation = {
                "obs_id": f"mock_obs_{ra}_{dec}",
                "target_name": "TEST_TARGET",
                "ra": ra,
                "dec": dec,
                "mission": "HST",
                "instrument": "ACS/WFC",
                "filters": "F814W",
                "exposure_time": 600.0,
                "obs_date": datetime.now().isoformat(),
                "data_rights": data_rights,
                "obs_collection": "HST",
                "dataURL": "https://mast.stsci.edu/...",
                "filesize": 1024000,  # bytes
            }

            self.logger.info(
                f"Queried MAST for position ({ra}, {dec}) with radius {radius}°"
            )
            return [mock_observation]

        except Exception as e:
            self.logger.error(f"Error querying MAST observations: {e}")
            raise

    async def query_new_observations(
        self,
        last_check: datetime,
        missions: list[str] | None = None,
        max_results: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query for new observations since last check.

        RESEARCH:
        - MAST time-based queries
        - Date formats accepted by MAST API
        - How to filter by ingestion date vs observation date

        Args:
            last_check: Only return observations newer than this
            missions: Filter by specific missions
            max_results: Maximum number of results to return

        Returns:
            List of new observation metadata

        USE CASE: Scheduled ingestion - check for new data daily
        """
        try:
            # TODO: Research time-based MAST queries
            # Key questions:
            # - Does MAST support queries by ingestion date?
            # - What date format does the API expect?
            # - How to handle pagination for large result sets?

            # Convert datetime to MAST format
            # Research: astropy.time.Time conversion to MJD or ISO format

            self.logger.info(f"Querying MAST for observations since {last_check}")

            # PLACEHOLDER: Return empty list for now
            return []

        except Exception as e:
            self.logger.error(f"Error querying new MAST observations: {e}")
            raise

    async def get_observation_metadata(self, obs_id: str) -> dict[str, Any]:
        """Get detailed metadata for a specific observation.

        RESEARCH:
        - MAST observation ID formats
        - Available metadata fields
        - How to query by observation ID

        Args:
            obs_id: MAST observation identifier

        Returns:
            Detailed observation metadata

        USE CASE: Get full metadata before downloading files
        """
        try:
            # TODO: Implement observation-specific metadata query
            # Research: Observations.query_criteria() with obsid parameter

            self.logger.info(f"Getting metadata for observation {obs_id}")

            # PLACEHOLDER
            return {"obs_id": obs_id, "status": "found", "metadata_complete": True}

        except Exception as e:
            self.logger.error(f"Error getting observation metadata: {e}")
            raise

    async def get_data_products(self, obs_id: str) -> list[dict[str, Any]]:
        """Get list of data products for an observation.

        RESEARCH:
        - Observations.get_product_list()
        - Data product types (science, calibration, preview)
        - File extensions and formats available

        Args:
            obs_id: MAST observation identifier

        Returns:
            List of available data products

        USE CASE: Choose which files to download (raw vs calibrated)
        """
        try:
            # TODO: Research Observations.get_product_list()
            # Key considerations:
            # - Filter by product type (science vs calibration)
            # - File size considerations for bulk downloads
            # - Preview images vs full FITS files

            self.logger.info(f"Getting data products for observation {obs_id}")

            # PLACEHOLDER
            return [
                {
                    "productFilename": f"{obs_id}_drz.fits",
                    "productType": "SCIENCE",
                    "size": 1024000,
                    "dataURI": f"mast:HST/product/{obs_id}_drz.fits",
                }
            ]

        except Exception as e:
            self.logger.error(f"Error getting data products: {e}")
            raise

    async def download_files(
        self, data_products: list[dict[str, Any]], download_dir: str, curl: bool = False
    ) -> list[str]:
        """Download FITS files from MAST.

        RESEARCH:
        - Observations.download_products()
        - Bulk download strategies
        - Authentication for proprietary data
        - Download resumption for large files

        Args:
            data_products: List of data products to download
            download_dir: Local directory for downloads
            curl: Use curl for downloading (faster for bulk)

        Returns:
            List of downloaded file paths

        USE CASE: Batch download of observation files for processing
        """
        try:
            # TODO: Implement file download logic
            # Research areas:
            # - How to handle large file downloads efficiently
            # - Progress tracking for long downloads
            # - Error handling and retry logic
            # - Disk space checking before download

            self.logger.info(
                f"Downloading {len(data_products)} files to {download_dir}"
            )

            downloaded_files = []
            for product in data_products:
                # PLACEHOLDER: Mock download
                filename = product.get("productFilename", "unknown.fits")
                filepath = f"{download_dir}/{filename}"
                downloaded_files.append(filepath)
                self.logger.debug(f"Downloaded {filename}")

            return downloaded_files

        except Exception as e:
            self.logger.error(f"Error downloading files: {e}")
            raise

    async def query_missions_and_instruments(self) -> dict[str, list[str]]:
        """Get available missions and their instruments.

        RESEARCH:
        - MAST mission list API
        - Instrument capabilities per mission
        - Data availability timelines

        Returns:
            Dictionary mapping missions to instrument lists

        USE CASE: System configuration - know what data sources are available
        """
        try:
            # TODO: Research how to get mission/instrument listings
            # This helps users configure which surveys to monitor

            # PLACEHOLDER: Common missions
            return {
                "HST": ["ACS/WFC", "WFC3/UVIS", "WFC3/IR", "WFPC2"],
                "JWST": ["NIRCam", "MIRI", "NIRSpec", "NIRISS"],
                "TESS": ["Camera"],
                "Kepler": ["Photometer"],
                "K2": ["Photometer"],
            }

        except Exception as e:
            self.logger.error(f"Error querying missions: {e}")
            raise


# INTEGRATION EXAMPLE:
# This shows how the MAST client would be used in the observation ingestion flow


async def example_usage():
    """Example usage of MAST client for AstrID ingestion."""

    client = MASTClient()

    # Use case 1: Survey a specific sky region
    observations = await client.query_observations_by_position(
        ra=180.0,  # degrees
        dec=45.0,  # degrees
        radius=0.5,  # degrees
        missions=["HST", "JWST"],
        start_time=datetime.now() - timedelta(days=30),
    )

    # Use case 2: Check for new observations
    await client.query_new_observations(
        last_check=datetime.now() - timedelta(days=1), missions=["TESS"]
    )

    # Use case 3: Download specific observation
    if observations:
        obs_id = observations[0]["obs_id"]
        products = await client.get_data_products(obs_id)
        files = await client.download_files(products, "/tmp/downloads")
        print(f"Downloaded {len(files)} files")


# TODO: Integration points with AstrID:
# 1. Connect to src/adapters/scheduler/flows/process_new.py ingest_window()
# 2. Store downloaded files using R2 client
# 3. Create Observation records in database
# 4. Trigger preprocessing workflows
# 5. Add configuration for target sky regions and missions
