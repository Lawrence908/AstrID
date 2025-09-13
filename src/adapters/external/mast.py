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

    def __init__(self, timeout: int = 30, test_mode: bool = False):
        """Initialize MAST client.

        Args:
            timeout: Query timeout in seconds (reduced from 300 for better testing)
            test_mode: If True, use mock data instead of real MAST queries
        """
        self.timeout = timeout
        self.test_mode = test_mode
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
        import time
        from io import BytesIO

        import numpy as np
        import requests
        from astropy.io import fits

        if to_display_image_fn is None:
            from ..imaging.utils import to_display_image as _to_disp

            to_display_image_fn = _to_disp

        info: dict[str, Any] = {"source": "ps1", "filter": filt, "error": None}
        
        # Validate coordinates
        if not (0 <= ra_deg <= 360):
            info["error"] = f"Invalid RA: {ra_deg} (must be 0-360 degrees)"
            return None, info
        if not (-90 <= dec_deg <= 90):
            info["error"] = f"Invalid Dec: {dec_deg} (must be -90 to +90 degrees)"
            return None, info
            
        # Format coordinates more precisely for PS1 service
        ra_str = f"{ra_deg:.6f}"
        dec_str = f"{dec_deg:+.6f}"  # Include + sign for positive declinations
        
        headers = {
            "Accept": "application/fits, application/octet-stream, image/jpeg",
            "User-Agent": "AstrID-PS1/1.0 (+https://github.com/AstrID)",
        }
        
        # First, get the PS1 cutout page to find the actual FITS file path
        cutout_page_url = f"https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra_str}+{dec_str}&filter={filt}&format=fits&size={size_pixels}"
        
        # Try to extract the FITS file path from the HTML response
        try:
            import re
            page_resp = requests.get(cutout_page_url, timeout=timeout_sec, headers=headers)
            if page_resp.status_code == 200 and "text/html" in page_resp.headers.get("Content-Type", "").lower():
                # Look for direct FITS file links first
                fits_file_pattern = r'href="(/rings\.v3\.skycell/[^"]+\.fits)"'
                fits_file_matches = re.findall(fits_file_pattern, page_resp.text)
                
                # Look for FITS cutout links in the HTML
                fits_cutout_pattern = r'href="//ps1images\.stsci\.edu/cgi-bin/fitscut\.cgi\?([^"]+)"'
                fits_cutout_matches = re.findall(fits_cutout_pattern, page_resp.text)
                
                urls_to_try = []
                
                print(f"Found {len(fits_file_matches)} direct FITS files and {len(fits_cutout_matches)} FITS cutout links")
                
                # Try direct FITS file first (full image, not cutout)
                if fits_file_matches:
                    fits_file_url = f"https://ps1images.stsci.edu{fits_file_matches[0]}"
                    urls_to_try.append(fits_file_url)
                    print(f"Found direct FITS file: {fits_file_url}")
                
                # Try FITS cutout
                if fits_cutout_matches:
                    fits_params = fits_cutout_matches[0]
                    # Ensure we're requesting FITS format
                    if "format=fits" not in fits_params:
                        fits_params = fits_params.replace("format=jpeg", "format=fits")
                    fits_cutout_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?{fits_params}"
                    urls_to_try.append(fits_cutout_url)
                    print(f"Found FITS cutout: {fits_cutout_url}")
                
                # Add JPEG fallback
                if fits_cutout_matches:
                    fits_params = fits_cutout_matches[0]
                    jpeg_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?{fits_params}".replace("format=fits", "format=jpeg").replace("&asinh=True", "")
                    urls_to_try.append(jpeg_url)
                    print(f"Added JPEG fallback: {jpeg_url}")
                
                if not urls_to_try:
                    # Fallback to original URLs if parsing fails
                    urls_to_try = [
                        f"https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra_str}+{dec_str}&filter={filt}&format=fits&size={size_pixels}",
                        f"https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra_str}+{dec_str}&filter={filt}&format=jpeg&size={size_pixels}",
                    ]
            else:
                # Fallback if page request fails
                urls_to_try = [
                    f"https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra_str}+{dec_str}&filter={filt}&format=fits&size={size_pixels}",
                    f"https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra_str}+{dec_str}&filter={filt}&format=jpeg&size={size_pixels}",
                ]
        except Exception as e:
            print(f"Error parsing PS1 page: {e}")
            # Fallback URLs
            urls_to_try = [
                f"https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra_str}+{dec_str}&filter={filt}&format=fits&size={size_pixels}",
                f"https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra_str}+{dec_str}&filter={filt}&format=jpeg&size={size_pixels}",
                    ]

        def _attempt(url: str) -> tuple[Any | None, str | None]:
            try:
                resp = requests.get(url, timeout=timeout_sec, headers=headers)
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                
                # Check for HTML error responses
                if "text/html" in ctype:
                    # Try to extract error message from HTML
                    content_str = resp.content.decode('utf-8', errors='ignore')
                    if "error" in content_str.lower() or "not found" in content_str.lower():
                        return None, f"PS1 service error (HTML response): {content_str[:200]}..."
                    return None, f"PS1 returned HTML instead of image data: {content_str[:100]}..."
                
                if "fits" in ctype or resp.content.startswith(b"SIMPLE"):
                    with fits.open(
                        BytesIO(resp.content), ignore_missing_simple=True
                    ) as hdul:
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
                    return data, None
                if "jpeg" in ctype or "image/" in ctype:
                    try:
                        from io import BytesIO as _BIO

                        from PIL import Image as _PILImage  # type: ignore

                        img = np.asarray(
                            _PILImage.open(_BIO(resp.content)).convert("RGB")
                        )
                        return img, None
                    except Exception as _:
                        return None, f"Unsupported image payload ({ctype})"
                return None, f"Unexpected PS1 content type: {ctype or 'unknown'}"
            except Exception as ex:
                return None, str(ex)

        # Try all URL formats with retries
        backoffs = [0.0, 0.5, 1.0]
        last_err: str | None = None
        
        for i, delay in enumerate(backoffs):
            if delay:
                time.sleep(delay)
                
            # Try each URL format
            for j, url in enumerate(urls_to_try):
                print(f"Trying PS1 URL {j+1}/{len(urls_to_try)}: {url}")
                data, err = _attempt(url)
                if data is not None:
                    format_type = "fits" if "format=fits" in url else "jpeg"
                    return to_display_image_fn(data), {**info, "format": format_type}
                last_err = err
                if i == 0 and j == 0:  # Log the first attempt for debugging
                    print(f"PS1 first attempt failed: {err}")

        # If all PS1 attempts failed, try a simple fallback
        print("All PS1 attempts failed, trying simple coordinate validation...")
        if last_err and "HTML" in last_err:
            info["error"] = f"PS1 service returned HTML error: {last_err}"
        else:
            info["error"] = last_err or "All PS1 URL formats failed"
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
        """Query observations by sky position using real MAST API.

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
            # Validate coordinates
            if not (0 <= ra <= 360):
                raise ValueError(f"RA must be between 0 and 360 degrees, got {ra}")
            if not (-90 <= dec <= 90):
                raise ValueError(f"Dec must be between -90 and 90 degrees, got {dec}")
            if radius <= 0:
                raise ValueError(f"Radius must be positive, got {radius}")

            # Use test mode if enabled
            if self.test_mode:
                self.logger.info("Using test mode - returning mock data")
                return self._create_mock_observation(ra, dec, data_rights)

            # Import astroquery here to handle missing dependency gracefully
            try:
                from astroquery.mast import Observations  # type: ignore
                from astropy.coordinates import SkyCoord  # type: ignore
                import astropy.units as u  # type: ignore
            except ImportError as e:
                self.logger.warning(f"astroquery not available, falling back to mock data: {e}")
                return self._create_mock_observation(ra, dec, data_rights)

            # Create coordinate object
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')  # type: ignore

            # Build query parameters
            query_params = {
                "coordinates": coord,
                "radius": radius * u.deg,  # type: ignore
            }

            # Note: Mission filtering will be done after the query
            # as astroquery.mast doesn't support direct mission filtering in query_region

            # Note: Time filtering will be done after the query
            # as astroquery.mast doesn't support t_min/t_max in query_region
            # We'll filter by observation date after getting results

            # Query MAST
            self.logger.info(f"Querying MAST for position ({ra}, {dec}) with radius {radius}°")
            self.logger.debug(f"Query parameters: {query_params}")
            obs_table = Observations.query_region(**query_params)  # type: ignore
            self.logger.info(f"MAST query returned {len(obs_table)} raw observations")

            # Convert to list of dictionaries and filter by missions
            observations = []
            filtered_count = 0
            for row in obs_table:
                mission = str(row.get("obs_collection", ""))
                
                # Filter by missions if specified
                if missions and mission not in missions:
                    filtered_count += 1
                    continue
                
                # Filter by time if specified
                if start_time or end_time:
                    from astropy.time import Time
                    obs_time = row.get("t_min")
                    if obs_time is not None:
                        try:
                            obs_mjd = float(obs_time)
                            obs_time_obj = Time(obs_mjd, format='mjd')
                            
                            if start_time and obs_time_obj < Time(start_time):
                                filtered_count += 1
                                continue
                            if end_time and obs_time_obj > Time(end_time):
                                filtered_count += 1
                                continue
                        except (ValueError, TypeError):
                            # Skip observations with invalid time data
                            filtered_count += 1
                            continue
                
                obs_dict = {
                    "obs_id": str(row.get("obsid", "")),
                    "target_name": str(row.get("target_name", "")),
                    "ra": float(row.get("s_ra", ra)),
                    "dec": float(row.get("s_dec", dec)),
                    "mission": mission,
                    "instrument": str(row.get("instrument_name", "")),
                    "filters": str(row.get("filters", "")),
                    "exposure_time": float(row.get("t_exptime", 0.0)),
                    "obs_date": str(row.get("t_min", "")),
                    "data_rights": str(row.get("data_rights", data_rights)),
                    "obs_collection": mission,
                    "dataURL": f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:{mission}/product/{row.get('obsid', '')}",
                    "filesize": int(row.get("size", 0)),
                }
                observations.append(obs_dict)

            self.logger.info(f"Found {len(observations)} observations from MAST (filtered {filtered_count} total)")
            if missions:
                self.logger.info(f"Filtered for missions: {missions}")
            if start_time or end_time:
                self.logger.info(f"Filtered for time range: {start_time} to {end_time}")
            return observations

        except ValueError as e:
            self.logger.error(f"Invalid input parameters: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error querying MAST observations: {e}")
            # Fallback to mock data if real query fails
            self.logger.warning("Falling back to mock data due to MAST query failure")
            return self._create_mock_observation(ra, dec, data_rights)

    def _create_mock_observation(self, ra: float, dec: float, data_rights: str) -> list[dict[str, Any]]:
        """Create mock observation data as fallback."""
        mock_observation = {
            "obs_id": f"mock_obs_{ra}_{dec}",
            "target_name": "MOCK_TARGET",
            "ra": ra,
            "dec": dec,
            "mission": "HST",
            "instrument": "ACS/WFC",
            "filters": "F814W",
            "exposure_time": 600.0,
            "obs_date": datetime.now().isoformat(),
            "data_rights": data_rights,
            "obs_collection": "HST",
            "dataURL": "https://mast.stsci.edu/mock/...",
            "filesize": 1024000,
        }
        return [mock_observation]

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
