"""
IRSA (Infrared Science Archive) API client for astronomical observation data.

IRSA provides access to:
- ZTF (Zwicky Transient Facility) - 2018+ optical transients
- PTF (Palomar Transient Factory) - 2009-2017 optical transients
- WISE/NEOWISE - Infrared all-sky surveys
- 2MASS - Near-infrared survey
- Spitzer - Infrared space telescope

RESEARCH DOCS:
- IRSA IBE API: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html
- ZTF Metadata: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_metadata.html
- Astroquery IRSA: https://astroquery.readthedocs.io/en/latest/ipac/irsa/irsa.html

PYTHON PACKAGES:
- astroquery.ipac.irsa.ibe: Primary interface for IRSA IBE queries
- astropy: For coordinate handling and time conversions
"""

import logging
from datetime import datetime
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
    from astroquery.ipac.irsa.ibe import Ibe
except ImportError:
    Ibe = None
    logging.warning(_INSTALL_HINT)

logger = logging.getLogger(__name__)


class IRSAClient:
    """Client for querying IRSA astronomical archive."""

    SUPPORTED_MISSIONS = ["ZTF", "PTF", "WISE", "NEOWISE", "2MASS", "SPITZER"]

    def __init__(self, timeout: int = 30, test_mode: bool = False):
        """Initialize IRSA client.

        Args:
            timeout: Query timeout in seconds
            test_mode: If True, use mock data instead of real IRSA queries
        """
        self.timeout = timeout
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)

        if Ibe is None and not test_mode:
            raise ImportError(_INSTALL_HINT)

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
        """Query observations by sky position using IRSA IBE API.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            missions: List of mission names (e.g., ['ZTF', 'PTF', 'WISE'])
            start_time: Start of observation time range
            end_time: End of observation time range
            data_rights: 'PUBLIC' or 'EXCLUSIVE' (not used for IRSA, kept for compatibility)

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
                import astropy.units as u  # type: ignore
                from astropy.coordinates import SkyCoord  # type: ignore
                from astropy.time import Time  # type: ignore
            except ImportError as e:
                self.logger.warning(
                    f"astroquery not available, falling back to mock data: {e}"
                )
                return self._create_mock_observation(ra, dec, data_rights)

            # Filter missions to IRSA-supported ones
            if missions:
                missions = [m.upper() for m in missions]
                irsa_missions = [
                    m for m in missions if m in [x.upper() for x in self.SUPPORTED_MISSIONS]
                ]
                if not irsa_missions:
                    self.logger.warning(
                        f"No IRSA-supported missions in {missions}, returning empty results"
                    )
                    return []
            else:
                irsa_missions = self.SUPPORTED_MISSIONS

            # Create coordinate object
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")  # type: ignore

            # Query IRSA for each mission
            all_observations = []
            for mission in irsa_missions:
                mission_obs = await self._query_mission(
                    mission, coord, radius, start_time, end_time
                )
                all_observations.extend(mission_obs)

            self.logger.info(
                f"Found {len(all_observations)} observations from IRSA"
            )
            if missions:
                self.logger.info(f"Filtered for missions: {irsa_missions}")
            if start_time or end_time:
                self.logger.info(f"Filtered for time range: {start_time} to {end_time}")

            return all_observations

        except ValueError as e:
            self.logger.error(f"Invalid input parameters: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error querying IRSA observations: {e}")
            # Fallback to mock data if real query fails
            self.logger.warning("Falling back to mock data due to IRSA query failure")
            return self._create_mock_observation(ra, dec, data_rights)

    async def _query_mission(
        self,
        mission: str,
        coord: Any,  # SkyCoord
        radius: float,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> list[dict[str, Any]]:
        """Query a specific mission from IRSA.

        Args:
            mission: Mission name (ZTF, PTF, WISE, etc.)
            coord: SkyCoord object
            radius: Search radius in degrees
            start_time: Start time filter
            end_time: End time filter

        Returns:
            List of observation dictionaries
        """
        mission_upper = mission.upper()

        if mission_upper == "ZTF":
            return await self._query_ztf(coord, radius, start_time, end_time)
        elif mission_upper in ["PTF", "WISE", "NEOWISE", "2MASS", "SPITZER"]:
            # TODO: Implement other missions as needed
            self.logger.warning(f"Mission {mission} not yet implemented, skipping")
            return []
        else:
            self.logger.warning(f"Unknown IRSA mission: {mission}")
            return []

    async def _query_ztf(
        self,
        coord: Any,  # SkyCoord
        radius: float,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> list[dict[str, Any]]:
        """Query ZTF science images from IRSA.

        Args:
            coord: SkyCoord object
            radius: Search radius in degrees
            start_time: Start time filter
            end_time: End time filter

        Returns:
            List of observation dictionaries
        """
        import asyncio
        from astropy.time import Time  # type: ignore

        observations = []

        try:
            # Build WHERE clause for time filtering
            where_clauses = []
            if start_time:
                start_jd = Time(start_time).jd
                where_clauses.append(f"obsjd >= {start_jd}")
            if end_time:
                end_jd = Time(end_time).jd
                where_clauses.append(f"obsjd <= {end_jd}")

            # Quality filter: exclude bad images (infobits < 33554432 is common threshold)
            where_clauses.append("infobits < 33554432")

            where_clause = " AND ".join(where_clauses) if where_clauses else None

            # Query ZTF science images
            self.logger.debug(
                f"Querying ZTF science images at ({coord.ra.deg}, {coord.dec.deg}) "
                f"with radius {radius}°"
            )

            # Convert radius to box size (IBE uses width/height)
            # Use string format to avoid Quantity truthiness issues
            box_size_deg = radius * 2  # Convert radius to box width/height
            width_str = f"{box_size_deg} deg"
            height_str = f"{box_size_deg} deg"

            # Run synchronous IBE query in executor to avoid blocking
            def _run_query():
                return Ibe.query_region(
                    coordinate=coord,
                    mission="ztf",
                    dataset="products",
                    table="sci",
                    width=width_str,  # Use string format instead of Quantity
                    height=height_str,  # Use string format instead of Quantity
                    where=where_clause,
                )

            # Execute in thread pool to avoid blocking event loop
            # Use get_running_loop() for Python 3.7+ compatibility
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Fallback if no event loop is running
                loop = asyncio.get_event_loop()
            obs_table = await loop.run_in_executor(None, _run_query)

            # Handle empty or None results
            if obs_table is None:
                self.logger.warning("ZTF query returned None")
                return []

            self.logger.info(f"ZTF query returned {len(obs_table)} raw observations")

            # Convert to standardized observation format
            for row in obs_table:
                # Build ZTF FITS URL
                file_url = self._build_ztf_url(row, table="sci")

                # Convert obsjd to datetime string
                obs_time = Time(row["obsjd"], format="jd")
                obs_date_str = obs_time.iso

                obs_dict = {
                    "obs_id": str(row.get("obsid", "")),
                    "target_name": str(row.get("target_name", "")),
                    "ra": float(row.get("ra", coord.ra.deg)),
                    "dec": float(row.get("dec", coord.dec.deg)),
                    "mission": "ZTF",
                    "instrument": "ZTF Camera",
                    "filters": str(row.get("filtercode", "")),
                    "exposure_time": float(row.get("exptime", 0.0)),
                    "obs_date": obs_date_str,
                    "data_rights": "PUBLIC",
                    "obs_collection": "ZTF",
                    "dataURL": file_url,
                    "filesize": int(row.get("size", 0)) if "size" in row.colnames else 0,
                }
                observations.append(obs_dict)

        except Exception as e:
            self.logger.error(f"Error querying ZTF: {e}", exc_info=True)
            # Return empty list on error rather than failing completely

        return observations

    def _build_ztf_url(self, row: Any, table: str = "sci") -> str:
        """Build ZTF FITS file URL from metadata row.

        Args:
            row: Table row from IBE query
            table: Table type ('sci', 'ref', 'deep', etc.)

        Returns:
            Full URL to ZTF FITS file
        """
        # ZTF URL pattern from IRSA documentation:
        # https://irsa.ipac.caltech.edu/ibe/data/ztf/products/{table}/
        # {year}/{monthday}/{fracday}/ztf_{filefracday}_{field}_{filtercode}_c{ccdid}_{imgtypecode}_q{qid}_{suffix}

        filefracday = str(row["filefracday"])
        year = filefracday[:4]
        monthday = filefracday[4:8]
        fracday = filefracday[8:14]

        field = f"{int(row['field']):06d}"
        filtercode = str(row["filtercode"])
        ccdid = f"{int(row['ccdid']):02d}"
        imgtypecode = str(row["imgtypecode"])
        qid = int(row["qid"])

        # Suffix depends on table type
        if table == "sci":
            suffix = "sciimg.fits"
        elif table == "ref":
            suffix = "refimg.fits"
        elif table == "deep":
            suffix = "diffimg.fits"
        else:
            suffix = "sciimg.fits"  # Default

        url = (
            f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/{table}/"
            f"{year}/{monthday}/{fracday}/"
            f"ztf_{filefracday}_{field}_{filtercode}_c{ccdid}_{imgtypecode}_q{qid}_{suffix}"
        )

        return url

    def _create_mock_observation(
        self, ra: float, dec: float, data_rights: str
    ) -> list[dict[str, Any]]:
        """Create mock observation data as fallback."""
        mock_observation = {
            "obs_id": f"mock_irsa_{ra}_{dec}",
            "target_name": "MOCK_TARGET",
            "ra": ra,
            "dec": dec,
            "mission": "ZTF",
            "instrument": "ZTF Camera",
            "filters": "zg",
            "exposure_time": 30.0,
            "obs_date": datetime.now().isoformat(),
            "data_rights": data_rights,
            "obs_collection": "ZTF",
            "dataURL": "https://irsa.ipac.caltech.edu/mock/...",
            "filesize": 1024000,
        }
        return [mock_observation]
