"""Sloan Digital Sky Survey (SDSS) data adapter."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from src.core.logging import configure_domain_logger
from src.domains.observations.adapters.base_adapter import SurveyAdapter
from src.domains.observations.schema import ObservationCreate


class SDSSAdapter(SurveyAdapter):
    """Adapter for Sloan Digital Sky Survey observation data."""

    def __init__(self):
        """Initialize SDSS adapter."""
        super().__init__("SDSS")
        self.logger = configure_domain_logger("observations.adapters.sdss")

        # SDSS filter mapping to standard names
        self.filter_mapping = {
            "u": "u",
            "g": "g",
            "r": "r",
            "i": "i",
            "z": "z",
            "u'": "u",  # SDSS prime filters
            "g'": "g",
            "r'": "r",
            "i'": "i",
            "z'": "z",
        }

        # SDSS surveys and data releases
        self.surveys = {
            "SDSS": "Sloan Digital Sky Survey",
            "SDSS-I": "SDSS-I Legacy Survey",
            "SDSS-II": "SDSS-II Legacy and SEGUE",
            "SDSS-III": "SDSS-III BOSS and APOGEE",
            "SDSS-IV": "SDSS-IV eBOSS and APOGEE-2",
            "SDSS-V": "SDSS-V Milky Way Mapper",
        }

        # SDSS imaging runs
        self.imaging_surveys = [
            "SDSS Legacy Survey",
            "SDSS Stripe 82",
            "SEGUE",
        ]

    async def normalize_observation_data(
        self, raw_data: dict[str, Any], survey_id: UUID
    ) -> ObservationCreate:
        """Normalize SDSS observation data to standard format.

        Args:
            raw_data: Raw SDSS observation data
            survey_id: Survey UUID

        Returns:
            Normalized observation data

        Raises:
            ValueError: For invalid or incomplete SDSS data
        """
        self.logger.debug(
            f"Normalizing SDSS observation: {raw_data.get('field', 'unknown')}"
        )

        try:
            # Validate required fields
            await self.validate_survey_specific_data(raw_data)

            # Extract basic observation information
            observation_id = self._extract_observation_id(raw_data)
            ra, dec = self._extract_coordinates(raw_data)
            observation_time = self._extract_observation_time(raw_data)
            filter_band = self._extract_filter_band(raw_data)
            exposure_time = self._extract_exposure_time(raw_data)
            fits_url = self._extract_fits_url(raw_data)

            # Extract optional metadata
            pixel_scale = self.calculate_pixel_scale(raw_data)
            image_width, image_height = self.extract_image_dimensions(raw_data)
            airmass = self.calculate_airmass(raw_data)
            seeing = self.extract_seeing(raw_data)

            # Create normalized observation
            observation = ObservationCreate(
                survey_id=survey_id,
                observation_id=observation_id,
                ra=ra,
                dec=dec,
                observation_time=observation_time,
                filter_band=filter_band,
                exposure_time=exposure_time,
                fits_url=fits_url,
                pixel_scale=pixel_scale,
                image_width=image_width,
                image_height=image_height,
                airmass=airmass,
                seeing=seeing,
            )

            self.logger.debug(
                f"Successfully normalized SDSS observation: {observation_id}"
            )
            return observation

        except Exception as e:
            self.logger.error(f"Failed to normalize SDSS observation: {e}")
            raise ValueError(f"SDSS data normalization failed: {e}") from e

    async def extract_metadata(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Extract SDSS-specific metadata.

        Args:
            raw_data: Raw SDSS observation data

        Returns:
            SDSS-specific metadata
        """
        self.logger.debug("Extracting SDSS metadata")

        try:
            # SDSS run/field/camcol identification
            run = raw_data.get("run")
            field = raw_data.get("field")
            camcol = raw_data.get("camcol")

            # Data release information
            data_release = raw_data.get("data_release") or raw_data.get("dr")

            # Photometric information
            photoobj_id = raw_data.get("photoobj_id") or raw_data.get("objid")

            # Quality flags
            clean_flag = raw_data.get("clean")
            star_flag = raw_data.get("star")
            galaxy_flag = raw_data.get("galaxy")

            # Photometry
            petrorad = raw_data.get("petrorad")
            petromag = raw_data.get("petromag")
            modelmag = raw_data.get("modelmag")

            # Processing information
            processing_version = raw_data.get("rerun")
            calibration_version = raw_data.get("calibration_version")

            metadata = {
                "run": run,
                "field": field,
                "camcol": camcol,
                "data_release": data_release,
                "photoobj_id": photoobj_id,
                "clean_flag": clean_flag,
                "star_flag": star_flag,
                "galaxy_flag": galaxy_flag,
                "petrosian_radius": petrorad,
                "petrosian_magnitude": petromag,
                "model_magnitude": modelmag,
                "processing_version": processing_version,
                "calibration_version": calibration_version,
                "survey": "SDSS",
                "observatory": "Apache Point Observatory",
                "telescope": "SDSS 2.5m",
                "instrument": "Photometric Camera",
            }

            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            self.logger.debug(f"Extracted SDSS metadata for run {run}, field {field}")
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to extract SDSS metadata: {e}")
            raise ValueError(f"SDSS metadata extraction failed: {e}") from e

    async def validate_survey_specific_data(self, data: dict[str, Any]) -> bool:
        """Validate SDSS-specific data requirements.

        Args:
            data: SDSS observation data to validate

        Returns:
            True if data is valid

        Raises:
            ValueError: For validation failures
        """
        errors = []

        # Check for SDSS identification fields (need at least one set)
        has_run_field = all(f in data for f in ["run", "field", "camcol"])
        has_objid = "objid" in data or "photoobj_id" in data

        if not (has_run_field or has_objid):
            errors.append(
                "Missing SDSS identification: need either (run,field,camcol) or objid"
            )

        # Check for coordinate information
        if not any(f in data for f in ["ra", "dec", "s_ra", "s_dec"]):
            errors.append("Missing coordinate information")

        # Validate filter
        filter_name = data.get("filter") or data.get("filters")
        if filter_name and not self._is_valid_sdss_filter(filter_name):
            errors.append(f"Unknown SDSS filter: {filter_name}")

        # Validate coordinates if present
        ra = data.get("ra") or data.get("s_ra")
        dec = data.get("dec") or data.get("s_dec")
        if ra is not None and not (0 <= ra <= 360):
            errors.append(f"Invalid RA: {ra}")
        if dec is not None and not (-90 <= dec <= 90):
            errors.append(f"Invalid Dec: {dec}")

        # Validate SDSS-specific fields
        run = data.get("run")
        if run is not None and (run < 0 or run > 10000):  # Reasonable range
            errors.append(f"Invalid SDSS run: {run}")

        field = data.get("field")
        if field is not None and (field < 0 or field > 1000):  # Reasonable range
            errors.append(f"Invalid SDSS field: {field}")

        camcol = data.get("camcol")
        if camcol is not None and camcol not in [1, 2, 3, 4, 5, 6]:
            errors.append(f"Invalid SDSS camcol: {camcol}")

        if errors:
            error_msg = "; ".join(errors)
            self.logger.error(f"SDSS data validation failed: {error_msg}")
            raise ValueError(f"SDSS data validation failed: {error_msg}")

        return True

    def get_supported_filters(self) -> list[str]:
        """Get SDSS supported filters.

        Returns:
            List of SDSS filter names
        """
        return list(self.filter_mapping.keys())

    def get_data_requirements(self) -> dict[str, Any]:
        """Get SDSS data requirements.

        Returns:
            SDSS-specific data requirements
        """
        base_requirements = super().get_data_requirements()
        base_requirements["survey_specific"] = {
            "identification_fields": {
                "option1": ["run", "field", "camcol"],
                "option2": ["objid"],
            },
            "optional_fields": [
                "data_release",
                "clean",
                "star",
                "galaxy",
                "petrorad",
                "petromag",
                "modelmag",
                "rerun",
                "calibration_version",
            ],
            "supported_filters": self.get_supported_filters(),
            "surveys": list(self.surveys.keys()),
            "valid_camcol": [1, 2, 3, 4, 5, 6],
        }
        return base_requirements

    def map_filter_band(self, survey_filter: str) -> str:
        """Map SDSS filter to standard name.

        Args:
            survey_filter: SDSS filter name

        Returns:
            Standard filter name
        """
        return self.filter_mapping.get(survey_filter, survey_filter)

    def _extract_observation_id(self, raw_data: dict[str, Any]) -> str:
        """Extract SDSS observation ID."""
        # Try different ID formats
        objid = raw_data.get("objid") or raw_data.get("photoobj_id")
        if objid:
            return f"sdss-{objid}"

        # Build from run/field/camcol
        run = raw_data.get("run")
        field = raw_data.get("field")
        camcol = raw_data.get("camcol")
        filter_name = raw_data.get("filter") or raw_data.get("filters", "")

        if all(x is not None for x in [run, field, camcol]):
            return f"sdss-{run}-{field}-{camcol}-{filter_name}"

        raise ValueError("Cannot construct SDSS observation ID")

    def _extract_coordinates(self, raw_data: dict[str, Any]) -> tuple[float, float]:
        """Extract coordinates from SDSS data."""
        ra = raw_data.get("ra") or raw_data.get("s_ra")
        dec = raw_data.get("dec") or raw_data.get("s_dec")

        if ra is None or dec is None:
            raise ValueError("Missing SDSS coordinates")

        return float(ra), float(dec)

    def _extract_observation_time(self, raw_data: dict[str, Any]) -> datetime:
        """Extract observation time from SDSS data."""
        # SDSS uses Modified Julian Date
        mjd = raw_data.get("mjd") or raw_data.get("t_min")
        if mjd is not None:
            # Convert MJD to datetime (UTC)
            timestamp = (float(mjd) - 2440587.5) * 86400.0
            return datetime.fromtimestamp(timestamp, tz=UTC)

        # Fallback: try to extract from date fields
        obs_date = raw_data.get("date_obs") or raw_data.get("observation_date")
        if obs_date:
            if isinstance(obs_date, str):
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        return datetime.strptime(obs_date, fmt)
                    except ValueError:
                        continue

        # Default fallback - use a representative SDSS date
        return datetime(2000, 1, 1)  # SDSS started around 2000

    def _extract_filter_band(self, raw_data: dict[str, Any]) -> str:
        """Extract and normalize filter band."""
        filter_raw = raw_data.get("filter") or raw_data.get("filters")
        if not filter_raw:
            # Default to 'r' if not specified (SDSS r-band is common)
            filter_raw = "r"

        # Map to standard filter name
        return self.map_filter_band(filter_raw)

    def _extract_exposure_time(self, raw_data: dict[str, Any]) -> float:
        """Extract exposure time from SDSS data."""
        exp_time = raw_data.get("exptime") or raw_data.get("t_exptime")
        if exp_time is not None and exp_time > 0:
            return float(exp_time)

        # SDSS typical exposure time is 53.9 seconds per frame
        return 53.9

    def _extract_fits_url(self, raw_data: dict[str, Any]) -> str:
        """Extract or construct FITS data URL."""
        # Try direct URL first
        fits_url = (
            raw_data.get("dataURL")
            or raw_data.get("data_uri")
            or raw_data.get("fits_url")
        )
        if fits_url:
            return fits_url

        # Construct SDSS data URL
        run = raw_data.get("run")
        field = raw_data.get("field")
        camcol = raw_data.get("camcol")
        filter_name = raw_data.get("filter", "r")

        if all(x is not None for x in [run, field, camcol]):
            # SDSS Data Archive Server URL format
            return f"https://data.sdss.org/sas/dr16/boss/photoObj/frames/{run}/{camcol}/frame-{filter_name}-{run:06d}-{camcol}-{field:04d}.fits.bz2"

        # Fallback generic URL
        return f"https://data.sdss.org/sas/dr16/sdss/{raw_data.get('objid', 'unknown')}"

    def _is_valid_sdss_filter(self, filter_name: str) -> bool:
        """Check if filter is a known SDSS filter."""
        return filter_name in self.filter_mapping

    def calculate_pixel_scale(self, raw_data: dict[str, Any]) -> float | None:
        """Calculate SDSS pixel scale."""
        pixel_scale = super().calculate_pixel_scale(raw_data)
        if pixel_scale:
            return pixel_scale

        # SDSS pixel scale is 0.396 arcsec/pixel
        return 0.396

    def extract_image_dimensions(
        self, raw_data: dict[str, Any]
    ) -> tuple[int | None, int | None]:
        """Extract SDSS image dimensions."""
        width, height = super().extract_image_dimensions(raw_data)
        if width and height:
            return width, height

        # SDSS images are typically 2048x1489 pixels (camcol dependent)
        # But frame images are 2048x1489, while field images can be larger
        return 2048, 1489

    def extract_seeing(self, raw_data: dict[str, Any]) -> float | None:
        """Extract seeing from SDSS data."""
        # Try SDSS-specific seeing fields
        seeing = (
            raw_data.get("seeing")
            or raw_data.get("psf_fwhm")
            or raw_data.get("psfwidth")
            or super().extract_seeing(raw_data)
        )

        if seeing:
            return float(seeing)

        # SDSS typical seeing is around 1.4 arcseconds
        return None  # Don't assume a default
