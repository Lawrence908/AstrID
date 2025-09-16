"""Legacy Survey of Space and Time (LSST) data adapter."""

from datetime import datetime
from typing import Any
from uuid import UUID

from src.core.logging import configure_domain_logger
from src.domains.observations.adapters.base_adapter import SurveyAdapter
from src.domains.observations.schema import ObservationCreate


class LSSTAdapter(SurveyAdapter):
    """Adapter for Legacy Survey of Space and Time observation data."""

    def __init__(self):
        """Initialize LSST adapter."""
        super().__init__("LSST")
        self.logger = configure_domain_logger("observations.adapters.lsst")

        # LSST filter mapping to standard names
        self.filter_mapping = {
            "u": "u",
            "g": "g",
            "r": "r",
            "i": "i",
            "z": "z",
            "y": "y",
        }

        # LSST cameras and detectors
        self.cameras = {
            "LSSTCam": "LSST Camera",
            "ComCam": "Commissioning Camera",
            "AuxTel": "Auxiliary Telescope Camera",
        }

        # LSST surveys and programs
        self.surveys = {
            "WFD": "Wide-Fast-Deep Survey",
            "DDF": "Deep Drilling Fields",
            "NES": "Northern Ecliptic Spur",
            "SCP": "South Celestial Pole",
            "GP": "Galactic Plane",
            "Commissioning": "Commissioning Survey",
        }

    async def normalize_observation_data(
        self, raw_data: dict[str, Any], survey_id: UUID
    ) -> ObservationCreate:
        """Normalize LSST observation data to standard format.

        Args:
            raw_data: Raw LSST observation data
            survey_id: Survey UUID

        Returns:
            Normalized observation data

        Raises:
            ValueError: For invalid or incomplete LSST data
        """
        self.logger.debug(
            f"Normalizing LSST observation: {raw_data.get('obsid', 'unknown')}"
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
                f"Successfully normalized LSST observation: {observation_id}"
            )
            return observation

        except Exception as e:
            self.logger.error(f"Failed to normalize LSST observation: {e}")
            raise ValueError(f"LSST data normalization failed: {e}") from e

    async def extract_metadata(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Extract LSST-specific metadata.

        Args:
            raw_data: Raw LSST observation data

        Returns:
            LSST-specific metadata
        """
        self.logger.debug("Extracting LSST metadata")

        try:
            # LSST observation identification
            obsid = raw_data.get("obsid") or raw_data.get("observation_id")
            visit_id = raw_data.get("visit") or raw_data.get("visitid")
            exposure_id = raw_data.get("exposure") or raw_data.get("exposureid")

            # Survey program information
            survey_program = raw_data.get("survey") or raw_data.get("program")
            observation_type = raw_data.get("obstype") or raw_data.get(
                "observation_type"
            )

            # Instrument and detector
            camera = raw_data.get("camera") or raw_data.get("instrument")
            detector = raw_data.get("detector")
            raft = raw_data.get("raft")
            ccd = raw_data.get("ccd")

            # Observing conditions
            moon_illumination = raw_data.get("moonillum") or raw_data.get(
                "moon_illumination"
            )
            moon_separation = raw_data.get("moonsep") or raw_data.get("moon_separation")
            sun_alt = raw_data.get("sunalt") or raw_data.get("sun_altitude")

            # Image quality
            psf_sigma = raw_data.get("psf_sigma")
            sky_background = raw_data.get("skybackground") or raw_data.get("sky_level")
            zero_point = raw_data.get("zeropoint") or raw_data.get("photometric_zp")

            # Processing information
            processing_version = raw_data.get("procvers") or raw_data.get(
                "processing_version"
            )
            calibration_version = raw_data.get("calibvers") or raw_data.get(
                "calibration_version"
            )

            metadata = {
                "obsid": obsid,
                "visit_id": visit_id,
                "exposure_id": exposure_id,
                "survey_program": survey_program,
                "observation_type": observation_type,
                "camera": camera,
                "detector": detector,
                "raft": raft,
                "ccd": ccd,
                "moon_illumination": moon_illumination,
                "moon_separation_deg": moon_separation,
                "sun_altitude_deg": sun_alt,
                "psf_sigma_arcsec": psf_sigma,
                "sky_background": sky_background,
                "photometric_zero_point": zero_point,
                "processing_version": processing_version,
                "calibration_version": calibration_version,
                "survey": "LSST",
                "observatory": "Vera C. Rubin Observatory",
                "telescope": "Simonyi Survey Telescope",
                "instrument": camera,
            }

            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            self.logger.debug(f"Extracted LSST metadata for obsid {obsid}")
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to extract LSST metadata: {e}")
            raise ValueError(f"LSST metadata extraction failed: {e}") from e

    async def validate_survey_specific_data(self, data: dict[str, Any]) -> bool:
        """Validate LSST-specific data requirements.

        Args:
            data: LSST observation data to validate

        Returns:
            True if data is valid

        Raises:
            ValueError: For validation failures
        """
        errors = []

        # Check for LSST identification
        has_obsid = any(f in data for f in ["obsid", "observation_id"])
        has_visit = any(
            f in data for f in ["visit", "visitid", "exposure", "exposureid"]
        )

        if not (has_obsid or has_visit):
            errors.append(
                "Missing LSST identification: need obsid or visit/exposure ID"
            )

        # Check for coordinate information
        if not any(f in data for f in ["ra", "dec", "s_ra", "s_dec"]):
            errors.append("Missing coordinate information")

        # Validate filter
        filter_name = data.get("filter") or data.get("filters")
        if filter_name and not self._is_valid_lsst_filter(filter_name):
            errors.append(f"Unknown LSST filter: {filter_name}")

        # Validate coordinates if present
        ra = data.get("ra") or data.get("s_ra")
        dec = data.get("dec") or data.get("s_dec")
        if ra is not None and not (0 <= ra <= 360):
            errors.append(f"Invalid RA: {ra}")
        if dec is not None and not (-90 <= dec <= 90):
            errors.append(f"Invalid Dec: {dec}")

        # Validate camera
        camera = data.get("camera") or data.get("instrument")
        if camera and camera not in self.cameras:
            self.logger.warning(f"Unknown LSST camera: {camera}")

        # Validate exposure time
        exp_time = data.get("exptime") or data.get("t_exptime")
        if exp_time is not None and exp_time <= 0:
            errors.append(f"Invalid exposure time: {exp_time}")

        # Validate airmass if present
        airmass = data.get("airmass")
        if airmass is not None and (airmass < 1.0 or airmass > 5.0):
            errors.append(f"Invalid airmass: {airmass}")

        if errors:
            error_msg = "; ".join(errors)
            self.logger.error(f"LSST data validation failed: {error_msg}")
            raise ValueError(f"LSST data validation failed: {error_msg}")

        return True

    def get_supported_filters(self) -> list[str]:
        """Get LSST supported filters.

        Returns:
            List of LSST filter names
        """
        return list(self.filter_mapping.keys())

    def get_data_requirements(self) -> dict[str, Any]:
        """Get LSST data requirements.

        Returns:
            LSST-specific data requirements
        """
        base_requirements = super().get_data_requirements()
        base_requirements["survey_specific"] = {
            "identification_fields": {
                "option1": ["obsid"],
                "option2": ["visit", "exposure"],
            },
            "optional_fields": [
                "survey",
                "program",
                "obstype",
                "camera",
                "detector",
                "raft",
                "ccd",
                "moonillum",
                "moonsep",
                "sunalt",
                "psf_sigma",
                "skybackground",
                "zeropoint",
                "procvers",
                "calibvers",
            ],
            "supported_filters": self.get_supported_filters(),
            "surveys": list(self.surveys.keys()),
            "cameras": list(self.cameras.keys()),
        }
        return base_requirements

    def map_filter_band(self, survey_filter: str) -> str:
        """Map LSST filter to standard name.

        Args:
            survey_filter: LSST filter name

        Returns:
            Standard filter name
        """
        return self.filter_mapping.get(survey_filter, survey_filter)

    def _extract_observation_id(self, raw_data: dict[str, Any]) -> str:
        """Extract LSST observation ID."""
        # Try different ID formats
        obsid = raw_data.get("obsid") or raw_data.get("observation_id")
        if obsid:
            return f"lsst-{obsid}"

        # Build from visit/exposure
        visit = raw_data.get("visit") or raw_data.get("visitid")
        exposure = raw_data.get("exposure") or raw_data.get("exposureid")
        detector = raw_data.get("detector", "")

        if visit:
            if detector:
                return f"lsst-{visit}-{detector}"
            return f"lsst-{visit}"
        elif exposure:
            if detector:
                return f"lsst-exp-{exposure}-{detector}"
            return f"lsst-exp-{exposure}"

        raise ValueError("Cannot construct LSST observation ID")

    def _extract_coordinates(self, raw_data: dict[str, Any]) -> tuple[float, float]:
        """Extract coordinates from LSST data."""
        ra = raw_data.get("ra") or raw_data.get("s_ra")
        dec = raw_data.get("dec") or raw_data.get("s_dec")

        if ra is None or dec is None:
            raise ValueError("Missing LSST coordinates")

        return float(ra), float(dec)

    def _extract_observation_time(self, raw_data: dict[str, Any]) -> datetime:
        """Extract observation time from LSST data."""
        # Try different time fields
        mjd = raw_data.get("mjd") or raw_data.get("t_min")
        if mjd is not None:
            # Convert MJD to datetime
            timestamp = (float(mjd) - 2440587.5) * 86400.0
            return datetime.fromtimestamp(timestamp)

        # Try ISO format dates
        obs_date = (
            raw_data.get("date_obs")
            or raw_data.get("observation_date")
            or raw_data.get("dateobs")
        )
        if obs_date:
            if isinstance(obs_date, str):
                # Try common date formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        return datetime.strptime(obs_date, fmt)
                    except ValueError:
                        continue

        # Default fallback - LSST operations start
        return datetime(2024, 1, 1)  # LSST operations timeline

    def _extract_filter_band(self, raw_data: dict[str, Any]) -> str:
        """Extract and normalize filter band."""
        filter_raw = raw_data.get("filter") or raw_data.get("filters")
        if not filter_raw:
            raise ValueError("Missing LSST filter information")

        # Map to standard filter name
        return self.map_filter_band(filter_raw)

    def _extract_exposure_time(self, raw_data: dict[str, Any]) -> float:
        """Extract exposure time from LSST data."""
        exp_time = raw_data.get("exptime") or raw_data.get("t_exptime")
        if exp_time is not None and exp_time > 0:
            return float(exp_time)

        # LSST typical exposure times
        filter_name = raw_data.get("filter", "")
        if filter_name == "u":
            return 30.0  # u-band needs longer exposure
        else:
            return 15.0  # Standard LSST visit time

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

        # Construct LSST data URL based on available IDs
        obsid = raw_data.get("obsid")
        visit = raw_data.get("visit")
        detector = raw_data.get("detector")

        if obsid:
            return f"https://data.lsst.cloud/api/v1/obscore/obs/{obsid}"
        elif visit and detector:
            return f"https://data.lsst.cloud/api/v1/visit/{visit}/detector/{detector}"
        elif visit:
            return f"https://data.lsst.cloud/api/v1/visit/{visit}"

        # Fallback generic URL
        return f"https://data.lsst.cloud/api/v1/obs/{raw_data.get('observation_id', 'unknown')}"

    def _is_valid_lsst_filter(self, filter_name: str) -> bool:
        """Check if filter is a known LSST filter."""
        return filter_name in self.filter_mapping

    def calculate_pixel_scale(self, raw_data: dict[str, Any]) -> float | None:
        """Calculate LSST pixel scale."""
        pixel_scale = super().calculate_pixel_scale(raw_data)
        if pixel_scale:
            return pixel_scale

        # LSST pixel scale is 0.2 arcsec/pixel
        return 0.2

    def extract_image_dimensions(
        self, raw_data: dict[str, Any]
    ) -> tuple[int | None, int | None]:
        """Extract LSST image dimensions."""
        width, height = super().extract_image_dimensions(raw_data)
        if width and height:
            return width, height

        # LSST CCD dimensions are 4096x4096 pixels
        # But full focal plane images would be much larger
        camera = raw_data.get("camera", "LSSTCam")
        if camera == "LSSTCam":
            return 4096, 4096  # Single CCD
        elif camera == "ComCam":
            return 4096, 4096  # ComCam CCD
        elif camera == "AuxTel":
            return 2048, 2048  # AuxTel camera

        return 4096, 4096  # Default

    def extract_seeing(self, raw_data: dict[str, Any]) -> float | None:
        """Extract seeing from LSST data."""
        # Try LSST-specific seeing fields
        seeing = (
            raw_data.get("seeing")
            or raw_data.get("fwhm_eff")
            or raw_data.get("psf_fwhm")
            or super().extract_seeing(raw_data)
        )

        if seeing:
            return float(seeing)

        # Convert PSF sigma to FWHM if available
        psf_sigma = raw_data.get("psf_sigma")
        if psf_sigma:
            # FWHM = 2.35 * sigma for Gaussian PSF
            return float(psf_sigma) * 2.35

        return None  # Don't assume a default

    def calculate_airmass(self, raw_data: dict[str, Any]) -> float | None:
        """Calculate or extract airmass from LSST data."""
        # Try direct airmass field
        airmass = super().calculate_airmass(raw_data)
        if airmass:
            return airmass

        # Calculate from altitude if available
        alt = raw_data.get("altitude") or raw_data.get("alt")
        if alt is not None:
            import math

            alt_rad = math.radians(float(alt))
            if alt_rad > 0:  # Above horizon
                return 1.0 / math.sin(alt_rad)

        # Calculate from zenith distance
        zenith_dist = raw_data.get("zenith_distance") or raw_data.get("zd")
        if zenith_dist is not None:
            import math

            zd_rad = math.radians(float(zenith_dist))
            if zd_rad < math.radians(80):  # Reasonable limit
                return 1.0 / math.cos(zd_rad)

        return None
