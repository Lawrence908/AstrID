"""Hubble Space Telescope (HST) data adapter."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from src.core.logging import configure_domain_logger
from src.domains.observations.adapters.base_adapter import SurveyAdapter
from src.domains.observations.schema import ObservationCreate


class HSTAdapter(SurveyAdapter):
    """Adapter for Hubble Space Telescope observation data."""

    def __init__(self):
        """Initialize HST adapter."""
        super().__init__("HST")
        self.logger = configure_domain_logger("observations.adapters.hst")

        # HST filter mapping to standard names
        self.filter_mapping = {
            # ACS filters
            "F435W": "B",
            "F475W": "g",
            "F555W": "V",
            "F606W": "V",
            "F625W": "r",
            "F775W": "i",
            "F814W": "I",
            "F850LP": "z",
            # WFC3 UVIS filters
            "F200LP": "UV",
            "F218W": "UV",
            "F225W": "UV",
            "F275W": "UV",
            "F336W": "U",
            "F390W": "U",
            "F438W": "B",
            "F467M": "B",
            "F547M": "V",
            "F621M": "r",
            "F689M": "R",
            "F763M": "i",
            "F845M": "z",
            # WFC3 IR filters
            "F105W": "Y",
            "F110W": "J",
            "F125W": "J",
            "F140W": "J",
            "F160W": "H",
            # WFPC2 filters
            "F300W": "U",
            "F450W": "B",
            "F702W": "R",
        }

        # HST instruments
        self.instruments = {
            "ACS": "Advanced Camera for Surveys",
            "WFC3": "Wide Field Camera 3",
            "WFPC2": "Wide Field and Planetary Camera 2",
            "NICMOS": "Near Infrared Camera and Multi-Object Spectrometer",
            "STIS": "Space Telescope Imaging Spectrograph",
            "COS": "Cosmic Origins Spectrograph",
            "FGS": "Fine Guidance Sensors",
        }

    async def normalize_observation_data(
        self, raw_data: dict[str, Any], survey_id: UUID
    ) -> ObservationCreate:
        """Normalize HST observation data to standard format.

        Args:
            raw_data: Raw HST observation data from MAST
            survey_id: Survey UUID

        Returns:
            Normalized observation data

        Raises:
            ValueError: For invalid or incomplete HST data
        """
        self.logger.debug(
            f"Normalizing HST observation: {raw_data.get('obs_id', 'unknown')}"
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
            airmass = None  # HST is in space, no airmass
            seeing = self._extract_seeing_equivalent(raw_data)

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
                f"Successfully normalized HST observation: {observation_id}"
            )
            return observation

        except Exception as e:
            self.logger.error(f"Failed to normalize HST observation: {e}")
            raise ValueError(f"HST data normalization failed: {e}") from e

    async def extract_metadata(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Extract HST-specific metadata.

        Args:
            raw_data: Raw HST observation data

        Returns:
            HST-specific metadata
        """
        self.logger.debug("Extracting HST metadata")

        try:
            instrument = raw_data.get("instrument_name", "unknown")
            detector = raw_data.get("detector", "unknown")
            proposal_id = raw_data.get("proposal_id")
            target_name = raw_data.get("target_name")

            # HST-specific fields
            expflag = raw_data.get("expflag")  # Exposure flag
            cal_level = raw_data.get("calib_level", 0)  # Calibration level

            # Image quality metrics
            cosmic_ray_flag = raw_data.get("cr_flag")
            quality_flag = raw_data.get("quality")

            # Pointing information
            position_angle = raw_data.get("orientat")
            aperture = raw_data.get("aperture")

            metadata = {
                "instrument": instrument,
                "instrument_description": self.instruments.get(instrument, instrument),
                "detector": detector,
                "proposal_id": proposal_id,
                "target_name": target_name,
                "exposure_flag": expflag,
                "calibration_level": cal_level,
                "cosmic_ray_flag": cosmic_ray_flag,
                "quality_flag": quality_flag,
                "position_angle_deg": position_angle,
                "aperture": aperture,
                "mission": "HST",
                "observatory": "Hubble Space Telescope",
            }

            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            self.logger.debug(f"Extracted HST metadata for {instrument}/{detector}")
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to extract HST metadata: {e}")
            raise ValueError(f"HST metadata extraction failed: {e}") from e

    async def validate_survey_specific_data(self, data: dict[str, Any]) -> bool:
        """Validate HST-specific data requirements.

        Args:
            data: HST observation data to validate

        Returns:
            True if data is valid

        Raises:
            ValueError: For validation failures
        """
        errors = []

        # Check for required HST fields
        required_fields = ["obs_id", "s_ra", "s_dec", "t_min", "filters", "t_exptime"]
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate HST instrument
        instrument = data.get("instrument_name")
        if instrument and instrument not in self.instruments:
            errors.append(f"Unknown HST instrument: {instrument}")

        # Validate filter
        filter_name = data.get("filters")
        if filter_name and not self._is_valid_hst_filter(filter_name):
            self.logger.warning(f"Unknown HST filter: {filter_name}")

        # Validate coordinates
        ra = data.get("s_ra")
        dec = data.get("s_dec")
        if ra is not None and not (0 <= ra <= 360):
            errors.append(f"Invalid RA: {ra}")
        if dec is not None and not (-90 <= dec <= 90):
            errors.append(f"Invalid Dec: {dec}")

        # Validate exposure time
        exp_time = data.get("t_exptime")
        if exp_time is not None and exp_time <= 0:
            errors.append(f"Invalid exposure time: {exp_time}")

        if errors:
            error_msg = "; ".join(errors)
            self.logger.error(f"HST data validation failed: {error_msg}")
            raise ValueError(f"HST data validation failed: {error_msg}")

        return True

    def get_supported_filters(self) -> list[str]:
        """Get HST supported filters.

        Returns:
            List of HST filter names
        """
        return list(self.filter_mapping.keys())

    def get_data_requirements(self) -> dict[str, Any]:
        """Get HST data requirements.

        Returns:
            HST-specific data requirements
        """
        base_requirements = super().get_data_requirements()
        base_requirements["survey_specific"] = {
            "required_fields": [
                "instrument_name",
                "obs_collection",
            ],
            "optional_fields": [
                "detector",
                "proposal_id",
                "target_name",
                "expflag",
                "calib_level",
                "orientat",
                "aperture",
            ],
            "supported_instruments": list(self.instruments.keys()),
            "supported_filters": self.get_supported_filters(),
        }
        return base_requirements

    def map_filter_band(self, survey_filter: str) -> str:
        """Map HST filter to standard photometric band.

        Args:
            survey_filter: HST filter name

        Returns:
            Standard filter name
        """
        return self.filter_mapping.get(survey_filter, survey_filter)

    def _extract_observation_id(self, raw_data: dict[str, Any]) -> str:
        """Extract HST observation ID."""
        obs_id = raw_data.get("obs_id") or raw_data.get("obsid")
        if not obs_id:
            raise ValueError("Missing HST observation ID")
        return str(obs_id)

    def _extract_coordinates(self, raw_data: dict[str, Any]) -> tuple[float, float]:
        """Extract coordinates from HST data."""
        ra = raw_data.get("s_ra")
        dec = raw_data.get("s_dec")

        if ra is None or dec is None:
            raise ValueError("Missing HST coordinates")

        return float(ra), float(dec)

    def _extract_observation_time(self, raw_data: dict[str, Any]) -> datetime:
        """Extract observation time from HST data."""
        time_mjd = raw_data.get("t_min")
        if time_mjd is None:
            raise ValueError("Missing HST observation time")

        # Convert MJD to datetime (UTC)
        timestamp = (float(time_mjd) - 2440587.5) * 86400.0
        return datetime.fromtimestamp(timestamp, tz=UTC)

    def _extract_filter_band(self, raw_data: dict[str, Any]) -> str:
        """Extract and normalize filter band."""
        filter_raw = raw_data.get("filters")
        if not filter_raw:
            raise ValueError("Missing HST filter information")

        # Map to standard filter name
        return self.map_filter_band(filter_raw)

    def _extract_exposure_time(self, raw_data: dict[str, Any]) -> float:
        """Extract exposure time from HST data."""
        exp_time = raw_data.get("t_exptime")
        if exp_time is None or exp_time <= 0:
            raise ValueError("Invalid HST exposure time")

        return float(exp_time)

    def _extract_fits_url(self, raw_data: dict[str, Any]) -> str:
        """Extract FITS data URL."""
        fits_url = raw_data.get("dataURL") or raw_data.get("data_uri")
        if not fits_url:
            # Construct MAST URL if not provided
            obs_id = raw_data.get("obs_id")
            if obs_id:
                fits_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={obs_id}"
            else:
                raise ValueError("Missing HST FITS URL")

        return fits_url

    def _extract_seeing_equivalent(self, raw_data: dict[str, Any]) -> float | None:
        """Extract HST point spread function width as seeing equivalent."""
        # HST doesn't have atmospheric seeing, but has PSF characteristics
        # This could be instrument-dependent PSF FWHM
        psf_fwhm = raw_data.get("psf_fwhm")
        if psf_fwhm:
            return float(psf_fwhm)

        # Default HST PSF values by instrument (approximate)
        instrument = raw_data.get("instrument_name")
        if instrument == "ACS":
            return 0.05  # ~0.05" for ACS
        elif instrument == "WFC3":
            detector = raw_data.get("detector")
            if detector == "UVIS":
                return 0.04  # ~0.04" for WFC3/UVIS
            elif detector == "IR":
                return 0.13  # ~0.13" for WFC3/IR
        elif instrument == "WFPC2":
            return 0.08  # ~0.08" for WFPC2

        return None

    def _is_valid_hst_filter(self, filter_name: str) -> bool:
        """Check if filter is a known HST filter."""
        return filter_name in self.filter_mapping

    def calculate_pixel_scale(self, raw_data: dict[str, Any]) -> float | None:
        """Calculate HST pixel scale based on instrument and detector."""
        pixel_scale = super().calculate_pixel_scale(raw_data)
        if pixel_scale:
            return pixel_scale

        # HST pixel scales by instrument/detector (arcsec/pixel)
        instrument = raw_data.get("instrument_name")
        detector = raw_data.get("detector")

        if instrument == "ACS":
            if detector == "WFC":
                return 0.05  # ACS/WFC
            elif detector == "HRC":
                return 0.025  # ACS/HRC
        elif instrument == "WFC3":
            if detector == "UVIS":
                return 0.04  # WFC3/UVIS
            elif detector == "IR":
                return 0.13  # WFC3/IR
        elif instrument == "WFPC2":
            return 0.1  # WFPC2
        elif instrument == "NICMOS":
            return 0.2  # NICMOS (varies by camera)

        return None
