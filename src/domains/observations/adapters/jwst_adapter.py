"""James Webb Space Telescope (JWST) data adapter."""

from datetime import datetime
from typing import Any
from uuid import UUID

from src.core.logging import configure_domain_logger
from src.domains.observations.adapters.base_adapter import SurveyAdapter
from src.domains.observations.schema import ObservationCreate


class JWSTAdapter(SurveyAdapter):
    """Adapter for James Webb Space Telescope observation data."""

    def __init__(self):
        """Initialize JWST adapter."""
        super().__init__("JWST")
        self.logger = configure_domain_logger("observations.adapters.jwst")

        # JWST filter mapping to standard names
        self.filter_mapping = {
            # NIRCam filters
            "F070W": "0.7μm",
            "F090W": "0.9μm",
            "F115W": "1.15μm",
            "F150W": "1.5μm",
            "F200W": "2.0μm",
            "F277W": "2.77μm",
            "F356W": "3.56μm",
            "F444W": "4.44μm",
            "F480M": "4.8μm",
            # NIRSpec filters
            "F100LP": "1.0μm-LP",
            "F170LP": "1.7μm-LP",
            "F290LP": "2.9μm-LP",
            # MIRI filters
            "F560W": "5.6μm",
            "F770W": "7.7μm",
            "F1000W": "10.0μm",
            "F1130W": "11.3μm",
            "F1280W": "12.8μm",
            "F1500W": "15.0μm",
            "F1800W": "18.0μm",
            "F2100W": "21.0μm",
            "F2550W": "25.5μm",
            # FGS filters
            "FGS": "FGS",
        }

        # JWST instruments
        self.instruments = {
            "NIRCAM": "Near Infrared Camera",
            "NIRSPEC": "Near Infrared Spectrograph",
            "MIRI": "Mid-Infrared Instrument",
            "FGS": "Fine Guidance Sensor",
            "NIRISS": "Near Infrared Imager and Slitless Spectrograph",
        }

        # JWST detectors
        self.detectors = {
            "NIRCAM": [
                "NRCA1",
                "NRCA2",
                "NRCA3",
                "NRCA4",
                "NRCB1",
                "NRCB2",
                "NRCB3",
                "NRCB4",
            ],
            "NIRSPEC": ["NRS1", "NRS2"],
            "MIRI": ["MIRIMAGE", "MIRIFULONG", "MIRIFUSHORT"],
            "FGS": ["GUIDER1", "GUIDER2"],
            "NIRISS": ["NIS"],
        }

    async def normalize_observation_data(
        self, raw_data: dict[str, Any], survey_id: UUID
    ) -> ObservationCreate:
        """Normalize JWST observation data to standard format.

        Args:
            raw_data: Raw JWST observation data from MAST
            survey_id: Survey UUID

        Returns:
            Normalized observation data

        Raises:
            ValueError: For invalid or incomplete JWST data
        """
        self.logger.debug(
            f"Normalizing JWST observation: {raw_data.get('obs_id', 'unknown')}"
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
            airmass = None  # JWST is in space, no airmass
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
                f"Successfully normalized JWST observation: {observation_id}"
            )
            return observation

        except Exception as e:
            self.logger.error(f"Failed to normalize JWST observation: {e}")
            raise ValueError(f"JWST data normalization failed: {e}") from e

    async def extract_metadata(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Extract JWST-specific metadata.

        Args:
            raw_data: Raw JWST observation data

        Returns:
            JWST-specific metadata
        """
        self.logger.debug("Extracting JWST metadata")

        try:
            instrument = raw_data.get("instrument_name", "unknown")
            detector = raw_data.get("detector", "unknown")
            proposal_id = raw_data.get("proposal_id")
            target_name = raw_data.get("target_name")

            # JWST-specific fields
            program_id = raw_data.get("project")
            visit_id = raw_data.get("visit_id")

            # Calibration information
            cal_level = raw_data.get("calib_level", 0)
            intent = raw_data.get("intent_type")  # science, calibration, etc.

            # Pointing and orientation
            position_angle = raw_data.get("orientat")
            aperture = raw_data.get("aperture")

            # Data processing
            pmap_version = raw_data.get("pmap_version")
            crds_version = raw_data.get("crds_version")

            metadata = {
                "instrument": instrument,
                "instrument_description": self.instruments.get(instrument, instrument),
                "detector": detector,
                "proposal_id": proposal_id,
                "program_id": program_id,
                "visit_id": visit_id,
                "target_name": target_name,
                "calibration_level": cal_level,
                "intent_type": intent,
                "position_angle_deg": position_angle,
                "aperture": aperture,
                "pmap_version": pmap_version,
                "crds_version": crds_version,
                "mission": "JWST",
                "observatory": "James Webb Space Telescope",
            }

            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            self.logger.debug(f"Extracted JWST metadata for {instrument}/{detector}")
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to extract JWST metadata: {e}")
            raise ValueError(f"JWST metadata extraction failed: {e}") from e

    async def validate_survey_specific_data(self, data: dict[str, Any]) -> bool:
        """Validate JWST-specific data requirements.

        Args:
            data: JWST observation data to validate

        Returns:
            True if data is valid

        Raises:
            ValueError: For validation failures
        """
        errors = []

        # Check for required JWST fields
        required_fields = ["obs_id", "s_ra", "s_dec", "t_min", "filters", "t_exptime"]
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate JWST instrument
        instrument = data.get("instrument_name")
        if instrument and instrument not in self.instruments:
            errors.append(f"Unknown JWST instrument: {instrument}")

        # Validate detector for instrument
        detector = data.get("detector")
        if instrument and detector:
            valid_detectors = self.detectors.get(instrument, [])
            if valid_detectors and detector not in valid_detectors:
                errors.append(
                    f"Invalid detector {detector} for instrument {instrument}"
                )

        # Validate filter
        filter_name = data.get("filters")
        if filter_name and not self._is_valid_jwst_filter(filter_name):
            self.logger.warning(f"Unknown JWST filter: {filter_name}")

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
            self.logger.error(f"JWST data validation failed: {error_msg}")
            raise ValueError(f"JWST data validation failed: {error_msg}")

        return True

    def get_supported_filters(self) -> list[str]:
        """Get JWST supported filters.

        Returns:
            List of JWST filter names
        """
        return list(self.filter_mapping.keys())

    def get_data_requirements(self) -> dict[str, Any]:
        """Get JWST data requirements.

        Returns:
            JWST-specific data requirements
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
                "program_id",
                "visit_id",
                "target_name",
                "intent_type",
                "calib_level",
                "orientat",
                "aperture",
                "pmap_version",
                "crds_version",
            ],
            "supported_instruments": list(self.instruments.keys()),
            "supported_filters": self.get_supported_filters(),
            "detectors_by_instrument": self.detectors,
        }
        return base_requirements

    def map_filter_band(self, survey_filter: str) -> str:
        """Map JWST filter to standard name.

        Args:
            survey_filter: JWST filter name

        Returns:
            Standard filter name
        """
        return self.filter_mapping.get(survey_filter, survey_filter)

    def _extract_observation_id(self, raw_data: dict[str, Any]) -> str:
        """Extract JWST observation ID."""
        obs_id = raw_data.get("obs_id") or raw_data.get("obsid")
        if not obs_id:
            raise ValueError("Missing JWST observation ID")
        return str(obs_id)

    def _extract_coordinates(self, raw_data: dict[str, Any]) -> tuple[float, float]:
        """Extract coordinates from JWST data."""
        ra = raw_data.get("s_ra")
        dec = raw_data.get("s_dec")

        if ra is None or dec is None:
            raise ValueError("Missing JWST coordinates")

        return float(ra), float(dec)

    def _extract_observation_time(self, raw_data: dict[str, Any]) -> datetime:
        """Extract observation time from JWST data."""
        time_mjd = raw_data.get("t_min")
        if time_mjd is None:
            raise ValueError("Missing JWST observation time")

        # Convert MJD to datetime
        timestamp = (float(time_mjd) - 2440587.5) * 86400.0
        return datetime.fromtimestamp(timestamp)

    def _extract_filter_band(self, raw_data: dict[str, Any]) -> str:
        """Extract and normalize filter band."""
        filter_raw = raw_data.get("filters")
        if not filter_raw:
            raise ValueError("Missing JWST filter information")

        # Map to standard filter name
        return self.map_filter_band(filter_raw)

    def _extract_exposure_time(self, raw_data: dict[str, Any]) -> float:
        """Extract exposure time from JWST data."""
        exp_time = raw_data.get("t_exptime")
        if exp_time is None or exp_time <= 0:
            raise ValueError("Invalid JWST exposure time")

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
                raise ValueError("Missing JWST FITS URL")

        return fits_url

    def _extract_seeing_equivalent(self, raw_data: dict[str, Any]) -> float | None:
        """Extract JWST point spread function width as seeing equivalent."""
        # JWST doesn't have atmospheric seeing, but has PSF characteristics
        psf_fwhm = raw_data.get("psf_fwhm")
        if psf_fwhm:
            return float(psf_fwhm)

        # Default JWST PSF values by instrument and wavelength (approximate)
        instrument = raw_data.get("instrument_name")
        filter_name = raw_data.get("filters", "")

        if instrument == "NIRCAM":
            # NIRCam PSF varies with wavelength
            if "F070W" in filter_name or "F090W" in filter_name:
                return 0.032  # ~0.032" at 1 μm
            elif "F115W" in filter_name or "F150W" in filter_name:
                return 0.040  # ~0.040" at 1.5 μm
            elif "F200W" in filter_name:
                return 0.063  # ~0.063" at 2 μm
            elif "F277W" in filter_name or "F356W" in filter_name:
                return 0.090  # ~0.090" at 3-4 μm
            elif "F444W" in filter_name:
                return 0.140  # ~0.140" at 4.4 μm
            else:
                return 0.070  # Average

        elif instrument == "MIRI":
            # MIRI PSF varies significantly with wavelength
            if "F560W" in filter_name:
                return 0.18  # ~0.18" at 5.6 μm
            elif "F770W" in filter_name:
                return 0.24  # ~0.24" at 7.7 μm
            elif "F1000W" in filter_name:
                return 0.31  # ~0.31" at 10 μm
            elif "F1130W" in filter_name:
                return 0.35  # ~0.35" at 11.3 μm
            elif "F1280W" in filter_name:
                return 0.40  # ~0.40" at 12.8 μm
            elif "F1500W" in filter_name:
                return 0.47  # ~0.47" at 15 μm
            elif "F1800W" in filter_name:
                return 0.56  # ~0.56" at 18 μm
            elif "F2100W" in filter_name:
                return 0.66  # ~0.66" at 21 μm
            elif "F2550W" in filter_name:
                return 0.80  # ~0.80" at 25.5 μm
            else:
                return 0.40  # Average

        elif instrument == "NIRSPEC":
            return 0.10  # ~0.10" typical for NIRSpec
        elif instrument == "NIRISS":
            return 0.065  # ~0.065" typical for NIRISS
        elif instrument == "FGS":
            return 0.069  # ~0.069" for FGS

        return None

    def _is_valid_jwst_filter(self, filter_name: str) -> bool:
        """Check if filter is a known JWST filter."""
        return filter_name in self.filter_mapping

    def calculate_pixel_scale(self, raw_data: dict[str, Any]) -> float | None:
        """Calculate JWST pixel scale based on instrument and detector."""
        pixel_scale = super().calculate_pixel_scale(raw_data)
        if pixel_scale:
            return pixel_scale

        # JWST pixel scales by instrument (arcsec/pixel)
        instrument = raw_data.get("instrument_name")

        if instrument == "NIRCAM":
            # NIRCam has two channels with different pixel scales
            detector = raw_data.get("detector", "")
            if "LONG" in detector or any(x in detector for x in ["NRCA5", "NRCB5"]):
                return 0.063  # Long wavelength channel
            else:
                return 0.031  # Short wavelength channel
        elif instrument == "MIRI":
            return 0.11  # MIRI imager
        elif instrument == "NIRSPEC":
            return 0.10  # NIRSpec
        elif instrument == "NIRISS":
            return 0.065  # NIRISS
        elif instrument == "FGS":
            return 0.069  # FGS

        return None
