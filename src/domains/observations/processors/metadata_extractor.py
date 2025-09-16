"""Enhanced metadata extraction for FITS astronomical observations."""

import io
import logging

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Enhanced metadata extractor for comprehensive FITS analysis."""

    def __init__(self):
        """Initialize metadata extractor."""
        self.logger = logging.getLogger(__name__)

        # Known telescope and instrument configurations
        self.instrument_configs = {
            "hst": {
                "keywords": ["TELESCOP", "INSTRUME", "DETECTOR"],
                "filters": ["FILTER1", "FILTER2", "FILTNAM1", "FILTNAM2"],
                "exposure": ["EXPTIME", "TEXPTIME"],
            },
            "jwst": {
                "keywords": ["TELESCOP", "INSTRUME", "DETECTOR"],
                "filters": ["FILTER", "PUPIL"],
                "exposure": ["EFFEXPTM", "DURATION"],
            },
            "spitzer": {
                "keywords": ["TELESCOP", "INSTRUME", "CHNLNUM"],
                "filters": ["FILTER", "WAVELNTH"],
                "exposure": ["EXPTIME", "FRAMTIME"],
            },
            "generic": {
                "keywords": ["TELESCOP", "INSTRUME", "DETECTOR"],
                "filters": ["FILTER", "BAND", "FILTNAM"],
                "exposure": ["EXPTIME", "EXPOSURE"],
            },
        }

    def extract_photometric_parameters(self, fits_data: bytes) -> dict:
        """Extract photometric calibration and measurement parameters.

        Args:
            fits_data: FITS file data as bytes

        Returns:
            dict: Photometric parameters and calibration info
        """
        parameters = {
            "photometric_system": None,
            "zero_point": None,
            "zero_point_error": None,
            "extinction_coefficient": None,
            "airmass": None,
            "filter_band": None,
            "effective_wavelength": None,
            "aperture_corrections": {},
            "color_terms": {},
            "photometric_quality": None,
            "calibration_source": None,
            "saturation_level": None,
            "gain": None,
            "read_noise": None,
        }

        try:
            with fits.open(io.BytesIO(fits_data)) as hdul:
                header = hdul[0].header

                # Extract basic photometric info
                parameters["filter_band"] = self._get_header_value(
                    header, ["FILTER", "BAND", "FILTNAM", "FILTER1"], default="unknown"
                )

                parameters["airmass"] = self._get_header_value(
                    header, ["AIRMASS", "SECZ"], dtype=float
                )

                # Zero point and calibration
                parameters["zero_point"] = self._get_header_value(
                    header, ["MAGZPT", "ZPOINT", "ZP"], dtype=float
                )

                parameters["zero_point_error"] = self._get_header_value(
                    header, ["MAGZPTER", "ZPOINTER", "ZP_ERR"], dtype=float
                )

                # Extinction
                parameters["extinction_coefficient"] = self._get_header_value(
                    header, ["EXTINCT", "EXTCOEFF", "KEXT"], dtype=float
                )

                # Detector parameters
                parameters["gain"] = self._get_header_value(
                    header, ["GAIN", "EGAIN", "CCDGAIN"], dtype=float
                )

                parameters["read_noise"] = self._get_header_value(
                    header, ["RDNOISE", "READNOIS", "RON"], dtype=float
                )

                parameters["saturation_level"] = self._get_header_value(
                    header, ["SATURATE", "SATLEVEL", "DATAMAX"], dtype=float
                )

                # Wavelength information
                parameters["effective_wavelength"] = self._get_header_value(
                    header, ["WAVELENG", "WAVELEN", "EFF_WAVE"], dtype=float
                )

                # Photometric system
                parameters["photometric_system"] = self._get_header_value(
                    header, ["PHOTSYS", "MAGSYS"], default="unknown"
                )

                # Quality assessment
                parameters["photometric_quality"] = self._assess_photometric_quality(
                    header
                )

                # Calibration source
                parameters["calibration_source"] = self._get_header_value(
                    header, ["CALREF", "PHOTREF", "STDFIELD"]
                )

                # Extract aperture corrections if available
                parameters["aperture_corrections"] = self._extract_aperture_corrections(
                    header
                )

                # Extract color terms if available
                parameters["color_terms"] = self._extract_color_terms(header)

                self.logger.debug(
                    f"Extracted photometric parameters for filter {parameters['filter_band']}"
                )

        except Exception as e:
            self.logger.error(f"Error extracting photometric parameters: {e}")
            parameters["extraction_error"] = str(e)

        return parameters

    def extract_observing_conditions(self, fits_data: bytes) -> dict:
        """Extract observing conditions and atmospheric parameters.

        Args:
            fits_data: FITS file data as bytes

        Returns:
            dict: Observing conditions and environmental data
        """
        conditions = {
            "seeing": None,
            "airmass": None,
            "sky_brightness": None,
            "cloud_cover": None,
            "humidity": None,
            "temperature": None,
            "pressure": None,
            "wind_speed": None,
            "wind_direction": None,
            "moon_phase": None,
            "moon_separation": None,
            "sun_elevation": None,
            "observation_date": None,
            "observation_time": None,
            "mjd": None,
            "weather_quality": None,
            "atmospheric_transparency": None,
        }

        try:
            with fits.open(io.BytesIO(fits_data)) as hdul:
                header = hdul[0].header

                # Seeing conditions
                conditions["seeing"] = self._get_header_value(
                    header, ["SEEING", "FWHM", "PSF_FWHM"], dtype=float
                )

                # Airmass
                conditions["airmass"] = self._get_header_value(
                    header, ["AIRMASS", "SECZ", "AIRM"], dtype=float
                )

                # Sky conditions
                conditions["sky_brightness"] = self._get_header_value(
                    header, ["SKYBRITE", "SKYBRGHT", "SKYVAL"], dtype=float
                )

                # Weather parameters
                conditions["cloud_cover"] = self._get_header_value(
                    header, ["CLOUDS", "CLOUDCOV", "WEATHER"], dtype=float
                )

                conditions["humidity"] = self._get_header_value(
                    header, ["HUMIDITY", "HUMID", "RH"], dtype=float
                )

                conditions["temperature"] = self._get_header_value(
                    header, ["TEMP", "TEMPERAT", "AIRTEMP"], dtype=float
                )

                conditions["pressure"] = self._get_header_value(
                    header, ["PRESSURE", "PRESS", "AIRPRESS"], dtype=float
                )

                # Wind conditions
                conditions["wind_speed"] = self._get_header_value(
                    header, ["WINDSPD", "WINDVEL", "WIND"], dtype=float
                )

                conditions["wind_direction"] = self._get_header_value(
                    header, ["WINDDIR", "WINDDIRN"], dtype=float
                )

                # Time and date
                obs_date = self._get_header_value(header, ["DATE-OBS", "DATE_OBS"])
                obs_time = self._get_header_value(header, ["TIME-OBS", "TIME_OBS"])

                if obs_date:
                    conditions["observation_date"] = obs_date
                    if obs_time:
                        conditions["observation_time"] = obs_time
                        try:
                            # Parse full datetime
                            if "T" in obs_date:
                                dt = Time(obs_date)
                            else:
                                dt = Time(f"{obs_date}T{obs_time}")
                            conditions["mjd"] = float(dt.mjd)
                        except Exception:
                            pass

                # MJD if available directly
                if conditions["mjd"] is None:
                    conditions["mjd"] = self._get_header_value(
                        header, ["MJD", "MJD-OBS", "MJDOBS"], dtype=float
                    )

                # Moon conditions
                conditions["moon_phase"] = self._get_header_value(
                    header, ["MOONPHSE", "MOONPH", "LUNATION"], dtype=float
                )

                conditions["moon_separation"] = self._get_header_value(
                    header, ["MOONSEP", "MOONANGL"], dtype=float
                )

                # Sun elevation
                conditions["sun_elevation"] = self._get_header_value(
                    header, ["SUNELEV", "SUNANGL", "SUNALT"], dtype=float
                )

                # Quality assessments
                conditions["weather_quality"] = self._assess_weather_quality(conditions)
                conditions["atmospheric_transparency"] = (
                    self._assess_atmospheric_transparency(conditions)
                )

                self.logger.debug("Extracted observing conditions")

        except Exception as e:
            self.logger.error(f"Error extracting observing conditions: {e}")
            conditions["extraction_error"] = str(e)

        return conditions

    def extract_instrument_parameters(self, fits_data: bytes) -> dict:
        """Extract instrument-specific parameters and configuration.

        Args:
            fits_data: FITS file data as bytes

        Returns:
            dict: Instrument parameters and configuration
        """
        parameters = {
            "telescope": None,
            "instrument": None,
            "detector": None,
            "pixel_scale": None,
            "field_of_view": None,
            "focal_length": None,
            "f_ratio": None,
            "binning": None,
            "read_mode": None,
            "exposure_time": None,
            "number_of_exposures": None,
            "filter_wheel_position": None,
            "instrument_mode": None,
            "detector_temperature": None,
            "optics_temperature": None,
            "focus_position": None,
            "instrument_config": None,
        }

        try:
            with fits.open(io.BytesIO(fits_data)) as hdul:
                header = hdul[0].header

                # Basic instrument identification
                parameters["telescope"] = self._get_header_value(
                    header, ["TELESCOP", "SCOPE", "TELNAME"]
                )

                parameters["instrument"] = self._get_header_value(
                    header, ["INSTRUME", "INSTRMNT", "CAMERA"]
                )

                parameters["detector"] = self._get_header_value(
                    header, ["DETECTOR", "DETNAME", "CCD"]
                )

                # Optical parameters
                parameters["pixel_scale"] = self._calculate_pixel_scale_from_header(
                    header
                )
                parameters["focal_length"] = self._get_header_value(
                    header, ["FOCALLEN", "FOCUS", "FL"], dtype=float
                )

                parameters["f_ratio"] = self._get_header_value(
                    header, ["FRATIO", "F_RATIO", "FNUM"], dtype=float
                )

                # Detector configuration
                parameters["binning"] = self._extract_binning(header)
                parameters["read_mode"] = self._get_header_value(
                    header, ["READMODE", "RDMODE", "AMPMODE"]
                )

                # Exposure parameters
                parameters["exposure_time"] = self._get_header_value(
                    header, ["EXPTIME", "EXPOSURE", "TEXPTIME"], dtype=float
                )

                parameters["number_of_exposures"] = self._get_header_value(
                    header, ["NEXP", "NUMEXP", "NFRAMES"], dtype=int
                )

                # Instrument settings
                parameters["filter_wheel_position"] = self._get_header_value(
                    header, ["FWPOS", "FILPOS", "FILTERP"], dtype=int
                )

                parameters["instrument_mode"] = self._get_header_value(
                    header, ["INSTMODE", "MODE", "OBSMODE"]
                )

                # Temperature monitoring
                parameters["detector_temperature"] = self._get_header_value(
                    header, ["DETTEMP", "CCDTEMP", "TEMP_DET"], dtype=float
                )

                parameters["optics_temperature"] = self._get_header_value(
                    header, ["OPTTEMP", "TEMP_OPT"], dtype=float
                )

                # Focus information
                parameters["focus_position"] = self._get_header_value(
                    header, ["FOCUSPOS", "FOCUS", "FOCPOS"], dtype=float
                )

                # Field of view calculation
                if (
                    parameters["pixel_scale"]
                    and "NAXIS1" in header
                    and "NAXIS2" in header
                ):
                    width_arcmin = (header["NAXIS1"] * parameters["pixel_scale"]) / 60
                    height_arcmin = (header["NAXIS2"] * parameters["pixel_scale"]) / 60
                    parameters["field_of_view"] = {
                        "width_arcmin": width_arcmin,
                        "height_arcmin": height_arcmin,
                        "area_sq_arcmin": width_arcmin * height_arcmin,
                    }

                # Determine instrument configuration
                parameters["instrument_config"] = self._identify_instrument_config(
                    parameters["telescope"], parameters["instrument"]
                )

                self.logger.debug(
                    f"Extracted parameters for {parameters['telescope']}/{parameters['instrument']}"
                )

        except Exception as e:
            self.logger.error(f"Error extracting instrument parameters: {e}")
            parameters["extraction_error"] = str(e)

        return parameters

    def extract_quality_metrics(self, fits_data: bytes) -> dict:
        """Extract image quality metrics and statistics.

        Args:
            fits_data: FITS file data as bytes

        Returns:
            dict: Quality metrics and image statistics
        """
        metrics = {
            "signal_to_noise": None,
            "background_level": None,
            "background_rms": None,
            "saturation_fraction": None,
            "cosmic_ray_count": None,
            "bad_pixel_fraction": None,
            "point_source_fwhm": None,
            "ellipticity": None,
            "streak_detection": None,
            "image_statistics": {},
            "quality_flags": [],
            "overall_quality_score": None,
        }

        try:
            with fits.open(io.BytesIO(fits_data)) as hdul:
                header = hdul[0].header
                image_data = hdul[0].data

                if image_data is not None:
                    # Basic image statistics
                    metrics["image_statistics"] = {
                        "mean": float(np.mean(image_data)),
                        "median": float(np.median(image_data)),
                        "std": float(np.std(image_data)),
                        "min": float(np.min(image_data)),
                        "max": float(np.max(image_data)),
                        "percentile_05": float(np.percentile(image_data, 5)),
                        "percentile_95": float(np.percentile(image_data, 95)),
                    }

                    # Background estimation
                    metrics["background_level"] = metrics["image_statistics"]["median"]
                    metrics["background_rms"] = self._estimate_background_rms(
                        image_data
                    )

                    # Saturation analysis
                    saturation_level = self._get_header_value(
                        header, ["SATURATE", "SATLEVEL", "DATAMAX"], dtype=float
                    )
                    if saturation_level:
                        saturated_pixels = np.sum(image_data >= saturation_level * 0.95)
                        metrics["saturation_fraction"] = float(
                            saturated_pixels / image_data.size
                        )

                    # Signal-to-noise estimation
                    if metrics["background_rms"]:
                        signal = (
                            metrics["image_statistics"]["percentile_95"]
                            - metrics["background_level"]
                        )
                        metrics["signal_to_noise"] = signal / metrics["background_rms"]

                # Extract quality metrics from header
                metrics["point_source_fwhm"] = self._get_header_value(
                    header, ["SEEING", "FWHM", "PSF_FWHM"], dtype=float
                )

                metrics["ellipticity"] = self._get_header_value(
                    header, ["ELLIPTIC", "ELONG", "PSF_ELL"], dtype=float
                )

                # Quality flags from header
                quality_keywords = ["QUALITY", "DATAQUAL", "IMGQUAL"]
                for keyword in quality_keywords:
                    if keyword in header:
                        metrics["quality_flags"].append(f"{keyword}: {header[keyword]}")

                # Calculate overall quality score
                metrics["overall_quality_score"] = self._calculate_quality_score(
                    metrics
                )

                self.logger.debug("Extracted quality metrics")

        except Exception as e:
            self.logger.error(f"Error extracting quality metrics: {e}")
            metrics["extraction_error"] = str(e)

        return metrics

    def extract_astrometric_solution(self, fits_data: bytes) -> dict:
        """Extract astrometric calibration and WCS solution information.

        Args:
            fits_data: FITS file data as bytes

        Returns:
            dict: Astrometric solution parameters and quality metrics
        """
        solution = {
            "wcs_present": False,
            "coordinate_system": None,
            "reference_frame": None,
            "equinox": None,
            "projection_type": None,
            "reference_pixel": None,
            "reference_coordinates": None,
            "pixel_scale": None,
            "rotation_angle": None,
            "distortion_model": None,
            "astrometric_residuals": None,
            "star_matches": None,
            "catalog_reference": None,
            "solution_quality": None,
        }

        try:
            with fits.open(io.BytesIO(fits_data)) as hdul:
                header = hdul[0].header

                # Check for WCS presence
                wcs_keywords = ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2"]
                solution["wcs_present"] = all(kw in header for kw in wcs_keywords)

                if solution["wcs_present"]:
                    # Extract WCS information
                    solution["coordinate_system"] = header.get("CTYPE1", "").split("-")[
                        0
                    ]
                    solution["projection_type"] = (
                        header.get("CTYPE1", "").split("-")[-1]
                        if "-" in header.get("CTYPE1", "")
                        else None
                    )

                    solution["reference_frame"] = header.get(
                        "RADESYS", header.get("RADECSYS")
                    )
                    solution["equinox"] = self._get_header_value(
                        header, ["EQUINOX", "EPOCH"], dtype=float
                    )

                    # Reference pixel and coordinates
                    solution["reference_pixel"] = [
                        self._get_header_value(header, ["CRPIX1"], dtype=float),
                        self._get_header_value(header, ["CRPIX2"], dtype=float),
                    ]

                    solution["reference_coordinates"] = [
                        self._get_header_value(header, ["CRVAL1"], dtype=float),
                        self._get_header_value(header, ["CRVAL2"], dtype=float),
                    ]

                    # Calculate pixel scale and rotation
                    try:
                        wcs = WCS(header)
                        solution["pixel_scale"] = self._calculate_wcs_pixel_scale(wcs)
                        solution["rotation_angle"] = self._calculate_wcs_rotation(wcs)
                    except Exception as e:
                        self.logger.warning(f"Could not calculate WCS parameters: {e}")

                    # Distortion information
                    distortion_keywords = [
                        "A_ORDER",
                        "B_ORDER",
                        "SIP",
                        "TPV",
                        "TAN-SIP",
                    ]
                    for keyword in distortion_keywords:
                        if keyword in header or any(
                            keyword in key for key in header.keys()
                        ):
                            solution["distortion_model"] = keyword
                            break

                    # Astrometric quality metrics
                    solution["astrometric_residuals"] = self._get_header_value(
                        header, ["ASTRRMS", "ASTRMS", "WCSERR"], dtype=float
                    )

                    solution["star_matches"] = self._get_header_value(
                        header, ["ASTRNMAT", "NSTARS", "WCSAXES"], dtype=int
                    )

                    solution["catalog_reference"] = self._get_header_value(
                        header, ["ASTREF", "CATALOG", "WCSCAT"]
                    )

                    # Assess solution quality
                    solution["solution_quality"] = self._assess_astrometric_quality(
                        solution, header
                    )

                self.logger.debug(
                    f"Extracted astrometric solution (WCS present: {solution['wcs_present']})"
                )

        except Exception as e:
            self.logger.error(f"Error extracting astrometric solution: {e}")
            solution["extraction_error"] = str(e)

        return solution

    def calculate_completeness_score(self, metadata: dict) -> float:
        """Calculate metadata completeness score.

        Args:
            metadata: Combined metadata dictionary

        Returns:
            float: Completeness score between 0.0 and 1.0
        """
        # Define essential metadata categories and weights
        categories = {
            "photometric": 0.25,
            "observing_conditions": 0.25,
            "instrument": 0.25,
            "astrometric": 0.25,
        }

        category_scores = {}

        # Score photometric completeness
        if "photometric" in metadata:
            phot = metadata["photometric"]
            essential_phot = ["filter_band", "exposure_time", "airmass"]
            present = sum(1 for key in essential_phot if phot.get(key) is not None)
            category_scores["photometric"] = present / len(essential_phot)

        # Score observing conditions completeness
        if "observing_conditions" in metadata:
            obs = metadata["observing_conditions"]
            essential_obs = ["seeing", "airmass", "observation_date"]
            present = sum(1 for key in essential_obs if obs.get(key) is not None)
            category_scores["observing_conditions"] = present / len(essential_obs)

        # Score instrument completeness
        if "instrument" in metadata:
            inst = metadata["instrument"]
            essential_inst = ["telescope", "instrument", "detector"]
            present = sum(1 for key in essential_inst if inst.get(key) is not None)
            category_scores["instrument"] = present / len(essential_inst)

        # Score astrometric completeness
        if "astrometric" in metadata:
            astrom = metadata["astrometric"]
            if astrom.get("wcs_present", False):
                category_scores["astrometric"] = 1.0
            else:
                category_scores["astrometric"] = 0.0

        # Calculate weighted average
        total_score = 0.0
        for category, weight in categories.items():
            total_score += category_scores.get(category, 0.0) * weight

        return total_score

    # Helper methods
    def _get_header_value(
        self, header: fits.Header, keywords: list, default=None, dtype=None
    ):
        """Get value from header using multiple possible keywords."""
        for keyword in keywords:
            if keyword in header:
                value = header[keyword]
                if value is not None and str(value).strip() != "":
                    try:
                        if dtype is not None:
                            return dtype(value)
                        return value
                    except (ValueError, TypeError):
                        continue
        return default

    def _calculate_pixel_scale_from_header(self, header: fits.Header) -> float | None:
        """Calculate pixel scale from header information."""
        try:
            # Try CD matrix first
            if "CD1_1" in header and "CD2_2" in header:
                cd11 = header["CD1_1"]
                cd22 = header["CD2_2"]
                pixel_scale = np.sqrt(abs(cd11 * cd22)) * 3600  # Convert to arcsec
                return float(pixel_scale)

            # Try CDELT values
            elif "CDELT1" in header and "CDELT2" in header:
                cdelt1 = header["CDELT1"]
                cdelt2 = header["CDELT2"]
                pixel_scale = np.sqrt(abs(cdelt1 * cdelt2)) * 3600  # Convert to arcsec
                return float(pixel_scale)

            # Try PIXSCALE or SECPIX keywords
            elif "PIXSCALE" in header:
                return float(header["PIXSCALE"])
            elif "SECPIX" in header:
                return float(header["SECPIX"])

        except Exception:
            pass

        return None

    def _extract_binning(self, header: fits.Header) -> dict | None:
        """Extract CCD binning information."""
        binning = {}

        # Try various binning keywords
        if "BINX" in header and "BINY" in header:
            binning["x"] = int(header["BINX"])
            binning["y"] = int(header["BINY"])
        elif "XBINNING" in header and "YBINNING" in header:
            binning["x"] = int(header["XBINNING"])
            binning["y"] = int(header["YBINNING"])
        elif "BINNING" in header:
            # Assume square binning
            bin_val = int(header["BINNING"])
            binning["x"] = bin_val
            binning["y"] = bin_val

        return binning if binning else None

    def _identify_instrument_config(self, telescope: str, instrument: str) -> str:
        """Identify instrument configuration type."""
        if not telescope or not instrument:
            return "generic"

        telescope_lower = telescope.lower()

        if "hst" in telescope_lower or "hubble" in telescope_lower:
            return "hst"
        elif "jwst" in telescope_lower or "webb" in telescope_lower:
            return "jwst"
        elif "spitzer" in telescope_lower:
            return "spitzer"
        else:
            return "generic"

    def _estimate_background_rms(self, image_data: np.ndarray) -> float:
        """Estimate background RMS using sigma-clipped statistics."""
        try:
            # Use central region to avoid edge effects
            h, w = image_data.shape
            central_region = image_data[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

            # Sigma clipping to remove sources
            mean = np.median(central_region)
            std = np.std(central_region)

            # 3-sigma clip
            mask = np.abs(central_region - mean) < 3 * std
            return float(np.std(central_region[mask]))
        except Exception:
            return float(np.std(image_data))

    def _extract_aperture_corrections(self, header: fits.Header) -> dict:
        """Extract aperture correction information."""
        corrections = {}

        # Look for aperture correction keywords
        for key in header.keys():
            if "APCOR" in key or "APERCOR" in key:
                corrections[key] = header[key]

        return corrections

    def _extract_color_terms(self, header: fits.Header) -> dict:
        """Extract color term coefficients."""
        color_terms = {}

        # Look for color term keywords
        for key in header.keys():
            if "COLOR" in key or "COLTERM" in key:
                color_terms[key] = header[key]

        return color_terms

    def _assess_photometric_quality(self, header: fits.Header) -> str | None:
        """Assess photometric quality based on header information."""
        # This is a simplified assessment
        if "MAGZPT" in header and "MAGZPTER" in header:
            zp_error = header["MAGZPTER"]
            if zp_error < 0.05:
                return "excellent"
            elif zp_error < 0.1:
                return "good"
            else:
                return "poor"
        return None

    def _assess_weather_quality(self, conditions: dict) -> str | None:
        """Assess weather quality based on observing conditions."""
        seeing = conditions.get("seeing")
        cloud_cover = conditions.get("cloud_cover")

        if seeing and seeing < 1.0 and (not cloud_cover or cloud_cover < 10):
            return "excellent"
        elif seeing and seeing < 1.5 and (not cloud_cover or cloud_cover < 30):
            return "good"
        elif seeing and seeing < 2.0:
            return "fair"
        else:
            return "poor"

    def _assess_atmospheric_transparency(self, conditions: dict) -> str | None:
        """Assess atmospheric transparency."""
        extinction = conditions.get("extinction_coefficient")
        if extinction:
            if extinction < 0.15:
                return "excellent"
            elif extinction < 0.25:
                return "good"
            else:
                return "poor"
        return None

    def _calculate_quality_score(self, metrics: dict) -> float:
        """Calculate overall quality score from metrics."""
        scores = []

        # Signal-to-noise contribution
        snr = metrics.get("signal_to_noise")
        if snr:
            scores.append(min(snr / 100, 1.0))  # Normalize to 1.0 at SNR=100

        # Background stability
        bg_rms = metrics.get("background_rms")
        bg_level = metrics.get("background_level")
        if bg_rms and bg_level and bg_level > 0:
            stability = 1.0 - min(bg_rms / bg_level, 1.0)
            scores.append(stability)

        # Saturation check
        sat_frac = metrics.get("saturation_fraction")
        if sat_frac is not None:
            scores.append(1.0 - min(sat_frac * 10, 1.0))  # Penalize saturation

        return np.mean(scores) if scores else 0.5

    def _calculate_wcs_pixel_scale(self, wcs: WCS) -> float:
        """Calculate pixel scale from WCS object."""
        try:
            if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
                cd_matrix = wcs.wcs.cd
                pixel_scale = np.sqrt(np.abs(np.linalg.det(cd_matrix))) * 3600
            elif hasattr(wcs.wcs, "cdelt") and wcs.wcs.cdelt is not None:
                pixel_scale = (
                    np.sqrt(np.abs(wcs.wcs.cdelt[0] * wcs.wcs.cdelt[1])) * 3600
                )
            else:
                # Fallback calculation
                pixel_scale = np.sqrt(
                    wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
                    * wcs.proj_plane_pixel_scales()[1].to(u.arcsec).value
                )
            return float(pixel_scale)
        except Exception:
            return None

    def _calculate_wcs_rotation(self, wcs: WCS) -> float:
        """Calculate rotation angle from WCS object."""
        try:
            if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
                cd_matrix = wcs.wcs.cd
                rotation = np.arctan2(cd_matrix[0, 1], cd_matrix[0, 0]) * 180 / np.pi
            else:
                rotation = 0.0  # Assume no rotation if CD matrix not available
            return float(rotation)
        except Exception:
            return None

    def _assess_astrometric_quality(
        self, solution: dict, header: fits.Header
    ) -> str | None:
        """Assess astrometric solution quality."""
        if not solution.get("wcs_present"):
            return "no_solution"

        residuals = solution.get("astrometric_residuals")
        if residuals:
            if residuals < 0.1:  # arcsec
                return "excellent"
            elif residuals < 0.5:
                return "good"
            elif residuals < 1.0:
                return "fair"
            else:
                return "poor"

        return "unknown"
