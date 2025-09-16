"""Metadata extraction from astronomical observation data and FITS files."""

import tempfile
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from pydantic import BaseModel

from src.core.logging import configure_domain_logger


class FITSHeaderMetadata(BaseModel):
    """Structured metadata extracted from FITS headers."""

    # Basic image information
    naxis1: int | None = None
    naxis2: int | None = None
    bitpix: int | None = None

    # WCS information
    crval1: float | None = None
    crval2: float | None = None
    crpix1: float | None = None
    crpix2: float | None = None
    cdelt1: float | None = None
    cdelt2: float | None = None
    cd1_1: float | None = None
    cd1_2: float | None = None
    cd2_1: float | None = None
    cd2_2: float | None = None
    ctype1: str | None = None
    ctype2: str | None = None

    # Observation information
    object_name: str | None = None
    telescope: str | None = None
    instrument: str | None = None
    filter_name: str | None = None
    exposure_time: float | None = None
    date_obs: str | None = None
    mjd_obs: float | None = None

    # Image quality
    seeing: float | None = None
    airmass: float | None = None
    sky_level: float | None = None

    # Photometric calibration
    zero_point: float | None = None
    mag_limit: float | None = None


class WCSInformation(BaseModel):
    """World Coordinate System information."""

    is_valid: bool
    coordinate_system: str | None = None
    projection: str | None = None
    reference_ra: float | None = None
    reference_dec: float | None = None
    pixel_scale_x: float | None = None
    pixel_scale_y: float | None = None
    rotation_angle: float | None = None
    field_of_view_x: float | None = None
    field_of_view_y: float | None = None


class PhotometricParameters(BaseModel):
    """Photometric calibration and quality parameters."""

    zero_point: float | None = None
    zero_point_error: float | None = None
    magnitude_limit: float | None = None
    sky_background: float | None = None
    sky_background_rms: float | None = None
    saturation_level: float | None = None
    dynamic_range: float | None = None


class QualityMetrics(BaseModel):
    """Image quality metrics."""

    seeing_fwhm: float | None = None
    ellipticity: float | None = None
    airmass: float | None = None
    cloud_cover: float | None = None
    moon_illumination: float | None = None
    moon_separation: float | None = None
    stellar_density: int | None = None
    cosmic_ray_count: int | None = None


class MetadataExtractor:
    """Extract metadata from astronomical observation data and FITS files."""

    def __init__(self):
        """Initialize metadata extractor."""
        self.logger = configure_domain_logger("observations.extractors.metadata")

    async def extract_fits_headers(self, fits_data: bytes) -> dict[str, Any]:
        """Extract FITS header information.

        Args:
            fits_data: Raw FITS file data

        Returns:
            Dictionary containing extracted header information

        Raises:
            ValueError: For invalid FITS data
        """
        self.logger.debug("Extracting FITS header metadata")

        try:
            # Write to temporary file for astropy processing
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_file:
                temp_file.write(fits_data)
                temp_path = temp_file.name

            try:
                # Open FITS file and extract headers
                with fits.open(temp_path) as hdul:
                    # Get primary header
                    primary_hdu = hdul[0]
                    header = primary_hdu.header

                    # Extract structured metadata
                    metadata = FITSHeaderMetadata(
                        # Image dimensions
                        naxis1=header.get("NAXIS1"),
                        naxis2=header.get("NAXIS2"),
                        bitpix=header.get("BITPIX"),
                        # WCS keywords
                        crval1=header.get("CRVAL1"),
                        crval2=header.get("CRVAL2"),
                        crpix1=header.get("CRPIX1"),
                        crpix2=header.get("CRPIX2"),
                        cdelt1=header.get("CDELT1"),
                        cdelt2=header.get("CDELT2"),
                        cd1_1=header.get("CD1_1"),
                        cd1_2=header.get("CD1_2"),
                        cd2_1=header.get("CD2_1"),
                        cd2_2=header.get("CD2_2"),
                        ctype1=header.get("CTYPE1"),
                        ctype2=header.get("CTYPE2"),
                        # Observation information
                        object_name=header.get("OBJECT"),
                        telescope=header.get("TELESCOP"),
                        instrument=header.get("INSTRUME"),
                        filter_name=header.get("FILTER") or header.get("FILTNAM1"),
                        exposure_time=header.get("EXPTIME"),
                        date_obs=header.get("DATE-OBS"),
                        mjd_obs=header.get("MJD-OBS"),
                        # Image quality
                        seeing=header.get("SEEING") or header.get("FWHM"),
                        airmass=header.get("AIRMASS"),
                        sky_level=header.get("SKYLEVEL") or header.get("SKYADU"),
                        # Photometry
                        zero_point=header.get("MAGZPT") or header.get("ZEROPNT"),
                        mag_limit=header.get("MAGLIMIT") or header.get("LIMMAG"),
                    )

                    # Convert to dictionary and add raw header
                    result = metadata.model_dump()
                    result["raw_header"] = dict(header)
                    result["header_comments"] = dict(header.comments)

                    self.logger.debug(
                        f"Extracted metadata from FITS with {len(header)} header cards"
                    )
                    return result

            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.error(f"Failed to extract FITS headers: {e}")
            raise ValueError(f"Invalid FITS data: {e}") from e

    async def extract_wcs_information(self, fits_data: bytes) -> dict[str, Any]:
        """Extract World Coordinate System information.

        Args:
            fits_data: Raw FITS file data

        Returns:
            Dictionary containing WCS information

        Raises:
            ValueError: For invalid FITS data or WCS
        """
        self.logger.debug("Extracting WCS information")

        try:
            # Write to temporary file for astropy processing
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_file:
                temp_file.write(fits_data)
                temp_path = temp_file.name

            try:
                # Open FITS file and extract WCS
                with fits.open(temp_path) as hdul:
                    primary_hdu = hdul[0]
                    header = primary_hdu.header

                    try:
                        wcs = WCS(header)

                        # Check if WCS is valid
                        if not wcs.has_celestial:
                            wcs_info = WCSInformation(is_valid=False)
                        else:
                            # Extract pixel scales
                            try:
                                pixel_scales = proj_plane_pixel_scales(wcs)
                                pixel_scale_x = (
                                    float(pixel_scales[0]) * 3600
                                )  # Convert to arcsec
                                pixel_scale_y = (
                                    float(pixel_scales[1]) * 3600
                                )  # Convert to arcsec
                            except Exception:
                                pixel_scale_x = pixel_scale_y = None

                            # Calculate field of view
                            naxis1 = header.get("NAXIS1", 0)
                            naxis2 = header.get("NAXIS2", 0)
                            fov_x = (
                                (pixel_scale_x * naxis1 / 3600)
                                if pixel_scale_x and naxis1
                                else None
                            )
                            fov_y = (
                                (pixel_scale_y * naxis2 / 3600)
                                if pixel_scale_y and naxis2
                                else None
                            )

                            # Extract coordinate system and projection
                            ctype1 = header.get("CTYPE1", "")
                            coordinate_system = "ICRS"  # Default
                            if "GLON" in ctype1:
                                coordinate_system = "Galactic"
                            elif "ELON" in ctype1:
                                coordinate_system = "Ecliptic"

                            projection = "TAN"  # Default
                            if "SIN" in ctype1:
                                projection = "SIN"
                            elif "ARC" in ctype1:
                                projection = "ARC"
                            elif "STG" in ctype1:
                                projection = "STG"

                            wcs_info = WCSInformation(
                                is_valid=True,
                                coordinate_system=coordinate_system,
                                projection=projection,
                                reference_ra=wcs.wcs.crval[0]
                                if len(wcs.wcs.crval) > 0
                                else None,
                                reference_dec=wcs.wcs.crval[1]
                                if len(wcs.wcs.crval) > 1
                                else None,
                                pixel_scale_x=pixel_scale_x,
                                pixel_scale_y=pixel_scale_y,
                                rotation_angle=None,  # Could be calculated from CD matrix
                                field_of_view_x=fov_x,
                                field_of_view_y=fov_y,
                            )

                    except Exception as e:
                        self.logger.warning(f"Failed to parse WCS: {e}")
                        wcs_info = WCSInformation(is_valid=False)

                    result = wcs_info.model_dump()
                    self.logger.debug(
                        f"Extracted WCS information: valid={wcs_info.is_valid}"
                    )
                    return result

            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.error(f"Failed to extract WCS information: {e}")
            raise ValueError(f"WCS extraction failed: {e}") from e

    async def extract_photometric_parameters(self, fits_data: bytes) -> dict[str, Any]:
        """Extract photometric calibration parameters.

        Args:
            fits_data: Raw FITS file data

        Returns:
            Dictionary containing photometric parameters

        Raises:
            ValueError: For invalid FITS data
        """
        self.logger.debug("Extracting photometric parameters")

        try:
            # Write to temporary file for astropy processing
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_file:
                temp_file.write(fits_data)
                temp_path = temp_file.name

            try:
                # Open FITS file and extract photometric info
                with fits.open(temp_path) as hdul:
                    primary_hdu = hdul[0]
                    header = primary_hdu.header
                    data = primary_hdu.data

                    # Extract photometric keywords
                    zero_point = (
                        header.get("MAGZPT")
                        or header.get("ZEROPNT")
                        or header.get("PHOTZP")
                    )
                    zero_point_error = header.get("MAGZPTER") or header.get("ZPTERR")
                    magnitude_limit = (
                        header.get("MAGLIMIT")
                        or header.get("LIMMAG")
                        or header.get("MAGLIM")
                    )

                    # Sky background
                    sky_background = (
                        header.get("SKYLEVEL")
                        or header.get("SKYADU")
                        or header.get("BACKGROUND")
                    )
                    sky_background_rms = header.get("SKYSIGMA") or header.get("SKYRMS")

                    # Saturation and dynamic range
                    saturation_level = header.get("SATURATE") or header.get("SATLEVEL")

                    # Calculate dynamic range if we have data
                    dynamic_range = None
                    if data is not None and saturation_level:
                        try:
                            import numpy as np

                            data_min = np.min(
                                data[data > 0]
                            )  # Avoid zero or negative values
                            dynamic_range = float(saturation_level) / float(data_min)
                        except Exception:
                            pass

                    photometric_params = PhotometricParameters(
                        zero_point=zero_point,
                        zero_point_error=zero_point_error,
                        magnitude_limit=magnitude_limit,
                        sky_background=sky_background,
                        sky_background_rms=sky_background_rms,
                        saturation_level=saturation_level,
                        dynamic_range=dynamic_range,
                    )

                    result = photometric_params.model_dump()
                    self.logger.debug("Extracted photometric parameters")
                    return result

            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.error(f"Failed to extract photometric parameters: {e}")
            raise ValueError(f"Photometric extraction failed: {e}") from e

    async def extract_quality_metrics(self, fits_data: bytes) -> dict[str, Any]:
        """Extract image quality metrics.

        Args:
            fits_data: Raw FITS file data

        Returns:
            Dictionary containing quality metrics

        Raises:
            ValueError: For invalid FITS data
        """
        self.logger.debug("Extracting quality metrics")

        try:
            # Write to temporary file for astropy processing
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_file:
                temp_file.write(fits_data)
                temp_path = temp_file.name

            try:
                # Open FITS file and extract quality info
                with fits.open(temp_path) as hdul:
                    primary_hdu = hdul[0]
                    header = primary_hdu.header

                    # Image quality metrics
                    seeing_fwhm = (
                        header.get("SEEING")
                        or header.get("FWHM")
                        or header.get("SEEPIX")
                    )
                    ellipticity = header.get("ELLIPTIC") or header.get("ELLIP")
                    airmass = header.get("AIRMASS")

                    # Observing conditions
                    cloud_cover = header.get("CLOUDCOV") or header.get("CLOUDS")
                    moon_illumination = header.get("MOONILLM") or header.get("MOONPHAS")
                    moon_separation = header.get("MOONSEP") or header.get("MOONDIST")

                    # Source counts
                    stellar_density = header.get("STARDENS") or header.get("NSTARS")
                    cosmic_ray_count = header.get("NCOSMIC") or header.get("CRCOUNT")

                    quality_metrics = QualityMetrics(
                        seeing_fwhm=seeing_fwhm,
                        ellipticity=ellipticity,
                        airmass=airmass,
                        cloud_cover=cloud_cover,
                        moon_illumination=moon_illumination,
                        moon_separation=moon_separation,
                        stellar_density=stellar_density,
                        cosmic_ray_count=cosmic_ray_count,
                    )

                    result = quality_metrics.model_dump()
                    self.logger.debug("Extracted quality metrics")
                    return result

            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.error(f"Failed to extract quality metrics: {e}")
            raise ValueError(f"Quality metrics extraction failed: {e}") from e

    async def extract_all_metadata(self, fits_data: bytes) -> dict[str, Any]:
        """Extract all available metadata from FITS file.

        Args:
            fits_data: Raw FITS file data

        Returns:
            Dictionary containing all extracted metadata

        Raises:
            ValueError: For invalid FITS data
        """
        self.logger.info("Extracting all metadata from FITS file")

        try:
            # Extract all metadata types
            fits_headers = await self.extract_fits_headers(fits_data)
            wcs_info = await self.extract_wcs_information(fits_data)
            photometric_params = await self.extract_photometric_parameters(fits_data)
            quality_metrics = await self.extract_quality_metrics(fits_data)

            # Combine all metadata
            all_metadata = {
                "fits_headers": fits_headers,
                "wcs_information": wcs_info,
                "photometric_parameters": photometric_params,
                "quality_metrics": quality_metrics,
                "extraction_summary": {
                    "file_size_bytes": len(fits_data),
                    "has_valid_wcs": wcs_info.get("is_valid", False),
                    "has_photometric_calibration": photometric_params.get("zero_point")
                    is not None,
                    "has_quality_info": quality_metrics.get("seeing_fwhm") is not None,
                },
            }

            self.logger.info("Successfully extracted all metadata")
            return all_metadata

        except Exception as e:
            self.logger.error(f"Failed to extract metadata: {e}")
            raise

    def validate_fits_file(self, fits_data: bytes) -> bool:
        """Validate if data is a valid FITS file.

        Args:
            fits_data: Raw file data to validate

        Returns:
            True if valid FITS file
        """
        try:
            # Check FITS magic number
            if len(fits_data) < 80:
                return False

            # FITS files start with "SIMPLE  = "
            header_start = fits_data[:80].decode("ascii", errors="ignore")
            if not header_start.startswith("SIMPLE  ="):
                return False

            # Try to open with astropy
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_file:
                temp_file.write(fits_data)
                temp_path = temp_file.name

            try:
                with fits.open(temp_path) as hdul:
                    # If we can read the headers, it's likely valid
                    primary_hdu = hdul[0]
                    _ = primary_hdu.header
                    return True
            finally:
                Path(temp_path).unlink(missing_ok=True)

        except Exception:
            return False

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats.

        Returns:
            List of supported format extensions
        """
        return [".fits", ".fit", ".fts"]

    def estimate_processing_time(self, file_size_bytes: int) -> float:
        """Estimate processing time for metadata extraction.

        Args:
            file_size_bytes: Size of FITS file in bytes

        Returns:
            Estimated processing time in seconds
        """
        # Rough estimate: ~1 second per 10 MB
        base_time = 0.5  # Base overhead
        size_factor = file_size_bytes / (10 * 1024 * 1024)  # 10 MB chunks
        return base_time + (size_factor * 1.0)
