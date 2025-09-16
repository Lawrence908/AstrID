"""Tests for metadata extraction functionality."""

import tempfile

import numpy as np
import pytest
from astropy.io import fits

from src.domains.observations.extractors.metadata_extractor import (
    FITSHeaderMetadata,
    MetadataExtractor,
    PhotometricParameters,
    QualityMetrics,
    WCSInformation,
)


class TestMetadataExtractor:
    """Test suite for metadata extractor."""

    @pytest.fixture
    def extractor(self):
        """Create metadata extractor for testing."""
        return MetadataExtractor()

    @pytest.fixture
    def sample_fits_data(self):
        """Create sample FITS data for testing."""
        # Create a simple FITS file in memory
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100
        header["BITPIX"] = -32
        header["CRVAL1"] = 210.8023
        header["CRVAL2"] = 54.3489
        header["CRPIX1"] = 50.0
        header["CRPIX2"] = 50.0
        header["CDELT1"] = -0.0001
        header["CDELT2"] = 0.0001
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["OBJECT"] = "NGC5194"
        header["TELESCOP"] = "HST"
        header["INSTRUME"] = "ACS"
        header["FILTER"] = "F606W"
        header["EXPTIME"] = 507.0
        header["DATE-OBS"] = "2020-05-15T12:30:45"
        header["MJD-OBS"] = 58849.52
        header["SEEING"] = 0.05
        header["AIRMASS"] = 1.0
        header["MAGZPT"] = 25.96
        header["SKYLEVEL"] = 123.4

        # Create data array
        data = np.random.normal(1000, 100, (100, 100)).astype(np.float32)

        # Create primary HDU
        hdu = fits.PrimaryHDU(data=data, header=header)

        # Write to bytes
        with tempfile.NamedTemporaryFile() as temp_file:
            hdu.writeto(temp_file, overwrite=True)
            temp_file.seek(0)
            fits_data = temp_file.read()

        return fits_data

    @pytest.fixture
    def invalid_fits_data(self):
        """Create invalid FITS data for testing."""
        return b"This is not a FITS file"

    @pytest.mark.asyncio
    async def test_extract_fits_headers_success(self, extractor, sample_fits_data):
        """Test successful FITS header extraction."""
        result = await extractor.extract_fits_headers(sample_fits_data)

        assert isinstance(result, dict)
        assert result["naxis1"] == 100
        assert result["naxis2"] == 100
        assert result["bitpix"] == -32
        assert result["crval1"] == 210.8023
        assert result["crval2"] == 54.3489
        assert result["object_name"] == "NGC5194"
        assert result["telescope"] == "HST"
        assert result["instrument"] == "ACS"
        assert result["filter_name"] == "F606W"
        assert result["exposure_time"] == 507.0
        assert result["seeing"] == 0.05
        assert result["airmass"] == 1.0
        assert result["zero_point"] == 25.96
        assert result["sky_level"] == 123.4
        assert "raw_header" in result
        assert "header_comments" in result

    @pytest.mark.asyncio
    async def test_extract_fits_headers_invalid_data(
        self, extractor, invalid_fits_data
    ):
        """Test FITS header extraction with invalid data."""
        with pytest.raises(ValueError, match="Invalid FITS data"):
            await extractor.extract_fits_headers(invalid_fits_data)

    @pytest.mark.asyncio
    async def test_extract_wcs_information_success(self, extractor, sample_fits_data):
        """Test successful WCS information extraction."""
        result = await extractor.extract_wcs_information(sample_fits_data)

        assert isinstance(result, dict)
        assert result["is_valid"] is True
        assert result["coordinate_system"] == "ICRS"
        assert result["projection"] == "TAN"
        assert result["reference_ra"] == 210.8023
        assert result["reference_dec"] == 54.3489
        assert result["pixel_scale_x"] is not None
        assert result["pixel_scale_y"] is not None
        assert result["field_of_view_x"] is not None
        assert result["field_of_view_y"] is not None

    @pytest.mark.asyncio
    async def test_extract_wcs_information_invalid_wcs(self, extractor):
        """Test WCS extraction with invalid WCS data."""
        # Create FITS with invalid WCS
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100
        # Missing critical WCS keywords

        data = np.zeros((100, 100), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data, header=header)

        with tempfile.NamedTemporaryFile() as temp_file:
            hdu.writeto(temp_file, overwrite=True)
            temp_file.seek(0)
            fits_data = temp_file.read()

        result = await extractor.extract_wcs_information(fits_data)
        assert result["is_valid"] is False

    @pytest.mark.asyncio
    async def test_extract_photometric_parameters_success(
        self, extractor, sample_fits_data
    ):
        """Test successful photometric parameter extraction."""
        result = await extractor.extract_photometric_parameters(sample_fits_data)

        assert isinstance(result, dict)
        assert result["zero_point"] == 25.96
        assert result["sky_background"] == 123.4
        assert result["zero_point_error"] is None  # Not provided in sample
        assert result["magnitude_limit"] is None  # Not provided in sample

    @pytest.mark.asyncio
    async def test_extract_quality_metrics_success(self, extractor, sample_fits_data):
        """Test successful quality metrics extraction."""
        result = await extractor.extract_quality_metrics(sample_fits_data)

        assert isinstance(result, dict)
        assert result["seeing_fwhm"] == 0.05
        assert result["airmass"] == 1.0
        assert result["ellipticity"] is None  # Not provided in sample
        assert result["cloud_cover"] is None  # Not provided in sample

    @pytest.mark.asyncio
    async def test_extract_all_metadata_success(self, extractor, sample_fits_data):
        """Test extraction of all metadata types."""
        result = await extractor.extract_all_metadata(sample_fits_data)

        assert isinstance(result, dict)
        assert "fits_headers" in result
        assert "wcs_information" in result
        assert "photometric_parameters" in result
        assert "quality_metrics" in result
        assert "extraction_summary" in result

        summary = result["extraction_summary"]
        assert summary["file_size_bytes"] == len(sample_fits_data)
        assert summary["has_valid_wcs"] is True
        assert summary["has_photometric_calibration"] is True
        assert summary["has_quality_info"] is True

    def test_validate_fits_file_valid(self, extractor, sample_fits_data):
        """Test FITS file validation with valid data."""
        is_valid = extractor.validate_fits_file(sample_fits_data)
        assert is_valid is True

    def test_validate_fits_file_invalid(self, extractor, invalid_fits_data):
        """Test FITS file validation with invalid data."""
        is_valid = extractor.validate_fits_file(invalid_fits_data)
        assert is_valid is False

    def test_validate_fits_file_too_short(self, extractor):
        """Test FITS file validation with too short data."""
        short_data = b"SIMPLE"  # Less than 80 bytes
        is_valid = extractor.validate_fits_file(short_data)
        assert is_valid is False

    def test_validate_fits_file_wrong_header(self, extractor):
        """Test FITS file validation with wrong header."""
        wrong_header = (
            b"COMPLEX =                    T" + b" " * 50
        )  # 80 bytes but wrong header
        is_valid = extractor.validate_fits_file(wrong_header)
        assert is_valid is False

    def test_get_supported_formats(self, extractor):
        """Test getting supported file formats."""
        formats = extractor.get_supported_formats()
        assert ".fits" in formats
        assert ".fit" in formats
        assert ".fts" in formats

    def test_estimate_processing_time(self, extractor):
        """Test processing time estimation."""
        # Small file
        small_time = extractor.estimate_processing_time(1024 * 1024)  # 1 MB
        assert small_time > 0.5
        assert small_time < 2.0

        # Large file
        large_time = extractor.estimate_processing_time(100 * 1024 * 1024)  # 100 MB
        assert large_time > small_time

    @pytest.mark.asyncio
    async def test_extract_with_missing_keywords(self, extractor):
        """Test extraction with missing optional keywords."""
        # Create minimal FITS file
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 50
        header["NAXIS2"] = 50
        # Only minimal required keywords

        data = np.zeros((50, 50), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data, header=header)

        with tempfile.NamedTemporaryFile() as temp_file:
            hdu.writeto(temp_file, overwrite=True)
            temp_file.seek(0)
            fits_data = temp_file.read()

        # Should not raise errors, but many fields will be None
        headers = await extractor.extract_fits_headers(fits_data)
        assert headers["naxis1"] == 50
        assert headers["naxis2"] == 50
        assert headers["object_name"] is None
        assert headers["telescope"] is None

    @pytest.mark.asyncio
    async def test_extract_with_galactic_coordinates(self, extractor):
        """Test extraction with Galactic coordinate system."""
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100
        header["CRVAL1"] = 120.0
        header["CRVAL2"] = 30.0
        header["CRPIX1"] = 50.0
        header["CRPIX2"] = 50.0
        header["CDELT1"] = -0.0001
        header["CDELT2"] = 0.0001
        header["CTYPE1"] = "GLON-TAN"
        header["CTYPE2"] = "GLAT-TAN"

        data = np.zeros((100, 100), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data, header=header)

        with tempfile.NamedTemporaryFile() as temp_file:
            hdu.writeto(temp_file, overwrite=True)
            temp_file.seek(0)
            fits_data = temp_file.read()

        wcs_info = await extractor.extract_wcs_information(fits_data)
        assert wcs_info["coordinate_system"] == "Galactic"
        assert wcs_info["projection"] == "TAN"

    @pytest.mark.asyncio
    async def test_extract_with_sin_projection(self, extractor):
        """Test extraction with SIN projection."""
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 0.0
        header["CRPIX1"] = 50.0
        header["CRPIX2"] = 50.0
        header["CDELT1"] = -0.0001
        header["CDELT2"] = 0.0001
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"

        data = np.zeros((100, 100), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data, header=header)

        with tempfile.NamedTemporaryFile() as temp_file:
            hdu.writeto(temp_file, overwrite=True)
            temp_file.seek(0)
            fits_data = temp_file.read()

        wcs_info = await extractor.extract_wcs_information(fits_data)
        assert wcs_info["projection"] == "SIN"

    @pytest.mark.asyncio
    async def test_dynamic_range_calculation(self, extractor):
        """Test dynamic range calculation."""
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 10
        header["NAXIS2"] = 10
        header["SATURATE"] = 65000

        # Create data with known range
        data = np.full((10, 10), 100.0, dtype=np.float32)
        data[0, 0] = 10.0  # Minimum value
        hdu = fits.PrimaryHDU(data=data, header=header)

        with tempfile.NamedTemporaryFile() as temp_file:
            hdu.writeto(temp_file, overwrite=True)
            temp_file.seek(0)
            fits_data = temp_file.read()

        photometric = await extractor.extract_photometric_parameters(fits_data)
        assert photometric["saturation_level"] == 65000
        assert photometric["dynamic_range"] is not None
        assert photometric["dynamic_range"] > 1000  # 65000 / ~10


class TestMetadataModels:
    """Test the metadata model classes."""

    def test_fits_header_metadata_creation(self):
        """Test creation of FITSHeaderMetadata."""
        metadata = FITSHeaderMetadata(
            naxis1=100,
            naxis2=100,
            crval1=210.8,
            crval2=54.3,
            object_name="NGC5194",
            telescope="HST",
        )

        assert metadata.naxis1 == 100
        assert metadata.naxis2 == 100
        assert metadata.object_name == "NGC5194"
        assert metadata.telescope == "HST"

    def test_wcs_information_creation(self):
        """Test creation of WCSInformation."""
        wcs = WCSInformation(
            is_valid=True,
            coordinate_system="ICRS",
            reference_ra=210.8,
            reference_dec=54.3,
        )

        assert wcs.is_valid is True
        assert wcs.coordinate_system == "ICRS"
        assert wcs.reference_ra == 210.8

    def test_photometric_parameters_creation(self):
        """Test creation of PhotometricParameters."""
        params = PhotometricParameters(
            zero_point=25.96, sky_background=123.4, magnitude_limit=24.5
        )

        assert params.zero_point == 25.96
        assert params.sky_background == 123.4
        assert params.magnitude_limit == 24.5

    def test_quality_metrics_creation(self):
        """Test creation of QualityMetrics."""
        metrics = QualityMetrics(seeing_fwhm=0.8, airmass=1.2, cloud_cover=0.1)

        assert metrics.seeing_fwhm == 0.8
        assert metrics.airmass == 1.2
        assert metrics.cloud_cover == 0.1
