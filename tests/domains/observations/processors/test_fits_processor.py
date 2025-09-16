"""Tests for FITS processor."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from src.domains.observations.processors.fits_processor import FITSProcessor


class TestFITSProcessor:
    """Test cases for FITSProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FITSProcessor()

    def create_test_fits_file(self, temp_dir: str) -> str:
        """Create a test FITS file for testing."""
        # Create test image data
        data = np.random.random((100, 100)).astype(np.float32)

        # Create basic header
        header = fits.Header()
        header["SIMPLE"] = True
        header["BITPIX"] = -32
        header["NAXIS"] = 2
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 0.0
        header["CRPIX1"] = 50.0
        header["CRPIX2"] = 50.0
        header["CDELT1"] = -0.001
        header["CDELT2"] = 0.001
        header["EXPTIME"] = 60.0
        header["FILTER"] = "V"

        # Create FITS file
        hdu = fits.PrimaryHDU(data, header=header)
        fits_file = Path(temp_dir) / "test.fits"
        hdu.writeto(fits_file, overwrite=True)

        return str(fits_file)

    def test_validate_fits_structure_valid_file(self):
        """Test FITS structure validation with valid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fits_file = self.create_test_fits_file(temp_dir)

            result = self.processor.validate_fits_structure(fits_file)
            assert result is True

    def test_validate_fits_structure_nonexistent_file(self):
        """Test FITS structure validation with non-existent file."""
        result = self.processor.validate_fits_structure("/nonexistent/file.fits")
        assert result is False

    def test_read_fits_with_validation_valid_file(self):
        """Test reading FITS file with validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fits_file = self.create_test_fits_file(temp_dir)

            image_data, metadata = self.processor.read_fits_with_validation(fits_file)

            # Check image data
            assert image_data is not None
            assert image_data.shape == (100, 100)
            assert image_data.dtype == np.float32

            # Check metadata
            assert "primary_header" in metadata
            assert "file_info" in metadata
            assert "processing_info" in metadata

            # Check specific header values
            header = metadata["primary_header"]
            assert header["NAXIS1"] == 100
            assert header["NAXIS2"] == 100
            assert header["FILTER"] == "V"

    def test_read_fits_with_validation_invalid_file(self):
        """Test reading invalid FITS file."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            # Write invalid data
            tmp.write(b"This is not a FITS file")
            tmp.flush()

            with pytest.raises(ValueError):
                self.processor.read_fits_with_validation(tmp.name)

    def test_write_fits_with_metadata(self):
        """Test writing FITS file with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            data = np.random.random((50, 50)).astype(np.float32)
            headers = {"EXPTIME": 120.0, "FILTER": "R", "OBJECT": "Test Object"}

            output_file = Path(temp_dir) / "output.fits"

            # Write FITS file
            self.processor.write_fits_with_metadata(data, headers, str(output_file))

            # Verify file was created and read it back
            assert output_file.exists()

            with fits.open(output_file) as hdul:
                # Check data
                assert np.array_equal(hdul[0].data, data)

                # Check headers
                header = hdul[0].header
                assert header["EXPTIME"] == 120.0
                assert header["FILTER"] == "R"
                assert header["OBJECT"] == "Test Object"
                assert "ORIGIN" in header
                assert "DATE" in header

    def test_extract_all_headers(self):
        """Test extracting all headers from FITS file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fits_file = self.create_test_fits_file(temp_dir)

            headers = self.processor.extract_all_headers(fits_file)

            # Should have at least primary HDU
            assert len(headers) >= 1
            assert "HDU_0" in headers

            # Check primary HDU structure
            primary_hdu = headers["HDU_0"]
            assert "header" in primary_hdu
            assert "data_info" in primary_hdu
            assert primary_hdu["data_info"]["has_data"] is True
            assert primary_hdu["data_info"]["shape"] == (100, 100)

    def test_verify_fits_integrity(self):
        """Test FITS integrity verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fits_file = self.create_test_fits_file(temp_dir)

            results = self.processor.verify_fits_integrity(fits_file)

            # Check integrity results
            assert results["file_exists"] is True
            assert results["is_valid_fits"] is True
            assert results["structure_valid"] is True
            assert results["data_readable"] is True
            assert results["header_complete"] is True
            assert len(results["errors"]) == 0

    def test_optimize_fits_file(self):
        """Test FITS file optimization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = self.create_test_fits_file(temp_dir)
            output_file = Path(temp_dir) / "optimized.fits"

            # Optimize file
            results = self.processor.optimize_fits_file(
                str(input_file), str(output_file)
            )

            # Check results
            assert output_file.exists()
            assert "compression_ratio" in results
            assert "size_reduction_percent" in results
            assert results["compression_ratio"] > 0
            assert results["processing_time"] > 0

    def test_write_fits_with_compression(self):
        """Test writing compressed FITS file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
            headers = {"EXPTIME": 60.0}

            output_file = Path(temp_dir) / "compressed.fits"

            # Write compressed FITS file
            self.processor.write_fits_with_metadata(
                data, headers, str(output_file), compress=True
            )

            # Verify file was created
            assert output_file.exists()

            # Read back and verify
            with fits.open(output_file) as hdul:
                assert np.array_equal(hdul[0].data, data)
                assert hdul[0].header["EXPTIME"] == 60.0

    def test_write_fits_invalid_data(self):
        """Test writing FITS with invalid data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "invalid.fits"

            # Test with empty data
            with pytest.raises(ValueError):
                self.processor.write_fits_with_metadata(
                    np.array([]), {}, str(output_file)
                )

            # Test with None data
            with pytest.raises(ValueError):
                self.processor.write_fits_with_metadata(None, {}, str(output_file))
