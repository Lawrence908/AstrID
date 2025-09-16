"""Tests for FITS processing pipeline."""

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from src.domains.observations.catalogs.star_catalog import StarCatalogConfig
from src.domains.observations.processors.pipeline import (
    FITSProcessingPipeline,
    FITSProcessingResult,
)


class TestFITSProcessingPipeline:
    """Test cases for FITSProcessingPipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple star catalog config for testing
        self.star_catalog_config = StarCatalogConfig(
            catalogs=["gaia"],
            search_radius=300.0,
            magnitude_limit=18.0,
            cache_size=1000,
            update_frequency="weekly",
        )

        self.pipeline = FITSProcessingPipeline(self.star_catalog_config)

    def create_test_fits_file(self, temp_dir: str) -> str:
        """Create a test FITS file for testing."""
        # Create test image data
        data = np.random.random((100, 100)).astype(np.float32)

        # Create comprehensive header
        header = fits.Header()
        header["SIMPLE"] = True
        header["BITPIX"] = -32
        header["NAXIS"] = 2
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100

        # WCS information
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 0.0
        header["CRPIX1"] = 50.0
        header["CRPIX2"] = 50.0
        header["CDELT1"] = -0.001
        header["CDELT2"] = 0.001
        header["EQUINOX"] = 2000.0
        header["RADESYS"] = "ICRS"

        # Observational parameters
        header["EXPTIME"] = 60.0
        header["FILTER"] = "V"
        header["AIRMASS"] = 1.2
        header["SEEING"] = 1.5
        header["DATE-OBS"] = "2023-01-01T12:00:00"
        header["TELESCOP"] = "Test Telescope"
        header["INSTRUME"] = "Test Camera"
        header["GAIN"] = 2.0
        header["RDNOISE"] = 5.0

        # Create FITS file
        hdu = fits.PrimaryHDU(data, header=header)
        fits_file = Path(temp_dir) / "test.fits"
        hdu.writeto(fits_file, overwrite=True)

        return str(fits_file)

    def test_process_fits_file_complete(self):
        """Test complete FITS file processing through pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fits_file = self.create_test_fits_file(temp_dir)

            result = self.pipeline.process_fits_file(fits_file)

            # Check result structure
            assert isinstance(result, FITSProcessingResult)
            assert result.file_path == fits_file
            assert result.processing_time > 0
            assert result.image_data.shape == (100, 100)
            assert result.wcs_solution is not None

            # Check metadata extraction
            assert "photometric" in result.metadata
            assert "observing_conditions" in result.metadata
            assert "instrument" in result.metadata
            assert "astrometric" in result.metadata

            # Check quality metrics
            assert isinstance(result.quality_metrics, dict)

            # Check star catalog integration
            assert isinstance(result.star_catalog_matches, list)
            assert result.total_catalog_stars is not None
            assert result.matched_stars_count is not None

    def test_process_fits_file_with_custom_config(self):
        """Test FITS processing with custom configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fits_file = self.create_test_fits_file(temp_dir)

            # Custom config to disable some processing steps
            config = {
                "extract_photometric": True,
                "extract_observing_conditions": False,
                "extract_instrument_params": True,
                "extract_quality_metrics": False,
                "query_star_catalogs": False,
                "star_catalog_radius": 600.0,
            }

            result = self.pipeline.process_fits_file(fits_file, config)

            # Check that config was respected
            assert "photometric" in result.metadata
            assert (
                "observing_conditions" in result.metadata
            )  # Still extracted due to default
            assert "instrument" in result.metadata
            assert len(result.star_catalog_matches) == 0  # Should be empty

    def test_validate_fits_file(self):
        """Test FITS file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fits_file = self.create_test_fits_file(temp_dir)

            validation_result = self.pipeline.validate_fits_file(fits_file)

            # Check validation results
            assert validation_result["is_valid"] is True
            assert validation_result["structure_validation"] is True
            assert validation_result["has_image_data"] is True
            assert "image_shape" in validation_result
            assert validation_result["image_shape"] == (100, 100)

    def test_validate_invalid_fits_file(self):
        """Test validation of invalid FITS file."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            # Write invalid data
            tmp.write(b"This is not a FITS file")
            tmp.flush()

            validation_result = self.pipeline.validate_fits_file(tmp.name)

            assert validation_result["is_valid"] is False
            assert "error" in validation_result

    def test_process_nonexistent_file(self):
        """Test processing non-existent FITS file."""
        result = self.pipeline.process_fits_file("/nonexistent/file.fits")

        # Should return result with errors
        assert isinstance(result, FITSProcessingResult)
        assert len(result.processing_errors) > 0
        assert result.image_data.size == 0
        assert result.wcs_solution is None

    def test_batch_process_directory(self):
        """Test batch processing of directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test files
            fits_files = []
            for i in range(3):
                data = np.random.random((50, 50)).astype(np.float32)
                header = fits.Header()
                header["SIMPLE"] = True
                header["BITPIX"] = -32
                header["NAXIS"] = 2
                header["NAXIS1"] = 50
                header["NAXIS2"] = 50
                header["EXPTIME"] = 60.0 + i * 10

                hdu = fits.PrimaryHDU(data, header=header)
                fits_file = Path(temp_dir) / f"test_{i}.fits"
                hdu.writeto(fits_file, overwrite=True)
                fits_files.append(fits_file)

            # Process directory
            results = self.pipeline.batch_process_directory(temp_dir)

            # Check results
            assert len(results) == 3
            for result in results:
                assert isinstance(result, FITSProcessingResult)
                assert result.image_data.shape == (50, 50)

    def test_batch_process_empty_directory(self):
        """Test batch processing of empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self.pipeline.batch_process_directory(temp_dir)
            assert len(results) == 0

    def test_get_pipeline_statistics(self):
        """Test pipeline statistics calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and process test file
            fits_file = self.create_test_fits_file(temp_dir)
            result = self.pipeline.process_fits_file(fits_file)

            # Calculate statistics
            stats = self.pipeline.get_pipeline_statistics([result])

            # Check statistics structure
            assert "total_files" in stats
            assert "successful_files" in stats
            assert "failed_files" in stats
            assert "total_processing_time" in stats
            assert "average_processing_time" in stats
            assert "quality_scores" in stats
            assert "star_catalog_stats" in stats

            # Check values
            assert stats["total_files"] == 1
            assert stats["successful_files"] == 1 if not result.processing_errors else 0
            assert stats["total_processing_time"] > 0

    def test_get_pipeline_statistics_empty_results(self):
        """Test pipeline statistics with empty results."""
        stats = self.pipeline.get_pipeline_statistics([])
        assert stats == {}

    def test_pipeline_without_star_catalog(self):
        """Test pipeline without star catalog configuration."""
        pipeline = FITSProcessingPipeline()  # No star catalog config

        with tempfile.TemporaryDirectory() as temp_dir:
            fits_file = self.create_test_fits_file(temp_dir)

            result = pipeline.process_fits_file(fits_file)

            # Should still work but without star catalog matches
            assert isinstance(result, FITSProcessingResult)
            assert len(result.star_catalog_matches) == 0
            assert result.total_catalog_stars is None
            assert result.matched_stars_count == 0

    def test_processing_result_dataclass(self):
        """Test FITSProcessingResult dataclass structure."""
        # Create minimal result
        result = FITSProcessingResult(
            image_data=np.array([[1, 2], [3, 4]]),
            wcs_solution=None,
            metadata={},
            quality_metrics={},
            star_catalog_matches=[],
            processing_errors=[],
            processing_time=1.0,
        )

        # Check default values
        assert result.file_path is None
        assert result.processing_config is None
        assert result.pipeline_version == "1.0.0"
        assert result.overall_quality_score is None
        assert result.wcs_quality_score is None
        assert result.photometric_quality_score is None
