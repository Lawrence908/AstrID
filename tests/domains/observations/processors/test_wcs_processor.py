"""Tests for WCS processor."""

import numpy as np
import pytest
from astropy.wcs import WCS

from src.domains.observations.processors.wcs_processor import WCSProcessor


class TestWCSProcessor:
    """Test cases for WCSProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = WCSProcessor()

    def create_test_wcs(self) -> WCS:
        """Create a test WCS object."""
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [180.0, 0.0]  # RA, Dec in degrees
        wcs.wcs.crpix = [50.0, 50.0]  # Reference pixel
        wcs.wcs.cdelt = [-0.001, 0.001]  # Pixel scale in degrees
        wcs.wcs.cunit = ["deg", "deg"]
        return wcs

    def test_validate_wcs_solution_valid(self):
        """Test WCS validation with valid WCS."""
        wcs = self.create_test_wcs()

        result = self.processor.validate_wcs_solution(wcs)
        assert result is True

    def test_validate_wcs_solution_invalid(self):
        """Test WCS validation with invalid WCS."""
        # Test with None
        result = self.processor.validate_wcs_solution(None)
        assert result is False

        # Test with incomplete WCS
        wcs = WCS(naxis=1)  # Only 1 dimension
        result = self.processor.validate_wcs_solution(wcs)
        assert result is False

    def test_pixel_to_world_coordinates(self):
        """Test pixel to world coordinate conversion."""
        wcs = self.create_test_wcs()

        # Test single coordinate
        pixel_coords = np.array([[50.0, 50.0]])  # Reference pixel
        ra, dec = self.processor.pixel_to_world_coordinates(pixel_coords, wcs)

        assert len(ra) == 1
        assert len(dec) == 1
        assert abs(ra[0] - 180.0) < 1e-6  # Should be close to reference RA
        assert abs(dec[0] - 0.0) < 1e-6  # Should be close to reference Dec

    def test_world_to_pixel_coordinates(self):
        """Test world to pixel coordinate conversion."""
        wcs = self.create_test_wcs()

        # Test conversion back to pixels
        world_coords = (np.array([180.0]), np.array([0.0]))
        pixel_coords = self.processor.world_to_pixel_coordinates(world_coords, wcs)

        assert pixel_coords.shape == (1, 2)
        assert (
            abs(pixel_coords[0, 0] - 50.0) < 1e-6
        )  # Should be close to reference pixel
        assert abs(pixel_coords[0, 1] - 50.0) < 1e-6

    def test_pixel_to_world_multiple_coordinates(self):
        """Test pixel to world conversion with multiple coordinates."""
        wcs = self.create_test_wcs()

        # Test multiple coordinates
        pixel_coords = np.array(
            [
                [50.0, 50.0],  # Reference pixel
                [60.0, 60.0],  # Offset pixel
                [40.0, 40.0],  # Another offset
            ]
        )

        ra, dec = self.processor.pixel_to_world_coordinates(pixel_coords, wcs)

        assert len(ra) == 3
        assert len(dec) == 3
        # First coordinate should be close to reference
        assert abs(ra[0] - 180.0) < 1e-6
        assert abs(dec[0] - 0.0) < 1e-6

    def test_calculate_sky_region_bounds(self):
        """Test sky region bounds calculation."""
        wcs = self.create_test_wcs()
        image_shape = (100, 100)

        bounds = self.processor.calculate_sky_region_bounds(wcs, image_shape)

        # Check that bounds dictionary has expected keys
        expected_keys = [
            "ra_min",
            "ra_max",
            "dec_min",
            "dec_max",
            "center_ra",
            "center_dec",
            "width_deg",
            "height_deg",
            "area_sq_deg",
            "pixel_scale_arcsec",
            "image_shape",
            "corners_ra",
            "corners_dec",
        ]

        for key in expected_keys:
            assert key in bounds

        # Check reasonable values
        assert bounds["image_shape"] == image_shape
        assert bounds["ra_min"] < bounds["ra_max"]
        assert bounds["dec_min"] < bounds["dec_max"]
        assert bounds["width_deg"] > 0
        assert bounds["height_deg"] > 0
        assert bounds["area_sq_deg"] > 0

    def test_transform_coordinates_same_system(self):
        """Test coordinate transformation within same system."""
        coords = (180.0, 0.0)

        result = self.processor.transform_coordinates(coords, "icrs", "icrs")

        # Should return same coordinates
        assert result == coords

    def test_transform_coordinates_different_systems(self):
        """Test coordinate transformation between different systems."""
        coords = (180.0, 0.0)  # RA, Dec in ICRS

        # Transform from ICRS to Galactic
        galactic_coords = self.processor.transform_coordinates(
            coords, "icrs", "galactic"
        )

        # Should get different coordinates
        assert galactic_coords != coords
        assert len(galactic_coords) == 2
        assert isinstance(galactic_coords[0], float)
        assert isinstance(galactic_coords[1], float)

    def test_transform_coordinates_invalid_system(self):
        """Test coordinate transformation with invalid system."""
        coords = (180.0, 0.0)

        with pytest.raises(ValueError):
            self.processor.transform_coordinates(coords, "invalid", "icrs")

        with pytest.raises(ValueError):
            self.processor.transform_coordinates(coords, "icrs", "invalid")

    def test_assess_wcs_quality(self):
        """Test WCS quality assessment."""
        wcs = self.create_test_wcs()
        image_shape = (100, 100)

        quality = self.processor.assess_wcs_quality(wcs, image_shape)

        # Check that quality dictionary has expected structure
        expected_keys = [
            "is_valid",
            "completeness_score",
            "accuracy_score",
            "reliability_score",
            "overall_score",
            "issues",
            "recommendations",
        ]

        for key in expected_keys:
            assert key in quality

        # Valid WCS should have good scores
        assert quality["is_valid"] is True
        assert 0 <= quality["completeness_score"] <= 1
        assert 0 <= quality["accuracy_score"] <= 1
        assert 0 <= quality["reliability_score"] <= 1
        assert 0 <= quality["overall_score"] <= 1

    def test_assess_wcs_quality_invalid_wcs(self):
        """Test WCS quality assessment with invalid WCS."""
        wcs = WCS(naxis=1)  # Invalid WCS

        quality = self.processor.assess_wcs_quality(wcs)

        assert quality["is_valid"] is False
        assert quality["overall_score"] == 0.0
        assert len(quality["issues"]) > 0

    def test_pixel_to_world_empty_coordinates(self):
        """Test pixel to world conversion with empty coordinates."""
        wcs = self.create_test_wcs()
        pixel_coords = np.array([]).reshape(0, 2)

        with pytest.raises(ValueError):
            self.processor.pixel_to_world_coordinates(pixel_coords, wcs)

    def test_world_to_pixel_mismatched_arrays(self):
        """Test world to pixel conversion with mismatched array lengths."""
        wcs = self.create_test_wcs()

        # Mismatched RA and Dec arrays
        world_coords = (np.array([180.0, 181.0]), np.array([0.0]))

        with pytest.raises(ValueError):
            self.processor.world_to_pixel_coordinates(world_coords, wcs)
