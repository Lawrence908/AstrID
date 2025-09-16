"""Tests for star catalog integration."""

import tempfile

import numpy as np
import pytest
from astropy.wcs import WCS

from src.domains.observations.catalogs.star_catalog import (
    StarCatalog,
    StarCatalogConfig,
)


class TestStarCatalog:
    """Test cases for StarCatalog."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = StarCatalogConfig(
            catalogs=["gaia", "tycho2"],
            search_radius=300.0,
            magnitude_limit=18.0,
            cache_size=1000,
            update_frequency="weekly",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            self.config.cache_directory = temp_dir
            self.catalog = StarCatalog(self.config)

    def create_test_wcs(self) -> WCS:
        """Create a test WCS object."""
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [180.0, 0.0]
        wcs.wcs.crpix = [50.0, 50.0]
        wcs.wcs.cdelt = [-0.001, 0.001]
        return wcs

    def test_star_catalog_config(self):
        """Test StarCatalogConfig dataclass."""
        config = StarCatalogConfig(
            catalogs=["gaia"],
            search_radius=300.0,
            magnitude_limit=20.0,
            cache_size=5000,
            update_frequency="daily",
        )

        assert config.catalogs == ["gaia"]
        assert config.search_radius == 300.0
        assert config.magnitude_limit == 20.0
        assert config.cache_size == 5000
        assert config.update_frequency == "daily"

    def test_query_stars_in_region(self):
        """Test querying stars in a region."""
        # Query around a test position
        ra, dec, radius = 180.0, 0.0, 300.0

        stars = self.catalog.query_stars_in_region(ra, dec, radius)

        # Should return list of star dictionaries
        assert isinstance(stars, list)

        # Check star structure if any returned
        if stars:
            star = stars[0]
            assert isinstance(star, dict)
            assert "ra" in star
            assert "dec" in star
            assert "catalog" in star
            assert "id" in star
            assert "magnitude" in star

    def test_query_stars_invalid_coordinates(self):
        """Test querying with invalid coordinates."""
        # Invalid RA
        with pytest.raises(ValueError):
            self.catalog.query_stars_in_region(-10.0, 0.0, 300.0)

        with pytest.raises(ValueError):
            self.catalog.query_stars_in_region(400.0, 0.0, 300.0)

        # Invalid Dec
        with pytest.raises(ValueError):
            self.catalog.query_stars_in_region(180.0, -100.0, 300.0)

        with pytest.raises(ValueError):
            self.catalog.query_stars_in_region(180.0, 100.0, 300.0)

        # Invalid radius
        with pytest.raises(ValueError):
            self.catalog.query_stars_in_region(180.0, 0.0, -10.0)

    def test_match_stars_to_image(self):
        """Test matching catalog stars to image coordinates."""
        # Get some test stars
        stars = self.catalog.query_stars_in_region(180.0, 0.0, 300.0)

        if not stars:
            # Create mock stars for testing
            stars = [
                {
                    "ra": 180.0,
                    "dec": 0.0,
                    "magnitude": 15.0,
                    "catalog": "gaia",
                    "id": "test1",
                },
                {
                    "ra": 180.01,
                    "dec": 0.01,
                    "magnitude": 16.0,
                    "catalog": "gaia",
                    "id": "test2",
                },
            ]

        # Create test WCS and image shape
        wcs = self.create_test_wcs()
        image_shape = (100, 100)

        matched_stars = self.catalog.match_stars_to_image(stars, wcs, image_shape)

        # Check matched stars structure
        assert isinstance(matched_stars, list)

        for star in matched_stars:
            assert "pixel_x" in star
            assert "pixel_y" in star
            assert "in_image" in star
            assert isinstance(star["pixel_x"], float)
            assert isinstance(star["pixel_y"], float)
            assert isinstance(star["in_image"], bool)

    def test_calculate_star_magnitudes(self):
        """Test calculating star magnitudes for specific filter."""
        # Create test stars with different magnitude types
        stars = [
            {"g_mag": 15.0, "v_mag": 15.2, "r_mag": 14.8},
            {"g_mag": 16.0, "v_mag": 16.1, "b_mag": 16.5},
            {"magnitude": 14.5},  # Generic magnitude
        ]

        # Test different filter bands
        g_mags = self.catalog.calculate_star_magnitudes(stars, "G")
        v_mags = self.catalog.calculate_star_magnitudes(stars, "V")

        assert len(g_mags) == 3
        assert len(v_mags) == 3

        # Check specific values
        assert g_mags[0] == 15.0  # Direct G magnitude
        assert v_mags[0] == 15.2  # Direct V magnitude
        assert not np.isnan(g_mags[2])  # Should fallback to generic magnitude

    def test_cross_match_catalogs(self):
        """Test cross-matching between catalogs."""
        # Create two sets of test stars
        stars1 = [
            {"ra": 180.0, "dec": 0.0, "catalog": "gaia", "id": "gaia1"},
            {"ra": 180.1, "dec": 0.1, "catalog": "gaia", "id": "gaia2"},
        ]

        stars2 = [
            {
                "ra": 180.0001,
                "dec": 0.0001,
                "catalog": "tycho2",
                "id": "tyc1",
            },  # Close match
            {"ra": 181.0, "dec": 1.0, "catalog": "tycho2", "id": "tyc2"},  # No match
        ]

        matches = self.catalog.cross_match_catalogs(stars1, stars2, tolerance=5.0)

        # Should find at least one match
        assert len(matches) >= 1

        # Check match structure
        if matches:
            match_pair = matches[0]
            assert len(match_pair) == 2
            assert "match_distance_arcsec" in match_pair[0]
            assert "match_distance_arcsec" in match_pair[1]

    def test_get_star_by_id(self):
        """Test getting specific star by ID."""
        # This is a mock implementation, so it will return None
        star = self.catalog.get_star_by_id("test_id", "gaia")

        # Should return None in mock implementation
        assert star is None

    def test_cache_functionality(self):
        """Test star catalog caching."""
        # Query the same region twice
        ra, dec, radius = 180.0, 0.0, 300.0

        # First query (will be cached)
        stars1 = self.catalog.query_stars_in_region(ra, dec, radius)

        # Second query (should use cache)
        stars2 = self.catalog.query_stars_in_region(ra, dec, radius)

        # Results should be identical
        assert len(stars1) == len(stars2)

    def test_clear_cache(self):
        """Test clearing the cache."""
        # Query to populate cache
        self.catalog.query_stars_in_region(180.0, 0.0, 300.0)

        # Clear cache
        self.catalog.clear_cache()

        # Should not raise any errors
        assert True

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        # Query to populate cache
        self.catalog.query_stars_in_region(180.0, 0.0, 300.0)

        stats = self.catalog.get_cache_stats()

        # Check statistics structure
        expected_keys = [
            "cached_regions",
            "cached_individual_stars",
            "total_cached_stars",
            "cache_size_mb",
        ]

        for key in expected_keys:
            assert key in stats
