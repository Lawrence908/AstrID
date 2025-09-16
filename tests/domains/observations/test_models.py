"""Tests for observation domain models."""

from datetime import datetime

from src.domains.observations.models import Observation, ObservationStatus, Survey


class TestObservationModel:
    """Tests for Observation model business logic."""

    def test_validate_coordinates_valid(self):
        """Test coordinate validation with valid coordinates."""
        observation = Observation(
            ra=180.0,
            dec=45.0,
            observation_time=datetime.now(),
            filter_band="g",
            exposure_time=300.0,
            fits_url="http://example.com/test.fits",
            observation_id="test_obs_001",
        )

        assert observation.validate_coordinates() is True

    def test_validate_coordinates_invalid_ra(self):
        """Test coordinate validation with invalid RA."""
        observation = Observation(
            ra=400.0,  # Invalid: > 360
            dec=45.0,
            observation_time=datetime.now(),
            filter_band="g",
            exposure_time=300.0,
            fits_url="http://example.com/test.fits",
            observation_id="test_obs_002",
        )

        assert observation.validate_coordinates() is False

    def test_validate_coordinates_invalid_dec(self):
        """Test coordinate validation with invalid Dec."""
        observation = Observation(
            ra=180.0,
            dec=100.0,  # Invalid: > 90
            observation_time=datetime.now(),
            filter_band="g",
            exposure_time=300.0,
            fits_url="http://example.com/test.fits",
            observation_id="test_obs_003",
        )

        assert observation.validate_coordinates() is False

    def test_calculate_airmass_with_existing_value(self):
        """Test airmass calculation when value already exists."""
        observation = Observation(
            ra=180.0,
            dec=45.0,
            airmass=1.5,
            observation_time=datetime.now(),
            filter_band="g",
            exposure_time=300.0,
            fits_url="http://example.com/test.fits",
            observation_id="test_obs_004",
        )

        result = observation.calculate_airmass()
        assert result == 1.5

    def test_calculate_airmass_without_existing_value(self):
        """Test airmass calculation when no value exists."""
        observation = Observation(
            ra=180.0,
            dec=45.0,
            observation_time=datetime.now(),
            filter_band="g",
            exposure_time=300.0,
            fits_url="http://example.com/test.fits",
            observation_id="test_obs_005",
        )

        result = observation.calculate_airmass(observatory_lat=30.0)
        assert isinstance(result, float)
        assert result > 0

    def test_get_processing_status(self):
        """Test processing status information retrieval."""
        observation = Observation(
            ra=180.0,
            dec=45.0,
            observation_time=datetime.now(),
            filter_band="g",
            exposure_time=300.0,
            fits_url="http://example.com/test.fits",
            observation_id="test_obs_006",
            status=ObservationStatus.INGESTED,
        )

        status_info = observation.get_processing_status()

        assert "status" in status_info
        assert "status_description" in status_info
        assert "can_process" in status_info
        assert "next_stage" in status_info
        assert "processing_metadata" in status_info

        assert status_info["status"] == ObservationStatus.INGESTED
        assert status_info["next_stage"] == "preprocessing"

    def test_get_sky_region_bounds(self):
        """Test sky region bounds calculation."""
        observation = Observation(
            ra=180.0,
            dec=45.0,
            observation_time=datetime.now(),
            filter_band="g",
            exposure_time=300.0,
            fits_url="http://example.com/test.fits",
            observation_id="test_obs_007",
        )

        bounds = observation.get_sky_region_bounds(radius_degrees=0.1)

        assert "ra_min" in bounds
        assert "ra_max" in bounds
        assert "dec_min" in bounds
        assert "dec_max" in bounds
        assert "center_ra" in bounds
        assert "center_dec" in bounds
        assert "radius_degrees" in bounds

        assert bounds["center_ra"] == 180.0
        assert bounds["center_dec"] == 45.0
        assert bounds["radius_degrees"] == 0.1


class TestSurveyModel:
    """Tests for Survey model business logic."""

    def test_get_survey_stats(self):
        """Test survey statistics retrieval."""
        survey = Survey(name="Test Survey", description="A test survey", is_active=True)

        stats = survey.get_survey_stats()

        assert "survey_id" in stats
        assert "name" in stats
        assert "description" in stats
        assert "is_active" in stats
        assert "observation_counts" in stats

        assert stats["name"] == "Test Survey"
        assert stats["is_active"] is True

    def test_is_configured_for_ingestion_valid(self):
        """Test ingestion configuration check with valid configuration."""
        survey = Survey(
            name="Test Survey",
            description="A test survey",
            is_active=True,
            base_url="http://example.com",
        )

        assert survey.is_configured_for_ingestion() is True

    def test_is_configured_for_ingestion_invalid(self):
        """Test ingestion configuration check with invalid configuration."""
        survey = Survey(
            name="",  # Empty name
            description="A test survey",
            is_active=True,
        )

        assert survey.is_configured_for_ingestion() is False

    def test_get_capabilities(self):
        """Test survey capabilities retrieval."""
        survey = Survey(
            name="Test Survey",
            description="A test survey",
            is_active=True,
            base_url="http://example.com",
            api_endpoint="http://api.example.com",
        )

        capabilities = survey.get_capabilities()

        assert "can_ingest" in capabilities
        assert "has_api" in capabilities
        assert "has_base_url" in capabilities
        assert "is_active" in capabilities
        assert "configuration_score" in capabilities

        assert capabilities["can_ingest"] is True
        assert capabilities["has_api"] is True
        assert capabilities["has_base_url"] is True
        assert capabilities["is_active"] is True
        assert isinstance(capabilities["configuration_score"], float)

    def test_calculate_configuration_score(self):
        """Test configuration score calculation."""
        survey = Survey(
            name="Test Survey",
            description="A test survey",
            is_active=True,
            base_url="http://example.com",
            api_endpoint="http://api.example.com",
        )

        score = survey._calculate_configuration_score()

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == 1.0  # All fields are configured
