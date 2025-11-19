"""Tests for observation validators."""

from datetime import datetime
from uuid import uuid4

import pytest

from src.domains.observations.validators import (
    CoordinateValidationError,
    ExposureTimeValidationError,
    FilterBandValidationError,
    MetadataValidationError,
    ObservationValidationError,
    ObservationValidator,
)


class TestObservationValidator:
    """Tests for ObservationValidator."""

    def setup_method(self):
        """Set up test validator."""
        self.validator = ObservationValidator()

    def test_validate_coordinates_valid(self):
        """Test coordinate validation with valid coordinates."""
        # Should not raise any exception
        self.validator.validate_coordinates(180.0, 45.0)
        self.validator.validate_coordinates(0.0, -90.0)
        self.validator.validate_coordinates(360.0, 90.0)

    def test_validate_coordinates_invalid_ra(self):
        """Test coordinate validation with invalid RA."""
        with pytest.raises(CoordinateValidationError):
            self.validator.validate_coordinates(-1.0, 45.0)

        with pytest.raises(CoordinateValidationError):
            self.validator.validate_coordinates(361.0, 45.0)

    def test_validate_coordinates_invalid_dec(self):
        """Test coordinate validation with invalid Dec."""
        with pytest.raises(CoordinateValidationError):
            self.validator.validate_coordinates(180.0, -91.0)

        with pytest.raises(CoordinateValidationError):
            self.validator.validate_coordinates(180.0, 91.0)

    def test_validate_coordinates_invalid_type(self):
        """Test coordinate validation with invalid types."""
        with pytest.raises(CoordinateValidationError):
            self.validator.validate_coordinates("180", 45.0)

        with pytest.raises(CoordinateValidationError):
            self.validator.validate_coordinates(180.0, "45")

    def test_validate_exposure_time_valid(self):
        """Test exposure time validation with valid values."""
        self.validator.validate_exposure_time(300.0)
        self.validator.validate_exposure_time(0.01)
        self.validator.validate_exposure_time(3600.0)

    def test_validate_exposure_time_invalid(self):
        """Test exposure time validation with invalid values."""
        with pytest.raises(ExposureTimeValidationError):
            self.validator.validate_exposure_time(0.0)

        with pytest.raises(ExposureTimeValidationError):
            self.validator.validate_exposure_time(-100.0)

        with pytest.raises(ExposureTimeValidationError):
            self.validator.validate_exposure_time(100000.0)  # Too long

    def test_validate_filter_band_valid(self):
        """Test filter band validation with valid bands."""
        self.validator.validate_filter_band("g")
        self.validator.validate_filter_band("r")
        self.validator.validate_filter_band("V")
        self.validator.validate_filter_band("F606W")

    def test_validate_filter_band_invalid(self):
        """Test filter band validation with invalid bands."""
        with pytest.raises(FilterBandValidationError):
            self.validator.validate_filter_band("invalid_band")

        with pytest.raises(FilterBandValidationError):
            self.validator.validate_filter_band("")

        with pytest.raises(FilterBandValidationError):
            self.validator.validate_filter_band(123)

    def test_validate_fits_url_valid(self):
        """Test FITS URL validation with valid URLs."""
        self.validator.validate_fits_url("http://example.com/test.fits")
        self.validator.validate_fits_url(
            "https://data.example.com/observations/obs001.fits"
        )
        self.validator.validate_fits_url("http://localhost:9001/test.fits")

    def test_validate_fits_url_invalid(self):
        """Test FITS URL validation with invalid URLs."""
        with pytest.raises(ObservationValidationError):
            self.validator.validate_fits_url("not_a_url")

        with pytest.raises(ObservationValidationError):
            self.validator.validate_fits_url("")

        with pytest.raises(ObservationValidationError):
            self.validator.validate_fits_url("ftp://example.com/test.fits")

    def test_validate_observation_id_valid(self):
        """Test observation ID validation with valid IDs."""
        self.validator.validate_observation_id("obs_001")
        self.validator.validate_observation_id("HST-2024-001-G141")
        self.validator.validate_observation_id("a" * 255)  # Max length

    def test_validate_observation_id_invalid(self):
        """Test observation ID validation with invalid IDs."""
        with pytest.raises(ObservationValidationError):
            self.validator.validate_observation_id("")

        with pytest.raises(ObservationValidationError):
            self.validator.validate_observation_id("a" * 256)  # Too long

        with pytest.raises(ObservationValidationError):
            self.validator.validate_observation_id(123)

    def test_validate_survey_id_valid(self):
        """Test survey ID validation with valid UUIDs."""
        valid_uuid = uuid4()
        self.validator.validate_survey_id(valid_uuid)
        self.validator.validate_survey_id(str(valid_uuid))

    def test_validate_survey_id_invalid(self):
        """Test survey ID validation with invalid UUIDs."""
        with pytest.raises(ObservationValidationError):
            self.validator.validate_survey_id("not_a_uuid")

        with pytest.raises(ObservationValidationError):
            self.validator.validate_survey_id(123)

    def test_validate_observation_time_valid(self):
        """Test observation time validation with valid times."""
        valid_time = datetime(2024, 1, 1, 12, 0, 0)
        self.validator.validate_observation_time(valid_time)

    def test_validate_observation_time_invalid(self):
        """Test observation time validation with invalid times."""
        with pytest.raises(ObservationValidationError):
            self.validator.validate_observation_time("2024-01-01")

        # Future time
        with pytest.raises(ObservationValidationError):
            future_time = datetime(2030, 1, 1, 12, 0, 0)
            self.validator.validate_observation_time(future_time)

        # Too old
        with pytest.raises(ObservationValidationError):
            old_time = datetime(1980, 1, 1, 12, 0, 0)
            self.validator.validate_observation_time(old_time)

    def test_validate_metadata_completeness_valid(self):
        """Test metadata completeness validation with valid data."""
        valid_data = {
            "survey_id": uuid4(),
            "observation_id": "test_obs",
            "ra": 180.0,
            "dec": 45.0,
            "observation_time": datetime.now(),
            "filter_band": "g",
            "exposure_time": 300.0,
            "fits_url": "http://example.com/test.fits",
        }

        # Should not raise any exception
        self.validator.validate_metadata_completeness(valid_data)

    def test_validate_metadata_completeness_missing_fields(self):
        """Test metadata completeness validation with missing fields."""
        incomplete_data = {
            "survey_id": uuid4(),
            "observation_id": "test_obs",
            # Missing required fields
        }

        with pytest.raises(MetadataValidationError):
            self.validator.validate_metadata_completeness(incomplete_data)

    def test_validate_observation_data_complete(self):
        """Test complete observation data validation."""
        valid_data = {
            "survey_id": uuid4(),
            "observation_id": "test_obs",
            "ra": 180.0,
            "dec": 45.0,
            "observation_time": datetime(2024, 1, 1, 12, 0, 0),
            "filter_band": "g",
            "exposure_time": 300.0,
            "fits_url": "http://example.com/test.fits",
            "airmass": 1.2,
            "seeing": 0.8,
        }

        # Should not raise any exception
        self.validator.validate_observation_data(valid_data)

    def test_validate_observation_data_invalid(self):
        """Test complete observation data validation with invalid data."""
        invalid_data = {
            "survey_id": uuid4(),
            "observation_id": "test_obs",
            "ra": 400.0,  # Invalid RA
            "dec": 45.0,
            "observation_time": datetime(2024, 1, 1, 12, 0, 0),
            "filter_band": "invalid_band",  # Invalid filter
            "exposure_time": -100.0,  # Invalid exposure time
            "fits_url": "not_a_url",  # Invalid URL
        }

        with pytest.raises(ObservationValidationError):
            self.validator.validate_observation_data(invalid_data)
