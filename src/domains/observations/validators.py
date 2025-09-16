"""Validation logic for observations domain."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import UUID

from src.core.exceptions import AstrIDException


class ObservationValidationError(AstrIDException):
    """Raised when observation validation fails."""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(
            message=message,
            details={
                "field": field,
                "value": str(value) if value is not None else None,
            },
            error_code="OBSERVATION_VALIDATION_ERROR",
        )


class CoordinateValidationError(ObservationValidationError):
    """Raised when coordinate validation fails."""

    def __init__(self, message: str, ra: float | None = None, dec: float | None = None):
        self.ra = ra
        self.dec = dec
        super().__init__(
            message=message, field="coordinates", value={"ra": ra, "dec": dec}
        )


class ExposureTimeValidationError(ObservationValidationError):
    """Raised when exposure time validation fails."""

    def __init__(self, message: str, exposure_time: float):
        super().__init__(message=message, field="exposure_time", value=exposure_time)


class FilterBandValidationError(ObservationValidationError):
    """Raised when filter band validation fails."""

    def __init__(
        self, message: str, filter_band: str, valid_bands: list[str] | None = None
    ):
        self.valid_bands = valid_bands
        super().__init__(
            message=message,
            field="filter_band",
            value={"provided": filter_band, "valid_bands": valid_bands},
        )


class MetadataValidationError(ObservationValidationError):
    """Raised when metadata validation fails."""

    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        invalid_fields: dict[str, Any] | None = None,
    ):
        self.missing_fields = missing_fields or []
        self.invalid_fields = invalid_fields or {}
        super().__init__(
            message=message,
            field="metadata",
            value={"missing": missing_fields, "invalid": invalid_fields},
        )


class ObservationValidator:
    """Validator for observation data."""

    # Common filter bands across astronomical surveys
    VALID_FILTER_BANDS = {
        # Optical bands
        "U",
        "B",
        "V",
        "R",
        "I",  # Johnson-Cousins
        "u",
        "g",
        "r",
        "i",
        "z",  # SDSS/Pan-STARRS
        "u'",
        "g'",
        "r'",
        "i'",
        "z'",  # SDSS with primes
        "w1",
        "w2",
        "w3",
        "w4",  # WISE
        "F606W",
        "F814W",
        "F555W",
        "F439W",  # HST common filters
        # Add more as needed for specific surveys
        "clear",
        "white",
        "CLEAR",
        "WHITE",  # Clear/white light
    }

    def __init__(self, survey_filter_bands: set[str] | None = None):
        """Initialize validator with optional survey-specific filter bands.

        Args:
            survey_filter_bands: Set of valid filter bands for a specific survey
        """
        self.survey_filter_bands = survey_filter_bands or self.VALID_FILTER_BANDS

    def validate_coordinates(self, ra: float, dec: float) -> None:
        """Validate astronomical coordinates.

        Args:
            ra: Right Ascension in degrees (0-360)
            dec: Declination in degrees (-90 to 90)

        Raises:
            CoordinateValidationError: If coordinates are invalid
        """
        if not isinstance(ra, int | float):
            raise CoordinateValidationError(
                "Right Ascension must be a number", ra=ra, dec=dec
            )

        if not isinstance(dec, int | float):
            raise CoordinateValidationError(
                "Declination must be a number", ra=ra, dec=dec
            )

        if not (0 <= ra <= 360):
            raise CoordinateValidationError(
                f"Right Ascension must be between 0 and 360 degrees, got {ra}",
                ra=ra,
                dec=dec,
            )

        if not (-90 <= dec <= 90):
            raise CoordinateValidationError(
                f"Declination must be between -90 and 90 degrees, got {dec}",
                ra=ra,
                dec=dec,
            )

    def validate_exposure_time(self, exposure_time: float) -> None:
        """Validate exposure time.

        Args:
            exposure_time: Exposure time in seconds

        Raises:
            ExposureTimeValidationError: If exposure time is invalid
        """
        if not isinstance(exposure_time, int | float):
            raise ExposureTimeValidationError(
                "Exposure time must be a number", exposure_time
            )

        if exposure_time <= 0:
            raise ExposureTimeValidationError(
                f"Exposure time must be positive, got {exposure_time}", exposure_time
            )

        # Reasonable bounds checking (0.001 seconds to 1 day)
        if exposure_time < 0.001:
            raise ExposureTimeValidationError(
                f"Exposure time too short (minimum 0.001s), got {exposure_time}",
                exposure_time,
            )

        if exposure_time > 86400:  # 1 day in seconds
            raise ExposureTimeValidationError(
                f"Exposure time too long (maximum 24 hours), got {exposure_time}",
                exposure_time,
            )

    def validate_filter_band(self, filter_band: str) -> None:
        """Validate filter band.

        Args:
            filter_band: Filter band designation

        Raises:
            FilterBandValidationError: If filter band is invalid
        """
        if not isinstance(filter_band, str):
            raise FilterBandValidationError(
                "Filter band must be a string",
                filter_band,
                list(self.survey_filter_bands),
            )

        if not filter_band.strip():
            raise FilterBandValidationError(
                "Filter band cannot be empty",
                filter_band,
                list(self.survey_filter_bands),
            )

        # Check against known filter bands (case-insensitive)
        filter_band_upper = filter_band.upper()
        valid_bands_upper = {band.upper() for band in self.survey_filter_bands}

        if filter_band_upper not in valid_bands_upper:
            raise FilterBandValidationError(
                f"Unknown filter band '{filter_band}'. Valid bands: {sorted(self.survey_filter_bands)}",
                filter_band,
                list(self.survey_filter_bands),
            )

    def validate_fits_url(self, fits_url: str) -> None:
        """Validate FITS file URL.

        Args:
            fits_url: URL to FITS file

        Raises:
            ObservationValidationError: If URL is invalid
        """
        if not isinstance(fits_url, str):
            raise ObservationValidationError(
                "FITS URL must be a string", field="fits_url", value=fits_url
            )

        if not fits_url.strip():
            raise ObservationValidationError(
                "FITS URL cannot be empty", field="fits_url", value=fits_url
            )

        # Basic URL validation
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        if not url_pattern.match(fits_url):
            raise ObservationValidationError(
                f"Invalid FITS URL format: {fits_url}", field="fits_url", value=fits_url
            )

    def validate_observation_id(self, observation_id: str) -> None:
        """Validate observation ID format.

        Args:
            observation_id: Observation identifier

        Raises:
            ObservationValidationError: If observation ID is invalid
        """
        if not isinstance(observation_id, str):
            raise ObservationValidationError(
                "Observation ID must be a string",
                field="observation_id",
                value=observation_id,
            )

        if not observation_id.strip():
            raise ObservationValidationError(
                "Observation ID cannot be empty",
                field="observation_id",
                value=observation_id,
            )

        # Length validation
        if len(observation_id) > 255:
            raise ObservationValidationError(
                f"Observation ID too long (maximum 255 characters), got {len(observation_id)}",
                field="observation_id",
                value=observation_id,
            )

    def validate_survey_id(self, survey_id: UUID | str) -> None:
        """Validate survey ID.

        Args:
            survey_id: Survey UUID

        Raises:
            ObservationValidationError: If survey ID is invalid
        """
        if isinstance(survey_id, str):
            try:
                UUID(survey_id)
            except ValueError as e:
                raise ObservationValidationError(
                    f"Invalid survey ID format: {survey_id}",
                    field="survey_id",
                    value=survey_id,
                ) from e
        elif not isinstance(survey_id, UUID):
            raise ObservationValidationError(
                "Survey ID must be a UUID", field="survey_id", value=survey_id
            )

    def validate_observation_time(self, observation_time: datetime) -> None:
        """Validate observation time.

        Args:
            observation_time: Observation timestamp

        Raises:
            ObservationValidationError: If observation time is invalid
        """
        if not isinstance(observation_time, datetime):
            raise ObservationValidationError(
                "Observation time must be a datetime object",
                field="observation_time",
                value=observation_time,
            )

        # Check if observation time is in the future (with some tolerance)
        now = (
            datetime.now(observation_time.tzinfo)
            if observation_time.tzinfo
            else datetime.now()
        )
        if observation_time > now:
            raise ObservationValidationError(
                f"Observation time cannot be in the future: {observation_time}",
                field="observation_time",
                value=observation_time,
            )

        # Check if observation time is too old (e.g., before 1990)
        if observation_time.year < 1990:
            raise ObservationValidationError(
                f"Observation time too old (before 1990): {observation_time}",
                field="observation_time",
                value=observation_time,
            )

    def validate_metadata_completeness(
        self, data: dict[str, Any], required_fields: list[str] | None = None
    ) -> None:
        """Validate metadata completeness.

        Args:
            data: Observation data dictionary
            required_fields: List of required field names

        Raises:
            MetadataValidationError: If required metadata is missing
        """
        if required_fields is None:
            required_fields = [
                "survey_id",
                "observation_id",
                "ra",
                "dec",
                "observation_time",
                "filter_band",
                "exposure_time",
                "fits_url",
            ]

        missing_fields = []
        invalid_fields = {}

        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif data[field] is None:
                missing_fields.append(field)
            elif isinstance(data[field], str) and not data[field].strip():
                invalid_fields[field] = "empty string"

        if missing_fields or invalid_fields:
            raise MetadataValidationError(
                f"Metadata validation failed. Missing: {missing_fields}, Invalid: {invalid_fields}",
                missing_fields=missing_fields,
                invalid_fields=invalid_fields,
            )

    def validate_observation_data(self, data: dict[str, Any]) -> None:
        """Comprehensive validation of observation data.

        Args:
            data: Complete observation data dictionary

        Raises:
            ObservationValidationError: If any validation fails
        """
        # Validate required fields exist
        self.validate_metadata_completeness(data)

        # Validate individual fields
        self.validate_survey_id(data["survey_id"])
        self.validate_observation_id(data["observation_id"])
        self.validate_coordinates(data["ra"], data["dec"])
        self.validate_observation_time(data["observation_time"])
        self.validate_filter_band(data["filter_band"])
        self.validate_exposure_time(data["exposure_time"])
        self.validate_fits_url(data["fits_url"])

        # Optional field validation
        if "airmass" in data and data["airmass"] is not None:
            if not isinstance(data["airmass"], int | float) or data["airmass"] <= 0:
                raise ObservationValidationError(
                    f"Airmass must be a positive number, got {data['airmass']}",
                    field="airmass",
                    value=data["airmass"],
                )

        if "seeing" in data and data["seeing"] is not None:
            if not isinstance(data["seeing"], int | float) or data["seeing"] <= 0:
                raise ObservationValidationError(
                    f"Seeing must be a positive number, got {data['seeing']}",
                    field="seeing",
                    value=data["seeing"],
                )
