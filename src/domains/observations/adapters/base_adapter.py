"""Base adapter interface for survey-specific data normalization."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from src.domains.observations.schema import ObservationCreate


class SurveyAdapter(ABC):
    """Abstract base class for survey-specific data adapters."""

    def __init__(self, survey_name: str):
        """Initialize adapter.

        Args:
            survey_name: Name of the survey this adapter handles
        """
        self.survey_name = survey_name

    @abstractmethod
    async def normalize_observation_data(
        self, raw_data: dict[str, Any], survey_id: UUID
    ) -> ObservationCreate:
        """Normalize raw observation data to our standard format.

        Args:
            raw_data: Raw observation data from the survey
            survey_id: Survey UUID to associate with observation

        Returns:
            Normalized observation data

        Raises:
            ValueError: For invalid or incomplete data
        """
        pass

    @abstractmethod
    async def extract_metadata(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Extract survey-specific metadata from raw data.

        Args:
            raw_data: Raw observation data

        Returns:
            Extracted metadata dictionary

        Raises:
            ValueError: For invalid data format
        """
        pass

    @abstractmethod
    async def validate_survey_specific_data(self, data: dict[str, Any]) -> bool:
        """Validate survey-specific data requirements.

        Args:
            data: Data to validate

        Returns:
            True if data meets survey requirements

        Raises:
            ValueError: For validation failures with specific messages
        """
        pass

    def get_supported_filters(self) -> list[str]:
        """Get list of supported filter bands for this survey.

        Returns:
            List of filter band names
        """
        return []

    def get_data_requirements(self) -> dict[str, Any]:
        """Get data requirements for this survey.

        Returns:
            Dictionary describing required and optional fields
        """
        return {
            "required_fields": [
                "observation_id",
                "ra",
                "dec",
                "observation_time",
                "filter_band",
                "exposure_time",
                "fits_url",
            ],
            "optional_fields": [
                "pixel_scale",
                "image_width",
                "image_height",
                "airmass",
                "seeing",
            ],
            "survey_specific": {},
        }

    def map_filter_band(self, survey_filter: str) -> str:
        """Map survey-specific filter names to standard names.

        Args:
            survey_filter: Survey-specific filter name

        Returns:
            Standardized filter name
        """
        # Default implementation returns input unchanged
        return survey_filter

    def calculate_pixel_scale(self, raw_data: dict[str, Any]) -> float | None:
        """Calculate pixel scale from raw data.

        Args:
            raw_data: Raw observation data

        Returns:
            Pixel scale in arcsec/pixel or None if not calculable
        """
        # Default implementation
        return raw_data.get("pixel_scale") or raw_data.get("s_resolution")

    def extract_image_dimensions(
        self, raw_data: dict[str, Any]
    ) -> tuple[int | None, int | None]:
        """Extract image dimensions from raw data.

        Args:
            raw_data: Raw observation data

        Returns:
            Tuple of (width, height) or (None, None) if not available
        """
        # Default implementation
        width = raw_data.get("image_width") or raw_data.get("naxis1")
        height = raw_data.get("image_height") or raw_data.get("naxis2")

        if width is not None:
            width = int(width)
        if height is not None:
            height = int(height)

        return width, height

    def calculate_airmass(self, raw_data: dict[str, Any]) -> float | None:
        """Calculate or extract airmass from raw data.

        Args:
            raw_data: Raw observation data

        Returns:
            Airmass value or None if not available
        """
        # Default implementation
        airmass = raw_data.get("airmass")
        if airmass is not None:
            return float(airmass)
        return None

    def extract_seeing(self, raw_data: dict[str, Any]) -> float | None:
        """Extract seeing measurement from raw data.

        Args:
            raw_data: Raw observation data

        Returns:
            Seeing in arcseconds or None if not available
        """
        # Default implementation
        seeing = raw_data.get("seeing") or raw_data.get("fwhm")
        if seeing is not None:
            return float(seeing)
        return None
