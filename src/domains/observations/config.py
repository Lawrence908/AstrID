"""Configuration settings for survey integration."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class SurveyConfig:
    """Configuration for external survey integration."""

    survey_id: str
    api_endpoint: str
    authentication: dict[str, Any]
    supported_filters: list[str]
    coordinate_system: str
    data_format: str
    rate_limits: dict[str, Any]
    retry_policy: dict[str, Any]


class MASTConfig(BaseModel):
    """MAST API configuration."""

    base_url: str = "https://mast.stsci.edu/api/v0.1"
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    max_results_per_request: int = 50000
    supported_missions: list[str] = ["HST", "JWST", "KEPLER", "TESS", "GALEX"]


class SkyViewConfig(BaseModel):
    """SkyView service configuration."""

    base_url: str = "https://skyview.gsfc.nasa.gov/current/cgi"
    timeout: float = 60.0
    max_retries: int = 3
    rate_limit_delay: float = 2.0
    default_image_size: float = 0.25  # degrees
    default_pixels: int = 512
    supported_formats: list[str] = ["FITS", "JPEG", "GIF"]
    supported_coordinate_systems: list[str] = ["J2000", "B1950", "Galactic", "Ecliptic"]
    supported_projections: list[str] = ["Tan", "Sin", "Arc"]


class SurveyIntegrationConfig(BaseModel):
    """Overall survey integration configuration."""

    mast: MASTConfig = MASTConfig()
    skyview: SkyViewConfig = SkyViewConfig()

    # Global settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_downloads: int = 5
    temp_storage_path: str = "/tmp/astrid_survey_data"

    # Error handling
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    timeout_tolerance: float = 1.5  # Multiplier for timeout values

    # Data validation
    validate_coordinates: bool = True
    validate_exposure_times: bool = True
    min_exposure_time: float = 0.1
    max_exposure_time: float = 86400.0  # 24 hours

    # Survey-specific configurations
    hst_config: dict[str, Any] = {
        "pixel_scales": {
            "ACS/WFC": 0.05,
            "ACS/HRC": 0.025,
            "WFC3/UVIS": 0.04,
            "WFC3/IR": 0.13,
            "WFPC2": 0.1,
            "NICMOS": 0.2,
        },
        "typical_seeing": {
            "ACS": 0.05,
            "WFC3/UVIS": 0.04,
            "WFC3/IR": 0.13,
            "WFPC2": 0.08,
        },
    }

    jwst_config: dict[str, Any] = {
        "pixel_scales": {
            "NIRCAM/SHORT": 0.031,
            "NIRCAM/LONG": 0.063,
            "MIRI": 0.11,
            "NIRSPEC": 0.10,
            "NIRISS": 0.065,
            "FGS": 0.069,
        },
        "psf_models": {
            "NIRCAM": "wavelength_dependent",
            "MIRI": "wavelength_dependent",
            "NIRSPEC": "fixed",
            "NIRISS": "fixed",
            "FGS": "fixed",
        },
    }

    sdss_config: dict[str, Any] = {
        "pixel_scale": 0.396,
        "image_size": [2048, 1489],
        "typical_seeing": 1.4,
        "data_releases": ["DR16", "DR17", "DR18"],
        "base_urls": {
            "DR16": "https://data.sdss.org/sas/dr16",
            "DR17": "https://data.sdss.org/sas/dr17",
            "DR18": "https://data.sdss.org/sas/dr18",
        },
    }

    lsst_config: dict[str, Any] = {
        "pixel_scale": 0.2,
        "ccd_size": [4096, 4096],
        "exposure_times": {
            "u": 30.0,
            "g": 15.0,
            "r": 15.0,
            "i": 15.0,
            "z": 15.0,
            "y": 15.0,
        },
        "data_service_url": "https://data.lsst.cloud/api/v1",
    }


# Default configuration instance
DEFAULT_CONFIG = SurveyIntegrationConfig()


def get_survey_config(survey_name: str) -> dict[str, Any]:
    """Get configuration for a specific survey.

    Args:
        survey_name: Name of the survey (HST, JWST, SDSS, LSST)

    Returns:
        Survey-specific configuration dictionary

    Raises:
        ValueError: For unknown survey names
    """
    config_map = {
        "HST": DEFAULT_CONFIG.hst_config,
        "JWST": DEFAULT_CONFIG.jwst_config,
        "SDSS": DEFAULT_CONFIG.sdss_config,
        "LSST": DEFAULT_CONFIG.lsst_config,
    }

    if survey_name not in config_map:
        raise ValueError(f"Unknown survey: {survey_name}")

    return config_map[survey_name]


def get_mast_config() -> MASTConfig:
    """Get MAST API configuration.

    Returns:
        MAST configuration object
    """
    return DEFAULT_CONFIG.mast


def get_skyview_config() -> SkyViewConfig:
    """Get SkyView configuration.

    Returns:
        SkyView configuration object
    """
    return DEFAULT_CONFIG.skyview


def validate_survey_config(config: dict[str, Any]) -> bool:
    """Validate survey configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: For invalid configuration
    """
    required_keys = ["pixel_scale"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate pixel scale
    pixel_scale = config.get("pixel_scale")
    if isinstance(pixel_scale, int | float):
        if pixel_scale <= 0:
            raise ValueError("Pixel scale must be positive")
    elif isinstance(pixel_scale, dict):
        # Multiple pixel scales for different instruments
        for instrument, scale in pixel_scale.items():
            if scale <= 0:
                raise ValueError(f"Pixel scale for {instrument} must be positive")
    else:
        raise ValueError("Pixel scale must be a number or dictionary")

    return True


def get_error_handling_config() -> dict[str, Any]:
    """Get error handling configuration.

    Returns:
        Error handling configuration dictionary
    """
    return {
        "max_retry_attempts": DEFAULT_CONFIG.max_retry_attempts,
        "retry_backoff_factor": DEFAULT_CONFIG.retry_backoff_factor,
        "timeout_tolerance": DEFAULT_CONFIG.timeout_tolerance,
        "enable_graceful_degradation": True,
        "log_all_errors": True,
        "error_notification_threshold": 10,  # errors per hour
    }


def get_rate_limiting_config() -> dict[str, Any]:
    """Get rate limiting configuration.

    Returns:
        Rate limiting configuration dictionary
    """
    return {
        "mast_delay": DEFAULT_CONFIG.mast.rate_limit_delay,
        "skyview_delay": DEFAULT_CONFIG.skyview.rate_limit_delay,
        "max_concurrent_requests": DEFAULT_CONFIG.max_concurrent_downloads,
        "burst_allowance": 5,  # requests
        "cooldown_period": 60,  # seconds
    }
