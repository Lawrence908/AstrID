"""Survey configuration schemas for centralized management of sky regions and missions."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class SkyRegion(BaseModel):
    """Configuration for a sky region to monitor."""

    name: str = Field(..., description="Human-readable name for the region")
    ra_center: float = Field(
        ..., ge=0, le=360, description="Right ascension center in degrees"
    )
    dec_center: float = Field(
        ..., ge=-90, le=90, description="Declination center in degrees"
    )
    radius: float = Field(..., gt=0, le=180, description="Search radius in degrees")
    priority: int = Field(
        default=1, ge=1, le=10, description="Priority level (1=highest, 10=lowest)"
    )
    is_active: bool = Field(
        default=True, description="Whether this region is actively monitored"
    )

    class Config:
        from_attributes = True


class MissionConfig(BaseModel):
    """Configuration for a specific mission/survey."""

    name: str = Field(..., description="Mission name (e.g., 'HST', 'JWST', 'TESS')")
    instruments: list[str] = Field(
        default_factory=list, description="Available instruments"
    )
    filters: list[str] = Field(default_factory=list, description="Available filters")
    min_exposure_time: float = Field(
        default=0.0, ge=0, description="Minimum exposure time in seconds"
    )
    max_exposure_time: float = Field(
        default=3600.0, ge=0, description="Maximum exposure time in seconds"
    )
    data_rights: str = Field(default="PUBLIC", description="Data access rights")
    is_active: bool = Field(
        default=True, description="Whether this mission is actively queried"
    )

    class Config:
        from_attributes = True


class SurveyConfiguration(BaseModel):
    """Complete survey configuration for a monitoring campaign."""

    name: str = Field(..., description="Configuration name")
    description: str | None = Field(None, description="Configuration description")

    # Sky regions to monitor
    sky_regions: list[SkyRegion] = Field(
        default_factory=list, description="Sky regions to monitor"
    )

    # Missions to query
    missions: list[MissionConfig] = Field(
        default_factory=list, description="Missions to query"
    )

    # Query parameters
    query_radius: float = Field(
        default=0.1, gt=0, le=10, description="Default query radius in degrees"
    )
    time_window_days: int = Field(
        default=30, gt=0, le=365, description="Time window for new observations in days"
    )
    max_observations_per_run: int = Field(
        default=100, gt=0, le=1000, description="Max observations per ingestion run"
    )

    # Processing parameters
    auto_download: bool = Field(
        default=True, description="Automatically download FITS files"
    )
    auto_preprocess: bool = Field(
        default=True, description="Automatically trigger preprocessing"
    )
    quality_threshold: float = Field(
        default=0.8, ge=0, le=1, description="Minimum quality threshold"
    )

    # Scheduling
    schedule_cron: str = Field(
        default="0 2 * * *", description="Cron expression for scheduling"
    )
    is_active: bool = Field(
        default=True, description="Whether this configuration is active"
    )

    # Optional fields for database storage
    id: UUID | None = Field(default=None, description="Configuration ID")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    updated_at: datetime | None = Field(
        default=None, description="Last update timestamp"
    )

    class Config:
        from_attributes = True

    @field_validator("sky_regions")
    @classmethod
    def validate_sky_regions(cls, v: list[SkyRegion]) -> list[SkyRegion]:
        """Validate sky regions don't overlap excessively."""
        if len(v) > 50:
            raise ValueError("Maximum 50 sky regions allowed per configuration")
        return v

    @field_validator("missions")
    @classmethod
    def validate_missions(cls, v: list[MissionConfig]) -> list[MissionConfig]:
        """Validate missions configuration."""
        if len(v) > 20:
            raise ValueError("Maximum 20 missions allowed per configuration")
        return v


class SurveyConfigurationCreate(BaseModel):
    """Schema for creating a new survey configuration."""

    name: str
    description: str | None = None
    sky_regions: list[SkyRegion] = Field(default_factory=list)
    missions: list[MissionConfig] = Field(default_factory=list)
    query_radius: float = 0.1
    time_window_days: int = 30
    max_observations_per_run: int = 100
    auto_download: bool = True
    auto_preprocess: bool = True
    quality_threshold: float = 0.8
    schedule_cron: str = "0 2 * * *"
    is_active: bool = True


class SurveyConfigurationUpdate(BaseModel):
    """Schema for updating an existing survey configuration."""

    name: str | None = None
    description: str | None = None
    sky_regions: list[SkyRegion] | None = None
    missions: list[MissionConfig] | None = None
    query_radius: float | None = None
    time_window_days: int | None = None
    max_observations_per_run: int | None = None
    auto_download: bool | None = None
    auto_preprocess: bool | None = None
    quality_threshold: float | None = None
    schedule_cron: str | None = None
    is_active: bool | None = None


class SurveyConfigurationRead(BaseModel):
    """Schema for reading survey configuration data."""

    id: UUID = Field(..., description="Configuration ID")
    name: str = Field(..., description="Configuration name")
    description: str | None = Field(None, description="Configuration description")
    sky_regions: list[SkyRegion] = Field(
        default_factory=list, description="Sky regions to monitor"
    )
    missions: list[MissionConfig] = Field(
        default_factory=list, description="Missions to query"
    )
    query_radius: float = Field(
        default=0.1, gt=0, le=10, description="Default query radius in degrees"
    )
    time_window_days: int = Field(
        default=30, gt=0, le=365, description="Time window for new observations in days"
    )
    max_observations_per_run: int = Field(
        default=100, gt=0, le=1000, description="Max observations per ingestion run"
    )
    auto_download: bool = Field(
        default=True, description="Automatically download FITS files"
    )
    auto_preprocess: bool = Field(
        default=True, description="Automatically trigger preprocessing"
    )
    quality_threshold: float = Field(
        default=0.8, ge=0, le=1, description="Minimum quality threshold"
    )
    schedule_cron: str = Field(
        default="0 2 * * *", description="Cron expression for scheduling"
    )
    is_active: bool = Field(
        default=True, description="Whether this configuration is active"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class SurveyConfigurationListParams(BaseModel):
    """Parameters for survey configuration queries."""

    name: str | None = None
    is_active: bool | None = None
    limit: int = 100
    offset: int = 0

    class Config:
        from_attributes = True


# Default configurations for common use cases
DEFAULT_SURVEY_CONFIGURATIONS = {
    "hst_galaxy_survey": SurveyConfigurationCreate(
        name="HST Galaxy Survey",
        description="Survey of nearby galaxies using HST observations",
        sky_regions=[
            SkyRegion(
                name="M31 Field",
                ra_center=10.6847,
                dec_center=41.2691,
                radius=1.0,
                priority=1,
            ),
            SkyRegion(
                name="M33 Field",
                ra_center=23.4621,
                dec_center=30.6602,
                radius=0.5,
                priority=2,
            ),
            SkyRegion(
                name="M51 Field",
                ra_center=202.4696,
                dec_center=47.1953,
                radius=0.3,
                priority=3,
            ),
        ],
        missions=[
            MissionConfig(
                name="HST",
                instruments=["ACS/WFC", "WFC3/UVIS", "WFC3/IR"],
                filters=["F814W", "F606W", "F435W"],
                min_exposure_time=100,
                max_exposure_time=1800,
            ),
        ],
        query_radius=0.1,
        time_window_days=30,
        max_observations_per_run=50,
        schedule_cron="0 2 * * *",
    ),
    "jwst_exoplanet_survey": SurveyConfigurationCreate(
        name="JWST Exoplanet Survey",
        description="JWST observations for exoplanet detection and characterization",
        sky_regions=[
            SkyRegion(
                name="Kepler Field",
                ra_center=290.0,
                dec_center=44.5,
                radius=2.0,
                priority=1,
            ),
            SkyRegion(
                name="TESS Sectors",
                ra_center=0.0,
                dec_center=0.0,
                radius=30.0,
                priority=2,
            ),
        ],
        missions=[
            MissionConfig(
                name="JWST",
                instruments=["NIRCam", "MIRI", "NIRSpec"],
                filters=["F200W", "F356W", "F444W"],
                min_exposure_time=60,
                max_exposure_time=3600,
            ),
        ],
        query_radius=0.2,
        time_window_days=7,
        max_observations_per_run=25,
        schedule_cron="0 */6 * * *",  # Every 6 hours
    ),
    "tess_continuous_survey": SurveyConfigurationCreate(
        name="TESS Continuous Survey",
        description="Continuous monitoring of TESS observations for transient detection",
        sky_regions=[
            SkyRegion(
                name="TESS Northern",
                ra_center=0.0,
                dec_center=30.0,
                radius=30.0,
                priority=1,
            ),
            SkyRegion(
                name="TESS Southern",
                ra_center=0.0,
                dec_center=-30.0,
                radius=30.0,
                priority=1,
            ),
        ],
        missions=[
            MissionConfig(
                name="TESS",
                instruments=["Camera"],
                filters=["TESS"],
                min_exposure_time=120,
                max_exposure_time=1800,
            ),
        ],
        query_radius=0.5,
        time_window_days=1,
        max_observations_per_run=200,
        schedule_cron="0 */2 * * *",  # Every 2 hours
        auto_preprocess=False,  # TESS data is already processed
    ),
}
