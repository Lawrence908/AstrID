"""Pydantic schemas for observations data validation."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, field_validator
from pydantic.fields import FieldInfo

from src.domains.observations.models import ObservationStatus


class SurveyBase(BaseModel):
    """Base schema with shared survey attributes."""

    name: str
    description: str | None = None
    base_url: str | None = None
    api_endpoint: str | None = None
    is_active: bool = True

    class Config:
        from_attributes = True


class SurveyCreate(SurveyBase):
    """Schema for creating a new survey."""

    pass


class SurveyUpdate(BaseModel):
    """Schema for updating an existing survey."""

    name: str | None = None
    description: str | None = None
    base_url: str | None = None
    api_endpoint: str | None = None
    is_active: bool | None = None

    class Config:
        from_attributes = True


class SurveyRead(SurveyBase):
    """Schema for reading survey data."""

    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SurveyListParams(BaseModel):
    """Parameters for survey queries."""

    name: str | None = None
    is_active: bool | None = None
    limit: int = 100
    offset: int = 0

    class Config:
        from_attributes = True


class ObservationBase(BaseModel):
    """Base schema with shared observation attributes."""

    observation_id: str
    ra: float
    dec: float
    observation_time: datetime
    filter_band: str
    exposure_time: float
    fits_url: str

    # Optional metadata
    pixel_scale: float | None = None
    image_width: int | None = None
    image_height: int | None = None
    airmass: float | None = None
    seeing: float | None = None

    class Config:
        from_attributes = True


class ObservationCreate(ObservationBase):
    """Schema for creating a new observation."""

    survey_id: UUID

    class Config:
        from_attributes = True


class ObservationUpdate(BaseModel):
    """Schema for updating an existing observation."""

    observation_id: str | None = None
    ra: float | None = None
    dec: float | None = None
    observation_time: datetime | None = None
    filter_band: str | None = None
    exposure_time: float | None = None
    fits_url: str | None = None
    pixel_scale: float | None = None
    image_width: int | None = None
    image_height: int | None = None
    airmass: float | None = None
    seeing: float | None = None
    status: ObservationStatus | None = None
    fits_file_path: str | None = None
    thumbnail_path: str | None = None

    class Config:
        from_attributes = True


class ObservationRead(ObservationBase):
    """Schema for reading observation data."""

    id: UUID
    survey_id: UUID
    status: ObservationStatus
    fits_file_path: str | None = None
    thumbnail_path: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ObservationWithSurveyRead(ObservationRead):
    """Schema for reading observation data with survey information."""

    survey: SurveyRead

    class Config:
        from_attributes = True


class ObservationListParams(BaseModel):
    """Parameters for observation queries."""

    survey_id: UUID | None = None
    status: ObservationStatus | None = None
    filter_band: str | None = None
    ra_min: float | None = None
    ra_max: float | None = None
    dec_min: float | None = None
    dec_max: float | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    limit: int = 100
    offset: int = 0

    class Config:
        from_attributes = True

    @field_validator("ra_max")  # type: ignore[misc]
    @classmethod
    def ra_max_must_be_greater_than_ra_min(
        cls, v: float | None, info: FieldInfo
    ) -> float | None:
        if (
            v is not None
            and hasattr(info, "data")
            and "ra_min" in info.data
            and info.data["ra_min"] is not None
        ):
            if v <= info.data["ra_min"]:
                raise ValueError("ra_max must be greater than ra_min")
        return v

    @field_validator("dec_max")  # type: ignore[misc]
    @classmethod
    def dec_max_must_be_greater_than_dec_min(
        cls, v: float | None, info: FieldInfo
    ) -> float | None:
        if (
            v is not None
            and hasattr(info, "data")
            and "dec_min" in info.data
            and info.data["dec_min"] is not None
        ):
            if v <= info.data["dec_min"]:
                raise ValueError("dec_max must be greater than dec_min")
        return v
