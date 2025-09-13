"""Pydantic schemas for catalog data validation."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ProcessingJobStatus(str, Enum):
    """Processing job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingJobType(str, Enum):
    """Processing job types."""

    INGEST = "ingest"
    PREPROCESS = "preprocess"
    DIFFERENCE = "difference"
    INFERENCE = "inference"
    VALIDATE = "validate"
    NOTIFY = "notify"


class SystemConfigBase(BaseModel):
    """Base schema with shared system config attributes."""

    key: str = Field(..., min_length=1, max_length=100, description="Configuration key")
    value: dict[str, Any] = Field(..., description="Configuration value")
    description: str | None = Field(None, description="Configuration description")
    is_active: bool = Field(True, description="Whether the configuration is active")

    class Config:
        from_attributes = True


class SystemConfigCreate(SystemConfigBase):
    """Schema for creating a new system configuration."""

    pass


class SystemConfigUpdate(BaseModel):
    """Schema for updating an existing system configuration."""

    key: str | None = Field(None, min_length=1, max_length=100)
    value: dict[str, Any] | None = None
    description: str | None = None
    is_active: bool | None = None

    class Config:
        from_attributes = True


class SystemConfigRead(SystemConfigBase):
    """Schema for reading system configuration data."""

    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SystemConfigListParams(BaseModel):
    """Parameters for system configuration queries."""

    key: str | None = Field(None, description="Filter by configuration key")
    is_active: bool | None = Field(None, description="Filter by active status")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    class Config:
        from_attributes = True


class ProcessingJobBase(BaseModel):
    """Base schema with shared processing job attributes."""

    job_type: ProcessingJobType = Field(..., description="Type of processing job")
    entity_id: UUID = Field(..., description="ID of the entity being processed")
    entity_type: str = Field(
        ..., min_length=1, max_length=50, description="Type of entity being processed"
    )
    priority: int = Field(
        0, description="Job priority (higher number = higher priority)"
    )
    max_retries: int = Field(
        3, ge=0, le=10, description="Maximum number of retry attempts"
    )

    class Config:
        from_attributes = True


class ProcessingJobCreate(ProcessingJobBase):
    """Schema for creating a new processing job."""

    workflow_id: str | None = Field(None, max_length=100, description="Workflow ID")
    task_id: str | None = Field(None, max_length=100, description="Task ID")
    scheduled_at: datetime | None = Field(None, description="When to schedule the job")

    class Config:
        from_attributes = True


class ProcessingJobUpdate(BaseModel):
    """Schema for updating an existing processing job."""

    status: ProcessingJobStatus | None = None
    priority: int | None = None
    workflow_id: str | None = Field(None, max_length=100)
    task_id: str | None = Field(None, max_length=100)
    retry_count: int | None = Field(None, ge=0, description="Number of retry attempts")
    max_retries: int | None = Field(None, ge=0, le=10)
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class ProcessingJobRead(ProcessingJobBase):
    """Schema for reading processing job data."""

    id: UUID
    status: ProcessingJobStatus
    workflow_id: str | None = None
    task_id: str | None = None
    retry_count: int
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProcessingJobListParams(BaseModel):
    """Parameters for processing job queries."""

    job_type: ProcessingJobType | None = Field(None, description="Filter by job type")
    entity_id: UUID | None = Field(None, description="Filter by entity ID")
    entity_type: str | None = Field(None, description="Filter by entity type")
    status: ProcessingJobStatus | None = Field(None, description="Filter by status")
    priority: int | None = Field(None, description="Filter by priority")
    workflow_id: str | None = Field(None, description="Filter by workflow ID")
    date_from: datetime | None = Field(None, description="Start date")
    date_to: datetime | None = Field(None, description="End date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    class Config:
        from_attributes = True


class AuditLogBase(BaseModel):
    """Base schema with shared audit log attributes."""

    entity_type: str = Field(
        ..., min_length=1, max_length=50, description="Type of entity"
    )
    entity_id: UUID = Field(..., description="ID of the entity")
    action: str = Field(
        ..., min_length=1, max_length=50, description="Action performed"
    )
    user_id: str | None = Field(
        None, max_length=100, description="ID of the user who performed the action"
    )
    old_values: dict[str, Any] | None = Field(None, description="Previous values")
    new_values: dict[str, Any] | None = Field(None, description="New values")
    change_summary: str | None = Field(None, description="Summary of changes")
    ip_address: str | None = Field(
        None, max_length=45, description="IP address of the user"
    )
    user_agent: str | None = Field(None, description="User agent string")

    class Config:
        from_attributes = True


class AuditLogCreate(AuditLogBase):
    """Schema for creating a new audit log entry."""

    pass


class AuditLogRead(AuditLogBase):
    """Schema for reading audit log data."""

    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class AuditLogListParams(BaseModel):
    """Parameters for audit log queries."""

    entity_type: str | None = Field(None, description="Filter by entity type")
    entity_id: UUID | None = Field(None, description="Filter by entity ID")
    action: str | None = Field(None, description="Filter by action")
    user_id: str | None = Field(None, description="Filter by user ID")
    date_from: datetime | None = Field(None, description="Start date")
    date_to: datetime | None = Field(None, description="End date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    class Config:
        from_attributes = True
