"""Catalog API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.core.db.session import get_db
from src.domains.catalog.service import (
    AuditLogService,
    ProcessingJobService,
    SystemConfigService,
)

router = APIRouter()


# SystemConfig Models
class SystemConfigCreate(BaseModel):
    """Create system config request model."""

    key: str
    value: str
    category: str
    description: str | None = None


class SystemConfigResponse(BaseModel):
    """System config response model."""

    id: str
    key: str
    value: str
    category: str
    description: str | None
    created_at: str
    updated_at: str


# ProcessingJob Models
class ProcessingJobCreate(BaseModel):
    """Create processing job request model."""

    job_type: str
    status: str
    parameters: dict
    priority: int = 0


class ProcessingJobResponse(BaseModel):
    """Processing job response model."""

    id: str
    job_type: str
    status: str
    parameters: dict
    priority: int
    created_at: str
    updated_at: str


# AuditLog Models
class AuditLogCreate(BaseModel):
    """Create audit log request model."""

    action: str
    resource_type: str
    resource_id: str
    details: dict


class AuditLogResponse(BaseModel):
    """Audit log response model."""

    id: str
    action: str
    resource_type: str
    resource_id: str
    details: dict
    timestamp: str


# SystemConfig Routes
@router.post("/configs", response_model=SystemConfigResponse)
async def create_system_config(config: SystemConfigCreate, db=Depends(get_db)):
    """Create a new system configuration."""
    service = SystemConfigService(db)
    return await service.create_config(config)


@router.get("/configs", response_model=list[SystemConfigResponse])
async def list_system_configs(
    key: str | None = Query(None, description="Filter by key"),
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(100, le=1000, description="Maximum number of configs"),
    offset: int = Query(0, ge=0, description="Number of configs to skip"),
    db=Depends(get_db),
):
    """List system configurations with optional filtering."""
    service = SystemConfigService(db)
    return await service.list_configs(
        key=key, category=category, limit=limit, offset=offset
    )


@router.get("/configs/{config_id}", response_model=SystemConfigResponse)
async def get_system_config(config_id: str, db=Depends(get_db)):
    """Get a specific system configuration by ID."""
    service = SystemConfigService(db)
    config = await service.get_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="System configuration not found")
    return config


# ProcessingJob Routes
@router.post("/jobs", response_model=ProcessingJobResponse)
async def create_processing_job(job: ProcessingJobCreate, db=Depends(get_db)):
    """Create a new processing job."""
    service = ProcessingJobService(db)
    return await service.create_job(job)


@router.get("/jobs", response_model=list[ProcessingJobResponse])
async def list_processing_jobs(
    job_type: str | None = Query(None, description="Filter by job type"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, le=1000, description="Maximum number of jobs"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    db=Depends(get_db),
):
    """List processing jobs with optional filtering."""
    service = ProcessingJobService(db)
    return await service.list_jobs(
        job_type=job_type, status=status, limit=limit, offset=offset
    )


@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
async def get_processing_job(job_id: str, db=Depends(get_db)):
    """Get a specific processing job by ID."""
    service = ProcessingJobService(db)
    job = await service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    return job


# AuditLog Routes
@router.post("/audit-logs", response_model=AuditLogResponse)
async def create_audit_log(log: AuditLogCreate, db=Depends(get_db)):
    """Create a new audit log entry."""
    service = AuditLogService(db)
    return await service.create_log(log)


@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def list_audit_logs(
    action: str | None = Query(None, description="Filter by action"),
    resource_type: str | None = Query(None, description="Filter by resource type"),
    limit: int = Query(100, le=1000, description="Maximum number of logs"),
    offset: int = Query(0, ge=0, description="Number of logs to skip"),
    db=Depends(get_db),
):
    """List audit logs with optional filtering."""
    service = AuditLogService(db)
    return await service.list_logs(
        action=action, resource_type=resource_type, limit=limit, offset=offset
    )


@router.get("/audit-logs/{log_id}", response_model=AuditLogResponse)
async def get_audit_log(log_id: str, db=Depends(get_db)):
    """Get a specific audit log by ID."""
    service = AuditLogService(db)
    log = await service.get_log(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Audit log not found")
    return log
