"""Catalog repository for API routes."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.catalog.crud import AuditLogCRUD, ProcessingJobCRUD, SystemConfigCRUD
from src.domains.catalog.schema import (
    AuditLogCreate,
    AuditLogListParams,
    ProcessingJobCreate,
    ProcessingJobListParams,
    SystemConfigCreate,
    SystemConfigListParams,
)


class SystemConfigRepository:
    """Repository for SystemConfig operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, config_data: SystemConfigCreate):
        """Create a new system configuration."""
        return await SystemConfigCRUD.create(self.db, config_data)

    async def get_by_id(self, config_id: str):
        """Get system configuration by ID."""
        from uuid import UUID

        return await SystemConfigCRUD.get_by_id(self.db, UUID(config_id))

    async def get_by_key(self, key: str):
        """Get system configuration by key."""
        return await SystemConfigCRUD.get_by_key(self.db, key)

    async def list(
        self,
        key: str | None = None,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List system configurations with filtering."""
        params = SystemConfigListParams(
            key=key, category=category, limit=limit, offset=offset
        )
        configs, _ = await SystemConfigCRUD.get_many(self.db, params)
        return configs


class ProcessingJobRepository:
    """Repository for ProcessingJob operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, job_data: ProcessingJobCreate):
        """Create a new processing job."""
        return await ProcessingJobCRUD.create(self.db, job_data)

    async def get_by_id(self, job_id: str):
        """Get processing job by ID."""
        from uuid import UUID

        return await ProcessingJobCRUD.get_by_id(self.db, UUID(job_id))

    async def list(
        self,
        job_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List processing jobs with filtering."""
        params = ProcessingJobListParams(
            job_type=job_type, status=status, limit=limit, offset=offset
        )
        jobs, _ = await ProcessingJobCRUD.get_many(self.db, params)
        return jobs


class AuditLogRepository:
    """Repository for AuditLog operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, log_data: AuditLogCreate):
        """Create a new audit log entry."""
        return await AuditLogCRUD.create(self.db, log_data)

    async def get_by_id(self, log_id: str):
        """Get audit log by ID."""
        from uuid import UUID

        return await AuditLogCRUD.get_by_id(self.db, UUID(log_id))

    async def list(
        self,
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List audit logs with filtering."""
        params = AuditLogListParams(
            action=action, resource_type=resource_type, limit=limit, offset=offset
        )
        logs, _ = await AuditLogCRUD.get_many(self.db, params)
        return logs
