"""Catalog service layer for business logic."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.catalog.repository import (
    AuditLogRepository,
    ProcessingJobRepository,
    SystemConfigRepository,
)
from src.domains.catalog.schema import (
    AuditLogCreate,
    ProcessingJobCreate,
    SystemConfigCreate,
)


class SystemConfigService:
    """Service for SystemConfig business logic."""

    def __init__(self, db: AsyncSession):
        self.repository = SystemConfigRepository(db)
        self.logger = configure_domain_logger("catalog.system_config")

    async def create_config(self, config_data: SystemConfigCreate):
        """Create a new system configuration with business logic."""
        self.logger.info(
            f"Creating system config: key={config_data.key}, category={config_data.category}"
        )
        try:
            # TODO: Add validation, default values, etc.
            result = await self.repository.create(config_data)
            self.logger.info(
                f"Successfully created system config: id={result.id}, key={result.key}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create system config: key={config_data.key}, error={str(e)}"
            )
            raise

    async def get_config(self, config_id: str):
        """Get system configuration by ID."""
        self.logger.debug(f"Retrieving system config by ID: {config_id}")
        result = await self.repository.get_by_id(config_id)
        if result:
            self.logger.debug(f"Found system config: id={result.id}, key={result.key}")
        else:
            self.logger.warning(f"System config not found: id={config_id}")
        return result

    async def get_config_by_key(self, key: str):
        """Get system configuration by key."""
        self.logger.debug(f"Retrieving system config by key: {key}")
        result = await self.repository.get_by_key(key)
        if result:
            self.logger.debug(f"Found system config: id={result.id}, key={result.key}")
        else:
            self.logger.warning(f"System config not found: key={key}")
        return result

    async def list_configs(
        self,
        key: str | None = None,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List system configurations."""
        self.logger.debug(
            f"Listing system configs: key={key}, category={category}, limit={limit}, offset={offset}"
        )
        result = await self.repository.list(
            key=key, category=category, limit=limit, offset=offset
        )
        self.logger.debug(f"Retrieved {len(result)} system configs")
        return result


class ProcessingJobService:
    """Service for ProcessingJob business logic."""

    def __init__(self, db: AsyncSession):
        self.repository = ProcessingJobRepository(db)
        self.logger = configure_domain_logger("catalog.processing_job")

    async def create_job(self, job_data: ProcessingJobCreate):
        """Create a new processing job with business logic."""
        self.logger.info(
            f"Creating processing job: type={job_data.job_type}, priority={job_data.priority}"
        )
        try:
            # TODO: Add job validation, status initialization, etc.
            result = await self.repository.create(job_data)
            self.logger.info(
                f"Successfully created processing job: id={result.id}, type={result.job_type}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create processing job: type={job_data.job_type}, error={str(e)}"
            )
            raise

    async def get_job(self, job_id: str):
        """Get processing job by ID."""
        self.logger.debug(f"Retrieving processing job by ID: {job_id}")
        result = await self.repository.get_by_id(job_id)
        if result:
            self.logger.debug(
                f"Found processing job: id={result.id}, type={result.job_type}, status={result.status}"
            )
        else:
            self.logger.warning(f"Processing job not found: id={job_id}")
        return result

    async def list_jobs(
        self,
        job_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List processing jobs."""
        self.logger.debug(
            f"Listing processing jobs: type={job_type}, status={status}, limit={limit}, offset={offset}"
        )
        result = await self.repository.list(
            job_type=job_type, status=status, limit=limit, offset=offset
        )
        self.logger.debug(f"Retrieved {len(result)} processing jobs")
        return result


class AuditLogService:
    """Service for AuditLog business logic."""

    def __init__(self, db: AsyncSession):
        self.repository = AuditLogRepository(db)
        self.logger = configure_domain_logger("catalog.audit_log")

    async def create_log(self, log_data: AuditLogCreate):
        """Create a new audit log entry with business logic."""
        self.logger.info(
            f"Creating audit log: action={log_data.action}, resource_type={log_data.resource_type}"
        )
        try:
            # TODO: Add automatic timestamp, user context, etc.
            result = await self.repository.create(log_data)
            self.logger.info(
                f"Successfully created audit log: id={result.id}, action={result.action}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create audit log: action={log_data.action}, error={str(e)}"
            )
            raise

    async def get_log(self, log_id: str):
        """Get audit log by ID."""
        self.logger.debug(f"Retrieving audit log by ID: {log_id}")
        result = await self.repository.get_by_id(log_id)
        if result:
            self.logger.debug(
                f"Found audit log: id={result.id}, action={result.action}"
            )
        else:
            self.logger.warning(f"Audit log not found: id={log_id}")
        return result

    async def list_logs(
        self,
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List audit logs."""
        self.logger.debug(
            f"Listing audit logs: action={action}, resource_type={resource_type}, limit={limit}, offset={offset}"
        )
        result = await self.repository.list(
            action=action, resource_type=resource_type, limit=limit, offset=offset
        )
        self.logger.debug(f"Retrieved {len(result)} audit logs")
        return result
