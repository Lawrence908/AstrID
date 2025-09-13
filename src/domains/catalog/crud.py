"""CRUD operations for catalog domain."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db.exceptions import create_db_error
from src.domains.catalog.models import AuditLog, ProcessingJob, SystemConfig
from src.domains.catalog.schema import (
    AuditLogCreate,
    AuditLogListParams,
    ProcessingJobCreate,
    ProcessingJobListParams,
    ProcessingJobUpdate,
    SystemConfigCreate,
    SystemConfigListParams,
    SystemConfigUpdate,
)


class SystemConfigCRUD:
    """CRUD operations for SystemConfig model."""

    def __init__(self):
        """Initialize the CRUD class."""
        self.model = SystemConfig
        self.logger = logging.getLogger(__name__)

    async def create(
        self, db: AsyncSession, config_data: SystemConfigCreate
    ) -> SystemConfig:
        """Create a new system configuration."""
        try:
            config = SystemConfig(**config_data.model_dump())
            db.add(config)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(config)
            self.logger.info(f"Successfully created system config with ID: {config.id}")
            return config
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating system config: {str(e)}")
            raise create_db_error(f"Failed to create system config: {str(e)}", e) from e

    async def get_by_id(self, db: AsyncSession, config_id: UUID) -> SystemConfig | None:
        """Get system configuration by ID."""
        try:
            result = await db.execute(
                select(SystemConfig).where(SystemConfig.id == config_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting system config with ID {config_id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to get system config with ID {config_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_by_key(db: AsyncSession, key: str) -> SystemConfig | None:
        """Get system configuration by key."""
        result = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_many(
        db: AsyncSession, params: SystemConfigListParams
    ) -> tuple[list[SystemConfig], int]:
        """Get multiple system configurations with pagination."""
        # Build query
        query = select(SystemConfig)
        count_query = select(func.count(SystemConfig.id))

        # Apply filters
        conditions = []
        if params.key:
            conditions.append(SystemConfig.key.ilike(f"%{params.key}%"))
        if params.category:
            conditions.append(SystemConfig.category == params.category)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(SystemConfig.key).offset(params.offset).limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        configs = result.scalars().all()

        return list(configs), total

    @staticmethod
    async def update(
        db: AsyncSession, config_id: UUID, config_data: SystemConfigUpdate
    ) -> SystemConfig | None:
        """Update a system configuration."""
        config = await SystemConfigCRUD.get_by_id(db, config_id)
        if not config:
            return None

        update_data = config_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(config, field, value)

        await db.commit()
        await db.refresh(config)
        return config

    @staticmethod
    async def delete(db: AsyncSession, config_id: UUID) -> bool:
        """Delete a system configuration."""
        config = await SystemConfigCRUD.get_by_id(db, config_id)
        if not config:
            return False

        await db.delete(config)
        await db.commit()
        return True


class ProcessingJobCRUD:
    """CRUD operations for ProcessingJob model."""

    def __init__(self):
        """Initialize the CRUD class."""
        self.model = ProcessingJob
        self.logger = logging.getLogger(__name__)

    async def create(
        self, db: AsyncSession, job_data: ProcessingJobCreate
    ) -> ProcessingJob:
        """Create a new processing job."""
        try:
            job = ProcessingJob(**job_data.model_dump())
            db.add(job)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(job)
            self.logger.info(f"Successfully created processing job with ID: {job.id}")
            return job
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating processing job: {str(e)}")
            raise create_db_error(
                f"Failed to create processing job: {str(e)}", e
            ) from e

    async def get_by_id(self, db: AsyncSession, job_id: UUID) -> ProcessingJob | None:
        """Get processing job by ID."""
        try:
            result = await db.execute(
                select(ProcessingJob).where(ProcessingJob.id == job_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting processing job with ID {job_id}: {str(e)}"
            )
            raise create_db_error(
                f"Failed to get processing job with ID {job_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: ProcessingJobListParams
    ) -> tuple[list[ProcessingJob], int]:
        """Get multiple processing jobs with pagination."""
        # Build query
        query = select(ProcessingJob)
        count_query = select(func.count(ProcessingJob.id))

        # Apply filters
        conditions = []
        if params.job_type:
            conditions.append(ProcessingJob.job_type == params.job_type)
        if params.status:
            conditions.append(ProcessingJob.status == params.status.value)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(ProcessingJob.created_at.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        jobs = result.scalars().all()

        return list(jobs), total

    @staticmethod
    async def update(
        db: AsyncSession, job_id: UUID, job_data: ProcessingJobUpdate
    ) -> ProcessingJob | None:
        """Update a processing job."""
        job = await ProcessingJobCRUD.get_by_id(db, job_id)
        if not job:
            return None

        update_data = job_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(job, field, value)

        await db.commit()
        await db.refresh(job)
        return job

    @staticmethod
    async def delete(db: AsyncSession, job_id: UUID) -> bool:
        """Delete a processing job."""
        job = await ProcessingJobCRUD.get_by_id(db, job_id)
        if not job:
            return False

        await db.delete(job)
        await db.commit()
        return True


class AuditLogCRUD:
    """CRUD operations for AuditLog model."""

    def __init__(self):
        """Initialize the CRUD class."""
        self.model = AuditLog
        self.logger = logging.getLogger(__name__)

    async def create(self, db: AsyncSession, log_data: AuditLogCreate) -> AuditLog:
        """Create a new audit log entry."""
        try:
            log = AuditLog(**log_data.model_dump())
            db.add(log)
            await db.flush()  # Flush to get the ID without committing
            await db.refresh(log)
            self.logger.info(f"Successfully created audit log with ID: {log.id}")
            return log
        except (SQLAlchemyError, IntegrityError) as e:
            await db.rollback()
            self.logger.error(f"Error creating audit log: {str(e)}")
            raise create_db_error(f"Failed to create audit log: {str(e)}", e) from e

    async def get_by_id(self, db: AsyncSession, log_id: UUID) -> AuditLog | None:
        """Get audit log by ID."""
        try:
            result = await db.execute(select(AuditLog).where(AuditLog.id == log_id))
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting audit log with ID {log_id}: {str(e)}")
            raise create_db_error(
                f"Failed to get audit log with ID {log_id}: {str(e)}", e
            ) from e

    @staticmethod
    async def get_many(
        db: AsyncSession, params: AuditLogListParams
    ) -> tuple[list[AuditLog], int]:
        """Get multiple audit logs with pagination."""
        # Build query
        query = select(AuditLog)
        count_query = select(func.count(AuditLog.id))

        # Apply filters
        conditions = []
        if params.action:
            conditions.append(AuditLog.action == params.action)
        if params.resource_type:
            conditions.append(AuditLog.resource_type == params.resource_type)

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        query = (
            query.order_by(AuditLog.timestamp.desc())
            .offset(params.offset)
            .limit(params.limit)
        )

        # Execute query
        result = await db.execute(query)
        logs = result.scalars().all()

        return list(logs), total
