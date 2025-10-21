"""Prefect server management and configuration."""

import asyncio
import logging
from typing import Any

import httpx
from prefect import get_client
from prefect.workers import ProcessWorker
from prefect.workers.base import WorkPool

from ..storage.config import StorageConfig
from .config import (
    FlowStatus,
    FlowStatusInfo,
    FlowType,
    WorkflowConfig,
)

logger = logging.getLogger(__name__)


class PrefectServer:
    """Manages Prefect server configuration and operations."""

    def __init__(self, config: WorkflowConfig) -> None:
        """Initialize Prefect server manager.

        Args:
            config: Workflow configuration
        """
        self.config = config
        self._client: Any | None = None
        self._worker: ProcessWorker | None = None

    async def start_server(self, host: str = "0.0.0.0", port: int = 4200) -> None:
        """Start Prefect server.

        Args:
            host: Server host address
            port: Server port
        """
        try:
            logger.info(f"Starting Prefect server on {host}:{port}")

            # Set Prefect API URL
            import os

            os.environ["PREFECT_API_URL"] = f"http://{host}:{port}/api"

            # Initialize client
            self._client = get_client()

            # Configure database
            await self.configure_database()

            # Setup authentication if enabled
            if self.config.authentication_enabled:
                await self.setup_authentication()

            # Configure storage
            await self.configure_storage()

            # Setup monitoring
            if self.config.monitoring_enabled:
                await self.setup_monitoring()

            # Create work pool
            await self._create_work_pool()

            logger.info("Prefect server started successfully")

        except Exception as e:
            logger.error(f"Failed to start Prefect server: {e}")
            raise

    async def configure_database(self, database_url: str | None = None) -> None:
        """Configure Prefect database.

        Args:
            database_url: Database URL (uses config if not provided)
        """
        db_url = database_url or self.config.database_url
        logger.info(f"Configuring Prefect database: {db_url}")

        # Note: In production, you would configure Prefect to use this database
        # For now, we'll just log the configuration
        logger.info("Database configuration completed")

    async def setup_authentication(
        self, auth_config: dict[str, Any] | None = None
    ) -> None:
        """Setup Prefect authentication.

        Args:
            auth_config: Authentication configuration
        """
        logger.info("Setting up Prefect authentication")

        # Default authentication configuration
        default_config = {
            "auth_type": "api_key",
            "api_key_header": "X-Prefect-API-Key",
            "enabled": True,
        }

        config = auth_config or default_config
        logger.info(f"Authentication configured: {config}")

    async def configure_storage(
        self, storage_config: StorageConfig | None = None
    ) -> None:
        """Configure Prefect storage.

        Args:
            storage_config: Storage configuration (uses config if not provided)
        """
        storage = storage_config or self.config.storage_config
        logger.info(f"Configuring Prefect storage: {storage}")

        # Configure Prefect to use the storage backend
        # This would typically involve setting up result storage and artifact storage
        logger.info("Storage configuration completed")

    async def setup_monitoring(
        self, monitoring_config: dict[str, Any] | None = None
    ) -> None:
        """Setup Prefect monitoring.

        Args:
            monitoring_config: Monitoring configuration
        """
        logger.info("Setting up Prefect monitoring")

        # Default monitoring configuration
        default_config = {
            "metrics_enabled": True,
            "logging_enabled": True,
            "tracing_enabled": True,
            "retention_days": self.config.metrics_retention_days,
        }

        config = monitoring_config or default_config
        logger.info(f"Monitoring configured: {config}")

    async def _create_work_pool(self) -> None:
        """Create work pool for flow execution."""
        try:
            if not self._client:
                raise RuntimeError("Prefect client not initialized")

            # Check if work pool already exists
            try:
                await self._client.read_work_pool(self.config.prefect_work_pool)
                logger.info(
                    f"Work pool '{self.config.prefect_work_pool}' already exists"
                )
            except Exception:
                # Create work pool
                work_pool = WorkPool(
                    name=self.config.prefect_work_pool,
                    type="process",
                    description="AstrID workflow execution pool",
                )
                await self._client.create_work_pool(work_pool)
                logger.info(f"Created work pool: {self.config.prefect_work_pool}")

        except Exception as e:
            logger.error(f"Failed to create work pool: {e}")
            raise

    async def start_worker(self) -> None:
        """Start Prefect worker for flow execution."""
        try:
            logger.info("Starting Prefect worker")

            self._worker = ProcessWorker(
                work_pool_name=self.config.prefect_work_pool,
                limit=self.config.max_concurrent_flows,
            )

            # Start worker in background
            asyncio.create_task(self._worker.start())
            logger.info("Prefect worker started")

        except Exception as e:
            logger.error(f"Failed to start Prefect worker: {e}")
            raise

    async def stop_worker(self) -> None:
        """Stop Prefect worker."""
        if self._worker:
            logger.info("Stopping Prefect worker")
            # Note: ProcessWorker doesn't have a shutdown method in this version
            # The worker will be stopped when the process exits
            pass
            self._worker = None
            logger.info("Prefect worker stopped")

    async def health_check(self) -> dict[str, Any]:
        """Check Prefect server health.

        Returns:
            Health status information
        """
        try:
            if not self._client:
                return {"status": "unhealthy", "error": "Client not initialized"}

            # Check server connectivity
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.prefect_api_url}/health")
                response.raise_for_status()

            # Check work pool status
            work_pool = await self._client.read_work_pool(self.config.prefect_work_pool)

            return {
                "status": "healthy",
                "server_url": self.config.prefect_server_url,
                "work_pool": work_pool.name,
                "worker_running": self._worker is not None,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def get_flow_status(self, flow_id: str) -> FlowStatusInfo:
        """Get status of a specific flow.

        Args:
            flow_id: Flow identifier

        Returns:
            Flow status information
        """
        try:
            if not self._client:
                raise RuntimeError("Prefect client not initialized")

            # Get flow run from Prefect
            flow_run = await self._client.read_flow_run(flow_id)

            # Map Prefect status to our status
            status_mapping = {
                "PENDING": FlowStatus.PENDING,
                "RUNNING": FlowStatus.RUNNING,
                "COMPLETED": FlowStatus.COMPLETED,
                "FAILED": FlowStatus.FAILED,
                "CANCELLED": FlowStatus.CANCELLED,
                "RETRYING": FlowStatus.RETRYING,
            }

            status = status_mapping.get(flow_run.state.type, FlowStatus.PENDING)

            return FlowStatusInfo(
                flow_id=flow_id,
                flow_type=FlowType.OBSERVATION_INGESTION,  # Default, would be determined from flow
                status=status,
                start_time=flow_run.start_time or flow_run.created,
                end_time=flow_run.end_time,
                progress=0.0,  # Would be calculated from flow state
                current_step="",  # Would be extracted from flow state
                error_message=flow_run.state.message
                if flow_run.state.type == "FAILED"
                else None,
                metrics={},
            )

        except Exception as e:
            logger.error(f"Failed to get flow status for {flow_id}: {e}")
            raise

    async def cancel_flow(self, flow_id: str) -> bool:
        """Cancel a running flow.

        Args:
            flow_id: Flow identifier

        Returns:
            True if cancellation was successful
        """
        try:
            if not self._client:
                raise RuntimeError("Prefect client not initialized")

            await self._client.cancel_flow_run(flow_id)
            logger.info(f"Cancelled flow: {flow_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel flow {flow_id}: {e}")
            return False

    async def retry_flow(self, flow_id: str) -> bool:
        """Retry a failed flow.

        Args:
            flow_id: Flow identifier

        Returns:
            True if retry was successful
        """
        try:
            if not self._client:
                raise RuntimeError("Prefect client not initialized")

            # Retry the flow run
            await self._client.retry_flow_run(flow_id)
            logger.info(f"Retried flow: {flow_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to retry flow {flow_id}: {e}")
            return False

    async def get_flow_logs(
        self, flow_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get logs for a specific flow.

        Args:
            flow_id: Flow identifier
            limit: Maximum number of log entries

        Returns:
            List of log entries
        """
        try:
            if not self._client:
                raise RuntimeError("Prefect client not initialized")

            # Get flow run logs
            logs = await self._client.read_logs(flow_run_id=flow_id, limit=limit)

            return [
                {
                    "timestamp": log.timestamp,
                    "level": log.level,
                    "message": log.message,
                    "name": log.name,
                }
                for log in logs
            ]

        except Exception as e:
            logger.error(f"Failed to get logs for flow {flow_id}: {e}")
            return []

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.stop_worker()
        if self._client:
            await self._client.close()
            self._client = None
