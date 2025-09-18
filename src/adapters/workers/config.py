"""Dramatiq worker configuration and management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend

from src.core.logging import configure_domain_logger


class WorkerType(str, Enum):
    """Worker type enumeration."""

    OBSERVATION_INGESTION = "observation_ingestion"
    PREPROCESSING = "preprocessing"
    DIFFERENCING = "differencing"
    DETECTION = "detection"
    CURATION = "curation"
    NOTIFICATION = "notification"


@dataclass
class WorkerConfig:
    """Worker configuration settings."""

    broker_url: str
    result_backend: str
    max_retries: int = 3
    retry_delay: int = 1000  # milliseconds
    worker_timeout: int = 300  # seconds
    max_memory: int = 1024  # MB
    max_cpu: int = 80  # percentage
    concurrency: int = 4
    prefetch_multiplier: int = 2
    queue_name: str = "default"
    priority: int = 0
    enabled: bool = True


@dataclass
class TaskQueue:
    """Task queue configuration."""

    queue_name: str
    worker_type: WorkerType
    priority: int
    max_retries: int
    timeout: int
    concurrency: int
    enabled: bool


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""

    worker_id: str
    worker_type: WorkerType
    status: str  # IDLE, BUSY, ERROR, STOPPED
    tasks_processed: int
    tasks_failed: int
    average_processing_time: float
    memory_usage: float
    cpu_usage: float
    last_heartbeat: str


class WorkerManager:
    """Manages Dramatiq workers and configuration."""

    def __init__(self):
        self.logger = configure_domain_logger("workers.manager")
        self.broker = None
        self.result_backend = None
        self.actors = {}

    def setup_broker(self, config: WorkerConfig) -> None:
        """Set up Redis broker and result backend."""
        self.logger.info(f"Setting up Redis broker: {config.broker_url}")

        try:
            # Create Redis broker
            self.broker = RedisBroker(url=config.broker_url)

            # Create result backend
            self.result_backend = RedisBackend(url=config.result_backend)
            self.broker.add_middleware(Results(backend=self.result_backend))

            # Set the broker
            dramatiq.set_broker(self.broker)

            self.logger.info(
                "Successfully configured Dramatiq broker and result backend"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup broker: {e}")
            raise

    def create_actor(
        self,
        queue_name: str,
        worker_type: WorkerType,
        max_retries: int = 3,
        time_limit: int = 300000,  # milliseconds
        min_backoff: int = 1000,  # milliseconds
        max_backoff: int = 10000,  # milliseconds
    ) -> Any:
        """Create a Dramatiq actor with configuration."""
        self.logger.info(f"Creating actor for queue: {queue_name}, type: {worker_type}")

        try:
            actor = cast(
                Any,
                dramatiq.actor(
                    queue_name=queue_name,
                    max_retries=max_retries,
                    time_limit=time_limit,
                    min_backoff=min_backoff,
                    max_backoff=max_backoff,
                ),
            )

            self.actors[queue_name] = {
                "actor": actor,
                "worker_type": worker_type,
                "max_retries": max_retries,
                "time_limit": time_limit,
            }

            self.logger.info(f"Successfully created actor for {queue_name}")
            return actor

        except Exception as e:
            self.logger.error(f"Failed to create actor for {queue_name}: {e}")
            raise

    def get_worker_metrics(self, worker_id: str) -> WorkerMetrics | None:
        """Get metrics for a specific worker."""
        # This would typically query the broker for worker metrics
        # For now, return a placeholder
        return WorkerMetrics(
            worker_id=worker_id,
            worker_type=WorkerType.OBSERVATION_INGESTION,
            status="IDLE",
            tasks_processed=0,
            tasks_failed=0,
            average_processing_time=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            last_heartbeat="2024-01-01T00:00:00Z",
        )

    def get_queue_status(self) -> dict[str, Any]:
        """Get status of all queues."""
        # This would typically query the broker for queue status
        # For now, return a placeholder
        return {
            "queues": list(self.actors.keys()),
            "total_actors": len(self.actors),
            "broker_connected": self.broker is not None,
        }


def get_worker_config() -> WorkerConfig:
    """Get worker configuration from environment variables."""
    return WorkerConfig(
        broker_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        result_backend=os.getenv("REDIS_RESULT_URL", "redis://localhost:6379/1"),
        max_retries=int(os.getenv("WORKER_MAX_RETRIES", "3")),
        retry_delay=int(os.getenv("WORKER_RETRY_DELAY", "1000")),
        worker_timeout=int(os.getenv("WORKER_TIMEOUT", "300")),
        max_memory=int(os.getenv("WORKER_MAX_MEMORY", "1024")),
        max_cpu=int(os.getenv("WORKER_MAX_CPU", "80")),
        concurrency=int(os.getenv("WORKER_CONCURRENCY", "4")),
        prefetch_multiplier=int(os.getenv("WORKER_PREFETCH_MULTIPLIER", "2")),
    )


def get_task_queues() -> list[TaskQueue]:
    """Get configured task queues."""
    return [
        TaskQueue(
            queue_name="observation_ingestion",
            worker_type=WorkerType.OBSERVATION_INGESTION,
            priority=1,
            max_retries=3,
            timeout=300,
            concurrency=2,
            enabled=True,
        ),
        TaskQueue(
            queue_name="preprocessing",
            worker_type=WorkerType.PREPROCESSING,
            priority=2,
            max_retries=3,
            timeout=600,
            concurrency=2,
            enabled=True,
        ),
        TaskQueue(
            queue_name="differencing",
            worker_type=WorkerType.DIFFERENCING,
            priority=3,
            max_retries=3,
            timeout=900,
            concurrency=1,
            enabled=True,
        ),
        TaskQueue(
            queue_name="detection",
            worker_type=WorkerType.DETECTION,
            priority=4,
            max_retries=3,
            timeout=1200,
            concurrency=1,
            enabled=True,
        ),
        TaskQueue(
            queue_name="curation",
            worker_type=WorkerType.CURATION,
            priority=5,
            max_retries=2,
            timeout=300,
            concurrency=1,
            enabled=True,
        ),
        TaskQueue(
            queue_name="notification",
            worker_type=WorkerType.NOTIFICATION,
            priority=6,
            max_retries=2,
            timeout=60,
            concurrency=3,
            enabled=True,
        ),
    ]


# Global worker manager instance
worker_manager = WorkerManager()
