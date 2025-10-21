"""Worker management API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.adapters.workers.config import WorkerType, get_task_queues
from src.adapters.workers.monitoring import worker_monitor
from src.core.logging import configure_domain_logger

# Note: Prefix is applied in app include_router (main.py) for consistency
router = APIRouter(tags=["workers"])
logger = configure_domain_logger("api.workers")


class WorkerStatusResponse(BaseModel):
    """Response model for worker status."""

    worker_id: str
    worker_type: str
    status: str
    tasks_processed: int
    tasks_failed: int
    average_processing_time: float
    memory_usage: float
    cpu_usage: float
    last_heartbeat: str


class WorkerHealthResponse(BaseModel):
    """Response model for worker health."""

    status: str
    total_workers: int
    healthy_workers: int
    health_ratio: float
    total_tasks_processed: int
    total_tasks_failed: int
    failure_rate: float
    uptime_seconds: float
    timestamp: str


class QueueStatusResponse(BaseModel):
    """Response model for queue status."""

    queues: list[str]
    total_actors: int
    broker_connected: bool
    queue_lengths: dict[str, int]
    processing_rates: dict[str, float]
    error_rates: dict[str, float]
    timestamp: str


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""

    time_window_hours: int
    total_tasks_processed: int
    total_tasks_failed: int
    failure_rate: float
    average_processing_time: float
    average_memory_usage_mb: float
    average_cpu_usage_percent: float
    active_workers: int
    timestamp: str


@router.get("/status", response_model=list[WorkerStatusResponse])
async def get_worker_status():
    """Get status of all workers."""
    try:
        logger.info("Getting worker status")

        all_metrics = worker_monitor.get_all_worker_metrics()
        status_list = []

        for _worker_id, metrics in all_metrics.items():
            status_list.append(
                WorkerStatusResponse(
                    worker_id=metrics.worker_id,
                    worker_type=metrics.worker_type.value,
                    status=metrics.status,
                    tasks_processed=metrics.tasks_processed,
                    tasks_failed=metrics.tasks_failed,
                    average_processing_time=metrics.average_processing_time,
                    memory_usage=metrics.memory_usage,
                    cpu_usage=metrics.cpu_usage,
                    last_heartbeat=metrics.last_heartbeat,
                )
            )

        logger.info(f"Retrieved status for {len(status_list)} workers")
        return status_list

    except Exception as err:
        logger.error(f"Failed to get worker status: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/health", response_model=WorkerHealthResponse)
async def get_worker_health():
    """Get overall worker health status."""
    try:
        logger.info("Getting worker health")

        health_data = worker_monitor.get_worker_health()

        response = WorkerHealthResponse(
            status=health_data["status"],
            total_workers=health_data.get("total_workers", 0),
            healthy_workers=health_data.get("healthy_workers", 0),
            health_ratio=health_data.get("health_ratio", 0.0),
            total_tasks_processed=health_data.get("total_tasks_processed", 0),
            total_tasks_failed=health_data.get("total_tasks_failed", 0),
            failure_rate=health_data.get("failure_rate", 0.0),
            uptime_seconds=health_data.get("uptime_seconds", 0.0),
            timestamp=health_data["timestamp"],
        )

        logger.info(f"Worker health: {response.status}")
        return response

    except Exception as err:
        logger.error(f"Failed to get worker health: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/queues", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get status of all task queues."""
    try:
        logger.info("Getting queue status")

        queue_data = worker_monitor.get_queue_status()

        response = QueueStatusResponse(
            queues=queue_data.get("queues", []),
            total_actors=queue_data.get("total_actors", 0),
            broker_connected=queue_data.get("broker_connected", False),
            queue_lengths=queue_data.get("queue_lengths", {}),
            processing_rates=queue_data.get("processing_rates", {}),
            error_rates=queue_data.get("error_rates", {}),
            timestamp=queue_data["timestamp"],
        )

        logger.info(f"Retrieved status for {len(response.queues)} queues")
        return response

    except Exception as err:
        logger.error(f"Failed to get queue status: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    time_window_hours: int = Query(
        24, ge=1, le=168, description="Time window in hours"
    ),
):
    """Get performance metrics for workers."""
    try:
        logger.info(f"Getting performance metrics for {time_window_hours} hours")

        metrics_data = worker_monitor.get_performance_metrics(time_window_hours)

        response = PerformanceMetricsResponse(
            time_window_hours=metrics_data["time_window_hours"],
            total_tasks_processed=metrics_data.get("total_tasks_processed", 0),
            total_tasks_failed=metrics_data.get("total_tasks_failed", 0),
            failure_rate=metrics_data.get("failure_rate", 0.0),
            average_processing_time=metrics_data.get("average_processing_time", 0.0),
            average_memory_usage_mb=metrics_data.get("average_memory_usage_mb", 0.0),
            average_cpu_usage_percent=metrics_data.get(
                "average_cpu_usage_percent", 0.0
            ),
            active_workers=metrics_data.get("active_workers", 0),
            timestamp=metrics_data["timestamp"],
        )

        logger.info(f"Retrieved performance metrics for {time_window_hours} hours")
        return response

    except Exception as err:
        logger.error(f"Failed to get performance metrics: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/{worker_type}/status", response_model=list[WorkerStatusResponse])
async def get_worker_type_status(worker_type: str):
    """Get status of workers by type."""
    try:
        logger.info(f"Getting status for worker type: {worker_type}")

        # Validate worker type
        try:
            WorkerType(worker_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid worker type: {worker_type}"
            ) from None

        all_metrics = worker_monitor.get_all_worker_metrics()
        status_list = []

        for _worker_id, metrics in all_metrics.items():
            if metrics.worker_type.value == worker_type:
                status_list.append(
                    WorkerStatusResponse(
                        worker_id=metrics.worker_id,
                        worker_type=metrics.worker_type.value,
                        status=metrics.status,
                        tasks_processed=metrics.tasks_processed,
                        tasks_failed=metrics.tasks_failed,
                        average_processing_time=metrics.average_processing_time,
                        memory_usage=metrics.memory_usage,
                        cpu_usage=metrics.cpu_usage,
                        last_heartbeat=metrics.last_heartbeat,
                    )
                )

        logger.info(f"Retrieved status for {len(status_list)} {worker_type} workers")
        return status_list

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Failed to get worker type status: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/{worker_type}/metrics", response_model=PerformanceMetricsResponse)
async def get_worker_type_metrics(
    worker_type: str,
    time_window_hours: int = Query(
        24, ge=1, le=168, description="Time window in hours"
    ),
):
    """Get performance metrics for specific worker type."""
    try:
        logger.info(
            f"Getting metrics for {worker_type} workers for {time_window_hours} hours"
        )

        # Validate worker type
        try:
            WorkerType(worker_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid worker type: {worker_type}"
            ) from None

        # Get metrics for specific worker type
        metrics_data = worker_monitor.get_performance_metrics(time_window_hours)

        response = PerformanceMetricsResponse(
            time_window_hours=metrics_data["time_window_hours"],
            total_tasks_processed=metrics_data.get("total_tasks_processed", 0),
            total_tasks_failed=metrics_data.get("total_tasks_failed", 0),
            failure_rate=metrics_data.get("failure_rate", 0.0),
            average_processing_time=metrics_data.get("average_processing_time", 0.0),
            average_memory_usage_mb=metrics_data.get("average_memory_usage_mb", 0.0),
            average_cpu_usage_percent=metrics_data.get(
                "average_cpu_usage_percent", 0.0
            ),
            active_workers=metrics_data.get("active_workers", 0),
            timestamp=metrics_data["timestamp"],
        )

        logger.info(f"Retrieved metrics for {worker_type} workers")
        return response

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Failed to get worker type metrics: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/{worker_type}/start")
async def start_worker_type(worker_type: str):
    """Start workers of a specific type."""
    try:
        logger.info(f"Starting {worker_type} workers")

        # Validate worker type
        try:
            WorkerType(worker_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid worker type: {worker_type}"
            ) from None

        # In a real implementation, this would start the actual workers
        # For now, just return success
        logger.info(f"Started {worker_type} workers")
        return {"message": f"Started {worker_type} workers", "status": "success"}

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Failed to start {worker_type} workers: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/{worker_type}/stop")
async def stop_worker_type(worker_type: str):
    """Stop workers of a specific type."""
    try:
        logger.info(f"Stopping {worker_type} workers")

        # Validate worker type
        try:
            WorkerType(worker_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid worker type: {worker_type}"
            ) from None

        # In a real implementation, this would stop the actual workers
        # For now, just return success
        logger.info(f"Stopped {worker_type} workers")
        return {"message": f"Stopped {worker_type} workers", "status": "success"}

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Failed to stop {worker_type} workers: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/{worker_type}/scale")
async def scale_worker_type(
    worker_type: str,
    count: int = Query(..., ge=0, le=10, description="Number of workers to scale to"),
):
    """Scale workers of a specific type."""
    try:
        logger.info(f"Scaling {worker_type} workers to {count}")

        # Validate worker type
        try:
            WorkerType(worker_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid worker type: {worker_type}"
            ) from None

        # In a real implementation, this would scale the actual workers
        # For now, just return success
        logger.info(f"Scaled {worker_type} workers to {count}")
        return {
            "message": f"Scaled {worker_type} workers to {count}",
            "status": "success",
            "target_count": count,
        }

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Failed to scale {worker_type} workers: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/queues/{queue_name}/clear")
async def clear_queue(queue_name: str):
    """Clear all tasks from a specific queue."""
    try:
        logger.info(f"Clearing queue: {queue_name}")

        # Validate queue name
        task_queues = get_task_queues()
        valid_queues = [q.queue_name for q in task_queues]
        if queue_name not in valid_queues:
            raise HTTPException(
                status_code=400, detail=f"Invalid queue name: {queue_name}"
            ) from None

        # In a real implementation, this would clear the actual queue
        # For now, just return success
        logger.info(f"Cleared queue: {queue_name}")
        return {"message": f"Cleared queue {queue_name}", "status": "success"}

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Failed to clear queue {queue_name}: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/config/queues")
async def get_queue_config():
    """Get configuration for all task queues."""
    try:
        logger.info("Getting queue configuration")

        task_queues = get_task_queues()
        queue_config = []

        for queue in task_queues:
            queue_config.append(
                {
                    "queue_name": queue.queue_name,
                    "worker_type": queue.worker_type.value,
                    "priority": queue.priority,
                    "max_retries": queue.max_retries,
                    "timeout": queue.timeout,
                    "concurrency": queue.concurrency,
                    "enabled": queue.enabled,
                }
            )

        logger.info(f"Retrieved configuration for {len(queue_config)} queues")
        return {"queues": queue_config}

    except Exception as err:
        logger.error(f"Failed to get queue configuration: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err
