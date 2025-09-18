"""Worker monitoring and metrics collection."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import psutil

from src.adapters.workers.config import WorkerMetrics, WorkerType, worker_manager
from src.core.logging import configure_domain_logger


class WorkerMonitor:
    """Monitor worker performance and health."""

    def __init__(self):
        self.logger = configure_domain_logger("workers.monitoring")
        self.metrics_history: dict[str, list[WorkerMetrics]] = {}
        self.start_time = datetime.now()

    def get_worker_metrics(self, worker_id: str) -> WorkerMetrics | None:
        """Get current metrics for a specific worker."""
        try:
            # Get system metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()

            # Calculate worker-specific metrics
            metrics = WorkerMetrics(
                worker_id=worker_id,
                worker_type=WorkerType.OBSERVATION_INGESTION,  # Default, would be determined by worker
                status="IDLE",  # Would be determined by actual worker state
                tasks_processed=0,  # Would be tracked by worker
                tasks_failed=0,  # Would be tracked by worker
                average_processing_time=0.0,  # Would be calculated from history
                memory_usage=memory_info.rss / 1024 / 1024,  # MB
                cpu_usage=cpu_percent,
                last_heartbeat=datetime.now().isoformat(),
            )

            # Store in history
            if worker_id not in self.metrics_history:
                self.metrics_history[worker_id] = []
            self.metrics_history[worker_id].append(metrics)

            # Keep only last 100 metrics per worker
            if len(self.metrics_history[worker_id]) > 100:
                self.metrics_history[worker_id] = self.metrics_history[worker_id][-100:]

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to get metrics for worker {worker_id}: {e}")
            return None

    def get_all_worker_metrics(self) -> dict[str, WorkerMetrics]:
        """Get metrics for all workers."""
        metrics = {}

        # In a real implementation, this would query all active workers
        # For now, return metrics for a single mock worker
        worker_id = "worker_1"
        worker_metrics = self.get_worker_metrics(worker_id)
        if worker_metrics:
            metrics[worker_id] = worker_metrics

        return metrics

    def get_worker_health(self) -> dict[str, Any]:
        """Get overall worker health status."""
        try:
            all_metrics = self.get_all_worker_metrics()

            if not all_metrics:
                return {
                    "status": "unhealthy",
                    "message": "No workers available",
                    "timestamp": datetime.now().isoformat(),
                }

            # Calculate health metrics
            total_workers = len(all_metrics)
            healthy_workers = sum(
                1 for m in all_metrics.values() if m.status != "ERROR"
            )
            total_tasks = sum(m.tasks_processed for m in all_metrics.values())
            total_failed = sum(m.tasks_failed for m in all_metrics.values())

            health_ratio = healthy_workers / total_workers if total_workers > 0 else 0
            failure_rate = total_failed / total_tasks if total_tasks > 0 else 0

            # Determine overall status
            if health_ratio >= 0.9 and failure_rate <= 0.05:
                status = "healthy"
            elif health_ratio >= 0.7 and failure_rate <= 0.1:
                status = "degraded"
            else:
                status = "unhealthy"

            return {
                "status": status,
                "total_workers": total_workers,
                "healthy_workers": healthy_workers,
                "health_ratio": health_ratio,
                "total_tasks_processed": total_tasks,
                "total_tasks_failed": total_failed,
                "failure_rate": failure_rate,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get worker health: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_queue_status(self) -> dict[str, Any]:
        """Get status of all task queues."""
        try:
            # Get queue status from worker manager
            queue_status = worker_manager.get_queue_status()

            # Add additional queue metrics
            queue_metrics = {
                "queues": queue_status.get("queues", []),
                "total_actors": queue_status.get("total_actors", 0),
                "broker_connected": queue_status.get("broker_connected", False),
                "queue_lengths": {},  # Would be populated from broker
                "processing_rates": {},  # Would be calculated from metrics
                "error_rates": {},  # Would be calculated from metrics
                "timestamp": datetime.now().isoformat(),
            }

            return queue_metrics

        except Exception as e:
            self.logger.error(f"Failed to get queue status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_performance_metrics(self, time_window_hours: int = 24) -> dict[str, Any]:
        """Get performance metrics for the specified time window."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

            # Filter metrics by time window
            recent_metrics = {}
            for worker_id, metrics_list in self.metrics_history.items():
                recent_metrics[worker_id] = [
                    m
                    for m in metrics_list
                    if datetime.fromisoformat(m.last_heartbeat) >= cutoff_time
                ]

            # Calculate performance metrics
            total_tasks = 0
            total_failed = 0
            total_processing_time = 0.0
            total_memory_usage = 0.0
            total_cpu_usage = 0.0

            for metrics_list in recent_metrics.values():
                if metrics_list:
                    latest = metrics_list[-1]
                    total_tasks += latest.tasks_processed
                    total_failed += latest.tasks_failed
                    total_processing_time += latest.average_processing_time
                    total_memory_usage += latest.memory_usage
                    total_cpu_usage += latest.cpu_usage

            worker_count = len([m for m in recent_metrics.values() if m])

            return {
                "time_window_hours": time_window_hours,
                "total_tasks_processed": total_tasks,
                "total_tasks_failed": total_failed,
                "failure_rate": total_failed / total_tasks if total_tasks > 0 else 0,
                "average_processing_time": total_processing_time / worker_count
                if worker_count > 0
                else 0,
                "average_memory_usage_mb": total_memory_usage / worker_count
                if worker_count > 0
                else 0,
                "average_cpu_usage_percent": total_cpu_usage / worker_count
                if worker_count > 0
                else 0,
                "active_workers": worker_count,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_worker_history(
        self, worker_id: str, limit: int = 100
    ) -> list[WorkerMetrics]:
        """Get historical metrics for a specific worker."""
        if worker_id not in self.metrics_history:
            return []

        return self.metrics_history[worker_id][-limit:]

    def clear_old_metrics(self, days_to_keep: int = 7) -> int:
        """Clear metrics older than specified days."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            cleared_count = 0

            for worker_id in list(self.metrics_history.keys()):
                original_count = len(self.metrics_history[worker_id])
                self.metrics_history[worker_id] = [
                    m
                    for m in self.metrics_history[worker_id]
                    if datetime.fromisoformat(m.last_heartbeat) >= cutoff_time
                ]
                cleared_count += original_count - len(self.metrics_history[worker_id])

                # Remove empty worker entries
                if not self.metrics_history[worker_id]:
                    del self.metrics_history[worker_id]

            self.logger.info(f"Cleared {cleared_count} old metrics entries")
            return cleared_count

        except Exception as e:
            self.logger.error(f"Failed to clear old metrics: {e}")
            return 0


# Global monitor instance
worker_monitor = WorkerMonitor()
