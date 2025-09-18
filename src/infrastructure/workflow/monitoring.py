"""Workflow monitoring and alerting system."""

import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx
from prefect import get_client

from .config import AlertConfig, WorkflowConfig

logger = logging.getLogger(__name__)


class WorkflowMonitoring:
    """Monitors workflow execution and manages alerts."""

    def __init__(self, config: WorkflowConfig) -> None:
        """Initialize workflow monitoring.

        Args:
            config: Workflow configuration
        """
        self.config = config
        self._client: Any | None = None
        self._alerts: dict[str, AlertConfig] = {}
        self._monitoring_task: asyncio.Task | None = None

    async def setup_flow_monitoring(self, flow_id: str) -> None:
        """Setup monitoring for a specific flow.

        Args:
            flow_id: Flow identifier
        """
        logger.info(f"Setting up monitoring for flow: {flow_id}")

        # Create default alert configuration
        alert_config = AlertConfig(flow_id=flow_id, alert_type="failure", enabled=True)

        self._alerts[flow_id] = alert_config
        logger.info(f"Monitoring setup completed for flow: {flow_id}")

    async def create_flow_alerts(self, flow_id: str, alert_config: AlertConfig) -> None:
        """Create alerts for a specific flow.

        Args:
            flow_id: Flow identifier
            alert_config: Alert configuration
        """
        logger.info(f"Creating alerts for flow: {flow_id}")

        # Store alert configuration
        self._alerts[flow_id] = alert_config

        # Setup monitoring if not already done
        if flow_id not in self._alerts:
            await self.setup_flow_monitoring(flow_id)

        logger.info(f"Alerts created for flow: {flow_id}")

    async def monitor_flow_performance(self, flow_id: str) -> dict[str, Any]:
        """Monitor performance metrics for a flow.

        Args:
            flow_id: Flow identifier

        Returns:
            Performance metrics
        """
        try:
            if not self._client:
                self._client = get_client()

            # Get flow run
            flow_run = await self._client.read_flow_run(flow_id)

            # Calculate performance metrics
            start_time = flow_run.start_time or flow_run.created
            end_time = flow_run.end_time or datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Get task run metrics
            task_runs = await self._client.read_task_runs(flow_run_id=flow_id)

            metrics = {
                "flow_id": flow_id,
                "status": flow_run.state.type,
                "duration_seconds": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat() if flow_run.end_time else None,
                "task_count": len(task_runs),
                "successful_tasks": len(
                    [tr for tr in task_runs if tr.state.type == "COMPLETED"]
                ),
                "failed_tasks": len(
                    [tr for tr in task_runs if tr.state.type == "FAILED"]
                ),
                "retry_count": sum(tr.run_count - 1 for tr in task_runs),
                "memory_usage": self._estimate_memory_usage(task_runs),
                "cpu_usage": self._estimate_cpu_usage(task_runs),
            }

            logger.debug(f"Performance metrics for {flow_id}: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to monitor flow performance {flow_id}: {e}")
            return {"flow_id": flow_id, "error": str(e)}

    async def alert_on_failure(self, flow_id: str, error: Exception) -> None:
        """Send alert when flow fails.

        Args:
            flow_id: Flow identifier
            error: Error that caused failure
        """
        logger.info(f"Sending failure alert for flow: {flow_id}")

        try:
            alert_config = self._alerts.get(flow_id)
            if not alert_config or not alert_config.enabled:
                return

            # Prepare alert message
            alert_message = {
                "flow_id": flow_id,
                "alert_type": "failure",
                "error_message": str(error),
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "high",
            }

            # Send alert via webhook if configured
            if alert_config.webhook_url:
                await self._send_webhook_alert(alert_config.webhook_url, alert_message)

            # Send email alert if configured
            if alert_config.email_recipients:
                await self._send_email_alert(
                    alert_config.email_recipients,
                    f"Flow Failure Alert: {flow_id}",
                    alert_message,
                )

            logger.info(f"Failure alert sent for flow: {flow_id}")

        except Exception as e:
            logger.error(f"Failed to send failure alert for {flow_id}: {e}")

    async def generate_flow_reports(
        self, flow_id: str, time_range: tuple[datetime, datetime]
    ) -> dict[str, Any]:
        """Generate comprehensive flow reports.

        Args:
            flow_id: Flow identifier
            time_range: Time range for report (start, end)

        Returns:
            Flow report data
        """
        logger.info(f"Generating flow report for {flow_id}")

        try:
            if not self._client:
                self._client = get_client()

            start_time, end_time = time_range

            # Get flow runs in time range
            flow_runs = await self._client.read_flow_runs(
                flow_id=flow_id, start_time=start_time, end_time=end_time
            )

            # Calculate statistics
            total_runs = len(flow_runs)
            successful_runs = len(
                [fr for fr in flow_runs if fr.state.type == "COMPLETED"]
            )
            failed_runs = len([fr for fr in flow_runs if fr.state.type == "FAILED"])
            success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0

            # Calculate average duration
            durations = []
            for fr in flow_runs:
                if fr.start_time and fr.end_time:
                    duration = (fr.end_time - fr.start_time).total_seconds()
                    durations.append(duration)

            avg_duration = sum(durations) / len(durations) if durations else 0

            # Get performance metrics for each run
            performance_metrics = []
            for fr in flow_runs:
                metrics = await self.monitor_flow_performance(fr.id)
                performance_metrics.append(metrics)

            report = {
                "flow_id": flow_id,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "summary": {
                    "total_runs": total_runs,
                    "successful_runs": successful_runs,
                    "failed_runs": failed_runs,
                    "success_rate": success_rate,
                    "average_duration_seconds": avg_duration,
                },
                "performance_metrics": performance_metrics,
                "generated_at": datetime.utcnow().isoformat(),
            }

            logger.info(f"Flow report generated for {flow_id}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate flow report for {flow_id}: {e}")
            return {"flow_id": flow_id, "error": str(e)}

    async def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        logger.info("Starting workflow monitoring")

        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already running")
            return

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Workflow monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        logger.info("Stopping workflow monitoring")

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("Workflow monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                # Monitor all configured flows
                for flow_id in self._alerts.keys():
                    await self._check_flow_health(flow_id)

                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _check_flow_health(self, flow_id: str) -> None:
        """Check health of a specific flow.

        Args:
            flow_id: Flow identifier
        """
        try:
            if not self._client:
                self._client = get_client()

            # Get flow run
            flow_run = await self._client.read_flow_run(flow_id)

            # Check for failures
            if flow_run.state.type == "FAILED":
                await self.alert_on_failure(flow_id, Exception(flow_run.state.message))

            # Check for timeouts
            if flow_run.state.type == "RUNNING":
                start_time = flow_run.start_time or flow_run.created
                duration = (datetime.utcnow() - start_time).total_seconds()

                if duration > self.config.flow_timeout:
                    await self.alert_on_failure(
                        flow_id, Exception(f"Flow timeout after {duration} seconds")
                    )

            # Check performance thresholds
            alert_config = self._alerts.get(flow_id)
            if alert_config and alert_config.threshold:
                metrics = await self.monitor_flow_performance(flow_id)
                if metrics.get("duration_seconds", 0) > alert_config.threshold:
                    await self._send_performance_alert(flow_id, metrics)

        except Exception as e:
            logger.error(f"Error checking flow health {flow_id}: {e}")

    async def _send_webhook_alert(
        self, webhook_url: str, message: dict[str, Any]
    ) -> None:
        """Send alert via webhook.

        Args:
            webhook_url: Webhook URL
            message: Alert message
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=message, timeout=10.0)
                response.raise_for_status()
                logger.debug(f"Webhook alert sent to {webhook_url}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    async def _send_email_alert(
        self, recipients: list[str], subject: str, message: dict[str, Any]
    ) -> None:
        """Send alert via email.

        Args:
            recipients: Email recipients
            subject: Email subject
            message: Alert message
        """
        # Note: In production, you would integrate with an email service
        # For now, we'll just log the alert
        logger.info(f"Email alert would be sent to {recipients}: {subject}")
        logger.info(f"Message: {message}")

    async def _send_performance_alert(
        self, flow_id: str, metrics: dict[str, Any]
    ) -> None:
        """Send performance alert.

        Args:
            flow_id: Flow identifier
            metrics: Performance metrics
        """
        logger.info(f"Performance alert for flow {flow_id}: {metrics}")

        alert_config = self._alerts.get(flow_id)
        if not alert_config or not alert_config.enabled:
            return

        alert_message = {
            "flow_id": flow_id,
            "alert_type": "performance",
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "medium",
        }

        # Send alert via webhook if configured
        if alert_config.webhook_url:
            await self._send_webhook_alert(alert_config.webhook_url, alert_message)

    def _estimate_memory_usage(self, task_runs: list[Any]) -> float:
        """Estimate memory usage from task runs.

        Args:
            task_runs: List of task runs

        Returns:
            Estimated memory usage in MB
        """
        # Simple estimation based on task count
        # In production, you would get actual memory usage from task metadata
        return len(task_runs) * 100.0  # 100MB per task

    def _estimate_cpu_usage(self, task_runs: list[Any]) -> float:
        """Estimate CPU usage from task runs.

        Args:
            task_runs: List of task runs

        Returns:
            Estimated CPU usage percentage
        """
        # Simple estimation based on task count
        # In production, you would get actual CPU usage from task metadata
        return min(len(task_runs) * 10.0, 100.0)  # 10% per task, max 100%

    async def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        await self.stop_monitoring()
        if self._client:
            await self._client.close()
            self._client = None
