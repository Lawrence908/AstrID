"""Workflow API endpoints for flow management and monitoring."""

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from src.core.dependencies import get_workflow_config

from ..config import AlertConfig, FlowStatus, FlowType, WorkflowConfig
from ..monitoring import WorkflowMonitoring
from ..prefect_server import PrefectServer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])


class FlowStartRequest(BaseModel):
    """Request to start a workflow."""

    flow_type: FlowType
    parameters: dict[str, Any]
    priority: int = 1


class FlowStatusResponse(BaseModel):
    """Response for flow status."""

    flow_id: str
    flow_type: FlowType
    status: FlowStatus
    start_time: datetime
    end_time: datetime | None = None
    progress: float = 0.0
    current_step: str = ""
    error_message: str | None = None
    metrics: dict[str, Any] | None = None


class FlowLogResponse(BaseModel):
    """Response for flow logs."""

    flow_id: str
    logs: list[dict[str, Any]]
    total_count: int


class FlowMetricsResponse(BaseModel):
    """Response for flow metrics."""

    flow_id: str
    metrics: dict[str, Any]


class FlowReportResponse(BaseModel):
    """Response for flow report."""

    flow_id: str
    report: dict[str, Any]


class AlertConfigRequest(BaseModel):
    """Request to configure flow alerts."""

    flow_id: str
    alert_type: str
    threshold: float | None = None
    webhook_url: str | None = None
    email_recipients: list[str] | None = None
    enabled: bool = True


# Dependency injection
async def get_prefect_server(
    config: WorkflowConfig = Depends(get_workflow_config),
) -> PrefectServer:
    """Get Prefect server instance."""
    return PrefectServer(config)


async def get_workflow_monitoring(
    config: WorkflowConfig = Depends(get_workflow_config),
) -> WorkflowMonitoring:
    """Get workflow monitoring instance."""
    return WorkflowMonitoring(config)


@router.get("/flows", response_model=list[FlowStatusResponse])
async def get_flows(
    status: FlowStatus | None = None,
    flow_type: FlowType | None = None,
    limit: int = 100,
    offset: int = 0,
    prefect_server: PrefectServer = Depends(get_prefect_server),
) -> list[FlowStatusResponse]:
    """Get list of workflows with optional filtering.

    Args:
        status: Filter by flow status
        flow_type: Filter by flow type
        limit: Maximum number of flows to return
        offset: Number of flows to skip

    Returns:
        List of flow status information
    """
    try:
        # Note: In production, you would implement proper filtering
        # For now, we'll return a placeholder response
        logger.info(f"Getting flows: status={status}, type={flow_type}, limit={limit}")

        # This would typically query Prefect for flow runs
        # For now, return empty list
        return []

    except Exception as e:
        logger.error(f"Failed to get flows: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/flows/{flow_type}/start", response_model=FlowStatusResponse)
async def start_flow(
    flow_type: FlowType,
    request: FlowStartRequest,
    background_tasks: BackgroundTasks,
    prefect_server: PrefectServer = Depends(get_prefect_server),
) -> FlowStatusResponse:
    """Start a new workflow.

    Args:
        flow_type: Type of flow to start
        request: Flow start request
        background_tasks: Background tasks

    Returns:
        Flow status information
    """
    try:
        logger.info(f"Starting flow: {flow_type}")

        # Start flow in background
        flow_id = f"{flow_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # In production, you would start the actual Prefect flow
        # For now, we'll return a mock response
        response = FlowStatusResponse(
            flow_id=flow_id,
            flow_type=flow_type,
            status=FlowStatus.PENDING,
            start_time=datetime.utcnow(),
            progress=0.0,
            current_step="Initializing",
            metrics={},
        )

        logger.info(f"Flow started: {flow_id}")
        return response

    except Exception as e:
        logger.error(f"Failed to start flow {flow_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/flows/{flow_id}/status", response_model=FlowStatusResponse)
async def get_flow_status(
    flow_id: str, prefect_server: PrefectServer = Depends(get_prefect_server)
) -> FlowStatusResponse:
    """Get status of a specific flow.

    Args:
        flow_id: Flow identifier
        prefect_server: Prefect server instance

    Returns:
        Flow status information
    """
    try:
        logger.info(f"Getting flow status: {flow_id}")

        # Get flow status from Prefect
        status_info = await prefect_server.get_flow_status(flow_id)

        response = FlowStatusResponse(
            flow_id=status_info.flow_id,
            flow_type=status_info.flow_type,
            status=status_info.status,
            start_time=status_info.start_time,
            end_time=status_info.end_time,
            progress=status_info.progress,
            current_step=status_info.current_step,
            error_message=status_info.error_message,
            metrics=status_info.metrics,
        )

        logger.info(f"Flow status retrieved: {flow_id}")
        return response

    except Exception as e:
        logger.error(f"Failed to get flow status {flow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/flows/{flow_id}/cancel")
async def cancel_flow(
    flow_id: str, prefect_server: PrefectServer = Depends(get_prefect_server)
) -> dict[str, Any]:
    """Cancel a running flow.

    Args:
        flow_id: Flow identifier
        prefect_server: Prefect server instance

    Returns:
        Cancellation result
    """
    try:
        logger.info(f"Cancelling flow: {flow_id}")

        # Cancel flow
        success = await prefect_server.cancel_flow(flow_id)

        if success:
            logger.info(f"Flow cancelled: {flow_id}")
            return {"flow_id": flow_id, "status": "cancelled"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel flow")

    except Exception as e:
        logger.error(f"Failed to cancel flow {flow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/flows/{flow_id}/logs", response_model=FlowLogResponse)
async def get_flow_logs(
    flow_id: str,
    limit: int = 100,
    prefect_server: PrefectServer = Depends(get_prefect_server),
) -> FlowLogResponse:
    """Get logs for a specific flow.

    Args:
        flow_id: Flow identifier
        limit: Maximum number of log entries
        prefect_server: Prefect server instance

    Returns:
        Flow logs
    """
    try:
        logger.info(f"Getting flow logs: {flow_id}")

        # Get flow logs from Prefect
        logs = await prefect_server.get_flow_logs(flow_id, limit)

        response = FlowLogResponse(flow_id=flow_id, logs=logs, total_count=len(logs))

        logger.info(f"Flow logs retrieved: {flow_id}")
        return response

    except Exception as e:
        logger.error(f"Failed to get flow logs {flow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/flows/{flow_id}/metrics", response_model=FlowMetricsResponse)
async def get_flow_metrics(
    flow_id: str, monitoring: WorkflowMonitoring = Depends(get_workflow_monitoring)
) -> FlowMetricsResponse:
    """Get performance metrics for a specific flow.

    Args:
        flow_id: Flow identifier
        monitoring: Workflow monitoring instance

    Returns:
        Flow metrics
    """
    try:
        logger.info(f"Getting flow metrics: {flow_id}")

        # Get flow metrics
        metrics = await monitoring.monitor_flow_performance(flow_id)

        response = FlowMetricsResponse(flow_id=flow_id, metrics=metrics)

        logger.info(f"Flow metrics retrieved: {flow_id}")
        return response

    except Exception as e:
        logger.error(f"Failed to get flow metrics {flow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/flows/{flow_id}/retry")
async def retry_flow(
    flow_id: str, prefect_server: PrefectServer = Depends(get_prefect_server)
) -> dict[str, Any]:
    """Retry a failed flow.

    Args:
        flow_id: Flow identifier
        prefect_server: Prefect server instance

    Returns:
        Retry result
    """
    try:
        logger.info(f"Retrying flow: {flow_id}")

        # Retry flow
        success = await prefect_server.retry_flow(flow_id)

        if success:
            logger.info(f"Flow retry initiated: {flow_id}")
            return {"flow_id": flow_id, "status": "retrying"}
        else:
            raise HTTPException(status_code=400, detail="Failed to retry flow")

    except Exception as e:
        logger.error(f"Failed to retry flow {flow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/flows/{flow_id}/alerts")
async def configure_flow_alerts(
    flow_id: str,
    request: AlertConfigRequest,
    monitoring: WorkflowMonitoring = Depends(get_workflow_monitoring),
) -> dict[str, Any]:
    """Configure alerts for a specific flow.

    Args:
        flow_id: Flow identifier
        request: Alert configuration request
        monitoring: Workflow monitoring instance

    Returns:
        Alert configuration result
    """
    try:
        logger.info(f"Configuring alerts for flow: {flow_id}")

        # Create alert configuration
        alert_config = AlertConfig(
            flow_id=flow_id,
            alert_type=request.alert_type,
            threshold=request.threshold,
            webhook_url=request.webhook_url,
            email_recipients=request.email_recipients or [],
            enabled=request.enabled,
        )

        # Configure alerts
        await monitoring.create_flow_alerts(flow_id, alert_config)

        logger.info(f"Alerts configured for flow: {flow_id}")
        return {"flow_id": flow_id, "status": "configured"}

    except Exception as e:
        logger.error(f"Failed to configure alerts for flow {flow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/flows/{flow_id}/report", response_model=FlowReportResponse)
async def get_flow_report(
    flow_id: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    monitoring: WorkflowMonitoring = Depends(get_workflow_monitoring),
) -> FlowReportResponse:
    """Get comprehensive report for a specific flow.

    Args:
        flow_id: Flow identifier
        start_time: Report start time
        end_time: Report end time
        monitoring: Workflow monitoring instance

    Returns:
        Flow report
    """
    try:
        logger.info(f"Getting flow report: {flow_id}")

        # Set default time range if not provided
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=7)
        if not end_time:
            end_time = datetime.utcnow()

        # Generate flow report
        report = await monitoring.generate_flow_reports(flow_id, (start_time, end_time))

        response = FlowReportResponse(flow_id=flow_id, report=report)

        logger.info(f"Flow report generated: {flow_id}")
        return response

    except Exception as e:
        logger.error(f"Failed to get flow report {flow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health")
async def health_check(
    prefect_server: PrefectServer = Depends(get_prefect_server),
) -> dict[str, Any]:
    """Check workflow system health.

    Args:
        prefect_server: Prefect server instance

    Returns:
        Health status
    """
    try:
        logger.info("Checking workflow system health")

        # Check Prefect server health
        health = await prefect_server.health_check()

        logger.info("Workflow system health check completed")
        return health

    except Exception as e:
        logger.error(f"Workflow system health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
