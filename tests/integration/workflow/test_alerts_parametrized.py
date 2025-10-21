import json
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from src.infrastructure.workflow.config import AlertConfig
from src.infrastructure.workflow.monitoring import WorkflowMonitoring

DATA_FILE = pathlib.Path("tests/data/golden/workflow_alert_cases.json")


@pytest.fixture(scope="module")
def alert_cases():
    return json.loads(DATA_FILE.read_text())


@pytest.fixture()
def workflow_monitoring(workflow_config):
    return WorkflowMonitoring(workflow_config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "duration,threshold,should_alert",
    [
        (case["duration"], case["threshold"], case["should_alert"])
        for case in json.loads(DATA_FILE.read_text())["duration_alerts"]
    ],
)
async def test_duration_alerts(
    workflow_monitoring: WorkflowMonitoring, duration, threshold, should_alert
):
    flow_id = "flow-test-duration"
    cfg = AlertConfig(
        flow_id=flow_id,
        alert_type="duration",
        threshold=threshold,
        enabled=True,
        webhook_url="https://example.com/webhook",
    )
    await workflow_monitoring.create_flow_alerts(flow_id, cfg)

    with patch.object(workflow_monitoring, "_get_flow_duration", return_value=duration):
        with patch.object(
            workflow_monitoring, "_send_webhook_alert", new=AsyncMock()
        ) as send:
            await workflow_monitoring._check_alerts(flow_id)
            assert send.await_count == (1 if should_alert else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,should_alert",
    [
        (case["status"], case["should_alert"])
        for case in json.loads(DATA_FILE.read_text())["status_alerts"]
    ],
)
async def test_failure_status_alerts(
    workflow_monitoring: WorkflowMonitoring, status, should_alert
):
    flow_id = "flow-test-status"
    cfg = AlertConfig(
        flow_id=flow_id,
        alert_type="failure",
        enabled=True,
        webhook_url="https://example.com/webhook",
    )
    await workflow_monitoring.create_flow_alerts(flow_id, cfg)

    with patch.object(workflow_monitoring, "_get_flow_status", return_value=status):
        with patch.object(
            workflow_monitoring, "_send_webhook_alert", new=AsyncMock()
        ) as send:
            await workflow_monitoring._check_alerts(flow_id)
            assert send.await_count == (1 if should_alert else 0)
