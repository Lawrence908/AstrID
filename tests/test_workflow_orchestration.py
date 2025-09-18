"""Tests for workflow orchestration system."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.infrastructure.storage.config import StorageConfig
from src.infrastructure.workflow.config import (
    AlertConfig,
    FlowStatus,
    FlowStatusInfo,
    FlowType,
    WorkflowConfig,
)
from src.infrastructure.workflow.monitoring import WorkflowMonitoring
from src.infrastructure.workflow.prefect_server import PrefectServer


@pytest.fixture
def storage_config():
    """Create storage configuration for testing."""
    return StorageConfig(
        r2_account_id="test-account",
        r2_access_key_id="test-key",
        r2_secret_access_key="test-secret",
        r2_bucket_name="test-bucket",
        r2_endpoint_url="https://test.r2.cloudflarestorage.com",
        dvc_remote_url="https://test-dvc.com",
        mlflow_artifact_root="s3://test-mlflow",
    )


@pytest.fixture
def workflow_config(storage_config):
    """Create workflow configuration for testing."""
    return WorkflowConfig(
        prefect_server_url="http://localhost:4200",
        database_url="postgresql+asyncpg://localhost/test",
        storage_config=storage_config,
        authentication_enabled=False,
        monitoring_enabled=True,
        alerting_enabled=True,
        max_concurrent_flows=5,
        flow_timeout=1800,
        retry_attempts=2,
    )


@pytest.fixture
def prefect_server(workflow_config):
    """Create Prefect server instance for testing."""
    return PrefectServer(workflow_config)


@pytest.fixture
def workflow_monitoring(workflow_config):
    """Create workflow monitoring instance for testing."""
    return WorkflowMonitoring(workflow_config)


class TestPrefectServer:
    """Test Prefect server functionality."""

    @pytest.mark.asyncio
    async def test_start_server(self, prefect_server):
        """Test starting Prefect server."""
        with patch(
            "src.infrastructure.workflow.prefect_server.get_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            await prefect_server.start_server("localhost", 4200)

            assert prefect_server._client is not None
            mock_get_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_configure_database(self, prefect_server):
        """Test database configuration."""
        await prefect_server.configure_database("postgresql://test")
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_setup_authentication(self, prefect_server):
        """Test authentication setup."""
        await prefect_server.setup_authentication()
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_configure_storage(self, prefect_server, storage_config):
        """Test storage configuration."""
        await prefect_server.configure_storage(storage_config)
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_setup_monitoring(self, prefect_server):
        """Test monitoring setup."""
        await prefect_server.setup_monitoring()
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_health_check_success(self, prefect_server):
        """Test successful health check."""
        with patch(
            "src.infrastructure.workflow.prefect_server.httpx.AsyncClient"
        ) as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            with patch.object(prefect_server, "_client") as mock_prefect_client:
                mock_work_pool = Mock()
                mock_work_pool.name = "test-pool"
                mock_prefect_client.read_work_pool = AsyncMock(
                    return_value=mock_work_pool
                )

                health = await prefect_server.health_check()

                assert health["status"] == "healthy"
                assert "server_url" in health
                assert "work_pool" in health

    @pytest.mark.asyncio
    async def test_health_check_failure(self, prefect_server):
        """Test health check failure."""
        with patch(
            "src.infrastructure.workflow.prefect_server.httpx.AsyncClient"
        ) as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                Exception("Connection failed")
            )

            health = await prefect_server.health_check()

            assert health["status"] == "unhealthy"
            assert "error" in health

    @pytest.mark.asyncio
    async def test_get_flow_status(self, prefect_server):
        """Test getting flow status."""
        flow_id = str(uuid4())

        with patch.object(prefect_server, "_client") as mock_client:
            mock_flow_run = Mock()
            mock_flow_run.state.type = "COMPLETED"
            mock_flow_run.start_time = datetime.utcnow()
            mock_flow_run.end_time = datetime.utcnow()
            mock_flow_run.created = datetime.utcnow()
            mock_flow_run.state.message = None
            mock_client.read_flow_run = AsyncMock(return_value=mock_flow_run)

            status = await prefect_server.get_flow_status(flow_id)

            assert status.flow_id == flow_id
            assert status.status == FlowStatus.COMPLETED
            assert status.start_time is not None

    @pytest.mark.asyncio
    async def test_cancel_flow(self, prefect_server):
        """Test cancelling a flow."""
        flow_id = str(uuid4())

        with patch.object(prefect_server, "_client") as mock_client:
            mock_client.cancel_flow_run = AsyncMock(return_value=None)

            result = await prefect_server.cancel_flow(flow_id)

            assert result is True
            mock_client.cancel_flow_run.assert_called_once_with(flow_id)

    @pytest.mark.asyncio
    async def test_retry_flow(self, prefect_server):
        """Test retrying a flow."""
        flow_id = str(uuid4())

        with patch.object(prefect_server, "_client") as mock_client:
            mock_client.retry_flow_run = AsyncMock(return_value=None)

            result = await prefect_server.retry_flow(flow_id)

            assert result is True
            mock_client.retry_flow_run.assert_called_once_with(flow_id)

    @pytest.mark.asyncio
    async def test_get_flow_logs(self, prefect_server):
        """Test getting flow logs."""
        flow_id = str(uuid4())

        with patch.object(prefect_server, "_client") as mock_client:
            mock_log = Mock()
            mock_log.timestamp = datetime.utcnow()
            mock_log.level = "INFO"
            mock_log.message = "Test log message"
            mock_log.name = "test_task"
            mock_client.read_logs = AsyncMock(return_value=[mock_log])

            logs = await prefect_server.get_flow_logs(flow_id, 10)

            assert len(logs) == 1
            assert logs[0]["message"] == "Test log message"
            assert logs[0]["level"] == "INFO"


class TestWorkflowMonitoring:
    """Test workflow monitoring functionality."""

    @pytest.mark.asyncio
    async def test_setup_flow_monitoring(self, workflow_monitoring):
        """Test setting up flow monitoring."""
        flow_id = str(uuid4())

        await workflow_monitoring.setup_flow_monitoring(flow_id)

        assert flow_id in workflow_monitoring._alerts
        assert workflow_monitoring._alerts[flow_id].enabled is True

    @pytest.mark.asyncio
    async def test_create_flow_alerts(self, workflow_monitoring):
        """Test creating flow alerts."""
        flow_id = str(uuid4())
        alert_config = AlertConfig(
            flow_id=flow_id,
            alert_type="failure",
            threshold=300.0,
            webhook_url="https://example.com/webhook",
            email_recipients=["test@example.com"],
            enabled=True,
        )

        await workflow_monitoring.create_flow_alerts(flow_id, alert_config)

        assert flow_id in workflow_monitoring._alerts
        assert workflow_monitoring._alerts[flow_id] == alert_config

    @pytest.mark.asyncio
    async def test_monitor_flow_performance(self, workflow_monitoring):
        """Test monitoring flow performance."""
        flow_id = str(uuid4())

        with patch.object(workflow_monitoring, "_client") as mock_client:
            mock_flow_run = Mock()
            mock_flow_run.start_time = datetime.utcnow()
            mock_flow_run.end_time = datetime.utcnow()
            mock_flow_run.created = datetime.utcnow()
            mock_flow_run.state.type = "COMPLETED"
            mock_client.read_flow_run = AsyncMock(return_value=mock_flow_run)

            mock_task_run = Mock()
            mock_task_run.state.type = "COMPLETED"
            mock_task_run.run_count = 1
            mock_client.read_task_runs = AsyncMock(return_value=[mock_task_run])

            metrics = await workflow_monitoring.monitor_flow_performance(flow_id)

            assert metrics["flow_id"] == flow_id
            assert metrics["status"] == "COMPLETED"
            assert "duration_seconds" in metrics
            assert "task_count" in metrics

    @pytest.mark.asyncio
    async def test_alert_on_failure(self, workflow_monitoring):
        """Test alerting on flow failure."""
        flow_id = str(uuid4())
        error = Exception("Test error")

        # Setup alert configuration
        alert_config = AlertConfig(
            flow_id=flow_id,
            alert_type="failure",
            enabled=True,
            webhook_url="https://example.com/webhook",
        )
        workflow_monitoring._alerts[flow_id] = alert_config

        with patch.object(workflow_monitoring, "_send_webhook_alert") as mock_webhook:
            await workflow_monitoring.alert_on_failure(flow_id, error)

            mock_webhook.assert_called_once()
            call_args = mock_webhook.call_args[0]
            assert call_args[0] == "https://example.com/webhook"
            assert call_args[1]["flow_id"] == flow_id
            assert call_args[1]["alert_type"] == "failure"

    @pytest.mark.asyncio
    async def test_generate_flow_reports(self, workflow_monitoring):
        """Test generating flow reports."""
        flow_id = str(uuid4())
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()

        with patch.object(workflow_monitoring, "_client") as mock_client:
            mock_flow_run = Mock()
            mock_flow_run.id = flow_id
            mock_flow_run.state.type = "COMPLETED"
            mock_flow_run.start_time = start_time
            mock_flow_run.end_time = end_time
            mock_client.read_flow_runs = AsyncMock(return_value=[mock_flow_run])

            with patch.object(
                workflow_monitoring, "monitor_flow_performance"
            ) as mock_perf:
                mock_perf.return_value = {"duration_seconds": 100.0}

                report = await workflow_monitoring.generate_flow_reports(
                    flow_id, (start_time, end_time)
                )

                assert report["flow_id"] == flow_id
                assert "summary" in report
                assert "performance_metrics" in report
                assert report["summary"]["total_runs"] == 1
                assert report["summary"]["successful_runs"] == 1

    @pytest.mark.asyncio
    async def test_start_monitoring(self, workflow_monitoring):
        """Test starting monitoring."""
        await workflow_monitoring.start_monitoring()

        assert workflow_monitoring._monitoring_task is not None
        assert not workflow_monitoring._monitoring_task.done()

        # Clean up
        await workflow_monitoring.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, workflow_monitoring):
        """Test stopping monitoring."""
        await workflow_monitoring.start_monitoring()
        await workflow_monitoring.stop_monitoring()

        assert workflow_monitoring._monitoring_task is None


class TestWorkflowConfig:
    """Test workflow configuration."""

    def test_workflow_config_creation(self, storage_config):
        """Test creating workflow configuration."""
        config = WorkflowConfig(
            prefect_server_url="http://localhost:4200",
            database_url="postgresql://localhost/test",
            storage_config=storage_config,
        )

        assert config.prefect_server_url == "http://localhost:4200"
        assert config.database_url == "postgresql://localhost/test"
        assert config.storage_config == storage_config
        assert config.authentication_enabled is True
        assert config.monitoring_enabled is True
        assert config.alerting_enabled is True
        assert config.max_concurrent_flows == 10
        assert config.flow_timeout == 3600
        assert config.retry_attempts == 3

    def test_workflow_config_defaults(self, storage_config):
        """Test workflow configuration defaults."""
        config = WorkflowConfig(
            prefect_server_url="http://localhost:4200",
            database_url="postgresql://localhost/test",
            storage_config=storage_config,
        )

        assert config.prefect_api_url == "http://localhost:4200/api"
        assert config.prefect_ui_url == "http://localhost:4200/ui"
        assert config.prefect_work_pool == "default"
        assert config.prefect_work_queue == "default"
        assert config.metrics_retention_days == 30
        assert config.alert_email_recipients == []


class TestFlowTypes:
    """Test flow type enumeration."""

    def test_flow_type_values(self):
        """Test flow type values."""
        assert FlowType.OBSERVATION_INGESTION.value == "observation_ingestion"
        assert FlowType.OBSERVATION_PREPROCESSING.value == "observation_preprocessing"
        assert FlowType.OBSERVATION_DIFFERENCING.value == "observation_differencing"
        assert FlowType.OBSERVATION_DETECTION.value == "observation_detection"
        assert FlowType.OBSERVATION_VALIDATION.value == "observation_validation"
        assert FlowType.MODEL_TRAINING.value == "model_training"
        assert FlowType.MODEL_EVALUATION.value == "model_evaluation"
        assert FlowType.MODEL_DEPLOYMENT.value == "model_deployment"
        assert FlowType.MODEL_RETRAINING.value == "model_retraining"
        assert (
            FlowType.HYPERPARAMETER_OPTIMIZATION.value == "hyperparameter_optimization"
        )
        assert FlowType.SYSTEM_MAINTENANCE.value == "system_maintenance"


class TestFlowStatus:
    """Test flow status enumeration."""

    def test_flow_status_values(self):
        """Test flow status values."""
        assert FlowStatus.PENDING.value == "PENDING"
        assert FlowStatus.RUNNING.value == "RUNNING"
        assert FlowStatus.COMPLETED.value == "COMPLETED"
        assert FlowStatus.FAILED.value == "FAILED"
        assert FlowStatus.CANCELLED.value == "CANCELLED"
        assert FlowStatus.RETRYING.value == "RETRYING"


class TestFlowStatusInfo:
    """Test flow status information."""

    def test_flow_status_info_creation(self):
        """Test creating flow status info."""
        flow_id = str(uuid4())
        start_time = datetime.utcnow()

        status_info = FlowStatusInfo(
            flow_id=flow_id,
            flow_type=FlowType.OBSERVATION_INGESTION,
            status=FlowStatus.RUNNING,
            start_time=start_time,
        )

        assert status_info.flow_id == flow_id
        assert status_info.flow_type == FlowType.OBSERVATION_INGESTION
        assert status_info.status == FlowStatus.RUNNING
        assert status_info.start_time == start_time
        assert status_info.end_time is None
        assert status_info.progress == 0.0
        assert status_info.current_step == ""
        assert status_info.error_message is None
        assert status_info.metrics == {}


class TestAlertConfig:
    """Test alert configuration."""

    def test_alert_config_creation(self):
        """Test creating alert configuration."""
        alert_config = AlertConfig(
            flow_id="test-flow",
            alert_type="failure",
            threshold=300.0,
            webhook_url="https://example.com/webhook",
            email_recipients=["test@example.com"],
            enabled=True,
        )

        assert alert_config.flow_id == "test-flow"
        assert alert_config.alert_type == "failure"
        assert alert_config.threshold == 300.0
        assert alert_config.webhook_url == "https://example.com/webhook"
        assert alert_config.email_recipients == ["test@example.com"]
        assert alert_config.enabled is True

    def test_alert_config_defaults(self):
        """Test alert configuration defaults."""
        alert_config = AlertConfig(flow_id="test-flow", alert_type="failure")

        assert alert_config.threshold is None
        assert alert_config.webhook_url is None
        assert alert_config.email_recipients == []
        assert alert_config.enabled is True
