"""Comprehensive API tests for workflows endpoints."""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.api.main import app
# Using db_session from main conftest.py
from tests.mocks.supabase import MockSupabaseClient


class TestWorkflowsAPI:
    """Test suite for workflows API endpoints."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    async def authenticated_client(self, client: AsyncClient):
        """Create authenticated test client."""
        client.headers.update({"Authorization": "Bearer test-token"})
        return client
    
    @pytest.fixture
    def sample_flow_data(self):
        """Sample flow data for testing."""
        return {
            "flow_type": "observation_processing",
            "parameters": {
                "observation_id": "123e4567-e89b-12d3-a456-426614174000",
                "preprocessing_enabled": True,
                "differencing_enabled": True,
                "detection_enabled": True
            }
        }
    
    async def test_list_flows_success(self, authenticated_client: AsyncClient):
        """Test successful flow listing."""
        response = await authenticated_client.get("/workflows/flows")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], list)
    
    async def test_start_flow_success(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test successful flow start."""
        response = await authenticated_client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "flow_id" in data["data"]
        assert data["data"]["flow_type"] == sample_flow_data["flow_type"]
    
    async def test_start_flow_validation_error(self, authenticated_client: AsyncClient):
        """Test flow start with validation errors."""
        invalid_data = {
            "invalid_parameter": "invalid_value"
        }
        
        response = await authenticated_client.post(
            "/workflows/flows/observation_processing/start",
            json=invalid_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["error_code"] == "VALIDATION_ERROR"
    
    async def test_get_flow_status_success(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test successful flow status retrieval."""
        # First start a flow
        start_response = await authenticated_client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        flow_id = start_response.json()["data"]["flow_id"]
        
        # Then get its status
        response = await authenticated_client.get(f"/workflows/flows/{flow_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "status" in data["data"]
        assert "flow_id" in data["data"]
    
    async def test_get_flow_status_not_found(self, authenticated_client: AsyncClient):
        """Test flow status retrieval with non-existent ID."""
        response = await authenticated_client.get("/workflows/flows/00000000-0000-0000-0000-000000000000/status")
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "RESOURCE_NOT_FOUND"
    
    async def test_cancel_flow_success(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test successful flow cancellation."""
        # First start a flow
        start_response = await authenticated_client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        flow_id = start_response.json()["data"]["flow_id"]
        
        # Then cancel it
        response = await authenticated_client.post(f"/workflows/flows/{flow_id}/cancel")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["status"] == "cancelled"
    
    async def test_retry_flow_success(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test successful flow retry."""
        # First start a flow
        start_response = await authenticated_client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        flow_id = start_response.json()["data"]["flow_id"]
        
        # Then retry it
        response = await authenticated_client.post(f"/workflows/flows/{flow_id}/retry")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "flow_id" in data["data"]
    
    async def test_get_flow_logs_success(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test successful flow logs retrieval."""
        # First start a flow
        start_response = await authenticated_client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        flow_id = start_response.json()["data"]["flow_id"]
        
        # Then get its logs
        response = await authenticated_client.get(f"/workflows/flows/{flow_id}/logs")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "logs" in data["data"]
        assert isinstance(data["data"]["logs"], list)
    
    async def test_get_flow_metrics_success(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test successful flow metrics retrieval."""
        # First start a flow
        start_response = await authenticated_client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        flow_id = start_response.json()["data"]["flow_id"]
        
        # Then get its metrics
        response = await authenticated_client.get(f"/workflows/flows/{flow_id}/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "metrics" in data["data"]
        assert "execution_time" in data["data"]["metrics"]
        assert "memory_usage" in data["data"]["metrics"]
    
    async def test_get_flow_report_success(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test successful flow report retrieval."""
        # First start a flow
        start_response = await authenticated_client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        flow_id = start_response.json()["data"]["flow_id"]
        
        # Then get its report
        response = await authenticated_client.get(f"/workflows/flows/{flow_id}/report")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "report" in data["data"]
        assert "summary" in data["data"]["report"]
    
    async def test_configure_flow_alerts_success(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test successful flow alert configuration."""
        # First start a flow
        start_response = await authenticated_client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        flow_id = start_response.json()["data"]["flow_id"]
        
        # Then configure alerts
        alert_config = {
            "enabled": True,
            "email_notifications": True,
            "webhook_url": "https://example.com/webhook",
            "alert_conditions": {
                "on_failure": True,
                "on_completion": False,
                "on_timeout": True
            }
        }
        
        response = await authenticated_client.post(
            f"/workflows/flows/{flow_id}/alerts",
            json=alert_config
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["alerts_enabled"] is True
    
    async def test_get_workflow_health_success(self, authenticated_client: AsyncClient):
        """Test successful workflow health check."""
        response = await authenticated_client.get("/workflows/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "status" in data["data"]
        assert "services" in data["data"]
    
    async def test_flow_parameter_validation(self, authenticated_client: AsyncClient):
        """Test flow parameter validation."""
        # Test missing required parameters
        response = await authenticated_client.post(
            "/workflows/flows/observation_processing/start",
            json={}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    async def test_flow_type_validation(self, authenticated_client: AsyncClient):
        """Test flow type validation."""
        response = await authenticated_client.post(
            "/workflows/flows/invalid_flow_type/start",
            json={"observation_id": "123e4567-e89b-12d3-a456-426614174000"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    async def test_unauthorized_access(self, client: AsyncClient, sample_flow_data: dict):
        """Test unauthorized access to protected endpoints."""
        response = await client.post(
            f"/workflows/flows/{sample_flow_data['flow_type']}/start",
            json=sample_flow_data["parameters"]
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_rate_limiting(self, authenticated_client: AsyncClient, sample_flow_data: dict):
        """Test rate limiting functionality."""
        # Make many requests quickly
        responses = []
        for _ in range(60):  # Exceed the 50/hour limit for workflows
            response = await authenticated_client.post(
                f"/workflows/flows/{sample_flow_data['flow_type']}/start",
                json=sample_flow_data["parameters"]
            )
            responses.append(response)
        
        # Check that some requests were rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0


class TestWorkflowsAPIIntegration:
    """Integration tests for workflows API."""
    
    async def test_complete_workflow_lifecycle(self, authenticated_client: AsyncClient):
        """Test complete workflow lifecycle."""
        # 1. Start a flow
        flow_data = {
            "flow_type": "observation_processing",
            "parameters": {
                "observation_id": "123e4567-e89b-12d3-a456-426614174000",
                "preprocessing_enabled": True,
                "differencing_enabled": True,
                "detection_enabled": True
            }
        }
        
        start_response = await authenticated_client.post(
            f"/workflows/flows/{flow_data['flow_type']}/start",
            json=flow_data["parameters"]
        )
        assert start_response.status_code == 200
        flow_id = start_response.json()["data"]["flow_id"]
        
        # 2. Check status
        status_response = await authenticated_client.get(f"/workflows/flows/{flow_id}/status")
        assert status_response.status_code == 200
        
        # 3. Get logs
        logs_response = await authenticated_client.get(f"/workflows/flows/{flow_id}/logs")
        assert logs_response.status_code == 200
        
        # 4. Get metrics
        metrics_response = await authenticated_client.get(f"/workflows/flows/{flow_id}/metrics")
        assert metrics_response.status_code == 200
        
        # 5. Get report
        report_response = await authenticated_client.get(f"/workflows/flows/{flow_id}/report")
        assert report_response.status_code == 200
        
        # 6. Cancel flow
        cancel_response = await authenticated_client.post(f"/workflows/flows/{flow_id}/cancel")
        assert cancel_response.status_code == 200
    
    async def test_workflow_error_handling(self, authenticated_client: AsyncClient):
        """Test error handling in workflow operations."""
        # Test invalid flow ID format
        response = await authenticated_client.get("/workflows/flows/invalid-id/status")
        assert response.status_code == 422  # Validation error
        
        # Test operations on non-existent flow
        response = await authenticated_client.post("/workflows/flows/00000000-0000-0000-0000-000000000000/cancel")
        assert response.status_code == 404
