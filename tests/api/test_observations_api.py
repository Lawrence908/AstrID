"""Comprehensive API tests for observations endpoints."""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.api.main import app
from src.domains.observations.schema import ObservationCreate, ObservationStatus
# Using db_session from main conftest.py
from tests.mocks.supabase import MockSupabaseClient


class TestObservationsAPI:
    """Test suite for observations API endpoints."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    async def authenticated_client(self, client: AsyncClient):
        """Create authenticated test client."""
        # Mock authentication
        client.headers.update({"Authorization": "Bearer test-token"})
        return client
    
    @pytest.fixture
    def sample_observation_data(self):
        """Sample observation data for testing."""
        return {
            "survey": "ZTF",
            "observation_id": "ZTF_20230101_000000",
            "ra": 180.5,
            "dec": 45.2,
            "observation_time": "2025-01-01T00:00:00Z",
            "filter_band": "r",
            "exposure_time": 30.0
        }
    
    async def test_create_observation_success(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test successful observation creation."""
        response = await authenticated_client.post("/observations", json=sample_observation_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["survey"] == sample_observation_data["survey"]
        assert data["data"]["observation_id"] == sample_observation_data["observation_id"]
        assert data["data"]["ra"] == sample_observation_data["ra"]
        assert data["data"]["dec"] == sample_observation_data["dec"]
    
    async def test_create_observation_validation_error(self, authenticated_client: AsyncClient):
        """Test observation creation with validation errors."""
        invalid_data = {
            "survey": "ZTF",
            "observation_id": "ZTF_20230101_000000",
            "ra": 400.0,  # Invalid RA (should be 0-360)
            "dec": 100.0,  # Invalid DEC (should be -90 to 90)
            "observation_time": "2025-01-01T00:00:00Z",
            "filter_band": "r",
            "exposure_time": 30.0
        }
        
        response = await authenticated_client.post("/observations", json=invalid_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["error_code"] == "VALIDATION_ERROR"
    
    async def test_get_observation_success(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test successful observation retrieval."""
        # First create an observation
        create_response = await authenticated_client.post("/observations", json=sample_observation_data)
        assert create_response.status_code == 201
        observation_id = create_response.json()["data"]["id"]
        
        # Then retrieve it
        response = await authenticated_client.get(f"/observations/{observation_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == observation_id
        assert data["data"]["survey"] == sample_observation_data["survey"]
    
    async def test_get_observation_not_found(self, authenticated_client: AsyncClient):
        """Test observation retrieval with non-existent ID."""
        response = await authenticated_client.get("/observations/00000000-0000-0000-0000-000000000000")
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "RESOURCE_NOT_FOUND"
    
    async def test_list_observations_success(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test successful observation listing."""
        # Create multiple observations
        for i in range(3):
            data = sample_observation_data.copy()
            data["observation_id"] = f"ZTF_20230101_00000{i}"
            await authenticated_client.post("/observations", json=data)
        
        # List observations
        response = await authenticated_client.get("/observations")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert len(data["data"]) >= 3
    
    async def test_list_observations_with_pagination(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test observation listing with pagination."""
        # Create multiple observations
        for i in range(5):
            data = sample_observation_data.copy()
            data["observation_id"] = f"ZTF_20230101_00000{i}"
            await authenticated_client.post("/observations", json=data)
        
        # List with pagination
        response = await authenticated_client.get("/observations?page=1&size=2")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "pagination" in data
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["size"] == 2
        assert len(data["data"]) == 2
    
    async def test_list_observations_with_filtering(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test observation listing with filtering."""
        # Create observations with different surveys
        ztf_data = sample_observation_data.copy()
        ztf_data["survey"] = "ZTF"
        await authenticated_client.post("/observations", json=ztf_data)
        
        lsst_data = sample_observation_data.copy()
        lsst_data["survey"] = "LSST"
        lsst_data["observation_id"] = "LSST_20230101_000000"
        await authenticated_client.post("/observations", json=lsst_data)
        
        # Filter by survey
        response = await authenticated_client.get("/observations?survey=ZTF")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert all(obs["survey"] == "ZTF" for obs in data["data"])
    
    async def test_update_observation_status_success(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test successful observation status update."""
        # Create an observation
        create_response = await authenticated_client.post("/observations", json=sample_observation_data)
        observation_id = create_response.json()["data"]["id"]
        
        # Update status
        status_update = {"status": "preprocessing"}
        response = await authenticated_client.put(f"/observations/{observation_id}/status", json=status_update)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "preprocessing"
    
    async def test_search_observations_by_coordinates(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test observation search by coordinates."""
        # Create an observation
        await authenticated_client.post("/observations", json=sample_observation_data)
        
        # Search by coordinates
        search_params = {
            "ra_min": 180.0,
            "ra_max": 181.0,
            "dec_min": 45.0,
            "dec_max": 46.0
        }
        response = await authenticated_client.get("/observations/search", params=search_params)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1
    
    async def test_get_observation_metrics(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test observation metrics retrieval."""
        # Create an observation
        create_response = await authenticated_client.post("/observations", json=sample_observation_data)
        observation_id = create_response.json()["data"]["id"]
        
        # Get metrics
        response = await authenticated_client.get(f"/observations/{observation_id}/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "metrics" in data["data"]
    
    async def test_unauthorized_access(self, client: AsyncClient, sample_observation_data: dict):
        """Test unauthorized access to protected endpoints."""
        response = await client.post("/observations", json=sample_observation_data)
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_rate_limiting(self, authenticated_client: AsyncClient, sample_observation_data: dict):
        """Test rate limiting functionality."""
        # Make many requests quickly
        responses = []
        for _ in range(150):  # Exceed the 100/hour limit for observations
            response = await authenticated_client.post("/observations", json=sample_observation_data)
            responses.append(response)
        
        # Check that some requests were rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0
        
        # Check rate limit headers
        for response in rate_limited_responses:
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers


class TestObservationsAPIIntegration:
    """Integration tests for observations API."""
    
    async def test_observation_workflow(self, authenticated_client: AsyncClient):
        """Test complete observation workflow."""
        # 1. Create observation
        observation_data = {
            "survey": "ZTF",
            "observation_id": "ZTF_20230101_000000",
            "ra": 180.5,
            "dec": 45.2,
            "observation_time": "2025-01-01T00:00:00Z",
            "filter_band": "r",
            "exposure_time": 30.0
        }
        
        create_response = await authenticated_client.post("/observations", json=observation_data)
        assert create_response.status_code == 201
        observation_id = create_response.json()["data"]["id"]
        
        # 2. Update status to preprocessing
        status_response = await authenticated_client.put(
            f"/observations/{observation_id}/status",
            json={"status": "preprocessing"}
        )
        assert status_response.status_code == 200
        
        # 3. Update status to differencing
        status_response = await authenticated_client.put(
            f"/observations/{observation_id}/status",
            json={"status": "differencing"}
        )
        assert status_response.status_code == 200
        
        # 4. Update status to detection
        status_response = await authenticated_client.put(
            f"/observations/{observation_id}/status",
            json={"status": "detection"}
        )
        assert status_response.status_code == 200
        
        # 5. Update status to completed
        status_response = await authenticated_client.put(
            f"/observations/{observation_id}/status",
            json={"status": "completed"}
        )
        assert status_response.status_code == 200
        
        # 6. Verify final state
        get_response = await authenticated_client.get(f"/observations/{observation_id}")
        assert get_response.status_code == 200
        assert get_response.json()["data"]["status"] == "completed"
    
    async def test_observation_error_handling(self, authenticated_client: AsyncClient):
        """Test error handling in observation operations."""
        # Test invalid observation ID format
        response = await authenticated_client.get("/observations/invalid-id")
        assert response.status_code == 422  # Validation error
        
        # Test invalid status update
        observation_data = {
            "survey": "ZTF",
            "observation_id": "ZTF_20230101_000000",
            "ra": 180.5,
            "dec": 45.2,
            "observation_time": "2025-01-01T00:00:00Z",
            "filter_band": "r",
            "exposure_time": 30.0
        }
        
        create_response = await authenticated_client.post("/observations", json=observation_data)
        observation_id = create_response.json()["data"]["id"]
        
        # Try invalid status
        status_response = await authenticated_client.put(
            f"/observations/{observation_id}/status",
            json={"status": "invalid_status"}
        )
        assert status_response.status_code == 400
