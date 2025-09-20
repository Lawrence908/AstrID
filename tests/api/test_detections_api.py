"""Comprehensive API tests for detections endpoints."""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.api.main import app
# Using db_session from main conftest.py
from tests.mocks.supabase import MockSupabaseClient


class TestDetectionsAPI:
    """Test suite for detections API endpoints."""
    
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
    def sample_detection_data(self):
        """Sample detection data for testing."""
        return {
            "observation_id": "123e4567-e89b-12d3-a456-426614174000",
            "ra": 180.5,
            "dec": 45.2,
            "confidence": 0.95,
            "magnitude": 18.5,
            "model_version": "v1.0.0"
        }
    
    async def test_run_inference_success(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test successful ML inference."""
        response = await authenticated_client.post("/detections/infer", json=sample_detection_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "detection_id" in data["data"]
        assert data["data"]["confidence"] == sample_detection_data["confidence"]
        assert data["data"]["ra"] == sample_detection_data["ra"]
        assert data["data"]["dec"] == sample_detection_data["dec"]
    
    async def test_run_inference_validation_error(self, authenticated_client: AsyncClient):
        """Test inference with validation errors."""
        invalid_data = {
            "observation_id": "invalid-uuid",
            "ra": 400.0,  # Invalid RA
            "dec": 100.0,  # Invalid DEC
            "confidence": 1.5,  # Invalid confidence (> 1.0)
            "magnitude": 18.5,
            "model_version": "v1.0.0"
        }
        
        response = await authenticated_client.post("/detections/infer", json=invalid_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["error_code"] == "VALIDATION_ERROR"
    
    async def test_get_detection_success(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test successful detection retrieval."""
        # First run inference
        infer_response = await authenticated_client.post("/detections/infer", json=sample_detection_data)
        detection_id = infer_response.json()["data"]["detection_id"]
        
        # Then retrieve it
        response = await authenticated_client.get(f"/detections/{detection_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == detection_id
        assert data["data"]["confidence"] == sample_detection_data["confidence"]
    
    async def test_get_detection_not_found(self, authenticated_client: AsyncClient):
        """Test detection retrieval with non-existent ID."""
        response = await authenticated_client.get("/detections/00000000-0000-0000-0000-000000000000")
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "RESOURCE_NOT_FOUND"
    
    async def test_list_detections_success(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test successful detection listing."""
        # Create multiple detections
        for i in range(3):
            data = sample_detection_data.copy()
            data["observation_id"] = f"123e4567-e89b-12d3-a456-42661417400{i}"
            await authenticated_client.post("/detections/infer", json=data)
        
        # List detections
        response = await authenticated_client.get("/detections")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert len(data["data"]) >= 3
    
    async def test_list_detections_with_filtering(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test detection listing with filtering."""
        # Create detections with different confidence levels
        high_conf_data = sample_detection_data.copy()
        high_conf_data["confidence"] = 0.95
        await authenticated_client.post("/detections/infer", json=high_conf_data)
        
        low_conf_data = sample_detection_data.copy()
        low_conf_data["confidence"] = 0.5
        low_conf_data["observation_id"] = "123e4567-e89b-12d3-a456-426614174001"
        await authenticated_client.post("/detections/infer", json=low_conf_data)
        
        # Filter by confidence
        response = await authenticated_client.get("/detections?min_confidence=0.9")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert all(det["confidence"] >= 0.9 for det in data["data"])
    
    async def test_validate_detection_success(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test successful detection validation."""
        # First run inference
        infer_response = await authenticated_client.post("/detections/infer", json=sample_detection_data)
        detection_id = infer_response.json()["data"]["detection_id"]
        
        # Then validate it
        validation_data = {
            "status": "validated",
            "validator_notes": "Confirmed anomaly",
            "validator_id": "curator_123"
        }
        response = await authenticated_client.put(f"/detections/{detection_id}/validate", json=validation_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "validated"
    
    async def test_get_detection_statistics(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test detection statistics retrieval."""
        # Create some detections first
        for i in range(5):
            data = sample_detection_data.copy()
            data["observation_id"] = f"123e4567-e89b-12d3-a456-42661417400{i}"
            await authenticated_client.post("/detections/infer", json=data)
        
        # Get statistics
        response = await authenticated_client.get("/detections/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "statistics" in data["data"]
        assert "total_detections" in data["data"]["statistics"]
        assert "average_confidence" in data["data"]["statistics"]
    
    async def test_detection_confidence_scoring(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test detection confidence scoring."""
        # Test with different confidence levels
        confidence_levels = [0.5, 0.7, 0.9, 0.95]
        
        for conf in confidence_levels:
            data = sample_detection_data.copy()
            data["confidence"] = conf
            data["observation_id"] = f"123e4567-e89b-12d3-a456-42661417400{int(conf*100)}"
            
            response = await authenticated_client.post("/detections/infer", json=data)
            assert response.status_code == 200
            
            detection_data = response.json()["data"]
            assert detection_data["confidence"] == conf
    
    async def test_detection_model_versioning(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test detection with different model versions."""
        model_versions = ["v1.0.0", "v1.1.0", "v2.0.0"]
        
        for version in model_versions:
            data = sample_detection_data.copy()
            data["model_version"] = version
            data["observation_id"] = f"123e4567-e89b-12d3-a456-42661417400{version.replace('.', '')}"
            
            response = await authenticated_client.post("/detections/infer", json=data)
            assert response.status_code == 200
            
            detection_data = response.json()["data"]
            assert "model_version" in detection_data
    
    async def test_detection_batch_processing(self, authenticated_client: AsyncClient):
        """Test batch detection processing."""
        batch_data = {
            "observations": [
                {
                    "observation_id": "123e4567-e89b-12d3-a456-426614174000",
                    "ra": 180.5,
                    "dec": 45.2,
                    "confidence": 0.95,
                    "magnitude": 18.5,
                    "model_version": "v1.0.0"
                },
                {
                    "observation_id": "123e4567-e89b-12d3-a456-426614174001",
                    "ra": 181.0,
                    "dec": 46.0,
                    "confidence": 0.85,
                    "magnitude": 19.0,
                    "model_version": "v1.0.0"
                }
            ]
        }
        
        response = await authenticated_client.post("/detections/batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert len(data["data"]) == 2
    
    async def test_unauthorized_access(self, client: AsyncClient, sample_detection_data: dict):
        """Test unauthorized access to protected endpoints."""
        response = await client.post("/detections/infer", json=sample_detection_data)
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_rate_limiting(self, authenticated_client: AsyncClient, sample_detection_data: dict):
        """Test rate limiting functionality."""
        # Make many requests quickly
        responses = []
        for _ in range(250):  # Exceed the 200/hour limit for detections
            response = await authenticated_client.post("/detections/infer", json=sample_detection_data)
            responses.append(response)
        
        # Check that some requests were rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0


class TestDetectionsAPIIntegration:
    """Integration tests for detections API."""
    
    async def test_detection_workflow(self, authenticated_client: AsyncClient):
        """Test complete detection workflow."""
        # 1. Run inference
        detection_data = {
            "observation_id": "123e4567-e89b-12d3-a456-426614174000",
            "ra": 180.5,
            "dec": 45.2,
            "confidence": 0.95,
            "magnitude": 18.5,
            "model_version": "v1.0.0"
        }
        
        infer_response = await authenticated_client.post("/detections/infer", json=detection_data)
        assert infer_response.status_code == 200
        detection_id = infer_response.json()["data"]["detection_id"]
        
        # 2. Validate detection
        validation_response = await authenticated_client.put(
            f"/detections/{detection_id}/validate",
            json={"status": "validated", "validator_notes": "Confirmed"}
        )
        assert validation_response.status_code == 200
        
        # 3. Get final detection
        get_response = await authenticated_client.get(f"/detections/{detection_id}")
        assert get_response.status_code == 200
        assert get_response.json()["data"]["status"] == "validated"
    
    async def test_detection_error_handling(self, authenticated_client: AsyncClient):
        """Test error handling in detection operations."""
        # Test invalid detection ID format
        response = await authenticated_client.get("/detections/invalid-id")
        assert response.status_code == 422  # Validation error
        
        # Test invalid validation status
        detection_data = {
            "observation_id": "123e4567-e89b-12d3-a456-426614174000",
            "ra": 180.5,
            "dec": 45.2,
            "confidence": 0.95,
            "magnitude": 18.5,
            "model_version": "v1.0.0"
        }
        
        infer_response = await authenticated_client.post("/detections/infer", json=detection_data)
        detection_id = infer_response.json()["data"]["detection_id"]
        
        # Try invalid validation status
        validation_response = await authenticated_client.put(
            f"/detections/{detection_id}/validate",
            json={"status": "invalid_status"}
        )
        assert validation_response.status_code == 400
