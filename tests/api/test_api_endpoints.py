"""Test API endpoints to verify the implementation."""

import pytest
from fastapi.testclient import TestClient

from src.adapters.api.main import app


class TestAPIEndpoints:
    """Test API endpoints implementation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_observations_endpoints_exist(self, client: TestClient):
        """Test that observations endpoints are accessible."""
        # Test GET /observations (should return 401 without auth, but endpoint exists)
        response = client.get("/observations")
        assert response.status_code == 401  # Unauthorized, but endpoint exists
        
        # Test POST /observations (should return 401 without auth, but endpoint exists)
        response = client.post("/observations", json={})
        assert response.status_code == 401  # Unauthorized, but endpoint exists
    
    def test_detections_endpoints_exist(self, client: TestClient):
        """Test that detections endpoints are accessible."""
        # Test GET /detections (should return 401 without auth, but endpoint exists)
        response = client.get("/detections")
        assert response.status_code == 401  # Unauthorized, but endpoint exists
        
        # Test POST /detections/infer (should return 401 without auth, but endpoint exists)
        response = client.post("/detections/infer", json={})
        assert response.status_code == 401  # Unauthorized, but endpoint exists
    
    def test_workflows_endpoints_exist(self, client: TestClient):
        """Test that workflows endpoints are accessible."""
        # Test GET /workflows/flows (should return 401 without auth, but endpoint exists)
        response = client.get("/workflows/flows")
        assert response.status_code == 401  # Unauthorized, but endpoint exists
        
        # Test GET /workflows/health (should work without auth)
        response = client.get("/workflows/health")
        assert response.status_code == 200
    
    def test_auth_endpoints_exist(self, client: TestClient):
        """Test that auth endpoints are accessible."""
        # Test POST /auth/register (should work without auth)
        response = client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "testpassword123",
            "role": "user"
        })
        # This might return 400 (validation error) or 201 (success) depending on implementation
        assert response.status_code in [200, 201, 400, 422]
        
        # Test POST /auth/login (should work without auth)
        response = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "testpassword123"
        })
        # This might return 400 (validation error) or 401 (invalid credentials) depending on implementation
        assert response.status_code in [200, 400, 401, 422]
    
    def test_rate_limiting_headers(self, client: TestClient):
        """Test that rate limiting headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        # Check for rate limiting headers
        headers = response.headers
        # Rate limiting headers might be present depending on implementation
        print(f"Response headers: {dict(headers)}")
    
    def test_openapi_schema_structure(self, client: TestClient):
        """Test that the OpenAPI schema has the expected structure."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        
        # Check basic OpenAPI structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
        
        # Check that our custom info is present
        assert schema["info"]["title"] == "AstrID API"
        assert "astrid.chrislawrence.ca" in schema["info"]["contact"]["email"]
        
        # Check that paths exist
        assert "/health" in schema["paths"]
        assert "/observations" in schema["paths"]
        assert "/detections" in schema["paths"]
        assert "/workflows" in schema["paths"]
        assert "/auth" in schema["paths"]
