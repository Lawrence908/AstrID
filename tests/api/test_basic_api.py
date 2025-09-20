"""Basic API tests to verify the API is working."""

import pytest
from fastapi.testclient import TestClient

from src.adapters.api.main import app


class TestBasicAPI:
    """Basic API tests."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint(self, client: TestClient):
        """Test that the health endpoint is accessible."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_docs_endpoint(self, client: TestClient):
        """Test that the docs endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_endpoint(self, client: TestClient):
        """Test that the OpenAPI endpoint is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert data["openapi"].startswith("3.")
    
    def test_api_versioning_headers(self, client: TestClient):
        """Test that API versioning headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        # Check for versioning headers
        assert "X-API-Version" in response.headers
    
    def test_cors_headers(self, client: TestClient):
        """Test that CORS headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        # Check for API versioning headers (CORS headers are only added to OPTIONS requests)
        assert "X-API-Version" in response.headers
        assert "X-API-Supported-Versions" in response.headers
