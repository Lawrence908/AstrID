"""Comprehensive API tests for authentication endpoints."""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.api.main import app
# Using db_session from main conftest.py
from tests.mocks.supabase import MockSupabaseClient


class TestAuthAPI:
    """Test suite for authentication API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_auth_data(self):
        """Sample authentication data for testing."""
        return {
            "email": "test@example.com",
            "password": "testpassword123",
            "role": "user"
        }
    
    def test_register_user_success(self, client: TestClient, sample_auth_data: dict):
        """Test successful user registration."""
        response = client.post("/auth/register", json=sample_auth_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "user_id" in data["data"]
        assert data["data"]["email"] == sample_auth_data["email"]
        assert "access_token" in data["data"]
        assert "refresh_token" in data["data"]
    
    async def test_register_user_validation_error(self, client: AsyncClient):
        """Test user registration with validation errors."""
        invalid_data = {
            "email": "invalid-email",  # Invalid email format
            "password": "123",  # Password too short
            "role": "invalid_role"  # Invalid role
        }
        
        response = await client.post("/auth/register", json=invalid_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["error_code"] == "VALIDATION_ERROR"
    
    async def test_register_user_duplicate_email(self, client: AsyncClient, sample_auth_data: dict):
        """Test user registration with duplicate email."""
        # First registration
        response1 = await client.post("/auth/register", json=sample_auth_data)
        assert response1.status_code == 201
        
        # Second registration with same email
        response2 = await client.post("/auth/register", json=sample_auth_data)
        
        assert response2.status_code == 409
        data = response2.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "RESOURCE_CONFLICT"
    
    async def test_login_success(self, client: AsyncClient, sample_auth_data: dict):
        """Test successful user login."""
        # First register a user
        await client.post("/auth/register", json=sample_auth_data)
        
        # Then login
        login_data = {
            "email": sample_auth_data["email"],
            "password": sample_auth_data["password"]
        }
        response = await client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "access_token" in data["data"]
        assert "refresh_token" in data["data"]
        assert "user" in data["data"]
    
    async def test_login_invalid_credentials(self, client: AsyncClient):
        """Test login with invalid credentials."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        
        response = await client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_refresh_token_success(self, client: AsyncClient, sample_auth_data: dict):
        """Test successful token refresh."""
        # First register and login
        await client.post("/auth/register", json=sample_auth_data)
        login_response = await client.post("/auth/login", json={
            "email": sample_auth_data["email"],
            "password": sample_auth_data["password"]
        })
        refresh_token = login_response.json()["data"]["refresh_token"]
        
        # Then refresh token
        response = await client.post("/auth/refresh", json={"refresh_token": refresh_token})
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "access_token" in data["data"]
        assert "refresh_token" in data["data"]
    
    async def test_refresh_token_invalid(self, client: AsyncClient):
        """Test token refresh with invalid token."""
        response = await client.post("/auth/refresh", json={"refresh_token": "invalid_token"})
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_logout_success(self, client: AsyncClient, sample_auth_data: dict):
        """Test successful user logout."""
        # First register and login
        await client.post("/auth/register", json=sample_auth_data)
        login_response = await client.post("/auth/login", json={
            "email": sample_auth_data["email"],
            "password": sample_auth_data["password"]
        })
        access_token = login_response.json()["data"]["access_token"]
        
        # Then logout
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.post("/auth/logout", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    async def test_logout_unauthorized(self, client: AsyncClient):
        """Test logout without authentication."""
        response = await client.post("/auth/logout")
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_get_user_profile_success(self, client: AsyncClient, sample_auth_data: dict):
        """Test successful user profile retrieval."""
        # First register and login
        await client.post("/auth/register", json=sample_auth_data)
        login_response = await client.post("/auth/login", json={
            "email": sample_auth_data["email"],
            "password": sample_auth_data["password"]
        })
        access_token = login_response.json()["data"]["access_token"]
        
        # Then get profile
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.get("/auth/profile", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["email"] == sample_auth_data["email"]
        assert data["data"]["role"] == sample_auth_data["role"]
    
    async def test_get_user_profile_unauthorized(self, client: AsyncClient):
        """Test user profile retrieval without authentication."""
        response = await client.get("/auth/profile")
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_update_user_profile_success(self, client: AsyncClient, sample_auth_data: dict):
        """Test successful user profile update."""
        # First register and login
        await client.post("/auth/register", json=sample_auth_data)
        login_response = await client.post("/auth/login", json={
            "email": sample_auth_data["email"],
            "password": sample_auth_data["password"]
        })
        access_token = login_response.json()["data"]["access_token"]
        
        # Then update profile
        update_data = {
            "name": "Test User",
            "organization": "Test Organization"
        }
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.put("/auth/profile", json=update_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == update_data["name"]
        assert data["data"]["organization"] == update_data["organization"]
    
    async def test_change_password_success(self, client: AsyncClient, sample_auth_data: dict):
        """Test successful password change."""
        # First register and login
        await client.post("/auth/register", json=sample_auth_data)
        login_response = await client.post("/auth/login", json={
            "email": sample_auth_data["email"],
            "password": sample_auth_data["password"]
        })
        access_token = login_response.json()["data"]["access_token"]
        
        # Then change password
        password_data = {
            "current_password": sample_auth_data["password"],
            "new_password": "newpassword123"
        }
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.put("/auth/password", json=password_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    async def test_change_password_invalid_current(self, client: AsyncClient, sample_auth_data: dict):
        """Test password change with invalid current password."""
        # First register and login
        await client.post("/auth/register", json=sample_auth_data)
        login_response = await client.post("/auth/login", json={
            "email": sample_auth_data["email"],
            "password": sample_auth_data["password"]
        })
        access_token = login_response.json()["data"]["access_token"]
        
        # Then try to change password with wrong current password
        password_data = {
            "current_password": "wrongpassword",
            "new_password": "newpassword123"
        }
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.put("/auth/password", json=password_data, headers=headers)
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_reset_password_request_success(self, client: AsyncClient, sample_auth_data: dict):
        """Test successful password reset request."""
        # First register a user
        await client.post("/auth/register", json=sample_auth_data)
        
        # Then request password reset
        reset_data = {"email": sample_auth_data["email"]}
        response = await client.post("/auth/reset-password", json=reset_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data["data"]
    
    async def test_reset_password_request_invalid_email(self, client: AsyncClient):
        """Test password reset request with invalid email."""
        reset_data = {"email": "nonexistent@example.com"}
        response = await client.post("/auth/reset-password", json=reset_data)
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "RESOURCE_NOT_FOUND"
    
    async def test_verify_email_success(self, client: AsyncClient, sample_auth_data: dict):
        """Test successful email verification."""
        # First register a user
        register_response = await client.post("/auth/register", json=sample_auth_data)
        verification_token = register_response.json()["data"]["verification_token"]
        
        # Then verify email
        response = await client.post("/auth/verify-email", json={"token": verification_token})
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    async def test_verify_email_invalid_token(self, client: AsyncClient):
        """Test email verification with invalid token."""
        response = await client.post("/auth/verify-email", json={"token": "invalid_token"})
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "AUTHENTICATION_ERROR"
    
    async def test_rate_limiting(self, client: AsyncClient, sample_auth_data: dict):
        """Test rate limiting functionality."""
        # Make many login attempts quickly
        responses = []
        for _ in range(150):  # Exceed the 100/hour limit for auth
            response = await client.post("/auth/login", json={
                "email": sample_auth_data["email"],
                "password": "wrongpassword"
            })
            responses.append(response)
        
        # Check that some requests were rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0


class TestAuthAPIIntegration:
    """Integration tests for authentication API."""
    
    async def test_complete_auth_workflow(self, client: AsyncClient):
        """Test complete authentication workflow."""
        # 1. Register user
        auth_data = {
            "email": "test@example.com",
            "password": "testpassword123",
            "role": "user"
        }
        
        register_response = await client.post("/auth/register", json=auth_data)
        assert register_response.status_code == 201
        access_token = register_response.json()["data"]["access_token"]
        
        # 2. Get profile
        headers = {"Authorization": f"Bearer {access_token}"}
        profile_response = await client.get("/auth/profile", headers=headers)
        assert profile_response.status_code == 200
        
        # 3. Update profile
        update_data = {"name": "Test User"}
        update_response = await client.put("/auth/profile", json=update_data, headers=headers)
        assert update_response.status_code == 200
        
        # 4. Change password
        password_data = {
            "current_password": auth_data["password"],
            "new_password": "newpassword123"
        }
        password_response = await client.put("/auth/password", json=password_data, headers=headers)
        assert password_response.status_code == 200
        
        # 5. Login with new password
        login_response = await client.post("/auth/login", json={
            "email": auth_data["email"],
            "password": "newpassword123"
        })
        assert login_response.status_code == 200
        
        # 6. Logout
        new_access_token = login_response.json()["data"]["access_token"]
        new_headers = {"Authorization": f"Bearer {new_access_token}"}
        logout_response = await client.post("/auth/logout", headers=new_headers)
        assert logout_response.status_code == 200
    
    async def test_auth_error_handling(self, client: AsyncClient):
        """Test error handling in authentication operations."""
        # Test invalid email format
        response = await client.post("/auth/register", json={
            "email": "invalid-email",
            "password": "testpassword123",
            "role": "user"
        })
        assert response.status_code == 400
        
        # Test weak password
        response = await client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "123",
            "role": "user"
        })
        assert response.status_code == 400
        
        # Test invalid role
        response = await client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "testpassword123",
            "role": "invalid_role"
        })
        assert response.status_code == 400
