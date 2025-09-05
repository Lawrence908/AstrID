"""Supabase authentication adapter for AstrID."""

import logging
from datetime import datetime
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from supabase import Client, create_client

from src.core.constants import (
    SUPABASE_JWT_SECRET,
    SUPABASE_KEY,
    SUPABASE_URL,
)

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Supabase client
supabase: Client | None = None


def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    global supabase
    if supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase URL and anon key must be configured")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase


class User(BaseModel):
    """User model from Supabase auth."""

    id: str
    email: str
    created_at: datetime
    updated_at: datetime
    email_confirmed_at: datetime | None = None
    last_sign_in_at: datetime | None = None
    role: str | None = None
    metadata: dict[str, Any] | None = None


class AuthService:
    """Authentication service using Supabase."""

    def __init__(self) -> None:
        self.client = get_supabase_client()
        self.jwt_secret = SUPABASE_JWT_SECRET

    async def verify_token(self, token: str) -> dict[str, Any]:
        """Verify JWT token and return payload."""
        try:
            if not self.jwt_secret:
                raise ValueError("JWT secret not configured")

            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                options={"verify_exp": True},
            )
            return payload  # type: ignore[no-any-return]
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            ) from None
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            ) from None

    async def get_user_from_token(self, token: str) -> User:
        """Get user information from JWT token."""
        payload = await self.verify_token(token)

        # Extract user info from JWT payload
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
            )

        # Get user from Supabase
        try:
            response = self.client.auth.admin.get_user_by_id(user_id)
            if not response.user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )

            user_data = response.user
            return User(
                id=user_data.id,
                email=user_data.email,
                created_at=datetime.fromisoformat(
                    user_data.created_at.replace("Z", "+00:00")
                ),
                updated_at=datetime.fromisoformat(
                    user_data.updated_at.replace("Z", "+00:00")
                ),
                email_confirmed_at=(
                    datetime.fromisoformat(
                        user_data.email_confirmed_at.replace("Z", "+00:00")
                    )
                    if user_data.email_confirmed_at
                    else None
                ),
                last_sign_in_at=(
                    datetime.fromisoformat(
                        user_data.last_sign_in_at.replace("Z", "+00:00")
                    )
                    if user_data.last_sign_in_at
                    else None
                ),
                role=(
                    user_data.user_metadata.get("role")
                    if user_data.user_metadata
                    else None
                ),
                metadata=user_data.user_metadata,
            )
        except Exception as e:
            logger.error(f"Error getting user from Supabase: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving user information",
            ) from e

    async def sign_up(
        self, email: str, password: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Sign up a new user."""
        try:
            response = self.client.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                    "options": {"data": metadata or {}},
                }
            )
            return response  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error signing up user: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Sign up failed: {str(e)}",
            ) from e

    async def sign_in(self, email: str, password: str) -> dict[str, Any]:
        """Sign in a user."""
        try:
            response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            return response  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error signing in user: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
            ) from e

    async def sign_out(self, token: str) -> None:
        """Sign out a user."""
        try:
            self.client.auth.sign_out()
        except Exception as e:
            logger.error(f"Error signing out user: {e}")
            # Don't raise exception for sign out failures

    async def reset_password(self, email: str) -> None:
        """Send password reset email."""
        try:
            self.client.auth.reset_password_email(email)
        except Exception as e:
            logger.error(f"Error sending password reset: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to send password reset email",
            ) from e

    async def update_user_metadata(
        self, user_id: str, metadata: dict[str, Any]
    ) -> None:
        """Update user metadata."""
        try:
            self.client.auth.admin.update_user_by_id(
                user_id, {"user_metadata": metadata}
            )
        except Exception as e:
            logger.error(f"Error updating user metadata: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update user metadata",
            ) from e


# Global auth service instance
auth_service = AuthService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Dependency to get current authenticated user."""
    token = credentials.credentials
    return await auth_service.get_user_from_token(token)


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User | None:
    """Dependency to get current user (optional authentication)."""
    if not credentials:
        return None

    try:
        return await auth_service.get_user_from_token(credentials.credentials)
    except HTTPException:
        return None


def require_role(required_role: str) -> Any:
    """Decorator to require specific role."""

    def role_checker(user: User = Depends(get_current_user)) -> User:
        user_role = user.role or "user"
        if user_role != required_role and user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role}",
            )
        return user

    return role_checker


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return user
