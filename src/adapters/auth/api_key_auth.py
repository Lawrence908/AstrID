"""API Key authentication dependencies."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.auth.api_key_service import APIKeyService
from src.adapters.auth.rbac import Permission, UserWithRole
from src.core.db.session import get_db


class APIKeyAuth:
    """API Key authentication result."""

    def __init__(self, api_key, permissions: list[str]):
        self.api_key = api_key
        self.permissions = permissions
        self.auth_method = "api_key"

    def has_permission(self, permission: str) -> bool:
        """Check if the API key has a specific permission."""
        return permission in self.permissions

    def has_any_permission(self, permissions: list[str]) -> bool:
        """Check if the API key has any of the specified permissions."""
        return any(perm in self.permissions for perm in permissions)

    def has_all_permissions(self, permissions: list[str]) -> bool:
        """Check if the API key has all of the specified permissions."""
        return all(perm in self.permissions for perm in permissions)


async def get_api_key_from_header(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> str | None:
    """Extract API key from X-API-Key header."""
    return x_api_key


async def validate_api_key(
    api_key: str | None = Depends(get_api_key_from_header),
    db: AsyncSession = Depends(get_db),
) -> APIKeyAuth | None:
    """Validate API key and return authentication info."""
    if not api_key:
        return None

    service = APIKeyService(db)
    validated_key = await service.validate_api_key(api_key)

    if not validated_key:
        return None

    return APIKeyAuth(validated_key, list(validated_key.permissions or []))


async def get_current_user_or_api_key(
    authorization: str | None = Header(None),
    api_key_auth: APIKeyAuth | None = Depends(validate_api_key),
    db: AsyncSession = Depends(get_db),
) -> UserWithRole | APIKeyAuth:
    """Get current user (JWT) or API key authentication."""

    # Try API key first
    if api_key_auth:
        return api_key_auth

    # Fall back to JWT authentication
    if authorization and authorization.startswith("Bearer "):
        try:
            # Extract token from authorization header
            authorization.split(" ")[1]  # Validate format but don't store
            # This would need to be adapted to work with your existing JWT auth
            # For now, we'll raise an exception to indicate JWT auth is not implemented here
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="JWT authentication not implemented in this context",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication",
            ) from e

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide either X-API-Key header or Authorization Bearer token",
    )


def require_permission_or_api_key(permission: Permission):
    """Dependency that requires either JWT user with permission or API key with permission."""

    async def check_permission(
        auth: UserWithRole | APIKeyAuth = Depends(get_current_user_or_api_key),
    ) -> UserWithRole | APIKeyAuth:
        if isinstance(auth, APIKeyAuth):
            # Check API key permissions
            if not auth.has_permission(permission.value):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key does not have required permission: {permission.value}",
                )
        else:
            # Check JWT user permissions (existing logic)
            if not auth.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User does not have required permission: {permission.value}",
                )

        return auth

    return check_permission


def require_any_permission_or_api_key(permissions: list[Permission]):
    """Dependency that requires either JWT user with any permission or API key with any permission."""

    async def check_permissions(
        auth: UserWithRole | APIKeyAuth = Depends(get_current_user_or_api_key),
    ) -> UserWithRole | APIKeyAuth:
        permission_values = [p.value for p in permissions]

        if isinstance(auth, APIKeyAuth):
            # Check API key permissions
            if not auth.has_any_permission(permission_values):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key does not have any of the required permissions: {permission_values}",
                )
        else:
            # Check JWT user permissions (existing logic)
            if not any(auth.has_permission(p) for p in permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User does not have any of the required permissions: {permission_values}",
                )

        return auth

    return check_permissions


def require_all_permissions_or_api_key(permissions: list[Permission]):
    """Dependency that requires either JWT user with all permissions or API key with all permissions."""

    async def check_permissions(
        auth: UserWithRole | APIKeyAuth = Depends(get_current_user_or_api_key),
    ) -> UserWithRole | APIKeyAuth:
        permission_values = [p.value for p in permissions]

        if isinstance(auth, APIKeyAuth):
            # Check API key permissions
            if not auth.has_all_permissions(permission_values):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key does not have all required permissions: {permission_values}",
                )
        else:
            # Check JWT user permissions (existing logic)
            if not all(auth.has_permission(p) for p in permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User does not have all required permissions: {permission_values}",
                )

        return auth

    return check_permissions
