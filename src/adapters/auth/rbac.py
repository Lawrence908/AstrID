"""Role-Based Access Control (RBAC) for AstrID."""

import logging
from enum import Enum

from fastapi import Depends, HTTPException, status
from pydantic import BaseModel

from src.adapters.auth.supabase_auth import User, get_current_user

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles in the system."""

    ADMIN = "admin"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(str, Enum):
    """Simplified 5-level permission system."""

    # Level 1: Guest - Read-only access to public data
    READ_PUBLIC = "read:public"

    # Level 2: Viewer - Read access to most data
    READ_DATA = "read:data"

    # Level 3: Analyst - Read/write access to analysis data
    WRITE_DATA = "write:data"

    # Level 4: Researcher - Full access to research operations
    MANAGE_OPERATIONS = "manage:operations"

    # Level 5: Admin - Full system access
    ADMIN_ACCESS = "admin:access"


# Role-Permission mapping - Simplified 5-level system
ROLE_PERMISSIONS: dict[Role, list[Permission]] = {
    Role.ADMIN: [
        # Level 5: Full system access
        Permission.ADMIN_ACCESS,
        Permission.MANAGE_OPERATIONS,
        Permission.WRITE_DATA,
        Permission.READ_DATA,
        Permission.READ_PUBLIC,
    ],
    Role.RESEARCHER: [
        # Level 4: Full research operations
        Permission.MANAGE_OPERATIONS,
        Permission.WRITE_DATA,
        Permission.READ_DATA,
        Permission.READ_PUBLIC,
    ],
    Role.ANALYST: [
        # Level 3: Read/write analysis data
        Permission.WRITE_DATA,
        Permission.READ_DATA,
        Permission.READ_PUBLIC,
    ],
    Role.VIEWER: [
        # Level 2: Read access to most data
        Permission.READ_DATA,
        Permission.READ_PUBLIC,
    ],
    Role.GUEST: [
        # Level 1: Read-only public data
        Permission.READ_PUBLIC,
    ],
}


class RBACService:
    """Role-Based Access Control service."""

    @staticmethod
    def get_user_role(user: User) -> Role:
        """Get user role from user metadata."""
        if not user.role:
            logger.debug(f"No role found for user {user.id}, defaulting to GUEST")
            return Role.GUEST

        try:
            role = Role(user.role)
            logger.debug(f"User {user.id} assigned role: {role.value}")
            return role
        except ValueError:
            logger.warning(
                f"Invalid role '{user.role}' for user {user.id}, defaulting to GUEST"
            )
            return Role.GUEST

    @staticmethod
    def get_user_permissions(user: User) -> list[Permission]:
        """Get all permissions for a user based on their role."""
        role = RBACService.get_user_role(user)
        permissions = ROLE_PERMISSIONS.get(role, [])
        logger.debug(
            f"User {user.id} with role {role.value} has permissions: {[p.value for p in permissions]}"
        )
        return permissions

    @staticmethod
    def has_permission(user: User, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        user_permissions = RBACService.get_user_permissions(user)
        return permission in user_permissions

    @staticmethod
    def has_any_permission(user: User, permissions: list[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        user_permissions = RBACService.get_user_permissions(user)
        return any(perm in user_permissions for perm in permissions)

    @staticmethod
    def has_all_permissions(user: User, permissions: list[Permission]) -> bool:
        """Check if user has all of the specified permissions."""
        user_permissions = RBACService.get_user_permissions(user)
        return all(perm in user_permissions for perm in permissions)


# Global RBAC service instance
rbac_service = RBACService()


def require_permission(permission: Permission):
    """Decorator to require a specific permission."""

    def permission_checker(
        current_user: User = Depends(get_current_user),
    ) -> UserWithRole:
        if not rbac_service.has_permission(current_user, permission):
            user_permissions = rbac_service.get_user_permissions(current_user)
            logger.warning(
                f"Permission denied: user_id={current_user.id}, required={permission.value}, user_permissions={[p.value for p in user_permissions]}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission.value}",
            )

        logger.debug(
            f"Permission granted: user_id={current_user.id}, permission={permission.value}"
        )
        return get_current_user_with_role(current_user)

    return permission_checker


def require_any_permission(permissions: list[Permission]):
    """Decorator to require any of the specified permissions."""

    def permission_checker(
        current_user: User = Depends(get_current_user),
    ) -> UserWithRole:
        if not rbac_service.has_any_permission(current_user, permissions):
            required_perms = [p.value for p in permissions]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required any of: {required_perms}",
            )
        return get_current_user_with_role(current_user)

    return permission_checker


def require_all_permissions(permissions: list[Permission]):
    """Decorator to require all of the specified permissions."""

    def permission_checker(
        current_user: User = Depends(get_current_user),
    ) -> UserWithRole:
        if not rbac_service.has_all_permissions(current_user, permissions):
            required_perms = [p.value for p in permissions]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required all of: {required_perms}",
            )
        return get_current_user_with_role(current_user)

    return permission_checker


def require_role(role: Role):
    """Decorator to require a specific role."""

    def role_checker(current_user: User = Depends(get_current_user)) -> UserWithRole:
        user_role = rbac_service.get_user_role(current_user)
        if user_role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {role.value}, Current: {user_role.value}",
            )
        return get_current_user_with_role(current_user)

    return role_checker


def require_admin():
    """Decorator to require admin role."""
    return require_role(Role.ADMIN)


def require_researcher_or_above():
    """Decorator to require researcher role or above."""

    def role_checker(current_user: User = Depends(get_current_user)) -> UserWithRole:
        user_role = rbac_service.get_user_role(current_user)
        if user_role not in [Role.ADMIN, Role.RESEARCHER]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: researcher or admin, Current: {user_role.value}",
            )
        return get_current_user_with_role(current_user)

    return role_checker


class UserWithRole(BaseModel):
    """User model with role information."""

    user: User
    role: Role
    permissions: list[Permission]

    def has_permission(self, permission: Permission) -> bool:
        """Check if the user has a specific permission."""
        return permission in self.permissions


def get_current_user_with_role(current_user: User) -> UserWithRole:
    """Get current user with role and permissions information."""
    role = rbac_service.get_user_role(current_user)
    permissions = rbac_service.get_user_permissions(current_user)

    return UserWithRole(
        user=current_user,
        role=role,
        permissions=permissions,
    )
