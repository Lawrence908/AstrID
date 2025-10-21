"""Authentication models for API keys and service accounts."""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID

from src.core.db.session import Base


class APIKey(Base):
    """API Key model for service authentication."""

    __tablename__ = "api_keys"

    id = Column(
        UUID(as_uuid=True), primary_key=True, default=lambda: secrets.token_hex(16)
    )
    name = Column(
        String(255), nullable=False, comment="Human-readable name for the API key"
    )
    description = Column(
        Text, nullable=True, comment="Description of the API key's purpose"
    )

    # Key management
    key_hash = Column(
        String(255), nullable=False, comment="Hashed version of the API key"
    )
    key_prefix = Column(
        String(8), nullable=False, comment="First 8 characters for identification"
    )

    # Permissions and access control
    permissions = Column(
        JSON, nullable=False, comment="List of permissions this key has"
    )
    scopes = Column(
        JSON, nullable=True, comment="Additional scopes for fine-grained access"
    )

    # Lifecycle management
    expires_at = Column(
        DateTime(timezone=True), nullable=True, comment="When this key expires"
    )
    last_used_at = Column(
        DateTime(timezone=True), nullable=True, comment="Last time this key was used"
    )
    usage_count = Column(
        String(20), default="0", comment="Number of times this key has been used"
    )

    # Ownership and audit
    created_by = Column(
        String(255), nullable=True, comment="User ID from Supabase auth"
    )
    is_active = Column(Boolean, default=True, comment="Whether this key is active")

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    # Note: No relationship to User since users are stored in Supabase auth

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name='{self.name}', prefix='{self.key_prefix}')>"

    def is_expired(self) -> bool:
        """Check if the API key has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if the API key is valid (active and not expired)."""
        return self.is_active and not self.is_expired()

    def has_permission(self, permission: str) -> bool:
        """Check if the API key has a specific permission."""
        if not self.is_valid():
            return False
        return permission in (self.permissions or [])

    def has_any_permission(self, permissions: list[str]) -> bool:
        """Check if the API key has any of the specified permissions."""
        if not self.is_valid():
            return False
        key_permissions = self.permissions or []
        return any(perm in key_permissions for perm in permissions)

    def has_all_permissions(self, permissions: list[str]) -> bool:
        """Check if the API key has all of the specified permissions."""
        if not self.is_valid():
            return False
        key_permissions = self.permissions or []
        return all(perm in key_permissions for perm in permissions)

    def update_usage(self) -> None:
        """Update the last used timestamp and increment usage count."""
        self.last_used_at = datetime.utcnow()
        try:
            self.usage_count = str(int(self.usage_count) + 1)
        except (ValueError, TypeError):
            self.usage_count = "1"

    @classmethod
    def generate_key(cls) -> tuple[str, str, str]:
        """Generate a new API key and return (full_key, key_hash, key_prefix)."""
        import hashlib

        # Generate a secure random key
        full_key = f"astrid_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        key_prefix = full_key[:8]
        return full_key, key_hash, key_prefix

    def to_dict(self) -> dict[str, Any]:
        """Convert API key to dictionary (excluding sensitive data)."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "key_prefix": self.key_prefix,
            "permissions": self.permissions,
            "scopes": self.scopes,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat()
            if self.last_used_at
            else None,
            "usage_count": self.usage_count,
            "is_active": self.is_active,
            "is_expired": self.is_expired(),
            "is_valid": self.is_valid(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
