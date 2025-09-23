"""API Key management service."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.adapters.auth.models import APIKey


class APIKeyService:
    """Service for managing API keys."""

    def __init__(self, db_session: Session | AsyncSession):
        self.db = db_session

    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _verify_key(self, key: str, key_hash: str) -> bool:
        """Verify an API key against its hash."""
        return self._hash_key(key) == key_hash

    async def create_api_key(
        self,
        name: str,
        permissions: list[str],
        description: str | None = None,
        expires_in_days: int | None = None,
        created_by: str | None = None,
        scopes: list[str] | None = None,
    ) -> tuple[str, APIKey]:
        """Create a new API key.

        Returns:
            tuple: (full_key, api_key_model)
        """
        # Generate the key
        full_key, key_hash, key_prefix = APIKey.generate_key()

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create the API key record
        api_key = APIKey(
            name=name,
            description=description,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=permissions,
            scopes=scopes or [],
            expires_at=expires_at,
            created_by=created_by,
        )

        if isinstance(self.db, AsyncSession):
            self.db.add(api_key)
            await self.db.flush()
            await self.db.refresh(api_key)
        else:
            self.db.add(api_key)
            self.db.flush()
            self.db.refresh(api_key)

        return full_key, api_key

    async def get_api_key_by_id(self, key_id: UUID) -> APIKey | None:
        """Get an API key by its ID."""
        if isinstance(self.db, AsyncSession):
            result = await self.db.execute(select(APIKey).where(APIKey.id == key_id))
            return result.scalar_one_or_none()
        else:
            return self.db.query(APIKey).filter(APIKey.id == key_id).first()

    async def get_api_key_by_key(self, key: str) -> APIKey | None:
        """Get an API key by the actual key value."""
        key_prefix = key[:8]

        if isinstance(self.db, AsyncSession):
            result = await self.db.execute(
                select(APIKey).where(APIKey.key_prefix == key_prefix, APIKey.is_active)
            )
            api_keys = result.scalars().all()
        else:
            api_keys = (
                self.db.query(APIKey)
                .filter(APIKey.key_prefix == key_prefix, APIKey.is_active)
                .all()
            )

        # Find the matching key by verifying the hash
        for api_key in api_keys:
            if self._verify_key(key, str(api_key.key_hash)):
                return api_key

        return None

    async def list_api_keys(
        self,
        created_by: str | None = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[APIKey]:
        """List API keys with optional filtering."""
        query = select(APIKey)

        if created_by:
            query = query.where(APIKey.created_by == created_by)

        if active_only:
            query = query.where(APIKey.is_active)

        query = query.offset(offset).limit(limit).order_by(APIKey.created_at.desc())

        if isinstance(self.db, AsyncSession):
            result = await self.db.execute(query)
            return list(result.scalars().all())
        else:
            return list(self.db.execute(query).scalars().all())

    async def update_api_key_usage(self, api_key: APIKey) -> None:
        """Update the usage statistics for an API key."""
        api_key.update_usage()

        if isinstance(self.db, AsyncSession):
            await self.db.commit()
        else:
            self.db.commit()

    async def revoke_api_key(self, key_id: UUID) -> bool:
        """Revoke an API key by setting it as inactive."""
        api_key = await self.get_api_key_by_id(key_id)
        if not api_key:
            return False

        api_key.is_active = False
        api_key.updated_at = datetime.utcnow()

        if isinstance(self.db, AsyncSession):
            await self.db.commit()
        else:
            self.db.commit()

        return True

    async def extend_api_key(
        self,
        key_id: UUID,
        expires_in_days: int,
    ) -> bool:
        """Extend the expiration date of an API key."""
        api_key = await self.get_api_key_by_id(key_id)
        if not api_key:
            return False

        new_expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        api_key.expires_at = new_expires_at
        api_key.updated_at = datetime.utcnow()

        if isinstance(self.db, AsyncSession):
            await self.db.commit()
        else:
            self.db.commit()

        return True

    async def update_api_key_permissions(
        self,
        key_id: UUID,
        permissions: list[str],
    ) -> bool:
        """Update the permissions of an API key."""
        api_key = await self.get_api_key_by_id(key_id)
        if not api_key:
            return False

        api_key.permissions = permissions
        api_key.updated_at = datetime.utcnow()

        if isinstance(self.db, AsyncSession):
            await self.db.commit()
        else:
            self.db.commit()

        return True

    async def validate_api_key(
        self,
        key: str,
        required_permissions: list[str] | None = None,
    ) -> APIKey | None:
        """Validate an API key and check permissions."""
        api_key = await self.get_api_key_by_key(key)
        if not api_key:
            return None

        if not api_key.is_valid():
            return None

        # Check permissions if required
        if required_permissions:
            if not api_key.has_all_permissions(required_permissions):
                return None

        # Update usage statistics
        await self.update_api_key_usage(api_key)

        return api_key


# Predefined permission sets for common use cases
PERMISSION_SETS = {
    "read_only": [
        "read:data",
        "read:public",
    ],
    "training_pipeline": [
        "read:data",
        "manage:operations",
        "training:data_collection",
    ],
    "prefect_workflows": [
        "read:data",
        "manage:operations",
        "training:data_collection",
        "observations:ingest",
        "preprocessing:process",
        "differencing:process",
        "detection:process",
    ],
    "full_access": [
        "read:data",
        "write:data",
        "manage:operations",
        "admin:access",
        "training:data_collection",
        "observations:ingest",
        "preprocessing:process",
        "differencing:process",
        "detection:process",
    ],
}


def get_permission_set(set_name: str) -> list[str]:
    """Get a predefined set of permissions."""
    return PERMISSION_SETS.get(set_name, [])
