"""API Key management routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.auth.api_key_auth import APIKeyAuth
from src.adapters.auth.api_key_service import APIKeyService, get_permission_set
from src.adapters.auth.rbac import Permission, UserWithRole, require_permission
from src.core.api.response_wrapper import ResponseEnvelope, create_response
from src.core.db.session import get_db

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


# Request/Response Models
class CreateAPIKeyRequest(BaseModel):
    """Request model for creating an API key."""

    name: str = Field(..., description="Human-readable name for the API key")
    description: str | None = Field(
        None, description="Description of the API key's purpose"
    )
    permissions: list[str] | None = Field(
        None, description="List of permissions for this key"
    )
    permission_set: str | None = Field(
        None, description="Use a predefined permission set"
    )
    expires_in_days: int | None = Field(
        None, description="Number of days until expiration"
    )
    scopes: list[str] | None = Field(None, description="Additional scopes")


class CreateAPIKeyResponse(BaseModel):
    """Response model for creating an API key."""

    id: str
    name: str
    description: str | None
    key: str  # Only returned once during creation
    key_prefix: str
    permissions: list[str]
    scopes: list[str]
    expires_at: str | None
    created_at: str


class APIKeyResponse(BaseModel):
    """Response model for API key details (without the actual key)."""

    id: str
    name: str
    description: str | None
    key_prefix: str
    permissions: list[str]
    scopes: list[str]
    expires_at: str | None
    last_used_at: str | None
    usage_count: str
    is_active: bool
    is_expired: bool
    is_valid: bool
    created_at: str
    updated_at: str


class UpdateAPIKeyRequest(BaseModel):
    """Request model for updating an API key."""

    name: str | None = None
    description: str | None = None
    permissions: list[str] | None = None
    scopes: list[str] | None = None


class ExtendAPIKeyRequest(BaseModel):
    """Request model for extending an API key."""

    expires_in_days: int = Field(..., description="Number of days to extend")


@router.post(
    "/",
    response_model=ResponseEnvelope[CreateAPIKeyResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    request: CreateAPIKeyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.ADMIN_ACCESS)),
):
    """Create a new API key."""

    # Check permissions
    if isinstance(current_user, APIKeyAuth):
        if not current_user.has_permission("admin_access"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API keys can only be created by admin users",
            )
    else:
        if not current_user.has_permission(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to create API keys",
            )

    # Determine permissions
    permissions = request.permissions or []
    if request.permission_set:
        predefined_permissions = get_permission_set(request.permission_set)
        if predefined_permissions:
            permissions = predefined_permissions
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown permission set: {request.permission_set}",
            )

    if not permissions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one permission is required",
        )

    # Get creator ID
    creator_id = None
    if isinstance(current_user, UserWithRole):
        creator_id = str(current_user.user.id)

    # Create the API key
    service = APIKeyService(db)
    full_key, api_key = await service.create_api_key(
        name=request.name,
        description=request.description,
        permissions=permissions,
        expires_in_days=request.expires_in_days,
        created_by=creator_id,
        scopes=request.scopes,
    )

    # Commit the transaction
    await db.commit()

    response = CreateAPIKeyResponse(
        id=str(api_key.id),
        name=api_key.name,
        description=api_key.description,
        key=full_key,  # Only returned once
        key_prefix=api_key.key_prefix,
        permissions=api_key.permissions,
        scopes=api_key.scopes or [],
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        created_at=api_key.created_at.isoformat(),
    )

    return create_response(response)


@router.get(
    "/",
    response_model=ResponseEnvelope[list[APIKeyResponse]],
)
async def list_api_keys(
    limit: int = 100,
    offset: int = 0,
    active_only: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.ADMIN_ACCESS)),
):
    """List API keys."""

    # Check permissions
    if isinstance(current_user, APIKeyAuth):
        if not current_user.has_permission("admin_access"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to list API keys",
            )
    else:
        if not current_user.has_permission(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to list API keys",
            )

    # Get creator filter
    creator_id = None
    if isinstance(current_user, UserWithRole):
        creator_id = str(current_user.user.id)

    service = APIKeyService(db)
    api_keys = await service.list_api_keys(
        created_by=creator_id,
        active_only=active_only,
        limit=limit,
        offset=offset,
    )

    response = [
        APIKeyResponse(
            id=str(key.id),
            name=key.name,
            description=key.description,
            key_prefix=key.key_prefix,
            permissions=key.permissions,
            scopes=key.scopes or [],
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            usage_count=key.usage_count,
            is_active=key.is_active,
            is_expired=key.is_expired(),
            is_valid=key.is_valid(),
            created_at=key.created_at.isoformat(),
            updated_at=key.updated_at.isoformat(),
        )
        for key in api_keys
    ]

    return create_response(response)


@router.get(
    "/{key_id}",
    response_model=ResponseEnvelope[APIKeyResponse],
)
async def get_api_key(
    key_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.ADMIN_ACCESS)),
):
    """Get a specific API key by ID."""

    # Check permissions
    if isinstance(current_user, APIKeyAuth):
        if not current_user.has_permission("admin_access"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to view API keys",
            )
    else:
        if not current_user.has_permission(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to view API keys",
            )

    service = APIKeyService(db)
    api_key = await service.get_api_key_by_id(key_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    response = APIKeyResponse(
        id=str(api_key.id),
        name=api_key.name,
        description=api_key.description,
        key_prefix=api_key.key_prefix,
        permissions=api_key.permissions,
        scopes=api_key.scopes or [],
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        usage_count=api_key.usage_count,
        is_active=api_key.is_active,
        is_expired=api_key.is_expired(),
        is_valid=api_key.is_valid(),
        created_at=api_key.created_at.isoformat(),
        updated_at=api_key.updated_at.isoformat(),
    )

    return create_response(response)


@router.put(
    "/{key_id}",
    response_model=ResponseEnvelope[APIKeyResponse],
)
async def update_api_key(
    key_id: UUID,
    request: UpdateAPIKeyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.ADMIN_ACCESS)),
):
    """Update an API key."""

    # Check permissions
    if isinstance(current_user, APIKeyAuth):
        if not current_user.has_permission("admin_access"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to update API keys",
            )
    else:
        if not current_user.has_permission(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to update API keys",
            )

    service = APIKeyService(db)
    api_key = await service.get_api_key_by_id(key_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    # Update fields
    if request.name is not None:
        api_key.name = request.name
    if request.description is not None:
        api_key.description = request.description
    if request.permissions is not None:
        api_key.permissions = request.permissions
    if request.scopes is not None:
        api_key.scopes = request.scopes

    await db.commit()
    await db.refresh(api_key)

    response = APIKeyResponse(
        id=str(api_key.id),
        name=api_key.name,
        description=api_key.description,
        key_prefix=api_key.key_prefix,
        permissions=api_key.permissions,
        scopes=api_key.scopes or [],
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        usage_count=api_key.usage_count,
        is_active=api_key.is_active,
        is_expired=api_key.is_expired(),
        is_valid=api_key.is_valid(),
        created_at=api_key.created_at.isoformat(),
        updated_at=api_key.updated_at.isoformat(),
    )

    return create_response(response)


@router.post(
    "/{key_id}/revoke",
    response_model=ResponseEnvelope[dict[str, str]],
)
async def revoke_api_key(
    key_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.ADMIN_ACCESS)),
):
    """Revoke an API key."""

    # Check permissions
    if isinstance(current_user, APIKeyAuth):
        if not current_user.has_permission("admin_access"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to revoke API keys",
            )
    else:
        if not current_user.has_permission(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to revoke API keys",
            )

    service = APIKeyService(db)
    success = await service.revoke_api_key(key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    return create_response({"message": "API key revoked successfully"})


@router.post(
    "/{key_id}/extend",
    response_model=ResponseEnvelope[dict[str, str]],
)
async def extend_api_key(
    key_id: UUID,
    request: ExtendAPIKeyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserWithRole = Depends(require_permission(Permission.ADMIN_ACCESS)),
):
    """Extend an API key expiration."""

    # Check permissions
    if isinstance(current_user, APIKeyAuth):
        if not current_user.has_permission("admin_access"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to extend API keys",
            )
    else:
        if not current_user.has_permission(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to extend API keys",
            )

    service = APIKeyService(db)
    success = await service.extend_api_key(key_id, request.expires_in_days)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    return create_response({"message": "API key extended successfully"})


@router.get(
    "/permission-sets",
    response_model=ResponseEnvelope[dict[str, list[str]]],
)
async def get_permission_sets():
    """Get available permission sets."""
    from src.adapters.auth.api_key_service import PERMISSION_SETS

    return create_response(PERMISSION_SETS)
