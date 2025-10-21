"""Authentication API routes for AstrID."""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from src.adapters.auth.rbac import (
    Permission,
    UserWithRole,
    require_permission,
)
from src.adapters.auth.supabase_auth import User, auth_service
from src.core.api.response_wrapper import ResponseEnvelope, create_response
from src.core.exceptions import (
    AstrIDException,
    AuthenticationError,
    EmailNotConfirmedError,
    PasswordUpdateError,
    ResourceNotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["authentication"])


def safe_datetime_convert(dt_str_or_obj):
    """Safely convert datetime strings or objects to datetime."""
    if dt_str_or_obj is None:
        return None
    if isinstance(dt_str_or_obj, datetime):
        return dt_str_or_obj
    if isinstance(dt_str_or_obj, str):
        return datetime.fromisoformat(dt_str_or_obj.replace("Z", "+00:00"))
    return None


# Request/Response models
class SignUpRequest(BaseModel):
    """Sign up request model."""

    email: EmailStr
    password: str
    metadata: dict[str, Any] | None = None


class SignInRequest(BaseModel):
    """Sign in request model."""

    email: EmailStr
    password: str


class SignUpResponse(BaseModel):
    """Sign up response model."""

    user: User
    session: dict[str, Any]
    message: str


class SignInResponse(BaseModel):
    """Sign in response model."""

    user: User
    session: dict[str, Any]
    access_token: str
    refresh_token: str


class UserResponse(BaseModel):
    """User response model."""

    id: str
    email: str
    created_at: datetime
    updated_at: datetime
    email_confirmed_at: datetime | None = None
    last_sign_in_at: datetime | None = None
    role: str | None = None
    metadata: dict[str, Any] | None = None


class PasswordResetRequest(BaseModel):
    """Password reset request model."""

    email: EmailStr


class UpdateUserRequest(BaseModel):
    """Update user request model."""

    metadata: dict[str, Any] | None = None


@router.post(
    "/signup",
    response_model=ResponseEnvelope[SignUpResponse],
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "User created successfully"},
        400: {"description": "Invalid signup data"},
        409: {"description": "User already exists"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def sign_up(request: SignUpRequest) -> ResponseEnvelope[SignUpResponse]:
    """Sign up a new user."""
    try:
        response = await auth_service.sign_up(
            email=request.email, password=request.password, metadata=request.metadata
        )

        if not response.user:  # type: ignore[attr-defined]
            raise AuthenticationError(
                message="Sign up failed - user creation unsuccessful",
                error_code="SIGNUP_FAILED",
                details={"email": request.email},
            )

        user_data = response.user  # type: ignore[attr-defined]
        user = User(
            id=user_data.id,
            email=user_data.email,
            created_at=safe_datetime_convert(user_data.created_at),
            updated_at=safe_datetime_convert(user_data.updated_at),
            email_confirmed_at=safe_datetime_convert(user_data.email_confirmed_at),
            last_sign_in_at=safe_datetime_convert(user_data.last_sign_in_at),
            role=(
                user_data.user_metadata.get("role") if user_data.user_metadata else None
            ),
            metadata=user_data.user_metadata,
        )

        signup_response = SignUpResponse(
            user=user,
            session=response.session.dict() if response.session else {},  # type: ignore[attr-defined]
            message="User created successfully. Please check your email to confirm your account.",
        )
        return create_response(signup_response, status_code=status.HTTP_201_CREATED)

    except AstrIDException as e:
        status_code = (
            400 if isinstance(e, ValidationError | AuthenticationError) else 500
        )
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except HTTPException:
        # Re-raise HTTPExceptions from auth service
        raise
    except Exception as e:
        logger.error(f"Sign up error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during sign up",
        ) from e


@router.post(
    "/signin",
    response_model=ResponseEnvelope[SignInResponse],
    responses={
        200: {"description": "User signed in successfully"},
        401: {"description": "Invalid credentials or email not confirmed"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def sign_in(request: SignInRequest) -> ResponseEnvelope[SignInResponse]:
    """Sign in a user."""
    try:
        response = await auth_service.sign_in(
            email=request.email, password=request.password
        )

        if not response.user or not response.session:  # type: ignore[attr-defined]
            raise AuthenticationError(
                message="Invalid credentials",
                error_code="INVALID_CREDENTIALS",
                details={"email": request.email},
            )

        # Check if email is confirmed
        if response.user.email_confirmed_at is None:  # type: ignore[attr-defined]
            raise EmailNotConfirmedError(
                message="Email not confirmed. Please check your email and confirm your account.",
                error_code="EMAIL_NOT_CONFIRMED",
                details={"email": request.email},
            )

        user_data = response.user  # type: ignore[attr-defined]
        user = User(
            id=user_data.id,
            email=user_data.email,
            created_at=safe_datetime_convert(user_data.created_at),
            updated_at=safe_datetime_convert(user_data.updated_at),
            email_confirmed_at=safe_datetime_convert(user_data.email_confirmed_at),
            last_sign_in_at=safe_datetime_convert(user_data.last_sign_in_at),
            role=(
                user_data.user_metadata.get("role") if user_data.user_metadata else None
            ),
            metadata=user_data.user_metadata,
        )

        signin_response = SignInResponse(
            user=user,
            session=response.session.dict(),  # type: ignore[attr-defined]
            access_token=response.session.access_token,  # type: ignore[attr-defined]
            refresh_token=response.session.refresh_token,  # type: ignore[attr-defined]
        )
        return create_response(signin_response)

    except AstrIDException as e:
        if isinstance(e, AuthenticationError | EmailNotConfirmedError):
            status_code = 401
        else:
            status_code = 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except HTTPException:
        # Re-raise HTTPExceptions from auth service
        raise
    except Exception as e:
        logger.error(f"Sign in error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during sign in",
        ) from e


@router.post(
    "/signout",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "User signed out successfully"},
        401: {"description": "Not authenticated"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def sign_out(
    current_user: UserWithRole = Depends(require_permission(Permission.READ_PUBLIC)),
):
    """Sign out the current user."""
    try:
        await auth_service.sign_out(current_user.user.id)
        return create_response({"message": "Signed out successfully"})
    except HTTPException as e:
        logger.warning(f"Sign out warning: {e.detail}")
        # Don't raise exception for sign out failures
        return create_response({"message": "Signed out successfully"})
    except Exception as e:
        logger.error(f"Sign out error: {e}")
        # Don't raise exception for sign out failures
        return create_response({"message": "Signed out successfully"})


@router.post(
    "/reset-password",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Password reset email sent"},
        400: {"description": "Invalid email or password update error"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def reset_password(request: PasswordResetRequest):
    """Send password reset email."""
    try:
        await auth_service.reset_password(request.email)
        return create_response({"message": "Password reset email sent"})
    except AstrIDException as e:
        status_code = (
            400 if isinstance(e, ValidationError | PasswordUpdateError) else 500
        )
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except HTTPException:
        # Re-raise HTTPExceptions from auth service
        raise
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during password reset",
        ) from e


@router.get(
    "/me",
    response_model=ResponseEnvelope[UserResponse],
    responses={
        200: {"description": "User information retrieved successfully"},
        401: {"description": "Not authenticated"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def get_current_user_info(
    current_user: UserWithRole = Depends(require_permission(Permission.READ_PUBLIC)),
) -> ResponseEnvelope[UserResponse]:
    """Get current user information."""
    logger.debug(
        f"get_current_user_info called: user_id={current_user.user.id}, role={current_user.role.value}"
    )

    user_response = UserResponse(
        id=current_user.user.id,
        email=current_user.user.email,
        created_at=current_user.user.created_at,
        updated_at=current_user.user.updated_at,
        email_confirmed_at=current_user.user.email_confirmed_at,
        last_sign_in_at=current_user.user.last_sign_in_at,
        role=current_user.user.role,
        metadata=current_user.user.metadata,
    )

    return create_response(user_response)


@router.put(
    "/me",
    response_model=ResponseEnvelope[UserResponse],
    responses={
        200: {"description": "User updated successfully"},
        400: {"description": "Invalid user data"},
        401: {"description": "Not authenticated"},
        404: {"description": "User not found"},
        500: {"description": "Internal server error"},
    },
)  # type: ignore[misc]
async def update_current_user(
    request: UpdateUserRequest,
    current_user: UserWithRole = Depends(require_permission(Permission.WRITE_DATA)),
) -> ResponseEnvelope[UserResponse]:
    """Update current user information."""
    try:
        if request.metadata:
            await auth_service.update_user_metadata(
                current_user.user.id, request.metadata
            )

        # Get updated user info
        try:
            updated_user = await auth_service.get_user_from_token(current_user.user.id)
        except HTTPException as e:
            if e.status_code == 404:
                raise ResourceNotFoundError(
                    message="User not found after update",
                    error_code="USER_NOT_FOUND",
                    details={"user_id": current_user.user.id},
                ) from e
            raise

        user_response = UserResponse(
            id=updated_user.id,
            email=updated_user.email,
            created_at=updated_user.created_at,
            updated_at=updated_user.updated_at,
            email_confirmed_at=updated_user.email_confirmed_at,
            last_sign_in_at=updated_user.last_sign_in_at,
            role=updated_user.role,
            metadata=updated_user.metadata,
        )
        return create_response(user_response)

    except AstrIDException as e:
        status_code = (
            400 if isinstance(e, ValidationError | ResourceNotFoundError) else 500
        )
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except HTTPException:
        # Re-raise HTTPExceptions from auth service
        raise
    except Exception as e:
        logger.error(f"Update user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during user update",
        ) from e


@router.get("/health", responses={200: {"description": "Auth service is healthy"}})  # type: ignore[misc]
async def auth_health():
    """Health check for auth service."""
    return create_response({"status": "healthy", "service": "auth"})
