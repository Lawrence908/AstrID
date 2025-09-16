"""
Storage API routes for AstrID.

This module provides HTTP endpoints for storage operations including
file upload/download, content-addressed storage, and dataset versioning.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.api.response_wrapper import create_response
from src.infrastructure.storage import (
    ContentAddressedStorage,
    DVCClient,
    R2StorageClient,
    StorageConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response


class FileUploadResponse(BaseModel):
    """Response model for file upload."""

    content_hash: str = Field(..., description="SHA-256 hash of uploaded content")
    object_key: str = Field(..., description="Storage object key")
    size_bytes: int = Field(..., description="File size in bytes")
    bucket: str = Field(..., description="Storage bucket name")


class FileMetadataResponse(BaseModel):
    """Response model for file metadata."""

    object_key: str = Field(..., description="Storage object key")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type")
    last_modified: str = Field(..., description="Last modification timestamp")
    etag: str = Field(..., description="Entity tag")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata"
    )


class DatasetVersionRequest(BaseModel):
    """Request model for dataset versioning."""

    dataset_path: str = Field(..., description="Path to dataset")
    message: str = Field(..., description="Version message")
    tag: str | None = Field(None, description="Optional version tag")


class DatasetVersionResponse(BaseModel):
    """Response model for dataset versioning."""

    version_id: str = Field(..., description="Version identifier")
    dataset_path: str = Field(..., description="Dataset path")
    message: str = Field(..., description="Version message")
    timestamp: str = Field(..., description="Version timestamp")


class DatasetListResponse(BaseModel):
    """Response model for dataset version listing."""

    versions: list[dict[str, Any]] = Field(..., description="List of dataset versions")


# Dependency injection for storage clients


def get_storage_config() -> StorageConfig:
    """Get storage configuration."""
    config = StorageConfig.from_env()
    config.validate()
    return config


def get_r2_client(
    config: StorageConfig = Depends(get_storage_config),
) -> R2StorageClient:
    """Get R2 storage client."""
    return R2StorageClient(config=config)


def get_cas_client(
    r2_client: R2StorageClient = Depends(get_r2_client),
    config: StorageConfig = Depends(get_storage_config),
) -> ContentAddressedStorage:
    """Get content-addressed storage client."""
    return ContentAddressedStorage(
        r2_client=r2_client,
        bucket=config.r2_bucket_name,
        prefix="cas/",
    )


def get_dvc_client(config: StorageConfig = Depends(get_storage_config)) -> DVCClient:
    """Get DVC client."""
    return DVCClient(config=config)


# Storage API endpoints


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    use_content_addressing: bool = Form(True),
    cas_client: ContentAddressedStorage = Depends(get_cas_client),
    r2_client: R2StorageClient = Depends(get_r2_client),
    config: StorageConfig = Depends(get_storage_config),
) -> dict[str, Any]:
    """Upload a file to storage.

    Args:
        file: File to upload
        metadata: Optional JSON metadata string
        use_content_addressing: Whether to use content-addressed storage
        cas_client: Content-addressed storage client
        r2_client: R2 storage client
        config: Storage configuration

    Returns:
        Upload response with file information
    """
    try:
        # Read file content
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty"
            )

        # Parse metadata if provided
        file_metadata = {}
        if metadata:
            try:
                import json

                file_metadata = json.loads(metadata)
            except json.JSONDecodeError as err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid metadata JSON",
                ) from err

        # Add file information to metadata
        file_metadata.update(
            {
                "original_filename": file.filename or "unknown",
                "content_type": file.content_type or "application/octet-stream",
            }
        )

        if use_content_addressing:
            # Use content-addressed storage
            content_hash = await cas_client.store_data(
                data=content,
                content_type=file.content_type or "application/octet-stream",
                metadata=file_metadata,
            )

            object_key = cas_client._get_object_key(content_hash)

            response_data = {
                "content_hash": content_hash,
                "object_key": object_key,
                "size_bytes": len(content),
                "bucket": config.r2_bucket_name,
            }
        else:
            # Use direct R2 storage
            object_key = f"uploads/{file.filename}"
            await r2_client.upload_file(
                bucket=config.r2_bucket_name,
                key=object_key,
                data=content,
                content_type=file.content_type,
                metadata=file_metadata,
            )

            # Calculate hash for response
            import hashlib

            content_hash = hashlib.sha256(content).hexdigest()

            response_data = {
                "content_hash": content_hash,
                "object_key": object_key,
                "size_bytes": len(content),
                "bucket": config.r2_bucket_name,
            }

        logger.info(f"Uploaded file: {file.filename} ({len(content)} bytes)")
        return create_response(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}",
        ) from e


@router.get("/download/{content_hash}")
async def download_file(
    content_hash: str,
    cas_client: ContentAddressedStorage = Depends(get_cas_client),
) -> StreamingResponse:
    """Download a file by content hash.

    Args:
        content_hash: SHA-256 content hash
        cas_client: Content-addressed storage client

    Returns:
        Streaming response with file content
    """
    try:
        # Check if content exists
        if not await cas_client.exists(content_hash):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content not found: {content_hash}",
            )

        # Get content metadata
        metadata = await cas_client.get_metadata(content_hash)

        # Retrieve content
        content = await cas_client.retrieve_data(content_hash)

        # Create streaming response
        import io

        # Determine filename and content type from metadata
        original_filename = metadata.get("metadata", {}).get(
            "original_filename", "download"
        )
        content_type = metadata.get("content_type", "application/octet-stream")

        return StreamingResponse(
            io.BytesIO(content),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={original_filename}",
                "Content-Length": str(len(content)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file: {str(e)}",
        ) from e


@router.delete("/{content_hash}")
async def delete_file(
    content_hash: str,
    cas_client: ContentAddressedStorage = Depends(get_cas_client),
) -> dict[str, Any]:
    """Delete a file by content hash.

    Args:
        content_hash: SHA-256 content hash
        cas_client: Content-addressed storage client

    Returns:
        Deletion confirmation
    """
    try:
        # Check if content exists
        if not await cas_client.exists(content_hash):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content not found: {content_hash}",
            )

        # Delete content
        success = await cas_client.delete_content(content_hash)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete content",
            )

        logger.info(f"Deleted content: {content_hash}")
        return create_response({"message": "Content deleted successfully"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}",
        ) from e


@router.get("/metadata/{content_hash}", response_model=FileMetadataResponse)
async def get_file_metadata(
    content_hash: str,
    cas_client: ContentAddressedStorage = Depends(get_cas_client),
) -> dict[str, Any]:
    """Get metadata for a file by content hash.

    Args:
        content_hash: SHA-256 content hash
        cas_client: Content-addressed storage client

    Returns:
        File metadata
    """
    try:
        # Check if content exists
        if not await cas_client.exists(content_hash):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content not found: {content_hash}",
            )

        # Get metadata
        metadata = await cas_client.get_metadata(content_hash)

        response_data = {
            "object_key": metadata["object_key"],
            "size_bytes": metadata["size_bytes"],
            "content_type": metadata["content_type"],
            "last_modified": metadata["last_modified"].isoformat(),
            "etag": metadata["etag"],
            "metadata": metadata.get("metadata", {}),
        }

        return create_response(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get file metadata: {str(e)}",
        ) from e


# Dataset versioning endpoints


@router.post("/datasets/{dataset_id}/version", response_model=DatasetVersionResponse)
async def version_dataset(
    dataset_id: str,
    request: DatasetVersionRequest,
    dvc_client: DVCClient = Depends(get_dvc_client),
) -> dict[str, Any]:
    """Create a new version of a dataset.

    Args:
        dataset_id: Dataset identifier
        request: Version request data
        dvc_client: DVC client

    Returns:
        Version information
    """
    try:
        # Initialize DVC repo if needed
        await dvc_client.init_repo()
        await dvc_client.configure_remote()

        # Create dataset version
        version_id = await dvc_client.version_dataset(
            dataset_path=request.dataset_path,
            message=request.message,
            tag=request.tag,
        )

        response_data = {
            "version_id": version_id,
            "dataset_path": request.dataset_path,
            "message": request.message,
            "timestamp": "",  # Will be filled by DVC client
        }

        logger.info(f"Created dataset version: {dataset_id}/{version_id}")
        return create_response(response_data)

    except Exception as e:
        logger.error(f"Error versioning dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to version dataset: {str(e)}",
        ) from e


@router.get("/datasets/{dataset_id}/versions", response_model=DatasetListResponse)
async def list_dataset_versions(
    dataset_id: str,
    dvc_client: DVCClient = Depends(get_dvc_client),
) -> dict[str, Any]:
    """List all versions of a dataset.

    Args:
        dataset_id: Dataset identifier
        dvc_client: DVC client

    Returns:
        List of dataset versions
    """
    try:
        # List dataset versions
        versions = await dvc_client.list_versions(dataset_path=dataset_id)

        response_data = {"versions": versions}

        return create_response(response_data)

    except Exception as e:
        logger.error(f"Error listing dataset versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list dataset versions: {str(e)}",
        ) from e


# Health check for storage services


@router.get("/health")
async def storage_health_check(
    config: StorageConfig = Depends(get_storage_config),
) -> dict[str, Any]:
    """Check health of storage services.

    Args:
        config: Storage configuration

    Returns:
        Health status of storage services
    """
    try:
        health_status = {"status": "healthy", "services": {}}

        # Check R2 connectivity
        try:
            r2_client = R2StorageClient(config=config)
            # Try listing files to test connectivity
            await r2_client.list_files(bucket=config.r2_bucket_name, max_keys=1)
            health_status["services"]["r2"] = {"status": "healthy"}
        except Exception as e:
            health_status["services"]["r2"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"

        # Check DVC availability
        try:
            # Basic DVC command test
            import subprocess

            result = subprocess.run(
                ["dvc", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                health_status["services"]["dvc"] = {"status": "healthy"}
            else:
                health_status["services"]["dvc"] = {"status": "unhealthy"}
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["dvc"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"

        return create_response(health_status)

    except Exception as e:
        logger.error(f"Error checking storage health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check storage health: {str(e)}",
        ) from e
