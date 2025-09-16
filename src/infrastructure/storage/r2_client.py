"""
Cloudflare R2 storage client for astronomical data and model artifacts.

This module provides an async interface to Cloudflare R2 object storage
with features specific to astronomical data management.
"""

import hashlib
import logging
import mimetypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import aioboto3
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    # Graceful fallback for development
    aioboto3 = None
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception
    Config = object
    logging.warning("boto3/aioboto3 not installed. Install with: uv add boto3 aioboto3")

from .config import StorageConfig

logger = logging.getLogger(__name__)


class R2StorageClient:
    """Async client for Cloudflare R2 object storage."""

    def __init__(
        self,
        config: StorageConfig | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        bucket_name: str | None = None,
        region: str = "auto",
    ):
        """Initialize R2 storage client.

        Args:
            config: Storage configuration object
            access_key_id: R2 access key (or use environment variable)
            secret_access_key: R2 secret key (or use environment variable)
            endpoint_url: R2 endpoint URL (or use environment variable)
            bucket_name: Default bucket name (or use environment variable)
            region: R2 region (always "auto" for R2)
        """
        if config:
            self.config = config
            self.access_key_id = config.r2_access_key_id
            self.secret_access_key = config.r2_secret_access_key
            self.endpoint_url = config.r2_endpoint_url
            self.bucket_name = config.r2_bucket_name
            self.region = config.r2_region
        else:
            # Fallback to individual parameters or defaults
            self.config = StorageConfig.from_env()
            self.access_key_id = access_key_id or self.config.r2_access_key_id
            self.secret_access_key = (
                secret_access_key or self.config.r2_secret_access_key
            )
            self.endpoint_url = endpoint_url or self.config.r2_endpoint_url
            self.bucket_name = bucket_name or self.config.r2_bucket_name
            self.region = region

        self.logger = logging.getLogger(__name__)

        if aioboto3 is None:
            raise ImportError(
                "aioboto3 package required. Install with: uv add aioboto3"
            )

        # Validate configuration
        if not all([self.access_key_id, self.secret_access_key, self.endpoint_url]):
            raise ValueError("R2 credentials and endpoint URL must be configured")

        # Configure boto settings
        if Config:
            self.boto_config = Config()
        else:
            self.boto_config = None

    def _get_session(self):
        """Get aioboto3 session with R2 configuration."""
        if aioboto3 is None:
            raise ImportError(
                "aioboto3 package required. Install with: uv add aioboto3"
            )
        return aioboto3.Session(
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
        )

    async def upload_file(
        self,
        bucket: str,
        key: str,
        data: bytes | Path | str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a file to R2 storage.

        Args:
            bucket: Bucket name
            key: Object key (path within bucket)
            data: File data as bytes, or path to local file
            content_type: MIME type (auto-detected if None)
            metadata: Custom metadata dictionary

        Returns:
            Object key of uploaded file

        Raises:
            FileNotFoundError: If local file doesn't exist
            ValueError: If upload fails validation
            ClientError: If R2 operation fails
        """
        try:
            # Handle different data types
            if isinstance(data, str | Path):
                local_path = Path(data)
                if not local_path.exists():
                    raise FileNotFoundError(f"Local file not found: {local_path}")

                # Auto-detect content type from file
                if content_type is None:
                    content_type, _ = mimetypes.guess_type(str(local_path))
                    if content_type is None:
                        # Default for FITS files
                        if local_path.suffix.lower() in [".fits", ".fit", ".fts"]:
                            content_type = "application/fits"
                        else:
                            content_type = "application/octet-stream"

                # Read file data
                with open(local_path, "rb") as f:
                    file_data = f.read()

                # Prepare metadata with file info
                upload_metadata = {
                    "original-filename": local_path.name,
                    "upload-timestamp": datetime.now(UTC).isoformat(),
                    "file-size": str(len(file_data)),
                }
            else:
                # Handle bytes data
                file_data = data
                content_type = content_type or "application/octet-stream"
                upload_metadata = {
                    "upload-timestamp": datetime.now(UTC).isoformat(),
                    "data-size": str(len(file_data)),
                }

            # Add user metadata
            if metadata:
                upload_metadata.update(metadata)

            # Calculate file hash for integrity
            file_hash = hashlib.sha256(file_data).hexdigest()
            upload_metadata["sha256"] = file_hash

            session = self._get_session()
            async with session.client(
                "s3", endpoint_url=self.endpoint_url, config=self.boto_config
            ) as s3_client:
                # Upload file
                await s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=file_data,
                    ContentType=content_type,
                    Metadata=upload_metadata,
                )

            self.logger.info(f"Uploaded to R2: {bucket}/{key}")
            return key

        except Exception as e:
            self.logger.error(f"Error uploading file to R2: {e}")
            raise

    async def download_file(self, bucket: str, key: str) -> bytes:
        """Download a file from R2 storage.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            File data as bytes

        Raises:
            ClientError: If object doesn't exist or download fails
        """
        try:
            session = self._get_session()
            async with session.client(
                "s3", endpoint_url=self.endpoint_url, config=self.boto_config
            ) as s3_client:
                response = await s3_client.get_object(Bucket=bucket, Key=key)
                data = await response["Body"].read()

                self.logger.info(f"Downloaded from R2: {bucket}/{key}")
                return data

        except Exception as e:
            self.logger.error(f"Error downloading file from R2: {e}")
            raise

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete a file from R2 storage.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            True if deletion successful

        Raises:
            ClientError: If deletion fails
        """
        try:
            session = self._get_session()
            async with session.client(
                "s3", endpoint_url=self.endpoint_url, config=self.boto_config
            ) as s3_client:
                await s3_client.delete_object(Bucket=bucket, Key=key)

            self.logger.info(f"Deleted from R2: {bucket}/{key}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting file from R2: {e}")
            raise

    async def list_files(
        self, bucket: str, prefix: str = "", max_keys: int = 1000
    ) -> list[str]:
        """List files in R2 bucket with optional prefix filter.

        Args:
            bucket: Bucket name
            prefix: Object key prefix filter
            max_keys: Maximum number of objects to return

        Returns:
            List of object keys

        Raises:
            ClientError: If listing fails
        """
        try:
            session = self._get_session()
            async with session.client(
                "s3", endpoint_url=self.endpoint_url, config=self.boto_config
            ) as s3_client:
                response = await s3_client.list_objects_v2(
                    Bucket=bucket, Prefix=prefix, MaxKeys=max_keys
                )

                keys = []
                for obj in response.get("Contents", []):
                    keys.append(obj["Key"])

                self.logger.info(
                    f"Listed {len(keys)} objects from {bucket} with prefix '{prefix}'"
                )
                return keys

        except Exception as e:
            self.logger.error(f"Error listing files from R2: {e}")
            raise

    async def get_file_metadata(self, bucket: str, key: str) -> dict[str, Any]:
        """Get metadata for an R2 object.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            Object metadata dictionary

        Raises:
            ClientError: If object doesn't exist or metadata retrieval fails
        """
        try:
            session = self._get_session()
            async with session.client(
                "s3", endpoint_url=self.endpoint_url, config=self.boto_config
            ) as s3_client:
                response = await s3_client.head_object(Bucket=bucket, Key=key)

                return {
                    "object_key": key,
                    "size_bytes": response["ContentLength"],
                    "content_type": response["ContentType"],
                    "last_modified": response["LastModified"],
                    "etag": response["ETag"].strip('"'),
                    "metadata": response.get("Metadata", {}),
                    "storage_class": response.get("StorageClass", "STANDARD"),
                }

        except Exception as e:
            self.logger.error(f"Error getting file metadata from R2: {e}")
            raise

    async def file_exists(self, bucket: str, key: str) -> bool:
        """Check if a file exists in R2 storage.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            True if file exists, False otherwise
        """
        try:
            await self.get_file_metadata(bucket, key)
            return True
        except ClientError:
            return False
        except Exception as e:
            self.logger.error(f"Error checking file existence in R2: {e}")
            return False

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        hash_sha256 = hashlib.sha256()

        # Read file in chunks to handle large files efficiently
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
