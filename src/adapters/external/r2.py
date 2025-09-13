"""
Cloudflare R2 storage client for astronomical data and model artifacts.

RESEARCH DOCS:
- Cloudflare R2 Documentation: https://developers.cloudflare.com/r2/
- R2 API Reference: https://developers.cloudflare.com/r2/api/
- S3 Compatibility: https://developers.cloudflare.com/r2/api/s3/api/
- boto3 S3 Client: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html

PYTHON PACKAGES:
- boto3: AWS SDK (works with R2 via S3 compatibility)
- aioboto3: Async version of boto3
- botocore: Low-level AWS SDK components

USE CASES FOR ASTRID:
1. Store FITS files from MAST/SkyView downloads
2. Store preprocessed and difference images
3. Store ML model artifacts and training data
4. Store detection cutouts and validation images
5. Implement content-addressable storage with hashing
6. Provide MLflow artifact backend storage
7. Cache reference images for difference imaging
"""

import hashlib
import logging
import mimetypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# TODO: Install these packages in your environment
# uv add boto3 aioboto3 botocore
try:
    import aioboto3
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    # Graceful fallback for development
    aioboto3 = None
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception
    logging.warning("boto3/aioboto3 not installed. Install with: uv add boto3 aioboto3")

from src.core.constants import (
    CLOUDFLARE_R2_ACCESS_KEY_ID,
    CLOUDFLARE_R2_BUCKET_NAME,
    CLOUDFLARE_R2_ENDPOINT_URL,
    CLOUDFLARE_R2_SECRET_ACCESS_KEY,
)

logger = logging.getLogger(__name__)


class R2StorageClient:
    """Async client for Cloudflare R2 object storage."""

    def __init__(
        self,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        bucket_name: str | None = None,
        region: str = "auto",
    ):
        """Initialize R2 storage client.

        RESEARCH:
        - R2 authentication setup
        - Endpoint URL format for your account
        - Bucket naming conventions

        Args:
            access_key_id: R2 access key (or use environment variable)
            secret_access_key: R2 secret key (or use environment variable)
            endpoint_url: R2 endpoint URL (or use environment variable)
            bucket_name: Default bucket name (or use environment variable)
            region: R2 region (always "auto" for R2)
        """
        self.access_key_id = access_key_id or CLOUDFLARE_R2_ACCESS_KEY_ID
        self.secret_access_key = secret_access_key or CLOUDFLARE_R2_SECRET_ACCESS_KEY
        self.endpoint_url = endpoint_url or CLOUDFLARE_R2_ENDPOINT_URL
        self.bucket_name = bucket_name or CLOUDFLARE_R2_BUCKET_NAME
        self.region = region
        self.logger = logging.getLogger(__name__)

        if aioboto3 is None:
            raise ImportError(
                "aioboto3 package required. Install with: uv add aioboto3"
            )

        # Validate configuration
        if not all([self.access_key_id, self.secret_access_key, self.endpoint_url]):
            raise ValueError("R2 credentials and endpoint URL must be configured")

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
        local_path: str | Path,
        object_key: str,
        bucket: str | None = None,
        metadata: dict[str, str] | None = None,
        content_type: str | None = None,
        storage_class: str = "STANDARD",
    ) -> dict[str, Any]:
        """Upload a file to R2 storage.

        RESEARCH:
        - R2 storage classes and pricing
        - Metadata limitations and best practices
        - Content-Type detection for FITS files
        - Multipart upload for large files

        Args:
            local_path: Path to local file
            object_key: R2 object key (path within bucket)
            bucket: Bucket name (uses default if None)
            metadata: Custom metadata dictionary
            content_type: MIME type (auto-detected if None)
            storage_class: R2 storage class

        Returns:
            Upload result with ETag and metadata

        USE CASE: Store FITS files from MAST downloads
        """
        try:
            bucket = bucket or self.bucket_name
            local_path = Path(local_path)

            if not local_path.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")

            # Auto-detect content type
            if content_type is None:
                content_type, _ = mimetypes.guess_type(str(local_path))
                if content_type is None:
                    # Default for FITS files
                    if local_path.suffix.lower() in [".fits", ".fit", ".fts"]:
                        content_type = "application/fits"
                    else:
                        content_type = "application/octet-stream"

            # Prepare metadata
            upload_metadata = {
                "original-filename": local_path.name,
                "upload-timestamp": datetime.now(UTC).isoformat(),
                "file-size": str(local_path.stat().st_size),
            }
            if metadata:
                upload_metadata.update(metadata)

            # TODO: Research aioboto3 async upload patterns
            # Key areas:
            # - Progress tracking for large files
            # - Multipart upload thresholds
            # - Error handling and retry logic
            # - Checksum validation

            session = self._get_session()
            async with session.resource(
                "s3", endpoint_url=self.endpoint_url
            ) as s3_client:
                # Calculate file hash for integrity
                file_hash = await self._calculate_file_hash(local_path)
                upload_metadata["sha256"] = file_hash

                # Upload file
                with open(local_path, "rb") as file_obj:
                    await s3_client.meta.client.upload_fileobj(
                        file_obj,
                        bucket,
                        object_key,
                        ExtraArgs={
                            "ContentType": content_type,
                            "Metadata": upload_metadata,
                            "StorageClass": storage_class,
                        },
                    )

                # Get object info after upload
                head_response = await s3_client.meta.client.head_object(
                    Bucket=bucket, Key=object_key
                )

                result = {
                    "bucket": bucket,
                    "object_key": object_key,
                    "etag": head_response["ETag"].strip('"'),
                    "size_bytes": head_response["ContentLength"],
                    "content_type": head_response["ContentType"],
                    "last_modified": head_response["LastModified"],
                    "metadata": head_response.get("Metadata", {}),
                    "sha256": file_hash,
                    "url": f"{self.endpoint_url}/{bucket}/{object_key}",
                }

                self.logger.info(f"Uploaded {local_path.name} to R2: {object_key}")
                return result

        except Exception as e:
            self.logger.error(f"Error uploading file to R2: {e}")
            raise

    async def download_file(
        self,
        object_key: str,
        local_path: str | Path,
        bucket: str | None = None,
        verify_checksum: bool = True,
    ) -> dict[str, Any]:
        """Download a file from R2 storage.

        RESEARCH:
        - Download resumption for large files
        - Parallel download strategies
        - Integrity verification methods

        Args:
            object_key: R2 object key
            local_path: Local destination path
            bucket: Bucket name (uses default if None)
            verify_checksum: Verify SHA256 checksum after download

        Returns:
            Download result with metadata

        USE CASE: Download reference images for difference imaging
        """
        try:
            bucket = bucket or self.bucket_name
            local_path = Path(local_path)

            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            session = self._get_session()
            async with session.resource(
                "s3", endpoint_url=self.endpoint_url
            ) as s3_client:
                # Get object metadata first
                head_response = await s3_client.meta.client.head_object(
                    Bucket=bucket, Key=object_key
                )
                expected_size = head_response["ContentLength"]
                expected_hash = head_response.get("Metadata", {}).get("sha256")

                # TODO: Implement download with progress tracking
                # Research areas:
                # - Streaming download for large files
                # - Progress callbacks
                # - Resume interrupted downloads
                # - Concurrent downloads for multiple files

                # Download file
                await s3_client.meta.client.download_file(
                    bucket, object_key, str(local_path)
                )

                # Verify download
                actual_size = local_path.stat().st_size
                if actual_size != expected_size:
                    raise ValueError(
                        f"Download size mismatch: expected {expected_size}, got {actual_size}"
                    )

                # Verify checksum if available
                if verify_checksum and expected_hash:
                    actual_hash = await self._calculate_file_hash(local_path)
                    if actual_hash != expected_hash:
                        raise ValueError("Checksum verification failed")

                result = {
                    "object_key": object_key,
                    "local_path": str(local_path),
                    "size_bytes": actual_size,
                    "checksum_verified": verify_checksum and bool(expected_hash),
                    "metadata": head_response.get("Metadata", {}),
                }

                self.logger.info(
                    f"Downloaded {object_key} from R2 to {local_path.name}"
                )
                return result

        except Exception as e:
            self.logger.error(f"Error downloading file from R2: {e}")
            raise

    async def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        bucket: str | None = None,
        metadata: dict[str, str] | None = None,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Upload bytes data directly to R2.

        Args:
            data: Bytes data to upload
            object_key: R2 object key
            bucket: Bucket name (uses default if None)
            metadata: Custom metadata
            content_type: MIME type

        Returns:
            Upload result

        USE CASE: Store processed image arrays or model predictions
        """
        try:
            bucket = bucket or self.bucket_name

            # Calculate hash
            data_hash = hashlib.sha256(data).hexdigest()

            # Prepare metadata
            upload_metadata = {
                "upload-timestamp": datetime.now(UTC).isoformat(),
                "data-size": str(len(data)),
                "sha256": data_hash,
            }
            if metadata:
                upload_metadata.update(metadata)

            session = self._get_session()
            async with session.resource(
                "s3", endpoint_url=self.endpoint_url
            ) as s3_client:
                await s3_client.meta.client.put_object(
                    Bucket=bucket,
                    Key=object_key,
                    Body=data,
                    ContentType=content_type,
                    Metadata=upload_metadata,
                )

                result = {
                    "bucket": bucket,
                    "object_key": object_key,
                    "size_bytes": len(data),
                    "sha256": data_hash,
                    "url": f"{self.endpoint_url}/{bucket}/{object_key}",
                }

                self.logger.info(f"Uploaded {len(data)} bytes to R2: {object_key}")
                return result

        except Exception as e:
            self.logger.error(f"Error uploading bytes to R2: {e}")
            raise

    async def list_objects(
        self, prefix: str = "", bucket: str | None = None, max_keys: int = 1000
    ) -> list[dict[str, Any]]:
        """List objects in R2 bucket.

        RESEARCH:
        - Pagination for large object lists
        - Efficient prefix-based filtering
        - Object metadata retrieval

        Args:
            prefix: Object key prefix filter
            bucket: Bucket name (uses default if None)
            max_keys: Maximum number of objects to return

        Returns:
            List of object information dictionaries

        USE CASE: Browse stored observations or find existing reference images
        """
        try:
            bucket = bucket or self.bucket_name

            session = self._get_session()
            async with session.resource(
                "s3", endpoint_url=self.endpoint_url
            ) as s3_client:
                response = await s3_client.meta.client.list_objects_v2(
                    Bucket=bucket, Prefix=prefix, MaxKeys=max_keys
                )

                objects = []
                for obj in response.get("Contents", []):
                    objects.append(
                        {
                            "key": obj["Key"],
                            "size_bytes": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "etag": obj["ETag"].strip('"'),
                            "storage_class": obj.get("StorageClass", "STANDARD"),
                        }
                    )

                self.logger.info(
                    f"Listed {len(objects)} objects with prefix '{prefix}'"
                )
                return objects

        except Exception as e:
            self.logger.error(f"Error listing R2 objects: {e}")
            raise

    async def delete_object(self, object_key: str, bucket: str | None = None) -> bool:
        """Delete an object from R2.

        Args:
            object_key: R2 object key to delete
            bucket: Bucket name (uses default if None)

        Returns:
            True if deletion successful

        USE CASE: Cleanup temporary files or remove outdated data
        """
        try:
            bucket = bucket or self.bucket_name

            session = self._get_session()
            async with session.resource(
                "s3", endpoint_url=self.endpoint_url
            ) as s3_client:
                await s3_client.meta.client.delete_object(Bucket=bucket, Key=object_key)

                self.logger.info(f"Deleted object from R2: {object_key}")
                return True

        except Exception as e:
            self.logger.error(f"Error deleting R2 object: {e}")
            raise

    async def get_object_metadata(
        self, object_key: str, bucket: str | None = None
    ) -> dict[str, Any]:
        """Get metadata for an R2 object.

        Args:
            object_key: R2 object key
            bucket: Bucket name (uses default if None)

        Returns:
            Object metadata dictionary

        USE CASE: Check file properties before downloading
        """
        try:
            bucket = bucket or self.bucket_name

            session = self._get_session()
            async with session.resource(
                "s3", endpoint_url=self.endpoint_url
            ) as s3_client:
                response = await s3_client.head_object(Bucket=bucket, Key=object_key)

                return {
                    "object_key": object_key,
                    "size_bytes": response["ContentLength"],
                    "content_type": response["ContentType"],
                    "last_modified": response["LastModified"],
                    "etag": response["ETag"].strip('"'),
                    "metadata": response.get("Metadata", {}),
                    "storage_class": response.get("StorageClass", "STANDARD"),
                }

        except Exception as e:
            self.logger.error(f"Error getting R2 object metadata: {e}")
            raise

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        hash_sha256 = hashlib.sha256()

        # Read file in chunks to handle large files
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()


class ContentAddressableStorage:
    """Content-addressable storage layer on top of R2."""

    def __init__(self, r2_client: R2StorageClient, prefix: str = "cas/"):
        """Initialize content-addressable storage.

        RESEARCH:
        - Content-addressable storage patterns
        - Hash-based deduplication
        - Performance considerations for lookups

        Args:
            r2_client: R2 storage client
            prefix: Prefix for content-addressable objects
        """
        self.r2_client = r2_client
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)

    async def store_file(
        self, local_path: str | Path, metadata: dict[str, str] | None = None
    ) -> tuple[str, str]:
        """Store file using content-addressable key.

        Args:
            local_path: Local file path
            metadata: Additional metadata

        Returns:
            Tuple of (content_hash, object_key)

        USE CASE: Deduplicated storage of FITS files and images
        """
        try:
            local_path = Path(local_path)

            # Calculate content hash
            content_hash = await self.r2_client._calculate_file_hash(local_path)

            # Create content-addressable key
            object_key = f"{self.prefix}{content_hash[:2]}/{content_hash}"

            # Check if object already exists
            try:
                await self.r2_client.get_object_metadata(object_key)
                self.logger.info(f"File already exists in CAS: {content_hash}")
                return content_hash, object_key
            except ClientError:
                # Object doesn't exist, proceed with upload
                pass

            # Upload with content hash metadata
            cas_metadata = {"content-hash": content_hash}
            if metadata:
                cas_metadata.update(metadata)

            await self.r2_client.upload_file(
                local_path=local_path, object_key=object_key, metadata=cas_metadata
            )

            self.logger.info(f"Stored file in CAS: {content_hash}")
            return content_hash, object_key

        except Exception as e:
            self.logger.error(f"Error storing file in CAS: {e}")
            raise

    async def retrieve_file(self, content_hash: str, local_path: str | Path) -> bool:
        """Retrieve file by content hash.

        Args:
            content_hash: SHA256 content hash
            local_path: Local destination path

        Returns:
            True if file retrieved successfully
        """
        try:
            object_key = f"{self.prefix}{content_hash[:2]}/{content_hash}"

            await self.r2_client.download_file(
                object_key=object_key, local_path=local_path, verify_checksum=True
            )

            return True

        except Exception as e:
            self.logger.error(f"Error retrieving file from CAS: {e}")
            return False


# INTEGRATION EXAMPLE:
# This shows how R2 storage would be used in AstrID


async def example_usage():
    """Example usage of R2 storage client for AstrID."""

    # Initialize client
    r2_client = R2StorageClient()

    # Use case 1: Store MAST download
    await r2_client.upload_file(
        local_path="/tmp/hst_observation.fits",
        object_key="observations/2024/01/hst_12345_drz.fits",
        metadata={
            "mission": "HST",
            "obs_id": "hst_12345",
            "instrument": "ACS/WFC",
            "filter": "F814W",
        },
    )

    # Use case 2: Content-addressable storage for deduplication
    cas = ContentAddressableStorage(r2_client)
    content_hash, object_key = await cas.store_file(
        "/tmp/reference_image.fits",
        metadata={"image_type": "reference", "survey": "DSS2_Red"},
    )

    # Use case 3: Store ML model artifacts
    await r2_client.upload_file(
        local_path="/models/unet_v1.0.keras",
        object_key="models/unet/v1.0/model.keras",
        metadata={
            "model_type": "unet",
            "version": "1.0",
            "training_date": "2024-01-15",
            "accuracy": "0.95",
        },
    )

    # Use case 4: List and manage stored data
    observations = await r2_client.list_objects(prefix="observations/2024/")
    print(f"Found {len(observations)} observation files")


# TODO: Integration points with AstrID:
# 1. Connect to MAST/SkyView clients for automatic file storage
# 2. Integrate with MLflow for artifact storage backend
# 3. Add R2 paths to observation database records
# 4. Implement cleanup policies for old files
# 5. Add monitoring and metrics for storage usage
# 6. Create backup and replication strategies
