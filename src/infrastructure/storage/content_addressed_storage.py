"""
Content-addressed storage implementation using SHA-256 hashing.

This module provides deduplication and content verification for stored files
by using their SHA-256 hash as the storage key.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

from .r2_client import R2StorageClient

logger = logging.getLogger(__name__)


class ContentAddressedStorage:
    """Content-addressable storage layer with deduplication."""

    def __init__(
        self,
        r2_client: R2StorageClient,
        bucket: str,
        prefix: str = "cas/",
    ):
        """Initialize content-addressable storage.

        Args:
            r2_client: R2 storage client instance
            bucket: Storage bucket name
            prefix: Prefix for content-addressable objects
        """
        self.r2_client = r2_client
        self.bucket = bucket
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)

    def get_content_hash(self, data: bytes) -> str:
        """Calculate SHA-256 hash of data.

        Args:
            data: Data bytes to hash

        Returns:
            SHA-256 hash as hex string
        """
        return hashlib.sha256(data).hexdigest()

    def _get_object_key(self, content_hash: str) -> str:
        """Generate object key from content hash.

        Uses a hierarchical structure for better performance:
        cas/ab/abcdef123... (first 2 chars as directory)

        Args:
            content_hash: SHA-256 hash

        Returns:
            Object key for storage
        """
        return f"{self.prefix}{content_hash[:2]}/{content_hash}"

    async def store_data(
        self,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store data using content addressing.

        Args:
            data: Data bytes to store
            content_type: MIME type of data
            metadata: Additional metadata

        Returns:
            Content hash of stored data
        """
        try:
            # Calculate content hash
            content_hash = self.get_content_hash(data)
            object_key = self._get_object_key(content_hash)

            # Check if object already exists (deduplication)
            if await self.r2_client.file_exists(self.bucket, object_key):
                self.logger.info(f"Data already exists in CAS: {content_hash}")
                return content_hash

            # Prepare metadata
            cas_metadata = {
                "content-hash": content_hash,
                "content-length": str(len(data)),
            }
            if metadata:
                cas_metadata.update(metadata)

            # Store data
            await self.r2_client.upload_file(
                bucket=self.bucket,
                key=object_key,
                data=data,
                content_type=content_type,
                metadata=cas_metadata,
            )

            self.logger.info(f"Stored data in CAS: {content_hash}")
            return content_hash

        except Exception as e:
            self.logger.error(f"Error storing data in CAS: {e}")
            raise

    async def retrieve_data(self, content_hash: str) -> bytes:
        """Retrieve data by content hash.

        Args:
            content_hash: SHA-256 content hash

        Returns:
            Retrieved data bytes

        Raises:
            ValueError: If content verification fails
            FileNotFoundError: If content hash not found
        """
        try:
            object_key = self._get_object_key(content_hash)

            # Download data
            data = await self.r2_client.download_file(self.bucket, object_key)

            # Verify content integrity
            actual_hash = self.get_content_hash(data)
            if actual_hash != content_hash:
                raise ValueError(
                    f"Content verification failed: expected {content_hash}, "
                    f"got {actual_hash}"
                )

            self.logger.info(f"Retrieved data from CAS: {content_hash}")
            return data

        except Exception as e:
            self.logger.error(f"Error retrieving data from CAS: {e}")
            raise

    async def store_file(
        self,
        file_path: str | Path,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store file using content addressing.

        Args:
            file_path: Path to file to store
            content_type: MIME type (auto-detected if None)
            metadata: Additional metadata

        Returns:
            Content hash of stored file
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read file data
            with open(file_path, "rb") as f:
                data = f.read()

            # Add file-specific metadata
            file_metadata = {
                "original-filename": file_path.name,
                "file-size": str(len(data)),
            }
            if metadata:
                file_metadata.update(metadata)

            # Store using content addressing
            content_hash = await self.store_data(
                data=data,
                content_type=content_type or "application/octet-stream",
                metadata=file_metadata,
            )

            self.logger.info(f"Stored file {file_path.name} in CAS: {content_hash}")
            return content_hash

        except Exception as e:
            self.logger.error(f"Error storing file in CAS: {e}")
            raise

    async def exists(self, content_hash: str) -> bool:
        """Check if content exists in storage.

        Args:
            content_hash: SHA-256 content hash

        Returns:
            True if content exists, False otherwise
        """
        try:
            object_key = self._get_object_key(content_hash)
            return await self.r2_client.file_exists(self.bucket, object_key)
        except Exception as e:
            self.logger.error(f"Error checking content existence: {e}")
            return False

    async def get_metadata(self, content_hash: str) -> dict[str, Any]:
        """Get metadata for stored content.

        Args:
            content_hash: SHA-256 content hash

        Returns:
            Content metadata dictionary
        """
        try:
            object_key = self._get_object_key(content_hash)
            return await self.r2_client.get_file_metadata(self.bucket, object_key)
        except Exception as e:
            self.logger.error(f"Error getting content metadata: {e}")
            raise

    async def delete_content(self, content_hash: str) -> bool:
        """Delete content by hash.

        Args:
            content_hash: SHA-256 content hash

        Returns:
            True if deletion successful
        """
        try:
            object_key = self._get_object_key(content_hash)
            result = await self.r2_client.delete_file(self.bucket, object_key)

            if result:
                self.logger.info(f"Deleted content from CAS: {content_hash}")

            return result
        except Exception as e:
            self.logger.error(f"Error deleting content from CAS: {e}")
            return False
