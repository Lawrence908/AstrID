"""Tests for content-addressed storage."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.infrastructure.storage.content_addressed_storage import ContentAddressedStorage


@pytest.fixture
def mock_r2_client():
    """Create mock R2 client."""
    return AsyncMock()


@pytest.fixture
def cas_client(mock_r2_client):
    """Create content-addressed storage client."""
    return ContentAddressedStorage(
        r2_client=mock_r2_client, bucket="test_bucket", prefix="cas/"
    )


class TestContentAddressedStorage:
    """Test content-addressed storage."""

    def test_get_content_hash(self, cas_client):
        """Test content hash calculation."""
        test_data = b"test content"
        expected_hash = hashlib.sha256(test_data).hexdigest()

        result = cas_client.get_content_hash(test_data)
        assert result == expected_hash

    def test_get_object_key(self, cas_client):
        """Test object key generation."""
        test_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        expected_key = (
            "cas/ab/abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        )

        result = cas_client._get_object_key(test_hash)
        assert result == expected_key

    @pytest.mark.asyncio
    async def test_store_data_new_content(self, cas_client, mock_r2_client):
        """Test storing new data."""
        test_data = b"new test content"
        test_hash = hashlib.sha256(test_data).hexdigest()
        test_metadata = {"source": "test"}

        # Mock file doesn't exist
        mock_r2_client.file_exists.return_value = False

        result = await cas_client.store_data(
            data=test_data, content_type="text/plain", metadata=test_metadata
        )

        assert result == test_hash

        # Verify R2 client calls
        mock_r2_client.file_exists.assert_called_once()
        mock_r2_client.upload_file.assert_called_once()

        # Check upload call arguments
        upload_call = mock_r2_client.upload_file.call_args
        assert upload_call[1]["bucket"] == "test_bucket"
        assert upload_call[1]["data"] == test_data
        assert upload_call[1]["content_type"] == "text/plain"
        assert upload_call[1]["metadata"]["content-hash"] == test_hash
        assert upload_call[1]["metadata"]["source"] == "test"

    @pytest.mark.asyncio
    async def test_store_data_existing_content(self, cas_client, mock_r2_client):
        """Test storing existing data (deduplication)."""
        test_data = b"existing test content"
        test_hash = hashlib.sha256(test_data).hexdigest()

        # Mock file already exists
        mock_r2_client.file_exists.return_value = True

        result = await cas_client.store_data(data=test_data)

        assert result == test_hash

        # Verify deduplication - no upload called
        mock_r2_client.file_exists.assert_called_once()
        mock_r2_client.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_data(self, cas_client, mock_r2_client):
        """Test retrieving data."""
        test_data = b"retrieved content"
        test_hash = hashlib.sha256(test_data).hexdigest()

        mock_r2_client.download_file.return_value = test_data

        result = await cas_client.retrieve_data(test_hash)

        assert result == test_data

        # Verify R2 client call
        expected_key = cas_client._get_object_key(test_hash)
        mock_r2_client.download_file.assert_called_once_with(
            "test_bucket", expected_key
        )

    @pytest.mark.asyncio
    async def test_retrieve_data_verification_failure(self, cas_client, mock_r2_client):
        """Test data retrieval with verification failure."""
        test_data = b"corrupted content"
        wrong_hash = "wrong_hash_value"

        mock_r2_client.download_file.return_value = test_data

        with pytest.raises(ValueError, match="Content verification failed"):
            await cas_client.retrieve_data(wrong_hash)

    @pytest.mark.asyncio
    async def test_store_file(self, cas_client, mock_r2_client):
        """Test storing file from path."""
        test_content = b"file content for storage"
        test_hash = hashlib.sha256(test_content).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()

            # Mock file doesn't exist
            mock_r2_client.file_exists.return_value = False

            result = await cas_client.store_file(
                file_path=tmp_file.name, content_type="text/plain"
            )

            assert result == test_hash

            # Verify upload was called with file metadata
            upload_call = mock_r2_client.upload_file.call_args
            assert (
                upload_call[1]["metadata"]["original-filename"]
                == Path(tmp_file.name).name
            )
            assert upload_call[1]["metadata"]["file-size"] == str(len(test_content))

    @pytest.mark.asyncio
    async def test_store_file_not_found(self, cas_client):
        """Test storing nonexistent file."""
        with pytest.raises(FileNotFoundError):
            await cas_client.store_file("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_exists(self, cas_client, mock_r2_client):
        """Test checking if content exists."""
        test_hash = "test_hash_value"

        mock_r2_client.file_exists.return_value = True

        result = await cas_client.exists(test_hash)

        assert result is True

        expected_key = cas_client._get_object_key(test_hash)
        mock_r2_client.file_exists.assert_called_once_with("test_bucket", expected_key)

    @pytest.mark.asyncio
    async def test_get_metadata(self, cas_client, mock_r2_client):
        """Test getting content metadata."""
        test_hash = "test_hash_value"
        test_metadata = {
            "object_key": "cas/te/test_hash_value",
            "size_bytes": 1024,
            "content_type": "text/plain",
        }

        mock_r2_client.get_file_metadata.return_value = test_metadata

        result = await cas_client.get_metadata(test_hash)

        assert result == test_metadata

        expected_key = cas_client._get_object_key(test_hash)
        mock_r2_client.get_file_metadata.assert_called_once_with(
            "test_bucket", expected_key
        )

    @pytest.mark.asyncio
    async def test_delete_content(self, cas_client, mock_r2_client):
        """Test deleting content."""
        test_hash = "test_hash_value"

        mock_r2_client.delete_file.return_value = True

        result = await cas_client.delete_content(test_hash)

        assert result is True

        expected_key = cas_client._get_object_key(test_hash)
        mock_r2_client.delete_file.assert_called_once_with("test_bucket", expected_key)

    @pytest.mark.asyncio
    async def test_delete_content_failure(self, cas_client, mock_r2_client):
        """Test deleting content with failure."""
        test_hash = "test_hash_value"

        mock_r2_client.delete_file.side_effect = Exception("Delete failed")

        result = await cas_client.delete_content(test_hash)

        assert result is False
