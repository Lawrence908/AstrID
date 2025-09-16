"""Tests for R2 storage client."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.storage.config import StorageConfig
from src.infrastructure.storage.r2_client import R2StorageClient


@pytest.fixture
def storage_config():
    """Create test storage configuration."""
    return StorageConfig(
        r2_account_id="test_account",
        r2_access_key_id="test_key_id",
        r2_secret_access_key="test_secret_key",
        r2_bucket_name="test_bucket",
        r2_endpoint_url="https://test.r2.cloudflarestorage.com",
        dvc_remote_url="s3://test-dvc",
        mlflow_artifact_root="s3://test-mlflow",
    )


@pytest.fixture
def r2_client(storage_config):
    """Create test R2 client."""
    with patch("src.infrastructure.storage.r2_client.aioboto3"):
        return R2StorageClient(config=storage_config)


class TestR2StorageClient:
    """Test R2 storage client."""

    def test_init_with_config(self, storage_config):
        """Test initialization with config object."""
        with patch("src.infrastructure.storage.r2_client.aioboto3"):
            client = R2StorageClient(config=storage_config)

            assert client.access_key_id == "test_key_id"
            assert client.secret_access_key == "test_secret_key"
            assert client.endpoint_url == "https://test.r2.cloudflarestorage.com"
            assert client.bucket_name == "test_bucket"

    def test_init_without_config(self):
        """Test initialization without config object."""
        with (
            patch("src.infrastructure.storage.r2_client.aioboto3"),
            patch(
                "src.infrastructure.storage.config.StorageConfig.from_env"
            ) as mock_from_env,
        ):
            mock_config = MagicMock()
            mock_config.r2_access_key_id = "env_key"
            mock_config.r2_secret_access_key = "env_secret"
            mock_config.r2_endpoint_url = "https://env.endpoint.com"
            mock_config.r2_bucket_name = "env_bucket"
            mock_config.r2_region = "auto"
            mock_from_env.return_value = mock_config

            client = R2StorageClient()

            assert client.access_key_id == "env_key"
            assert client.secret_access_key == "env_secret"

    def test_init_missing_credentials(self):
        """Test initialization with missing credentials."""
        config = StorageConfig(
            r2_account_id="test_account",
            r2_access_key_id="",  # Missing
            r2_secret_access_key="test_secret",
            r2_bucket_name="test_bucket",
            r2_endpoint_url="https://test.com",
            dvc_remote_url="s3://test-dvc",
            mlflow_artifact_root="s3://test-mlflow",
        )

        with patch("src.infrastructure.storage.r2_client.aioboto3"):
            with pytest.raises(
                ValueError, match="R2 credentials and endpoint URL must be configured"
            ):
                R2StorageClient(config=config)

    @pytest.mark.asyncio
    async def test_upload_file_bytes(self, r2_client):
        """Test uploading bytes data."""
        test_data = b"test file content"
        test_bucket = "test_bucket"
        test_key = "test/file.txt"

        mock_s3_client = AsyncMock()
        mock_session = AsyncMock()
        mock_session.client.return_value.__aenter__.return_value = mock_s3_client

        with patch.object(r2_client, "_get_session", return_value=mock_session):
            result = await r2_client.upload_file(
                bucket=test_bucket,
                key=test_key,
                data=test_data,
                content_type="text/plain",
            )

            assert result == test_key
            mock_s3_client.put_object.assert_called_once()

            # Verify call arguments
            call_args = mock_s3_client.put_object.call_args
            assert call_args[1]["Bucket"] == test_bucket
            assert call_args[1]["Key"] == test_key
            assert call_args[1]["Body"] == test_data
            assert call_args[1]["ContentType"] == "text/plain"

    @pytest.mark.asyncio
    async def test_upload_file_from_path(self, r2_client):
        """Test uploading file from local path."""
        test_content = b"test file content"

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()

            mock_s3_client = AsyncMock()
            mock_session = AsyncMock()
            mock_session.client.return_value.__aenter__.return_value = mock_s3_client

            with patch.object(r2_client, "_get_session", return_value=mock_session):
                result = await r2_client.upload_file(
                    bucket="test_bucket", key="test/file.txt", data=Path(tmp_file.name)
                )

                assert result == "test/file.txt"
                mock_s3_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_nonexistent_path(self, r2_client):
        """Test uploading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            await r2_client.upload_file(
                bucket="test_bucket",
                key="test/file.txt",
                data=Path("/nonexistent/file.txt"),
            )

    @pytest.mark.asyncio
    async def test_download_file(self, r2_client):
        """Test downloading file."""
        test_data = b"downloaded content"
        test_bucket = "test_bucket"
        test_key = "test/file.txt"

        mock_response = {"Body": AsyncMock()}
        mock_response["Body"].read.return_value = test_data

        mock_s3_client = AsyncMock()
        mock_s3_client.get_object.return_value = mock_response
        mock_session = AsyncMock()
        mock_session.client.return_value.__aenter__.return_value = mock_s3_client

        with patch.object(r2_client, "_get_session", return_value=mock_session):
            result = await r2_client.download_file(bucket=test_bucket, key=test_key)

            assert result == test_data
            mock_s3_client.get_object.assert_called_once_with(
                Bucket=test_bucket, Key=test_key
            )

    @pytest.mark.asyncio
    async def test_delete_file(self, r2_client):
        """Test deleting file."""
        test_bucket = "test_bucket"
        test_key = "test/file.txt"

        mock_s3_client = AsyncMock()
        mock_session = AsyncMock()
        mock_session.client.return_value.__aenter__.return_value = mock_s3_client

        with patch.object(r2_client, "_get_session", return_value=mock_session):
            result = await r2_client.delete_file(bucket=test_bucket, key=test_key)

            assert result is True
            mock_s3_client.delete_object.assert_called_once_with(
                Bucket=test_bucket, Key=test_key
            )

    @pytest.mark.asyncio
    async def test_list_files(self, r2_client):
        """Test listing files."""
        test_bucket = "test_bucket"
        test_prefix = "test/"

        mock_response = {
            "Contents": [
                {"Key": "test/file1.txt"},
                {"Key": "test/file2.txt"},
            ]
        }

        mock_s3_client = AsyncMock()
        mock_s3_client.list_objects_v2.return_value = mock_response
        mock_session = AsyncMock()
        mock_session.client.return_value.__aenter__.return_value = mock_s3_client

        with patch.object(r2_client, "_get_session", return_value=mock_session):
            result = await r2_client.list_files(bucket=test_bucket, prefix=test_prefix)

            assert result == ["test/file1.txt", "test/file2.txt"]
            mock_s3_client.list_objects_v2.assert_called_once_with(
                Bucket=test_bucket, Prefix=test_prefix, MaxKeys=1000
            )

    @pytest.mark.asyncio
    async def test_get_file_metadata(self, r2_client):
        """Test getting file metadata."""
        test_bucket = "test_bucket"
        test_key = "test/file.txt"

        mock_response = {
            "ContentLength": 1024,
            "ContentType": "text/plain",
            "LastModified": "2023-01-01T00:00:00Z",
            "ETag": '"abc123"',
            "Metadata": {"custom": "value"},
            "StorageClass": "STANDARD",
        }

        mock_s3_client = AsyncMock()
        mock_s3_client.head_object.return_value = mock_response
        mock_session = AsyncMock()
        mock_session.client.return_value.__aenter__.return_value = mock_s3_client

        with patch.object(r2_client, "_get_session", return_value=mock_session):
            result = await r2_client.get_file_metadata(bucket=test_bucket, key=test_key)

            assert result["object_key"] == test_key
            assert result["size_bytes"] == 1024
            assert result["content_type"] == "text/plain"
            assert result["etag"] == "abc123"
            assert result["metadata"] == {"custom": "value"}

    @pytest.mark.asyncio
    async def test_file_exists_true(self, r2_client):
        """Test file exists returns True."""
        with patch.object(
            r2_client, "get_file_metadata", return_value={"key": "value"}
        ):
            result = await r2_client.file_exists("test_bucket", "test_key")
            assert result is True

    @pytest.mark.asyncio
    async def test_file_exists_false(self, r2_client):
        """Test file exists returns False."""
        from botocore.exceptions import ClientError

        with patch.object(
            r2_client, "get_file_metadata", side_effect=ClientError({}, "HeadObject")
        ):
            result = await r2_client.file_exists("test_bucket", "test_key")
            assert result is False

    def test_calculate_file_hash(self, r2_client):
        """Test file hash calculation."""
        test_content = b"test content for hashing"
        expected_hash = hashlib.sha256(test_content).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()

            import asyncio

            result = asyncio.run(r2_client._calculate_file_hash(Path(tmp_file.name)))

            assert result == expected_hash
