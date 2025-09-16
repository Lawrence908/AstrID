"""Tests for storage configuration."""

from unittest.mock import patch

import pytest

from src.infrastructure.storage.config import StorageConfig


class TestStorageConfig:
    """Test storage configuration."""

    def test_from_env_with_all_variables(self):
        """Test creating config from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "CLOUDFLARE_ACCOUNT_ID": "test_account",
                "CLOUDFLARE_R2_ACCESS_KEY_ID": "test_key_id",
                "CLOUDFLARE_R2_SECRET_ACCESS_KEY": "test_secret",
                "CLOUDFLARE_R2_BUCKET_NAME": "test_bucket",
                "CLOUDFLARE_R2_ENDPOINT_URL": "https://test.r2.cloudflarestorage.com",
                "DVC_REMOTE_URL": "s3://test-dvc-bucket",
                "MLFLOW_ARTIFACT_ROOT": "s3://test-mlflow-bucket",
            },
        ):
            config = StorageConfig.from_env()

            assert config.r2_account_id == "test_account"
            assert config.r2_access_key_id == "test_key_id"
            assert config.r2_secret_access_key == "test_secret"
            assert config.r2_bucket_name == "test_bucket"
            assert config.r2_endpoint_url == "https://test.r2.cloudflarestorage.com"
            assert config.dvc_remote_url == "s3://test-dvc-bucket"
            assert config.mlflow_artifact_root == "s3://test-mlflow-bucket"

    def test_from_env_with_defaults(self):
        """Test creating config with default values."""
        with patch.dict("os.environ", {}, clear=True):
            config = StorageConfig.from_env()

            assert config.r2_bucket_name == "astrid"
            assert config.dvc_remote_url == "s3://astrid-data"
            assert config.mlflow_artifact_root == "s3://astrid-models"
            assert config.content_addressing_enabled is True
            assert config.deduplication_enabled is True

    def test_validate_success(self):
        """Test successful validation."""
        config = StorageConfig(
            r2_account_id="test_account",
            r2_access_key_id="test_key",
            r2_secret_access_key="test_secret",
            r2_bucket_name="test_bucket",
            r2_endpoint_url="https://test.com",
            dvc_remote_url="s3://test-bucket",
            mlflow_artifact_root="s3://test-mlflow",
        )

        # Should not raise any exception
        config.validate()

    def test_validate_missing_fields(self):
        """Test validation with missing required fields."""
        config = StorageConfig(
            r2_account_id="",  # Missing
            r2_access_key_id="test_key",
            r2_secret_access_key="",  # Missing
            r2_bucket_name="test_bucket",
            r2_endpoint_url="https://test.com",
            dvc_remote_url="s3://test-bucket",
            mlflow_artifact_root="s3://test-mlflow",
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "Missing required storage configuration fields" in str(exc_info.value)
        assert "r2_account_id" in str(exc_info.value)
        assert "r2_secret_access_key" in str(exc_info.value)
