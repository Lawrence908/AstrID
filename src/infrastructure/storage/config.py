"""Storage configuration for AstrID."""

from dataclasses import dataclass

from src.core.constants import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_R2_ACCESS_KEY_ID,
    CLOUDFLARE_R2_BUCKET_NAME,
    CLOUDFLARE_R2_CA_BUNDLE,
    CLOUDFLARE_R2_ENDPOINT_URL,
    CLOUDFLARE_R2_SECRET_ACCESS_KEY,
    CLOUDFLARE_R2_VERIFY_SSL,
    DVC_REMOTE_URL,
    MLFLOW_ARTIFACT_ROOT,
)


@dataclass
class StorageConfig:
    """Configuration for storage infrastructure."""

    # Required R2 Configuration
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str
    r2_endpoint_url: str

    # Required DVC Configuration
    dvc_remote_url: str

    # Required MLflow Configuration
    mlflow_artifact_root: str

    # Optional fields with defaults (must come after required fields)
    r2_region: str = "auto"
    dvc_remote_name: str = "r2"
    content_addressing_enabled: bool = True
    deduplication_enabled: bool = True
    r2_verify_ssl: bool = True
    r2_ca_bundle: str | None = None

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create storage config from environment variables."""
        return cls(
            r2_account_id=CLOUDFLARE_ACCOUNT_ID or "",
            r2_access_key_id=CLOUDFLARE_R2_ACCESS_KEY_ID or "",
            r2_secret_access_key=CLOUDFLARE_R2_SECRET_ACCESS_KEY or "",
            r2_bucket_name=CLOUDFLARE_R2_BUCKET_NAME or "astrid",
            r2_endpoint_url=CLOUDFLARE_R2_ENDPOINT_URL or "",
            dvc_remote_url=DVC_REMOTE_URL or "s3://astrid-data",
            mlflow_artifact_root=MLFLOW_ARTIFACT_ROOT or "s3://astrid-models",
            r2_verify_ssl=CLOUDFLARE_R2_VERIFY_SSL,
            r2_ca_bundle=CLOUDFLARE_R2_CA_BUNDLE if CLOUDFLARE_R2_VERIFY_SSL else None,
        )

    def validate(self) -> None:
        """Validate storage configuration."""
        required_fields = [
            "r2_account_id",
            "r2_access_key_id",
            "r2_secret_access_key",
            "r2_endpoint_url",
            "dvc_remote_url",
            "mlflow_artifact_root",
        ]

        missing_fields = []
        for field in required_fields:
            if not getattr(self, field):
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Missing required storage configuration fields: {missing_fields}"
            )
