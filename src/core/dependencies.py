"""Core dependencies for dependency injection."""

from pydantic_settings import BaseSettings

from ..infrastructure.storage.config import StorageConfig
from ..infrastructure.workflow.config import WorkflowConfig


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = "postgresql+asyncpg://localhost/astrid"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Cloudflare R2
    cloudflare_r2_endpoint_url: str = ""
    cloudflare_r2_access_key_id: str = ""
    cloudflare_r2_secret_access_key: str = ""

    # Prefect
    prefect_server_url: str = "http://localhost:4200"
    prefect_supabase_project_ref: str = ""
    prefect_supabase_password: str = ""
    prefect_supabase_host: str = ""

    # Workflow settings
    max_concurrent_flows: int = 10
    flow_timeout: int = 3600
    retry_attempts: int = 3

    class Config:
        env_file = ".env"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_storage_config() -> StorageConfig:
    """Get storage configuration."""
    settings = get_settings()

    return StorageConfig(
        r2_account_id="",
        r2_access_key_id=settings.cloudflare_r2_access_key_id,
        r2_secret_access_key=settings.cloudflare_r2_secret_access_key,
        r2_bucket_name="astrid-models",
        r2_endpoint_url=settings.cloudflare_r2_endpoint_url,
        dvc_remote_url="",
        mlflow_artifact_root="s3://astrid-models",
    )


def get_workflow_config() -> WorkflowConfig:
    """Get workflow configuration."""
    settings = get_settings()

    return WorkflowConfig(
        prefect_server_url=settings.prefect_server_url,
        database_url=settings.database_url,
        storage_config=get_storage_config(),
        authentication_enabled=True,
        monitoring_enabled=True,
        alerting_enabled=True,
        max_concurrent_flows=settings.max_concurrent_flows,
        flow_timeout=settings.flow_timeout,
        retry_attempts=settings.retry_attempts,
        prefect_api_url=f"{settings.prefect_server_url}/api",
        prefect_ui_url=f"{settings.prefect_server_url}/ui",
        prefect_work_pool="astrid-pool",
        prefect_work_queue="default",
    )
