"""Configuration management for AstrID application."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_test_url: str | None = None

    # Supabase
    supabase_url: str
    supabase_project_ref: str
    supabase_password: str | None = None
    supabase_host: str | None = None
    supabase_ssl_cert_path: str | None = None
    supabase_key: str | None = None
    supabase_service_role_key: str | None = None
    supabase_jwt_secret: str | None = None

    # Cloudflare R2 (S3-compatible)
    cloudflare_account_id: str
    cloudflare_r2_token_value: str
    cloudflare_r2_access_key_id: str
    cloudflare_r2_secret_access_key: str
    cloudflare_r2_bucket_name: str = "astrid"
    cloudflare_r2_endpoint_url: str
    cloudflare_eu_r2_endpoint_url: str

    # MLflow
    mlflow_tracking_uri: str
    mlflow_artifact_root: str = "s3://astrid-models"
    mlflow_s3_endpoint_url: str | None = None

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # DVC
    dvc_remote_name: str = "r2"
    dvc_remote_url: str = "s3://astrid-data"

    # Optional: Sentry
    sentry_dsn: str | None = None

    # External APIs
    astroquery_timeout: int = 300
    vizier_timeout: int = 300

    @property
    def supabase_database_url(self) -> str:
        """Build Supabase database URL from components."""
        if self.supabase_password and self.supabase_host:
            return f"postgresql+psycopg://postgres:{self.supabase_password}@{self.supabase_host}:5432/postgres"
        return str(self.database_url)

    @property
    def mlflow_tracking_uri_supabase(self) -> str:
        """Build MLflow tracking URI using Supabase database."""
        if self.supabase_password and self.supabase_host:
            return f"postgresql+psycopg://postgres:{self.supabase_password}@{self.supabase_host}:5432/postgres"
        return self.mlflow_tracking_uri

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
