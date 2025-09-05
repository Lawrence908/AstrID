"""Example usage of the constants file."""

from src.core.constants import (
    # Cloudflare R2
    CLOUDFLARE_R2_ACCESS_KEY_ID,
    CLOUDFLARE_R2_BUCKET_NAME,
    CLOUDFLARE_R2_ENDPOINT_URL,
    DEBUG,
    ENVIRONMENT,
    LOG_LEVEL,
    # MLflow
    MLFLOW_ARTIFACT_ROOT,
    # Supabase
    SUPABASE_URL,
    # Database
    get_database_url,
    get_db_config,
    get_mlflow_tracking_uri,
)


def example_database_usage() -> None:
    """Example of how to use database configuration."""
    db_url = get_database_url()
    print(f"Database URL: {db_url}")

    mlflow_uri = get_mlflow_tracking_uri()
    print(f"MLflow URI: {mlflow_uri}")

    db_config = get_db_config()
    print(f"Database Config: {db_config}")
    print(f"Database Host: {db_config['host']}")
    print(f"Database User: {db_config['user']}")
    print(f"Pool Size: {db_config['pool_size']}")


def example_r2_usage() -> None:
    """Example of how to use Cloudflare R2 configuration."""
    print(f"R2 Access Key: {CLOUDFLARE_R2_ACCESS_KEY_ID}")
    print(f"R2 Endpoint: {CLOUDFLARE_R2_ENDPOINT_URL}")
    print(f"R2 Bucket: {CLOUDFLARE_R2_BUCKET_NAME}")
    print(f"MLflow Artifact Root: {MLFLOW_ARTIFACT_ROOT}")


def example_app_config() -> None:
    """Example of how to use app configuration."""
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Supabase URL: {SUPABASE_URL}")


if __name__ == "__main__":
    example_database_usage()
    example_r2_usage()
    example_app_config()
