"""Constants and configuration for AstrID application."""

import os
from uuid import UUID

from dotenv import load_dotenv

load_dotenv()

# Application
APP_VERSION = "0.1.0"  # Using version from core/constants.py
APP_NAME = "AstrID"

# Environment
ENVIRONMENT = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGGING_LEVEL = "INFO"  # From core/constants.py
SENTRY_DSN = os.getenv("SENTRY_DSN")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# API metadata (from core/constants.py)
API_TITLE = "AstrID API"
API_DESCRIPTION = (
    "Astronomical Identification: Temporal Dataset Preparation and Anomaly Detection"
)

# CORS configuration (from core/constants.py)
CORS_ORIGINS = ["*"]  # Configure appropriately for production
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_USER = os.getenv("REDIS_USER", "default")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "foo")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Database Configuration
DATABASE_TEST_URL = os.getenv("DATABASE_TEST_URL")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_SSL_CERT_PATH = os.getenv("SUPABASE_SSL_CERT_PATH")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# Cloudflare R2 Configuration
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_R2_TOKEN_VALUE = os.getenv("CLOUDFLARE_R2_TOKEN_VALUE")
CLOUDFLARE_R2_ACCESS_KEY_ID = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID")
CLOUDFLARE_R2_SECRET_ACCESS_KEY = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
CLOUDFLARE_R2_BUCKET_NAME = os.getenv("CLOUDFLARE_R2_BUCKET_NAME", "astrid")
CLOUDFLARE_R2_ENDPOINT_URL = os.getenv("CLOUDFLARE_R2_ENDPOINT_URL")
CLOUDFLARE_EU_R2_ENDPOINT_URL = os.getenv("CLOUDFLARE_EU_R2_ENDPOINT_URL")

# MLflow Configuration
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://astrid-models")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# MLflow Supabase Configuration
MLFLOW_SUPABASE_URL = os.getenv("MLFLOW_SUPABASE_URL")
MLFLOW_SUPABASE_PROJECT_REF = os.getenv("MLFLOW_SUPABASE_PROJECT_REF")
MLFLOW_SUPABASE_PASSWORD = os.getenv("MLFLOW_SUPABASE_PASSWORD")
MLFLOW_SUPABASE_HOST = os.getenv("MLFLOW_SUPABASE_HOST")
MLFLOW_SUPABASE_SSL_CERT_PATH = os.getenv("MLFLOW_SUPABASE_SSL_CERT_PATH")
MLFLOW_SUPABASE_KEY = os.getenv("MLFLOW_SUPABASE_KEY")
MLFLOW_SUPABASE_SERVICE_ROLE_KEY = os.getenv("MLFLOW_SUPABASE_SERVICE_ROLE_KEY")
MLFLOW_SUPABASE_JWT_SECRET = os.getenv("MLFLOW_SUPABASE_JWT_SECRET")

# DVC Configuration
DVC_REMOTE_NAME = os.getenv("DVC_REMOTE_NAME", "r2")
DVC_REMOTE_URL = os.getenv("DVC_REMOTE_URL", "s3://astrid-data")

# Prefect Configuration
PREFECT_API_URL = os.getenv("PREFECT_API_URL")

# External APIs
ASTROQUERY_TIMEOUT = int(os.getenv("ASTROQUERY_TIMEOUT", "300"))
VIZIER_TIMEOUT = int(os.getenv("VIZIER_TIMEOUT", "300"))

# Twitter OAuth
TWITTER_APP_ID = os.getenv("TWITTER_APP_ID")
TWITTER_APP_SECRET = os.getenv("TWITTER_APP_SECRET")

# Database Pool Configuration
POOL_CONFIGS = {
    "development": {
        "pool_size": 5,
        "max_overflow": 10,
    },
    "production": {
        "pool_size": 3,
        "max_overflow": 2,
    },
}

# Get pool config for current environment
pool_config = POOL_CONFIGS.get(ENVIRONMENT, POOL_CONFIGS["development"])

# Database connection settings
DB_CONFIG = {
    "host": SUPABASE_HOST or "aws-0-us-west-1.pooler.supabase.com:5432",
    "database": "postgres",
    "user": f"postgres.{SUPABASE_PROJECT_REF}" if SUPABASE_PROJECT_REF else "postgres",
    "password": SUPABASE_PASSWORD,
    "ssl": "require",
    "ssl_cert_path": SUPABASE_SSL_CERT_PATH,
    # Connection pool settings
    "pool_size": pool_config["pool_size"],
    "max_overflow": pool_config["max_overflow"],
    "pool_timeout": 30,  # Seconds to wait for a connection from pool
    "pool_recycle": 1800,  # Recycle connections after 30 minutes
    "pool_pre_ping": True,  # Verify connections before use
    # Query timeout settings
    "command_timeout": 60,  # Default timeout for queries in seconds
    "echo": DEBUG,  # Enable echo in debug mode
}

# MLflow Database connection settings
MLFLOW_DB_CONFIG = {
    "host": MLFLOW_SUPABASE_HOST or "aws-0-us-west-1.pooler.supabase.com:5432",
    "database": "postgres",
    "user": f"postgres.{MLFLOW_SUPABASE_PROJECT_REF}"
    if MLFLOW_SUPABASE_PROJECT_REF
    else "postgres",
    "password": MLFLOW_SUPABASE_PASSWORD,
    "ssl": "require",
    "ssl_cert_path": MLFLOW_SUPABASE_SSL_CERT_PATH,
    # Connection pool settings
    "pool_size": pool_config["pool_size"],
    "max_overflow": pool_config["max_overflow"],
    "pool_timeout": 30,  # Seconds to wait for a connection from pool
    "pool_recycle": 1800,  # Recycle connections after 30 minutes
    "pool_pre_ping": True,  # Verify connections before use
    # Query timeout settings
    "command_timeout": 60,  # Default timeout for queries in seconds
    "echo": DEBUG,  # Enable echo in debug mode
}


def get_database_url() -> str:
    """Get the database URL."""
    return (
        f"postgresql+asyncpg://"
        f"{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )


def get_mlflow_tracking_uri() -> str:
    """Get the MLflow tracking URI."""
    return (
        f"postgresql+asyncpg://"
        f"{MLFLOW_DB_CONFIG['user']}:{MLFLOW_DB_CONFIG['password']}@"
        f"{MLFLOW_DB_CONFIG['host']}/{MLFLOW_DB_CONFIG['database']}"
    )


def get_db_config() -> dict:
    """Get the database configuration dictionary."""
    return DB_CONFIG.copy()


# System profile UUID for audit fields (created_by/updated_by)
SYSTEM_PROFILE_ID = UUID("00000000-0000-0000-0000-000000000000")

# Rate limiting configuration
DEFAULT_RATE_LIMITS = {
    "GLOBAL": "10000 per minute",
    "AUTHENTICATED": "500 per minute",
    "AUTH": "10 per minute",
    "HEALTH": "10 per minute",
}

RATE_LIMIT_HEADERS = {
    "X-RateLimit-Limit": True,
    "X-RateLimit-Remaining": True,
    "X-RateLimit-Reset": True,
}

RATE_LIMIT_LOGGING = {
    "ENABLED": True,
    "LOG_LEVEL": "WARNING",
}
