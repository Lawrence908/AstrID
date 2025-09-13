"""Logging configuration for AstrID application."""

import logging
import os
from logging.handlers import RotatingFileHandler

import sentry_sdk

from src.core.constants import APP_VERSION, DEBUG, ENVIRONMENT, LOG_LEVEL, SENTRY_DSN

_logging_initialized = False


def setup_logging() -> None:
    """Configure logging for the entire application."""
    global _logging_initialized

    # Skip if already initialized
    if _logging_initialized:
        return

    # Use LOG_DIR environment variable or default to "logs"
    log_dir = os.getenv("LOG_DIR", "logs")

    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Set propagate=True for all loggers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.propagate = True

    # Sentry SDK initialization (only if DSN is provided)
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=ENVIRONMENT,
            release=APP_VERSION,
            debug=DEBUG,
            # Additional Sentry configuration
            traces_sample_rate=0.1 if ENVIRONMENT == "production" else 1.0,
            profiles_sample_rate=0.1 if ENVIRONMENT == "production" else 1.0,
        )
        logging.info("Sentry SDK initialized successfully")
    else:
        logging.warning("Sentry DSN not provided, error tracking disabled")

    _logging_initialized = True
    logging.info(f"Logging initialized for {ENVIRONMENT} environment")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: The name of the logger (typically __name__)

    Returns:
        A configured logger instance
    """
    # Ensure logging is set up
    if not _logging_initialized:
        setup_logging()

    return logging.getLogger(name)


def configure_domain_logger(domain_name: str) -> logging.Logger:
    """Configure a logger specifically for a domain.

    Args:
        domain_name: The name of the domain (e.g., 'curation', 'detection')

    Returns:
        A configured logger for the domain
    """
    logger = get_logger(f"astrid.domains.{domain_name}")
    logger.info(f"Domain logger initialized for {domain_name}")
    return logger
