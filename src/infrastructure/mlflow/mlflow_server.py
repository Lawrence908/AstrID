"""
MLflow tracking server management for AstrID.

This module provides server management, health checks, and monitoring
for the MLflow tracking server with proper integration into the AstrID workflow.
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..storage.config import StorageConfig
from .config import MLflowConfig

logger = logging.getLogger(__name__)


@dataclass
class ServerStatus:
    """MLflow server status information."""

    is_running: bool
    health_status: str
    response_time_ms: float
    version: str | None = None
    error_message: str | None = None


class MLflowServer:
    """MLflow tracking server management."""

    def __init__(self, config: MLflowConfig):
        """Initialize MLflow server manager.

        Args:
            config: MLflow configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._process: subprocess.Popen | None = None

        # Configure requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def start_tracking_server(
        self, host: str, port: int, background: bool = True
    ) -> None:
        """Start MLflow tracking server.

        Args:
            host: Server host address
            port: Server port
            background: Whether to run server in background

        Raises:
            RuntimeError: If server fails to start
        """
        try:
            # Update config with provided host/port
            self.config.server_host = host
            self.config.server_port = port

            # Validate configuration
            self.config.validate()

            # Get server command
            cmd = self.config.get_server_command()
            self.logger.info(f"Starting MLflow server: {cmd}")

            if background:
                # Start server in background
                self._process = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**self.config.get_environment_variables()},
                )

                # Wait for server to start
                self._wait_for_server_startup()
                self.logger.info(f"MLflow server started on {host}:{port}")
            else:
                # Run server in foreground (blocking)
                subprocess.run(
                    cmd.split(),
                    env={**self.config.get_environment_variables()},
                    check=True,
                )

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to start MLflow server: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error starting MLflow server: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def stop_tracking_server(self) -> None:
        """Stop MLflow tracking server.

        Raises:
            RuntimeError: If server fails to stop
        """
        try:
            if self._process and self._process.poll() is None:
                self.logger.info("Stopping MLflow server...")
                self._process.terminate()

                # Wait for graceful shutdown
                try:
                    self._process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Server didn't stop gracefully, forcing kill")
                    self._process.kill()
                    self._process.wait()

                self._process = None
                self.logger.info("MLflow server stopped")
            else:
                self.logger.warning("MLflow server is not running")

        except Exception as e:
            error_msg = f"Failed to stop MLflow server: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def configure_artifact_store(self, storage_config: StorageConfig) -> None:
        """Configure MLflow artifact store with storage configuration.

        Args:
            storage_config: Storage configuration for artifact store
        """
        try:
            # Update MLflow config with storage settings
            self.config.storage_config = storage_config
            self.config.artifact_root = storage_config.mlflow_artifact_root

            # Set environment variables for S3/R2 configuration
            env_vars = self.config.get_environment_variables()
            for key, value in env_vars.items():
                if key.startswith(("MLFLOW_S3_", "AWS_")):
                    import os

                    os.environ[key] = value

            self.logger.info(
                f"Configured artifact store: {storage_config.mlflow_artifact_root}"
            )

        except Exception as e:
            error_msg = f"Failed to configure artifact store: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def setup_database_backend(self, database_url: str) -> None:
        """Set up database backend for MLflow.

        Args:
            database_url: Database connection URL
        """
        try:
            self.config.database_url = database_url

            # Test database connection
            self._test_database_connection(database_url)

            self.logger.info(f"Configured database backend: {database_url}")

        except Exception as e:
            error_msg = f"Failed to setup database backend: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def configure_authentication(self, auth_config: dict[str, Any]) -> None:
        """Configure MLflow authentication.

        Args:
            auth_config: Authentication configuration
        """
        try:
            self.config.auth_config = auth_config
            self.config.authentication_enabled = True

            self.logger.info("Configured MLflow authentication")

        except Exception as e:
            error_msg = f"Failed to configure authentication: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_server_status(self) -> ServerStatus:
        """Get MLflow server status.

        Returns:
            Server status information
        """
        try:
            # Check if process is running
            is_running = self._process is not None and self._process.poll() is None

            if not is_running:
                return ServerStatus(
                    is_running=False,
                    health_status="stopped",
                    response_time_ms=0.0,
                    error_message="Server process not running",
                )

            # Test server health
            start_time = time.time()
            try:
                response = self.session.get(
                    f"{self.config.tracking_uri}/health", timeout=5
                )
                response_time_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    health_data = response.json()
                    return ServerStatus(
                        is_running=True,
                        health_status="healthy",
                        response_time_ms=response_time_ms,
                        version=health_data.get("version"),
                    )
                else:
                    return ServerStatus(
                        is_running=True,
                        health_status="unhealthy",
                        response_time_ms=response_time_ms,
                        error_message=f"HTTP {response.status_code}",
                    )

            except requests.RequestException as e:
                response_time_ms = (time.time() - start_time) * 1000
                return ServerStatus(
                    is_running=True,
                    health_status="unreachable",
                    response_time_ms=response_time_ms,
                    error_message=str(e),
                )

        except Exception as e:
            return ServerStatus(
                is_running=False,
                health_status="error",
                response_time_ms=0.0,
                error_message=str(e),
            )

    def _wait_for_server_startup(self, timeout: int = 60) -> None:
        """Wait for server to start up.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If server doesn't start within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.session.get(
                    f"{self.config.tracking_uri}/health", timeout=2
                )
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            time.sleep(1)

        # If we get here, server didn't start
        if self._process:
            self._process.terminate()
            self._process = None

        raise RuntimeError(f"MLflow server failed to start within {timeout} seconds")

    def _test_database_connection(self, database_url: str) -> None:
        """Test database connection.

        Args:
            database_url: Database connection URL

        Raises:
            RuntimeError: If database connection fails
        """
        try:
            # Parse database URL to extract connection details
            if database_url.startswith("postgresql"):
                import psycopg2

                # Test PostgreSQL connection
                conn = psycopg2.connect(database_url)
                conn.close()
            elif database_url.startswith("sqlite"):
                import sqlite3

                # Test SQLite connection
                conn = sqlite3.connect(database_url.replace("sqlite:///", ""))
                conn.close()
            else:
                self.logger.warning(f"Unknown database type in URL: {database_url}")

        except ImportError as e:
            raise RuntimeError(f"Database driver not available: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {e}") from e

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_tracking_server()
