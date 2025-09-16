"""
DVC (Data Version Control) client for dataset versioning with R2 backend.

This module provides an interface to DVC for managing dataset versions,
lineage tracking, and metadata management with Cloudflare R2 as the storage backend.
"""

import asyncio
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import StorageConfig

logger = logging.getLogger(__name__)


class DVCClient:
    """Client for DVC dataset versioning operations."""

    def __init__(
        self,
        config: StorageConfig,
        repo_path: str | Path | None = None,
    ):
        """Initialize DVC client.

        Args:
            config: Storage configuration
            repo_path: Path to DVC repository (defaults to current directory)
        """
        self.config = config
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.logger = logging.getLogger(__name__)

        # Ensure DVC is available
        try:
            result = subprocess.run(
                ["dvc", "--version"], capture_output=True, text=True, check=True
            )
            self.logger.info(f"DVC version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise RuntimeError(
                "DVC not found. Install with: uv add dvc[s3] or pip install dvc[s3]"
            ) from err

    async def _run_dvc_command(
        self, command: list[str], cwd: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run DVC command asynchronously.

        Args:
            command: DVC command and arguments
            cwd: Working directory (defaults to repo_path)
            check: Whether to raise exception on non-zero exit

        Returns:
            Completed process result
        """
        full_command = ["dvc"] + command
        work_dir = cwd or self.repo_path

        self.logger.debug(f"Running DVC command: {' '.join(full_command)}")

        # Run command in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        process = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                full_command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=check,
                env=self._get_dvc_env(),
            ),
        )

        if process.returncode != 0 and check:
            self.logger.error(f"DVC command failed: {process.stderr}")
            raise subprocess.CalledProcessError(
                process.returncode, full_command, process.stdout, process.stderr
            )

        return process

    def _get_dvc_env(self) -> dict[str, str]:
        """Get environment variables for DVC commands.

        Returns:
            Environment variables dictionary
        """
        import os

        env = os.environ.copy()

        # Set AWS credentials for S3-compatible storage
        env.update(
            {
                "AWS_ACCESS_KEY_ID": self.config.r2_access_key_id,
                "AWS_SECRET_ACCESS_KEY": self.config.r2_secret_access_key,
                "AWS_ENDPOINT_URL": self.config.r2_endpoint_url,
                "AWS_REGION": self.config.r2_region,
            }
        )

        return env

    async def init_repo(self) -> bool:
        """Initialize DVC repository.

        Returns:
            True if initialization successful
        """
        try:
            await self._run_dvc_command(["init", "--no-scm"])
            self.logger.info("DVC repository initialized")
            return True
        except subprocess.CalledProcessError as e:
            if "already initialized" in e.stderr:
                self.logger.info("DVC repository already initialized")
                return True
            raise

    async def configure_remote(self) -> bool:
        """Configure R2 remote for DVC.

        Returns:
            True if configuration successful
        """
        try:
            # Add remote
            await self._run_dvc_command(
                [
                    "remote",
                    "add",
                    "-d",
                    self.config.dvc_remote_name,
                    self.config.dvc_remote_url,
                ]
            )

            # Configure remote settings for R2
            remote_config_commands = [
                [
                    "remote",
                    "modify",
                    self.config.dvc_remote_name,
                    "endpointurl",
                    self.config.r2_endpoint_url,
                ],
                [
                    "remote",
                    "modify",
                    self.config.dvc_remote_name,
                    "access_key_id",
                    self.config.r2_access_key_id,
                ],
                [
                    "remote",
                    "modify",
                    self.config.dvc_remote_name,
                    "secret_access_key",
                    self.config.r2_secret_access_key,
                ],
                [
                    "remote",
                    "modify",
                    self.config.dvc_remote_name,
                    "region",
                    self.config.r2_region,
                ],
            ]

            for cmd in remote_config_commands:
                await self._run_dvc_command(cmd)

            self.logger.info(f"DVC remote '{self.config.dvc_remote_name}' configured")
            return True

        except subprocess.CalledProcessError as e:
            if "already exists" in e.stderr:
                self.logger.info(
                    f"DVC remote '{self.config.dvc_remote_name}' already exists"
                )
                return True
            raise

    async def add_dataset(
        self, dataset_path: str | Path, remote: str | None = None
    ) -> str:
        """Add dataset to DVC tracking.

        Args:
            dataset_path: Path to dataset directory or file
            remote: Remote name (uses default if None)

        Returns:
            DVC file path (.dvc file)
        """
        try:
            dataset_path = Path(dataset_path)
            remote = remote or self.config.dvc_remote_name

            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

            # Add dataset to DVC
            await self._run_dvc_command(["add", str(dataset_path)])

            # Find the generated .dvc file
            dvc_file = dataset_path.with_suffix(dataset_path.suffix + ".dvc")
            if not dvc_file.exists():
                # For directories, DVC creates .dvc file with directory name
                if dataset_path.is_dir():
                    dvc_file = dataset_path.parent / f"{dataset_path.name}.dvc"

            if not dvc_file.exists():
                raise FileNotFoundError("DVC file not found after adding dataset")

            self.logger.info(f"Added dataset to DVC: {dataset_path} -> {dvc_file}")
            return str(dvc_file)

        except Exception as e:
            self.logger.error(f"Error adding dataset to DVC: {e}")
            raise

    async def version_dataset(
        self, dataset_path: str | Path, message: str, tag: str | None = None
    ) -> str:
        """Create a version of the dataset.

        Args:
            dataset_path: Path to dataset
            message: Version message
            tag: Optional version tag

        Returns:
            Version identifier (timestamp-based)
        """
        try:
            dataset_path = Path(dataset_path)

            # Generate version identifier
            version_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            if tag:
                version_id = f"{tag}_{version_id}"

            # Create metadata file
            metadata = {
                "version_id": version_id,
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
                "dataset_path": str(dataset_path),
                "tag": tag,
            }

            metadata_file = dataset_path.parent / f".dvc_metadata_{version_id}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            # Add metadata to DVC
            await self._run_dvc_command(["add", str(metadata_file)])

            # Push to remote
            await self.push_dataset(str(dataset_path))
            await self.push_dataset(str(metadata_file))

            self.logger.info(f"Created dataset version: {version_id}")
            return version_id

        except Exception as e:
            self.logger.error(f"Error versioning dataset: {e}")
            raise

    async def pull_dataset(self, dataset_id: str, target_path: str | Path) -> None:
        """Pull dataset from remote storage.

        Args:
            dataset_id: Dataset identifier or DVC file path
            target_path: Local target path
        """
        try:
            target_path = Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Pull using DVC
            await self._run_dvc_command(["pull", dataset_id])

            self.logger.info(f"Pulled dataset {dataset_id} to {target_path}")

        except Exception as e:
            self.logger.error(f"Error pulling dataset: {e}")
            raise

    async def push_dataset(self, dataset_id: str) -> None:
        """Push dataset to remote storage.

        Args:
            dataset_id: Dataset identifier or DVC file path
        """
        try:
            await self._run_dvc_command(["push", dataset_id])
            self.logger.info(f"Pushed dataset: {dataset_id}")

        except Exception as e:
            self.logger.error(f"Error pushing dataset: {e}")
            raise

    async def list_versions(self, dataset_path: str | Path) -> list[dict[str, Any]]:
        """List all versions of a dataset.

        Args:
            dataset_path: Path to dataset

        Returns:
            List of version metadata dictionaries
        """
        try:
            dataset_path = Path(dataset_path)
            versions = []

            # Find metadata files
            metadata_pattern = ".dvc_metadata_*.json"
            metadata_files = list(dataset_path.parent.glob(metadata_pattern))

            for metadata_file in metadata_files:
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    versions.append(metadata)
                except Exception as e:
                    self.logger.warning(
                        f"Error reading metadata file {metadata_file}: {e}"
                    )

            # Sort by timestamp
            versions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            self.logger.info(
                f"Found {len(versions)} versions for dataset: {dataset_path}"
            )
            return versions

        except Exception as e:
            self.logger.error(f"Error listing dataset versions: {e}")
            raise

    async def get_dataset_status(
        self, dataset_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Get status of datasets in DVC repository.

        Args:
            dataset_path: Specific dataset path (optional)

        Returns:
            Status information dictionary
        """
        try:
            cmd = ["status"]
            if dataset_path:
                cmd.append(str(dataset_path))

            result = await self._run_dvc_command(cmd, check=False)

            status_info = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "up_to_date": result.returncode == 0
                and "Data and pipelines are up to date" in result.stdout,
            }

            return status_info

        except Exception as e:
            self.logger.error(f"Error getting dataset status: {e}")
            raise

    async def remove_dataset(
        self, dataset_path: str | Path, keep_local: bool = True
    ) -> bool:
        """Remove dataset from DVC tracking.

        Args:
            dataset_path: Path to dataset
            keep_local: Whether to keep local files

        Returns:
            True if removal successful
        """
        try:
            cmd = ["remove", str(dataset_path)]
            if keep_local:
                cmd.append("--keep")

            await self._run_dvc_command(cmd)
            self.logger.info(f"Removed dataset from DVC: {dataset_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error removing dataset from DVC: {e}")
            return False
