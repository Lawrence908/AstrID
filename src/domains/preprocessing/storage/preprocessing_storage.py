"""Storage system for preprocessing results and artifacts."""

import gzip
import hashlib
import json
import logging
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class StorageCompressionLevel(int, Enum):
    """Compression levels for storage."""

    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9


class StorageFormat(str, Enum):
    """Storage formats for different data types."""

    NUMPY = "numpy"
    FITS = "fits"
    JSON = "json"
    PICKLE = "pickle"


class PreprocessingStorage:
    """Storage system for preprocessing results and artifacts."""

    def __init__(
        self,
        base_path: str = "/tmp/astrid/preprocessing",
        compression_level: int = StorageCompressionLevel.MEDIUM,
        enable_versioning: bool = True,
    ) -> None:
        """
        Initialize the preprocessing storage system.

        Args:
            base_path: Base directory for storage
            compression_level: Default compression level
            enable_versioning: Whether to enable versioning
        """
        self.base_path = Path(base_path)
        self.compression_level = compression_level
        self.enable_versioning = enable_versioning
        self.logger = logger.getChild(self.__class__.__name__)

        # Create directory structure
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Set up the directory structure for storage."""
        directories = [
            "images",
            "parameters",
            "metrics",
            "metadata",
            "archive",
            "versions",
        ]

        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)

    def store_processed_image(
        self,
        image: NDArray[np.floating],
        metadata: dict[str, Any],
        observation_id: UUID,
    ) -> str:
        """
        Store a processed image with metadata.

        Args:
            image: Processed image array
            metadata: Image metadata
            observation_id: Associated observation ID

        Returns:
            Storage ID for retrieving the image

        Raises:
            IOError: If storage operation fails
        """
        try:
            # Generate unique storage ID
            storage_id = str(uuid4())

            # Create storage paths
            image_path = self.base_path / "images" / f"{storage_id}.npz"
            metadata_path = self.base_path / "metadata" / f"{storage_id}.json"

            # Add storage metadata
            full_metadata = {
                "storage_id": storage_id,
                "observation_id": str(observation_id),
                "stored_at": datetime.utcnow().isoformat(),
                "image_shape": image.shape,
                "image_dtype": str(image.dtype),
                "compression_level": self.compression_level,
                "checksum": self._calculate_checksum(image),
                **metadata,
            }

            # Store image with compression
            if self.compression_level > 0:
                np.savez_compressed(image_path, image=image)
            else:
                np.savez(image_path, image=image)

            # Store metadata
            with open(metadata_path, "w") as f:
                json.dump(full_metadata, f, indent=2, default=str)

            # Handle versioning
            if self.enable_versioning:
                self._store_version(storage_id, observation_id, "image")

            self.logger.info(f"Stored processed image: {storage_id}")
            return storage_id

        except Exception as e:
            self.logger.error(f"Failed to store processed image: {e}")
            raise OSError(f"Storage operation failed: {e}") from e

    def retrieve_processed_image(
        self, storage_id: str
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """
        Retrieve a processed image and its metadata.

        Args:
            storage_id: Storage ID of the image

        Returns:
            Tuple of (image array, metadata)

        Raises:
            FileNotFoundError: If image is not found
            IOError: If retrieval operation fails
        """
        try:
            # Construct paths
            image_path = self.base_path / "images" / f"{storage_id}.npz"
            metadata_path = self.base_path / "metadata" / f"{storage_id}.json"

            # Check if files exist
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {storage_id}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {storage_id}")

            # Load image
            with np.load(image_path) as data:
                image = data["image"]

            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Verify checksum
            if "checksum" in metadata:
                current_checksum = self._calculate_checksum(image)
                if current_checksum != metadata["checksum"]:
                    self.logger.warning(f"Checksum mismatch for {storage_id}")

            self.logger.info(f"Retrieved processed image: {storage_id}")
            return image, metadata

        except Exception as e:
            self.logger.error(f"Failed to retrieve processed image: {e}")
            raise OSError(f"Retrieval operation failed: {e}") from e

    def store_processing_parameters(
        self, parameters: dict[str, Any], observation_id: UUID
    ) -> str:
        """
        Store processing parameters.

        Args:
            parameters: Processing parameters
            observation_id: Associated observation ID

        Returns:
            Storage ID for retrieving the parameters
        """
        try:
            storage_id = str(uuid4())
            parameters_path = self.base_path / "parameters" / f"{storage_id}.json"

            # Add metadata
            full_parameters = {
                "storage_id": storage_id,
                "observation_id": str(observation_id),
                "stored_at": datetime.utcnow().isoformat(),
                "parameters": parameters,
            }

            # Store parameters
            with open(parameters_path, "w") as f:
                json.dump(full_parameters, f, indent=2, default=str)

            # Handle versioning
            if self.enable_versioning:
                self._store_version(storage_id, observation_id, "parameters")

            self.logger.info(f"Stored processing parameters: {storage_id}")
            return storage_id

        except Exception as e:
            self.logger.error(f"Failed to store processing parameters: {e}")
            raise OSError(f"Storage operation failed: {e}") from e

    def retrieve_processing_parameters(self, observation_id: UUID) -> dict[str, Any]:
        """
        Retrieve processing parameters for an observation.

        Args:
            observation_id: Observation ID

        Returns:
            Processing parameters dictionary
        """
        try:
            # Find parameters file
            parameters_dir = self.base_path / "parameters"
            observation_str = str(observation_id)

            for param_file in parameters_dir.glob("*.json"):
                with open(param_file) as f:
                    data = json.load(f)
                    if data.get("observation_id") == observation_str:
                        return data.get("parameters", {})

            raise FileNotFoundError(
                f"Parameters not found for observation: {observation_id}"
            )

        except Exception as e:
            self.logger.error(f"Failed to retrieve processing parameters: {e}")
            raise OSError(f"Retrieval operation failed: {e}") from e

    def store_processing_metrics(
        self, metrics: dict[str, Any], observation_id: UUID
    ) -> str:
        """
        Store processing metrics.

        Args:
            metrics: Processing metrics
            observation_id: Associated observation ID

        Returns:
            Storage ID for retrieving the metrics
        """
        try:
            storage_id = str(uuid4())
            metrics_path = self.base_path / "metrics" / f"{storage_id}.json"

            # Add metadata
            full_metrics = {
                "storage_id": storage_id,
                "observation_id": str(observation_id),
                "stored_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
            }

            # Store metrics
            with open(metrics_path, "w") as f:
                json.dump(full_metrics, f, indent=2, default=str)

            # Handle versioning
            if self.enable_versioning:
                self._store_version(storage_id, observation_id, "metrics")

            self.logger.info(f"Stored processing metrics: {storage_id}")
            return storage_id

        except Exception as e:
            self.logger.error(f"Failed to store processing metrics: {e}")
            raise OSError(f"Storage operation failed: {e}") from e

    def archive_processed_data(self, observation_id: UUID) -> None:
        """
        Archive all processed data for an observation.

        Args:
            observation_id: Observation ID to archive
        """
        try:
            observation_str = str(observation_id)
            archive_dir = self.base_path / "archive" / observation_str
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Archive all related files
            for directory in ["images", "parameters", "metrics", "metadata"]:
                source_dir = self.base_path / directory

                for file_path in source_dir.glob("*.json"):
                    # Check if file is related to this observation
                    try:
                        with open(file_path) as f:
                            data = json.load(f)
                            if data.get("observation_id") == observation_str:
                                # Copy to archive with compression
                                archive_file = (
                                    archive_dir / f"{directory}_{file_path.name}.gz"
                                )
                                with open(file_path, "rb") as f_in:
                                    with gzip.open(archive_file, "wb") as f_out:
                                        f_out.write(f_in.read())
                    except (json.JSONDecodeError, KeyError):
                        continue

                # Archive numpy files
                if directory == "images":
                    for npz_file in source_dir.glob("*.npz"):
                        # Load metadata to check observation ID
                        storage_id = npz_file.stem
                        metadata_file = (
                            self.base_path / "metadata" / f"{storage_id}.json"
                        )

                        if metadata_file.exists():
                            try:
                                with open(metadata_file) as f:
                                    metadata = json.load(f)
                                    if (
                                        metadata.get("observation_id")
                                        == observation_str
                                    ):
                                        # Copy image file
                                        archive_file = (
                                            archive_dir / f"image_{npz_file.name}"
                                        )
                                        import shutil

                                        shutil.copy2(npz_file, archive_file)
                            except (json.JSONDecodeError, KeyError):
                                continue

            # Create archive manifest
            manifest = {
                "observation_id": observation_str,
                "archived_at": datetime.utcnow().isoformat(),
                "files": [f.name for f in archive_dir.iterdir()],
            }

            manifest_path = archive_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            self.logger.info(
                f"Archived processed data for observation: {observation_id}"
            )

        except Exception as e:
            self.logger.error(f"Failed to archive processed data: {e}")
            raise OSError(f"Archive operation failed: {e}") from e

    def _store_version(
        self, storage_id: str, observation_id: UUID, data_type: str
    ) -> None:
        """Store version information."""
        try:
            versions_path = self.base_path / "versions" / f"{observation_id}.json"

            # Load existing versions or create new
            if versions_path.exists():
                with open(versions_path) as f:
                    versions = json.load(f)
            else:
                versions = {"observation_id": str(observation_id), "versions": []}

            # Add new version
            version_info = {
                "storage_id": storage_id,
                "data_type": data_type,
                "created_at": datetime.utcnow().isoformat(),
                "version": len(versions["versions"]) + 1,
            }

            versions["versions"].append(version_info)

            # Save versions
            with open(versions_path, "w") as f:
                json.dump(versions, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to store version information: {e}")

    def _calculate_checksum(self, data: NDArray[np.floating]) -> str:
        """Calculate MD5 checksum for data integrity."""
        return hashlib.md5(data.tobytes()).hexdigest()

    def get_storage_statistics(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary containing storage statistics
        """
        try:
            stats = {
                "total_images": 0,
                "total_parameters": 0,
                "total_metrics": 0,
                "total_size_bytes": 0,
                "oldest_entry": None,
                "newest_entry": None,
            }

            # Count files and calculate sizes
            for directory in ["images", "parameters", "metrics", "metadata"]:
                dir_path = self.base_path / directory
                if dir_path.exists():
                    files = list(dir_path.iterdir())

                    if directory == "images":
                        stats["total_images"] = len(
                            [f for f in files if f.suffix == ".npz"]
                        )
                    elif directory == "parameters":
                        stats["total_parameters"] = len(
                            [f for f in files if f.suffix == ".json"]
                        )
                    elif directory == "metrics":
                        stats["total_metrics"] = len(
                            [f for f in files if f.suffix == ".json"]
                        )

                    # Calculate total size
                    for file_path in files:
                        if file_path.is_file():
                            stats["total_size_bytes"] += file_path.stat().st_size

                            # Track oldest and newest
                            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if (
                                stats["oldest_entry"] is None
                                or mtime < stats["oldest_entry"]
                            ):
                                stats["oldest_entry"] = mtime
                            if (
                                stats["newest_entry"] is None
                                or mtime > stats["newest_entry"]
                            ):
                                stats["newest_entry"] = mtime

            # Convert dates to ISO format
            if stats["oldest_entry"]:
                stats["oldest_entry"] = stats["oldest_entry"].isoformat()
            if stats["newest_entry"]:
                stats["newest_entry"] = stats["newest_entry"].isoformat()

            # Calculate size in human readable format
            stats["total_size_mb"] = stats["total_size_bytes"] / (1024 * 1024)
            stats["total_size_gb"] = stats["total_size_mb"] / 1024

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {e}")
            return {"error": str(e)}

    def cleanup_old_data(self, days_old: int = 30) -> Mapping[str, int | str]:
        """
        Clean up data older than specified days.

        Args:
            days_old: Age threshold in days

        Returns:
            Dictionary containing cleanup statistics
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            cleanup_stats = {"files_removed": 0, "bytes_freed": 0}

            # Clean up old files
            for directory in ["images", "parameters", "metrics", "metadata"]:
                dir_path = self.base_path / directory
                if dir_path.exists():
                    for file_path in dir_path.iterdir():
                        if file_path.is_file():
                            # Check modification time
                            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if mtime < cutoff_date:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                cleanup_stats["files_removed"] += 1
                                cleanup_stats["bytes_freed"] += file_size

            self.logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return {"error": str(e)}
