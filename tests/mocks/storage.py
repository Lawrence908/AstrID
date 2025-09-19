"""Mock storage client for testing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class MockStorageClient:
    """Mock cloud storage client for testing."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.upload_delay = 0.0
        self.download_delay = 0.0
        self.error_on_upload = False
        self.error_on_download = False

    async def upload_file(
        self,
        file_path: str | Path,
        remote_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Mock file upload."""
        if self.error_on_upload:
            raise Exception("Simulated upload error")

        if self.upload_delay > 0:
            await asyncio.sleep(self.upload_delay)

        # Simulate reading local file
        if isinstance(file_path, str):
            file_path = Path(file_path)

        content = b"mock_file_content_" + str(file_path).encode()
        self.files[remote_path] = content

        if metadata:
            self.metadata[remote_path] = metadata
        else:
            self.metadata[remote_path] = {
                "size": len(content),
                "content_type": "application/octet-stream",
                "uploaded_at": "2024-09-19T12:00:00Z",
            }

        return f"mock://storage/{remote_path}"

    async def download_file(
        self, remote_path: str, local_path: str | Path | None = None
    ) -> bytes:
        """Mock file download."""
        if self.error_on_download:
            raise Exception("Simulated download error")

        if self.download_delay > 0:
            await asyncio.sleep(self.download_delay)

        if remote_path not in self.files:
            raise FileNotFoundError(f"File not found: {remote_path}")

        content = self.files[remote_path]

        if local_path:
            # Simulate writing to local file
            if isinstance(local_path, str):
                local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(content)

        return content

    async def delete_file(self, remote_path: str) -> bool:
        """Mock file deletion."""
        if remote_path in self.files:
            del self.files[remote_path]
            if remote_path in self.metadata:
                del self.metadata[remote_path]
            return True
        return False

    async def file_exists(self, remote_path: str) -> bool:
        """Check if file exists."""
        return remote_path in self.files

    async def list_files(self, prefix: str = "", limit: int | None = None) -> list[str]:
        """List files with optional prefix filter."""
        files = [path for path in self.files.keys() if path.startswith(prefix)]

        if limit:
            files = files[:limit]

        return files

    async def get_file_metadata(self, remote_path: str) -> dict[str, Any]:
        """Get file metadata."""
        if remote_path not in self.metadata:
            raise FileNotFoundError(f"File not found: {remote_path}")
        return self.metadata[remote_path].copy()

    async def copy_file(self, source_path: str, dest_path: str) -> bool:
        """Copy file within storage."""
        if source_path not in self.files:
            return False

        self.files[dest_path] = self.files[source_path]
        if source_path in self.metadata:
            self.metadata[dest_path] = self.metadata[source_path].copy()
        return True

    async def get_signed_url(self, remote_path: str, expires_in: int = 3600) -> str:
        """Generate a signed URL."""
        return f"https://mock-storage.example.com/{remote_path}?expires={expires_in}"

    def set_upload_delay(self, delay: float) -> None:
        """Set artificial delay for uploads."""
        self.upload_delay = delay

    def set_download_delay(self, delay: float) -> None:
        """Set artificial delay for downloads."""
        self.download_delay = delay

    def simulate_upload_error(self, should_error: bool = True) -> None:
        """Enable/disable upload error simulation."""
        self.error_on_upload = should_error

    def simulate_download_error(self, should_error: bool = True) -> None:
        """Enable/disable download error simulation."""
        self.error_on_download = should_error

    def clear_storage(self) -> None:
        """Clear all stored files."""
        self.files.clear()
        self.metadata.clear()

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(len(content) for content in self.files.values())
        return {
            "file_count": len(self.files),
            "total_size": total_size,
            "files": list(self.files.keys()),
        }
