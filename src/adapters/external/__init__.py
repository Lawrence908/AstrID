"""
External adapters for AstrID.

This package contains adapters for external services and APIs including:
- MAST (Mikulski Archive for Space Telescopes)
- SkyView (NASA's SkyView Virtual Observatory)
- Cloudflare R2 storage
"""

# Import the main classes for easier access
from .mast import MASTClient
from .r2 import ContentAddressableStorage, R2StorageClient
from .skyview import SkyViewClient

__all__ = [
    "MASTClient",
    "R2StorageClient",
    "ContentAddressableStorage",
    "SkyViewClient",
]
