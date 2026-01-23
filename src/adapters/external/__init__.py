"""
External adapters for AstrID.

This package contains adapters for external services and APIs including:
- MAST (Mikulski Archive for Space Telescopes)
- IRSA (Infrared Science Archive) - ZTF, PTF, WISE, etc.
- ArchiveRouter - Unified interface for querying multiple archives
- SkyView (NASA's SkyView Virtual Observatory)

Note: R2 storage has been moved to src/infrastructure/storage/
"""

# Import the main classes for easier access
from .archive_router import ArchiveRouter
from .irsa import IRSAClient
from .mast import MASTClient
from .skyview import SkyViewClient

__all__ = [
    "ArchiveRouter",
    "IRSAClient",
    "MASTClient",
    "SkyViewClient",
]
