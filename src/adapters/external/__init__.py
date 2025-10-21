"""
External adapters for AstrID.

This package contains adapters for external services and APIs including:
- MAST (Mikulski Archive for Space Telescopes)
- SkyView (NASA's SkyView Virtual Observatory)

Note: R2 storage has been moved to src/infrastructure/storage/
"""

# Import the main classes for easier access
from .mast import MASTClient
from .skyview import SkyViewClient

__all__ = [
    "MASTClient",
    "SkyViewClient",
]
