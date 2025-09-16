"""FITS processing pipeline components for observations domain."""

from .fits_processor import FITSProcessor
from .metadata_extractor import MetadataExtractor
from .pipeline import FITSProcessingPipeline, FITSProcessingResult
from .wcs_processor import WCSProcessor

__all__ = [
    "FITSProcessor",
    "WCSProcessor",
    "MetadataExtractor",
    "FITSProcessingPipeline",
    "FITSProcessingResult",
]
