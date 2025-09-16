"""Image processing utilities for the preprocessing domain."""

from .fits_processing import AdvancedFITSProcessor
from .astronomical_image_processing import (
    AstronomicalImageProcessor,
    ImageDifferencingProcessor,
    SourceDetectionProcessor
)
from .testing_framework import (
    ProcessingPipeline,
    ProcessingResult,
    TestDatasetGenerator,
    ProcessingBenchmark,
    PerformanceAnalyzer,
    ConfigurationManager
)

__all__ = [
    "AdvancedFITSProcessor",
    "AstronomicalImageProcessor",
    "ImageDifferencingProcessor", 
    "SourceDetectionProcessor",
    "ProcessingPipeline",
    "ProcessingResult",
    "TestDatasetGenerator",
    "ProcessingBenchmark",
    "PerformanceAnalyzer",
    "ConfigurationManager"
]
