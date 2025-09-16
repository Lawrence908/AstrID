"""Image processing utilities for the preprocessing domain."""

from .astronomical_image_processing import (
    AstronomicalImageProcessor,
    ImageDifferencingProcessor,
    SourceDetectionProcessor,
)
from .fits_processing import AdvancedFITSProcessor
from .opencv_processor import (
    ContrastMethod,
    EdgeDetectionMethod,
    FilterType,
    MorphologicalOperation,
    NoiseRemovalMethod,
    OpenCVProcessor,
)
from .scikit_processor import (
    FeatureDetector,
    FootprintShape,
    MorphologyOperation,
    RestorationMethod,
    ScikitProcessor,
    SegmentationMethod,
)
from .testing_framework import (
    ConfigurationManager,
    PerformanceAnalyzer,
    ProcessingBenchmark,
    ProcessingPipeline,
    ProcessingResult,
    TestDatasetGenerator,
)

__all__ = [
    "AdvancedFITSProcessor",
    "AstronomicalImageProcessor",
    "ImageDifferencingProcessor",
    "SourceDetectionProcessor",
    "OpenCVProcessor",
    "MorphologicalOperation",
    "EdgeDetectionMethod",
    "FilterType",
    "ContrastMethod",
    "NoiseRemovalMethod",
    "ScikitProcessor",
    "SegmentationMethod",
    "FeatureDetector",
    "MorphologyOperation",
    "RestorationMethod",
    "FootprintShape",
    "ProcessingPipeline",
    "ProcessingResult",
    "TestDatasetGenerator",
    "ProcessingBenchmark",
    "PerformanceAnalyzer",
    "ConfigurationManager",
]
