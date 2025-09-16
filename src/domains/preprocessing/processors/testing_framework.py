"""Testing and validation framework for preprocessing pipelines.

This module provides comprehensive testing, benchmarking, and validation
utilities for astronomical image processing workflows.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.domains.detection.processors.anomaly_detection import AnomalyDetector

from .astronomical_image_processing import (
    AstronomicalImageProcessor,
    ImageDifferencingProcessor,
    SourceDetectionProcessor,
)


@dataclass
class ProcessingResult:
    """Container for processing results and metrics."""

    input_image: np.ndarray
    processed_image: np.ndarray
    quality_metrics: dict[str, float]
    processing_time: float
    method: str
    timestamp: str
    metadata: dict[str, Any]


class ProcessingPipeline:
    """Complete processing pipeline with validation and testing."""

    def __init__(self, config: dict | None = None):
        self.config = config or self.default_config()
        self.results_history = []
        self.image_processor = AstronomicalImageProcessor()
        self.differencing_processor = ImageDifferencingProcessor()
        self.source_processor = SourceDetectionProcessor()
        self.anomaly_detector = AnomalyDetector()

    def default_config(self) -> dict:
        """Default processing configuration."""
        return {
            "preprocessing": {
                "bias_correction": True,
                "flat_correction": True,
                "dark_correction": True,
                "cosmic_ray_removal": True,
                "background_subtraction": True,
                "noise_reduction": True,
            },
            "differencing": {"method": "zogy", "threshold": 3.0, "min_area": 5},
            "anomaly_detection": {"threshold": 0.5, "use_unet": False, "use_ml": True},
            "quality_thresholds": {
                "min_snr": 5.0,
                "min_contrast": 0.1,
                "max_noise": 50.0,
            },
        }

    def process_single_image(
        self,
        image: np.ndarray,
        reference_image: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> ProcessingResult:
        """Process a single image through the complete pipeline."""
        start_time = time.time()
        timestamp = datetime.utcnow().isoformat()

        # Step 1: Enhanced preprocessing
        processed_img, quality_metrics = (
            self.image_processor.enhance_astronomical_image(
                image, **self.config["preprocessing"]
            )
        )

        # Step 2: Image differencing (if reference available)
        differencing_metrics = {}
        if reference_image is not None:
            diff_img, diff_metrics = (
                self.differencing_processor.perform_image_differencing(
                    processed_img,
                    reference_image,
                    method=self.config["differencing"]["method"],
                )
            )
            differencing_metrics = diff_metrics

            # Step 3: Source detection
            sources, source_mask = self.source_processor.detect_sources_in_difference(
                diff_img,
                threshold=self.config["differencing"]["threshold"],
                min_area=self.config["differencing"]["min_area"],
            )
            quality_metrics["num_sources"] = len(sources)
            quality_metrics["max_source_significance"] = (
                max([s["max_significance"] for s in sources]) if sources else 0.0
            )

        # Step 4: Anomaly detection
        anomaly_results = {}
        if self.config["anomaly_detection"]["use_ml"]:
            anomaly_results = self.anomaly_detector.detect_anomalies_ml(processed_img)

        # Combine all metrics
        all_metrics = {**quality_metrics, **differencing_metrics, **anomaly_results}

        processing_time = time.time() - start_time

        result = ProcessingResult(
            input_image=image,
            processed_image=processed_img,
            quality_metrics=all_metrics,
            processing_time=processing_time,
            method=self.config["differencing"]["method"],
            timestamp=timestamp,
            metadata=metadata or {},
        )

        self.results_history.append(result)
        return result

    def batch_process(
        self,
        images: list[np.ndarray],
        reference_images: list[np.ndarray] | None = None,
        metadata_list: list[dict] | None = None,
    ) -> list[ProcessingResult]:
        """Process multiple images in batch."""
        results = []

        for i, img in enumerate(images):
            ref_img = reference_images[i] if reference_images else None
            meta = metadata_list[i] if metadata_list else None

            result = self.process_single_image(img, ref_img, meta)
            results.append(result)

            print(
                f"Processed image {i+1}/{len(images)} - Quality: {result.quality_metrics.get('snr', 0):.2f}"
            )

        return results

    def validate_quality(self, result: ProcessingResult) -> dict[str, bool]:
        """Validate processing quality against thresholds."""
        thresholds = self.config["quality_thresholds"]
        validation = {}

        validation["snr_acceptable"] = (
            result.quality_metrics.get("snr", 0) >= thresholds["min_snr"]
        )
        validation["contrast_acceptable"] = (
            result.quality_metrics.get("contrast", 0) >= thresholds["min_contrast"]
        )
        validation["noise_acceptable"] = (
            result.quality_metrics.get("std", 0) <= thresholds["max_noise"]
        )
        validation["processing_time_acceptable"] = (
            result.processing_time <= 30.0
        )  # 30 seconds max

        validation["overall_acceptable"] = all(validation.values())

        return validation

    def generate_quality_report(self) -> pd.DataFrame:
        """Generate comprehensive quality report."""
        if not self.results_history:
            return pd.DataFrame()

        data = []
        for result in self.results_history:
            validation = self.validate_quality(result)
            row = {
                "timestamp": result.timestamp,
                "method": result.method,
                "processing_time": result.processing_time,
                "snr": result.quality_metrics.get("snr", 0),
                "contrast": result.quality_metrics.get("contrast", 0),
                "noise_std": result.quality_metrics.get("std", 0),
                "num_sources": result.quality_metrics.get("num_sources", 0),
                "anomaly_score": result.quality_metrics.get(
                    "combined_anomaly_score", 0
                ),
                "quality_passed": validation["overall_acceptable"],
            }
            data.append(row)

        return pd.DataFrame(data)

    def save_results(self, output_dir: str = "processing_results"):
        """Save processing results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save quality report
        report_df = self.generate_quality_report()
        report_df.to_csv(output_path / "quality_report.csv", index=False)

        # Save individual results
        for i, result in enumerate(self.results_history):
            result_dir = output_path / f"result_{i:03d}"
            result_dir.mkdir(exist_ok=True)

            # Save images
            np.save(result_dir / "input.npy", result.input_image)
            np.save(result_dir / "processed.npy", result.processed_image)

            # Save metadata
            with open(result_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "quality_metrics": result.quality_metrics,
                        "processing_time": result.processing_time,
                        "method": result.method,
                        "timestamp": result.timestamp,
                        "metadata": result.metadata,
                    },
                    f,
                    indent=2,
                )

        print(f"Results saved to {output_path}")


class TestDatasetGenerator:
    """Generate synthetic test datasets for validation."""

    def create_test_dataset(
        self,
        num_images: int = 10,
        image_size: tuple[int, int] = (256, 256),
        noise_level: float = 0.1,
    ) -> list[np.ndarray]:
        """Create synthetic test dataset."""
        images = []

        for _ in range(num_images):
            # Create synthetic astronomical image
            img = np.random.normal(100, 20, image_size)

            # Add some structure (galaxy-like)
            y, x = np.ogrid[: image_size[0], : image_size[1]]
            center_y, center_x = image_size[0] // 2, image_size[1] // 2

            # Add elliptical structure
            a, b = 30, 20
            ellipse = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2
            img += 50 * np.exp(-ellipse)

            # Add noise
            img += np.random.normal(0, noise_level * np.std(img), image_size)

            # Add some random sources
            for _ in range(np.random.randint(3, 8)):
                y_src, x_src = (
                    np.random.randint(0, image_size[0]),
                    np.random.randint(0, image_size[1]),
                )
                flux = np.random.uniform(20, 100)

                # Add bounds checking to prevent IndexError
                y_min = max(0, y_src - 2)
                y_max = min(image_size[0], y_src + 3)
                x_min = max(0, x_src - 2)
                x_max = min(image_size[1], x_src + 3)

                img[y_min:y_max, x_min:x_max] += flux

            images.append(img)

        return images


class ProcessingBenchmark:
    """Benchmark different processing methods."""

    def benchmark_processing_methods(
        self, images: list[np.ndarray], reference_images: list[np.ndarray] | None = None
    ) -> pd.DataFrame:
        """Benchmark different processing methods."""
        methods = ["zogy", "classic", "optimal"]
        results = []

        for method in methods:
            config = {
                "differencing": {"method": method, "threshold": 3.0, "min_area": 5},
                "preprocessing": {
                    "bias_correction": True,
                    "flat_correction": True,
                    "dark_correction": True,
                    "cosmic_ray_removal": True,
                    "background_subtraction": True,
                    "noise_reduction": True,
                },
                "anomaly_detection": {
                    "threshold": 0.5,
                    "use_unet": False,
                    "use_ml": True,
                },
                "quality_thresholds": {
                    "min_snr": 5.0,
                    "min_contrast": 0.1,
                    "max_noise": 50.0,
                },
            }

            pipeline = ProcessingPipeline(config)

            start_time = time.time()
            batch_results = pipeline.batch_process(images, reference_images)
            total_time = time.time() - start_time

            # Calculate average metrics
            avg_snr = np.mean([r.quality_metrics.get("snr", 0) for r in batch_results])
            avg_processing_time = np.mean([r.processing_time for r in batch_results])
            avg_sources = np.mean(
                [r.quality_metrics.get("num_sources", 0) for r in batch_results]
            )

            results.append(
                {
                    "method": method,
                    "total_time": total_time,
                    "avg_processing_time": avg_processing_time,
                    "avg_snr": avg_snr,
                    "avg_sources": avg_sources,
                    "num_images": len(images),
                }
            )

        return pd.DataFrame(results)


class PerformanceAnalyzer:
    """Analyze processing performance and provide recommendations."""

    def analyze_processing_performance(
        self, pipeline: ProcessingPipeline
    ) -> dict[str, Any]:
        """Analyze processing performance and provide recommendations."""
        if not pipeline.results_history:
            return {"error": "No processing results available"}

        report_df = pipeline.generate_quality_report()

        analysis = {
            "summary": {
                "total_images": len(report_df),
                "success_rate": float(report_df["quality_passed"].mean()),
                "avg_processing_time": float(report_df["processing_time"].mean()),
                "avg_snr": float(report_df["snr"].mean()),
                "avg_contrast": float(report_df["contrast"].mean()),
            },
            "performance_metrics": {
                "fastest_processing": float(report_df["processing_time"].min()),
                "slowest_processing": float(report_df["processing_time"].max()),
                "highest_snr": float(report_df["snr"].max()),
                "lowest_snr": float(report_df["snr"].min()),
                "most_sources_detected": int(report_df["num_sources"].max()),
            },
            "recommendations": [],
        }

        # Generate recommendations
        if report_df["quality_passed"].mean() < 0.8:
            analysis["recommendations"].append(
                "Consider adjusting quality thresholds - success rate is low"
            )

        if report_df["processing_time"].mean() > 20:
            analysis["recommendations"].append(
                "Processing time is high - consider optimizing algorithms"
            )

        if report_df["snr"].mean() < 10:
            analysis["recommendations"].append(
                "Low SNR detected - check preprocessing parameters"
            )

        if report_df["contrast"].mean() < 0.2:
            analysis["recommendations"].append(
                "Low contrast - consider adjusting image enhancement"
            )

        return analysis


class ConfigurationManager:
    """Manage processing configurations for different use cases."""

    def create_processing_configuration(
        self,
        image_type: str = "galaxy",
        quality_priority: str = "high",
        speed_priority: str = "medium",
    ) -> dict:
        """Create processing configuration based on requirements."""
        configs = {
            "galaxy": {
                "preprocessing": {
                    "bias_correction": True,
                    "flat_correction": True,
                    "dark_correction": True,
                    "cosmic_ray_removal": True,
                    "background_subtraction": True,
                    "noise_reduction": True,
                },
                "differencing": {"method": "zogy", "threshold": 3.0, "min_area": 5},
            },
            "star_field": {
                "preprocessing": {
                    "bias_correction": True,
                    "flat_correction": True,
                    "dark_correction": False,
                    "cosmic_ray_removal": True,
                    "background_subtraction": True,
                    "noise_reduction": False,
                },
                "differencing": {"method": "optimal", "threshold": 2.5, "min_area": 3},
            },
            "nebula": {
                "preprocessing": {
                    "bias_correction": True,
                    "flat_correction": True,
                    "dark_correction": True,
                    "cosmic_ray_removal": True,
                    "background_subtraction": True,
                    "noise_reduction": True,
                },
                "differencing": {"method": "classic", "threshold": 4.0, "min_area": 8},
            },
        }

        base_config = configs.get(image_type, configs["galaxy"])

        # Adjust based on quality priority
        if quality_priority == "high":
            base_config["quality_thresholds"] = {
                "min_snr": 10.0,
                "min_contrast": 0.2,
                "max_noise": 30.0,
            }
        elif quality_priority == "medium":
            base_config["quality_thresholds"] = {
                "min_snr": 5.0,
                "min_contrast": 0.1,
                "max_noise": 50.0,
            }
        else:  # low
            base_config["quality_thresholds"] = {
                "min_snr": 3.0,
                "min_contrast": 0.05,
                "max_noise": 100.0,
            }

        # Adjust based on speed priority
        if speed_priority == "high":
            base_config["preprocessing"]["noise_reduction"] = False
            base_config["preprocessing"]["cosmic_ray_removal"] = False
        elif speed_priority == "low":
            base_config["preprocessing"]["noise_reduction"] = True
            base_config["preprocessing"]["cosmic_ray_removal"] = True

        # Add anomaly detection configuration
        base_config["anomaly_detection"] = {
            "threshold": 0.5,
            "use_unet": False,
            "use_ml": True,
        }

        return base_config
