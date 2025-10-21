"""Image normalization and scaling for astronomical images."""

import logging
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage import exposure, transform

logger = logging.getLogger(__name__)


class NormalizationMethod(str, Enum):
    """Image normalization methods."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    QUANTILE = "quantile"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    ADAPTIVE_HISTOGRAM = "adaptive_histogram"
    PERCENTILE = "percentile"


class ScalingMethod(str, Enum):
    """Image scaling methods."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    AREA = "area"


class HistogramMethod(str, Enum):
    """Histogram processing methods."""

    EQUALIZATION = "equalization"
    ADAPTIVE = "adaptive"
    CLAHE = "clahe"
    STRETCHING = "stretching"


class ImageNormalizer:
    """Advanced image normalizer for astronomical images."""

    def __init__(self) -> None:
        """Initialize the image normalizer."""
        self.logger = logger.getChild(self.__class__.__name__)

    def normalize_intensity(
        self, image: NDArray[np.floating], method: str
    ) -> NDArray[np.floating]:
        """
        Normalize image intensity using various methods.

        Args:
            image: Input image array
            method: Normalization method to use

        Returns:
            Normalized image array

        Raises:
            ValueError: If method is not supported
        """
        try:
            if method == NormalizationMethod.MIN_MAX:
                # Min-max normalization to [0, 1]
                img_min, img_max = np.min(image), np.max(image)
                if img_max == img_min:
                    return np.zeros_like(image)
                normalized = (image - img_min) / (img_max - img_min)

            elif method == NormalizationMethod.Z_SCORE:
                # Z-score normalization (zero mean, unit variance)
                mean, std = np.mean(image), np.std(image)
                if std == 0:
                    return np.zeros_like(image)
                normalized = (image - mean) / std

            elif method == NormalizationMethod.ROBUST:
                # Robust normalization using median and MAD
                median = np.median(image)
                mad = np.median(np.abs(image - median))
                if mad == 0:
                    return np.zeros_like(image)
                normalized = (image - median) / (
                    1.4826 * mad
                )  # 1.4826 for normal distribution

            elif method == NormalizationMethod.QUANTILE:
                # Quantile-based normalization (1st to 99th percentile)
                q1, q99 = np.percentile(image, [1, 99])
                if q99 == q1:
                    return np.zeros_like(image)
                normalized = np.clip((image - q1) / (q99 - q1), 0, 1)

            elif method == NormalizationMethod.HISTOGRAM_EQUALIZATION:
                # Histogram equalization
                normalized = exposure.equalize_hist(image)

            elif method == NormalizationMethod.ADAPTIVE_HISTOGRAM:
                # Adaptive histogram equalization (CLAHE)
                normalized = exposure.equalize_adapthist(image, clip_limit=0.03)

            elif method == NormalizationMethod.PERCENTILE:
                # Percentile-based normalization (5th to 95th percentile)
                p5, p95 = np.percentile(image, [5, 95])
                if p95 == p5:
                    return np.zeros_like(image)
                normalized = np.clip((image - p5) / (p95 - p5), 0, 1)

            else:
                raise ValueError(f"Unsupported normalization method: {method}")

            return normalized.astype(image.dtype)

        except Exception as e:
            self.logger.error(f"Intensity normalization failed: {e}")
            raise

    def scale_image(
        self, image: NDArray[np.floating], target_size: tuple[int, int], method: str
    ) -> NDArray[np.floating]:
        """
        Scale image to target size using various interpolation methods.

        Args:
            image: Input image array
            target_size: Target size as (height, width)
            method: Scaling method to use

        Returns:
            Scaled image array

        Raises:
            ValueError: If method is not supported
        """
        try:
            target_height, target_width = target_size
            current_height, current_width = image.shape

            # Calculate scaling factors
            scale_y = target_height / current_height
            scale_x = target_width / current_width

            # Map scaling methods to skimage orders
            method_map = {
                ScalingMethod.NEAREST: 0,
                ScalingMethod.BILINEAR: 1,
                ScalingMethod.BICUBIC: 3,
                ScalingMethod.LANCZOS: 3,  # Use cubic as approximation
                ScalingMethod.AREA: 1,  # Use bilinear as approximation
            }

            if method not in method_map:
                raise ValueError(f"Unsupported scaling method: {method}")

            # Perform scaling
            scaled = transform.rescale(
                image,
                scale=(scale_y, scale_x),
                order=method_map[method],
                preserve_range=True,
                anti_aliasing=True,
            )

            return scaled.astype(image.dtype)

        except Exception as e:
            self.logger.error(f"Image scaling failed: {e}")
            raise

    def normalize_histogram(
        self, image: NDArray[np.floating], method: str
    ) -> NDArray[np.floating]:
        """
        Normalize image histogram using various methods.

        Args:
            image: Input image array
            method: Histogram normalization method

        Returns:
            Histogram-normalized image array

        Raises:
            ValueError: If method is not supported
        """
        try:
            # Ensure image is in the correct range for scikit-image histogram functions
            # Convert to 0-1 range using min-max normalization
            img_min, img_max = np.min(image), np.max(image)
            if img_max == img_min:
                img_norm = np.zeros_like(image)
            else:
                img_norm = (image - img_min) / (img_max - img_min)

            if method == HistogramMethod.EQUALIZATION:
                # Global histogram equalization
                normalized = exposure.equalize_hist(img_norm)

            elif method == HistogramMethod.ADAPTIVE:
                # Adaptive histogram equalization
                normalized = exposure.equalize_adapthist(img_norm)

            elif method == HistogramMethod.CLAHE:
                # Contrast Limited Adaptive Histogram Equalization
                normalized = exposure.equalize_adapthist(
                    img_norm, clip_limit=0.03, nbins=256
                )

            elif method == HistogramMethod.STRETCHING:
                # Contrast stretching
                p2, p98 = np.percentile(img_norm, (2, 98))
                normalized = exposure.rescale_intensity(img_norm, in_range=(p2, p98))

            else:
                raise ValueError(f"Unsupported histogram method: {method}")

            return normalized.astype(image.dtype)

        except Exception as e:
            self.logger.error(f"Histogram normalization failed: {e}")
            raise

    def apply_z_score_normalization(
        self, image: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Apply z-score normalization (zero mean, unit variance).

        Args:
            image: Input image array

        Returns:
            Z-score normalized image array
        """
        try:
            mean = np.mean(image)
            std = np.std(image)

            if std == 0:
                self.logger.warning("Standard deviation is zero, returning zero array")
                return np.zeros_like(image)

            normalized = (image - mean) / std
            return normalized.astype(image.dtype)

        except Exception as e:
            self.logger.error(f"Z-score normalization failed: {e}")
            raise

    def normalize_to_reference(
        self, image: NDArray[np.floating], reference: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Normalize image to match reference image statistics.

        Args:
            image: Input image to normalize
            reference: Reference image for target statistics

        Returns:
            Normalized image matching reference statistics
        """
        try:
            # Calculate statistics
            img_mean, img_std = np.mean(image), np.std(image)
            ref_mean, ref_std = np.mean(reference), np.std(reference)

            if img_std == 0:
                self.logger.warning("Input image has zero standard deviation")
                return np.full_like(image, ref_mean)

            # Normalize to reference statistics
            normalized = (image - img_mean) / img_std * ref_std + ref_mean

            return normalized.astype(image.dtype)

        except Exception as e:
            self.logger.error(f"Reference normalization failed: {e}")
            raise

    def apply_adaptive_normalization(
        self, image: NDArray[np.floating], window_size: int = 64
    ) -> NDArray[np.floating]:
        """
        Apply adaptive normalization using local statistics.

        Args:
            image: Input image array
            window_size: Size of the local window for statistics

        Returns:
            Adaptively normalized image array
        """
        try:
            # Pad image to handle borders
            pad_size = window_size // 2
            padded = np.pad(image, pad_size, mode="reflect")

            # Initialize output
            normalized = np.zeros_like(image)

            # Process each pixel with local window
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # Extract local window
                    window = padded[i : i + window_size, j : j + window_size]

                    # Calculate local statistics
                    local_mean = np.mean(window)
                    local_std = np.std(window)

                    # Normalize pixel using local statistics
                    if local_std > 0:
                        normalized[i, j] = (image[i, j] - local_mean) / local_std
                    else:
                        normalized[i, j] = 0

            return normalized.astype(image.dtype)

        except Exception as e:
            self.logger.error(f"Adaptive normalization failed: {e}")
            raise

    def apply_robust_scaling(
        self,
        image: NDArray[np.floating],
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> NDArray[np.floating]:
        """
        Apply robust scaling using quantiles.

        Args:
            image: Input image array
            quantile_range: Quantile range for scaling (default: IQR)

        Returns:
            Robustly scaled image array
        """
        try:
            q_min, q_max = quantile_range

            # Calculate quantiles
            q1 = np.percentile(image, q_min)
            q3 = np.percentile(image, q_max)
            median = np.median(image)

            # Calculate IQR
            iqr = q3 - q1

            if iqr == 0:
                self.logger.warning("IQR is zero, using min-max scaling")
                return self.normalize_intensity(image, NormalizationMethod.MIN_MAX)

            # Apply robust scaling
            scaled = (image - median) / iqr

            return scaled.astype(image.dtype)

        except Exception as e:
            self.logger.error(f"Robust scaling failed: {e}")
            raise

    def assess_normalization_quality(
        self, original: NDArray[np.floating], normalized: NDArray[np.floating]
    ) -> dict[str, float]:
        """
        Assess the quality of normalization.

        Args:
            original: Original image array
            normalized: Normalized image array

        Returns:
            Dictionary containing quality metrics
        """
        try:
            metrics = {}

            # Basic statistics comparison
            metrics["original_mean"] = float(np.mean(original))
            metrics["original_std"] = float(np.std(original))
            metrics["normalized_mean"] = float(np.mean(normalized))
            metrics["normalized_std"] = float(np.std(normalized))

            # Dynamic range
            metrics["original_range"] = float(np.max(original) - np.min(original))
            metrics["normalized_range"] = float(np.max(normalized) - np.min(normalized))

            # Information preservation (correlation)
            correlation = np.corrcoef(original.flatten(), normalized.flatten())[0, 1]
            metrics["correlation"] = float(correlation)

            # Entropy comparison
            try:
                from skimage.measure import shannon_entropy

                metrics["original_entropy"] = float(shannon_entropy(original))
                metrics["normalized_entropy"] = float(shannon_entropy(normalized))
                metrics["entropy_ratio"] = float(
                    metrics["normalized_entropy"] / metrics["original_entropy"]
                    if metrics["original_entropy"] > 0
                    else 0
                )
            except ImportError:
                self.logger.warning("Skimage not available for entropy calculation")

            # Contrast improvement
            original_contrast = (
                np.std(original) / np.mean(original) if np.mean(original) > 0 else 0
            )
            normalized_contrast = (
                np.std(normalized) / np.mean(normalized)
                if np.mean(normalized) > 0
                else 0
            )
            metrics["contrast_improvement"] = float(
                normalized_contrast / original_contrast if original_contrast > 0 else 0
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            raise

    def get_normalization_info(self) -> dict[str, Any]:
        """
        Get information about available normalization methods.

        Returns:
            Dictionary containing method information
        """
        return {
            "normalization_methods": [method.value for method in NormalizationMethod],
            "scaling_methods": [method.value for method in ScalingMethod],
            "histogram_methods": [method.value for method in HistogramMethod],
            "recommended_methods": {
                "astronomical_images": ["quantile", "robust", "z_score"],
                "general_purpose": ["min_max", "histogram_equalization"],
                "high_dynamic_range": ["adaptive_histogram", "clahe"],
            },
        }
