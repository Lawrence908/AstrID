"""Scikit-image based advanced image processing for astronomical images."""

import logging
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.feature import (
    blob_dog,
    blob_doh,
    blob_log,
    corner_harris,
    corner_peaks,
    match_template,
)
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
)

try:
    from skimage.morphology.footprints import rectangle, square
except ImportError:
    # Fallback for older scikit-image versions
    def rectangle(height, width):
        return np.ones((height, width), dtype=bool)

    def square(size):
        return np.ones((size, size), dtype=bool)


from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    denoise_tv_chambolle,
)
from skimage.segmentation import (
    felzenszwalb,
    quickshift,
    slic,
    watershed,
)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class SegmentationMethod(str, Enum):
    """Image segmentation methods."""

    WATERSHED = "watershed"
    SLIC = "slic"
    FELZENSZWALB = "felzenszwalb"
    QUICKSHIFT = "quickshift"
    THRESHOLD = "threshold"


class FeatureDetector(str, Enum):
    """Feature detection methods."""

    BLOB_LOG = "blob_log"
    BLOB_DOG = "blob_dog"
    BLOB_DOH = "blob_doh"
    CORNER_HARRIS = "corner_harris"
    TEMPLATE_MATCH = "template_match"


class MorphologyOperation(str, Enum):
    """Morphological operations."""

    EROSION = "erosion"
    DILATION = "dilation"
    OPENING = "opening"
    CLOSING = "closing"
    GRADIENT = "gradient"


class RestorationMethod(str, Enum):
    """Image restoration methods."""

    WIENER = "wiener"
    DENOISE_BILATERAL = "denoise_bilateral"
    DENOISE_NL_MEANS = "denoise_nl_means"
    DENOISE_TV_CHAMBOLLE = "denoise_tv_chambolle"


class FootprintShape(str, Enum):
    """Morphological footprint shapes."""

    DISK = "disk"
    SQUARE = "square"
    RECTANGLE = "rectangle"


class ScikitProcessor:
    """Scikit-image based processor for advanced astronomical image processing."""

    def __init__(self) -> None:
        """Initialize the Scikit-image processor."""
        self.logger = logger.getChild(self.__class__.__name__)

    def segment_image(
        self, image: NDArray[np.floating], method: str, parameters: dict[str, Any]
    ) -> NDArray[np.integer]:
        """
        Segment the image using various algorithms.

        Args:
            image: Input image array
            method: Segmentation method to use
            parameters: Method-specific parameters

        Returns:
            Segmented image with labeled regions

        Raises:
            ValueError: If method is not supported
        """
        try:
            # Normalize image to 0-1 range for segmentation
            img_norm = self._normalize_image(image)

            if method == SegmentationMethod.WATERSHED:
                # Watershed segmentation
                markers = parameters.get("markers", None)
                if markers is None:
                    # Generate markers automatically
                    from skimage.filters import sobel

                    _elevation_map = sobel(img_norm)
                    markers = np.zeros_like(img_norm, dtype=int)
                    markers[img_norm < 0.3] = 1
                    markers[img_norm > 0.7] = 2

                segmented = watershed(img_norm, markers)

            elif method == SegmentationMethod.SLIC:
                # SLIC superpixel segmentation
                n_segments = parameters.get("n_segments", 100)
                compactness = parameters.get("compactness", 10)
                # Handle 2D grayscale images by specifying channel_axis=None
                segmented = slic(
                    img_norm,
                    n_segments=n_segments,
                    compactness=compactness,
                    channel_axis=None,
                )

            elif method == SegmentationMethod.FELZENSZWALB:
                # Felzenszwalb's efficient graph-based segmentation
                scale = parameters.get("scale", 100)
                sigma = parameters.get("sigma", 0.5)
                min_size = parameters.get("min_size", 50)
                segmented = felzenszwalb(
                    img_norm,
                    scale=scale,
                    sigma=sigma,
                    min_size=min_size,
                    channel_axis=None,
                )

            elif method == SegmentationMethod.QUICKSHIFT:
                # Quickshift segmentation (doesn't support channel_axis parameter)
                kernel_size = parameters.get("kernel_size", 3)
                max_dist = parameters.get("max_dist", 6)
                ratio = parameters.get("ratio", 0.5)
                segmented = quickshift(
                    img_norm, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio
                )

            elif method == SegmentationMethod.THRESHOLD:
                # Simple threshold segmentation
                threshold = parameters.get("threshold", 0.5)
                segmented = (img_norm > threshold).astype(int)

            else:
                raise ValueError(f"Unsupported segmentation method: {method}")

            return segmented.astype(np.int32)

        except Exception as e:
            self.logger.error(f"Image segmentation failed: {e}")
            raise

    def detect_features(
        self, image: NDArray[np.floating], detector: str, parameters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Detect features in the image.

        Args:
            image: Input image array
            detector: Feature detector to use
            parameters: Detector-specific parameters

        Returns:
            List of detected features with properties

        Raises:
            ValueError: If detector is not supported
        """
        try:
            img_norm = self._normalize_image(image)
            features = []

            if detector == FeatureDetector.BLOB_LOG:
                # Laplacian of Gaussian blob detection
                min_sigma = parameters.get("min_sigma", 1)
                max_sigma = parameters.get("max_sigma", 30)
                num_sigma = parameters.get("num_sigma", 10)
                threshold = parameters.get("threshold", 0.1)

                blobs = blob_log(
                    img_norm,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold,
                )

                for blob in blobs:
                    y, x, r = blob
                    features.append(
                        {
                            "type": "blob_log",
                            "x": float(x),
                            "y": float(y),
                            "radius": float(r * np.sqrt(2)),
                            "intensity": float(img_norm[int(y), int(x)]),
                        }
                    )

            elif detector == FeatureDetector.BLOB_DOG:
                # Difference of Gaussian blob detection
                min_sigma = parameters.get("min_sigma", 1)
                max_sigma = parameters.get("max_sigma", 30)
                threshold = parameters.get("threshold", 0.1)

                blobs = blob_dog(
                    img_norm,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    threshold=threshold,
                )

                for blob in blobs:
                    y, x, r = blob
                    features.append(
                        {
                            "type": "blob_dog",
                            "x": float(x),
                            "y": float(y),
                            "radius": float(r * np.sqrt(2)),
                            "intensity": float(img_norm[int(y), int(x)]),
                        }
                    )

            elif detector == FeatureDetector.BLOB_DOH:
                # Determinant of Hessian blob detection
                min_sigma = parameters.get("min_sigma", 1)
                max_sigma = parameters.get("max_sigma", 30)
                threshold = parameters.get("threshold", 0.01)

                blobs = blob_doh(
                    img_norm,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    threshold=threshold,
                )

                for blob in blobs:
                    y, x, r = blob
                    features.append(
                        {
                            "type": "blob_doh",
                            "x": float(x),
                            "y": float(y),
                            "radius": float(r),
                            "intensity": float(img_norm[int(y), int(x)]),
                        }
                    )

            elif detector == FeatureDetector.CORNER_HARRIS:
                # Harris corner detection
                min_distance = parameters.get("min_distance", 5)
                threshold_abs = parameters.get("threshold_abs", 0.1)

                corners = corner_peaks(
                    corner_harris(img_norm),
                    min_distance=min_distance,
                    threshold_abs=threshold_abs,
                )

                for corner in corners:
                    y, x = corner
                    features.append(
                        {
                            "type": "corner_harris",
                            "x": float(x),
                            "y": float(y),
                            "response": float(corner_harris(img_norm)[y, x]),
                            "intensity": float(img_norm[y, x]),
                        }
                    )

            elif detector == FeatureDetector.TEMPLATE_MATCH:
                # Template matching
                template = parameters.get("template")
                if template is None:
                    raise ValueError("Template is required for template matching")

                result = match_template(img_norm, template)

                # Find local maxima using a simple approach
                from scipy import ndimage

                local_maxima = ndimage.maximum_filter(result, size=20) == result
                local_maxima = local_maxima & (result > 0.3)  # threshold

                peaks = np.where(local_maxima)
                for i in range(len(peaks[0])):
                    y, x = peaks[0][i], peaks[1][i]
                    features.append(
                        {
                            "type": "template_match",
                            "x": float(x),
                            "y": float(y),
                            "correlation": float(result[y, x]),
                            "intensity": float(img_norm[y, x]),
                        }
                    )

            else:
                raise ValueError(f"Unsupported feature detector: {detector}")

            return features

        except Exception as e:
            self.logger.error(f"Feature detection failed: {e}")
            raise

    def apply_morphology(
        self, image: NDArray[np.floating], operation: str, footprint: NDArray[np.bool_]
    ) -> NDArray[np.floating]:
        """
        Apply morphological operations to the image.

        Args:
            image: Input image array
            operation: Morphological operation to apply
            footprint: Structural element for the operation

        Returns:
            Processed image array

        Raises:
            ValueError: If operation is not supported
        """
        try:
            # Convert to binary image for morphological operations
            binary_img = image > np.mean(image)

            if operation == MorphologyOperation.EROSION:
                result = binary_erosion(binary_img, footprint)

            elif operation == MorphologyOperation.DILATION:
                result = binary_dilation(binary_img, footprint)

            elif operation == MorphologyOperation.OPENING:
                result = binary_opening(binary_img, footprint)

            elif operation == MorphologyOperation.CLOSING:
                result = binary_closing(binary_img, footprint)

            elif operation == MorphologyOperation.GRADIENT:
                dilated = binary_dilation(binary_img, footprint)
                eroded = binary_erosion(binary_img, footprint)
                result = dilated ^ eroded

            else:
                raise ValueError(f"Unsupported morphological operation: {operation}")

            # Convert back to original image range
            return result.astype(image.dtype) * (
                np.max(image) - np.min(image)
            ) + np.min(image)

        except Exception as e:
            self.logger.error(f"Morphological operation failed: {e}")
            raise

    def measure_image_properties(
        self, image: NDArray[np.floating], properties: list[str]
    ) -> dict[str, Any]:
        """
        Measure various properties of the image.

        Args:
            image: Input image array
            properties: List of properties to measure

        Returns:
            Dictionary containing measured properties
        """
        try:
            measurements = {}

            # Basic statistics
            if "mean" in properties:
                measurements["mean"] = float(np.mean(image))
            if "std" in properties:
                measurements["std"] = float(np.std(image))
            if "min" in properties:
                measurements["min"] = float(np.min(image))
            if "max" in properties:
                measurements["max"] = float(np.max(image))
            if "median" in properties:
                measurements["median"] = float(np.median(image))

            # Entropy and information measures
            if "entropy" in properties:
                from skimage.measure import shannon_entropy

                measurements["entropy"] = float(shannon_entropy(image))

            # Regional properties
            if any(
                prop in properties
                for prop in ["area", "perimeter", "centroid", "eccentricity"]
            ):
                # Create labeled regions
                binary_img = image > np.mean(image)
                labeled_img = label(binary_img)
                regions = regionprops(labeled_img, intensity_image=image)

                if "area" in properties:
                    measurements["total_area"] = sum(
                        [region.area for region in regions]
                    )
                    measurements["regions_count"] = len(regions)

                if "perimeter" in properties:
                    measurements["total_perimeter"] = sum(
                        [region.perimeter for region in regions]
                    )

                if "centroid" in properties:
                    centroids = [region.centroid for region in regions]
                    measurements["centroids"] = [
                        [float(c[0]), float(c[1])] for c in centroids
                    ]

                if "eccentricity" in properties:
                    eccentricities = [region.eccentricity for region in regions]
                    measurements["mean_eccentricity"] = (
                        float(np.mean(eccentricities)) if eccentricities else 0.0
                    )

            # Texture measures
            if "local_binary_pattern" in properties:
                from skimage.feature import local_binary_pattern

                lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
                measurements["lbp_histogram"] = np.histogram(lbp, bins=10)[0].tolist()

            return measurements

        except Exception as e:
            self.logger.error(f"Image property measurement failed: {e}")
            raise

    def restore_image(
        self, image: NDArray[np.floating], method: str, parameters: dict[str, Any]
    ) -> NDArray[np.floating]:
        """
        Restore/denoise the image using various algorithms.

        Args:
            image: Input image array
            method: Restoration method to use
            parameters: Method-specific parameters

        Returns:
            Restored image array

        Raises:
            ValueError: If method is not supported
        """
        try:
            img_norm = self._normalize_image(image)

            if method == RestorationMethod.WIENER:
                # Wiener filter restoration - using a simple implementation
                # since scikit-image's wiener has different parameters
                from scipy import ndimage

                # Simple deconvolution approximation
                restored = ndimage.gaussian_filter(img_norm, sigma=1.0)

            elif method == RestorationMethod.DENOISE_BILATERAL:
                # Bilateral denoising
                sigma_color = parameters.get("sigma_color", 0.05)
                sigma_spatial = parameters.get("sigma_spatial", 15)
                restored = denoise_bilateral(
                    img_norm, sigma_color=sigma_color, sigma_spatial=sigma_spatial
                )

            elif method == RestorationMethod.DENOISE_NL_MEANS:
                # Non-local means denoising
                patch_size = parameters.get("patch_size", 7)
                patch_distance = parameters.get("patch_distance", 11)
                h = parameters.get("h", 0.1)
                restored = denoise_nl_means(
                    img_norm, patch_size=patch_size, patch_distance=patch_distance, h=h
                )

            elif method == RestorationMethod.DENOISE_TV_CHAMBOLLE:
                # Total variation denoising
                weight = parameters.get("weight", 0.1)
                restored = denoise_tv_chambolle(img_norm, weight=weight)

            else:
                raise ValueError(f"Unsupported restoration method: {method}")

            # Scale back to original range
            return self._denormalize_image(restored, image)

        except Exception as e:
            self.logger.error(f"Image restoration failed: {e}")
            raise

    def classify_pixels(
        self,
        image: NDArray[np.floating],
        classifier: str,
        features: NDArray[np.floating],
    ) -> NDArray[np.integer]:
        """
        Classify pixels using machine learning.

        Args:
            image: Input image array
            classifier: Type of classifier to use
            features: Feature array for classification

        Returns:
            Classified image with pixel labels

        Raises:
            ValueError: If classifier is not supported
        """
        try:
            h, w = image.shape
            img_norm = self._normalize_image(image)

            if classifier == "kmeans":
                # K-means clustering
                n_clusters = features.shape[0] if len(features.shape) > 1 else 3
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)

                # Reshape image for clustering
                pixels = img_norm.reshape(-1, 1)
                labels = kmeans.fit_predict(pixels)
                classified = labels.reshape(h, w)

            elif classifier == "random_forest":
                # Random forest classification (requires training data)
                if features.shape[0] < img_norm.size:
                    raise ValueError("Insufficient training features for random forest")

                rf = RandomForestClassifier(n_estimators=100, random_state=42)

                # For demonstration, use intensity-based features
                X = img_norm.ravel().reshape(-1, 1)
                y = np.digitize(X.ravel(), bins=np.linspace(0, 1, 4)) - 1

                rf.fit(X, y)
                classified = rf.predict(X).reshape(h, w)

            else:
                raise ValueError(f"Unsupported classifier: {classifier}")

            return classified.astype(np.int32)

        except Exception as e:
            self.logger.error(f"Pixel classification failed: {e}")
            raise

    def create_footprint(
        self, shape: str, size: int, **kwargs: Any
    ) -> NDArray[np.bool_]:
        """
        Create morphological footprint/structural element.

        Args:
            shape: Shape of the footprint
            size: Size of the footprint
            **kwargs: Additional shape-specific parameters

        Returns:
            Boolean array representing the footprint

        Raises:
            ValueError: If shape is not supported
        """
        try:
            if shape == FootprintShape.DISK:
                result = disk(size)

            elif shape == FootprintShape.SQUARE:
                result = square(size)

            elif shape == FootprintShape.RECTANGLE:
                height = kwargs.get("height", size)
                width = kwargs.get("width", size)
                result = rectangle(height, width)

            else:
                raise ValueError(f"Unsupported footprint shape: {shape}")

            # Ensure result is boolean array
            return np.asarray(result, dtype=np.bool_)

        except Exception as e:
            self.logger.error(f"Footprint creation failed: {e}")
            raise

    def _normalize_image(self, image: NDArray[np.floating]) -> NDArray[np.floating]:
        """Normalize image to 0-1 range."""
        img_min, img_max = np.min(image), np.max(image)
        if img_max == img_min:
            return np.zeros_like(image)
        return (image - img_min) / (img_max - img_min)

    def _denormalize_image(
        self, normalized: NDArray[np.floating], original: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Denormalize image back to original range."""
        orig_min, orig_max = np.min(original), np.max(original)
        return normalized * (orig_max - orig_min) + orig_min

    def get_processing_info(self) -> dict[str, Any]:
        """
        Get information about available processing operations.

        Returns:
            Dictionary containing operation information
        """
        return {
            "segmentation_methods": [method.value for method in SegmentationMethod],
            "feature_detectors": [detector.value for detector in FeatureDetector],
            "morphology_operations": [op.value for op in MorphologyOperation],
            "restoration_methods": [method.value for method in RestorationMethod],
            "footprint_shapes": [shape.value for shape in FootprintShape],
            "scikit_image_version": "0.21.0+",  # Version varies with installation
        }
