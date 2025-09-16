"""OpenCV-based image processing for astronomical images."""

import logging
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Handle OpenCV import with fallback for headless environments
try:
    import cv2
except ImportError as e:
    if "libGL" in str(e):
        # Headless environment - install opencv-python-headless instead
        print("Warning: OpenCV GUI libraries not available. Using headless mode.")
        try:
            import cv2
        except ImportError as err:
            raise ImportError(
                "OpenCV not available. Install opencv-python-headless for headless environments."
            ) from err
    else:
        raise

logger = logging.getLogger(__name__)


class MorphologicalOperation(str, Enum):
    """Morphological operations supported by OpenCV."""

    OPENING = "opening"
    CLOSING = "closing"
    EROSION = "erosion"
    DILATION = "dilation"
    GRADIENT = "gradient"
    TOP_HAT = "top_hat"
    BLACK_HAT = "black_hat"


class EdgeDetectionMethod(str, Enum):
    """Edge detection methods."""

    CANNY = "canny"
    SOBEL = "sobel"
    LAPLACIAN = "laplacian"
    SCHARR = "scharr"


class FilterType(str, Enum):
    """Filter types for image processing."""

    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"
    BOX = "box"
    MOTION_BLUR = "motion_blur"


class ContrastMethod(str, Enum):
    """Contrast enhancement methods."""

    CLAHE = "clahe"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    GAMMA_CORRECTION = "gamma_correction"
    ADAPTIVE_HISTOGRAM = "adaptive_histogram"


class NoiseRemovalMethod(str, Enum):
    """Noise removal methods."""

    GAUSSIAN_BLUR = "gaussian_blur"
    NON_LOCAL_MEANS = "non_local_means"
    FAST_NON_LOCAL_MEANS = "fast_non_local_means"
    BILATERAL_FILTER = "bilateral_filter"


class OpenCVProcessor:
    """OpenCV-based image processor for astronomical images."""

    def __init__(self) -> None:
        """Initialize the OpenCV processor."""
        self.logger = logger.getChild(self.__class__.__name__)

    def apply_morphological_operations(
        self, image: NDArray[np.floating], operation: str, kernel_size: int = 5
    ) -> NDArray[np.floating]:
        """
        Apply morphological operations to the image.

        Args:
            image: Input image array
            operation: Type of morphological operation
            kernel_size: Size of the morphological kernel

        Returns:
            Processed image array

        Raises:
            ValueError: If operation is not supported
        """
        try:
            # Convert to uint8 for OpenCV operations
            img_uint8 = self._to_uint8(image)

            # Create morphological kernel
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )

            # Apply morphological operation
            op_map = {
                MorphologicalOperation.OPENING: cv2.MORPH_OPEN,
                MorphologicalOperation.CLOSING: cv2.MORPH_CLOSE,
                MorphologicalOperation.EROSION: cv2.MORPH_ERODE,
                MorphologicalOperation.DILATION: cv2.MORPH_DILATE,
                MorphologicalOperation.GRADIENT: cv2.MORPH_GRADIENT,
                MorphologicalOperation.TOP_HAT: cv2.MORPH_TOPHAT,
                MorphologicalOperation.BLACK_HAT: cv2.MORPH_BLACKHAT,
            }

            if operation not in op_map:
                raise ValueError(f"Unsupported morphological operation: {operation}")

            processed = cv2.morphologyEx(img_uint8, op_map[operation], kernel)

            # Convert back to original data type and range
            return self._from_uint8(processed, image)

        except Exception as e:
            self.logger.error(f"Morphological operation failed: {e}")
            raise

    def detect_edges(
        self,
        image: NDArray[np.floating],
        method: str,
        threshold1: float = 100.0,
        threshold2: float = 200.0,
    ) -> NDArray[np.floating]:
        """
        Detect edges in the image.

        Args:
            image: Input image array
            method: Edge detection method
            threshold1: First threshold for Canny or low threshold for Sobel
            threshold2: Second threshold for Canny or high threshold for Sobel

        Returns:
            Edge-detected image array

        Raises:
            ValueError: If method is not supported
        """
        try:
            img_uint8 = self._to_uint8(image)

            if method == EdgeDetectionMethod.CANNY:
                edges = cv2.Canny(img_uint8, int(threshold1), int(threshold2))

            elif method == EdgeDetectionMethod.SOBEL:
                grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(grad_x**2 + grad_y**2)
                edges = np.clip(edges, threshold1, threshold2)
                edges = ((edges - threshold1) / (threshold2 - threshold1) * 255).astype(
                    np.uint8
                )

            elif method == EdgeDetectionMethod.LAPLACIAN:
                edges = cv2.Laplacian(img_uint8, cv2.CV_64F)
                edges = np.uint8(np.absolute(edges))

            elif method == EdgeDetectionMethod.SCHARR:
                grad_x = cv2.Scharr(img_uint8, cv2.CV_64F, 1, 0)
                grad_y = cv2.Scharr(img_uint8, cv2.CV_64F, 0, 1)
                edges = np.sqrt(grad_x**2 + grad_y**2)
                edges = np.uint8(np.clip(edges, 0, 255))

            else:
                raise ValueError(f"Unsupported edge detection method: {method}")

            return self._from_uint8(edges, image)

        except Exception as e:
            self.logger.error(f"Edge detection failed: {e}")
            raise

    def apply_filters(
        self, image: NDArray[np.floating], filter_type: str, kernel_size: int = 5
    ) -> NDArray[np.floating]:
        """
        Apply various filters to the image.

        Args:
            image: Input image array
            filter_type: Type of filter to apply
            kernel_size: Size of the filter kernel

        Returns:
            Filtered image array

        Raises:
            ValueError: If filter type is not supported
        """
        try:
            img_uint8 = self._to_uint8(image)

            if filter_type == FilterType.GAUSSIAN:
                filtered = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)

            elif filter_type == FilterType.MEDIAN:
                filtered = cv2.medianBlur(img_uint8, kernel_size)

            elif filter_type == FilterType.BILATERAL:
                filtered = cv2.bilateralFilter(img_uint8, kernel_size, 75, 75)

            elif filter_type == FilterType.BOX:
                filtered = cv2.boxFilter(img_uint8, -1, (kernel_size, kernel_size))

            elif filter_type == FilterType.MOTION_BLUR:
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                filtered = cv2.filter2D(img_uint8, -1, kernel)

            else:
                raise ValueError(f"Unsupported filter type: {filter_type}")

            return self._from_uint8(filtered, image)

        except Exception as e:
            self.logger.error(f"Filter application failed: {e}")
            raise

    def perform_geometric_transforms(
        self, image: NDArray[np.floating], transform: dict[str, Any]
    ) -> NDArray[np.floating]:
        """
        Perform geometric transformations on the image.

        Args:
            image: Input image array
            transform: Dictionary containing transformation parameters
                      Supported keys: 'rotation', 'translation', 'scale', 'shear'

        Returns:
            Transformed image array
        """
        try:
            img_uint8 = self._to_uint8(image)
            h, w = img_uint8.shape
            center = (w // 2, h // 2)

            # Start with identity matrix
            M = np.eye(2, 3, dtype=np.float32)

            # Apply rotation
            if "rotation" in transform:
                angle = transform["rotation"]
                cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                rotation_matrix = np.array(
                    [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                )
                M[:2, :2] = rotation_matrix @ M[:2, :2]

            # Apply scaling
            if "scale" in transform:
                scale = transform["scale"]
                if isinstance(scale, int | float):
                    scale_x = scale_y = scale
                else:
                    scale_x, scale_y = scale
                scale_matrix = np.array([[scale_x, 0], [0, scale_y]], dtype=np.float32)
                M[:2, :2] = scale_matrix @ M[:2, :2]

            # Apply translation
            if "translation" in transform:
                tx, ty = transform["translation"]
                M[:2, 2] += [tx, ty]

            # Apply shear
            if "shear" in transform:
                shear = transform["shear"]
                if isinstance(shear, int | float):
                    shear_x = shear_y = shear
                else:
                    shear_x, shear_y = shear
                shear_matrix = np.array([[1, shear_x], [shear_y, 1]], dtype=np.float32)
                M[:2, :2] = shear_matrix @ M[:2, :2]

            # Center the transformation
            offset = np.array(center) - M[:2, :2] @ np.array(center)
            M[:2, 2] += offset

            # Apply transformation
            transformed = cv2.warpAffine(img_uint8, M, (w, h))

            return self._from_uint8(transformed, image)

        except Exception as e:
            self.logger.error(f"Geometric transformation failed: {e}")
            raise

    def enhance_contrast(
        self, image: NDArray[np.floating], method: str, alpha: float = 1.0
    ) -> NDArray[np.floating]:
        """
        Enhance image contrast.

        Args:
            image: Input image array
            method: Contrast enhancement method
            alpha: Enhancement parameter (varies by method)

        Returns:
            Contrast-enhanced image array

        Raises:
            ValueError: If method is not supported
        """
        try:
            img_uint8 = self._to_uint8(image)

            if method == ContrastMethod.CLAHE:
                clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(8, 8))
                enhanced = clahe.apply(img_uint8)

            elif method == ContrastMethod.HISTOGRAM_EQUALIZATION:
                enhanced = cv2.equalizeHist(img_uint8)

            elif method == ContrastMethod.GAMMA_CORRECTION:
                # Gamma correction: I_out = I_in^(1/gamma)
                gamma = alpha
                inv_gamma = 1.0 / gamma
                table = np.array(
                    [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
                ).astype(np.uint8)
                enhanced = cv2.LUT(img_uint8, table)

            elif method == ContrastMethod.ADAPTIVE_HISTOGRAM:
                # Adaptive histogram equalization
                enhanced = cv2.createCLAHE(
                    clipLimit=alpha, tileGridSize=(16, 16)
                ).apply(img_uint8)

            else:
                raise ValueError(f"Unsupported contrast enhancement method: {method}")

            return self._from_uint8(enhanced, image)

        except Exception as e:
            self.logger.error(f"Contrast enhancement failed: {e}")
            raise

    def remove_noise(
        self, image: NDArray[np.floating], method: str, strength: float = 10.0
    ) -> NDArray[np.floating]:
        """
        Remove noise from the image.

        Args:
            image: Input image array
            method: Noise removal method
            strength: Denoising strength parameter

        Returns:
            Denoised image array

        Raises:
            ValueError: If method is not supported
        """
        try:
            img_uint8 = self._to_uint8(image)

            if method == NoiseRemovalMethod.GAUSSIAN_BLUR:
                kernel_size = (
                    int(strength) if int(strength) % 2 == 1 else int(strength) + 1
                )
                denoised = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)

            elif method == NoiseRemovalMethod.NON_LOCAL_MEANS:
                denoised = cv2.fastNlMeansDenoising(img_uint8, None, strength, 7, 21)

            elif method == NoiseRemovalMethod.FAST_NON_LOCAL_MEANS:
                denoised = cv2.fastNlMeansDenoising(img_uint8, None, strength, 3, 7)

            elif method == NoiseRemovalMethod.BILATERAL_FILTER:
                denoised = cv2.bilateralFilter(
                    img_uint8, int(strength), strength * 2, strength * 2
                )

            else:
                raise ValueError(f"Unsupported noise removal method: {method}")

            return self._from_uint8(denoised, image)

        except Exception as e:
            self.logger.error(f"Noise removal failed: {e}")
            raise

    def _to_uint8(self, image: NDArray[np.floating]) -> NDArray[np.uint8]:
        """
        Convert floating point image to uint8.

        Args:
            image: Input floating point image

        Returns:
            uint8 image
        """
        # Normalize to 0-1 range
        img_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
        # Convert to 0-255 range
        return (img_norm * 255).astype(np.uint8)

    def _from_uint8(
        self, img_uint8: Any, original: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Convert uint8 image back to original data type and range.

        Args:
            img_uint8: uint8 image to convert
            original: Original image for reference range

        Returns:
            Image in original data type and range
        """
        # Ensure numpy array of expected dtype
        img_uint8_np = np.asarray(img_uint8, dtype=np.uint8)

        # Convert to 0-1 range
        img_norm = img_uint8_np.astype(np.float64) / 255.0

        # Scale to original range
        orig_min, orig_max = np.min(original), np.max(original)
        img_scaled = img_norm * (orig_max - orig_min) + orig_min

        return img_scaled.astype(original.dtype)

    def get_processing_info(self) -> dict[str, Any]:
        """
        Get information about available processing operations.

        Returns:
            Dictionary containing operation information
        """
        return {
            "morphological_operations": [op.value for op in MorphologicalOperation],
            "edge_detection_methods": [method.value for method in EdgeDetectionMethod],
            "filter_types": [ft.value for ft in FilterType],
            "contrast_methods": [method.value for method in ContrastMethod],
            "noise_removal_methods": [method.value for method in NoiseRemovalMethod],
            "opencv_version": cv2.__version__,
        }
