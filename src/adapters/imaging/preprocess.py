"""
Lightweight imaging preprocessing utilities inspired by zOLD/imageProcessing.py.

Focuses on safe, dependency-light operations suitable for notebooks and services.
"""

from __future__ import annotations

from typing import Final

import cv2  # type: ignore
import numpy as np

__all__: Final = [
    "convert_to_grayscale",
    "apply_gaussian_blur",
    "apply_threshold",
    "apply_morphology",
    "normalize_image",
    "preprocess_image",
]


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert possibly RGB image to 2D grayscale robustly.

    Handles float arrays without requiring OpenCV conversion by averaging channels.
    For uint8 inputs, uses cv2 for performance when available.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]

    # If float types, avoid cv2 depth issues and convert via weighted mean
    if np.issubdtype(image.dtype, np.floating):
        # Assume RGB ordering for general images; if not sure, average is fine
        if image.shape[-1] >= 3:
            # Rec. 709 luma weights
            r, g, b = image[..., 0], image[..., 1], image[..., 2]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.mean(image, axis=-1)

    # For integer images, try cv2; fallback to mean if conversion fails
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        return np.mean(image, axis=-1)


def apply_gaussian_blur(
    image: np.ndarray, kernel_size: tuple[int, int] = (5, 5)
) -> np.ndarray:
    return cv2.GaussianBlur(image, kernel_size, 0)


def apply_threshold(image: np.ndarray, threshold_value: int = 30) -> np.ndarray:
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


def apply_morphology(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (3, 3),
    dilate_iter: int = 3,
    erode_iter: int = 3,
) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=dilate_iter)
    eroded = cv2.erode(dilated, kernel, iterations=erode_iter)
    return eroded


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(float)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))
    if max_val <= min_val:
        return np.zeros_like(image, dtype=float)
    return (image - min_val) / (max_val - min_val)


def preprocess_image(
    image: np.ndarray,
    *,
    kernel_size: tuple[int, int] = (1, 1),
    threshold_value: int = 100,
) -> np.ndarray:
    # Ensure 2D grayscale base
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 3 and image.shape[-1] == 3:
        gray = convert_to_grayscale(image)
    elif image.ndim == 2:
        gray = image
    else:
        # Fallback: mean across channels
        gray = np.mean(image, axis=-1)

    blurred = apply_gaussian_blur(gray, kernel_size)
    norm = normalize_image(blurred)
    # Scale to 0-255 for thresholding
    binary = apply_threshold((norm * 255).astype(np.uint8), threshold_value)
    morphed = apply_morphology(binary)
    return normalize_image(morphed)
