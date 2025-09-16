"""WCS alignment and image registration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.wcs import WCS
from scipy.signal import correlate2d
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray
    aligned_wcs: WCS
    metrics: dict[str, float]


class WCSAligner:
    """Align images using WCS information and image registration techniques."""

    def align_to_reference_image(
        self, image: np.ndarray, reference: np.ndarray, wcs: WCS
    ) -> tuple[np.ndarray, WCS]:
        """Align image to reference using subpixel phase correlation."""
        shift, error, _ = phase_cross_correlation(reference, image, upsample_factor=10)
        transform = AffineTransform(translation=(-shift[1], -shift[0]))
        aligned = warp(image, transform.inverse, preserve_range=True)
        # WCS translation adjustment (approximate)
        adjusted_wcs = wcs.deepcopy()
        adjusted_wcs.wcs.crpix += shift[::-1]
        return aligned.astype(image.dtype), adjusted_wcs

    def register_multiple_images(
        self, images: list[np.ndarray], wcs_list: list[WCS]
    ) -> tuple[list[np.ndarray], list[WCS]]:
        """Register a list of images to the first image as reference."""
        if not images:
            return [], []
        ref = images[0]
        ref_wcs = wcs_list[0]
        aligned_images = [ref]
        aligned_wcs_list = [ref_wcs]
        for img, w in zip(images[1:], wcs_list[1:], strict=False):
            aligned, adj_wcs = self.align_to_reference_image(img, ref, w)
            aligned_images.append(aligned)
            aligned_wcs_list.append(adj_wcs)
        return aligned_images, aligned_wcs_list

    def calculate_alignment_transform(
        self, source_coords: np.ndarray, target_coords: np.ndarray
    ) -> AffineTransform:
        """Compute affine transform from matched coordinate pairs."""
        if source_coords.shape != target_coords.shape or source_coords.shape[1] != 2:
            raise ValueError("coords must be Nx2 and shapes must match")
        # Solve least squares for affine parameters
        src = np.hstack([source_coords, np.ones((source_coords.shape[0], 1))])
        tx, _, _, _ = np.linalg.lstsq(src, target_coords, rcond=None)
        matrix = np.eye(3)
        matrix[:2, :3] = tx.T
        return AffineTransform(matrix=matrix)

    def apply_alignment_transform(
        self, image: np.ndarray, transform: AffineTransform
    ) -> np.ndarray:
        """Apply an affine transform to an image."""
        return warp(image, transform.inverse, preserve_range=True).astype(image.dtype)

    def validate_alignment_quality(
        self, aligned_image: np.ndarray, reference: np.ndarray
    ) -> dict:
        """Return simple alignment quality metrics based on cross-correlation."""
        # Normalize
        a = (aligned_image - np.mean(aligned_image)) / (np.std(aligned_image) + 1e-9)
        r = (reference - np.mean(reference)) / (np.std(reference) + 1e-9)
        corr = correlate2d(r, a, mode="valid")
        peak = float(np.max(corr))
        return {"xcorr_peak": peak}
