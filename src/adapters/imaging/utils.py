"""
Shared imaging utilities for FITS and display normalization.

This module provides helpers to convert arbitrary FITS arrays into
displayable numpy arrays suitable for matplotlib, handling a variety of
data dimensionalities robustly.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__: Final = [
    "to_display_image",
]


def to_display_image(array: np.ndarray) -> np.ndarray:
    """Convert a FITS array to a displayable image (HxW or HxWx3) with robust normalization.

    Accepts shapes like (H,W), (C,H,W), (H,W,C), or higher; handles alpha.
    Always returns float image in [0,1] for plt.imshow.
    """
    data = np.asarray(array)

    # Handle scalar and 1D safely
    if data.ndim == 0:
        return np.zeros((1, 1), dtype=float)
    if data.ndim == 1:
        vec = data.astype(float)
        finite_mask = np.isfinite(vec)
        if np.any(finite_mask):
            lo, hi = np.percentile(vec[finite_mask], (1, 99.5))
            if hi <= lo:
                hi = lo + 1e-9
            vec = np.clip((vec - lo) / (hi - lo), 0, 1)
        else:
            vec = np.zeros_like(vec, dtype=float)
        return vec.reshape(-1, 1)

    # Collapse singleton dims once
    data = np.squeeze(data)

    # Reduce higher-dimensional arrays by averaging leading axes until <=3 dims
    while data.ndim > 3:
        data = np.nanmean(data, axis=0)
        data = np.squeeze(data)

    if data.ndim == 2:
        finite_mask = np.isfinite(data)
        if np.any(finite_mask):
            lo, hi = np.percentile(data[finite_mask], (1, 99.5))
            if hi <= lo:
                hi = lo + 1e-9
            data = np.clip((data - lo) / (hi - lo), 0, 1)
        else:
            data = np.zeros_like(data, dtype=float)
        return data

    # Handle 3D: try to identify channel dimension; otherwise collapse to grayscale
    if data.ndim == 3:
        shape = data.shape
        # If planes-first (C,H,W) with square planes, move channels to last
        if shape[0] in (1, 2, 3, 4) and shape[1] == shape[2]:
            data = np.moveaxis(data, 0, -1)
        # If channels at end but >3, drop extras
        if data.shape[-1] >= 4:
            data = data[..., :3]
        elif data.shape[-1] == 2:
            y = data[..., 0]
            data = np.stack([y, y, y], axis=-1)
        elif data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
        # If none of the dims look like channels, collapse first dim
        if data.shape[-1] > 4 and (3 not in data.shape and 4 not in data.shape):
            data = np.nanmean(data, axis=-1)
            return to_display_image(data)

        rgb = data.astype(float)
        finite_mask = np.isfinite(rgb)
        if np.any(finite_mask):
            lo, hi = np.percentile(rgb[finite_mask], (1, 99.5))
            if hi <= lo:
                hi = lo + 1e-9
            rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
        else:
            rgb = np.zeros_like(rgb, dtype=float)
        return rgb

    # As a final fallback, convert anything else to 2D grayscale via mean
    data2d = np.nanmean(data, axis=tuple(range(max(0, data.ndim - 2))))
    if data2d.ndim != 2:
        data2d = np.atleast_2d(data2d)
    finite_mask = np.isfinite(data2d)
    if np.any(finite_mask):
        lo, hi = np.percentile(data2d[finite_mask], (1, 99.5))
        if hi <= lo:
            hi = lo + 1e-9
        data2d = np.clip((data2d - lo) / (hi - lo), 0, 1)
    else:
        data2d = np.zeros_like(data2d, dtype=float)
    return data2d
