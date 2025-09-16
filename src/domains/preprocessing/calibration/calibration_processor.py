"""Calibration processor for astronomical image preprocessing.

Implements bias, dark, and flat-field calibration with robust master frame
creation, validation, and simple uncertainty propagation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clip


@dataclass
class CalibrationUncertainty:
    """Container for propagated calibration uncertainties."""

    bias_std: float | None = None
    dark_std: float | None = None
    flat_std: float | None = None


class CalibrationValidationError(ValueError):
    """Raised when calibration frames are invalid."""


class CalibrationProcessor:
    """Applies calibration steps and builds master calibration frames."""

    def _validate_same_shape(
        self, a: np.ndarray, b: np.ndarray, name_a: str, name_b: str
    ) -> None:
        if a.shape != b.shape:
            raise CalibrationValidationError(
                f"Shape mismatch: {name_a}{a.shape} != {name_b}{b.shape}"
            )

    def _validate_nonzero(self, frame: np.ndarray, name: str) -> None:
        if not np.isfinite(frame).all():
            raise CalibrationValidationError(f"{name} contains non-finite values")
        if np.all(frame == 0):
            raise CalibrationValidationError(f"{name} is all zeros")

    def _robust_mean(self, frames: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Compute a sigma-clipped mean and return mean and per-pixel std estimate.

        Returns (mean_frame, robust_std_estimate), where robust_std_estimate is the
        global median of per-pixel standard deviations after clipping.
        """
        stack = np.stack(frames, axis=0).astype(float)
        clipped = sigma_clip(stack, sigma=3.0, axis=0, maxiters=5, cenfunc="median")
        mean_frame = np.nanmean(clipped.filled(np.nan), axis=0)
        std_frame = np.nanstd(clipped.filled(np.nan), axis=0)
        robust_std = float(np.nanmedian(std_frame))
        return mean_frame, robust_std

    # --- Application methods ---
    def apply_bias_correction(
        self, image: np.ndarray, bias_frame: np.ndarray
    ) -> np.ndarray:
        self._validate_same_shape(image, bias_frame, "image", "bias_frame")
        self._validate_nonzero(bias_frame, "bias_frame")
        return image.astype(float) - bias_frame.astype(float)

    def apply_dark_correction(
        self, image: np.ndarray, dark_frame: np.ndarray, exposure_time: float
    ) -> np.ndarray:
        if exposure_time <= 0:
            raise CalibrationValidationError("exposure_time must be > 0")
        self._validate_same_shape(image, dark_frame, "image", "dark_frame")
        self._validate_nonzero(dark_frame, "dark_frame")
        # Scale dark current to the image exposure
        return image.astype(float) - dark_frame.astype(float) * exposure_time

    def apply_flat_correction(
        self, image: np.ndarray, flat_frame: np.ndarray
    ) -> np.ndarray:
        self._validate_same_shape(image, flat_frame, "image", "flat_frame")
        self._validate_nonzero(flat_frame, "flat_frame")
        # Normalize flat to unit mean to avoid flux scaling
        norm_flat = flat_frame.astype(float)
        norm = np.mean(norm_flat)
        if not np.isfinite(norm) or norm == 0:
            raise CalibrationValidationError("flat_frame mean is invalid or zero")
        norm_flat = norm_flat / norm
        return image.astype(float) / (norm_flat + 1e-12)

    # --- Master frame creation ---
    def create_master_bias(self, bias_frames: list[np.ndarray]) -> np.ndarray:
        if not bias_frames:
            raise CalibrationValidationError("No bias frames provided")
        ref_shape = bias_frames[0].shape
        if any(f.shape != ref_shape for f in bias_frames):
            raise CalibrationValidationError("Bias frames have inconsistent shapes")
        master_bias, _ = self._robust_mean(bias_frames)
        self._validate_nonzero(master_bias, "master_bias")
        return master_bias

    def create_master_dark(
        self, dark_frames: list[np.ndarray], exposure_times: list[float]
    ) -> np.ndarray:
        if not dark_frames:
            raise CalibrationValidationError("No dark frames provided")
        if len(dark_frames) != len(exposure_times):
            raise CalibrationValidationError(
                "dark_frames and exposure_times length mismatch"
            )
        if any(t <= 0 for t in exposure_times):
            raise CalibrationValidationError("All exposure_times must be > 0")
        ref_shape = dark_frames[0].shape
        if any(f.shape != ref_shape for f in dark_frames):
            raise CalibrationValidationError("Dark frames have inconsistent shapes")

        # Normalize dark frames to per-second dark current, then robust-mean
        per_sec = [
            f.astype(float) / t
            for f, t in zip(dark_frames, exposure_times, strict=False)
        ]
        master_dark, _ = self._robust_mean(per_sec)
        self._validate_nonzero(master_dark, "master_dark")
        return master_dark

    def create_master_flat(
        self, flat_frames: list[np.ndarray], bias_frame: np.ndarray
    ) -> np.ndarray:
        if not flat_frames:
            raise CalibrationValidationError("No flat frames provided")
        ref_shape = flat_frames[0].shape
        if any(f.shape != ref_shape for f in flat_frames):
            raise CalibrationValidationError("Flat frames have inconsistent shapes")
        # Bias-correct each flat, normalize to unit mean, then robust-mean
        corrected = []
        for f in flat_frames:
            self._validate_same_shape(f, bias_frame, "flat", "bias_frame")
            bc = f.astype(float) - bias_frame.astype(float)
            m = np.mean(bc)
            if not np.isfinite(m) or m == 0:
                raise CalibrationValidationError(
                    "Flat frame mean after bias-correction is invalid or zero"
                )
            corrected.append(bc / m)
        master_flat, _ = self._robust_mean(corrected)
        self._validate_nonzero(master_flat, "master_flat")
        return master_flat

    # --- Quality checks and uncertainty ---
    def validate_calibration_quality(self, frame: np.ndarray) -> dict[str, float]:
        """Return basic quality metrics for a calibration frame."""
        metrics: dict[str, float] = {
            "mean": float(np.mean(frame)),
            "std": float(np.std(frame)),
            "min": float(np.min(frame)),
            "max": float(np.max(frame)),
            "dynamic_range": float(np.max(frame) - np.min(frame)),
        }
        return metrics

    def propagate_uncertainty(
        self,
        image: np.ndarray,
        master_bias: np.ndarray | None = None,
        master_dark: np.ndarray | None = None,
        master_flat: np.ndarray | None = None,
        exposure_time: float | None = None,
    ) -> CalibrationUncertainty:
        """Simple uncertainty propagation for calibrated image components.

        Uses global robust std estimates as proxies for per-pixel uncertainties.
        """
        bias_std = float(np.std(master_bias)) if master_bias is not None else None
        dark_std = None
        if master_dark is not None:
            scale = exposure_time if exposure_time is not None else 1.0
            dark_std = float(np.std(master_dark) * scale)
        flat_std = float(np.std(master_flat)) if master_flat is not None else None
        return CalibrationUncertainty(
            bias_std=bias_std, dark_std=dark_std, flat_std=flat_std
        )
