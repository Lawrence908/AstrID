"""
Supernova image differencing pipeline.

Core pipeline classes for astronomical image differencing:
- WCS alignment via reprojection
- Background subtraction
- PSF estimation and matching
- Flux normalization
- Optimal differencing
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from reproject import reproject_interp

logger = logging.getLogger(__name__)


# Mission filter definitions
MISSION_FILTERS = {
    "SWIFT": {
        "filters": ["uvw2", "uvm2", "uvw1", "uuu", "ubb", "uvv"],
        "preferred": ["uuu", "uvw1", "uvm2", "uvw2", "ubb", "uvv"],
        "pattern": "SWIFT_*.fits",
    },
    "GALEX": {
        "filters": ["nuv", "fuv", "nd", "fd", "ng", "fg"],
        "preferred": ["nuv", "fuv", "nd", "fd", "ng", "fg"],
        "pattern": "GALEX_*.fits*",
    },
    "PS1": {
        "filters": ["g", "r", "i", "z", "y"],
        "preferred": ["g", "r", "i", "z", "y"],
        "pattern": "PS1_*.fits*",
    },
}


def detect_mission_from_filename(filename: str) -> str | None:
    """Detect mission from filename."""
    name = Path(filename).name.upper()
    if name.startswith("SWIFT_"):
        return "SWIFT"
    elif name.startswith("GALEX_"):
        return "GALEX"
    elif name.startswith("PS1_"):
        return "PS1"
    return None


@dataclass
class DifferencingResult:
    """Container for differencing pipeline results."""

    sn_name: str
    mission_name: str
    filter_name: str
    ref_date: str
    sci_date: str

    # Metrics
    overlap_fraction: float
    ref_fwhm: float
    sci_fwhm: float
    flux_scale: float
    sig_max: float
    sig_min: float
    n_detections: int

    # File paths (relative)
    difference_file: str
    significance_file: str
    mask_file: str | None

    # SN position
    sn_pixel: tuple[float, float] | None = None


class SNDifferencingPipeline:
    """
    Supernova image differencing pipeline.

    Implements proper astronomical image differencing with:
    - WCS alignment via reprojection
    - Background subtraction
    - PSF estimation and matching
    - Flux normalization
    - Optimal differencing
    """

    def __init__(
        self,
        psf_fwhm: float = 4.0,
        background_box_size: int = 64,
        detection_threshold: float = 5.0,
        mask_radius: int = 10,
    ):
        """
        Initialize pipeline.

        Args:
            psf_fwhm: Expected PSF FWHM in pixels
            background_box_size: Box size for background estimation
            detection_threshold: SNR threshold for source detection
            mask_radius: Radius of mask around known SN position
        """
        self.psf_fwhm = psf_fwhm
        self.background_box_size = background_box_size
        self.detection_threshold = detection_threshold
        self.mask_radius = mask_radius

    def load_fits(self, filepath: Path) -> tuple[np.ndarray, dict, WCS]:
        """Load FITS image, header, and WCS."""
        with fits.open(filepath, memmap=True) as hdul:
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    # Use float32 instead of float64 to save memory
                    data = hdu.data.astype(np.float32)
                    header = dict(hdu.header)
                    if len(data.shape) == 3:
                        data = data[0]
                    try:
                        wcs = WCS(hdu.header, naxis=2)
                    except Exception:
                        wcs = None
                    return data, header, wcs
        raise ValueError(f"No image data found in {filepath}")

    def estimate_background(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate and subtract background."""
        mask = ~np.isfinite(image)
        image_clean = np.where(mask, np.nanmedian(image), image)

        try:
            bkg = Background2D(
                image_clean,
                box_size=self.background_box_size,
                filter_size=3,
                bkg_estimator=MedianBackground(),
                exclude_percentile=10,
            )
            bkg_subtracted = image - bkg.background
            bkg_rms = bkg.background_rms.copy()  # Copy to avoid keeping reference
            del bkg  # Clean up background object
        except Exception:
            mean, median, std = sigma_clipped_stats(image_clean, sigma=3.0)
            bkg_subtracted = image - median
            bkg_rms = np.full_like(image, std, dtype=np.float32)

        return bkg_subtracted, bkg_rms

    def estimate_psf_fwhm(self, image: np.ndarray) -> float:
        """Estimate PSF FWHM from bright stars."""
        try:
            mean, median, std = sigma_clipped_stats(image, sigma=3.0)
            daofind = DAOStarFinder(fwhm=self.psf_fwhm, threshold=10 * std)
            sources = daofind(image - median)

            if sources is not None and len(sources) > 3:
                fwhm_estimate = np.median(sources["sharpness"]) * self.psf_fwhm * 2
                return max(1.5, min(fwhm_estimate, 10.0))
        except Exception:
            pass

        return self.psf_fwhm

    def match_psf(
        self, image: np.ndarray, current_fwhm: float, target_fwhm: float
    ) -> np.ndarray:
        """Convolve image to match target PSF."""
        if target_fwhm <= current_fwhm:
            return image

        sigma_current = current_fwhm / 2.355
        sigma_target = target_fwhm / 2.355
        sigma_kernel = np.sqrt(sigma_target**2 - sigma_current**2)

        kernel = Gaussian2DKernel(sigma_kernel)
        # Use preserve_nan instead of nan_treatment to reduce memory usage
        result = convolve_fft(image, kernel, allow_huge=True, preserve_nan=True)
        return result

    def normalize_flux(
        self, science: np.ndarray, reference: np.ndarray, footprint: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Normalize science image flux to match reference."""
        mask = (footprint > 0.5) & np.isfinite(science) & np.isfinite(reference)

        if np.sum(mask) < 100:
            return science, 1.0, 0.0

        clip = SigmaClip(sigma=3.0, maxiters=5)
        sci_med = np.median(clip(science[mask]).compressed())
        ref_med = np.median(clip(reference[mask]).compressed())

        scale = ref_med / sci_med if sci_med > 0 else 1.0
        normalized = science * scale
        offset = ref_med - np.median(normalized[mask])
        normalized = normalized + offset

        return normalized, scale, offset

    def create_sn_mask(
        self, shape: tuple[int, int], sn_pixel: tuple[float, float] | None
    ) -> np.ndarray:
        """Create binary mask at SN position."""
        mask = np.zeros(shape, dtype=np.float32)

        if sn_pixel is not None:
            x, y = int(sn_pixel[0]), int(sn_pixel[1])

            # Create circular mask
            yy, xx = np.ogrid[: shape[0], : shape[1]]
            distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            mask[distance <= self.mask_radius] = 1.0

        return mask

    def process(
        self,
        ref_path: Path,
        sci_path: Path,
        sn_name: str = "unknown",
        sn_coords: SkyCoord | None = None,
        mission_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, DifferencingResult]:
        """
        Run full differencing pipeline.

        Returns:
            (difference_image, significance_map, sn_mask, result_metadata)
        """
        # Extract filter and mission from path
        filter_name = "unknown"
        mission_name = detect_mission_from_filename(str(sci_path))
        
        if mission_name and mission_name in MISSION_FILTERS:
            config = MISSION_FILTERS[mission_name]
            for filt in config["filters"]:
                if filt in str(sci_path).lower():
                    filter_name = filt
                    break

        logger.info(f"Processing {sn_name} ({mission_name or 'UNKNOWN'} {filter_name} filter)")

        # Load images
        ref_data, ref_header, ref_wcs = self.load_fits(ref_path)
        sci_data, sci_header, sci_wcs = self.load_fits(sci_path)

        ref_date = ref_header.get("DATE-OBS", "N/A")[:10]
        sci_date = sci_header.get("DATE-OBS", "N/A")[:10]

        # WCS alignment
        sci_aligned, footprint = reproject_interp(
            (sci_data, sci_wcs), ref_wcs, shape_out=ref_data.shape
        )
        sci_aligned = np.where(np.isfinite(sci_aligned), sci_aligned, 0)
        overlap_frac = np.sum(footprint > 0.5) / footprint.size * 100

        # Background subtraction
        ref_bkg_sub, ref_noise = self.estimate_background(ref_data)
        sci_bkg_sub, sci_noise = self.estimate_background(sci_aligned)

        # PSF estimation and matching
        ref_fwhm = self.estimate_psf_fwhm(ref_bkg_sub)
        sci_fwhm = self.estimate_psf_fwhm(sci_bkg_sub)
        target_fwhm = max(ref_fwhm, sci_fwhm)

        ref_matched = self.match_psf(ref_bkg_sub, ref_fwhm, target_fwhm)
        sci_matched = self.match_psf(sci_bkg_sub, sci_fwhm, target_fwhm)

        # Flux normalization
        sci_normalized, scale, offset = self.normalize_flux(
            sci_matched, ref_matched, footprint
        )

        # Compute difference
        diff = sci_normalized - ref_matched
        noise_combined = np.sqrt(sci_noise**2 + ref_noise**2)
        noise_combined = np.where(noise_combined > 0, noise_combined, 1e-10)
        significance = diff / noise_combined

        # Detect sources
        valid_mask = footprint > 0.5
        try:
            mean, median, std = sigma_clipped_stats(significance[valid_mask], sigma=3.0)
            daofind = DAOStarFinder(
                fwhm=self.psf_fwhm,
                threshold=self.detection_threshold * std,
                sharplo=0.2,
                sharphi=1.0,
            )
            sources = daofind(np.where(valid_mask, significance - median, 0))
            n_detections = len(sources) if sources is not None else 0
        except Exception:
            n_detections = 0

        # Get SN pixel position
        sn_pixel = None
        if sn_coords is not None and ref_wcs is not None:
            try:
                x, y = ref_wcs.world_to_pixel(sn_coords)
                sn_pixel = (float(x), float(y))
            except Exception:
                pass

        # Create mask
        sn_mask = self.create_sn_mask(diff.shape, sn_pixel)

        # Metrics
        result = DifferencingResult(
            sn_name=sn_name,
            mission_name=mission_name or detect_mission_from_filename(str(sci_path)) or "UNKNOWN",
            filter_name=filter_name,
            ref_date=ref_date,
            sci_date=sci_date,
            overlap_fraction=float(overlap_frac),
            ref_fwhm=float(ref_fwhm),
            sci_fwhm=float(sci_fwhm),
            flux_scale=float(scale),
            sig_max=float(np.nanmax(significance[valid_mask])),
            sig_min=float(np.nanmin(significance[valid_mask])),
            n_detections=n_detections,
            difference_file="",  # Filled in by caller
            significance_file="",
            mask_file="",
            sn_pixel=sn_pixel,
        )

        logger.info(
            f"  Complete: overlap={overlap_frac:.1f}%, max_sig={result.sig_max:.1f}σ"
        )

        # Clean up large intermediate arrays
        del ref_data, sci_data, sci_aligned, footprint
        del ref_bkg_sub, sci_bkg_sub, ref_noise, sci_noise
        del ref_matched, sci_matched, sci_normalized
        gc.collect()

        return diff, significance, sn_mask, result
