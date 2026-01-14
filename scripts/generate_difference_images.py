#!/usr/bin/env python3
"""
Generate difference images for supernova training.

Processes all same-mission image pairs through the differencing pipeline
and outputs training-ready difference images with masks.

Usage:
    python scripts/generate_difference_images.py --input-dir output/fits_training --output-dir output/difference_images
    python scripts/generate_difference_images.py --input-dir output/fits_training --output-dir output/difference_images --visualize
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
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

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Known SN coordinates (add more as needed)
SN_COORDINATES = {
    "2014J": SkyCoord("9h 55m 42.217s", "+69d 40m 26.56s", frame="icrs"),
    # Add coordinates for other SNe from catalogs
}


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
        "pattern": "GALEX_*.fits*",  # May be .fits or .fits.gz
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


def find_matching_filter_pair(
    sn_name: str, base_dir: Path
) -> tuple[Path, Path, str, str] | None:
    """Find a reference/science pair with matching mission and filter.
    
    Returns:
        (ref_path, sci_path, filter_name, mission) or None
    """
    ref_dir = base_dir / sn_name / "reference"
    sci_dir = base_dir / sn_name / "science"

    if not ref_dir.exists() or not sci_dir.exists():
        return None

    # Group files by mission
    ref_by_mission: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sci_by_mission: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # Process reference files
    for f in ref_dir.glob("*.fits*"):
        mission = detect_mission_from_filename(f.name)
        if mission is None:
            continue
        config = MISSION_FILTERS.get(mission)
        if config is None:
            continue

        for filt in config["filters"]:
            if filt in f.name.lower():
                ref_by_mission[mission][filt].append(f)
                break

    # Process science files
    for f in sci_dir.glob("*.fits*"):
        mission = detect_mission_from_filename(f.name)
        if mission is None:
            continue
        config = MISSION_FILTERS.get(mission)
        if config is None:
            continue

        for filt in config["filters"]:
            if filt in f.name.lower():
                sci_by_mission[mission][filt].append(f)
                break

    # Find same-mission pairs with matching filters
    for mission in ref_by_mission.keys() & sci_by_mission.keys():
        ref_filters = set(ref_by_mission[mission].keys())
        sci_filters = set(sci_by_mission[mission].keys())
        common_filters = ref_filters & sci_filters

        if not common_filters:
            continue

        config = MISSION_FILTERS[mission]
        # Use preferred filter order
        for preferred in config["preferred"]:
            if preferred in common_filters:
                ref_path = ref_by_mission[mission][preferred][0]
                sci_path = sci_by_mission[mission][preferred][0]
                return ref_path, sci_path, preferred, mission

        # Fallback to first common filter
        filt = list(common_filters)[0]
        ref_path = ref_by_mission[mission][filt][0]
        sci_path = sci_by_mission[mission][filt][0]
        return ref_path, sci_path, filt, mission

    return None


def get_same_mission_sne(manifest_path: Path) -> list[str]:
    """Get list of SNe with same-mission pairs (SWIFT-SWIFT, GALEX-GALEX, PS1-PS1, etc.)."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    same_mission = []
    for entry in manifest:
        ref_missions = set()
        sci_missions = set()

        for f in entry.get("reference_files", []):
            mission = detect_mission_from_filename(f)
            if mission:
                ref_missions.add(mission)

        for f in entry.get("science_files", []):
            mission = detect_mission_from_filename(f)
            if mission:
                sci_missions.add(mission)

        # Check for any overlapping missions
        common_missions = ref_missions & sci_missions
        if common_missions:
            same_mission.append(entry["sn_name"])

    return same_mission


def main():
    parser = argparse.ArgumentParser(
        description="Generate difference images for training"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/fits_training"),
        help="Directory with organized FITS files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/difference_images"),
        help="Output directory for difference images",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to training manifest (default: input-dir/training_manifest.json)",
    )
    parser.add_argument(
        "--sn",
        nargs="+",
        default=None,
        help="Specific SNe to process (default: all same-mission)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization plots"
    )
    parser.add_argument(
        "--psf-fwhm", type=float, default=4.0, help="Expected PSF FWHM in pixels"
    )
    parser.add_argument(
        "--mask-radius", type=int, default=10, help="Radius of SN mask in pixels"
    )
    parser.add_argument(
        "--mission",
        type=str,
        default=None,
        choices=["SWIFT", "GALEX", "PS1"],
        help="Filter to specific mission (default: all missions)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of SNe to process before forcing garbage collection (default: 10)",
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = args.input_dir
    output_dir = args.output_dir
    manifest_path = args.manifest or (input_dir / "training_manifest.json")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get SNe to process
    if args.sn:
        sne_to_process = args.sn
    else:
        sne_to_process = get_same_mission_sne(manifest_path)
        
        # Filter by mission if specified
        if args.mission:
            filtered = []
            for sn_name in sne_to_process:
                pair = find_matching_filter_pair(sn_name, input_dir)
                if pair and pair[3] == args.mission:  # pair[3] is mission_name
                    filtered.append(sn_name)
            sne_to_process = filtered
            logger.info(f"Filtered to {args.mission} mission: {len(sne_to_process)} SNe")

    logger.info(f"Processing {len(sne_to_process)} SNe: {sne_to_process}")

    # Initialize pipeline
    pipeline = SNDifferencingPipeline(
        psf_fwhm=args.psf_fwhm, mask_radius=args.mask_radius
    )

    # Process each SN
    results = []

    for idx, sn_name in enumerate(sne_to_process, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {sn_name}")

        pair = find_matching_filter_pair(sn_name, input_dir)
        if pair is None:
            logger.warning("  No matching filter pair found, skipping")
            continue

        ref_path, sci_path, filter_name, mission_name = pair

        try:
            diff, sig, mask, result = pipeline.process(
                ref_path=ref_path,
                sci_path=sci_path,
                sn_name=sn_name,
                sn_coords=SN_COORDINATES.get(sn_name),
                mission_name=mission_name,
            )

            # Create output directory for this SN
            sn_output_dir = output_dir / sn_name
            sn_output_dir.mkdir(exist_ok=True)

            # Save difference image
            diff_file = sn_output_dir / f"{sn_name}_{mission_name}_{filter_name}_diff.fits"
            hdu_diff = fits.PrimaryHDU(diff.astype(np.float32))
            hdu_diff.header["SN_NAME"] = sn_name
            hdu_diff.header["MISSION"] = mission_name
            hdu_diff.header["FILTER"] = filter_name
            hdu_diff.header["REF_DATE"] = result.ref_date
            hdu_diff.header["SCI_DATE"] = result.sci_date
            hdu_diff.header["OVERLAP"] = result.overlap_fraction
            hdu_diff.header["SIG_MAX"] = result.sig_max
            if result.sn_pixel:
                hdu_diff.header["SN_X"] = result.sn_pixel[0]
                hdu_diff.header["SN_Y"] = result.sn_pixel[1]
            hdu_diff.writeto(diff_file, overwrite=True)

            # Save significance map
            sig_file = sn_output_dir / f"{sn_name}_{mission_name}_{filter_name}_sig.fits"
            hdu_sig = fits.PrimaryHDU(sig.astype(np.float32))
            hdu_sig.header["SN_NAME"] = sn_name
            hdu_sig.header["MISSION"] = mission_name
            hdu_sig.header["FILTER"] = filter_name
            hdu_sig.writeto(sig_file, overwrite=True)

            # Save mask
            mask_file = sn_output_dir / f"{sn_name}_{mission_name}_{filter_name}_mask.fits"
            hdu_mask = fits.PrimaryHDU(mask.astype(np.float32))
            hdu_mask.header["SN_NAME"] = sn_name
            hdu_mask.header["MISSION"] = mission_name
            hdu_mask.header["FILTER"] = filter_name
            hdu_mask.writeto(mask_file, overwrite=True)

            # Update result with file paths
            result.difference_file = str(diff_file.relative_to(output_dir))
            result.significance_file = str(sig_file.relative_to(output_dir))
            result.mask_file = str(mask_file.relative_to(output_dir))

            results.append(asdict(result))

            logger.info(f"  Saved: {diff_file.name}")

            # Visualization
            if args.visualize:
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # Use non-interactive backend
                    import matplotlib.pyplot as plt

                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    fig.suptitle(f"SN {sn_name} ({filter_name})")

                    # Difference
                    vmax = np.nanpercentile(np.abs(diff), 99)
                    axes[0].imshow(
                        diff, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax
                    )
                    axes[0].set_title("Difference")

                    # Significance
                    vmax_sig = min(np.nanpercentile(np.abs(sig), 99), 15)
                    axes[1].imshow(
                        sig,
                        origin="lower",
                        cmap="RdBu_r",
                        vmin=-vmax_sig,
                        vmax=vmax_sig,
                    )
                    axes[1].set_title(f"Significance (max={result.sig_max:.1f}σ)")

                    # Mask
                    axes[2].imshow(mask, origin="lower", cmap="gray")
                    axes[2].set_title("SN Mask")

                    # Mark SN position
                    if result.sn_pixel:
                        for ax in axes[:3]:
                            ax.scatter(
                                [result.sn_pixel[0]],
                                [result.sn_pixel[1]],
                                marker="+",
                                s=100,
                                c="lime",
                                linewidths=2,
                            )

                    # Histogram
                    valid = sig[np.isfinite(sig)]
                    axes[3].hist(
                        valid.flatten(), bins=100, range=(-10, 10), density=True
                    )
                    axes[3].axvline(5, color="r", linestyle="--", label="5σ")
                    axes[3].set_xlabel("Significance (σ)")
                    axes[3].set_title("Distribution")
                    axes[3].legend()

                    plt.tight_layout()
                    plt.savefig(
                        sn_output_dir / f"{sn_name}_{mission_name}_{filter_name}_viz.png", dpi=150
                    )
                    plt.close(fig)
                    plt.clf()
                    del fig, axes

                except Exception as e:
                    logger.warning(f"  Visualization failed: {e}")

            # Clean up memory after each SN
            del diff, sig, mask, result
            gc.collect()

        except Exception as e:
            logger.error(f"  Processing failed: {e}")
            import traceback

            traceback.print_exc()
            # Clean up on error too
            gc.collect()

        # Force garbage collection after each batch
        if idx % args.batch_size == 0:
            logger.info(f"  [Batch {idx}/{len(sne_to_process)}] Running garbage collection...")
            gc.collect()

    # Save results summary
    summary = {
        "pipeline_version": "1.0",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_processed": len(results),
        "results": results,
    }

    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Processed: {len(results)} SNe")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Summary: {summary_file}")

    # Print training data summary
    logger.info("\n📊 Training Data Summary:")
    by_mission = defaultdict(list)
    for r in results:
        # Extract mission from difference_file path
        mission = "UNKNOWN"
        for m in ["SWIFT", "GALEX", "PS1"]:
            if m in r.get("difference_file", ""):
                mission = m
                break
        by_mission[mission].append(r)
        
    for mission, mission_results in sorted(by_mission.items()):
        logger.info(f"\n  {mission} ({len(mission_results)} pairs):")
        for r in mission_results:
            logger.info(
                f"    {r['sn_name']}: {r['filter_name']}, sig_max={r['sig_max']:.1f}σ, "
                f"overlap={r['overlap_fraction']:.1f}%, mask={'yes' if r['sn_pixel'] else 'no'}"
            )


if __name__ == "__main__":
    main()
