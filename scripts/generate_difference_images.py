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

import sys

import argparse
import gc
import json
import logging
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domains.differencing.pipeline import (
    MISSION_FILTERS,
    DifferencingResult,
    SNDifferencingPipeline,
    detect_mission_from_filename,
)

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
    """Get list of SNe with same-mission pairs from the training manifest."""
    if not manifest_path.exists():
        return []
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


def discover_sne_from_input_dir(input_dir: Path) -> list[str]:
    """Discover SNe that have a same-mission ref/sci pair on disk (ignore manifest).
    Use when manifest is missing or stale so we process every SN that has data.
    """
    discovered = []
    for sn_dir in sorted(input_dir.iterdir()):
        if not sn_dir.is_dir():
            continue
        sn_name = sn_dir.name
        if find_matching_filter_pair(sn_name, input_dir) is not None:
            discovered.append(sn_name)
    return discovered


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
        n_from_manifest = len(sne_to_process)
        # If manifest is missing or only lists some SNe, discover from directory
        # so we process every SN that has ref/sci same-mission pair on disk
        discovered = discover_sne_from_input_dir(input_dir)
        for sn_name in discovered:
            if sn_name not in sne_to_process:
                sne_to_process.append(sn_name)
        sne_to_process = sorted(sne_to_process)
        if len(sne_to_process) > n_from_manifest:
            logger.info(
                "Manifest had %d same-mission SNe; discovered %d more from directory (total %d)",
                n_from_manifest,
                len(sne_to_process) - n_from_manifest,
                len(sne_to_process),
            )

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
