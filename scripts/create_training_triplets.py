#!/usr/bin/env python3
"""
Create training triplets (reference, science, difference) from processed FITS files.

This script generates normalized image triplets suitable for CNN training:
- Reads FITS files from fits_training and difference_images directories
- Creates 63x63 pixel cutouts centered on supernova positions
- Generates both "real" (at SN position) and "bogus" (random positions) samples
- Saves as compressed numpy arrays for efficient loading during training

Usage:
    # Test on 2014j dataset
    python scripts/create_training_triplets.py \
        --fits-dir output/datasets/sn2014j/fits_training \
        --diff-dir output/datasets/sn2014j/difference_images \
        --output-dir output/datasets/sn2014j/training_triplets \
        --cutout-size 63 \
        --bogus-ratio 1.0

    # Process full dataset
    python scripts/create_training_triplets.py \
        --fits-dir output/fits_training \
        --diff-dir output/difference_images \
        --output-dir output/training_triplets \
        --cutout-size 63 \
        --bogus-ratio 1.0 \
        --augment
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TripletStats:
    """Statistics for triplet generation."""

    sn_processed: int = 0
    real_samples: int = 0
    bogus_samples: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ImageTriplet:
    """Container for a single training triplet."""

    sn_name: str
    mission: str
    filter_name: str
    reference: np.ndarray  # (H, W)
    science: np.ndarray  # (H, W)
    difference: np.ndarray  # (H, W)
    label: int  # 1=real, 0=bogus
    center_x: float
    center_y: float
    metadata: dict[str, Any] = field(default_factory=dict)


def load_fits_data(fits_path: Path) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load FITS file and extract data + metadata."""
    try:
        with fits.open(fits_path) as hdul:
            # Try primary HDU first
            data = hdul[0].data
            header = dict(hdul[0].header)

            # If primary is empty, try HDU 1 (common for SWIFT .img files)
            if data is None and len(hdul) > 1:
                data = hdul[1].data
                header.update(dict(hdul[1].header))

            if data is None:
                logger.warning(f"No data found in {fits_path}")
                return None, {}

            data = data.astype(np.float32)

            # Handle different dimensionalities
            if data.ndim > 2:
                # Take first 2D slice
                data = data[0] if data.ndim == 3 else data[0, 0]

            return data, header
    except Exception as e:
        logger.warning(f"Failed to load {fits_path}: {e}")
        return None, {}


def normalize_image(image: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """Normalize image to [0, 1] range with robust clipping."""
    # Handle NaN/Inf
    finite_mask = np.isfinite(image)
    if not np.any(finite_mask):
        return np.zeros_like(image)

    # Robust normalization using percentiles
    valid_data = image[finite_mask]
    vmin = np.percentile(valid_data, 1)
    vmax = np.percentile(valid_data, clip_percentile)

    if vmax <= vmin:
        vmax = vmin + 1e-9

    # Normalize
    normalized = np.clip((image - vmin) / (vmax - vmin), 0, 1)
    normalized[~finite_mask] = 0  # Set invalid pixels to 0

    return normalized.astype(np.float32)


def ref_pixel_to_science_pixel(
    ref_x: float,
    ref_y: float,
    ref_header: dict[str, Any],
    sci_header: dict[str, Any],
) -> tuple[float, float] | None:
    """Convert reference-image pixel coords to science-image pixel coords via WCS.

    The difference image (and SN_X, SN_Y) are on the reference grid. The science
    FITS has its own WCS; using ref coords there would cut the wrong sky location.
    Returns (sci_x, sci_y) or None if WCS conversion fails.
    """
    try:
        ref_wcs = WCS(ref_header, naxis=2)
        sci_wcs = WCS(sci_header, naxis=2)
        if not ref_wcs.has_celestial or not sci_wcs.has_celestial:
            return None
        world = ref_wcs.pixel_to_world(ref_x, ref_y)
        sci_xy = sci_wcs.world_to_pixel(world)
        return (float(sci_xy[0]), float(sci_xy[1]))
    except Exception:
        return None


def extract_cutout(
    image: np.ndarray, center_x: float, center_y: float, size: int = 63
) -> np.ndarray | None:
    """Extract a square cutout centered at (center_x, center_y)."""
    h, w = image.shape
    half_size = size // 2

    # Calculate bounds
    x_min = int(center_x - half_size)
    x_max = int(center_x + half_size + 1)
    y_min = int(center_y - half_size)
    y_max = int(center_y + half_size + 1)

    # Check if cutout is within bounds
    if x_min < 0 or x_max > w or y_min < 0 or y_max > h:
        return None

    cutout = image[y_min:y_max, x_min:x_max]

    # Ensure exact size (handle edge cases)
    if cutout.shape != (size, size):
        return None

    return cutout


def generate_bogus_positions(
    image_shape: tuple[int, int],
    sn_position: tuple[float, float],
    n_bogus: int,
    cutout_size: int,
    min_distance: float = 100.0,
) -> list[tuple[float, float]]:
    """Generate random positions for bogus samples, avoiding SN position."""
    h, w = image_shape
    half_size = cutout_size // 2
    positions = []

    sn_x, sn_y = sn_position
    max_attempts = n_bogus * 10  # Prevent infinite loops

    for _ in range(max_attempts):
        if len(positions) >= n_bogus:
            break

        # Random position with valid cutout bounds
        x = random.uniform(half_size, w - half_size)
        y = random.uniform(half_size, h - half_size)

        # Check distance from SN
        dist = np.sqrt((x - sn_x) ** 2 + (y - sn_y) ** 2)
        if dist < min_distance:
            continue

        # Check distance from other bogus positions (avoid clustering)
        too_close = False
        for px, py in positions:
            if np.sqrt((x - px) ** 2 + (y - py) ** 2) < cutout_size:
                too_close = True
                break

        if not too_close:
            positions.append((x, y))

    return positions


def create_triplet(
    sn_name: str,
    mission: str,
    filter_name: str,
    ref_path: Path,
    sci_path: Path,
    diff_path: Path,
    center_x: float,
    center_y: float,
    cutout_size: int,
    label: int,
) -> ImageTriplet | None:
    """Create a single training triplet from FITS files.

    center_x, center_y are in reference/difference image pixel coordinates (same grid).
    The science FITS has a different WCS; we convert ref pixel -> sky -> science pixel
    so all three cutouts are centered on the same sky position.
    """
    # Load images
    ref_data, ref_header = load_fits_data(ref_path)
    sci_data, sci_header = load_fits_data(sci_path)
    diff_data, diff_header = load_fits_data(diff_path)

    if ref_data is None or sci_data is None or diff_data is None:
        return None

    # Science image has its own WCS — convert ref-pixel center to science-pixel center
    sci_center_x, sci_center_y = center_x, center_y
    sci_coords = ref_pixel_to_science_pixel(
        center_x, center_y, ref_header, sci_header
    )
    if sci_coords is not None:
        sci_center_x, sci_center_y = sci_coords

    # Extract cutouts: ref and diff share reference grid; science uses science grid
    ref_cutout = extract_cutout(ref_data, center_x, center_y, cutout_size)
    sci_cutout = extract_cutout(sci_data, sci_center_x, sci_center_y, cutout_size)
    diff_cutout = extract_cutout(diff_data, center_x, center_y, cutout_size)

    if ref_cutout is None or sci_cutout is None or diff_cutout is None:
        return None

    # Normalize each channel independently
    ref_norm = normalize_image(ref_cutout)
    sci_norm = normalize_image(sci_cutout)
    diff_norm = normalize_image(diff_cutout)

    # Create metadata
    metadata = {
        "ref_date": ref_header.get("DATE-OBS", "unknown"),
        "sci_date": sci_header.get("DATE-OBS", "unknown"),
        "overlap": diff_header.get("OVERLAP", 0.0),
        "sig_max": diff_header.get("SIG_MAX", 0.0),
    }

    return ImageTriplet(
        sn_name=sn_name,
        mission=mission,
        filter_name=filter_name,
        reference=ref_norm,
        science=sci_norm,
        difference=diff_norm,
        label=label,
        center_x=center_x,
        center_y=center_y,
        metadata=metadata,
    )


def augment_triplet(triplet: ImageTriplet) -> list[ImageTriplet]:
    """Create augmented versions of a triplet (rotation, flipping)."""
    augmented = [triplet]  # Include original

    # 90° rotations
    for k in [1, 2, 3]:
        aug = ImageTriplet(
            sn_name=triplet.sn_name,
            mission=triplet.mission,
            filter_name=triplet.filter_name,
            reference=np.rot90(triplet.reference, k),
            science=np.rot90(triplet.science, k),
            difference=np.rot90(triplet.difference, k),
            label=triplet.label,
            center_x=triplet.center_x,
            center_y=triplet.center_y,
            metadata={**triplet.metadata, "augmentation": f"rot{k*90}"},
        )
        augmented.append(aug)

    # Horizontal flip
    aug = ImageTriplet(
        sn_name=triplet.sn_name,
        mission=triplet.mission,
        filter_name=triplet.filter_name,
        reference=np.fliplr(triplet.reference),
        science=np.fliplr(triplet.science),
        difference=np.fliplr(triplet.difference),
        label=triplet.label,
        center_x=triplet.center_x,
        center_y=triplet.center_y,
        metadata={**triplet.metadata, "augmentation": "flip_lr"},
    )
    augmented.append(aug)

    # Vertical flip
    aug = ImageTriplet(
        sn_name=triplet.sn_name,
        mission=triplet.mission,
        filter_name=triplet.filter_name,
        reference=np.flipud(triplet.reference),
        science=np.flipud(triplet.science),
        difference=np.flipud(triplet.difference),
        label=triplet.label,
        center_x=triplet.center_x,
        center_y=triplet.center_y,
        metadata={**triplet.metadata, "augmentation": "flip_ud"},
    )
    augmented.append(aug)

    return augmented


def process_sn(
    sn_name: str,
    fits_dir: Path,
    diff_dir: Path,
    cutout_size: int,
    bogus_ratio: float,
    augment: bool,
) -> list[ImageTriplet]:
    """Process a single supernova to generate training triplets."""
    triplets = []

    # Find difference image
    sn_diff_dir = diff_dir / sn_name
    if not sn_diff_dir.exists():
        logger.warning(f"No difference images found for {sn_name}")
        return triplets

    diff_files = list(sn_diff_dir.glob("*_diff.fits"))
    if not diff_files:
        logger.warning(f"No *_diff.fits files found for {sn_name}")
        return triplets

    # Process each filter
    for diff_file in diff_files:
        # Parse filename: {sn_name}_{mission}_{filter}_diff.fits
        parts = diff_file.stem.split("_")
        if len(parts) < 4:
            continue

        mission = parts[-3]
        filter_name = parts[-2]

        # Load difference image to get SN position
        diff_data, diff_header = load_fits_data(diff_file)
        if diff_data is None:
            continue

        # Get SN pixel position from header
        sn_x = diff_header.get("SN_X")
        sn_y = diff_header.get("SN_Y")

        if sn_x is None or sn_y is None:
            # Use image center as fallback
            logger.warning(
                f"No SN position in header for {sn_name}, using image center"
            )
            sn_x = diff_data.shape[1] / 2
            sn_y = diff_data.shape[0] / 2

        # Find corresponding reference and science images
        # Files are named like: SWIFT_sw00032503001uuu_sk.fits
        # Filter name is embedded in the observation ID
        sn_fits_dir = fits_dir / sn_name
        ref_files = list(sn_fits_dir.glob(f"reference/{mission}_*{filter_name}*.fits"))
        sci_files = list(sn_fits_dir.glob(f"science/{mission}_*{filter_name}*.fits"))

        if not ref_files or not sci_files:
            logger.warning(f"Missing ref/sci files for {sn_name} {mission} {filter_name}")
            continue

        ref_path = ref_files[0]
        sci_path = sci_files[0]

        # Create REAL sample at SN position
        real_triplet = create_triplet(
            sn_name=sn_name,
            mission=mission,
            filter_name=filter_name,
            ref_path=ref_path,
            sci_path=sci_path,
            diff_path=diff_file,
            center_x=sn_x,
            center_y=sn_y,
            cutout_size=cutout_size,
            label=1,  # Real
        )

        if real_triplet:
            if augment:
                triplets.extend(augment_triplet(real_triplet))
            else:
                triplets.append(real_triplet)

        # Generate BOGUS samples
        n_bogus = int(bogus_ratio * (len(augment_triplet(real_triplet)) if augment else 1))
        bogus_positions = generate_bogus_positions(
            image_shape=diff_data.shape,
            sn_position=(sn_x, sn_y),
            n_bogus=n_bogus,
            cutout_size=cutout_size,
        )

        for bx, by in bogus_positions:
            bogus_triplet = create_triplet(
                sn_name=sn_name,
                mission=mission,
                filter_name=filter_name,
                ref_path=ref_path,
                sci_path=sci_path,
                diff_path=diff_file,
                center_x=bx,
                center_y=by,
                cutout_size=cutout_size,
                label=0,  # Bogus
            )

            if bogus_triplet:
                triplets.append(bogus_triplet)

    return triplets


def visualize_triplet(
    triplet: ImageTriplet, output_path: Path, show_metadata: bool = True
) -> None:
    """Create a visualization PNG for a single triplet.

    Reference and science panels use gray; the difference panel uses RdBu_r
    so positive residuals (new source) appear red and negative (over-subtraction) blue.
    Overlap is shown correctly whether the header stores fraction (0-1) or percentage (0-100).

    Args:
        triplet: ImageTriplet to visualize
        output_path: Path to save PNG
        show_metadata: Whether to include metadata in title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    images = [triplet.reference, triplet.science, triplet.difference]
    titles = ["Reference (pre-SN)", "Science (with SN)", "Difference (sci - ref)"]
    # Use diverging colormap for difference so positive (new source) vs negative (residuals) is clear
    cmaps = ["gray", "gray", "RdBu_r"]
    vmins = [0, 0, 0]
    vmaxs = [1, 1, 1]
    
    for ax, img, title, cmap, vmin, vmax in zip(axes, images, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(img, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Mark center with crosshair
        h, w = img.shape
        ax.plot([w/2], [h/2], "r+", markersize=15, markeredgewidth=2)
    
    # Create title with metadata
    label_str = "REAL" if triplet.label == 1 else "BOGUS"
    label_color = "green" if triplet.label == 1 else "red"
    
    if show_metadata:
        title_parts = [
            f"{triplet.sn_name} | {triplet.mission} {triplet.filter_name}",
            f"Position: ({triplet.center_x:.1f}, {triplet.center_y:.1f})",
        ]
        
        if "sig_max" in triplet.metadata:
            sig_max = triplet.metadata["sig_max"]
            if sig_max > 1e6:
                title_parts.append(f"σ_max: {sig_max:.2e}")
            else:
                title_parts.append(f"σ_max: {sig_max:.1f}")
        
        if "overlap" in triplet.metadata:
            overlap = triplet.metadata["overlap"]
            # Header may store fraction (0–1) or percentage (0–100)
            overlap_pct = overlap * 100 if overlap <= 1 else overlap
            title_parts.append(f"overlap: {overlap_pct:.1f}%")
        
        title = " | ".join(title_parts)
    else:
        title = f"{triplet.sn_name} - {triplet.mission} {triplet.filter_name}"
    
    fig.suptitle(
        f"{title}\nLabel: {label_str}",
        fontsize=14,
        fontweight="bold",
        color=label_color,
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_triplets(
    triplets: list[ImageTriplet], 
    output_dir: Path, 
    split: str = "train",
    visualize: bool = False,
) -> None:
    """Save triplets as compressed numpy arrays and optionally create visualizations.
    
    Args:
        triplets: List of ImageTriplet objects
        output_dir: Output directory
        split: Dataset split name (train/val/test)
        visualize: If True, create PNG visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate by label
    real_triplets = [t for t in triplets if t.label == 1]
    bogus_triplets = [t for t in triplets if t.label == 0]

    logger.info(f"Saving {len(real_triplets)} real and {len(bogus_triplets)} bogus samples")

    # Create visualization directory if needed
    if visualize:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        real_viz_dir = viz_dir / "real"
        bogus_viz_dir = viz_dir / "bogus"
        real_viz_dir.mkdir(exist_ok=True)
        bogus_viz_dir.mkdir(exist_ok=True)
        
        logger.info(f"Creating visualizations in {viz_dir}")

    # Stack into arrays
    if real_triplets:
        real_images = np.stack(
            [
                np.stack([t.reference, t.science, t.difference], axis=0)
                for t in real_triplets
            ],
            axis=0,
        )  # (N, 3, H, W)
        real_labels = np.ones(len(real_triplets), dtype=np.int32)
        real_metadata = [
            {
                "sn_name": t.sn_name,
                "mission": t.mission,
                "filter": t.filter_name,
                "center_x": t.center_x,
                "center_y": t.center_y,
                **t.metadata,
            }
            for t in real_triplets
        ]

        np.savez_compressed(
            output_dir / f"{split}_real.npz",
            images=real_images,
            labels=real_labels,
        )
        with open(output_dir / f"{split}_real_metadata.json", "w") as f:
            json.dump(real_metadata, f, indent=2)
        
        # Create visualizations for real samples
        if visualize:
            for i, triplet in enumerate(real_triplets):
                aug_suffix = ""
                if "augmentation" in triplet.metadata:
                    aug_suffix = f"_{triplet.metadata['augmentation']}"
                
                viz_filename = f"{triplet.sn_name}_{triplet.mission}_{triplet.filter_name}_real_{i:03d}{aug_suffix}.png"
                viz_path = real_viz_dir / viz_filename
                visualize_triplet(triplet, viz_path)

    if bogus_triplets:
        bogus_images = np.stack(
            [
                np.stack([t.reference, t.science, t.difference], axis=0)
                for t in bogus_triplets
            ],
            axis=0,
        )
        bogus_labels = np.zeros(len(bogus_triplets), dtype=np.int32)
        bogus_metadata = [
            {
                "sn_name": t.sn_name,
                "mission": t.mission,
                "filter": t.filter_name,
                "center_x": t.center_x,
                "center_y": t.center_y,
                **t.metadata,
            }
            for t in bogus_triplets
        ]

        np.savez_compressed(
            output_dir / f"{split}_bogus.npz",
            images=bogus_images,
            labels=bogus_labels,
        )
        with open(output_dir / f"{split}_bogus_metadata.json", "w") as f:
            json.dump(bogus_metadata, f, indent=2)
        
        # Create visualizations for bogus samples
        if visualize:
            for i, triplet in enumerate(bogus_triplets):
                viz_filename = f"{triplet.sn_name}_{triplet.mission}_{triplet.filter_name}_bogus_{i:03d}.png"
                viz_path = bogus_viz_dir / viz_filename
                visualize_triplet(triplet, viz_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create training triplets from FITS files"
    )
    parser.add_argument(
        "--fits-dir",
        type=Path,
        required=True,
        help="Directory containing organized FITS files (output of organize_training_pairs.py)",
    )
    parser.add_argument(
        "--diff-dir",
        type=Path,
        required=True,
        help="Directory containing difference images (output of generate_difference_images.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for training triplets",
    )
    parser.add_argument(
        "--cutout-size",
        type=int,
        default=63,
        help="Size of cutouts (default: 63x63 pixels)",
    )
    parser.add_argument(
        "--bogus-ratio",
        type=float,
        default=1.0,
        help="Ratio of bogus to real samples (default: 1.0 for balanced dataset)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation (rotations, flips)",
    )
    parser.add_argument(
        "--sn",
        nargs="+",
        default=None,
        help="Only process specific supernovae (space-separated names)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create PNG visualizations of all triplets",
    )

    args = parser.parse_args()

    if not args.fits_dir.exists():
        logger.error(f"FITS directory does not exist: {args.fits_dir}")
        return

    if not args.diff_dir.exists():
        logger.error(f"Difference directory does not exist: {args.diff_dir}")
        return

    logger.info(f"Processing triplets from:")
    logger.info(f"  FITS: {args.fits_dir}")
    logger.info(f"  Diff: {args.diff_dir}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Cutout size: {args.cutout_size}x{args.cutout_size}")
    logger.info(f"  Bogus ratio: {args.bogus_ratio}")
    logger.info(f"  Augmentation: {args.augment}")

    # Find all SNe to process
    if args.sn:
        sne_to_process = args.sn
    else:
        # Get all SNe from difference directory
        sne_to_process = [
            d.name for d in args.diff_dir.iterdir() if d.is_dir()
        ]

    logger.info(f"Found {len(sne_to_process)} supernovae to process")

    # Process each SN
    all_triplets = []
    stats = TripletStats()

    for sn_name in sorted(sne_to_process):
        logger.info(f"Processing {sn_name}...")
        try:
            triplets = process_sn(
                sn_name=sn_name,
                fits_dir=args.fits_dir,
                diff_dir=args.diff_dir,
                cutout_size=args.cutout_size,
                bogus_ratio=args.bogus_ratio,
                augment=args.augment,
            )

            if triplets:
                all_triplets.extend(triplets)
                stats.sn_processed += 1
                real_count = sum(1 for t in triplets if t.label == 1)
                bogus_count = sum(1 for t in triplets if t.label == 0)
                stats.real_samples += real_count
                stats.bogus_samples += bogus_count
                logger.info(f"  Created {real_count} real + {bogus_count} bogus samples")
            else:
                stats.failed += 1
                logger.warning(f"  No triplets created for {sn_name}")

        except Exception as e:
            stats.failed += 1
            stats.errors.append(f"{sn_name}: {e}")
            logger.error(f"  Failed to process {sn_name}: {e}")

    # Save all triplets
    if all_triplets:
        logger.info(f"\nSaving {len(all_triplets)} total triplets...")
        save_triplets(all_triplets, args.output_dir, split="train", visualize=args.visualize)

        # Save summary
        summary = {
            "total_sne": len(sne_to_process),
            "processed": stats.sn_processed,
            "failed": stats.failed,
            "real_samples": stats.real_samples,
            "bogus_samples": stats.bogus_samples,
            "total_samples": len(all_triplets),
            "cutout_size": args.cutout_size,
            "bogus_ratio": args.bogus_ratio,
            "augmentation": args.augment,
            "errors": stats.errors,
        }

        with open(args.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("TRIPLET GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\n📊 SUMMARY")
        logger.info("-" * 40)
        logger.info(f"  SNe processed:     {stats.sn_processed}/{len(sne_to_process)}")
        logger.info(f"  Real samples:      {stats.real_samples}")
        logger.info(f"  Bogus samples:     {stats.bogus_samples}")
        logger.info(f"  Total samples:     {len(all_triplets)}")
        logger.info(f"  Class balance:     {stats.bogus_samples/max(stats.real_samples,1):.2f}:1")
        if stats.failed > 0:
            logger.info(f"\n⚠️  Failed: {stats.failed}")
        logger.info(f"\n📁 Output: {args.output_dir}")
        logger.info("=" * 70)

    else:
        logger.error("No triplets were created!")


if __name__ == "__main__":
    main()
