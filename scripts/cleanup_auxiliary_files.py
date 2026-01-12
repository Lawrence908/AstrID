#!/usr/bin/env python3
"""
Clean up auxiliary FITS files that aren't needed for differencing.

Removes files that don't have valid WCS headers (masks, weights, metadata, etc.)
and keeps only the main science image files.

Usage:
    python scripts/cleanup_auxiliary_files.py --input-dir output/fits_downloads
    python scripts/cleanup_auxiliary_files.py --input-dir output/fits_downloads --dry-run
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.wcs import WCS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Files to keep (main science images)
KEEP_PATTERNS = [
    ".fits",
    ".img",  # SWIFT UVOT files
]

# Files to remove (auxiliary files)
REMOVE_PATTERNS = [
    ".mask.fits",
    ".wt.fits",
    ".weight.fits",
    ".pswarp.mdc",
    ".target.psf",
    ".unconv.",
    ".cmf",
    ".jpg",
    ".png",
    ".num.fits",
    ".exp.fits",
    ".expwt.fits",
]


def has_valid_wcs(file_path: Path) -> bool:
    """Check if a FITS file has valid WCS information."""
    try:
        with fits.open(file_path) as hdul:
            for hdu in hdul:
                if hdu.data is None:
                    continue
                if len(hdu.data.shape) < 2:
                    continue
                try:
                    wcs = WCS(hdu.header)
                    if wcs.has_celestial:
                        return True
                except Exception:
                    continue
        return False
    except Exception:
        return False


def should_keep_file(file_path: Path, check_wcs: bool = False) -> tuple[bool, str]:
    """
    Determine if a file should be kept.

    Args:
        file_path: Path to file
        check_wcs: If True, verify WCS for ambiguous files (slower)

    Returns:
        (keep, reason)
    """
    name_lower = file_path.name.lower()

    # Fast pattern-based removal (no file I/O)
    for pattern in REMOVE_PATTERNS:
        if pattern in name_lower:
            return False, f"matches remove pattern: {pattern}"

    # Remove non-FITS files (JPG, PNG, etc.)
    if not any(ext in name_lower for ext in [".fits", ".img", ".fit"]):
        return False, "not a FITS file"

    # For ambiguous FITS files, check WCS if requested
    # Most files can be determined by filename alone
    if check_wcs and name_lower.endswith((".fits", ".fits.gz", ".img", ".fit")):
        # Only check WCS for files that might be science images
        # Skip obvious auxiliary files (already filtered above)
        try:
            if has_valid_wcs(file_path):
                return True, "has valid WCS"
            else:
                return False, "no valid WCS (auxiliary file)"
        except Exception:
            # If we can't read it, assume it's being written or corrupted
            return True, "unreadable (may be in use)"

    # Default: keep FITS files that passed pattern checks
    return True, "keep"


def cleanup_directory(
    directory: Path, dry_run: bool = False, check_wcs: bool = False
) -> dict[str, Any]:
    """Clean up auxiliary files in a directory."""
    stats = {
        "total_files": 0,
        "kept": 0,
        "removed": 0,
        "errors": 0,
        "space_freed_mb": 0,
    }

    # Process files in batches for progress reporting
    files = list(directory.rglob("*"))
    total = len([f for f in files if f.is_file()])

    logger.info(f"Scanning {total} files...")

    for _i, file_path in enumerate(files):
        if not file_path.is_file():
            continue

        stats["total_files"] += 1

        # Progress update every 1000 files
        if stats["total_files"] % 1000 == 0:
            logger.info(
                f"Processed {stats['total_files']}/{total} files... "
                f"({stats['removed']} removed, {stats['space_freed_mb']:.1f} MB freed)"
            )

        try:
            keep, reason = should_keep_file(file_path, check_wcs=check_wcs)
        except Exception:
            # Skip files that can't be accessed (may be in use)
            stats["kept"] += 1
            continue

        if keep:
            stats["kept"] += 1
        else:
            stats["removed"] += 1
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                stats["space_freed_mb"] += file_size_mb
            except Exception:
                # File may have been deleted or is in use
                pass

            if dry_run:
                if stats["removed"] <= 20:  # Only show first 20 in dry-run
                    logger.info(f"[DRY RUN] Would remove: {file_path.name} ({reason})")
            else:
                try:
                    file_path.unlink()
                except FileNotFoundError:
                    # Already deleted (race condition with download)
                    pass
                except PermissionError:
                    # File in use - skip it
                    stats["errors"] += 1
                    stats["removed"] -= 1  # Don't count as removed
                    stats["kept"] += 1
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
                    stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean up auxiliary FITS files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded FITS files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be removed without deleting",
    )
    parser.add_argument(
        "--check-wcs",
        action="store_true",
        help="Also check WCS for ambiguous files (slower but more accurate)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Directory not found: {args.input_dir}")
        return

    logger.info(f"Cleaning up auxiliary files in: {args.input_dir}")
    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be deleted")
    if args.check_wcs:
        logger.info("WCS checking enabled (slower but more accurate)")
    else:
        logger.info("Fast mode: using filename patterns only (recommended)")

    stats = cleanup_directory(
        args.input_dir, dry_run=args.dry_run, check_wcs=args.check_wcs
    )

    logger.info("\n" + "=" * 60)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files scanned: {stats['total_files']}")
    logger.info(f"Files kept: {stats['kept']}")
    logger.info(
        f"Files {'would be ' if args.dry_run else ''}removed: {stats['removed']}"
    )
    logger.info(
        f"Space {'would be ' if args.dry_run else ''}freed: {stats['space_freed_mb']:.1f} MB ({stats['space_freed_mb']/1024:.2f} GB)"
    )
    if stats["errors"] > 0:
        logger.warning(f"Errors: {stats['errors']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
