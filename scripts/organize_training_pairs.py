#!/usr/bin/env python3
"""
Organize validated supernova pairs for training.

Reads the audit report and copies/symlinks complete reference/science pairs
to a flat training directory structure. Optionally decompresses .fits.gz files
and handles .img files (Swift UVOT format, which are FITS-compatible).

Usage:
    python scripts/organize_training_pairs.py --input-dir output/fits_downloads --output-dir output/fits_training
    python scripts/organize_training_pairs.py --input-dir output/fits_downloads --output-dir output/fits_training --decompress
    python scripts/organize_training_pairs.py --input-dir output/fits_downloads --output-dir output/fits_training --symlink
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".fits", ".fits.gz", ".fit", ".img")


@dataclass
class OrganizeStats:
    """Statistics for the organization process."""

    sn_processed: int = 0
    sn_skipped: int = 0
    reference_files_copied: int = 0
    science_files_copied: int = 0
    files_decompressed: int = 0
    files_renamed: int = 0
    errors: list[str] = field(default_factory=list)


def load_audit_report(audit_path: Path) -> dict[str, Any] | None:
    """Load the audit report JSON."""
    if not audit_path.exists():
        return None
    with open(audit_path) as f:
        return json.load(f)


def get_clean_filename(original_path: Path, mission: str | None = None) -> str:
    """
    Generate a cleaner filename from the original MAST path.

    Original: mastDownload/SWIFT/00032503002/sw00032503002uw1_sk.img
    Clean: SWIFT_00032503002_uw1_sk.fits

    Original: mastDownload/PS1/rings.v3.skycell.1043.044.wrp.z.56435_60074/rings.v3.skycell.1043.044.wrp.z.56435_60074.fits
    Clean: PS1_skycell.1043.044_z_56435_60074.fits
    """
    name = original_path.name
    parts = original_path.parts

    # Try to extract mission from path
    if mission is None:
        for i, part in enumerate(parts):
            if part == "mastDownload" and i + 1 < len(parts):
                mission = parts[i + 1]
                break

    # Handle .fits.gz -> .fits
    if name.lower().endswith(".fits.gz"):
        name = name[:-3]  # Remove .gz
    # Handle .img -> .fits (Swift UVOT files are FITS-compatible)
    elif name.lower().endswith(".img"):
        name = name[:-4] + ".fits"

    # Add mission prefix if we have it and it's not already there
    if mission and not name.upper().startswith(mission.upper()):
        name = f"{mission}_{name}"

    return name


def extract_mission_from_path(file_path: Path) -> str | None:
    """Extract mission name from MAST download path."""
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part == "mastDownload" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def copy_or_link_file(
    src: Path,
    dst: Path,
    use_symlink: bool = False,
    decompress: bool = False,
) -> bool:
    """
    Copy or symlink a file, optionally decompressing .fits.gz.

    Returns True if successful.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file/symlink to avoid "same file" errors
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if decompress and src.name.lower().endswith(".fits.gz"):
            # Decompress to .fits
            dst_fits = dst.with_suffix("") if dst.suffix == ".gz" else dst
            if not dst_fits.name.endswith(".fits"):
                dst_fits = dst_fits.with_suffix(".fits")
            with gzip.open(src, "rb") as f_in:
                with open(dst_fits, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True

        if use_symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

        return True

    except Exception as exc:
        logger.warning(f"Failed to copy {src}: {exc}")
        return False


def organize_sn_pair(
    sn_name: str,
    input_dir: Path,
    output_dir: Path,
    use_symlink: bool = False,
    decompress: bool = False,
    stats: OrganizeStats | None = None,
) -> dict[str, Any]:
    """
    Organize a single supernova's reference and science files.

    Returns a manifest entry for this SN.
    """
    if stats is None:
        stats = OrganizeStats()

    sn_input = input_dir / sn_name
    sn_output = output_dir / sn_name

    manifest_entry: dict[str, Any] = {
        "sn_name": sn_name,
        "reference_files": [],
        "science_files": [],
        "errors": [],
    }

    # Process reference files
    ref_input = sn_input / "reference"
    ref_output = sn_output / "reference"
    if ref_input.exists():
        for src_file in ref_input.rglob("*"):
            if not src_file.is_file():
                continue
            if not any(
                src_file.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS
            ):
                continue

            # Skip weight and mask files for now (focus on science images)
            name_lower = src_file.name.lower()
            if ".wt." in name_lower or ".mask." in name_lower or ".num." in name_lower:
                continue
            if name_lower.endswith(".wt.fits") or name_lower.endswith(".mask.fits"):
                continue

            mission = extract_mission_from_path(src_file)
            clean_name = get_clean_filename(src_file, mission)
            dst_file = ref_output / clean_name

            # Handle decompression naming
            if decompress and src_file.name.lower().endswith(".fits.gz"):
                dst_file = ref_output / clean_name.replace(".fits.gz", ".fits")

            if copy_or_link_file(src_file, dst_file, use_symlink, decompress):
                stats.reference_files_copied += 1
                manifest_entry["reference_files"].append(
                    str(dst_file.relative_to(output_dir))
                )
                if decompress and src_file.name.lower().endswith(".fits.gz"):
                    stats.files_decompressed += 1
            else:
                manifest_entry["errors"].append(f"Failed to copy: {src_file}")

    # Process science files
    sci_input = sn_input / "science"
    sci_output = sn_output / "science"
    if sci_input.exists():
        for src_file in sci_input.rglob("*"):
            if not src_file.is_file():
                continue
            if not any(
                src_file.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS
            ):
                continue

            # Skip weight and mask files
            name_lower = src_file.name.lower()
            if ".wt." in name_lower or ".mask." in name_lower or ".num." in name_lower:
                continue
            if name_lower.endswith(".wt.fits") or name_lower.endswith(".mask.fits"):
                continue

            mission = extract_mission_from_path(src_file)
            clean_name = get_clean_filename(src_file, mission)
            dst_file = sci_output / clean_name

            # Handle decompression naming
            if decompress and src_file.name.lower().endswith(".fits.gz"):
                dst_file = sci_output / clean_name.replace(".fits.gz", ".fits")

            if copy_or_link_file(src_file, dst_file, use_symlink, decompress):
                stats.science_files_copied += 1
                manifest_entry["science_files"].append(
                    str(dst_file.relative_to(output_dir))
                )
                if decompress and src_file.name.lower().endswith(".fits.gz"):
                    stats.files_decompressed += 1
            else:
                manifest_entry["errors"].append(f"Failed to copy: {src_file}")

    return manifest_entry


def run_organize(
    input_dir: Path,
    output_dir: Path,
    audit_report: dict[str, Any] | None = None,
    use_symlink: bool = False,
    decompress: bool = False,
    sn_filter: list[str] | None = None,
) -> tuple[list[dict[str, Any]], OrganizeStats]:
    """
    Organize all complete SN pairs from input to output directory.

    Returns manifest entries and statistics.
    """
    stats = OrganizeStats()
    manifest: list[dict[str, Any]] = []

    # Determine which SNe to process
    if audit_report:
        # Use audit report to find complete pairs
        complete_pairs = [
            entry["sn_name"]
            for entry in audit_report.get("entries", [])
            if entry.get("is_complete_pair", False)
        ]
    else:
        # Scan directory for SNe with both reference and science
        complete_pairs = []
        for sn_dir in input_dir.iterdir():
            if not sn_dir.is_dir():
                continue
            ref_dir = sn_dir / "reference"
            sci_dir = sn_dir / "science"
            if ref_dir.exists() and sci_dir.exists():
                # Check they have files
                ref_files = list(ref_dir.rglob("*.fits")) + list(ref_dir.rglob("*.img"))
                sci_files = list(sci_dir.rglob("*.fits")) + list(sci_dir.rglob("*.img"))
                if ref_files and sci_files:
                    complete_pairs.append(sn_dir.name)

    # Apply filter if provided
    if sn_filter:
        complete_pairs = [sn for sn in complete_pairs if sn in sn_filter]

    logger.info(f"Found {len(complete_pairs)} complete pairs to organize")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each SN
    for sn_name in sorted(complete_pairs):
        logger.info(f"Organizing {sn_name}...")
        entry = organize_sn_pair(
            sn_name=sn_name,
            input_dir=input_dir,
            output_dir=output_dir,
            use_symlink=use_symlink,
            decompress=decompress,
            stats=stats,
        )
        manifest.append(entry)
        stats.sn_processed += 1

        if entry["errors"]:
            stats.errors.extend(entry["errors"])

    return manifest, stats


def print_summary(stats: OrganizeStats, output_dir: Path) -> None:
    """Print organization summary."""
    logger.info("\n" + "=" * 70)
    logger.info("ORGANIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\n📊 SUMMARY")
    logger.info("-" * 40)
    logger.info(f"  SNe processed:            {stats.sn_processed}")
    logger.info(f"  Reference files copied:   {stats.reference_files_copied}")
    logger.info(f"  Science files copied:     {stats.science_files_copied}")
    logger.info(f"  Files decompressed:       {stats.files_decompressed}")
    total = stats.reference_files_copied + stats.science_files_copied
    logger.info(f"  Total files:              {total}")

    if stats.errors:
        logger.info(f"\n⚠️  Errors: {len(stats.errors)}")
        for err in stats.errors[:5]:
            logger.info(f"    - {err}")
        if len(stats.errors) > 5:
            logger.info(f"    ... and {len(stats.errors) - 5} more")

    logger.info(f"\n📁 Output directory: {output_dir}")
    logger.info("=" * 70)


def save_manifest(manifest: list[dict[str, Any]], output_path: Path) -> None:
    """Save the training manifest JSON."""
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\n📄 Manifest saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize validated supernova pairs for training."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/fits_downloads"),
        help="Directory containing downloaded SN subdirectories (default: output/fits_downloads)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/fits_training"),
        help="Output directory for organized training data (default: output/fits_training)",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=None,
        help="Path to audit report JSON (default: <input-dir>/audit_report.json)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress .fits.gz files during copy",
    )
    parser.add_argument(
        "--sn",
        nargs="+",
        default=None,
        help="Only process specific supernovae (space-separated names)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before organizing",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    # Load audit report if available
    audit_path = args.audit_report or (args.input_dir / "audit_report.json")
    audit_report = load_audit_report(audit_path)
    if audit_report:
        logger.info(f"Loaded audit report from: {audit_path}")
    else:
        logger.info("No audit report found, scanning directories directly")

    # Clean output if requested
    if args.clean and args.output_dir.exists():
        logger.info(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    logger.info(f"Organizing from: {args.input_dir}")
    logger.info(f"Organizing to:   {args.output_dir}")
    if args.symlink:
        logger.info("Using symlinks (no file copy)")
    if args.decompress:
        logger.info("Decompressing .fits.gz files")

    manifest, stats = run_organize(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        audit_report=audit_report,
        use_symlink=args.symlink,
        decompress=args.decompress,
        sn_filter=args.sn,
    )

    print_summary(stats, args.output_dir)

    # Save manifest
    manifest_path = args.output_dir / "training_manifest.json"
    save_manifest(manifest, manifest_path)


if __name__ == "__main__":
    main()
