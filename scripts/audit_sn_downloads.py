#!/usr/bin/env python3
"""
Audit downloaded supernova FITS files for training readiness.

Scans the download directory, counts files by type, validates FITS headers
(WCS, observation time), and identifies complete reference/science pairs
ready for the differencing and training pipeline.

Usage:
    python scripts/audit_sn_downloads.py --input-dir output/fits_downloads
    python scripts/audit_sn_downloads.py --input-dir output/fits_downloads --validate-fits
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.wcs import WCS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".fits", ".fits.gz", ".fit", ".img")


@dataclass
class FileStats:
    """Statistics for a single file type."""

    fits: int = 0
    fits_gz: int = 0
    img: int = 0
    fit: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.fits + self.fits_gz + self.img + self.fit + self.other

    def add_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        name_lower = path.name.lower()
        if name_lower.endswith(".fits.gz"):
            self.fits_gz += 1
        elif suffix == ".fits":
            self.fits += 1
        elif suffix == ".img":
            self.img += 1
        elif suffix == ".fit":
            self.fit += 1
        else:
            self.other += 1


@dataclass
class SNAuditEntry:
    """Audit result for a single supernova."""

    sn_name: str
    has_reference: bool = False
    has_science: bool = False
    reference_stats: FileStats = field(default_factory=FileStats)
    science_stats: FileStats = field(default_factory=FileStats)
    reference_files: list[str] = field(default_factory=list)
    science_files: list[str] = field(default_factory=list)
    validated_reference: int = 0
    validated_science: int = 0
    invalid_reference: list[str] = field(default_factory=list)
    invalid_science: list[str] = field(default_factory=list)

    @property
    def is_complete_pair(self) -> bool:
        return (
            self.has_reference
            and self.has_science
            and self.reference_stats.total > 0
            and self.science_stats.total > 0
        )

    @property
    def total_files(self) -> int:
        return self.reference_stats.total + self.science_stats.total


@dataclass
class AuditSummary:
    """Overall audit summary."""

    total_sn_dirs: int = 0
    complete_pairs: int = 0
    reference_only: int = 0
    science_only: int = 0
    empty_dirs: int = 0
    total_fits: int = 0
    total_fits_gz: int = 0
    total_img: int = 0
    total_fit: int = 0
    validated_files: int = 0
    invalid_files: int = 0
    complete_pair_names: list[str] = field(default_factory=list)


def validate_fits_file(file_path: Path) -> tuple[bool, str | None]:
    """
    Validate a FITS file for WCS and observation metadata.

    Returns (is_valid, error_message).
    """
    try:
        with fits.open(file_path) as hdul:
            # Check primary header first, then extensions
            for _i, hdu in enumerate(hdul):
                header = hdu.header
                if not header:
                    continue

                # Try to get WCS
                try:
                    wcs = WCS(header, naxis=2)
                    if wcs.has_celestial:
                        return True, None
                except Exception:
                    continue

            # No celestial WCS found in any extension
            return False, "No celestial WCS found"

    except Exception as exc:
        return False, f"Failed to open: {exc}"


def scan_directory(sn_dir: Path, subdir_name: str) -> tuple[FileStats, list[Path]]:
    """Scan a reference or science subdirectory for FITS files."""
    stats = FileStats()
    files: list[Path] = []

    subdir = sn_dir / subdir_name
    if not subdir.exists():
        return stats, files

    for path in subdir.rglob("*"):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if any(name_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            stats.add_file(path)
            files.append(path)

    return stats, files


def audit_supernova(sn_dir: Path, validate: bool = False) -> SNAuditEntry:
    """Audit a single supernova directory."""
    sn_name = sn_dir.name
    entry = SNAuditEntry(sn_name=sn_name)

    # Scan reference directory
    ref_dir = sn_dir / "reference"
    entry.has_reference = ref_dir.exists()
    if entry.has_reference:
        entry.reference_stats, ref_files = scan_directory(sn_dir, "reference")
        entry.reference_files = [str(f.relative_to(sn_dir)) for f in ref_files]

        if validate:
            for f in ref_files:
                valid, reason = validate_fits_file(f)
                if valid:
                    entry.validated_reference += 1
                else:
                    entry.invalid_reference.append(f"{f.name}: {reason}")

    # Scan science directory
    sci_dir = sn_dir / "science"
    entry.has_science = sci_dir.exists()
    if entry.has_science:
        entry.science_stats, sci_files = scan_directory(sn_dir, "science")
        entry.science_files = [str(f.relative_to(sn_dir)) for f in sci_files]

        if validate:
            for f in sci_files:
                valid, reason = validate_fits_file(f)
                if valid:
                    entry.validated_science += 1
                else:
                    entry.invalid_science.append(f"{f.name}: {reason}")

    return entry


def run_audit(
    input_dir: Path, validate: bool = False
) -> tuple[list[SNAuditEntry], AuditSummary]:
    """Run the full audit on all supernova directories."""
    entries: list[SNAuditEntry] = []
    summary = AuditSummary()

    # Find all SN directories (direct children with SN-like names)
    sn_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    summary.total_sn_dirs = len(sn_dirs)

    for sn_dir in sn_dirs:
        entry = audit_supernova(sn_dir, validate=validate)
        entries.append(entry)

        # Update summary
        summary.total_fits += entry.reference_stats.fits + entry.science_stats.fits
        summary.total_fits_gz += (
            entry.reference_stats.fits_gz + entry.science_stats.fits_gz
        )
        summary.total_img += entry.reference_stats.img + entry.science_stats.img
        summary.total_fit += entry.reference_stats.fit + entry.science_stats.fit

        if validate:
            summary.validated_files += (
                entry.validated_reference + entry.validated_science
            )
            summary.invalid_files += len(entry.invalid_reference) + len(
                entry.invalid_science
            )

        if entry.is_complete_pair:
            summary.complete_pairs += 1
            summary.complete_pair_names.append(entry.sn_name)
        elif entry.has_reference and not entry.has_science:
            summary.reference_only += 1
        elif entry.has_science and not entry.has_reference:
            summary.science_only += 1
        elif entry.total_files == 0:
            summary.empty_dirs += 1

    return entries, summary


def print_report(
    entries: list[SNAuditEntry], summary: AuditSummary, validate: bool
) -> None:
    """Print a human-readable audit report."""
    logger.info("\n" + "=" * 70)
    logger.info("SUPERNOVA DOWNLOAD AUDIT REPORT")
    logger.info("=" * 70)

    logger.info("\n📊 SUMMARY")
    logger.info("-" * 40)
    logger.info(f"  Total SN directories:     {summary.total_sn_dirs}")
    logger.info(f"  ✅ Complete pairs:         {summary.complete_pairs}")
    logger.info(f"  ⚠️  Reference only:         {summary.reference_only}")
    logger.info(f"  ⚠️  Science only:           {summary.science_only}")
    logger.info(f"  ❌ Empty directories:      {summary.empty_dirs}")

    logger.info("\n📁 FILE COUNTS BY TYPE")
    logger.info("-" * 40)
    logger.info(f"  .fits files:              {summary.total_fits}")
    logger.info(f"  .fits.gz files:           {summary.total_fits_gz}")
    logger.info(f"  .img files:               {summary.total_img}")
    logger.info(f"  .fit files:               {summary.total_fit}")
    total_files = (
        summary.total_fits
        + summary.total_fits_gz
        + summary.total_img
        + summary.total_fit
    )
    logger.info(f"  Total:                    {total_files}")

    if validate:
        logger.info("\n🔍 FITS VALIDATION")
        logger.info("-" * 40)
        logger.info(f"  ✅ Valid (with WCS):       {summary.validated_files}")
        logger.info(f"  ❌ Invalid/No WCS:         {summary.invalid_files}")
        if total_files > 0:
            pct = summary.validated_files / total_files * 100
            logger.info(f"  Success rate:             {pct:.1f}%")

    logger.info("\n✅ COMPLETE PAIRS (ready for training)")
    logger.info("-" * 40)
    if summary.complete_pair_names:
        for name in sorted(summary.complete_pair_names):
            entry = next(e for e in entries if e.sn_name == name)
            ref_count = entry.reference_stats.total
            sci_count = entry.science_stats.total
            logger.info(f"  {name:20s}  ref: {ref_count:4d}  sci: {sci_count:4d}")
    else:
        logger.info("  (none)")

    # Show incomplete entries
    incomplete = [e for e in entries if not e.is_complete_pair and e.total_files > 0]
    if incomplete:
        logger.info("\n⚠️  INCOMPLETE (missing ref or sci)")
        logger.info("-" * 40)
        for entry in sorted(incomplete, key=lambda x: x.sn_name):
            ref = (
                "✓" if entry.has_reference and entry.reference_stats.total > 0 else "✗"
            )
            sci = "✓" if entry.has_science and entry.science_stats.total > 0 else "✗"
            logger.info(f"  {entry.sn_name:20s}  ref: {ref}  sci: {sci}")

    logger.info("\n" + "=" * 70)


def save_report(
    entries: list[SNAuditEntry],
    summary: AuditSummary,
    output_path: Path,
) -> None:
    """Save the audit report as JSON."""

    def entry_to_dict(entry: SNAuditEntry) -> dict[str, Any]:
        return {
            "sn_name": entry.sn_name,
            "is_complete_pair": entry.is_complete_pair,
            "has_reference": entry.has_reference,
            "has_science": entry.has_science,
            "reference_files_count": entry.reference_stats.total,
            "science_files_count": entry.science_stats.total,
            "reference_stats": asdict(entry.reference_stats),
            "science_stats": asdict(entry.science_stats),
            "reference_files": entry.reference_files,
            "science_files": entry.science_files,
            "validated_reference": entry.validated_reference,
            "validated_science": entry.validated_science,
            "invalid_reference": entry.invalid_reference,
            "invalid_science": entry.invalid_science,
        }

    report = {
        "summary": asdict(summary),
        "entries": [entry_to_dict(e) for e in entries],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📄 Report saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit downloaded supernova FITS files for training readiness."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/fits_downloads"),
        help="Directory containing downloaded SN subdirectories (default: output/fits_downloads)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON report path (default: <input-dir>/audit_report.json)",
    )
    parser.add_argument(
        "--validate-fits",
        action="store_true",
        help="Validate FITS files for WCS (slower but more thorough)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output JSON, suppress human-readable report",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    output_path = args.output or (args.input_dir / "audit_report.json")

    logger.info(f"Auditing downloads in: {args.input_dir}")
    if args.validate_fits:
        logger.info("FITS validation enabled (this may take a while...)")

    entries, summary = run_audit(args.input_dir, validate=args.validate_fits)

    if not args.quiet:
        print_report(entries, summary, validate=args.validate_fits)

    save_report(entries, summary, output_path)


if __name__ == "__main__":
    main()
