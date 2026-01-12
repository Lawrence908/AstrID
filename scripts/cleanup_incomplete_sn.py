#!/usr/bin/env python3
"""
Analyze and optionally delete incomplete SN folders (missing reference or science).

Usage:
    # Analyze only (dry run)
    python scripts/cleanup_incomplete_sn.py --input-dir output/fits_downloads

    # Delete incomplete folders
    python scripts/cleanup_incomplete_sn.py --input-dir output/fits_downloads --delete
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_incomplete_sn(input_dir: Path) -> dict:
    """Analyze which SNe are incomplete."""
    stats = {
        "complete": [],
        "ref_only": [],
        "sci_only": [],
        "none": [],
        "ref_file_counts": defaultdict(int),
        "sci_file_counts": defaultdict(int),
        "total_size": defaultdict(int),
    }

    for sn_dir in sorted(input_dir.iterdir()):
        if not sn_dir.is_dir():
            continue
        sn_name = sn_dir.name
        ref_dir = sn_dir / "reference"
        sci_dir = sn_dir / "science"

        ref_files = (
            list(ref_dir.rglob("*.fits*")) + list(ref_dir.rglob("*.img"))
            if ref_dir.exists()
            else []
        )
        sci_files = (
            list(sci_dir.rglob("*.fits*")) + list(sci_dir.rglob("*.img"))
            if sci_dir.exists()
            else []
        )

        ref_count = len(ref_files)
        sci_count = len(sci_files)

        # Calculate size
        total_size = 0
        for f in ref_files + sci_files:
            try:
                total_size += f.stat().st_size
            except Exception:
                pass

        if ref_count > 0 and sci_count > 0:
            stats["complete"].append(sn_name)
        elif ref_count > 0:
            stats["ref_only"].append(sn_name)
            stats["ref_file_counts"][ref_count] += 1
            stats["total_size"][sn_name] = total_size
        elif sci_count > 0:
            stats["sci_only"].append(sn_name)
            stats["sci_file_counts"][sci_count] += 1
            stats["total_size"][sn_name] = total_size
        else:
            stats["none"].append(sn_name)
            stats["total_size"][sn_name] = total_size

    return stats


def format_size(bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and optionally delete incomplete SN folders"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/fits_downloads"),
        help="Directory containing SN subdirectories",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete incomplete folders (default: dry run)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    logger.info(f"Analyzing: {args.input_dir}")
    stats = analyze_incomplete_sn(args.input_dir)

    print("\n" + "=" * 60)
    print("INCOMPLETE SN ANALYSIS")
    print("=" * 60)
    print(f"\n✅ Complete pairs: {len(stats['complete'])}")
    print(f"📁 Reference only: {len(stats['ref_only'])}")
    print(f"📁 Science only: {len(stats['sci_only'])}")
    print(f"📁 No structure: {len(stats['none'])}")
    incomplete_total = (
        len(stats["ref_only"]) + len(stats["sci_only"]) + len(stats["none"])
    )
    print(f"\n❌ Total incomplete: {incomplete_total}")

    if incomplete_total > 0:
        total_size = sum(stats["total_size"].values())
        print(f"💾 Total size of incomplete folders: {format_size(total_size)}")

        if stats["ref_only"]:
            print(f"\n📁 Reference-only SNe ({len(stats['ref_only'])}):")
            print(f"   First 10: {stats['ref_only'][:10]}")

        if stats["sci_only"]:
            print(f"\n📁 Science-only SNe ({len(stats['sci_only'])}):")
            print(f"   First 10: {stats['sci_only'][:10]}")

        if stats["none"]:
            print(f"\n📁 No structure SNe ({len(stats['none'])}):")
            print(f"   First 10: {stats['none'][:10]}")

        if args.delete:
            print("\n" + "=" * 60)
            print("DELETING INCOMPLETE FOLDERS")
            print("=" * 60)
            deleted = 0
            freed_space = 0

            for sn_name in stats["ref_only"] + stats["sci_only"] + stats["none"]:
                sn_dir = args.input_dir / sn_name
                if sn_dir.exists():
                    try:
                        # Calculate size before deletion
                        size = stats["total_size"].get(sn_name, 0)
                        shutil.rmtree(sn_dir)
                        deleted += 1
                        freed_space += size
                        logger.info(f"Deleted: {sn_name} ({format_size(size)})")
                    except Exception as exc:
                        logger.error(f"Failed to delete {sn_name}: {exc}")

            print(f"\n✅ Deleted {deleted} incomplete folders")
            print(f"💾 Freed space: {format_size(freed_space)}")
        else:
            print("\n⚠️  DRY RUN - No folders deleted")
            print("   Use --delete to actually delete incomplete folders")


if __name__ == "__main__":
    main()
