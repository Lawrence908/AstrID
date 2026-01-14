#!/usr/bin/env python3
"""
Analyze download failures to understand why SNe didn't get complete pairs.

This script examines the fits_downloads directory and compares it with the
query results to identify why downloads failed or resulted in incomplete pairs.

Usage:
    python3 scripts/analyze_download_failures.py \
        --download-dir output/fits_downloads \
        --query-results output/sn_queries_same_mission.json
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def count_fits_files(directory: Path) -> int:
    """Count FITS/IMG files in a directory."""
    if not directory.exists():
        return 0
    count = 0
    for ext in ["*.fits", "*.fits.gz", "*.img", "*.img.gz"]:
        count += len(list(directory.rglob(ext)))
    return count


def analyze_sn_download(
    sn_name: str,
    download_dir: Path,
    query_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a single SN's download status."""
    sn_dir = download_dir / sn_name
    
    result = {
        "sn_name": sn_name,
        "exists": sn_dir.exists(),
        "ref_files": 0,
        "sci_files": 0,
        "is_complete": False,
        "status": "unknown",
        "missions_downloaded": set(),
        "missions_expected": set(),
        "mission_mismatch": False,
    }
    
    if not sn_dir.exists():
        result["status"] = "not_downloaded"
        return result
    
    # Count files
    ref_dir = sn_dir / "reference"
    sci_dir = sn_dir / "science"
    result["ref_files"] = count_fits_files(ref_dir)
    result["sci_files"] = count_fits_files(sci_dir)
    result["is_complete"] = result["ref_files"] > 0 and result["sci_files"] > 0
    
    # Detect missions from directory structure
    if ref_dir.exists():
        for mission_dir in ref_dir.glob("mastDownload/*"):
            if mission_dir.is_dir():
                result["missions_downloaded"].add(mission_dir.name)
    if sci_dir.exists():
        for mission_dir in sci_dir.glob("mastDownload/*"):
            if mission_dir.is_dir():
                result["missions_downloaded"].add(mission_dir.name)
    
    # Get expected missions from query
    if query_entry:
        ref_missions = {
            obs.get("mission", "").upper()
            for obs in query_entry.get("reference_observations", [])
            if obs.get("mission")
        }
        sci_missions = {
            obs.get("mission", "").upper()
            for obs in query_entry.get("science_observations", [])
            if obs.get("mission")
        }
        result["missions_expected"] = ref_missions & sci_missions
        
        # Check for mission mismatch
        if result["ref_files"] == 0 and result["sci_files"] > 0:
            # Science-only: likely ref missions don't match sci missions
            result["status"] = "science_only"
            result["mission_mismatch"] = not bool(ref_missions & sci_missions)
        elif result["ref_files"] > 0 and result["sci_files"] == 0:
            result["status"] = "reference_only"
            result["mission_mismatch"] = not bool(ref_missions & sci_missions)
        elif result["is_complete"]:
            result["status"] = "complete"
        else:
            result["status"] = "empty"
    else:
        if result["ref_files"] == 0 and result["sci_files"] > 0:
            result["status"] = "science_only"
        elif result["ref_files"] > 0 and result["sci_files"] == 0:
            result["status"] = "reference_only"
        elif result["is_complete"]:
            result["status"] = "complete"
        else:
            result["status"] = "empty"
    
    return result


def analyze_downloads(
    download_dir: Path,
    query_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze all downloads and categorize failures."""
    
    # Build query lookup
    query_lookup = {}
    if query_results:
        for entry in query_results:
            sn_name = entry.get("sn_name")
            if sn_name:
                query_lookup[sn_name] = entry
    
    # Get all SN directories
    sn_dirs = [d for d in download_dir.iterdir() if d.is_dir() and d.name.startswith("20")]
    
    results = []
    stats = {
        "total_sne": len(sn_dirs),
        "complete": 0,
        "science_only": 0,
        "reference_only": 0,
        "empty": 0,
        "not_downloaded": 0,
        "mission_mismatch": 0,
        "mission_mismatch_examples": [],
    }
    
    for sn_dir in sorted(sn_dirs):
        sn_name = sn_dir.name
        query_entry = query_lookup.get(sn_name)
        
        result = analyze_sn_download(sn_name, download_dir, query_entry)
        results.append(result)
        
        # Update stats
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        
        if result["mission_mismatch"]:
            stats["mission_mismatch"] += 1
            if len(stats["mission_mismatch_examples"]) < 10:
                stats["mission_mismatch_examples"].append({
                    "sn_name": sn_name,
                    "ref_files": result["ref_files"],
                    "sci_files": result["sci_files"],
                    "missions_expected": sorted(result["missions_expected"]),
                    "missions_downloaded": sorted(result["missions_downloaded"]),
                })
    
    return {
        "results": results,
        "stats": stats,
    }


def print_analysis(analysis: dict[str, Any]):
    """Print analysis results."""
    stats = analysis["stats"]
    
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Total SNe downloaded: {stats['total_sne']}")
    logger.info("")
    logger.info("Status breakdown:")
    logger.info(f"  ✅ Complete (ref + sci): {stats['complete']} ({100*stats['complete']/stats['total_sne']:.1f}%)")
    logger.info(f"  ⚠️  Science only: {stats['science_only']} ({100*stats['science_only']/stats['total_sne']:.1f}%)")
    logger.info(f"  ⚠️  Reference only: {stats['reference_only']} ({100*stats['reference_only']/stats['total_sne']:.1f}%)")
    logger.info(f"  ❌ Empty: {stats['empty']} ({100*stats['empty']/stats['total_sne']:.1f}%)")
    logger.info("")
    logger.info(f"Mission mismatches detected: {stats['mission_mismatch']}")
    
    if stats["mission_mismatch_examples"]:
        logger.info("\nMission mismatch examples:")
        for ex in stats["mission_mismatch_examples"][:5]:
            logger.info(f"  {ex['sn_name']}:")
            logger.info(f"    Files: {ex['ref_files']} ref, {ex['sci_files']} sci")
            logger.info(f"    Expected missions: {ex['missions_expected']}")
            logger.info(f"    Downloaded missions: {ex['missions_downloaded']}")
    
    logger.info("=" * 70)
    
    # Calculate improvement potential
    incomplete = stats['science_only'] + stats['reference_only'] + stats['empty']
    mission_mismatch_pct = 100 * stats['mission_mismatch'] / incomplete if incomplete > 0 else 0
    
    logger.info("\n📊 IMPROVEMENT POTENTIAL")
    logger.info("-" * 70)
    logger.info(f"Current success rate: {stats['complete']}/{stats['total_sne']} = {100*stats['complete']/stats['total_sne']:.1f}%")
    logger.info(f"Incomplete downloads: {incomplete}")
    logger.info(f"  Caused by mission mismatch: {stats['mission_mismatch']} ({mission_mismatch_pct:.1f}%)")
    logger.info(f"  Other causes: {incomplete - stats['mission_mismatch']}")
    logger.info("")
    logger.info(f"Expected after filtering: {stats['complete'] + stats['mission_mismatch']}/{stats['total_sne']} = {100*(stats['complete'] + stats['mission_mismatch'])/stats['total_sne']:.1f}%")
    logger.info(f"Improvement: +{stats['mission_mismatch']} complete pairs (+{100*stats['mission_mismatch']/stats['complete']:.0f}%)")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze download failures and mission mismatches"
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("output/fits_downloads"),
        help="Directory with downloaded FITS files",
    )
    parser.add_argument(
        "--query-results",
        type=Path,
        default=None,
        help="Query results JSON file (for mission mismatch detection)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save detailed analysis to JSON file",
    )
    
    args = parser.parse_args()
    
    # Load query results if provided
    query_results = None
    if args.query_results and args.query_results.exists():
        logger.info(f"Loading query results from {args.query_results}")
        with open(args.query_results) as f:
            query_results = json.load(f)
        logger.info(f"Loaded {len(query_results)} query entries")
    
    # Analyze downloads
    logger.info(f"Analyzing downloads in {args.download_dir}")
    analysis = analyze_downloads(args.download_dir, query_results)
    
    # Print results
    print_analysis(analysis)
    
    # Save detailed results if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            # Convert sets to lists for JSON serialization
            for result in analysis["results"]:
                result["missions_downloaded"] = sorted(result["missions_downloaded"])
                result["missions_expected"] = sorted(result["missions_expected"])
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"\n✅ Detailed analysis saved to {args.output}")


if __name__ == "__main__":
    main()
