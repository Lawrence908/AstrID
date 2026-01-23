#!/usr/bin/env python3
"""
Per-SN pipeline runner that processes each supernova through all stages sequentially.

This approach:
- Processes one SN at a time: query → filter → download → organize → differencing
- Much lower memory footprint (only one SN's data in memory at a time)
- Easier to resume (just skip completed SNe)
- Better for incremental progress and debugging
- Can see results immediately

Usage:
    python scripts/run_pipeline_per_sn.py --config configs/galex_golden_era.yaml
    python scripts/run_pipeline_per_sn.py --config configs/galex_golden_era.yaml --resume
    python scripts/run_pipeline_per_sn.py --config configs/galex_golden_era.yaml --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add scripts directory to path for query_sn_fits_from_catalog
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from src.pipeline.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    """Load supernova catalog entries."""
    from query_sn_fits_from_catalog import parse_catalog_file
    
    entries = parse_catalog_file(catalog_path)
    logger.info(f"Loaded {len(entries)} entries from catalog")
    return entries


def filter_entries_by_year(
    entries: list[dict[str, Any]], min_year: int | None, max_year: int | None
) -> list[dict[str, Any]]:
    """Filter entries by discovery year."""
    if not min_year and not max_year:
        return entries
    
    filtered = []
    for entry in entries:
        if not entry.get("discovery_date"):
            continue
        year = entry["discovery_date"].year
        if min_year and year < min_year:
            continue
        if max_year and year > max_year:
            continue
        filtered.append(entry)
    
    year_range = []
    if min_year:
        year_range.append(f">= {min_year}")
    if max_year:
        year_range.append(f"<= {max_year}")
    logger.info(f"Filtered to {len(filtered)} entries with year {' and '.join(year_range)}")
    return filtered


def load_checkpoint(checkpoint_file: Path) -> set[str]:
    """Load completed SN names from checkpoint."""
    if not checkpoint_file.exists():
        return set()
    
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
            return set(data.get("completed", []))
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set()


def save_checkpoint(checkpoint_file: Path, completed: set[str]):
    """Save checkpoint of completed SNe."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump({"completed": list(completed)}, f, indent=2)


async def query_single_sn(
    entry: dict[str, Any],
    missions: list[str],
    days_before: int,
    days_after: int,
    radius_deg: float,
) -> dict[str, Any] | None:
    """Query MAST for a single supernova."""
    from query_sn_fits_from_catalog import query_mast_for_supernova
    
    try:
        result = await query_mast_for_supernova(
            entry,
            missions=missions,
            days_before=days_before,
            days_after=days_after,
            radius_deg=radius_deg,
            disable_time_filter=False,
        )
        return result
    except Exception as e:
        logger.error(f"Error querying {entry['sn_name']}: {e}")
        return None


def filter_same_mission_pairs(query_result: dict[str, Any]) -> dict[str, Any]:
    """Filter for same-mission pairs (in-memory version)."""
    ref_obs = query_result.get("reference_observations", [])
    sci_obs = query_result.get("science_observations", [])
    
    # Group by mission
    ref_by_mission: dict[str, list[dict]] = {}
    sci_by_mission: dict[str, list[dict]] = {}
    
    for obs in ref_obs:
        mission = obs.get("mission", "UNKNOWN")
        if mission not in ref_by_mission:
            ref_by_mission[mission] = []
        ref_by_mission[mission].append(obs)
    
    for obs in sci_obs:
        mission = obs.get("mission", "UNKNOWN")
        if mission not in sci_by_mission:
            sci_by_mission[mission] = []
        sci_by_mission[mission].append(obs)
    
    # Find missions with both reference and science
    common_missions = set(ref_by_mission.keys()) & set(sci_by_mission.keys())
    
    if not common_missions:
        return {
            "sn_name": query_result["sn_name"],
            "reference_observations": [],
            "science_observations": [],
            "same_mission_pairs": [],
        }
    
    # Use first common mission
    mission = sorted(common_missions)[0]
    
    return {
        "sn_name": query_result["sn_name"],
        "reference_observations": ref_by_mission[mission],
        "science_observations": sci_by_mission[mission],
        "same_mission_pairs": [mission],
    }


def download_single_sn(
    filtered_result: dict[str, Any],
    output_dir: Path,
    max_obs: int,
    max_products: int,
) -> bool:
    """Download FITS files for a single SN."""
    # Create temporary query file for this SN
    temp_query_file = output_dir / "temp_query.json"
    temp_query_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_query_file, "w") as f:
        json.dump([filtered_result], f, indent=2, default=str)
    
    # Run download script
    download_script = project_root / "scripts" / "download_sn_fits.py"
    cmd = [
        sys.executable,
        str(download_script),
        "--query-results",
        str(temp_query_file),
        "--output-dir",
        str(output_dir),
        "--max-obs",
        str(max_obs),
        "--max-products-per-obs",
        str(max_products),
        "--filter-has-both",
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root)
        # Clean up temp file
        temp_query_file.unlink()
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed for {filtered_result['sn_name']}: {e.stderr}")
        temp_query_file.unlink(missing_ok=True)
        return False


def organize_single_sn(
    sn_name: str,
    input_dir: Path,
    output_dir: Path,
    decompress: bool = False,
    symlink: bool = False,
) -> bool:
    """Organize training pairs for a single SN."""
    organize_script = project_root / "scripts" / "organize_training_pairs.py"
    
    # Check if SN has data
    sn_input_dir = input_dir / sn_name
    if not sn_input_dir.exists():
        logger.warning(f"No data directory for {sn_name}")
        return False
    
    cmd = [
        sys.executable,
        str(organize_script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--sn",
        sn_name,  # Process only this SN
    ]
    
    if decompress:
        cmd.append("--decompress")
    if symlink:
        cmd.append("--symlink")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Organize failed for {sn_name}: {e.stderr}")
        return False


def differencing_single_sn(
    sn_name: str,
    input_dir: Path,
    output_dir: Path,
    visualize: bool = False,
) -> bool:
    """Generate difference images for a single SN."""
    differencing_script = project_root / "scripts" / "generate_difference_images.py"
    
    # Check if SN has organized data
    sn_input_dir = input_dir / sn_name
    if not sn_input_dir.exists():
        logger.warning(f"No organized data for {sn_name}")
        return False
    
    cmd = [
        sys.executable,
        str(differencing_script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--sn",
        sn_name,  # Process only this SN
    ]
    
    if visualize:
        cmd.append("--visualize")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Differencing failed for {sn_name}: {e.stderr}")
        return False


async def process_single_sn(
    entry: dict[str, Any],
    config: PipelineConfig,
    checkpoint_file: Path,
    completed: set[str],
    decompress: bool = False,
    symlink: bool = False,
    visualize: bool = False,
) -> tuple[bool, str]:
    """Process a single SN through all pipeline stages.
    
    Returns:
        (success, reason) where success is True if completed, False otherwise,
        and reason is "completed", "no_data", "no_pairs", or "error"
    """
    sn_name = entry["sn_name"]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {sn_name}")
    logger.info(f"{'='*60}")
    
    # Stage 1: Query
    logger.info(f"[{sn_name}] Stage 1/5: Querying MAST...")
    query_result = await query_single_sn(
        entry,
        missions=config.query.missions,
        days_before=config.query.days_before,
        days_after=config.query.days_after,
        radius_deg=config.query.radius_deg,
    )
    
    if not query_result:
        logger.warning(f"[{sn_name}] Query failed, skipping")
        return (False, "error")
    
    ref_count = len(query_result.get("reference_observations", []))
    sci_count = len(query_result.get("science_observations", []))
    logger.info(f"[{sn_name}] Found {ref_count} reference, {sci_count} science observations")
    
    if ref_count == 0 or sci_count == 0:
        logger.warning(f"[{sn_name}] Missing reference or science observations, skipping")
        return (False, "no_data")
    
    # Stage 2: Filter
    logger.info(f"[{sn_name}] Stage 2/5: Filtering same-mission pairs...")
    filtered_result = filter_same_mission_pairs(query_result)
    
    if not filtered_result.get("same_mission_pairs"):
        logger.warning(f"[{sn_name}] No same-mission pairs found, skipping")
        return (False, "no_pairs")
    
    # Stage 3: Download
    logger.info(f"[{sn_name}] Stage 3/5: Downloading FITS files...")
    if not download_single_sn(
        filtered_result,
        config.output.fits_downloads,
        max_obs=config.download.max_obs_per_type,
        max_products=config.download.max_products_per_obs,
    ):
        logger.warning(f"[{sn_name}] Download failed, skipping")
        return (False, "error")
    
    # Stage 4: Organize
    logger.info(f"[{sn_name}] Stage 4/5: Organizing training pairs...")
    if not organize_single_sn(
        sn_name,
        config.output.fits_downloads,
        config.output.fits_training,
        decompress=decompress,
        symlink=symlink,
    ):
        logger.warning(f"[{sn_name}] Organize failed, skipping")
        return (False, "error")
    
    # Stage 5: Differencing
    logger.info(f"[{sn_name}] Stage 5/5: Generating difference images...")
    if not differencing_single_sn(
        sn_name,
        config.output.fits_training,
        config.output.difference_images,
        visualize=visualize,
    ):
        logger.warning(f"[{sn_name}] Differencing failed, skipping")
        return (False, "error")
    
    logger.info(f"[{sn_name}] ✅ Completed all stages!")
    
    # Save checkpoint
    completed.add(sn_name)
    save_checkpoint(checkpoint_file, completed)
    
    # Clean up memory
    del query_result, filtered_result
    gc.collect()
    
    return (True, "completed")


async def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline per-SN (one SN at a time through all stages)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("resources/sncat_compiled.txt"),
        help="Path to supernova catalog",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of SNe to process",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in catalog",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress .fits.gz files",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks to save disk space",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = PipelineConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration: {config.dataset_name}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Load catalog
    if not args.catalog.exists():
        logger.error(f"Catalog file not found: {args.catalog}")
        sys.exit(1)
    
    entries = load_catalog(args.catalog)
    
    # Filter by year
    entries = filter_entries_by_year(
        entries, config.query.min_year, config.query.max_year
    )
    
    # Apply start index and limit
    if args.start_index > 0:
        entries = entries[args.start_index:]
        logger.info(f"Starting from index {args.start_index}")
    
    if args.limit:
        entries = entries[:args.limit]
        logger.info(f"Limited to {args.limit} entries")
    
    # Load checkpoint
    checkpoint_file = config.output.checkpoint
    completed = load_checkpoint(checkpoint_file) if args.resume else set()
    logger.info(f"Found {len(completed)} already completed SNe")
    
    # Process each SN
    success_count = 0
    skip_count = 0
    no_data_count = 0
    no_pairs_count = 0
    error_count = 0
    
    for i, entry in enumerate(entries, 1):
        sn_name = entry["sn_name"]
        
        if sn_name in completed:
            logger.info(f"[{i}/{len(entries)}] Skipping {sn_name} (already completed)")
            skip_count += 1
            continue
        
        logger.info(f"\n[{i}/{len(entries)}] Processing {sn_name}")
        
        try:
            success, reason = await process_single_sn(
                entry,
                config,
                checkpoint_file,
                completed,
                decompress=args.decompress,
                symlink=args.symlink,
                visualize=args.visualize,
            )
            
            if success:
                success_count += 1
            elif reason == "no_data":
                no_data_count += 1
            elif reason == "no_pairs":
                no_pairs_count += 1
            else:  # error
                error_count += 1
        except Exception as e:
            logger.error(f"Error processing {sn_name}: {e}")
            error_count += 1
        
        # Periodic memory cleanup
        if i % 5 == 0:
            gc.collect()
            logger.debug(f"Ran garbage collection after {i} SNe")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total SNe: {len(entries)}")
    logger.info(f"✅ Completed: {success_count}")
    logger.info(f"⏭️  Already completed: {skip_count}")
    logger.info(f"📭 No data available: {no_data_count} (expected - not all SNe have observations)")
    logger.info(f"🔍 No same-mission pairs: {no_pairs_count} (expected - not all have matching missions)")
    logger.info(f"❌ Errors: {error_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
