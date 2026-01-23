#!/usr/bin/env python3
"""
Chunked version of query_sn_fits_from_catalog.py that processes in batches
and saves incrementally to avoid memory issues.

This script:
1. Processes supernovae in chunks (e.g., 250 at a time)
2. Saves each chunk to a separate file
3. Can resume from a checkpoint
4. Combines all chunks at the end
"""

import argparse
import asyncio
import gc
import json
import logging

# Import from the original script
import sys
from pathlib import Path
from typing import Any

# Add project root to path for src imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from query_sn_fits_from_catalog import (  # noqa: E402
    parse_catalog_file,
    query_mast_for_supernova,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_file: Path, reset: bool = False) -> set[str]:
    """Load already processed supernova names from checkpoint."""
    if reset and checkpoint_file.exists():
        logger.info(f"Resetting checkpoint: {checkpoint_file}")
        checkpoint_file.unlink()
        return set()

    if not checkpoint_file.exists():
        return set()

    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
            return set(data.get("processed", []))
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set()


def save_checkpoint(checkpoint_file: Path, processed: set[str]):
    """Save checkpoint of processed supernovae."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump({"processed": list(processed)}, f, indent=2)


def save_chunk(chunk_file: Path, results: list[dict[str, Any]]):
    """Save a chunk of results to a file."""
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunk_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved chunk: {chunk_file} ({len(results)} entries)")


def combine_chunks(chunk_files: list[Path], output_file: Path):
    """Combine all chunk files into a single output file."""
    logger.info(f"Combining {len(chunk_files)} chunks into {output_file}")

    all_results = []
    for chunk_file in sorted(chunk_files):
        if not chunk_file.exists():
            logger.warning(f"Chunk file not found: {chunk_file}")
            continue

        try:
            with open(chunk_file) as f:
                chunk_data = json.load(f)
                all_results.extend(chunk_data)
                logger.info(
                    f"  Loaded {len(chunk_data)} entries from {chunk_file.name}"
                )
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_file}: {e}")
            continue

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Combined {len(all_results)} total entries into {output_file}")
    return all_results


async def process_chunk(
    entries: list[dict[str, Any]],
    chunk_num: int,
    chunk_size: int,
    checkpoint_file: Path,
    chunk_dir: Path,
    missions_to_query: list[str] | None,
    days_before: int,
    days_after: int,
    radius_deg: float,
    disable_time_filter: bool,
    reset_checkpoint: bool = False,
) -> Path:
    """Process a single chunk of entries."""
    start_idx = chunk_num * chunk_size
    end_idx = min(start_idx + chunk_size, len(entries))
    chunk_entries = entries[start_idx:end_idx]

    logger.info(
        f"\n{'='*60}\n"
        f"Processing chunk {chunk_num + 1}: entries {start_idx + 1}-{end_idx} "
        f"({len(chunk_entries)} supernovae)\n"
        f"{'='*60}"
    )

    # Check if chunk file already exists with data
    chunk_file = chunk_dir / f"chunk_{chunk_num:03d}.json"
    existing_chunk_data = []
    if chunk_file.exists() and not reset_checkpoint:
        try:
            with open(chunk_file) as f:
                existing_chunk_data = json.load(f)
            if existing_chunk_data and len(existing_chunk_data) > 0:
                logger.info(
                    f"Chunk {chunk_num + 1} already has {len(existing_chunk_data)} entries, skipping"
                )
                return chunk_file
        except Exception as e:
            logger.warning(f"Could not read existing chunk file: {e}")

    # Load checkpoint to skip already processed
    processed = load_checkpoint(checkpoint_file, reset=reset_checkpoint)

    results = []
    for i, entry in enumerate(chunk_entries, start=start_idx + 1):
        sn_name = entry["sn_name"]

        # Skip if already processed (only if we have valid chunk data)
        if (
            sn_name in processed
            and existing_chunk_data
            and len(existing_chunk_data) > 0
        ):
            logger.info(f"[{i}/{len(entries)}] Skipping {sn_name} (already processed)")
            continue

        logger.info(f"\n[{i}/{len(entries)}] Processing {sn_name}")

        try:
            result = await query_mast_for_supernova(
                entry,
                missions=missions_to_query,
                days_before=days_before,
                days_after=days_after,
                radius_deg=radius_deg,
                disable_time_filter=disable_time_filter,
            )
            results.append(result)
            processed.add(sn_name)

            # Save checkpoint after each entry
            save_checkpoint(checkpoint_file, processed)

            # Brief summary
            ref_count = len(result["reference_observations"])
            sci_count = len(result["science_observations"])
            logger.info(
                f"{sn_name}: {ref_count} reference, {sci_count} science observations"
            )
            
            # Clean up memory after each query to prevent accumulation
            if i % 10 == 0:  # Every 10 supernovae
                gc.collect()
                logger.debug(f"Ran garbage collection after {i} supernovae")
        except Exception as e:
            logger.error(f"Error processing {sn_name}: {e}")
            # Still save a result entry with error
            results.append(
                {
                    "sn_name": sn_name,
                    "ra_deg": entry.get("ra_deg"),
                    "dec_deg": entry.get("dec_deg"),
                    "discovery_date": str(entry.get("discovery_date", "")),
                    "reference_observations": [],
                    "science_observations": [],
                    "errors": [str(e)],
                }
            )
            processed.add(sn_name)
            save_checkpoint(checkpoint_file, processed)

    # Save chunk
    chunk_file = chunk_dir / f"chunk_{chunk_num:03d}.json"
    save_chunk(chunk_file, results)

    return chunk_file


async def main():
    parser = argparse.ArgumentParser(
        description="Query MAST for supernova FITS files (chunked version)"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("resources/sncat_compiled.txt"),
        help="Path to supernova catalog file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sn_queries_chunked.json"),
        help="Final combined output JSON file",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Number of supernovae per chunk (default: 250)",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("output/chunks"),
        help="Directory to save chunk files (default: output/chunks)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/checkpoint.json"),
        help="Checkpoint file to track progress (default: output/checkpoint.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of supernovae to process",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip first N entries before processing (default: 0)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Minimum discovery year to process",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Maximum discovery year to process",
    )
    parser.add_argument(
        "--missions",
        nargs="+",
        default=["TESS", "GALEX", "PS1", "SWIFT"],
        help="Space telescope missions to query",
    )
    parser.add_argument(
        "--days-before",
        type=int,
        default=1095,
        help="Days before discovery to search for reference images",
    )
    parser.add_argument(
        "--days-after",
        type=int,
        default=730,
        help="Days after discovery to search for science images",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.1,
        help="Search radius in degrees",
    )
    parser.add_argument(
        "--no-time-filter",
        action="store_true",
        help="Disable time-based filtering",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only combine existing chunks, don't process new ones",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear checkpoint and start fresh (ignores existing checkpoint)",
    )

    args = parser.parse_args()

    # Parse catalog
    if not args.catalog.exists():
        logger.error(f"Catalog file not found: {args.catalog}")
        return

    entries = parse_catalog_file(args.catalog)

    # Filter by year if specified
    if args.min_year or args.max_year:
        filtered = []
        for entry in entries:
            if not entry["discovery_date"]:
                continue
            year = entry["discovery_date"].year
            if args.min_year and year < args.min_year:
                continue
            if args.max_year and year > args.max_year:
                continue
            filtered.append(entry)
        entries = filtered
        year_range = []
        if args.min_year:
            year_range.append(f">= {args.min_year}")
        if args.max_year:
            year_range.append(f"<= {args.max_year}")
        logger.info(f"Filtered to {len(entries)} entries with year {' and '.join(year_range)}")

    # Skip entries if start-index specified
    if args.start_index > 0:
        if args.start_index >= len(entries):
            logger.error(
                f"Start index {args.start_index} exceeds total entries {len(entries)}"
            )
            return
        entries = entries[args.start_index :]
        logger.info(
            f"Skipped first {args.start_index} entries, processing {len(entries)} remaining"
        )

    # Limit entries if specified
    if args.limit:
        entries = entries[: args.limit]
        logger.info(f"Limited to {args.limit} entries")

    # Handle mission filtering
    missions_to_query = args.missions
    if args.missions and args.missions[0].upper() == "NONE":
        missions_to_query = None
        logger.info("Mission filtering disabled")

    # Calculate chunks
    chunk_size = args.chunk_size
    num_chunks = (len(entries) + chunk_size - 1) // chunk_size

    logger.info(f"\n{'='*60}")
    logger.info("CHUNKED PROCESSING SETUP")
    logger.info(f"{'='*60}")
    logger.info(f"Total entries: {len(entries)}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Number of chunks: {num_chunks}")
    logger.info(f"Chunk directory: {args.chunk_dir}")
    logger.info(f"Checkpoint file: {args.checkpoint}")
    logger.info(f"{'='*60}\n")

    if args.combine_only:
        # Just combine existing chunks
        chunk_files = sorted(args.chunk_dir.glob("chunk_*.json"))
        if not chunk_files:
            logger.error(f"No chunk files found in {args.chunk_dir}")
            return
        combine_chunks(chunk_files, args.output)
        return

    # Reset checkpoint if requested
    if args.reset_checkpoint:
        if args.checkpoint.exists():
            logger.info(f"Clearing checkpoint: {args.checkpoint}")
            args.checkpoint.unlink()
        # Also clear empty chunk files
        for chunk_file in args.chunk_dir.glob("chunk_*.json"):
            try:
                with open(chunk_file) as f:
                    data = json.load(f)
                    if not data or len(data) == 0:
                        logger.info(f"Removing empty chunk: {chunk_file}")
                        chunk_file.unlink()
            except Exception:
                pass

    # Process chunks
    chunk_files = []
    for chunk_num in range(num_chunks):
        chunk_file = await process_chunk(
            entries,
            chunk_num,
            chunk_size,
            args.checkpoint,
            args.chunk_dir,
            missions_to_query,
            args.days_before,
            args.days_after,
            args.radius,
            args.no_time_filter,
            args.reset_checkpoint,
        )
        chunk_files.append(chunk_file)

    # Combine all chunks
    all_results = combine_chunks(chunk_files, args.output)

    # Print summary
    total_ref = sum(len(r["reference_observations"]) for r in all_results)
    total_sci = sum(len(r["science_observations"]) for r in all_results)
    with_errors = sum(1 for r in all_results if r.get("errors"))
    viable = sum(
        1
        for r in all_results
        if r.get("reference_observations") and r.get("science_observations")
    )

    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total supernovae processed: {len(all_results)}")
    if len(all_results) > 0:
        logger.info(
            f"✅ Viable (both ref & sci): {viable} ({viable/len(all_results)*100:.1f}%)"
        )
    else:
        logger.warning(
            "⚠️  No results found! Check checkpoint or reset with --reset-checkpoint"
        )
    logger.info(f"Total reference observations: {total_ref}")
    logger.info(f"Total science observations: {total_sci}")
    logger.info(f"Entries with errors: {with_errors}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
