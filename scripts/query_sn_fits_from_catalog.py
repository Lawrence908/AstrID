#!/usr/bin/env python3
"""
Query MAST for FITS files based on supernova catalog entries.

This script:
1. Parses the supernova catalog (sncat_latest_view.txt)
2. Extracts coordinates and discovery dates
3. Queries MAST for reference images (before discovery) and science images (after discovery)
4. Downloads FITS files for ZOGY differencing pipeline

Usage:
    python scripts/query_sn_fits_from_catalog.py --catalog resources/sncat_latest_view.txt --limit 10
"""

import argparse
import asyncio
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_ra_dec(ra_str: str, dec_str: str) -> tuple[float, float] | None:
    """Parse RA/Dec from catalog format (HH MM SS.SS, +/-DD MM SS.S) to degrees.
    
    Args:
        ra_str: Right ascension string like "15 02 48.00" or "00 42 43.00"
        dec_str: Declination string like "-41 57 00.0" or "+41 16 04.0"
    
    Returns:
        Tuple of (ra_deg, dec_deg) or None if parsing fails
    """
    try:
        # Parse RA: HH MM SS.SS -> degrees
        ra_parts = ra_str.strip().split()
        if len(ra_parts) != 3:
            return None
        
        ra_h = float(ra_parts[0])
        ra_m = float(ra_parts[1])
        ra_s = float(ra_parts[2])
        ra_deg = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0  # Convert hours to degrees
        
        # Parse Dec: +/-DD MM SS.S -> degrees
        dec_parts = dec_str.strip().split()
        if len(dec_parts) != 3:
            return None
        
        dec_sign = 1.0
        dec_d = dec_parts[0]
        if dec_d.startswith("-"):
            dec_sign = -1.0
            dec_d = dec_d[1:]
        elif dec_d.startswith("+"):
            dec_d = dec_d[1:]
        
        dec_d = float(dec_d)
        dec_m = float(dec_parts[1])
        dec_s = float(dec_parts[2])
        dec_deg = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)
        
        return (ra_deg, dec_deg)
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse coordinates: {ra_str}, {dec_str}: {e}")
        return None


def parse_date(date_str: str) -> datetime | None:
    """Parse date from catalog format (YYYY-MM-DD).
    
    Args:
        date_str: Date string like "1885-08-17" or empty string
    
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str or date_str.strip() == "":
        return None
    
    try:
        # Handle YYYY-MM-DD format
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        logger.debug(f"Failed to parse date: {date_str}")
        return None


def parse_catalog_line(line: str, headers: list[str]) -> dict[str, Any] | None:
    """Parse a single line from the catalog.
    
    Args:
        line: Pipe-delimited catalog line
        headers: List of column headers
    
    Returns:
        Dictionary with parsed values or None if parsing fails
    """
    # Split by pipe, handling quoted fields
    parts = [p.strip() for p in line.split("|")]
    
    if len(parts) != len(headers):
        return None
    
    # Create dictionary
    entry = dict(zip(headers, parts))
    
    # Extract key fields
    sn_name = entry.get("sn_name", "").strip()
    sn_ra_str = entry.get("sn_ra", "").strip()
    sn_dec_str = entry.get("sn_dec", "").strip()
    disc_date_str = entry.get("disc_date", "").strip()
    max_date_str = entry.get("max_date", "").strip()
    
    # Skip if missing essential data
    if not sn_name or not sn_ra_str or not sn_dec_str:
        return None
    
    # Parse coordinates
    coords = parse_ra_dec(sn_ra_str, sn_dec_str)
    if coords is None:
        return None
    
    ra_deg, dec_deg = coords
    
    # Parse dates
    disc_date = parse_date(disc_date_str)
    max_date = parse_date(max_date_str)
    
    # Use discovery date as primary, fallback to max_date
    discovery_date = disc_date if disc_date else max_date
    
    return {
        "sn_name": sn_name,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "discovery_date": discovery_date,
        "disc_date": disc_date,
        "max_date": max_date,
        "sn_type": entry.get("sn_type", "").strip(),
        "gal_name": entry.get("gal_name", "").strip(),
        "raw_entry": entry,
    }


def parse_catalog_file(catalog_path: Path) -> list[dict[str, Any]]:
    """Parse the supernova catalog file.
    
    Args:
        catalog_path: Path to the catalog file
    
    Returns:
        List of parsed supernova entries
    """
    logger.info(f"Parsing catalog file: {catalog_path}")
    
    with open(catalog_path, "r") as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        logger.error("Catalog file appears to be empty or malformed")
        return []
    
    # Parse header line
    header_line = lines[0].strip()
    headers = [h.strip() for h in header_line.split("|")]
    
    # Skip separator line (line 1)
    entries = []
    for i, line in enumerate(lines[2:], start=2):  # Start from line 2 (0-indexed)
        line = line.strip()
        if not line:
            continue
        
        entry = parse_catalog_line(line, headers)
        if entry:
            entries.append(entry)
        else:
            logger.debug(f"Skipping malformed line {i}: {line[:100]}")
    
    logger.info(f"Parsed {len(entries)} valid supernova entries from catalog")
    return entries


async def query_mast_for_supernova(
    sn_entry: dict[str, Any],
    missions: list[str] | None = None,
    days_before: int = 365,
    days_after: int = 365,
    radius_deg: float = 0.1,
) -> dict[str, Any]:
    """Query MAST for reference and science images for a supernova.
    
    Args:
        sn_entry: Parsed supernova entry from catalog
        missions: List of missions to query (e.g., ['HST', 'JWST', 'TESS'])
        days_before: Days before discovery to search for reference images
        days_after: Days after discovery to search for science images
        radius_deg: Search radius in degrees
    
    Returns:
        Dictionary with reference and science observations
    """
    from src.adapters.external.mast import MASTClient
    
    sn_name = sn_entry["sn_name"]
    ra_deg = sn_entry["ra_deg"]
    dec_deg = sn_entry["dec_deg"]
    discovery_date = sn_entry["discovery_date"]
    
    logger.info(f"Querying MAST for {sn_name} at ({ra_deg:.4f}, {dec_deg:.4f})")
    
    result = {
        "sn_name": sn_name,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "discovery_date": discovery_date.isoformat() if discovery_date else None,
        "reference_observations": [],
        "science_observations": [],
        "errors": [],
    }
    
    if discovery_date is None:
        result["errors"].append("No discovery date available")
        logger.warning(f"{sn_name}: No discovery date, skipping time-based queries")
        return result
    
    # Calculate time windows
    ref_start = discovery_date - timedelta(days=days_before)
    ref_end = discovery_date - timedelta(days=1)  # Day before discovery
    sci_start = discovery_date
    sci_end = discovery_date + timedelta(days=days_after)
    
    client = MASTClient()
    
    try:
        # Query for reference images (before discovery)
        logger.info(f"{sn_name}: Querying reference images ({ref_start.date()} to {ref_end.date()})")
        ref_obs = await client.query_observations_by_position(
            ra=ra_deg,
            dec=dec_deg,
            radius=radius_deg,
            missions=missions,
            start_time=ref_start,
            end_time=ref_end,
        )
        result["reference_observations"] = ref_obs
        logger.info(f"{sn_name}: Found {len(ref_obs)} reference observations")
        
        # Query for science images (after discovery)
        logger.info(f"{sn_name}: Querying science images ({sci_start.date()} to {sci_end.date()})")
        sci_obs = await client.query_observations_by_position(
            ra=ra_deg,
            dec=dec_deg,
            radius=radius_deg,
            missions=missions,
            start_time=sci_start,
            end_time=sci_end,
        )
        result["science_observations"] = sci_obs
        logger.info(f"{sn_name}: Found {len(sci_obs)} science observations")
        
    except Exception as e:
        error_msg = f"Error querying MAST for {sn_name}: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
    
    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query MAST for FITS files based on supernova catalog"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("resources/sncat_latest_view.txt"),
        help="Path to supernova catalog file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of supernovae to process (for testing)",
    )
    parser.add_argument(
        "--missions",
        nargs="+",
        default=["HST", "JWST", "TESS"],
        help="Space missions to query (default: HST JWST TESS)",
    )
    parser.add_argument(
        "--days-before",
        type=int,
        default=365,
        help="Days before discovery to search for reference images (default: 365)",
    )
    parser.add_argument(
        "--days-after",
        type=int,
        default=365,
        help="Days after discovery to search for science images (default: 365)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.1,
        help="Search radius in degrees (default: 0.1)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Minimum discovery year to process (e.g., 2000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sn_mast_queries.json"),
        help="Output JSON file for results",
    )
    
    args = parser.parse_args()
    
    # Parse catalog
    if not args.catalog.exists():
        logger.error(f"Catalog file not found: {args.catalog}")
        return
    
    entries = parse_catalog_file(args.catalog)
    
    # Filter by year if specified
    if args.min_year:
        filtered = []
        for entry in entries:
            if entry["discovery_date"] and entry["discovery_date"].year >= args.min_year:
                filtered.append(entry)
        entries = filtered
        logger.info(f"Filtered to {len(entries)} entries with year >= {args.min_year}")
    
    # Limit entries if specified
    if args.limit:
        entries = entries[: args.limit]
        logger.info(f"Limited to {args.limit} entries")
    
    # Query MAST for each supernova
    results = []
    for i, entry in enumerate(entries, 1):
        logger.info(f"\n[{i}/{len(entries)}] Processing {entry['sn_name']}")
        
        result = await query_mast_for_supernova(
            entry,
            missions=args.missions,
            days_before=args.days_before,
            days_after=args.days_after,
            radius_deg=args.radius,
        )
        results.append(result)
        
        # Brief summary
        ref_count = len(result["reference_observations"])
        sci_count = len(result["science_observations"])
        logger.info(
            f"{entry['sn_name']}: {ref_count} reference, {sci_count} science observations"
        )
    
    # Save results
    import json
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {args.output}")
    
    # Print summary
    total_ref = sum(len(r["reference_observations"]) for r in results)
    total_sci = sum(len(r["science_observations"]) for r in results)
    with_errors = sum(1 for r in results if r["errors"])
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total supernovae processed: {len(results)}")
    logger.info(f"Total reference observations: {total_ref}")
    logger.info(f"Total science observations: {total_sci}")
    logger.info(f"Entries with errors: {with_errors}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

