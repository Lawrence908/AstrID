#!/usr/bin/env python3
"""
Compile supernova catalog from multiple sources.

This script downloads supernova data from:
1. Open Supernova Catalog (OSC) - https://sne.space
2. TNS (Transient Name Server) - via API (optional, requires API key)
3. Existing catalog (sncat_latest_view.txt)

It merges all sources, deduplicates, and outputs in the same format.
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from astropy import units as u
from astropy.coordinates import Angle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Open Supernova Catalog API endpoint
OSC_API_BASE = "https://api.sne.space"
OSC_DOWNLOAD_URL = (
    "https://sne.space/astrocats/astrocats/supernovae/output/json/supernovae.json.zip"
)

# TNS API endpoint (requires API key)
TNS_API_BASE = "https://www.wis-tns.org/api/get"


def deg_to_hms(ra_deg: float) -> str:
    """Convert RA from degrees to HH MM SS.SS format."""
    angle = Angle(ra_deg, unit=u.deg)
    hms = angle.to_string(unit=u.hour, sep=" ", precision=2)
    return hms


def deg_to_dms(dec_deg: float) -> str:
    """Convert Dec from degrees to +/-DD MM SS.S format."""
    angle = Angle(dec_deg, unit=u.deg)
    dms = angle.to_string(unit=u.deg, sep=" ", precision=1, alwayssign=True)
    # Format: +DD MM SS.S or -DD MM SS.S
    return dms


def parse_osc_date(date_str: str | None) -> str | None:
    """Parse date from OSC format to YYYY-MM-DD."""
    if not date_str:
        return None

    # OSC dates can be in various formats
    # Try common formats
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str[:10], fmt[:8])
            return dt.strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            continue

    # Try to extract year-month-day pattern
    match = re.search(r"(\d{4})[-\/](\d{2})[-\/](\d{2})", date_str)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return None


def download_osc_catalog(output_dir: Path) -> list[dict[str, Any]]:
    """Download and parse Open Supernova Catalog.

    Returns:
        List of supernova entries
    """
    logger.info("Downloading Open Supernova Catalog...")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "osc_catalog.json"

    # Try to use cached version first
    if cache_file.exists():
        logger.info("Using cached OSC catalog...")
        try:
            with open(cache_file) as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} supernovae from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will download fresh")
            data = None
    else:
        data = None

    # Download if no cache or cache is old
    if data is None:
        # Try direct JSON download from GitHub (most reliable)
        try:
            logger.info("Downloading OSC catalog from GitHub...")
            # OSC data is available via GitHub
            with httpx.Client(timeout=300.0) as client:
                response = client.get(
                    "https://raw.githubusercontent.com/astrocatalogs/supernovae/master/output/json/supernovae.json"
                )
                if response.status_code == 200:
                    logger.info(
                        "Downloading large file, this may take a few minutes..."
                    )
                    data = response.json()
                    with open(cache_file, "w") as f:
                        json.dump(data, f)
                    logger.info(f"Downloaded {len(data)} supernovae from OSC")
                else:
                    logger.error(f"Download failed with status {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Failed to download OSC catalog: {e}")
            return []

    # Parse OSC data
    # OSC format: { "SN2011fe": { "ra": [{ "value": "14:03:05.81", ... }], ... }, ... }
    entries = []
    for sn_name, sn_data in data.items():
        try:
            # Extract coordinates - OSC uses lists of dicts with "value" keys
            ra_list = sn_data.get("ra", [])
            dec_list = sn_data.get("dec", [])

            if not ra_list or not dec_list:
                continue

            # Get first value
            ra_value = (
                ra_list[0].get("value", "")
                if isinstance(ra_list[0], dict)
                else str(ra_list[0])
            )
            dec_value = (
                dec_list[0].get("value", "")
                if isinstance(dec_list[0], dict)
                else str(dec_list[0])
            )

            if not ra_value or not dec_value:
                continue

            # Convert to degrees
            try:
                # Try parsing as degrees first
                ra_deg = float(ra_value)
                dec_deg = float(dec_value)
            except (ValueError, TypeError):
                # Try parsing as hourangle/deg strings
                try:
                    ra_angle = Angle(ra_value, unit=u.hourangle)
                    dec_angle = Angle(dec_value, unit=u.deg)
                    ra_deg = ra_angle.deg
                    dec_deg = dec_angle.deg
                except Exception as e:
                    logger.debug(f"Could not parse coordinates for {sn_name}: {e}")
                    continue

            # Extract supernova type
            claimedtype = sn_data.get("claimedtype", [])
            sn_type = ""
            if claimedtype:
                if isinstance(claimedtype, list):
                    if claimedtype and isinstance(claimedtype[0], dict):
                        sn_type = str(claimedtype[0].get("value", ""))
                    elif claimedtype:
                        sn_type = str(claimedtype[0])
                else:
                    sn_type = str(claimedtype)

            # Discovery date
            discovery = sn_data.get("discoverdate", [])
            disc_date = None
            if discovery:
                if isinstance(discovery, list) and discovery:
                    disc_date_str = (
                        discovery[0].get("value", "")
                        if isinstance(discovery[0], dict)
                        else str(discovery[0])
                    )
                else:
                    disc_date_str = str(discovery)
                disc_date = parse_osc_date(disc_date_str)

            # Maximum date
            maxdate = sn_data.get("maxdate", [])
            max_date = None
            if maxdate:
                if isinstance(maxdate, list) and maxdate:
                    max_date_str = (
                        maxdate[0].get("value", "")
                        if isinstance(maxdate[0], dict)
                        else str(maxdate[0])
                    )
                else:
                    max_date_str = str(maxdate)
                max_date = parse_osc_date(max_date_str)

            # Host galaxy
            host = sn_data.get("host", [])
            gal_name = ""
            if host:
                if isinstance(host, list) and host:
                    gal_name = (
                        host[0].get("value", "")
                        if isinstance(host[0], dict)
                        else str(host[0])
                    )
                else:
                    gal_name = str(host)

            entries.append(
                {
                    "sn_name": sn_name,
                    "sn_ra": deg_to_hms(ra_deg),
                    "sn_dec": deg_to_dms(dec_deg),
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "disc_date": disc_date,
                    "max_date": max_date,
                    "sn_type": sn_type,
                    "gal_name": gal_name,
                    "source": "OSC",
                }
            )
        except Exception as e:
            logger.debug(f"Error parsing OSC entry {sn_name}: {e}")
            continue

    logger.info(f"Parsed {len(entries)} entries from OSC")
    return entries


def download_tns_catalog(api_key: str | None, output_dir: Path) -> list[dict[str, Any]]:
    """Download supernova catalog from TNS (Transient Name Server).

    Args:
        api_key: TNS API key (optional)
        output_dir: Directory to cache data

    Returns:
        List of supernova entries
    """
    if not api_key:
        logger.info("Skipping TNS (no API key provided)")
        return []

    logger.info("Downloading TNS catalog...")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "tns_catalog.json"

    try:
        # TNS API requires authentication
        headers = {
            "User-Agent": "AstrID/1.0",
        }

        # Get list of objects (this is a simplified example - TNS API may require more setup)
        # Note: TNS API documentation should be consulted for exact endpoint format
        with httpx.Client(timeout=120.0) as client:
            response = client.get(
                f"{TNS_API_BASE}/object",
                headers=headers,
                params={"api_key": api_key, "format": "json"},
            )

            if response.status_code == 200:
                data = response.json()
                with open(cache_file, "w") as f:
                    json.dump(data, f)
            else:
                logger.warning(f"TNS API returned status {response.status_code}")
                if cache_file.exists():
                    logger.info("Using cached TNS catalog...")
                    with open(cache_file) as f:
                        data = json.load(f)
                else:
                    return []
    except Exception as e:
        logger.warning(f"TNS download failed: {e}")
        if cache_file.exists():
            logger.info("Using cached TNS catalog...")
            with open(cache_file) as f:
                data = json.load(f)
        else:
            return []

    # Parse TNS data
    entries = []
    # TNS format may vary - adjust parsing as needed
    # This is a placeholder structure
    for obj in data.get("data", []):
        try:
            ra = obj.get("ra")
            dec = obj.get("dec")
            if not ra or not dec:
                continue

            ra_deg = float(ra)
            dec_deg = float(dec)

            entries.append(
                {
                    "sn_name": obj.get("name", ""),
                    "sn_ra": deg_to_hms(ra_deg),
                    "sn_dec": deg_to_dms(dec_deg),
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "disc_date": obj.get("discovery_date"),
                    "max_date": None,
                    "sn_type": obj.get("type", ""),
                    "gal_name": obj.get("host_name", ""),
                    "source": "TNS",
                }
            )
        except Exception as e:
            logger.debug(f"Error parsing TNS entry: {e}")
            continue

    logger.info(f"Parsed {len(entries)} entries from TNS")
    return entries


def parse_existing_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    """Parse existing catalog file.

    Returns:
        List of supernova entries
    """
    logger.info(f"Parsing existing catalog: {catalog_path}")

    if not catalog_path.exists():
        logger.warning(f"Catalog file not found: {catalog_path}")
        return []

    entries = []
    with open(catalog_path) as f:
        lines = f.readlines()

    if len(lines) < 2:
        return []

    # Parse header
    header_line = lines[0].strip()
    headers = [h.strip() for h in header_line.split("|")]

    # Skip separator line
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != len(headers):
            continue

        entry = dict(zip(headers, parts, strict=False))

        sn_name = entry.get("sn_name", "").strip()
        sn_ra = entry.get("sn_ra", "").strip()
        sn_dec = entry.get("sn_dec", "").strip()

        if not sn_name or not sn_ra or not sn_dec:
            continue

        # Convert RA/Dec to degrees for deduplication
        try:
            ra_angle = Angle(sn_ra, unit=u.hourangle)
            dec_angle = Angle(sn_dec, unit=u.deg)
            ra_deg = ra_angle.deg
            dec_deg = dec_angle.deg
        except (ValueError, TypeError) as exc:
            logger.debug("Skipping %s due to invalid coordinates (%s)", sn_name, exc)
            continue

        entries.append(
            {
                "sn_name": sn_name,
                "sn_ra": sn_ra,
                "sn_dec": sn_dec,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "disc_date": entry.get("disc_date", "").strip() or None,
                "max_date": entry.get("max_date", "").strip() or None,
                "sn_type": entry.get("sn_type", "").strip(),
                "gal_name": entry.get("gal_name", "").strip(),
                "source": "existing",
            }
        )

    logger.info(f"Parsed {len(entries)} entries from existing catalog")
    return entries


def deduplicate_entries(
    entries: list[dict[str, Any]], coord_tolerance: float = 0.001
) -> list[dict[str, Any]]:
    """Deduplicate entries by coordinates and name.

    Args:
        entries: List of supernova entries
        coord_tolerance: Coordinate matching tolerance in degrees

    Returns:
        Deduplicated list
    """
    logger.info(f"Deduplicating {len(entries)} entries...")

    # Group by name first
    by_name = defaultdict(list)
    for entry in entries:
        by_name[entry["sn_name"].upper()].append(entry)

    # Group by coordinates
    by_coords = defaultdict(list)
    for entry in entries:
        # Round coordinates to tolerance
        ra_rounded = round(entry["ra_deg"] / coord_tolerance) * coord_tolerance
        dec_rounded = round(entry["dec_deg"] / coord_tolerance) * coord_tolerance
        key = (ra_rounded, dec_rounded)
        by_coords[key].append(entry)

    # Merge duplicates
    seen = set()
    merged = []

    for entry in entries:
        # Check by name
        name_key = entry["sn_name"].upper()
        if name_key in seen:
            continue

        # Check by coordinates
        ra_rounded = round(entry["ra_deg"] / coord_tolerance) * coord_tolerance
        dec_rounded = round(entry["dec_deg"] / coord_tolerance) * coord_tolerance
        coord_key = (ra_rounded, dec_rounded)

        # Find duplicates
        duplicates = []
        if name_key in by_name:
            duplicates.extend(by_name[name_key])
        if coord_key in by_coords:
            duplicates.extend(by_coords[coord_key])

        # Remove self
        duplicates = [d for d in duplicates if d != entry]

        if duplicates:
            # Merge: prefer existing catalog, then OSC, then TNS
            source_priority = {"existing": 0, "OSC": 1, "TNS": 2}
            all_entries = [entry] + duplicates
            all_entries.sort(key=lambda x: source_priority.get(x.get("source", ""), 99))

            # Merge data
            merged_entry = all_entries[0].copy()
            for dup in all_entries[1:]:
                # Fill in missing fields
                if not merged_entry.get("disc_date") and dup.get("disc_date"):
                    merged_entry["disc_date"] = dup["disc_date"]
                if not merged_entry.get("max_date") and dup.get("max_date"):
                    merged_entry["max_date"] = dup["max_date"]
                if not merged_entry.get("sn_type") and dup.get("sn_type"):
                    merged_entry["sn_type"] = dup["sn_type"]
                if not merged_entry.get("gal_name") and dup.get("gal_name"):
                    merged_entry["gal_name"] = dup["gal_name"]

            merged.append(merged_entry)
            seen.add(name_key)
            seen.add(coord_key)
        else:
            merged.append(entry)
            seen.add(name_key)
            seen.add(coord_key)

    logger.info(f"Deduplicated to {len(merged)} entries")
    return merged


def format_catalog_entry(entry: dict[str, Any]) -> str:
    """Format entry as pipe-delimited line matching original format."""
    # Create empty fields matching original format
    fields = [
        entry.get("sn_name", ""),
        entry.get("gal_name", ""),
        "",  # sn_gal_uncert
        "",  # gal_ra
        "",  # gal_dec
        "",  # gal_mag
        "",  # gal_mband
        "",  # gal_mag_ref
        "",  # gal_posang
        "",  # gal_inclination
        "",  # redshift
        "",  # gal_type
        "",  # gal_logab
        "",  # gal_log10d
        "",  # gal_type_code
        "",  # sn_offset_ew
        "",  # sn_offset_ns
        "",  # sn_mag
        "",  # sn_mflag
        "",  # sn_mband
        "",  # sn_qmag
        "",  # sn_optdisc
        entry.get("max_date", "") or "",
        entry.get("disc_date", "") or "",
        entry.get("sn_type", ""),
        entry.get("sn_ra", ""),
        entry.get("sn_dec", ""),
        "",  # sn_uncert
        "",  # surveys
        "",  # discoverer
        "",  # sn_comment
        "",  # gal_comment
        "",  # mod_date_arr
    ]

    return " | ".join(f"{f}" for f in fields)


def write_catalog(entries: list[dict[str, Any]], output_path: Path):
    """Write catalog in original format."""
    logger.info(f"Writing {len(entries)} entries to {output_path}")

    # Header matching original format
    header = "        sn_name        |            gal_name            | sn_gal_uncert |    gal_ra    |   gal_dec   | gal_mag | gal_mband | gal_mag_ref | gal_posang | gal_inclination | redshift  | gal_type | gal_logab | gal_log10d | gal_type_code | sn_offset_ew | sn_offset_ns | sn_mag | sn_mflag | sn_mband |  sn_qmag  | sn_optdisc |  max_date  | disc_date  | sn_type  |    sn_ra     |   sn_dec    |    sn_uncert    |   surveys   |                                                                                                                                                                                         discoverer                                                                                                                                                                                         |                                                                                                                                                                                                                                                                                                                       sn_comment                                                                                                                                                                                                                                                                                                                        |                                                                                                                                            gal_comment                                                                                                                                            |                        mod_date_arr                         "
    separator = "-----------------------+--------------------------------+---------------+--------------+-------------+---------+-----------+-------------+------------+-----------------+-----------+----------+-----------+------------+---------------+--------------+--------------+--------+----------+----------+-----------+------------+------------+------------+----------+--------------+-------------+-----------------+-------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------"

    with open(output_path, "w") as f:
        f.write(header + "\n")
        f.write(separator + "\n")

        # Sort by discovery date (most recent first, then by name)
        sorted_entries = sorted(
            entries,
            key=lambda x: (
                x.get("disc_date") or x.get("max_date") or "0000-01-01",
                x.get("sn_name", ""),
            ),
            reverse=True,
        )

        for entry in sorted_entries:
            line = format_catalog_entry(entry)
            f.write(line + "\n")

    logger.info(f"Catalog written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile supernova catalog from multiple sources"
    )
    parser.add_argument(
        "--existing-catalog",
        type=Path,
        default=Path("resources/sncat_latest_view.txt"),
        help="Path to existing catalog file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/sncat_compiled.txt"),
        help="Output catalog path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/catalog_cache"),
        help="Directory to cache downloaded catalogs",
    )
    parser.add_argument(
        "--tns-api-key",
        type=str,
        default=None,
        help="TNS API key (optional)",
    )
    parser.add_argument(
        "--skip-osc",
        action="store_true",
        help="Skip Open Supernova Catalog download",
    )
    parser.add_argument(
        "--skip-tns",
        action="store_true",
        help="Skip TNS download",
    )
    parser.add_argument(
        "--coord-tolerance",
        type=float,
        default=0.001,
        help="Coordinate matching tolerance in degrees for deduplication",
    )

    args = parser.parse_args()

    all_entries = []

    # Parse existing catalog
    if args.existing_catalog.exists():
        existing_entries = parse_existing_catalog(args.existing_catalog)
        all_entries.extend(existing_entries)
        logger.info(f"Loaded {len(existing_entries)} entries from existing catalog")

    # Download OSC
    if not args.skip_osc:
        osc_entries = download_osc_catalog(args.cache_dir)
        all_entries.extend(osc_entries)
        logger.info(f"Added {len(osc_entries)} entries from OSC")

    # Download TNS
    if not args.skip_tns and args.tns_api_key:
        tns_entries = download_tns_catalog(args.tns_api_key, args.cache_dir)
        all_entries.extend(tns_entries)
        logger.info(f"Added {len(tns_entries)} entries from TNS")

    # Deduplicate
    merged_entries = deduplicate_entries(all_entries, args.coord_tolerance)

    # Write output
    write_catalog(merged_entries, args.output)

    logger.info(f"Compilation complete: {len(merged_entries)} unique supernovae")
    logger.info(
        f"  - From existing: {sum(1 for e in merged_entries if e.get('source') == 'existing')}"
    )
    logger.info(
        f"  - From OSC: {sum(1 for e in merged_entries if e.get('source') == 'OSC')}"
    )
    logger.info(
        f"  - From TNS: {sum(1 for e in merged_entries if e.get('source') == 'TNS')}"
    )


if __name__ == "__main__":
    main()
