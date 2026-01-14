#!/usr/bin/env python3
"""
Filter query results to keep only observations from matching missions.

This script solves the problem where identify_same_mission_pairs.py identifies
SNe with overlapping missions but doesn't filter the observation lists. This
leads to downloads of incompatible data (e.g., GALEX ref + SWIFT sci).

Usage:
    python3 scripts/filter_same_mission_observations.py \
        --input output/sn_queries_same_mission.json \
        --output output/sn_queries_filtered.json \
        --require-both-missions

    # Or create separate entries per mission for multi-mission SNe
    python3 scripts/filter_same_mission_observations.py \
        --input output/sn_queries_same_mission.json \
        --output output/sn_queries_filtered.json \
        --split-by-mission
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


def filter_observations_by_mission(
    entry: dict[str, Any],
    split_by_mission: bool = False,
) -> list[dict[str, Any]]:
    """
    Filter observations to keep only matching missions.

    Args:
        entry: Query result entry for a single SN
        split_by_mission: If True, create separate entries per mission

    Returns:
        List of filtered entries (1 entry if not splitting, N entries if splitting by mission)
    """
    sn_name = entry.get("sn_name", "UNKNOWN")
    ref_obs = entry.get("reference_observations", [])
    sci_obs = entry.get("science_observations", [])

    if not ref_obs or not sci_obs:
        return []

    # Group observations by mission
    ref_by_mission = defaultdict(list)
    sci_by_mission = defaultdict(list)

    for obs in ref_obs:
        mission = obs.get("mission", "").upper()
        if mission:
            ref_by_mission[mission].append(obs)

    for obs in sci_obs:
        mission = obs.get("mission", "").upper()
        if mission:
            sci_by_mission[mission].append(obs)

    # Find common missions
    common_missions = set(ref_by_mission.keys()) & set(sci_by_mission.keys())

    if not common_missions:
        return []

    # Create filtered entries
    filtered_entries = []

    if split_by_mission:
        # Create separate entry per mission
        for mission in sorted(common_missions):
            filtered_entry = {
                **entry,
                "sn_name": f"{sn_name}_{mission}",
                "original_sn_name": sn_name,
                "mission": mission,
                "reference_observations": ref_by_mission[mission],
                "science_observations": sci_by_mission[mission],
            }
            filtered_entries.append(filtered_entry)
    else:
        # Combine all common-mission observations into single entry
        filtered_ref = []
        filtered_sci = []

        for mission in common_missions:
            filtered_ref.extend(ref_by_mission[mission])
            filtered_sci.extend(sci_by_mission[mission])

        filtered_entry = {
            **entry,
            "reference_observations": filtered_ref,
            "science_observations": filtered_sci,
            "common_missions": sorted(common_missions),
        }
        filtered_entries.append(filtered_entry)

    return filtered_entries


def filter_query_results(
    query_results: list[dict[str, Any]],
    split_by_mission: bool = False,
    require_both_missions: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Filter all query results to keep only matching missions.

    Args:
        query_results: List of query result entries
        split_by_mission: Create separate entries per mission
        require_both_missions: Require both ref and sci for each mission

    Returns:
        Tuple of (filtered_results, stats)
    """
    filtered_results = []
    stats = {
        "input_sne": len(query_results),
        "output_entries": 0,
        "sne_with_matches": 0,
        "sne_without_matches": 0,
        "multi_mission_sne": 0,
        "missions": defaultdict(int),
        "ref_obs_before": 0,
        "ref_obs_after": 0,
        "sci_obs_before": 0,
        "sci_obs_after": 0,
    }

    for entry in query_results:
        sn_name = entry.get("sn_name", "UNKNOWN")
        ref_obs_count = len(entry.get("reference_observations", []))
        sci_obs_count = len(entry.get("science_observations", []))

        stats["ref_obs_before"] += ref_obs_count
        stats["sci_obs_before"] += sci_obs_count

        filtered_entries = filter_observations_by_mission(entry, split_by_mission)

        if filtered_entries:
            stats["sne_with_matches"] += 1
            if len(filtered_entries) > 1:
                stats["multi_mission_sne"] += 1

            for filtered_entry in filtered_entries:
                # Count missions
                if split_by_mission:
                    mission = filtered_entry.get("mission")
                    if mission:
                        stats["missions"][mission] += 1
                else:
                    for mission in filtered_entry.get("common_missions", []):
                        stats["missions"][mission] += 1

                # Count observations
                ref_count = len(filtered_entry.get("reference_observations", []))
                sci_count = len(filtered_entry.get("science_observations", []))
                stats["ref_obs_after"] += ref_count
                stats["sci_obs_after"] += sci_count

                filtered_results.append(filtered_entry)
        else:
            stats["sne_without_matches"] += 1
            logger.debug(f"No matching missions for {sn_name}")

    stats["output_entries"] = len(filtered_results)
    stats["missions"] = dict(stats["missions"])

    return filtered_results, stats


def print_stats(stats: dict[str, Any], split_by_mission: bool):
    """Print filtering statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("MISSION FILTERING RESULTS")
    logger.info("=" * 70)
    logger.info(f"Input SNe: {stats['input_sne']}")
    logger.info(f"SNe with matching missions: {stats['sne_with_matches']}")
    logger.info(f"SNe without matches: {stats['sne_without_matches']}")

    if split_by_mission:
        logger.info(f"Multi-mission SNe (split): {stats['multi_mission_sne']}")
        logger.info(f"Output entries (after splitting): {stats['output_entries']}")
    else:
        logger.info(f"Multi-mission SNe (combined): {stats['multi_mission_sne']}")
        logger.info(f"Output entries: {stats['output_entries']}")

    logger.info("")
    logger.info("Observations filtered:")
    logger.info(f"  Reference: {stats['ref_obs_before']} → {stats['ref_obs_after']} "
                f"({100 * stats['ref_obs_after'] / stats['ref_obs_before']:.1f}% kept)")
    logger.info(f"  Science:   {stats['sci_obs_before']} → {stats['sci_obs_after']} "
                f"({100 * stats['sci_obs_after'] / stats['sci_obs_before']:.1f}% kept)")

    logger.info("")
    logger.info("Breakdown by mission:")
    for mission, count in sorted(stats["missions"].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {mission}: {count} entries")

    logger.info("=" * 70)

    # Calculate expected improvement
    success_rate = 100 * stats["sne_with_matches"] / stats["input_sne"] if stats["input_sne"] > 0 else 0
    logger.info(f"\n✅ Expected success rate: {success_rate:.1f}%")
    logger.info(f"   (vs current 25% = 56/225)")
    logger.info(f"   Expected complete pairs: ~{int(stats['sne_with_matches'] * 0.75)}-{stats['sne_with_matches']}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter observations to keep only matching missions"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input query results JSON (e.g., sn_queries_same_mission.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output filtered JSON file",
    )
    parser.add_argument(
        "--split-by-mission",
        action="store_true",
        help="Create separate entries per mission (e.g., 2009nz_SWIFT, 2009nz_PS1)",
    )
    parser.add_argument(
        "--require-both-missions",
        action="store_true",
        help="Require both ref and sci observations for each mission (default: True)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics without writing output file",
    )

    args = parser.parse_args()

    # Load query results
    logger.info(f"Loading query results from {args.input}")
    with open(args.input) as f:
        query_results = json.load(f)

    logger.info(f"Loaded {len(query_results)} entries")

    # Filter results
    filtered_results, stats = filter_query_results(
        query_results,
        split_by_mission=args.split_by_mission,
        require_both_missions=args.require_both_missions,
    )

    # Print statistics
    print_stats(stats, args.split_by_mission)

    # Save results
    if not args.stats_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(filtered_results, f, indent=2, default=str)

        logger.info(f"\n✅ Filtered results saved to {args.output}")
        logger.info(f"   Entries: {len(filtered_results)}")
        logger.info(f"   Ready for download with --filter-has-both flag")


if __name__ == "__main__":
    main()
