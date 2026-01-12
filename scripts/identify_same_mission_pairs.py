#!/usr/bin/env python3
"""
Identify same-mission pairs from query results.

Analyzes the query results JSON to find supernovae where reference and science
observations share at least one mission (e.g., SWIFT-SWIFT, PS1-PS1).

Usage:
    python scripts/identify_same_mission_pairs.py \
        --input output/sn_queries_remaining.json \
        --output output/same_mission_pairs.json
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


def identify_same_mission_pairs(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Identify SNe with same-mission pairs.

    Returns:
        Dictionary with:
        - same_mission_sne: List of SN names with same-mission pairs
        - pairs_by_mission: Dict mapping mission -> list of SN names
        - detailed_results: Full results with mission overlap info
    """
    same_mission_sne = []
    pairs_by_mission = defaultdict(list)
    detailed_results = []

    for entry in query_results:
        sn_name = entry.get("sn_name", "UNKNOWN")
        ref_obs = entry.get("reference_observations", [])
        sci_obs = entry.get("science_observations", [])

        if not ref_obs or not sci_obs:
            continue

        # Get unique missions from reference and science observations
        ref_missions = {
            obs.get("mission", "").upper() for obs in ref_obs if obs.get("mission")
        }
        sci_missions = {
            obs.get("mission", "").upper() for obs in sci_obs if obs.get("mission")
        }

        # Find overlapping missions
        common_missions = ref_missions & sci_missions

        if common_missions:
            same_mission_sne.append(sn_name)

            # Track by mission
            for mission in common_missions:
                pairs_by_mission[mission].append(sn_name)

            # Count observations per mission
            ref_by_mission = defaultdict(int)
            sci_by_mission = defaultdict(int)

            for obs in ref_obs:
                mission = obs.get("mission", "").upper()
                if mission:
                    ref_by_mission[mission] += 1

            for obs in sci_obs:
                mission = obs.get("mission", "").upper()
                if mission:
                    sci_by_mission[mission] += 1

            detailed_results.append(
                {
                    "sn_name": sn_name,
                    "ra_deg": entry.get("ra_deg"),
                    "dec_deg": entry.get("dec_deg"),
                    "discovery_date": entry.get("discovery_date"),
                    "common_missions": sorted(common_missions),
                    "ref_missions": sorted(ref_missions),
                    "sci_missions": sorted(sci_missions),
                    "ref_counts_by_mission": dict(ref_by_mission),
                    "sci_counts_by_mission": dict(sci_by_mission),
                    "total_ref_obs": len(ref_obs),
                    "total_sci_obs": len(sci_obs),
                }
            )

    return {
        "same_mission_sne": sorted(same_mission_sne),
        "pairs_by_mission": {k: sorted(v) for k, v in pairs_by_mission.items()},
        "detailed_results": detailed_results,
        "summary": {
            "total_sne_analyzed": len(query_results),
            "same_mission_count": len(same_mission_sne),
            "same_mission_percentage": round(
                100 * len(same_mission_sne) / len(query_results), 1
            )
            if query_results
            else 0,
            "missions_found": sorted(pairs_by_mission.keys()),
            "counts_by_mission": {k: len(v) for k, v in pairs_by_mission.items()},
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Identify same-mission pairs from query results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input query results JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file with same-mission pairs",
    )

    args = parser.parse_args()

    # Load query results
    logger.info(f"Loading query results from {args.input}")
    with open(args.input) as f:
        query_results = json.load(f)

    logger.info(f"Analyzing {len(query_results)} supernovae...")

    # Identify same-mission pairs
    results = identify_same_mission_pairs(query_results)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SAME-MISSION PAIRS ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total SNe analyzed: {results['summary']['total_sne_analyzed']}")
    logger.info(
        f"Same-mission pairs found: {results['summary']['same_mission_count']} ({results['summary']['same_mission_percentage']}%)"
    )
    logger.info("")
    logger.info("Breakdown by mission:")
    for mission, count in sorted(results["summary"]["counts_by_mission"].items()):
        logger.info(f"  {mission}: {count} SNe")
    logger.info("=" * 60)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n✅ Results saved to {args.output}")
    logger.info(f"   Same-mission SNe: {len(results['same_mission_sne'])}")
    logger.info(f"   Detailed results: {len(results['detailed_results'])} entries")


if __name__ == "__main__":
    main()
