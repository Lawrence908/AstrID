#!/usr/bin/env python3
"""
Filter query results to only include specified SN names.

Usage:
    python scripts/filter_query_results.py \
        --input output/sn_queries_remaining.json \
        --sn-list output/same_mission_pairs.json \
        --output output/sn_queries_same_mission.json
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Filter query results to specific SN names"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input query results JSON file",
    )
    parser.add_argument(
        "--sn-list",
        type=Path,
        required=True,
        help="JSON file with 'same_mission_sne' list or list of SN names",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output filtered JSON file",
    )

    args = parser.parse_args()

    # Load query results
    logger.info(f"Loading query results from {args.input}")
    with open(args.input) as f:
        query_results = json.load(f)

    # Load SN list
    logger.info(f"Loading SN list from {args.sn_list}")
    with open(args.sn_list) as f:
        sn_data = json.load(f)

    # Extract SN names (handle both formats)
    if isinstance(sn_data, dict) and "same_mission_sne" in sn_data:
        sn_names = set(sn_data["same_mission_sne"])
    elif isinstance(sn_data, list):
        sn_names = set(sn_data)
    else:
        raise ValueError(f"Unexpected format in {args.sn_list}")

    logger.info(f"Filtering to {len(sn_names)} SN names...")

    # Filter query results
    filtered = [entry for entry in query_results if entry.get("sn_name") in sn_names]

    logger.info(f"Found {len(filtered)} matching entries")

    # Save filtered results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(filtered, f, indent=2, default=str)

    logger.info(f"✅ Filtered results saved to {args.output}")


if __name__ == "__main__":
    main()
