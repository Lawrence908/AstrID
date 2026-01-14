#!/usr/bin/env python3
"""
Unified pipeline runner that orchestrates all stages from a YAML configuration.

This script executes the complete supernova data acquisition pipeline:
1. Query MAST archive for observations
2. Filter for same-mission pairs
3. Download FITS files
4. Organize into training structure
5. Generate difference images

Usage:
    python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml
    python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --resume
    python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --dry-run
    python scripts/run_pipeline_from_config.py --config configs/swift_uv_dataset.yaml --stage download
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_stage(
    stage_name: str,
    config: PipelineConfig,
    dry_run: bool = False,
    **kwargs: Any,
) -> bool:
    """Run a single pipeline stage.

    Args:
        stage_name: Name of stage to run (query, filter, download, organize, differencing)
        config: Pipeline configuration
        dry_run: If True, only print what would be executed
        **kwargs: Additional arguments to pass to stage

    Returns:
        True if stage completed successfully, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage: {stage_name.upper()}")
    logger.info(f"{'='*60}")

    if stage_name == "query":
        return run_query_stage(config, dry_run=dry_run, **kwargs)
    elif stage_name == "filter":
        return run_filter_stage(config, dry_run=dry_run, **kwargs)
    elif stage_name == "download":
        return run_download_stage(config, dry_run=dry_run, **kwargs)
    elif stage_name == "organize":
        return run_organize_stage(config, dry_run=dry_run, **kwargs)
    elif stage_name == "differencing":
        return run_differencing_stage(config, dry_run=dry_run, **kwargs)
    else:
        logger.error(f"Unknown stage: {stage_name}")
        return False


def run_query_stage(
    config: PipelineConfig, dry_run: bool = False, resume: bool = False, **kwargs: Any
) -> bool:
    """Run the query stage."""
    script_path = project_root / "scripts" / "query_sn_fits_chunked.py"
    catalog_path = project_root / "resources" / "sncat_compiled.txt"

    if not catalog_path.exists():
        logger.error(f"Catalog file not found: {catalog_path}")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--catalog",
        str(catalog_path),
        "--output",
        str(config.output.query_results),
        "--chunk-size",
        str(config.query.chunk_size),
        "--checkpoint",
        str(config.output.checkpoint),
        "--chunk-dir",
        str(config.output.chunk_dir),
        "--days-before",
        str(config.query.days_before),
        "--days-after",
        str(config.query.days_after),
        "--radius",
        str(config.query.radius_deg),
        "--missions",
        *config.query.missions,
    ]

    if config.query.min_year:
        cmd.extend(["--min-year", str(config.query.min_year)])
    if config.query.max_year:
        cmd.extend(["--max-year", str(config.query.max_year)])
    if config.query.start_index > 0:
        cmd.extend(["--start-index", str(config.query.start_index)])
    if config.query.limit:
        cmd.extend(["--limit", str(config.query.limit)])
    if resume:
        logger.info("Resuming from checkpoint...")
    else:
        cmd.append("--reset-checkpoint")

    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute query stage")
        return True

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)
        logger.info("✅ Query stage completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Query stage failed: {e}")
        return False


def run_filter_stage(
    config: PipelineConfig, dry_run: bool = False, **kwargs: Any
) -> bool:
    """Run the same-mission filtering stage."""
    script_path = project_root / "scripts" / "identify_same_mission_pairs.py"

    # Output for filtered results
    filtered_output = config.output.query_results.parent / "same_mission_pairs.json"

    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(config.output.query_results),
        "--output",
        str(filtered_output),
    ]

    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute filter stage")
        return True

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)
        logger.info("✅ Filter stage completed successfully")
        logger.info(f"Filtered results saved to: {filtered_output}")

        # Update config to use filtered results for download stage
        if filtered_output.exists():
            with open(filtered_output) as f:
                filter_data = json.load(f)
            logger.info(
                f"Found {len(filter_data.get('same_mission_sne', []))} same-mission pairs"
            )

        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Filter stage failed: {e}")
        return False


def run_download_stage(
    config: PipelineConfig, dry_run: bool = False, **kwargs: Any
) -> bool:
    """Run the download stage."""
    script_path = project_root / "scripts" / "download_sn_fits.py"

    # Use filtered results if available, otherwise use query results
    filtered_output = config.output.query_results.parent / "same_mission_pairs.json"
    if filtered_output.exists():
        # Need to extract the detailed results from same_mission_pairs.json
        # For now, use the original query results
        input_file = config.output.query_results
    else:
        input_file = config.output.query_results

    cmd = [
        sys.executable,
        str(script_path),
        "--query-results",
        str(input_file),
        "--output-dir",
        str(config.output.fits_downloads),
        "--max-obs",
        str(config.download.max_obs_per_type),
        "--max-products-per-obs",
        str(config.download.max_products_per_obs),
    ]

    if config.download.require_same_mission:
        cmd.append("--filter-has-both")
    if not config.download.verify_fits:
        cmd.append("--no-verify")
    if config.download.include_auxiliary:
        cmd.append("--include-auxiliary")
    if config.download.skip_reference:
        cmd.append("--skip-reference")
    if config.download.skip_science:
        cmd.append("--skip-science")
    if dry_run:
        cmd.append("--dry-run")

    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute download stage")
        return True

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)
        logger.info("✅ Download stage completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Download stage failed: {e}")
        return False


def run_organize_stage(
    config: PipelineConfig, dry_run: bool = False, **kwargs: Any
) -> bool:
    """Run the organization stage."""
    script_path = project_root / "scripts" / "organize_training_pairs.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir",
        str(config.output.fits_downloads),
        "--output-dir",
        str(config.output.fits_training),
        "--clean",
    ]

    # Add optional flags
    if kwargs.get("symlink", False):
        cmd.append("--symlink")
    if kwargs.get("decompress", False):
        cmd.append("--decompress")

    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute organize stage")
        return True

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)
        logger.info("✅ Organize stage completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Organize stage failed: {e}")
        return False


def run_differencing_stage(
    config: PipelineConfig, dry_run: bool = False, **kwargs: Any
) -> bool:
    """Run the differencing stage."""
    script_path = project_root / "scripts" / "generate_difference_images.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir",
        str(config.output.fits_training),
        "--output-dir",
        str(config.output.difference_images),
    ]

    if kwargs.get("visualize", False):
        cmd.append("--visualize")
    if kwargs.get("mission"):
        cmd.extend(["--mission", kwargs["mission"]])

    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute differencing stage")
        return True

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)
        logger.info("✅ Differencing stage completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Differencing stage failed: {e}")
        return False


def generate_pipeline_report(config: PipelineConfig) -> None:
    """Generate a summary report of the pipeline execution."""
    report_path = config.output.difference_images / "pipeline_report.json"

    report = {
        "dataset_name": config.dataset_name,
        "description": config.description,
        "configuration": config.to_dict(),
        "output_paths": {
            "query_results": str(config.output.query_results),
            "fits_downloads": str(config.output.fits_downloads),
            "fits_training": str(config.output.fits_training),
            "difference_images": str(config.output.difference_images),
        },
    }

    # Try to load statistics from each stage
    if config.output.query_results.exists():
        try:
            with open(config.output.query_results) as f:
                query_data = json.load(f)
                report["query_stats"] = {
                    "total_sne": len(query_data),
                    "viable_pairs": sum(
                        1
                        for entry in query_data
                        if entry.get("reference_observations")
                        and entry.get("science_observations")
                    ),
                }
        except Exception as e:
            logger.warning(f"Could not load query stats: {e}")

    if (config.output.difference_images / "processing_summary.json").exists():
        try:
            with open(
                config.output.difference_images / "processing_summary.json"
            ) as f:
                diff_data = json.load(f)
                report["differencing_stats"] = {
                    "n_processed": diff_data.get("n_processed", 0),
                    "results": diff_data.get("results", []),
                }
        except Exception as e:
            logger.warning(f"Could not load differencing stats: {e}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📄 Pipeline report saved to: {report_path}")


def main() -> None:
    """Main entry point for pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run supernova data acquisition pipeline from YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["query", "filter", "download", "organize", "differencing"],
        default=None,
        help="Run only a specific stage (default: run all stages)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (for query stage)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be executed without running",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks in organize stage (saves disk space)",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress .fits.gz files in organize stage",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations in differencing stage",
    )
    parser.add_argument(
        "--mission",
        type=str,
        help="Filter to specific mission in differencing stage",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = PipelineConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration: {config.dataset_name}")
        if config.description:
            logger.info(f"Description: {config.description}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Validate configuration
    warnings = config.validate()
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Determine which stages to run
    if args.stage:
        stages = [args.stage]
    else:
        stages = ["query", "filter", "download", "organize", "differencing"]

    # Run stages
    success = True
    for stage in stages:
        stage_kwargs = {
            "symlink": args.symlink,
            "decompress": args.decompress,
            "visualize": args.visualize,
            "mission": args.mission,
        }
        if not run_stage(
            stage,
            config,
            dry_run=args.dry_run,
            resume=args.resume,
            **stage_kwargs,
        ):
            success = False
            if not args.dry_run:
                logger.error(f"Pipeline failed at stage: {stage}")
                break

    # Generate report
    if success and not args.dry_run:
        generate_pipeline_report(config)

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ Pipeline completed successfully!")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("❌ Pipeline failed")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
