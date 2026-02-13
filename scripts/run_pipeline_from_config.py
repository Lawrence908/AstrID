#!/usr/bin/env python3
"""
Unified pipeline runner that orchestrates all stages from a YAML configuration.

This script executes the complete supernova data acquisition pipeline:
1. Query MAST archive for observations
2. Filter for same-mission pairs
3. Download FITS files
4. Organize into training structure
5. Generate difference images

Progress/resume:
    Progress is saved to pipeline_progress.json (next to query results). On the next
    run, download/organize/differencing skip SNe already completed and only process
    the remaining list, making reruns much faster. Use --no-skip-completed to reprocess all.

Usage:
    python scripts/run_pipeline_from_config.py --config configs/full_catalog.yaml
    python scripts/run_pipeline_from_config.py --config configs/full_catalog.yaml --resume
    python scripts/run_pipeline_from_config.py --config configs/full_catalog.yaml --dry-run
    python scripts/run_pipeline_from_config.py --config configs/full_catalog.yaml --stage download
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Exit code when process is killed by SIGKILL (e.g. OOM killer)
_SIGKILL_EXIT = -(signal.SIGKILL) if hasattr(signal, "SIGKILL") else -9
_SIGKILL_EXIT_ALT = 128 + 9  # 137, some systems report this

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Progress file lives next to query results (per-dataset)
PROGRESS_FILENAME = "pipeline_progress.json"
SAME_MISSION_PAIRS_FILENAME = "same_mission_pairs.json"


def _log_sigkill_hint(exc: subprocess.CalledProcessError) -> None:
    """If the subprocess died with SIGKILL, log a clear OOM/tmux hint."""
    r = getattr(exc, "returncode", None)
    if r is None:
        return
    if r == _SIGKILL_EXIT or r == _SIGKILL_EXIT_ALT:
        logger.error(
            "Process was killed with SIGKILL (signal 9). This is usually the Linux "
            "OOM killer. Check: dmesg | tail -30 (look for 'Out of memory' / 'oom-killer'). "
            "Try smaller --chunk-size in the config or run with more RAM/swap."
        )


def progress_path(config: PipelineConfig) -> Path:
    """Path to the pipeline progress file for this config."""
    return config.output.query_results.parent / PROGRESS_FILENAME


def same_mission_pairs_path(config: PipelineConfig) -> Path:
    """Path to same_mission_pairs.json produced by filter stage."""
    return config.output.query_results.parent / SAME_MISSION_PAIRS_FILENAME


def load_progress(config: PipelineConfig) -> dict[str, Any]:
    """Load pipeline progress (completed/attempted SNe). Returns empty dict if missing."""
    path = progress_path(config)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load progress from {path}: {e}")
        return {}


def save_progress(config: PipelineConfig, data: dict[str, Any]) -> None:
    """Save pipeline progress."""
    path = progress_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    if "config_path" not in data:
        data["config_path"] = str(config.output.query_results)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Progress saved to {path}")


def get_full_and_remaining_sne(
    config: PipelineConfig, skip_completed: bool = True
) -> tuple[list[str] | None, list[str] | None, set[str]]:
    """Get full SN list and remaining SNe to process (excluding completed).

    Reads same_mission_pairs.json (from filter stage) and pipeline_progress.json.

    Returns:
        (full_sne, remaining_sne, completed_set)
        - full_sne: list of all same-mission SNe (None if filter not run yet)
        - remaining_sne: list to process this run (None = process all)
        - completed_set: set of SN names already completed through differencing
    """
    pairs_path = same_mission_pairs_path(config)
    if not pairs_path.exists():
        return None, None, set()

    try:
        with open(pairs_path) as f:
            data = json.load(f)
        full_sne = data.get("same_mission_sne", [])
        if not full_sne:
            return [], [], set()
        full_sne = sorted(full_sne)
    except Exception as e:
        logger.warning(f"Could not load same_mission_pairs from {pairs_path}: {e}")
        return None, None, set()

    if not skip_completed:
        return full_sne, full_sne, set()

    progress = load_progress(config)
    completed_set = set(progress.get("completed_sne", []))
    remaining = [s for s in full_sne if s not in completed_set]
    return full_sne, remaining, completed_set


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
        _log_sigkill_hint(e)
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
        _log_sigkill_hint(e)
        return False


def run_download_stage(
    config: PipelineConfig, dry_run: bool = False, **kwargs: Any
) -> bool:
    """Run the download stage."""
    skip_completed = kwargs.get("skip_completed", True)
    full_sne, remaining_sne, completed_set = get_full_and_remaining_sne(
        config, skip_completed=skip_completed
    )
    if full_sne is not None and not remaining_sne:
        logger.info(
            "All SNe already completed (see %s); skipping download.",
            progress_path(config),
        )
        return True

    script_path = project_root / "scripts" / "download_sn_fits.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--query-results",
        str(config.output.query_results),
        "--output-dir",
        str(config.output.fits_downloads),
        "--max-obs",
        str(config.download.max_obs_per_type),
        "--max-products-per-obs",
        str(config.download.max_products_per_obs),
    ]

    if completed_set and full_sne is not None:
        cmd.extend(["--skip-sn"] + sorted(completed_set))
        logger.info("Resuming: skipping %d already-completed SNe", len(completed_set))

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
        _log_sigkill_hint(e)
        return False


def run_organize_stage(
    config: PipelineConfig, dry_run: bool = False, **kwargs: Any
) -> bool:
    """Run the organization stage."""
    skip_completed = kwargs.get("skip_completed", True)
    full_sne, remaining_sne, completed_set = get_full_and_remaining_sne(
        config, skip_completed=skip_completed
    )
    if full_sne is not None and not remaining_sne:
        logger.info(
            "All SNe already completed (see %s); skipping organize.",
            progress_path(config),
        )
        return True

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

    if remaining_sne and full_sne is not None:
        cmd.extend(["--sn"] + remaining_sne)
        logger.info("Resuming: organizing only %d remaining SNe", len(remaining_sne))

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
        _log_sigkill_hint(e)
        return False


def run_differencing_stage(
    config: PipelineConfig, dry_run: bool = False, **kwargs: Any
) -> bool:
    """Run the differencing stage."""
    skip_completed = kwargs.get("skip_completed", True)
    full_sne, remaining_sne, completed_set = get_full_and_remaining_sne(
        config, skip_completed=skip_completed
    )
    if full_sne is not None and not remaining_sne:
        logger.info(
            "All SNe already completed (see %s); skipping differencing.",
            progress_path(config),
        )
        return True

    script_path = project_root / "scripts" / "generate_difference_images.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir",
        str(config.output.fits_training),
        "--output-dir",
        str(config.output.difference_images),
    ]

    if remaining_sne and full_sne is not None:
        cmd.extend(["--sn"] + remaining_sne)
        logger.info("Resuming: differencing only %d remaining SNe", len(remaining_sne))

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
        if result.returncode == 0:
            _update_progress_after_differencing(config)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Differencing stage failed: {e}")
        _log_sigkill_hint(e)
        return False


def _update_progress_after_differencing(config: PipelineConfig) -> None:
    """Read processing_summary.json and merge successfully processed SNe into pipeline progress."""
    summary_path = config.output.difference_images / "processing_summary.json"
    if not summary_path.exists():
        return
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        results = summary.get("results", [])
        newly_completed = [r["sn_name"] for r in results if r.get("sn_name")]
        if not newly_completed:
            return
        progress = load_progress(config)
        completed = set(progress.get("completed_sne", []))
        completed.update(newly_completed)
        progress["completed_sne"] = sorted(completed)
        save_progress(config, progress)
        logger.info("Progress updated: %d SNe now marked completed (total %d)", len(newly_completed), len(completed))
    except Exception as e:
        logger.warning("Could not update progress from differencing summary: %s", e)


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
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Do not skip SNe already in pipeline_progress.json (reprocess all)",
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

    # Log resume status when progress exists
    if not args.no_skip_completed:
        full_sne, remaining_sne, completed_set = get_full_and_remaining_sne(
            config, skip_completed=True
        )
        if full_sne is not None and completed_set:
            n_remaining = len(full_sne) - len(completed_set)
            logger.info(
                "Resume: %d SNe already completed, %d remaining (progress: %s)",
                len(completed_set),
                n_remaining,
                progress_path(config),
            )

    # Determine which stages to run
    if args.stage:
        stages = [args.stage]
    else:
        stages = ["query", "filter", "download", "organize", "differencing"]

    # Run stages
    success = True
    skip_completed = not args.no_skip_completed
    for stage in stages:
        stage_kwargs = {
            "symlink": args.symlink,
            "decompress": args.decompress,
            "visualize": args.visualize,
            "mission": args.mission,
            "skip_completed": skip_completed,
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
