#!/usr/bin/env python3
"""
Robust supernova FITS downloader with validation, filtering, and reporting.

The script reads query output produced by `query_sn_fits_from_catalog.py`,
filters out observations that cannot be downloaded, and fetches only the
imaging products required for building before/after pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / heuristics for filtering data products
# ---------------------------------------------------------------------------

IMAGING_PRODUCT_KEYWORDS = [
    "image",
    "img",
    "int",
    "rrhr",
    "skybg",
    "flags",
    "stk",
    "skycell",
    "unconv",
    "mast:ps1",
    ".fits",
    ".fits.gz",
    ".img",
]

SPECTROSCOPY_PRODUCT_KEYWORDS = [
    "spec",
    "spectrum",
    "spectra",
    "spall",
    "spplate",
    "spz",
    "spAllLine",
    "allspec",
]

CATALOG_PRODUCT_KEYWORDS = [
    "catalog",
    "combined",
    "_v5_",
    "table",
    "summary",
]

# Auxiliary file patterns to exclude (unless --include-auxiliary is set)
AUXILIARY_FILE_PATTERNS = [
    ".mask.fits",
    ".wt.fits",
    ".weight.fits",
    ".pswarp.mdc",
    ".cmf",
    ".unconv.",
    ".target.psf",
    ".exp.fits",
    ".expwt.fits",
    ".num.fits",
    ".jpg",
    ".png",
    ".gif",
    ".jpeg",
    "preview",
    "thumbnail",
]

# Keywords in description/subgroup that indicate auxiliary data
AUXILIARY_DESCRIPTION_KEYWORDS = [
    "auxiliary",
    "metadata",
    "weight",
    "mask",
    "background",
    "noise",
]

MAX_FILE_SIZE_MB = 500
DEFAULT_MAX_OBS = 3


@dataclass
class PrefilterStats:
    total_supernovae: int = 0
    viable_supernovae: int = 0
    reference_only: int = 0
    science_only: int = 0
    neither: int = 0
    total_reference_obs: int = 0
    total_science_obs: int = 0


@dataclass
class DownloadStats:
    reference_files: int = 0
    science_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    complete_pairs: int = 0
    errors: int = 0
    warning_counts: dict[str, int] = None

    def __post_init__(self):
        if self.warning_counts is None:
            object.__setattr__(self, "warning_counts", defaultdict(int))


def human_readable_size(num_bytes: Any) -> str:
    """Convert byte counts to a readable string."""
    if num_bytes in (None, ""):
        return "unknown size"
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "unknown size"
    units = ["bytes", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "bytes":
                return f"{size:.0f} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def validate_observation_data(observation: dict[str, Any]) -> bool:
    """Quick check to ensure an observation has a downloadable data URL."""
    if not observation:
        return False
    data_url = observation.get("dataURL") or observation.get("dataurl")
    return bool(observation.get("obs_id") and observation.get("mission") and data_url)


def prefilter_query_results(
    query_results: Sequence[dict[str, Any]],
    require_both: bool,
) -> tuple[list[dict[str, Any]], PrefilterStats]:
    """Filter observations without download URLs and compute stats."""
    stats = PrefilterStats(total_supernovae=len(query_results))
    filtered: list[dict[str, Any]] = []

    for entry in query_results:
        sn_name = entry.get("sn_name", "UNKNOWN")
        ref_obs = [
            obs
            for obs in entry.get("reference_observations", [])
            if validate_observation_data(obs)
        ]
        sci_obs = [
            obs
            for obs in entry.get("science_observations", [])
            if validate_observation_data(obs)
        ]

        if not ref_obs and not sci_obs:
            stats.neither += 1
            continue

        if require_both:
            if not ref_obs and sci_obs:
                stats.science_only += 1
                continue
            if not sci_obs and ref_obs:
                stats.reference_only += 1
                continue

        if not ref_obs:
            stats.reference_only += 1
        if not sci_obs:
            stats.science_only += 1

        stats.viable_supernovae += 1
        stats.total_reference_obs += len(ref_obs)
        stats.total_science_obs += len(sci_obs)

        filtered.append(
            {
                **entry,
                "sn_name": sn_name,
                "reference_observations": ref_obs,
                "science_observations": sci_obs,
            }
        )

    return filtered, stats


def log_prefilter_report(stats: PrefilterStats, mode_label: str) -> None:
    """Print a detailed pre-filtering report."""
    logger.info("\n" + "=" * 60)
    logger.info("PRE-FILTERING QUERY RESULTS")
    logger.info("=" * 60)
    logger.info("Mode: %s", mode_label)
    logger.info("")
    logger.info("Filtering results:")
    logger.info("  Total supernovae in query: %s", stats.total_supernovae)
    logger.info("  ✅ With both ref & sci (viable): %s", stats.viable_supernovae)
    logger.info("  ⚠️  Reference only: %s", stats.reference_only)
    logger.info("  ⚠️  Science only: %s", stats.science_only)
    logger.info("  ❌ Neither: %s", stats.neither)
    logger.info("")
    logger.info("  Total viable reference observations: %s", stats.total_reference_obs)
    logger.info("  Total viable science observations: %s", stats.total_science_obs)
    logger.info("")
    logger.info("  Supernovae to process: %s", stats.viable_supernovae)
    logger.info("=" * 60)


def filter_products(table: Any, include_auxiliary: bool = False) -> Any:
    """Return only imaging products under size threshold.

    Args:
        table: Product table from MAST query
        include_auxiliary: If True, include auxiliary files (masks, weights, etc.)
                          If False (default), exclude auxiliary files to reduce download size

    Returns:
        Filtered product table
    """
    if len(table) == 0:
        return table

    indices: list[int] = []
    excluded_auxiliary = 0

    for idx, row in enumerate(table):
        filename = str(row.get("productFilename", "")).lower()
        subgroup = str(row.get("productSubGroupDescription", "")).lower()
        description = str(row.get("description", "")).lower()

        if not filename:
            continue

        # Filter spectroscopy products
        if any(keyword in filename for keyword in SPECTROSCOPY_PRODUCT_KEYWORDS):
            continue
        if any(keyword in description for keyword in SPECTROSCOPY_PRODUCT_KEYWORDS):
            continue

        # Filter catalog products
        if any(keyword in filename for keyword in CATALOG_PRODUCT_KEYWORDS):
            continue

        # Check if product looks like imaging data
        looks_imaging = any(
            keyword in filename for keyword in IMAGING_PRODUCT_KEYWORDS
        ) or any(keyword in subgroup for keyword in IMAGING_PRODUCT_KEYWORDS)
        if not looks_imaging:
            continue

        # Filter auxiliary files (unless explicitly included)
        if not include_auxiliary:
            is_auxiliary = (
                any(pattern in filename for pattern in AUXILIARY_FILE_PATTERNS)
                or any(
                    keyword in description for keyword in AUXILIARY_DESCRIPTION_KEYWORDS
                )
                or any(
                    keyword in subgroup for keyword in AUXILIARY_DESCRIPTION_KEYWORDS
                )
            )
            if is_auxiliary:
                excluded_auxiliary += 1
                continue

        # Filter by file size
        file_size = row.get("size") or row.get("filesize") or row.get("fileSize")
        size_mb = None
        if file_size:
            size_mb = float(file_size) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                continue

        indices.append(idx)

    # At this point, indices contains imaging, non-auxiliary products under the
    # size limit. To further reduce redundancy (especially for PS1), keep at
    # most one "best" product per band (g, r, i, z, y) per observation.
    #
    # We infer band from common PS1 filename patterns like:
    #   rings.v3.skycell.2426.020.wrp.g.55191_54928.fits
    #   rings.v3.skycell.2426.020.stk.r.55191_56472.fits
    #
    # For missions where we cannot infer a band, we keep all products.
    if indices:
        band_best: dict[str, tuple[int, float]] = {}
        unbanded_indices: list[int] = []

        for idx in indices:
            row = table[idx]
            filename = str(row.get("productFilename", "")).lower()

            band: str | None = None
            for marker in (".wrp.", ".stk."):
                if marker in filename:
                    pos = filename.find(marker) + len(marker)
                    if pos < len(filename):
                        band = filename[pos]
                    break

            # Use file size as a proxy for "best" quality when available
            file_size = row.get("size") or row.get("filesize") or row.get("fileSize")
            size_bytes = float(file_size) if file_size not in (None, "") else 0.0

            if band is None:
                # For products where we can't confidently infer a band, keep them all
                unbanded_indices.append(idx)
            else:
                current_best = band_best.get(band)
                if current_best is None or size_bytes > current_best[1]:
                    band_best[band] = (idx, size_bytes)

        # Combine all unbanded products with the single best product per band
        selected_indices = set(unbanded_indices)
        selected_indices.update(best_idx for best_idx, _ in band_best.values())
        indices = sorted(selected_indices)

    if excluded_auxiliary > 0:
        logger.debug("Excluded %s auxiliary files from download", excluded_auxiliary)

    return table[indices] if indices else table[:0]


def extract_manifest_paths(manifest: Any) -> list[Path]:
    """Extract downloaded file paths from astroquery manifest."""
    paths: list[Path] = []
    if manifest is None:
        return paths

    if hasattr(manifest, "colnames") and "Local Path" in manifest.colnames:
        for row in manifest:
            local_path = row.get("Local Path")
            if local_path:
                path_obj = Path(local_path)
                if path_obj.exists():
                    paths.append(path_obj)
    elif isinstance(manifest, dict) and "Local Path" in manifest:
        for local_path in manifest.get("Local Path", []):
            path_obj = Path(local_path)
            if path_obj.exists():
                paths.append(path_obj)
    elif hasattr(manifest, "to_pandas"):
        df = manifest.to_pandas()
        if "Local Path" in df.columns:
            for local_path in df["Local Path"].dropna():
                path_obj = Path(str(local_path))
                if path_obj.exists():
                    paths.append(path_obj)
    return paths


def verify_fits_file(
    file_path: Path, warning_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    """Open a FITS file and verify it contains WCS information.

    Suppresses FITS header warnings and tracks them for summary reporting.
    """
    if warning_counts is None:
        warning_counts = defaultdict(int)

    info = {
        "file": str(file_path),
        "valid": False,
        "has_wcs": False,
        "observation_time": None,
        "filter": None,
        "error": None,
    }
    try:
        # Suppress FITS header warnings during verification
        with warnings.catch_warnings():
            # Filter out common FITS header warnings that are harmless
            warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
            # CardWarning may not exist in all astropy versions
            try:
                warnings.filterwarnings("ignore", category=fits.card.CardWarning)
            except AttributeError:
                pass  # CardWarning doesn't exist in this astropy version
            warnings.filterwarnings(
                "ignore", message=".*unrecognized non-standard convention.*"
            )
            warnings.filterwarnings("ignore", message=".*Unprintable string.*")
            warnings.filterwarnings(
                "ignore", message=".*FITS header values must contain.*"
            )

            # Track FITSFixedWarning but don't show them
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=FITSFixedWarning)
                warnings.simplefilter("always", category=fits.verify.VerifyWarning)
                # CardWarning may not exist in all astropy versions
                try:
                    warnings.simplefilter("always", category=fits.card.CardWarning)
                except AttributeError:
                    pass  # CardWarning doesn't exist in this astropy version

                with fits.open(file_path) as hdul:
                    header = hdul[0].header
                    info["observation_time"] = header.get("DATE-OBS") or header.get(
                        "MJD-OBS"
                    )
                    info["filter"] = (
                        header.get("FILTER") or header.get("FILT") or header.get("BAND")
                    )
                    try:
                        wcs = WCS(header)
                        info["has_wcs"] = wcs.has_celestial
                    except Exception as exc:  # pragma: no cover - defensive
                        info["error"] = f"WCS parsing error: {exc}"
                        info["has_wcs"] = False
                    info["valid"] = bool(info["has_wcs"])

                # Count warnings by type
                for warning in w:
                    warning_type = type(warning.category).__name__
                    warning_counts[warning_type] += 1
    except Exception as exc:  # pragma: no cover - defensive
        info["error"] = str(exc)
    return info


def download_products_for_observation(
    obs_id: str,
    mission: str,
    destination_dir: Path,
    include_auxiliary: bool = False,
) -> list[Path]:
    """Download imaging products for a single observation.

    Args:
        obs_id: Observation ID from MAST
        mission: Mission name (e.g., 'PS1', 'GALEX', 'SWIFT')
        destination_dir: Directory to save downloaded files
        include_auxiliary: If True, include auxiliary files (masks, weights, etc.)

    Returns:
        List of downloaded file paths
    """
    from astroquery.mast import Observations

    products = Observations.get_product_list(obs_id)
    total = len(products)
    if total == 0:
        logger.warning("No products found for %s %s", mission, obs_id)
        return []

    filtered = filter_products(products, include_auxiliary=include_auxiliary)
    logger.info(
        "Found %s imaging products for %s (filtered from %s total)",
        len(filtered),
        obs_id,
        total,
    )

    if len(filtered) == 0:
        logger.warning("No imaging products remained after filtering for %s", obs_id)
        return []

    destination_dir.mkdir(parents=True, exist_ok=True)

    manifest = Observations.download_products(
        filtered,
        download_dir=str(destination_dir),
        cache=True,
    )
    downloaded = extract_manifest_paths(manifest)
    logger.info("Downloaded %s files for %s", len(downloaded), obs_id)
    return downloaded


def run_dry_run(
    sn_name: str,
    observations: Sequence[dict[str, Any]],
    obs_type: str,
) -> None:
    """Log what would be downloaded during a dry run."""
    logger.info(
        "[DRY RUN] Would download %s %s observations", len(observations), obs_type
    )
    for idx, obs in enumerate(observations, 1):
        mission = obs.get("mission", "Unknown")
        obs_id = obs.get("obs_id", "Unknown")
        filesize = obs.get("filesize")
        logger.info(
            "  [%s/%s] %s %s (%s)",
            idx,
            len(observations),
            mission,
            obs_id,
            human_readable_size(filesize),
        )


def process_observation_batch(
    sn_dir: Path,
    observations: Sequence[dict[str, Any]],
    obs_type: str,
    verify_fits: bool,
    dry_run: bool,
    include_auxiliary: bool = False,
    warning_counts: dict[str, int] | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Download and verify all observations for a single SN/type.

    Args:
        sn_dir: Base directory for this supernova
        observations: List of observation metadata dicts
        obs_type: 'reference' or 'science'
        verify_fits: Whether to verify FITS files after download
        dry_run: If True, log what would be downloaded without fetching
        include_auxiliary: If True, include auxiliary files in downloads
        warning_counts: Dictionary to track warning counts

    Returns:
        Tuple of (downloaded files, verified files, invalid files, error messages)
    """
    if warning_counts is None:
        warning_counts = defaultdict(int)

    downloaded: list[str] = []
    verified: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    errors: list[str] = []

    if dry_run:
        run_dry_run(sn_dir.name, observations, obs_type)
        return downloaded, verified, invalid, errors

    target_dir = sn_dir / obs_type
    for idx, obs in enumerate(observations, 1):
        obs_id = str(obs.get("obs_id"))
        mission = obs.get("mission", "Unknown")
        logger.info(
            "[%s/%s] %s: %s %s",
            idx,
            len(observations),
            obs_type.capitalize(),
            mission,
            obs_id,
        )

        try:
            files = download_products_for_observation(
                obs_id, mission, target_dir, include_auxiliary=include_auxiliary
            )
            downloaded.extend(str(f) for f in files)

            if verify_fits:
                for file_path in files:
                    info = verify_fits_file(file_path, warning_counts=warning_counts)
                    if info["valid"]:
                        verified.append(info)
                        logger.info("  ✅ Verified: %s", Path(info["file"]).name)
                    else:
                        invalid.append(info)
                        logger.warning(
                            "  ⚠️  Invalid FITS: %s - %s",
                            Path(info["file"]).name,
                            info["error"],
                        )
        except Exception as exc:  # pragma: no cover - defensive
            message = f"Failed to download {mission} {obs_id}: {exc}"
            logger.error(message)
            errors.append(message)

    return downloaded, verified, invalid, errors


def build_observation_subset(
    observations: Sequence[dict[str, Any]],
    max_obs: int,
) -> list[dict[str, Any]]:
    """Return the first N observations, preserving existing order."""
    return list(observations[:max_obs])


def build_mode_label(require_both: bool) -> str:
    return (
        "Require both reference AND science"
        if require_both
        else "Allow partial (reference OR science)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download supernova FITS files from MAST results"
    )
    parser.add_argument(
        "--query-results",
        type=Path,
        required=True,
        help="Path to JSON produced by query_sn_fits_from_catalog.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fits"),
        help="Directory where downloads are stored (default: data/fits)",
    )
    parser.add_argument(
        "--max-obs",
        type=int,
        default=DEFAULT_MAX_OBS,
        help="Maximum observations per type per SN (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of supernovae processed",
    )
    parser.add_argument(
        "--min-viable-pairs",
        type=int,
        default=None,
        help="Abort if fewer than this many viable SN would be processed",
    )
    parser.add_argument(
        "--require-both",
        action="store_true",
        help="Explicitly require reference AND science observations",
    )
    parser.add_argument(
        "--filter-has-both",
        action="store_true",
        dest="require_both",
        help="Alias for --require-both: filter to SNe with both reference AND science observations",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow SN with only reference OR only science observations",
    )
    parser.add_argument(
        "--verify-fits",
        dest="verify_fits",
        action="store_true",
        default=True,
        help="Verify downloaded FITS files (default: on)",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify_fits",
        action="store_false",
        help="Skip FITS verification to save time",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview downloads without fetching files",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip downloading reference observations",
    )
    parser.add_argument(
        "--skip-science",
        action="store_true",
        help="Skip downloading science observations",
    )
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="Include auxiliary files (masks, weights, etc.) in downloads. "
        "Default: exclude to reduce download size by 60-80%%",
    )

    args = parser.parse_args()

    if not args.query_results.exists():
        logger.error("Query results file not found: %s", args.query_results)
        raise SystemExit(1)

    with open(args.query_results) as handle:
        query_results = json.load(handle)

    require_both = args.require_both or not args.allow_partial
    mode_label = build_mode_label(require_both)

    filtered_results, pre_stats = prefilter_query_results(
        query_results, require_both=require_both
    )
    log_prefilter_report(pre_stats, mode_label)

    # Log auxiliary file filtering mode
    if args.include_auxiliary:
        logger.info("\n⚠️  Auxiliary files WILL be downloaded (--include-auxiliary)")
        logger.info("   This will increase download size significantly.")
    else:
        logger.info("\n✅ Auxiliary files will be EXCLUDED (default behavior)")
        logger.info("   Expected reduction: 60-80%% in download size")
        logger.info("   Use --include-auxiliary to download all files if needed.")

    if args.min_viable_pairs and pre_stats.viable_supernovae < args.min_viable_pairs:
        logger.error(
            "Only %s viable supernovae found, below --min-viable-pairs=%s. Aborting.",
            pre_stats.viable_supernovae,
            args.min_viable_pairs,
        )
        raise SystemExit(1)

    if args.limit:
        filtered_results = filtered_results[: args.limit]

    if not filtered_results:
        logger.warning("⚠️  No supernovae to process after filtering!")
        logger.info(
            "Try running with --allow-partial or adjusting your query parameters."
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    download_results: list[dict[str, Any]] = []
    stats = DownloadStats()

    for idx, sn_entry in enumerate(filtered_results, 1):
        sn_name = sn_entry.get("sn_name", f"SN_{idx}")
        logger.info("\n[%s/%s] Processing %s", idx, len(filtered_results), sn_name)

        sn_dir = args.output_dir / sn_name.replace("/", "_")
        sn_dir.mkdir(parents=True, exist_ok=True)

        ref_obs = build_observation_subset(
            sn_entry.get("reference_observations", []), max_obs=args.max_obs
        )
        sci_obs = build_observation_subset(
            sn_entry.get("science_observations", []), max_obs=args.max_obs
        )

        reference_files: list[str] = []
        science_files: list[str] = []
        verified_files: list[dict[str, Any]] = []
        invalid_files: list[dict[str, Any]] = []
        errors: list[str] = []

        if not args.skip_reference and ref_obs:
            ref_downloads, ref_verified, ref_invalid, ref_errors = (
                process_observation_batch(
                    sn_dir,
                    ref_obs,
                    "reference",
                    args.verify_fits,
                    args.dry_run,
                    include_auxiliary=args.include_auxiliary,
                    warning_counts=stats.warning_counts,
                )
            )
            reference_files.extend(ref_downloads)
            verified_files.extend(ref_verified)
            invalid_files.extend(ref_invalid)
            errors.extend(ref_errors)

        if not args.skip_science and sci_obs:
            sci_downloads, sci_verified, sci_invalid, sci_errors = (
                process_observation_batch(
                    sn_dir,
                    sci_obs,
                    "science",
                    args.verify_fits,
                    args.dry_run,
                    include_auxiliary=args.include_auxiliary,
                    warning_counts=stats.warning_counts,
                )
            )
            science_files.extend(sci_downloads)
            verified_files.extend(sci_verified)
            invalid_files.extend(sci_invalid)
            errors.extend(sci_errors)

        stats.reference_files += len(reference_files)
        stats.science_files += len(science_files)
        stats.valid_files += len([item for item in verified_files if item.get("valid")])
        stats.invalid_files += len(invalid_files)
        stats.errors += len(errors)
        if reference_files and science_files:
            stats.complete_pairs += 1

        download_results.append(
            {
                "sn_name": sn_name,
                "reference_files": reference_files,
                "science_files": science_files,
                "verified_files": verified_files,
                "invalid_files": invalid_files,
                "errors": errors,
            }
        )

    results_file = args.output_dir / "download_results.json"
    with open(results_file, "w") as handle:
        json.dump(download_results, handle, indent=2)

    logger.info("\nDownload results saved to: %s", results_file)
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info("Supernovae processed: %s", len(download_results))
    logger.info("  ✅ Complete before/after pairs: %s", stats.complete_pairs)
    logger.info("")
    logger.info("Files downloaded:")
    logger.info("  Reference files: %s", stats.reference_files)
    logger.info("  Science files: %s", stats.science_files)
    logger.info("  Total: %s", stats.reference_files + stats.science_files)
    logger.info("")
    logger.info("FITS validation:")
    logger.info("  ✅ Valid files: %s", stats.valid_files)
    logger.info("  ❌ Invalid files: %s", stats.invalid_files)
    total_checked = stats.valid_files + stats.invalid_files
    success_rate = (stats.valid_files / total_checked * 100) if total_checked else 0.0
    logger.info("  Success rate: %.1f%%", success_rate)
    logger.info("")
    if stats.warning_counts and sum(stats.warning_counts.values()) > 0:
        logger.info("FITS Header Warnings (suppressed during processing):")
        total_warnings = sum(stats.warning_counts.values())
        logger.info("  Total warnings suppressed: %s", total_warnings)
        for warning_type, count in sorted(
            stats.warning_counts.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info("    %s: %s", warning_type, count)
        logger.info("  Note: These warnings are harmless and typically occur due to")
        logger.info("        non-standard metadata formats (especially in PS1 data).")
        logger.info("        Files are still valid and usable for processing.")
        logger.info("")
    logger.info("⚠️  Total errors: %s", stats.errors)
    logger.info("")
    logger.info("Output directory: %s", args.output_dir)
    logger.info("=" * 60)
    if stats.complete_pairs:
        logger.info(
            "\n✅ Ready for differencing: %s supernovae with complete before/after pairs",
            stats.complete_pairs,
        )
        logger.info(
            "   Next step: Run ingest_sn_fits_to_pipeline.py to process these files"
        )
    else:
        logger.info(
            "\n⚠️  No complete before/after pairs were created. Consider relaxing filters."
        )


if __name__ == "__main__":
    main()
