#!/usr/bin/env python3
"""
Ingest downloaded supernova FITS files into the AstrID pipeline.

Enhancements:
* Accepts the download manifest (`download_results.json`) produced by the downloader
* Handles `.fits.gz` and `.img` files in addition to plain `.fits`
* Optional WCS verification before ingesting to avoid downstream failures
* Flexible CLI that works even if the original query JSON is unavailable
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from astropy.io import fits
from astropy.wcs import WCS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".fits", ".fits.gz", ".fit", ".img")


def verify_fits_file(file_path: Path) -> tuple[bool, str | None]:
    """Verify the FITS file has celestial WCS coordinates."""
    try:
        with fits.open(file_path) as hdul:
            header = hdul[0].header
            try:
                wcs = WCS(header)
                if wcs.has_celestial:
                    return True, None
                return False, "Missing celestial WCS"
            except Exception as exc:  # pragma: no cover - defensive
                return False, f"WCS parse error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Failed to open FITS: {exc}"


def resolve_file(path_value: str, base_dir: Path) -> Path:
    """Resolve a relative or absolute path to a Path instance."""
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def files_from_manifest(
    entry: dict[str, Any], base_dir: Path
) -> tuple[list[Path], list[Path]]:
    """Extract reference/science files from a download manifest entry."""
    reference_files = [
        resolve_file(path_str, base_dir)
        for path_str in entry.get("reference_files", [])
    ]
    science_files = [
        resolve_file(path_str, base_dir) for path_str in entry.get("science_files", [])
    ]
    return reference_files, science_files


def discover_files_on_disk(
    sn_name: str, base_dir: Path
) -> tuple[list[Path], list[Path]]:
    """Fallback discovery by scanning the filesystem."""
    sn_dir = base_dir / sn_name.replace("/", "_")
    reference_files: list[Path] = []
    science_files: list[Path] = []

    ref_dir = sn_dir / "reference"
    if ref_dir.exists():
        reference_files = sorted(
            [
                path
                for path in ref_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )

    sci_dir = sn_dir / "science"
    if sci_dir.exists():
        science_files = sorted(
            [
                path
                for path in sci_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )

    return reference_files, science_files


def load_download_manifest(
    path: Path | None, base_dir: Path
) -> dict[str, dict[str, Any]]:
    """Load download_results.json if it exists."""
    manifest_path = path or (base_dir / "download_results.json")
    if manifest_path and manifest_path.exists():
        with open(manifest_path) as handle:
            manifest_list = json.load(handle)
        return {entry["sn_name"]: entry for entry in manifest_list}
    return {}


async def ingest_supernova_pair(
    sn_name: str,
    reference_files: list[Path],
    science_files: list[Path],
    survey_id: UUID,
    verify_wcs: bool,
) -> dict[str, Any]:
    """Ingest a supernova's reference and science images into the pipeline."""
    from src.adapters.database.session import AsyncSessionLocal
    from src.adapters.storage.r2_client import R2Client
    from src.adapters.workers.preprocessing.preprocessing_workers import (
        preprocess_observation,
    )
    from src.domains.observations.ingestion.services.data_ingestion import (
        DataIngestionService,
    )
    from src.domains.observations.schema import ObservationCreate
    from src.domains.observations.service import ObservationService

    logger.info(f"\n{'='*60}")
    logger.info("Ingesting %s", sn_name)
    logger.info("%s", "=" * 60)
    logger.info("Reference files: %s", len(reference_files))
    logger.info("Science files: %s", len(science_files))

    result: dict[str, Any] = {
        "sn_name": sn_name,
        "reference_observations": [],
        "science_observations": [],
        "errors": [],
    }

    ingestion_service = DataIngestionService()
    r2_client = R2Client()

    async def _ingest_file(file_path: Path, obs_type: str) -> None:
        if not file_path.exists():
            msg = f"{obs_type} file not found: {file_path}"
            logger.warning(msg)
            result["errors"].append(msg)
            return

        if verify_wcs:
            valid, reason = verify_fits_file(file_path)
            if not valid:
                msg = f"Skipped {file_path.name} - {reason}"
                logger.warning(msg)
                result["errors"].append(msg)
                return

        logger.info("Ingesting %s: %s", obs_type, file_path.name)

        with fits.open(file_path) as hdul:
            header = hdul[0].header
            ra = header.get("RA", header.get("CRVAL1", 0.0))
            dec = header.get("DEC", header.get("CRVAL2", 0.0))
            filter_band = header.get("FILTER", header.get("FILT", "r"))
            exposure_time = header.get("EXPTIME", header.get("EXPOSURE", 300.0))

        with open(file_path, "rb") as fh:
            fits_data = fh.read()

        r2_path = f"observations/{sn_name}/{obs_type}/{file_path.name}"
        await r2_client.upload_file(r2_path, fits_data, content_type="application/fits")

        observation_id = f"{sn_name}_{obs_type}_{file_path.stem}"

        async with AsyncSessionLocal() as db:
            obs_service = ObservationService(db)
            obs_create = ObservationCreate(
                survey_id=survey_id,
                observation_id=observation_id,
                ra=float(ra),
                dec=float(dec),
                observation_time=None,
                filter_band=str(filter_band),
                exposure_time=float(exposure_time),
                fits_url=r2_path,
            )
            observation = await obs_service.create_observation(obs_create)
            await ingestion_service.ingest_observations_by_position(
                ra=float(ra),
                dec=float(dec),
                survey_id=survey_id,
                radius=0.01,
            )

            if obs_type == "science":
                await obs_service.mark_observation_as_ingested(observation.id)
                preprocess_observation.send(str(observation.id))

            result[f"{obs_type}_observations"].append(
                {
                    "observation_id": str(observation.id),
                    "file": str(file_path),
                    "fits_url": r2_path,
                }
            )
            logger.info("✅ Ingested %s observation: %s", obs_type, observation.id)

    for ref_file in reference_files:
        await _ingest_file(ref_file, "reference")
    for sci_file in science_files:
        await _ingest_file(sci_file, "science")

    return result


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest downloaded supernova FITS files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/fits"),
        help="Directory containing downloaded FITS files (default: data/fits)",
    )
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Path to download_results.json (default: <input-dir>/download_results.json)",
    )
    parser.add_argument(
        "--query-results",
        type=Path,
        default=None,
        help="Optional query JSON for reference (used if manifest missing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for ingestion_results.json (default: input dir)",
    )
    parser.add_argument(
        "--survey-id",
        type=str,
        required=True,
        help="Survey UUID in the database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of supernovae to ingest",
    )
    parser.add_argument(
        "--filter-has-both",
        action="store_true",
        help="Only ingest supernovae with both reference and science files",
    )
    parser.add_argument(
        "--verify-wcs",
        action="store_true",
        help="Verify FITS files contain WCS before ingesting",
    )

    args = parser.parse_args()

    base_dir = args.input_dir
    if args.fits_dir:
        base_dir = args.fits_dir

    if not base_dir.exists():
        logger.error("Input directory does not exist: %s", base_dir)
        raise SystemExit(1)

    download_manifest = load_download_manifest(args.results_file, base_dir)

    query_results: list[dict[str, Any]] = []
    if args.query_results and args.query_results.exists():
        with open(args.query_results) as handle:
            query_results = json.load(handle)

    # Determine SN order of operations
    if download_manifest:
        sn_queue = list(download_manifest.keys())
    elif query_results:
        sn_queue = [entry["sn_name"] for entry in query_results]
    else:
        sn_queue = [path.name for path in base_dir.iterdir() if path.is_dir()]

    if args.limit:
        sn_queue = sn_queue[: args.limit]

    try:
        survey_id = UUID(args.survey_id)
    except ValueError as exc:
        logger.error("Invalid survey UUID: %s", exc)
        raise SystemExit(1) from exc

    ingestion_results: list[dict[str, Any]] = []

    for idx, sn_name in enumerate(sn_queue, 1):
        logger.info("\n[%s/%s] Preparing %s", idx, len(sn_queue), sn_name)
        manifest_entry = download_manifest.get(sn_name)

        if manifest_entry:
            reference_files, science_files = files_from_manifest(
                manifest_entry, base_dir
            )
        else:
            reference_files, science_files = discover_files_on_disk(sn_name, base_dir)

        if not reference_files and not science_files:
            logger.warning("  ❌ No files found for %s under %s", sn_name, base_dir)
            continue

        if args.filter_has_both and (not reference_files or not science_files):
            logger.warning(
                "  ⚠️  Skipping %s because it does not have both reference and science files",
                sn_name,
            )
            continue

        logger.info("  Reference files ready: %s", len(reference_files))
        logger.info("  Science files ready: %s", len(science_files))

        ingestion_result = await ingest_supernova_pair(
            sn_name=sn_name,
            reference_files=reference_files,
            science_files=science_files,
            survey_id=survey_id,
            verify_wcs=args.verify_wcs,
        )
        ingestion_results.append(ingestion_result)

    output_dir = args.output_dir or base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "ingestion_results.json"
    with open(results_file, "w") as handle:
        json.dump(ingestion_results, handle, indent=2, default=str)

    logger.info("\nIngestion results saved to: %s", results_file)

    total_ref = sum(len(entry["reference_observations"]) for entry in ingestion_results)
    total_sci = sum(len(entry["science_observations"]) for entry in ingestion_results)
    total_errors = sum(len(entry["errors"]) for entry in ingestion_results)

    logger.info("\n" + "=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Supernovae processed: %s", len(ingestion_results))
    logger.info("Reference observations ingested: %s", total_ref)
    logger.info("Science observations ingested: %s", total_sci)
    logger.info("Total errors: %s", total_errors)
    logger.info("=" * 60)
    logger.info(
        "\n✅ Observations have been queued for preprocessing → differencing → ML pipeline"
    )
