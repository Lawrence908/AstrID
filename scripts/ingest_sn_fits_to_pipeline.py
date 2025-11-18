#!/usr/bin/env python3
"""
Ingest downloaded supernova FITS files into the AstrID pipeline.

This script bridges the gap between:
1. SN catalog querying/downloading (query_sn_fits_from_catalog.py, download_sn_fits.py)
2. The main AstrID observation ingestion pipeline

It takes downloaded FITS files and:
- Stores them in R2
- Creates observation records in the database
- Triggers preprocessing → differencing → ML pipeline

Usage:
    python scripts/ingest_sn_fits_to_pipeline.py \
        --query-results output/sn_queries_all_missions.json \
        --fits-dir data/fits \
        --survey-id <survey-uuid>
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def ingest_supernova_pair(
    sn_name: str,
    reference_files: list[str],
    science_files: list[str],
    survey_id: UUID,
    fits_base_dir: Path,
) -> dict[str, Any]:
    """Ingest a supernova's reference and science images into the pipeline.
    
    Args:
        sn_name: Supernova name
        reference_files: List of reference FITS file paths
        science_files: List of science FITS file paths
        survey_id: Survey UUID in the database
        fits_base_dir: Base directory containing FITS files
    
    Returns:
        Dictionary with ingestion results
    """
    from src.domains.observations.ingestion.services.data_ingestion import (
        DataIngestionService,
    )
    from src.adapters.storage.r2_client import R2Client
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Ingesting {sn_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Reference files: {len(reference_files)}")
    logger.info(f"Science files: {len(science_files)}")
    
    result = {
        "sn_name": sn_name,
        "reference_observations": [],
        "science_observations": [],
        "errors": [],
    }
    
    ingestion_service = DataIngestionService()
    r2_client = R2Client()
    
    # Ingest reference images
    for ref_file in reference_files:
        ref_path = fits_base_dir / ref_file if not Path(ref_file).is_absolute() else Path(ref_file)
        
        if not ref_path.exists():
            error_msg = f"Reference file not found: {ref_path}"
            logger.warning(error_msg)
            result["errors"].append(error_msg)
            continue
        
        try:
            logger.info(f"Ingesting reference: {ref_path.name}")
            
            # Read FITS metadata
            from astropy.io import fits
            with fits.open(ref_path) as hdul:
                header = hdul[0].header
                
                # Extract metadata
                ra = header.get("RA", header.get("CRVAL1", 0.0))
                dec = header.get("DEC", header.get("CRVAL2", 0.0))
                obs_time = header.get("DATE-OBS", header.get("MJD-OBS", ""))
                filter_band = header.get("FILTER", header.get("FILT", "r"))
                exposure_time = header.get("EXPTIME", header.get("EXPOSURE", 300.0))
            
            # Store in R2
            with open(ref_path, "rb") as f:
                fits_data = f.read()
            
            r2_path = f"observations/{sn_name}/reference/{ref_path.name}"
            await r2_client.upload_file(r2_path, fits_data, content_type="application/fits")
            
            # Create observation record
            observation_data = {
                "survey_id": str(survey_id),
                "observation_id": f"{sn_name}_ref_{ref_path.stem}",
                "ra": float(ra),
                "dec": float(dec),
                "observation_time": obs_time if obs_time else None,
                "filter_band": str(filter_band),
                "exposure_time": float(exposure_time),
                "fits_url": r2_path,
                "observation_type": "reference",
                "sn_name": sn_name,
            }
            
            # Use ingestion service to create observation
            obs_create = await ingestion_service.ingest_observations_by_position(
                ra=float(ra),
                dec=float(dec),
                survey_id=survey_id,
                radius=0.01,  # Small radius for exact match
            )
            
            # For now, create observation directly
            from src.domains.observations.schema import ObservationCreate
            from src.domains.observations.service import ObservationService
            from src.adapters.database.session import AsyncSessionLocal
            
            async with AsyncSessionLocal() as db:
                obs_service = ObservationService(db)
                obs_create_obj = ObservationCreate(
                    survey_id=survey_id,
                    observation_id=observation_data["observation_id"],
                    ra=float(ra),
                    dec=float(dec),
                    observation_time=None,  # Will need to parse obs_time
                    filter_band=str(filter_band),
                    exposure_time=float(exposure_time),
                    fits_url=r2_path,
                )
                
                observation = await obs_service.create_observation(obs_create_obj)
                result["reference_observations"].append({
                    "observation_id": str(observation.id),
                    "file": str(ref_path),
                })
                logger.info(f"✅ Ingested reference observation: {observation.id}")
        
        except Exception as e:
            error_msg = f"Failed to ingest reference {ref_path.name}: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
    
    # Ingest science images
    for sci_file in science_files:
        sci_path = fits_base_dir / sci_file if not Path(sci_file).is_absolute() else Path(sci_file)
        
        if not sci_path.exists():
            error_msg = f"Science file not found: {sci_path}"
            logger.warning(error_msg)
            result["errors"].append(error_msg)
            continue
        
        try:
            logger.info(f"Ingesting science: {sci_path.name}")
            
            # Read FITS metadata
            from astropy.io import fits
            with fits.open(sci_path) as hdul:
                header = hdul[0].header
                
                # Extract metadata
                ra = header.get("RA", header.get("CRVAL1", 0.0))
                dec = header.get("DEC", header.get("CRVAL2", 0.0))
                obs_time = header.get("DATE-OBS", header.get("MJD-OBS", ""))
                filter_band = header.get("FILTER", header.get("FILT", "r"))
                exposure_time = header.get("EXPTIME", header.get("EXPOSURE", 300.0))
            
            # Store in R2
            with open(sci_path, "rb") as f:
                fits_data = f.read()
            
            r2_path = f"observations/{sn_name}/science/{sci_path.name}"
            await r2_client.upload_file(r2_path, fits_data, content_type="application/fits")
            
            # Create observation record
            from src.domains.observations.schema import ObservationCreate
            from src.domains.observations.service import ObservationService
            from src.adapters.database.session import AsyncSessionLocal
            
            async with AsyncSessionLocal() as db:
                obs_service = ObservationService(db)
                obs_create_obj = ObservationCreate(
                    survey_id=survey_id,
                    observation_id=f"{sn_name}_sci_{sci_path.stem}",
                    ra=float(ra),
                    dec=float(dec),
                    observation_time=None,  # Will need to parse obs_time
                    filter_band=str(filter_band),
                    exposure_time=float(exposure_time),
                    fits_url=r2_path,
                )
                
                observation = await obs_service.create_observation(obs_create_obj)
                
                # Mark as ingested and trigger preprocessing
                await obs_service.mark_observation_as_ingested(observation.id)
                
                # Trigger preprocessing (which will trigger differencing → ML)
                from src.adapters.workers.preprocessing.preprocessing_workers import (
                    preprocess_observation,
                )
                preprocess_observation.send(str(observation.id))
                
                result["science_observations"].append({
                    "observation_id": str(observation.id),
                    "file": str(sci_path),
                    "preprocessing_triggered": True,
                })
                logger.info(f"✅ Ingested science observation: {observation.id} (preprocessing queued)")
        
        except Exception as e:
            error_msg = f"Failed to ingest science {sci_path.name}: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
    
    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest supernova FITS files into AstrID pipeline"
    )
    parser.add_argument(
        "--query-results",
        type=Path,
        required=True,
        help="JSON file from query_sn_fits_from_catalog.py",
    )
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=Path("data/fits"),
        help="Directory containing downloaded FITS files",
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
        help="Limit number of supernovae to process",
    )
    parser.add_argument(
        "--filter-has-both",
        action="store_true",
        help="Only process supernovae with both reference and science images",
    )
    
    args = parser.parse_args()
    
    # Load query results
    if not args.query_results.exists():
        logger.error(f"Query results file not found: {args.query_results}")
        return
    
    logger.info(f"Loading query results from: {args.query_results}")
    with open(args.query_results) as f:
        query_results = json.load(f)
    
    logger.info(f"Loaded {len(query_results)} supernova query results")
    
    # Filter results if requested
    if args.filter_has_both:
        filtered = []
        for result in query_results:
            ref_count = len(result.get("reference_observations", []))
            sci_count = len(result.get("science_observations", []))
            if ref_count > 0 and sci_count > 0:
                filtered.append(result)
        query_results = filtered
        logger.info(f"Filtered to {len(query_results)} supernovae with both reference and science observations")
    
    # Limit if specified
    if args.limit:
        query_results = query_results[:args.limit]
        logger.info(f"Limited to {args.limit} supernovae")
    
    # Parse survey ID
    try:
        survey_id = UUID(args.survey_id)
    except ValueError:
        logger.error(f"Invalid survey UUID: {args.survey_id}")
        return
    
    # Process each supernova
    ingestion_results = []
    for i, sn_result in enumerate(query_results, 1):
        sn_name = sn_result["sn_name"]
        logger.info(f"\n[{i}/{len(query_results)}] Processing {sn_name}")
        
        # Get file paths from download results if available
        # Otherwise, construct paths from observation IDs
        reference_files = []
        science_files = []
        
        # Try to find downloaded files
        sn_dir = args.fits_dir / sn_name.replace("/", "_")
        if sn_dir.exists():
            ref_dir = sn_dir / "reference"
            sci_dir = sn_dir / "science"
            
            if ref_dir.exists():
                reference_files = [str(f.relative_to(args.fits_dir)) for f in ref_dir.glob("*.fits")]
            if sci_dir.exists():
                science_files = [str(f.relative_to(args.fits_dir)) for f in sci_dir.glob("*.fits")]
        
        if not reference_files and not science_files:
            logger.warning(f"No FITS files found for {sn_name} in {sn_dir}")
            logger.info("Tip: Run download_sn_fits.py first to download the files")
            continue
        
        result = await ingest_supernova_pair(
            sn_name=sn_name,
            reference_files=reference_files,
            science_files=science_files,
            survey_id=survey_id,
            fits_base_dir=args.fits_dir,
        )
        ingestion_results.append(result)
    
    # Save results
    results_file = args.fits_dir / "ingestion_results.json"
    with open(results_file, "w") as f:
        json.dump(ingestion_results, f, indent=2, default=str)
    
    logger.info(f"\nIngestion results saved to: {results_file}")
    
    # Print summary
    total_ref = sum(len(r["reference_observations"]) for r in ingestion_results)
    total_sci = sum(len(r["science_observations"]) for r in ingestion_results)
    total_errors = sum(len(r["errors"]) for r in ingestion_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Supernovae processed: {len(ingestion_results)}")
    logger.info(f"Reference observations ingested: {total_ref}")
    logger.info(f"Science observations ingested: {total_sci}")
    logger.info(f"Total errors: {total_errors}")
    logger.info("=" * 60)
    logger.info("\n✅ Observations have been queued for preprocessing → differencing → ML pipeline")


if __name__ == "__main__":
    asyncio.run(main())

