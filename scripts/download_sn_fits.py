#!/usr/bin/env python3
"""
Download FITS files for supernovae based on MAST query results.

This script reads the JSON output from query_sn_fits_from_catalog.py and downloads
the actual FITS files for reference and science images.

Usage:
    python scripts/download_sn_fits.py --query-results output/sn_mast_queries.json --output-dir data/fits
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_observation_fits(
    obs_id: str,
    mission: str,
    output_dir: Path,
    product_type: str = "SCIENCE",
) -> list[Path]:
    """Download FITS files for a single observation.
    
    Args:
        obs_id: MAST observation ID
        mission: Mission name (e.g., 'HST', 'JWST')
        output_dir: Directory to save FITS files
        product_type: Type of product to download ('SCIENCE', 'DRZ', etc.)
    
    Returns:
        List of downloaded file paths
    """
    try:
        from astroquery.mast import Observations
        
        logger.info(f"Getting data products for {mission} observation: {obs_id}")
        
        # Get list of available products
        products = Observations.get_product_list(obs_id)
        
        if len(products) == 0:
            logger.warning(f"No data products found for {obs_id}")
            return []
        
        # Filter for science products (FITS files)
        # Look for science products, drizzled images, or FITS files
        science_products = products[
            (products['productSubGroupDescription'] == 'SCI') |
            (products['productSubGroupDescription'] == 'DRZ') |
            (products['productSubGroupDescription'].str.contains('FITS', case=False, na=False))
        ]
        
        if len(science_products) == 0:
            # Fallback: take first product that's a FITS file
            fits_products = products[products['productFilename'].str.endswith('.fits', case=False, na=False)]
            if len(fits_products) > 0:
                science_products = fits_products[:1]
                logger.info(f"Using first available FITS product for {obs_id}")
            else:
                logger.warning(f"No FITS products found for {obs_id}")
                return []
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download products
        logger.info(f"Downloading {len(science_products)} products for {obs_id}")
        manifest = Observations.download_products(
            science_products,
            download_dir=str(output_dir),
            mrp_only=False  # Download all products, not just minimum recommended
        )
        
        # Find downloaded files
        downloaded_files = []
        if 'Local Path' in manifest.columns:
            for path in manifest['Local Path']:
                if path and Path(path).exists():
                    downloaded_files.append(Path(path))
        else:
            # Search for downloaded files
            for product in science_products:
                filename = product.get('productFilename', '')
                if filename:
                    file_path = output_dir / filename
                    if file_path.exists():
                        downloaded_files.append(file_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} files for {obs_id}")
        return downloaded_files
        
    except ImportError:
        logger.error("astroquery.mast not available. Install with: pip install astroquery")
        return []
    except Exception as e:
        logger.error(f"Error downloading {obs_id}: {e}")
        return []


async def download_supernova_fits(
    sn_result: dict[str, Any],
    output_base_dir: Path,
    download_reference: bool = True,
    download_science: bool = True,
    max_obs_per_type: int = 5,
) -> dict[str, Any]:
    """Download FITS files for a single supernova.
    
    Args:
        sn_result: Supernova query result from query_sn_fits_from_catalog.py
        output_base_dir: Base directory for downloads
        download_reference: Whether to download reference images
        download_science: Whether to download science images
        max_obs_per_type: Maximum observations to download per type
    
    Returns:
        Dictionary with download results
    """
    sn_name = sn_result["sn_name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {sn_name}")
    logger.info(f"{'='*60}")
    
    result = {
        "sn_name": sn_name,
        "reference_files": [],
        "science_files": [],
        "errors": [],
    }
    
    # Create subdirectory for this supernova
    sn_dir = output_base_dir / sn_name.replace("/", "_")
    sn_dir.mkdir(parents=True, exist_ok=True)
    
    # Download reference images
    if download_reference:
        ref_obs = sn_result.get("reference_observations", [])[:max_obs_per_type]
        logger.info(f"Downloading {len(ref_obs)} reference observations")
        
        ref_dir = sn_dir / "reference"
        for i, obs in enumerate(ref_obs, 1):
            obs_id = obs.get("obs_id", "")
            mission = obs.get("mission", "Unknown")
            
            logger.info(f"[{i}/{len(ref_obs)}] Reference: {mission} {obs_id}")
            
            try:
                files = download_observation_fits(obs_id, mission, ref_dir)
                result["reference_files"].extend([str(f) for f in files])
            except Exception as e:
                error_msg = f"Failed to download reference {obs_id}: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
    
    # Download science images
    if download_science:
        sci_obs = sn_result.get("science_observations", [])[:max_obs_per_type]
        logger.info(f"Downloading {len(sci_obs)} science observations")
        
        sci_dir = sn_dir / "science"
        for i, obs in enumerate(sci_obs, 1):
            obs_id = obs.get("obs_id", "")
            mission = obs.get("mission", "Unknown")
            
            logger.info(f"[{i}/{len(sci_obs)}] Science: {mission} {obs_id}")
            
            try:
                files = download_observation_fits(obs_id, mission, sci_dir)
                result["science_files"].extend([str(f) for f in files])
            except Exception as e:
                error_msg = f"Failed to download science {obs_id}: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
    
    # Summary
    logger.info(f"{sn_name}: {len(result['reference_files'])} reference, {len(result['science_files'])} science files")
    
    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download FITS files for supernovae from MAST query results"
    )
    parser.add_argument(
        "--query-results",
        type=Path,
        required=True,
        help="JSON file from query_sn_fits_from_catalog.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fits"),
        help="Base directory for downloaded FITS files",
    )
    parser.add_argument(
        "--max-obs",
        type=int,
        default=5,
        help="Maximum observations to download per type (reference/science) per supernova",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip downloading reference images",
    )
    parser.add_argument(
        "--skip-science",
        action="store_true",
        help="Skip downloading science images",
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
        help="Only process supernovae with both reference and science observations",
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
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download FITS files for each supernova
    download_results = []
    for i, sn_result in enumerate(query_results, 1):
        logger.info(f"\n[{i}/{len(query_results)}] Processing supernova")
        
        result = await download_supernova_fits(
            sn_result,
            args.output_dir,
            download_reference=not args.skip_reference,
            download_science=not args.skip_science,
            max_obs_per_type=args.max_obs,
        )
        download_results.append(result)
    
    # Save download results
    results_file = args.output_dir / "download_results.json"
    with open(results_file, "w") as f:
        json.dump(download_results, f, indent=2, default=str)
    
    logger.info(f"\nDownload results saved to: {results_file}")
    
    # Print summary
    total_ref = sum(len(r["reference_files"]) for r in download_results)
    total_sci = sum(len(r["science_files"]) for r in download_results)
    total_errors = sum(len(r["errors"]) for r in download_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Supernovae processed: {len(download_results)}")
    logger.info(f"Total reference files: {total_ref}")
    logger.info(f"Total science files: {total_sci}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

