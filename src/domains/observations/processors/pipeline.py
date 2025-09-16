"""FITS processing pipeline integration and result structures."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.wcs import WCS

from ..catalogs.star_catalog import StarCatalog, StarCatalogConfig
from .fits_processor import FITSProcessor
from .metadata_extractor import MetadataExtractor
from .wcs_processor import WCSProcessor

logger = logging.getLogger(__name__)


@dataclass
class FITSProcessingResult:
    """Complete FITS processing pipeline result."""

    # Core data
    image_data: np.ndarray
    wcs_solution: WCS
    metadata: dict[str, Any]
    quality_metrics: dict[str, Any]
    star_catalog_matches: list[dict[str, Any]]
    processing_errors: list[str]
    processing_time: float

    # Additional processing info
    file_path: str | None = None
    processing_config: dict[str, Any] | None = None
    pipeline_version: str = "1.0.0"

    # Quality assessments
    overall_quality_score: float | None = None
    wcs_quality_score: float | None = None
    photometric_quality_score: float | None = None

    # Star catalog information
    catalog_query_radius: float | None = None
    total_catalog_stars: int | None = None
    matched_stars_count: int | None = None


class FITSProcessingPipeline:
    """Integrated FITS processing pipeline combining all components."""

    def __init__(self, star_catalog_config: StarCatalogConfig | None = None):
        """Initialize the FITS processing pipeline.

        Args:
            star_catalog_config: Configuration for star catalog integration
        """
        self.logger = logging.getLogger(__name__)

        # Initialize processors
        self.fits_processor = FITSProcessor()
        self.wcs_processor = WCSProcessor()
        self.metadata_extractor = MetadataExtractor()

        # Initialize star catalog if configured
        self.star_catalog = None
        if star_catalog_config:
            self.star_catalog = StarCatalog(star_catalog_config)

        # Default processing configuration
        self.default_config = {
            "extract_photometric": True,
            "extract_observing_conditions": True,
            "extract_instrument_params": True,
            "extract_quality_metrics": True,
            "extract_astrometric_solution": True,
            "query_star_catalogs": True,
            "match_stars_to_image": True,
            "calculate_quality_scores": True,
            "star_catalog_radius": 300.0,  # arcseconds
        }

    def process_fits_file(
        self, file_path: str, config: dict[str, Any] | None = None
    ) -> FITSProcessingResult:
        """Process a FITS file through the complete pipeline.

        Args:
            file_path: Path to FITS file
            config: Processing configuration options

        Returns:
            FITSProcessingResult with all processing outputs

        Raises:
            ValueError: If file is invalid or processing fails
        """
        start_time = time.time()
        processing_config = {**self.default_config, **(config or {})}
        errors = []

        try:
            self.logger.info(f"Starting FITS processing pipeline for: {file_path}")

            # Step 1: Read and validate FITS file
            try:
                image_data, file_metadata = (
                    self.fits_processor.read_fits_with_validation(file_path)
                )
                self.logger.debug("FITS file read and validated successfully")
            except Exception as e:
                error_msg = f"FITS reading failed: {str(e)}"
                errors.append(error_msg)
                raise ValueError(error_msg) from e

            # Step 2: Extract WCS solution
            try:
                # Get WCS from file metadata
                if "primary_header" in file_metadata:
                    wcs = WCS(file_metadata["primary_header"])
                    wcs_valid = self.wcs_processor.validate_wcs_solution(wcs)

                    if not wcs_valid:
                        errors.append("WCS solution validation failed")
                        wcs = None
                else:
                    wcs = None
                    errors.append("No WCS information found in FITS header")

            except Exception as e:
                wcs = None
                error_msg = f"WCS extraction failed: {str(e)}"
                errors.append(error_msg)
                self.logger.warning(error_msg)

            # Step 3: Extract comprehensive metadata
            metadata = {}
            fits_data = open(file_path, "rb").read()

            if processing_config.get("extract_photometric", True):
                try:
                    metadata["photometric"] = (
                        self.metadata_extractor.extract_photometric_parameters(
                            fits_data
                        )
                    )
                except Exception as e:
                    errors.append(f"Photometric extraction failed: {str(e)}")
                    metadata["photometric"] = {}

            if processing_config.get("extract_observing_conditions", True):
                try:
                    metadata["observing_conditions"] = (
                        self.metadata_extractor.extract_observing_conditions(fits_data)
                    )
                except Exception as e:
                    errors.append(f"Observing conditions extraction failed: {str(e)}")
                    metadata["observing_conditions"] = {}

            if processing_config.get("extract_instrument_params", True):
                try:
                    metadata["instrument"] = (
                        self.metadata_extractor.extract_instrument_parameters(fits_data)
                    )
                except Exception as e:
                    errors.append(f"Instrument parameters extraction failed: {str(e)}")
                    metadata["instrument"] = {}

            if processing_config.get("extract_astrometric_solution", True):
                try:
                    metadata["astrometric"] = (
                        self.metadata_extractor.extract_astrometric_solution(fits_data)
                    )
                except Exception as e:
                    errors.append(f"Astrometric solution extraction failed: {str(e)}")
                    metadata["astrometric"] = {}

            # Add file metadata
            metadata["file_info"] = file_metadata

            # Step 4: Extract quality metrics
            quality_metrics = {}
            if processing_config.get("extract_quality_metrics", True):
                try:
                    quality_metrics = self.metadata_extractor.extract_quality_metrics(
                        fits_data
                    )
                except Exception as e:
                    errors.append(f"Quality metrics extraction failed: {str(e)}")

            # Step 5: Star catalog integration
            star_matches = []
            catalog_query_radius = None
            total_catalog_stars = None

            if (
                self.star_catalog
                and processing_config.get("query_star_catalogs", True)
                and wcs is not None
            ):
                try:
                    # Calculate sky region bounds
                    sky_bounds = self.wcs_processor.calculate_sky_region_bounds(
                        wcs, image_data.shape
                    )

                    # Query star catalogs
                    catalog_query_radius = processing_config.get(
                        "star_catalog_radius", 300.0
                    )
                    catalog_stars = self.star_catalog.query_stars_in_region(
                        ra=sky_bounds["center_ra"],
                        dec=sky_bounds["center_dec"],
                        radius=catalog_query_radius,
                    )
                    total_catalog_stars = len(catalog_stars)

                    # Match stars to image
                    if processing_config.get("match_stars_to_image", True):
                        star_matches = self.star_catalog.match_stars_to_image(
                            catalog_stars, wcs, image_data.shape
                        )

                    self.logger.debug(
                        f"Star catalog integration: {total_catalog_stars} queried, "
                        f"{len(star_matches)} matched"
                    )

                except Exception as e:
                    error_msg = f"Star catalog integration failed: {str(e)}"
                    errors.append(error_msg)
                    self.logger.warning(error_msg)

            # Step 6: Calculate overall quality scores
            wcs_quality_score = None
            photometric_quality_score = None
            overall_quality_score = None

            if processing_config.get("calculate_quality_scores", True):
                try:
                    # WCS quality assessment
                    if wcs is not None:
                        wcs_quality = self.wcs_processor.assess_wcs_quality(
                            wcs, image_data.shape
                        )
                        wcs_quality_score = wcs_quality.get("overall_score")

                    # Photometric quality (simplified)
                    if "photometric" in metadata:
                        phot_data = metadata["photometric"]
                        if phot_data.get("zero_point") is not None:
                            zp_error = phot_data.get("zero_point_error", 0.1)
                            photometric_quality_score = max(0, 1.0 - zp_error / 0.2)

                    # Overall quality (metadata completeness + individual scores)
                    completeness_score = (
                        self.metadata_extractor.calculate_completeness_score(metadata)
                    )
                    quality_scores = [completeness_score]

                    if wcs_quality_score is not None:
                        quality_scores.append(wcs_quality_score)
                    if photometric_quality_score is not None:
                        quality_scores.append(photometric_quality_score)

                    overall_quality_score = np.mean(quality_scores)

                except Exception as e:
                    error_msg = f"Quality score calculation failed: {str(e)}"
                    errors.append(error_msg)
                    self.logger.warning(error_msg)

            # Create processing result
            processing_time = time.time() - start_time

            result = FITSProcessingResult(
                image_data=image_data,
                wcs_solution=wcs,
                metadata=metadata,
                quality_metrics=quality_metrics,
                star_catalog_matches=star_matches,
                processing_errors=errors,
                processing_time=processing_time,
                file_path=file_path,
                processing_config=processing_config,
                overall_quality_score=overall_quality_score,
                wcs_quality_score=wcs_quality_score,
                photometric_quality_score=photometric_quality_score,
                catalog_query_radius=catalog_query_radius,
                total_catalog_stars=total_catalog_stars,
                matched_stars_count=len(star_matches),
            )

            self.logger.info(
                f"FITS processing completed: {file_path} "
                f"({processing_time:.3f}s, quality={overall_quality_score:.2f})"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"FITS processing pipeline failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)

            # Return partial result even on failure
            return FITSProcessingResult(
                image_data=np.array([]),
                wcs_solution=None,
                metadata={},
                quality_metrics={},
                star_catalog_matches=[],
                processing_errors=errors,
                processing_time=processing_time,
                file_path=file_path,
                processing_config=processing_config,
            )

    def validate_fits_file(self, file_path: str) -> dict[str, Any]:
        """Quick validation of FITS file without full processing.

        Args:
            file_path: Path to FITS file

        Returns:
            dict: Validation results and basic file info
        """
        try:
            # Basic structure validation
            structure_valid = self.fits_processor.validate_fits_structure(file_path)

            # Integrity check
            integrity_results = self.fits_processor.verify_fits_integrity(file_path)

            # Quick metadata peek
            headers = self.fits_processor.extract_all_headers(file_path)

            validation_result = {
                "is_valid": structure_valid
                and integrity_results.get("data_readable", False),
                "structure_validation": structure_valid,
                "integrity_check": integrity_results,
                "header_count": len(headers),
                "has_image_data": any(
                    hdu_info.get("data_info", {}).get("has_data", False)
                    for hdu_info in headers.values()
                ),
                "validation_time": time.time(),
            }

            # Add basic image info if available
            primary_hdu = headers.get("HDU_0", {})
            if primary_hdu.get("data_info", {}).get("has_data"):
                validation_result["image_shape"] = primary_hdu["data_info"]["shape"]
                validation_result["image_dtype"] = primary_hdu["data_info"]["dtype"]

            return validation_result

        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "validation_time": time.time(),
            }

    def batch_process_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.fits",
        config: dict[str, Any] | None = None,
    ) -> list[FITSProcessingResult]:
        """Process all FITS files in a directory.

        Args:
            directory_path: Directory containing FITS files
            file_pattern: File pattern to match (e.g., "*.fits", "*.fit")
            config: Processing configuration

        Returns:
            List of FITSProcessingResult objects
        """
        from pathlib import Path

        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")

            # Find matching files
            fits_files = list(directory.glob(file_pattern))

            if not fits_files:
                self.logger.warning(
                    f"No files matching {file_pattern} found in {directory_path}"
                )
                return []

            self.logger.info(
                f"Processing {len(fits_files)} FITS files from {directory_path}"
            )

            results = []
            for fits_file in fits_files:
                try:
                    result = self.process_fits_file(str(fits_file), config)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process {fits_file}: {e}")
                    # Add failed result
                    results.append(
                        FITSProcessingResult(
                            image_data=np.array([]),
                            wcs_solution=None,
                            metadata={},
                            quality_metrics={},
                            star_catalog_matches=[],
                            processing_errors=[f"Processing failed: {str(e)}"],
                            processing_time=0.0,
                            file_path=str(fits_file),
                        )
                    )

            return results

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise

    def get_pipeline_statistics(
        self, results: list[FITSProcessingResult]
    ) -> dict[str, Any]:
        """Calculate statistics across multiple processing results.

        Args:
            results: List of processing results

        Returns:
            dict: Pipeline statistics and summaries
        """
        if not results:
            return {}

        try:
            stats = {
                "total_files": len(results),
                "successful_files": sum(1 for r in results if not r.processing_errors),
                "failed_files": sum(1 for r in results if r.processing_errors),
                "total_processing_time": sum(r.processing_time for r in results),
                "average_processing_time": np.mean(
                    [r.processing_time for r in results]
                ),
                "quality_scores": {
                    "overall": [
                        r.overall_quality_score
                        for r in results
                        if r.overall_quality_score is not None
                    ],
                    "wcs": [
                        r.wcs_quality_score
                        for r in results
                        if r.wcs_quality_score is not None
                    ],
                    "photometric": [
                        r.photometric_quality_score
                        for r in results
                        if r.photometric_quality_score is not None
                    ],
                },
                "star_catalog_stats": {
                    "total_queried": sum(r.total_catalog_stars or 0 for r in results),
                    "total_matched": sum(r.matched_stars_count or 0 for r in results),
                    "average_matches_per_image": np.mean(
                        [r.matched_stars_count or 0 for r in results]
                    ),
                },
                "common_errors": {},
            }

            # Calculate quality score statistics
            for quality_type, scores in stats["quality_scores"].items():
                if scores:
                    stats["quality_scores"][f"{quality_type}_mean"] = np.mean(scores)
                    stats["quality_scores"][f"{quality_type}_std"] = np.std(scores)
                    stats["quality_scores"][f"{quality_type}_min"] = np.min(scores)
                    stats["quality_scores"][f"{quality_type}_max"] = np.max(scores)

            # Analyze common errors
            all_errors = []
            for result in results:
                all_errors.extend(result.processing_errors)

            if all_errors:
                from collections import Counter

                error_counts = Counter(all_errors)
                stats["common_errors"] = dict(error_counts.most_common(10))

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating pipeline statistics: {e}")
            return {"error": str(e)}
