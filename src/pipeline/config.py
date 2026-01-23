"""Pipeline configuration system with YAML support.

This module provides dataclasses for configuring the supernova data acquisition
pipeline, with validation and YAML loading capabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class QueryConfig:
    """Configuration for archive queries (MAST and/or IRSA)."""

    missions: list[str]
    filters: list[str] | None = None
    archives: list[str] | None = None  # Optional: ["MAST", "IRSA"] or None for auto-routing
    min_year: int = 2005
    max_year: int | None = None
    days_before: int = 1095  # ~3 years for reference images
    days_after: int = 730  # ~2 years for science images
    radius_deg: float = 0.1
    chunk_size: int = 250
    start_index: int = 0
    limit: int | None = None

    def __post_init__(self) -> None:
        """Validate query configuration."""
        if not self.missions:
            raise ValueError("At least one mission must be specified")
        if self.min_year < 1900 or self.min_year > 2100:
            raise ValueError(f"Invalid min_year: {self.min_year}")
        if self.max_year is not None and self.max_year < self.min_year:
            raise ValueError(
                f"max_year ({self.max_year}) must be >= min_year ({self.min_year})"
            )
        if self.days_before < 0 or self.days_after < 0:
            raise ValueError("days_before and days_after must be non-negative")
        if self.radius_deg <= 0:
            raise ValueError("radius_deg must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


@dataclass
class DownloadConfig:
    """Configuration for FITS file downloads."""

    max_obs_per_type: int = 5
    max_products_per_obs: int = 3
    include_auxiliary: bool = False
    require_same_mission: bool = True
    verify_fits: bool = True
    skip_reference: bool = False
    skip_science: bool = False

    def __post_init__(self) -> None:
        """Validate download configuration."""
        if self.max_obs_per_type <= 0:
            raise ValueError("max_obs_per_type must be positive")
        if self.max_products_per_obs <= 0:
            raise ValueError("max_products_per_obs must be positive")


@dataclass
class QualityConfig:
    """Configuration for quality filtering."""

    min_overlap_fraction: float = 0.85
    max_file_size_mb: float = 500.0
    verify_wcs: bool = True

    def __post_init__(self) -> None:
        """Validate quality configuration."""
        if not 0.0 <= self.min_overlap_fraction <= 1.0:
            raise ValueError(
                f"min_overlap_fraction must be between 0 and 1, got {self.min_overlap_fraction}"
            )
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")


@dataclass
class OutputConfig:
    """Configuration for output paths."""

    query_results: Path
    fits_downloads: Path
    fits_training: Path
    difference_images: Path
    checkpoint: Path | None = None
    chunk_dir: Path | None = None

    def __post_init__(self) -> None:
        """Convert string paths to Path objects and set defaults."""
        # Convert strings to Path if needed
        if isinstance(self.query_results, str):
            object.__setattr__(self, "query_results", Path(self.query_results))
        if isinstance(self.fits_downloads, str):
            object.__setattr__(self, "fits_downloads", Path(self.fits_downloads))
        if isinstance(self.fits_training, str):
            object.__setattr__(self, "fits_training", Path(self.fits_training))
        if isinstance(self.difference_images, str):
            object.__setattr__(
                self, "difference_images", Path(self.difference_images)
            )
        if isinstance(self.checkpoint, str):
            object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        if isinstance(self.chunk_dir, str):
            object.__setattr__(self, "chunk_dir", Path(self.chunk_dir))

        # Set defaults if not provided
        if self.checkpoint is None:
            object.__setattr__(
                self, "checkpoint", self.query_results.parent / "checkpoint.json"
            )
        if self.chunk_dir is None:
            object.__setattr__(
                self, "chunk_dir", self.query_results.parent / "chunks"
            )


@dataclass
class PipelineConfig:
    """Complete pipeline configuration.

    This is the main configuration class that combines all sub-configurations
    for the entire data acquisition pipeline.
    """

    dataset_name: str
    description: str = ""
    query: QueryConfig = field(default_factory=QueryConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            PipelineConfig instance with loaded values

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a YAML dictionary")

        # Extract top-level fields
        dataset_name = data.get("dataset_name", "unnamed_dataset")
        description = data.get("description", "")

        # Build QueryConfig
        query_data = data.get("query", {})
        query_config = QueryConfig(
            missions=query_data.get("missions", ["SWIFT", "PS1", "GALEX"]),
            filters=query_data.get("filters"),
            archives=query_data.get("archives"),  # Optional: auto-routed if None
            min_year=query_data.get("min_year", 2005),
            max_year=query_data.get("max_year"),
            days_before=query_data.get("days_before", 1095),
            days_after=query_data.get("days_after", 730),
            radius_deg=query_data.get("radius_deg", 0.1),
            chunk_size=query_data.get("chunk_size", 250),
            start_index=query_data.get("start_index", 0),
            limit=query_data.get("limit"),
        )

        # Build DownloadConfig
        download_data = data.get("download", {})
        download_config = DownloadConfig(
            max_obs_per_type=download_data.get("max_obs_per_type", 5),
            max_products_per_obs=download_data.get("max_products_per_obs", 3),
            include_auxiliary=download_data.get("include_auxiliary", False),
            require_same_mission=download_data.get("require_same_mission", True),
            verify_fits=download_data.get("verify_fits", True),
            skip_reference=download_data.get("skip_reference", False),
            skip_science=download_data.get("skip_science", False),
        )

        # Build QualityConfig
        quality_data = data.get("quality", {})
        quality_config = QualityConfig(
            min_overlap_fraction=quality_data.get("min_overlap_fraction", 0.85),
            max_file_size_mb=quality_data.get("max_file_size_mb", 500.0),
            verify_wcs=quality_data.get("verify_wcs", True),
        )

        # Build OutputConfig
        output_data = data.get("output", {})
        if not output_data:
            # Generate default paths based on dataset name
            base_dir = Path("output") / "datasets" / dataset_name
            output_config = OutputConfig(
                query_results=base_dir / "queries.json",
                fits_downloads=base_dir / "fits_downloads",
                fits_training=base_dir / "fits_training",
                difference_images=base_dir / "difference_images",
            )
        else:
            output_config = OutputConfig(
                query_results=output_data.get(
                    "query_results", f"output/datasets/{dataset_name}/queries.json"
                ),
                fits_downloads=output_data.get(
                    "fits_downloads",
                    f"output/datasets/{dataset_name}/fits_downloads",
                ),
                fits_training=output_data.get(
                    "fits_training",
                    f"output/datasets/{dataset_name}/fits_training",
                ),
                difference_images=output_data.get(
                    "difference_images",
                    f"output/datasets/{dataset_name}/difference_images",
                ),
                checkpoint=output_data.get("checkpoint"),
                chunk_dir=output_data.get("chunk_dir"),
            )

        return cls(
            dataset_name=dataset_name,
            description=description,
            query=query_config,
            download=download_config,
            quality=quality_config,
            output=output_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "description": self.description,
            "query": {
                "missions": self.query.missions,
                "filters": self.query.filters,
                "archives": self.query.archives,
                "min_year": self.query.min_year,
                "max_year": self.query.max_year,
                "days_before": self.query.days_before,
                "days_after": self.query.days_after,
                "radius_deg": self.query.radius_deg,
                "chunk_size": self.query.chunk_size,
                "start_index": self.query.start_index,
                "limit": self.query.limit,
            },
            "download": {
                "max_obs_per_type": self.download.max_obs_per_type,
                "max_products_per_obs": self.download.max_products_per_obs,
                "include_auxiliary": self.download.include_auxiliary,
                "require_same_mission": self.download.require_same_mission,
                "verify_fits": self.download.verify_fits,
                "skip_reference": self.download.skip_reference,
                "skip_science": self.download.skip_science,
            },
            "quality": {
                "min_overlap_fraction": self.quality.min_overlap_fraction,
                "max_file_size_mb": self.quality.max_file_size_mb,
                "verify_wcs": self.quality.verify_wcs,
            },
            "output": {
                "query_results": str(self.output.query_results),
                "fits_downloads": str(self.output.fits_downloads),
                "fits_training": str(self.output.fits_training),
                "difference_images": str(self.output.difference_images),
                "checkpoint": str(self.output.checkpoint) if self.output.checkpoint else None,
                "chunk_dir": str(self.output.chunk_dir) if self.output.chunk_dir else None,
            },
        }

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings/errors.

        Returns:
            List of validation messages (empty if valid)
        """
        warnings: list[str] = []

        # Check output directories are writable
        if self.output:
            for path_name, path in [
                ("fits_downloads", self.output.fits_downloads),
                ("fits_training", self.output.fits_training),
                ("difference_images", self.output.difference_images),
            ]:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.parent.exists():
                        warnings.append(
                            f"Cannot create parent directory for {path_name}: {path.parent}"
                        )
                except Exception as e:
                    warnings.append(
                        f"Cannot write to {path_name} directory {path.parent}: {e}"
                    )

        # Check mission names are valid
        valid_missions = {
            # MAST missions
            "SWIFT", "PS1", "GALEX", "TESS", "HST", "JWST", "SDSS", "HLA", "HLSP",
            # IRSA missions
            "ZTF", "PTF", "WISE", "NEOWISE", "2MASS", "SPITZER",
        }
        invalid_missions = set(m.upper() for m in self.query.missions) - {m.upper() for m in valid_missions}
        if invalid_missions:
            warnings.append(
                f"Unknown missions (may still work): {invalid_missions}. "
                f"Known missions: {sorted(valid_missions)}"
            )

        return warnings
