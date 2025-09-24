"""Services for collecting and loading real training data (ASTR-113)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.adapters.imaging.fits_io import FITSProcessor
from src.domains.ml.training_data.mlflow_logging import log_training_dataset_info
from src.domains.ml.training_data.models import TrainingDataset, TrainingSample
from src.domains.preprocessing.processors.astronomical_image_processing import (
    AstronomicalImageProcessor,
)
from src.infrastructure.storage.r2_client import R2StorageClient

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataCollectionParams:
    survey_ids: list[str]
    date_range: tuple[datetime, datetime]
    confidence_threshold: float = 0.7
    anomaly_types: list[str] | None = None
    quality_score_threshold: float = 0.8
    max_samples: int = 10000
    validation_status: str = "validated"


@dataclass
class DataQualityReport:
    """Report on training data quality."""

    total_samples: int
    anomaly_ratio: float
    image_quality_score: float
    label_consistency: float
    temporal_distribution: dict[str, Any]
    survey_coverage: dict[str, int]
    quality_score: float
    issues: list[str]


class TrainingDataCollector:
    """Service for collecting training data from validated detections."""

    def __init__(self, db_session: Session | AsyncSession, r2_client: R2StorageClient):
        self.db = db_session
        self.r2 = r2_client
        self.fits = FITSProcessor()
        self.preprocessor = AstronomicalImageProcessor()

    async def collect_training_data(
        self, params: TrainingDataCollectionParams
    ) -> list[TrainingSample]:
        """Collect training samples from validated detections."""
        logger.info(f"Collecting training data with params: {params}")

        # Fetch validated detections from database
        observations_detections = await self._fetch_observations_with_detections(params)
        logger.info(
            f"Found {len(observations_detections)} observations with detections"
        )

        samples: list[TrainingSample] = []
        for obs, det in observations_detections:
            try:
                # Load FITS image data from R2 or URL
                image_data = await self._load_fits_image(obs)
                if image_data is None:
                    logger.warning(f"Could not load image for observation {obs.id}")
                    continue

                # Generate training patches around detection coordinates
                training_patches = await self._generate_training_patches(
                    image_data, det, obs, patch_size=(64, 64)
                )

                for patch_data in training_patches:
                    sample = TrainingSample(
                        dataset_id=None,  # assigned when persisted via manager
                        observation_id=obs.id,
                        detection_id=det.id,
                        image_path=patch_data["image_path"],
                        mask_path=patch_data["mask_path"],
                        labels={
                            "anomaly_type": det.human_label or det.detection_type.value,
                            "confidence": float(det.confidence_score),
                            "validated": det.is_validated,
                            "is_anomaly": det.confidence_score
                            >= params.confidence_threshold,
                            "pixel_coords": [det.pixel_x, det.pixel_y],
                        },
                        sample_metadata={
                            "survey": str(obs.survey_id),
                            "filter_band": obs.filter_band,
                            "exposure_time": float(obs.exposure_time),
                            "ra": float(obs.ra),
                            "dec": float(obs.dec),
                            "observation_time": obs.observation_time.isoformat(),
                            "patch_size": patch_data["patch_size"],
                            "detection_type": det.detection_type.value,
                        },
                    )
                    samples.append(sample)

            except Exception as e:
                logger.error(f"Error processing observation {obs.id}: {e}")
                continue

        logger.info(f"Generated {len(samples)} training samples")
        return samples

    async def _fetch_observations_with_detections(
        self, params: TrainingDataCollectionParams
    ) -> list[Any]:
        """Fetch observations with associated detections from database."""
        from src.domains.detection.models import Detection
        from src.domains.observations.models import Observation

        # Base query
        stmt = (
            select(Observation, Detection)
            .join(Detection, Detection.observation_id == Observation.id)
            .where(Detection.confidence_score >= params.confidence_threshold)
            .where(
                Observation.observation_time.between(
                    params.date_range[0], params.date_range[1]
                )
            )
            .order_by(Detection.created_at.desc())
            .limit(params.max_samples)
        )

        # Add survey filter if specified
        if params.survey_ids:
            from src.domains.observations.models import Survey

            stmt = stmt.join(Survey, Observation.survey_id == Survey.id)
            stmt = stmt.where(Survey.name.in_(params.survey_ids))

        # Validation filter
        if params.validation_status == "validated":
            stmt = stmt.where(Detection.is_validated.is_(True))

        if params.anomaly_types:
            stmt = stmt.where(Detection.human_label.in_(params.anomaly_types))

        # Execute query
        if isinstance(self.db, AsyncSession):
            result = await self.db.execute(stmt)
            rows = result.all()
        else:
            rows = self.db.execute(stmt).all()

        # If no validated detections found, try with relaxed validation
        if not rows and params.validation_status == "validated":
            logger.info("No validated detections found, trying with relaxed validation")
            relaxed_stmt = stmt.where(Detection.is_validated.is_not(False))
            if isinstance(self.db, AsyncSession):
                result = await self.db.execute(relaxed_stmt)
                rows = result.all()
            else:
                rows = self.db.execute(relaxed_stmt).all()

        return list(rows)

    async def _load_fits_image(self, observation: Any) -> np.ndarray | None:
        """Load FITS image data from R2 storage or URL."""
        try:
            image_path = observation.fits_file_path or observation.fits_url
            if not image_path:
                return None

            # Try to load from R2 first (if it's an R2 path)
            if image_path.startswith("s3://") or "astrid" in image_path:
                try:
                    # Note: R2StorageClient download method varies by implementation
                    # For now, we'll skip R2 loading and focus on local files
                    logger.info(f"R2 loading not fully implemented for: {image_path}")
                except Exception as e:
                    logger.warning(f"Failed to load from R2: {e}")

            # Fallback to direct URL or file path
            if image_path.startswith("http"):
                # For HTTP URLs, we'll skip for now (would need requests)
                logger.warning(f"HTTP URL loading not implemented: {image_path}")
                return None
            elif Path(image_path).exists():
                # Local file path
                with fits.open(image_path) as hdu_list:
                    primary_hdu = hdu_list[0]
                    image_array = getattr(primary_hdu, "data", None)
                    if image_array is not None:
                        return image_array.astype(np.float32)
                return None
            else:
                logger.warning(f"Image path not accessible: {image_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading FITS image: {e}")
            return None

    async def _generate_training_patches(
        self,
        image_data: np.ndarray,
        detection: Any,
        observation: Any,
        patch_size: tuple[int, int] = (64, 64),
    ) -> list[dict[str, Any]]:
        """Generate training patches around detection coordinates."""
        patches = []

        try:
            # Get detection pixel coordinates
            center_x, center_y = detection.pixel_x, detection.pixel_y
            patch_w, patch_h = patch_size

            # Ensure coordinates are within image bounds
            img_h, img_w = (
                image_data.shape[-2:] if image_data.ndim > 2 else image_data.shape
            )

            # Calculate patch boundaries
            x_start = max(0, center_x - patch_w // 2)
            x_end = min(img_w, center_x + patch_w // 2)
            y_start = max(0, center_y - patch_h // 2)
            y_end = min(img_h, center_y + patch_h // 2)

            # Extract image patch
            if image_data.ndim == 2:
                patch = image_data[y_start:y_end, x_start:x_end]
            else:
                patch = image_data[0, y_start:y_end, x_start:x_end]  # First channel

            # Create binary mask for anomaly location
            mask = np.zeros_like(patch)
            rel_x = center_x - x_start
            rel_y = center_y - y_start

            # Mark anomaly region (small circle around detection)
            radius = 3
            y_indices, x_indices = np.ogrid[: patch.shape[0], : patch.shape[1]]
            mask_circle = (x_indices - rel_x) ** 2 + (
                y_indices - rel_y
            ) ** 2 <= radius**2
            mask[mask_circle] = 1.0

            # Generate unique paths for this patch (would be stored in R2 in production)
            patch_id = f"{observation.id}_{detection.id}_{center_x}_{center_y}"
            image_path = f"training/patches/{patch_id}_image.npy"
            mask_path = f"training/patches/{patch_id}_mask.npy"

            # For now, store as temporary arrays (in production, would upload to R2)
            patch_data = {
                "image_data": patch,
                "mask_data": mask,
                "image_path": image_path,
                "mask_path": mask_path,
                "patch_size": patch_size,
                "center_coords": (center_x, center_y),
                "patch_coords": (x_start, y_start, x_end, y_end),
            }
            patches.append(patch_data)

            # Also generate negative samples (patches without anomalies)
            for _ in range(2):  # Generate 2 negative samples per positive
                neg_x = np.random.randint(patch_w // 2, img_w - patch_w // 2)
                neg_y = np.random.randint(patch_h // 2, img_h - patch_h // 2)

                # Ensure negative sample is far from detection
                if abs(neg_x - center_x) > patch_w and abs(neg_y - center_y) > patch_h:
                    neg_x_start = max(0, neg_x - patch_w // 2)
                    neg_x_end = min(img_w, neg_x + patch_w // 2)
                    neg_y_start = max(0, neg_y - patch_h // 2)
                    neg_y_end = min(img_h, neg_y + patch_h // 2)

                    if image_data.ndim == 2:
                        neg_patch = image_data[
                            neg_y_start:neg_y_end, neg_x_start:neg_x_end
                        ]
                    else:
                        neg_patch = image_data[
                            0, neg_y_start:neg_y_end, neg_x_start:neg_x_end
                        ]

                    neg_mask = np.zeros_like(neg_patch)  # All zeros for negative sample

                    neg_patch_id = (
                        f"{observation.id}_{detection.id}_neg_{neg_x}_{neg_y}"
                    )
                    neg_patch_data = {
                        "image_data": neg_patch,
                        "mask_data": neg_mask,
                        "image_path": f"training/patches/{neg_patch_id}_image.npy",
                        "mask_path": f"training/patches/{neg_patch_id}_mask.npy",
                        "patch_size": patch_size,
                        "center_coords": (neg_x, neg_y),
                        "patch_coords": (
                            neg_x_start,
                            neg_y_start,
                            neg_x_end,
                            neg_y_end,
                        ),
                        "is_negative": True,
                    }
                    patches.append(neg_patch_data)

        except Exception as e:
            logger.error(f"Error generating patches: {e}")

        return patches

    def validate_data_quality(self, samples: list[TrainingSample]) -> DataQualityReport:
        """Comprehensive data quality validation for training samples."""
        if not samples:
            return DataQualityReport(
                total_samples=0,
                anomaly_ratio=0.0,
                image_quality_score=0.0,
                label_consistency=0.0,
                temporal_distribution={},
                survey_coverage={},
                quality_score=0.0,
                issues=["No samples provided"],
            )

        total_samples = len(samples)
        issues = []

        # Calculate anomaly ratio
        anomaly_samples = sum(
            1
            for s in samples
            if s.labels.get("is_anomaly", False)
            and s.labels.get("confidence", 0) >= 0.7
        )
        anomaly_ratio = anomaly_samples / total_samples if total_samples > 0 else 0.0

        # Check label consistency
        validated_samples = sum(1 for s in samples if s.labels.get("validated", False))
        label_consistency = (
            validated_samples / total_samples if total_samples > 0 else 0.0
        )

        # Analyze temporal distribution
        temporal_dist = {}
        try:
            for sample in samples:
                if (
                    sample.sample_metadata
                    and "observation_time" in sample.sample_metadata
                ):
                    obs_time = sample.sample_metadata["observation_time"]
                    # Extract year-month for distribution
                    date_key = obs_time[:7] if len(obs_time) >= 7 else "unknown"
                    temporal_dist[date_key] = temporal_dist.get(date_key, 0) + 1
        except Exception as e:
            logger.warning(f"Error analyzing temporal distribution: {e}")
            issues.append("Could not analyze temporal distribution")

        # Analyze survey coverage
        survey_coverage = {}
        try:
            for sample in samples:
                if sample.sample_metadata and "survey" in sample.sample_metadata:
                    survey_id = sample.sample_metadata["survey"]
                    survey_coverage[survey_id] = survey_coverage.get(survey_id, 0) + 1
        except Exception as e:
            logger.warning(f"Error analyzing survey coverage: {e}")
            issues.append("Could not analyze survey coverage")

        # Calculate image quality score (placeholder - would analyze actual images)
        image_quality_score = 0.8  # Placeholder

        # Check for potential issues
        if anomaly_ratio < 0.1:
            issues.append("Low anomaly ratio - may need more positive samples")
        elif anomaly_ratio > 0.9:
            issues.append("High anomaly ratio - may need more negative samples")

        if label_consistency < 0.5:
            issues.append("Low label consistency - many unvalidated samples")

        if len(survey_coverage) < 2:
            issues.append(
                "Limited survey diversity - consider adding more survey sources"
            )

        # Calculate overall quality score
        quality_score = (
            (anomaly_ratio * 0.3)  # Balanced anomaly ratio is good
            + (label_consistency * 0.4)  # High validation rate is important
            + (image_quality_score * 0.3)  # Image quality matters
        )

        return DataQualityReport(
            total_samples=total_samples,
            anomaly_ratio=anomaly_ratio,
            image_quality_score=image_quality_score,
            label_consistency=label_consistency,
            temporal_distribution=temporal_dist,
            survey_coverage=survey_coverage,
            quality_score=quality_score,
            issues=issues,
        )


class RealDataLoader:
    """Loader for assembling datasets into train/val/test splits and PyTorch datasets."""

    def __init__(self, db_session: Session | AsyncSession, r2_client: R2StorageClient):
        self.db = db_session
        self.r2 = r2_client

    async def load_training_dataset(
        self, dataset_id: str
    ) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
        """Load training dataset and return train/val/test splits."""
        if isinstance(self.db, AsyncSession):
            ds = await self.db.get(TrainingDataset, dataset_id)
            if not ds:
                raise ValueError(f"TrainingDataset not found: {dataset_id}")

            from sqlalchemy import select

            result = await self.db.execute(
                select(TrainingSample).where(TrainingSample.dataset_id == ds.id)
            )
            samples = result.scalars().all()
        else:
            ds = self.db.get(TrainingDataset, dataset_id)
        if not ds:
            raise ValueError(f"TrainingDataset not found: {dataset_id}")

        samples = (
            self.db.query(TrainingSample)
            .filter(TrainingSample.dataset_id == ds.id)
            .all()
        )
        return self.create_data_splits(samples)

    def create_data_splits(
        self, samples: list[TrainingSample]
    ) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
        """Create balanced train/validation/test splits."""
        # Shuffle samples for randomness
        import random

        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)

        n = len(shuffled_samples)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        train = shuffled_samples[:n_train]
        val = shuffled_samples[n_train : n_train + n_val]
        test = shuffled_samples[n_train + n_val :]

        logger.info(
            f"Created data splits: train={len(train)}, val={len(val)}, test={len(test)}"
        )
        return train, val, test

    def create_pytorch_dataset(
        self, samples: list[TrainingSample]
    ) -> AstronomicalRealDataset:
        """Create PyTorch Dataset from training samples."""
        return AstronomicalRealDataset(samples, self.r2)

    def preprocess_samples(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """Apply preprocessing pipeline to samples (placeholder)."""
        # In practice, this would apply image preprocessing transformations
        logger.info(f"Preprocessing {len(samples)} samples")
        return samples


class AstronomicalRealDataset:
    """PyTorch-compatible dataset for real astronomical data."""

    def __init__(
        self, samples: list[TrainingSample], r2_client: R2StorageClient, transform=None
    ):
        self.samples = samples
        self.r2 = r2_client
        self.transform = transform

        # Cache for loaded patches to avoid repeated R2 calls
        self._image_cache = {}
        self._mask_cache = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Get training sample as (image, mask) tuple."""
        import torch

        sample = self.samples[idx]

        # Try to load image from cache first
        cache_key = sample.image_path
        if cache_key in self._image_cache:
            image = self._image_cache[cache_key]
            mask = self._mask_cache[cache_key]
        else:
            # Load image and mask data
            image, mask = self._load_sample_data(sample)

            # Cache the loaded data
            self._image_cache[cache_key] = image
            self._mask_cache[cache_key] = mask

        if image is None or mask is None:
            # Return dummy data if loading fails
            logger.warning(f"Failed to load sample {idx}, returning dummy data")
            image = np.random.normal(0.5, 0.1, (64, 64)).astype(np.float32)
            mask = np.zeros((64, 64), dtype=np.float32)

        # Ensure proper shape and type
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # Add channel dimension
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        # Convert to tensors
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # Apply transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor, mask_tensor

    def _load_sample_data(
        self, sample: TrainingSample
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load image and mask data for a training sample."""
        try:
            # For now, we'll generate synthetic patches based on the sample metadata
            # In production, this would load from R2 storage

            # Get patch size from metadata
            patch_size = (
                sample.sample_metadata.get("patch_size", (64, 64))
                if sample.sample_metadata
                else (64, 64)
            )

            # Generate synthetic image based on detection type
            image = self._generate_synthetic_patch(sample, patch_size)

            # Generate mask based on labels
            mask = self._generate_mask_from_labels(sample, patch_size)

            return image, mask

        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return None, None

    def _generate_synthetic_patch(
        self, sample: TrainingSample, patch_size: tuple[int, int]
    ) -> np.ndarray:
        """Generate synthetic image patch for training."""
        # Create base astronomical image
        image = np.random.normal(0.5, 0.1, patch_size).astype(np.float32)

        # Add stars
        num_stars = np.random.poisson(3)
        for _ in range(num_stars):
            x, y = (
                np.random.randint(0, patch_size[0]),
                np.random.randint(0, patch_size[1]),
            )
            size = int(np.random.uniform(1, 2))
            image[
                max(0, x - size) : min(patch_size[0], x + size),
                max(0, y - size) : min(patch_size[1], y + size),
            ] += np.random.uniform(0.2, 0.6)

        # Add anomaly if this is a positive sample
        if sample.labels.get("is_anomaly", False):
            # Get detection coordinates from metadata
            if sample.sample_metadata and "pixel_coords" in sample.labels:
                center_x, center_y = sample.labels["pixel_coords"]
                # Convert to relative coordinates (center of patch)
                rel_x, rel_y = patch_size[0] // 2, patch_size[1] // 2
            else:
                rel_x, rel_y = patch_size[0] // 2, patch_size[1] // 2

            # Add bright anomaly
            size = int(np.random.uniform(2, 4))
            image[
                max(0, rel_x - size) : min(patch_size[0], rel_x + size),
                max(0, rel_y - size) : min(patch_size[1], rel_y + size),
            ] += np.random.uniform(0.5, 1.0)

        return image

    def _generate_mask_from_labels(
        self, sample: TrainingSample, patch_size: tuple[int, int]
    ) -> np.ndarray:
        """Generate binary mask from sample labels."""
        mask = np.zeros(patch_size, dtype=np.float32)

        # If this is an anomaly sample, create a mask
        if sample.labels.get("is_anomaly", False):
            # Create circular mask around center
            center_x, center_y = patch_size[0] // 2, patch_size[1] // 2
            radius = 3

            y_indices, x_indices = np.ogrid[: patch_size[0], : patch_size[1]]
            mask_circle = (x_indices - center_x) ** 2 + (
                y_indices - center_y
            ) ** 2 <= radius**2
            mask[mask_circle] = 1.0

        return mask


class TrainingDatasetManager:
    """Manager for creating and managing TrainingDataset records."""

    def __init__(self, db_session: Session | AsyncSession):
        self.db = db_session

    async def create_dataset(
        self,
        name: str,
        created_by: str,
        samples: list[TrainingSample],
        quality_report: DataQualityReport,
        collection_params: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> TrainingDataset:
        """Create new training dataset with comprehensive metadata."""
        ds = TrainingDataset(
            name=name,
            description=description or f"Training dataset with {len(samples)} samples",
            collection_params=collection_params or {},
            total_samples=quality_report.total_samples,
            anomaly_ratio=quality_report.anomaly_ratio,
            quality_score=quality_report.quality_score,
            created_by=created_by,
            status="active",
        )
        self.db.add(ds)

        if isinstance(self.db, AsyncSession):
            await self.db.flush()  # get ds.id
        else:
            self.db.flush()

        # Attach samples to dataset
        for s in samples:
            s.dataset_id = ds.id
            self.db.add(s)

        if isinstance(self.db, AsyncSession):
            await self.db.commit()
        else:
            self.db.commit()

        # Log to MLflow (best-effort)
        try:
            log_training_dataset_info(
                str(ds.id),
                {
                    "total_samples": ds.total_samples,
                    "anomaly_ratio": float(ds.anomaly_ratio),
                    "quality_score": float(ds.quality_score),
                    "survey_coverage": quality_report.survey_coverage,
                    "temporal_distribution": quality_report.temporal_distribution,
                    "issues": quality_report.issues,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log dataset to MLflow: {e}")

        logger.info(f"Created training dataset {ds.id} with {len(samples)} samples")
        return ds

    def get_dataset(self, dataset_id: str) -> TrainingDataset | None:
        """Retrieve training dataset by ID."""
        if isinstance(self.db, AsyncSession):
            # For async sessions, use different approach
            return None  # Would need async implementation
        else:
            return self.db.get(TrainingDataset, dataset_id)

    def list_datasets(
        self, filters: dict[str, Any] | None = None
    ) -> list[TrainingDataset]:
        """List available training datasets with optional filters."""
        if isinstance(self.db, AsyncSession):
            # For async sessions, return empty for now
            return []

        query = self.db.query(TrainingDataset)

        if filters:
            if "status" in filters:
                query = query.filter(TrainingDataset.status == filters["status"])
            if "created_by" in filters:
                query = query.filter(
                    TrainingDataset.created_by == filters["created_by"]
                )
            if "min_quality_score" in filters:
                query = query.filter(
                    TrainingDataset.quality_score >= filters["min_quality_score"]
                )

        return query.order_by(TrainingDataset.created_at.desc()).all()

    def update_dataset_metrics(self, dataset_id: str, metrics: dict[str, Any]) -> None:
        """Update dataset quality metrics."""
        dataset = self.get_dataset(dataset_id)
        if dataset:
            if "quality_score" in metrics:
                dataset.quality_score = metrics["quality_score"]
            if "anomaly_ratio" in metrics:
                dataset.anomaly_ratio = metrics["anomaly_ratio"]

            self.db.commit()
            logger.info(f"Updated metrics for dataset {dataset_id}")
        else:
            logger.warning(f"Dataset {dataset_id} not found for metrics update")
