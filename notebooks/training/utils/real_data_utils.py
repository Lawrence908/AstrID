"""Real data loading utilities for training pipeline integration."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.core.db.session import AsyncSessionLocal
from src.domains.ml.training_data.services import (
    AstronomicalRealDataset,
    RealDataLoader,
    TrainingDataCollectionParams,
    TrainingDataCollector,
    TrainingDatasetManager,
)
from src.infrastructure.storage.r2_client import R2StorageClient

logger = logging.getLogger(__name__)


@dataclass
class RealDataConfig:
    """Configuration for real data loading."""

    survey_ids: list[str] = None
    confidence_threshold: float = 0.7
    max_samples: int = 1000
    date_range_days: int = 180
    validation_status: str = "validated"
    anomaly_types: list[str] | None = None

    def __post_init__(self):
        if self.survey_ids is None:
            self.survey_ids = ["hst", "jwst", "skyview"]


class RealAstronomicalDataset(Dataset):
    """PyTorch Dataset wrapper for real astronomical data."""

    def __init__(self, real_dataset: AstronomicalRealDataset, transform=None):
        self.real_dataset = real_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.real_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get training sample as (image, mask) tensors."""
        image, mask = self.real_dataset[idx]

        if self.transform:
            # Apply transforms to both image and mask
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


async def load_real_training_data(
    config: RealDataConfig,
    dataset_name: str | None = None,
    created_by: str = "training_pipeline",
) -> tuple[Dataset, Dataset, Dataset]:
    """Load real astronomical training data and create train/val/test datasets.

    Args:
        config: Configuration for data collection
        dataset_name: Name for the dataset (auto-generated if None)
        created_by: Creator identifier

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Loading real astronomical training data...")

    # Create database session and R2 client
    db_session = AsyncSessionLocal()
    r2_client = R2StorageClient()

    try:
        # Set up collection parameters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.date_range_days)

        collection_params = TrainingDataCollectionParams(
            survey_ids=config.survey_ids,
            date_range=(start_date, end_date),
            confidence_threshold=config.confidence_threshold,
            max_samples=config.max_samples,
            validation_status=config.validation_status,
            anomaly_types=config.anomaly_types,
        )

        # Collect training data
        collector = TrainingDataCollector(db_session, r2_client)
        samples = await collector.collect_training_data(collection_params)

        if not samples:
            logger.warning("No samples collected, falling back to synthetic data")
            return _create_fallback_synthetic_datasets(config)

        # Validate data quality
        quality_report = collector.validate_data_quality(samples)
        logger.info(f"Data quality report: {quality_report.quality_score:.3f} score")

        if quality_report.issues:
            logger.warning(f"Data quality issues: {quality_report.issues}")

        # Create and save dataset
        manager = TrainingDatasetManager(db_session)
        dataset_name = (
            dataset_name or f"real_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        training_dataset = await manager.create_dataset(
            name=dataset_name,
            created_by=created_by,
            samples=samples,
            quality_report=quality_report,
            collection_params=collection_params.__dict__,
            description=f"Real astronomical data collected on {datetime.now().isoformat()}",
        )

        logger.info(
            f"Created training dataset {training_dataset.id} with {len(samples)} samples"
        )

        # Create data splits
        data_loader = RealDataLoader(db_session, r2_client)
        train_samples, val_samples, test_samples = data_loader.create_data_splits(
            samples
        )

        # Create PyTorch datasets
        train_dataset = RealAstronomicalDataset(
            data_loader.create_pytorch_dataset(train_samples)
        )
        val_dataset = RealAstronomicalDataset(
            data_loader.create_pytorch_dataset(val_samples)
        )
        test_dataset = RealAstronomicalDataset(
            data_loader.create_pytorch_dataset(test_samples)
        )

        logger.info(
            f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        logger.error(f"Error loading real data: {e}")
        logger.info("Falling back to synthetic data generation")
        return _create_fallback_synthetic_datasets(config)

    finally:
        await db_session.close()


def _create_fallback_synthetic_datasets(
    config: RealDataConfig,
) -> tuple[Dataset, Dataset, Dataset]:
    """Create synthetic datasets as fallback when real data is not available."""
    from sklearn.model_selection import train_test_split

    from notebooks.training.utils.training_utils import (
        AstronomicalDataset,
        create_data_transforms,
        load_sample_data,
    )

    logger.info("Creating fallback synthetic datasets...")

    # Generate synthetic data
    sample_images, sample_masks = load_sample_data(
        num_samples=config.max_samples, image_size=(64, 64)
    )

    # Create data transforms
    train_transform, val_transform = create_data_transforms()

    # Create train/val/test splits
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        sample_images, sample_masks, test_size=0.3, random_state=42
    )
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=0.5, random_state=42
    )

    # Create PyTorch datasets
    train_dataset = AstronomicalDataset(
        train_images, train_masks, transform=train_transform
    )
    val_dataset = AstronomicalDataset(val_images, val_masks, transform=val_transform)
    test_dataset = AstronomicalDataset(test_images, test_masks, transform=val_transform)

    logger.info(
        f"Created synthetic datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def create_real_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 2,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


async def create_real_data_collection_demo() -> dict[str, Any]:
    """Demo function to show real data collection capabilities."""
    logger.info("Running real data collection demo...")

    config = RealDataConfig(
        survey_ids=["hst", "jwst"],
        confidence_threshold=0.6,
        max_samples=100,
        date_range_days=365,
        validation_status="validated",
    )

    # Create training datasets
    train_dataset, val_dataset, test_dataset = await load_real_training_data(
        config=config, dataset_name="demo_real_data", created_by="demo"
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_real_data_loaders(
        train_dataset, val_dataset, test_dataset
    )

    # Sample a batch to verify data loading
    train_batch = next(iter(train_loader))
    images, masks = train_batch

    demo_results = {
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "batch_shape": {"images": list(images.shape), "masks": list(masks.shape)},
        "data_types": {"images": str(images.dtype), "masks": str(masks.dtype)},
        "value_ranges": {
            "images": [float(images.min()), float(images.max())],
            "masks": [float(masks.min()), float(masks.max())],
        },
    }

    logger.info(f"Demo results: {demo_results}")
    return demo_results


# Convenience function for easy integration
async def get_real_training_data(
    max_samples: int = 1000, confidence_threshold: float = 0.7, batch_size: int = 2
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Simple interface to get real training data loaders.

    Args:
        max_samples: Maximum number of samples to collect
        confidence_threshold: Minimum confidence for detections
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = RealDataConfig(
        max_samples=max_samples, confidence_threshold=confidence_threshold
    )

    # Load datasets
    train_dataset, val_dataset, test_dataset = await load_real_training_data(config)

    # Create data loaders
    return create_real_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
