"""Training utilities for the AstrID model training notebook."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Dataset

from src.core.gpu_monitoring import EnergyConsumption
from src.domains.detection.metrics.detection_metrics import DetectionMetrics
from src.domains.preprocessing.processors.astronomical_image_processing import (
    AstronomicalImageProcessor,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics container."""

    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Advanced metrics
    auroc: float = 0.0
    auprc: float = 0.0
    mcc: float = 0.0
    balanced_accuracy: float = 0.0

    # Calibration metrics
    ece: float = 0.0
    brier_score: float = 0.0

    # Performance metrics
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    throughput_items_per_s: float = 0.0

    # Energy metrics
    energy_consumed_wh: float = 0.0
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    carbon_footprint_g: float = 0.0

    # Training metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0


class AstronomicalDataset(Dataset):
    """Dataset for astronomical images with preprocessing integration."""

    def __init__(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        preprocessing_service: AstronomicalImageProcessor | None = None,
        transform: transforms.Compose | None = None,
    ):
        self.images = images
        self.masks = masks
        self.preprocessing_service = preprocessing_service
        self.transform = transform

        # Validate data
        assert len(images) == len(masks), "Images and masks must have same length"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        # Apply preprocessing if available
        if self.preprocessing_service:
            try:
                # Apply astronomical preprocessing pipeline
                processed_image, _ = (
                    self.preprocessing_service.enhance_astronomical_image(
                        image,
                        bias_correction=True,
                        flat_correction=True,
                        dark_correction=True,
                        cosmic_ray_removal=True,
                        background_subtraction=True,
                        noise_reduction=True,
                    )
                )
                image = processed_image
            except Exception as e:
                logger.warning(f"Preprocessing failed for sample {idx}: {e}")

        # Ensure proper data types and shapes
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # Add channel dimension
        elif image.ndim == 3 and image.shape[-1] == 3:
            # Convert RGB to grayscale
            image = np.mean(image, axis=-1, keepdims=True)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        # Ensure binary mask {0,1}
        mask = (mask > 0.5).astype(np.float32)

        # Convert to tensors
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # Apply transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor, mask_tensor


def create_data_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Create data augmentation transforms for training and validation."""

    # Training transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ]
    )

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])

    return train_transform, val_transform


def load_sample_data(
    num_samples: int = 100, image_size: tuple[int, int] = (512, 512)
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load sample astronomical data for demonstration."""

    images = []
    masks = []

    for _ in range(num_samples):
        # Create synthetic astronomical image
        image = np.random.normal(0.5, 0.1, image_size)

        # Add some structure (stars, galaxies, etc.)
        # Stars
        num_stars = np.random.poisson(5)
        for _ in range(num_stars):
            x, y = (
                np.random.randint(0, image_size[0]),
                np.random.randint(0, image_size[1]),
            )
            size = int(np.random.uniform(1, 3))
            image[
                max(0, x - size) : min(image_size[0], x + size),
                max(0, y - size) : min(image_size[1], y + size),
            ] += np.random.uniform(0.2, 0.8)

        # Add some anomalies (transients)
        if np.random.random() < 0.3:  # 30% chance of anomaly
            x, y = (
                np.random.randint(0, image_size[0]),
                np.random.randint(0, image_size[1]),
            )
            size = int(np.random.uniform(2, 5))
            image[
                max(0, x - size) : min(image_size[0], x + size),
                max(0, y - size) : min(image_size[1], y + size),
            ] += np.random.uniform(0.5, 1.0)

        # Create corresponding mask
        mask = np.zeros(image_size)
        if np.random.random() < 0.3:  # 30% chance of anomaly
            x, y = (
                np.random.randint(0, image_size[0]),
                np.random.randint(0, image_size[1]),
            )
            size = int(np.random.uniform(2, 5))
            mask[
                max(0, x - size) : min(image_size[0], x + size),
                max(0, y - size) : min(image_size[1], y + size),
            ] = 1.0

        images.append(image)
        masks.append(mask)

    return images, masks


class MetricsCalculator:
    """Comprehensive metrics calculation for model evaluation."""

    def __init__(self):
        self.detection_metrics = DetectionMetrics()

    def calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
    ) -> dict[str, float]:
        """Calculate comprehensive classification metrics."""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Advanced metrics
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auroc = 0.0

        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            auprc = np.trapz(recall_curve, precision_curve)
        except ValueError:
            auprc = 0.0

        mcc = matthews_corrcoef(y_true, y_pred)

        # Balanced accuracy
        balanced_accuracy = (
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            + recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        ) / 2

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "mcc": float(mcc),
            "balanced_accuracy": float(balanced_accuracy),
        }

    def calculate_calibration_metrics(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> dict[str, float]:
        """Calculate calibration metrics."""

        try:
            # Expected Calibration Error (ECE)
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_scores, n_bins=10
            )
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

            # Brier Score
            brier_score = np.mean((y_scores - y_true) ** 2)

        except (ValueError, ZeroDivisionError):
            ece = 0.0
            brier_score = 0.0

        return {"ece": float(ece), "brier_score": float(brier_score)}

    def calculate_performance_metrics(
        self, inference_times: list[float], batch_size: int
    ) -> dict[str, float]:
        """Calculate performance metrics."""

        if not inference_times:
            return {
                "latency_ms_p50": 0.0,
                "latency_ms_p95": 0.0,
                "throughput_items_per_s": 0.0,
            }

        inference_times_ms = np.array(inference_times) * 1000  # Convert to ms

        # Latency percentiles
        latency_p50 = np.percentile(inference_times_ms, 50)
        latency_p95 = np.percentile(inference_times_ms, 95)

        # Throughput
        avg_inference_time = np.mean(inference_times)
        throughput = batch_size / avg_inference_time if avg_inference_time > 0 else 0.0

        return {
            "latency_ms_p50": float(latency_p50),
            "latency_ms_p95": float(latency_p95),
            "throughput_items_per_s": float(throughput),
        }

    def calculate_energy_metrics(
        self, energy_consumption: EnergyConsumption
    ) -> dict[str, float]:
        """Calculate energy consumption metrics."""

        carbon_footprint_kg = (
            energy_consumption.carbon_footprint_kg
            if energy_consumption.carbon_footprint_kg is not None
            else 0.0
        )

        return {
            "energy_consumed_wh": float(energy_consumption.total_energy_wh),
            "avg_power_w": float(energy_consumption.average_power_watts),
            "peak_power_w": float(energy_consumption.peak_power_watts),
            "carbon_footprint_g": float(carbon_footprint_kg * 1000),  # grams
        }


class TrainingVisualizer:
    """Visualization utilities for training monitoring."""

    def __init__(self):
        self.fig_size = (15, 10)

    def plot_training_curves(
        self,
        train_losses: list[float],
        val_losses: list[float],
        train_metrics: list[dict[str, float]] | None = None,
        val_metrics: list[dict[str, float]] | None = None,
    ) -> None:
        """Plot training and validation curves."""

        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)

        # Loss curves
        axes[0, 0].plot(train_losses, label="Training Loss", color="blue")
        axes[0, 0].plot(val_losses, label="Validation Loss", color="red")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        if train_metrics and val_metrics:
            train_acc = [m.get("accuracy", 0) for m in train_metrics]
            val_acc = [m.get("accuracy", 0) for m in val_metrics]

            axes[0, 1].plot(train_acc, label="Training Accuracy", color="blue")
            axes[0, 1].plot(val_acc, label="Validation Accuracy", color="red")
            axes[0, 1].set_title("Training and Validation Accuracy")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # F1 Score curves
        if train_metrics and val_metrics:
            train_f1 = [m.get("f1_score", 0) for m in train_metrics]
            val_f1 = [m.get("f1_score", 0) for m in val_metrics]

            axes[1, 0].plot(train_f1, label="Training F1", color="blue")
            axes[1, 0].plot(val_f1, label="Validation F1", color="red")
            axes[1, 0].set_title("Training and Validation F1 Score")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("F1 Score")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Learning rate curve
        if train_metrics:
            lr = [m.get("learning_rate", 0) for m in train_metrics]
            axes[1, 1].plot(lr, label="Learning Rate", color="green")
            axes[1, 1].set_title("Learning Rate Schedule")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix."""

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> None:
        """Plot ROC curve."""

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_precision_recall_curve(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> None:
        """Plot Precision-Recall curve."""

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = np.trapz(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall, precision, color="blue", label=f"PR Curve (AUPRC = {auprc:.3f})"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_energy_consumption(self, energy_data: list[dict[str, float]]) -> None:
        """Plot energy consumption over time."""

        if not energy_data:
            return

        timestamps = list(range(len(energy_data)))
        power_values = [d.get("power_w", 0) for d in energy_data]
        energy_values = [d.get("energy_wh", 0) for d in energy_data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Power consumption
        ax1.plot(timestamps, power_values, color="red", linewidth=2)
        ax1.set_title("GPU Power Consumption Over Time")
        ax1.set_xlabel("Time (samples)")
        ax1.set_ylabel("Power (W)")
        ax1.grid(True)

        # Cumulative energy
        ax2.plot(timestamps, energy_values, color="blue", linewidth=2)
        ax2.set_title("Cumulative Energy Consumption")
        ax2.set_xlabel("Time (samples)")
        ax2.set_ylabel("Energy (Wh)")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


class ModelCheckpoint:
    """Model checkpointing utilities."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: TrainingMetrics,
        is_best: bool = False,
    ) -> str:
        """Save model checkpoint."""

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved at epoch {epoch}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Load model checkpoint."""

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
