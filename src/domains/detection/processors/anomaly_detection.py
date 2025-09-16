"""Anomaly detection processors for astronomical data.

This module contains machine learning-based anomaly detection algorithms
for identifying unusual patterns in astronomical images.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# Only import heavy dependencies when actually used
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


class SimpleUNet(nn.Module):
    """Simplified U-Net architecture for anomaly detection."""

    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        return self.final(d1)


class AnomalyDetector:
    """Anomaly detection system using multiple approaches."""

    def __init__(self, model_path: str | None = None):
        self.unet_model = None
        self.isolation_forest = None
        self.one_class_svm = None
        self.scaler = StandardScaler()

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load pre-trained U-Net model."""
        if not TORCH_AVAILABLE or torch is None:
            print("PyTorch not available. U-Net model loading disabled.")
            return

        try:
            self.unet_model = SimpleUNet()
            self.unet_model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.unet_model.eval()
            print(f"Loaded U-Net model from {model_path}")
        except Exception as e:
            print(f"Failed to load U-Net model: {e}")

    def train_anomaly_models(self, normal_images: list[np.ndarray]):
        """Train anomaly detection models on normal images."""
        print("Training anomaly detection models...")

        # Extract features from normal images
        features = []
        for img in normal_images:
            # Extract statistical features
            feat = self.extract_image_features(img)
            features.append(feat)

        features = np.array(features)

        # Train Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(features)

        # Train One-Class SVM
        self.one_class_svm = OneClassSVM(nu=0.1, kernel="rbf")
        self.one_class_svm.fit(features)

        print("Anomaly detection models trained successfully")

    def extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image for anomaly detection."""
        features = []

        # Statistical features
        features.extend(
            [
                np.mean(image),
                np.std(image),
                np.min(image),
                np.max(image),
                np.median(image),
                np.percentile(image, 25),
                np.percentile(image, 75),
            ]
        )

        # Texture features (simplified)
        features.extend(
            [
                np.var(image),  # Variance
                np.mean(np.gradient(image)),  # Gradient mean
                np.std(np.gradient(image)),  # Gradient std
            ]
        )

        # Histogram features
        hist, _ = np.histogram(image.flatten(), bins=10)
        features.extend(hist / np.sum(hist))  # Normalized histogram

        return np.array(features)

    def detect_anomalies_unet(
        self, image: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, float]:
        """Detect anomalies using U-Net model."""
        if not TORCH_AVAILABLE or torch is None or F is None:
            raise ValueError("PyTorch not available for U-Net inference")

        if self.unet_model is None:
            raise ValueError("U-Net model not loaded")

        # Preprocess image
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Convert to tensor
        image_tensor = torch.FloatTensor(image)

        # Run inference
        with torch.no_grad():
            output = self.unet_model(image_tensor)
            anomaly_map = F.softmax(output, dim=1)[:, 1]  # Get anomaly probability
            anomaly_score = float(torch.max(anomaly_map))

        return anomaly_map[0].numpy(), anomaly_score

    def detect_anomalies_ml(self, image: np.ndarray) -> dict[str, float]:
        """Detect anomalies using traditional ML methods."""
        features = self.extract_image_features(image)
        features = features.reshape(1, -1)

        results = {}

        if self.isolation_forest is not None:
            if_score = self.isolation_forest.decision_function(features)[0]
            results["isolation_forest_score"] = float(if_score)
            results["isolation_forest_anomaly"] = float(if_score < 0)

        if self.one_class_svm is not None:
            svm_score = self.one_class_svm.decision_function(features)[0]
            results["one_class_svm_score"] = float(svm_score)
            results["one_class_svm_anomaly"] = float(svm_score < 0)

        return results

    def comprehensive_anomaly_detection(
        self, image: np.ndarray
    ) -> dict[str, float | np.ndarray]:
        """Perform comprehensive anomaly detection using all methods."""
        results = {}

        # U-Net detection
        if TORCH_AVAILABLE and self.unet_model is not None:
            try:
                anomaly_map, unet_score = self.detect_anomalies_unet(image)
                results["unet_anomaly_map"] = anomaly_map
                results["unet_anomaly_score"] = unet_score
            except Exception as e:
                print(f"U-Net detection failed: {e}")
                results["unet_anomaly_score"] = 0.0
        else:
            results["unet_anomaly_score"] = 0.0

        # Traditional ML detection
        ml_results = self.detect_anomalies_ml(image)
        results.update(ml_results)

        # Combined score
        scores = [
            v for k, v in results.items() if "score" in k and isinstance(v, int | float)
        ]
        if scores:
            results["combined_anomaly_score"] = float(np.mean(scores))

        return results


class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies for training and testing."""

    def create_synthetic_anomaly_dataset(
        self,
        normal_images: list[np.ndarray],
        num_anomalies: int = 50,
        anomaly_types: list[str] | None = None,
    ) -> tuple[list[np.ndarray], list[int]]:
        """Create synthetic anomaly dataset for training."""
        if anomaly_types is None:
            anomaly_types = ["bright_spots", "dark_spots", "streaks"]

        images = []
        labels = []  # 0 = normal, 1 = anomaly

        # Add normal images
        for img in normal_images:
            images.append(img)
            labels.append(0)

        # Generate synthetic anomalies
        for i in range(num_anomalies):
            # Select random normal image
            base_img = normal_images[i % len(normal_images)].copy()
            anomaly_type = anomaly_types[i % len(anomaly_types)]

            # Add anomaly
            if anomaly_type == "bright_spots":
                # Add bright spots
                y, x = (
                    np.random.randint(0, base_img.shape[0]),
                    np.random.randint(0, base_img.shape[1]),
                )

                # Add bounds checking to prevent IndexError
                y_min = max(0, y - 2)
                y_max = min(base_img.shape[0], y + 3)
                x_min = max(0, x - 2)
                x_max = min(base_img.shape[1], x + 3)

                base_img[y_min:y_max, x_min:x_max] += np.random.uniform(50, 100)
            elif anomaly_type == "dark_spots":
                # Add dark spots
                y, x = (
                    np.random.randint(0, base_img.shape[0]),
                    np.random.randint(0, base_img.shape[1]),
                )

                # Add bounds checking to prevent IndexError
                y_min = max(0, y - 2)
                y_max = min(base_img.shape[0], y + 3)
                x_min = max(0, x - 2)
                x_max = min(base_img.shape[1], x + 3)

                base_img[y_min:y_max, x_min:x_max] -= np.random.uniform(50, 100)
            elif anomaly_type == "streaks":
                # Add streaks
                y1, x1 = (
                    np.random.randint(0, base_img.shape[0]),
                    np.random.randint(0, base_img.shape[1]),
                )
                y2, x2 = (
                    np.random.randint(0, base_img.shape[0]),
                    np.random.randint(0, base_img.shape[1]),
                )
                import cv2

                cv2.line(
                    base_img, (x1, y1), (x2, y2), (int(np.random.uniform(50, 100)),), 2
                )

            images.append(base_img)
            labels.append(1)

        return images, labels


class AnomalyDetectionEvaluator:
    """Evaluate anomaly detection performance."""

    def evaluate_anomaly_detection(
        self,
        detector: AnomalyDetector,
        test_images: list[np.ndarray],
        test_labels: list[int],
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate anomaly detection performance."""
        predictions = []

        for img in test_images:
            results = detector.comprehensive_anomaly_detection(img)
            score = results.get("combined_anomaly_score", 0.0)
            predictions.append(1 if score > threshold else 0)

        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
