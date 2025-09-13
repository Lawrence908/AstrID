"""U-Net model adapter for astronomical image segmentation."""

import logging
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class UNetModel:
    """U-Net model wrapper for astronomical image segmentation."""

    def __init__(
        self,
        model_path: str | None = None,
        input_shape: tuple[int, int, int] = (512, 512, 3),
    ):
        """Initialize U-Net model.

        Args:
            model_path: Path to saved model file (.keras or .h5)
            input_shape: Input image shape (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None

        if model_path:
            self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self) -> None:
        """Build U-Net model architecture."""
        logger.info("Building U-Net model")

        # Input layer
        inputs = keras.layers.Input(shape=self.input_shape)

        # Encoder (downsampling path)
        conv1 = self._conv_block(inputs, 64)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self._conv_block(pool1, 128)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self._conv_block(pool2, 256)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self._conv_block(pool3, 512)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bridge
        conv5 = self._conv_block(pool4, 1024)

        # Decoder (upsampling path)
        up6 = keras.layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = keras.layers.concatenate([up6, conv4], axis=-1)
        conv6 = self._conv_block(up6, 512)

        up7 = keras.layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = keras.layers.concatenate([up7, conv3], axis=-1)
        conv7 = self._conv_block(up7, 256)

        up8 = keras.layers.UpSampling2D(size=(2, 2))(conv7)
        up8 = keras.layers.concatenate([up8, conv2], axis=-1)
        conv8 = self._conv_block(up8, 128)

        up9 = keras.layers.UpSampling2D(size=(2, 2))(conv8)
        up9 = keras.layers.concatenate([up9, conv1], axis=-1)
        conv9 = self._conv_block(up9, 64)

        # Output layer
        outputs = keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(conv9)

        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        logger.info("U-Net model built successfully")

    def _conv_block(self, inputs: tf.Tensor, filters: int) -> tf.Tensor:
        """Convolutional block with batch normalization and dropout.

        Args:
            inputs: Input tensor
            filters: Number of filters

        Returns:
            Output tensor
        """
        conv = keras.layers.Conv2D(filters, (3, 3), padding="same")(inputs)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.Activation("relu")(conv)
        conv = keras.layers.Conv2D(filters, (3, 3), padding="same")(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.Activation("relu")(conv)
        conv = keras.layers.Dropout(0.2)(conv)
        return conv

    def load_model(self, model_path: str) -> None:
        """Load pre-trained model from file.

        Args:
            model_path: Path to saved model file
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Building new model instead")
            self.build_model()

    def save_model(self, model_path: str) -> None:
        """Save model to file.

        Args:
            model_path: Path to save model file
        """
        if self.model:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.

        Args:
            image: Input image array

        Returns:
            Preprocessed image array
        """
        # Ensure image has correct shape
        if image.ndim == 2:
            # Grayscale to RGB
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            # Single channel to RGB
            image = np.repeat(image, 3, axis=2)

        # Resize if necessary
        if image.shape[:2] != self.input_shape[:2]:
            image = tf.image.resize(image, self.input_shape[:2])

        # Normalize to [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def predict(
        self, image: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on image.

        Args:
            image: Input image array
            threshold: Confidence threshold for segmentation

        Returns:
            Tuple of (raw_predictions, binary_mask)
        """
        if self.model is None:
            raise ValueError("Model not loaded or built")

        # Preprocess image
        preprocessed = self.preprocess_image(image)

        # Run inference
        raw_predictions = self.model.predict(preprocessed, verbose=0)

        # Remove batch dimension
        raw_predictions = raw_predictions[0]

        # Apply threshold
        binary_mask = (raw_predictions > threshold).astype(np.uint8)

        return raw_predictions, binary_mask

    def extract_star_predictions(
        self, prediction: np.ndarray, threshold: float = 0.5
    ) -> tuple[list, np.ndarray]:
        """Extract star predictions from segmentation output.

        Args:
            prediction: Raw prediction array
            threshold: Confidence threshold

        Returns:
            Tuple of (star_locations, prediction_mask)
        """
        # Normalize prediction to [0, 1]
        prediction = (prediction - prediction.min()) / (
            prediction.max() - prediction.min()
        )

        # Ensure 2D
        if prediction.ndim == 3:
            prediction = prediction[:, :, 0]

        # Threshold
        stars = np.argwhere(prediction > threshold)

        # Create prediction mask
        prediction_mask = np.zeros_like(prediction, dtype=np.uint8)

        # Extract star locations
        star_locations = []
        for star_location in stars:
            y, x = star_location
            star_locations.append((x, y))
            prediction_mask[y, x] = 1

        return star_locations, prediction_mask

    def get_model_summary(self) -> str:
        """Get model architecture summary.

        Returns:
            Model summary string
        """
        if self.model:
            # Capture model summary
            summary_list = []
            self.model.summary(print_fn=lambda x: summary_list.append(x))
            return "\n".join(summary_list)
        return "Model not built or loaded"

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"error": "Model not built or loaded"}

        return {
            "input_shape": self.input_shape,
            "output_shape": self.model.output_shape,
            "total_params": self.model.count_params(),
            "trainable_params": sum(
                [tf.keras.backend.count_params(w) for w in self.model.trainable_weights]
            ),
            "non_trainable_params": sum(
                [
                    tf.keras.backend.count_params(w)
                    for w in self.model.non_trainable_weights
                ]
            ),
        }
