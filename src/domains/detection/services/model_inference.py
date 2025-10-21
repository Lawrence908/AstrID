"""Model inference service for detection domain.

Bridges domain entity/config with adapter-side ML model and MLflow registry.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, cast

import numpy as np

try:
    import mlflow
    from mlflow import pyfunc

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    mlflow = None
    pyfunc = None
    MLFLOW_AVAILABLE = False
try:  # MLflow error type for model registry
    from mlflow.exceptions import RestException as MlflowRestException  # type: ignore
except Exception:
    MlflowRestException = Exception  # type: ignore

from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.detection.config import ModelConfig
from src.domains.detection.ml_entities import UNetModelEntity
from src.domains.detection.models import DetectionType
from src.domains.detection.repository import DetectionRepository
from src.domains.detection.schema import DetectionCreate
from src.domains.detection.scorers.confidence import compute_confidence

try:
    from src.core.mlflow_energy import MLflowEnergyTracker
except Exception:
    MLflowEnergyTracker = None  # type: ignore


@lru_cache(maxsize=4)
def _load_model_cached(
    model_name: str, model_version: str, height: int, width: int
) -> Any:
    """Module-level cached model loader to avoid method-level caching (ruff B019).

    Cache key includes model name, version, and input size.
    """
    if not MLFLOW_AVAILABLE:
        # Fall back to local adapter model construction to keep dev flowing
        from src.adapters.ml.unet import UNetModel  # local lightweight import

        model = UNetModel(input_shape=(height, width, 3)).model
        return model

    model_uri = f"models:/{model_name}/{model_version}"
    try:
        keras_api = getattr(mlflow, "keras", None)
        if keras_api is not None:
            loaded = cast(Any, keras_api).load_model(model_uri)
        else:
            raise RuntimeError("mlflow.keras not available")
    except (MlflowRestException, Exception):
        # Fallback to pyfunc (works if logged as generic model)
        try:
            loaded = (
                cast(Any, pyfunc).load_model(model_uri) if pyfunc is not None else None
            )
        except Exception:
            loaded = None
    # Final safety: if nothing loaded, build local adapter model
    if loaded is None:
        from src.adapters.ml.unet import UNetModel

        loaded = UNetModel(input_shape=(height, width, 3)).model
    return loaded


class ModelInferenceService:
    """Service responsible for loading models and running inference."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.entity = UNetModelEntity(self.config)

    # Caching: one loaded model per (name, version, input_size) at module level
    def _load_model(self, model_name: str, model_version: str) -> Any:
        h, w = int(self.config.input_size[0]), int(self.config.input_size[1])
        return _load_model_cached(model_name, model_version, h, w)

    def warm_up(self) -> None:
        """Warm-up by performing a dummy forward pass."""
        model = self._load_model(self.config.model_name, self.config.model_version)
        dummy = np.zeros(
            (1, self.config.input_size[0], self.config.input_size[1], 3),
            dtype=np.float32,
        )
        try:
            if hasattr(model, "predict"):
                model.predict(dummy, verbose=0)
            elif callable(model):
                model(dummy)
        except Exception:
            # Non-fatal in warm-up
            pass

    def _preprocess_batch(self, images: list[np.ndarray]) -> np.ndarray:
        processed: list[np.ndarray] = []
        for img in images:
            arr = img
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if self.config.resize and arr.shape[:2] != tuple(self.config.input_size):
                try:
                    import cv2  # optional, faster than TF here

                    arr = cv2.resize(
                        arr, self.config.input_size[::-1], interpolation=cv2.INTER_AREA
                    )
                except Exception:
                    # Fallback to PIL (nearest)
                    from PIL import Image  # lightweight fallback

                    target_size: tuple[int, int] = (
                        int(self.config.input_size[1]),
                        int(self.config.input_size[0]),
                    )
                    arr = np.array(Image.fromarray(arr).resize(target_size))
            arr = arr.astype(np.float32)
            if self.config.normalize and arr.max() > 1.0:
                arr = arr / 255.0
            processed.append(arr)
        batch = np.stack(processed, axis=0)
        return batch

    def _postprocess(self, preds: np.ndarray, threshold: float) -> dict[str, Any]:
        # Ensure channel-last and remove batch if needed
        if preds.ndim == 4 and preds.shape[0] == 1:
            preds = preds[0]
        if preds.ndim == 3 and preds.shape[-1] == 1:
            prob = preds[:, :, 0]
        else:
            prob = preds if preds.ndim == 2 else preds.mean(axis=-1)

        mask = (prob >= threshold).astype(np.uint8)
        return {"probability": prob, "mask": mask}

    def infer_batch(
        self,
        images: list[np.ndarray],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run batch inference and return outputs with metrics."""
        thr = threshold if threshold is not None else self.entity.confidence_threshold
        model = self._load_model(self.config.model_name, self.config.model_version)
        batch = self._preprocess_batch(images)

        t0 = time.perf_counter()
        if hasattr(model, "predict"):
            preds = model.predict(batch, verbose=0)
        else:
            preds = model(batch)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Postprocess per item
        results = []
        for i in range(len(images)):
            post = self._postprocess(preds[i], thr)
            score = compute_confidence(post["probability"])  # baseline scoring
            post["confidence"] = score
            results.append(post)

        return {
            "results": results,
            "metrics": {"latency_ms": latency_ms, "batch_size": len(images)},
        }

    async def infer_and_persist_candidates(
        self,
        db: AsyncSession,
        observation_id: str,
        difference_image: np.ndarray,
        candidates: list[dict[str, Any]],
        *,
        model_run_id: str,
    ) -> dict[str, Any]:
        """Run inference on a difference image, score ASTR-79 candidates, and persist detections.

        Expects each candidate dict to at least contain pixel_x, pixel_y, and optionally RA/Dec.
        """
        # Run single-image inference
        out = self.infer_batch([difference_image])
        prob = cast(np.ndarray, out["results"][0]["probability"])  # HxW
        latency_ms = float(out["metrics"]["latency_ms"])

        # Minimal MLflow logging (latency + counts) if client available
        if MLFLOW_AVAILABLE and mlflow is not None:
            try:
                mlflow.set_experiment("inference")
                with mlflow.start_run():
                    mlflow.log_metric("latency_ms", latency_ms)
                    mlflow.log_param("model_version", self.config.model_version)
                    mlflow.log_param("num_candidates", len(candidates))

                    # Log minimal artifacts so we can verify R2 uploads
                    try:
                        import os
                        import tempfile

                        prob_preview = (prob * 255).astype("uint8")
                        with tempfile.TemporaryDirectory() as td:
                            prob_path = os.path.join(td, "probability.npy")
                            mask_path = os.path.join(td, "mask_preview.png")
                            np.save(prob_path, prob)
                            # Save a small PNG preview of the mask
                            from PIL import Image

                            Image.fromarray(
                                (
                                    prob_preview
                                    > int(self.config.confidence_threshold * 255)
                                ).astype("uint8")
                                * 255
                            ).save(mask_path)
                            mlflow.log_artifact(prob_path, artifact_path="outputs")
                            mlflow.log_artifact(mask_path, artifact_path="outputs")
                    except Exception:
                        pass
            except Exception:
                pass

        repo = DetectionRepository(db)
        created_ids: list[str] = []

        # Sample confidence around candidate pixels
        h, w = prob.shape[:2]
        for cand in candidates:
            x = int(cand.get("pixel_x", cand.get("x", 0)))
            y = int(cand.get("pixel_y", cand.get("y", 0)))
            x0, x1 = max(0, x - 1), min(w, x + 2)
            y0, y1 = max(0, y - 1), min(h, y + 2)
            patch = prob[y0:y1, x0:x1]
            conf = float(patch.max()) if patch.size else 0.0

            det = DetectionCreate(
                observation_id=cast(Any, observation_id if observation_id else None),
                model_run_id=cast(Any, model_run_id),
                ra=float(cand.get("ra", 0.0)),
                dec=float(cand.get("dec", 0.0)),
                pixel_x=x,
                pixel_y=y,
                confidence_score=conf,
                detection_type=DetectionType.TRANSIENT,
                model_version=self.config.model_version,
                inference_time_ms=int(latency_ms),
                prediction_metadata={"window": [x0, y0, x1, y1]},
            )
            created = await repo.create(det)
            created_ids.append(str(created.id))  # type: ignore[attr-defined]

        return {"latency_ms": latency_ms, "created_detection_ids": created_ids}
