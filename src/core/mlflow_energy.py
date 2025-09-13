"""MLflow integration for energy consumption tracking and carbon footprint analysis."""

import logging
from typing import Any

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore

from src.core.gpu_monitoring import EnergyConsumption

logger = logging.getLogger(__name__)


class MLflowEnergyTracker:
    """Track energy consumption and carbon footprint in MLflow experiments."""

    def __init__(self, experiment_name: str | None = None):
        """
        Initialize MLflow energy tracker.

        Args:
            experiment_name: MLflow experiment name for energy tracking
        """
        self.experiment_name = experiment_name or "energy-tracking"

        if not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow not available. Energy tracking will be logged locally only."
            )

    def log_training_energy(
        self,
        energy_consumption: EnergyConsumption,
        model_version: str,
        model_params: dict[str, Any] | None = None,
        performance_metrics: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str | None:
        """
        Log training energy consumption to MLflow.

        Args:
            energy_consumption: Energy consumption data
            model_version: Model version identifier
            model_params: Model hyperparameters
            performance_metrics: Model performance metrics
            run_id: Existing MLflow run ID (optional)

        Returns:
            MLflow run ID if successful, None otherwise
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Logging energy metrics locally.")
            self._log_energy_locally(energy_consumption, model_version, "training")
            return None

        try:
            # Set or create experiment
            mlflow.set_experiment(self.experiment_name)

            # Use existing run or create new one
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    self._log_energy_metrics(
                        energy_consumption, model_version, "training"
                    )
                    if model_params:
                        self._log_model_params(model_params)
                    if performance_metrics:
                        self._log_performance_metrics(performance_metrics)
                return run_id
            else:
                with mlflow.start_run() as run:
                    self._log_energy_metrics(
                        energy_consumption, model_version, "training"
                    )
                    if model_params:
                        self._log_model_params(model_params)
                    if performance_metrics:
                        self._log_performance_metrics(performance_metrics)
                    return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to log training energy to MLflow: {e}")
            self._log_energy_locally(energy_consumption, model_version, "training")
            return None

    def log_inference_energy(
        self,
        energy_consumption: EnergyConsumption,
        model_version: str,
        inference_metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str | None:
        """
        Log inference energy consumption to MLflow.

        Args:
            energy_consumption: Energy consumption data
            model_version: Model version identifier
            inference_metadata: Inference run metadata
            run_id: Existing MLflow run ID (optional)

        Returns:
            MLflow run ID if successful, None otherwise
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Logging energy metrics locally.")
            self._log_energy_locally(energy_consumption, model_version, "inference")
            return None

        try:
            # Set or create experiment
            mlflow.set_experiment(self.experiment_name)

            # Use existing run or create new one
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    self._log_energy_metrics(
                        energy_consumption, model_version, "inference"
                    )
                    if inference_metadata:
                        for key, value in inference_metadata.items():
                            mlflow.log_param(f"inference_{key}", value)
                return run_id
            else:
                with mlflow.start_run() as run:
                    self._log_energy_metrics(
                        energy_consumption, model_version, "inference"
                    )
                    if inference_metadata:
                        for key, value in inference_metadata.items():
                            mlflow.log_param(f"inference_{key}", value)
                    return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to log inference energy to MLflow: {e}")
            self._log_energy_locally(energy_consumption, model_version, "inference")
            return None

    def _log_energy_metrics(
        self, energy_consumption: EnergyConsumption, model_version: str, stage: str
    ) -> None:
        """Log energy metrics to current MLflow run."""
        # Core energy metrics
        mlflow.log_metric(f"{stage}_energy_wh", energy_consumption.total_energy_wh)
        mlflow.log_metric(
            f"{stage}_avg_power_w", energy_consumption.average_power_watts
        )
        mlflow.log_metric(f"{stage}_peak_power_w", energy_consumption.peak_power_watts)
        mlflow.log_metric(f"{stage}_min_power_w", energy_consumption.min_power_watts)
        mlflow.log_metric(f"{stage}_duration_s", energy_consumption.duration_seconds)

        # Carbon footprint
        if energy_consumption.carbon_footprint_kg is not None:
            mlflow.log_metric(
                f"{stage}_carbon_footprint_kg", energy_consumption.carbon_footprint_kg
            )
            mlflow.log_metric(
                f"{stage}_carbon_footprint_g",
                energy_consumption.carbon_footprint_kg * 1000,
            )

        # Efficiency metrics
        if energy_consumption.duration_seconds > 0:
            energy_per_second = energy_consumption.total_energy_wh / (
                energy_consumption.duration_seconds / 3600
            )
            mlflow.log_metric(
                f"{stage}_energy_efficiency_wh_per_hour", energy_per_second
            )

        # Metadata
        mlflow.log_param(f"{stage}_model_version", model_version)
        mlflow.log_param(f"{stage}_power_samples", energy_consumption.total_samples)

        # Tags for filtering
        mlflow.set_tag("energy_tracked", "true")
        mlflow.set_tag("stage", stage)
        mlflow.set_tag("model_version", model_version)

        logger.info(
            f"Logged {stage} energy metrics to MLflow: "
            f"{energy_consumption.total_energy_wh:.3f} Wh, "
            f"{energy_consumption.carbon_footprint_kg:.6f} kg CO2"
        )

    def _log_model_params(self, model_params: dict[str, Any]) -> None:
        """Log model parameters to MLflow."""
        for key, value in model_params.items():
            mlflow.log_param(f"model_{key}", value)

    def _log_performance_metrics(self, performance_metrics: dict[str, Any]) -> None:
        """Log model performance metrics to MLflow."""
        for key, value in performance_metrics.items():
            if isinstance(value, int | float):
                mlflow.log_metric(key, value)
            else:
                mlflow.log_param(key, str(value))

    def _log_energy_locally(
        self, energy_consumption: EnergyConsumption, model_version: str, stage: str
    ) -> None:
        """Log energy metrics locally when MLflow is not available."""
        logger.info(
            f"Local energy log [{stage}] - Model: {model_version}, "
            f"Energy: {energy_consumption.total_energy_wh:.3f} Wh, "
            f"Avg Power: {energy_consumption.average_power_watts:.1f} W, "
            f"Peak Power: {energy_consumption.peak_power_watts:.1f} W, "
            f"Duration: {energy_consumption.duration_seconds:.1f}s, "
            f"Carbon: {energy_consumption.carbon_footprint_kg:.6f} kg CO2"
        )


def log_energy_comparison(
    baseline_energy: EnergyConsumption,
    current_energy: EnergyConsumption,
    baseline_model: str,
    current_model: str,
    experiment_name: str = "energy-comparison",
) -> None:
    """
    Log energy consumption comparison between models.

    Args:
        baseline_energy: Baseline model energy consumption
        current_energy: Current model energy consumption
        baseline_model: Baseline model version
        current_model: Current model version
        experiment_name: MLflow experiment name
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Logging comparison locally.")
        _log_comparison_locally(
            baseline_energy, current_energy, baseline_model, current_model
        )
        return

    try:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log both energy consumptions
            mlflow.log_metric("baseline_energy_wh", baseline_energy.total_energy_wh)
            mlflow.log_metric("current_energy_wh", current_energy.total_energy_wh)

            # Calculate and log differences
            energy_diff = (
                current_energy.total_energy_wh - baseline_energy.total_energy_wh
            )
            energy_diff_pct = (energy_diff / baseline_energy.total_energy_wh) * 100

            mlflow.log_metric("energy_difference_wh", energy_diff)
            mlflow.log_metric("energy_difference_percent", energy_diff_pct)

            # Carbon footprint comparison
            if (
                baseline_energy.carbon_footprint_kg
                and current_energy.carbon_footprint_kg
            ):
                carbon_diff = (
                    current_energy.carbon_footprint_kg
                    - baseline_energy.carbon_footprint_kg
                )
                carbon_diff_pct = (
                    carbon_diff / baseline_energy.carbon_footprint_kg
                ) * 100

                mlflow.log_metric("carbon_difference_kg", carbon_diff)
                mlflow.log_metric("carbon_difference_percent", carbon_diff_pct)

            # Model versions
            mlflow.log_param("baseline_model", baseline_model)
            mlflow.log_param("current_model", current_model)

            # Tags
            mlflow.set_tag("comparison", "energy")
            mlflow.set_tag("baseline_model", baseline_model)
            mlflow.set_tag("current_model", current_model)

            logger.info(
                f"Energy comparison logged: {current_model} vs {baseline_model}, "
                f"Difference: {energy_diff:+.3f} Wh ({energy_diff_pct:+.1f}%)"
            )

    except Exception as e:
        logger.error(f"Failed to log energy comparison to MLflow: {e}")
        _log_comparison_locally(
            baseline_energy, current_energy, baseline_model, current_model
        )


def _log_comparison_locally(
    baseline_energy: EnergyConsumption,
    current_energy: EnergyConsumption,
    baseline_model: str,
    current_model: str,
) -> None:
    """Log energy comparison locally when MLflow is not available."""
    energy_diff = current_energy.total_energy_wh - baseline_energy.total_energy_wh
    energy_diff_pct = (energy_diff / baseline_energy.total_energy_wh) * 100

    logger.info(
        f"Local energy comparison - {current_model} vs {baseline_model}: "
        f"Current: {current_energy.total_energy_wh:.3f} Wh, "
        f"Baseline: {baseline_energy.total_energy_wh:.3f} Wh, "
        f"Difference: {energy_diff:+.3f} Wh ({energy_diff_pct:+.1f}%)"
    )


# Convenience function for integration with training flows
def create_energy_tracker(experiment_name: str | None = None) -> MLflowEnergyTracker:
    """Create an MLflow energy tracker instance."""
    return MLflowEnergyTracker(experiment_name=experiment_name)
