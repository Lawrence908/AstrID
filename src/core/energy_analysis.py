"""Energy consumption analysis and reporting utilities for ML workloads."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None  # type: ignore
    sns = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class EnergyReport:
    """Energy consumption analysis report."""

    total_energy_wh: float
    total_carbon_kg: float
    average_power_w: float
    peak_power_w: float
    total_training_runs: int
    total_inference_runs: int
    cost_estimate_usd: float
    efficiency_metrics: dict[str, float]
    time_period: str
    generated_at: datetime


class EnergyAnalyzer:
    """Analyze energy consumption patterns and generate reports."""

    def __init__(
        self,
        electricity_cost_per_kwh: float = 0.12,  # USD per kWh
        carbon_intensity_kg_per_kwh: float = 0.233,  # US average
    ):
        """
        Initialize energy analyzer.

        Args:
            electricity_cost_per_kwh: Cost of electricity in USD per kWh
            carbon_intensity_kg_per_kwh: Carbon intensity in kg CO2 per kWh
        """
        self.electricity_cost_per_kwh = electricity_cost_per_kwh
        self.carbon_intensity = carbon_intensity_kg_per_kwh

    def analyze_training_energy(
        self, training_data: list[dict[str, Any]], time_period_days: int = 30
    ) -> EnergyReport:
        """
        Analyze energy consumption for training runs.

        Args:
            training_data: List of training run data with energy metrics
            time_period_days: Analysis time period in days

        Returns:
            Energy analysis report
        """
        if not training_data:
            return self._create_empty_report("training", time_period_days)

        df = pd.DataFrame(training_data)

        # Calculate totals
        total_energy_wh = df["energy_consumption.total_energy_wh"].sum()
        total_carbon_kg = df["energy_consumption.carbon_footprint_kg"].sum()
        avg_power_w = df["energy_consumption.average_power_watts"].mean()
        peak_power_w = df["energy_consumption.peak_power_watts"].max()

        # Calculate cost
        cost_estimate_usd = (total_energy_wh / 1000) * self.electricity_cost_per_kwh

        # Efficiency metrics
        efficiency_metrics = self._calculate_training_efficiency(df)

        return EnergyReport(
            total_energy_wh=total_energy_wh,
            total_carbon_kg=total_carbon_kg,
            average_power_w=avg_power_w,
            peak_power_w=peak_power_w,
            total_training_runs=len(training_data),
            total_inference_runs=0,
            cost_estimate_usd=cost_estimate_usd,
            efficiency_metrics=efficiency_metrics,
            time_period=f"{time_period_days} days",
            generated_at=datetime.utcnow(),
        )

    def analyze_inference_energy(
        self, inference_data: list[dict[str, Any]], time_period_days: int = 30
    ) -> EnergyReport:
        """
        Analyze energy consumption for inference runs.

        Args:
            inference_data: List of inference run data with energy metrics
            time_period_days: Analysis time period in days

        Returns:
            Energy analysis report
        """
        if not inference_data:
            return self._create_empty_report("inference", time_period_days)

        df = pd.DataFrame(inference_data)

        # Calculate totals
        total_energy_wh = df["energy_consumed_wh"].sum()
        total_carbon_g = df["carbon_footprint_g"].sum()
        total_carbon_kg = total_carbon_g / 1000
        avg_power_w = df["avg_power_draw_w"].mean()
        peak_power_w = df["peak_power_draw_w"].max()

        # Calculate cost
        cost_estimate_usd = (total_energy_wh / 1000) * self.electricity_cost_per_kwh

        # Efficiency metrics
        efficiency_metrics = self._calculate_inference_efficiency(df)

        return EnergyReport(
            total_energy_wh=total_energy_wh,
            total_carbon_kg=total_carbon_kg,
            average_power_w=avg_power_w,
            peak_power_w=peak_power_w,
            total_training_runs=0,
            total_inference_runs=len(inference_data),
            cost_estimate_usd=cost_estimate_usd,
            efficiency_metrics=efficiency_metrics,
            time_period=f"{time_period_days} days",
            generated_at=datetime.utcnow(),
        )

    def compare_models_energy(
        self, model_data: list[dict[str, Any]], metric_key: str = "validation_accuracy"
    ) -> dict[str, Any]:
        """
        Compare energy efficiency across different models.

        Args:
            model_data: List of model data with performance and energy metrics
            metric_key: Performance metric to use for efficiency comparison

        Returns:
            Model comparison analysis
        """
        if not model_data:
            return {"error": "No model data provided"}

        df = pd.DataFrame(model_data)

        # Calculate energy efficiency (performance per Wh)
        df["energy_efficiency"] = (
            df[metric_key] / df["energy_consumption.total_energy_wh"]
        )

        # Rank models by efficiency
        df_sorted = df.sort_values("energy_efficiency", ascending=False)

        comparison = {
            "most_efficient_model": {
                "model_version": df_sorted.iloc[0]["model_version"],
                "efficiency": df_sorted.iloc[0]["energy_efficiency"],
                "energy_wh": df_sorted.iloc[0]["energy_consumption.total_energy_wh"],
                "performance": df_sorted.iloc[0][metric_key],
            },
            "least_efficient_model": {
                "model_version": df_sorted.iloc[-1]["model_version"],
                "efficiency": df_sorted.iloc[-1]["energy_efficiency"],
                "energy_wh": df_sorted.iloc[-1]["energy_consumption.total_energy_wh"],
                "performance": df_sorted.iloc[-1][metric_key],
            },
            "average_efficiency": df["energy_efficiency"].mean(),
            "total_models": len(model_data),
            "efficiency_improvement_potential": (
                df_sorted.iloc[0]["energy_efficiency"] - df["energy_efficiency"].mean()
            )
            / df["energy_efficiency"].mean()
            * 100,
        }

        return comparison

    def generate_carbon_footprint_report(
        self, training_data: list[dict[str, Any]], inference_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Generate comprehensive carbon footprint report.

        Args:
            training_data: Training run energy data
            inference_data: Inference run energy data

        Returns:
            Carbon footprint analysis
        """
        # Calculate training carbon footprint
        training_carbon = sum(
            run.get("energy_consumption", {}).get("carbon_footprint_kg", 0)
            for run in training_data
        )

        # Calculate inference carbon footprint
        inference_carbon = sum(
            run.get("carbon_footprint_g", 0) / 1000  # Convert g to kg
            for run in inference_data
        )

        total_carbon_kg = training_carbon + inference_carbon

        # Carbon equivalencies for context
        equivalencies = self._calculate_carbon_equivalencies(total_carbon_kg)

        return {
            "total_carbon_footprint_kg": total_carbon_kg,
            "training_carbon_kg": training_carbon,
            "inference_carbon_kg": inference_carbon,
            "carbon_per_training_run_kg": training_carbon / len(training_data)
            if training_data
            else 0,
            "carbon_per_inference_run_g": (inference_carbon * 1000)
            / len(inference_data)
            if inference_data
            else 0,
            "equivalencies": equivalencies,
            "reduction_recommendations": self._get_carbon_reduction_tips(
                total_carbon_kg
            ),
        }

    def _calculate_training_efficiency(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate training efficiency metrics."""
        return {
            "energy_per_epoch_wh": (
                df["energy_consumption.total_energy_wh"] / df["epochs_completed"]
            ).mean(),
            "energy_per_accuracy_point_wh": (
                df["energy_consumption.total_energy_wh"] / df["validation_accuracy"]
            ).mean(),
            "power_efficiency_score": (
                df["validation_accuracy"] / df["energy_consumption.average_power_watts"]
            ).mean(),
        }

    def _calculate_inference_efficiency(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate inference efficiency metrics."""
        return {
            "energy_per_prediction_wh": (
                df["energy_consumed_wh"] / df["total_predictions"]
            ).mean(),
            "predictions_per_wh": (
                df["total_predictions"] / df["energy_consumed_wh"]
            ).mean(),
            "power_per_prediction_w": (
                df["avg_power_draw_w"] / df["total_predictions"]
            ).mean(),
        }

    def _calculate_carbon_equivalencies(self, carbon_kg: float) -> dict[str, float]:
        """Calculate carbon footprint equivalencies for context."""
        return {
            "miles_driven_gasoline_car": carbon_kg * 2.31,  # miles
            "smartphone_charges": carbon_kg * 121.64,  # charges
            "hours_of_led_bulb": carbon_kg * 12195.12,  # hours
            "trees_needed_to_offset": carbon_kg * 0.0365,  # trees for 1 year
        }

    def _get_carbon_reduction_tips(self, carbon_kg: float) -> list[str]:
        """Get carbon footprint reduction recommendations."""
        tips = [
            "Consider training during off-peak hours when grid carbon intensity is lower",
            "Implement early stopping to avoid unnecessary training epochs",
            "Use model pruning and quantization to reduce inference energy",
            "Batch inference requests to improve GPU utilization efficiency",
        ]

        if carbon_kg > 1.0:
            tips.extend(
                [
                    "Consider using renewable energy sources for training",
                    "Implement model distillation to create smaller, more efficient models",
                    "Use cloud providers with strong renewable energy commitments",
                ]
            )

        return tips

    def _create_empty_report(
        self, report_type: str, time_period_days: int
    ) -> EnergyReport:
        """Create empty energy report."""
        return EnergyReport(
            total_energy_wh=0.0,
            total_carbon_kg=0.0,
            average_power_w=0.0,
            peak_power_w=0.0,
            total_training_runs=0 if report_type == "inference" else 0,
            total_inference_runs=0 if report_type == "training" else 0,
            cost_estimate_usd=0.0,
            efficiency_metrics={},
            time_period=f"{time_period_days} days",
            generated_at=datetime.utcnow(),
        )

    def export_report_to_dict(self, report: EnergyReport) -> dict[str, Any]:
        """Export energy report to dictionary format."""
        return {
            "total_energy_wh": report.total_energy_wh,
            "total_carbon_kg": report.total_carbon_kg,
            "average_power_w": report.average_power_w,
            "peak_power_w": report.peak_power_w,
            "total_training_runs": report.total_training_runs,
            "total_inference_runs": report.total_inference_runs,
            "cost_estimate_usd": report.cost_estimate_usd,
            "efficiency_metrics": report.efficiency_metrics,
            "time_period": report.time_period,
            "generated_at": report.generated_at.isoformat(),
        }


def create_energy_dashboard_data(
    training_data: list[dict[str, Any]],
    inference_data: list[dict[str, Any]],
    time_period_days: int = 30,
) -> dict[str, Any]:
    """
    Create data structure for energy consumption dashboard.

    Args:
        training_data: Training run energy data
        inference_data: Inference run energy data
        time_period_days: Analysis time period

    Returns:
        Dashboard data structure
    """
    analyzer = EnergyAnalyzer()

    # Generate reports
    training_report = analyzer.analyze_training_energy(training_data, time_period_days)
    inference_report = analyzer.analyze_inference_energy(
        inference_data, time_period_days
    )
    carbon_report = analyzer.generate_carbon_footprint_report(
        training_data, inference_data
    )

    # Combine all models for comparison
    all_models = training_data + inference_data
    model_comparison = analyzer.compare_models_energy(all_models) if all_models else {}

    return {
        "summary": {
            "total_energy_wh": training_report.total_energy_wh
            + inference_report.total_energy_wh,
            "total_carbon_kg": training_report.total_carbon_kg
            + inference_report.total_carbon_kg,
            "total_cost_usd": training_report.cost_estimate_usd
            + inference_report.cost_estimate_usd,
            "total_runs": training_report.total_training_runs
            + inference_report.total_inference_runs,
        },
        "training_analysis": analyzer.export_report_to_dict(training_report),
        "inference_analysis": analyzer.export_report_to_dict(inference_report),
        "carbon_footprint": carbon_report,
        "model_comparison": model_comparison,
        "generated_at": datetime.utcnow().isoformat(),
        "time_period": f"{time_period_days} days",
    }


def generate_energy_summary_text(dashboard_data: dict[str, Any]) -> str:
    """
    Generate human-readable energy consumption summary.

    Args:
        dashboard_data: Dashboard data from create_energy_dashboard_data

    Returns:
        Human-readable summary text
    """
    summary = dashboard_data["summary"]
    carbon = dashboard_data["carbon_footprint"]

    text = f"""
# Energy Consumption Report ({dashboard_data['time_period']})

## Summary
- **Total Energy Consumed**: {summary['total_energy_wh']:.2f} Wh
- **Total Carbon Footprint**: {summary['total_carbon_kg']:.4f} kg CO2
- **Estimated Cost**: ${summary['total_cost_usd']:.2f}
- **Total ML Runs**: {summary['total_runs']}

## Carbon Impact Context
Your ML workloads' carbon footprint is equivalent to:
- 🚗 {carbon['equivalencies']['miles_driven_gasoline_car']:.1f} miles driven in a gasoline car
- 📱 {carbon['equivalencies']['smartphone_charges']:.0f} smartphone charges
- 💡 {carbon['equivalencies']['hours_of_led_bulb']:.0f} hours of LED bulb usage
- 🌳 {carbon['equivalencies']['trees_needed_to_offset']:.3f} trees needed for 1 year to offset

## Recommendations
"""

    for tip in carbon.get("reduction_recommendations", []):
        text += f"- {tip}\n"

    return text
