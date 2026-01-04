#!/usr/bin/env python3
"""Example script demonstrating energy tracking capabilities for ML workloads."""

import asyncio
import json
from datetime import datetime

from src.core.energy_analysis import (
    EnergyAnalyzer,
    create_energy_dashboard_data,
    generate_energy_summary_text,
)
from src.core.gpu_monitoring import GPUPowerMonitor
from src.core.mlflow_energy import create_energy_tracker


async def demo_training_energy_tracking():
    """Demonstrate energy tracking during a mock training session."""
    print("🔋 Starting GPU energy tracking demo for model training...")

    # Initialize monitoring
    gpu_monitor = GPUPowerMonitor(sampling_interval=0.5)  # Sample every 0.5 seconds
    energy_tracker = create_energy_tracker("demo-training")

    print("📊 Starting GPU power monitoring...")
    await gpu_monitor.start_monitoring()

    # Simulate training workload
    print("🤖 Simulating model training (10 seconds)...")
    await asyncio.sleep(10)

    # Stop monitoring and get results
    print("⏹️  Stopping monitoring and calculating energy consumption...")
    energy_consumption = await gpu_monitor.stop_monitoring()

    # Log to MLflow
    model_version = f"demo_model_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_params = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
    }
    performance_metrics = {
        "validation_accuracy": 0.92,
        "validation_loss": 0.08,
    }

    mlflow_run_id = energy_tracker.log_training_energy(
        energy_consumption=energy_consumption,
        model_version=model_version,
        model_params=model_params,
        performance_metrics=performance_metrics,
    )

    # Print results
    print("\n" + "=" * 60)
    print("🎯 ENERGY TRACKING RESULTS")
    print("=" * 60)
    print(f"⚡ Energy Consumed: {energy_consumption.total_energy_wh:.3f} Wh")
    print(f"🔌 Average Power: {energy_consumption.average_power_watts:.1f} W")
    print(f"📈 Peak Power: {energy_consumption.peak_power_watts:.1f} W")
    print(f"⏱️  Duration: {energy_consumption.duration_seconds:.1f} seconds")
    print(f"🌍 Carbon Footprint: {energy_consumption.carbon_footprint_kg:.6f} kg CO2")
    print(f"📊 Power Samples: {energy_consumption.total_samples}")

    if mlflow_run_id:
        print(f"📝 MLflow Run ID: {mlflow_run_id}")

    return {
        "model_version": model_version,
        "energy_consumption": {
            "total_energy_wh": energy_consumption.total_energy_wh,
            "average_power_watts": energy_consumption.average_power_watts,
            "peak_power_watts": energy_consumption.peak_power_watts,
            "duration_seconds": energy_consumption.duration_seconds,
            "carbon_footprint_kg": energy_consumption.carbon_footprint_kg,
        },
        **performance_metrics,
        "epochs_completed": 50,
        "mlflow_run_id": mlflow_run_id,
    }


def demo_energy_analysis(training_runs_data):
    """Demonstrate energy analysis capabilities."""
    print("\n" + "=" * 60)
    print("📈 ENERGY ANALYSIS DEMO")
    print("=" * 60)

    # Create analyzer
    analyzer = EnergyAnalyzer(
        electricity_cost_per_kwh=0.12,  # $0.12 per kWh
        carbon_intensity_kg_per_kwh=0.233,  # US average
    )

    # Analyze training energy
    training_report = analyzer.analyze_training_energy(
        training_runs_data, time_period_days=1
    )

    print("📊 Training Energy Analysis:")
    print(f"   Total Energy: {training_report.total_energy_wh:.3f} Wh")
    print(f"   Carbon Footprint: {training_report.total_carbon_kg:.6f} kg CO2")
    print(f"   Estimated Cost: ${training_report.cost_estimate_usd:.4f}")
    print(f"   Training Runs: {training_report.total_training_runs}")

    # Create dashboard data
    dashboard_data = create_energy_dashboard_data(
        training_data=training_runs_data,
        inference_data=[],  # No inference data in this demo
        time_period_days=1,
    )

    # Generate summary text
    summary_text = generate_energy_summary_text(dashboard_data)
    print("\n📋 ENERGY REPORT:")
    print(summary_text)

    return dashboard_data


async def main():
    """Run the energy tracking demonstration."""
    print("🚀 AstrID Energy Tracking System Demo")
    print("=" * 60)

    # Check if nvidia-smi is available
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_count = len(
                [line for line in result.stdout.strip().split("\n") if line.strip()]
            )
            print(f"✅ Found {gpu_count} GPU(s) for monitoring")
        else:
            print("⚠️  No GPUs detected - will simulate energy data")
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - will simulate energy data")

    # Run training demo
    training_data = await demo_training_energy_tracking()

    # Run analysis demo
    dashboard_data = demo_energy_analysis([training_data])

    # Save results to file
    output_file = (
        f"energy_tracking_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_file}")
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print(
        "1. Run your actual training with: python -m src.adapters.scheduler.flows.model_training"
    )
    print("2. Check MLflow UI at http://localhost:9003 for energy metrics")
    print("3. Use the energy analysis tools in your notebooks")


if __name__ == "__main__":
    asyncio.run(main())
