"""GPU power consumption monitoring utilities for energy tracking."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""

    timestamp: datetime
    gpu_id: int
    power_draw_watts: float
    gpu_utilization_percent: float
    memory_utilization_percent: float
    temperature_celsius: float
    memory_used_mb: float
    memory_total_mb: float


@dataclass
class EnergyConsumption:
    """Energy consumption summary."""

    duration_seconds: float
    average_power_watts: float
    total_energy_wh: float  # Watt-hours
    peak_power_watts: float
    min_power_watts: float
    total_samples: int
    carbon_footprint_kg: float | None = None  # kg CO2 equivalent


class GPUPowerMonitor:
    """Monitor GPU power consumption during ML workloads."""

    def __init__(
        self,
        sampling_interval: float = 1.0,
        carbon_intensity_kg_per_kwh: float = 0.233,  # US average
    ):
        """
        Initialize GPU power monitor.

        Args:
            sampling_interval: Seconds between power measurements
            carbon_intensity_kg_per_kwh: Carbon intensity (kg CO2/kWh)
        """
        self.sampling_interval = sampling_interval
        self.carbon_intensity = carbon_intensity_kg_per_kwh
        self.metrics_history: list[GPUMetrics] = []
        self.is_monitoring = False
        # Timestamp of the last reset to delineate segments (e.g., per-epoch)
        self._last_reset_timestamp: datetime | None = None

    async def _get_gpu_metrics(self) -> list[GPUMetrics]:
        """Get current GPU metrics using nvidia-smi."""
        try:
            # Query nvidia-smi for power and utilization data
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,power.draw,utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.warning(f"nvidia-smi failed: {stderr.decode()}")
                return []

            metrics = []
            timestamp = datetime.utcnow()

            for line in stdout.decode().strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 7:
                    continue

                try:
                    gpu_id = int(parts[0])
                    power_draw = (
                        float(parts[1]) if parts[1] != "[Not Supported]" else 0.0
                    )
                    gpu_util = float(parts[2]) if parts[2] != "[Not Supported]" else 0.0
                    mem_util = float(parts[3]) if parts[3] != "[Not Supported]" else 0.0
                    temperature = (
                        float(parts[4]) if parts[4] != "[Not Supported]" else 0.0
                    )
                    mem_used = float(parts[5])
                    mem_total = float(parts[6])

                    metrics.append(
                        GPUMetrics(
                            timestamp=timestamp,
                            gpu_id=gpu_id,
                            power_draw_watts=power_draw,
                            gpu_utilization_percent=gpu_util,
                            memory_utilization_percent=mem_util,
                            temperature_celsius=temperature,
                            memory_used_mb=mem_used,
                            memory_total_mb=mem_total,
                        )
                    )

                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse GPU metrics line '{line}': {e}")
                    continue

            return metrics

        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return []

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("🔄 Starting GPU monitoring loop")
        try:
            while self.is_monitoring:
                try:
                    metrics = await self._get_gpu_metrics()
                    self.metrics_history.extend(metrics)

                    # Log current power draw
                    if metrics:
                        total_power = sum(m.power_draw_watts for m in metrics)
                        logger.info(
                            f"🔋 GPU power draw: {total_power:.1f}W across {len(metrics)} GPUs (samples: {len(self.metrics_history)})"
                        )
                    else:
                        logger.warning("No GPU metrics collected in this cycle")

                    await asyncio.sleep(self.sampling_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop iteration: {e}")
                    await asyncio.sleep(self.sampling_interval)
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
        finally:
            logger.info("🔄 GPU monitoring loop ended")

    async def start_monitoring(self) -> None:
        """Start monitoring GPU power consumption."""
        if self.is_monitoring:
            logger.warning("GPU monitoring already running")
            return

        self.is_monitoring = True
        self.metrics_history.clear()
        self._last_reset_timestamp = None

        # Check if nvidia-smi is available
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--list-gpus",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error("nvidia-smi not available. GPU monitoring disabled.")
                self.is_monitoring = False
                return

            gpu_count = len(
                [line for line in stdout.decode().strip().split("\n") if line.strip()]
            )
            logger.info(f"Starting GPU power monitoring for {gpu_count} GPUs")

        except Exception as e:
            logger.error(f"Failed to check GPU availability: {e}")
            self.is_monitoring = False
            return

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"🔄 Created monitoring task: {self._monitor_task}")

    async def stop_monitoring(self) -> EnergyConsumption:
        """Stop monitoring and return energy consumption summary."""
        if not self.is_monitoring:
            logger.warning("GPU monitoring not running")
            return EnergyConsumption(0, 0, 0, 0, 0, 0)

        self.is_monitoring = False

        if hasattr(self, "_monitor_task"):
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        return self._calculate_energy_consumption()

    def reset_history(self) -> None:
        """Reset collected metrics to start a new measurement segment."""
        logger.info(
            f"🔄 Resetting GPU monitoring history (had {len(self.metrics_history)} samples)"
        )
        logger.info(
            f"🔄 Monitoring status: is_monitoring={self.is_monitoring}, task_running={hasattr(self, '_monitor_task') and not self._monitor_task.done()}"
        )
        self.metrics_history.clear()
        self._last_reset_timestamp = datetime.utcnow()

    def summarize(self) -> EnergyConsumption:
        """Summarize energy consumption from currently collected samples without stopping."""
        logger.info(
            f"📊 Summarizing GPU energy from {len(self.metrics_history)} samples"
        )
        return self._calculate_energy_consumption()

    def _calculate_energy_consumption(self) -> EnergyConsumption:
        """Calculate total energy consumption from collected metrics."""
        if not self.metrics_history:
            logger.warning("No GPU metrics history available for energy calculation")
            return EnergyConsumption(0, 0, 0, 0, 0, 0)

        # Group metrics by timestamp to get total power per sample
        power_samples = {}
        for metric in self.metrics_history:
            timestamp_key = metric.timestamp.isoformat()
            if timestamp_key not in power_samples:
                power_samples[timestamp_key] = {
                    "timestamp": metric.timestamp,
                    "total_power": 0.0,
                }
            power_samples[timestamp_key]["total_power"] += metric.power_draw_watts

        samples = list(power_samples.values())
        if len(samples) < 2:
            return EnergyConsumption(0, 0, 0, 0, 0, len(samples))

        # Calculate duration
        start_time = min(s["timestamp"] for s in samples)
        end_time = max(s["timestamp"] for s in samples)
        duration_seconds = (end_time - start_time).total_seconds()

        # Calculate power statistics
        power_values = [s["total_power"] for s in samples]
        avg_power = sum(power_values) / len(power_values)
        peak_power = max(power_values)
        min_power = min(power_values)

        # Calculate total energy (Wh = W * h)
        total_energy_wh = avg_power * (duration_seconds / 3600)

        # Calculate carbon footprint
        carbon_footprint_kg = (total_energy_wh / 1000) * self.carbon_intensity

        logger.info(
            f"Energy consumption: {total_energy_wh:.3f} Wh, "
            f"Avg power: {avg_power:.1f}W, "
            f"Carbon footprint: {carbon_footprint_kg:.6f} kg CO2"
        )

        return EnergyConsumption(
            duration_seconds=duration_seconds,
            average_power_watts=avg_power,
            total_energy_wh=total_energy_wh,
            peak_power_watts=peak_power,
            min_power_watts=min_power,
            total_samples=len(samples),
            carbon_footprint_kg=carbon_footprint_kg,
        )

    @asynccontextmanager
    async def monitor_context(self) -> AsyncGenerator[None, None]:
        """Context manager for monitoring GPU power during a block of code."""
        await self.start_monitoring()
        try:
            yield
        finally:
            energy_consumption = await self.stop_monitoring()
            logger.info(f"GPU monitoring completed: {energy_consumption}")


# Convenience functions
async def monitor_gpu_during_training(
    training_function, *args, sampling_interval: float = 1.0, **kwargs
) -> tuple[Any, EnergyConsumption]:
    """
    Monitor GPU power consumption during model training.

    Returns:
        Tuple of (training_result, energy_consumption)
    """
    monitor = GPUPowerMonitor(sampling_interval=sampling_interval)

    async with monitor.monitor_context():
        result = await training_function(*args, **kwargs)

    energy_consumption = monitor._calculate_energy_consumption()
    return result, energy_consumption


def create_energy_metrics_dict(energy: EnergyConsumption) -> dict[str, Any]:
    """Convert EnergyConsumption to dictionary for database storage."""
    return {
        "duration_seconds": energy.duration_seconds,
        "average_power_watts": energy.average_power_watts,
        "total_energy_wh": energy.total_energy_wh,
        "peak_power_watts": energy.peak_power_watts,
        "min_power_watts": energy.min_power_watts,
        "total_samples": energy.total_samples,
        "carbon_footprint_kg": energy.carbon_footprint_kg,
    }
