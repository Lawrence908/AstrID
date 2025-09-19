"""Mock MLflow client for testing."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock


class MockMLflowClient:
    """Mock MLflow client for testing."""

    def __init__(self):
        self.experiments: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.models: dict[str, dict[str, Any]] = {}
        self.artifacts: dict[str, bytes] = {}
        self.current_run_id: str | None = None
        self.error_on_operation: dict[str, bool] = {}

    def create_experiment(self, name: str, artifact_location: str | None = None) -> str:
        """Create a new experiment."""
        if self.error_on_operation.get("create_experiment", False):
            raise Exception("Simulated experiment creation error")

        experiment_id = f"exp_{len(self.experiments)}"
        self.experiments[experiment_id] = {
            "experiment_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location or f"mock://artifacts/{name}",
            "lifecycle_stage": "active",
            "creation_time": int(time.time() * 1000),
        }
        return experiment_id

    def get_experiment_by_name(self, name: str) -> dict[str, Any] | None:
        """Get experiment by name."""
        for exp in self.experiments.values():
            if exp["name"] == name:
                return exp
        return None

    def create_run(
        self,
        experiment_id: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> MagicMock:
        """Create a new run."""
        if self.error_on_operation.get("create_run", False):
            raise Exception("Simulated run creation error")

        run_id = f"run_{len(self.runs)}"
        self.current_run_id = run_id

        run_data = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "name": run_name or f"run_{run_id}",
            "status": "RUNNING",
            "start_time": int(time.time() * 1000),
            "end_time": None,
            "params": {},
            "metrics": {},
            "tags": tags or {},
            "artifacts": [],
        }
        self.runs[run_id] = run_data

        # Return mock run object
        run_mock = MagicMock()
        run_mock.info.run_id = run_id
        run_mock.info.experiment_id = experiment_id
        run_mock.info.status = "RUNNING"
        return run_mock

    def start_run(
        self,
        run_id: str | None = None,
        experiment_id: str | None = None,
        run_name: str | None = None,
    ) -> MagicMock:
        """Start a run."""
        if run_id and run_id in self.runs:
            self.current_run_id = run_id
            run_data = self.runs[run_id]
        else:
            return self.create_run(experiment_id or "default", run_name)

        run_mock = MagicMock()
        run_mock.info.run_id = run_id
        run_mock.info.experiment_id = run_data["experiment_id"]
        run_mock.info.status = "RUNNING"
        return run_mock

    def end_run(self, run_id: str | None = None, status: str = "FINISHED") -> None:
        """End a run."""
        target_run_id = run_id or self.current_run_id
        if target_run_id and target_run_id in self.runs:
            self.runs[target_run_id]["status"] = status
            self.runs[target_run_id]["end_time"] = int(time.time() * 1000)

        if target_run_id == self.current_run_id:
            self.current_run_id = None

    def log_param(self, key: str, value: Any, run_id: str | None = None) -> None:
        """Log a parameter."""
        if self.error_on_operation.get("log_param", False):
            raise Exception("Simulated param logging error")

        target_run_id = run_id or self.current_run_id
        if target_run_id and target_run_id in self.runs:
            self.runs[target_run_id]["params"][key] = str(value)

    def log_params(self, params: dict[str, Any], run_id: str | None = None) -> None:
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value, run_id)

    def log_metric(
        self, key: str, value: float, step: int | None = None, run_id: str | None = None
    ) -> None:
        """Log a metric."""
        if self.error_on_operation.get("log_metric", False):
            raise Exception("Simulated metric logging error")

        target_run_id = run_id or self.current_run_id
        if target_run_id and target_run_id in self.runs:
            if key not in self.runs[target_run_id]["metrics"]:
                self.runs[target_run_id]["metrics"][key] = []

            metric_entry = {
                "value": value,
                "timestamp": int(time.time() * 1000),
                "step": step or 0,
            }
            self.runs[target_run_id]["metrics"][key].append(metric_entry)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step, run_id)

    def set_tag(self, key: str, value: str, run_id: str | None = None) -> None:
        """Set a tag."""
        target_run_id = run_id or self.current_run_id
        if target_run_id and target_run_id in self.runs:
            self.runs[target_run_id]["tags"][key] = value

    def set_tags(self, tags: dict[str, str], run_id: str | None = None) -> None:
        """Set multiple tags."""
        for key, value in tags.items():
            self.set_tag(key, value, run_id)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log an artifact."""
        if self.error_on_operation.get("log_artifact", False):
            raise Exception("Simulated artifact logging error")

        target_run_id = run_id or self.current_run_id
        if target_run_id and target_run_id in self.runs:
            artifact_key = f"{target_run_id}/{artifact_path or local_path}"

            # Simulate reading file content
            self.artifacts[artifact_key] = f"mock_content_of_{local_path}".encode()
            self.runs[target_run_id]["artifacts"].append(artifact_key)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log a model."""
        target_run_id = run_id or self.current_run_id
        if target_run_id and target_run_id in self.runs:
            model_key = f"{target_run_id}/{artifact_path}"
            self.artifacts[model_key] = b"mock_model_content"
            self.runs[target_run_id]["artifacts"].append(model_key)

            if registered_model_name:
                self.register_model(model_key, registered_model_name)

    def register_model(
        self, model_uri: str, name: str, tags: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Register a model."""
        if name not in self.models:
            self.models[name] = {"name": name, "versions": []}

        version = len(self.models[name]["versions"]) + 1
        model_version = {
            "name": name,
            "version": str(version),
            "source": model_uri,
            "stage": "None",
            "tags": tags or {},
            "creation_timestamp": int(time.time() * 1000),
        }

        self.models[name]["versions"].append(model_version)
        return model_version

    def transition_model_version_stage(
        self, name: str, version: str, stage: str
    ) -> dict[str, Any]:
        """Transition model version stage."""
        if name in self.models:
            for model_version in self.models[name]["versions"]:
                if model_version["version"] == version:
                    model_version["stage"] = stage
                    return model_version

        raise Exception(f"Model version {name}:{version} not found")

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Get run by ID."""
        if run_id in self.runs:
            return self.runs[run_id]
        raise Exception(f"Run {run_id} not found")

    def search_runs(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str = "",
        max_results: int = 1000,
    ) -> list[dict[str, Any]]:
        """Search runs."""
        runs = list(self.runs.values())

        if experiment_ids:
            runs = [r for r in runs if r["experiment_id"] in experiment_ids]

        # Simple filter implementation
        if filter_string:
            # For testing, just return all runs
            pass

        return runs[:max_results]

    def simulate_error(self, operation: str, should_error: bool = True) -> None:
        """Enable/disable error simulation for specific operations."""
        self.error_on_operation[operation] = should_error

    def clear_all_data(self) -> None:
        """Clear all stored data."""
        self.experiments.clear()
        self.runs.clear()
        self.models.clear()
        self.artifacts.clear()
        self.current_run_id = None

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "experiments": len(self.experiments),
            "runs": len(self.runs),
            "models": len(self.models),
            "artifacts": len(self.artifacts),
            "current_run_id": self.current_run_id,
        }
