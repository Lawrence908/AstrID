"""Mock Prefect client for testing."""

from __future__ import annotations

import asyncio
import time
from typing import Any


class MockPrefectClient:
    """Mock Prefect client for testing."""

    def __init__(self):
        self.flows: dict[str, dict[str, Any]] = {}
        self.flow_runs: dict[str, dict[str, Any]] = {}
        self.deployments: dict[str, dict[str, Any]] = {}
        self.task_runs: dict[str, dict[str, Any]] = {}
        self.work_queues: dict[str, dict[str, Any]] = {}
        self.error_on_operation: dict[str, bool] = {}

    async def create_flow(
        self, name: str, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Create a new flow."""
        if self.error_on_operation.get("create_flow", False):
            raise Exception("Simulated flow creation error")

        flow_id = f"flow_{len(self.flows)}"
        flow_data = {
            "id": flow_id,
            "name": name,
            "tags": tags or [],
            "created": time.time(),
            "updated": time.time(),
        }
        self.flows[flow_id] = flow_data
        return flow_data

    async def create_flow_run(
        self,
        flow_id: str | None = None,
        name: str | None = None,
        parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        state: str = "PENDING",
    ) -> dict[str, Any]:
        """Create a new flow run."""
        if self.error_on_operation.get("create_flow_run", False):
            raise Exception("Simulated flow run creation error")

        flow_run_id = f"flow_run_{len(self.flow_runs)}"
        flow_run_data = {
            "id": flow_run_id,
            "flow_id": flow_id,
            "name": name or f"flow-run-{flow_run_id}",
            "parameters": parameters or {},
            "tags": tags or [],
            "state": {"type": state, "name": state, "timestamp": time.time()},
            "created": time.time(),
            "start_time": None,
            "end_time": None,
            "total_run_time": 0.0,
            "estimated_run_time": 0.0,
            "estimated_start_time_delta": 0.0,
        }
        self.flow_runs[flow_run_id] = flow_run_data
        return flow_run_data

    async def read_flow_run(self, flow_run_id: str) -> dict[str, Any]:
        """Read a flow run by ID."""
        if flow_run_id not in self.flow_runs:
            raise Exception(f"Flow run {flow_run_id} not found")
        return self.flow_runs[flow_run_id]

    async def set_flow_run_state(
        self, flow_run_id: str, state: str, message: str | None = None
    ) -> dict[str, Any]:
        """Set flow run state."""
        if flow_run_id not in self.flow_runs:
            raise Exception(f"Flow run {flow_run_id} not found")

        flow_run = self.flow_runs[flow_run_id]
        old_state = flow_run["state"]["type"]

        flow_run["state"] = {
            "type": state,
            "name": state,
            "message": message,
            "timestamp": time.time(),
        }

        # Update timing information
        if state == "RUNNING" and old_state == "PENDING":
            flow_run["start_time"] = time.time()
        elif state in ["COMPLETED", "FAILED", "CANCELLED"] and flow_run["start_time"]:
            flow_run["end_time"] = time.time()
            flow_run["total_run_time"] = flow_run["end_time"] - flow_run["start_time"]

        return flow_run

    async def create_deployment(
        self,
        flow_id: str,
        name: str,
        work_queue_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        schedule: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new deployment."""
        if self.error_on_operation.get("create_deployment", False):
            raise Exception("Simulated deployment creation error")

        deployment_id = f"deployment_{len(self.deployments)}"
        deployment_data = {
            "id": deployment_id,
            "flow_id": flow_id,
            "name": name,
            "work_queue_name": work_queue_name,
            "parameters": parameters or {},
            "schedule": schedule,
            "is_schedule_active": schedule is not None,
            "created": time.time(),
            "updated": time.time(),
        }
        self.deployments[deployment_id] = deployment_data
        return deployment_data

    async def create_work_queue(
        self, name: str, description: str | None = None, is_paused: bool = False
    ) -> dict[str, Any]:
        """Create a new work queue."""
        work_queue_id = f"queue_{len(self.work_queues)}"
        work_queue_data = {
            "id": work_queue_id,
            "name": name,
            "description": description,
            "is_paused": is_paused,
            "created": time.time(),
            "updated": time.time(),
        }
        self.work_queues[work_queue_id] = work_queue_data
        return work_queue_data

    async def read_work_queue(self, work_queue_id: str) -> dict[str, Any]:
        """Read a work queue by ID."""
        if work_queue_id not in self.work_queues:
            raise Exception(f"Work queue {work_queue_id} not found")
        return self.work_queues[work_queue_id]

    async def get_flow_runs(
        self,
        flow_filter: dict[str, Any] | None = None,
        flow_run_filter: dict[str, Any] | None = None,
        sort: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get flow runs with filtering."""
        runs = list(self.flow_runs.values())

        # Apply filters (simplified for testing)
        if flow_filter:
            if "name" in flow_filter:
                flow_names = flow_filter["name"]["any_"]
                runs = [r for r in runs if r.get("name") in flow_names]

        if flow_run_filter:
            if "state" in flow_run_filter:
                states = flow_run_filter["state"]["type"]["any_"]
                runs = [r for r in runs if r["state"]["type"] in states]

        # Apply sorting
        if sort == "CREATED_DESC":
            runs.sort(key=lambda x: x["created"], reverse=True)
        elif sort == "START_TIME_ASC":
            runs.sort(key=lambda x: x.get("start_time", 0))

        # Apply pagination
        return runs[offset : offset + limit]

    async def cancel_flow_run(self, flow_run_id: str) -> dict[str, Any]:
        """Cancel a flow run."""
        return await self.set_flow_run_state(flow_run_id, "CANCELLED")

    async def pause_deployment(self, deployment_id: str) -> None:
        """Pause a deployment."""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["is_schedule_active"] = False

    async def resume_deployment(self, deployment_id: str) -> None:
        """Resume a deployment."""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["is_schedule_active"] = True

    async def create_task_run(
        self,
        flow_run_id: str,
        task_key: str,
        name: str | None = None,
        state: str = "PENDING",
    ) -> dict[str, Any]:
        """Create a task run."""
        task_run_id = f"task_run_{len(self.task_runs)}"
        task_run_data = {
            "id": task_run_id,
            "flow_run_id": flow_run_id,
            "task_key": task_key,
            "name": name or task_key,
            "state": {"type": state, "name": state, "timestamp": time.time()},
            "created": time.time(),
            "start_time": None,
            "end_time": None,
            "total_run_time": 0.0,
        }
        self.task_runs[task_run_id] = task_run_data
        return task_run_data

    async def set_task_run_state(
        self, task_run_id: str, state: str, message: str | None = None
    ) -> dict[str, Any]:
        """Set task run state."""
        if task_run_id not in self.task_runs:
            raise Exception(f"Task run {task_run_id} not found")

        task_run = self.task_runs[task_run_id]
        old_state = task_run["state"]["type"]

        task_run["state"] = {
            "type": state,
            "name": state,
            "message": message,
            "timestamp": time.time(),
        }

        # Update timing information
        if state == "RUNNING" and old_state == "PENDING":
            task_run["start_time"] = time.time()
        elif state in ["COMPLETED", "FAILED"] and task_run["start_time"]:
            task_run["end_time"] = time.time()
            task_run["total_run_time"] = task_run["end_time"] - task_run["start_time"]

        return task_run

    async def get_server_info(self) -> dict[str, Any]:
        """Get server information."""
        return {
            "server_type": "mock",
            "version": "2.14.0-mock",
            "api_version": "0.8.4",
            "database_version": "mock",
            "created": time.time(),
        }

    async def hello(self) -> str:
        """Health check endpoint."""
        return "Hello from mock Prefect server!"

    def simulate_error(self, operation: str, should_error: bool = True) -> None:
        """Enable/disable error simulation for specific operations."""
        self.error_on_operation[operation] = should_error

    def clear_all_data(self) -> None:
        """Clear all stored data."""
        self.flows.clear()
        self.flow_runs.clear()
        self.deployments.clear()
        self.task_runs.clear()
        self.work_queues.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "flows": len(self.flows),
            "flow_runs": len(self.flow_runs),
            "deployments": len(self.deployments),
            "task_runs": len(self.task_runs),
            "work_queues": len(self.work_queues),
        }

    async def simulate_flow_execution(
        self, flow_run_id: str, duration: float = 1.0, success: bool = True
    ) -> None:
        """Simulate flow execution with realistic state transitions."""
        await self.set_flow_run_state(flow_run_id, "RUNNING")
        await asyncio.sleep(duration)

        final_state = "COMPLETED" if success else "FAILED"
        await self.set_flow_run_state(flow_run_id, final_state)
