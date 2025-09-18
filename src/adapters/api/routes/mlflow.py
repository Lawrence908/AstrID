"""
MLflow API routes for AstrID.

This module provides REST API endpoints for MLflow integration including
experiment management, model registry, and model versioning.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.api.response_wrapper import create_response
from src.infrastructure.mlflow import (
    ExperimentTracker,
    MLflowConfig,
    MLflowServer,
    ModelRegistry,
    ModelVersioning,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/mlflow", tags=["mlflow"])

# Global instances (would be injected via dependency injection in production)
_mlflow_config: MLflowConfig | None = None
_experiment_tracker: ExperimentTracker | None = None
_model_registry: ModelRegistry | None = None
_model_versioning: ModelVersioning | None = None
_mlflow_server: MLflowServer | None = None


def get_mlflow_config() -> MLflowConfig:
    """Get MLflow configuration."""
    global _mlflow_config
    if _mlflow_config is None:
        _mlflow_config = MLflowConfig.from_env()
    return _mlflow_config


def get_experiment_tracker() -> ExperimentTracker:
    """Get experiment tracker instance."""
    global _experiment_tracker
    if _experiment_tracker is None:
        config = get_mlflow_config()
        _experiment_tracker = ExperimentTracker(config)
    return _experiment_tracker


def get_model_registry() -> ModelRegistry:
    """Get model registry instance."""
    global _model_registry
    if _model_registry is None:
        config = get_mlflow_config()
        _model_registry = ModelRegistry(config)
    return _model_registry


def get_model_versioning() -> ModelVersioning:
    """Get model versioning instance."""
    global _model_versioning
    if _model_versioning is None:
        config = get_mlflow_config()
        _model_versioning = ModelVersioning(config)
    return _model_versioning


def get_mlflow_server() -> MLflowServer:
    """Get MLflow server instance."""
    global _mlflow_server
    if _mlflow_server is None:
        config = get_mlflow_config()
        _mlflow_server = MLflowServer(config)
    return _mlflow_server


# Pydantic models for request/response
class ExperimentCreateRequest(BaseModel):
    """Experiment creation request."""

    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")


class RunCreateRequest(BaseModel):
    """Run creation request."""

    experiment_id: str = Field(..., description="Experiment ID")
    run_name: str | None = Field(None, description="Run name")
    tags: dict[str, str] | None = Field(None, description="Run tags")


class ParameterLogRequest(BaseModel):
    """Parameter logging request."""

    run_id: str = Field(..., description="Run ID")
    parameters: dict[str, Any] = Field(..., description="Parameters to log")


class MetricLogRequest(BaseModel):
    """Metric logging request."""

    run_id: str = Field(..., description="Run ID")
    metrics: dict[str, float] = Field(..., description="Metrics to log")
    step: int | None = Field(None, description="Step number")


class ArtifactLogRequest(BaseModel):
    """Artifact logging request."""

    run_id: str = Field(..., description="Run ID")
    artifacts: dict[str, str] = Field(..., description="Artifact paths")


class ModelRegisterRequest(BaseModel):
    """Model registration request."""

    model_path: str = Field(..., description="Path to model artifact")
    model_name: str = Field(..., description="Model name")
    run_id: str = Field(..., description="MLflow run ID")
    description: str | None = Field(None, description="Model description")
    tags: dict[str, str] | None = Field(None, description="Model tags")


class ModelStageTransitionRequest(BaseModel):
    """Model stage transition request."""

    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Target stage")
    archive_existing_versions: bool = Field(
        True, description="Archive existing versions"
    )


class ModelVersionCreateRequest(BaseModel):
    """Model version creation request."""

    model_name: str = Field(..., description="Model name")
    model_data: str = Field(..., description="Base64 encoded model data")
    metadata: dict[str, Any] = Field(..., description="Model metadata")
    version_type: str = Field("patch", description="Version type (major, minor, patch)")


# Server management endpoints
@router.get("/server/status")
async def get_server_status(
    server: MLflowServer = Depends(get_mlflow_server),
) -> dict[str, Any]:
    """Get MLflow server status."""
    try:
        status = server.get_server_status()
        return create_response(
            data={
                "is_running": status.is_running,
                "health_status": status.health_status,
                "response_time_ms": status.response_time_ms,
                "version": status.version,
                "error_message": status.error_message,
            },
            message="Server status retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to get server status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get server status: {str(e)}",
        ) from e


@router.post("/server/start")
async def start_server(
    host: str = Query("0.0.0.0", description="Server host"),
    port: int = Query(5000, description="Server port"),
    server: MLflowServer = Depends(get_mlflow_server),
) -> dict[str, Any]:
    """Start MLflow server."""
    try:
        server.start_tracking_server(host, port)
        return create_response(
            data={"host": host, "port": port},
            message="MLflow server started successfully",
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start server: {str(e)}",
        ) from e


@router.post("/server/stop")
async def stop_server(
    server: MLflowServer = Depends(get_mlflow_server),
) -> dict[str, Any]:
    """Stop MLflow server."""
    try:
        server.stop_tracking_server()
        return create_response(data={}, message="MLflow server stopped successfully")
    except Exception as e:
        logger.error(f"Failed to stop server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop server: {str(e)}",
        ) from e


# Experiment management endpoints
@router.post("/experiments")
async def create_experiment(
    request: ExperimentCreateRequest,
    tracker: ExperimentTracker = Depends(get_experiment_tracker),
) -> dict[str, Any]:
    """Create a new experiment."""
    try:
        experiment_id = tracker.create_experiment(request.name, request.description)
        return create_response(
            data={"experiment_id": experiment_id, "name": request.name},
            message="Experiment created successfully",
        )
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create experiment: {str(e)}",
        ) from e


@router.get("/experiments")
async def list_experiments(
    tracker: ExperimentTracker = Depends(get_experiment_tracker),
) -> dict[str, Any]:
    """List all experiments."""
    try:
        experiments = tracker.list_experiments()
        return create_response(
            data=[
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "description": exp.description,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "creation_time": exp.creation_time,
                    "last_update_time": exp.last_update_time,
                }
                for exp in experiments
            ],
            message="Experiments retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiments: {str(e)}",
        ) from e


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str, tracker: ExperimentTracker = Depends(get_experiment_tracker)
) -> dict[str, Any]:
    """Get experiment information."""
    try:
        experiment = tracker.get_experiment(experiment_id)
        return create_response(
            data={
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "description": experiment.description,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
                "creation_time": experiment.creation_time,
                "last_update_time": experiment.last_update_time,
            },
            message="Experiment retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to get experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment: {str(e)}",
        ) from e


# Run management endpoints
@router.post("/experiments/{experiment_id}/runs")
async def start_run(
    experiment_id: str,
    request: RunCreateRequest,
    tracker: ExperimentTracker = Depends(get_experiment_tracker),
) -> dict[str, Any]:
    """Start a new run."""
    try:
        run_id = tracker.start_run(experiment_id, request.run_name, request.tags)
        return create_response(
            data={"run_id": run_id, "experiment_id": experiment_id},
            message="Run started successfully",
        )
    except Exception as e:
        logger.error(f"Failed to start run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start run: {str(e)}",
        ) from e


@router.get("/experiments/{experiment_id}/runs")
async def list_runs(
    experiment_id: str,
    filter_string: str = Query("", description="MLflow filter string"),
    max_results: int = Query(1000, description="Maximum number of results"),
    tracker: ExperimentTracker = Depends(get_experiment_tracker),
) -> dict[str, Any]:
    """List runs in an experiment."""
    try:
        runs = tracker.search_runs(experiment_id, filter_string, max_results)
        return create_response(
            data=[
                {
                    "run_id": run.run_id,
                    "experiment_id": run.experiment_id,
                    "name": run.name,
                    "status": run.status,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "user_id": run.user_id,
                    "tags": run.tags,
                }
                for run in runs
            ],
            message="Runs retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list runs: {str(e)}",
        ) from e


@router.post("/runs/{run_id}/parameters")
async def log_parameters(
    run_id: str,
    request: ParameterLogRequest,
    tracker: ExperimentTracker = Depends(get_experiment_tracker),
) -> dict[str, Any]:
    """Log parameters to a run."""
    try:
        tracker.log_parameters(request.parameters, run_id)
        return create_response(
            data={"run_id": run_id, "parameters": request.parameters},
            message="Parameters logged successfully",
        )
    except Exception as e:
        logger.error(f"Failed to log parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log parameters: {str(e)}",
        ) from e


@router.post("/runs/{run_id}/metrics")
async def log_metrics(
    run_id: str,
    request: MetricLogRequest,
    tracker: ExperimentTracker = Depends(get_experiment_tracker),
) -> dict[str, Any]:
    """Log metrics to a run."""
    try:
        tracker.log_metrics(request.metrics, request.step, run_id)
        return create_response(
            data={"run_id": run_id, "metrics": request.metrics},
            message="Metrics logged successfully",
        )
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log metrics: {str(e)}",
        ) from e


@router.post("/runs/{run_id}/artifacts")
async def log_artifacts(
    run_id: str,
    request: ArtifactLogRequest,
    tracker: ExperimentTracker = Depends(get_experiment_tracker),
) -> dict[str, Any]:
    """Log artifacts to a run."""
    try:
        tracker.log_artifacts(request.artifacts, run_id)
        return create_response(
            data={"run_id": run_id, "artifacts": request.artifacts},
            message="Artifacts logged successfully",
        )
    except Exception as e:
        logger.error(f"Failed to log artifacts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log artifacts: {str(e)}",
        ) from e


@router.post("/runs/{run_id}/end")
async def end_run(
    run_id: str,
    status: str = Query("FINISHED", description="Run status"),
    tracker: ExperimentTracker = Depends(get_experiment_tracker),
) -> dict[str, Any]:
    """End a run."""
    try:
        tracker.end_run(run_id, status)
        return create_response(
            data={"run_id": run_id, "status": status}, message="Run ended successfully"
        )
    except Exception as e:
        logger.error(f"Failed to end run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end run: {str(e)}",
        ) from e


# Model registry endpoints
@router.post("/models/register")
async def register_model(
    request: ModelRegisterRequest, registry: ModelRegistry = Depends(get_model_registry)
) -> dict[str, Any]:
    """Register a model in the model registry."""
    try:
        version = registry.register_model(
            request.model_path,
            request.model_name,
            request.run_id,
            request.description,
            request.tags,
        )
        return create_response(
            data={"model_name": request.model_name, "version": version},
            message="Model registered successfully",
        )
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register model: {str(e)}",
        ) from e


@router.get("/models")
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
) -> dict[str, Any]:
    """List all registered models."""
    try:
        models = registry.list_registered_models()
        return create_response(
            data=[
                {
                    "name": model.name,
                    "description": model.description,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "tags": model.tags,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.stage,
                            "description": v.description,
                            "creation_timestamp": v.creation_timestamp,
                        }
                        for v in model.latest_versions
                    ]
                    if model.latest_versions
                    else [],
                }
                for model in models
            ],
            message="Models retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        ) from e


@router.get("/models/{model_name}/versions")
async def list_model_versions(
    model_name: str, registry: ModelRegistry = Depends(get_model_registry)
) -> dict[str, Any]:
    """List versions of a model."""
    try:
        versions = registry.list_model_versions(model_name)
        return create_response(
            data=[
                {
                    "name": v.name,
                    "version": v.version,
                    "stage": v.stage,
                    "description": v.description,
                    "user_id": v.user_id,
                    "creation_timestamp": v.creation_timestamp,
                    "last_updated_timestamp": v.last_updated_timestamp,
                    "run_id": v.run_id,
                    "status": v.status,
                    "tags": v.tags,
                }
                for v in versions
            ],
            message="Model versions retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list model versions: {str(e)}",
        ) from e


@router.post("/models/{model_name}/versions/{version}/transition")
async def transition_model_stage(
    model_name: str,
    version: str,
    request: ModelStageTransitionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
) -> dict[str, Any]:
    """Transition model version to a new stage."""
    try:
        registry.transition_model_stage(
            model_name, version, request.stage, request.archive_existing_versions
        )
        return create_response(
            data={"model_name": model_name, "version": version, "stage": request.stage},
            message="Model stage transitioned successfully",
        )
    except Exception as e:
        logger.error(f"Failed to transition model stage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transition model stage: {str(e)}",
        ) from e


@router.get("/models/{model_name}/latest")
async def get_latest_model(
    model_name: str,
    stage: str = Query("None", description="Model stage"),
    registry: ModelRegistry = Depends(get_model_registry),
) -> dict[str, Any]:
    """Get the latest model version."""
    try:
        model = registry.get_latest_model(model_name, stage)
        return create_response(
            data={
                "name": model.name,
                "version": model.version,
                "stage": model.stage,
                "description": model.description,
                "user_id": model.user_id,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "run_id": model.run_id,
                "status": model.status,
                "tags": model.tags,
            },
            message="Latest model retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to get latest model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get latest model: {str(e)}",
        ) from e


@router.get("/models/{model_name}/versions/{version}")
async def get_model_version(
    model_name: str, version: str, registry: ModelRegistry = Depends(get_model_registry)
) -> dict[str, Any]:
    """Get model version information."""
    try:
        model_version = registry.get_model_version(model_name, version)
        return create_response(
            data={
                "name": model_version.name,
                "version": model_version.version,
                "stage": model_version.stage,
                "description": model_version.description,
                "user_id": model_version.user_id,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "run_id": model_version.run_id,
                "status": model_version.status,
                "tags": model_version.tags,
            },
            message="Model version retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to get model version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model version: {str(e)}",
        ) from e


@router.delete("/models/{model_name}/versions/{version}")
async def delete_model_version(
    model_name: str, version: str, registry: ModelRegistry = Depends(get_model_registry)
) -> dict[str, Any]:
    """Delete a model version."""
    try:
        registry.delete_model_version(model_name, version)
        return create_response(
            data={"model_name": model_name, "version": version},
            message="Model version deleted successfully",
        )
    except Exception as e:
        logger.error(f"Failed to delete model version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model version: {str(e)}",
        ) from e


# Model versioning endpoints
@router.post("/models/{model_name}/versions")
async def create_model_version(
    model_name: str,
    request: ModelVersionCreateRequest,
    versioning: ModelVersioning = Depends(get_model_versioning),
) -> dict[str, Any]:
    """Create a new model version."""
    try:
        import base64

        # Decode base64 model data
        model_data = base64.b64decode(request.model_data)

        # Convert version type string to enum
        from ...infrastructure.mlflow.model_versioning import VersionType

        version_type = VersionType(request.version_type)

        version = versioning.create_model_version(
            model_name, model_data, request.metadata, version_type
        )

        return create_response(
            data={"model_name": model_name, "version": version},
            message="Model version created successfully",
        )
    except Exception as e:
        logger.error(f"Failed to create model version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model version: {str(e)}",
        ) from e


@router.get("/models/{model_name}/versions/{version}/data")
async def get_model_version_data(
    model_name: str,
    version: str,
    versioning: ModelVersioning = Depends(get_model_versioning),
) -> dict[str, Any]:
    """Get model version data."""
    try:
        model_data = versioning.get_model_version(model_name, version)

        # Encode as base64 for JSON response
        import base64

        encoded_data = base64.b64encode(model_data).decode("utf-8")

        return create_response(
            data={
                "model_name": model_name,
                "version": version,
                "data": encoded_data,
                "size_bytes": len(model_data),
            },
            message="Model version data retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to get model version data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model version data: {str(e)}",
        ) from e


@router.get("/models/{model_name}/versions/{version1}/compare/{version2}")
async def compare_model_versions(
    model_name: str,
    version1: str,
    version2: str,
    versioning: ModelVersioning = Depends(get_model_versioning),
) -> dict[str, Any]:
    """Compare two model versions."""
    try:
        comparison = versioning.compare_model_versions(model_name, version1, version2)
        return create_response(
            data=comparison, message="Model versions compared successfully"
        )
    except Exception as e:
        logger.error(f"Failed to compare model versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare model versions: {str(e)}",
        ) from e


@router.post("/models/{model_name}/versions/{version}/rollback")
async def rollback_model_version(
    model_name: str,
    version: str,
    versioning: ModelVersioning = Depends(get_model_versioning),
) -> dict[str, Any]:
    """Rollback model to a specific version."""
    try:
        versioning.rollback_model_version(model_name, version)
        return create_response(
            data={"model_name": model_name, "version": version},
            message="Model rolled back successfully",
        )
    except Exception as e:
        logger.error(f"Failed to rollback model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback model: {str(e)}",
        ) from e


@router.post("/models/{model_name}/versions/{version}/archive")
async def archive_model_version(
    model_name: str,
    version: str,
    versioning: ModelVersioning = Depends(get_model_versioning),
) -> dict[str, Any]:
    """Archive a model version."""
    try:
        versioning.archive_model_version(model_name, version)
        return create_response(
            data={"model_name": model_name, "version": version},
            message="Model version archived successfully",
        )
    except Exception as e:
        logger.error(f"Failed to archive model version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to archive model version: {str(e)}",
        ) from e
