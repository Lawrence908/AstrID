"""Central import file for all SQLAlchemy database models."""

# Import all domain models to ensure they're registered with SQLAlchemy
from src.domains.catalog.models import AuditLog, ProcessingJob, SystemConfig
from src.domains.curation.models import Alert, ValidationEvent
from src.domains.detection.models import Detection, Model, ModelRun
from src.domains.differencing.models import Candidate, DifferenceRun
from src.domains.observations.models import Observation, Survey
from src.domains.preprocessing.models import PreprocessRun

# ExampleModel removed - migration to domain models complete

# Export all models for Alembic autogenerate
__all__ = [
    # Observations domain
    "Survey",
    "Observation",
    # Detection domain
    "Model",
    "ModelRun",
    "Detection",
    # Preprocessing domain
    "PreprocessRun",
    # Differencing domain
    "DifferenceRun",
    "Candidate",
    # Curation domain
    "ValidationEvent",
    "Alert",
    # Catalog domain
    "SystemConfig",
    "ProcessingJob",
    "AuditLog",
    # Legacy models removed after migration
]
