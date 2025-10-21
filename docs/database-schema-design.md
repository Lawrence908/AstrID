# AstrID Database Schema Design

## Overview

This document outlines the comprehensive database schema for the AstrID astronomical identification system. The schema follows Domain-Driven Design principles and supports the complete data pipeline from observation ingestion to detection validation.

**Implementation Status**: This schema is implemented using SQLAlchemy models across multiple domain modules. The actual implementation uses Python enums, proper relationships, and follows modern SQLAlchemy patterns.

## Core Entities and Relationships

### 1. Surveys and Observations

**Domain**: `src/domains/observations/models.py`

```python
# Survey information
class Survey(Base):
    __tablename__ = "surveys"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    base_url: Mapped[Optional[str]] = mapped_column(String(500))
    api_endpoint: Mapped[Optional[str]] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Astronomical observations
class Observation(Base):
    __tablename__ = "observations"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    survey_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("surveys.id"), nullable=False)
    observation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Astronomical coordinates
    ra: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    dec: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    observation_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Observation properties
    filter_band: Mapped[str] = mapped_column(String(50), nullable=False)
    exposure_time: Mapped[float] = mapped_column(Numeric(10, 3), nullable=False)
    fits_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    
    # Image metadata
    pixel_scale: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    image_width: Mapped[Optional[int]] = mapped_column(Integer)
    image_height: Mapped[Optional[int]] = mapped_column(Integer)
    airmass: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    seeing: Mapped[Optional[float]] = mapped_column(Numeric(6, 3))
    
    # Processing status
    status: Mapped[ObservationStatus] = mapped_column(String(50), default=ObservationStatus.INGESTED)
    
    # Storage references
    fits_file_path: Mapped[Optional[str]] = mapped_column(String(500))
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Status enum
class ObservationStatus(str, Enum):
    INGESTED = "ingested"
    PREPROCESSING = "preprocessing"
    PREPROCESSED = "preprocessed"
    DIFFERENCING = "differencing"
    DIFFERENCED = "differenced"
    FAILED = "failed"
    ARCHIVED = "archived"
```

**Constraints and Indexes**:
- RA/Dec coordinate validation (0-360, -90 to 90)
- Exposure time > 0, airmass > 0, seeing > 0
- Spatial indexes on RA/Dec for efficient cone searches
- Unique constraint on (survey_id, observation_id)

### 2. Processing Pipeline

**Domain**: `src/domains/preprocessing/models.py`

```python
# Preprocessing runs
class PreprocessRun(Base):
    __tablename__ = "preprocess_runs"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    observation_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("observations.id"), nullable=False)
    
    status: Mapped[PreprocessRunStatus] = mapped_column(String(50), default=PreprocessRunStatus.PENDING)
    
    # Processing details
    calibration_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    wcs_aligned: Mapped[bool] = mapped_column(Boolean, default=False)
    registration_quality: Mapped[Optional[float]] = mapped_column(Numeric(5, 3))
    
    # Output files
    processed_fits_path: Mapped[Optional[str]] = mapped_column(String(500))
    wcs_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    processing_time_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Status enum
class PreprocessRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

**Domain**: `src/domains/differencing/models.py`

```python
# Differencing runs
class DifferenceRun(Base):
    __tablename__ = "difference_runs"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    observation_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("observations.id"), nullable=False)
    reference_observation_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("observations.id"))
    
    status: Mapped[DifferenceRunStatus] = mapped_column(
        SQLAlchemyEnum(DifferenceRunStatus, name="difference_run_status"),
        nullable=False,
        default=DifferenceRunStatus.PENDING
    )
    
    # Algorithm details
    algorithm: Mapped[str] = mapped_column(String(50), default="zogy")
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Output files
    difference_image_path: Mapped[Optional[str]] = mapped_column(String(500))
    significance_map_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Quality metrics
    noise_level: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    detection_threshold: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    candidates_found: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    processing_time_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Candidates (potential transients from differencing)
class Candidate(Base):
    __tablename__ = "candidates"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    difference_run_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("difference_runs.id"), nullable=False)
    observation_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("observations.id"), nullable=False)
    
    # Position
    ra: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    dec: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    pixel_x: Mapped[int] = mapped_column(Integer, nullable=False)
    pixel_y: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Detection properties
    flux: Mapped[Optional[float]] = mapped_column(Numeric(15, 6))
    flux_error: Mapped[Optional[float]] = mapped_column(Numeric(15, 6))
    significance: Mapped[Optional[float]] = mapped_column(Numeric(10, 3))
    snr: Mapped[Optional[float]] = mapped_column(Numeric(8, 3))
    
    # Shape parameters
    fwhm: Mapped[Optional[float]] = mapped_column(Numeric(8, 3))
    ellipticity: Mapped[Optional[float]] = mapped_column(Numeric(5, 3))
    position_angle: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    
    # Quality flags
    is_saturated: Mapped[bool] = mapped_column(Boolean, default=False)
    is_blended: Mapped[bool] = mapped_column(Boolean, default=False)
    is_edge: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Status
    status: Mapped[CandidateStatus] = mapped_column(
        SQLAlchemyEnum(CandidateStatus, name="candidate_status"),
        nullable=False,
        default=CandidateStatus.PENDING
    )
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Status enums
class DifferenceRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class CandidateStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DETECTED = "detected"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
```

**Indexes and Relationships**:
- Spatial indexes on RA/Dec for candidates
- Indexes on significance, status, and difference_run_id
- Proper SQLAlchemy relationships between observations, difference runs, and candidates
```

### 3. Machine Learning and Detection

**Domain**: `src/domains/detection/models.py`

```python
# ML Models registry
class Model(Base):
    __tablename__ = "models"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[ModelType] = mapped_column(
        SQLAlchemyEnum(ModelType, name="model_type"),
        nullable=False
    )
    
    # Model metadata
    architecture: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    hyperparameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    training_dataset: Mapped[Optional[str]] = mapped_column(String(200))
    training_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Performance metrics
    precision: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    recall: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    f1_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    accuracy: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    
    # Deployment
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    deployment_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Storage
    model_path: Mapped[Optional[str]] = mapped_column(String(500))
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Model inference runs
class ModelRun(Base):
    __tablename__ = "model_runs"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    observation_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("observations.id"))
    
    # Inference details
    input_image_path: Mapped[Optional[str]] = mapped_column(String(500))
    output_mask_path: Mapped[Optional[str]] = mapped_column(String(500))
    confidence_map_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Performance
    inference_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    memory_usage_mb: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Results
    total_predictions: Mapped[Optional[int]] = mapped_column(Integer)
    high_confidence_predictions: Mapped[Optional[int]] = mapped_column(Integer)
    
    status: Mapped[ModelRunStatus] = mapped_column(
        SQLAlchemyEnum(ModelRunStatus, name="model_run_status"),
        nullable=False,
        default=ModelRunStatus.PENDING
    )
    
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Detections (ML-identified anomalies)
class Detection(Base):
    __tablename__ = "detections"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_run_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_runs.id"), nullable=False)
    observation_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("observations.id"), nullable=False)
    
    # Position
    ra: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    dec: Mapped[float] = mapped_column(Numeric(10, 7), nullable=False)
    pixel_x: Mapped[int] = mapped_column(Integer, nullable=False)
    pixel_y: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Detection properties
    confidence_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    detection_type: Mapped[DetectionType] = mapped_column(
        SQLAlchemyEnum(DetectionType, name="detection_type"),
        nullable=False
    )
    
    # ML metadata
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    inference_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    prediction_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Status and workflow
    status: Mapped[DetectionStatus] = mapped_column(
        SQLAlchemyEnum(DetectionStatus, name="detection_status"),
        nullable=False,
        default=DetectionStatus.PENDING
    )
    
    # Validation
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_confidence: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    human_label: Mapped[Optional[str]] = mapped_column(String(50))
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Enums
class ModelType(str, Enum):
    UNET = "unet"
    ANOMALY_DETECTOR = "anomaly_detector"
    CLASSIFIER = "classifier"
    SEGMENTATION = "segmentation"

class ModelRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class DetectionStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"
    ARCHIVED = "archived"

class DetectionType(str, Enum):
    SUPERNOVA = "supernova"
    VARIABLE = "variable"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"
    ARTIFACT = "artifact"
```

**Key Differences from Original Schema**:
- Removed `candidate_id` foreign key from `Detection` model (not implemented)
- Added proper SQLAlchemy relationships and constraints
- Uses Python enums with SQLAlchemy Enum types
- Confidence score validation (0-1 range) implemented as CheckConstraint
```

### 4. Validation and Curation

**Domain**: `src/domains/curation/models.py`

```python
# Validation events (human review)
class ValidationEvent(Base):
    __tablename__ = "validation_events"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    detection_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("detections.id"), nullable=False)
    validator_id: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Validation results
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(50))
    confidence_level: Mapped[Optional[ConfidenceLevel]] = mapped_column(
        SQLAlchemyEnum(ConfidenceLevel, name="confidence_level"),
        nullable=True
    )
    
    # Additional information
    notes: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSONB)
    
    # Quality assessment
    image_quality: Mapped[Optional[QualityLevel]] = mapped_column(
        SQLAlchemyEnum(QualityLevel, name="image_quality"),
        nullable=True
    )
    detection_quality: Mapped[Optional[QualityLevel]] = mapped_column(
        SQLAlchemyEnum(QualityLevel, name="detection_quality"),
        nullable=True
    )
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Alerts and notifications
class Alert(Base):
    __tablename__ = "alerts"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    detection_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("detections.id"), nullable=False)
    
    # Alert configuration
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    priority: Mapped[AlertPriority] = mapped_column(
        SQLAlchemyEnum(AlertPriority, name="alert_priority"),
        nullable=False,
        default=AlertPriority.MEDIUM
    )
    
    # Content
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    alert_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Delivery
    status: Mapped[AlertStatus] = mapped_column(
        SQLAlchemyEnum(AlertStatus, name="alert_status"),
        nullable=False,
        default=AlertStatus.PENDING
    )
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    delivery_attempts: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Enums
class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERT = "expert"

class QualityLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class AlertPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

**Key Differences from Original Schema**:
- Field name change: `metadata` → `alert_metadata` in Alert model
- Uses Python enums with SQLAlchemy Enum types instead of CHECK constraints
- Proper SQLAlchemy relationships with Detection model
```

### 5. System and Audit Tables

**Domain**: `src/domains/catalog/models.py`

```python
# Processing jobs (for workflow tracking)
class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Job details
    status: Mapped[ProcessingJobStatus] = mapped_column(
        SQLAlchemyEnum(ProcessingJobStatus, name="processing_job_status"),
        nullable=False,
        default=ProcessingJobStatus.PENDING
    )
    priority: Mapped[int] = mapped_column(Integer, default=0)
    
    # Workflow tracking
    workflow_id: Mapped[Optional[str]] = mapped_column(String(100))
    task_id: Mapped[Optional[str]] = mapped_column(String(100))
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    
    # Timing
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# System configuration
class SystemConfig(Base):
    __tablename__ = "system_config"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    value: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Audit log for important operations
class AuditLog(Base):
    __tablename__ = "audit_log"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Change details
    old_values: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    new_values: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    change_summary: Mapped[Optional[str]] = mapped_column(Text)
    
    # Context
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv6 max length
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

# Enums
class ProcessingJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

**Key Differences from Original Schema**:
- IP address field changed from `INET` to `String(45)` for IPv6 compatibility
- Uses Python enums with SQLAlchemy Enum types
- Proper SQLAlchemy type annotations and relationships
```

## Key Design Decisions

### 1. UUID Primary Keys
- All tables use UUID primary keys for better distributed system support
- Enables easier data sharding and replication
- Prevents enumeration attacks
- Implemented using `UUID(as_uuid=True)` in SQLAlchemy

### 2. JSONB for Flexible Metadata
- Used for model parameters, WCS info, and other variable data
- Enables schema evolution without migrations
- PostgreSQL JSONB provides excellent query performance
- Implemented using `JSONB` type in SQLAlchemy

### 3. Python Enums with SQLAlchemy Integration
- Consistent status tracking across all processing stages using Python enums
- Enables workflow state management with type safety
- Clear audit trail of processing progress
- Implemented using `SQLAlchemyEnum` for database-level validation

### 4. Spatial Indexing
- RA/Dec coordinates indexed for efficient spatial queries
- Supports cone searches and region-based filtering
- Essential for astronomical data access patterns
- Implemented using `Index` objects in SQLAlchemy `__table_args__`

### 5. Domain-Driven Design Structure
- Models organized by domain (observations, preprocessing, differencing, detection, curation, catalog)
- Clear separation of concerns and bounded contexts
- Proper SQLAlchemy relationships between domains

### 6. Type Safety and Modern SQLAlchemy
- Uses SQLAlchemy 2.0+ `Mapped` type annotations
- Proper nullable/optional field handling
- Automatic timestamp management with `server_default=func.now()`
- Check constraints for data validation (e.g., confidence scores, coordinates)

### 7. Audit and Tracking
- Comprehensive audit logging for data lineage
- Processing job tracking for workflow management
- Error handling and retry mechanisms
- IP address support for both IPv4 and IPv6

### 8. ML Integration
- Model registry for versioning and deployment
- Inference tracking for performance monitoring
- Confidence scoring and validation workflows
- Proper relationship between models, runs, and detections

## Implementation Status

###  Completed Models
All models from the original schema design have been implemented:

1. ** Core entities**: `Survey`, `Observation` (observations domain)
2. ** Processing pipeline**: `PreprocessRun` (preprocessing domain), `DifferenceRun`, `Candidate` (differencing domain)
3. ** ML integration**: `Model`, `ModelRun`, `Detection` (detection domain)
4. ** Validation and curation**: `ValidationEvent`, `Alert` (curation domain)
5. ** System tables**: `SystemConfig`, `ProcessingJob`, `AuditLog` (catalog domain)

###  Migration Strategy
The models are ready for database migration using Alembic:

1. **Phase 1**: Core entities (surveys, observations) -  Ready
2. **Phase 2**: Processing pipeline (preprocess_runs, difference_runs, candidates) -  Ready
3. **Phase 3**: ML integration (models, model_runs, detections) -  Ready
4. **Phase 4**: Validation and curation (validation_events, alerts) -  Ready
5. **Phase 5**: System tables (processing_jobs, audit_log, system_config) -  Ready

### 📋 Next Steps
1. Generate Alembic migrations for all models
2. Set up database connection and test migrations
3. Implement repository patterns for each domain
4. Add comprehensive tests for all models

## Performance Considerations

- **Partitioning**: Consider partitioning large tables by date (e.g., observations by observation_time)
- **Indexing**: Strategic indexes implemented via SQLAlchemy `Index` objects:
  - Spatial indexes on RA/Dec coordinates for cone searches
  - Status indexes for workflow queries
  - Foreign key indexes for relationship queries
- **Archiving**: Implement data archiving for old observations using status field
- **Caching**: Redis caching for frequently accessed metadata (surveys, models)
- **Connection Pooling**: Configure SQLAlchemy connection pool for optimal performance
- **Query Optimization**: Use SQLAlchemy's relationship loading strategies (lazy, eager, select)

## Security Considerations

- **Row Level Security**: Implement RLS for multi-tenant scenarios using PostgreSQL policies
- **Data Encryption**: Encrypt sensitive metadata fields in JSONB columns
- **Access Control**: Role-based access to different data types using application-level authorization
- **Audit Trail**: Comprehensive logging via `AuditLog` model for all data modifications
- **Input Validation**: SQLAlchemy type validation and check constraints prevent invalid data
- **SQL Injection Prevention**: SQLAlchemy ORM provides built-in protection against SQL injection

## SQLAlchemy-Specific Implementation Notes

### Model Relationships
- Proper bidirectional relationships using `back_populates`
- Lazy loading by default with options for eager loading when needed
- Foreign key constraints enforced at database level

### Type Safety
- `Mapped` type annotations provide IDE support and runtime type checking
- Optional fields properly handled with `Optional[Type]` annotations
- Enum types provide compile-time and runtime validation

### Migration Management
- Alembic integration for schema versioning
- Automatic migration generation from model changes
- Rollback capabilities for safe schema evolution
