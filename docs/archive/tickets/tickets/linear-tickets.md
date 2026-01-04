# AstrID Linear Tickets & Feature Tracking

## Project Overview

**Project**: AstrID - Astronomical Identification System  
**Team**: Chris Lawrence (Lead Developer)  
**Timeline**: September 2025 - April 2026  
**Status**: Core Development Phase  
**Progress**: 22/33 tickets completed (66.7%)  

## Recent Accomplishments

### September 16, 2025 - ASTR-73 Completed!
**Major Milestone**: Core observation domain fully implemented and tested
- **Domain Models**: Rich business logic for Observation and Survey entities
- **Domain Events**: Complete event-driven architecture foundation
- **Validation System**: Comprehensive data validation with custom exceptions
- **Repository Layer**: Enhanced repository with all required query methods
- **Service Layer**: Business logic with transaction management
- **API Integration**: New endpoints for status, search, metrics, and validation
- **Testing**: Comprehensive test coverage with integration notebook
- **Production Ready**: Full type annotations, documentation, and error handling

**Impact**: This provides the foundational domain model that all other services will build upon. The observation processing pipeline is now ready for integration with preprocessing, differencing, and ML detection services.

### September 16, 2025 - ASTR-76 Completed!
**Major Milestone**: Image Preprocessing Services fully implemented and tested
- Calibration: Bias/Dark/Flat master creation, application, validation, uncertainty
- Alignment: WCS alignment, multi-image registration, quality metrics
- Quality: Background/noise/cosmic rays/flatness/saturation, scoring
- Pipeline: Orchestration with hooks and metrics
- API: Preprocess/status/calibration-frame/quality/configure endpoints

**Impact**: Science-ready calibrated and aligned images enable the Differencing and Detection domains. This unblocks ASTR-78 (Image Differencing).

### September 17, 2025 - ASTR-88 Completed!
**Major Milestone**: MLflow Integration fully implemented and tested
- **MLflowServer**: Complete server management with health checks and monitoring
- **ExperimentTracker**: Comprehensive experiment and run management with parameter/metric logging
- **ModelRegistry**: Full model registry with versioning, stage transitions, and lineage tracking
- **ModelVersioning**: Semantic versioning system with deployment tracking and performance monitoring
- **API Integration**: 20+ REST API endpoints for all MLflow operations
- **R2 Storage**: Cloudflare R2 integration for MLflow artifact storage
- **Docker Integration**: MLflow server container in docker-compose.yaml
- **Production Ready**: Full error handling, logging, and type annotations

**Impact**: Complete ML infrastructure for experiment tracking, model management, and versioning. This enables ASTR-89 (Model Training Pipeline) and provides the foundation for all ML operations in AstrID.

### September 17, 2025 - ASTR-80 & ASTR-81 Completed!
**Major Milestone**: Complete ML Detection Pipeline fully implemented and tested
- **U-Net Integration**: Complete model loading, inference, and persistence with GPU acceleration
- **Detection Pipeline**: Comprehensive DetectionService with validation, storage, and metrics
- **Database Integration**: Full integration with Survey, Observation, Model, and ModelRun entities
- **MLflow Integration**: Complete experiment tracking and run management
- **API Endpoints**: 7 REST endpoints for all detection operations
- **Production Ready**: Full error handling, logging, and type annotations

**Impact**: Complete end-to-end detection pipeline from candidates → ML inference → detection persistence. This provides the core anomaly detection capability that enables the full AstrID workflow.

### September 17, 2025 - ASTR-91 Completed!
**Major Milestone**: Workflow Orchestration fully implemented and tested
- **Prefect Server**: Complete server management with health checks and monitoring
- **Observation Flows**: Comprehensive processing workflows for ingestion, preprocessing, differencing, detection, and validation
- **ML Training Flows**: Complete model training, hyperparameter optimization, evaluation, deployment, and retraining workflows
- **Monitoring System**: Real-time monitoring, performance tracking, and alerting capabilities
- **REST API**: 10+ endpoints for flow management, monitoring, and health checks
- **Docker Integration**: Prefect server and worker containers with retry logic
- **Production Ready**: Full error handling, logging, and type annotations

**Impact**: Complete workflow orchestration system enabling automated processing pipelines for observations and ML operations. This provides the foundation for scalable, reliable, and monitored data processing workflows.

## Linear Project Configuration

### Projects
- **ASTRID-INFRA**: Infrastructure & DevOps
- **ASTRID-CORE**: Core Domain Implementation  
- **ASTRID-API**: API & Web Interface
- **ASTRID-ML**: Machine Learning & Models
- **ASTRID-WORK**: Workflow & Orchestration
- **ASTRID-TEST**: Testing & Quality
- **ASTRID-DEPLOY**: Deployment & Operations
- **ASTRID-DOCS**: Documentation & Training

### Labels
- `infrastructure` - Core system setup
- `database` - Database related work
- `api` - API development
- `ml` - Machine learning features
- `ui` - User interface
- `testing` - Testing and QA
- `deployment` - Deployment and ops
- `documentation` - Docs and training
- `high-priority` - Critical path items
- `blocked` - Waiting on dependencies

### Priority Levels
- **P1** (Critical): Blocking other work, must be done first
- **P2** (High): Important for current phase
- **P3** (Medium): Important but not blocking
- **P4** (Low): Nice to have, can be deferred

## Epic: Foundation & Infrastructure Setup

### Core Infrastructure

#### ASTR-69: Development Environment Setup **COMPLETED**
- **Project**: ASTRID-INFRA
- **Priority**: P1 (Critical)
- **Labels**: `infrastructure`, `high-priority`
- **Estimated Time**: 2 days
- **Dependencies**: None
- **Description**: Set up complete development environment for AstrID project
- **Subtasks**:
  - [ ] Install Python 3.11 and uv package manager
  - [ ] Configure pre-commit hooks (Ruff, Black, MyPy)
  - [ ] Set up Docker development environment
  - [ ] Configure environment variables and secrets

#### ASTR-70: Database Setup and Migrations **COMPLETED**
- **Project**: ASTRID-INFRA
- **Priority**: P1 (Critical)
- **Labels**: `infrastructure`, `database`, `high-priority`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-69
- **Description**: Design and implement database schema with migration system
- **Subtasks**:
  - [ ] Design database schema for observations, detections, etc.
  - [ ] Implement SQLAlchemy 2 models
  - [ ] Create Alembic migration scripts
  - [ ] Set up test database configuration

#### ASTR-71: Cloud Storage Integration **COMPLETED**
- **Project**: ASTRID-INFRA
- **Priority**: P2 (High)
- **Labels**: `infrastructure`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-69
- **Description**: Configure cloud storage for datasets and artifacts
- **Subtasks**:
  - [ ] Configure Cloudflare R2 (S3-compatible) storage
  - [ ] Implement storage client with content addressing
  - [ ] Set up DVC for dataset versioning
  - [ ] Configure MLflow artifact storage

### Authentication & Security

#### ASTR-72: Supabase Integration **COMPLETED**
- **Project**: ASTRID-INFRA
- **Priority**: P2 (High)
- **Labels**: `infrastructure`, `security`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-69
- **Description**: Implement authentication and authorization system
- **Subtasks**:
  - [ ] Set up Supabase project
  - [ ] Implement JWT authentication
  - [ ] Create role-based access control
  - [ ] Add API key management

## Epic: Core Domain Implementation

### Observations Domain

#### ASTR-73: Observation Models and Services **COMPLETED**
- **Project**: ASTRID-CORE
- **Priority**: P1 (Critical) 
- **Labels**: `core-domain`, `high-priority`
- **Estimated Time**: 3 days **Actual: 2 days**
- **Dependencies**: ASTR-70 Complete
- **Description**: Implement core observation domain models and business logic
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 16, 2025
- **Subtasks**:
  - [x] Implement Observation domain models with business logic methods
    - [x] validate_coordinates() - Validates astronomical coordinates
    - [x] calculate_airmass() - Calculates observation airmass
    - [x] get_processing_status() - Returns detailed processing status
    - [x] get_sky_region_bounds() - Gets spatial bounds around observation
  - [x] Implement Survey domain models with business logic methods
    - [x] get_survey_stats() - Returns survey statistics
    - [x] is_configured_for_ingestion() - Checks ingestion readiness
    - [x] get_capabilities() - Returns survey capabilities
  - [x] Create observation repository interface with all required methods
    - [x] get_by_survey() - Get observations by survey ID
    - [x] get_by_status() - Get observations by processing status  
    - [x] update_status() - Update observation status
    - [x] get_by_coordinates() - Spatial coordinate search
    - [x] get_observations_for_processing() - Get ready observations
    - [x] count_by_survey() & count_by_status() - Count methods
  - [x] Implement observation service layer with enhanced business logic
    - [x] validate_observation_data() - Comprehensive data validation
    - [x] calculate_observation_metrics() - Calculate derived metrics
    - [x] process_observation_status_change() - Handle status transitions
    - [x] handle_observation_failure() - Process failures with events
    - [x] get_survey_observation_summary() - Generate survey summaries
    - [x] Transaction management and proper error handling
  - [x] Add comprehensive observation validation logic
    - [x] ObservationValidator with coordinate, exposure time, filter validation
    - [x] Custom validation exceptions (CoordinateValidationError, etc.)
    - [x] Metadata completeness validation
    - [x] Business rule enforcement
  - [x] Implement domain events for workflow orchestration
    - [x] ObservationIngested, ObservationStatusChanged, ObservationFailed
    - [x] ObservationProcessingStarted, ObservationProcessingCompleted
    - [x] ObservationValidationFailed, ObservationArchived
  - [x] Add enhanced API endpoints
    - [x] PUT /observations/{id}/status - Update observation status
    - [x] GET /observations/search - Coordinate-based search
    - [x] GET /observations/metrics/{id} - Get observation metrics
    - [x] GET /observations/survey/{id}/summary - Survey summaries
    - [x] POST /observations/validate - Data validation endpoint
  - [x] Create comprehensive test coverage
    - [x] Unit tests for all domain model methods
    - [x] Validation system tests with error cases
    - [x] Integration testing notebook (astr73_testing.ipynb)
    - [x] All 12 core features tested and validated

**Key Achievements**:
- **Complete Domain-Driven Design implementation** with rich business logic
- **Event-driven architecture** ready for workflow orchestration  
- **Comprehensive validation** with proper error handling
- **Enhanced repository pattern** with all required query methods
- **Transaction management** for data consistency
- **Full API integration** with new endpoints
- **Production-ready code** with proper type annotations and documentation

#### ASTR-74: Survey Integration **COMPLETED**
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `integration`
- **Estimated Time**: 4 days **Actual: 2 days**
- **Dependencies**: ASTR-73 Complete
- **Description**: Integrate with external astronomical survey APIs
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 16, 2025
- **Subtasks**:
  - [x] Integrate with MAST API for observations
  - [x] Integrate with SkyView for image data
  - [x] Implement survey-specific adapters
  - [x] Add observation metadata extraction

#### ASTR-75: FITS Processing Pipeline **COMPLETED**
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `data-processing`
- **Estimated Time**: 3 days **Actual: 2 days**
- **Dependencies**: ASTR-73 Complete
- **Description**: Implement FITS file processing and WCS handling
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 16, 2025
- **Subtasks**:
  - [x] Implement FITS file reading and writing
  - [x] Add WCS (World Coordinate System) handling
  - [x] Create image metadata extraction
  - [x] Implement star catalog integration

### Preprocessing Domain

#### ASTR-76: Image Preprocessing Services
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `image-processing`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-75
- **Description**: Implement image calibration and preprocessing pipeline
 - **Status**: FULLY IMPLEMENTED & TESTED (September 16, 2025)
- **Subtasks**:
  - [x] Implement bias/dark/flat calibration
  - [x] Add WCS alignment and registration
  - [x] Create image quality assessment
  - [x] Implement preprocessing pipeline orchestration
 
**Next Up**: Begin ASTR-78 (Image Differencing Algorithms)

#### ASTR-77: Astronomical Image Processing **COMPLETED**
- **Project**: ASTRID-CORE
- **Priority**: P3 (Medium)
- **Labels**: `core-domain`, `image-processing`
- **Estimated Time**: 3 days **Actual: 1 day**
- **Dependencies**: ASTR-76 Complete
- **Description**: Advanced image processing with OpenCV and scikit-image
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 16, 2025
- **Subtasks**:
  - [x] Integrate OpenCV for image manipulation
  - [x] Add scikit-image for advanced processing
  - [x] Implement image normalization and scaling
  - [x] Create preprocessing result storage

**Key Achievements**:
- **Complete OpenCV processor** with morphological operations, edge detection, filters, transforms, contrast enhancement, and noise removal
- **Full scikit-image processor** with segmentation, feature detection, morphology, measurements, restoration, and classification
- **Advanced image normalizer** with intensity normalization, scaling, histogram processing, z-score, and reference normalization
- **Preprocessing storage system** with compressed storage, metadata management, archival, and version control
- **Comprehensive API endpoints** for all image processing operations
- **Full test coverage** with performance benchmarks and integration testing
- **Production-ready** with proper error handling, logging, and documentation

### Differencing Domain

#### ASTR-78: Image Differencing Algorithms **COMPLETED**
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `algorithm`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-76 Complete
- **Description**: Implement image differencing algorithms for anomaly detection
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 16, 2025
- **Subtasks**:
  - [x] Implement ZOGY algorithm
  - [x] Add classic differencing methods
  - [x] Create reference image selection logic
  - [x] Implement difference image generation

#### ASTR-79: Source Extraction **COMPLETED**
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `algorithm`
- **Estimated Time**: 3 days **Actual: 2 days**
- **Dependencies**: ASTR-78 Complete
- **Description**: Extract and analyze sources from difference images
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Integrate SEP for source extraction
  - [x] Add photutils for additional analysis
  - [x] Implement candidate filtering
  - [x] Create candidate scoring system

### Detection Domain

#### ASTR-80: U-Net Model Integration **COMPLETED**
- **Project**: ASTRID-ML
- **Priority**: P2 (High)
- **Labels**: `ml`, `model-integration`
- **Estimated Time**: 3 days **Actual: 3 days**
- **Dependencies**: ASTR-79 Complete
- **Description**: Integrate existing U-Net model into new architecture
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Port existing U-Net model to new architecture
  - [x] Implement model loading and inference
  - [x] Add confidence scoring
  - [x] Create model performance tracking

**Key Achievements**:
- **Complete U-Net Integration** with model loading, inference, and persistence
- **Database Integration** with Survey, Observation, Model, and ModelRun creation
- **MLflow Integration** with run tracking and experiment management
- **GPU Support** with CUDA acceleration on RTX 3080
- **End-to-End Pipeline** from candidates → inference → detection persistence
- **Production Ready** with proper error handling, logging, and type annotations

#### ASTR-81: Anomaly Detection Pipeline **COMPLETED**
- **Project**: ASTRID-ML
- **Priority**: P2 (High)
- **Labels**: `ml`, `pipeline`
- **Estimated Time**: 3 days **Actual: 3 days**
- **Dependencies**: ASTR-80 Complete
- **Description**: Complete anomaly detection service implementation
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Implement detection service layer
  - [x] Add detection validation logic
  - [x] Create detection result storage
  - [x] Implement detection metrics calculation

**Key Achievements**:
- **Complete DetectionService** with comprehensive methods for anomaly processing
- **Advanced DetectionValidator** with quality assessment, duplicate checking, and false positive filtering
- **Comprehensive DetectionStorage** with indexing, analytics, and advanced querying capabilities
- **Full DetectionMetrics** with precision, recall, F1, AUC, latency, and throughput calculations
- **Complete API Endpoints** with 7 REST endpoints for all detection operations
- **Production Ready** with proper error handling, logging, and comprehensive data structures

### Curation Domain

#### ASTR-82: Human Validation System
- **Project**: ASTRID-CORE
- **Priority**: P3 (Medium)
- **Labels**: `core-domain`, `ui`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-81
- **Description**: Create human validation interface for detected anomalies
- **Subtasks**:
  - [ ] Create validation interface
  - [ ] Implement curator management
  - [ ] Add validation event tracking
  - [ ] Create feedback collection system

#### ASTR-111: Expert Review Process for Anomaly Confirmation
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `curation`, `expert-review`
- **Estimated Time**: 5 days
- **Dependencies**: ASTR-81, ASTR-82
- **Description**: Implement expert review workflow for anomaly validation and scientific importance rating
- **Subtasks**:
  - [ ] Set up expert notification system and alerts
  - [ ] Create expert review interface and dashboard
  - [ ] Implement anomaly rating and confirmation system
  - [ ] Add high-priority discovery tracking and management

### Catalog Domain

#### ASTR-83: Data Cataloging
- **Project**: ASTRID-CORE
- **Priority**: P3 (Medium)
- **Labels**: `core-domain`, `data`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-82
- **Description**: Implement data cataloging and export functionality
- **Subtasks**:
  - [ ] Implement catalog entry creation
  - [ ] Add analytics and reporting
  - [ ] Create data export functionality
  - [ ] Implement catalog search and filtering



## Epic: API & Web Interface

### FastAPI Implementation

#### ASTR-84: Core API Endpoints
- **Project**: ASTRID-API
- **Priority**: P1 (Critical)
- **Labels**: `api`, `high-priority`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-73, ASTR-81
- **Description**: Implement core API endpoints for all domains
- **Status**: **COMPLETED**
- **Subtasks**:
  - [x] Implement observations endpoints
  - [x] Add detections endpoints
  - [x] Create streaming endpoints (SSE)
  - [x] Add health check and monitoring endpoints



#### ASTR-85: API Documentation and Testing **COMPLETED**
- **Project**: ASTRID-API
- **Priority**: P2 (High)
- **Labels**: `api`, `testing`
- **Estimated Time**: 2 days **Actual: 1 day**
- **Dependencies**: ASTR-84 Complete
- **Description**: Comprehensive API documentation and testing
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Add comprehensive API documentation
  - [x] Implement API testing suite
  - [x] Add API versioning
  - [x] Create API rate limiting

#### ASTR-114: Real Data Ingestion Service & Worker Orchestration
- **Project**: ASTRID-API, ASTRID-WORK, ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `api`, `workflow`, `data-pipeline`, `ingestion`
- **Estimated Time**: 3-5 days
- **Dependencies**: ASTR-73, ASTR-76, ASTR-88, ASTR-91
- **Description**: Implement a thin ingestion service and background worker flow that fetches real survey imagery (SkyView/MAST), stores FITS to R2, and persists Observation records and manifests for training.
- **Status**: PLANNED
- **Subtasks**:
  - [ ] API: POST `/observations/ingest/position` (validate RA/Dec/FOV/survey, enqueue job)
  - [ ] Service: `ObservationIngestionService` composing `SkyViewClient`/`MASTClient`, `FITSProcessor`, `R2StorageClient`
  - [ ] Worker: `ingest_position_task` with retries + idempotency; store metadata and R2 keys
  - [ ] Manifest: write minimal JSON manifest entries for training datasets
  - [ ] Dataset Builder: extend `training/datasets/collect` to consume manifest or query by region/time
  - [ ] Docs: update notebooks to call API endpoints (not direct clients)

### Frontend Dashboard

#### ASTR-86: Next.js Dashboard Setup **COMPLETED**
- **Project**: ASTRID-API
- **Priority**: P3 (Medium)
- **Labels**: `ui`, `frontend`
- **Estimated Time**: 3 days **Actual: 2 days**
- **Dependencies**: ASTR-84 Complete
- **Description**: Set up Next.js dashboard with authentication
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Set up Next.js project with TypeScript
  - [x] Implement Tailwind CSS styling
  - [x] Create responsive layout components
  - [x] Add authentication integration

#### ASTR-87: Dashboard Features
- **Project**: ASTRID-API
- **Priority**: P3 (Medium)
- **Labels**: `ui`, `frontend`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-86
- **Description**: Implement core dashboard functionality
- **Subtasks**:
  - [ ] Create observation overview dashboard
  - [ ] Implement detection visualization
  - [ ] Add real-time streaming updates
  - [ ] Create user management interface

#### ASTR-108: Email Notification for Detected Anomalies
- **Project**: ASTRID-API
- **Priority**: P2 (High)
- **Labels**: `api`, `notifications`, `email`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-81
- **Description**: Implement SendGrid email notification system for anomaly detection alerts
- **Subtasks**:
  - [ ] Set up SendGrid integration and configuration
  - [ ] Define criteria for notable anomaly detection
  - [ ] Create email templates for anomaly alerts
  - [ ] Implement notification toggle and preferences

#### ASTR-109: Anomaly Timeline Feature for React Frontend
- **Project**: ASTRID-API
- **Priority**: P2 (High)
- **Labels**: `ui`, `frontend`, `timeline`, `visualization`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-81, ASTR-84
- **Description**: Add interactive timeline view to display detected anomalies over time
- **Subtasks**:
  - [ ] Create timeline component with real-time updates
  - [ ] Implement anomaly detail views and navigation
  - [ ] Add external resource links and sky map integration
  - [ ] Design responsive timeline layout for homepage

#### ASTR-110: Virtual Sky Observatory 3D Visualization (Stretch Goal)
- **Project**: ASTRID-API
- **Priority**: P4 (Low)
- **Labels**: `ui`, `frontend`, `3d`, `visualization`, `stretch-goal`
- **Estimated Time**: 8-12 weeks
- **Dependencies**: ASTR-81, ASTR-84, ASTR-109
- **Description**: Create immersive 3D virtual sky observatory for exploring astronomical data
- **Subtasks**:
  - [ ] Research and select 3D visualization technology (Three.js/WebGL)
  - [ ] Design 3D sky sphere with coordinate mapping system
  - [ ] Implement interactive 3D interface with camera controls
  - [ ] Integrate astronomical image projection and anomaly highlighting

#### ASTR-105: Testing & Diagnostics Page
- **Project**: ASTRID-API
- **Priority**: P3 (Medium)
- **Labels**: `ui`, `frontend`, `testing`, `ops`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-84, ASTR-93
- **Description**: Add a frontend page to run smoke checks and surface system health for troubleshooting. Hooks into API health, workers, queues, MLflow, storage, and database checks.
- **Subtasks**:
  - [ ] Create `/dashboard/testing` page with sections and results panel
  - [ ] Wire to API health endpoints (API, DB, storage, MLflow, Prefect)
  - [ ] Wire to workers/queues status endpoints and quick actions (read-only)
  - [ ] Add run-all tests button, per-test execution, and live logs
  - [ ] Add degraded state banner and error summaries
  - [ ] Persist last-run results (localStorage) and export JSON
  - [ ] Add route link from dashboard and Workers page

## Epic: Machine Learning & Model Management

### ML Infrastructure

#### ASTR-88: MLflow Integration **COMPLETED**
- **Project**: ASTRID-ML
- **Priority**: P2 (High)
- **Labels**: `ml`, `infrastructure`
- **Estimated Time**: 2 days **Actual: 2 days**
- **Dependencies**: ASTR-71 Complete
- **Description**: Set up MLflow for experiment tracking and model management
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Set up MLflow tracking server
  - [x] Implement experiment tracking
  - [x] Add model registry functionality
  - [x] Create model versioning system

**Key Achievements**:
- **Complete MLflow Infrastructure** with server management, health checks, and monitoring
- **Comprehensive Experiment Tracking** with run management, parameter/metric logging, and artifact management
- **Full Model Registry** with versioning, stage transitions, and model lineage tracking
- **Semantic Versioning System** with major.minor.patch versioning and deployment tracking
- **REST API Integration** with 20+ endpoints for all MLflow operations
- **R2 Storage Integration** for MLflow artifact storage with Cloudflare R2
- **Docker Integration** with MLflow server container in docker-compose.yaml
- **Production Ready** with proper error handling, logging, and type annotations

#### ASTR-89: Model Training Pipeline
- **Project**: ASTRID-ML
- **Priority**: P3 (Medium)
- **Labels**: `ml`, `pipeline`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-88
- **Description**: Automated model training and optimization workflows
- **Subtasks**:
  - [ ] Implement automated training workflows
  - [ ] Add hyperparameter optimization
  - [ ] Create model evaluation metrics
  - [ ] Implement model deployment automation

#### ASTR-106: Training Notebook for Model Training and MLflow Logging
- **Project**: ASTRID-ML
- **Priority**: P2 (High)
- **Labels**: `ml`, `notebook`, `training`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-88
- **Description**: Create Jupyter notebook for initial model training with MLflow integration
- **Subtasks**:
  - [ ] Set up data loading and preprocessing steps
  - [ ] Implement training loop with MLflow logging
  - [ ] Add metric tracking and validation
  - [ ] Create troubleshooting and debugging tools

#### ASTR-113: Real Data Loading Integration for Training Pipeline
- **Project**: ASTRID-ML
- **Priority**: P2 (High)
- **Labels**: `ml`, `data-pipeline`, `training`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-106, ASTR-88, ASTR-81, ASTR-76, ASTR-73
- **Description**: Implement real data loading functions that integrate with the complete AstrID workflow pipeline, enabling training on validated astronomical observations and human-labeled detections
- **Subtasks**:
  - [ ] Create TrainingDataCollector service for harvesting validated detections
  - [ ] Implement RealDataLoader class to replace synthetic data generation
  - [ ] Add TrainingDataset entity for dataset versioning and tracking
  - [ ] Integrate with human validation labels from ASTR-82
  - [ ] Add data pipeline orchestration with Dramatiq workers and Prefect flows
  - [ ] Enhance MLflow integration for real data metrics and lineage

### Model Operations

#### ASTR-90: Model Serving
- **Project**: ASTRID-ML
- **Priority**: P3 (Medium)
- **Labels**: `ml`, `mlops`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-89
- **Description**: Production model serving and monitoring
- **Subtasks**:
  - [ ] Implement model inference endpoints
  - [ ] Add model performance monitoring
  - [ ] Create A/B testing framework
  - [ ] Implement model rollback capabilities

## Epic: Workflow & Orchestration

### Prefect Integration

#### ASTR-91: Workflow Orchestration **COMPLETED**
- **Project**: ASTRID-WORK
- **Priority**: P2 (High)
- **Labels**: `workflow`, `orchestration`
- **Estimated Time**: 3 days **Actual: 3 days**
- **Dependencies**: ASTR-69 Complete
- **Description**: Set up Prefect for workflow orchestration
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Set up Prefect server
  - [x] Implement observation processing flows
  - [x] Add model training workflows
  - [x] Create monitoring and alerting

### Background Processing

#### ASTR-92: Dramatiq Workers **COMPLETED**
- **Project**: ASTRID-WORK
- **Priority**: P2 (High)
- **Labels**: `workflow`, `background`
- **Estimated Time**: 4 days **Actual: 3 days**
- **Dependencies**: ASTR-91 Complete
- **Description**: Implement background processing workers
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Implement observation ingestion workers
  - [x] Add preprocessing workers
  - [x] Create differencing workers
  - [x] Implement detection workers

#### ASTR-107: Task Scheduler for Automated Inference Runs
- **Project**: ASTRID-WORK
- **Priority**: P2 (High)
- **Labels**: `workflow`, `scheduler`, `automation`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-91
- **Description**: Set up task scheduler for automated inference pipeline execution
- **Subtasks**:
  - [ ] Review existing Docker and Prefect scheduling configuration
  - [ ] Configure cron jobs or task scheduler for inference runs
  - [ ] Set up log capture and monitoring for scheduled runs
  - [ ] Test and validate automated execution

#### ASTR-104: Workers Operations Page & Dashboard Integration
- **Project**: ASTRID-API
- **Priority**: P3 (Medium)
- **Labels**: `ui`, `frontend`, `workflow`, `ops`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-92, ASTR-84
- **Description**: Create a dedicated Workers page for operations and monitoring, and integrate a lightweight Workers card on the main dashboard that links to it.
- **Subtasks**:
  - [ ] Add Start/Stop/Scale quick actions per worker type (uses `/workers/{worker_type}/start|stop|scale`)
  - [ ] Add Clear Queue action per queue (uses `/workers/queues/{queue_name}/clear`)
  - [ ] Dedicated page at `/dashboard/workers` listing workers and queues with actions
  - [ ] Expand Workers card (dashboard) to show summary stats and link to page
  - [ ] Add mini trend charts for processed tasks and failure rate (1h/6h/24h)
  - [ ] Add polling controls (interval selector, pause/resume) and degraded state banner
  - [ ] Toast notifications and error handling for operations

## Epic: Testing & Quality Assurance

### Testing Infrastructure

#### ASTR-93: Test Framework Setup **COMPLETED**
- **Project**: ASTRID-TEST
- **Priority**: P1 (Critical)
- **Labels**: `testing`, `high-priority`
- **Estimated Time**: 2 days **Actual: 1 day**
- **Dependencies**: ASTR-69 Complete
- **Description**: Set up comprehensive testing infrastructure
- **Status**: **FULLY IMPLEMENTED & TESTED**
- **Completion Date**: September 17, 2025
- **Subtasks**:
  - [x] Configure pytest with async support
  - [x] Set up test database fixtures
  - [x] Implement mock services
  - [x] Add test coverage reporting

#### ASTR-94: Test Implementation
- **Project**: ASTRID-TEST
- **Priority**: P2 (High)
- **Labels**: `testing`
- **Estimated Time**: 5 days
- **Dependencies**: ASTR-93
- **Description**: Implement comprehensive test coverage
- **Subtasks**:
  - [ ] Write unit tests for all domains
  - [ ] Add integration tests for API
  - [ ] Implement end-to-end tests
  - [ ] Add performance and load tests

### Code Quality

#### ASTR-95: Code Quality Tools
- **Project**: ASTRID-TEST
- **Priority**: P2 (High)
- **Labels**: `testing`, `quality`
- **Estimated Time**: 1 day
- **Dependencies**: ASTR-69
- **Description**: Configure code quality and formatting tools
- **Status**: **COMPLETED**
- **Subtasks**:
  - [x] Configure Ruff for linting
  - [x] Set up MyPy for type checking
  - [x] Implement Black code formatting
  - [x] Add pre-commit hooks

## Epic: Deployment & Operations

### Containerization

#### ASTR-96: Docker Setup
- **Project**: ASTRID-DEPLOY
- **Priority**: P2 (High)
- **Labels**: `deployment`, `docker`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-84
- **Description**: Containerize all services for deployment
- **Status**: **COMPLETED**
- **Subtasks**:
  - [x] Create API Dockerfile
  - [x] Create worker Dockerfile
  - [x] Set up Docker Compose for development
  - [x] Implement health checks

### Production Deployment

#### ASTR-97: Production Setup
- **Project**: ASTRID-DEPLOY
- **Priority**: P3 (Medium)
- **Labels**: `deployment`, `production`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-96
- **Description**: Configure production environment and monitoring
- **Subtasks**:
  - [ ] Configure production environment
  - [ ] Set up monitoring and logging
  - [ ] Implement backup and recovery
  - [ ] Add performance monitoring

### CI/CD Pipeline

#### ASTR-98: GitHub Actions
- **Project**: ASTRID-DEPLOY
- **Priority**: P2 (High)
- **Labels**: `deployment`, `ci-cd`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-93
- **Description**: Set up automated CI/CD pipeline
- **Subtasks**:
  - [ ] Set up automated testing
  - [ ] Add code quality checks
  - [ ] Implement automated deployment
  - [ ] Add security scanning

## Epic: Documentation & Training

### Documentation

#### ASTR-99: Technical Documentation
- **Project**: ASTRID-DOCS
- **Priority**: P3 (Medium)
- **Labels**: `documentation`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-84
- **Description**: Create comprehensive technical documentation
- **Subtasks**:
  - [ ] Write API documentation
  - [ ] Create architecture documentation
  - [ ] Add deployment guides
  - [ ] Write user manuals

### Training & Knowledge Transfer

#### ASTR-100: User Training
- **Project**: ASTRID-DOCS
- **Priority**: P4 (Low)
- **Labels**: `documentation`, `training`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-99
- **Description**: Create user training materials and onboarding
- **Subtasks**:
  - [ ] Create user training materials
  - [ ] Implement onboarding process
  - [ ] Add help and support documentation
  - [ ] Create video tutorials

#### ASTR-101: GPU Energy Tracking for ML Workloads
- **Project**: ASTRID-WORK
- **Priority**: P3 (Medium)
- **Labels**: `mlops`, `monitoring`, `improvement`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-88
- **Description**: Implement GPU energy tracking for ML workloads
- **Status**: **COMPLETED**
- **Subtasks**:
  - [x] Implement GPU energy monitoring
  - [x] Add MLflow integration for energy metrics
  - [x] Create energy consumption reporting
  - [x] Add carbon footprint tracking
  - [ ] Ensure metrics are being recorded on model runs

#### ASTR-102: Refine Model Performance Tracking
- **Project**: ASTRID-ML
- **Priority**: P3 (Medium)
- **Labels**: `ml`, `monitoring`, `metrics`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-88, ASTR-81
- **Description**: Iterate through a full `ModelRun` session and ensure all desired performance metrics in database fields are populated per run; enrich MLflow logging and API responses.
- **Subtasks**:
  - [ ] Audit current `Model` and `ModelRun` metric fields
  - [ ] Add missing calculations (precision/recall/F1/AUC/latency/throughput)
  - [ ] Ensure metrics recorded during pipeline and persisted to DB
  - [ ] Align MLflow metrics/tags with DB schema
  - [ ] Update API responses to include full metrics set

#### ASTR-103: Supabase Connection Pooling Hardening
- **Project**: ASTRID-INFRA
- **Priority**: P2 (High)
- **Labels**: `infrastructure`, `database`, `stability`
- **Estimated Time**: 1 day
- **Dependencies**: ASTR-88, ASTR-91
- **Description**: Minimize per-service connection pools, align MLflow/Prefect DB URLs with `pool_size=1&max_overflow=0` and timeouts, and validate optional Transaction pooling compatibility.
- **Subtasks**:
  - [ ] Update MLflow backend-store URI with pool params
  - [ ] Update Prefect DB URL with pool params
  - [ ] Document Transaction pooling compatibility
  - [ ] Add verification steps and health checks

#### ASTR-112: Data Backup and Disaster Recovery System
- **Project**: ASTRID-INFRA
- **Priority**: P4 (Low)
- **Labels**: `infrastructure`, `backup`, `disaster-recovery`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-70, ASTR-71, ASTR-88
- **Description**: Implement comprehensive backup and disaster recovery system for data protection and business continuity
- **Subtasks**:
  - [ ] Set up automated database backup system
  - [ ] Implement cloud storage redundancy and versioning
  - [ ] Create ML model and artifact backup procedures
  - [ ] Add disaster recovery testing and documentation


---


## Priority Matrix

### High Priority (Phase 1: Sept-Oct 2025)
- **ASTR-69**: Development environment setup (P1) **COMPLETED**
- **ASTR-70**: Database setup (P1) **COMPLETED**
- **ASTR-73**: Observation models (P1) **COMPLETED**
- **ASTR-84**: Core API endpoints (P1) **COMPLETED**
- **ASTR-88**: MLflow Integration (P2) **COMPLETED**
- **ASTR-80**: U-Net Model Integration (P2) **COMPLETED**
- **ASTR-81**: Anomaly Detection Pipeline (P2) **COMPLETED**
- **ASTR-93**: Test framework (P1)

### Medium Priority (Phase 2: Nov-Dec 2025)
- **ASTR-76**: Image preprocessing (P2) **COMPLETED**
- **ASTR-78**: Image differencing (P2) **COMPLETED**
- **ASTR-80**: U-Net integration (P2) **COMPLETED**
- **ASTR-81**: Anomaly Detection Pipeline (P2) **COMPLETED**
- **ASTR-88**: MLflow setup (P2) **COMPLETED**
- **ASTR-91**: Workflow orchestration (P2)

### Lower Priority (Phase 3: Jan-Apr 2026)
- **ASTR-86**: Frontend dashboard (P3)
- **ASTR-82**: Human validation (P3)
- **ASTR-83**: Data cataloging (P3)
- **ASTR-97**: Production deployment (P3)
- **ASTR-99**: Documentation (P3)
- **ASTR-101**: GPU Energy Tracking (P3) **COMPLETED**

## Success Metrics

### Technical Metrics
- **Code Coverage**: >90% test coverage
- **API Response Time**: <200ms for 95% of requests
- **Model Accuracy**: >85% precision for anomaly detection
- **System Uptime**: >99.5% availability

### Business Metrics
- **Processing Throughput**: >1000 observations/day
- **Detection Rate**: >50% of actual anomalies detected
- **False Positive Rate**: <10% of detections
- **User Adoption**: >5 active users within 3 months

## Risk Assessment

### High Risk
- **ML Model Performance**: U-Net may not generalize well to new data
- **Data Quality**: External survey data may be inconsistent
- **Scalability**: System may not handle high observation volumes

### Medium Risk
- **Integration Complexity**: Multiple external APIs may cause reliability issues
- **Performance**: Real-time processing may be resource-intensive
- **Security**: Astronomical data may have privacy implications

### Low Risk
- **Technology Stack**: Well-established technologies with good community support
- **Development Process**: Clear methodology and experienced developer
- **Documentation**: Comprehensive planning and documentation

## Next Steps

### Immediate Actions (This Week)
1. Set up Linear workspace and projects
2. Create labels and configure team settings
3. Use automation scripts to create initial tickets
4. Set up development environment

### Week 1-2
1. Implement core domain models
2. Set up database and migrations
3. Create basic API structure
4. Begin U-Net model integration

### Month 1 Goals
1. Complete foundation infrastructure
2. Implement observation domain
3. Set up basic API endpoints
4. Begin preprocessing pipeline

### Current Focus (September 2025)
1. Complete MLflow integration (ASTR-88)
2. Complete U-Net model integration (ASTR-80)
3. Complete anomaly detection pipeline (ASTR-81)
4. Complete source extraction (ASTR-79)
5. Begin workflow orchestration (ASTR-91)
6. Implement test framework (ASTR-93)