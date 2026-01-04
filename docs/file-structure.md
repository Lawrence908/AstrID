AstrID File Structure

This document provides a high-level, hierarchical view of the repository with brief descriptions for key directories and files. Use it as context during development and future prompts.

Root
- `alembic/`: Database migrations managed by Alembic.
- `alembic.ini`: Alembic configuration file.
- `astrid-aliases.sh`: Shell aliases to streamline local development.
- `astrid.egg-info/`: Package metadata generated during builds/installs.
- `certs/`: Certificates and related assets for local/dev services.
- `CONFIGURATION_FIXES.md`: Notes on configuration issues and fixes.
- `docker-compose-reference.md`: Reference compose configuration notes.
- `docker-compose.override.yml`: Local overrides for Docker Compose.
- `docker-compose.yaml`: Top-level Docker Compose services definition.
- `docker-setup.md`: Guide for setting up Docker environment.
- `Dockerfile.api`: Dockerfile for the API service.
- `Dockerfile.api.optimized`: Size/performance optimized API image.
- `Dockerfile.base`: Base image with shared dependencies.
- `Dockerfile.prefect`: Dockerfile for workflow/orchestration services.
- `Dockerfile.prefect.optimized`: Optimized Prefect image.
- `Dockerfile.worker`: Dockerfile for background workers.
- `docs/`: Project documentation (architecture, guides, design, API, etc.).
  - `api/`: API-specific docs and references.
  - `architecture.md`: System architecture overview.
  - `database-schema-design.md`: Database schema design notes.
  - `design-overview.md`: High-level design decisions and rationale.
  - `development.md`: Developer setup and workflows.
  - `diagrams/`: Architecture and workflow diagrams.
  - `file-structure.md`: This file; repository layout and descriptions.
  - `guides/`: How-to guides and operational runbooks.
  - `logging-guide.md`: Logging patterns and guidance.
  - `research/`: Research papers and pipeline guides.
  - `tech-stack.md`: Technology stack summary and versions.
  - `test-framework-summary.md`: Testing strategy and tooling.
  - `archive/`: Archived documentation (historical reference).
- `examples/`: Example scripts and usage snippets.
- `frontend/`: Web UI application (separate frontend codebase).
- `htmlcov/`: Test coverage HTML reports.
- `jwst.png`: Project image/logo.
- `LICENSE`: MIT license file.
- `logs/`: Log outputs from local/dev runs.
- `notebooks/`: Jupyter notebooks and experimentation assets.
  - `supernova_training_pipeline.ipynb`: Main training pipeline notebook.
  - `fits_viewer.ipynb`: FITS file viewing utility.
  - `training/`: Training workflows and experimentation notebooks.
  - `archive/`: Archived notebooks (historical reference).
- `pyproject.toml`: Python project configuration, dependencies, and tooling.
- `README.md`: Project overview, features, and quickstart.
- `scripts/`: Operational and developer scripts.
  - `README.md`: Comprehensive scripts documentation.
  - `download_sn_fits.py`: FITS downloader from MAST.
  - `query_sn_fits_from_catalog.py`: MAST query script.
  - `query_sn_fits_chunked.py`: Chunked queries for large datasets.
  - `generate_difference_images.py`: Difference image generation.
  - `organize_training_pairs.py`: Training data organization.
  - `compile_supernova_catalog.py`: Catalog compilation.
  - `audit_sn_downloads.py`: Download audit utility.
  - `archive/`: Archived scripts (historical reference).
- `src/`: Application source code (DDD-aligned: domains, adapters, core, infrastructure).
  - `__init__.py`: Marks `src` as a package.
  - `adapters/`: Framework and external integration layer.
    - `__init__.py`: Package marker.
    - `api/`: FastAPI application and API plumbing.
      - `main.py`: FastAPI app bootstrap and server initialization.
      - `docs.py`: OpenAPI/Docs configuration and helpers.
      - `rate_limiting.py`: API rate limiting utilities/middleware.
      - `routes/`: API route modules grouped by domain.
      - `versioning.py`: API version handling and helpers.
    - `auth/`: Authentication and authorization adapters.
      - `api_key_auth.py`: API key auth backend and validation.
      - `api_key_service.py`: API key creation/management services.
      - `models.py`: Auth-related data models.
      - `rbac.py`: Role-based access control policies/utilities.
      - `supabase_auth.py`: Supabase integration for auth.
    - `external/`: 3rd-party data sources integrations.
      - `mast.py`: MAST archive client helpers.
      - `r2.py`: Cloudflare R2 (S3) client and utilities.
      - `skyview.py`: SkyView data retrieval client.
      - `vizier.py`: VizieR catalog integration.
    - `imaging/`: Astronomical imaging utilities.
      - `fits_io.py`: FITS file I/O and metadata handling.
      - `preprocess.py`: Image preprocessing routines.
      - `utils.py`: Shared imaging helpers/utilities.
    - `ml/`: ML model adapters and interfaces.
      - `unet.py`: U-Net model definition/wrappers.
    - `scheduler/`: Flow scheduling and deployment.
      - `config.py`: Scheduler configuration.
      - `deploy.py`: Flow deployment utilities.
      - `flows/`: Scheduling-related flows.
    - `workers/`: Background worker processes and tasks.
      - `config.py`: Worker configuration options.
      - `monitoring.py`: Worker health/metrics monitoring.
      - `start_workers.py`: Entry point to start worker processes.
      - `tasks.py`: Shared task definitions for workers.
      - `training_data.py`: Worker tasks for training data generation.
      - `curation/`, `detection/`, `differencing/`, `ingestion/`, `preprocessing/`: Specialized worker task modules.
  - `core/`: Cross-cutting concerns and shared core utilities.
    - `api/`: API-level shared helpers.
      - `response_wrapper.py`: Standard API response formatting.
    - `constants.py`: Global constants and configuration values.
    - `db/`: Database abstractions and session management.
      - `base_crud.py`: Base CRUD operations using SQLAlchemy.
      - `exceptions.py`: Database error types/handling.
      - `models.py`: ORM models shared across domains.
      - `session.py`: Session/engine initialization and utilities.
    - `dependencies.py`: Dependency injection wiring for API/services.
    - `energy_analysis.py`: Energy usage analysis utilities.
    - `exceptions.py`: Common exception types.
    - `gpu_monitoring.py`: GPU utilization monitoring helpers.
    - `logging.py`: Logging configuration setup.
    - `logging_examples.py`: Examples of structured logging usage.
    - `logging_utils.py`: Logging utilities and helpers.
    - `mlflow_energy.py`: MLflow integration for energy tracking.
  - `domains/`: Business logic by bounded context (DDD).
    - `__init__.py`: Package marker.
    - `catalog/`: Data cataloging domain.
      - `api/`, `crud.py`, `models.py`, `repository.py`, `schema.py`, `service.py`: Catalog endpoints, persistence, schemas, and services.
    - `curation/`: Human validation/curation domain.
      - `api/`, `crud.py`, `models.py`, `repository.py`, `schema.py`, `service.py`: Curation endpoints, persistence, schemas, and services.
    - `differencing/`: Image differencing domain.
      - `api/`, `crud.py`, `models.py`, `repository.py`, `schema.py`, `service.py`: Differencing endpoints, persistence, schemas, and services.
    - `detection/`: Anomaly detection domain.
      - `config.py`, `entities.py`, `models.py`: Core detection types/configs.
      - `analyzers/`, `extractors/`, `filters/`, `metrics/`, `ml_entities/`, `processors/`, `scorers/`, `services/`, `storage/`, `validators/`: Detection pipeline components and utilities.
      - `api/`, `crud.py`, `repository.py`, `schema.py`, `service.py`: API and persistence for detections.
    - `ml/`: ML workflows and training data domain.
      - `flows/`: Orchestrated ML flows.
      - `training_data/`: Training data generation/transforms.
    - `observations/`: Observations ingestion and management domain.
      - `adapters/`, `api/`, `catalogs/`, `extractors/`, `flows/`, `ingestion/`, `integrations/`, `processors/`: Observation-related integrations and pipelines.
      - `config.py`, `crud.py`, `events.py`, `exceptions.py`, `models.py`, `repository.py`, `schema.py`, `service.py`: Core observation logic and API.
      - `survey_config.py`, `survey_config_service.py`: Survey configuration models/services.
      - `testing/`: Test utilities for observation flows.
      - `validators.py`: Validation utilities.
    - `preprocessing/`: Image preprocessing domain.
      - `alignment/`, `calibration/`, `normalizers/`, `pipeline/`, `processors/`, `quality/`, `storage/`: Preprocessing components and stages.
      - `api/`, `crud.py`, `models.py`, `repository.py`, `schema.py`, `service.py`: API and persistence for preprocessing.
  - `infrastructure/`: Infrastructure integrations and platform services.
    - `__init__.py`: Package marker.
    - `mlflow/`: MLflow server and model registry tooling.
      - `config.py`: MLflow configuration.
      - `experiment_tracker.py`: Experiment tracking utilities.
      - `mlflow_server.py`: MLflow server management.
      - `model_registry.py`: Model registry helpers.
      - `model_versioning.py`: Versioning utilities for models.
    - `storage/`: Storage clients and abstractions.
      - `config.py`: Storage configuration.
      - `content_addressed_storage.py`: CAS layer for deduplication.
      - `dvc_client.py`: DVC client helpers.
      - `mlflow_storage.py`: MLflow artifact storage integration.
      - `r2_client.py`: Cloudflare R2 storage client.
      - `README.md`: Storage module notes.
    - `workflow/`: Workflow orchestration infrastructure.
      - `api/`: Workflow-related API surface.
      - `config.py`: Prefect/workflow configuration.
      - `monitoring.py`: Workflow/flow monitoring utilities.
      - `prefect_server.py`: Local Prefect server management.
- `start-dev-with-frontend.sh`: Start backend and frontend for development.
- `start-dev.sh`: Start backend-only dev environment.
- `start-frontend.sh`: Start frontend-only dev environment.
- `tests/`: Test suite (unit, integration, e2e) and fixtures.
  - `api/`: API tests.
  - `conftest.py`: Pytest configuration/fixtures.
  - `data/`: Test data assets.
  - `domains/`: Domain-specific tests.
  - `e2e/`: End-to-end tests.
  - `fixtures/`: Shared test fixtures.
  - `infrastructure/`: Infrastructure integration tests.
  - `integration/`: Integration tests across modules.
  - `mocks/`: Test doubles and mock objects.
  - `scripts/`: Helper scripts for tests.
  - `test_astr92_workers.py`, `test_framework_validation.py`, `test_workflow_orchestration.py`: Notable test entry points.
  - `unit/`: Unit tests by module.
  - `utils.py`: Test utilities.
- `uv.lock`: Dependency lock file for the UV package manager.
- `zOLD/`: Archived/legacy content kept for reference.

ASCII hierarchy view

```text
AstrID/
├── alembic/                               # Database migrations (Alembic)
├── alembic.ini                            # Alembic configuration
├── astrid-aliases.sh                      # Dev convenience aliases
├── astrid.egg-info/                       # Build/install metadata
├── certs/                                 # Local/dev certificates
├── CONFIGURATION_FIXES.md                 # Configuration fixes and notes
├── docker-compose-reference.md            # Reference compose notes
├── docker-compose.override.yml            # Local compose overrides
├── docker-compose.yaml                    # Compose services definition
├── docker-setup.md                        # Docker setup guide
├── Dockerfile.api                         # API service image
├── Dockerfile.api.optimized               # Optimized API image
├── Dockerfile.base                        # Base image with shared deps
├── Dockerfile.prefect                     # Orchestration image (Prefect)
├── Dockerfile.prefect.optimized           # Optimized Prefect image
├── Dockerfile.worker                      # Worker service image
├── docs/                                  # Project documentation
│   ├── api/                               # API docs/reference
│   ├── architecture.md                    # System architecture overview
│   ├── database-schema-design.md          # Database schema design
│   ├── deep-research.md                   # Research notes
│   ├── design-overview.md                 # Design decisions
│   ├── development.md                     # Dev setup and workflows
│   ├── diagrams/                          # Architecture/workflow diagrams
│   ├── file-structure.md                  # This file
│   ├── guides/                            # How-to guides
│   ├── linear/                            # Ticketing/planning notes
│   ├── logging-guide.md                   # Logging guidance
│   ├── meetings/                          # Meeting notes
│   ├── tech-stack.md                      # Tech stack summary
│   ├── test-framework-summary.md          # Testing strategy
│   ├── tickets/                           # Ticket drafts
│   ├── training-data-pipeline.md          # Training data pipeline
│   ├── viu/                               # Visualization/UI docs
│   ├── workflow-orchestration-setup.md    # Prefect/flows setup
│   └── workflow-orchestration.md          # Orchestration overview
├── examples/                              # Example scripts/snippets
├── frontend/                              # Web UI application
├── htmlcov/                               # Test coverage reports
├── jwst.png                               # Project image/logo
├── LICENSE                                # MIT License
├── logs/                                  # Local/dev logs
├── notebooks/                             # Jupyter notebooks & helpers
│   ├── API_KEYS_IMPLEMENTATION.md         # API key usage in notebooks
│   ├── auth_helper.py                     # Notebook auth helpers
│   ├── auth_template.py                   # Auth template for notebooks
│   ├── create_api_key.py                  # Create API keys via API
│   ├── data/                              # Sample data
│   ├── logs/                              # Notebook logs
│   ├── ml_training_data/                  # ML training data exploration
│   ├── notebook_processing_results/       # Notebook run outputs
│   ├── training/                          # Training experimentation
│   ├── test_storage/                      # Storage test notebooks
│   ├── processing.ipynb                   # Processing walkthrough
│   ├── simple_ingestion_test.ipynb        # Ingestion test
│   ├── training_data_pipeline_test.ipynb  # Pipeline test
│   ├── astrXX_testing.ipynb               # Scenario-specific tests
│   ├── test_api_keys.py                   # API keys tests
│   └── test_training_pipeline.py          # Training pipeline tests
├── pyproject.toml                         # Project config & dependencies
├── README.md                              # Project overview and quickstart
├── scripts/                               # Dev/ops scripts
├── src/                                   # Application source code
│   ├── __init__.py                        # Package marker
│   ├── adapters/                          # Framework/external integrations
│   │   ├── __init__.py                    # Package marker
│   │   ├── api/                           # FastAPI app and plumbing
│   │   │   ├── main.py                    # App bootstrap & server init
│   │   │   ├── docs.py                    # OpenAPI/docs configuration
│   │   │   ├── rate_limiting.py           # Rate limiting middleware
│   │   │   ├── routes/                    # Route modules by domain
│   │   │   └── versioning.py              # API version handling
│   │   ├── auth/                          # AuthN/AuthZ adapters
│   │   │   ├── api_key_auth.py            # API key backend & validation
│   │   │   ├── api_key_service.py         # API key management
│   │   │   ├── models.py                  # Auth-related models
│   │   │   ├── rbac.py                    # Role-based access control
│   │   │   └── supabase_auth.py           # Supabase auth integration
│   │   ├── external/                      # Third-party data sources
│   │   │   ├── mast.py                    # MAST archive client
│   │   │   ├── r2.py                      # Cloudflare R2 client
│   │   │   ├── skyview.py                 # SkyView client
│   │   │   └── vizier.py                  # VizieR catalog integration
│   │   ├── imaging/                       # Astronomical imaging utils
│   │   │   ├── fits_io.py                 # FITS I/O & metadata
│   │   │   ├── preprocess.py              # Preprocessing routines
│   │   │   └── utils.py                   # Shared imaging helpers
│   │   ├── ml/                            # ML model adapters
│   │   │   └── unet.py                    # U-Net model wrapper
│   │   ├── scheduler/                     # Flow scheduling/deployment
│   │   │   ├── config.py                  # Scheduler config
│   │   │   ├── deploy.py                  # Flow deployment utils
│   │   │   └── flows/                     # Scheduling flows
│   │   └── workers/                       # Background workers & tasks
│   │       ├── config.py                  # Worker configuration
│   │       ├── monitoring.py              # Worker health/metrics
│   │       ├── start_workers.py           # Workers entry point
│   │       ├── tasks.py                   # Shared task definitions
│   │       ├── training_data.py           # Training data tasks
│   │       ├── curation/                  # Curation tasks
│   │       ├── detection/                 # Detection tasks
│   │       ├── differencing/              # Differencing tasks
│   │       ├── ingestion/                 # Ingestion tasks
│   │       └── preprocessing/             # Preprocessing tasks
│   ├── core/                              # Cross-cutting core utilities
│   │   ├── api/                           # API-level helpers
│   │   │   └── response_wrapper.py        # Standard API responses
│   │   ├── constants.py                   # Global constants/config
│   │   ├── db/                            # DB abstractions & sessions
│   │   │   ├── base_crud.py               # Base CRUD (SQLAlchemy)
│   │   │   ├── exceptions.py              # DB error types
│   │   │   ├── models.py                  # Shared ORM models
│   │   │   └── session.py                 # Engine/session init
│   │   ├── dependencies.py                # Dependency injection wiring
│   │   ├── energy_analysis.py             # Energy usage analysis
│   │   ├── exceptions.py                  # Common exception types
│   │   ├── gpu_monitoring.py              # GPU monitoring helpers
│   │   ├── logging.py                     # Logging configuration
│   │   ├── logging_examples.py            # Structured logging examples
│   │   └── logging_utils.py               # Logging helpers/utilities
│   ├── domains/                           # Business logic (DDD)
│   │   ├── __init__.py                    # Package marker
│   │   ├── catalog/                       # Data cataloging
│   │   │   ├── api/                       # Catalog endpoints
│   │   │   ├── crud.py                    # Catalog CRUD
│   │   │   ├── models.py                  # Catalog models
│   │   │   ├── repository.py              # Catalog repository
│   │   │   ├── schema.py                  # Catalog schemas
│   │   │   └── service.py                 # Catalog services
│   │   ├── curation/                      # Human validation/curation
│   │   │   ├── api/                       # Curation endpoints
│   │   │   ├── crud.py                    # Curation CRUD
│   │   │   ├── models.py                  # Curation models
│   │   │   ├── repository.py              # Curation repository
│   │   │   ├── schema.py                  # Curation schemas
│   │   │   └── service.py                 # Curation services
│   │   ├── differencing/                  # Image differencing
│   │   │   ├── api/                       # Differencing endpoints
│   │   │   ├── crud.py                    # Differencing CRUD
│   │   │   ├── models.py                  # Differencing models
│   │   │   ├── repository.py              # Differencing repository
│   │   │   ├── schema.py                  # Differencing schemas
│   │   │   └── service.py                 # Differencing services
│   │   ├── detection/                     # Anomaly detection
│   │   │   ├── config.py                  # Detection config
│   │   │   ├── entities.py                # Detection entities
│   │   │   ├── models.py                  # Detection models
│   │   │   ├── analyzers/                 # Analysis components
│   │   │   ├── extractors/                # Feature extraction
│   │   │   ├── filters/                   # Filtering components
│   │   │   ├── metrics/                   # Metrics computation
│   │   │   ├── ml_entities/               # ML entity definitions
│   │   │   ├── processors/                # Processing stages
│   │   │   ├── scorers/                   # Scoring components
│   │   │   ├── services/                  # Detection services
│   │   │   ├── storage/                   # Detection storage
│   │   │   ├── validators/                # Validation utilities
│   │   │   ├── api/                       # Detection endpoints
│   │   │   ├── crud.py                    # Detection CRUD
│   │   │   ├── repository.py              # Detection repository
│   │   │   ├── schema.py                  # Detection schemas
│   │   │   └── service.py                 # Detection services
│   │   ├── ml/                            # ML workflows & training data
│   │   │   ├── flows/                     # Orchestrated ML flows
│   │   │   └── training_data/             # Training data transforms
│   │   ├── observations/                  # Observations management
│   │   │   ├── adapters/                  # Observation adapters
│   │   │   ├── api/                       # Observation endpoints
│   │   │   ├── catalogs/                  # Linked catalogs
│   │   │   ├── extractors/                # Feature extraction
│   │   │   ├── flows/                     # Observation flows
│   │   │   ├── ingestion/                 # Ingestion components
│   │   │   ├── integrations/              # External integrations
│   │   │   ├── processors/                # Processing components
│   │   │   ├── config.py                  # Observation config
│   │   │   ├── crud.py                    # Observation CRUD
│   │   │   ├── events.py                  # Domain events
│   │   │   ├── exceptions.py              # Domain exceptions
│   │   │   ├── models.py                  # Observation models
│   │   │   ├── repository.py              # Observation repository
│   │   │   ├── schema.py                  # Observation schemas
│   │   │   ├── service.py                 # Observation services
│   │   │   ├── survey_config.py           # Survey config model
│   │   │   ├── survey_config_service.py   # Survey config services
│   │   │   ├── testing/                   # Test utilities
│   │   │   └── validators.py              # Validation utilities
│   │   └── preprocessing/                 # Image preprocessing
│   │       ├── alignment/                 # Alignment components
│   │       ├── calibration/               # Calibration components
│   │       ├── normalizers/               # Normalization components
│   │       ├── pipeline/                  # Preprocessing pipeline
│   │       ├── processors/                # Processing components
│   │       ├── quality/                   # Quality checks
│   │       ├── storage/                   # Preprocessing storage
│   │       ├── api/                       # Preprocessing endpoints
│   │       ├── crud.py                    # Preprocessing CRUD
│   │       ├── models.py                  # Preprocessing models
│   │       ├── repository.py              # Preprocessing repository
│   │       └── schema.py                  # Preprocessing schemas
│   └── infrastructure/                    # Infra integrations/services
│       ├── __init__.py                    # Package marker
│       ├── mlflow/                        # MLflow server & registry
│       │   ├── config.py                  # MLflow configuration
│       │   ├── experiment_tracker.py      # Experiment tracking utils
│       │   ├── mlflow_server.py           # MLflow server management
│       │   ├── model_registry.py          # Model registry helpers
│       │   └── model_versioning.py        # Model versioning utilities
│       ├── storage/                       # Storage clients/abstractions
│       │   ├── config.py                  # Storage configuration
│       │   ├── content_addressed_storage.py # CAS layer
│       │   ├── dvc_client.py              # DVC client helpers
│       │   ├── mlflow_storage.py          # MLflow artifact storage
│       │   ├── r2_client.py               # Cloudflare R2 client
│       │   └── README.md                  # Storage notes
│       └── workflow/                      # Workflow orchestration
│           ├── api/                       # Workflow API surface
│           ├── config.py                  # Prefect/workflow config
│           ├── monitoring.py              # Flow monitoring
│           └── prefect_server.py          # Local Prefect server
├── start-dev-with-frontend.sh             # Start backend + frontend
├── start-dev.sh                           # Start backend-only dev
├── start-frontend.sh                      # Start frontend-only dev
├── tests/                                 # Test suite & fixtures
│   ├── api/                               # API tests
│   ├── conftest.py                        # Pytest config/fixtures
│   ├── data/                              # Test data assets
│   ├── domains/                           # Domain-specific tests
│   ├── e2e/                               # End-to-end tests
│   ├── fixtures/                          # Shared test fixtures
│   ├── infrastructure/                    # Infra integration tests
│   ├── integration/                       # Cross-module integration
│   ├── mocks/                             # Test doubles/mocks
│   ├── scripts/                           # Helper test scripts
│   ├── test_astr92_workers.py             # Workers tests
│   ├── test_framework_validation.py       # Framework validations
│   ├── test_workflow_orchestration.py     # Orchestration tests
│   ├── unit/                              # Unit tests by module
│   └── utils.py                           # Test utilities
├── uv.lock                                # UV package lockfile
└── zOLD/                                  # Archived/legacy content
```

Additional API route contents

```text
src/adapters/api/routes/
├── api_keys.py                # API key management endpoints
├── auth.py                    # Authentication endpoints
├── health.py                  # Health/ready/liveness checks
├── mlflow.py                  # MLflow-related endpoints
├── storage.py                 # Storage interactions (R2/DVC/etc.)
├── stream.py                  # Server-Sent Events / streaming
└── workers.py                 # Worker control/monitoring

src/domains/catalog/api/
├── __init__.py                # Package marker
└── routes.py                  # Catalog domain endpoints

src/domains/curation/api/
├── __init__.py                # Package marker
└── routes.py                  # Curation domain endpoints

src/domains/differencing/api/
├── __init__.py                # Package marker
└── routes.py                  # Differencing domain endpoints

src/domains/detection/api/
├── __init__.py                # Package marker
└── routes.py                  # Detection domain endpoints

src/domains/observations/api/
├── __init__.py                # Package marker
├── routes.py                  # Observation domain endpoints
└── survey_config_routes.py    # Survey configuration endpoints

src/domains/preprocessing/api/
├── __init__.py                # Package marker
└── routes.py                  # Preprocessing domain endpoints
```
