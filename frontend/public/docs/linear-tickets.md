# AstrID Linear Tickets & Feature Tracking

## Project Overview

**Project**: AstrID - Astronomical Identification System  
**Team**: Chris Lawrence (Lead Developer)  
**Timeline**: September 2025 - April 2026  
**Status**: Planning & Setup Phase  

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

### 🏗️ Core Infrastructure

#### ASTR-69: Development Environment Setup
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

#### ASTR-70: Database Setup and Migrations
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

#### ASTR-71: Cloud Storage Integration
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

### 🔐 Authentication & Security

#### ASTR-72: Supabase Integration
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

### 📡 Observations Domain

#### ASTR-73: Observation Models and Services
- **Project**: ASTRID-CORE
- **Priority**: P1 (Critical)
- **Labels**: `core-domain`, `high-priority`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-70
- **Description**: Implement core observation domain models and business logic
- **Subtasks**:
  - [ ] Implement Observation domain models
  - [ ] Create observation repository interface
  - [ ] Implement observation service layer
  - [ ] Add observation validation logic

#### ASTR-74: Survey Integration
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `integration`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-73
- **Description**: Integrate with external astronomical survey APIs
- **Subtasks**:
  - [ ] Integrate with MAST API for observations
  - [ ] Integrate with SkyView for image data
  - [ ] Implement survey-specific adapters
  - [ ] Add observation metadata extraction

#### ASTR-75: FITS Processing Pipeline
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `data-processing`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-73
- **Description**: Implement FITS file processing and WCS handling
- **Subtasks**:
  - [ ] Implement FITS file reading and writing
  - [ ] Add WCS (World Coordinate System) handling
  - [ ] Create image metadata extraction
  - [ ] Implement star catalog integration

### 🖼️ Preprocessing Domain

#### ASTR-76: Image Preprocessing Services
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `image-processing`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-75
- **Description**: Implement image calibration and preprocessing pipeline
- **Subtasks**:
  - [ ] Implement bias/dark/flat calibration
  - [ ] Add WCS alignment and registration
  - [ ] Create image quality assessment
  - [ ] Implement preprocessing pipeline orchestration

#### ASTR-77: Astronomical Image Processing
- **Project**: ASTRID-CORE
- **Priority**: P3 (Medium)
- **Labels**: `core-domain`, `image-processing`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-76
- **Description**: Advanced image processing with OpenCV and scikit-image
- **Subtasks**:
  - [ ] Integrate OpenCV for image manipulation
  - [ ] Add scikit-image for advanced processing
  - [ ] Implement image normalization and scaling
  - [ ] Create preprocessing result storage

### 🔍 Differencing Domain

#### ASTR-78: Image Differencing Algorithms
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `algorithm`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-76
- **Description**: Implement image differencing algorithms for anomaly detection
- **Subtasks**:
  - [ ] Implement ZOGY algorithm
  - [ ] Add classic differencing methods
  - [ ] Create reference image selection logic
  - [ ] Implement difference image generation

#### ASTR-79: Source Extraction
- **Project**: ASTRID-CORE
- **Priority**: P2 (High)
- **Labels**: `core-domain`, `algorithm`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-78
- **Description**: Extract and analyze sources from difference images
- **Subtasks**:
  - [ ] Integrate SEP for source extraction
  - [ ] Add photutils for additional analysis
  - [ ] Implement candidate filtering
  - [ ] Create candidate scoring system

### 🤖 Detection Domain

#### ASTR-80: U-Net Model Integration
- **Project**: ASTRID-ML
- **Priority**: P2 (High)
- **Labels**: `ml`, `model-integration`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-79
- **Description**: Integrate existing U-Net model into new architecture
- **Subtasks**:
  - [ ] Port existing U-Net model to new architecture
  - [ ] Implement model loading and inference
  - [ ] Add confidence scoring
  - [ ] Create model performance tracking

#### ASTR-81: Anomaly Detection Pipeline
- **Project**: ASTRID-ML
- **Priority**: P2 (High)
- **Labels**: `ml`, `pipeline`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-80
- **Description**: Complete anomaly detection service implementation
- **Subtasks**:
  - [ ] Implement detection service layer
  - [ ] Add detection validation logic
  - [ ] Create detection result storage
  - [ ] Implement detection metrics calculation

### 👥 Curation Domain

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

### 📊 Catalog Domain

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

### 🌐 FastAPI Implementation

#### ASTR-84: Core API Endpoints
- **Project**: ASTRID-API
- **Priority**: P1 (Critical)
- **Labels**: `api`, `high-priority`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-73, ASTR-81
- **Description**: Implement core API endpoints for all domains
- **Subtasks**:
  - [ ] Implement observations endpoints
  - [ ] Add detections endpoints
  - [ ] Create streaming endpoints (SSE)
  - [ ] Add health check and monitoring endpoints



#### ASTR-85: API Documentation and Testing
- **Project**: ASTRID-API
- **Priority**: P2 (High)
- **Labels**: `api`, `testing`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-84
- **Description**: Comprehensive API documentation and testing
- **Subtasks**:
  - [ ] Add comprehensive API documentation
  - [ ] Implement API testing suite
  - [ ] Add API versioning
  - [ ] Create API rate limiting

### 🎨 Frontend Dashboard

#### ASTR-86: Next.js Dashboard Setup
- **Project**: ASTRID-API
- **Priority**: P3 (Medium)
- **Labels**: `ui`, `frontend`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-84
- **Description**: Set up Next.js dashboard with authentication
- **Subtasks**:
  - [ ] Set up Next.js project with TypeScript
  - [ ] Implement Tailwind CSS styling
  - [ ] Create responsive layout components
  - [ ] Add authentication integration

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

## Epic: Machine Learning & Model Management

### 🧠 ML Infrastructure

#### ASTR-88: MLflow Integration
- **Project**: ASTRID-ML
- **Priority**: P2 (High)
- **Labels**: `ml`, `infrastructure`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-71
- **Description**: Set up MLflow for experiment tracking and model management
- **Subtasks**:
  - [ ] Set up MLflow tracking server
  - [ ] Implement experiment tracking
  - [ ] Add model registry functionality
  - [ ] Create model versioning system

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

### 🔄 Model Operations

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

### ⚙️ Prefect Integration

#### ASTR-91: Workflow Orchestration
- **Project**: ASTRID-WORK
- **Priority**: P2 (High)
- **Labels**: `workflow`, `orchestration`
- **Estimated Time**: 3 days
- **Dependencies**: ASTR-69
- **Description**: Set up Prefect for workflow orchestration
- **Subtasks**:
  - [ ] Set up Prefect server
  - [ ] Implement observation processing flows
  - [ ] Add model training workflows
  - [ ] Create monitoring and alerting

### 🚀 Background Processing

#### ASTR-92: Dramatiq Workers
- **Project**: ASTRID-WORK
- **Priority**: P2 (High)
- **Labels**: `workflow`, `background`
- **Estimated Time**: 4 days
- **Dependencies**: ASTR-91
- **Description**: Implement background processing workers
- **Subtasks**:
  - [ ] Implement observation ingestion workers
  - [ ] Add preprocessing workers
  - [ ] Create differencing workers
  - [ ] Implement detection workers

## Epic: Testing & Quality Assurance

### 🧪 Testing Infrastructure

#### ASTR-93: Test Framework Setup
- **Project**: ASTRID-TEST
- **Priority**: P1 (Critical)
- **Labels**: `testing`, `high-priority`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-69
- **Description**: Set up comprehensive testing infrastructure
- **Subtasks**:
  - [ ] Configure pytest with async support
  - [ ] Set up test database fixtures
  - [ ] Implement mock services
  - [ ] Add test coverage reporting

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

### 🔍 Code Quality

#### ASTR-95: Code Quality Tools
- **Project**: ASTRID-TEST
- **Priority**: P2 (High)
- **Labels**: `testing`, `quality`
- **Estimated Time**: 1 day
- **Dependencies**: ASTR-69
- **Description**: Configure code quality and formatting tools
- **Subtasks**:
  - [ ] Configure Ruff for linting
  - [ ] Set up MyPy for type checking
  - [ ] Implement Black code formatting
  - [ ] Add pre-commit hooks

## Epic: Deployment & Operations

### 🐳 Containerization

#### ASTR-96: Docker Setup
- **Project**: ASTRID-DEPLOY
- **Priority**: P2 (High)
- **Labels**: `deployment`, `docker`
- **Estimated Time**: 2 days
- **Dependencies**: ASTR-84
- **Description**: Containerize all services for deployment
- **Subtasks**:
  - [ ] Create API Dockerfile
  - [ ] Create worker Dockerfile
  - [ ] Set up Docker Compose for development
  - [ ] Implement health checks

### 🚀 Production Deployment

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

### 🔄 CI/CD Pipeline

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

### 📚 Documentation

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

### 🎓 Training & Knowledge Transfer

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


---


## Priority Matrix

### 🔴 High Priority (Phase 1: Sept-Oct 2025)
- **ASTR-69**: Development environment setup (P1)
- **ASTR-70**: Database setup (P1)
- **ASTR-73**: Observation models (P1)
- **ASTR-84**: Core API endpoints (P1)
- **ASTR-93**: Test framework (P1)

### 🟡 Medium Priority (Phase 2: Nov-Dec 2025)
- **ASTR-76**: Image preprocessing (P2)
- **ASTR-78**: Image differencing (P2)
- **ASTR-80**: U-Net integration (P2)
- **ASTR-88**: MLflow setup (P2)
- **ASTR-91**: Workflow orchestration (P2)

### 🟢 Lower Priority (Phase 3: Jan-Apr 2026)
- **ASTR-86**: Frontend dashboard (P3)
- **ASTR-82**: Human validation (P3)
- **ASTR-83**: Data cataloging (P3)
- **ASTR-97**: Production deployment (P3)
- **ASTR-99**: Documentation (P3)

## Success Metrics

### 🎯 Technical Metrics
- **Code Coverage**: >90% test coverage
- **API Response Time**: <200ms for 95% of requests
- **Model Accuracy**: >85% precision for anomaly detection
- **System Uptime**: >99.5% availability

### 📊 Business Metrics
- **Processing Throughput**: >1000 observations/day
- **Detection Rate**: >50% of actual anomalies detected
- **False Positive Rate**: <10% of detections
- **User Adoption**: >5 active users within 3 months

## Risk Assessment

### 🚨 High Risk
- **ML Model Performance**: U-Net may not generalize well to new data
- **Data Quality**: External survey data may be inconsistent
- **Scalability**: System may not handle high observation volumes

### ⚠️ Medium Risk
- **Integration Complexity**: Multiple external APIs may cause reliability issues
- **Performance**: Real-time processing may be resource-intensive
- **Security**: Astronomical data may have privacy implications

### ✅ Low Risk
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