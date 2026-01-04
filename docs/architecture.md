# AstrID Architecture Overview

## System Architecture

AstrID follows a **Domain-Driven Design (DDD)** approach with a clean separation between business logic and infrastructure concerns. The system is built around bounded contexts that represent different areas of astronomical data processing.

> **Note**: For design principles and rationale, see [`design-overview.md`](design-overview.md). For technology choices, see [`tech-stack.md`](tech-stack.md).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                          │
│                    (Next.js + Tailwind)                        │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                          │
│                    (FastAPI + Auth)                             │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Domain Services Layer                        │
│              (Pure business logic, no frameworks)               │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                          │
│           (Databases, Storage, Message Queues)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Bounded Contexts

### 1. Observations Context

**Purpose**: Manage astronomical observations from various surveys

**Core Entities**:
- `Observation`: Astronomical image with metadata
- `Survey`: Source survey information
- `ObservationRun`: Processing execution details

**Key Operations**:
- Fetch new observations from external APIs
- Validate observation metadata
- Track processing status
- Manage observation lifecycle

### 2. Preprocessing Context

**Purpose**: Prepare astronomical images for analysis

**Core Entities**:
- `PreprocessRun`: Preprocessing execution details
- `CalibrationData`: Bias, dark, flat calibration frames
- `WCSInfo`: World Coordinate System information

**Key Operations**:
- Apply calibration corrections
- Perform WCS alignment
- Image registration and normalization
- Quality assessment

### 3. Differencing Context

**Purpose**: Detect changes between observations

**Core Entities**:
- `DifferenceRun`: Differencing execution details
- `Candidate`: Potential transient source
- `ReferenceImage`: Baseline image for comparison

**Key Operations**:
- ZOGY algorithm implementation
- Source extraction (SEP/photutils)
- Candidate filtering and scoring
- Difference image generation

### 4. Detection Context

**Purpose**: Machine learning-based anomaly detection

**Core Entities**:
- `Detection`: Identified astronomical anomaly
- `ModelRun`: ML model execution details
- `InferenceResult`: Model prediction output

**Key Operations**:
- U-Net model inference
- Confidence scoring
- Anomaly classification
- Model performance tracking

### 5. Curation Context

**Purpose**: Human validation and curation

**Core Entities**:
- `ValidationEvent`: Human review event
- `Curator`: Person performing validation
- `ValidationLabel`: Assigned classification

**Key Operations**:
- Human review interface
- Label assignment
- Quality control
- Feedback collection

### 6. Catalog Context

**Purpose**: Persistent storage and analytics

**Core Entities**:
- `CatalogEntry`: Finalized detection record
- `Analytics`: Statistical summaries
- `ExportJob`: Data export requests

**Key Operations**:
- Data persistence
- Statistical analysis
- Report generation
- Data export

## Data Flow Architecture

### Event-Driven Pipeline

```
Observation Ingested
       │
       ▼
┌─────────────────┐
│  Preprocessing  │ ──→ Observation Preprocessed
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   Differencing  │ ──→ Candidates Found
└─────────────────┘
       │
       ▼
┌─────────────────┐
│    Inference    │ ──→ Detections Scored
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   Validation    │ ──→ Detections Validated
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   Cataloging    │ ──→ Data Persisted
└─────────────────┘
```

### Message Flow

1. **Redis Pub/Sub**: Lightweight event broadcasting
2. **Dramatiq Workers**: Background task processing
3. **Prefect Flows**: Complex workflow orchestration
4. **Database Events**: Change notification triggers

## Technology Integration

### Database Layer

- **PostgreSQL**: Primary relational database
- **SQLAlchemy 2**: Modern async ORM
- **Alembic**: Database migration management
- **Connection Pooling**: Optimized database connections

### Storage Layer

- **Cloudflare R2**: S3-compatible object storage
- **Content Addressing**: Hash-based file naming
- **Versioning**: Dataset and model versioning
- **Lifecycle Management**: Automatic cleanup policies

### ML Infrastructure

- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control
- **TensorFlow/Keras**: Model training and inference
- **Model Serving**: RESTful inference endpoints

### Message Infrastructure

- **Redis**: Message broker and caching
- **Dramatiq**: Background task processing
- **Prefect**: Workflow orchestration
- **Event Sourcing**: Audit trail and replay capability

## Security Architecture

### Authentication & Authorization

- **Supabase JWT**: Secure token-based authentication
- **Role-Based Access Control**: Granular permission system
- **API Key Management**: Service-to-service authentication
- **Audit Logging**: Comprehensive access tracking

### Data Security

- **Encryption at Rest**: Database and storage encryption
- **Encryption in Transit**: TLS for all communications
- **Secrets Management**: Environment-based configuration
- **Input Validation**: Comprehensive data sanitization

## Scalability Considerations

### Horizontal Scaling

- **Stateless Services**: API and worker services
- **Load Balancing**: Multiple service instances
- **Database Sharding**: Partitioned data storage
- **Cache Distribution**: Redis cluster support

### Performance Optimization

- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Database and HTTP connection reuse
- **Batch Processing**: Efficient bulk operations
- **CDN Integration**: Global content distribution

## Monitoring & Observability

### Health Monitoring

- **Service Health Checks**: Built-in health endpoints
- **Dependency Monitoring**: Database, Redis, storage health
- **Performance Metrics**: Response times and throughput
- **Resource Utilization**: CPU, memory, disk usage

### Logging & Tracing

- **Structured Logging**: JSON-formatted log output
- **Log Aggregation**: Centralized log collection
- **Distributed Tracing**: Request flow tracking
- **Error Reporting**: Sentry integration

### Metrics & Alerting

- **Prometheus Metrics**: Time-series data collection
- **Grafana Dashboards**: Visualization and monitoring
- **Alert Rules**: Automated notification triggers
- **SLA Monitoring**: Service level agreement tracking

## Deployment Architecture

### Development Environment

- **Docker Compose**: Local service orchestration
- **Volume Mounts**: Live code reloading
- **Environment Variables**: Configuration management
- **Health Checks**: Service dependency validation

### Production Environment

- **Container Orchestration**: Kubernetes or Docker Swarm
- **Service Mesh**: Inter-service communication
- **Load Balancing**: Traffic distribution
- **Auto-scaling**: Dynamic resource allocation

### CI/CD Pipeline

- **GitHub Actions**: Automated testing and deployment
- **Docker Registry**: Container image management
- **Environment Promotion**: Staging to production
- **Rollback Capability**: Quick issue resolution

## Future Architecture Considerations

### Microservices Evolution

- **Service Decomposition**: Further domain separation
- **API Gateway**: Centralized routing and auth
- **Service Mesh**: Advanced networking features
- **Event Sourcing**: Complete audit trail

### Cloud-Native Features

- **Serverless Functions**: Event-driven processing
- **Managed Services**: Database and storage services
- **Auto-scaling**: Dynamic resource management
- **Multi-region**: Global deployment

### Advanced ML Features

- **Model Serving**: Dedicated inference services
- **A/B Testing**: Model performance comparison
- **Online Learning**: Continuous model updates
- **Federated Learning**: Distributed model training
