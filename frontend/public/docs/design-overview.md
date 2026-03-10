# AstrID Design Overview

## System Vision

AstrID is an astronomical identification system that processes observations from multiple surveys to detect transient astronomical phenomena. The system follows Domain-Driven Design principles with a clean separation between business logic and infrastructure concerns.

## Architecture Principles

### 1. Domain-Driven Design (DDD)
- **Bounded Contexts**: Clear separation between different areas of astronomical data processing
- **Domain Models**: Rich business objects that encapsulate behavior
- **Ubiquitous Language**: Consistent terminology across code and documentation

### 2. Event-Driven Architecture
- **Asynchronous Processing**: Non-blocking operations for scalability
- **Event Sourcing**: Complete audit trail of system changes
- **Message Queues**: Redis and Dramatiq for reliable message processing

### 3. Clean Architecture
- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Interface Segregation**: Small, focused interfaces
- **Single Responsibility**: Each component has one reason to change

## Data Flow Pipeline

The AstrID system processes astronomical data through a well-defined pipeline:

```
External Surveys → Observation Ingestion → Preprocessing → Differencing → Detection → Validation → Cataloging
```

### Stage 1: Observation Ingestion
- **Input**: External survey APIs (MAST, SkyView)
- **Process**: Download FITS files and metadata
- **Output**: Stored observations with complete metadata
- **Key Components**: MAST Client, SkyView Client, R2 Storage

### Stage 2: Preprocessing
- **Input**: Raw FITS files
- **Process**: Calibration, WCS alignment, image registration
- **Output**: Processed images ready for analysis
- **Key Components**: Calibration Engine, WCS Handler

### Stage 3: Differencing
- **Input**: Processed observations and reference images
- **Process**: ZOGY algorithm, source extraction, candidate filtering
- **Output**: Difference images and candidate sources
- **Key Components**: ZOGY Algorithm, SEP Integration

### Stage 4: Detection
- **Input**: Difference images and candidates
- **Process**: U-Net model inference, anomaly detection
- **Output**: Scored detections with confidence levels
- **Key Components**: U-Net Model, MLflow Integration

### Stage 5: Validation
- **Input**: ML detections
- **Process**: Human review, label assignment, quality assessment
- **Output**: Validated detections with human labels
- **Key Components**: Human Interface, Validation Service

### Stage 6: Cataloging
- **Input**: Validated detections
- **Process**: Analytics, reporting, data export
- **Output**: Final catalog entries and alerts
- **Key Components**: Analytics Engine, Export Service

## Technology Stack

> **Note**: For detailed technology stack information, see [`tech-stack.md`](tech-stack.md).

**Key Technologies:**
- **Backend**: Python 3.11+, FastAPI, SQLAlchemy 2, PostgreSQL, Redis, Dramatiq, Prefect
- **ML/AI**: TensorFlow/Keras, MLflow, DVC, U-Net
- **Storage**: Cloudflare R2 (S3-compatible), Content-addressable storage
- **Frontend**: Next.js, Tailwind CSS, TypeScript
- **External Services**: MAST API, SkyView, Astronomical Catalogs

## Domain Structure

### 1. Observations Domain
**Purpose**: Manage astronomical observations from various surveys
**Key Entities**: Survey, Observation
**Responsibilities**:
- Fetch observations from external APIs
- Validate observation metadata
- Track processing status
- Manage observation lifecycle

### 2. Preprocessing Domain
**Purpose**: Prepare astronomical images for analysis
**Key Entities**: PreprocessRun, CalibrationData, WCSInfo
**Responsibilities**:
- Apply calibration corrections
- Perform WCS alignment
- Image registration and normalization
- Quality assessment

### 3. Differencing Domain
**Purpose**: Detect changes between observations
**Key Entities**: DifferenceRun, Candidate, ReferenceImage
**Responsibilities**:
- ZOGY algorithm implementation
- Source extraction (SEP/photutils)
- Candidate filtering and scoring
- Difference image generation

### 4. Detection Domain
**Purpose**: Machine learning-based anomaly detection
**Key Entities**: Detection, ModelRun, Model
**Responsibilities**:
- U-Net model inference
- Confidence scoring
- Anomaly classification
- Model performance tracking

### 5. Curation Domain
**Purpose**: Human validation and curation
**Key Entities**: ValidationEvent, Alert, Curator
**Responsibilities**:
- Human review interface
- Label assignment
- Quality control
- Feedback collection

### 6. Catalog Domain
**Purpose**: Persistent storage and analytics
**Key Entities**: CatalogEntry, Analytics, ExportJob
**Responsibilities**:
- Data persistence
- Statistical analysis
- Report generation
- Data export

## Development Phases

### Phase 1: Foundation (P1 Critical)
- **ASTR-69**: Development environment setup
- **ASTR-70**: Database setup and migrations
- **ASTR-73**: Observation domain models
- **ASTR-84**: Core API endpoints
- **ASTR-93**: Test framework setup

### Phase 2: Core Features (P2 High)
- **ASTR-71**: Cloud storage integration
- **ASTR-74**: Survey integration (MAST/SkyView)
- **ASTR-75**: FITS processing pipeline
- **ASTR-76**: Image preprocessing services
- **ASTR-78**: Image differencing algorithms
- **ASTR-80**: U-Net model integration
- **ASTR-88**: MLflow setup
- **ASTR-91**: Workflow orchestration

### Phase 3: Advanced Features (P3 Medium)
- **ASTR-82**: Human validation system
- **ASTR-83**: Data cataloging
- **ASTR-86**: Frontend dashboard
- **ASTR-97**: Production deployment

## Key Design Decisions

### 1. Event-Driven Processing
**Decision**: Use Redis pub/sub and Dramatiq workers for background processing
**Rationale**: Enables scalable, asynchronous processing with proper error handling and retry mechanisms

### 2. Content-Addressable Storage
**Decision**: Use hash-based file naming for R2 storage
**Rationale**: Enables deduplication, integrity verification, and efficient caching

### 3. Domain-Driven Design
**Decision**: Organize code around business domains rather than technical layers
**Rationale**: Improves maintainability, testability, and business alignment

### 4. Async-First Architecture
**Decision**: Use async/await throughout the Python codebase
**Rationale**: Enables high concurrency and efficient resource utilization

### 5. ML Model Registry
**Decision**: Use MLflow for model management and versioning
**Rationale**: Enables experiment tracking, model versioning, and production deployment

## Scalability Considerations

### Horizontal Scaling
- **Stateless Services**: API and worker services can be scaled horizontally
- **Load Balancing**: Multiple service instances with load balancing
- **Database Sharding**: Partitioned data storage for large datasets
- **Cache Distribution**: Redis cluster support for high availability

### Performance Optimization
- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Database and HTTP connection reuse
- **Batch Processing**: Efficient bulk operations
- **CDN Integration**: Global content distribution

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

## Next Steps

1. **Review Design Diagrams**: Study the PlantUML diagrams in `docs/diagrams/`
2. **Set Up Development Environment**: Follow ASTR-69 ticket
3. **Implement Database Schema**: Follow ASTR-70 ticket
4. **Create Domain Models**: Follow ASTR-73 ticket
5. **Build API Endpoints**: Follow ASTR-84 ticket

## Resources

- **Architecture Overview**: [`architecture.md`](architecture.md) - System structure and components
- **Technology Stack**: [`tech-stack.md`](tech-stack.md) - Detailed technology choices
- **File Structure**: [`file-structure.md`](file-structure.md) - Repository layout
- **Database Schema**: [`database-schema-design.md`](database-schema-design.md) - Database design
- **Architecture Diagrams**: `docs/diagrams/`
- **Development Guide**: [`development.md`](development.md) - Setup and workflows
