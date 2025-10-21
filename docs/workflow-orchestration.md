# Workflow Orchestration with Prefect

This document describes the workflow orchestration system implemented for AstrID using Prefect 2.x.

## Overview

The workflow orchestration system provides automated processing pipelines for:
- **Observation Processing**: Ingestion, preprocessing, differencing, detection, and validation
- **Model Training**: Training, hyperparameter optimization, evaluation, deployment, and retraining
- **Monitoring & Alerting**: Real-time monitoring, performance tracking, and failure alerts

## Architecture

### Components

1. **Prefect Server**: Central orchestration engine
2. **Prefect Workers**: Execute flows in parallel
3. **Workflow Flows**: Domain-specific processing pipelines
4. **Monitoring System**: Health checks, metrics, and alerting
5. **API Endpoints**: REST API for flow management

### Directory Structure

```
src/infrastructure/workflow/
├── __init__.py
├── config.py              # Configuration and data models
├── prefect_server.py      # Prefect server management
├── monitoring.py          # Monitoring and alerting
└── api/
    ├── __init__.py
    └── workflow_api.py    # REST API endpoints

src/domains/observations/flows/
├── __init__.py
└── observation_flows.py   # Observation processing workflows

src/domains/ml/flows/
├── __init__.py
└── training_flows.py      # ML training workflows
```

## Configuration

### Environment Variables

```bash
# Prefect Server
PREFECT_SERVER_URL=http://localhost:4200
PREFECT_API_URL=http://localhost:4200/api
PREFECT_UI_URL=http://localhost:4200/ui

# Database
PREFECT_SUPABASE_PROJECT_REF=your_project_ref
PREFECT_SUPABASE_PASSWORD=your_password
PREFECT_SUPABASE_HOST=your_host

# Workflow Settings
MAX_CONCURRENT_FLOWS=10
FLOW_TIMEOUT=3600
RETRY_ATTEMPTS=3
```

### WorkflowConfig

```python
@dataclass
class WorkflowConfig:
    prefect_server_url: str
    database_url: str
    storage_config: StorageConfig
    authentication_enabled: bool = True
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    max_concurrent_flows: int = 10
    flow_timeout: int = 3600
    retry_attempts: int = 3
```

## Workflow Types

### Observation Processing Flows

1. **observation_ingestion_flow**: Ingest new observations
2. **observation_preprocessing_flow**: Calibrate and align images
3. **observation_differencing_flow**: Create difference images
4. **observation_detection_flow**: Detect anomalies using ML
5. **observation_validation_flow**: Validate processing results
6. **complete_observation_processing_flow**: End-to-end processing

### ML Training Flows

1. **model_training_flow**: Train new ML models
2. **hyperparameter_optimization_flow**: Optimize model parameters
3. **model_evaluation_flow**: Evaluate model performance
4. **model_deployment_flow**: Deploy models to production
5. **model_retraining_flow**: Retrain models with new data

## API Endpoints

### Flow Management

- `GET /workflows/flows` - List workflows
- `POST /workflows/flows/{flow_type}/start` - Start a workflow
- `GET /workflows/flows/{flow_id}/status` - Get flow status
- `POST /workflows/flows/{flow_id}/cancel` - Cancel a flow
- `POST /workflows/flows/{flow_id}/retry` - Retry a failed flow

### Monitoring

- `GET /workflows/flows/{flow_id}/logs` - Get flow logs
- `GET /workflows/flows/{flow_id}/metrics` - Get flow metrics
- `GET /workflows/flows/{flow_id}/report` - Get flow report
- `POST /workflows/flows/{flow_id}/alerts` - Configure alerts
- `GET /workflows/health` - System health check

## Usage Examples

### Starting a Workflow

```python
from src.infrastructure.workflow.prefect_server import PrefectServer
from src.infrastructure.workflow.config import WorkflowConfig

# Configure workflow
config = WorkflowConfig(
    prefect_server_url="http://localhost:4200",
    database_url="postgresql://localhost/astrid",
    storage_config=storage_config
)

# Start server
server = PrefectServer(config)
await server.start_server()

# Start observation processing
from src.domains.observations.flows import observation_ingestion_flow

observation_data = {
    "survey_id": "survey_123",
    "coordinates": {"ra": 180.0, "dec": 30.0},
    "exposure_time": 300.0,
    "filter_name": "r"
}

result = await observation_ingestion_flow(observation_data, observation_service)
```

### Monitoring Workflows

```python
from src.infrastructure.workflow.monitoring import WorkflowMonitoring

# Setup monitoring
monitoring = WorkflowMonitoring(config)
await monitoring.start_monitoring()

# Get flow status
status = await monitoring.monitor_flow_performance("flow_123")
print(f"Flow status: {status['status']}")
print(f"Duration: {status['duration_seconds']}s")

# Generate report
report = await monitoring.generate_flow_reports(
    "flow_123", 
    (start_time, end_time)
)
```

## Docker Deployment

### Prefect Server

```yaml
# docker-compose.yaml
services:
  prefect:
    image: python:3.11-slim
    ports:
      - "4200:4200"
    environment:
      - PREFECT_API_URL=http://localhost:4200/api
      - PREFECT_SERVER_DATABASE_CONNECTION_URL=postgresql+asyncpg://...
    command: prefect server start --host 0.0.0.0 --port 4200

  prefect-worker:
    build:
      context: .
      dockerfile: Dockerfile.prefect
    environment:
      - PREFECT_API_URL=http://prefect:4200/api
      - PREFECT_WORK_POOL_NAME=astrid-pool
    depends_on:
      - prefect
```

### Starting the System

```bash
# Start Prefect server and worker
docker-compose up prefect prefect-worker

# Start the main API
docker-compose up api
```

## Monitoring and Alerting

### Health Checks

The system provides comprehensive health checks for:
- Prefect server connectivity
- Database connections
- Work pool status
- Worker availability

### Metrics

Tracked metrics include:
- Flow execution time
- Success/failure rates
- Resource usage (CPU, memory)
- Task completion rates
- Retry counts

### Alerts

Configure alerts for:
- Flow failures
- Performance degradation
- Timeout violations
- Resource exhaustion

## Error Handling

### Retry Logic

- Automatic retries for transient failures
- Configurable retry attempts and delays
- Exponential backoff for repeated failures

### Failure Recovery

- Graceful degradation on service failures
- Flow state persistence across restarts
- Manual intervention capabilities

## Testing

### Unit Tests

```bash
# Run workflow tests
pytest tests/test_workflow_orchestration.py -v
```

### Integration Tests

```bash
# Test with Prefect server
pytest tests/test_workflow_integration.py -v
```

## Performance Considerations

### Parallel Execution

- Concurrent task execution within flows
- Multiple worker processes
- Resource pooling and management

### Scalability

- Horizontal scaling with multiple workers
- Load balancing across work pools
- Database connection pooling

### Optimization

- Flow result caching
- Incremental processing
- Resource cleanup and garbage collection

## Security

### Authentication

- API key authentication
- Role-based access control
- Secure credential management

### Data Protection

- Encrypted data transmission
- Secure storage of sensitive data
- Audit logging and compliance

## Troubleshooting

### Common Issues

1. **Prefect server not starting**
   - Check database connectivity
   - Verify environment variables
   - Check port availability

2. **Workers not connecting**
   - Verify Prefect API URL
   - Check work pool configuration
   - Review worker logs

3. **Flow execution failures**
   - Check flow logs for errors
   - Verify service dependencies
   - Review resource constraints

### Debugging

```bash
# Check Prefect server logs
docker logs astrid-prefect-dev

# Check worker logs
docker logs astrid-prefect-worker-dev

# Check flow status
curl http://localhost:4200/api/flow_runs/{flow_id}
```

## Future Enhancements

### Planned Features

1. **Advanced Scheduling**: Cron-based scheduling, conditional execution
2. **Flow Dependencies**: Complex workflow graphs with dependencies
3. **Resource Management**: Dynamic resource allocation and scaling
4. **Advanced Monitoring**: Real-time dashboards and analytics
5. **Flow Templates**: Reusable workflow templates and components

### Integration Opportunities

1. **CI/CD Integration**: Automated testing and deployment workflows
2. **Data Pipeline Integration**: ETL workflows for data processing
3. **ML Pipeline Integration**: Automated model training and deployment
4. **Notification Integration**: Slack, email, and webhook notifications
