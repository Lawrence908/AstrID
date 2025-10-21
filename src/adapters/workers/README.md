# AstrID Dramatiq Workers

This module implements comprehensive background processing workers for the AstrID astronomical identification system using Dramatiq.

## Overview

The workers handle the complete pipeline from observation ingestion through detection and curation:

1. **Observation Ingestion** - Ingest observations from external sources
2. **Preprocessing** - Apply calibration, alignment, and quality assessment
3. **Differencing** - Create difference images and extract candidates
4. **Detection** - Run ML inference to detect anomalies
5. **Curation** - Prepare results for human validation

## Architecture

### Worker Types

- `observation_ingestion` - High priority, handles data ingestion
- `preprocessing` - Medium priority, image processing tasks
- `differencing` - Medium priority, image differencing algorithms
- `detection` - High priority, ML inference tasks
- `curation` - Low priority, human validation preparation
- `notification` - Low priority, alert and notification tasks

### Queue Configuration

Each worker type has its own queue with specific configuration:

```python
TaskQueue(
    queue_name="observation_ingestion",
    worker_type=WorkerType.OBSERVATION_INGESTION,
    priority=1,
    max_retries=3,
    timeout=300,
    concurrency=2,
    enabled=True,
)
```

## Usage

### Starting Workers

```bash
# Start all workers
python -m src.adapters.workers.start_workers

# Start specific worker types
python -m src.adapters.workers.start_workers --worker-types observation_ingestion preprocessing

# Start with custom configuration
python -m src.adapters.workers.start_workers --processes 2 --threads 8 --log-level DEBUG
```

### API Endpoints

The workers expose monitoring and management endpoints:

- `GET /workers/status` - Get status of all workers
- `GET /workers/health` - Get overall worker health
- `GET /workers/queues` - Get queue status
- `GET /workers/metrics` - Get performance metrics
- `POST /workers/{worker_type}/start` - Start specific worker type
- `POST /workers/{worker_type}/stop` - Stop specific worker type
- `POST /workers/{worker_type}/scale` - Scale worker count

### Programmatic Usage

```python
from src.adapters.workers.ingestion.observation_workers import ingest_observation
from src.adapters.workers.preprocessing.preprocessing_workers import preprocess_observation

# Send tasks to queues
ingest_observation.send({
    "observation_id": "obs_123",
    "survey_id": "survey_456",
    "ra": 180.0,
    "dec": 30.0,
    "observation_time": "2024-01-01T12:00:00Z"
})

preprocess_observation.send("obs_123")
```

## Worker Details

### Observation Ingestion Workers

**File**: `src/adapters/workers/ingestion/observation_workers.py`

**Tasks**:
- `ingest_observation(observation_data)` - Main ingestion task
- `batch_ingest_observations(observation_data_list)` - Batch processing
- `validate_observation_data(observation_data)` - Data validation
- `process_observation_metadata(observation_data)` - Metadata processing
- `store_observation_files(observation_data)` - File storage

**Features**:
- Comprehensive data validation
- Metadata enhancement
- Cloud storage integration
- Automatic preprocessing triggering

### Preprocessing Workers

**File**: `src/adapters/workers/preprocessing/preprocessing_workers.py`

**Tasks**:
- `preprocess_observation(observation_id)` - Main preprocessing task
- `apply_calibration(observation_id, calibration_frames)` - Calibration
- `align_observation(observation_id, reference_id)` - Image alignment
- `assess_quality(observation_id)` - Quality assessment
- `batch_preprocess_observations(observation_ids)` - Batch processing

**Features**:
- Bias/dark/flat correction
- WCS alignment
- Quality metrics calculation
- Automatic differencing triggering

### Differencing Workers

**File**: `src/adapters/workers/differencing/differencing_workers.py`

**Tasks**:
- `create_difference_image(observation_id, reference_id)` - Main differencing
- `apply_differencing_algorithm(observation_id, algorithm)` - Algorithm application
- `validate_difference_image(difference_id)` - Quality validation
- `extract_sources(difference_id)` - Source extraction
- `batch_difference_observations(observation_ids)` - Batch processing

**Features**:
- ZOGY algorithm implementation
- Classic differencing methods
- Source extraction (SEP/photutils)
- Automatic detection triggering

### Detection Workers

**File**: `src/adapters/workers/detection/detection_workers.py`

**Tasks**:
- `detect_anomalies(difference_id, model_id)` - Main detection task
- `run_ml_inference(difference_id, model_id)` - ML inference
- `validate_detections(detection_id)` - Detection validation
- `calculate_detection_metrics(detection_id)` - Metrics calculation
- `store_detection_results(detection_id)` - Result storage
- `batch_detect_anomalies(difference_ids, model_id)` - Batch processing

**Features**:
- U-Net model integration
- GPU acceleration
- Confidence scoring
- Performance metrics
- Automatic curation triggering

### Curation Workers

**File**: `src/adapters/workers/curation/curation_workers.py`

**Tasks**:
- `curate_detections(detection_id)` - Main curation task
- `create_validation_events(detection_id)` - Event creation
- `generate_alerts(detection_id)` - Alert generation
- `prepare_curation_interface(detection_id)` - Interface preparation
- `send_notifications(detection_id)` - Notification sending

**Features**:
- Human validation preparation
- Alert generation
- Interface data preparation
- Multi-channel notifications

## Configuration

### Environment Variables

```bash
# Redis configuration
REDIS_URL=redis://localhost:6379/0
REDIS_RESULT_URL=redis://localhost:6379/1

# Worker configuration
WORKER_MAX_RETRIES=3
WORKER_RETRY_DELAY=1000
WORKER_TIMEOUT=300
WORKER_MAX_MEMORY=1024
WORKER_MAX_CPU=80
WORKER_CONCURRENCY=4
WORKER_PREFETCH_MULTIPLIER=2
```

### Worker Configuration

```python
from src.adapters.workers.config import get_worker_config, get_task_queues

# Get configuration
config = get_worker_config()
queues = get_task_queues()

# Customize configuration
config.max_retries = 5
config.worker_timeout = 600
```

## Monitoring

### Health Checks

```python
from src.adapters.workers.monitoring import worker_monitor

# Get worker health
health = worker_monitor.get_worker_health()
print(f"Status: {health['status']}")
print(f"Healthy workers: {health['healthy_workers']}/{health['total_workers']}")

# Get performance metrics
metrics = worker_monitor.get_performance_metrics(time_window_hours=24)
print(f"Tasks processed: {metrics['total_tasks_processed']}")
print(f"Failure rate: {metrics['failure_rate']:.2%}")
```

### Metrics Collection

The workers automatically collect:
- Task processing counts
- Failure rates
- Processing times
- Memory usage
- CPU usage
- Queue lengths

## Error Handling

### Retry Logic

- Exponential backoff for transient failures
- Configurable max retries per queue
- Dead letter queue for permanent failures

### Error Recovery

- Automatic worker restart on crashes
- Queue-level error isolation
- Comprehensive error logging

## Development

### Adding New Workers

1. Create worker class in appropriate module
2. Define Dramatiq actors
3. Add to task queues configuration
4. Update monitoring if needed

### Testing

```bash
# Run worker tests
pytest tests/adapters/workers/

# Test specific worker type
pytest tests/adapters/workers/test_observation_workers.py
```

### Debugging

```bash
# Start workers with debug logging
python -m src.adapters.workers.start_workers --log-level DEBUG

# Monitor specific queue
python -m src.adapters.workers.start_workers --queues observation_ingestion
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

CMD ["python", "-m", "src.adapters.workers.start_workers"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: astrid-workers
spec:
  replicas: 3
  selector:
    matchLabels:
      app: astrid-workers
  template:
    metadata:
      labels:
        app: astrid-workers
    spec:
      containers:
      - name: workers
        image: astrid/workers:latest
        command: ["python", "-m", "src.adapters.workers.start_workers"]
        env:
        - name: REDIS_URL
          value: "redis://redis:6379/0"
```

## Troubleshooting

### Common Issues

1. **Workers not starting**: Check Redis connection
2. **Tasks not processing**: Verify queue configuration
3. **High memory usage**: Adjust concurrency settings
4. **Task failures**: Check error logs and retry configuration

### Logs

```bash
# View worker logs
docker logs astrid-workers

# Follow logs in real-time
docker logs -f astrid-workers
```

### Performance Tuning

- Adjust `concurrency` based on CPU cores
- Tune `prefetch_multiplier` for throughput
- Monitor memory usage and adjust `max_memory`
- Use separate queues for different priorities
