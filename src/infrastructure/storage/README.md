# AstrID Storage Infrastructure

This module provides comprehensive cloud storage infrastructure for the AstrID project, implementing all requirements from ticket ASTR-71.

## Overview

The storage infrastructure consists of the following components:

- **R2StorageClient**: Direct interface to Cloudflare R2 object storage
- **ContentAddressedStorage**: Deduplication layer using SHA-256 content addressing
- **DVCClient**: Dataset versioning and lineage tracking
- **MLflowArtifactStorage**: ML model artifact storage with R2 backend
- **StorageConfig**: Configuration management for all storage services

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   API Endpoints     │    │  Content Addressed  │    │    DVC Client       │
│   /storage/*        │    │      Storage        │    │  Dataset Versioning │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        │
            ┌─────────────────────────────────────────────────────────┐
            │                R2StorageClient                          │
            │            Cloudflare R2 Interface                     │
            └─────────────────────────────────────────────────────────┘
                                        │
            ┌─────────────────────────────────────────────────────────┐
            │              Cloudflare R2 Object Storage               │
            │                 S3-Compatible API                      │
            └─────────────────────────────────────────────────────────┘
```

## Bucket Structure

The storage is organized into the following bucket structure:

```
astrid-storage/
├── cas/                           # Content-addressed storage
│   ├── ab/                        # First 2 chars of hash
│   │   └── ab123...456            # Full SHA-256 hash
├── raw-observations/              # Original FITS files
├── processed-observations/        # Calibrated and preprocessed data
├── difference-images/             # Image differencing results
├── detections/                    # Detection results and metadata
├── models/                        # ML model artifacts
├── datasets/                      # Training and validation datasets
├── artifacts/                     # MLflow artifacts
└── temp/                          # Temporary processing files
```

## Configuration

### Environment Variables

Set these environment variables for storage configuration:

```bash
# Cloudflare R2 Configuration
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_R2_ACCESS_KEY_ID=your_access_key
CLOUDFLARE_R2_SECRET_ACCESS_KEY=your_secret_key
CLOUDFLARE_R2_BUCKET_NAME=astrid-storage
CLOUDFLARE_R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com

# DVC Configuration
DVC_REMOTE_URL=s3://astrid-data
DVC_REMOTE_NAME=r2

# MLflow Configuration
MLFLOW_ARTIFACT_ROOT=s3://astrid-models
```

### Usage Example

```python
from src.infrastructure.storage import (
    StorageConfig,
    R2StorageClient,
    ContentAddressedStorage,
    DVCClient,
    MLflowArtifactStorage
)

# Initialize configuration
config = StorageConfig.from_env()
config.validate()

# Initialize clients
r2_client = R2StorageClient(config=config)
cas = ContentAddressedStorage(
    r2_client=r2_client,
    bucket=config.r2_bucket_name,
    prefix="cas/"
)
dvc_client = DVCClient(config=config)
```

## API Endpoints

The storage infrastructure provides the following REST API endpoints:

### File Operations

- `POST /storage/upload` - Upload file with optional content addressing
- `GET /storage/download/{content_hash}` - Download file by content hash
- `DELETE /storage/{content_hash}` - Delete file by content hash
- `GET /storage/metadata/{content_hash}` - Get file metadata

### Dataset Versioning

- `POST /storage/datasets/{dataset_id}/version` - Create dataset version
- `GET /storage/datasets/{dataset_id}/versions` - List dataset versions

### Health Check

- `GET /storage/health` - Check storage services health

## Components

### R2StorageClient

Direct interface to Cloudflare R2 with the following methods:

- `upload_file()` - Upload files or bytes data
- `download_file()` - Download files 
- `delete_file()` - Delete files
- `list_files()` - List files with prefix filtering
- `get_file_metadata()` - Get file metadata
- `file_exists()` - Check if file exists

### ContentAddressedStorage

Provides deduplication using SHA-256 content addressing:

- `store_data()` - Store data with automatic deduplication
- `retrieve_data()` - Retrieve data by content hash
- `store_file()` - Store file from local path
- `exists()` - Check if content exists
- `get_metadata()` - Get content metadata
- `delete_content()` - Delete content by hash

### DVCClient

Dataset versioning and lineage tracking:

- `init_repo()` - Initialize DVC repository
- `configure_remote()` - Configure R2 as DVC remote
- `add_dataset()` - Add dataset to DVC tracking
- `version_dataset()` - Create dataset version
- `pull_dataset()` - Pull dataset from remote
- `push_dataset()` - Push dataset to remote
- `list_versions()` - List dataset versions

### MLflowArtifactStorage

ML model artifact management with R2 backend:

- `store_model_artifact()` - Store model artifacts
- `retrieve_model_artifact()` - Retrieve model artifacts
- `list_model_artifacts()` - List artifacts in experiment
- `get_artifact_metadata()` - Get artifact metadata
- `configure_experiment_tracking()` - Configure experiments

## Features

### Content Addressing & Deduplication

- SHA-256 content hashing for unique identification
- Automatic deduplication of identical files
- Content verification on retrieval
- Hierarchical storage structure for performance

### Error Handling & Retry Logic

- Exponential backoff for network operations
- Comprehensive error handling for all storage operations
- Graceful degradation when storage is unavailable
- Detailed logging for all operations

### Security

- Encrypted data at rest (R2 default encryption)
- Secure credential management via environment variables
- Access control through R2 IAM policies
- Audit logging for all storage operations

## Testing

The storage infrastructure includes comprehensive unit and integration tests:

```bash
# Run storage tests
pytest tests/infrastructure/storage/

# Run specific test modules
pytest tests/infrastructure/storage/test_r2_client.py
pytest tests/infrastructure/storage/test_content_addressed_storage.py
pytest tests/infrastructure/storage/test_config.py
```

## Development

### Prerequisites

Install required dependencies:

```bash
uv add boto3 aioboto3 dvc[s3] mlflow
```

### Local Development

For local development, you can use MinIO as an S3-compatible storage backend:

```bash
# Start MinIO server
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  quay.io/minio/minio server /data --console-address ":9001"

# Configure environment for local development
export CLOUDFLARE_R2_ENDPOINT_URL=http://localhost:9000
export CLOUDFLARE_R2_ACCESS_KEY_ID=minioadmin
export CLOUDFLARE_R2_SECRET_ACCESS_KEY=minioadmin
```

## Performance Considerations

- Files are uploaded/downloaded in chunks for memory efficiency
- Content addressing provides natural deduplication
- Hierarchical bucket structure improves listing performance
- Async/await pattern for non-blocking I/O operations

## Monitoring & Observability

- Structured logging with correlation IDs
- Metrics for upload/download operations
- Health check endpoints for service monitoring
- Integration with Sentry for error tracking

## Future Enhancements

- Implement lifecycle policies for automated cleanup
- Add compression for large files
- Implement multi-part uploads for very large files
- Add caching layer for frequently accessed content
- Implement backup and replication strategies
