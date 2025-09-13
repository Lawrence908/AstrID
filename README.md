# AstrID: Astronomical Identification System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Astronomical Identification: Temporal Dataset Preparation and Anomaly Detection**

AstrID is a comprehensive system for processing astronomical observations, detecting anomalies, and managing temporal datasets for astronomical research. Built with modern Python technologies and designed for scalability and scientific rigor.

## Features

- **Temporal Dataset Preparation**: Robust pipelines for processing time-series astronomical observations
- **Anomaly Detection**: U-Net based machine learning models for detecting transient events
- **Real-time Processing**: FastAPI-based API with Server-Sent Events for live detection streaming
- **Scalable Architecture**: Event-driven architecture with Redis, PostgreSQL, and Cloudflare R2
- **ML Experiment Tracking**: MLflow integration for model versioning and experiment management
- **Production Ready**: Docker containerization with health checks and monitoring

## Architecture

AstrID follows Domain-Driven Design (DDD) principles with a clean separation of concerns:

```
src/
├── domains/           # Pure business logic
│   ├── observations/  # Observation management
│   ├── preprocessing/ # Image preprocessing
│   ├── differencing/  # Image differencing
│   ├── detection/     # Anomaly detection
│   ├── curation/      # Human validation
│   └── catalog/       # Data cataloging
├── adapters/          # Framework integrations
│   ├── api/           # FastAPI application
│   ├── db/            # Database layer
│   ├── storage/       # Cloud storage
│   ├── ml/            # ML model interfaces
│   ├── imaging/       # Astronomical image processing
│   └── scheduler/     # Workflow orchestration
└── utils/             # Shared utilities
```

## Tech Stack

- **Language**: Python 3.11+
- **Web Framework**: FastAPI with async support
- **Database**: PostgreSQL with SQLAlchemy 2 + Alembic
- **Message Queue**: Redis + Dramatiq
- **ML Framework**: TensorFlow 2.13 + Keras
- **Storage**: Cloudflare R2 (S3-compatible)
- **Orchestration**: Prefect for workflows
- **Containerization**: Docker + Docker Compose
- **Quality**: Ruff, MyPy, Black, pre-commit

## Data Pipeline

AstrID implements a comprehensive data processing pipeline:

1. **Ingestion**: Fetch observations from astronomical surveys (MAST, SkyView)
2. **Preprocessing**: Calibration, registration, and WCS alignment
3. **Differencing**: ZOGY algorithm and source extraction
4. **Inference**: U-Net model for anomaly detection
5. **Validation**: Human review and curation
6. **Cataloging**: Persistent storage and analytics

## Machine Learning

### U-Net Model

The system uses a U-Net architecture for astronomical image segmentation:

- **Input**: Astronomical images (FITS format)
- **Output**: Segmentation masks identifying potential anomalies
- **Training**: Supervised learning with pixel-level annotations
- **Inference**: Real-time processing with configurable confidence thresholds

### Model Management

- **Versioning**: MLflow for experiment tracking and model registry
- **Artifacts**: DVC for dataset and model versioning
- **Deployment**: Automated model promotion based on performance metrics

## API Overview

AstrID provides a comprehensive REST API for:

- **Observations**: Management and synchronization of astronomical observations
- **Detections**: Anomaly detection inference and validation
- **Streaming**: Real-time detection streaming via Server-Sent Events
- **Catalog**: Data cataloging and retrieval

## Monitoring & Observability

- **Health Checks**: Built-in health endpoints for all services
- **Logging**: Structured logging with structlog
- **Metrics**: Prometheus metrics (planned)
- **Tracing**: Distributed tracing (planned)


## Documentation

For detailed documentation, development guides, and deployment information, please visit our documentation site or check the project wiki.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- **Email**: chris@chrislawrence.ca

---

**AstrID** - Advancing astronomical discovery through intelligent data processing and machine learning.
