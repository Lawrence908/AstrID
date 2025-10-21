# AstrID Training Data Pipeline

## Overview

This document outlines the complete data pipeline for training the U-Net anomaly detection model, from raw astronomical observations to curated training datasets. The pipeline integrates with the existing AstrID workflow to provide real, validated data for model training.

## Quick Runbook: Preparing Real Data for Training

Follow these steps to populate real detections and create a training dataset consumable by the notebooks `notebooks/ml_training_data/real_data_loading.ipynb` and `notebooks/training/model_training.ipynb`.

### 0) Prerequisites
- Services running: api, worker, prefect, mlflow, redis
- Database migrated (includes training tables)
- Environment variables set for Supabase and R2

Verify health:
- GET http://127.0.0.1:8000/health
- MLflow UI: http://localhost:5000
- Prefect UI: http://localhost:4200

### 1) Ingest observations
Goal: create `observations` rows and store raw FITS in R2.

Options:
- Prefect flow (recommended): trigger observation ingestion in UI or via your existing endpoint/CLI.
- API/Worker (if you have direct endpoints): use your ingestion route to pull a small time window.

Confirm:
- Query DB table `observations` has new rows with recent `observation_time`.
- R2 shows raw FITS for those observations.

### 2) Preprocess images
Goal: create `PreprocessRun` rows and processed FITS in R2.

- Trigger preprocessing (Prefect/worker) for the ingested observations.

Confirm:
- DB contains `preprocess_runs` with status completed.
- R2 has processed FITS paths referenced by the runs.

### 3) Difference and extract sources
Goal: create `DifferenceRun` and `Candidate` rows; store difference images in R2.

- Trigger differencing (Prefect/worker). Ensure reference selection is configured.

Confirm:
- DB has `difference_runs` completed, and `candidates` present.
- R2 has difference FITS referenced by runs.

### 4) Run ML detection
Goal: populate `detections` with confidence scores; optionally mark some validated.

- Trigger detection (Prefect/worker). This runs the U-Net inference over difference images.

Confirm:
- DB contains `detections` with `confidence_score > 0` and recent timestamps.
- Optional: set `is_validated=true` on a subset (via your validation UI/flow) to test the validated-only path.

### 5) Create training dataset (API)
Goal: collect detections in a time window into a `TrainingDataset` + `TrainingSample`s and log minimal stats to MLflow.

Endpoint:
- POST http://127.0.0.1:8000/training/datasets/collect

Body example:
```json
{
  "survey_ids": ["hst"],
  "start": "2024-01-01T00:00:00",
  "end": "2024-12-31T23:59:59",
  "confidence_threshold": 0.7,
  "max_samples": 200,
  "name": "hst_2024_training"
}
```

Behavior:
- Collector pulls detections joined to observations in the time range.
- Prefers validated detections; if none, falls back to high-confidence non-validated.
- Creates one `training_datasets` row and associated `training_samples` rows.
- Logs dataset metrics to MLflow (best-effort).

Confirm:
- GET http://127.0.0.1:8000/training/datasets → returns newly created dataset.
- GET http://127.0.0.1:8000/training/datasets/{dataset_id} → shows counts.
- MLflow run named `training_data_{dataset_id}` appears in UI.

Tips if `total_samples` is 0:
- Expand the date range.
- Lower `confidence_threshold` (e.g., 0.5) to pick up more detections.
- Ensure detection pipeline actually ran for the window.

### 6) Load in notebook (smoke test)
Notebook: `notebooks/ml_training_data/real_data_loading.ipynb`

Steps:
1. Ensure `API_BASE = "http://127.0.0.1:8000"` (no `/v1`).
2. Run the cell that POSTs `/training/datasets/collect` (or copy the returned `dataset_id`).
3. Run the list/detail cells to verify the dataset.

Outcome: you get a `dataset_id` and basic counts. This verifies the pipeline up to dataset creation.

### 7) Train with real data
Notebook: `notebooks/training/model_training.ipynb`

Integrate dataset_id:
- Add a small helper cell that fetches samples for `dataset_id` using an API route (or extend the API to provide sample listings) and replaces the synthetic data loader with real paths + masks.

Checklist:
- Confirm GPU/MLflow settings.
- Start training; monitor in MLflow.

---
## Pipeline Architecture

### 1. Data Sources and Ingestion

#### External Survey APIs
- **MAST (Mikulski Archive for Space Telescopes)**
- **SkyView Virtual Observatory**
- **Other survey databases** (SDSS, Pan-STARRS, etc.)

#### Observation Ingestion Flow
```
External Survey APIs → MAST Client → R2 Storage → PostgreSQL Database
```

**Key Components:**
- `Survey` entities containing metadata
- `Observation` records with FITS file references
- Raw FITS files stored in Cloudflare R2
- Database records with status tracking

### 2. Preprocessing Pipeline

#### Calibration and Alignment
```
Raw FITS → Calibration Engine → WCS Alignment → Image Registration → Processed FITS
```

**Process Steps:**
1. **Bias/Dark/Flat Correction** (if calibration frames available)
2. **WCS (World Coordinate System) Alignment**
3. **Astrometric Image Registration**
4. **Quality Assessment and Validation**

**Key Components:**
- `PreprocessRun` entities tracking processing status
- `CalibrationEngine` for bias/dark/flat correction
- `PreprocessingService` orchestrating the pipeline
- Processed images stored in R2 with metadata

### 3. Image Differencing Pipeline

#### Reference Selection and ZOGY Algorithm
```
Processed Image → Reference Selection → ZOGY Algorithm → Difference Image → Source Extraction
```

**Process Steps:**
1. **Reference Image Selection** (historical survey data or SkyView)
2. **ZOGY Algorithm Application** for optimal image differencing
3. **Source Extraction** using SEP/photutils
4. **Candidate Filtering** based on quality thresholds

**Key Components:**
- `DifferenceRun` entities
- `ZOGYAlgorithm` implementation
- `DifferencingService` orchestration
- `Candidate` entities for detected sources

### 4. Machine Learning Detection Pipeline

#### U-Net Model Training Data Preparation
```
Difference Images → U-Net Inference → Detection Scoring → Human Validation → Training Labels
```

**Process Steps:**
1. **U-Net Model Loading** (current version)
2. **Inference on Difference Images**
3. **Confidence Scoring** of detections
4. **Human Validation** and labeling
5. **Training Data Curation**

**Key Components:**
- `Model` and `ModelRun` entities
- `Detection` entities with confidence scores
- `ValidationEvent` entities for human review
- Labeled training datasets

## Training Data Flow Integration

### 1. Data Collection Phase

#### Automated Data Harvesting
```python
# Pseudo-code for training data collection
def collect_training_data():
    # Get validated detections from the last 6 months
    validated_detections = get_validated_detections(
        status='validated',
        date_range=last_6_months,
        confidence_threshold=0.7
    )
    
    # For each validated detection, collect:
    for detection in validated_detections:
        # Original observation
        observation = get_observation(detection.observation_id)
        raw_fits = download_from_r2(observation.raw_fits_path)
        
        # Processed image
        preprocess_run = get_preprocess_run(observation.id)
        processed_fits = download_from_r2(preprocess_run.processed_fits_path)
        
        # Difference image
        difference_run = get_difference_run(observation.id)
        difference_fits = download_from_r2(difference_run.difference_fits_path)
        
        # Reference image
        reference_fits = download_from_r2(difference_run.reference_fits_path)
        
        # Human validation labels
        validation = get_validation_event(detection.id)
        labels = {
            'is_anomaly': validation.is_valid,
            'anomaly_type': validation.anomaly_type,
            'confidence': validation.confidence,
            'reviewer_notes': validation.notes
        }
        
        yield TrainingSample(
            raw_image=raw_fits,
            processed_image=processed_fits,
            difference_image=difference_fits,
            reference_image=reference_fits,
            labels=labels,
            metadata=detection.metadata
        )
```

### 2. Data Preprocessing for Training

#### Astronomical Image Processing Integration
```python
# Integration with existing preprocessing service
from src.domains.preprocessing.processors.astronomical_image_processing import AstronomicalImageProcessor

def prepare_training_data(samples):
    processor = AstronomicalImageProcessor()
    
    processed_samples = []
    for sample in samples:
        # Apply consistent preprocessing
        enhanced_image = processor.enhance_astronomical_image(
            sample.difference_image,
            calibration_params=sample.calibration_params
        )
        
        # Create training mask from validation labels
        mask = create_training_mask(
            sample.labels,
            image_shape=enhanced_image.shape
        )
        
        processed_samples.append(TrainingSample(
            image=enhanced_image,
            mask=mask,
            labels=sample.labels,
            metadata=sample.metadata
        ))
    
    return processed_samples
```

### 3. Training Dataset Creation

#### PyTorch Dataset Implementation
```python
class AstrIDTrainingDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = load_fits_image(sample.image_path)
        mask = load_fits_image(sample.mask_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return {
            'image': image,
            'mask': mask,
            'labels': sample.labels,
            'metadata': sample.metadata
        }
```

## Integration with Existing Workflow

### 1. Dramatiq Worker Integration

#### Training Data Collection Worker
```python
@dramatiq.actor(queue_name='training_data')
def collect_training_data_worker(collection_params):
    """Worker to collect and prepare training data from validated detections."""
    
    # Get validated detections
    detections = get_validated_detections(collection_params)
    
    # Collect associated data
    training_samples = []
    for detection in detections:
        sample = collect_detection_data(detection)
        training_samples.append(sample)
    
    # Store training dataset
    dataset_id = store_training_dataset(training_samples)
    
    # Trigger training pipeline
    trigger_training_pipeline.send(dataset_id)
```

### 2. MLflow Integration

#### Experiment Tracking for Training Data
```python
def log_training_dataset_info(dataset_id, samples):
    """Log training dataset information to MLflow."""
    
    with mlflow.start_run(run_name=f"training_data_{dataset_id}"):
        # Log dataset statistics
        mlflow.log_metric("total_samples", len(samples))
        mlflow.log_metric("anomaly_ratio", calculate_anomaly_ratio(samples))
        mlflow.log_metric("data_quality_score", calculate_quality_score(samples))
        
        # Log data distribution
        log_data_distribution(samples)
        
        # Log sample images
        log_sample_images(samples, max_samples=10)
        
        # Log dataset metadata
        mlflow.log_params({
            "collection_date": datetime.now().isoformat(),
            "validation_threshold": 0.7,
            "time_range_days": 180,
            "survey_sources": get_survey_sources(samples)
        })
```

### 3. Database Schema Extensions

#### Training Dataset Storage
```sql
-- Training dataset storage
CREATE TABLE training_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    collection_params JSONB NOT NULL,
    total_samples INTEGER NOT NULL,
    anomaly_ratio FLOAT NOT NULL,
    quality_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'active'
);

-- Training samples linking
CREATE TABLE training_samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES training_datasets(id),
    detection_id UUID REFERENCES detections(id),
    image_path VARCHAR(500) NOT NULL,
    mask_path VARCHAR(500) NOT NULL,
    labels JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Training run tracking
CREATE TABLE training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES training_datasets(id),
    model_id UUID REFERENCES models(id),
    mlflow_run_id VARCHAR(255),
    training_params JSONB NOT NULL,
    performance_metrics JSONB,
    status VARCHAR(50) DEFAULT 'running',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);
```

## Training Pipeline Orchestration

### 1. Prefect Flow Integration

#### Training Data Collection Flow
```python
from prefect import flow, task

@task
def collect_validated_detections(collection_params):
    """Collect validated detections for training."""
    return get_validated_detections(collection_params)

@task
def prepare_training_samples(detections):
    """Prepare training samples from detections."""
    return [prepare_training_sample(d) for d in detections]

@task
def create_training_dataset(samples, dataset_name):
    """Create and store training dataset."""
    return store_training_dataset(samples, dataset_name)

@flow
def training_data_collection_flow(collection_params):
    """Orchestrate training data collection."""
    
    # Collect detections
    detections = collect_validated_detections(collection_params)
    
    # Prepare samples
    samples = prepare_training_samples(detections)
    
    # Create dataset
    dataset_id = create_training_dataset(samples, collection_params['name'])
    
    # Trigger training
    trigger_training_pipeline.send(dataset_id)
    
    return dataset_id
```

### 2. Training Pipeline Integration

#### Updated Training Notebook
```python
# In the training notebook, replace synthetic data with real data
def load_training_data(dataset_id):
    """Load real training data from the pipeline."""
    
    # Get dataset from database
    dataset = get_training_dataset(dataset_id)
    
    # Load samples
    samples = get_training_samples(dataset_id)
    
    # Create PyTorch datasets
    train_dataset = AstrIDTrainingDataset(
        samples['train'],
        transform=train_transform
    )
    val_dataset = AstrIDTrainingDataset(
        samples['val'],
        transform=val_transform
    )
    test_dataset = AstrIDTrainingDataset(
        samples['test'],
        transform=val_transform
    )
    
    return train_dataset, val_dataset, test_dataset
```

## Quality Assurance and Validation

### 1. Data Quality Metrics

#### Training Data Validation
```python
def validate_training_data(samples):
    """Validate training data quality."""
    
    metrics = {
        'total_samples': len(samples),
        'anomaly_ratio': calculate_anomaly_ratio(samples),
        'image_quality_score': calculate_image_quality(samples),
        'label_consistency': calculate_label_consistency(samples),
        'temporal_distribution': analyze_temporal_distribution(samples),
        'survey_coverage': analyze_survey_coverage(samples)
    }
    
    # Quality thresholds
    if metrics['anomaly_ratio'] < 0.1 or metrics['anomaly_ratio'] > 0.9:
        raise ValueError("Anomaly ratio out of acceptable range")
    
    if metrics['image_quality_score'] < 0.7:
        raise ValueError("Image quality below threshold")
    
    return metrics
```

### 2. Continuous Data Monitoring

#### Training Data Health Checks
```python
@task
def monitor_training_data_health():
    """Monitor training data health and quality."""
    
    # Check for new validated detections
    new_detections = get_recent_validated_detections(days=7)
    
    # Assess data quality
    quality_metrics = validate_training_data(new_detections)
    
    # Log to MLflow
    log_data_health_metrics(quality_metrics)
    
    # Alert if quality drops
    if quality_metrics['image_quality_score'] < 0.8:
        send_quality_alert(quality_metrics)
```

## Implementation Roadmap

### Phase 1: Data Collection Infrastructure
1. **Database Schema Updates** - Add training dataset tables
2. **Data Collection Worker** - Dramatiq worker for data harvesting
3. **Quality Validation** - Data quality assessment tools

### Phase 2: Training Pipeline Integration
1. **Prefect Flow Updates** - Integrate training data collection
2. **MLflow Integration** - Enhanced experiment tracking
3. **Training Notebook Updates** - Replace synthetic data with real data

### Phase 3: Monitoring and Optimization
1. **Data Health Monitoring** - Continuous quality assessment
2. **Performance Optimization** - Efficient data loading and processing
3. **Automated Retraining** - Trigger retraining on new data

## Benefits of Real Data Integration

1. **Authentic Training Data** - Real astronomical observations and validated labels
2. **Consistent Preprocessing** - Same pipeline used for inference and training
3. **Quality Assurance** - Human-validated labels ensure accuracy
4. **Continuous Learning** - New data automatically incorporated
5. **Reproducibility** - Full traceability from observation to training sample
6. **Performance Monitoring** - Real-world performance metrics

## Conclusion

This pipeline ensures that the U-Net model is trained on real, validated astronomical data that flows through the complete AstrID workflow. The integration maintains data consistency, quality, and traceability while providing a robust foundation for continuous model improvement.
