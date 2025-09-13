# AstrID - Consolidated Database Models

This file contains all SQLAlchemy database models from the AstrID astronomical identification system, consolidated from the domain-driven design structure.

## Original Structure

- `src/domains/observations/models.py`
- `src/domains/preprocessing/models.py`
- `src/domains/differencing/models.py`
- `src/domains/detection/models.py`
- `src/domains/curation/models.py`
- `src/domains/catalog/models.py`

> **Note:** This consolidated version is for reference and discussion purposes only.

## Database Schema Overview

The AstrID system uses a domain-driven design approach with 6 main domains:

1. **Observations** - Raw astronomical data ingestion
2. **Preprocessing** - Image calibration and preparation
3. **Differencing** - Image subtraction and candidate detection
4. **Detection** - ML-based anomaly detection
5. **Curation** - Human validation and alerting
6. **Catalog** - System management and workflow tracking

---

## Observations Domain

### Models: 2

#### Survey
**Purpose:** Survey information and metadata

**Key Fields:**
- `name` - Survey identifier
- `description` - Survey description
- `base_url` - Data source URL
- `api_endpoint` - API access point
- `is_active` - Survey status

#### Observation
**Purpose:** Individual astronomical observations

**Key Fields:**
- `observation_id` - External observation identifier
- `ra`, `dec` - Right ascension and declination
- `observation_time` - When the observation was taken
- `filter_band` - Photometric filter used
- `exposure_time` - Exposure duration
- `fits_url` - Link to raw FITS file
- `pixel_scale` - Arcseconds per pixel
- `airmass`, `seeing` - Atmospheric conditions
- `status` - Processing status

**Constraints:**
- RA: 0° to 360°
- Dec: -90° to +90°
- Exposure time > 0
- Airmass > 0
- Seeing > 0

---

## Preprocessing Domain

### Models: 1

#### PreprocessRun
**Purpose:** Image preprocessing workflow tracking

**Key Fields:**
- `observation_id` - Reference to observation
- `status` - Processing status (pending, running, completed, failed)
- `calibration_applied` - Whether calibration was applied
- `wcs_aligned` - Whether WCS alignment was performed
- `registration_quality` - Quality metric for alignment
- `processed_fits_path` - Path to processed image
- `wcs_info` - World Coordinate System metadata
- `processing_time_seconds` - Duration of processing
- `error_message` - Error details if failed
- `retry_count` - Number of retry attempts

---

## Differencing Domain

### Models: 2

#### DifferenceRun
**Purpose:** Image differencing operations

**Key Fields:**
- `observation_id` - Target observation
- `reference_observation_id` - Reference image
- `algorithm` - Differencing algorithm used (default: "zogy")
- `parameters` - Algorithm-specific parameters
- `difference_image_path` - Path to difference image
- `significance_map_path` - Path to significance map
- `noise_level` - Measured noise level
- `detection_threshold` - Threshold used for detection
- `candidates_found` - Number of candidates detected

#### Candidate
**Purpose:** Potential transients from differencing

**Key Fields:**
- `difference_run_id` - Parent differencing run
- `observation_id` - Source observation
- `ra`, `dec` - Sky coordinates
- `pixel_x`, `pixel_y` - Pixel coordinates
- `flux`, `flux_error` - Photometric measurements
- `significance` - Statistical significance
- `snr` - Signal-to-noise ratio
- `fwhm` - Full width at half maximum
- `ellipticity` - Source ellipticity
- `position_angle` - Orientation angle
- `is_saturated`, `is_blended`, `is_edge` - Quality flags
- `status` - Processing status

---

## Detection Domain

### Models: 3

#### Model
**Purpose:** ML model registry and metadata

**Key Fields:**
- `name`, `version` - Model identification
- `model_type` - Type of model (unet, anomaly_detector, classifier, segmentation)
- `architecture` - Model architecture details
- `hyperparameters` - Training hyperparameters
- `training_dataset` - Dataset used for training
- `training_metrics` - Training performance metrics
- `precision`, `recall`, `f1_score`, `accuracy` - Performance metrics
- `is_active` - Whether model is currently deployed
- `deployment_date` - When model was deployed
- `model_path` - Path to model file
- `mlflow_run_id` - MLflow tracking ID

#### ModelRun
**Purpose:** Model inference execution tracking

**Key Fields:**
- `model_id` - Reference to model
- `observation_id` - Input observation
- `input_image_path` - Path to input image
- `output_mask_path` - Path to output segmentation mask
- `confidence_map_path` - Path to confidence map
- `inference_time_ms` - Inference duration
- `memory_usage_mb` - Memory consumption
- `total_predictions` - Number of predictions made
- `high_confidence_predictions` - High-confidence predictions count
- `status` - Run status

#### Detection
**Purpose:** ML-identified anomalies

**Key Fields:**
- `model_run_id` - Parent model run
- `observation_id` - Source observation
- `ra`, `dec` - Sky coordinates
- `pixel_x`, `pixel_y` - Pixel coordinates
- `confidence_score` - ML confidence (0-1)
- `detection_type` - Type of detection (supernova, variable, transient, unknown, artifact)
- `model_version` - Model version used
- `inference_time_ms` - Time for this specific detection
- `prediction_metadata` - Additional ML metadata
- `status` - Detection status (pending, validating, validated, rejected, confirmed, archived)
- `is_validated` - Whether human-validated
- `validation_confidence` - Human validation confidence
- `human_label` - Human-assigned label

**Constraints:**
- `confidence_score` between 0 and 1

---

## Curation Domain

### Models: 2

#### ValidationEvent
**Purpose:** Human validation of detections

**Key Fields:**
- `detection_id` - Reference to detection
- `validator_id` - Human validator identifier
- `is_valid` - Whether detection is valid
- `label` - Human-assigned label
- `confidence_level` - Validator confidence (low, medium, high, expert)
- `notes` - Additional notes
- `tags` - Categorization tags
- `image_quality` - Quality assessment of image
- `detection_quality` - Quality assessment of detection

#### Alert
**Purpose:** Notifications and alerts

**Key Fields:**
- `detection_id` - Source detection
- `alert_type` - Type of alert
- `priority` - Alert priority (low, medium, high, critical)
- `title` - Alert title
- `message` - Alert message content
- `alert_metadata` - Additional alert data
- `status` - Delivery status (pending, sent, delivered, failed, cancelled)
- `sent_at` - When alert was sent
- `delivery_attempts` - Number of delivery attempts
- `error_message` - Delivery error details

---

## Catalog Domain (System Tables)

### Models: 3

#### SystemConfig
**Purpose:** System configuration storage

**Key Fields:**
- `key` - Configuration key (unique)
- `value` - Configuration value (JSON)
- `description` - Configuration description
- `is_active` - Whether configuration is active

#### ProcessingJob
**Purpose:** Workflow and job tracking

**Key Fields:**
- `job_type` - Type of job
- `entity_id` - ID of entity being processed
- `entity_type` - Type of entity
- `status` - Job status (pending, running, completed, failed, cancelled)
- `priority` - Job priority
- `workflow_id` - Workflow identifier
- `task_id` - Task identifier
- `retry_count` - Number of retries
- `max_retries` - Maximum retry limit
- `scheduled_at` - When job was scheduled
- `started_at` - When job started
- `completed_at` - When job completed
- `error_message` - Error details
- `error_details` - Additional error information

#### AuditLog
**Purpose:** System audit trail

**Key Fields:**
- `entity_type` - Type of entity modified
- `entity_id` - ID of entity modified
- `action` - Action performed
- `user_id` - User who performed action
- `old_values` - Previous values (JSON)
- `new_values` - New values (JSON)
- `change_summary` - Human-readable summary
- `ip_address` - Source IP address
- `user_agent` - Client user agent

---

## Database Summary

### Model Count by Domain
- **Observations:** 2 models
- **Preprocessing:** 1 model
- **Differencing:** 2 models
- **Detection:** 3 models
- **Curation:** 2 models
- **Catalog:** 3 models

**Total:** 13 models across 6 domains

### Key Features
- ✅ UUID primary keys for all models
- ✅ SQLAlchemy 2.0+ with Mapped type annotations
- ✅ Python enums with SQLAlchemy Enum integration
- ✅ JSONB fields for flexible metadata storage
- ✅ Proper relationships and foreign key constraints
- ✅ Spatial indexing on RA/Dec coordinates
- ✅ Comprehensive audit and workflow tracking
- ✅ ML model registry and inference tracking
- ✅ Human validation and curation workflows

### Data Flow
```
Survey → Observation → PreprocessRun → DifferenceRun → Candidate → ModelRun → Detection → ValidationEvent/Alert
```

This represents a complete astronomical transient detection pipeline from raw observations through ML-based detection to human validation and alerting.

---

## Technical Implementation

### Database Technology
- **Database:** PostgreSQL
- **ORM:** SQLAlchemy 2.0+
- **Primary Keys:** UUID4
- **Spatial Data:** RA/Dec coordinates with spatial indexing
- **Flexible Data:** JSONB for metadata and configuration
- **Enums:** PostgreSQL native enums for status fields

### Indexing Strategy
- Spatial indexes on RA/Dec coordinates for fast sky queries
- Status indexes for workflow filtering
- Foreign key indexes for relationship queries
- Composite indexes for common query patterns

### Constraints
- Coordinate bounds validation (RA: 0-360°, Dec: -90° to +90°)
- Positive value constraints (exposure time, airmass, seeing)
- Confidence score bounds (0-1)
- Unique constraints on critical fields

This schema supports the complete lifecycle of astronomical transient detection, from raw data ingestion through machine learning inference to human validation and alert generation.
EOF
