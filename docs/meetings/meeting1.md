# **AstrID Project Overview** - *Meeting 1*

## Project Goal
Detect astronomical anomalies (supernovae, transients) using ML on time-series imaging data

## Core ML Strategy
- **Primary Method**: U-Net deep learning for image segmentation
- **Backup Methods**: Isolation Forest, One-Class SVM  
- **Image Differencing**: ZOGY algorithm for optimal subtraction
- **Training Data**: Synthetic anomalies + real survey data (SDSS, ZTF)

## Key Tools & Integration

### MLflow Usage
- Experiment tracking for model training runs
- Model registry and version control
- Artifact storage (Cloudflare R2)

### Prefect Usage  
- Automated processing pipelines
- Scheduled model retraining
- System monitoring and health checks

## Data Pipeline
1. FITS ingestion → 2. Preprocessing → 3. Differencing → 4. ML detection → 5. Human validation

## Main Research Questions
- How to establish ground truth for rare astronomical events?
- Optimal preprocessing for anomaly detection accuracy?
- Scalability for LSST-scale data (millions of alerts/night)?
- Balance between automation and human expert validation?
- Model retraining strategy as new data becomes available?

## Development Plan
- **Phase 1**: Infrastructure setup (current)
- **Phase 2**: ML pipeline integration 
- **Phase 3**: Production deployment
