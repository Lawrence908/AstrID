# **AstrID** – *Meeting 4*

## Objective
Focus on training data strategies to help the model distinguish genuine astronomical transients from noise, artifacts, and false positives in difference images.

## Recap
- Goal: Detect astronomical anomalies (transients) from imaging time-series using difference images
- Current challenge: Teaching the model what constitutes a "real" anomaly versus instrumental noise or processing artifacts
- Pipeline status: Image differencing (ZOGY) working; need robust training data for U-Net model

## The Core Challenge: Real vs. False Anomalies

### Problem Statement
When subtracting two telescope images, most differences are **not** real astronomical events:
- **Camera noise**: Random pixel variations, read noise, dark current
- **Alignment artifacts**: Slight misalignments between epochs create spurious differences  
- **Cosmic rays**: High-energy particles hitting the detector
- **Processing artifacts**: Scaling differences, background variations
- **Atmospheric effects**: Seeing variations, transparency changes

The model must learn to distinguish these from genuine transients (supernovae, variable stars, etc.).

## Training Data Strategy

### 1. Synthetic Anomaly Generation
- **Artificial transients**: Inject realistic supernova light curves into difference images
- **Variable star patterns**: Create periodic brightness changes with proper stellar profiles
- **Moving objects**: Generate asteroid/comet trails with appropriate motion
- **Realistic noise**: Use actual survey noise characteristics for synthetic backgrounds

### 2. Real-Bogus Classification Dataset
- **Curated examples**: Expert-labeled difference images from historical surveys
- **False positive catalog**: Document common artifact types (cosmic rays, alignment issues)
- **True positive catalog**: Known transients with confirmed follow-up observations
- **Ambiguous cases**: Borderline examples for model uncertainty training

### 3. Data Augmentation Techniques
- **Noise injection**: Add realistic noise patterns to clean difference images
- **Artifact simulation**: Generate common false positive patterns
- **Multi-epoch training**: Use temporal sequences to improve discrimination
- **Cross-survey validation**: Train on multiple survey characteristics

## Evaluation Metrics (Enhanced)

### Primary Metrics
- **Precision**: Of flagged candidates, what fraction are genuine transients?
- **Recall**: Of real transients present, what fraction do we detect?
- **False positive rate**: Average spurious detections per image (target: <1 per image)
- **Localization accuracy**: IoU overlap with true transient positions

### Advanced Metrics
- **Artifact rejection rate**: How well does the model ignore common false positives?
- **Transient type classification**: Can the model distinguish supernovae from variable stars?
- **Confidence calibration**: Do model confidence scores correlate with actual accuracy?
- **Processing efficiency**: Detection time per image on target hardware

## Implementation Plan

### Phase 1: Data Collection (Week 1-2)
1. **Synthetic data generation**: Create 1000+ difference images with injected transients
2. **Historical data mining**: Curate 500+ real examples from ZTF, Pan-STARRS archives
3. **Artifact cataloging**: Document and label common false positive patterns
4. **Expert validation**: Have astronomers review and label ambiguous cases

### Phase 2: Model Training (Week 3-4)
1. **U-Net architecture**: Implement segmentation model for anomaly detection
2. **Multi-class training**: Train to distinguish transient types and artifacts
3. **Confidence estimation**: Add uncertainty quantification to model outputs
4. **Ensemble methods**: Combine U-Net with traditional ML approaches

### Phase 3: Validation & Iteration (Week 5-6)
1. **Cross-validation**: Test on held-out survey data
2. **Expert review**: Astronomer evaluation of model outputs
3. **Performance analysis**: Detailed metrics on different transient types
4. **Model refinement**: Iterate based on failure cases and expert feedback

## Technical Implementation

### Data Pipeline Enhancements
- **Quality scoring**: Pre-filter difference images by alignment and noise metrics
- **Multi-band training**: Use color information to improve discrimination
- **Temporal context**: Include light curve information when available
- **Survey-specific models**: Adapt to different telescope characteristics

### MLflow Integration
- **Experiment tracking**: Log training runs with synthetic vs. real data ratios
- **Model versioning**: Track performance across different training strategies
- **Artifact storage**: Save training datasets and model weights to R2
- **Performance monitoring**: Track metrics over time as new data arrives

## Risks & Mitigation Strategies

### Data Quality Risks
- **Synthetic bias**: Model may not generalize to real survey conditions
- **Expert labeling**: Inconsistent or incomplete human annotations
- **Class imbalance**: Far more false positives than real transients

### Mitigation Approaches
- **Realistic simulation**: Use actual survey parameters for synthetic data
- **Multiple experts**: Cross-validate human labels with multiple astronomers
- **Balanced sampling**: Stratify training data to ensure adequate positive examples
- **Active learning**: Iteratively improve model with expert feedback

## Success Criteria

### Short-term (2 weeks)
- Training dataset with 1000+ labeled examples (synthetic + real)
- Baseline U-Net model achieving >80% precision on validation set
- False positive rate <1 per image on test data

### Medium-term (1 month)
- Model performance comparable to human expert classification
- Successful detection of known transients in historical data
- Integration with existing pipeline for automated processing

## Requests for Advisor
- **Data access**: Help identifying best sources for historical transient examples
- **Expert collaboration**: Connections to astronomers for data labeling
- **Validation strategy**: Feedback on evaluation metrics and success criteria
- **Scientific priorities**: Guidance on which transient types to prioritize
- **Performance targets**: Realistic expectations for false positive rates

## Next Steps
- Begin synthetic data generation with realistic survey parameters
- Contact survey teams for access to historical difference images
- Set up expert labeling workflow for ground truth creation
- Implement baseline U-Net architecture for anomaly detection


