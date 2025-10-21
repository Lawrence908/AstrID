# **AstrID – Week 4 Plan** - *Meeting 4*

## Objective
Develop training data strategies to teach the model to distinguish genuine astronomical transients from noise and artifacts in difference images.

## Core Challenge
Most differences in subtracted images are **not** real astronomical events (camera noise, alignment artifacts, cosmic rays). The model must learn to distinguish these from genuine transients.

## Training Data Strategy

### 1. Synthetic Anomaly Generation
- **Artificial transients**: Inject realistic supernova light curves into difference images
- **Variable star patterns**: Create periodic brightness changes with proper stellar profiles
- **Moving objects**: Generate asteroid/comet trails with appropriate motion
- **Realistic noise**: Use actual survey noise characteristics for synthetic backgrounds

### 2. Real-Bogus Classification
- **Curated examples**: Expert-labeled difference images from historical surveys
- **False positive catalog**: Document common artifact types (cosmic rays, alignment issues)
- **True positive catalog**: Known transients with confirmed follow-up observations
- **Ambiguous cases**: Borderline examples for model uncertainty training

### 3. Data Augmentation
- **Noise injection**: Add realistic noise patterns to clean difference images
- **Artifact simulation**: Generate common false positive patterns
- **Multi-epoch training**: Use temporal sequences to improve discrimination
- **Cross-survey validation**: Train on multiple survey characteristics

## Evaluation Metrics
- **Precision**: Of flagged candidates, what fraction are genuine transients?
- **Recall**: Of real transients present, what fraction do we detect?
- **False positive rate**: Average spurious detections per image (target: <1 per image)
- **Localization accuracy**: IoU overlap with true transient positions
- **Artifact rejection rate**: How well does the model ignore common false positives?

## Implementation Plan

### Week 1-2: Data Collection
1. **Synthetic data generation**: Create 1000+ difference images with injected transients
2. **Historical data mining**: Curate 500+ real examples from ZTF, Pan-STARRS archives
3. **Artifact cataloging**: Document and label common false positive patterns
4. **Expert validation**: Have astronomers review and label ambiguous cases

### Week 3-4: Model Training
1. **U-Net architecture**: Implement segmentation model for anomaly detection
2. **Multi-class training**: Train to distinguish transient types and artifacts
3. **Confidence estimation**: Add uncertainty quantification to model outputs
4. **Ensemble methods**: Combine U-Net with traditional ML approaches

### Week 5-6: Validation & Iteration
1. **Cross-validation**: Test on held-out survey data
2. **Expert review**: Astronomer evaluation of model outputs
3. **Performance analysis**: Detailed metrics on different transient types
4. **Model refinement**: Iterate based on failure cases and expert feedback

## Success Criteria
- **Short-term (2 weeks)**: Training dataset with 1000+ labeled examples; Baseline U-Net model achieving >80% precision; False positive rate <1 per image
- **Medium-term (1 month)**: Model performance comparable to human expert classification; Successful detection of known transients; Integration with existing pipeline

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
