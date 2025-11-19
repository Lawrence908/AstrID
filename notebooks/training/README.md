# AstrID Model Training Notebook

## ASTR-106: Training Notebook for Model Training and MLflow Logging

This directory contains the comprehensive training notebook and utilities for the U-Net anomaly detection model with full MLflow integration, GPU energy tracking, and performance metrics.

## Features

### ✅ Complete MLflow Integration (ASTR-88)
- Experiment tracking and management
- Model registry and versioning
- Artifact storage with R2 backend
- Comprehensive parameter and metric logging

### ✅ GPU Energy Tracking (ASTR-101)
- Real-time GPU power monitoring
- Carbon footprint calculation
- Energy consumption analysis
- MLflow integration for energy metrics

### ✅ Comprehensive Performance Metrics (ASTR-102)
- Classification metrics (accuracy, precision, recall, F1, AUROC, AUPRC, MCC)
- Calibration metrics (ECE, Brier score)
- Performance metrics (latency, throughput)
- Confusion matrices and ROC/PR curves

### ✅ Data Preprocessing Integration (ASTR-76)
- Astronomical image preprocessing pipeline
- Data augmentation and normalization
- Quality assessment and validation

### ✅ Training Infrastructure
- U-Net model architecture
- Early stopping and checkpointing
- Learning rate scheduling
- Comprehensive visualization tools

## File Structure

```
notebooks/training/
├── model_training.ipynb          # Main training notebook
├── utils/
│   ├── training_utils.py         # Core training utilities
│   ├── performance_metrics.py    # ASTR-102 metrics implementation
│   └── training_manager.py       # Comprehensive training manager
├── config/
│   └── training_config.yaml      # Training configuration templates
└── docs/
    └── training_guide.md         # Detailed training guide
```

## Quick Start

1. **Prerequisites**
   ```bash
   # Ensure MLflow server is running
   docker-compose up mlflow-server
   
   # Install dependencies
   uv sync
   ```

2. **Run Training**
   ```bash
   # Start Jupyter notebook
   jupyter notebook notebooks/training/model_training.ipynb
   
   # Or run programmatically
   python -m notebooks.training.model_training
   ```

3. **Monitor Progress**
   - MLflow UI: http://localhost:9003
   - Training logs: Available in MLflow UI
   - Model checkpoints: `checkpoints/` directory

## Configuration

### Training Parameters
- **Model Architecture**: U-Net with configurable depth and filters
- **Training**: Adam optimizer with learning rate scheduling
- **Data**: Configurable batch size, validation split, augmentation
- **Energy Tracking**: GPU power monitoring with carbon footprint calculation

### MLflow Integration
- **Experiment Tracking**: Automatic experiment creation and run management
- **Model Registry**: Model versioning and stage management
- **Artifacts**: Comprehensive artifact logging (metrics, curves, models)

### Performance Metrics
- **Classification**: Accuracy, precision, recall, F1 (macro/micro/weighted)
- **Advanced**: AUROC, AUPRC, MCC, balanced accuracy
- **Calibration**: ECE, Brier score
- **Performance**: Latency (P50, P95), throughput

## Usage Examples

### Basic Training
```python
from notebooks.training.utils.training_manager import TrainingManager

# Initialize training manager
training_manager = TrainingManager(
    config=config,
    experiment_tracker=experiment_tracker,
    model_registry=model_registry,
    mlflow_client=mlflow_client,
    gpu_monitor=gpu_monitor,
    energy_tracker=energy_tracker
)

# Start training
run_id = await training_manager.start_training_run(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler
)
```

### Custom Metrics
```python
from notebooks.training.utils.performance_metrics import ComprehensiveMetricsCalculator

# Calculate comprehensive metrics
metrics_calculator = ComprehensiveMetricsCalculator()
metrics = metrics_calculator.calculate_all_metrics(
    y_true, y_pred, y_scores, inference_times, batch_size
)
```

### Energy Analysis
```python
from src.core.energy_analysis import EnergyAnalyzer

# Analyze energy consumption
energy_analyzer = EnergyAnalyzer()
analysis = energy_analyzer.analyze_energy_consumption(energy_data)
```

## Troubleshooting

### Common Issues

1. **MLflow Connection Error**
   - Ensure MLflow server is running: `docker-compose up mlflow-server`
   - Check tracking URI: `http://localhost:9003`

2. **GPU Energy Tracking Issues**
   - Install nvidia-smi: `nvidia-smi`
   - Check GPU availability: `torch.cuda.is_available()`

3. **Memory Issues**
   - Reduce batch size in configuration
   - Enable gradient checkpointing
   - Use mixed precision training

4. **Training Convergence**
   - Adjust learning rate
   - Modify early stopping patience
   - Check data quality and preprocessing

### Debug Tools

The notebook includes comprehensive debugging tools:
- Model prediction inspection
- Training issue analysis
- Performance bottleneck identification
- Data quality assessment

## Integration Points

### ASTR-88 (MLflow Integration)
- Uses existing MLflow infrastructure
- Leverages experiment tracking and model registry
- Integrates with R2 artifact storage

### ASTR-101 (GPU Energy Tracking)
- Real-time power monitoring during training
- Carbon footprint calculation and reporting
- MLflow integration for energy metrics

### ASTR-102 (Performance Metrics)
- Comprehensive metrics calculation
- Database integration for metric storage
- MLflow artifact logging

### ASTR-76 (Preprocessing)
- Astronomical image preprocessing pipeline
- Quality assessment and validation
- Data augmentation strategies

## Performance Considerations

### Memory Optimization
- Gradient checkpointing for large models
- Mixed precision training
- Efficient data loading with multiple workers

### GPU Utilization
- Automatic GPU detection and usage
- Energy monitoring and optimization
- Batch size optimization

### Training Efficiency
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing for recovery

## Monitoring and Alerting

### Training Monitoring
- Real-time loss and metric tracking
- GPU utilization monitoring
- Energy consumption tracking

### MLflow Integration
- Automatic experiment logging
- Model versioning and registry
- Artifact storage and retrieval

### Performance Tracking
- Latency and throughput monitoring
- Memory usage tracking
- Error rate monitoring

## Best Practices

### Model Training
1. Start with small datasets for experimentation
2. Use validation data for hyperparameter tuning
3. Monitor for overfitting with early stopping
4. Save model checkpoints regularly

### Energy Efficiency
1. Monitor GPU power consumption
2. Optimize batch sizes for efficiency
3. Use appropriate learning rates
4. Consider model compression techniques

### Experiment Management
1. Use descriptive experiment names
2. Tag runs with relevant metadata
3. Document hyperparameter choices
4. Compare results across experiments

## Contributing

When extending the training notebook:

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include unit tests for new utilities
4. Update documentation
5. Ensure MLflow integration compatibility

## License

This training notebook is part of the AstrID project and follows the same licensing terms.
