"""Training manager for comprehensive model training with MLflow and energy tracking."""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from src.core.gpu_monitoring import GPUPowerMonitor, EnergyConsumption
from src.core.mlflow_energy import MLflowEnergyTracker
from src.core.energy_analysis import EnergyAnalyzer
from src.infrastructure.mlflow import ExperimentTracker, ModelRegistry
from performance_metrics import ComprehensiveMetricsCalculator, MLflowMetricsLogger
from training_utils import TrainingMetrics, ModelCheckpoint, TrainingVisualizer

logger = logging.getLogger(__name__)


class TrainingManager:
    """Comprehensive training manager with MLflow and energy tracking."""
    
    def __init__(self, 
                 config: Any,
                 experiment_tracker: ExperimentTracker,
                 model_registry: ModelRegistry,
                 mlflow_client: MlflowClient,
                 gpu_monitor: Optional[GPUPowerMonitor] = None,
                 energy_tracker: Optional[MLflowEnergyTracker] = None):
        
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.model_registry = model_registry
        self.mlflow_client = mlflow_client
        self.gpu_monitor = gpu_monitor
        self.energy_tracker = energy_tracker
        
        # Initialize components
        self.metrics_calculator = ComprehensiveMetricsCalculator()
        self.checkpoint_manager = ModelCheckpoint()
        self.visualizer = TrainingVisualizer()
        
        # Training state
        self.current_run_id = None
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    async def start_training_run(self, 
                               model: nn.Module,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               optimizer: optim.Optimizer,
                               criterion: nn.Module,
                               scheduler: Optional[Any] = None) -> str:
        """Start a comprehensive training run with full tracking."""
        
        # Start MLflow run
        self.current_run_id = self.experiment_tracker.start_run(
            experiment_id=self.config.experiment_id,
            run_name=self.config.run_name,
            tags=self.config.tags
        )
        
        # Log hyperparameters
        hyperparams = {
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'num_epochs': self.config.num_epochs,
            'weight_decay': self.config.weight_decay,
            'gradient_clip_norm': self.config.gradient_clip_norm,
            'model_architecture': 'unet',
            'input_size': str(self.config.input_size),
            'initial_filters': self.config.initial_filters,
            'depth': self.config.depth
        }
        
        self.experiment_tracker.log_parameters(hyperparams, self.current_run_id)
        
        # Initialize metrics logger
        metrics_logger = MLflowMetricsLogger(self.mlflow_client, self.current_run_id)
        
        # Start GPU monitoring if enabled
        if self.gpu_monitor and self.config.enable_energy_tracking:
            await self.gpu_monitor.start_monitoring()
            logger.info("🔋 GPU energy monitoring started")
        
        # Training loop
        model = model.to(self.device)
        start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                
                # Training phase
                train_metrics = await self._train_epoch(
                    model, train_loader, optimizer, criterion, epoch
                )
                
                # Validation phase
                val_metrics = await self._validate_epoch(
                    model, val_loader, criterion, epoch
                )
                
                # Update learning rate
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_metrics['loss'])
                    else:
                        scheduler.step()
                
                # Update training history
                self.training_history['train_losses'].append(train_metrics['loss'])
                self.training_history['val_losses'].append(val_metrics['loss'])
                self.training_history['train_metrics'].append(train_metrics)
                self.training_history['val_metrics'].append(val_metrics)
                self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                # Log metrics to MLflow
                metrics_logger.log_scalar_metrics(train_metrics, step=epoch)
                metrics_logger.log_scalar_metrics(val_metrics, step=epoch)
                metrics_logger.log_scalar_metrics({
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                # Check for early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    best_metrics = TrainingMetrics(
                        accuracy=val_metrics.get('accuracy', 0.0),
                        precision=val_metrics.get('precision_macro', 0.0),
                        recall=val_metrics.get('recall_macro', 0.0),
                        f1_score=val_metrics.get('f1_macro', 0.0),
                        auroc=val_metrics.get('auroc', 0.0),
                        auprc=val_metrics.get('auprc', 0.0),
                        mcc=val_metrics.get('mcc', 0.0),
                        balanced_accuracy=val_metrics.get('balanced_accuracy', 0.0),
                        ece=val_metrics.get('ece', 0.0),
                        brier_score=val_metrics.get('brier_score', 0.0),
                        train_loss=train_metrics['loss'],
                        val_loss=val_metrics['loss'],
                        learning_rate=optimizer.param_groups[0]['lr'],
                        epoch=epoch
                    )
                    
                    self.checkpoint_manager.save_checkpoint(
                        model, optimizer, epoch, best_metrics, is_best=True
                    )
                else:
                    self.early_stopping_counter += 1
                
                # Checkpoint frequency
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    checkpoint_metrics = TrainingMetrics(
                        accuracy=val_metrics.get('accuracy', 0.0),
                        precision=val_metrics.get('precision_macro', 0.0),
                        recall=val_metrics.get('recall_macro', 0.0),
                        f1_score=val_metrics.get('f1_macro', 0.0),
                        train_loss=train_metrics['loss'],
                        val_loss=val_metrics['loss'],
                        learning_rate=optimizer.param_groups[0]['lr'],
                        epoch=epoch
                    )
                    
                    self.checkpoint_manager.save_checkpoint(
                        model, optimizer, epoch, checkpoint_metrics, is_best=False
                    )
                
                # Early stopping check
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
                logger.info(f"Epoch {epoch+1} completed - Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics.get('accuracy', 0.0):.4f}")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Stop GPU monitoring
            if self.gpu_monitor and self.config.enable_energy_tracking:
                energy_consumption = await self.gpu_monitor.stop_monitoring()
                logger.info(f"🔋 Training energy consumption: {energy_consumption.total_energy_wh:.3f} Wh")
                
                # Log energy metrics
                energy_metrics = {
                    'training_energy_wh': energy_consumption.total_energy_wh,
                    'training_avg_power_w': energy_consumption.average_power_watts,
                    'training_peak_power_w': energy_consumption.peak_power_watts,
                    'training_duration_seconds': time.time() - start_time,
                    'training_carbon_footprint_kg': energy_consumption.carbon_footprint_kg
                }
                
                metrics_logger.log_scalar_metrics(energy_metrics)
                
                # Log to energy tracker if available
                if self.energy_tracker:
                    self.energy_tracker.log_training_energy(
                        energy_consumption=energy_consumption,
                        model_version=f"{self.config.model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        model_params=hyperparams,
                        performance_metrics=val_metrics
                    )
        
        # Final evaluation and model registration
        await self._finalize_training(model, val_loader, metrics_logger)
        
        return self.current_run_id
    
    async def _train_epoch(self, 
                          model: nn.Module,
                          train_loader: DataLoader,
                          optimizer: optim.Optimizer,
                          criterion: nn.Module,
                          epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_scores = []
        inference_times = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            start_time = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            
            optimizer.step()
            
            # Record metrics
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            total_loss += loss.item()
            
            # Convert outputs for metrics calculation
            predictions = (torch.sigmoid(output) > self.config.confidence_threshold).float()
            scores = torch.sigmoid(output).cpu().detach().numpy().flatten()
            
            all_predictions.extend(predictions.cpu().detach().numpy().flatten())
            all_targets.extend(target.cpu().detach().numpy().flatten())
            all_scores.extend(scores)
            
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        
        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_scores = np.array(all_scores)
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            all_targets, all_predictions, all_scores, inference_times, self.config.batch_size
        )
        
        metrics['loss'] = avg_loss
        return metrics
    
    async def _validate_epoch(self, 
                             model: nn.Module,
                             val_loader: DataLoader,
                             criterion: nn.Module,
                             epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_scores = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                start_time = time.time()
                output = model(data)
                loss = criterion(output, target)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                total_loss += loss.item()
                
                # Convert outputs for metrics calculation
                predictions = (torch.sigmoid(output) > self.config.confidence_threshold).float()
                scores = torch.sigmoid(output).cpu().detach().numpy().flatten()
                
                all_predictions.extend(predictions.cpu().detach().numpy().flatten())
                all_targets.extend(target.cpu().detach().numpy().flatten())
                all_scores.extend(scores)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        
        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_scores = np.array(all_scores)
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            all_targets, all_predictions, all_scores, inference_times, self.config.batch_size
        )
        
        metrics['loss'] = avg_loss
        return metrics
    
    async def _finalize_training(self, 
                               model: nn.Module,
                               val_loader: DataLoader,
                               metrics_logger: MLflowMetricsLogger) -> None:
        """Finalize training with comprehensive evaluation."""
        
        logger.info("🔍 Performing final evaluation...")
        
        # Load best model
        best_checkpoint_path = self.checkpoint_manager.checkpoint_dir / "best_model.pt"
        if best_checkpoint_path.exists():
            checkpoint = self.checkpoint_manager.load_checkpoint(str(best_checkpoint_path))
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Loaded best model for final evaluation")
        
        # Final evaluation
        model.eval()
        all_predictions = []
        all_targets = []
        all_scores = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                predictions = (torch.sigmoid(output) > self.config.confidence_threshold).float()
                scores = torch.sigmoid(output).cpu().detach().numpy().flatten()
                
                all_predictions.extend(predictions.cpu().detach().numpy().flatten())
                all_targets.extend(target.cpu().detach().numpy().flatten())
                all_scores.extend(scores)
        
        # Calculate final metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_scores = np.array(all_scores)
        
        final_metrics = self.metrics_calculator.calculate_all_metrics(
            all_targets, all_predictions, all_scores, inference_times, self.config.batch_size
        )
        
        # Create metrics summary
        metrics_summary = self.metrics_calculator.create_metrics_summary(final_metrics)
        
        # Log final metrics and artifacts
        metrics_logger.log_scalar_metrics(final_metrics)
        metrics_logger.log_artifacts(metrics_summary)
        
        # Log model to MLflow
        model_path = f"models/{self.config.model_name}_{self.current_run_id}"
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model_path,
            registered_model_name=self.config.model_name
        )
        
        # Register model in model registry
        try:
            model_version = self.model_registry.register_model(
                model_path=f"runs:/{self.current_run_id}/{model_path}",
                model_name=self.config.model_name,
                run_id=self.current_run_id,
                description=f"U-Net anomaly detection model trained on {datetime.now().strftime('%Y-%m-%d')}",
                tags=self.config.tags
            )
            logger.info(f"✅ Model registered with version: {model_version}")
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
        
        # End MLflow run
        self.experiment_tracker.end_run(self.current_run_id, "FINISHED")
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📊 Final metrics: Accuracy={final_metrics.get('accuracy', 0.0):.4f}, "
                   f"F1={final_metrics.get('f1_macro', 0.0):.4f}, "
                   f"AUROC={final_metrics.get('auroc', 0.0):.4f}")
    
    def plot_training_summary(self) -> None:
        """Plot comprehensive training summary."""
        
        self.visualizer.plot_training_curves(
            self.training_history['train_losses'],
            self.training_history['val_losses'],
            self.training_history['train_metrics'],
            self.training_history['val_metrics']
        )
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        
        return {
            'run_id': self.current_run_id,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.training_history['train_losses']),
            'final_train_loss': self.training_history['train_losses'][-1] if self.training_history['train_losses'] else 0.0,
            'final_val_loss': self.training_history['val_losses'][-1] if self.training_history['val_losses'] else 0.0,
            'final_val_metrics': self.training_history['val_metrics'][-1] if self.training_history['val_metrics'] else {},
            'training_history': self.training_history
        }
