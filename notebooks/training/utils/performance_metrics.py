"""Comprehensive performance metrics for ASTR-102 integration."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.domains.detection.models import Model, ModelRun
from src.domains.detection.metrics.detection_metrics import DetectionMetrics

logger = logging.getLogger(__name__)


class ComprehensiveMetricsCalculator:
    """Comprehensive metrics calculator implementing ASTR-102 requirements."""
    
    def __init__(self):
        self.detection_metrics = DetectionMetrics()
    
    def calculate_all_metrics(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            y_scores: np.ndarray,
                            inference_times: List[float] = None,
                            batch_size: int = 1) -> Dict[str, Any]:
        """Calculate all required metrics for ASTR-102."""
        
        metrics = {}
        
        # Classification metrics
        metrics.update(self.calculate_classification_metrics(y_true, y_pred, y_scores))
        
        # Calibration metrics
        metrics.update(self.calculate_calibration_metrics(y_true, y_scores))
        
        # Performance metrics
        if inference_times:
            metrics.update(self.calculate_performance_metrics(inference_times, batch_size))
        
        # Additional metrics for comprehensive tracking
        metrics.update(self.calculate_additional_metrics(y_true, y_pred, y_scores))
        
        return metrics
    
    def calculate_classification_metrics(self, 
                                       y_true: np.ndarray, 
                                       y_pred: np.ndarray, 
                                       y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (macro, micro, weighted)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Advanced metrics
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auroc = 0.0
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            auprc = np.trapz(recall_curve, precision_curve)
        except ValueError:
            auprc = 0.0
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'precision_micro': float(precision_micro),
            'precision_weighted': float(precision_weighted),
            'recall_macro': float(recall_macro),
            'recall_micro': float(recall_micro),
            'recall_weighted': float(recall_weighted),
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'f1_weighted': float(f1_weighted),
            'auroc': float(auroc),
            'auprc': float(auprc),
            'mcc': float(mcc),
            'balanced_accuracy': float(balanced_acc)
        }
    
    def calculate_calibration_metrics(self, 
                                    y_true: np.ndarray, 
                                    y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics."""
        
        try:
            # Expected Calibration Error (ECE)
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_scores, n_bins=10
            )
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Brier Score
            brier_score = brier_score_loss(y_true, y_scores)
            
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Calibration metrics calculation failed: {e}")
            ece = 0.0
            brier_score = 0.0
        
        return {
            'ece': float(ece),
            'brier_score': float(brier_score)
        }
    
    def calculate_performance_metrics(self, 
                                    inference_times: List[float], 
                                    batch_size: int) -> Dict[str, float]:
        """Calculate performance metrics."""
        
        if not inference_times:
            return {
                'latency_ms_p50': 0.0,
                'latency_ms_p95': 0.0,
                'throughput_items_per_s': 0.0
            }
        
        inference_times_ms = np.array(inference_times) * 1000  # Convert to ms
        
        # Latency percentiles
        latency_p50 = np.percentile(inference_times_ms, 50)
        latency_p95 = np.percentile(inference_times_ms, 95)
        
        # Throughput
        avg_inference_time = np.mean(inference_times)
        throughput = batch_size / avg_inference_time if avg_inference_time > 0 else 0.0
        
        return {
            'latency_ms_p50': float(latency_p50),
            'latency_ms_p95': float(latency_p95),
            'throughput_items_per_s': float(throughput)
        }
    
    def calculate_additional_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray, 
                                   y_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate additional metrics for comprehensive tracking."""
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve data
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            fpr, tpr, roc_thresholds = np.array([]), np.array([]), np.array([])
            roc_auc = 0.0
        
        # Precision-Recall curve data
        try:
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
            pr_auc = np.trapz(recall_curve, precision_curve)
        except ValueError:
            precision_curve, recall_curve, pr_thresholds = np.array([]), np.array([]), np.array([])
            pr_auc = 0.0
        
        return {
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': float(roc_auc)
            },
            'pr_curve': {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'auc': float(pr_auc)
            }
        }
    
    def create_metrics_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive metrics summary for MLflow artifacts."""
        
        summary = {
            'classification_metrics': {
                'accuracy': metrics.get('accuracy', 0.0),
                'precision_macro': metrics.get('precision_macro', 0.0),
                'recall_macro': metrics.get('recall_macro', 0.0),
                'f1_macro': metrics.get('f1_macro', 0.0),
                'auroc': metrics.get('auroc', 0.0),
                'auprc': metrics.get('auprc', 0.0),
                'mcc': metrics.get('mcc', 0.0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0.0)
            },
            'calibration_metrics': {
                'ece': metrics.get('ece', 0.0),
                'brier_score': metrics.get('brier_score', 0.0)
            },
            'performance_metrics': {
                'latency_ms_p50': metrics.get('latency_ms_p50', 0.0),
                'latency_ms_p95': metrics.get('latency_ms_p95', 0.0),
                'throughput_items_per_s': metrics.get('throughput_items_per_s', 0.0)
            },
            'confusion_matrix': metrics.get('confusion_matrix', []),
            'roc_curve': metrics.get('roc_curve', {}),
            'pr_curve': metrics.get('pr_curve', {})
        }
        
        return summary


class MLflowMetricsLogger:
    """MLflow metrics logging utilities for ASTR-102."""
    
    def __init__(self, mlflow_client, run_id: str):
        self.mlflow_client = mlflow_client
        self.run_id = run_id
    
    def log_scalar_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log scalar metrics to MLflow."""
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                self.mlflow_client.log_metric(
                    run_id=self.run_id,
                    key=metric_name,
                    value=float(metric_value),
                    step=step
                )
    
    def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Log artifacts to MLflow."""
        
        import json
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save confusion matrix
            if 'confusion_matrix' in artifacts:
                cm_path = os.path.join(temp_dir, 'confusion_matrix.json')
                with open(cm_path, 'w') as f:
                    json.dump(artifacts['confusion_matrix'], f)
                self.mlflow_client.log_artifact(self.run_id, cm_path)
            
            # Save ROC curve
            if 'roc_curve' in artifacts:
                roc_path = os.path.join(temp_dir, 'roc_curve.json')
                with open(roc_path, 'w') as f:
                    json.dump(artifacts['roc_curve'], f)
                self.mlflow_client.log_artifact(self.run_id, roc_path)
            
            # Save PR curve
            if 'pr_curve' in artifacts:
                pr_path = os.path.join(temp_dir, 'pr_curve.json')
                with open(pr_path, 'w') as f:
                    json.dump(artifacts['pr_curve'], f)
                self.mlflow_client.log_artifact(self.run_id, pr_path)
            
            # Save metrics summary
            summary_path = os.path.join(temp_dir, 'metrics_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(artifacts, f)
            self.mlflow_client.log_artifact(self.run_id, summary_path)
    
    def log_model_parameters(self, model_params: Dict[str, Any]) -> None:
        """Log model parameters to MLflow."""
        
        for param_name, param_value in model_params.items():
            if isinstance(param_value, (str, int, float, bool)):
                self.mlflow_client.log_param(
                    run_id=self.run_id,
                    key=param_name,
                    value=param_value
                )
            else:
                # Convert complex types to strings
                self.mlflow_client.log_param(
                    run_id=self.run_id,
                    key=param_name,
                    value=str(param_value)
                )


class DatabaseMetricsUpdater:
    """Database metrics updater for ASTR-102 integration."""
    
    def __init__(self, db_session):
        self.db_session = db_session
    
    async def update_model_metrics(self, 
                                 model_id: str, 
                                 metrics: Dict[str, float]) -> None:
        """Update model metrics in database."""
        
        try:
            # Update Model table with training metrics
            model = await self.db_session.get(Model, model_id)
            if model:
                # Update scalar metrics
                model.precision = metrics.get('precision_macro', 0.0)
                model.recall = metrics.get('recall_macro', 0.0)
                model.f1_score = metrics.get('f1_macro', 0.0)
                model.accuracy = metrics.get('accuracy', 0.0)
                
                # Update training_metrics JSONB field with comprehensive metrics
                if not model.training_metrics:
                    model.training_metrics = {}
                
                model.training_metrics.update({
                    'auroc': metrics.get('auroc', 0.0),
                    'auprc': metrics.get('auprc', 0.0),
                    'mcc': metrics.get('mcc', 0.0),
                    'balanced_accuracy': metrics.get('balanced_accuracy', 0.0),
                    'ece': metrics.get('ece', 0.0),
                    'brier_score': metrics.get('brier_score', 0.0),
                    'last_updated': datetime.now().isoformat()
                })
                
                await self.db_session.commit()
                logger.info(f"Updated model metrics for model {model_id}")
        
        except Exception as e:
            logger.error(f"Failed to update model metrics: {e}")
            await self.db_session.rollback()
            raise
    
    async def update_model_run_metrics(self, 
                                     model_run_id: str, 
                                     metrics: Dict[str, float]) -> None:
        """Update model run metrics in database."""
        
        try:
            # Update ModelRun table with inference metrics
            model_run = await self.db_session.get(ModelRun, model_run_id)
            if model_run:
                # Update performance metrics
                model_run.inference_time_ms = int(metrics.get('latency_ms_p50', 0))
                
                # Update additional metrics in a JSONB field if available
                # (This would require extending the ModelRun schema)
                
                await self.db_session.commit()
                logger.info(f"Updated model run metrics for run {model_run_id}")
        
        except Exception as e:
            logger.error(f"Failed to update model run metrics: {e}")
            await self.db_session.rollback()
            raise


class MetricsValidator:
    """Metrics validation utilities for ASTR-102."""
    
    @staticmethod
    def validate_metrics(metrics: Dict[str, Any]) -> List[str]:
        """Validate metrics for completeness and correctness."""
        
        errors = []
        
        # Required classification metrics
        required_metrics = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'auroc', 'auprc', 'mcc', 'balanced_accuracy'
        ]
        
        for metric in required_metrics:
            if metric not in metrics:
                errors.append(f"Missing required metric: {metric}")
            elif not isinstance(metrics[metric], (int, float)):
                errors.append(f"Invalid metric type for {metric}: {type(metrics[metric])}")
            elif np.isnan(metrics[metric]):
                errors.append(f"NaN value for metric: {metric}")
        
        # Validate metric ranges
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auroc', 'auprc', 'balanced_accuracy']:
            if metric in metrics and not (0 <= metrics[metric] <= 1):
                errors.append(f"Metric {metric} out of range [0, 1]: {metrics[metric]}")
        
        # Validate performance metrics
        if 'latency_ms_p50' in metrics and metrics['latency_ms_p50'] < 0:
            errors.append(f"Invalid latency_p50: {metrics['latency_ms_p50']}")
        
        if 'throughput_items_per_s' in metrics and metrics['throughput_items_per_s'] < 0:
            errors.append(f"Invalid throughput: {metrics['throughput_items_per_s']}")
        
        return errors
    
    @staticmethod
    def check_metrics_completeness(metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Check completeness of metrics against ASTR-102 requirements."""
        
        completeness = {
            'classification_metrics': all(metric in metrics for metric in [
                'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                'auroc', 'auprc', 'mcc', 'balanced_accuracy'
            ]),
            'calibration_metrics': all(metric in metrics for metric in [
                'ece', 'brier_score'
            ]),
            'performance_metrics': all(metric in metrics for metric in [
                'latency_ms_p50', 'latency_ms_p95', 'throughput_items_per_s'
            ]),
            'artifacts': all(artifact in metrics for artifact in [
                'confusion_matrix', 'roc_curve', 'pr_curve'
            ])
        }
        
        return completeness
