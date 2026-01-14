"""
Monitoring integration for individual trainers.

This module provides integration hooks for trainers to report metrics
and diagnostic information to both the enhanced monitoring system and
the existing diagnostic framework.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .two_phase_metrics import TwoPhaseMetricsTracker

# Import existing diagnostic system
try:
    from diagnostic.training_integration import TrainingLoopIntegration
    from diagnostic.diagnostic_engine import BatchData
    from diagnostic.logging_utils import MetricsLogger
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False


@dataclass
class TrainingStepMetrics:
    """Metrics for a single training step/batch."""
    step: int
    epoch: int
    loss: float
    batch_size: int
    processing_time: float
    samples_processed: int
    
    # Phase-specific metrics
    accuracy: Optional[float] = None
    triplets_created: Optional[int] = None
    avg_positive_similarity: Optional[float] = None
    avg_negative_similarity: Optional[float] = None
    
    # Additional metrics
    memory_usage_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gradient_norm: Optional[float] = None


class MonitoringIntegration:
    """
    Integration layer for connecting trainers with monitoring systems.
    
    Provides a unified interface for trainers to report metrics and
    receive diagnostic feedback without tight coupling to monitoring systems.
    """
    
    def __init__(self, 
                 metrics_tracker: Optional[TwoPhaseMetricsTracker] = None,
                 diagnostic_integration: Optional[TrainingLoopIntegration] = None,
                 diagnostic_logger: Optional[MetricsLogger] = None,
                 phase: str = "training"):
        """
        Initialize monitoring integration.
        
        Args:
            metrics_tracker: Optional metrics tracker instance
            diagnostic_integration: Optional diagnostic integration instance
            diagnostic_logger: Optional diagnostic logger instance
            phase: Training phase identifier
        """
        self.metrics_tracker = metrics_tracker
        self.diagnostic_integration = diagnostic_integration
        self.diagnostic_logger = diagnostic_logger
        self.phase = phase
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.last_batch_time = None
        self.batch_count = 0
        self.epoch_start_time = None
        
        # Diagnostic alert handling
        if self.diagnostic_integration:
            self.diagnostic_integration.add_alert_callback(self._handle_diagnostic_alert)
    
    def start_epoch(self, epoch: int) -> None:
        """Signal the start of a new epoch."""
        self.epoch_start_time = time.time()
        self.batch_count = 0
        
        self.logger.info(f"Starting epoch {epoch} monitoring")
    
    def end_epoch(self, epoch: int, epoch_loss: float, epoch_accuracy: Optional[float] = None) -> None:
        """Signal the end of an epoch."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            
            # Report to diagnostics
            if self.diagnostics:
                self.diagnostics.log_epoch_timing(epoch_time, epoch)
            
            # Report to metrics tracker
            if self.metrics_tracker:
                batch_metrics = {'total_samples': self.batch_count}  # Approximate
                
                if self.phase == "self_supervised":
                    self.metrics_tracker.log_phase1_epoch(epoch, epoch_loss, batch_metrics)
                elif self.phase == "fine_tuning" and epoch_accuracy is not None:
                    self.metrics_tracker.log_phase2_epoch(epoch, epoch_loss, epoch_accuracy, batch_metrics)
            
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    def log_training_step(self, metrics: TrainingStepMetrics, 
                         model: Optional[torch.nn.Module] = None,
                         batch_data: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Log a training step and perform health checks.
        
        Args:
            metrics: Training step metrics
            model: Optional model for gradient checking
            batch_data: Optional batch data for quality checks
            
        Returns:
            List of diagnostic alerts (if any)
        """
        alerts = []
        
        # Track batch timing
        if self.last_batch_time:
            batch_time = time.time() - self.last_batch_time
            
            if self.diagnostics:
                timing_alert = self.diagnostics.log_batch_timing(
                    batch_time, metrics.step, metrics.epoch
                )
                if timing_alert:
                    alerts.append(timing_alert)
        
        self.last_batch_time = time.time()
        self.batch_count += 1
        
        # Report to metrics tracker
        if self.metrics_tracker:
            batch_metrics = {
                'samples_processed': metrics.samples_processed,
                'processing_time': metrics.processing_time
            }
            
            if self.phase == "self_supervised":
                # Add contrastive-specific metrics
                if metrics.triplets_created is not None:
                    batch_metrics['triplets_created'] = metrics.triplets_created
                if metrics.avg_positive_similarity is not None:
                    batch_metrics['avg_positive_similarity'] = metrics.avg_positive_similarity
                if metrics.avg_negative_similarity is not None:
                    batch_metrics['avg_negative_similarity'] = metrics.avg_negative_similarity
                
                self.metrics_tracker.log_phase1_batch(metrics.step, metrics.loss, batch_metrics)
            
            elif self.phase == "fine_tuning" and metrics.accuracy is not None:
                self.metrics_tracker.log_phase2_batch(
                    metrics.step, metrics.loss, metrics.accuracy, batch_metrics
                )
        
        # Perform diagnostic health checks using existing diagnostic system
        if self.diagnostic_integration and DIAGNOSTICS_AVAILABLE:
            try:
                # Log training step to diagnostic system
                diagnostic_report = self.diagnostic_integration.on_training_step(
                    step=metrics.step,
                    loss=metrics.loss,
                    model=model,
                    batch_data=batch_data
                )
                
                if diagnostic_report:
                    alerts.append(diagnostic_report)
                    
                    # Log diagnostic results
                    if self.diagnostic_logger:
                        self.diagnostic_logger.log_diagnostic_report(diagnostic_report, metrics.step)
                
            except Exception as e:
                self.logger.warning(f"Diagnostic integration error: {e}")
        
        # Log detailed metrics to diagnostic logger
        if self.diagnostic_logger:
            try:
                self.diagnostic_logger.log_training_step(
                    step=metrics.step,
                    loss=metrics.loss,
                    gradient_norm=metrics.gradient_norm,
                    additional_metrics={
                        'accuracy': metrics.accuracy,
                        'batch_size': metrics.batch_size,
                        'processing_time': metrics.processing_time,
                        'samples_processed': metrics.samples_processed,
                        'triplets_created': metrics.triplets_created,
                        'avg_positive_similarity': metrics.avg_positive_similarity,
                        'avg_negative_similarity': metrics.avg_negative_similarity
                    }
                )
            except Exception as e:
                self.logger.warning(f"Diagnostic logging error: {e}")
        
        return alerts
    
    def update_cache_stats(self, cache_stats: Dict[str, Any]) -> None:
        """Update embedding cache statistics."""
        if self.metrics_tracker and self.phase == "self_supervised":
            self.metrics_tracker.update_phase1_cache_stats(cache_stats)
    
    def update_model_info(self, model_info: Dict[str, Any]) -> None:
        """Update model information."""
        if self.metrics_tracker and self.phase == "fine_tuning":
            self.metrics_tracker.update_phase2_model_info(model_info)
    
    def log_checkpoint_save(self, epoch: int, checkpoint_path: str, metrics: Dict[str, Any]) -> None:
        """Log checkpoint save event."""
        self.logger.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
        
        # Log to structured logger if available through metrics tracker
        if self.metrics_tracker and hasattr(self.metrics_tracker, 'logger'):
            self.metrics_tracker.logger.log_checkpoint_save(epoch, checkpoint_path, metrics)
    
    def log_convergence_event(self, epoch: int, event_type: str, details: Dict[str, Any]) -> None:
        """Log convergence-related events."""
        self.logger.info(f"Convergence event at epoch {epoch}: {event_type}")
        
        if event_type == "converged":
            self.logger.info(f"Training converged with details: {details}")
        elif event_type == "stagnated":
            self.logger.warning(f"Training stagnated with details: {details}")
        elif event_type == "diverged":
            self.logger.error(f"Training diverged with details: {details}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        status = {
            'phase': self.phase,
            'batch_count': self.batch_count,
            'monitoring_active': True
        }
        
        if self.metrics_tracker:
            status['metrics_summary'] = self.metrics_tracker.get_current_metrics_summary()
        
        if self.diagnostic_integration:
            try:
                status['diagnostic_performance'] = self.diagnostic_integration.get_performance_stats()
            except Exception as e:
                status['diagnostic_error'] = str(e)
        
        return status
    
    def _handle_diagnostic_alert(self, alert) -> None:
        """Handle diagnostic alerts with appropriate logging and actions."""
        # Log alert with appropriate level
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        self.logger.log(
            log_level, 
            f"DIAGNOSTIC ALERT [{alert.alert_type}]: {alert.message}"
        )
        
        # Take automatic actions for critical alerts
        if alert.severity == "critical":
            self._handle_critical_alert(alert)
    
    def _handle_critical_alert(self, alert) -> None:
        """Handle critical alerts that may require immediate action."""
        if alert.alert_type == "gradient_explosion":
            self.logger.critical(
                "Gradient explosion detected! Consider stopping training and "
                "reducing learning rate or implementing gradient clipping."
            )
        
        elif alert.alert_type == "memory_leak_detected":
            self.logger.critical(
                "Memory leak detected! Training may need to be restarted. "
                "Check for unclosed resources and tensor accumulation."
            )
        
        elif alert.alert_type == "loss_divergence":
            self.logger.critical(
                "Loss divergence detected! Training is likely unstable. "
                "Consider reducing learning rate or reverting to last checkpoint."
            )
    
    def create_training_report(self) -> Dict[str, Any]:
        """Create a comprehensive training report."""
        report = {
            'phase': self.phase,
            'batch_count': self.batch_count,
            'monitoring_summary': self.get_current_status()
        }
        
        if self.diagnostic_integration:
            try:
                report['diagnostic_performance'] = self.diagnostic_integration.get_performance_stats()
            except Exception as e:
                report['diagnostic_error'] = str(e)
        
        if self.diagnostic_logger:
            try:
                report['recent_anomalies'] = self.diagnostic_logger.get_recent_anomalies(last_n_steps=50)
                report['training_summary'] = self.diagnostic_logger.get_training_step_summary(self.batch_count)
            except Exception as e:
                report['metrics_error'] = str(e)
        
        return report


def create_monitoring_integration(metrics_tracker: Optional[TwoPhaseMetricsTracker] = None,
                                diagnostic_integration: Optional[TrainingLoopIntegration] = None,
                                diagnostic_logger: Optional[MetricsLogger] = None,
                                phase: str = "training") -> MonitoringIntegration:
    """
    Factory function to create monitoring integration.
    
    Args:
        metrics_tracker: Optional metrics tracker
        diagnostic_integration: Optional diagnostic integration
        diagnostic_logger: Optional diagnostic logger
        phase: Training phase identifier
        
    Returns:
        Configured MonitoringIntegration instance
    """
    return MonitoringIntegration(
        metrics_tracker=metrics_tracker,
        diagnostic_integration=diagnostic_integration,
        diagnostic_logger=diagnostic_logger,
        phase=phase
    )