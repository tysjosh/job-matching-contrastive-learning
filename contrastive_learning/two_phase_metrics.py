"""
Enhanced metrics tracking and monitoring for 2-phase training.

This module provides comprehensive metrics collection, analysis, and reporting
for both self-supervised pre-training and supervised fine-tuning phases.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading
from collections import defaultdict, deque

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .logging_utils import StructuredLogger, MemoryMonitor


@dataclass
class PhaseMetrics:
    """Metrics for a single training phase."""
    phase_name: str
    phase_type: str  # "self_supervised" or "fine_tuning"
    start_time: float
    end_time: Optional[float] = None
    total_epochs: int = 0
    total_batches: int = 0
    total_samples: int = 0
    
    # Loss tracking
    epoch_losses: List[float] = field(default_factory=list)
    batch_losses: List[float] = field(default_factory=list)
    best_loss: float = float('inf')
    final_loss: Optional[float] = None
    
    # Phase-specific metrics
    phase_specific_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Convergence tracking
    loss_convergence_window: int = 10
    convergence_threshold: float = 0.001
    is_converged: bool = False
    convergence_epoch: Optional[int] = None
    
    def get_duration(self) -> Optional[float]:
        """Get phase duration in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def add_epoch_loss(self, loss: float, epoch: int) -> None:
        """Add epoch loss and check for convergence."""
        self.epoch_losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Check for convergence
        if not self.is_converged and len(self.epoch_losses) >= self.loss_convergence_window:
            recent_losses = self.epoch_losses[-self.loss_convergence_window:]
            loss_std = np.std(recent_losses) if len(recent_losses) > 1 else float('inf')
            
            if loss_std < self.convergence_threshold:
                self.is_converged = True
                self.convergence_epoch = epoch
    
    def add_batch_loss(self, loss: float) -> None:
        """Add batch loss."""
        self.batch_losses.append(loss)
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get comprehensive loss statistics."""
        if not self.epoch_losses:
            return {}
        
        return {
            'mean_loss': float(np.mean(self.epoch_losses)),
            'std_loss': float(np.std(self.epoch_losses)),
            'min_loss': float(np.min(self.epoch_losses)),
            'max_loss': float(np.max(self.epoch_losses)),
            'final_loss': self.epoch_losses[-1] if self.epoch_losses else None,
            'best_loss': self.best_loss,
            'loss_improvement': self.epoch_losses[0] - self.epoch_losses[-1] if len(self.epoch_losses) > 1 else 0.0,
            'convergence_epoch': self.convergence_epoch,
            'is_converged': self.is_converged
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ContrastivePhaseMetrics(PhaseMetrics):
    """Metrics specific to contrastive learning (self-supervised) phase."""
    
    def __post_init__(self):
        self.phase_type = "self_supervised"
        # Initialize contrastive-specific metrics
        self.phase_specific_metrics.update({
            'total_triplets_created': 0,
            'avg_positive_similarity': [],
            'avg_negative_similarity': [],
            'embedding_cache_stats': {},
            'global_negative_sampling_stats': {},
            'augmentation_stats': {}
        })
    
    def add_contrastive_batch_metrics(self, batch_metrics: Dict[str, Any]) -> None:
        """Add contrastive learning specific batch metrics."""
        self.total_batches += 1
        self.total_samples += batch_metrics.get('samples_processed', 0)
        self.phase_specific_metrics['total_triplets_created'] += batch_metrics.get('triplets_created', 0)
        
        # Track similarity statistics
        if 'avg_positive_similarity' in batch_metrics:
            self.phase_specific_metrics['avg_positive_similarity'].append(batch_metrics['avg_positive_similarity'])
        
        if 'avg_negative_similarity' in batch_metrics:
            self.phase_specific_metrics['avg_negative_similarity'].append(batch_metrics['avg_negative_similarity'])
    
    def update_cache_stats(self, cache_stats: Dict[str, Any]) -> None:
        """Update embedding cache statistics."""
        self.phase_specific_metrics['embedding_cache_stats'] = cache_stats
    
    def update_augmentation_stats(self, aug_stats: Dict[str, Any]) -> None:
        """Update augmentation statistics."""
        self.phase_specific_metrics['augmentation_stats'] = aug_stats
    
    def get_contrastive_summary(self) -> Dict[str, Any]:
        """Get contrastive learning specific summary."""
        summary = self.get_loss_statistics()
        
        # Add contrastive-specific metrics
        if self.phase_specific_metrics['avg_positive_similarity']:
            summary['avg_positive_similarity'] = float(np.mean(self.phase_specific_metrics['avg_positive_similarity']))
        
        if self.phase_specific_metrics['avg_negative_similarity']:
            summary['avg_negative_similarity'] = float(np.mean(self.phase_specific_metrics['avg_negative_similarity']))
        
        summary.update({
            'total_triplets': self.phase_specific_metrics['total_triplets_created'],
            'triplets_per_sample': (
                self.phase_specific_metrics['total_triplets_created'] / self.total_samples 
                if self.total_samples > 0 else 0
            ),
            'cache_hit_rate': self.phase_specific_metrics['embedding_cache_stats'].get('hit_rate', 0.0),
            'cache_time_saved': self.phase_specific_metrics['embedding_cache_stats'].get('time_saved_seconds', 0.0)
        })
        
        return summary


@dataclass
class FineTuningPhaseMetrics(PhaseMetrics):
    """Metrics specific to fine-tuning phase."""
    
    def __post_init__(self):
        self.phase_type = "fine_tuning"
        # Initialize fine-tuning specific metrics
        self.phase_specific_metrics.update({
            'epoch_accuracies': [],
            'batch_accuracies': [],
            'best_accuracy': 0.0,
            'final_accuracy': None,
            'classification_stats': {},
            'frozen_parameters': 0,
            'trainable_parameters': 0
        })
    
    def add_finetuning_batch_metrics(self, batch_metrics: Dict[str, Any]) -> None:
        """Add fine-tuning specific batch metrics."""
        self.total_batches += 1
        self.total_samples += batch_metrics.get('samples_processed', 0)
        
        # Track accuracy
        if 'accuracy' in batch_metrics:
            self.phase_specific_metrics['batch_accuracies'].append(batch_metrics['accuracy'])
    
    def add_epoch_accuracy(self, accuracy: float, epoch: int) -> None:
        """Add epoch accuracy."""
        self.phase_specific_metrics['epoch_accuracies'].append(accuracy)
        if accuracy > self.phase_specific_metrics['best_accuracy']:
            self.phase_specific_metrics['best_accuracy'] = accuracy
    
    def update_model_info(self, model_info: Dict[str, Any]) -> None:
        """Update model parameter information."""
        self.phase_specific_metrics['frozen_parameters'] = model_info.get('frozen_parameters', 0)
        self.phase_specific_metrics['trainable_parameters'] = model_info.get('trainable_parameters', 0)
    
    def get_finetuning_summary(self) -> Dict[str, Any]:
        """Get fine-tuning specific summary."""
        summary = self.get_loss_statistics()
        
        # Add fine-tuning specific metrics
        accuracies = self.phase_specific_metrics['epoch_accuracies']
        if accuracies:
            summary.update({
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'best_accuracy': self.phase_specific_metrics['best_accuracy'],
                'final_accuracy': accuracies[-1] if accuracies else None,
                'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0.0
            })
        
        summary.update({
            'frozen_parameters': self.phase_specific_metrics['frozen_parameters'],
            'trainable_parameters': self.phase_specific_metrics['trainable_parameters'],
            'parameter_efficiency': (
                self.phase_specific_metrics['trainable_parameters'] / 
                (self.phase_specific_metrics['frozen_parameters'] + self.phase_specific_metrics['trainable_parameters'])
                if (self.phase_specific_metrics['frozen_parameters'] + self.phase_specific_metrics['trainable_parameters']) > 0 
                else 0.0
            )
        })
        
        return summary


class TwoPhaseMetricsTracker:
    """
    Comprehensive metrics tracker for 2-phase training.
    
    Tracks metrics for both phases, provides real-time monitoring,
    and generates comprehensive reports comparing both phases.
    """
    
    def __init__(self, output_dir: Union[str, Path], enable_real_time_monitoring: bool = True):
        """
        Initialize the metrics tracker.
        
        Args:
            output_dir: Directory for saving metrics and reports
            enable_real_time_monitoring: Whether to enable real-time monitoring
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize structured logging
        self.logger = StructuredLogger(
            name="two_phase_metrics",
            log_dir=self.output_dir / "logs"
        )
        self.memory_monitor = MemoryMonitor()
        
        # Phase metrics
        self.phase1_metrics: Optional[ContrastivePhaseMetrics] = None
        self.phase2_metrics: Optional[FineTuningPhaseMetrics] = None
        self.current_phase: Optional[str] = None
        
        # Pipeline metrics
        self.pipeline_start_time: Optional[float] = None
        self.pipeline_end_time: Optional[float] = None
        self.phase_transition_time: Optional[float] = None
        
        # Real-time monitoring
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Metrics history for trend analysis
        self.loss_history = deque(maxlen=1000)  # Keep last 1000 loss values
        self.accuracy_history = deque(maxlen=1000)  # Keep last 1000 accuracy values
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.logger.info("TwoPhaseMetricsTracker initialized")
    
    def start_pipeline(self) -> None:
        """Start pipeline metrics tracking."""
        with self._lock:
            self.pipeline_start_time = time.time()
        
        self.logger.logger.info("Started 2-phase training pipeline metrics tracking")
        
        if self.enable_real_time_monitoring:
            self._start_monitoring_thread()
    
    def start_phase1(self, phase_name: str = "self_supervised_pretraining") -> None:
        """Start Phase 1 (contrastive learning) metrics tracking."""
        with self._lock:
            self.phase1_metrics = ContrastivePhaseMetrics(
                phase_name=phase_name,
                phase_type="self_supervised",
                start_time=time.time()
            )
            self.current_phase = "phase1"
        
        self.logger.logger.info(f"Started Phase 1 metrics tracking: {phase_name}")
    
    def start_phase2(self, phase_name: str = "supervised_finetuning") -> None:
        """Start Phase 2 (fine-tuning) metrics tracking."""
        with self._lock:
            # Record phase transition time
            if self.phase1_metrics and self.phase1_metrics.end_time:
                self.phase_transition_time = time.time() - self.phase1_metrics.end_time
            
            self.phase2_metrics = FineTuningPhaseMetrics(
                phase_name=phase_name,
                phase_type="fine_tuning",
                start_time=time.time()
            )
            self.current_phase = "phase2"
        
        self.logger.logger.info(f"Started Phase 2 metrics tracking: {phase_name}")
    
    def end_phase1(self) -> None:
        """End Phase 1 metrics tracking."""
        with self._lock:
            if self.phase1_metrics:
                self.phase1_metrics.end_time = time.time()
                if self.phase1_metrics.epoch_losses:
                    self.phase1_metrics.final_loss = self.phase1_metrics.epoch_losses[-1]
        
        self.logger.logger.info("Ended Phase 1 metrics tracking")
        self._save_phase_metrics("phase1")
    
    def end_phase2(self) -> None:
        """End Phase 2 metrics tracking."""
        with self._lock:
            if self.phase2_metrics:
                self.phase2_metrics.end_time = time.time()
                if self.phase2_metrics.epoch_losses:
                    self.phase2_metrics.final_loss = self.phase2_metrics.epoch_losses[-1]
                if self.phase2_metrics.phase_specific_metrics['epoch_accuracies']:
                    self.phase2_metrics.phase_specific_metrics['final_accuracy'] = \
                        self.phase2_metrics.phase_specific_metrics['epoch_accuracies'][-1]
        
        self.logger.logger.info("Ended Phase 2 metrics tracking")
        self._save_phase_metrics("phase2")
    
    def end_pipeline(self) -> None:
        """End pipeline metrics tracking."""
        with self._lock:
            self.pipeline_end_time = time.time()
            self.current_phase = None
        
        if self.enable_real_time_monitoring:
            self._stop_monitoring_thread()
        
        self.logger.logger.info("Ended 2-phase training pipeline metrics tracking")
        self._generate_comprehensive_report()
    
    def log_phase1_epoch(self, epoch: int, loss: float, batch_metrics: Dict[str, Any]) -> None:
        """Log Phase 1 epoch completion."""
        if not self.phase1_metrics:
            return
        
        with self._lock:
            self.phase1_metrics.add_epoch_loss(loss, epoch)
            self.phase1_metrics.total_epochs = epoch + 1
            self.loss_history.append(('phase1', epoch, loss))
        
        # Log convergence status
        if self.phase1_metrics.is_converged and self.phase1_metrics.convergence_epoch == epoch:
            self.logger.logger.info(f"Phase 1 converged at epoch {epoch} with loss {loss:.6f}")
        
        self.logger.logger.info(
            f"Phase 1 Epoch {epoch + 1}: loss={loss:.6f}, "
            f"best_loss={self.phase1_metrics.best_loss:.6f}, "
            f"samples={batch_metrics.get('total_samples', 0)}"
        )
    
    def log_phase1_batch(self, batch_idx: int, loss: float, batch_metrics: Dict[str, Any]) -> None:
        """Log Phase 1 batch completion."""
        if not self.phase1_metrics:
            return
        
        with self._lock:
            self.phase1_metrics.add_batch_loss(loss)
            self.phase1_metrics.add_contrastive_batch_metrics(batch_metrics)
    
    def log_phase2_epoch(self, epoch: int, loss: float, accuracy: float, batch_metrics: Dict[str, Any]) -> None:
        """Log Phase 2 epoch completion."""
        if not self.phase2_metrics:
            return
        
        with self._lock:
            self.phase2_metrics.add_epoch_loss(loss, epoch)
            self.phase2_metrics.add_epoch_accuracy(accuracy, epoch)
            self.phase2_metrics.total_epochs = epoch + 1
            self.loss_history.append(('phase2', epoch, loss))
            self.accuracy_history.append(('phase2', epoch, accuracy))
        
        # Log convergence status
        if self.phase2_metrics.is_converged and self.phase2_metrics.convergence_epoch == epoch:
            self.logger.logger.info(f"Phase 2 converged at epoch {epoch} with loss {loss:.6f}")
        
        self.logger.logger.info(
            f"Phase 2 Epoch {epoch + 1}: loss={loss:.6f}, accuracy={accuracy:.4f}, "
            f"best_accuracy={self.phase2_metrics.phase_specific_metrics['best_accuracy']:.4f}"
        )
    
    def log_phase2_batch(self, batch_idx: int, loss: float, accuracy: float, batch_metrics: Dict[str, Any]) -> None:
        """Log Phase 2 batch completion."""
        if not self.phase2_metrics:
            return
        
        with self._lock:
            self.phase2_metrics.add_batch_loss(loss)
            batch_metrics['accuracy'] = accuracy
            self.phase2_metrics.add_finetuning_batch_metrics(batch_metrics)
    
    def update_phase1_cache_stats(self, cache_stats: Dict[str, Any]) -> None:
        """Update Phase 1 embedding cache statistics."""
        if self.phase1_metrics:
            self.phase1_metrics.update_cache_stats(cache_stats)
    
    def update_phase2_model_info(self, model_info: Dict[str, Any]) -> None:
        """Update Phase 2 model information."""
        if self.phase2_metrics:
            self.phase2_metrics.update_model_info(model_info)
    
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary for both phases."""
        summary = {
            'pipeline_status': {
                'current_phase': self.current_phase,
                'pipeline_duration': (
                    time.time() - self.pipeline_start_time 
                    if self.pipeline_start_time else None
                ),
                'phase_transition_time': self.phase_transition_time
            },
            'phase1_summary': None,
            'phase2_summary': None
        }
        
        if self.phase1_metrics:
            summary['phase1_summary'] = self.phase1_metrics.get_contrastive_summary()
        
        if self.phase2_metrics:
            summary['phase2_summary'] = self.phase2_metrics.get_finetuning_summary()
        
        return summary
    
    def _save_phase_metrics(self, phase: str) -> None:
        """Save metrics for a specific phase."""
        if phase == "phase1" and self.phase1_metrics:
            metrics_file = self.output_dir / "phase1_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.phase1_metrics.to_dict(), f, indent=2, default=str)
        
        elif phase == "phase2" and self.phase2_metrics:
            metrics_file = self.output_dir / "phase2_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.phase2_metrics.to_dict(), f, indent=2, default=str)
    
    def _generate_comprehensive_report(self) -> None:
        """Generate comprehensive comparison report for both phases."""
        report = {
            'pipeline_summary': {
                'total_duration': (
                    self.pipeline_end_time - self.pipeline_start_time 
                    if self.pipeline_start_time and self.pipeline_end_time else None
                ),
                'phase_transition_time': self.phase_transition_time,
                'phases_completed': []
            },
            'phase_comparison': {},
            'recommendations': []
        }
        
        # Add phase summaries
        if self.phase1_metrics:
            report['pipeline_summary']['phases_completed'].append('phase1')
            report['phase_comparison']['phase1'] = self.phase1_metrics.get_contrastive_summary()
        
        if self.phase2_metrics:
            report['pipeline_summary']['phases_completed'].append('phase2')
            report['phase_comparison']['phase2'] = self.phase2_metrics.get_finetuning_summary()
        
        # Generate recommendations
        self._generate_training_recommendations(report)
        
        # Save comprehensive report
        report_file = self.output_dir / "comprehensive_training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.logger.info(f"Comprehensive training report saved: {report_file}")
    
    def _generate_training_recommendations(self, report: Dict[str, Any]) -> None:
        """Generate training recommendations based on metrics."""
        recommendations = []
        
        # Phase 1 recommendations
        if 'phase1' in report['phase_comparison']:
            phase1_metrics = report['phase_comparison']['phase1']
            
            if not phase1_metrics.get('is_converged', False):
                recommendations.append({
                    'type': 'convergence',
                    'phase': 'phase1',
                    'message': 'Phase 1 did not converge. Consider increasing epochs or adjusting learning rate.',
                    'severity': 'warning'
                })
            
            cache_hit_rate = phase1_metrics.get('cache_hit_rate', 0.0)
            if cache_hit_rate < 0.5:
                recommendations.append({
                    'type': 'performance',
                    'phase': 'phase1',
                    'message': f'Low embedding cache hit rate ({cache_hit_rate:.2%}). Consider increasing cache size.',
                    'severity': 'info'
                })
        
        # Phase 2 recommendations
        if 'phase2' in report['phase_comparison']:
            phase2_metrics = report['phase_comparison']['phase2']
            
            final_accuracy = phase2_metrics.get('final_accuracy', 0.0)
            if final_accuracy < 0.7:
                recommendations.append({
                    'type': 'performance',
                    'phase': 'phase2',
                    'message': f'Low final accuracy ({final_accuracy:.2%}). Consider unfreezing more layers or adjusting learning rate.',
                    'severity': 'warning'
                })
            
            accuracy_improvement = phase2_metrics.get('accuracy_improvement', 0.0)
            if accuracy_improvement < 0.05:
                recommendations.append({
                    'type': 'convergence',
                    'phase': 'phase2',
                    'message': f'Small accuracy improvement ({accuracy_improvement:.2%}). Pre-trained model may already be well-suited for the task.',
                    'severity': 'info'
                })
        
        # Cross-phase recommendations
        if 'phase1' in report['phase_comparison'] and 'phase2' in report['phase_comparison']:
            phase1_duration = self.phase1_metrics.get_duration() if self.phase1_metrics else 0
            phase2_duration = self.phase2_metrics.get_duration() if self.phase2_metrics else 0
            
            if phase2_duration and phase1_duration and phase2_duration > phase1_duration * 2:
                recommendations.append({
                    'type': 'efficiency',
                    'phase': 'both',
                    'message': 'Phase 2 took significantly longer than Phase 1. Consider reducing fine-tuning epochs or batch size.',
                    'severity': 'info'
                })
        
        report['recommendations'] = recommendations
    
    def _start_monitoring_thread(self) -> None:
        """Start real-time monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.logger.info("Started real-time monitoring thread")
    
    def _stop_monitoring_thread(self) -> None:
        """Stop real-time monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.logger.info("Stopped real-time monitoring thread")
    
    def _monitoring_loop(self) -> None:
        """Real-time monitoring loop."""
        while self.monitoring_active:
            try:
                # Log current status every 30 seconds
                summary = self.get_current_metrics_summary()
                
                # Log memory usage
                memory_info = self.memory_monitor.get_memory_usage()
                self.logger.log_memory_usage(
                    f"monitoring_{self.current_phase or 'idle'}",
                    memory_info["system_memory_mb"] or 0,
                    memory_info["gpu_memory_mb"]
                )
                
                # Save periodic checkpoint of metrics
                if self.current_phase:
                    checkpoint_file = self.output_dir / f"metrics_checkpoint_{self.current_phase}.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(summary, f, indent=2, default=str)
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def export_metrics_for_visualization(self) -> Dict[str, Any]:
        """Export metrics in format suitable for visualization tools."""
        export_data = {
            'timeline': [],
            'loss_curves': {
                'phase1': [],
                'phase2': []
            },
            'accuracy_curve': [],
            'summary_stats': self.get_current_metrics_summary()
        }
        
        # Export loss timeline
        for phase, epoch, loss in self.loss_history:
            export_data['loss_curves'][phase].append({'epoch': epoch, 'loss': loss})
        
        # Export accuracy timeline (Phase 2 only)
        for phase, epoch, accuracy in self.accuracy_history:
            if phase == 'phase2':
                export_data['accuracy_curve'].append({'epoch': epoch, 'accuracy': accuracy})
        
        # Create timeline of major events
        if self.phase1_metrics:
            export_data['timeline'].append({
                'event': 'phase1_start',
                'timestamp': self.phase1_metrics.start_time,
                'description': 'Self-supervised pre-training started'
            })
            
            if self.phase1_metrics.end_time:
                export_data['timeline'].append({
                    'event': 'phase1_end',
                    'timestamp': self.phase1_metrics.end_time,
                    'description': f'Pre-training completed with final loss: {self.phase1_metrics.final_loss:.6f}'
                })
        
        if self.phase2_metrics:
            export_data['timeline'].append({
                'event': 'phase2_start',
                'timestamp': self.phase2_metrics.start_time,
                'description': 'Supervised fine-tuning started'
            })
            
            if self.phase2_metrics.end_time:
                final_acc = self.phase2_metrics.phase_specific_metrics.get('final_accuracy', 0.0)
                export_data['timeline'].append({
                    'event': 'phase2_end',
                    'timestamp': self.phase2_metrics.end_time,
                    'description': f'Fine-tuning completed with final accuracy: {final_acc:.4f}'
                })
        
        return export_data