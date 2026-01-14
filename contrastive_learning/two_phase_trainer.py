"""
TwoPhaseTrainer - Orchestrator for 2-phase training strategy.

This module implements the main orchestrator that coordinates self-supervised 
pre-training followed by supervised fine-tuning for career-aware contrastive learning.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .data_structures import TrainingConfig, TrainingResults
from .trainer import ContrastiveLearningTrainer
from .fine_tuning_trainer import FineTuningTrainer
from .logging_utils import setup_training_logger, MemoryMonitor
from .two_phase_metrics import TwoPhaseMetricsTracker

# Import existing diagnostic system
try:
    from diagnostic.training_integration import create_training_integration, DiagnosticTriggerConfig
    from diagnostic.diagnostic_engine import DiagnosticConfig
    from diagnostic.logging_utils import MetricsLogger
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    logger.warning("Diagnostic system not available - limited monitoring capabilities")

logger = logging.getLogger(__name__)


@dataclass
class TwoPhaseResults:
    """Results from complete 2-phase training pipeline."""
    pretraining_results: TrainingResults
    finetuning_results: Dict[str, Any]
    pretrained_model_path: str
    final_model_path: str
    phase_transition_time: float
    total_training_time: float
    phase1_config: Dict[str, Any]
    phase2_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pretraining_results': self.pretraining_results.to_dict(),
            'finetuning_results': self.finetuning_results,
            'pretrained_model_path': self.pretrained_model_path,
            'final_model_path': self.final_model_path,
            'phase_transition_time': self.phase_transition_time,
            'total_training_time': self.total_training_time,
            'phase1_config': self.phase1_config,
            'phase2_config': self.phase2_config
        }
    
    def save_json(self, file_path: str) -> None:
        """Save results to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class TwoPhaseTrainer:
    """
    Orchestrator for 2-phase training strategy.
    
    Coordinates self-supervised pre-training using augmentation-generated labels,
    followed by supervised fine-tuning on labeled data for downstream tasks.
    """
    
    def __init__(self, 
                 pretrain_config: TrainingConfig, 
                 finetune_config: TrainingConfig,
                 output_dir: str = "two_phase_output"):
        """
        Initialize the TwoPhaseTrainer.
        
        Args:
            pretrain_config: Configuration for self-supervised pre-training phase
            finetune_config: Configuration for supervised fine-tuning phase
            output_dir: Base directory for saving outputs from both phases
            
        Raises:
            ImportError: If PyTorch is not available
            ValueError: If configurations are invalid or incompatible
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TwoPhaseTrainer. "
                "Install with: pip install torch"
            )
        
        self.pretrain_config = pretrain_config
        self.finetune_config = finetune_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configurations
        self._validate_configurations()
        
        # Setup output directories for each phase
        self.phase1_output_dir = self.output_dir / "phase1_pretraining"
        self.phase2_output_dir = self.output_dir / "phase2_finetuning"
        self.phase1_output_dir.mkdir(parents=True, exist_ok=True)
        self.phase2_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize structured logging
        self.structured_logger = setup_training_logger(
            name="two_phase_training",
            log_dir=self.output_dir / "logs",
            log_level=logging.INFO
        )
        self.memory_monitor = MemoryMonitor()
        
        # Initialize enhanced metrics tracking
        self.metrics_tracker = TwoPhaseMetricsTracker(
            output_dir=self.output_dir / "metrics",
            enable_real_time_monitoring=True
        )
        
        # Initialize diagnostic integration if available
        self.diagnostic_integration = None
        self.diagnostic_logger = None
        
        if DIAGNOSTICS_AVAILABLE:
            try:
                # Configure diagnostic system for 2-phase training
                diagnostic_config = DiagnosticConfig(
                    enable_batch_analysis=True,
                    enable_loss_analysis=True,
                    enable_embedding_analysis=True,
                    compute_gradients=False,  # Avoid affecting training state
                    log_level="INFO"
                )
                
                trigger_config = DiagnosticTriggerConfig(
                    loss_stagnation_steps=50,
                    gradient_explosion_threshold=10.0,
                    periodic_check_frequency=100,
                    quick_check_frequency=25,
                    enable_async_diagnostics=True,
                    max_diagnostic_overhead_ms=50.0
                )
                
                self.diagnostic_integration = create_training_integration(
                    diagnostic_config=diagnostic_config,
                    trigger_config=trigger_config,
                    enable_async=True
                )
                
                # Initialize diagnostic metrics logger
                self.diagnostic_logger = MetricsLogger(
                    log_dir=self.output_dir / "diagnostic_logs",
                    real_time_logging=True,
                    anomaly_detection=True
                )
                
                logger.info("Diagnostic integration enabled for 2-phase training")
                
            except Exception as e:
                logger.warning(f"Failed to initialize diagnostic integration: {e}")
                self.diagnostic_integration = None
                self.diagnostic_logger = None
        
        # Training state
        self.phase1_completed = False
        self.phase2_completed = False
        self.pretrained_model_path = None
        self.final_model_path = None
        
        # Metrics tracking
        self.training_metrics = {
            'pipeline_start_time': None,
            'pipeline_end_time': None,
            'phase1_start_time': None,
            'phase1_end_time': None,
            'phase2_start_time': None,
            'phase2_end_time': None,
            'phase_transition_time': None
        }
        
        # Log initialization
        self.structured_logger.logger.info("TwoPhaseTrainer initialized")
        self.structured_logger.logger.info(f"Output directory: {self.output_dir}")
        self.structured_logger.logger.info(f"Phase 1 config: {pretrain_config.training_phase}")
        self.structured_logger.logger.info(f"Phase 2 config: {finetune_config.training_phase}")
        
        # Log initial memory usage
        memory_info = self.memory_monitor.get_memory_usage()
        self.structured_logger.log_memory_usage(
            "initialization",
            memory_info["system_memory_mb"] or 0,
            memory_info["gpu_memory_mb"]
        )
    
    def _validate_configurations(self):
        """
        Validate configuration compatibility between phases.
        
        Raises:
            ValueError: If configurations are invalid or incompatible
        """
        # Validate Phase 1 configuration
        if self.pretrain_config.training_phase != "self_supervised":
            raise ValueError(
                f"Phase 1 must use training_phase='self_supervised', "
                f"got: {self.pretrain_config.training_phase}"
            )
        
        # Validate Phase 2 configuration
        if self.finetune_config.training_phase != "fine_tuning":
            raise ValueError(
                f"Phase 2 must use training_phase='fine_tuning', "
                f"got: {self.finetune_config.training_phase}"
            )
        
        # Validate text encoder compatibility
        if self.pretrain_config.text_encoder_model != self.finetune_config.text_encoder_model:
            raise ValueError(
                f"Text encoder models must match between phases. "
                f"Phase 1: {self.pretrain_config.text_encoder_model}, "
                f"Phase 2: {self.finetune_config.text_encoder_model}"
            )
        
        # Validate Phase 1 specific settings
        if not self.pretrain_config.use_augmentation_labels_only:
            logger.warning(
                "Phase 1 should typically use use_augmentation_labels_only=True "
                "for pure self-supervised learning"
            )
        
        if self.pretrain_config.augmentation_positive_ratio == 0.0:
            raise ValueError(
                "Phase 1 augmentation_positive_ratio cannot be 0.0 for self-supervised learning"
            )
        
        # Validate Phase 2 specific settings
        if self.finetune_config.pretrained_model_path is not None:
            logger.warning(
                "Phase 2 pretrained_model_path will be overridden with Phase 1 output"
            )
        
        # Validate compatibility of key parameters
        compatibility_checks = [
            ('batch_size', 'Batch sizes should be compatible for memory consistency'),
            ('text_encoder_device', 'Text encoder devices should match'),
        ]
        
        for param, message in compatibility_checks:
            phase1_val = getattr(self.pretrain_config, param, None)
            phase2_val = getattr(self.finetune_config, param, None)
            
            if phase1_val != phase2_val and phase1_val is not None and phase2_val is not None:
                logger.warning(f"{message}. Phase 1: {phase1_val}, Phase 2: {phase2_val}")
        
        logger.info("Configuration validation completed successfully")
    
    def _save_phase_configs(self):
        """Save configurations for both phases."""
        # Save Phase 1 config
        phase1_config_path = self.phase1_output_dir / "pretraining_config.json"
        with open(phase1_config_path, 'w') as f:
            json.dump(asdict(self.pretrain_config), f, indent=2)
        
        # Save Phase 2 config
        phase2_config_path = self.phase2_output_dir / "finetuning_config.json"
        with open(phase2_config_path, 'w') as f:
            json.dump(asdict(self.finetune_config), f, indent=2)
        
        logger.info(f"Phase configurations saved to {self.output_dir}")
    
    def _get_pretrained_model_path(self) -> str:
        """
        Get the path to the pre-trained model from Phase 1.
        
        Returns:
            Path to the best checkpoint from Phase 1
            
        Raises:
            FileNotFoundError: If no suitable checkpoint is found
        """
        # Look for checkpoints in Phase 1 output directory
        checkpoint_pattern = "checkpoint_*.pt"
        checkpoint_files = list(self.phase1_output_dir.glob(checkpoint_pattern))
        
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {self.phase1_output_dir}. "
                f"Phase 1 may not have completed successfully."
            )
        
        # Find the best checkpoint (lowest loss or most recent)
        best_checkpoint = None
        best_loss = float('inf')
        
        for checkpoint_file in checkpoint_files:
            try:
                # Try to extract loss from filename or load checkpoint to check
                if 'best' in checkpoint_file.name.lower():
                    best_checkpoint = checkpoint_file
                    break
                
                # If no 'best' checkpoint, use the most recent one
                if best_checkpoint is None or checkpoint_file.stat().st_mtime > best_checkpoint.stat().st_mtime:
                    best_checkpoint = checkpoint_file
            
            except Exception as e:
                logger.warning(f"Error checking checkpoint {checkpoint_file}: {e}")
                continue
        
        if best_checkpoint is None:
            raise FileNotFoundError(
                f"No valid checkpoint found in {self.phase1_output_dir}"
            )
        
        logger.info(f"Selected pre-trained model: {best_checkpoint}")
        return str(best_checkpoint)
    
    def run_two_phase_training(self, 
                             unlabeled_data_path: Union[str, Path], 
                             labeled_data_path: Union[str, Path]) -> TwoPhaseResults:
        """
        Execute the complete 2-phase training pipeline.
        
        Args:
            unlabeled_data_path: Path to unlabeled augmented data for self-supervised pre-training
            labeled_data_path: Path to labeled data for supervised fine-tuning
            
        Returns:
            TwoPhaseResults containing comprehensive results from both phases
            
        Raises:
            FileNotFoundError: If data files don't exist
            Exception: If either phase fails
        """
        unlabeled_data_path = Path(unlabeled_data_path)
        labeled_data_path = Path(labeled_data_path)
        
        # Validate input files
        if not unlabeled_data_path.exists():
            raise FileNotFoundError(f"Unlabeled data file not found: {unlabeled_data_path}")
        
        if not labeled_data_path.exists():
            raise FileNotFoundError(f"Labeled data file not found: {labeled_data_path}")
        
        # Start pipeline timing
        self.training_metrics['pipeline_start_time'] = time.time()
        
        # Start training session
        session_id = self.structured_logger.start_training_session({
            "pipeline_type": "two_phase_training",
            "unlabeled_data_path": str(unlabeled_data_path),
            "labeled_data_path": str(labeled_data_path),
            "phase1_config": asdict(self.pretrain_config),
            "phase2_config": asdict(self.finetune_config),
            "output_dir": str(self.output_dir)
        })
        
        # Start enhanced metrics tracking
        self.metrics_tracker.start_pipeline()
        
        try:
            # Save configurations
            self._save_phase_configs()
            
            # Phase 1: Self-supervised pre-training
            logger.info("=" * 60)
            logger.info("STARTING PHASE 1: SELF-SUPERVISED PRE-TRAINING")
            logger.info("=" * 60)
            
            # Start Phase 1 metrics and diagnostics
            self.metrics_tracker.start_phase1("self_supervised_pretraining")
            
            if self.diagnostic_logger:
                self.diagnostic_logger.log_metric("phase_start", "self_supervised", category="pipeline")
            
            phase1_results = self._run_phase1(unlabeled_data_path)
            
            # End Phase 1 metrics
            self.metrics_tracker.end_phase1()
            
            if self.diagnostic_logger:
                self.diagnostic_logger.log_metric("phase_end", "self_supervised", category="pipeline")
                self.diagnostic_logger.log_metric("phase1_final_loss", phase1_results.final_loss, category="results")
            
            # Phase transition
            transition_start = time.time()
            self.pretrained_model_path = self._get_pretrained_model_path()
            
            # Update Phase 2 config with pre-trained model path
            self.finetune_config.pretrained_model_path = self.pretrained_model_path
            
            self.training_metrics['phase_transition_time'] = time.time() - transition_start
            
            logger.info("=" * 60)
            logger.info("STARTING PHASE 2: SUPERVISED FINE-TUNING")
            logger.info("=" * 60)
            
            # Start Phase 2 metrics and diagnostics
            self.metrics_tracker.start_phase2("supervised_finetuning")
            
            if self.diagnostic_logger:
                self.diagnostic_logger.log_metric("phase_start", "fine_tuning", category="pipeline")
            
            # Phase 2: Supervised fine-tuning
            phase2_results = self._run_phase2(labeled_data_path)
            
            # End Phase 2 metrics
            self.metrics_tracker.end_phase2()
            
            if self.diagnostic_logger:
                self.diagnostic_logger.log_metric("phase_end", "fine_tuning", category="pipeline")
                self.diagnostic_logger.log_metric("phase2_final_accuracy", phase2_results.get('final_accuracy', 0.0), category="results")
            
            # Pipeline completion
            self.training_metrics['pipeline_end_time'] = time.time()
            total_training_time = (
                self.training_metrics['pipeline_end_time'] - 
                self.training_metrics['pipeline_start_time']
            )
            
            # Create comprehensive results
            results = TwoPhaseResults(
                pretraining_results=phase1_results,
                finetuning_results=phase2_results,
                pretrained_model_path=self.pretrained_model_path,
                final_model_path=phase2_results.get('final_model_path', ''),
                phase_transition_time=self.training_metrics['phase_transition_time'],
                total_training_time=total_training_time,
                phase1_config=asdict(self.pretrain_config),
                phase2_config=asdict(self.finetune_config)
            )
            
            # Save comprehensive results
            self._save_pipeline_results(results)
            
            # End enhanced metrics tracking
            self.metrics_tracker.end_pipeline()
            
            # Generate comprehensive diagnostic report
            if self.diagnostic_logger:
                self.diagnostic_logger.log_metric("pipeline_completed", True, category="pipeline")
                self.diagnostic_logger.log_metric("total_training_time", total_training_time, category="performance")
                
                # Export metrics for analysis
                try:
                    metrics_csv_path = self.output_dir / "training_metrics.csv"
                    self.diagnostic_logger.export_metrics_csv(metrics_csv_path)
                    logger.info(f"Training metrics exported to {metrics_csv_path}")
                except Exception as e:
                    logger.warning(f"Failed to export metrics: {e}")
            
            # Stop diagnostic integration
            if self.diagnostic_integration:
                try:
                    self.diagnostic_integration.stop_async_diagnostics()
                    
                    # Get performance stats
                    perf_stats = self.diagnostic_integration.get_performance_stats()
                    logger.info(f"Diagnostic performance: {perf_stats}")
                    
                except Exception as e:
                    logger.warning(f"Error stopping diagnostic integration: {e}")
            
            # End training session
            self.structured_logger.end_training_session({
                "pipeline_completed": True,
                "total_training_time": total_training_time,
                "phase1_final_loss": phase1_results.final_loss,
                "phase2_final_accuracy": phase2_results.get('final_accuracy', 0.0),
                "pretrained_model_path": self.pretrained_model_path,
                "final_model_path": results.final_model_path
            })
            
            logger.info("=" * 60)
            logger.info("2-PHASE TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total training time: {total_training_time:.2f} seconds")
            logger.info(f"Phase 1 final loss: {phase1_results.final_loss:.4f}")
            logger.info(f"Phase 2 final accuracy: {phase2_results.get('final_accuracy', 0.0):.4f}")
            logger.info(f"Pre-trained model: {self.pretrained_model_path}")
            logger.info(f"Final model: {results.final_model_path}")
            
            return results
        
        except Exception as e:
            # Handle pipeline failure
            logger.error(f"2-phase training pipeline failed: {e}")
            
            # Save emergency state
            try:
                self._save_emergency_state(e)
            except Exception as save_error:
                logger.error(f"Failed to save emergency state: {save_error}")
            
            # End session with error
            self.structured_logger.end_training_session({
                "pipeline_failed": True,
                "error": str(e),
                "phase1_completed": self.phase1_completed,
                "phase2_completed": self.phase2_completed,
                "pretrained_model_path": self.pretrained_model_path
            })
            
            raise
    
    def _run_phase1(self, unlabeled_data_path: Path) -> TrainingResults:
        """
        Execute Phase 1: Self-supervised pre-training.
        
        Args:
            unlabeled_data_path: Path to unlabeled augmented data
            
        Returns:
            TrainingResults from pre-training phase
            
        Raises:
            Exception: If Phase 1 training fails
        """
        self.training_metrics['phase1_start_time'] = time.time()
        
        try:
            # Initialize contrastive learning trainer for Phase 1
            phase1_trainer = ContrastiveLearningTrainer(
                config=self.pretrain_config,
                output_dir=str(self.phase1_output_dir)
            )
            
            logger.info(f"Phase 1 trainer initialized with config: {self.pretrain_config.training_phase}")
            logger.info(f"Using augmentation labels only: {self.pretrain_config.use_augmentation_labels_only}")
            logger.info(f"Augmentation positive ratio: {self.pretrain_config.augmentation_positive_ratio}")
            
            # Run self-supervised pre-training
            phase1_results = phase1_trainer.train(unlabeled_data_path)
            
            self.training_metrics['phase1_end_time'] = time.time()
            phase1_time = (
                self.training_metrics['phase1_end_time'] - 
                self.training_metrics['phase1_start_time']
            )
            
            self.phase1_completed = True
            
            logger.info(f"Phase 1 completed successfully in {phase1_time:.2f} seconds")
            logger.info(f"Phase 1 final loss: {phase1_results.final_loss:.4f}")
            logger.info(f"Phase 1 total samples: {phase1_results.total_samples}")
            
            return phase1_results
        
        except Exception as e:
            logger.error(f"Phase 1 (self-supervised pre-training) failed: {e}")
            raise Exception(f"Phase 1 failed: {e}") from e
    
    def _run_phase2(self, labeled_data_path: Path) -> Dict[str, Any]:
        """
        Execute Phase 2: Supervised fine-tuning.
        
        Args:
            labeled_data_path: Path to labeled data for fine-tuning
            
        Returns:
            Dictionary containing fine-tuning results
            
        Raises:
            Exception: If Phase 2 training fails
        """
        self.training_metrics['phase2_start_time'] = time.time()
        
        try:
            # Initialize fine-tuning trainer for Phase 2
            phase2_trainer = FineTuningTrainer(
                config=self.finetune_config,
                output_dir=str(self.phase2_output_dir)
            )
            
            logger.info(f"Phase 2 trainer initialized with pre-trained model: {self.pretrained_model_path}")
            logger.info(f"Freeze contrastive layers: {self.finetune_config.freeze_contrastive_layers}")
            logger.info(f"Classification dropout: {self.finetune_config.classification_dropout}")
            
            # Run supervised fine-tuning
            phase2_results = phase2_trainer.train(labeled_data_path)
            
            self.training_metrics['phase2_end_time'] = time.time()
            phase2_time = (
                self.training_metrics['phase2_end_time'] - 
                self.training_metrics['phase2_start_time']
            )
            
            self.phase2_completed = True
            
            # Set final model path
            self.final_model_path = str(self.phase2_output_dir / "final_model.pt")
            phase2_results['final_model_path'] = self.final_model_path
            
            logger.info(f"Phase 2 completed successfully in {phase2_time:.2f} seconds")
            logger.info(f"Phase 2 final accuracy: {phase2_results.get('final_accuracy', 0.0):.4f}")
            logger.info(f"Phase 2 best accuracy: {phase2_results.get('best_accuracy', 0.0):.4f}")
            
            return phase2_results
        
        except Exception as e:
            logger.error(f"Phase 2 (supervised fine-tuning) failed: {e}")
            raise Exception(f"Phase 2 failed: {e}") from e
    
    def _save_pipeline_results(self, results: TwoPhaseResults):
        """Save comprehensive pipeline results."""
        results_path = self.output_dir / "two_phase_results.json"
        results.save_json(str(results_path))
        
        # Also save a summary report
        summary_path = self.output_dir / "pipeline_summary.json"
        summary = {
            "pipeline_type": "two_phase_training",
            "total_training_time": results.total_training_time,
            "phase_transition_time": results.phase_transition_time,
            "phase1_summary": {
                "final_loss": results.pretraining_results.final_loss,
                "training_time": results.pretraining_results.training_time,
                "total_samples": results.pretraining_results.total_samples,
                "total_batches": results.pretraining_results.total_batches
            },
            "phase2_summary": {
                "final_accuracy": results.finetuning_results.get('final_accuracy', 0.0),
                "best_accuracy": results.finetuning_results.get('best_accuracy', 0.0),
                "final_loss": results.finetuning_results.get('final_loss', 0.0),
                "training_time": results.finetuning_results.get('training_time', 0.0),
                "total_samples": results.finetuning_results.get('total_samples', 0)
            },
            "model_paths": {
                "pretrained_model": results.pretrained_model_path,
                "final_model": results.final_model_path
            },
            "configurations": {
                "phase1_training_phase": results.phase1_config.get('training_phase'),
                "phase1_use_augmentation_labels_only": results.phase1_config.get('use_augmentation_labels_only'),
                "phase1_augmentation_positive_ratio": results.phase1_config.get('augmentation_positive_ratio'),
                "phase2_training_phase": results.phase2_config.get('training_phase'),
                "phase2_freeze_contrastive_layers": results.phase2_config.get('freeze_contrastive_layers'),
                "phase2_classification_dropout": results.phase2_config.get('classification_dropout')
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Pipeline results saved to {results_path}")
        logger.info(f"Pipeline summary saved to {summary_path}")
    
    def _save_emergency_state(self, error: Exception):
        """Save emergency state when pipeline fails."""
        emergency_state = {
            "error": str(error),
            "error_type": type(error).__name__,
            "phase1_completed": self.phase1_completed,
            "phase2_completed": self.phase2_completed,
            "pretrained_model_path": self.pretrained_model_path,
            "final_model_path": self.final_model_path,
            "training_metrics": self.training_metrics,
            "phase1_config": asdict(self.pretrain_config),
            "phase2_config": asdict(self.finetune_config)
        }
        
        emergency_path = self.output_dir / "emergency_state.json"
        with open(emergency_path, 'w') as f:
            json.dump(emergency_state, f, indent=2)
        
        logger.info(f"Emergency state saved to {emergency_path}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status and progress.
        
        Returns:
            Dictionary containing current training status
        """
        return {
            "phase1_completed": self.phase1_completed,
            "phase2_completed": self.phase2_completed,
            "pretrained_model_path": self.pretrained_model_path,
            "final_model_path": self.final_model_path,
            "training_metrics": self.training_metrics,
            "output_directories": {
                "base": str(self.output_dir),
                "phase1": str(self.phase1_output_dir),
                "phase2": str(self.phase2_output_dir)
            }
        }
    
    def resume_from_checkpoint(self, checkpoint_dir: Union[str, Path]) -> bool:
        """
        Resume training from a previous checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint state
            
        Returns:
            True if resume was successful, False otherwise
        """
        checkpoint_dir = Path(checkpoint_dir)
        emergency_state_path = checkpoint_dir / "emergency_state.json"
        
        if not emergency_state_path.exists():
            logger.error(f"No emergency state found in {checkpoint_dir}")
            return False
        
        try:
            with open(emergency_state_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.phase1_completed = state.get('phase1_completed', False)
            self.phase2_completed = state.get('phase2_completed', False)
            self.pretrained_model_path = state.get('pretrained_model_path')
            self.final_model_path = state.get('final_model_path')
            self.training_metrics.update(state.get('training_metrics', {}))
            
            logger.info(f"Successfully resumed from checkpoint: {checkpoint_dir}")
            logger.info(f"Phase 1 completed: {self.phase1_completed}")
            logger.info(f"Phase 2 completed: {self.phase2_completed}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return False