"""
Training Strategy Pattern Implementation for ML Pipeline.

This module defines the abstract base class and concrete implementations for different
training strategies (single-phase and two-phase) to provide a unified interface
for the ML Pipeline orchestrator.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING

from .data_structures import TrainingConfig, TrainingResults
from .pipeline_config import PipelineConfig

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from .evaluator import ContrastiveEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainingStrategyResult:
    """
    Unified result format for all training strategies.
    
    This class provides a consistent interface for training results regardless
    of whether single-phase or two-phase training was used.
    """
    # Core training results
    final_model_path: str
    best_model_path: Optional[str] = None
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    validation_metrics: Optional[Dict[str, Any]] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    
    # Training metadata
    training_mode: str = "single_phase"  # "single_phase" or "two_phase"
    total_training_time: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    # Two-phase specific results (None for single-phase)
    phase1_results: Optional[TrainingResults] = None
    phase2_results: Optional[Dict[str, Any]] = None
    phase_transition_metrics: Optional[Dict[str, Any]] = None
    comparative_analysis: Optional[Dict[str, Any]] = None
    pretrained_model_path: Optional[str] = None
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'final_model_path': self.final_model_path,
            'best_model_path': self.best_model_path,
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'validation_metrics': self.validation_metrics,
            'checkpoint_paths': self.checkpoint_paths,
            'training_mode': self.training_mode,
            'total_training_time': self.total_training_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'errors': self.errors,
            'warnings': self.warnings
        }
        
        # Add two-phase specific fields if present
        if self.training_mode == "two_phase":
            result.update({
                'phase1_results': self.phase1_results.to_dict() if self.phase1_results else None,
                'phase2_results': self.phase2_results,
                'phase_transition_metrics': self.phase_transition_metrics,
                'comparative_analysis': self.comparative_analysis,
                'pretrained_model_path': self.pretrained_model_path
            })
        
        return result
    
    def has_errors(self) -> bool:
        """Check if there were any errors during training."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there were any warnings during training."""
        return len(self.warnings) > 0
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        logger.error(f"Training strategy error: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(f"Training strategy warning: {warning}")


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.
    
    This class defines the common interface that all training strategies must implement,
    providing unified error handling, logging, and validation functionality.
    """
    
    def __init__(self, config: PipelineConfig, output_dir: Path):
        """
        Initialize the training strategy.
        
        Args:
            config: Pipeline configuration containing training parameters
            output_dir: Directory for saving training outputs
        """
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize result tracking
        self.result = TrainingStrategyResult(final_model_path="")
        
        # Validate configuration
        validation_errors = self.validate_configuration()
        if validation_errors:
            for error in validation_errors:
                self.result.add_error(error)
            raise ValueError(f"Configuration validation failed: {validation_errors}")
    
    @abstractmethod
    def execute_training(self, data_splits: Dict[str, str]) -> TrainingStrategyResult:
        """
        Execute the training strategy with the provided data splits.
        
        Args:
            data_splits: Dictionary containing paths to training data splits
                        Expected keys: 'train', 'validation', 'test'
        
        Returns:
            TrainingStrategyResult containing comprehensive training results
            
        Raises:
            ValueError: If data splits are invalid
            FileNotFoundError: If required data files don't exist
            Exception: If training fails
        """
        pass
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the configuration for this training strategy.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Basic configuration validation
        if not self.config.training_config_path:
            errors.append("training_config_path is required")
        
        if self.config.training_overrides and not isinstance(self.config.training_overrides, dict):
            errors.append("training_overrides must be a dictionary")
        
        return errors
    
    def validate_data_splits(self, data_splits: Dict[str, str]) -> List[str]:
        """
        Validate the provided data splits.
        
        Args:
            data_splits: Dictionary containing data split paths
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required splits
        required_splits = ['train', 'validation']
        for split_name in required_splits:
            if split_name not in data_splits:
                errors.append(f"Missing required data split: {split_name}")
            elif not data_splits[split_name]:
                errors.append(f"Empty path for data split: {split_name}")
            else:
                # Check if file exists
                split_path = Path(data_splits[split_name])
                if not split_path.exists():
                    errors.append(f"Data file not found: {data_splits[split_name]}")
        
        return errors
    
    def _setup_training_logging(self, strategy_name: str) -> None:
        """
        Setup logging for training execution.
        
        Args:
            strategy_name: Name of the training strategy for log identification
        """
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING {strategy_name.upper()} TRAINING STRATEGY")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Configuration: {self.config.experiment_name}")
    
    def _finalize_training_results(self, start_time: float, end_time: float) -> None:
        """
        Finalize training results with timing and metadata.
        
        Args:
            start_time: Training start timestamp
            end_time: Training end timestamp
        """
        from datetime import datetime
        
        self.result.total_training_time = end_time - start_time
        self.result.start_time = datetime.fromtimestamp(start_time).isoformat()
        self.result.end_time = datetime.fromtimestamp(end_time).isoformat()
        
        # Log completion
        self.logger.info("=" * 80)
        self.logger.info("TRAINING STRATEGY COMPLETED")
        self.logger.info(f"Total training time: {self.result.total_training_time:.2f} seconds")
        self.logger.info(f"Final model: {self.result.final_model_path}")
        if self.result.best_model_path:
            self.logger.info(f"Best model: {self.result.best_model_path}")
        self.logger.info("=" * 80)
    
    def _handle_training_error(self, error: Exception, phase: str = "training") -> None:
        """
        Handle training errors with proper logging and result tracking.
        
        Args:
            error: The exception that occurred
            phase: The phase where the error occurred
        """
        error_msg = f"Error during {phase}: {str(error)}"
        self.result.add_error(error_msg)
        self.logger.error(error_msg, exc_info=True)
    
    def _load_training_config(self) -> TrainingConfig:
        """
        Load and validate the training configuration.
        
        Returns:
            TrainingConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(self.config.training_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Training config not found: {config_path}")
        
        try:
            training_config = TrainingConfig.from_json(str(config_path))
            
            # Apply overrides
            for key, value in self.config.training_overrides.items():
                if hasattr(training_config, key):
                    setattr(training_config, key, value)
                    self.logger.info(f"Applied override: {key} = {value}")
                else:
                    self.result.add_warning(f"Unknown override parameter: {key}")
            
            return training_config
            
        except Exception as e:
            raise ValueError(f"Failed to load training config: {e}")
    
    def _save_strategy_results(self) -> str:
        """
        Save the training strategy results to file.
        
        Returns:
            Path to the saved results file
        """
        results_path = self.output_dir / "training_strategy_results.json"
        
        try:
            import json
            with open(results_path, 'w') as f:
                json.dump(self.result.to_dict(), f, indent=2)
            
            self.logger.info(f"Training strategy results saved to: {results_path}")
            return str(results_path)
            
        except Exception as e:
            self.result.add_warning(f"Failed to save results: {e}")
            return ""


class SinglePhaseStrategy(TrainingStrategy):
    """
    Single-phase training strategy using ContrastiveLearningTrainer.
    
    This strategy wraps the existing contrastive learning training logic
    to maintain backward compatibility while providing the unified interface.
    """
    
    def __init__(self, config: PipelineConfig, output_dir: Path):
        """
        Initialize the single-phase training strategy.
        
        Args:
            config: Pipeline configuration
            output_dir: Directory for saving training outputs
        """
        super().__init__(config, output_dir)
        self.result.training_mode = "single_phase"
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration for single-phase training.
        
        Returns:
            List of validation error messages
        """
        errors = super().validate_configuration()
        
        # Single-phase specific validation
        if hasattr(self.config, 'two_phase_training') and self.config.two_phase_training.enabled:
            errors.append("SinglePhaseStrategy cannot be used with two-phase training enabled")
        
        return errors
    
    def execute_training(self, data_splits: Dict[str, str]) -> TrainingStrategyResult:
        """
        Execute single-phase contrastive learning training.
        
        Args:
            data_splits: Dictionary containing data split paths
            
        Returns:
            TrainingStrategyResult with single-phase training results
        """
        start_time = time.time()
        
        try:
            # Setup logging
            self._setup_training_logging("SINGLE-PHASE CONTRASTIVE LEARNING")
            
            # Validate data splits
            validation_errors = self.validate_data_splits(data_splits)
            if validation_errors:
                for error in validation_errors:
                    self.result.add_error(error)
                raise ValueError(f"Data validation failed: {validation_errors}")
            
            # Load training configuration
            training_config = self._load_training_config()
            
            # Execute training with validation
            training_results = self._run_single_phase_training(
                training_config, 
                data_splits['train'], 
                data_splits['validation']
            )
            
            # Populate result
            self.result.final_model_path = training_results['final_model_path']
            self.result.best_model_path = training_results['best_model_path']
            self.result.training_history = training_results['training_history']
            self.result.validation_history = training_results['validation_history']
            self.result.validation_metrics = training_results['validation_metrics']
            self.result.checkpoint_paths = training_results['checkpoint_paths']
            
            # Finalize results
            end_time = time.time()
            self._finalize_training_results(start_time, end_time)
            
            # Save results
            self._save_strategy_results()
            
            return self.result
            
        except Exception as e:
            self._handle_training_error(e, "single-phase training")
            end_time = time.time()
            self._finalize_training_results(start_time, end_time)
            raise
    
    def _run_single_phase_training(self, training_config: TrainingConfig, 
                                 train_data_path: str, val_data_path: str) -> Dict[str, Any]:
        """
        Run the single-phase training using ContrastiveLearningTrainer.
        
        Args:
            training_config: Training configuration
            train_data_path: Path to training data
            val_data_path: Path to validation data
            
        Returns:
            Dictionary containing training results
        """
        from .trainer import ContrastiveLearningTrainer
        from .evaluator import ContrastiveEvaluator, EvaluationConfig
        from torch.utils.data import DataLoader
        from datetime import datetime
        
        # Setup trainer
        training_output_dir = self.output_dir / "training"
        training_output_dir.mkdir(exist_ok=True)
        
        trainer = ContrastiveLearningTrainer(
            training_config, 
            output_dir=str(training_output_dir)
        )
        
        # Create validation data loader
        val_loader = self._create_data_loader(val_data_path, training_config.batch_size)
        
        # Setup validation evaluator
        eval_config = EvaluationConfig(
            metrics=self.config.validation.metrics,
            save_embeddings=False,  # Don't save embeddings during validation
            save_predictions=self.config.validation.save_predictions,
            generate_visualizations=self.config.validation.generate_plots,
            batch_size=self.config.validation.batch_size or training_config.batch_size,
            device=self.config.device
        )
        evaluator = ContrastiveEvaluator(eval_config, text_encoder=trainer.text_encoder)
        
        # Run enhanced training loop with validation
        return self._enhanced_training_loop(
            trainer, train_data_path, val_loader, evaluator, training_output_dir
        )
    
    def _enhanced_training_loop(self, trainer, train_data_path: str, val_loader,
                              evaluator, output_dir: Path) -> Dict[str, Any]:
        """
        Enhanced training loop with validation and early stopping.
        
        This method replicates the training loop from MLPipeline to ensure
        identical behavior for backward compatibility.
        """
        # Initialize tracking variables
        best_metric = float('-inf') if self.config.early_stopping.mode == 'max' else float('inf')
        patience_counter = 0
        training_history = []
        validation_history = []
        checkpoint_paths = []
        
        self.logger.info(f"Starting training with early stopping:")
        self.logger.info(f"   Metric: {self.config.early_stopping.metric}")
        self.logger.info(f"   Patience: {self.config.early_stopping.patience}")
        self.logger.info(f"   Mode: {self.config.early_stopping.mode}")
        
        # Start training
        epoch = 0
        training_complete = False
        
        try:
            while epoch < trainer.config.num_epochs and not training_complete:
                epoch_start_time = time.time()
                
                self.logger.info(f"Epoch {epoch + 1}/{trainer.config.num_epochs}")
                
                # Training phase
                train_metrics = trainer.train_epoch(train_data_path, epoch)
                
                # Validation phase
                val_metrics = None
                if self._should_validate(epoch):
                    self.logger.info("Running validation...")
                    val_output_dir = output_dir / f"validation_epoch_{epoch + 1}"
                    val_results = evaluator.evaluate_model(
                        trainer.model, val_loader, str(val_output_dir)
                    )
                    val_metrics = val_results.metrics
                    
                    validation_history.append({
                        'epoch': epoch + 1,
                        'metrics': val_metrics,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Record training history
                epoch_duration = time.time() - epoch_start_time
                epoch_record = {
                    'epoch': epoch + 1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'duration': epoch_duration,
                    'timestamp': datetime.now().isoformat()
                }
                training_history.append(epoch_record)
                
                # Checkpointing
                if self._should_save_checkpoint(epoch):
                    checkpoint_path = self._save_checkpoint(
                        trainer, epoch, train_metrics, val_metrics, output_dir
                    )
                    checkpoint_paths.append(checkpoint_path)
                
                # Early stopping logic
                if val_metrics and self.config.early_stopping.enabled:
                    current_metric = val_metrics.get(self.config.early_stopping.metric)
                    
                    if current_metric is not None:
                        improved = self._check_improvement(current_metric, best_metric)
                        
                        if improved:
                            best_metric = current_metric
                            patience_counter = 0
                            
                            # Save best model
                            best_model_path = self._save_best_checkpoint(
                                trainer, epoch, val_metrics, output_dir
                            )
                            self.logger.info(
                                f"New best model saved: {self.config.early_stopping.metric} = {current_metric:.4f}"
                            )
                        else:
                            patience_counter += 1
                            self.logger.info(f"Patience: {patience_counter}/{self.config.early_stopping.patience}")
                            
                            if patience_counter >= self.config.early_stopping.patience:
                                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                                training_complete = True
                
                epoch += 1
            
            # Save final model
            final_model_path = self._save_final_checkpoint(trainer, epoch - 1, output_dir)
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            final_model_path = self._save_checkpoint(trainer, epoch, {}, {}, output_dir)
        
        # Find best model path
        best_model_path = None
        for checkpoint_path in checkpoint_paths:
            if 'best' in checkpoint_path:
                best_model_path = checkpoint_path
                break
        
        return {
            'final_model_path': final_model_path,
            'best_model_path': best_model_path,
            'training_history': training_history,
            'validation_history': validation_history,
            'validation_metrics': validation_history[-1]['metrics'] if validation_history else None,
            'checkpoint_paths': checkpoint_paths
        }
    
    def _create_data_loader(self, data_path: str, batch_size: int):
        """Create data loader for given dataset."""
        from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader
        import json
        
        self.logger.info(f"Creating data loader for: {data_path}")
        
        class JSONLDataset(Dataset):
            def __init__(self, file_path):
                self.samples = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                sample = json.loads(line)
                                # Convert to expected format
                                processed_sample = {
                                    'resume': sample.get('resume_text', ''),
                                    'job': sample.get('job_text', ''),
                                    'label': sample.get('label', 1)
                                }
                                self.samples.append(processed_sample)
                            except json.JSONDecodeError:
                                continue
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        try:
            dataset = JSONLDataset(data_path)
            if len(dataset) == 0:
                self.logger.warning(f"No valid samples found in {data_path}")
                return None
            
            data_loader = PyTorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # Don't shuffle validation data
                num_workers=0   # Use single thread for compatibility
            )
            
            self.logger.info(f"Created data loader with {len(dataset)} samples")
            return data_loader
            
        except Exception as e:
            self.logger.error(f"Failed to create data loader: {e}")
            return None
    
    def _should_validate(self, epoch: int) -> bool:
        """Check if validation should be run for this epoch."""
        if self.config.validation.frequency == "every_epoch":
            return True
        elif self.config.validation.frequency == "every_n_epochs":
            return (epoch + 1) % self.config.validation.frequency_value == 0
        return False
    
    def _should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved for this epoch."""
        return (epoch + 1) % self.config.checkpointing.save_frequency == 0
    
    def _save_checkpoint(self, trainer, epoch: int, train_metrics: Dict, 
                        val_metrics: Dict, output_dir: Path) -> str:
        """Save model checkpoint."""
        combined_metrics = {**train_metrics, **val_metrics}
        checkpoint_path = trainer.save_checkpoint(epoch, combined_metrics)
        self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")
        return checkpoint_path
    
    def _save_best_checkpoint(self, trainer, epoch: int, val_metrics: Dict, output_dir: Path) -> str:
        """Save best model checkpoint."""
        checkpoint_path = trainer.save_checkpoint(epoch, val_metrics)
        return checkpoint_path
    
    def _save_final_checkpoint(self, trainer, epoch: int, output_dir: Path) -> str:
        """Save final model checkpoint."""
        final_metrics = {"epoch": epoch, "final": True}
        checkpoint_path = trainer.save_checkpoint(epoch, final_metrics)
        return checkpoint_path
    
    def _check_improvement(self, current_metric: float, best_metric: float) -> bool:
        """Check if current metric is an improvement over best metric."""
        if self.config.early_stopping.mode == 'max':
            return current_metric > best_metric + self.config.early_stopping.min_delta
        else:
            return current_metric < best_metric - self.config.early_stopping.min_delta


class TwoPhaseStrategy(TrainingStrategy):
    """
    Two-phase training strategy using TwoPhaseTrainer.
    
    This strategy orchestrates the execution of both self-supervised pre-training
    and supervised fine-tuning phases with checkpoint transfer and comprehensive
    metrics collection.
    """
    
    def __init__(self, config: PipelineConfig, output_dir: Path):
        """
        Initialize the two-phase training strategy.
        
        Args:
            config: Pipeline configuration with two-phase training enabled
            output_dir: Directory for saving training outputs
        """
        super().__init__(config, output_dir)
        self.result.training_mode = "two_phase"
        
        # Validate two-phase configuration
        if not hasattr(config, 'two_phase_training') or not config.two_phase_training.enabled:
            raise ValueError("TwoPhaseStrategy requires two_phase_training to be enabled in config")
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration for two-phase training.
        
        Returns:
            List of validation error messages
        """
        errors = super().validate_configuration()
        
        # Two-phase specific validation
        if not hasattr(self.config, 'two_phase_training'):
            errors.append("two_phase_training configuration is required")
            return errors
        
        two_phase_config = self.config.two_phase_training
        
        # Validate two-phase configuration
        config_errors = two_phase_config.validate_config()
        errors.extend(config_errors)
        
        # Check phase configuration files exist
        if two_phase_config.phase1_config_path:
            phase1_path = Path(two_phase_config.phase1_config_path)
            if not phase1_path.exists():
                errors.append(f"Phase 1 config file not found: {two_phase_config.phase1_config_path}")
        
        if two_phase_config.phase2_config_path:
            phase2_path = Path(two_phase_config.phase2_config_path)
            if not phase2_path.exists():
                errors.append(f"Phase 2 config file not found: {two_phase_config.phase2_config_path}")
        
        return errors
    
    def validate_data_splits(self, data_splits: Dict[str, str]) -> List[str]:
        """
        Validate data splits for two-phase training.
        
        Args:
            data_splits: Dictionary containing data split paths
            
        Returns:
            List of validation error messages
        """
        errors = super().validate_data_splits(data_splits)
        
        # Two-phase training may need different data requirements
        # depending on the data strategy configuration
        two_phase_config = self.config.two_phase_training
        
        # Check if we need augmented data for phase 1
        if two_phase_config.phase1_data_strategy == "augmentation_only":
            # We'll need to ensure the training data contains augmented samples
            # This validation could be enhanced to check data format
            pass
        
        # Check if we need labeled data for phase 2
        if two_phase_config.phase2_data_strategy == "labeled_only":
            # We'll need to ensure the training data contains labeled samples
            # This validation could be enhanced to check data format
            pass
        
        return errors
    
    def execute_training(self, data_splits: Dict[str, str]) -> TrainingStrategyResult:
        """
        Execute two-phase training strategy.
        
        Args:
            data_splits: Dictionary containing data split paths
            
        Returns:
            TrainingStrategyResult with two-phase training results
        """
        start_time = time.time()
        
        try:
            # Setup logging
            self._setup_training_logging("TWO-PHASE CONTRASTIVE LEARNING")
            
            # Validate data splits
            validation_errors = self.validate_data_splits(data_splits)
            if validation_errors:
                for error in validation_errors:
                    self.result.add_error(error)
                raise ValueError(f"Data validation failed: {validation_errors}")
            
            # Load phase configurations
            phase1_config, phase2_config = self._load_phase_configurations()
            
            # Validate phase compatibility
            compatibility_errors = self._validate_phase_compatibility(phase1_config, phase2_config)
            if compatibility_errors:
                for error in compatibility_errors:
                    self.result.add_error(error)
                raise ValueError(f"Phase compatibility validation failed: {compatibility_errors}")
            
            # Execute two-phase training
            two_phase_results = self._run_two_phase_training(
                phase1_config, phase2_config, data_splits
            )
            
            # Populate unified result structure
            self._populate_result_from_two_phase(two_phase_results)
            
            # Generate comparative analysis
            self._generate_comparative_analysis(two_phase_results)
            
            # Finalize results
            end_time = time.time()
            self._finalize_training_results(start_time, end_time)
            
            # Save results
            self._save_strategy_results()
            
            return self.result
            
        except Exception as e:
            self._handle_training_error(e, "two-phase training")
            end_time = time.time()
            self._finalize_training_results(start_time, end_time)
            raise
    
    def _load_phase_configurations(self) -> tuple[TrainingConfig, TrainingConfig]:
        """
        Load configurations for both training phases.
        
        Returns:
            Tuple of (phase1_config, phase2_config)
        """
        two_phase_config = self.config.two_phase_training
        
        # Load Phase 1 configuration
        phase1_path = Path(two_phase_config.phase1_config_path)
        if not phase1_path.exists():
            raise FileNotFoundError(f"Phase 1 config not found: {phase1_path}")
        
        phase1_config = TrainingConfig.from_json(str(phase1_path))
        self.logger.info(f"Loaded Phase 1 config: {phase1_path}")
        
        # Load Phase 2 configuration
        phase2_path = Path(two_phase_config.phase2_config_path)
        if not phase2_path.exists():
            raise FileNotFoundError(f"Phase 2 config not found: {phase2_path}")
        
        phase2_config = TrainingConfig.from_json(str(phase2_path))
        self.logger.info(f"Loaded Phase 2 config: {phase2_path}")
        
        # Apply any overrides from pipeline config
        for key, value in self.config.training_overrides.items():
            if hasattr(phase1_config, key):
                setattr(phase1_config, key, value)
                self.logger.info(f"Applied Phase 1 override: {key} = {value}")
            if hasattr(phase2_config, key):
                setattr(phase2_config, key, value)
                self.logger.info(f"Applied Phase 2 override: {key} = {value}")
        
        return phase1_config, phase2_config
    
    def _validate_phase_compatibility(self, phase1_config: TrainingConfig, 
                                    phase2_config: TrainingConfig) -> List[str]:
        """
        Validate compatibility between phase configurations.
        
        Args:
            phase1_config: Phase 1 training configuration
            phase2_config: Phase 2 training configuration
            
        Returns:
            List of compatibility error messages
        """
        errors = []
        
        # Check text encoder compatibility
        if phase1_config.text_encoder_model != phase2_config.text_encoder_model:
            errors.append(
                f"Text encoder mismatch: Phase 1 uses '{phase1_config.text_encoder_model}', "
                f"Phase 2 uses '{phase2_config.text_encoder_model}'"
            )
        
        # Check projection dimension compatibility (if both have it)
        if (hasattr(phase1_config, 'projection_dim') and 
            hasattr(phase2_config, 'projection_dim')):
            if phase1_config.projection_dim != phase2_config.projection_dim:
                errors.append(
                    f"Projection dimension mismatch: Phase 1 uses {phase1_config.projection_dim}, "
                    f"Phase 2 uses {phase2_config.projection_dim}"
                )
        
        # Validate training phases
        if phase1_config.training_phase not in ["self_supervised", "supervised"]:
            errors.append(f"Invalid Phase 1 training_phase: {phase1_config.training_phase}")
        
        if phase2_config.training_phase not in ["fine_tuning", "supervised"]:
            errors.append(f"Invalid Phase 2 training_phase: {phase2_config.training_phase}")
        
        # Check that Phase 2 is configured for fine-tuning if Phase 1 is self-supervised
        if (phase1_config.training_phase == "self_supervised" and 
            phase2_config.training_phase != "fine_tuning"):
            self.result.add_warning(
                "Phase 1 is self-supervised but Phase 2 is not configured for fine-tuning"
            )
        
        return errors
    
    def _run_two_phase_training(self, phase1_config: TrainingConfig, phase2_config: TrainingConfig,
                              data_splits: Dict[str, str]) -> 'TwoPhaseResults':
        """
        Execute the two-phase training using TwoPhaseTrainer.
        
        Args:
            phase1_config: Configuration for Phase 1
            phase2_config: Configuration for Phase 2
            data_splits: Data split paths
            
        Returns:
            TwoPhaseResults from the training
        """
        from .two_phase_trainer import TwoPhaseTrainer
        
        # Setup output directories for phases
        phase1_output_dir = self.output_dir / "phase1_pretraining"
        phase2_output_dir = self.output_dir / "phase2_finetuning"
        
        # Initialize TwoPhaseTrainer
        trainer = TwoPhaseTrainer(
            pretrain_config=phase1_config,
            finetune_config=phase2_config,
            output_dir=str(self.output_dir)
        )
        
        # Determine data paths based on strategy
        unlabeled_data_path = self._get_phase1_data_path(data_splits)
        labeled_data_path = self._get_phase2_data_path(data_splits)
        
        self.logger.info(f"Phase 1 data: {unlabeled_data_path}")
        self.logger.info(f"Phase 2 data: {labeled_data_path}")
        
        # Execute two-phase training
        results = trainer.run_two_phase_training(
            unlabeled_data_path=unlabeled_data_path,
            labeled_data_path=labeled_data_path
        )
        
        return results
    
    def _get_phase1_data_path(self, data_splits: Dict[str, str]) -> str:
        """
        Get the appropriate data path for Phase 1 based on data strategy.
        
        Args:
            data_splits: Available data splits
            
        Returns:
            Path to data for Phase 1
        """
        strategy = self.config.two_phase_training.phase1_data_strategy
        
        if strategy == "augmentation_only":
            # Use training data (assuming it contains augmented samples)
            return data_splits['train']
        elif strategy == "all_data":
            # Use all available training data
            return data_splits['train']
        else:
            raise ValueError(f"Unknown Phase 1 data strategy: {strategy}")
    
    def _get_phase2_data_path(self, data_splits: Dict[str, str]) -> str:
        """
        Get the appropriate data path for Phase 2 based on data strategy.
        
        Args:
            data_splits: Available data splits
            
        Returns:
            Path to data for Phase 2
        """
        strategy = self.config.two_phase_training.phase2_data_strategy
        
        if strategy == "labeled_only":
            # Use training data (assuming it contains labeled samples)
            return data_splits['train']
        elif strategy == "all_data":
            # Use all available training data
            return data_splits['train']
        else:
            raise ValueError(f"Unknown Phase 2 data strategy: {strategy}")
    
    def _populate_result_from_two_phase(self, two_phase_results: 'TwoPhaseResults') -> None:
        """
        Populate the unified result structure from TwoPhaseResults.
        
        Args:
            two_phase_results: Results from TwoPhaseTrainer
        """
        # Set primary model paths
        self.result.final_model_path = two_phase_results.final_model_path
        self.result.best_model_path = two_phase_results.final_model_path  # Final model is the best for two-phase
        self.result.pretrained_model_path = two_phase_results.pretrained_model_path
        
        # Set phase-specific results
        self.result.phase1_results = two_phase_results.pretraining_results
        self.result.phase2_results = two_phase_results.finetuning_results
        
        # Set phase transition metrics
        self.result.phase_transition_metrics = {
            'phase_transition_time': two_phase_results.phase_transition_time,
            'pretrained_model_path': two_phase_results.pretrained_model_path,
            'checkpoint_transfer_success': True  # Assume success if we got results
        }
        
        # Combine training histories (if available)
        if hasattr(two_phase_results.pretraining_results, 'training_history'):
            self.result.training_history.extend(two_phase_results.pretraining_results.training_history)
        
        if 'training_history' in two_phase_results.finetuning_results:
            self.result.training_history.extend(two_phase_results.finetuning_results['training_history'])
        
        # Set validation metrics from Phase 2 (final phase)
        if 'validation_metrics' in two_phase_results.finetuning_results:
            self.result.validation_metrics = two_phase_results.finetuning_results['validation_metrics']
        
        # Combine checkpoint paths
        if hasattr(two_phase_results.pretraining_results, 'checkpoint_paths'):
            self.result.checkpoint_paths.extend(two_phase_results.pretraining_results.checkpoint_paths)
        
        if 'checkpoint_paths' in two_phase_results.finetuning_results:
            self.result.checkpoint_paths.extend(two_phase_results.finetuning_results['checkpoint_paths'])
    
    def _generate_comparative_analysis(self, two_phase_results: 'TwoPhaseResults') -> None:
        """
        Generate comparative analysis between phases.
        
        Args:
            two_phase_results: Results from both phases
        """
        analysis = {
            'phase_comparison': {
                'phase1_final_loss': two_phase_results.pretraining_results.final_loss,
                'phase1_training_time': two_phase_results.pretraining_results.training_time,
                'phase1_total_samples': two_phase_results.pretraining_results.total_samples,
                'phase2_training_time': two_phase_results.finetuning_results.get('training_time', 0),
                'total_training_time': two_phase_results.total_training_time,
                'phase_transition_time': two_phase_results.phase_transition_time
            },
            'efficiency_metrics': {
                'phase1_samples_per_second': (
                    two_phase_results.pretraining_results.total_samples / 
                    two_phase_results.pretraining_results.training_time
                    if two_phase_results.pretraining_results.training_time > 0 else 0
                ),
                'total_samples_processed': two_phase_results.pretraining_results.total_samples,
                'phase_transition_overhead': (
                    two_phase_results.phase_transition_time / 
                    two_phase_results.total_training_time * 100
                    if two_phase_results.total_training_time > 0 else 0
                )
            }
        }
        
        # Add performance comparison if metrics are available
        if 'validation_metrics' in two_phase_results.finetuning_results:
            analysis['performance_metrics'] = {
                'final_validation_metrics': two_phase_results.finetuning_results['validation_metrics']
            }
        
        self.result.comparative_analysis = analysis
        
        # Log key insights
        self.logger.info("=" * 60)
        self.logger.info("TWO-PHASE TRAINING ANALYSIS")
        self.logger.info("=" * 60)
        self.logger.info(f"Phase 1 (Pre-training): {analysis['phase_comparison']['phase1_training_time']:.2f}s")
        self.logger.info(f"Phase 2 (Fine-tuning): {analysis['phase_comparison']['phase2_training_time']:.2f}s")
        self.logger.info(f"Phase transition: {analysis['phase_comparison']['phase_transition_time']:.2f}s")
        self.logger.info(f"Total training time: {analysis['phase_comparison']['total_training_time']:.2f}s")
        self.logger.info(f"Samples per second (Phase 1): {analysis['efficiency_metrics']['phase1_samples_per_second']:.2f}")
        self.logger.info("=" * 60)