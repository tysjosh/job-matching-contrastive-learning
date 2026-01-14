"""
TwoPhaseTrainingOrchestrator - Core orchestration logic for two-phase training.

This module implements the orchestrator that manages execution of both training phases,
handling checkpoint management, phase transitions, and comprehensive results collection.
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
from .pipeline_config import TwoPhaseTrainingConfig
from .logging_utils import setup_training_logger, MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class PhaseTransitionMetrics:
    """Metrics collected during phase transition."""
    checkpoint_path: str
    checkpoint_size_mb: float
    checkpoint_validation_time: float
    model_loading_time: float
    phase1_final_loss: float
    phase2_initial_setup_time: float
    memory_usage_before_mb: Optional[float] = None
    memory_usage_after_mb: Optional[float] = None
    gpu_memory_before_mb: Optional[float] = None
    gpu_memory_after_mb: Optional[float] = None


@dataclass
class TwoPhaseResults:
    """Comprehensive results from two-phase training."""
    # Phase results
    phase1_results: TrainingResults
    phase2_results: Dict[str, Any]
    
    # Model paths
    pretrained_model_path: str
    final_model_path: str
    
    # Timing metrics
    phase1_training_time: float
    phase2_training_time: float
    phase_transition_time: float
    total_training_time: float
    
    # Phase transition metrics
    phase_transition_metrics: PhaseTransitionMetrics
    
    # Comparative analysis
    comparative_analysis: Dict[str, Any]
    
    # Configuration used
    phase1_config: Dict[str, Any]
    phase2_config: Dict[str, Any]
    orchestrator_config: Dict[str, Any]
    
    # Resource usage
    peak_memory_usage_mb: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'phase1_results': self.phase1_results.to_dict(),
            'phase2_results': self.phase2_results,
            'pretrained_model_path': self.pretrained_model_path,
            'final_model_path': self.final_model_path,
            'phase1_training_time': self.phase1_training_time,
            'phase2_training_time': self.phase2_training_time,
            'phase_transition_time': self.phase_transition_time,
            'total_training_time': self.total_training_time,
            'phase_transition_metrics': asdict(self.phase_transition_metrics),
            'comparative_analysis': self.comparative_analysis,
            'phase1_config': self.phase1_config,
            'phase2_config': self.phase2_config,
            'orchestrator_config': self.orchestrator_config,
            'peak_memory_usage_mb': self.peak_memory_usage_mb,
            'peak_gpu_memory_mb': self.peak_gpu_memory_mb
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class TwoPhaseTrainingOrchestrator:
    """
    Core orchestrator for two-phase training strategy.
    
    This class manages the execution of both self-supervised pre-training and
    supervised fine-tuning phases, handling checkpoint management, phase transitions,
    and comprehensive results collection.
    """
    
    def __init__(self, 
                 phase1_config: TrainingConfig,
                 phase2_config: TrainingConfig,
                 orchestrator_config: TwoPhaseTrainingConfig,
                 output_dir: Union[str, Path] = "two_phase_output"):
        """
        Initialize the TwoPhaseTrainingOrchestrator.
        
        Args:
            phase1_config: Configuration for Phase 1 (self-supervised pre-training)
            phase2_config: Configuration for Phase 2 (supervised fine-tuning)
            orchestrator_config: Configuration for orchestration behavior
            output_dir: Base directory for saving outputs from both phases
            
        Raises:
            ImportError: If PyTorch is not available
            ValueError: If configurations are invalid or incompatible
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TwoPhaseTrainingOrchestrator. "
                "Install with: pip install torch"
            )
        
        self.phase1_config = phase1_config
        self.phase2_config = phase2_config
        self.orchestrator_config = orchestrator_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configurations
        self._validate_configurations()
        
        # Setup output directories for each phase
        self.phase1_output_dir = self.output_dir / orchestrator_config.phase1_output_dir
        self.phase2_output_dir = self.output_dir / orchestrator_config.phase2_output_dir
        self.phase1_output_dir.mkdir(parents=True, exist_ok=True)
        self.phase2_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize structured logging
        self.structured_logger = setup_training_logger(
            name="two_phase_orchestrator",
            log_dir=self.output_dir / "logs",
            log_level=logging.INFO
        )
        self.memory_monitor = MemoryMonitor()
        
        # Training state
        self.phase1_completed = False
        self.phase2_completed = False
        self.pretrained_model_path = None
        self.final_model_path = None
        
        # Metrics tracking
        self.training_metrics = {
            'orchestrator_start_time': None,
            'orchestrator_end_time': None,
            'phase1_start_time': None,
            'phase1_end_time': None,
            'phase2_start_time': None,
            'phase2_end_time': None,
            'phase_transition_start_time': None,
            'phase_transition_end_time': None,
            'peak_memory_usage_mb': 0.0,
            'peak_gpu_memory_mb': 0.0
        }
        
        # Log initialization
        self.structured_logger.logger.info("TwoPhaseTrainingOrchestrator initialized")
        self.structured_logger.logger.info(f"Output directory: {self.output_dir}")
        self.structured_logger.logger.info(f"Phase 1 config: {phase1_config.training_phase}")
        self.structured_logger.logger.info(f"Phase 2 config: {phase2_config.training_phase}")
        
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
        if self.phase1_config.training_phase != "self_supervised":
            raise ValueError(
                f"Phase 1 must use training_phase='self_supervised', "
                f"got: {self.phase1_config.training_phase}"
            )
        
        # Validate Phase 2 configuration
        if self.phase2_config.training_phase != "fine_tuning":
            raise ValueError(
                f"Phase 2 must use training_phase='fine_tuning', "
                f"got: {self.phase2_config.training_phase}"
            )
        
        # Validate text encoder compatibility
        if self.phase1_config.text_encoder_model != self.phase2_config.text_encoder_model:
            raise ValueError(
                f"Text encoder models must match between phases. "
                f"Phase 1: {self.phase1_config.text_encoder_model}, "
                f"Phase 2: {self.phase2_config.text_encoder_model}"
            )
        
        # Validate Phase 1 specific settings
        if not self.phase1_config.use_augmentation_labels_only:
            logger.warning(
                "Phase 1 should typically use use_augmentation_labels_only=True "
                "for pure self-supervised learning"
            )
        
        if self.phase1_config.augmentation_positive_ratio == 0.0:
            raise ValueError(
                "Phase 1 augmentation_positive_ratio cannot be 0.0 for self-supervised learning"
            )
        
        # Validate Phase 2 specific settings
        if self.phase2_config.pretrained_model_path is not None:
            logger.warning(
                "Phase 2 pretrained_model_path will be overridden with Phase 1 output"
            )
        
        # Validate compatibility of key parameters
        compatibility_checks = [
            ('batch_size', 'Batch sizes should be compatible for memory consistency'),
            ('text_encoder_device', 'Text encoder devices should match'),
        ]
        
        for param, message in compatibility_checks:
            phase1_val = getattr(self.phase1_config, param, None)
            phase2_val = getattr(self.phase2_config, param, None)
            
            if phase1_val != phase2_val and phase1_val is not None and phase2_val is not None:
                logger.warning(f"{message}. Phase 1: {phase1_val}, Phase 2: {phase2_val}")
        
        logger.info("Configuration validation completed successfully")
    
    def execute_phase1(self, train_data: str, val_data: str) -> TrainingResults:
        """
        Execute Phase 1: Self-supervised pre-training using ContrastiveLearningTrainer.
        
        Args:
            train_data: Path to training data for self-supervised learning
            val_data: Path to validation data
            
        Returns:
            TrainingResults from Phase 1
            
        Raises:
            FileNotFoundError: If data files don't exist
            Exception: If Phase 1 training fails
        """
        # Validate input files
        train_path = Path(train_data)
        val_path = Path(val_data)
        
        if not train_path.exists():
            raise FileNotFoundError(f"Phase 1 training data not found: {train_data}")
        if not val_path.exists():
            raise FileNotFoundError(f"Phase 1 validation data not found: {val_data}")
        
        self.training_metrics['phase1_start_time'] = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info("STARTING PHASE 1: SELF-SUPERVISED PRE-TRAINING")
            logger.info("=" * 60)
            logger.info(f"Training data: {train_data}")
            logger.info(f"Validation data: {val_data}")
            logger.info(f"Output directory: {self.phase1_output_dir}")
            
            # Initialize ContrastiveLearningTrainer for Phase 1
            phase1_trainer = ContrastiveLearningTrainer(
                config=self.phase1_config,
                output_dir=str(self.phase1_output_dir)
            )
            
            logger.info(f"Phase 1 trainer initialized")
            logger.info(f"Using augmentation labels only: {self.phase1_config.use_augmentation_labels_only}")
            logger.info(f"Augmentation positive ratio: {self.phase1_config.augmentation_positive_ratio}")
            
            # Execute self-supervised pre-training
            phase1_results = phase1_trainer.train(train_path)
            
            self.training_metrics['phase1_end_time'] = time.time()
            phase1_time = (
                self.training_metrics['phase1_end_time'] - 
                self.training_metrics['phase1_start_time']
            )
            
            self.phase1_completed = True
            
            logger.info(f"Phase 1 completed successfully in {phase1_time:.2f} seconds")
            logger.info(f"Phase 1 final loss: {phase1_results.final_loss:.4f}")
            logger.info(f"Phase 1 total samples: {phase1_results.total_samples}")
            
            # Update memory tracking
            memory_info = self.memory_monitor.get_memory_usage()
            if memory_info["system_memory_mb"]:
                self.training_metrics['peak_memory_usage_mb'] = max(
                    self.training_metrics['peak_memory_usage_mb'],
                    memory_info["system_memory_mb"]
                )
            if memory_info["gpu_memory_mb"]:
                self.training_metrics['peak_gpu_memory_mb'] = max(
                    self.training_metrics['peak_gpu_memory_mb'],
                    memory_info["gpu_memory_mb"]
                )
            
            return phase1_results
            
        except Exception as e:
            logger.error(f"Phase 1 (self-supervised pre-training) failed: {e}")
            raise Exception(f"Phase 1 failed: {e}") from e
    
    def execute_phase2(self, train_data: str, val_data: str, pretrained_model_path: str) -> Dict[str, Any]:
        """
        Execute Phase 2: Supervised fine-tuning using FineTuningTrainer.
        
        Args:
            train_data: Path to labeled training data for fine-tuning
            val_data: Path to validation data
            pretrained_model_path: Path to pre-trained model from Phase 1
            
        Returns:
            Dictionary containing Phase 2 results
            
        Raises:
            FileNotFoundError: If data files or pre-trained model don't exist
            Exception: If Phase 2 training fails
        """
        # Validate input files
        train_path = Path(train_data)
        val_path = Path(val_data)
        pretrained_path = Path(pretrained_model_path)
        
        if not train_path.exists():
            raise FileNotFoundError(f"Phase 2 training data not found: {train_data}")
        if not val_path.exists():
            raise FileNotFoundError(f"Phase 2 validation data not found: {val_data}")
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pre-trained model not found: {pretrained_model_path}")
        
        self.training_metrics['phase2_start_time'] = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info("STARTING PHASE 2: SUPERVISED FINE-TUNING")
            logger.info("=" * 60)
            logger.info(f"Training data: {train_data}")
            logger.info(f"Validation data: {val_data}")
            logger.info(f"Pre-trained model: {pretrained_model_path}")
            logger.info(f"Output directory: {self.phase2_output_dir}")
            
            # Update Phase 2 config with pre-trained model path
            self.phase2_config.pretrained_model_path = pretrained_model_path
            
            # Initialize FineTuningTrainer for Phase 2
            phase2_trainer = FineTuningTrainer(
                config=self.phase2_config,
                output_dir=str(self.phase2_output_dir)
            )
            
            logger.info(f"Phase 2 trainer initialized with pre-trained model: {pretrained_model_path}")
            logger.info(f"Freeze contrastive layers: {self.phase2_config.freeze_contrastive_layers}")
            logger.info(f"Classification dropout: {self.phase2_config.classification_dropout}")
            
            # Execute supervised fine-tuning
            phase2_results = phase2_trainer.train(train_path)
            
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
            
            # Update memory tracking
            memory_info = self.memory_monitor.get_memory_usage()
            if memory_info["system_memory_mb"]:
                self.training_metrics['peak_memory_usage_mb'] = max(
                    self.training_metrics['peak_memory_usage_mb'],
                    memory_info["system_memory_mb"]
                )
            if memory_info["gpu_memory_mb"]:
                self.training_metrics['peak_gpu_memory_mb'] = max(
                    self.training_metrics['peak_gpu_memory_mb'],
                    memory_info["gpu_memory_mb"]
                )
            
            return phase2_results
            
        except Exception as e:
            logger.error(f"Phase 2 (supervised fine-tuning) failed: {e}")
            raise Exception(f"Phase 2 failed: {e}") from e
    
    def run_complete_pipeline(self, data_splits: Dict[str, str]) -> TwoPhaseResults:
        """
        Coordinate both phases with checkpoint transfer and comprehensive results collection.
        
        Args:
            data_splits: Dictionary containing data split paths
                        Expected keys: 'train', 'validation', 'test'
                        For two-phase training:
                        - 'train' should contain augmented data for Phase 1
                        - 'labeled_train' should contain labeled data for Phase 2 (optional key)
        
        Returns:
            TwoPhaseResults containing comprehensive results from both phases
            
        Raises:
            ValueError: If required data splits are missing
            Exception: If either phase fails
        """
        # Validate required data splits
        required_splits = ['train', 'validation']
        for split_name in required_splits:
            if split_name not in data_splits:
                raise ValueError(f"Missing required data split: {split_name}")
            if not data_splits[split_name]:
                raise ValueError(f"Empty path for data split: {split_name}")
        
        # Determine data for each phase based on orchestrator configuration
        phase1_train_data = data_splits['train']  # Augmented data for self-supervised learning
        phase1_val_data = data_splits['validation']
        
        # For Phase 2, use labeled data if available, otherwise use same training data
        phase2_train_data = data_splits.get('labeled_train', data_splits['train'])
        phase2_val_data = data_splits['validation']
        
        self.training_metrics['orchestrator_start_time'] = time.time()
        
        try:
            logger.info("=" * 80)
            logger.info("STARTING TWO-PHASE TRAINING ORCHESTRATION")
            logger.info("=" * 80)
            logger.info(f"Phase 1 training data: {phase1_train_data}")
            logger.info(f"Phase 1 validation data: {phase1_val_data}")
            logger.info(f"Phase 2 training data: {phase2_train_data}")
            logger.info(f"Phase 2 validation data: {phase2_val_data}")
            
            # Execute Phase 1: Self-supervised pre-training
            phase1_results = self.execute_phase1(phase1_train_data, phase1_val_data)
            
            # Phase transition with checkpoint management
            transition_metrics = self._handle_phase_transition()
            
            # Execute Phase 2: Supervised fine-tuning
            phase2_results = self.execute_phase2(
                phase2_train_data, 
                phase2_val_data, 
                self.pretrained_model_path
            )
            
            # Collect comprehensive results
            results = self._collect_comprehensive_results(
                phase1_results, 
                phase2_results, 
                transition_metrics
            )
            
            self.training_metrics['orchestrator_end_time'] = time.time()
            
            logger.info("=" * 80)
            logger.info("TWO-PHASE TRAINING ORCHESTRATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total training time: {results.total_training_time:.2f} seconds")
            logger.info(f"Phase 1 final loss: {phase1_results.final_loss:.4f}")
            logger.info(f"Phase 2 final accuracy: {phase2_results.get('final_accuracy', 0.0):.4f}")
            logger.info(f"Pre-trained model: {self.pretrained_model_path}")
            logger.info(f"Final model: {results.final_model_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Two-phase training orchestration failed: {e}")
            
            # Save emergency state
            try:
                self._save_emergency_state(e)
            except Exception as save_error:
                logger.error(f"Failed to save emergency state: {save_error}")
            
            raise  
  
    def _handle_phase_transition(self) -> PhaseTransitionMetrics:
        """
        Handle checkpoint management and phase transition between Phase 1 and Phase 2.
        
        Returns:
            PhaseTransitionMetrics containing transition timing and validation info
            
        Raises:
            FileNotFoundError: If no suitable checkpoint is found
            Exception: If checkpoint validation fails
        """
        self.training_metrics['phase_transition_start_time'] = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info("HANDLING PHASE TRANSITION")
            logger.info("=" * 60)
            
            # Get memory usage before transition
            memory_before = self.memory_monitor.get_memory_usage()
            
            # Find and validate the best checkpoint from Phase 1
            checkpoint_path = self._get_pretrained_model_path()
            self.pretrained_model_path = checkpoint_path
            
            # Validate checkpoint compatibility
            validation_start = time.time()
            self._validate_checkpoint_compatibility(checkpoint_path)
            validation_time = time.time() - validation_start
            
            # Get checkpoint file size
            checkpoint_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
            
            # Test model loading time
            loading_start = time.time()
            self._test_checkpoint_loading(checkpoint_path)
            loading_time = time.time() - loading_start
            
            # Get memory usage after transition
            memory_after = self.memory_monitor.get_memory_usage()
            
            self.training_metrics['phase_transition_end_time'] = time.time()
            transition_time = (
                self.training_metrics['phase_transition_end_time'] - 
                self.training_metrics['phase_transition_start_time']
            )
            
            # Create transition metrics
            transition_metrics = PhaseTransitionMetrics(
                checkpoint_path=checkpoint_path,
                checkpoint_size_mb=checkpoint_size_mb,
                checkpoint_validation_time=validation_time,
                model_loading_time=loading_time,
                phase1_final_loss=0.0,  # Will be updated in collect_comprehensive_results
                phase2_initial_setup_time=0.0,  # Will be updated when Phase 2 starts
                memory_usage_before_mb=memory_before.get("system_memory_mb"),
                memory_usage_after_mb=memory_after.get("system_memory_mb"),
                gpu_memory_before_mb=memory_before.get("gpu_memory_mb"),
                gpu_memory_after_mb=memory_after.get("gpu_memory_mb")
            )
            
            logger.info(f"Phase transition completed in {transition_time:.2f} seconds")
            logger.info(f"Selected checkpoint: {checkpoint_path}")
            logger.info(f"Checkpoint size: {checkpoint_size_mb:.2f} MB")
            logger.info(f"Validation time: {validation_time:.2f} seconds")
            logger.info(f"Loading test time: {loading_time:.2f} seconds")
            
            return transition_metrics
            
        except Exception as e:
            logger.error(f"Phase transition failed: {e}")
            raise Exception(f"Phase transition failed: {e}") from e
    
    def _get_pretrained_model_path(self) -> str:
        """
        Get the path to the pre-trained model from Phase 1 based on checkpoint save strategy.
        
        Returns:
            Path to the selected checkpoint from Phase 1
            
        Raises:
            FileNotFoundError: If no suitable checkpoint is found
        """
        # Look for checkpoints in Phase 1 output directory
        checkpoint_pattern = "*.pt"
        checkpoint_files = list(self.phase1_output_dir.glob(checkpoint_pattern))
        
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {self.phase1_output_dir}. "
                f"Phase 1 may not have completed successfully."
            )
        
        # Select checkpoint based on save strategy
        strategy = self.orchestrator_config.checkpoint_save_strategy
        
        if strategy == "best_only":
            # Look for best checkpoint
            best_checkpoint = None
            for checkpoint_file in checkpoint_files:
                if 'best' in checkpoint_file.name.lower():
                    best_checkpoint = checkpoint_file
                    break
            
            if best_checkpoint:
                return str(best_checkpoint)
            else:
                logger.warning("No 'best' checkpoint found, using most recent checkpoint")
                # Fall through to last_only logic
        
        if strategy == "last_only" or strategy == "best_only":
            # Use the most recent checkpoint
            most_recent = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            return str(most_recent)
        
        elif strategy == "all_epochs":
            # Use the checkpoint with the highest epoch number
            best_checkpoint = None
            best_epoch = -1
            
            for checkpoint_file in checkpoint_files:
                try:
                    # Try to extract epoch number from filename
                    filename = checkpoint_file.name
                    if 'epoch_' in filename:
                        epoch_str = filename.split('epoch_')[1].split('_')[0].split('.')[0]
                        epoch_num = int(epoch_str)
                        if epoch_num > best_epoch:
                            best_epoch = epoch_num
                            best_checkpoint = checkpoint_file
                except (ValueError, IndexError):
                    continue
            
            if best_checkpoint:
                return str(best_checkpoint)
            else:
                # Fall back to most recent if epoch parsing fails
                most_recent = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
                return str(most_recent)
        
        else:
            raise ValueError(f"Unknown checkpoint save strategy: {strategy}")
    
    def _validate_checkpoint_compatibility(self, checkpoint_path: str) -> None:
        """
        Validate checkpoint compatibility between phases.
        
        Args:
            checkpoint_path: Path to the checkpoint to validate
            
        Raises:
            Exception: If checkpoint is incompatible
        """
        try:
            # Load checkpoint to inspect its structure
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate checkpoint structure
            required_keys = ['model_state_dict', 'config']
            for key in required_keys:
                if key not in checkpoint:
                    raise ValueError(f"Checkpoint missing required key: {key}")
            
            # Validate model architecture compatibility
            checkpoint_config = checkpoint.get('config', {})
            
            # Check text encoder compatibility
            if 'text_encoder_model' in checkpoint_config:
                checkpoint_encoder = checkpoint_config['text_encoder_model']
                if checkpoint_encoder != self.phase2_config.text_encoder_model:
                    logger.warning(
                        f"Text encoder mismatch: checkpoint has {checkpoint_encoder}, "
                        f"Phase 2 expects {self.phase2_config.text_encoder_model}"
                    )
            
            # Check embedding dimensions if available
            model_state = checkpoint['model_state_dict']
            if model_state:
                # Try to infer embedding dimension from model state
                for key, tensor in model_state.items():
                    if 'embedding' in key.lower() and tensor.dim() >= 2:
                        checkpoint_dim = tensor.shape[-1]
                        logger.info(f"Checkpoint embedding dimension: {checkpoint_dim}")
                        break
            
            logger.info("Checkpoint compatibility validation passed")
            
        except Exception as e:
            raise Exception(f"Checkpoint compatibility validation failed: {e}")
    
    def _test_checkpoint_loading(self, checkpoint_path: str) -> None:
        """
        Test checkpoint loading to ensure it can be loaded successfully.
        
        Args:
            checkpoint_path: Path to the checkpoint to test
            
        Raises:
            Exception: If checkpoint cannot be loaded
        """
        try:
            # Test loading the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Verify we can access the model state dict
            model_state = checkpoint.get('model_state_dict')
            if model_state is None:
                raise ValueError("Checkpoint does not contain model_state_dict")
            
            # Test that the state dict is not empty
            if not model_state:
                raise ValueError("Model state dict is empty")
            
            logger.info("Checkpoint loading test passed")
            
        except Exception as e:
            raise Exception(f"Checkpoint loading test failed: {e}")
    
    def _collect_comprehensive_results(self, 
                                     phase1_results: TrainingResults,
                                     phase2_results: Dict[str, Any],
                                     transition_metrics: PhaseTransitionMetrics) -> TwoPhaseResults:
        """
        Create unified results structure combining both phase outcomes with comprehensive analysis.
        
        Args:
            phase1_results: Results from Phase 1
            phase2_results: Results from Phase 2
            transition_metrics: Metrics from phase transition
            
        Returns:
            TwoPhaseResults containing comprehensive results and analysis
        """
        # Calculate timing metrics
        phase1_time = (
            self.training_metrics['phase1_end_time'] - 
            self.training_metrics['phase1_start_time']
        )
        phase2_time = (
            self.training_metrics['phase2_end_time'] - 
            self.training_metrics['phase2_start_time']
        )
        transition_time = (
            self.training_metrics['phase_transition_end_time'] - 
            self.training_metrics['phase_transition_start_time']
        )
        total_time = (
            self.training_metrics['orchestrator_end_time'] - 
            self.training_metrics['orchestrator_start_time']
        )
        
        # Update transition metrics with phase 1 final loss
        transition_metrics.phase1_final_loss = phase1_results.final_loss
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(
            phase1_results, phase2_results, transition_metrics
        )
        
        # Create comprehensive results
        results = TwoPhaseResults(
            phase1_results=phase1_results,
            phase2_results=phase2_results,
            pretrained_model_path=self.pretrained_model_path,
            final_model_path=self.final_model_path,
            phase1_training_time=phase1_time,
            phase2_training_time=phase2_time,
            phase_transition_time=transition_time,
            total_training_time=total_time,
            phase_transition_metrics=transition_metrics,
            comparative_analysis=comparative_analysis,
            phase1_config=asdict(self.phase1_config),
            phase2_config=asdict(self.phase2_config),
            orchestrator_config=asdict(self.orchestrator_config),
            peak_memory_usage_mb=self.training_metrics.get('peak_memory_usage_mb'),
            peak_gpu_memory_mb=self.training_metrics.get('peak_gpu_memory_mb')
        )
        
        # Save comprehensive results
        self._save_comprehensive_results(results)
        
        return results
    
    def _generate_comparative_analysis(self, 
                                     phase1_results: TrainingResults,
                                     phase2_results: Dict[str, Any],
                                     transition_metrics: PhaseTransitionMetrics) -> Dict[str, Any]:
        """
        Implement comparative analysis between pre-training and fine-tuning performance.
        
        Args:
            phase1_results: Results from Phase 1
            phase2_results: Results from Phase 2
            transition_metrics: Metrics from phase transition
            
        Returns:
            Dictionary containing comparative analysis
        """
        analysis = {
            'training_efficiency': {
                'phase1_samples_per_second': (
                    phase1_results.total_samples / self.training_metrics.get('phase1_end_time', 1) - 
                    self.training_metrics.get('phase1_start_time', 0)
                ) if phase1_results.total_samples > 0 else 0,
                'phase2_samples_per_second': (
                    phase2_results.get('total_samples', 0) / (
                        self.training_metrics.get('phase2_end_time', 1) - 
                        self.training_metrics.get('phase2_start_time', 0)
                    )
                ) if phase2_results.get('total_samples', 0) > 0 else 0,
                'phase1_time_percentage': (
                    self.training_metrics.get('phase1_end_time', 0) - 
                    self.training_metrics.get('phase1_start_time', 0)
                ) / (
                    self.training_metrics.get('orchestrator_end_time', 1) - 
                    self.training_metrics.get('orchestrator_start_time', 0)
                ) * 100,
                'phase2_time_percentage': (
                    self.training_metrics.get('phase2_end_time', 0) - 
                    self.training_metrics.get('phase2_start_time', 0)
                ) / (
                    self.training_metrics.get('orchestrator_end_time', 1) - 
                    self.training_metrics.get('orchestrator_start_time', 0)
                ) * 100,
                'transition_overhead_percentage': transition_metrics.checkpoint_validation_time / (
                    self.training_metrics.get('orchestrator_end_time', 1) - 
                    self.training_metrics.get('orchestrator_start_time', 0)
                ) * 100
            },
            'performance_metrics': {
                'phase1_final_loss': phase1_results.final_loss,
                'phase1_best_loss': getattr(phase1_results, 'best_loss', phase1_results.final_loss),
                'phase2_final_accuracy': phase2_results.get('final_accuracy', 0.0),
                'phase2_best_accuracy': phase2_results.get('best_accuracy', 0.0),
                'phase2_final_loss': phase2_results.get('final_loss', float('inf')),
                'improvement_from_pretraining': {
                    'accuracy_gain': phase2_results.get('final_accuracy', 0.0),
                    'loss_reduction': max(0, phase1_results.final_loss - phase2_results.get('final_loss', 0))
                }
            },
            'resource_utilization': {
                'phase1_peak_memory_mb': transition_metrics.memory_usage_before_mb,
                'phase2_peak_memory_mb': transition_metrics.memory_usage_after_mb,
                'memory_efficiency': {
                    'checkpoint_size_mb': transition_metrics.checkpoint_size_mb,
                    'memory_overhead_percentage': (
                        (transition_metrics.memory_usage_after_mb or 0) - 
                        (transition_metrics.memory_usage_before_mb or 0)
                    ) / max(1, transition_metrics.memory_usage_before_mb or 1) * 100
                },
                'gpu_utilization': {
                    'phase1_peak_gpu_mb': transition_metrics.gpu_memory_before_mb,
                    'phase2_peak_gpu_mb': transition_metrics.gpu_memory_after_mb
                }
            },
            'training_stability': {
                'phase1_epochs_completed': getattr(phase1_results, 'epochs_completed', 0),
                'phase2_epochs_completed': phase2_results.get('epochs_completed', 0),
                'phase1_convergence_rate': self._calculate_convergence_rate(phase1_results),
                'checkpoint_transfer_success': True,  # If we got here, transfer was successful
                'phase_transition_time_seconds': transition_metrics.checkpoint_validation_time + transition_metrics.model_loading_time
            }
        }
        
        return analysis
    
    def _calculate_convergence_rate(self, results: TrainingResults) -> float:
        """
        Calculate convergence rate from training results.
        
        Args:
            results: Training results to analyze
            
        Returns:
            Convergence rate (loss improvement per epoch)
        """
        try:
            if hasattr(results, 'epoch_losses') and len(results.epoch_losses) > 1:
                initial_loss = results.epoch_losses[0]
                final_loss = results.epoch_losses[-1]
                epochs = len(results.epoch_losses)
                return (initial_loss - final_loss) / epochs
            else:
                return 0.0
        except (AttributeError, IndexError, ZeroDivisionError):
            return 0.0
    
    def _save_comprehensive_results(self, results: TwoPhaseResults) -> None:
        """
        Save comprehensive results to multiple formats.
        
        Args:
            results: TwoPhaseResults to save
        """
        try:
            # Save main results file
            results_path = self.output_dir / "two_phase_orchestrator_results.json"
            results.save_to_file(results_path)
            
            # Save summary report
            summary_path = self.output_dir / "orchestrator_summary.json"
            summary = {
                "orchestrator_type": "two_phase_training",
                "total_training_time": results.total_training_time,
                "phase_transition_time": results.phase_transition_time,
                "phase1_summary": {
                    "final_loss": results.phase1_results.final_loss,
                    "training_time": results.phase1_training_time,
                    "total_samples": results.phase1_results.total_samples,
                    "total_batches": results.phase1_results.total_batches
                },
                "phase2_summary": {
                    "final_accuracy": results.phase2_results.get('final_accuracy', 0.0),
                    "best_accuracy": results.phase2_results.get('best_accuracy', 0.0),
                    "final_loss": results.phase2_results.get('final_loss', 0.0),
                    "training_time": results.phase2_training_time,
                    "total_samples": results.phase2_results.get('total_samples', 0)
                },
                "model_paths": {
                    "pretrained_model": results.pretrained_model_path,
                    "final_model": results.final_model_path
                },
                "resource_usage": {
                    "peak_memory_mb": results.peak_memory_usage_mb,
                    "peak_gpu_memory_mb": results.peak_gpu_memory_mb,
                    "checkpoint_size_mb": results.phase_transition_metrics.checkpoint_size_mb
                },
                "comparative_analysis_summary": {
                    "phase1_time_percentage": results.comparative_analysis['training_efficiency']['phase1_time_percentage'],
                    "phase2_time_percentage": results.comparative_analysis['training_efficiency']['phase2_time_percentage'],
                    "accuracy_improvement": results.comparative_analysis['performance_metrics']['improvement_from_pretraining']['accuracy_gain']
                }
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Comprehensive results saved to {results_path}")
            logger.info(f"Summary report saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save comprehensive results: {e}")
    
    def _save_emergency_state(self, error: Exception) -> None:
        """
        Save emergency state when orchestration fails.
        
        Args:
            error: The exception that caused the failure
        """
        emergency_state = {
            "error": str(error),
            "error_type": type(error).__name__,
            "phase1_completed": self.phase1_completed,
            "phase2_completed": self.phase2_completed,
            "pretrained_model_path": self.pretrained_model_path,
            "final_model_path": self.final_model_path,
            "training_metrics": self.training_metrics,
            "phase1_config": asdict(self.phase1_config),
            "phase2_config": asdict(self.phase2_config),
            "orchestrator_config": asdict(self.orchestrator_config)
        }
        
        emergency_path = self.output_dir / "orchestrator_emergency_state.json"
        with open(emergency_path, 'w') as f:
            json.dump(emergency_state, f, indent=2)
        
        logger.info(f"Emergency state saved to {emergency_path}")
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """
        Get current orchestration status and progress.
        
        Returns:
            Dictionary containing current orchestration status
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
            },
            "orchestrator_config": asdict(self.orchestrator_config)
        }