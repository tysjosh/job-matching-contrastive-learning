#!/usr/bin/env python3
"""
Pipeline configuration and data structures for complete ML pipeline
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class DataSplittingConfig:
    """Configuration for data splitting"""
    strategy: str = "sequential"  # sequential, random, stratified, user_based, temporal
    ratios: Dict[str, float] = field(default_factory=lambda: {
                                     "train": 0.7, "validation": 0.15, "test": 0.15})
    seed: int = 42
    min_samples_per_split: int = 1
    validate_splits: bool = True
    output_dir: str = "data_splits"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    enabled: bool = True
    patience: int = 5
    metric: str = "validation_loss"
    mode: str = "min"  # min or max
    min_delta: float = 0.001
    restore_best_weights: bool = True


@dataclass
class CheckpointingConfig:
    """Configuration for model checkpointing"""
    save_best: bool = True
    save_last: bool = True
    save_frequency: int = 1  # epochs
    metric: str = "validation_contrastive_accuracy"
    mode: str = "max"  # min or max
    filename_template: str = "checkpoint_epoch_{epoch}_{metric}_{value:.4f}.pt"


@dataclass
class ValidationConfig:
    """Configuration for validation during training"""
    frequency: str = "every_epoch"  # every_epoch, every_n_epochs, every_n_batches
    frequency_value: int = 1
    metrics: List[str] = field(default_factory=lambda: ["all"])
    save_predictions: bool = True
    generate_plots: bool = True
    batch_size: Optional[int] = None  # Use training batch size if None


@dataclass
class TestingConfig:
    """Configuration for final testing"""
    load_best_model: bool = True
    checkpoint_path: Optional[str] = None  # Override best model selection
    comprehensive_report: bool = True
    save_embeddings: bool = True
    save_predictions: bool = True
    error_analysis: bool = True
    generate_visualizations: bool = True
    batch_size: Optional[int] = None  # Use training batch size if None


@dataclass
class ReportingConfig:
    """Configuration for result reporting"""
    formats: List[str] = field(default_factory=lambda: ["json", "html"])
    include_visualizations: bool = True
    include_model_analysis: bool = True
    include_training_history: bool = True
    include_hyperparameters: bool = True
    output_dir: str = "pipeline_reports"
    report_name: Optional[str] = None  # Auto-generate if None


@dataclass
class HyperparameterOptimizationConfig:
    """Configuration for hyperparameter optimization"""
    enabled: bool = False
    n_trials: int = 50
    optimization_metric: str = "validation_contrastive_accuracy"
    optimization_direction: str = "maximize"  # maximize or minimize
    search_space: Dict[str, Any] = field(default_factory=dict)
    pruning: bool = True
    study_name: Optional[str] = None


@dataclass
class TwoPhaseTrainingConfig:
    """Configuration for two-phase training mode"""
    enabled: bool = False
    phase1_config_path: str = ""  # Self-supervised pre-training config
    phase2_config_path: str = ""  # Supervised fine-tuning config
    phase1_data_strategy: str = "augmentation_only"  # augmentation_only, all_data
    phase2_data_strategy: str = "labeled_only"  # labeled_only, all_data
    checkpoint_transfer: bool = True
    phase_validation: bool = True
    comparative_reporting: bool = True
    phase1_output_dir: str = "phase1_output"
    phase2_output_dir: str = "phase2_output"
    checkpoint_save_strategy: str = "best_only"  # best_only, all_epochs, last_only
    phase_transition_validation: bool = True
    
    def validate_config(self) -> List[str]:
        """Validate two-phase training configuration and return list of errors"""
        errors = []
        
        if self.enabled:
            if not self.phase1_config_path:
                errors.append("phase1_config_path is required when two-phase training is enabled")
            if not self.phase2_config_path:
                errors.append("phase2_config_path is required when two-phase training is enabled")
            
            # Validate data strategies
            valid_phase1_strategies = ["augmentation_only", "all_data"]
            if self.phase1_data_strategy not in valid_phase1_strategies:
                errors.append(f"phase1_data_strategy must be one of {valid_phase1_strategies}")
            
            valid_phase2_strategies = ["labeled_only", "all_data"]
            if self.phase2_data_strategy not in valid_phase2_strategies:
                errors.append(f"phase2_data_strategy must be one of {valid_phase2_strategies}")
            
            # Validate checkpoint save strategy
            valid_checkpoint_strategies = ["best_only", "all_epochs", "last_only"]
            if self.checkpoint_save_strategy not in valid_checkpoint_strategies:
                errors.append(f"checkpoint_save_strategy must be one of {valid_checkpoint_strategies}")
        
        return errors
    
    def validate_phase_compatibility(self, phase1_config: Dict[str, Any], phase2_config: Dict[str, Any]) -> List[str]:
        """Validate compatibility between phase configurations"""
        errors = []
        
        # Check model architecture compatibility
        if "model" in phase1_config and "model" in phase2_config:
            phase1_model = phase1_config["model"]
            phase2_model = phase2_config["model"]
            
            # Check embedding dimension compatibility
            if "embedding_dim" in phase1_model and "embedding_dim" in phase2_model:
                if phase1_model["embedding_dim"] != phase2_model["embedding_dim"]:
                    errors.append("Phase 1 and Phase 2 must have the same embedding_dim")
            
            # Check encoder compatibility
            if "encoder_name" in phase1_model and "encoder_name" in phase2_model:
                if phase1_model["encoder_name"] != phase2_model["encoder_name"]:
                    errors.append("Phase 1 and Phase 2 should use the same encoder for optimal transfer learning")
        
        # Check data format compatibility
        if "data" in phase1_config and "data" in phase2_config:
            phase1_data = phase1_config["data"]
            phase2_data = phase2_config["data"]
            
            # Check input format compatibility
            if "input_format" in phase1_data and "input_format" in phase2_data:
                if phase1_data["input_format"] != phase2_data["input_format"]:
                    errors.append("Phase 1 and Phase 2 must use the same input data format")
        
        return errors


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # Data management
    data_splitting: DataSplittingConfig = field(
        default_factory=DataSplittingConfig)

    # Training configuration (references existing TrainingConfig)
    training_config_path: str = "config/training_config.json"
    training_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Augmentation configuration
    augmentation_config_path: Optional[str] = None
    augmentation_quality_profile: str = "balanced"  # fast, balanced, high_quality
    
    # Two-phase training support
    two_phase_training: TwoPhaseTrainingConfig = field(default_factory=TwoPhaseTrainingConfig)
    
    # Enhanced config path handling for two-phase training
    phase1_config_path: Optional[str] = None    # Phase 1 config (overrides two_phase_training.phase1_config_path)
    phase2_config_path: Optional[str] = None    # Phase 2 config (overrides two_phase_training.phase2_config_path)

    # Pipeline components
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig)
    checkpointing: CheckpointingConfig = field(
        default_factory=CheckpointingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    # Advanced features
    hyperparameter_optimization: HyperparameterOptimizationConfig = field(
        default_factory=HyperparameterOptimizationConfig)

    # Pipeline settings
    output_base_dir: str = "pipeline_output"
    experiment_name: Optional[str] = None
    resume_from: Optional[str] = None  # Resume entire pipeline from checkpoint
    skip_data_splitting: bool = False  # Use existing splits
    skip_training: bool = False  # Only run evaluation
    skip_testing: bool = False  # Only run training and validation

    # Resource settings
    device: str = "cpu"
    num_workers: int = 4
    memory_limit: Optional[str] = None  # e.g., "8GB"

    def get_effective_phase1_config_path(self) -> Optional[str]:
        """Get the effective Phase 1 config path, considering overrides"""
        return self.phase1_config_path or self.two_phase_training.phase1_config_path or None
    
    def get_effective_phase2_config_path(self) -> Optional[str]:
        """Get the effective Phase 2 config path, considering overrides"""
        return self.phase2_config_path or self.two_phase_training.phase2_config_path or None
    
    def is_two_phase_training_enabled(self) -> bool:
        """Check if two-phase training is enabled and properly configured"""
        return (self.two_phase_training.enabled and 
                bool(self.get_effective_phase1_config_path()) and 
                bool(self.get_effective_phase2_config_path()))
    
    def validate_configuration(self) -> List[str]:
        """Validate the complete pipeline configuration and return list of errors"""
        errors = []
        
        # Validate two-phase training configuration
        two_phase_errors = self.two_phase_training.validate_config()
        errors.extend(two_phase_errors)
        
        # Additional validation for two-phase training
        if self.two_phase_training.enabled:
            effective_phase1_path = self.get_effective_phase1_config_path()
            effective_phase2_path = self.get_effective_phase2_config_path()
            
            if not effective_phase1_path:
                errors.append("Phase 1 config path must be specified when two-phase training is enabled")
            elif not Path(effective_phase1_path).exists():
                errors.append(f"Phase 1 config file does not exist: {effective_phase1_path}")
            
            if not effective_phase2_path:
                errors.append("Phase 2 config path must be specified when two-phase training is enabled")
            elif not Path(effective_phase2_path).exists():
                errors.append(f"Phase 2 config file does not exist: {effective_phase2_path}")
        
        # Validate single-phase training configuration
        elif not Path(self.training_config_path).exists():
            errors.append(f"Training config file does not exist: {self.training_config_path}")
        
        # Validate output directories
        if self.two_phase_training.enabled:
            phase1_output = Path(self.output_base_dir) / self.two_phase_training.phase1_output_dir
            phase2_output = Path(self.output_base_dir) / self.two_phase_training.phase2_output_dir
            
            if phase1_output == phase2_output:
                errors.append("Phase 1 and Phase 2 output directories must be different")
        
        return errors
    
    def validate_phase_compatibility(self) -> List[str]:
        """Validate compatibility between phase configurations if two-phase training is enabled"""
        errors = []
        
        if not self.two_phase_training.enabled:
            return errors
        
        phase1_path = self.get_effective_phase1_config_path()
        phase2_path = self.get_effective_phase2_config_path()
        
        if not phase1_path or not phase2_path:
            return errors  # Basic validation will catch missing paths
        
        try:
            # Load phase configurations
            with open(phase1_path, 'r') as f:
                phase1_config = json.load(f)
            
            with open(phase2_path, 'r') as f:
                phase2_config = json.load(f)
            
            # Use TwoPhaseTrainingConfig validation method
            compatibility_errors = self.two_phase_training.validate_phase_compatibility(
                phase1_config, phase2_config
            )
            errors.extend(compatibility_errors)
            
        except FileNotFoundError as e:
            errors.append(f"Could not load phase configuration file: {e}")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in phase configuration file: {e}")
        except Exception as e:
            errors.append(f"Error validating phase compatibility: {e}")
        
        return errors

    def save_to_file(self, file_path: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()

        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_file(cls, file_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "data_splitting": {
                "strategy": self.data_splitting.strategy,
                "ratios": self.data_splitting.ratios,
                "seed": self.data_splitting.seed,
                "min_samples_per_split": self.data_splitting.min_samples_per_split,
                "validate_splits": self.data_splitting.validate_splits,
                "output_dir": self.data_splitting.output_dir
            },
            "training_config_path": self.training_config_path,
            "training_overrides": self.training_overrides,
            "augmentation_config_path": self.augmentation_config_path,
            "augmentation_quality_profile": self.augmentation_quality_profile,
            "two_phase_training": {
                "enabled": self.two_phase_training.enabled,
                "phase1_config_path": self.two_phase_training.phase1_config_path,
                "phase2_config_path": self.two_phase_training.phase2_config_path,
                "phase1_data_strategy": self.two_phase_training.phase1_data_strategy,
                "phase2_data_strategy": self.two_phase_training.phase2_data_strategy,
                "checkpoint_transfer": self.two_phase_training.checkpoint_transfer,
                "phase_validation": self.two_phase_training.phase_validation,
                "comparative_reporting": self.two_phase_training.comparative_reporting,
                "phase1_output_dir": self.two_phase_training.phase1_output_dir,
                "phase2_output_dir": self.two_phase_training.phase2_output_dir,
                "checkpoint_save_strategy": self.two_phase_training.checkpoint_save_strategy,
                "phase_transition_validation": self.two_phase_training.phase_transition_validation
            },
            "phase1_config_path": self.phase1_config_path,
            "phase2_config_path": self.phase2_config_path,
            "early_stopping": {
                "enabled": self.early_stopping.enabled,
                "patience": self.early_stopping.patience,
                "metric": self.early_stopping.metric,
                "mode": self.early_stopping.mode,
                "min_delta": self.early_stopping.min_delta,
                "restore_best_weights": self.early_stopping.restore_best_weights
            },
            "checkpointing": {
                "save_best": self.checkpointing.save_best,
                "save_last": self.checkpointing.save_last,
                "save_frequency": self.checkpointing.save_frequency,
                "metric": self.checkpointing.metric,
                "mode": self.checkpointing.mode,
                "filename_template": self.checkpointing.filename_template
            },
            "validation": {
                "frequency": self.validation.frequency,
                "frequency_value": self.validation.frequency_value,
                "metrics": self.validation.metrics,
                "save_predictions": self.validation.save_predictions,
                "generate_plots": self.validation.generate_plots,
                "batch_size": self.validation.batch_size
            },
            "testing": {
                "load_best_model": self.testing.load_best_model,
                "checkpoint_path": self.testing.checkpoint_path,
                "comprehensive_report": self.testing.comprehensive_report,
                "save_embeddings": self.testing.save_embeddings,
                "save_predictions": self.testing.save_predictions,
                "error_analysis": self.testing.error_analysis,
                "generate_visualizations": self.testing.generate_visualizations,
                "batch_size": self.testing.batch_size
            },
            "reporting": {
                "formats": self.reporting.formats,
                "include_visualizations": self.reporting.include_visualizations,
                "include_model_analysis": self.reporting.include_model_analysis,
                "include_training_history": self.reporting.include_training_history,
                "include_hyperparameters": self.reporting.include_hyperparameters,
                "output_dir": self.reporting.output_dir,
                "report_name": self.reporting.report_name
            },
            "hyperparameter_optimization": {
                "enabled": self.hyperparameter_optimization.enabled,
                "n_trials": self.hyperparameter_optimization.n_trials,
                "optimization_metric": self.hyperparameter_optimization.optimization_metric,
                "optimization_direction": self.hyperparameter_optimization.optimization_direction,
                "search_space": self.hyperparameter_optimization.search_space,
                "pruning": self.hyperparameter_optimization.pruning,
                "study_name": self.hyperparameter_optimization.study_name
            },
            "output_base_dir": self.output_base_dir,
            "experiment_name": self.experiment_name,
            "resume_from": self.resume_from,
            "skip_data_splitting": self.skip_data_splitting,
            "skip_training": self.skip_training,
            "skip_testing": self.skip_testing,
            "device": self.device,
            "num_workers": self.num_workers,
            "memory_limit": self.memory_limit
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary"""

        # Create sub-configurations
        data_splitting = DataSplittingConfig(
            **config_dict.get("data_splitting", {}))
        early_stopping = EarlyStoppingConfig(
            **config_dict.get("early_stopping", {}))
        checkpointing = CheckpointingConfig(
            **config_dict.get("checkpointing", {}))
        validation = ValidationConfig(**config_dict.get("validation", {}))
        testing = TestingConfig(**config_dict.get("testing", {}))
        reporting = ReportingConfig(**config_dict.get("reporting", {}))
        hyperparameter_optimization = HyperparameterOptimizationConfig(
            **config_dict.get("hyperparameter_optimization", {}))
        two_phase_training = TwoPhaseTrainingConfig(
            **config_dict.get("two_phase_training", {}))

        return cls(
            data_splitting=data_splitting,
            training_config_path=config_dict.get(
                "training_config_path", "config/training_config.json"),
            training_overrides=config_dict.get("training_overrides", {}),
            augmentation_config_path=config_dict.get("augmentation_config_path"),
            augmentation_quality_profile=config_dict.get("augmentation_quality_profile", "balanced"),
            two_phase_training=two_phase_training,
            phase1_config_path=config_dict.get("phase1_config_path"),
            phase2_config_path=config_dict.get("phase2_config_path"),
            early_stopping=early_stopping,
            checkpointing=checkpointing,
            validation=validation,
            testing=testing,
            reporting=reporting,
            hyperparameter_optimization=hyperparameter_optimization,
            output_base_dir=config_dict.get(
                "output_base_dir", "pipeline_output"),
            experiment_name=config_dict.get("experiment_name"),
            resume_from=config_dict.get("resume_from"),
            skip_data_splitting=config_dict.get("skip_data_splitting", False),
            skip_training=config_dict.get("skip_training", False),
            skip_testing=config_dict.get("skip_testing", False),
            device=config_dict.get("device", "cpu"),
            num_workers=config_dict.get("num_workers", 4),
            memory_limit=config_dict.get("memory_limit")
        )


@dataclass
class PipelineResults:
    """Results from complete pipeline execution"""
    experiment_name: str
    start_time: str
    end_time: str
    total_duration: float

    # Data splitting results
    data_splits: Optional[Dict[str, str]] = None
    split_statistics: Optional[Dict[str, Any]] = None

    # Training results
    final_model_path: Optional[str] = None
    best_model_path: Optional[str] = None
    training_history: Optional[Dict[str, Any]] = None

    # Validation results
    validation_metrics: Optional[Dict[str, Any]] = None
    validation_history: Optional[List[Dict[str, Any]]] = None

    # Test results
    test_metrics: Optional[Dict[str, float]] = None
    test_detailed_metrics: Optional[Dict[str, Any]] = None
    test_predictions_path: Optional[str] = None
    test_embeddings_path: Optional[str] = None

    # Generated artifacts
    visualization_paths: List[str] = field(default_factory=list)
    report_paths: List[str] = field(default_factory=list)
    checkpoint_paths: List[str] = field(default_factory=list)

    # Pipeline metadata
    config_used: Optional[Dict[str, Any]] = None
    git_commit: Optional[str] = None
    python_version: str = ""
    packages_used: Dict[str, str] = field(default_factory=dict)

    # Two-phase training specific results (None for single-phase)
    training_mode: str = "single_phase"  # "single_phase" or "two_phase"
    phase1_results: Optional[Any] = None  # TrainingResults from Phase 1
    phase2_results: Optional[Dict[str, Any]] = None  # Results from Phase 2
    phase_transition_metrics: Optional[Dict[str, Any]] = None
    comparative_analysis: Optional[Dict[str, Any]] = None

    def save_to_file(self, file_path: str):
        """Save results to JSON file"""
        results_dict = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": self.total_duration,
            "data_splits": self.data_splits,
            "split_statistics": self.split_statistics,
            "final_model_path": self.final_model_path,
            "best_model_path": self.best_model_path,
            "training_history": self.training_history,
            "validation_metrics": self.validation_metrics,
            "validation_history": self.validation_history,
            "test_metrics": self.test_metrics,
            "test_detailed_metrics": self.test_detailed_metrics,
            "test_predictions_path": self.test_predictions_path,
            "test_embeddings_path": self.test_embeddings_path,
            "visualization_paths": self.visualization_paths,
            "report_paths": self.report_paths,
            "checkpoint_paths": self.checkpoint_paths,
            "config_used": self.config_used,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "packages_used": self.packages_used,
            "training_mode": self.training_mode,
            "phase1_results": self.phase1_results.to_dict() if self.phase1_results and hasattr(self.phase1_results, 'to_dict') else self.phase1_results,
            "phase2_results": self.phase2_results,
            "phase_transition_metrics": self.phase_transition_metrics,
            "comparative_analysis": self.comparative_analysis
        }

        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

    @classmethod
    def from_file(cls, file_path: str) -> 'PipelineResults':
        """Load results from JSON file"""
        with open(file_path, 'r') as f:
            results_dict = json.load(f)

        return cls(**results_dict)


def create_default_pipeline_config() -> PipelineConfig:
    """Create a default pipeline configuration"""
    return PipelineConfig(
        data_splitting=DataSplittingConfig(
            strategy="sequential",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=42
        ),
        training_config_path="config/training_config.json",
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=5,
            metric="validation_loss",
            mode="min"
        ),
        validation=ValidationConfig(
            frequency="every_epoch",
            metrics=["all"],
            save_predictions=True,
            generate_plots=True
        ),
        testing=TestingConfig(
            load_best_model=True,
            comprehensive_report=True,
            save_embeddings=True,
            generate_visualizations=True
        ),
        reporting=ReportingConfig(
            formats=["json", "html"],
            include_visualizations=True
        )
    )


def create_quick_pipeline_config() -> PipelineConfig:
    """Create a quick/minimal pipeline configuration for testing"""
    config = create_default_pipeline_config()

    # Reduce for speed
    config.early_stopping.patience = 3
    config.validation.generate_plots = False
    config.testing.save_embeddings = False
    config.testing.generate_visualizations = False
    config.reporting.formats = ["json"]
    config.reporting.include_visualizations = False

    return config


def main():
    """Generate example pipeline configurations"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate pipeline configuration files")
    parser.add_argument(
        "--output", "-o", default="config/pipeline_config.json", help="Output file path")
    parser.add_argument("--type", choices=["default", "quick", "comprehensive"], default="default",
                        help="Configuration type")

    args = parser.parse_args()

    # Create configuration based on type
    if args.type == "default":
        config = create_default_pipeline_config()
    elif args.type == "quick":
        config = create_quick_pipeline_config()
    elif args.type == "comprehensive":
        config = create_default_pipeline_config()
        # Add comprehensive settings
        config.hyperparameter_optimization.enabled = True
        config.hyperparameter_optimization.n_trials = 20
        config.testing.error_analysis = True
        config.reporting.formats = ["json", "html", "pdf"]

    # Save configuration
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config.save_to_file(str(output_path))

    print(f"Generated {args.type} pipeline configuration: {output_path}")
    print("\nConfiguration preview:")
    print(f"  Data splitting: {config.data_splitting.strategy}")
    print(
        f"  Early stopping: {'Enabled' if config.early_stopping.enabled else 'Disabled'}")
    print(
        f"  Hyperparameter optimization: {'Enabled' if config.hyperparameter_optimization.enabled else 'Disabled'}")
    print(f"  Report formats: {', '.join(config.reporting.formats)}")


if __name__ == "__main__":
    main()
