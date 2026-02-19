#!/usr/bin/env python3
"""
Command-line interface for contrastive learning training.

This module provides a comprehensive CLI for running contrastive learning training
with support for configuration files, environment variables, and various training
scenarios.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import torch

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .data_structures import TrainingConfig, TrainingResults
from .trainer import ContrastiveLearningTrainer
from .logging_utils import setup_training_logger
from .data_adapter import DataAdapter, DataAdapterConfig
from .pipeline import MLPipeline
from .pipeline_config import PipelineConfig, create_default_pipeline_config, create_quick_pipeline_config
from .evaluator import ContrastiveEvaluator, EvaluationConfig


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Contrastive Learning Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default config
  python -m contrastive_learning.cli train data/training_samples.jsonl

  # Training with custom config file
  python -m contrastive_learning.cli train data/training_samples.jsonl --config config/my_config.yaml

  # Training with specific parameters
  python -m contrastive_learning.cli train data/training_samples.jsonl --batch-size 512 --epochs 20

  # Resume training from checkpoint (Phase 1 only)
  python -m contrastive_learning.cli train data/training_samples.jsonl --resume checkpoints/checkpoint_epoch_5.pt

  # Run Phase 2 fine-tuning on pre-trained model
  python -m contrastive_learning.cli train data/training_samples.jsonl --config config/phase2_config.json

  # Run sequential two-phase training (Phase 1 → Phase 2)
  python -m contrastive_learning.cli train data/training_samples.jsonl --two-phase --config config/base_config.json

  # Generate example configuration files
  python -m contrastive_learning.cli generate-config --output config/example.yaml --format yaml

  # Evaluate a checkpoint with text-encoder baseline (cosine similarity only)
  python -m contrastive_learning evaluate --config config/phase2_finetuning_config.json --checkpoint phase2_finetuning/checkpoint_epoch_9.pt --output-dir baseline_evaluation --use-text-encoder-baseline

  # Convert new data format to training format
  python -m contrastive_learning.cli convert-data data/train_labeled_data.jsonl data/translated_eng_jd.csv data/translated_eng_resume.csv --output data/converted_training_data.jsonl

Environment Variables:
  CL_BATCH_SIZE          Override batch size
  CL_LEARNING_RATE       Override learning rate
  CL_NUM_EPOCHS          Override number of epochs
  CL_OUTPUT_DIR          Override output directory
  CL_LOG_LEVEL           Set logging level (DEBUG, INFO, WARNING, ERROR)
        """
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser(
        'train', help='Run contrastive learning training')
    train_parser.add_argument(
        'dataset', type=str, help='Path to training dataset (JSONL format)')
    train_parser.add_argument(
        '--config', '-c', type=str, help='Path to configuration file (YAML or JSON)')
    train_parser.add_argument('--output-dir', '-o', type=str, default='training_output',
                              help='Output directory for checkpoints and logs (default: training_output)')
    train_parser.add_argument(
        '--resume', '-r', type=str, help='Path to checkpoint file to resume from')

    # Training parameters
    train_parser.add_argument('--batch-size', type=int,
                              help='Batch size for training')
    train_parser.add_argument(
        '--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--epochs', type=int,
                              help='Number of training epochs')
    train_parser.add_argument(
        '--temperature', type=float, help='Temperature for contrastive loss')
    train_parser.add_argument('--negative-sampling-ratio', type=float,
                              help='Ratio of random vs pathway negatives (0.0-1.0)')
    train_parser.add_argument(
        '--pathway-weight', type=float, help='Weight for pathway-aware loss')

    # Career graph configuration
    train_parser.add_argument('--esco-graph-path', type=str,
                              help='Path to ESCO career graph file (.gexf format)')

    # Feature flags
    train_parser.add_argument('--no-pathway-negatives', action='store_true',
                              help='Disable pathway-aware negative sampling')
    train_parser.add_argument('--no-view-augmentation', action='store_true',
                              help='Disable view augmentation')
    train_parser.add_argument('--no-shuffle', action='store_true',
                              help='Disable data shuffling')
    train_parser.add_argument('--two-phase', action='store_true',
                              help='Run sequential two-phase training (Phase 1: self-supervised → Phase 2: fine-tuning)')

    # Logging and checkpointing
    train_parser.add_argument('--checkpoint-frequency', type=int,
                              help='Save checkpoint every N batches')
    train_parser.add_argument('--log-frequency', type=int,
                              help='Log progress every N batches')
    train_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              default='INFO', help='Logging level')

    # Generate config command
    config_parser = subparsers.add_parser('generate-config',
                                          help='Generate example configuration files')
    config_parser.add_argument('--output', '-o', type=str, required=True,
                               help='Output path for configuration file')
    config_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                               help='Configuration file format')
    config_parser.add_argument('--scenario', choices=['basic', 'large-dataset', 'fast-training', 'research'],
                               default='basic', help='Configuration scenario')

    # Validate config command
    validate_parser = subparsers.add_parser('validate-config',
                                            help='Validate configuration file')
    validate_parser.add_argument(
        'config', type=str, help='Path to configuration file')

    # Status command
    status_parser = subparsers.add_parser(
        'status', help='Check training status')
    status_parser.add_argument(
        'output_dir', type=str, help='Training output directory')

    # Convert data command
    convert_parser = subparsers.add_parser(
        'convert-data', help='Convert new data format to training format')
    convert_parser.add_argument(
        'labeled_data', type=str, help='Path to train_labeled_data.jsonl')
    convert_parser.add_argument(
        'job_descriptions', type=str, help='Path to translated_eng_jd.csv')
    convert_parser.add_argument(
        'resume_data', type=str, help='Path to translated_eng_resume.csv')
    convert_parser.add_argument('--output', '-o', type=str, default='data/converted_training_data.jsonl',
                                help='Output path for converted JSONL file')
    convert_parser.add_argument('--min-job-desc-length', type=int, default=50,
                                help='Minimum job description length (default: 50)')
    convert_parser.add_argument('--min-resume-exp-length', type=int, default=20,
                                help='Minimum resume experience length (default: 20)')
    convert_parser.add_argument('--max-samples-per-user', type=int, default=10,
                                help='Maximum samples per user to avoid bias (default: 10)')
    convert_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                                default='INFO', help='Logging level')

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline',
                                            help='Run complete train/validate/test pipeline')
    pipeline_parser.add_argument('dataset', type=str,
                                 help='Path to dataset (JSONL format) or training data if using separate files')
    pipeline_parser.add_argument('--config', '-c', type=str,
                                 help='Path to pipeline configuration file (JSON)')
    pipeline_parser.add_argument('--train-data', type=str,
                                 help='Path to training data (if using pre-split data)')
    pipeline_parser.add_argument('--val-data', type=str,
                                 help='Path to validation data (if using pre-split data)')
    pipeline_parser.add_argument('--test-data', type=str,
                                 help='Path to test data (if using pre-split data)')
    pipeline_parser.add_argument('--output-dir', '-o', type=str, default='pipeline_output',
                                 help='Base output directory for pipeline results')
    pipeline_parser.add_argument('--experiment-name', type=str,
                                 help='Name for this experiment (auto-generated if not provided)')
    pipeline_parser.add_argument('--quick', action='store_true',
                                 help='Use quick/minimal configuration for testing')
    pipeline_parser.add_argument('--skip-data-splitting', action='store_true',
                                 help='Skip data splitting (use existing splits)')
    pipeline_parser.add_argument('--skip-training', action='store_true',
                                 help='Skip training (only run evaluation)')
    pipeline_parser.add_argument('--skip-testing', action='store_true',
                                 help='Skip final testing (only run training and validation)')
    pipeline_parser.add_argument('--device', default='cpu',
                                 help='Device to use (cpu, cuda, mps)')
    pipeline_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                                 default='INFO', help='Logging level')

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate', help='Evaluate a checkpoint or text-encoder baseline')
    evaluate_parser.add_argument('--config', '-c', type=str, required=True,
                                 help='Path to training configuration file (YAML or JSON)')
    evaluate_parser.add_argument('--checkpoint', type=str,
                                 help='Path to model checkpoint (.pt)')
    evaluate_parser.add_argument('--output-dir', '-o', type=str, default='evaluation_output',
                                 help='Output directory for evaluation results')
    evaluate_parser.add_argument('--dataset', type=str,
                                 help='Path to evaluation dataset (JSONL format)')
    evaluate_parser.add_argument('--batch-size', type=int,
                                 help='Override evaluation batch size')
    evaluate_parser.add_argument('--device', type=str,
                                 help='Device to use (cpu, cuda, mps)')
    evaluate_parser.add_argument('--use-text-encoder-baseline', action='store_true',
                                 help='Skip contrastive model and evaluate using text encoder embeddings')

    return parser


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from file (YAML or JSON). Alias for load_config_from_file."""
    return load_config_from_file(config_path)


def load_config_from_file(config_path: str) -> TrainingConfig:
    """Load configuration from file (YAML or JSON)."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix.lower() in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install PyYAML")
        return TrainingConfig.from_yaml(str(config_path))
    elif config_path.suffix.lower() == '.json':
        return TrainingConfig.from_json(str(config_path))
    else:
        # Try to detect format from content
        try:
            return TrainingConfig.from_json(str(config_path))
        except json.JSONDecodeError:
            if YAML_AVAILABLE:
                return TrainingConfig.from_yaml(str(config_path))
            else:
                raise ValueError(
                    f"Unable to determine configuration file format: {config_path}")


def apply_env_overrides(config: TrainingConfig) -> TrainingConfig:
    """Apply environment variable overrides to configuration."""
    env_mappings = {
        'CL_BATCH_SIZE': ('batch_size', int),
        'CL_LEARNING_RATE': ('learning_rate', float),
        'CL_NUM_EPOCHS': ('num_epochs', int),
        'CL_TEMPERATURE': ('temperature', float),
        'CL_NEGATIVE_SAMPLING_RATIO': ('negative_sampling_ratio', float),
        'CL_PATHWAY_WEIGHT': ('pathway_weight', float),
        'CL_CHECKPOINT_FREQUENCY': ('checkpoint_frequency', int),
        'CL_LOG_FREQUENCY': ('log_frequency', int),
        'CL_MAX_RESUME_VIEWS': ('max_resume_views', int),
        'CL_MAX_JOB_VIEWS': ('max_job_views', int),
        'CL_USE_PATHWAY_NEGATIVES': ('use_pathway_negatives', lambda x: x.lower() == 'true'),
        'CL_USE_VIEW_AUGMENTATION': ('use_view_augmentation', lambda x: x.lower() == 'true'),
        'CL_SHUFFLE_DATA': ('shuffle_data', lambda x: x.lower() == 'true'),
        'CL_FALLBACK_ON_AUGMENTATION_FAILURE': ('fallback_on_augmentation_failure', lambda x: x.lower() == 'true'),
        'CL_HARD_NEGATIVE_MAX_DISTANCE': ('hard_negative_max_distance', float),
        'CL_MEDIUM_NEGATIVE_MAX_DISTANCE': ('medium_negative_max_distance', float),
        'CL_ESCO_GRAPH_PATH': ('esco_graph_path', str),
    }

    config_dict = config.to_dict()

    for env_var, (config_key, converter) in env_mappings.items():
        if env_var in os.environ:
            try:
                config_dict[config_key] = converter(os.environ[env_var])
                print(
                    f"Applied environment override: {config_key} = {config_dict[config_key]}")
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: Invalid value for {env_var}: {os.environ[env_var]} ({e})")

    return TrainingConfig.from_dict(config_dict)


def apply_cli_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply command-line argument overrides to configuration."""
    config_dict = config.to_dict()

    # Map CLI arguments to config keys
    cli_mappings = {
        'batch_size': 'batch_size',
        'learning_rate': 'learning_rate',
        'epochs': 'num_epochs',
        'temperature': 'temperature',
        'negative_sampling_ratio': 'negative_sampling_ratio',
        'pathway_weight': 'pathway_weight',
        'checkpoint_frequency': 'checkpoint_frequency',
        'log_frequency': 'log_frequency',
        'esco_graph_path': 'esco_graph_path',
    }

    for cli_arg, config_key in cli_mappings.items():
        if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
            config_dict[config_key] = getattr(args, cli_arg)

    # Handle boolean flags
    if hasattr(args, 'no_pathway_negatives') and args.no_pathway_negatives:
        config_dict['use_pathway_negatives'] = False

    if hasattr(args, 'no_view_augmentation') and args.no_view_augmentation:
        config_dict['use_view_augmentation'] = False

    if hasattr(args, 'no_shuffle') and args.no_shuffle:
        config_dict['shuffle_data'] = False

    return TrainingConfig.from_dict(config_dict)


def generate_example_config(scenario: str) -> Dict[str, Any]:
    """Generate example configuration for different scenarios."""
    base_config = {
        'batch_size': 256,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'temperature': 0.1,
        'negative_sampling_ratio': 0.7,
        'pathway_weight': 2.0,
        'use_pathway_negatives': True,
        'use_view_augmentation': True,
        'checkpoint_frequency': 1000,
        'log_frequency': 100,
        'shuffle_data': True,
        'max_resume_views': 5,
        'max_job_views': 5,
        'fallback_on_augmentation_failure': True,
        'hard_negative_max_distance': 2.0,
        'medium_negative_max_distance': 4.0,
        'esco_graph_path': None
    }

    if scenario == 'basic':
        return base_config

    elif scenario == 'large-dataset':
        return {
            **base_config,
            'batch_size': 512,
            'num_epochs': 5,
            'checkpoint_frequency': 500,
            'log_frequency': 50,
            'max_resume_views': 3,
            'max_job_views': 3,
        }

    elif scenario == 'fast-training':
        return {
            **base_config,
            'batch_size': 128,
            'learning_rate': 0.01,
            'num_epochs': 3,
            'use_view_augmentation': False,
            'checkpoint_frequency': 2000,
            'log_frequency': 200,
        }

    elif scenario == 'research':
        return {
            **base_config,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'num_epochs': 50,
            'temperature': 0.05,
            'negative_sampling_ratio': 0.5,
            'pathway_weight': 3.0,
            'checkpoint_frequency': 100,
            'log_frequency': 10,
            'max_resume_views': 10,
            'max_job_views': 10,
        }

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    try:
        # Set up logging
        log_level = getattr(logging, args.log_level.upper())
        logging.basicConfig(
            level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Load configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config = load_config_from_file(args.config)
        else:
            print("Using default configuration")
            config = TrainingConfig()

        # Apply environment variable overrides
        config = apply_env_overrides(config)

        # Apply CLI argument overrides
        config = apply_cli_overrides(config, args)

        print(f"Final configuration:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Use pathway negatives: {config.use_pathway_negatives}")
        print(f"  Use view augmentation: {config.use_view_augmentation}")

        # Create output directory
        output_dir = os.environ.get('CL_OUTPUT_DIR', args.output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Determine training mode and initialize appropriate trainer
        training_phase = config.training_phase
        print(f"Training phase: {training_phase}")
        print(f"Initializing trainer with output directory: {output_dir}")

        # Check if this is two-phase training (both configs provided or two-phase mode requested)
        is_two_phase = (
            hasattr(args, 'two_phase') and args.two_phase or
            training_phase == "two_phase" or
            (training_phase ==
             "fine_tuning" and not config.pretrained_model_path and not args.resume)
        )

        if is_two_phase:
            # Use TwoPhaseTrainingOrchestrator for sequential Phase 1 → Phase 2
            from .two_phase_orchestrator import TwoPhaseTrainingOrchestrator
            print("Using TwoPhaseTrainingOrchestrator for sequential training")
            print("  Phase 1: Self-supervised contrastive pre-training")
            print("  Phase 2: Supervised fine-tuning on labels")

            # Create orchestrator (it will handle both phases automatically)
            # We'll pass the main config and let orchestrator create phase-specific configs
            trainer = TwoPhaseTrainingOrchestrator(
                phase1_output_dir=f"{output_dir}_phase1",
                phase2_output_dir=f"{output_dir}_phase2",
                enable_validation=False,  # No separate validation data in this workflow
                enable_checkpointing=True,
                verbose=True
            )

            # Store config for two-phase execution
            trainer.base_config = config

        elif training_phase == "fine_tuning":
            # Use FineTuningTrainer for supervised fine-tuning only
            from .fine_tuning_trainer import FineTuningTrainer
            print("Using FineTuningTrainer for supervised fine-tuning")
            print(f"  Pre-trained model: {config.pretrained_model_path}")

            trainer = FineTuningTrainer(config, output_dir=output_dir)

            # Note: FineTuningTrainer automatically loads pretrained_model_path in __init__
            # So no need to call load_checkpoint separately

        else:
            # Use ContrastiveLearningTrainer for single-phase contrastive training
            print(
                f"Using ContrastiveLearningTrainer for {training_phase} training")

            trainer = ContrastiveLearningTrainer(
                config, output_dir=output_dir, esco_graph_path=getattr(args, 'esco_graph_path', None))

            # Resume from checkpoint if specified (only for ContrastiveLearningTrainer)
            if args.resume:
                print(f"Resuming training from checkpoint: {args.resume}")
                checkpoint_info = trainer.load_checkpoint(args.resume)
                print(
                    f"Resumed from epoch {checkpoint_info['epoch']} with loss {checkpoint_info['loss']:.6f}")

        # Validate dataset
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset file not found: {dataset_path}")
            return 1

        print(f"Starting training with dataset: {dataset_path}")

        # Run training based on trainer type
        if is_two_phase:
            # Two-phase training requires different execution
            # The orchestrator needs separate configs for each phase
            print("\n" + "="*80)
            print("STARTING TWO-PHASE TRAINING")
            print("="*80)

            # For now, use the same dataset for both phases
            # In production, you might want separate datasets
            from .data_structures import TrainingConfig

            # Create Phase 1 config (self-supervised)
            phase1_config = TrainingConfig(**{
                **{k: v for k, v in config.__dict__.items() if not k.startswith('_')},
                'training_phase': 'self_supervised',
                'use_augmentation_labels_only': True,
                'augmentation_positive_ratio': 0.5,
                'pretrained_model_path': None,
                'freeze_contrastive_layers': False
            })

            # Create Phase 2 config (fine-tuning)
            phase2_config = TrainingConfig(**{
                **{k: v for k, v in config.__dict__.items() if not k.startswith('_')},
                'training_phase': 'fine_tuning',
                'use_augmentation_labels_only': False,
                'augmentation_positive_ratio': 1.0,
                'freeze_contrastive_layers': True,
                'learning_rate': config.learning_rate * 0.5  # Lower LR for fine-tuning
            })

            # Execute Phase 1
            print("\n>>> Phase 1: Self-Supervised Pre-training")
            phase1_trainer = ContrastiveLearningTrainer(
                phase1_config,
                output_dir=f"{output_dir}_phase1",
                esco_graph_path=getattr(args, 'esco_graph_path', None)
            )
            phase1_results = phase1_trainer.train(dataset_path)

            print(
                f"\n✓ Phase 1 completed: loss {phase1_results.final_loss:.6f}, time {phase1_results.training_time:.2f}s")

            # Get best checkpoint from Phase 1
            phase1_checkpoint = f"{output_dir}_phase1/best_checkpoint.pt"
            phase2_config.pretrained_model_path = phase1_checkpoint

            # Execute Phase 2
            print("\n>>> Phase 2: Supervised Fine-tuning")
            from .fine_tuning_trainer import FineTuningTrainer
            phase2_trainer = FineTuningTrainer(
                phase2_config, output_dir=f"{output_dir}_phase2")
            phase2_results = phase2_trainer.train(dataset_path)

            print(
                f"\n✓ Phase 2 completed: loss {phase2_results.final_loss:.6f}, time {phase2_results.training_time:.2f}s")

            # Combine results for reporting
            total_time = phase1_results.training_time + phase2_results.training_time
            print("\n" + "="*80)
            print("TWO-PHASE TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(
                f"Phase 1 (Self-supervised): loss {phase1_results.final_loss:.6f}, {phase1_results.training_time:.2f}s")
            print(
                f"Phase 2 (Fine-tuning):      loss {phase2_results.final_loss:.6f}, {phase2_results.training_time:.2f}s")
            print(f"Total training time:        {total_time:.2f}s")
            print(f"Phase 1 checkpoint:         {phase1_checkpoint}")
            print(
                f"Final model:                {output_dir}_phase2/best_checkpoint.pt")

            # Use phase2 results as final results for return
            results = phase2_results

        else:
            # Single-phase training
            results = trainer.train(dataset_path)

            # Print results
            print("\nTraining completed successfully!")

            # Handle different return types (dict for fine-tuning, TrainingResults for others)
            if isinstance(results, dict):
                print(f"Final loss: {results.get('final_loss', 'N/A')}")
                print(
                    f"Final accuracy: {results.get('final_accuracy', 'N/A')}")
                print(
                    f"Training time: {results.get('training_time', 0):.2f} seconds")
                print(
                    f"Total samples processed: {results.get('total_samples', 'N/A')}")
                print(
                    f"Total batches processed: {results.get('total_batches', 'N/A')}")
            else:
                print(f"Final loss: {results.final_loss:.6f}")
                print(f"Training time: {results.training_time:.2f} seconds")
                print(f"Total samples processed: {results.total_samples}")
                print(f"Total batches processed: {results.total_batches}")
                print(f"Checkpoints saved: {len(results.checkpoint_paths)}")

        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 130
    except Exception as e:
        print(f"Error during training: {e}")
        logging.exception("Training failed")
        return 1


def cmd_generate_config(args: argparse.Namespace) -> int:
    """Execute the generate-config command."""
    try:
        # Generate configuration
        config_data = generate_example_config(args.scenario)

        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save configuration
        if args.format == 'yaml':
            if not YAML_AVAILABLE:
                print(
                    "Error: PyYAML is required for YAML format. Install with: pip install PyYAML")
                return 1

            with open(output_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        else:  # json
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)

        print(f"Generated {args.scenario} configuration: {output_path}")
        print(f"Format: {args.format}")

        return 0

    except Exception as e:
        print(f"Error generating configuration: {e}")
        return 1


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Execute the validate-config command."""
    try:
        # Load and validate configuration
        config = load_config_from_file(args.config)

        print(f"Configuration file is valid: {args.config}")
        print(f"Configuration summary:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Temperature: {config.temperature}")
        print(
            f"  Negative sampling ratio (deprecated, ignored): {config.negative_sampling_ratio}")
        print(f"  Pathway weight (deprecated, handled by selection): {config.pathway_weight}")
        print(f"  Ontology weight (sample-level): {config.ontology_weight}")
        print(f"  Use pathway negatives: {config.use_pathway_negatives}")
        print(f"  Use view augmentation: {config.use_view_augmentation}")

        return 0

    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Execute the status command."""
    try:
        output_dir = Path(args.output_dir)

        if not output_dir.exists():
            print(f"Output directory not found: {output_dir}")
            return 1

        # Check for configuration file
        config_file = output_dir / "training_config.json"
        if config_file.exists():
            print(f"Configuration: {config_file}")
        else:
            print("No configuration file found")

        # Check for checkpoints
        checkpoint_files = list(output_dir.glob("checkpoint_*.pt"))
        if checkpoint_files:
            print(f"Checkpoints found: {len(checkpoint_files)}")
            latest_checkpoint = max(
                checkpoint_files, key=lambda p: p.stat().st_mtime)
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoints found")

        # Check for results
        results_file = output_dir / "training_results.json"
        if results_file.exists():
            results = TrainingResults.from_json(str(results_file))
            print(f"Training results:")
            print(f"  Final loss: {results.final_loss:.6f}")
            print(f"  Training time: {results.training_time:.2f} seconds")
            print(f"  Total samples: {results.total_samples}")
            print(f"  Total batches: {results.total_batches}")
        else:
            print("No training results found")

        # Check for logs
        log_dir = output_dir / "logs"
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            print(f"Log files: {len(log_files)}")
        else:
            print("No log directory found")

        return 0

    except Exception as e:
        print(f"Error checking status: {e}")
        return 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Execute the pipeline command."""
    try:
        # Setup basic logging first
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

        logger.info("🚀 Starting ML Pipeline")

        # Load or create pipeline configuration
        if args.config:
            logger.info(f"Loading pipeline config from: {args.config}")
            config = PipelineConfig.from_file(args.config)
        elif args.quick:
            logger.info("Using quick pipeline configuration")
            config = create_quick_pipeline_config()
        else:
            logger.info("Using default pipeline configuration")
            config = create_default_pipeline_config()

        # Apply command line overrides
        if args.output_dir:
            config.output_base_dir = args.output_dir
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        if args.device:
            config.device = args.device
        if args.skip_data_splitting:
            config.skip_data_splitting = args.skip_data_splitting
        if args.skip_training:
            config.skip_training = args.skip_training
        if args.skip_testing:
            config.skip_testing = args.skip_testing

        # Initialize and run pipeline
        pipeline = MLPipeline(config)
        results = pipeline.run_complete_pipeline(
            dataset_path=args.dataset,
            train_data=args.train_data,
            val_data=args.val_data,
            test_data=args.test_data
        )

        # Print summary
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Experiment: {results.experiment_name}")
        print(f"⏱️  Duration: {results.total_duration:.2f} seconds")

        if results.test_metrics:
            print("\n📊 Final Test Results:")
            for metric, value in results.test_metrics.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.4f}")

        print(f"\n📋 Reports: {len(results.report_paths)} generated")
        print(
            f"📊 Visualizations: {len(results.visualization_paths)} generated")
        print("="*80)

        return 0

    except Exception as e:
        # Setup logger for error handling if not already done
        if 'logger' not in locals():
            logging.basicConfig(level=logging.ERROR)
            logger = logging.getLogger(__name__)

        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        return 1


def cmd_convert_data(args: argparse.Namespace) -> int:
    """Execute the convert-data command."""
    try:
        # Set up logging
        log_level = getattr(logging, args.log_level.upper())
        logging.basicConfig(
            level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        logger.info("Starting data conversion process...")

        # Validate input files exist
        for file_path in [args.labeled_data, args.job_descriptions, args.resume_data]:
            if not Path(file_path).exists():
                print(f"Error: Input file not found: {file_path}")
                return 1

        # Create adapter configuration
        config = DataAdapterConfig(
            train_labeled_path=args.labeled_data,
            job_descriptions_path=args.job_descriptions,
            resume_data_path=args.resume_data,
            output_path=args.output,
            min_job_description_length=args.min_job_desc_length,
            min_resume_experience_length=args.min_resume_exp_length,
            max_samples_per_user=args.max_samples_per_user
        )

        print(f"Converting data with configuration:")
        print(f"  Labeled data: {config.train_labeled_path}")
        print(f"  Job descriptions: {config.job_descriptions_path}")
        print(f"  Resume data: {config.resume_data_path}")
        print(f"  Output: {config.output_path}")
        print(
            f"  Min job description length: {config.min_job_description_length}")
        print(
            f"  Min resume experience length: {config.min_resume_experience_length}")
        print(f"  Max samples per user: {config.max_samples_per_user}")

        # Create adapter and convert data
        adapter = DataAdapter(config)
        adapter.load_data()

        # Print data statistics
        stats = adapter.get_data_statistics()
        print("\nData Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Convert and save
        adapter.convert_and_save()

        print(f"\nData conversion completed successfully!")
        print(f"Converted data saved to: {config.output_path}")

        # Validate output file was created
        output_path = Path(config.output_path)
        if output_path.exists():
            # Count lines to give feedback on conversion success
            with open(output_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"Output file contains {line_count} training samples")

        return 0

    except Exception as e:
        print(f"Error during data conversion: {e}")
        logging.exception("Data conversion failed")
        return 1


def _create_jsonl_dataloader(data_path: str, batch_size: int):
    """Create a DataLoader for JSONL evaluation data."""
    from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader

    class JSONLDataset(Dataset):
        def __init__(self, file_path: str):
            self.samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            sample = json.loads(line)
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

        def __getitem__(self, idx: int):
            return self.samples[idx]

    dataset = JSONLDataset(data_path)
    if len(dataset) == 0:
        logger.warning(f"No valid samples found in {data_path}")
        return None

    return PyTorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


def _load_model_from_checkpoint(checkpoint_path: str, config: TrainingConfig, device: str):
    """Load a model from checkpoint using the training config."""
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required unless using baseline mode.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            from .trainer import ContrastiveLearningTrainer
            temp_trainer = ContrastiveLearningTrainer(config)
            model = temp_trainer.model
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("No model found in checkpoint")
    else:
        model = checkpoint

    return model.to(device)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluate command."""
    try:
        config = load_config_from_file(args.config)

        if args.batch_size:
            config.batch_size = args.batch_size

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        dataset_path = args.dataset or "data/converted_training_data.jsonl"
        if not Path(dataset_path).exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {dataset_path}. "
                "Provide --dataset to specify the JSONL file.")

        data_loader = _create_jsonl_dataloader(dataset_path, config.batch_size)
        if data_loader is None:
            raise ValueError("No valid samples found for evaluation.")

        from sentence_transformers import SentenceTransformer
        text_encoder = SentenceTransformer(config.text_encoder_model)
        text_encoder.to(device)

        eval_config = EvaluationConfig(
            batch_size=config.batch_size,
            device=device,
            use_text_encoder_baseline=args.use_text_encoder_baseline
        )

        evaluator = ContrastiveEvaluator(eval_config, text_encoder=text_encoder)

        model = None
        if not args.use_text_encoder_baseline:
            model = _load_model_from_checkpoint(args.checkpoint, config, device)

        results = evaluator.evaluate_model(
            model, data_loader, args.output_dir)

        print("Evaluation metrics:")
        for metric_name, value in results.metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")

        return 0
    except Exception as e:
        print(f"Evaluation failed: {e}")
        logging.exception("Evaluation failed")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == 'train':
        return cmd_train(args)
    elif args.command == 'pipeline':
        return cmd_pipeline(args)
    elif args.command == 'generate-config':
        return cmd_generate_config(args)
    elif args.command == 'validate-config':
        return cmd_validate_config(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'convert-data':
        return cmd_convert_data(args)
    elif args.command == 'evaluate':
        return cmd_evaluate(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
