#!/usr/bin/env python3
"""
Complete training script for running the MLPipeline with full training workflow.
This script demonstrates how to run both single-phase and two-phase training.
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

from contrastive_learning.pipeline import MLPipeline
from contrastive_learning.pipeline_config import (
    PipelineConfig,
    TwoPhaseTrainingConfig,
    DataSplittingConfig,
    ValidationConfig,
    TestingConfig,
    EarlyStoppingConfig,
    CheckpointingConfig,
    ReportingConfig
)


def setup_logging():
    """Setup logging for the training script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def create_training_config():
    """Create a basic training configuration file if it doesn't exist"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    training_config_path = config_dir / "training_config.json"

    if not training_config_path.exists():
        training_config = {
            "model_architecture": "contrastive",
            "text_encoder_model": "sentence-transformers/all-mpnet-base-v2",
            "embedding_dim": 384,
            "projection_dim": 128,
            "temperature": 0.07,

            "num_epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.01,

            "device": "cpu",
            "save_frequency": 2,
            "validation_frequency": 1,

            "use_augmentation_labels_only": False,
            "augmentation_positive_ratio": 0.3
        }

        with open(training_config_path, 'w') as f:
            json.dump(training_config, f, indent=2)

        print(f"✅ Created training config: {training_config_path}")

    return str(training_config_path)


def run_single_phase_training(dataset_path: str, output_dir: str = "single_phase_output"):
    """Run single-phase training"""
    print("\n🚀 Starting Single-Phase Training")
    print("=" * 60)

    # Create training config if needed
    training_config_path = create_training_config()

    # Configure pipeline for single-phase training
    config = PipelineConfig(
        # Core configuration
        training_config_path=training_config_path,
        output_base_dir=output_dir,
        experiment_name=f"single_phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

        # Data splitting configuration
        data_splitting=DataSplittingConfig(
            strategy="random",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=42,
            validate_splits=True
        ),

        # Validation configuration
        validation=ValidationConfig(
            frequency="every_epoch",
            metrics=["contrastive_loss", "embedding_quality"],
            save_predictions=True,
            generate_plots=True
        ),

        # Testing configuration
        testing=TestingConfig(
            save_embeddings=True,
            save_predictions=True,
            generate_visualizations=True,
            batch_size=64
        ),

        # Early stopping
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            metric="contrastive_loss",
            patience=3,
            mode="min",
            min_delta=0.001
        ),

        # Checkpointing
        checkpointing=CheckpointingConfig(
            save_frequency=2,
            keep_best=True,
            keep_last=True
        ),

        # Reporting
        reporting=ReportingConfig(
            formats=["json", "html"],
            include_plots=True
        ),

        # Resource settings
        device="cpu",  # Change to "cuda" if you have GPU
        num_workers=4
    )

    try:
        # Initialize and run pipeline
        pipeline = MLPipeline(config)

        print(f"🎯 Training mode: {pipeline.detected_mode.value}")
        print(f"🛠️  Strategy: {pipeline.training_strategy.__class__.__name__}")

        # Run complete pipeline
        results = pipeline.run_complete_pipeline(dataset_path=dataset_path)

        print("\n✅ Single-Phase Training Completed!")
        print(f"📁 Final model: {results.final_model_path}")
        print(f"📁 Best model: {results.best_model_path}")
        print(f"⏱️  Total duration: {results.total_duration:.2f} seconds")
        print(f"📊 Reports generated: {len(results.report_paths)} files")

        return results

    except Exception as e:
        print(f"❌ Single-phase training failed: {e}")
        raise


def run_two_phase_training(dataset_path: str, output_dir: str = "two_phase_output"):
    """Run two-phase training"""
    print("\n🔄 Starting Two-Phase Training")
    print("=" * 60)

    # Ensure configuration files exist
    config_dir = Path("config")
    phase1_config_path = config_dir / "phase1_pretraining_config.json"
    phase2_config_path = config_dir / "phase2_finetuning_config.json"

    if not phase1_config_path.exists() or not phase2_config_path.exists():
        print("❌ Two-phase configuration files not found!")
        print("💡 Run the example_two_phase_config.py script first to create them")
        return None

    # Create fallback training config
    training_config_path = create_training_config()

    # Configure two-phase training
    two_phase_config = TwoPhaseTrainingConfig(
        enabled=True,
        phase1_config_path=str(phase1_config_path),
        phase2_config_path=str(phase2_config_path),
        phase1_output_dir="phase1_pretraining",
        phase2_output_dir="phase2_finetuning",
        phase1_data_strategy="augmentation_only",
        phase2_data_strategy="labeled_only",
        checkpoint_save_strategy="best_only"
    )

    # Configure pipeline for two-phase training
    config = PipelineConfig(
        # Core configuration
        training_config_path=training_config_path,  # Fallback config
        output_base_dir=output_dir,
        experiment_name=f"two_phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        two_phase_training=two_phase_config,

        # Data splitting configuration
        data_splitting=DataSplittingConfig(
            strategy="random",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=42,
            validate_splits=True
        ),

        # Validation configuration
        validation=ValidationConfig(
            frequency="every_epoch",
            metrics=["contrastive_loss", "embedding_quality",
                     "classification_accuracy"],
            save_predictions=True,
            generate_plots=True
        ),

        # Testing configuration
        testing=TestingConfig(
            save_embeddings=True,
            save_predictions=True,
            generate_visualizations=True,
            batch_size=64
        ),

        # Early stopping
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            metric="contrastive_loss",
            patience=5,
            mode="min",
            min_delta=0.001
        ),

        # Checkpointing
        checkpointing=CheckpointingConfig(
            save_frequency=1,
            keep_best=True,
            keep_last=True
        ),

        # Reporting
        reporting=ReportingConfig(
            formats=["json", "html"],
            include_plots=True
        ),

        # Resource settings
        device="cpu",  # Change to "cuda" if you have GPU
        num_workers=4
    )

    try:
        # Initialize and run pipeline
        pipeline = MLPipeline(config)

        print(f"🎯 Training mode: {pipeline.detected_mode.value}")
        print(f"🛠️  Strategy: {pipeline.training_strategy.__class__.__name__}")

        # Run complete pipeline
        results = pipeline.run_complete_pipeline(dataset_path=dataset_path)

        print("\n✅ Two-Phase Training Completed!")
        print(
            f"📁 Pre-trained model: {results.phase1_results.final_model_path if results.phase1_results else 'N/A'}")
        print(f"📁 Final model: {results.final_model_path}")
        print(f"📁 Best model: {results.best_model_path}")
        print(f"⏱️  Total duration: {results.total_duration:.2f} seconds")
        print(f"📊 Reports generated: {len(results.report_paths)} files")

        # Two-phase specific results
        if results.phase1_results:
            print(
                f"🔹 Phase 1 final loss: {results.phase1_results.final_loss:.4f}")
        if results.phase2_results:
            print(
                f"🔹 Phase 2 final accuracy: {results.phase2_results.get('final_accuracy', 'N/A')}")

        return results

    except Exception as e:
        print(f"❌ Two-phase training failed: {e}")
        raise


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Run MLPipeline Training")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset file (JSONL format)")
    parser.add_argument("--mode", choices=["single", "two-phase", "both"], default="single",
                        help="Training mode to run")
    parser.add_argument("--output-dir", default="pipeline_output",
                        help="Base output directory")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    print("🎯 MLPipeline Full Training Script")
    print("=" * 70)
    print(f"📁 Dataset: {args.dataset}")
    print(f"🎛️  Mode: {args.mode}")
    print(f"📂 Output: {args.output_dir}")

    # Validate dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset file not found: {args.dataset}")
        return 1

    try:
        if args.mode == "single":
            results = run_single_phase_training(
                args.dataset, f"{args.output_dir}/single_phase")

        elif args.mode == "two-phase":
            results = run_two_phase_training(
                args.dataset, f"{args.output_dir}/two_phase")

        elif args.mode == "both":
            print("🔄 Running both training modes for comparison...")

            # Run single-phase
            single_results = run_single_phase_training(
                args.dataset, f"{args.output_dir}/single_phase")

            # Run two-phase
            two_phase_results = run_two_phase_training(
                args.dataset, f"{args.output_dir}/two_phase")

            # Compare results
            print("\n📊 Training Comparison:")
            print(
                f"Single-phase duration: {single_results.total_duration:.2f}s")
            if two_phase_results:
                print(
                    f"Two-phase duration: {two_phase_results.total_duration:.2f}s")

        print("\n🎉 Training completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
