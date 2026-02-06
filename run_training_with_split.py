#!/usr/bin/env python3
"""
Training pipeline with enforced 80/10/10 train/validation/test splitting.

Phases:
- Phase 1: Contrastive pretraining on training set
- Phase 1 Eval: Embedding quality baseline on validation set
- Phase 2: Classification fine-tuning with validation monitoring
- Phase 2 Eval: Final results on test set (threshold tuned on validation)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from contrastive_learning.data_splitter import DataSplitter, SplitConfig
from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.fine_tuning_trainer import FineTuningTrainer
from contrastive_learning.trainer import ContrastiveLearningTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_dataset(dataset_path: str, output_dir: str, strategy: str, seed: int) -> dict:
    """Split dataset into train/validation/test sets (80/10/10)."""
    logger.info(f"Splitting {dataset_path} → {output_dir} (strategy={strategy}, seed={seed})")

    splitter = DataSplitter(SplitConfig(
        strategy=strategy,
        ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        seed=seed,
        validate_splits=True
    ))

    result = splitter.split_dataset(dataset_path, output_dir)

    # Log split statistics
    for name, info in result.statistics.get('splits', {}).items():
        logger.info(f"  {name}: {info['count']} samples ({info['percentage']:.1f}%)")

    # Check for data leakage
    if result.validation_report and result.validation_report.get('data_leakage'):
        leakage = result.validation_report['data_leakage']
        overlaps = [
            leakage.get('train_validation_overlap', 0),
            leakage.get('train_test_overlap', 0),
            leakage.get('validation_test_overlap', 0)
        ]
        if any(overlaps):
            logger.warning(f"⚠️ Data leakage detected: {overlaps}")
        else:
            logger.info("✓ No data leakage")

    return result.splits


def run_phase1(train_path: str, config_path: str, output_dir: str = "phase1_pretraining",
               validation_path: str = None):
    """Run Phase 1 contrastive pretraining."""
    logger.info(f"Phase 1: Training on {train_path}")

    config = TrainingConfig.from_json(config_path)
    
    # Override validation path if provided
    if validation_path:
        config.validation_path = validation_path
        logger.info(f"  Validation: {validation_path}")
    
    trainer = ContrastiveLearningTrainer(config=config, output_dir=output_dir)
    results = trainer.train(train_path)

    # Log validation loss if available
    if results.validation_losses:
        logger.info(f"✓ Phase 1 complete - train_loss: {results.final_loss:.6f}, "
                   f"val_loss: {results.validation_losses[-1]:.6f}, time: {results.training_time:.2f}s")
    else:
        logger.info(f"✓ Phase 1 complete - loss: {results.final_loss:.6f}, time: {results.training_time:.2f}s")
    
    # Prefer best_checkpoint.pt (based on validation loss) over last epoch checkpoint
    best_checkpoint_path = Path(output_dir) / "best_checkpoint.pt"
    if best_checkpoint_path.exists():
        logger.info(f"  Using best checkpoint: {best_checkpoint_path}")
        return str(best_checkpoint_path)
    
    # Fallback to last checkpoint if best doesn't exist
    return results.checkpoint_paths[-1] if results.checkpoint_paths else None


def run_phase2(train_path: str, config_path: str, pretrained_path: str,
               validation_path: str = None, output_dir: str = "phase2_finetuning"):
    """Run Phase 2 classification fine-tuning."""
    logger.info(f"Phase 2: Training on {train_path}, pretrained={pretrained_path}")

    config = TrainingConfig.from_json(config_path)
    config.pretrained_model_path = pretrained_path

    trainer = FineTuningTrainer(config=config, output_dir=output_dir)
    results = trainer.train(train_path, validation_data_path=validation_path)

    # Handle both dict and TrainingResults
    if isinstance(results, dict):
        checkpoints = results.get('checkpoint_paths', [])
        accuracy = results.get('final_accuracy', 'N/A')
        time_taken = results.get('training_time', 0)
    else:
        checkpoints = results.checkpoint_paths
        accuracy = results.final_accuracy
        time_taken = results.training_time

    logger.info(f"✓ Phase 2 complete - accuracy: {accuracy}, time: {time_taken:.2f}s")
    return checkpoints[-1] if checkpoints else None


def run_evaluation(script: str, dataset: str, checkpoint: str, config: str,
                   output_dir: str, validation_dataset: str = None) -> bool:
    """Run evaluation script."""
    cmd = ["python", script, "--dataset", dataset, "--checkpoint", checkpoint,
           "--config", config, "--output-dir", output_dir]

    if validation_dataset:
        cmd.extend(["--validation-dataset", validation_dataset])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Training with 80/10/10 data splitting")
    parser.add_argument('--dataset', required=True, help='Path to full dataset (JSONL)')
    parser.add_argument('--phase1-config', default='config/phase1_pretraining_config.json')
    parser.add_argument('--phase2-config', default='config/phase2_finetuning_config.json')
    parser.add_argument('--split-strategy', choices=['random', 'stratified', 'sequential'],
                        default='sequential')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--splits-dir', default='data_splits')
    parser.add_argument('--use-existing-splits', action='store_true')
    parser.add_argument('--train-file', type=str, help='Custom training file (overrides splits/train.jsonl)')
    parser.add_argument('--skip-phase1', action='store_true')
    parser.add_argument('--skip-phase2', action='store_true')
    parser.add_argument('--skip-phase1-eval', action='store_true')
    parser.add_argument('--skip-phase2-eval', action='store_true')
    parser.add_argument('--phase1-checkpoint', type=str, help='Existing Phase 1 checkpoint')
    parser.add_argument('--phase2-checkpoint', type=str, help='Existing Phase 2 checkpoint (skips Phase 2 training)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return 1

    # Step 1: Get or create splits
    if args.use_existing_splits:
        splits = {
            'train': str(Path(args.splits_dir) / 'train.jsonl'),
            'validation': str(Path(args.splits_dir) / 'validation.jsonl'),
            'test': str(Path(args.splits_dir) / 'test.jsonl')
        }
        for name, path in splits.items():
            if not Path(path).exists():
                logger.error(f"Split not found: {path}")
                return 1
        logger.info(f"Using existing splits from {args.splits_dir}")
    else:
        splits = split_dataset(args.dataset, args.splits_dir, args.split_strategy, args.split_seed)

    train_path, val_path, test_path = splits['train'], splits['validation'], splits['test']

    # Early exit if only splitting
    if args.skip_phase1 and args.skip_phase2 and args.skip_phase2_eval:
        logger.info("✅ Splitting complete, all training skipped")
        return 0

    # Step 2: Phase 1 training
    phase1_ckpt = args.phase1_checkpoint
    if not args.skip_phase1:
        if not Path(args.phase1_config).exists():
            logger.error(f"Config not found: {args.phase1_config}")
            return 1
        phase1_ckpt = run_phase1(train_path, args.phase1_config, validation_path=val_path)
    elif not phase1_ckpt or not Path(phase1_ckpt).exists():
        logger.error("Phase 1 checkpoint required when skipping Phase 1")
        return 1

    # Step 2.5: Phase 1 evaluation
    if not args.skip_phase1_eval and phase1_ckpt:
        if not run_evaluation("run_phase1_embedding_evaluation.py", val_path,
                              phase1_ckpt, args.phase1_config, "phase1_evaluation"):
            logger.warning("Phase 1 evaluation failed, continuing...")

    # Step 3: Phase 2 training
    phase2_ckpt = args.phase2_checkpoint
    if phase2_ckpt:
        # Use existing Phase 2 checkpoint (skip training)
        if not Path(phase2_ckpt).exists():
            logger.error(f"Phase 2 checkpoint not found: {phase2_ckpt}")
            return 1
        logger.info(f"Using existing Phase 2 checkpoint: {phase2_ckpt}")
    elif not args.skip_phase2:
        if not Path(args.phase2_config).exists():
            logger.error(f"Config not found: {args.phase2_config}")
            return 1
        phase2_ckpt = run_phase2(train_path, args.phase2_config, phase1_ckpt, val_path)

    # Step 4: Phase 2 evaluation
    if not args.skip_phase2_eval and phase2_ckpt:
        run_evaluation("run_phase2_classification_evaluation.py", test_path,
                       phase2_ckpt, args.phase2_config, "test_evaluation", val_path)

    # Summary
    logger.info("=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info(f"  Splits: {train_path}, {val_path}, {test_path}")
    if phase1_ckpt:
        logger.info(f"  Phase 1: {phase1_ckpt}")
    if phase2_ckpt:
        logger.info(f"  Phase 2: {phase2_ckpt}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
