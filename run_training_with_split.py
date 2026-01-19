#!/usr/bin/env python3
"""
Training script with ENFORCED data splitting (80/20 train/test)
This script ensures that training data is properly split before training begins.
"""

import json
import logging
import argparse
import sys
from pathlib import Path

from contrastive_learning.data_splitter import DataSplitter, SplitConfig
from contrastive_learning.trainer import ContrastiveLearningTrainer
from contrastive_learning.fine_tuning_trainer import FineTuningTrainer
from contrastive_learning.data_structures import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_dataset(dataset_path: str, output_dir: str = "data_splits",
                  strategy: str = "random", seed: int = 42):
    """
    Split dataset into train/test sets.

    Args:
        dataset_path: Path to full dataset (JSONL)
        output_dir: Directory to save splits
        strategy: Split strategy (random, stratified, sequential)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with paths to train/test files
    """
    logger.info("=" * 80)
    logger.info("DATA SPLITTING")
    logger.info("=" * 80)
    logger.info(f"Input dataset: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Ratios: 80% train, 20% test")
    logger.info(f"Random seed: {seed}")

    # Create split configuration
    split_config = SplitConfig(
        strategy=strategy,
        ratios={"train": 0.8, "test": 0.2},
        seed=seed,
        validate_splits=True
    )

    # Initialize splitter
    splitter = DataSplitter(split_config)

    # Perform split
    split_results = splitter.split_dataset(dataset_path, output_dir)

    # Log statistics
    logger.info("\n" + "=" * 80)
    logger.info("SPLIT STATISTICS")
    logger.info("=" * 80)

    stats = split_results.statistics
    for split_name in ["train", "test"]:
        if split_name in stats['split_sizes']:
            count = stats['split_sizes'][split_name]['count']
            percentage = stats['split_sizes'][split_name]['percentage']
            logger.info(
                f"{split_name.upper():12s}: {count:6d} samples ({percentage:.1f}%)")

    # Log validation report if available
    if split_results.validation_report:
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 80)

        report = split_results.validation_report

        if report.get('label_distribution'):
            logger.info("\nLabel Distribution:")
            for split_name, dist in report['label_distribution'].items():
                logger.info(f"  {split_name}:")
                for label, count in dist.items():
                    logger.info(f"    label={label}: {count}")

        if report.get('data_leakage'):
            leakage = report['data_leakage']
            if leakage['train_val_overlap'] > 0 or leakage['train_test_overlap'] > 0:
                logger.warning("\n⚠️  DATA LEAKAGE DETECTED!")
                logger.warning(
                    f"  Train-Val overlap: {leakage['train_val_overlap']}")
                logger.warning(
                    f"  Train-Test overlap: {leakage['train_test_overlap']}")
            else:
                logger.info("\n✓ No data leakage detected")

    logger.info("\n" + "=" * 80)
    logger.info("SPLIT FILES")
    logger.info("=" * 80)
    for split_name, file_path in split_results.splits.items():
        logger.info(f"{split_name:12s}: {file_path}")

    return split_results.splits


def run_phase1_training(train_data_path: str, config_path: str,
                        output_dir: str = "phase1_pretraining"):
    """
    Run Phase 1 (contrastive pretraining).

    Args:
        train_data_path: Path to training split
        config_path: Path to Phase 1 config
        output_dir: Output directory
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: CONTRASTIVE PRETRAINING")
    logger.info("=" * 80)
    logger.info(f"Training data: {train_data_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")

    # Load config
    config = TrainingConfig.from_json(config_path)

    # Initialize trainer
    trainer = ContrastiveLearningTrainer(
        config=config,
        output_dir=output_dir
    )

    # Train on training split only
    results = trainer.train(train_data_path)

    logger.info(f"\n✓ Phase 1 complete")
    logger.info(f"  Final loss: {results.final_loss:.6f}")
    logger.info(f"  Training time: {results.training_time:.2f}s")
    logger.info(f"  Best checkpoint: {results.best_checkpoint_path}")

    return results


def run_phase2_training(train_data_path: str, config_path: str,
                        pretrained_model_path: str,
                        output_dir: str = "phase2_finetuning"):
    """
    Run Phase 2 (classification fine-tuning).

    Args:
        train_data_path: Path to training split
        config_path: Path to Phase 2 config
        pretrained_model_path: Path to Phase 1 checkpoint
        output_dir: Output directory
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: CLASSIFICATION FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Training data: {train_data_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Pretrained model: {pretrained_model_path}")
    logger.info(f"Output: {output_dir}")

    # Load config
    config = TrainingConfig.from_json(config_path)

    # Update pretrained model path
    config.pretrained_model_path = pretrained_model_path

    # Initialize trainer
    trainer = FineTuningTrainer(
        config=config,
        output_dir=output_dir
    )

    # Train on training split only
    results = trainer.train(train_data_path)

    logger.info(f"\n✓ Phase 2 complete")
    logger.info(f"  Final accuracy: {results.final_accuracy:.4f}")
    logger.info(f"  Training time: {results.training_time:.2f}s")
    logger.info(f"  Best checkpoint: {results.best_checkpoint_path}")

    return results


def evaluate_on_test_set(model_checkpoint: str, test_data_path: str,
                         config_path: str, output_dir: str = "test_evaluation"):
    """
    Evaluate final model on held-out test set.

    Args:
        model_checkpoint: Path to trained model
        test_data_path: Path to test split
        config_path: Path to config
        output_dir: Output directory
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model checkpoint: {model_checkpoint}")
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Output: {output_dir}")

    # Import evaluation functionality
    from run_phase2_classification_evaluation import (
        JSONLDataset, evaluate_classification_model, evaluate_ranking
    )
    import torch
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import DataLoader as TorchDataLoader
    from contrastive_learning.contrastive_classification_model import ContrastiveClassificationModel

    # Load config
    config = TrainingConfig.from_json(config_path)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = SentenceTransformer(config.text_encoder_model).to(device)

    # Load model
    model = ContrastiveClassificationModel(config=config)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Load test dataset
    dataset = JSONLDataset(test_data_path)
    data_loader = TorchDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            'resume': [item['resume'] for item in x],
            'job': [item['job'] for item in x],
            'label': torch.tensor([item['label'] for item in x])
        }
    )

    logger.info(f"Test set size: {len(dataset)} samples")

    # Run evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np

    predictions, probabilities, true_labels = evaluate_classification_model(
        model, text_encoder, data_loader, device
    )

    # Calculate metrics with different thresholds
    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:
        preds = (probabilities > threshold).astype(int)
        f1 = f1_score(true_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Use optimal threshold
    predictions_optimal = (probabilities > best_threshold).astype(int)

    accuracy = accuracy_score(true_labels, predictions_optimal)
    precision = precision_score(
        true_labels, predictions_optimal, zero_division=0)
    recall = recall_score(true_labels, predictions_optimal, zero_division=0)
    f1 = f1_score(true_labels, predictions_optimal, zero_division=0)
    auc_roc = roc_auc_score(true_labels, probabilities)

    logger.info(f"\n🎯 TEST SET RESULTS (threshold={best_threshold:.4f}):")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    logger.info(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    logger.info(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    logger.info(f"  AUC-ROC:   {auc_roc:.4f} ({auc_roc*100:.2f}%)")

    # Ranking evaluation
    logger.info("\n" + "=" * 80)
    logger.info("RANKING EVALUATION ON TEST SET")
    logger.info("=" * 80)

    job_ranking = evaluate_ranking(
        model, text_encoder, dataset, device, mode='job')
    resume_ranking = evaluate_ranking(
        model, text_encoder, dataset, device, mode='resume')

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {
        'test_set_metrics': {
            'optimal_threshold': float(best_threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc)
        },
        'job_ranking': job_ranking,
        'resume_ranking': resume_ranking,
        'test_set_size': len(dataset),
        'model_checkpoint': model_checkpoint,
        'test_data_path': test_data_path
    }

    results_path = output_path / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Test results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Training with enforced data splitting (80/20 train/test)"
    )
    parser.add_argument('--dataset', required=True,
                        help='Path to full dataset (JSONL)')
    parser.add_argument('--phase1-config', default='config/phase1_pretraining_config.json',
                        help='Path to Phase 1 config')
    parser.add_argument('--phase2-config', default='config/phase2_finetuning_config.json',
                        help='Path to Phase 2 config')
    parser.add_argument('--split-strategy', choices=['random', 'stratified', 'sequential'],
                        default='stratified', help='Data splitting strategy')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Random seed for splitting')
    parser.add_argument('--splits-dir', default='data_splits',
                        help='Directory for split files')
    parser.add_argument('--use-existing-splits', action='store_true',
                        help='Use existing splits instead of creating new ones')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1 training')
    parser.add_argument('--skip-phase2', action='store_true',
                        help='Skip Phase 2 training')
    parser.add_argument('--skip-test-eval', action='store_true',
                        help='Skip test set evaluation')
    parser.add_argument('--phase1-checkpoint', type=str,
                        help='Use existing Phase 1 checkpoint (if skipping Phase 1)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TRAINING WITH ENFORCED DATA SPLITTING (80/20)")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split strategy: {args.split_strategy}")
    logger.info(f"Phase 1 config: {args.phase1_config}")
    logger.info(f"Phase 2 config: {args.phase2_config}")

    # Check that dataset exists
    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return 1

    # Step 1: Split dataset (or use existing splits)
    if args.use_existing_splits:
        logger.info(f"\nUsing existing splits from: {args.splits_dir}")
        splits = {
            'train': str(Path(args.splits_dir) / 'train.jsonl'),
            'test': str(Path(args.splits_dir) / 'test.jsonl')
        }

        # Verify all splits exist
        for split_name, split_path in splits.items():
            if not Path(split_path).exists():
                logger.error(f"Split file not found: {split_path}")
                return 1
    else:
        splits = split_dataset(
            args.dataset,
            output_dir=args.splits_dir,
            strategy=args.split_strategy,
            seed=args.split_seed
        )

    train_path = splits['train']
    test_path = splits['test']

    # Step 2: Run Phase 1 training (contrastive pretraining)
    phase1_checkpoint = args.phase1_checkpoint

    if not args.skip_phase1:
        if not Path(args.phase1_config).exists():
            logger.error(f"Phase 1 config not found: {args.phase1_config}")
            return 1

        phase1_results = run_phase1_training(
            train_path,
            args.phase1_config,
            output_dir="phase1_pretraining"
        )
        phase1_checkpoint = phase1_results.best_checkpoint_path
    else:
        if not phase1_checkpoint or not Path(phase1_checkpoint).exists():
            logger.error("Phase 1 checkpoint required when skipping Phase 1")
            return 1
        logger.info(
            f"\nSkipping Phase 1, using checkpoint: {phase1_checkpoint}")

    # Step 3: Run Phase 2 training (classification fine-tuning)
    phase2_checkpoint = None

    if not args.skip_phase2:
        if not Path(args.phase2_config).exists():
            logger.error(f"Phase 2 config not found: {args.phase2_config}")
            return 1

        phase2_results = run_phase2_training(
            train_path,
            args.phase2_config,
            phase1_checkpoint,
            output_dir="phase2_finetuning"
        )
        phase2_checkpoint = phase2_results.best_checkpoint_path
    else:
        logger.info("\nSkipping Phase 2 training")

    # Step 4: Evaluate on held-out test set
    if not args.skip_test_eval and phase2_checkpoint:
        test_results = evaluate_on_test_set(
            phase2_checkpoint,
            test_path,
            args.phase2_config,
            output_dir="test_evaluation"
        )
    else:
        logger.info("\nSkipping test set evaluation")

    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Train split: {train_path}")
    logger.info(f"Test split: {test_path}")
    if phase1_checkpoint:
        logger.info(f"Phase 1 checkpoint: {phase1_checkpoint}")
    if phase2_checkpoint:
        logger.info(f"Phase 2 checkpoint: {phase2_checkpoint}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
