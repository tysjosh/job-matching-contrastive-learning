#!/usr/bin/env python3
"""
ORCA training runner.

Runs the four-phase ORCA schedule (Ontology-Regularized Contrastive Alignment
with Uncertainty) on top of the existing OSCAR ``ContrastiveLearningTrainer``:

  * Phase 1 - reuse OSCAR ontology preprocessing (no gradient updates)
  * Phase 2 - encoder warmup (normal InfoNCE) + freeze a warmup-embedding snapshot
  * Phase 3 - train ONLY the ReliabilityMLP on the frozen warmup embeddings
  * Phase 4 - joint training (projection head + ReliabilityMLP, encoder frozen)
              with the reliability-calibrated OrcaLossEngine

ORCA is additive and config-gated: it activates only when ``orca_enabled`` is
true. This runner mirrors ``run_training_with_split.py`` for data handling but
drives training through ``orca.OrcaPhaseOrchestrator`` instead of
``trainer.train(...)``.

IMPORTANT — dataset format
--------------------------
ORCA's ontology signals (ReliabilityMLP features, weak targets, alignment)
require the *prepared* training format that carries ``skill_uris`` on each
resume/job and ontology scores in ``metadata`` — i.e. the ``data_splits_v6/`` /
``data_splits_v7/`` splits produced by ``scripts/prepare_training_data_v3.py``.
Do NOT point this at the raw ``combined_all_8000_enriched.jsonl`` (its ESCO data
is nested under ``esco_enrichment_v3`` and has no ``skill_uris`` on
resume/job) — ORCA's ontology signal would silently degrade. The runner checks
this and warns (or errors with ``--require-ontology``).

Examples
--------
Use the prepared v6 splits and run ORCA-Denominator (MVP):

    python run_orca_training.py \
        --use-existing-splits --splits-dir preprocess/data_splits_v6 \
        --config config/orca_denominator_config.json

Point directly at prepared train/validation files (no splitting):

    python run_orca_training.py \
        --train-file preprocess/data_splits_v6/train.jsonl \
        --validation-file preprocess/data_splits_v6/validation.jsonl \
        --config config/orca_denominator_config.json

Run the ORCA-Full variant (denominator + alignment + adaptive sampling):

    python run_orca_training.py \
        --use-existing-splits --splits-dir preprocess/data_splits_v6 \
        --config config/orca_denominator_config.json --variant full
"""

import argparse
import logging
import sys
from pathlib import Path

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.trainer import ContrastiveLearningTrainer
from orca.config import RECOGNIZED_VARIANTS, OrcaConfigError
from orca.orchestrator import OrcaPhaseError, OrcaPhaseOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _split_dataset(dataset_path: str, output_dir: str, strategy: str, seed: int) -> dict:
    """Split a full dataset into train/validation/test (80/10/10).

    Imported lazily so the runner still works for the --train-file /
    --use-existing-splits paths even if the splitter's optional deps are absent.
    """
    from contrastive_learning.data_splitter import DataSplitter, SplitConfig

    logger.info(
        "Splitting %s -> %s (strategy=%s, seed=%s)",
        dataset_path, output_dir, strategy, seed,
    )
    splitter = DataSplitter(SplitConfig(
        strategy=strategy,
        ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        seed=seed,
        validate_splits=True,
    ))
    result = splitter.split_dataset(dataset_path, output_dir)

    for name, info in result.statistics.get("splits", {}).items():
        logger.info("  %s: %s samples (%.1f%%)",
                    name, info["count"], info["percentage"])
    return result.splits


def _resolve_data(args) -> tuple:
    """Resolve (train_path, validation_path) from the CLI arguments."""
    # 1. Explicit training file wins (no splitting).
    if args.train_file:
        if not Path(args.train_file).exists():
            logger.error("Training file not found: %s", args.train_file)
            sys.exit(1)
        return args.train_file, args.validation_file

    # 2. Existing split directory.
    if args.use_existing_splits:
        train_path = str(Path(args.splits_dir) / "train.jsonl")
        val_path = str(Path(args.splits_dir) / "validation.jsonl")
        if not Path(train_path).exists():
            logger.error("Split not found: %s", train_path)
            sys.exit(1)
        val_path = val_path if Path(val_path).exists() else None
        logger.info("Using existing splits from %s", args.splits_dir)
        return train_path, val_path

    # 3. Split a full dataset.
    if not args.dataset:
        logger.error(
            "Provide one of --dataset, --train-file, or --use-existing-splits.")
        sys.exit(1)
    if not Path(args.dataset).exists():
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)
    splits = _split_dataset(
        args.dataset, args.splits_dir, args.split_strategy, args.split_seed)
    val_path = splits.get("validation")
    if val_path and not Path(val_path).exists():
        val_path = None
    return splits["train"], val_path


def _check_orca_data_quality(train_path: str, require: bool) -> None:
    """Warn (or error) if the training data lacks the ontology fields ORCA needs.

    ORCA's ontology features / weak targets rely on ``skill_uris`` on the resume
    and jobs. On the raw enriched dataset these are absent (nested under
    ``esco_enrichment_v3``), which silently degrades ORCA to career-distance
    proxies and random negative selection. Surface that early.
    """
    import json

    try:
        with open(train_path, "r") as f:
            first = f.readline()
        rec = json.loads(first) if first.strip() else {}
    except Exception as exc:
        logger.warning("Could not inspect training data quality (%s): %s",
                       train_path, exc)
        return

    resume = rec.get("resume", {}) if isinstance(rec, dict) else {}
    job = rec.get("job", {}) if isinstance(rec, dict) else {}
    r_uris = resume.get("skill_uris") if isinstance(resume, dict) else None
    j_uris = job.get("skill_uris") if isinstance(job, dict) else None

    if r_uris and j_uris:
        logger.info(
            "Ontology check OK: resume.skill_uris=%d, job.skill_uris=%d "
            "(ORCA ontology signal active).", len(r_uris), len(j_uris))
        return

    msg = (
        f"Training data '{train_path}' has no skill_uris on "
        f"{'resume' if not r_uris else 'job'} records. ORCA's ontology features "
        "and weak targets will degrade to career-distance proxies. Use the "
        "prepared v6/v7 splits (scripts/prepare_training_data_v3.py), e.g. "
        "preprocess/data_splits_v6/."
    )
    if require:
        logger.error(msg)
        sys.exit(1)
    logger.warning(msg)


def _load_config(args) -> TrainingConfig:
    """Load the TrainingConfig and apply CLI overrides / ORCA guards."""
    if not Path(args.config).exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    config = TrainingConfig.from_json(args.config)

    # ORCA must be enabled for the four-phase schedule to run.
    config.orca_enabled = True

    if args.variant:
        config.orca_variant = args.variant
    if args.seed is not None:
        config.training_seed = args.seed
    if args.warmup_epochs is not None:
        config.orca_warmup_epochs = args.warmup_epochs
    if args.reliability_epochs is not None:
        config.orca_reliability_epochs = args.reliability_epochs
    if args.joint_epochs is not None:
        config.orca_joint_epochs = args.joint_epochs

    if config.orca_variant not in RECOGNIZED_VARIANTS:
        logger.error(
            "Unrecognized orca_variant %r; must be one of %s",
            config.orca_variant, ", ".join(RECOGNIZED_VARIANTS))
        sys.exit(1)

    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the four-phase ORCA training schedule.")

    # Data source (choose one path).
    parser.add_argument("--dataset", help="Full dataset (JSONL) to split 80/10/10.")
    parser.add_argument("--train-file", help="Prepared training file (skips splitting).")
    parser.add_argument("--validation-file", help="Optional validation file (with --train-file).")
    parser.add_argument("--use-existing-splits", action="store_true",
                        help="Use train.jsonl/validation.jsonl from --splits-dir.")
    parser.add_argument("--splits-dir", default="data_splits",
                        help="Directory for splits (read or written).")
    parser.add_argument("--split-strategy",
                        choices=["random", "stratified", "sequential"],
                        default="sequential")
    parser.add_argument("--split-seed", type=int, default=42)

    # Config + output.
    parser.add_argument("--config", default="config/orca_denominator_config.json",
                        help="TrainingConfig JSON (orca_* fields honored).")
    parser.add_argument("--output-dir", default="orca_output",
                        help="Directory for checkpoints, logs, and results.")

    # Convenience overrides.
    parser.add_argument("--variant", choices=list(RECOGNIZED_VARIANTS),
                        help="Override orca_variant.")
    parser.add_argument("--seed", type=int, help="Override training_seed.")
    parser.add_argument("--warmup-epochs", type=int, help="Override orca_warmup_epochs.")
    parser.add_argument("--reliability-epochs", type=int,
                        help="Override orca_reliability_epochs.")
    parser.add_argument("--joint-epochs", type=int, help="Override orca_joint_epochs.")
    parser.add_argument("--require-ontology", action="store_true",
                        help="Error (instead of warn) if the data lacks skill_uris.")

    args = parser.parse_args()

    train_path, val_path = _resolve_data(args)
    _check_orca_data_quality(train_path, require=args.require_ontology)
    config = _load_config(args)
    if val_path:
        config.validation_path = val_path

    logger.info("=" * 60)
    logger.info("ORCA training")
    logger.info("  variant:        %s", config.orca_variant)
    logger.info("  train:          %s", train_path)
    logger.info("  validation:     %s", val_path or "(none)")
    logger.info("  output_dir:     %s", args.output_dir)
    logger.info("  seed:           %s", config.training_seed)
    logger.info("  phases (ep):    warmup=%s reliability=%s joint=%s",
                config.orca_warmup_epochs, config.orca_reliability_epochs,
                config.orca_joint_epochs)
    logger.info("=" * 60)

    # Build the standard OSCAR trainer, then drive it with the ORCA orchestrator.
    trainer = ContrastiveLearningTrainer(config=config, output_dir=args.output_dir)

    try:
        orchestrator = OrcaPhaseOrchestrator(config, trainer)
        warmup_store = orchestrator.run(train_path)
    except (OrcaConfigError, OrcaPhaseError) as exc:
        logger.error("ORCA configuration/phase error: %s", exc)
        return 1

    logger.info("=" * 60)
    logger.info("ORCA COMPLETE")
    logger.info("  completed phase: %s", orchestrator.current_phase.name)
    if warmup_store is not None:
        logger.info("  warmup snapshot: %d embeddings", len(warmup_store))
    logger.info("  outputs in:      %s", args.output_dir)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
