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


def _register_domain(config) -> None:
    """Import a non-career domain's adapter so the seam registry knows it.

    Must run **before** ``ContrastiveLearningTrainer`` is constructed: the trainer
    builds a ``DataLoader``, which resolves ``config.domain_adapter`` through
    ``get_domain_adapter`` immediately and raises ``KeyError`` for an unregistered
    name. Registration happens as an import-time side effect of the domain's
    ``record_adapter`` module, so importing it is sufficient.

    The career adapter registers when ``contrastive_learning.domain_adapters`` is
    imported, so the default path needs nothing here.
    """
    adapter = getattr(config, "domain_adapter", "career")
    if adapter == "career":
        return

    #: Domain name -> the module whose import registers its adapter.
    registrars = {
        "trials": "trials_domain.record_adapter",
        "cve": "cve_domain.record_adapter",
        "go_ppi": "go_ppi_domain.record_adapter",
    }
    module = registrars.get(adapter)
    if module is None:
        logger.error(
            "config.domain_adapter=%r is not a domain this runner knows how to "
            "register. Known: %s (plus 'career', registered by default).",
            adapter, ", ".join(sorted(registrars)))
        sys.exit(1)

    import importlib

    try:
        importlib.import_module(module)
    except ImportError as exc:
        logger.error(
            "domain_adapter=%r requires %s, which is not importable: %s",
            adapter, module, exc)
        sys.exit(1)
    logger.info("Registered the %r domain adapter via %s", adapter, module)


def _attach_domain(trainer, config) -> None:
    """Perform any domain-specific seam injection before training starts.

    The career domain needs nothing here: its ontology matcher is built inside
    ``BatchProcessor`` from the ``esco_*`` config paths. A non-career domain has
    to inject its own matcher and negative selector through the additive seams,
    and that injection has to happen *before* Phase 2 — the warmup snapshot is
    frozen from whatever negatives are in play, so attaching afterwards would
    train the warmup encoder on one negative distribution and then swap another
    in underneath the frozen embeddings.

    Raises:
        SystemExit: If the domain's wiring cannot be applied. This is deliberately
            fatal rather than a warning: an unwired trials run trains happily on
            random negatives with no ontology signal and produces
            plausible-looking numbers, which is the worst possible failure mode.
    """
    adapter = getattr(config, "domain_adapter", "career")
    if adapter not in ("trials", "go_ppi"):
        return

    if adapter == "trials":
        try:
            from trials_domain.run_config import attach_trials_domain
        except ImportError as exc:
            logger.error(
                "domain_adapter='trials' but trials_domain is not importable: %s", exc)
            sys.exit(1)
        attach_fn = attach_trials_domain
        ontology_label = "MeSH descriptors"
        ontology_count_key = "mesh_descriptors"
        split_dir_field = "trials_split_dir"
    else:
        try:
            from go_ppi_domain.run_config import attach_go_ppi_domain
        except ImportError as exc:
            logger.error(
                "domain_adapter='go_ppi' but go_ppi_domain is not importable: %s", exc)
            sys.exit(1)
        attach_fn = attach_go_ppi_domain
        ontology_label = "GO terms"
        ontology_count_key = "go_terms"
        split_dir_field = "go_ppi_split_dir"

    try:
        summary = attach_fn(trainer, config, split="train")
    except Exception as exc:
        logger.error("Failed to attach the %s domain: %s", adapter, exc)
        sys.exit(1)

    pools = summary.get("pools", {})
    logger.info(
        "%s domain attached: %d %s, %d anchor pools "
        "(%d grade-1 / %d grade-0 candidates), hard ratio %.2f -> %.2f",
        adapter, summary.get(ontology_count_key, 0), ontology_label,
        pools.get("topics", 0),
        pools.get("hard_total", 0),
        pools.get("easy_total", 0),
        summary.get("curriculum", {}).get("start_hard_ratio", 0.0),
        summary.get("curriculum", {}).get("end_hard_ratio", 0.0),
    )
    if not pools.get("topics"):
        logger.error(
            "The %s negative selector indexed 0 anchor pools. Negatives would "
            "silently fall back to other anchors' positives and the graded "
            "grade-1 judgments would never reach the loss. Check %s=%s contains "
            "negative_pools.jsonl for the train split.",
            adapter, split_dir_field, getattr(config, split_dir_field, "<unset>"))
        sys.exit(1)


def _verify_ontology_wiring(trainer, config, require: bool) -> None:
    """Assert an ontology matcher is actually live on the batch processor.

    ``_check_orca_data_quality`` only inspects the *data*. That is not sufficient:
    a run can carry perfectly good ``skill_uris`` and still have no matcher wired,
    in which case ``_compute_negative_ontology_features`` returns ``None`` and
    ORCA silently degrades to the blended ``career_distances`` proxy — losing the
    d_esco/d_isco decomposition the ontology ablations exist to measure. This
    checks the wiring itself.
    """
    batch_processor = getattr(trainer, "batch_processor", None)
    if batch_processor is None:
        return

    resolver = getattr(batch_processor, "_effective_skill_matcher", None)
    matcher = resolver() if callable(resolver) else getattr(
        batch_processor, "skill_matcher", None)

    if matcher is not None:
        logger.info(
            "Ontology wiring OK: %s active for negative selection and ORCA "
            "feature capture.", type(matcher).__name__)
        return

    msg = (
        "No ontology matcher is active on the BatchProcessor. ORCA's per-negative "
        "features (d_esco/d_isco/d_ot/s_esco/s_isco) cannot be computed and will "
        "degrade to the blended career_distances proxy, making the ontology "
        "ablations meaningless. For the career domain set esco_graph_path / "
        "esco_kg_path; for a non-career domain ensure its matcher is injected via "
        "BatchProcessor.set_ontology_matcher."
    )
    if require:
        logger.error(msg)
        sys.exit(1)
    logger.warning(msg)


def _check_orca_data_quality(train_path: str, require: bool) -> None:
    """Warn (or error) if the training data lacks the ontology fields ORCA needs.

    ORCA's ontology features / weak targets rely on ``skill_uris`` on the resume
    and jobs. On the raw enriched dataset these are absent (nested under
    ``esco_enrichment_v3``), which silently degrades ORCA to career-distance
    proxies and random negative selection. Surface that early.

    NOTE: this inspects the data only. It cannot tell whether a matcher is wired,
    so :func:`_verify_ontology_wiring` is the necessary companion check.
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
    logger.info("  domain:         %s", getattr(config, "domain_adapter", "career"))
    logger.info("  train:          %s", train_path)
    logger.info("  validation:     %s", val_path or "(none)")
    logger.info("  output_dir:     %s", args.output_dir)
    logger.info("  seed:           %s", config.training_seed)
    logger.info("  phases (ep):    warmup=%s reliability=%s joint=%s",
                config.orca_warmup_epochs, config.orca_reliability_epochs,
                config.orca_joint_epochs)
    logger.info("=" * 60)

    # Register a non-career domain adapter BEFORE the trainer is built: the
    # trainer's DataLoader resolves config.domain_adapter during construction.
    _register_domain(config)

    # Build the standard OSCAR trainer, then drive it with the ORCA orchestrator.
    trainer = ContrastiveLearningTrainer(config=config, output_dir=args.output_dir)

    # Domain-specific seam injection MUST precede the orchestrator run so the
    # Phase-2 warmup snapshot is captured from this domain's negatives.
    _attach_domain(trainer, config)
    _verify_ontology_wiring(trainer, config, require=args.require_ontology)

    try:
        orchestrator = OrcaPhaseOrchestrator(config, trainer)
        warmup_store = orchestrator.run(train_path)
    except (OrcaConfigError, OrcaPhaseError) as exc:
        logger.error("ORCA configuration/phase error: %s", exc)
        return 1

    # Save an evaluable checkpoint. The orchestrator drives the 4-phase schedule
    # through trainer.train_epoch (which does not checkpoint), so the runner owns
    # persisting the final projection head — exactly what the shared Phase-1
    # embedding evaluation loads (checkpoint['model_state_dict']).
    ckpt_path = _save_orca_checkpoint(trainer, args.output_dir, config, orchestrator)

    logger.info("=" * 60)
    logger.info("ORCA COMPLETE")
    logger.info("  completed phase: %s", orchestrator.current_phase.name)
    if warmup_store is not None:
        logger.info("  warmup snapshot: %d embeddings", len(warmup_store))
    logger.info("  checkpoint:      %s", ckpt_path)
    logger.info("  outputs in:      %s", args.output_dir)
    logger.info("=" * 60)
    return 0


def _save_orca_checkpoint(trainer, output_dir: str, config, orchestrator) -> str:
    """Persist the trained ORCA projection head as ``best_checkpoint.pt``.

    Writes the same ``model_state_dict`` key the Phase-1 embedding evaluation
    loads, so an ORCA run is evaluated by the identical script/metrics as the
    OSCAR / InfoNCE baselines. Also stores the ReliabilityMLP weights and the
    completed phase for provenance.
    """
    import torch
    from dataclasses import asdict

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "best_checkpoint.pt"

    reliability_model = getattr(orchestrator, "reliability_model", None)
    payload = {
        "model_state_dict": trainer.model.state_dict(),
        "reliability_model_state_dict": (
            reliability_model.state_dict() if reliability_model is not None else None
        ),
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else {},
        "orca_variant": getattr(config, "orca_variant", "denominator"),
        "completed_phase": orchestrator.current_phase.name,
        "training_seed": getattr(config, "training_seed", None),
    }
    torch.save(payload, ckpt_path)
    return str(ckpt_path)


if __name__ == "__main__":
    sys.exit(main())
