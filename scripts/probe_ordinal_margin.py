#!/usr/bin/env python3
"""Probe the ordinal-loss internals of a trained EO-* checkpoint.

Purpose
-------
EO-ON-B (+OSCAR sample weighting) and EO-ON-C (no curriculum) score almost
identically to each other and to their EO-OntNeg base on Table-4 metrics. There
are two competing explanations, mirroring probe_reliability.py's ER-DEN vs
ER-EXT investigation:

  1. The mechanisms these variants toggle are near-inert on this data/model:
     - EO-ON-B's `_compute_ontology_weight` collapses toward ~1.0 (no real
       reweighting happening), so the sample-level loss multiplier does nothing.
     - EO-ON-C's curriculum-switch change only shifts WHEN L2 turns on by a few
       epochs out of 15; the margin term stays small (ReLU-bounded, ~0.3) next
       to the InfoNCE term (L1, unbounded), so the shift barely matters either
       way.
  2. The mechanisms DO vary meaningfully but still don't move the saturating
     Table-4 metrics (AUC/Spearman/Cohen's d) at this data scale/seed count.

This script settles (1) vs (2) by loading a trained checkpoint, running a few
REAL training batches through the exact production path (batch processor ->
embeddings -> ContrastiveLossEngine._compute_ordinal_loss /
_compute_ontology_weight), and reporting:

  * the distribution of `ont_weight` actually produced by
    `_compute_ontology_weight` (mean/std/frac near 1.0) — settles whether
    EO-ON-B's sample weighting is inert;
  * the distribution of the ordinal_margin scalar (L2+L3 contribution) relative
    to L1 (InfoNCE) — settles whether the ordinal margin term is a meaningful
    fraction of the total loss or a rounding error next to L1;
  * whether L2 (good>potential margin) is actually being exercised (i.e.
    `is_full_phase` true and `mask_good_pot` non-empty) at the epoch of the
    loaded checkpoint, for both curriculum settings.

Run it ON THE GPU BOX where the checkpoints live, e.g.:

    d=results/research_runs/EO-ON-B__cnamuangtoun__s13
    python scripts/probe_ordinal_margin.py \
        --checkpoint $d/phase1_pretraining/best_checkpoint.pt \
        --config     $d/training_config.json \
        --data       preprocess/data_splits_v7/train.jsonl \
        --max-batches 20

Compare EO-ON-B vs EO-ON-C (same seed) — if ont_weight clusters near 1.0 for
EO-ON-B and the ordinal_margin/L1 ratio is tiny for both, that confirms
explanation (1): the toggled mechanisms are inert on this data, not that
ontology signal doesn't matter in principle.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure the repo root is importable when this script is run from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("probe_ordinal_margin")
logger.setLevel(logging.INFO)


def _load_config(config_path: str):
    from contrastive_learning.data_structures import TrainingConfig
    return TrainingConfig.from_json(config_path)


def _build(config, output_dir: str):
    from contrastive_learning.trainer import ContrastiveLearningTrainer
    return ContrastiveLearningTrainer(config=config, output_dir=output_dir)


def _load_weights(trainer, ckpt: dict) -> int:
    """Load the trained projection head; return the checkpoint's saved epoch."""
    msd = ckpt.get("model_state_dict")
    if msd is not None:
        missing, unexpected = trainer.model.load_state_dict(msd, strict=False)
        if missing or unexpected:
            logger.warning("load_state_dict: missing=%s unexpected=%s",
                           list(missing), list(unexpected))
    epoch = ckpt.get("epoch", 0)
    return epoch


def _maybe_load_global_pool(trainer, config, data_path: str) -> None:
    if not getattr(config, "global_negative_sampling", False):
        return
    try:
        trainer.global_job_pool = trainer.data_loader.load_global_job_pool(
            data_path, max_jobs=getattr(config, "global_negative_pool_size", 200))
        logger.info("Loaded global job pool: %d jobs", len(trainer.global_job_pool))
    except Exception as exc:
        logger.warning("Could not load global job pool (%s); using in-batch negatives.", exc)


def _collect(trainer, data_path: str, max_batches: int, epoch: int):
    """Run a few batches and capture ont_weight + L1/margin decomposition.

    Returns a dict of numpy arrays / scalars for reporting.
    """
    engine = trainer.loss_engine
    engine.current_epoch = epoch  # match the checkpoint's curriculum phase

    ont_weights = []
    l1_values = []
    margin_values = []
    l2_active_count = 0
    l2_possible_count = 0

    original_ont_weight = engine._compute_ontology_weight

    def _recording_ont_weight(view_metadata):
        w = original_ont_weight(view_metadata)
        ont_weights.append(w)
        return w

    engine._compute_ontology_weight = _recording_ont_weight  # type: ignore[assignment]

    original_ordinal = engine._compute_ordinal_loss

    def _recording_ordinal(triplets, embeddings):
        # Re-derive L1 and the margin term the same way _compute_ordinal_loss
        # does, so we can report them separately instead of just the sum.
        infonce_losses = []
        for triplet in triplets:
            try:
                tl = engine._compute_triplet_loss(triplet, embeddings)
                if tl is not None:
                    infonce_losses.append(tl)
            except Exception:
                continue
        l1_mean = torch.stack(infonce_losses).mean() if infonce_losses else None

        total = original_ordinal(triplets, embeddings)
        if l1_mean is not None:
            l1_values.append(float(l1_mean.detach().cpu()))
            margin_values.append(float((total - l1_mean).detach().cpu()))

        # Check whether L2 (good>potential) had any eligible pairs this batch,
        # independent of curriculum phase, so we can report "possible vs active".
        epoch_ratio = engine.current_epoch / max(1, engine.total_epochs)
        is_full_phase = epoch_ratio >= engine.ordinal_curriculum_switch
        nonlocal l2_active_count, l2_possible_count
        for triplet in triplets:
            vm = triplet.view_metadata
            pos_label = vm.get('positive_original_label', 'good_fit')
            neg_labels = vm.get('negative_original_labels', [])
            has_good = pos_label == 'good_fit' or 'good_fit' in neg_labels
            has_pot = pos_label == 'potential_fit' or 'potential_fit' in neg_labels
            if has_good and has_pot:
                l2_possible_count += 1
                if is_full_phase:
                    l2_active_count += 1
        return total

    engine._compute_ordinal_loss = _recording_ordinal  # type: ignore[assignment]

    trainer.model.eval()

    n = 0
    with torch.no_grad():
        for batch in trainer.data_loader.load_batches(data_path):
            triplets = trainer.batch_processor.process_batch(
                batch, trainer.global_job_pool, trainer.global_resume_pool)
            if not triplets:
                continue
            embeddings = trainer._generate_embeddings(triplets)
            if not embeddings:
                continue
            trainer.loss_engine.compute_loss(triplets, embeddings)
            n += 1
            if n >= max_batches:
                break

    engine._compute_ontology_weight = original_ont_weight  # restore
    engine._compute_ordinal_loss = original_ordinal

    return {
        "ont_weights": np.array(ont_weights) if ont_weights else np.array([1.0]),
        "l1_values": np.array(l1_values) if l1_values else np.array([0.0]),
        "margin_values": np.array(margin_values) if margin_values else np.array([0.0]),
        "l2_active_count": l2_active_count,
        "l2_possible_count": l2_possible_count,
        "n_batches": n,
    }


def _report(variant_label, epoch, stats):
    ow = stats["ont_weights"]
    l1 = stats["l1_values"]
    margin = stats["margin_values"]

    print("=" * 70)
    print(f"Ordinal-loss probe — {variant_label}  (checkpoint epoch={epoch}, "
          f"batches={stats['n_batches']})")
    print("=" * 70)

    print("ont_weight distribution (_compute_ontology_weight output, per triplet):")
    print(f"  mean   {ow.mean():.4f}")
    print(f"  std    {ow.std():.4f}")
    print(f"  min    {ow.min():.4f}")
    print(f"  max    {ow.max():.4f}")
    frac_near_1 = float(np.mean(np.abs(ow - 1.0) < 0.05))
    print(f"  frac within 0.05 of 1.0   {frac_near_1:.3f}")
    print("-" * 70)

    print("L1 (InfoNCE) vs ordinal margin term (L2+L3), per batch:")
    print(f"  L1 mean       {l1.mean():.4f}   std {l1.std():.4f}")
    print(f"  margin mean   {margin.mean():.4f}   std {margin.std():.4f}")
    ratio = margin.mean() / max(abs(l1.mean()), 1e-8)
    print(f"  margin / L1 ratio (mean)   {ratio:.4f}")
    print("-" * 70)

    print("L2 (good>potential) eligibility vs activation:")
    print(f"  batches/triplets with BOTH good_fit and potential_fit present: "
          f"{stats['l2_possible_count']}")
    print(f"  of those, L2 actually active this epoch (curriculum phase on): "
          f"{stats['l2_active_count']}")
    print("-" * 70)

    verdicts = []
    if ow.std() == 0.0 and abs(ow.mean() - 1.0) < 1e-9:
        verdicts.append(
            "ont_weight is EXACTLY 1.0 with zero variance -> ontology_weight is "
            "configured to 0.0 for this run (the early-return branch in "
            "_compute_ontology_weight), i.e. OSCAR sample weighting is OFF BY "
            "CONFIG, not collapsed by training. This is expected/uninformative "
            "unless this run is supposed to have ontology_weight > 0.")
    elif frac_near_1 > 0.7:
        verdicts.append(
            "ont_weight clusters near 1.0 (with some spread) for most triplets "
            "-> OSCAR sample weighting is enabled but has converged to a "
            "near-inert multiplier on this data; expect it to barely move "
            "metrics vs its ontology-negative base.")
    else:
        verdicts.append(
            "ont_weight varies meaningfully across triplets -> sample weighting "
            "is enabled and NOT inert; the tie is not explained by a collapsed "
            "weight.")

    if ratio < 0.05:
        verdicts.append(
            "ordinal margin term is a tiny fraction of L1 (InfoNCE) -> the "
            "curriculum-switch timing (EO-ON-C) mostly perturbs a term that "
            "barely contributes to the total loss either way.")
    else:
        verdicts.append(
            "ordinal margin term is a non-trivial fraction of L1 -> curriculum "
            "timing changes ARE moving a meaningful part of the loss; the tie "
            "on Table-4 metrics is not explained by margin negligibility.")

    print("VERDICT:")
    for v in verdicts:
        print("  -", v)
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best_checkpoint.pt of the run.")
    ap.add_argument("--config", required=True, help="training_config.json of the run.")
    ap.add_argument("--data", required=True, help="A prepared split (train.jsonl) with skill_uris.")
    ap.add_argument("--max-batches", type=int, default=20, help="How many batches to probe.")
    ap.add_argument("--output-dir", default="/tmp/ordinal_margin_probe",
                    help="Scratch dir for the trainer (no artifacts of interest written).")
    ap.add_argument("--label", default=None, help="Label for the report header (default: config path).")
    args = ap.parse_args()

    for p in (args.checkpoint, args.config, args.data):
        if not Path(p).exists():
            logger.error("Not found: %s", p)
            return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)

    config = _load_config(args.config)
    logger.info("Probing loss_type=%s curriculum_switch=%s on %s (device=%s)",
               getattr(config, "loss_type", None),
               getattr(config, "ordinal_curriculum_switch", None),
               args.data, device)

    trainer = _build(config, args.output_dir)
    epoch = _load_weights(trainer, ckpt)
    _maybe_load_global_pool(trainer, config, args.data)

    stats = _collect(trainer, args.data, args.max_batches, epoch)
    label = args.label or args.config
    _report(label, epoch, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
