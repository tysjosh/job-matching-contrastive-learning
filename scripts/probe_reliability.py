#!/usr/bin/env python3
"""Probe the learned ReliabilityMLP outputs of a trained ORCA checkpoint.

Purpose
-------
The ER-DEN (denominator) vs ER-EXT (external_weight) result is a tie. There are
two competing explanations:

  1. The learned per-negative reliabilities are near-constant (all ~equal, or all
     ~1.0). If so, denominator calibration collapses to a uniform scaling of the
     InfoNCE negative mass -> ~= Standard_InfoNCE, and ER-EXT's detached scalar
     multiplier under Adam is also ~= Standard_InfoNCE. Both land in the same
     place, so the tie is EXPECTED and says nothing about "placement".

  2. The reliabilities genuinely vary across negatives, in which case the tie is
     more interesting (placement really doesn't move the metric).

This script settles (1) vs (2) by loading a trained checkpoint, running a few
REAL training batches through the exact production path (batch processor ->
embeddings -> OrcaTrainerLossAdapter -> OrcaLossEngine._predict_reliability), and
reporting the distribution of the reliabilities the model actually produces.

Two statistics matter:
  * GLOBAL spread  : mean / std / min / max / percentiles over every
                     (anchor, negative) reliability. Tells you if values are
                     pinned near a single number (e.g. ~1.0 or ~0.5).
  * WITHIN-ANCHOR  : the std of reliabilities ACROSS the K negatives of the same
                     anchor, averaged over anchors. This is the quantity that
                     actually drives denominator calibration: if it is ~0, the
                     denominator is just uniformly scaled and ER-DEN == InfoNCE.

Run it ON THE GPU BOX where the checkpoints live, e.g.:

    d=results/research_runs/ER-DEN__cnamuangtoun__s13
    python scripts/probe_reliability.py \
        --checkpoint $d/phase1_pretraining/best_checkpoint.pt \
        --config     $d/training_config.json \
        --data       preprocess/data_splits_v7/train.jsonl \
        --max-batches 20

Compare ER-DEN vs ER-EXT (same seed) — they should produce similar reliability
distributions if the models are equivalent; the key output is whether the
WITHIN-ANCHOR std is meaningfully > 0.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the repo root is importable when this script is run from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Quiet the noisy trainer/bath-processor logs; we only care about our summary.
logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("probe_reliability")
logger.setLevel(logging.INFO)


def _load_config(config_path: str, ckpt: dict):
    """Load the run's TrainingConfig and force it onto the ORCA path.

    The saved training_config.json is the same artifact the ordinal evaluation
    consumes. We force orca_enabled=True and copy the checkpoint's recorded
    variant so the factory reconstructs the ReliabilityMLP with the exact
    feature layout the run trained with.
    """
    from contrastive_learning.data_structures import TrainingConfig

    config = TrainingConfig.from_json(config_path)
    config.orca_enabled = True
    variant = ckpt.get("orca_variant") or getattr(config, "orca_variant", "denominator")
    config.orca_variant = variant
    return config, variant


def _build(config, output_dir: str):
    """Construct the OSCAR trainer + ORCA orchestrator (no training run)."""
    from contrastive_learning.trainer import ContrastiveLearningTrainer
    from orca.orchestrator import OrcaPhaseOrchestrator

    trainer = ContrastiveLearningTrainer(config=config, output_dir=output_dir)
    orchestrator = OrcaPhaseOrchestrator(config, trainer)
    return trainer, orchestrator


def _load_weights(trainer, orchestrator, ckpt: dict) -> None:
    """Load the trained projection head + ReliabilityMLP weights."""
    msd = ckpt.get("model_state_dict")
    if msd is not None and getattr(trainer, "model", None) is not None:
        missing, unexpected = trainer.model.load_state_dict(msd, strict=False)
        if missing or unexpected:
            logger.warning("projection load_state_dict: missing=%s unexpected=%s",
                           list(missing), list(unexpected))

    rsd = ckpt.get("reliability_model_state_dict")
    rmodel = getattr(orchestrator, "reliability_model", None)
    if rsd is None:
        raise SystemExit(
            "Checkpoint has no reliability_model_state_dict; this is not an ORCA "
            "checkpoint (or reliability model was never trained).")
    if rmodel is None:
        raise SystemExit("Orchestrator built no reliability_model; check config.orca_variant.")
    rmodel.load_state_dict(rsd, strict=True)


def _maybe_load_global_pool(trainer, config, data_path: str) -> None:
    """Reproduce OSCAR's global negative pool so negative selection matches training."""
    if not getattr(config, "global_negative_sampling", False):
        return
    try:
        trainer.global_job_pool = trainer.data_loader.load_global_job_pool(
            data_path, max_jobs=getattr(config, "global_negative_pool_size", 200))
        logger.info("Loaded global job pool: %d jobs", len(trainer.global_job_pool))
    except Exception as exc:
        logger.warning("Could not load global job pool (%s); using in-batch negatives.", exc)


def _collect(trainer, orchestrator, data_path: str, max_batches: int):
    """Run a few batches and capture every reliability the model produces.

    Returns (all_values, within_anchor_stds) as numpy arrays.
    """
    engine = orchestrator.loss_engine

    captured = []  # list of 1-D numpy arrays, one per anchor (the K reliabilities)

    original_predict = engine._predict_reliability

    def _recording_predict(z_r, z_negs, scalar_features):
        r = original_predict(z_r, z_negs, scalar_features)  # (B, K)
        captured.append(r.detach().float().cpu().numpy().reshape(-1, r.shape[-1]))
        return r

    engine._predict_reliability = _recording_predict  # type: ignore[assignment]

    # Inject the ORCA adapter so trainer.active_loss_engine is the ORCA path
    # (warmup snapshot is irrelevant for reliability prediction — it only feeds
    # the BCE weak targets, which we are not measuring here).
    orchestrator._inject_orca_loss()

    # Move the ReliabilityMLP onto the trainer's device (the orchestrator does
    # this in phases 3/4, which the probe skips). Without it the CPU-resident
    # MLP receives CUDA embeddings and every triplet is skipped on a device
    # mismatch.
    orchestrator._move_reliability_to_device()

    # Deterministic, dropout-free forward.
    if getattr(trainer, "model", None) is not None:
        trainer.model.eval()
    if orchestrator.reliability_model is not None:
        orchestrator.reliability_model.eval()

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
            # Drives _predict_reliability via the recording hook.
            trainer.active_loss_engine.compute_loss(triplets, embeddings)
            n += 1
            if n >= max_batches:
                break

    engine._predict_reliability = original_predict  # restore

    if not captured:
        raise SystemExit("No reliabilities captured — no valid triplets/embeddings produced.")

    all_values = np.concatenate([row.reshape(-1) for row in captured])
    # Within-anchor std across the K negatives (only anchors with K >= 2).
    within = [np.std(row) for arr in captured for row in arr if row.size >= 2]
    within_stds = np.array(within) if within else np.array([0.0])
    return all_values, within_stds, n


def _report(variant, all_values, within_stds, n_batches):
    v = all_values
    print("=" * 66)
    print(f"Reliability probe — variant={variant}  (batches={n_batches}, "
          f"pairs={v.size})")
    print("=" * 66)
    print("GLOBAL distribution of r_psi over all (anchor, negative) pairs:")
    print(f"  mean   {v.mean():.4f}")
    print(f"  std    {v.std():.4f}")
    print(f"  min    {v.min():.4f}")
    print(f"  p05    {np.percentile(v, 5):.4f}")
    print(f"  p25    {np.percentile(v, 25):.4f}")
    print(f"  median {np.median(v):.4f}")
    print(f"  p75    {np.percentile(v, 75):.4f}")
    print(f"  p95    {np.percentile(v, 95):.4f}")
    print(f"  max    {v.max():.4f}")
    frac_near_1 = float(np.mean(v > 0.95))
    frac_near_floor = float(np.mean(v < 0.10))
    print(f"  frac > 0.95   {frac_near_1:.3f}")
    print(f"  frac < 0.10   {frac_near_floor:.3f}")
    print("-" * 66)
    print("WITHIN-ANCHOR spread (std across the K negatives per anchor):")
    print(f"  mean within-anchor std {within_stds.mean():.4f}")
    print(f"  median                 {np.median(within_stds):.4f}")
    print(f"  max                    {within_stds.max():.4f}")
    print("-" * 66)
    # Interpretation heuristic.
    if v.std() < 0.03 and within_stds.mean() < 0.03:
        verdict = ("NEAR-CONSTANT reliabilities. Denominator calibration is ~a "
                   "uniform scaling -> ER-DEN ~= Standard_InfoNCE, and ER-EXT's "
                   "detached scalar under Adam is also ~= Standard_InfoNCE. The "
                   "ER-DEN==ER-EXT tie is EXPECTED and is NOT evidence about "
                   "placement. Reframe the paper claim accordingly.")
    elif within_stds.mean() < 0.03:
        verdict = ("Reliabilities vary ACROSS anchors but are ~uniform WITHIN an "
                   "anchor's negative set. Denominator calibration still scales "
                   "each anchor's negatives ~uniformly, so per-negative "
                   "discrimination is weak — a likely contributor to the tie.")
    else:
        verdict = ("Reliabilities genuinely vary across negatives within an "
                   "anchor. The tie is therefore NOT explained by constant "
                   "reliabilities; placement genuinely does not move this metric "
                   "on this dataset.")
    print("VERDICT:", verdict)
    print("=" * 66)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best_checkpoint.pt of the run.")
    ap.add_argument("--config", required=True, help="training_config.json of the run.")
    ap.add_argument("--data", required=True, help="A prepared split (train.jsonl) with skill_uris.")
    ap.add_argument("--max-batches", type=int, default=20, help="How many batches to probe.")
    ap.add_argument("--output-dir", default="/tmp/orca_probe",
                    help="Scratch dir for the trainer (no artifacts of interest written).")
    args = ap.parse_args()

    for p in (args.checkpoint, args.config, args.data):
        if not Path(p).exists():
            logger.error("Not found: %s", p)
            return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)

    config, variant = _load_config(args.config, ckpt)
    logger.info("Probing variant=%s on %s (device=%s)", variant, args.data, device)

    trainer, orchestrator = _build(config, args.output_dir)
    _load_weights(trainer, orchestrator, ckpt)
    _maybe_load_global_pool(trainer, config, args.data)

    all_values, within_stds, n = _collect(trainer, orchestrator, args.data, args.max_batches)
    _report(variant, all_values, within_stds, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
