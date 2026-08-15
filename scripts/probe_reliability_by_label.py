#!/usr/bin/env python3
"""Mechanism validation: does ORCA's learned reliability recover HUMAN judgments
of partial fit, using labels the model never saw?

Why this exists
---------------
The headline ORCA result (+0.038 AUC over E4-OSCAR-Skill) is *indirect* evidence
for the paper's mechanism claim. It shows a metric moved; it does not show that
the ReliabilityMLP actually identifies false negatives. That gap is what a
reviewer attacks: the same gain is equally consistent with the auxiliary BCE term
regularizing the projection head, or with the 4-phase schedule, or with the extra
epochs.

This probe closes the gap by exploiting a gold standard that is already in the
dataset but is deliberately DISCARDED during training:

    original_label     binary label used in training
    good_fit        ->  1  (positive)
    potential_fit   ->  0  (negative)   <- 33% of all negatives; a PARTIAL match
    no_fit          ->  0  (negative)

The binary collapse throws away the potential_fit / no_fit distinction, and the
weak targets that supervise the ReliabilityMLP never see ``original_label``
either — ``r_ont`` comes from ESCO/ISCO distance and ``r_enc`` from warmup-encoder
cosine (see orca/weak_targets.py). So the three-class label is genuinely held-out
information for this model. If reliability recovers it, the model learned
something nobody told it, and the mechanism claim is supported directly.

The hypothesis (ORCA's premise) predicts a monotone ordering, since reliability
means "probability this is a TRUE negative":

    r(good_fit)  <  r(potential_fit)  <  r(no_fit)

What is reported
----------------
  * mean +/- std reliability per original_label class;
  * AUC using (1 - r) to discriminate potential_fit from no_fit among negatives
    -- THE headline number. ~0.5 means reliability carries no information about
    real ambiguity; 0.65-0.75 is strong direct support;
  * AUC for good_fit vs no_fit (an easier contrast; sanity check);
  * Spearman rho between r and the ordinal rank (good=0, potential=1, no_fit=2),
    which tests the full monotone ordering rather than one pairwise split;
  * the reliability spread, which also answers the question
    scripts/probe_reliability.py was written for: if the within-anchor std is
    ~0, the denominator is only a uniform rescale and the ER-DEN/ER-EXT tie is
    expected rather than interesting.

Run it ON THE GPU BOX where the checkpoints live. Use the TEST split so the
pairs are held out:

    d=results/research_runs/ER-DEN__cnamuangtoun__s13
    python scripts/probe_reliability_by_label.py \
        --checkpoint $d/phase1_pretraining/best_checkpoint.pt \
        --config     $d/training_config.json \
        --data       preprocess/data_splits_v7/test.jsonl \
        --json-out   $d/phase1_evaluation/reliability_by_label.json

Then aggregate across seeds with --json-out files, or just read the printed
table. Compare ER-DEN against ER-NOONTW/ER-NOONTALL: if removing the ontology
weak target collapses this AUC toward 0.5, the ontology is what makes reliability
learnable, which is a far stronger claim than its ~20% share of the AUC gain.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("probe_reliability_by_label")
logger.setLevel(logging.INFO)

# Ordinal rank for the monotonicity test: reliability should INCREASE with rank
# (a no_fit job is the most reliable negative; a good_fit job the least).
ORDINAL_RANK = {"good_fit": 0, "potential_fit": 1, "no_fit": 2}
CLASS_ORDER = ["good_fit", "potential_fit", "no_fit"]


# --------------------------------------------------------------------------- #
# Loading (mirrors scripts/probe_reliability.py so both probes share one path)
# --------------------------------------------------------------------------- #
def _load_config(config_path: str, ckpt: dict):
    from contrastive_learning.data_structures import TrainingConfig
    config = TrainingConfig.from_json(config_path)
    config.orca_enabled = True
    variant = ckpt.get("orca_variant") or getattr(config, "orca_variant", "denominator")
    config.orca_variant = variant
    return config, variant


def _build(config, output_dir: str):
    from contrastive_learning.trainer import ContrastiveLearningTrainer
    from orca.orchestrator import OrcaPhaseOrchestrator
    trainer = ContrastiveLearningTrainer(config=config, output_dir=output_dir)
    orchestrator = OrcaPhaseOrchestrator(config, trainer)
    return trainer, orchestrator


def _load_weights(trainer, orchestrator, ckpt: dict) -> None:
    msd = ckpt.get("model_state_dict")
    if msd is not None and getattr(trainer, "model", None) is not None:
        missing, unexpected = trainer.model.load_state_dict(msd, strict=False)
        if missing or unexpected:
            logger.warning("projection load_state_dict: missing=%s unexpected=%s",
                           list(missing), list(unexpected))
    rsd = ckpt.get("reliability_model_state_dict")
    if rsd is None:
        raise SystemExit(
            "Checkpoint has no reliability_model_state_dict; not an ORCA checkpoint.")
    rmodel = getattr(orchestrator, "reliability_model", None)
    if rmodel is None:
        raise SystemExit("Orchestrator built no reliability_model; check orca_variant.")
    rmodel.load_state_dict(rsd, strict=True)


# --------------------------------------------------------------------------- #
# Pair construction
# --------------------------------------------------------------------------- #
def _resume_key(resume: dict) -> str:
    """Stable grouping key for a resume (content-hashed)."""
    import hashlib
    return hashlib.md5(json.dumps(resume, sort_keys=True).encode()).hexdigest()


def _load_labeled_pairs(data_path: str):
    """Group the split's rows by resume, keeping each job's original_label.

    Returns ``{resume_key: {"resume": dict, "jobs": [(job_dict, original_label)]}}``.
    Only resumes carrying at least one ``potential_fit`` AND one ``no_fit`` job are
    kept, because those are the resumes that support the headline within-anchor
    contrast (both classes scored against the SAME anchor, so the comparison is
    not confounded by resume difficulty).
    """
    groups = defaultdict(lambda: {"resume": None, "jobs": []})
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            label = (row.get("metadata") or {}).get("original_label")
            if label not in ORDINAL_RANK:
                continue
            k = _resume_key(row["resume"])
            groups[k]["resume"] = row["resume"]
            groups[k]["jobs"].append((row["job"], label))

    usable = {}
    for k, g in groups.items():
        labels = {lab for _, lab in g["jobs"]}
        if "potential_fit" in labels and "no_fit" in labels:
            usable[k] = g
    return groups, usable


def _score_group(trainer, orchestrator, group) -> list:
    """Reliability for every (resume, job) pair in one group.

    Reuses the production path end to end: the trainer's own embedding generation
    (so the trained projection head is applied exactly as in training), OSCAR's
    ``_compute_negative_ontology_features`` for the ontology scalars, the ORCA
    adapter's ``_build_scalar_features`` for the width/ordering, and finally the
    engine's ``_predict_reliability``.

    Returns a list of ``(original_label, reliability)`` tuples.
    """
    from contrastive_learning.data_structures import ContrastiveTriplet, TrainingSample

    resume = group["resume"]
    jobs = [j for j, _ in group["jobs"]]
    labels = [lab for _, lab in group["jobs"]]

    # A triplet is only needed so the trainer embeds anchor + all candidate jobs
    # in one pass. ``positive`` is set to the first job purely to satisfy the
    # dataclass; every job also appears in ``negatives``, which is what we score.
    triplet = ContrastiveTriplet(
        anchor=resume,
        positive=jobs[0],
        negatives=list(jobs),
        career_distances=[0.0] * len(jobs),
        view_metadata={},
    )

    embeddings = trainer._generate_embeddings([triplet])
    if not embeddings:
        return []

    key_fn = trainer.embedding_cache.get_content_key
    anchor_key = key_fn(resume)
    if anchor_key not in embeddings:
        return []

    # Keep only jobs that actually got embedded, preserving label alignment.
    kept, kept_labels = [], []
    for job, lab in zip(jobs, labels):
        if key_fn(job) in embeddings:
            kept.append(job)
            kept_labels.append(lab)
    if not kept:
        return []

    # Ontology scalars via OSCAR's own producer (never reimplemented here).
    sample = TrainingSample(resume=resume, job=kept[0], label="negative",
                            sample_id="probe", metadata={})
    feats = trainer.batch_processor._compute_negative_ontology_features(sample, kept)

    adapter = orchestrator.loss_adapter
    scored_triplet = ContrastiveTriplet(
        anchor=resume, positive=kept[0], negatives=kept,
        career_distances=[0.0] * len(kept),
        view_metadata={"negative_ontology_features": feats} if feats else {},
    )

    z_r = embeddings[anchor_key]
    z_negs = torch.stack([embeddings[key_fn(j)] for j in kept], dim=0)
    device = z_negs.device
    scalar_features = adapter._build_scalar_features(
        scored_triplet, list(range(len(kept))), len(kept), z_negs.dtype, device)

    engine = orchestrator.loss_engine
    r = engine._predict_reliability(
        z_r.unsqueeze(0), z_negs.unsqueeze(0), scalar_features.unsqueeze(0))
    r = r.detach().float().cpu().numpy().reshape(-1)

    return list(zip(kept_labels, r.tolist()))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _auc(pos_scores, neg_scores) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-aware. NaN if either side is empty."""
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.empty_like(allv)
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    i = 0
    while i < sorted_v.size:  # average ranks within tie groups
        j = i
        while j + 1 < sorted_v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def _spearman(x, y) -> float:
    """Spearman rho without a scipy dependency."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty_like(v)
        sv = v[order]
        i = 0
        while i < sv.size:
            j = i
            while j + 1 < sv.size and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _report(variant, by_class, within_stds, n_groups, n_pairs):
    """Print the per-class table and the discrimination statistics."""
    pot = by_class.get("potential_fit", [])
    nof = by_class.get("no_fit", [])
    good = by_class.get("good_fit", [])

    print("=" * 74)
    print(f"Reliability vs HUMAN original_label — variant={variant}")
    print(f"anchors={n_groups}  scored pairs={n_pairs}")
    print("=" * 74)

    print(f"\n{'original_label':<16}{'n':>7}{'mean r':>10}{'std':>9}"
          f"{'p25':>9}{'median':>9}{'p75':>9}")
    print("-" * 74)
    for cls in CLASS_ORDER:
        v = np.asarray(by_class.get(cls, []), float)
        if v.size == 0:
            print(f"{cls:<16}{0:>7}{'n/a':>10}")
            continue
        print(f"{cls:<16}{v.size:>7}{v.mean():>10.4f}{v.std(ddof=1) if v.size > 1 else 0:>9.4f}"
              f"{np.percentile(v, 25):>9.4f}{np.median(v):>9.4f}{np.percentile(v, 75):>9.4f}")

    # Headline: does LOW reliability flag the human-labeled partial matches?
    auc_pot = _auc([-x for x in pot], [-x for x in nof])   # (1-r) ranking == -r
    auc_good = _auc([-x for x in good], [-x for x in nof])

    ranks, rvals = [], []
    for cls in CLASS_ORDER:
        for x in by_class.get(cls, []):
            ranks.append(ORDINAL_RANK[cls])
            rvals.append(x)
    rho = _spearman(ranks, rvals)

    print(f"\n{'discrimination':<46}{'value':>10}")
    print("-" * 74)
    print(f"{'AUC  (1-r) : potential_fit vs no_fit  [HEADLINE]':<46}{auc_pot:>10.4f}")
    print(f"{'AUC  (1-r) : good_fit vs no_fit  [sanity]':<46}{auc_good:>10.4f}")
    print(f"{'Spearman rho(r, ordinal rank)  [monotonicity]':<46}{rho:>10.4f}")

    gap = (np.mean(nof) - np.mean(pot)) if (pot and nof) else float("nan")
    print(f"{'mean r(no_fit) - mean r(potential_fit)':<46}{gap:>10.4f}")

    print(f"\n{'reliability spread':<46}{'value':>10}")
    print("-" * 74)
    allv = np.concatenate([np.asarray(v, float) for v in by_class.values() if len(v)])
    print(f"{'global std of r':<46}{allv.std(ddof=1) if allv.size > 1 else 0:>10.4f}")
    print(f"{'mean WITHIN-ANCHOR std of r':<46}{np.mean(within_stds):>10.4f}")

    print("\n" + "=" * 74)
    print("Interpretation")
    print("-" * 74)
    if np.isnan(auc_pot):
        print("  Not enough pairs in both classes to compute the headline AUC.")
    elif auc_pot >= 0.65:
        print(f"  AUC={auc_pot:.3f}: reliability RECOVERS human partial-fit judgments it")
        print("  was never trained on. This is direct support for the mechanism claim:")
        print("  the ontology + encoder weak signals suffice to identify ambiguous")
        print("  negatives, so ORCA's gain has an explanation, not just a correlation.")
    elif auc_pot >= 0.55:
        print(f"  AUC={auc_pot:.3f}: weak but non-trivial signal. Reliability is partially")
        print("  aligned with human ambiguity. Reportable, but do not overclaim the")
        print("  mechanism; pair it with the ER-SCHED schedule control.")
    else:
        print(f"  AUC={auc_pot:.3f}: reliability does NOT track real ambiguity. The +0.038")
        print("  AUC gain then needs another explanation (most likely the 4-phase")
        print("  schedule or the auxiliary BCE acting as a regularizer). Run the")
        print("  ER-SCHED control (orca_r_min=1.0, orca_eta_rel=0.0) before writing")
        print("  the mechanism story.")
    if np.mean(within_stds) < 1e-3:
        print("\n  WARNING: within-anchor std ~= 0, so the reliability-calibrated")
        print("  denominator is only a UNIFORM rescale of the negative mass. That makes")
        print("  the ER-DEN vs ER-EXT tie expected rather than informative.")
    print("=" * 74)

    return {
        "variant": variant,
        "n_anchors": n_groups,
        "n_pairs": n_pairs,
        "per_class": {
            cls: {
                "n": len(by_class.get(cls, [])),
                "mean": float(np.mean(by_class[cls])) if by_class.get(cls) else None,
                "std": float(np.std(by_class[cls], ddof=1)) if len(by_class.get(cls, [])) > 1 else None,
            } for cls in CLASS_ORDER
        },
        "auc_potential_vs_nofit": None if np.isnan(auc_pot) else auc_pot,
        "auc_good_vs_nofit": None if np.isnan(auc_good) else auc_good,
        "spearman_r_vs_ordinal_rank": None if np.isnan(rho) else rho,
        "mean_gap_nofit_minus_potential": None if np.isnan(gap) else float(gap),
        "global_std_r": float(allv.std(ddof=1)) if allv.size > 1 else 0.0,
        "mean_within_anchor_std_r": float(np.mean(within_stds)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate ORCA reliability against held-out human original_label.")
    ap.add_argument("--checkpoint", required=True, help="ORCA best_checkpoint.pt")
    ap.add_argument("--config", required=True, help="The run's training_config.json")
    ap.add_argument("--data", required=True,
                    help="Split JSONL with metadata.original_label (use the TEST split)")
    ap.add_argument("--max-anchors", type=int, default=0,
                    help="Cap the number of resumes scored (0 = all)")
    ap.add_argument("--json-out", default=None, help="Write the summary as JSON")
    ap.add_argument("--output-dir", default="results/_probe_by_label",
                    help="Scratch dir for trainer construction")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, variant = _load_config(args.config, ckpt)
    trainer, orchestrator = _build(config, args.output_dir)
    _load_weights(trainer, orchestrator, ckpt)

    # Build the ORCA engine/adapter, then move the MLP to the trainer's device
    # (phases 3/4 normally do this; this probe skips them).
    orchestrator._inject_orca_loss()
    orchestrator._move_reliability_to_device()

    if getattr(trainer, "model", None) is not None:
        trainer.model.eval()
    if orchestrator.reliability_model is not None:
        orchestrator.reliability_model.eval()

    all_groups, usable = _load_labeled_pairs(args.data)
    logger.info("resumes: %d total, %d with both potential_fit and no_fit",
                len(all_groups), len(usable))
    if not usable:
        raise SystemExit(
            "No resume carries both a potential_fit and a no_fit job; the "
            "within-anchor contrast is not computable on this split.")

    keys = sorted(usable)
    if args.max_anchors:
        keys = keys[:args.max_anchors]

    by_class = defaultdict(list)
    within_stds, n_pairs = [], 0
    with torch.no_grad():
        for i, k in enumerate(keys, 1):
            try:
                scored = _score_group(trainer, orchestrator, usable[k])
            except Exception as exc:
                logger.warning("anchor %s skipped: %s", k[:8], exc)
                continue
            if not scored:
                continue
            for lab, r in scored:
                by_class[lab].append(r)
            n_pairs += len(scored)
            vals = [r for _, r in scored]
            if len(vals) >= 2:
                within_stds.append(float(np.std(vals)))
            if i % 50 == 0:
                logger.info("scored %d/%d anchors (%d pairs)", i, len(keys), n_pairs)

    if not n_pairs:
        raise SystemExit("No pairs scored — check the checkpoint and split.")

    summary = _report(variant, by_class, within_stds or [0.0], len(keys), n_pairs)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
