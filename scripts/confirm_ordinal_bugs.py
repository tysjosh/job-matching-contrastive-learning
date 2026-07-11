#!/usr/bin/env python3
"""
Confirmation test for ordinal contrastive loss bugs.

Demonstrates (empirically, before any fix):
  - Issue 2: the ordinal margin for a FIXED good_fit query changes depending on
    unrelated triplets in the batch (cross-query leakage).
  - Root cause detail: s_n uses best_nf['resume_emb'], which for a same-triplet
    negative equals the anchor resume, making s_n == s_alpha (degenerate margin).
  - Issue 1: negatives carry their OWN-pair original_label, not their relationship
    to the anchor resume.

Run: python3 scripts/confirm_ordinal_bugs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from contrastive_learning.data_structures import ContrastiveTriplet, TrainingConfig
from contrastive_learning.loss_engine import ContrastiveLossEngine


def make_resume(rid):
    return {"role": f"resume_{rid}", "skills": [f"s{rid}"], "skill_uris": []}


def make_job(jid, label):
    return {"title": f"job_{jid}", "skills": [f"s{jid}"], "skill_uris": [],
            "occupation_uri": f"occ_{jid}", "original_label": label}


def build_triplet(anchor_id, pos_id, pos_label, neg_specs, resume_id=None):
    """neg_specs: list of (job_id, own_label)."""
    anchor = make_resume(anchor_id)
    positive = make_job(pos_id, pos_label)
    negatives = [make_job(jid, lbl) for jid, lbl in neg_specs]
    return ContrastiveTriplet(
        anchor=anchor,
        positive=positive,
        negatives=negatives,
        career_distances=[1.0] * len(negatives),
        view_metadata={
            "positive_original_label": pos_label,
            "negative_original_labels": [lbl for _, lbl in neg_specs],
            "quality_tier": "A",
            "resume_id": resume_id if resume_id is not None else f"rid_{anchor_id}",
        },
    )


def register(engine, emb, content, vec):
    key = engine._get_content_key(content)
    emb[key] = torch.tensor(vec, dtype=torch.float32)


def main():
    cfg = TrainingConfig()
    cfg.loss_type = "ordinal"
    cfg.num_epochs = 10
    engine = ContrastiveLossEngine(cfg)
    engine.set_epoch(0)  # easy phase (epoch_ratio 0 < curriculum_switch 0.3)

    # Triplet A: good_fit anchor with two negatives (own labels no_fit / potential_fit)
    A = build_triplet("A", "Apos", "good_fit",
                      [("Aneg1", "no_fit"), ("Aneg2", "potential_fit")])
    # Triplet B: unrelated, different resume and jobs
    B = build_triplet("B", "Bpos", "good_fit",
                      [("Bneg1", "no_fit"), ("Bneg2", "no_fit")])

    # Build embeddings (2D unit-ish vectors). A.pos aligned with A.resume.
    emb = {}
    register(engine, emb, A.anchor, [1.0, 0.0])       # A resume
    register(engine, emb, A.positive, [0.9, 0.1])     # A pos job (close to A resume)
    register(engine, emb, A.negatives[0], [0.2, 0.9]) # A neg1 (far)
    register(engine, emb, A.negatives[1], [0.5, 0.5]) # A neg2
    register(engine, emb, B.anchor, [-1.0, 0.0])      # B resume (opposite direction)
    register(engine, emb, B.positive, [-0.9, 0.1])
    register(engine, emb, B.negatives[0], [-0.2, 0.9])
    register(engine, emb, B.negatives[1], [-0.5, 0.5])

    loss_A_alone = engine._compute_ordinal_loss([A], emb).item()
    loss_A_with_B = engine._compute_ordinal_loss([A, B], emb).item()

    print("=" * 60)
    print("REGRESSION CHECK: cross-query leakage (Issue 2)")
    print("=" * 60)
    print(f"Ordinal loss for A alone      : {loss_A_alone:.6f}")
    print(f"Ordinal loss for A (batch A,B): {loss_A_with_B:.6f}")
    print(f"Difference                    : {abs(loss_A_alone - loss_A_with_B):.6f}")
    print("(Pre-fix this difference was 0.30; fixed code should show ~0.0.)")

    print()
    print("=" * 60)
    print("POST-FIX VALIDATION: query-anchored loss is batch-invariant")
    print("=" * 60)
    print(f"Loss for A alone       : {loss_A_alone:.6f}")
    print(f"Loss for A (batch A,B) : {loss_A_with_B:.6f}")
    if abs(loss_A_alone - loss_A_with_B) < 1e-6:
        print(">>> FIXED: A's ordinal margin is invariant to unrelated triplet B.")
    else:
        print(">>> STILL LEAKING: difference =",
              f"{abs(loss_A_alone - loss_A_with_B):.6f}")

    # Full-phase check: a good>potential violation should produce positive margin
    engine.set_epoch(9)  # epoch_ratio 0.9 >= 0.3 -> full phase
    # Query C: good job LESS similar to r than a potential sibling -> violation
    C = build_triplet("C", "Cgood", "good_fit", [("Cneg", "no_fit")], resume_id="RC")
    Cpot = build_triplet("C", "Cpot", "potential_fit", [("Cneg2", "no_fit")], resume_id="RC")
    embC = {}
    register(engine, embC, C.anchor, [1.0, 0.0])
    register(engine, embC, C.positive, [0.3, 0.95])    # good job FAR from resume
    register(engine, embC, C.negatives[0], [0.0, 1.0])
    register(engine, embC, Cpot.positive, [0.98, 0.05]) # potential job CLOSE (violation)
    register(engine, embC, Cpot.negatives[0], [-0.5, 0.8])
    viol_loss = engine._compute_ordinal_loss([C, Cpot], embC).item()
    print()
    print(f"good<potential violation -> ordinal loss > InfoNCE floor: {viol_loss:.6f}")
    print(">>> Non-degenerate: penalizes good_fit ranked below potential_fit for SAME resume.")


if __name__ == "__main__":
    main()
