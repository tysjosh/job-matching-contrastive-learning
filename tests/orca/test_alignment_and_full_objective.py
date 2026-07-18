#!/usr/bin/env python3
"""Unit tests for the ORCA ontology-alignment loss and full-objective assembly.

Components under test:
  * ``orca.alignment.OntologyAlignmentLoss`` (design section B.6, task 5.1).
  * ``orca.loss_engine.OrcaLossEngine.compute_loss`` full-objective assembly
    with the alignment term wired in (design section B.4, task 5.2).

Requirements exercised:
  * Requirement 4.3 — alignment loss math ``mean((cosine(z_a, z_b) - s_ont)**2)``
    and its inclusion (scaled by ``orca_lambda_align``) when
    ``orca_use_alignment`` is true.
  * Requirement 4.4 — with ``orca_use_alignment`` false the alignment term is
    excluded, yielding a total equal to setting ``orca_lambda_align = 0``.
  * Requirement 4.5 — a non-finite alignment term (hence a non-finite assembled
    total) yields a finite zero-gradient loss with a warning and does not raise.

These are example/edge-case unit tests (not Hypothesis property tests): they pin
the exact alignment math, the additive assembly ``loss_main + eta_rel*loss_rel
+ lambda_align*loss_align``, the alignment-off equivalence, and the non-finite
guard on the assembled total.

The engine is constructed *directly* (not via ``orca.factory.make_loss_engine``)
so these tests control ``orca_use_alignment`` independently of the MVP/variant
factory guard (task 6.1). Real ``ReliabilityMLP`` and ``WeakTargetBuilder``
instances are supplied; the reliability model is put in ``eval`` mode so dropout
is disabled and its predictions are deterministic across the independent
reconstructions below.

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_alignment_and_full_objective.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from contrastive_learning.data_structures import TrainingConfig
from orca.alignment import OntologyAlignmentLoss
from orca.loss_engine import OrcaLossEngine
from orca.reliability_model import ReliabilityMLP
from orca.types import JobPairs
from orca.weak_targets import WeakTargetBuilder


# Small, fixed problem dimensions (B anchors x K negatives, embed dim D, feature
# dim F, P job pairs) — deliberately tiny to keep the arithmetic transparent.
_B, _K, _D, _F, _P = 2, 3, 4, 5, 4

_RTOL = 1e-5
_ATOL = 1e-6


def _job_pairs(*, perfect: bool = False, non_finite: bool = False) -> JobPairs:
    """Build a small ``JobPairs`` batch.

    Args:
        perfect: When true, ``z_b == z_a`` (cosine == 1) and ``s_ont == 1`` so
            the alignment loss is ~0.
        non_finite: When true, inject an infinity into ``z_a`` so the alignment
            term (and hence the assembled total) becomes non-finite.
    """
    g = torch.Generator().manual_seed(7)
    z_a = torch.randn(_P, _D, generator=g)
    if perfect:
        z_b = z_a.clone()
        s_ont = torch.ones(_P)
    else:
        z_b = torch.randn(_P, _D, generator=g)
        s_ont = torch.rand(_P, generator=g)
    if non_finite:
        z_a = z_a.clone()
        z_a[0, 0] = float("inf")
    return JobPairs(z_a=z_a, z_b=z_b, s_ont=s_ont)


def _batch():
    """Reproducible ``(z_r, z_pos, z_negs, scalar_features, weak_targets)``."""
    g = torch.Generator().manual_seed(11)
    z_r = torch.randn(_B, _D, generator=g)
    z_pos = torch.randn(_B, _D, generator=g)
    z_negs = torch.randn(_B, _K, _D, generator=g)
    scalar_features = torch.randn(_B, _K, _F, generator=g)
    weak_targets = torch.rand(_B, _K, generator=g)  # r_tilde in [0, 1]
    return z_r, z_pos, z_negs, scalar_features, weak_targets


def _make_engine(*, use_alignment: bool, variant: str, lambda_align: float,
                 reliability_model=None, weak_builder=None) -> OrcaLossEngine:
    """Construct an ``OrcaLossEngine`` directly (bypassing the factory guard).

    Real ``ReliabilityMLP`` / ``WeakTargetBuilder`` instances are created unless
    shared ones are passed in (used by the alignment-off equivalence test so the
    two engines produce identical reliability predictions).
    """
    cfg = TrainingConfig(
        orca_enabled=True,
        orca_variant=variant,
        orca_use_alignment=use_alignment,
        orca_lambda_align=lambda_align,
        orca_eta_rel=0.5,
    )
    if reliability_model is None:
        torch.manual_seed(0)
        reliability_model = ReliabilityMLP(embed_dim=_D, feature_dim=_F)
        reliability_model.eval()  # disable dropout -> deterministic predictions
    if weak_builder is None:
        weak_builder = WeakTargetBuilder(cfg)
    return OrcaLossEngine(
        cfg, skill_matcher=None,
        reliability_model=reliability_model, weak_builder=weak_builder,
    )


# ---------------------------------------------------------------------------
# Requirement 4.3 — OntologyAlignmentLoss math
# ---------------------------------------------------------------------------

def test_alignment_loss_equals_mean_squared_cosine_gap():
    """loss_align == mean((cosine(z_a, z_b) - s_ont)**2) (Requirement 4.3)."""
    loss_fn = OntologyAlignmentLoss()
    pairs = _job_pairs()

    got = loss_fn(pairs)

    cos = F.cosine_similarity(pairs.z_a, pairs.z_b, dim=-1)
    expected = ((cos - pairs.s_ont) ** 2).mean()

    assert got.shape == ()
    assert torch.isfinite(got)
    assert torch.allclose(got, expected, rtol=_RTOL, atol=_ATOL)


def test_alignment_loss_perfect_alignment_is_zero():
    """When cosine == s_ont for every pair, the alignment loss is ~0."""
    loss_fn = OntologyAlignmentLoss()
    got = loss_fn(_job_pairs(perfect=True))

    assert got.shape == ()
    assert torch.allclose(got, torch.zeros(()), atol=1e-6)


def test_alignment_loss_empty_pairs_is_zero_scalar():
    """An empty pair batch yields a zero scalar (never a NaN mean)."""
    loss_fn = OntologyAlignmentLoss()
    empty = JobPairs(
        z_a=torch.empty(0, _D),
        z_b=torch.empty(0, _D),
        s_ont=torch.empty(0),
    )

    got = loss_fn(empty)

    assert got.shape == ()
    assert torch.isfinite(got)
    assert float(got) == 0.0


# ---------------------------------------------------------------------------
# Requirement 4.3 — full-objective assembly with alignment ON
# ---------------------------------------------------------------------------

def test_full_objective_assembly_with_alignment_on():
    """total == loss_main + eta_rel*loss_rel + lambda_align*loss_align (Req 4.3).

    The engine is built directly with a real ReliabilityMLP (eval mode) and a
    real WeakTargetBuilder; each term is reconstructed independently using the
    engine's own methods and summed to check the additive assembly.
    """
    engine = _make_engine(
        use_alignment=True, variant="full", lambda_align=0.3,
    )
    z_r, z_pos, z_negs, scalar_features, weak_targets = _batch()
    pairs = _job_pairs()

    total = engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features,
        weak_targets=weak_targets, job_pairs=pairs,
    )

    # Independently reconstruct each additive term (reliability is deterministic
    # because the model is in eval mode).
    reliability = engine._predict_reliability(z_r, z_negs, scalar_features)
    expected_main = engine.orca_loss(z_r, z_pos, z_negs, reliability).mean()
    expected_rel = engine.reliability_loss(reliability, weak_targets)
    expected_align = engine.alignment(pairs)
    expected_total = (
        expected_main
        + engine.eta_rel * expected_rel
        + engine.lambda_align * expected_align
    )

    assert engine.alignment is not None  # alignment wired in when use_alignment
    assert total.shape == ()
    assert torch.isfinite(total)
    assert torch.allclose(total, expected_total, rtol=_RTOL, atol=_ATOL)

    # The alignment term genuinely contributes (non-perfect pairs -> > 0), so
    # this is not vacuously equal to the no-alignment total.
    assert float(engine.lambda_align * expected_align) > 0.0


# ---------------------------------------------------------------------------
# Requirement 4.4 — alignment OFF reproduces the no-alignment total
# ---------------------------------------------------------------------------

def test_alignment_off_reproduces_no_alignment_total():
    """With alignment off, total == loss_main + eta_rel*loss_rel (Req 4.4)."""
    engine = _make_engine(
        use_alignment=False, variant="no_align", lambda_align=0.3,
    )
    z_r, z_pos, z_negs, scalar_features, weak_targets = _batch()
    pairs = _job_pairs()

    # job_pairs are supplied but must be ignored while alignment is off.
    total = engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features,
        weak_targets=weak_targets, job_pairs=pairs,
    )

    reliability = engine._predict_reliability(z_r, z_negs, scalar_features)
    expected_main = engine.orca_loss(z_r, z_pos, z_negs, reliability).mean()
    expected_rel = engine.reliability_loss(reliability, weak_targets)
    expected_total = expected_main + engine.eta_rel * expected_rel

    assert engine.alignment is None  # alignment not constructed when off
    assert torch.isfinite(total)
    assert torch.allclose(total, expected_total, rtol=_RTOL, atol=_ATOL)


def test_alignment_off_equals_alignment_on_with_lambda_align_zero():
    """Alignment off == alignment on with lambda_align=0 (Req 4.4).

    Both engines share the same ReliabilityMLP (eval) and WeakTargetBuilder so
    their reliability predictions and reliability BCE are identical; the only
    possible difference is the alignment term, which is zero on both sides.
    """
    torch.manual_seed(0)
    shared_model = ReliabilityMLP(embed_dim=_D, feature_dim=_F).eval()
    shared_builder = WeakTargetBuilder(TrainingConfig(orca_enabled=True))

    engine_off = _make_engine(
        use_alignment=False, variant="no_align", lambda_align=0.3,
        reliability_model=shared_model, weak_builder=shared_builder,
    )
    engine_on_zero = _make_engine(
        use_alignment=True, variant="full", lambda_align=0.0,
        reliability_model=shared_model, weak_builder=shared_builder,
    )

    z_r, z_pos, z_negs, scalar_features, weak_targets = _batch()
    pairs = _job_pairs()

    total_off = engine_off.compute_loss(
        z_r, z_pos, z_negs, scalar_features,
        weak_targets=weak_targets, job_pairs=pairs,
    )
    total_on_zero = engine_on_zero.compute_loss(
        z_r, z_pos, z_negs, scalar_features,
        weak_targets=weak_targets, job_pairs=pairs,
    )

    assert torch.allclose(total_off, total_on_zero, rtol=_RTOL, atol=_ATOL)


# ---------------------------------------------------------------------------
# Requirement 4.5 — non-finite guard on the assembled total loss
# ---------------------------------------------------------------------------

def test_non_finite_alignment_term_yields_zero_gradient_loss():
    """A non-finite alignment term -> finite zero-gradient total (Req 4.5).

    Feeding ``JobPairs`` with an infinite embedding drives the alignment term
    (and thus the assembled total) non-finite. ``compute_loss`` must not raise;
    it must return a finite, zero-valued loss that carries gradient (a fresh
    leaf) so model parameters stay unchanged for the step.
    """
    engine = _make_engine(
        use_alignment=True, variant="full", lambda_align=0.3,
    )
    z_r, z_pos, z_negs, scalar_features, weak_targets = _batch()
    bad_pairs = _job_pairs(non_finite=True)

    # Sanity: the alignment term really is non-finite for these pairs.
    assert not torch.isfinite(engine.alignment(bad_pairs)).all()

    # Must not raise.
    total = engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features,
        weak_targets=weak_targets, job_pairs=bad_pairs,
    )

    # Finite, zero-valued, gradient-carrying fallback (Requirement 4.5).
    assert torch.isfinite(total).all()
    assert float(total.detach()) == 0.0
    assert total.requires_grad

    # Backprop leaves parameters untouched: the fallback is disconnected from
    # the model, so every parameter gradient is None (or zero) after backward.
    total.backward()
    for param in engine.reliability_model.parameters():
        assert param.grad is None or torch.all(param.grad == 0)
