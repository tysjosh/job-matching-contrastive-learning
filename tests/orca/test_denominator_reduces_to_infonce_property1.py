#!/usr/bin/env python3
"""Property-based test for the ORCA reliability-calibrated denominator (Property 1).

Component under test: ``orca.loss_engine.OrcaLossEngine`` (design section B.4),
specifically the reduction of ``orca_loss`` to ``standard_infonce`` when every
per-negative reliability equals ``1.0``.

    Property 1 — Denominator reduces to standard InfoNCE
    For all anchors / positives / negatives, if every ``reliability_k = 1``,
    then ``orca_loss(z_r, z_pos, z_negs, ones) == standard_infonce(z_r, z_pos,
    z_negs)`` within a relative tolerance of ``1e-5`` (or an absolute tolerance
    of ``1e-6`` when the Standard_InfoNCE reference loss is tiny).

**Validates: Requirements 3.1**

The intuition (design B.4): the reliability-calibrated denominator
``pos + sum_k(clamp(reliability_k, r_min, 1.0) * exp(sim_k / tau))`` collapses to
the standard InfoNCE denominator ``pos + sum_k(exp(sim_k / tau))`` precisely when
every reliability clamps to ``1.0`` (``clamp(1.0, r_min, 1.0) == 1.0``). Because
``OrcaLossEngine`` reuses identical numerical-stability constants for both paths,
the two losses agree to floating-point tolerance across a wide range of shapes
and embedding magnitudes.

Hypothesis varies the batch size ``B``, the number of negatives ``K`` (including
the empty-negative case ``K = 0``, Requirement 3.4), the embedding dim ``D``, and
the embedding values (including large magnitudes that saturate the shared
``max_exp`` clamp identically on both paths).

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_denominator_reduces_to_infonce_property1.py
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from contrastive_learning.data_structures import TrainingConfig
from orca.loss_engine import OrcaLossEngine


# Tolerances from Requirement 3.1: relative 1e-5, or absolute 1e-6 for tiny
# reference magnitudes. ``torch.allclose(a, b, rtol, atol)`` tests
# ``|a - b| <= atol + rtol * |b|``, which reduces to ~atol when the reference
# ``b`` is tiny and to ~rtol*|b| when it is large — matching the requirement.
_RTOL = 1e-5
_ATOL = 1e-6


def _make_engine() -> OrcaLossEngine:
    """Construct a minimal MVP ORCA-Denominator engine.

    ``orca_loss`` / ``standard_infonce`` only consume ``z_r`` / ``z_pos`` /
    ``z_negs`` / ``reliability``; the reliability model and weak-target builder
    are unused by these methods, so ``None`` placeholders keep the fixture
    minimal (per the task's construction note).
    """
    cfg = TrainingConfig(orca_enabled=True, orca_variant="denominator")
    return OrcaLossEngine(
        cfg, skill_matcher=None, reliability_model=None, weak_builder=None,
    )


@st.composite
def _loss_inputs(draw):
    """Draw a ``(z_r, z_pos, z_negs, reliability_ones)`` tuple.

    ``K = 0`` is allowed to exercise the empty-negative denominator reduction
    (Requirement 3.4). Values span a wide magnitude range so the shared
    ``max_exp`` similarity clamp is exercised identically on both code paths.
    """
    B = draw(st.integers(min_value=1, max_value=4))
    K = draw(st.integers(min_value=0, max_value=5))
    D = draw(st.integers(min_value=1, max_value=8))

    value_st = st.floats(
        min_value=-50.0, max_value=50.0,
        allow_nan=False, allow_infinity=False, width=32,
    )

    def _tensor(*shape):
        n = 1
        for d in shape:
            n *= d
        flat = draw(st.lists(value_st, min_size=n, max_size=n))
        return torch.tensor(flat, dtype=torch.float32).reshape(shape)

    z_r = _tensor(B, D)
    z_pos = _tensor(B, D)
    z_negs = _tensor(B, K, D)
    reliability_ones = torch.ones(B, K, dtype=torch.float32)
    return z_r, z_pos, z_negs, reliability_ones


# Feature: orca, Property 1: Denominator reduces to standard InfoNCE
@settings(max_examples=250, deadline=None)
@given(_loss_inputs())
def test_denominator_reduces_to_standard_infonce(data):
    """orca_loss with all-ones reliability equals standard_infonce.

    Validates: Requirements 3.1
    """
    z_r, z_pos, z_negs, reliability_ones = data
    engine = _make_engine()

    orca = engine.orca_loss(z_r, z_pos, z_negs, reliability_ones)
    standard = engine.standard_infonce(z_r, z_pos, z_negs)

    # Same per-anchor shape and every element finite.
    assert orca.shape == standard.shape == z_r.shape[:-1]
    assert torch.isfinite(orca).all()
    assert torch.isfinite(standard).all()

    # Reduction holds within the Requirement 3.1 tolerances.
    assert torch.allclose(orca, standard, rtol=_RTOL, atol=_ATOL)


def test_empty_negative_set_reduces_to_positive_only():
    """With K = 0, both losses use the positive term alone (Requirement 3.4).

    Validates: Requirements 3.1
    """
    engine = _make_engine()
    z_r = torch.randn(3, 6)
    z_pos = torch.randn(3, 6)
    z_negs = torch.empty(3, 0, 6)
    reliability_ones = torch.ones(3, 0)

    orca = engine.orca_loss(z_r, z_pos, z_negs, reliability_ones)
    standard = engine.standard_infonce(z_r, z_pos, z_negs)

    assert orca.shape == standard.shape == (3,)
    assert torch.allclose(orca, standard, rtol=_RTOL, atol=_ATOL)
