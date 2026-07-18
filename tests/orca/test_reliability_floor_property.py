#!/usr/bin/env python3
"""Property-based test for the ORCA reliability floor (Property 2).

Component under test: ``orca.loss_engine.OrcaLossEngine`` (design section B.4).

This test uses Hypothesis to vary the per-negative reliabilities (including
``0``, negative values, and values greater than ``1`` — i.e. inputs *outside*
the ``[0, 1]`` interval), the reliability floor ``r_min``, the temperature, and
the batch/negative shapes and embedding values. It asserts Property 2:

    Property 2 — Reliability floor respected
    For all reliabilities (including 0), each value used in the denominator is
    ``>= r_min`` and ``<= 1.0``; the denominator never drops all negatives.

Concretely, this verifies that ``clamp_reliability`` maps every input into the
closed interval ``[r_min, 1.0]`` (Requirement 3.2), that no negative is ever
fully dropped (every clamped reliability is ``>= r_min > 0``), and that the
reliability-calibrated denominator's negative mass reflects the *clamped*
reliabilities — so a batch whose reliabilities are all ``0`` still contributes
the ``r_min``-weighted negative terms rather than collapsing to the positive
term alone.

**Validates: Requirements 3.2**

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_reliability_floor_property.py
"""

from __future__ import annotations

import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from contrastive_learning.data_structures import TrainingConfig
from orca.loss_engine import OrcaLossEngine


# Tolerance for comparing the engine's negative mass against an independently
# computed reference (absorbs float32 accumulation noise across the sum). The
# reference is built in the engine's exact operation order, so agreement is
# effectively exact; the relative tolerance is kept slightly above float32 eps
# only to stay robust to benign per-platform accumulation ordering.
_RTOL = 1e-4
_ATOL = 1e-6

_finite = dict(allow_nan=False, allow_infinity=False)


def _make_engine(*, r_min: float, temperature: float) -> OrcaLossEngine:
    """Construct a minimal ORCA-Denominator engine.

    Only ``clamp_reliability`` / ``orca_denominator`` are exercised here, so the
    reliability model and weak-target builder are unused and passed as ``None``.
    """
    cfg = TrainingConfig(
        orca_enabled=True,
        orca_variant="denominator",
        orca_r_min=r_min,
        temperature=temperature,
    )
    return OrcaLossEngine(cfg, skill_matcher=None, reliability_model=None,
                          weak_builder=None)


@st.composite
def _floor_cases(draw) -> dict:
    """Draw a reliability-floor test case.

    Reliabilities deliberately span *outside* ``[0, 1]`` — including exact ``0``,
    negatives, and values ``> 1`` — so the clamp is probed at and beyond both
    bounds. Shapes vary over batch size ``B`` and negative count ``K``.
    """
    B = draw(st.integers(min_value=1, max_value=4))
    K = draw(st.integers(min_value=1, max_value=6))
    D = draw(st.integers(min_value=1, max_value=8))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    # r_min strictly in (0, 1) so the "no negative fully dropped" clause has a
    # positive floor to assert against (the default is 0.05).
    r_min = draw(st.floats(min_value=1e-4, max_value=0.9, **_finite))
    temperature = draw(st.floats(min_value=1e-2, max_value=2.0, **_finite))

    # Reliabilities spanning well outside [0, 1], with 0 always representable.
    rel_value = st.floats(min_value=-5.0, max_value=5.0, **_finite)
    n = B * K
    rel_flat = draw(st.lists(rel_value, min_size=n, max_size=n))

    return {
        "B": B, "K": K, "D": D, "seed": seed,
        "r_min": r_min, "temperature": temperature,
        "rel_flat": rel_flat,
    }


def _make_embeddings(case: dict):
    """Reproducible anchor / positive / negative embeddings for a case."""
    g = torch.Generator().manual_seed(case["seed"])
    B, K, D = case["B"], case["K"], case["D"]
    z_r = torch.randn(B, D, generator=g)
    z_pos = torch.randn(B, D, generator=g)
    z_negs = torch.randn(B, K, D, generator=g)
    return z_r, z_pos, z_negs


# ---------------------------------------------------------------------------
# Property 2 — every value entering the denominator is in [r_min, 1.0]
# ---------------------------------------------------------------------------

@settings(max_examples=300, deadline=None)
@given(case=_floor_cases())
def test_clamp_bounds_and_no_negative_fully_dropped(case):
    """Property 2 (Validates: Requirements 3.2).

    ``clamp_reliability`` maps every reliability — including 0, negatives, and
    values above 1 — into ``[r_min, 1.0]``, and every clamped value is
    ``>= r_min > 0`` so no negative is ever fully dropped.
    """
    engine = _make_engine(r_min=case["r_min"], temperature=case["temperature"])
    reliability = torch.tensor(case["rel_flat"], dtype=torch.float32).reshape(
        case["B"], case["K"]
    )

    clamped = engine.clamp_reliability(reliability)

    # Every clamped value lies within the closed interval [r_min, 1.0].
    assert torch.isfinite(clamped).all()
    assert (clamped >= case["r_min"]).all()
    assert (clamped <= 1.0).all()

    # No negative is fully dropped: the floor is strictly positive.
    assert case["r_min"] > 0.0
    assert (clamped >= case["r_min"]).all()

    # Where the raw reliability was <= r_min (e.g. 0 or negative), the clamp
    # must lift it exactly to the floor, proving the floor is actually applied.
    below = reliability <= case["r_min"]
    if below.any():
        assert torch.allclose(
            clamped[below], torch.full_like(clamped[below], case["r_min"])
        )


# ---------------------------------------------------------------------------
# Property 2 (denominator clause) — negative mass reflects the CLAMPED values
# ---------------------------------------------------------------------------

@settings(max_examples=300, deadline=None)
@given(case=_floor_cases())
@example(
    # Pinned regression: tiny temperature + r_min < 1 + all-zero reliabilities.
    # exp(sim/tau) can underflow to (near-)0 in float32; weighting by r_min then
    # drives subnormals to exactly 0.0. This must NOT read as the floor dropping
    # a negative (see the substantial-mass guard below).
    case={
        "B": 1, "K": 3, "D": 1, "seed": 0,
        "r_min": 0.125, "temperature": 0.01,
        "rel_flat": [0.0, 0.0, 0.0],
    }
)
def test_denominator_negative_mass_uses_clamped_reliabilities(case):
    """Property 2 (Validates: Requirements 3.2).

    The reliability-calibrated denominator's negative mass equals the sum over
    negatives of ``clamp(reliability_k, r_min, 1.0) * exp(sim_k / tau)`` — so a
    batch with all-zero reliabilities still includes the ``r_min``-weighted
    negative terms rather than collapsing to the positive term alone.
    """
    engine = _make_engine(r_min=case["r_min"], temperature=case["temperature"])
    z_r, z_pos, z_negs = _make_embeddings(case)
    reliability = torch.tensor(case["rel_flat"], dtype=torch.float32).reshape(
        case["B"], case["K"]
    )

    pos, denom = engine.orca_denominator(z_r, z_pos, z_negs, reliability)

    # Independently reconstruct the clamped negative mass using the SAME clamps
    # the engine applies.
    clamped = torch.clamp(reliability, case["r_min"], 1.0)
    neg_sim = (z_r.unsqueeze(-2) * z_negs).sum(-1) / case["temperature"]
    neg_sim = torch.clamp(neg_sim, max=engine.max_exp)
    expected_neg = (clamped * torch.exp(neg_sim)).sum(-1)

    # Compare the denominator against ``pos + expected_neg`` directly. We add
    # (rather than recovering the mass via ``denom - pos``) because when the
    # positive term dominates, the subtraction suffers catastrophic float32
    # cancellation while the engine's own addition does not.
    assert torch.allclose(denom, pos + expected_neg, rtol=_RTOL, atol=_ATOL)

    # No negative is fully dropped BY THE FLOOR: every negative's weight is the
    # clamped reliability, which is ``>= r_min > 0`` (asserted directly on the
    # per-negative weights below). Hence wherever a negative carries
    # non-negligible ``exp(sim/tau)`` mass, the reliability-weighted mass is
    # strictly positive too.
    #
    # We guard on the *per-negative* raw exp mass exceeding a small epsilon
    # rather than on the unweighted row sum being ``> 0``. ``exp`` of a very
    # negative similarity (tiny temperature) can legitimately underflow to
    # exactly 0.0 — or to a subnormal that, once multiplied by ``r_min < 1``,
    # underflows the rest of the way to 0.0. That is float32 underflow, not the
    # floor dropping a negative. The epsilon (1e-30) sits far enough above the
    # smallest float32 subnormal (~1.4e-45) that weighting by any admissible
    # ``r_min`` (>= 1e-4) cannot underflow such a term to 0.0, so the guard
    # isolates exactly the terms whose weighted mass must remain strictly
    # positive.
    raw_neg_exp = torch.exp(neg_sim)
    substantial = (raw_neg_exp > 1e-30).any(dim=-1)
    assert (expected_neg[substantial] > 0.0).all()

    # The weight applied to each negative is the clamped reliability, floored at
    # r_min (this is the direct content of Requirement 3.2 at the denominator).
    assert (clamped >= case["r_min"]).all()
    assert (clamped <= 1.0).all()


def test_all_zero_reliabilities_still_contribute_negative_mass():
    """Focused edge case: all-zero reliabilities keep the r_min-weighted mass.

    Complements the property tests above by pinning the "no negative fully
    dropped" clause on the exact ``reliability == 0`` boundary.
    """
    engine = _make_engine(r_min=0.05, temperature=0.1)
    torch.manual_seed(0)
    z_r = torch.randn(3, 4)
    z_pos = torch.randn(3, 4)
    z_negs = torch.randn(3, 5, 4)
    reliability = torch.zeros(3, 5)

    pos, denom = engine.orca_denominator(z_r, z_pos, z_negs, reliability)

    # The negative mass equals r_min * sum_k exp(sim_k / tau) and is strictly
    # positive — every negative retains its r_min weight rather than being
    # dropped when its reliability is exactly 0.
    neg_sim = torch.clamp((z_r.unsqueeze(-2) * z_negs).sum(-1) / 0.1, max=engine.max_exp)
    expected_neg = (0.05 * torch.exp(neg_sim)).sum(-1)
    assert (expected_neg > 0.0).all()

    # The denominator is the positive term plus that r_min-weighted mass.
    assert torch.allclose(denom, pos + expected_neg, rtol=_RTOL, atol=_ATOL)


def test_underflowing_negatives_are_not_mistaken_for_a_dropped_floor():
    """Focused counterexample: tiny temperature drives ``exp(sim/tau)`` to 0.

    Pins the exact float32 edge case (temperature=0.01, r_min=0.125, all-zero
    reliabilities) behind the property test's counterexample: when the anchor is
    strongly *dissimilar* from its negatives, ``sim/tau`` falls far below zero
    and ``exp`` underflows to exactly 0.0 in float32. This is legitimate
    numerical underflow — the reliability floor still applies its ``r_min``
    weight — so the reference math (built in the engine's exact operation order)
    must agree with the engine, and the "no negative fully dropped" guard must
    not fire for terms whose raw mass has genuinely underflowed.
    """
    r_min, temperature = 0.125, 0.01
    engine = _make_engine(r_min=r_min, temperature=temperature)

    # Unit-norm anchor pointing opposite to each negative => strongly negative
    # similarities, which /tau pushes well below the underflow threshold.
    z_r = torch.tensor([[1.0, 0.0, 0.0]])
    z_pos = torch.tensor([[1.0, 0.0, 0.0]])
    z_negs = torch.tensor([[[-5.0, 0.0, 0.0],
                            [-6.0, 0.0, 0.0],
                            [-7.0, 0.0, 0.0]]])
    reliability = torch.zeros(1, 3)

    pos, denom = engine.orca_denominator(z_r, z_pos, z_negs, reliability)

    clamped = torch.clamp(reliability, r_min, 1.0)
    neg_sim = (z_r.unsqueeze(-2) * z_negs).sum(-1) / temperature
    neg_sim = torch.clamp(neg_sim, max=engine.max_exp)
    expected_neg = (clamped * torch.exp(neg_sim)).sum(-1)

    # The similarities genuinely underflow exp() to 0.0 in float32.
    assert (torch.exp(neg_sim) == 0.0).all()
    assert (expected_neg == 0.0).all()

    # Reference (engine order) still matches the engine: denom == pos here.
    assert torch.allclose(denom, pos + expected_neg, rtol=_RTOL, atol=_ATOL)

    # The floor is still applied to every negative's weight; underflow of the
    # raw exp mass is not the floor dropping a negative.
    assert (clamped >= r_min).all()
    assert (clamped <= 1.0).all()
