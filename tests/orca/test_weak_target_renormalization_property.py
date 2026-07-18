"""Property-based test for ORCA WeakTargetBuilder renormalization (Property 4).

Component under test: ``orca.weak_targets.WeakTargetBuilder`` (design section B.3).

This test uses Hypothesis to vary the weak-target blend weights
(``orca_lambda_ont/enc/hist``), the signal sharpness knobs (``orca_omega``,
``orca_beta``, ``orca_gamma_enc``), the input tensor shapes/values (ontology
distances ``d_esco``/``d_isco`` and warmup embeddings ``z_r``/``z_neg``), and
*which* reliability signals are present for the batch. It asserts Property 4:

    Property 4 — Weak-target renormalization
    For all subsets of present signals, the active blend weights sum to
    ``1.0`` (within ``1e-6``) and every ``r_tilde`` lies in ``[0, 1]``; and
    dropping the history signal on a static dataset yields the same
    ``r_tilde`` as setting ``orca_lambda_hist = 0`` and renormalizing over the
    ontology and encoder signals.

**Validates: Requirements 2.1**

At the MVP milestone ``history_reliability`` is stubbed to ``None`` (the history
signal is never present), so the present-signal subsets exercised here are
``{}``, ``{ont}``, ``{enc}``, and ``{ont, enc}``.

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_weak_target_renormalization_property.py
"""

from __future__ import annotations

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from contrastive_learning.data_structures import TrainingConfig
from orca.weak_targets import NEUTRAL_R_TILDE, WEIGHT_SUM_TOL, WeakTargetBuilder


# Comparison tolerance for r_tilde equality against the independently-computed
# convex combination (a hair looser than WEIGHT_SUM_TOL to absorb float32 noise
# accumulated across the blend).
_RTOL = 1e-5
_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Generation helpers
#
# We draw the blend weights, sharpness knobs, tensor shapes, and a torch seed
# (so the ontology distances and warmup embeddings vary in both shape and
# value while staying reproducible), plus two booleans choosing which signals
# are present for the batch.
# ---------------------------------------------------------------------------

_finite = dict(allow_nan=False, allow_infinity=False)


@st.composite
def weak_target_cases(draw) -> dict:
    return {
        "B": draw(st.integers(min_value=1, max_value=4)),
        "K": draw(st.integers(min_value=1, max_value=5)),
        "embed_dim": draw(st.integers(min_value=2, max_value=8)),
        "seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
        # Which signals are present. History is stubbed to None at the MVP, so
        # its presence is driven purely by the (ont, enc) selection here.
        "include_ont": draw(st.booleans()),
        "include_enc": draw(st.booleans()),
        # Blend weights — non-negative, occasionally all-zero to exercise the
        # equal-split fallback.
        "lambda_ont": draw(st.floats(min_value=0.0, max_value=10.0, **_finite)),
        "lambda_enc": draw(st.floats(min_value=0.0, max_value=10.0, **_finite)),
        "lambda_hist": draw(st.floats(min_value=0.0, max_value=10.0, **_finite)),
        # Sharpness knobs.
        "omega": draw(st.floats(min_value=0.0, max_value=1.0, **_finite)),
        "beta": draw(st.floats(min_value=0.0, max_value=5.0, **_finite)),
        "gamma": draw(st.floats(min_value=0.0, max_value=10.0, **_finite)),
    }


def _make_builder(case: dict, *, use_history: bool = False,
                  lambda_hist: float | None = None) -> WeakTargetBuilder:
    cfg = TrainingConfig(
        orca_enabled=True,
        orca_variant="denominator",
        orca_omega=case["omega"],
        orca_beta=case["beta"],
        orca_gamma_enc=case["gamma"],
        orca_lambda_ont=case["lambda_ont"],
        orca_lambda_enc=case["lambda_enc"],
        orca_lambda_hist=case["lambda_hist"] if lambda_hist is None else lambda_hist,
        orca_use_history=use_history,
    )
    return WeakTargetBuilder(cfg)


def _make_inputs(case: dict):
    """Build reproducible ontology distances and warmup embeddings for a case."""
    g = torch.Generator().manual_seed(case["seed"])
    B, K, D = case["B"], case["K"], case["embed_dim"]
    # Ontology distances are non-negative.
    d_esco = torch.rand(B, K, generator=g) * 5.0
    d_isco = torch.rand(B, K, generator=g) * 5.0
    z_r = torch.randn(B, K, D, generator=g)
    z_neg = torch.randn(B, K, D, generator=g)
    return d_esco, d_isco, z_r, z_neg


def _active_weights(case: dict, present) -> torch.Tensor:
    """The renormalized active blend weights the builder should use.

    Computed in float32 to match the builder's tensor dtype exactly — the
    blend weights can span a wide dynamic range, and float64 reference math
    would diverge from the float32 implementation near the underflow boundary
    (e.g. a lambda that underflows to 0 in float32 flips the equal-split
    fallback). Mirroring the dtype keeps the invariant check faithful.
    """
    lambdas = {"ont": case["lambda_ont"], "enc": case["lambda_enc"]}
    w = torch.tensor([lambdas[s] for s in present], dtype=torch.float32)
    total = w.sum()
    if total <= 0.0:
        return torch.ones_like(w) / w.numel()
    return w / total


# ---------------------------------------------------------------------------
# Property 4 — active weights sum to 1.0 and r_tilde in [0, 1]
# ---------------------------------------------------------------------------

@settings(max_examples=300, deadline=None)
@given(case=weak_target_cases())
def test_active_weights_sum_to_one_and_r_tilde_in_unit_interval(case):
    """Property 4 (Validates: Requirements 2.1).

    For every subset of present signals, the active blend weights the builder
    applies sum to ``1.0`` (within ``1e-6``) and every ``r_tilde`` value lies
    in ``[0, 1]``. We verify the weight-sum invariant *independently* by
    reconstructing the renormalized active weights and asserting that the
    builder's output equals the corresponding convex combination of the
    present signal reliabilities.
    """
    builder = _make_builder(case, use_history=False)
    d_esco, d_isco, z_r, z_neg = _make_inputs(case)

    kwargs = {}
    present = []
    if case["include_ont"]:
        kwargs.update(d_esco=d_esco, d_isco=d_isco)
        present.append("ont")
    if case["include_enc"]:
        kwargs.update(z_r_warmup=z_r, z_neg_warmup=z_neg)
        present.append("enc")

    r_tilde, signals_present = builder.build_targets(**kwargs)

    # The builder must report exactly the signals we supplied.
    assert signals_present["ont"] is bool(case["include_ont"])
    assert signals_present["enc"] is bool(case["include_enc"])
    assert signals_present["hist"] is False  # history stubbed off at MVP

    # r_tilde is always within the unit interval and finite.
    assert torch.isfinite(r_tilde).all()
    assert (r_tilde >= 0.0).all() and (r_tilde <= 1.0).all()

    if not present:
        # No signal present -> neutral default, no renormalization (Req 2.5).
        assert torch.allclose(r_tilde, torch.tensor(NEUTRAL_R_TILDE))
        return

    # Independently reconstruct the active weights and assert they sum to 1.0.
    weights = _active_weights(case, present)
    assert math.isclose(float(weights.sum()), 1.0, abs_tol=WEIGHT_SUM_TOL)

    # And assert the builder produced exactly that convex combination.
    terms = []
    if "ont" in present:
        terms.append(builder.ontology_reliability(d_esco, d_isco))
    if "enc" in present:
        terms.append(builder.encoder_reliability(z_r, z_neg))
    expected = sum(w * t for w, t in zip(weights.tolist(), terms))
    expected = torch.clamp(expected, 0.0, 1.0)
    assert torch.allclose(r_tilde, expected, rtol=_RTOL, atol=_ATOL)


# ---------------------------------------------------------------------------
# Property 4 (second clause) — dropping history == lambda_hist=0 + renormalize
# ---------------------------------------------------------------------------

@settings(max_examples=300, deadline=None)
@given(case=weak_target_cases())
def test_dropping_history_equals_lambda_hist_zero_and_renormalize(case):
    """Property 4 (Validates: Requirements 2.1, cf. Requirement 2.4).

    On a static dataset (no interaction history) with ``orca_use_history`` off,
    ``r_tilde`` is identical to the result of explicitly setting
    ``orca_lambda_hist = 0`` and renormalizing over the ontology and encoder
    signals — regardless of the configured ``orca_lambda_hist`` value.
    """
    d_esco, d_isco, z_r, z_neg = _make_inputs(case)

    # Both ontology and encoder signals present, history absent (static data).
    build_kwargs = dict(
        d_esco=d_esco, d_isco=d_isco, z_r_warmup=z_r, z_neg_warmup=z_neg,
    )

    # History dropped via use_history=False, keeping the configured lambda_hist.
    dropped = _make_builder(case, use_history=False).build(**build_kwargs)

    # Explicit lambda_hist=0 renormalized over ontology+encoder.
    zeroed = _make_builder(case, use_history=False, lambda_hist=0.0).build(**build_kwargs)

    assert torch.allclose(dropped, zeroed, rtol=_RTOL, atol=_ATOL)


# ---------------------------------------------------------------------------
# Focused edge case — no signal present yields the neutral default (Req 2.5),
# complementing the property tests above.
# ---------------------------------------------------------------------------

def test_no_signal_present_returns_neutral_default():
    """With neither ontology nor encoder inputs, r_tilde is the neutral 0.5."""
    builder = _make_builder(
        {"omega": 0.5, "beta": 1.0, "gamma": 5.0,
         "lambda_ont": 0.5, "lambda_enc": 0.5, "lambda_hist": 0.0},
        use_history=False,
    )
    r_tilde, signals_present = builder.build_targets()
    assert signals_present == {"ont": False, "enc": False, "hist": False}
    assert torch.allclose(r_tilde, torch.tensor(NEUTRAL_R_TILDE))
