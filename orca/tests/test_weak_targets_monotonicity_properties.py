"""Property-based tests for the monotonicity of ORCA's weak reliability signals.

Feature: orca, Task 2.5 — Property 8: Monotonicity of weak signals.

Design section B.3 defines the two weak reliability signals blended by
``WeakTargetBuilder``:

  * ``r_ont = 1 - exp(-beta * d_ont)`` with ``d_ont = (1 - omega) * d_esco +
    omega * d_isco``. With ``beta >= 0`` and non-negative distances this is a
    **non-decreasing** function of ``d_ont`` — a job that is farther away in
    ontology space is more likely a genuine (reliable) negative.
  * ``r_enc = 1 - sigmoid(gamma * cos)`` on the frozen warmup embeddings. With
    ``gamma >= 0`` this is a **non-increasing** function of the warmup-embedding
    cosine similarity — a job that looks semantically closer is a more
    suspicious (less reliable) negative.

These tests feed Hypothesis-ordered inputs to ``ontology_reliability`` and
``encoder_reliability`` and assert the monotonic relationships hold across
randomized configurations (``omega``, ``beta``, ``gamma``), embedding shapes,
and values.

Validates: Requirements 2.2
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from orca.weak_targets import WeakTargetBuilder

# Tolerance absorbing floating-point rounding in the monotonicity comparisons.
_TOL = 1e-6


def _cfg(omega: float = 0.5, beta: float = 1.0, gamma_enc: float = 5.0) -> SimpleNamespace:
    """Minimal config exposing only the ``orca_*`` fields the builder reads.

    ``WeakTargetBuilder`` reads its fields via ``getattr`` with defaults, so a
    ``SimpleNamespace`` is a faithful, dependency-free stand-in for
    ``TrainingConfig`` here. Only the signal-shaping fields matter for this
    property; the blend weights are left at neutral values.
    """
    return SimpleNamespace(
        orca_omega=omega,
        orca_beta=beta,
        orca_gamma_enc=gamma_enc,
        orca_lambda_ont=0.5,
        orca_lambda_enc=0.5,
        orca_lambda_hist=0.0,
        orca_use_history=False,
    )


# Non-negative ontology distances. Bounded to keep exp(-beta * d_ont) well
# within float range (the exponent is negative, so this never overflows).
_distance = st.floats(
    min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=32
)

# beta must be >= 0 for r_ont to be non-decreasing in d_ont (design B.3).
_beta = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=32)

# omega is the ISCO-vs-ESCO mix in d_ont; a proper convex weight in [0, 1].
_omega = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32)

# gamma must be >= 0 for r_enc to be non-increasing in cosine similarity.
_gamma = st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False, width=32)


# Feature: orca, Property 8: Monotonicity of weak signals (r_ont)
@settings(max_examples=200)
@given(d_onts=st.lists(_distance, min_size=2, max_size=30), omega=_omega, beta=_beta)
def test_r_ont_non_decreasing_in_d_ont(d_onts, omega, beta):
    """r_ont is non-decreasing in the blended ontology distance d_ont.

    Setting ``d_esco == d_isco == v`` makes ``d_ont == v`` for *any* ``omega``
    (the convex blend of equal values is that value), so a Hypothesis-ordered
    list of distances is exactly an ordered list of ``d_ont``. The resulting
    ``r_ont`` sequence must therefore be non-decreasing.

    Validates: Requirements 2.2
    """
    builder = WeakTargetBuilder(_cfg(omega=omega, beta=beta))

    ordered = sorted(d_onts)
    vals = torch.tensor(ordered, dtype=torch.float32)
    r_ont = builder.ontology_reliability(vals, vals)

    # Non-decreasing along ascending d_ont.
    diffs = r_ont[1:] - r_ont[:-1]
    assert torch.all(diffs >= -_TOL), (
        f"r_ont decreased as d_ont increased: min step {diffs.min().item()}"
    )

    # Signal stays a valid reliability in [0, 1].
    assert torch.all(r_ont >= -_TOL) and torch.all(r_ont <= 1.0 + _TOL)


@st.composite
def _embedding_pairs(draw):
    """Draw ``n`` anchor/negative embedding pairs of a shared dimension ``dim``."""
    dim = draw(st.integers(min_value=2, max_value=8))
    n = draw(st.integers(min_value=2, max_value=20))
    comp = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32)
    row = st.lists(comp, min_size=dim, max_size=dim)
    z_r = draw(st.lists(row, min_size=n, max_size=n))
    z_neg = draw(st.lists(row, min_size=n, max_size=n))
    return z_r, z_neg


# Feature: orca, Property 8: Monotonicity of weak signals (r_enc)
@settings(max_examples=200)
@given(pairs=_embedding_pairs(), gamma=_gamma)
def test_r_enc_non_increasing_in_cosine(pairs, gamma):
    """r_enc is non-increasing in warmup-embedding cosine similarity.

    Robust ordering approach (per the task's guidance): generate arbitrary
    embedding pairs, compute their *actual* cosine similarity, sort by that
    cosine, and assert ``r_enc`` is non-increasing along the ascending-cosine
    order — a higher similarity yields a reliability less than or equal to that
    at any lower similarity.

    Validates: Requirements 2.2
    """
    builder = WeakTargetBuilder(_cfg(gamma_enc=gamma))

    z_r = torch.tensor(pairs[0], dtype=torch.float32)
    z_neg = torch.tensor(pairs[1], dtype=torch.float32)

    cos = F.cosine_similarity(z_r, z_neg, dim=-1)
    r_enc = builder.encoder_reliability(z_r, z_neg)

    # Reorder both by ascending cosine similarity, then r_enc must not increase.
    order = torch.argsort(cos, stable=True)
    r_sorted = r_enc[order]
    diffs = r_sorted[1:] - r_sorted[:-1]
    assert torch.all(diffs <= _TOL), (
        f"r_enc increased as cosine increased: max step {diffs.max().item()}"
    )

    # Signal stays a valid reliability in [0, 1].
    assert torch.all(r_enc >= -_TOL) and torch.all(r_enc <= 1.0 + _TOL)
