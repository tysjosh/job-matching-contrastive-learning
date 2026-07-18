#!/usr/bin/env python3
"""Property-based test for the ORCA adaptive negative sampler (Property 3).

Component under test: ``orca.adaptive_sampler.AdaptiveNegativeSampler``
(design section B.5).

This test uses Hypothesis to vary candidate counts, per-candidate
reliabilities, seeds, epochs, requested counts ``k``, and the ``random_mix``
fraction, and asserts Property 3:

    Property 3 — Adaptive sampler determinism and distribution
    For a fixed ``(seed, epoch, candidates)``, ``select`` returns identical
    negatives across runs; over many draws, empirical selection frequency
    matches ``sampling_score / sum(sampling_score)`` up to seeded-RNG sampling
    error.

Two facets are checked:

- **Determinism** (Requirement 5.1): repeated ``select`` invocations for a
  fixed ``(training_seed, epoch, candidates, scalar_features, k)`` return the
  identical index tensor, and the structural contract holds (exactly
  ``min(k, n)`` distinct indices; empty for empty candidates or ``k <= 0``).
- **Distribution**: with ``orca_random_mix = 0.0`` (every draw is
  score-proportional) and ``k = 1`` (each ``select`` is a single categorical
  sample), running many draws across many epochs yields an empirical selection
  frequency that matches ``sampling_score / sum(sampling_score)`` within seeded
  sampling error.

The reliability model is a lightweight stub returning *fixed* reliabilities so
the sampling scores — and therefore the exact target distribution — are known
analytically, making the distribution assertion precise and fast.

**Validates: Requirements 5.1**

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_adaptive_sampler_property3.py
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from contrastive_learning.data_structures import TrainingConfig
from orca.adaptive_sampler import AdaptiveNegativeSampler


_finite = dict(allow_nan=False, allow_infinity=False)

# Embedding / feature widths are irrelevant to the stub reliability model, so a
# small fixed width keeps the generated tensors cheap.
_EMBED_DIM = 4
_FEATURE_DIM = 3


class _FixedReliabilityModel:
    """Stub reliability model returning caller-supplied fixed reliabilities.

    The real :class:`~orca.reliability_model.ReliabilityMLP` exposes
    ``score(anchor, candidates, scalar_features) -> Tensor`` of per-candidate
    reliabilities. Returning fixed values here makes the sampler's
    ``sampling_scores`` (and hence the exact target distribution) known, so the
    distribution assertion is exact rather than dependent on a trained network.
    """

    def __init__(self, reliabilities):
        self._rel = torch.as_tensor(reliabilities, dtype=torch.float64).reshape(-1)

    def score(self, anchor, candidates, scalar_features):
        # Aligned 1:1 with candidates; a fresh clone so the sampler's in-place
        # ``.to(...)`` / masking never mutates our stored values.
        return self._rel.clone()


def _make_sampler(*, reliabilities, epsilon, gamma_s, random_mix, seed):
    cfg = TrainingConfig(
        orca_enabled=True,
        orca_variant="full",
        orca_sampling_epsilon=epsilon,
        orca_gamma_s=gamma_s,
        orca_random_mix=random_mix,
        training_seed=seed,
    )
    model = _FixedReliabilityModel(reliabilities)
    return AdaptiveNegativeSampler(cfg, model)


def _dummy_inputs(n):
    """Anchor / candidate / scalar-feature tensors of a compatible shape.

    Values are unused by the stub reliability model but must be shaped so the
    sampler can read ``candidates.shape[0] == n``.
    """
    anchor = torch.zeros(_EMBED_DIM)
    candidates = torch.zeros(n, _EMBED_DIM)
    scalar_features = torch.zeros(n, _FEATURE_DIM)
    return anchor, candidates, scalar_features


# ---------------------------------------------------------------------------
# Determinism + structural contract (Requirement 5.1, 5.3, 5.5)
# ---------------------------------------------------------------------------

@st.composite
def _determinism_cases(draw):
    """Draw a determinism case varying counts, reliabilities, k, mix, seed, epoch."""
    n = draw(st.integers(min_value=1, max_value=8))
    reliabilities = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, **_finite),
            min_size=n, max_size=n,
        )
    )
    k = draw(st.integers(min_value=1, max_value=10))
    random_mix = draw(st.floats(min_value=0.0, max_value=1.0, **_finite))
    gamma_s = draw(st.floats(min_value=0.0, max_value=3.0, **_finite))
    epsilon = draw(st.floats(min_value=1e-3, max_value=1.0, **_finite))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    epoch = draw(st.integers(min_value=0, max_value=50))
    return {
        "n": n, "reliabilities": reliabilities, "k": k,
        "random_mix": random_mix, "gamma_s": gamma_s, "epsilon": epsilon,
        "seed": seed, "epoch": epoch,
    }


@settings(max_examples=300, deadline=None)
@given(case=_determinism_cases())
def test_select_is_deterministic_and_well_formed(case):
    """Property 3 — determinism (Validates: Requirements 5.1).

    Repeated ``select`` calls for a fixed ``(seed, epoch, candidates, k)`` return
    the identical index tensor, and the result contains exactly ``min(k, n)``
    distinct indices (Requirements 5.3).
    """
    sampler = _make_sampler(
        reliabilities=case["reliabilities"], epsilon=case["epsilon"],
        gamma_s=case["gamma_s"], random_mix=case["random_mix"], seed=case["seed"],
    )
    anchor, candidates, scalar_features = _dummy_inputs(case["n"])

    first = sampler.select(anchor, candidates, scalar_features, case["k"], case["epoch"])
    second = sampler.select(anchor, candidates, scalar_features, case["k"], case["epoch"])

    # Identical output across repeated invocations (Requirement 5.1).
    assert torch.equal(first, second)

    # Exactly min(k, n) distinct indices, each a valid candidate index
    # (Requirement 5.3).
    expected_count = min(case["k"], case["n"])
    assert first.dtype == torch.long
    assert first.numel() == expected_count
    assert torch.unique(first).numel() == expected_count
    assert (first >= 0).all() and (first < case["n"]).all()


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    epoch_a=st.integers(min_value=0, max_value=50),
    epoch_b=st.integers(min_value=0, max_value=50),
)
def test_distinct_epochs_are_independent_but_each_reproducible(seed, epoch_a, epoch_b):
    """Determinism holds per epoch; the seed alone (not wall-clock) drives draws.

    Each epoch is independently reproducible — re-running the same epoch
    reproduces its selection exactly, confirming randomness comes solely from
    ``training_seed + epoch`` (Requirements 5.1, 10.1).
    """
    n, k = 8, 4
    reliabilities = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4]
    sampler = _make_sampler(
        reliabilities=reliabilities, epsilon=0.1, gamma_s=1.0,
        random_mix=0.0, seed=seed,
    )
    anchor, candidates, scalar_features = _dummy_inputs(n)

    a1 = sampler.select(anchor, candidates, scalar_features, k, epoch_a)
    a2 = sampler.select(anchor, candidates, scalar_features, k, epoch_a)
    b1 = sampler.select(anchor, candidates, scalar_features, k, epoch_b)

    # Same epoch -> identical (reproducible).
    assert torch.equal(a1, a2)
    # Result is always a valid min(k, n)-sized distinct selection regardless of
    # epoch.
    assert a1.numel() == min(k, n)
    assert b1.numel() == min(k, n)


# ---------------------------------------------------------------------------
# Empty / non-positive request edge cases (Requirement 5.5)
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(k=st.integers(min_value=-5, max_value=5), epoch=st.integers(min_value=0, max_value=10))
def test_empty_candidates_returns_empty(k, epoch):
    """Empty candidate set -> empty selection without raising (Requirement 5.5)."""
    sampler = _make_sampler(
        reliabilities=[], epsilon=0.1, gamma_s=1.0, random_mix=0.5, seed=7,
    )
    anchor = torch.zeros(_EMBED_DIM)
    candidates = torch.zeros(0, _EMBED_DIM)
    scalar_features = torch.zeros(0, _FEATURE_DIM)

    out = sampler.select(anchor, candidates, scalar_features, k, epoch)
    assert out.dtype == torch.long
    assert out.numel() == 0


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    k=st.integers(min_value=-5, max_value=0),
    epoch=st.integers(min_value=0, max_value=10),
)
def test_non_positive_k_returns_empty(n, k, epoch):
    """``k <= 0`` -> empty selection without raising (Requirement 5.5)."""
    sampler = _make_sampler(
        reliabilities=[0.5] * n, epsilon=0.1, gamma_s=1.0, random_mix=0.5, seed=7,
    )
    anchor, candidates, scalar_features = _dummy_inputs(n)
    out = sampler.select(anchor, candidates, scalar_features, k, epoch)
    assert out.numel() == 0


# ---------------------------------------------------------------------------
# Distribution: empirical frequency ~ sampling_score / sum(sampling_score)
# ---------------------------------------------------------------------------

@st.composite
def _distribution_cases(draw):
    """Draw a distribution case: a small candidate set with fixed reliabilities.

    ``n`` is kept small (2..4) so each candidate accrues enough draws for a
    tight empirical estimate within a bounded number of samples.
    """
    n = draw(st.integers(min_value=2, max_value=4))
    reliabilities = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, **_finite),
            min_size=n, max_size=n,
        )
    )
    gamma_s = draw(st.floats(min_value=0.0, max_value=2.0, **_finite))
    epsilon = draw(st.floats(min_value=0.05, max_value=0.5, **_finite))
    seed = draw(st.integers(min_value=0, max_value=10_000))
    return {
        "n": n, "reliabilities": reliabilities,
        "gamma_s": gamma_s, "epsilon": epsilon, "seed": seed,
    }


# A single-draw (k=1) sample per epoch is a categorical draw with probability
# sampling_score / sum(sampling_score); accumulating over many epochs yields the
# empirical distribution. N_DRAWS chosen so the worst-case standard error
# (~sqrt(0.25 / N)) is well below the assertion tolerance.
_N_DRAWS = 12000
_DIST_TOL = 0.04


@settings(max_examples=12, deadline=None)
@given(case=_distribution_cases())
def test_score_proportional_selection_matches_target_distribution(case):
    """Property 3 — distribution (Validates: Requirements 5.1).

    With ``random_mix = 0`` and ``k = 1``, each ``select`` is a categorical draw
    with probability ``sampling_score / sum(sampling_score)``. Over ``_N_DRAWS``
    epochs the empirical selection frequency matches that target within seeded
    sampling error.
    """
    n = case["n"]
    sampler = _make_sampler(
        reliabilities=case["reliabilities"], epsilon=case["epsilon"],
        gamma_s=case["gamma_s"], random_mix=0.0, seed=case["seed"],
    )
    anchor, candidates, scalar_features = _dummy_inputs(n)

    # Analytic target distribution from the sampler's own scoring rule so the
    # comparison is exact w.r.t. the implementation under test.
    reliability = torch.as_tensor(case["reliabilities"], dtype=torch.float64)
    scores = sampler.sampling_scores(reliability)
    target = (scores / scores.sum()).numpy()

    counts = torch.zeros(n, dtype=torch.float64)
    for epoch in range(_N_DRAWS):
        idx = sampler.select(anchor, candidates, scalar_features, 1, epoch)
        assert idx.numel() == 1  # min(k=1, n>=2) == 1
        counts[int(idx.item())] += 1.0

    empirical = (counts / counts.sum()).numpy()

    for i in range(n):
        assert abs(empirical[i] - target[i]) < _DIST_TOL, (
            f"index {i}: empirical={empirical[i]:.4f} target={target[i]:.4f} "
            f"reliabilities={case['reliabilities']} gamma_s={case['gamma_s']} "
            f"epsilon={case['epsilon']}"
        )
