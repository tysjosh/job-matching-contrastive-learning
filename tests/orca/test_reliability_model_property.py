#!/usr/bin/env python3
"""Property-based tests for :class:`orca.reliability_model.ReliabilityMLP`.

Feature: orca, Task 2.2 — Property 5: ReliabilityMLP output range and shape.

These tests exercise ``ReliabilityMLP.forward`` across Hypothesis-generated
leading shapes, ``embed_dim`` / ``feature_dim`` widths, and tensor values
(including large magnitudes that saturate the final sigmoid) and assert the
output contract from Requirement 1.1:

- the output shape equals ``z_r.shape[:-1]`` (the shared leading dimensions,
  with the trailing ``embed_dim`` removed),
- every output element is finite and within the closed interval ``[0, 1]``, and
- mismatched leading dimensions of ``z_r`` / ``z_j`` / ``scalar_features`` raise
  a ``ValueError`` (Requirement 1.5) rather than returning a tensor.

Validates: Requirements 1.1
"""

from __future__ import annotations

import torch
from hypothesis import given, settings, strategies as st

from orca.reliability_model import ReliabilityMLP


# Bound the total number of elements so generated tensors stay small/fast.
_MAX_LEADING_DIMS = 3
_MAX_DIM_SIZE = 4


@st.composite
def _mlp_inputs(draw):
    """Draw a consistent (ReliabilityMLP, z_r, z_j, scalar_features) tuple.

    Leading dimensions are shared across all three inputs (the valid case).
    Tensor values span a wide magnitude range so the finiteness / range
    guarantees are exercised where the final sigmoid saturates toward 0 or 1.
    ``feature_dim`` may be 0 to cover the ORCA-NoOntology zero-feature layout.
    """
    embed_dim = draw(st.integers(min_value=1, max_value=8))
    feature_dim = draw(st.integers(min_value=0, max_value=6))

    n_leading = draw(st.integers(min_value=0, max_value=_MAX_LEADING_DIMS))
    leading = tuple(
        draw(st.integers(min_value=1, max_value=_MAX_DIM_SIZE)) for _ in range(n_leading)
    )

    # Wide value range including large magnitudes to probe sigmoid saturation.
    value_st = st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=32
    )

    def _tensor(trailing: int) -> torch.Tensor:
        shape = leading + (trailing,)
        n = 1
        for d in shape:
            n *= d
        flat = draw(st.lists(value_st, min_size=n, max_size=n))
        return torch.tensor(flat, dtype=torch.float32).reshape(shape)

    z_r = _tensor(embed_dim)
    z_j = _tensor(embed_dim)
    scalar_features = _tensor(feature_dim)

    model = ReliabilityMLP(embed_dim=embed_dim, feature_dim=feature_dim)
    model.eval()  # disable dropout so the forward pass is deterministic
    return model, z_r, z_j, scalar_features, leading


# Feature: orca, Property 5: ReliabilityMLP output range and shape
@settings(max_examples=200)
@given(_mlp_inputs())
def test_reliability_output_range_and_shape(data):
    """Output is finite, in [0, 1], and shaped as ``z_r.shape[:-1]``.

    Validates: Requirements 1.1
    """
    model, z_r, z_j, scalar_features, leading = data

    with torch.no_grad():
        out = model(z_r, z_j, scalar_features)

    # Requirement 1.1: shape equals the shared leading dims (embed_dim removed).
    assert out.shape == z_r.shape[:-1]
    assert tuple(out.shape) == leading

    # Requirement 1.2: all finite and within [0, 1].
    assert torch.isfinite(out).all()
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


@st.composite
def _mismatched_inputs(draw):
    """Draw inputs whose leading dimensions do NOT all match (Requirement 1.5)."""
    embed_dim = draw(st.integers(min_value=1, max_value=8))
    feature_dim = draw(st.integers(min_value=1, max_value=6))

    # A single shared batch dimension per tensor; force at least one to differ.
    b_r = draw(st.integers(min_value=1, max_value=_MAX_DIM_SIZE))
    b_j = draw(st.integers(min_value=1, max_value=_MAX_DIM_SIZE))
    b_f = draw(st.integers(min_value=1, max_value=_MAX_DIM_SIZE))
    # Reject the all-equal case; only mismatched configurations are of interest.
    if b_r == b_j == b_f:
        b_j = b_j % _MAX_DIM_SIZE + 1  # nudge one dimension so they differ

    z_r = torch.randn(b_r, embed_dim)
    z_j = torch.randn(b_j, embed_dim)
    scalar_features = torch.randn(b_f, feature_dim)

    model = ReliabilityMLP(embed_dim=embed_dim, feature_dim=feature_dim)
    model.eval()
    return model, z_r, z_j, scalar_features


# Feature: orca, Property 5: mismatched leading dims raise (Requirement 1.5)
@settings(max_examples=100)
@given(_mismatched_inputs())
def test_reliability_mismatched_leading_dims_raise(data):
    """Mismatched leading dims raise ValueError and return no tensor.

    Validates: Requirements 1.1
    """
    import pytest

    model, z_r, z_j, scalar_features = data
    with pytest.raises(ValueError):
        model(z_r, z_j, scalar_features)
