#!/usr/bin/env python3
"""Property-based test for ORCA variant application site (Property 6).

Component under test: ``orca.loss_engine.OrcaLossEngine.compute_loss`` variant
branching (design section B.4, task 6.1), specifically the headline contrast
between the ``external_weight`` and ``denominator`` variants.

    Property 6 — ORCA-ExternalWeight vs ORCA-Denominator differ only in
    application site.
    Given identical batches, seeds, inputs, and predicted reliabilities, the two
    variants apply reliability only at their variant-specific location:
      * ``external_weight`` -> ``mean(reliability) * Standard_InfoNCE`` (a scalar
        multiplier on the final loss), and
      * ``denominator``     -> ``orca_loss`` (reliability inside the
        Reliability_Calibrated_Denominator).
    Both variants produce identical ``r_tilde`` weak targets and identical
    ReliabilityMLP predictions within floating-point tolerance, so the assembled
    totals differ ONLY by where reliability enters.

**Validates: Requirements 6.1**

Construction note (per the task): the two engines are built *directly* (not via
``orca.factory.make_loss_engine``) so the ``external_weight`` and ``denominator``
variants can be exercised freely without the MVP/variant factory guard. Both
engines share the SAME ``ReliabilityMLP`` instance (in ``eval`` mode so dropout
is disabled and predictions are deterministic) and the SAME
``WeakTargetBuilder``, guaranteeing identical reliability predictions and
identical ``r_tilde`` on identical inputs.

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_external_weight_vs_denominator_property6.py
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from contrastive_learning.data_structures import TrainingConfig
from orca.loss_engine import OrcaLossEngine
from orca.reliability_model import ReliabilityMLP
from orca.weak_targets import WeakTargetBuilder


_RTOL = 1e-5
_ATOL = 1e-6


def _make_paired_engines(embed_dim: int, feature_dim: int, eta_rel: float):
    """Build ``(external_weight_engine, denominator_engine)`` sharing state.

    Both engines are constructed directly with the SAME ``ReliabilityMLP``
    (``eval`` mode -> deterministic, dropout off) and the SAME
    ``WeakTargetBuilder``, so on identical inputs they emit identical reliability
    predictions and identical ``r_tilde`` (Requirement 6.3). Only ``orca_variant``
    differs between the two configs.
    """
    torch.manual_seed(0)
    shared_model = ReliabilityMLP(embed_dim=embed_dim, feature_dim=feature_dim)
    shared_model.eval()  # disable dropout -> deterministic predictions

    def _cfg(variant: str) -> TrainingConfig:
        return TrainingConfig(
            orca_enabled=True,
            orca_variant=variant,
            orca_eta_rel=eta_rel,
            orca_use_alignment=False,
        )

    shared_builder = WeakTargetBuilder(_cfg("denominator"))

    ext_engine = OrcaLossEngine(
        _cfg("external_weight"), skill_matcher=None,
        reliability_model=shared_model, weak_builder=shared_builder,
    )
    denom_engine = OrcaLossEngine(
        _cfg("denominator"), skill_matcher=None,
        reliability_model=shared_model, weak_builder=shared_builder,
    )
    return ext_engine, denom_engine


@st.composite
def _variant_inputs(draw):
    """Draw a full batch plus the shared engine dimensions.

    Varies the batch size ``B``, the number of negatives ``K`` (``>= 1`` so the
    reliability tensor and its mean are well-defined), the embedding dim ``D``,
    the ontology feature dim ``F`` (``0`` covers the NoOntology-style zero-width
    layout), the reliability BCE weight ``eta_rel``, and all tensor values.
    """
    B = draw(st.integers(min_value=1, max_value=4))
    K = draw(st.integers(min_value=1, max_value=5))
    D = draw(st.integers(min_value=1, max_value=8))
    F = draw(st.integers(min_value=0, max_value=6))
    eta_rel = draw(st.floats(min_value=0.0, max_value=2.0, width=32))

    # Moderate embedding magnitudes keep the InfoNCE math finite; the shared
    # clamps behave identically on both paths regardless.
    value_st = st.floats(
        min_value=-10.0, max_value=10.0,
        allow_nan=False, allow_infinity=False, width=32,
    )
    # Ontology distances are non-negative by construction (design B.3).
    dist_st = st.floats(
        min_value=0.0, max_value=5.0,
        allow_nan=False, allow_infinity=False, width=32,
    )

    def _tensor(strategy, *shape):
        n = 1
        for d in shape:
            n *= d
        if n == 0:
            return torch.zeros(shape, dtype=torch.float32)
        flat = draw(st.lists(strategy, min_size=n, max_size=n))
        return torch.tensor(flat, dtype=torch.float32).reshape(shape)

    z_r = _tensor(value_st, B, D)
    z_pos = _tensor(value_st, B, D)
    z_negs = _tensor(value_st, B, K, D)
    scalar_features = _tensor(value_st, B, K, F)
    d_esco = _tensor(dist_st, B, K)
    d_isco = _tensor(dist_st, B, K)

    return {
        "z_r": z_r, "z_pos": z_pos, "z_negs": z_negs,
        "scalar_features": scalar_features,
        "d_esco": d_esco, "d_isco": d_isco,
        "embed_dim": D, "feature_dim": F, "eta_rel": eta_rel,
    }


# Feature: orca, Property 6: ExternalWeight vs Denominator application site
@settings(max_examples=200, deadline=None)
@given(_variant_inputs())
def test_external_weight_vs_denominator_application_site(data):
    """Both variants apply reliability only at their site; supervision is shared.

    Validates: Requirements 6.1
    """
    ext_engine, denom_engine = _make_paired_engines(
        data["embed_dim"], data["feature_dim"], data["eta_rel"],
    )

    z_r, z_pos, z_negs = data["z_r"], data["z_pos"], data["z_negs"]
    scalar_features = data["scalar_features"]
    weak_inputs = {"d_esco": data["d_esco"], "d_isco": data["d_isco"]}

    # ---- Identical ReliabilityMLP predictions (Requirement 6.3). ------------
    reliability_ext = ext_engine._predict_reliability(z_r, z_negs, scalar_features)
    reliability_denom = denom_engine._predict_reliability(z_r, z_negs, scalar_features)
    assert reliability_ext.shape == reliability_denom.shape == z_negs.shape[:-1]
    assert torch.allclose(reliability_ext, reliability_denom, rtol=_RTOL, atol=_ATOL)

    # ---- Identical r_tilde weak targets (Requirement 6.3). ------------------
    r_tilde_ext = ext_engine.weak_builder.build(**weak_inputs)
    r_tilde_denom = denom_engine.weak_builder.build(**weak_inputs)
    assert torch.allclose(r_tilde_ext, r_tilde_denom, rtol=_RTOL, atol=_ATOL)

    # ---- external_weight main loss == mean(reliability) * standard_infonce. -
    total_ext = ext_engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features, weak_target_inputs=weak_inputs,
    )
    expected_ext_main = (
        reliability_ext.detach().mean()
        * ext_engine.standard_infonce(z_r, z_pos, z_negs).mean()
    )

    # ---- denominator main loss == orca_loss(...).mean(). --------------------
    total_denom = denom_engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features, weak_target_inputs=weak_inputs,
    )
    expected_denom_main = denom_engine.orca_loss(
        z_r, z_pos, z_negs, reliability_denom,
    ).mean()

    # ---- Shared reliability supervision term (identical on both variants). --
    rel_loss_ext = ext_engine.reliability_loss(reliability_ext, r_tilde_ext)
    rel_loss_denom = denom_engine.reliability_loss(reliability_denom, r_tilde_denom)
    assert torch.allclose(rel_loss_ext, rel_loss_denom, rtol=_RTOL, atol=_ATOL)

    eta = data["eta_rel"]

    # Each assembled total decomposes into its variant-specific main term plus
    # the identical eta_rel * reliability-BCE supervision term.
    assert torch.allclose(
        total_ext, expected_ext_main + eta * rel_loss_ext, rtol=_RTOL, atol=_ATOL,
    )
    assert torch.allclose(
        total_denom, expected_denom_main + eta * rel_loss_denom,
        rtol=_RTOL, atol=_ATOL,
    )

    # The sole difference between the two assembled totals is the application
    # site: stripping each variant's main term leaves the identical supervision
    # residual (Requirement 6.1 / 6.3).
    residual_ext = total_ext - expected_ext_main
    residual_denom = total_denom - expected_denom_main
    assert torch.allclose(residual_ext, residual_denom, rtol=_RTOL, atol=_ATOL)


def test_application_site_example_fixed_batch():
    """Deterministic example pinning the external_weight vs denominator sites.

    Validates: Requirements 6.1
    """
    embed_dim, feature_dim, eta_rel = 4, 5, 0.5
    ext_engine, denom_engine = _make_paired_engines(embed_dim, feature_dim, eta_rel)

    g = torch.Generator().manual_seed(11)
    B, K, D, Fd = 2, 3, embed_dim, feature_dim
    z_r = torch.randn(B, D, generator=g)
    z_pos = torch.randn(B, D, generator=g)
    z_negs = torch.randn(B, K, D, generator=g)
    scalar_features = torch.randn(B, K, Fd, generator=g)
    weak_inputs = {
        "d_esco": torch.rand(B, K, generator=g) * 3.0,
        "d_isco": torch.rand(B, K, generator=g) * 3.0,
    }

    reliability = ext_engine._predict_reliability(z_r, z_negs, scalar_features)
    reliability_d = denom_engine._predict_reliability(z_r, z_negs, scalar_features)
    assert torch.allclose(reliability, reliability_d, rtol=_RTOL, atol=_ATOL)

    total_ext = ext_engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features, weak_target_inputs=weak_inputs,
    )
    total_denom = denom_engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features, weak_target_inputs=weak_inputs,
    )

    r_tilde = ext_engine.weak_builder.build(**weak_inputs)
    expected_ext_main = (
        reliability.detach().mean()
        * ext_engine.standard_infonce(z_r, z_pos, z_negs).mean()
    )
    expected_denom_main = denom_engine.orca_loss(
        z_r, z_pos, z_negs, reliability,
    ).mean()
    rel_loss = ext_engine.reliability_loss(reliability, r_tilde)

    assert torch.allclose(
        total_ext, expected_ext_main + eta_rel * rel_loss, rtol=_RTOL, atol=_ATOL,
    )
    assert torch.allclose(
        total_denom, expected_denom_main + eta_rel * rel_loss, rtol=_RTOL, atol=_ATOL,
    )

    # Application site genuinely matters here: with reliability not all ones and
    # negatives present, the two main terms differ, so the totals differ.
    assert not torch.allclose(
        expected_ext_main, expected_denom_main, rtol=_RTOL, atol=_ATOL,
    )
