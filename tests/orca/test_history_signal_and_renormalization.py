#!/usr/bin/env python3
"""Unit tests for the ORCA history weak-target signal and renormalization.

Component under test: ``orca.weak_targets.WeakTargetBuilder`` — specifically the
``history_reliability`` signal (design section B.3, task 7.1) and its
integration into ``build`` / ``build_targets`` with weight renormalization.

Requirements exercised:
  * Requirement 2.1 — active blend weights renormalize to sum to ``1.0`` (within
    ``1e-6``) over the *present* signals and ``r_tilde`` stays in ``[0, 1]``.
    Here the history signal is exercised as a third present signal, so the
    active set can be ``{ont, enc, hist}``.
  * Requirement 2.3 — a missing history signal is dropped and its weight
    redistributed across the remaining present signals so the active weights
    again sum to ``1.0``.
  * Requirement 2.4 — with ``orca_use_history`` false the history term is
    excluded, yielding a ``r_tilde`` equal to setting ``orca_lambda_hist = 0``
    and renormalizing over the ontology and encoder signals; and this equals the
    ``orca_use_history`` true result when the history signal is absent.

These are example/edge-case unit tests (not Hypothesis property tests): they pin
the documented ``history_reliability`` input contract (single string /
dict / enum-like status and the batch list/tuple form), the neutral fill for
unrecognized-within-batch tokens, the missing-signal ``None`` return, and the
renormalization equivalences above.

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_history_signal_and_renormalization.py
"""

from __future__ import annotations

import torch

from contrastive_learning.data_structures import TrainingConfig
from orca.weak_targets import (
    NEUTRAL_R_TILDE,
    WEIGHT_SUM_TOL,
    WeakTargetBuilder,
)


# Small, fixed problem dimensions (B anchors x K negatives, embed dim D).
_B, _K, _D = 2, 3, 4

_RTOL = 1e-5
_ATOL = 1e-6


# ---------------------------------------------------------------------------
# An enum-like status object exposing a status-bearing attribute, per the
# documented ``history_reliability`` input contract (``value``/``name``/
# ``type``/``status``). Mirrors how a real Enum member would present.
# ---------------------------------------------------------------------------
class _StatusEnumLike:
    def __init__(self, value: str):
        self.value = value
        self.name = value


def _make_builder(
    *,
    use_history: bool,
    lambda_ont: float = 0.5,
    lambda_enc: float = 0.3,
    lambda_hist: float = 0.2,
    omega: float = 0.5,
    beta: float = 1.0,
    gamma: float = 5.0,
) -> WeakTargetBuilder:
    cfg = TrainingConfig(
        orca_enabled=True,
        orca_variant="denominator",
        orca_omega=omega,
        orca_beta=beta,
        orca_gamma_enc=gamma,
        orca_lambda_ont=lambda_ont,
        orca_lambda_enc=lambda_enc,
        orca_lambda_hist=lambda_hist,
        orca_use_history=use_history,
    )
    return WeakTargetBuilder(cfg)


def _inputs(seed: int = 3):
    """Reproducible ontology distances and warmup embeddings."""
    g = torch.Generator().manual_seed(seed)
    d_esco = torch.rand(_B, _K, generator=g) * 5.0
    d_isco = torch.rand(_B, _K, generator=g) * 5.0
    z_r = torch.randn(_B, _K, _D, generator=g)
    z_neg = torch.randn(_B, _K, _D, generator=g)
    return d_esco, d_isco, z_r, z_neg


# ===========================================================================
# History semantics — Requirement 2.1
# ===========================================================================

def test_history_reliability_recognized_single_statuses():
    """later_positive -> 0.0, repeatedly_negative -> 1.0 (single status)."""
    builder = _make_builder(use_history=True)

    later = builder.history_reliability("later_positive")
    repeated = builder.history_reliability("repeatedly_negative")

    assert later is not None and repeated is not None
    assert float(later) == 0.0
    assert float(repeated) == 1.0


def test_history_reliability_is_case_insensitive():
    """Status strings are matched case-insensitively (with surrounding space)."""
    builder = _make_builder(use_history=True)

    assert float(builder.history_reliability("LATER_POSITIVE")) == 0.0
    assert float(builder.history_reliability("  Repeatedly_Negative  ")) == 1.0


def test_history_reliability_accepts_dict_type_and_status_keys():
    """A mapping carrying a 'type' or 'status' key is accepted."""
    builder = _make_builder(use_history=True)

    assert float(builder.history_reliability({"type": "later_positive"})) == 0.0
    assert float(builder.history_reliability({"status": "repeatedly_negative"})) == 1.0


def test_history_reliability_accepts_enum_like_object():
    """An enum-like object exposing value/name is accepted."""
    builder = _make_builder(use_history=True)

    assert float(builder.history_reliability(_StatusEnumLike("later_positive"))) == 0.0
    assert float(builder.history_reliability(_StatusEnumLike("repeatedly_negative"))) == 1.0


def test_history_reliability_none_and_unrecognized_return_none():
    """None or an unrecognized status is treated as missing (returns None)."""
    builder = _make_builder(use_history=True)

    assert builder.history_reliability(None) is None
    assert builder.history_reliability("ghosted") is None
    assert builder.history_reliability({"type": "unknown"}) is None
    assert builder.history_reliability({"other_key": "later_positive"}) is None
    assert builder.history_reliability(_StatusEnumLike("nope")) is None


# ---------------------------------------------------------------------------
# History semantics — batch (list/tuple) form
# ---------------------------------------------------------------------------

def test_history_reliability_batch_maps_recognized_tokens():
    """A batch of recognized tokens maps 1:1 to 0.0/1.0."""
    builder = _make_builder(use_history=True)

    tokens = ["later_positive", "repeatedly_negative", "later_positive"]
    r_hist = builder.history_reliability(tokens)

    assert r_hist is not None
    assert r_hist.shape == (3,)
    assert torch.equal(r_hist, torch.tensor([0.0, 1.0, 0.0]))


def test_history_reliability_batch_fills_unrecognized_with_neutral():
    """Unrecognized tokens *within* an otherwise-present batch -> neutral 0.5."""
    builder = _make_builder(use_history=True)

    tokens = ["later_positive", "ghosted", None, "repeatedly_negative"]
    r_hist = builder.history_reliability(tokens)

    assert r_hist is not None
    expected = torch.tensor([0.0, NEUTRAL_R_TILDE, NEUTRAL_R_TILDE, 1.0])
    assert torch.equal(r_hist, expected)
    # Stays within the unit interval (Req 2.1).
    assert (r_hist >= 0.0).all() and (r_hist <= 1.0).all()


def test_history_reliability_batch_all_unrecognized_or_empty_returns_none():
    """A batch with no recognized token, or an empty batch, is missing (None)."""
    builder = _make_builder(use_history=True)

    assert builder.history_reliability([]) is None
    assert builder.history_reliability(()) is None
    assert builder.history_reliability(["ghosted", None, {"type": "x"}]) is None


def test_history_reliability_batch_accepts_mixed_token_representations():
    """A batch may mix strings, dicts, and enum-like tokens."""
    builder = _make_builder(use_history=True)

    tokens = [
        "later_positive",
        {"status": "repeatedly_negative"},
        _StatusEnumLike("later_positive"),
    ]
    r_hist = builder.history_reliability(tokens)

    assert torch.equal(r_hist, torch.tensor([0.0, 1.0, 0.0]))


# ===========================================================================
# Blending with history ON — Requirement 2.1
# ===========================================================================

def test_build_includes_history_and_renormalizes_three_signals():
    """With history ON and present, all three signals blend and renormalize.

    ``signals_present['hist']`` is true, the active weights
    (lambda_ont, lambda_enc, lambda_hist) renormalize to sum 1.0, ``r_tilde``
    lies in ``[0, 1]``, and equals the independently-computed convex
    combination (Requirement 2.1).
    """
    lambda_ont, lambda_enc, lambda_hist = 0.5, 0.3, 0.2
    builder = _make_builder(
        use_history=True,
        lambda_ont=lambda_ont, lambda_enc=lambda_enc, lambda_hist=lambda_hist,
    )
    d_esco, d_isco, z_r, z_neg = _inputs()
    # Per-pair history tokens aligned with the K negatives of a single anchor
    # row; broadcasts against the (B, K) ontology/encoder signals.
    tokens = ["later_positive", "repeatedly_negative", "later_positive"]

    r_tilde, signals_present = builder.build_targets(
        d_esco=d_esco, d_isco=d_isco,
        z_r_warmup=z_r, z_neg_warmup=z_neg,
        interaction=tokens,
    )

    assert signals_present == {"ont": True, "enc": True, "hist": True}
    assert torch.isfinite(r_tilde).all()
    assert (r_tilde >= 0.0).all() and (r_tilde <= 1.0).all()

    # Independently reconstruct the renormalized active weights.
    w = torch.tensor([lambda_ont, lambda_enc, lambda_hist], dtype=torch.float32)
    w = w / w.sum()
    assert abs(float(w.sum()) - 1.0) <= WEIGHT_SUM_TOL

    r_ont = builder.ontology_reliability(d_esco, d_isco)
    r_enc = builder.encoder_reliability(z_r, z_neg)
    r_hist = builder.history_reliability(tokens)
    expected = w[0] * r_ont + w[1] * r_enc + w[2] * r_hist
    expected = torch.clamp(expected, 0.0, 1.0)

    assert torch.allclose(r_tilde, expected, rtol=_RTOL, atol=_ATOL)


# ===========================================================================
# Missing-history redistribution — Requirement 2.3
# ===========================================================================

def test_missing_history_redistributes_weight_to_ont_and_enc():
    """History ON but missing -> excluded, weight redistributed (Req 2.3).

    With ``use_history=True`` but ``interaction=None`` (or an unrecognized
    status), the history signal is dropped, the active weights over
    ontology+encoder renormalize to sum 1.0, and the result equals the
    no-history blend built with ``use_history=False``.
    """
    builder_on = _make_builder(use_history=True)
    builder_off = _make_builder(use_history=False)
    d_esco, d_isco, z_r, z_neg = _inputs()

    build_kwargs = dict(
        d_esco=d_esco, d_isco=d_isco, z_r_warmup=z_r, z_neg_warmup=z_neg,
    )

    for missing_interaction in (None, "ghosted", ["ghosted", None]):
        r_tilde_on, present_on = builder_on.build_targets(
            interaction=missing_interaction, **build_kwargs
        )
        # History dropped -> only ontology + encoder active.
        assert present_on == {"ont": True, "enc": True, "hist": False}

        # Active weights over the two present signals sum to 1.0.
        w = torch.tensor([builder_on.lambda_ont, builder_on.lambda_enc],
                         dtype=torch.float32)
        w = w / w.sum()
        assert abs(float(w.sum()) - 1.0) <= WEIGHT_SUM_TOL

        # Equals the no-history (use_history=False) blend on the same inputs.
        r_tilde_off = builder_off.build(**build_kwargs)
        assert torch.allclose(r_tilde_on, r_tilde_off, rtol=_RTOL, atol=_ATOL)


# ===========================================================================
# orca_use_history=False reproduces ontology+encoder-only r_tilde — Req 2.4
# ===========================================================================

def test_use_history_false_equals_lambda_hist_zero_renormalized():
    """use_history=False == lambda_hist=0 renormalized over ont+enc (Req 2.4)."""
    d_esco, d_isco, z_r, z_neg = _inputs()
    build_kwargs = dict(
        d_esco=d_esco, d_isco=d_isco, z_r_warmup=z_r, z_neg_warmup=z_neg,
    )
    # A history token is supplied but must be ignored while history is off.
    tokens = ["later_positive", "repeatedly_negative", "later_positive"]

    off = _make_builder(use_history=False, lambda_hist=0.2).build(
        interaction=tokens, **build_kwargs
    )
    # Explicit lambda_hist=0, still history-off, renormalized over ont+enc.
    zeroed = _make_builder(use_history=False, lambda_hist=0.0).build(
        interaction=tokens, **build_kwargs
    )

    assert torch.allclose(off, zeroed, rtol=_RTOL, atol=_ATOL)


def test_use_history_false_equals_use_history_true_when_history_absent():
    """When history is absent, use_history true and false agree (Req 2.4)."""
    d_esco, d_isco, z_r, z_neg = _inputs()
    build_kwargs = dict(
        d_esco=d_esco, d_isco=d_isco, z_r_warmup=z_r, z_neg_warmup=z_neg,
    )

    # use_history=True but no interaction -> history absent.
    on_absent = _make_builder(use_history=True).build(**build_kwargs)
    # use_history=False -> history excluded by config.
    off = _make_builder(use_history=False).build(**build_kwargs)

    assert torch.allclose(on_absent, off, rtol=_RTOL, atol=_ATOL)


def test_use_history_false_matches_direct_ont_enc_convex_combination():
    """use_history=False r_tilde == direct ont+enc convex combination (Req 2.4)."""
    lambda_ont, lambda_enc = 0.5, 0.3
    builder = _make_builder(
        use_history=False, lambda_ont=lambda_ont, lambda_enc=lambda_enc,
        lambda_hist=0.2,  # configured but must be ignored while history is off
    )
    d_esco, d_isco, z_r, z_neg = _inputs()

    r_tilde = builder.build(
        d_esco=d_esco, d_isco=d_isco, z_r_warmup=z_r, z_neg_warmup=z_neg,
    )

    # Renormalize only over ontology + encoder (history weight excluded).
    w = torch.tensor([lambda_ont, lambda_enc], dtype=torch.float32)
    w = w / w.sum()
    r_ont = builder.ontology_reliability(d_esco, d_isco)
    r_enc = builder.encoder_reliability(z_r, z_neg)
    expected = torch.clamp(w[0] * r_ont + w[1] * r_enc, 0.0, 1.0)

    assert torch.allclose(r_tilde, expected, rtol=_RTOL, atol=_ATOL)
