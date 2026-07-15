"""Tests for Stage 2 decision calibration helpers (imbalance-collapse fix).

Component under test: the validation-fitted decision-calibration helpers on
``CVEStage2Trainer`` in ``cve_domain/stage2.py``:

* ``_macro_f1`` — unweighted mean per-class F1 (must match the reporter's metric,
  so the threshold we pick optimizes the exact number being reported);
* ``_best_binary_threshold`` — the probability threshold maximizing macro-F1,
  which recovers the rare positive class that a fixed 0.5 threshold collapses;
* ``_label_bool`` — label coercion.

These are pure / torch-free helpers, so they are tested directly without building
a model. The end-to-end fit (``_calibrate_heads``) and predict-time application
are exercised on the L40S run via the ``calibration_report.json`` artifact.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_stage2_threshold_calibration.py
"""

from __future__ import annotations

import pytest

from cve_domain.stage2 import CVEStage2Trainer


# ---------------------------------------------------------------------------
# _macro_f1
# ---------------------------------------------------------------------------
def test_macro_f1_perfect_prediction():
    y = [0, 1, 1, 0, 1]
    assert CVEStage2Trainer._macro_f1(y, y, 2) == pytest.approx(1.0)


def test_macro_f1_all_majority_collapse_binary():
    """All-negative predictions on a rare-positive set → the collapse signature.

    12 negatives + 1 positive, predict all-negative: negative F1 = 1.0, positive
    F1 = 0.0, macro-F1 = 0.5 — exactly the in_kev 0.4988 pattern from the runs.
    """
    # ~1:212 skew (the in_kev ratio): negative F1 ≈ 0.9976, positive F1 = 0,
    # macro ≈ 0.4988 — the exact value seen in the runs.
    y_true = [0] * 212 + [1]
    y_pred = [0] * 213
    assert CVEStage2Trainer._macro_f1(y_true, y_pred, 2) == pytest.approx(0.4988, abs=1e-3)


def test_macro_f1_all_majority_collapse_4class():
    """Predict the majority band for everything on a 4-class skew.

    ~91% majority: majority F1 ≈ 0.95, others 0 → macro-F1 ≈ 0.24, matching the
    priority_band 0.239 collapse.
    """
    y_true = [3] * 91 + [0, 1, 2] * 3
    y_pred = [3] * len(y_true)
    f1 = CVEStage2Trainer._macro_f1(y_true, y_pred, 4)
    assert 0.20 < f1 < 0.28


def test_macro_f1_matches_manual_two_class():
    # tp=1, fp=1, fn=1 for class 1; symmetric for class 0.
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    # Each class: precision=0.5, recall=0.5, f1=0.5 → macro 0.5.
    assert CVEStage2Trainer._macro_f1(y_true, y_pred, 2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _best_binary_threshold
# ---------------------------------------------------------------------------
def test_best_threshold_recovers_rare_positive():
    """A separable-but-low-probability positive is recovered below 0.5.

    Negatives sit at prob ~0.1, the single positive at ~0.4. A fixed 0.5 predicts
    all-negative (macro-F1 0.5); a threshold in (0.1, 0.4] separates them (1.0).
    """
    probs = [0.05, 0.08, 0.10, 0.12, 0.09, 0.40]
    ys = [0, 0, 0, 0, 0, 1]
    best_t, f1_half, best_f1 = CVEStage2Trainer._best_binary_threshold(probs, ys)

    assert f1_half < 0.5                  # 0.5 threshold collapses to all-negative
    assert best_f1 == pytest.approx(1.0)  # calibration perfectly separates
    assert best_f1 > f1_half              # calibration strictly improves macro-F1
    assert 0.12 <= best_t <= 0.40         # threshold lands between the classes


def test_best_threshold_no_improvement_when_inseparable():
    """When classes overlap completely, calibration can't beat the collapse.

    All probabilities identical → no threshold separates → best == 0.5 baseline.
    This is the flat-logit case the diagnostic flags as an embedding ceiling.
    """
    probs = [0.3] * 10
    ys = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    _, f1_half, best_f1 = CVEStage2Trainer._best_binary_threshold(probs, ys)
    assert best_f1 == pytest.approx(f1_half)


# ---------------------------------------------------------------------------
# _label_bool
# ---------------------------------------------------------------------------
def test_label_bool_coercion():
    assert CVEStage2Trainer._label_bool(True) is True
    assert CVEStage2Trainer._label_bool(False) is False
    assert CVEStage2Trainer._label_bool(1) is True
    assert CVEStage2Trainer._label_bool(0) is False
    assert CVEStage2Trainer._label_bool(None) is None
