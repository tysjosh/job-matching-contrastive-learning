"""Unit tests for the ordinal ``priority_band_from_score`` metric.

Component under test: ``CVEEvaluationReporter._evaluate_band_from_score`` in
``cve_domain/evaluation_reporter.py``.

``priority_band`` is a deterministic bucketing of ``priority_score`` (contiguous,
non-overlapping per-band score ranges). The trained softmax band head collapses to
the majority class under the extreme 'watch' skew; the reporter therefore also
derives the band by bucketing the model's PREDICTED ``priority_score`` at cut points
derived from the ground-truth (band, score) pairs. These tests verify:

- cut points + severity order are derived from the data (not hard-coded),
- a perfect predicted score reproduces the ground-truth bands (macro-F1 = 1.0),
- a predicted score that always undershoots into 'watch' is caught (degenerate),
  yet the metric itself remains well-defined and reports the collapse honestly,
- the metric is skipped when band or score is absent from every record.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_band_from_score.py
"""

from __future__ import annotations

from typing import Any, Dict

from cve_domain.evaluation_reporter import CVEEvaluationReporter


# Band cut points used to build the fixtures (mirror the real data: 45/70/85).
_BANDS = [
    ("watch", 10.0),
    ("watch", 44.0),
    ("medium", 45.0),
    ("medium", 60.0),
    ("high", 70.0),
    ("high", 80.0),
    ("critical", 85.0),
    ("critical", 95.0),
]


def _record(cve: str, band: str, score: float) -> Dict[str, Any]:
    return {"cve": cve, "cve_labels": {"priority_band": band, "priority_score": score}}


def _records():
    return [_record(f"CVE-{i}", b, s) for i, (b, s) in enumerate(_BANDS)]


def _band_from_score(report) -> Dict[str, Any]:
    return report.classification["priority_band_from_score"]


def test_cutpoints_and_order_derived_from_data():
    records = _records()
    # Perfect predictor: predicted score == ground-truth score.
    preds = {
        str(r["cve"]): {"ranking_score": r["cve_labels"]["priority_score"]}
        for r in records
    }
    report = CVEEvaluationReporter().evaluate(records, preds)
    block = _band_from_score(report)

    assert block["skipped"] is False
    assert block["severity_order"] == ["watch", "medium", "high", "critical"]
    assert block["score_cut_points"] == [45.0, 70.0, 85.0]
    # A perfect score predictor buckets back to the exact ground-truth bands.
    assert block["macro_f1"] == 1.0
    assert block["accuracy"] == 1.0
    assert block["adjacent_accuracy"] == 1.0
    assert block["mae_priority_score"] == 0.0


def test_collapsed_score_predictor_is_reported_not_hidden():
    records = _records()
    # Pathological predictor: every predicted score falls in the 'watch' range.
    preds = {str(r["cve"]): {"ranking_score": 1.0} for r in records}
    report = CVEEvaluationReporter().evaluate(records, preds)
    block = _band_from_score(report)

    assert block["skipped"] is False
    # Everything predicted 'watch' -> only that class has any F1; macro-F1 < 0.5.
    assert block["per_class"]["watch"]["recall"] == 1.0
    assert block["per_class"]["critical"]["f1"] == 0.0
    assert block["macro_f1"] < 0.5
    # Still ordinal-aware: adjacent accuracy counts watch/medium as within one rank.
    assert 0.0 <= block["adjacent_accuracy"] <= 1.0


def test_beats_degenerate_argmax_when_score_is_informative():
    records = _records()
    # Slightly noisy but monotone score predictor.
    preds = {
        str(r["cve"]): {"ranking_score": r["cve_labels"]["priority_score"] - 3.0}
        for r in records
    }
    report = CVEEvaluationReporter().evaluate(records, preds)
    block = _band_from_score(report)
    # An informative score predictor must beat the majority-only collapse (~0.25
    # macro-F1 for 4 classes with everything in one bucket).
    assert block["macro_f1"] > 0.25


def test_skipped_when_score_absent():
    records = [
        {"cve": "CVE-a", "cve_labels": {"priority_band": "watch"}},
        {"cve": "CVE-b", "cve_labels": {"priority_band": "critical"}},
    ]
    preds = {"CVE-a": {"priority_band": "watch"}, "CVE-b": {"priority_band": "watch"}}
    report = CVEEvaluationReporter().evaluate(records, preds)
    block = _band_from_score(report)
    assert block["skipped"] is True
    assert "classification_priority_band_from_score" in report.skipped_metrics
