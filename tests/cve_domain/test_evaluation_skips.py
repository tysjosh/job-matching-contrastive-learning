"""Unit tests for CVEEvaluationReporter metric skips (task 11.3).

Component under test: ``CVEEvaluationReporter.evaluate`` in
``cve_domain/evaluation_reporter.py``.

These example-based tests cover the two skip conditions of the evaluation
reporter (Requirement 10):

- **Absent target label (Req 10.5):** when a label required by a metric is
  absent from *every* test-split record, that metric is skipped and the skip +
  reason is recorded in the report (via ``skipped_metrics`` and an inline
  ``{"skipped": True, "reason": ...}`` marker on the metric block). Covered:
  * no ``priority_score`` on any record -> ranking skipped,
  * no ``in_kev`` on any record -> in_kev classification skipped,
  * no ``priority_band`` on any record -> band classification AND
    embedding-separation skipped.
- **Empty test split (Req 10.6):** an empty ``test_records`` sequence yields
  status ``"empty_test_split"``, all metrics skipped, the condition recorded,
  and ``evaluation_report.json`` is still written when ``output_dir`` is given.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_evaluation_skips.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from cve_domain.evaluation_reporter import CVEEvaluationReporter


# ---------------------------------------------------------------------------
# Helpers to build test-split CVE_View_Records and predictions.
# ---------------------------------------------------------------------------
def _record(cve: str, **labels: Any) -> Dict[str, Any]:
    """A test-split CVE_View_Record carrying only the given ground-truth labels.

    Labels are placed in the canonical ``cve_labels`` object. Omitting a label
    keyword means that label is *absent* from the record (the condition Req 10.5
    is about).
    """
    return {"cve": cve, "cve_labels": dict(labels)}


def _predictions(*cves: str) -> Dict[str, Dict[str, Any]]:
    """Full predictions (ranking score, in_kev, band, embedding) for each cve.

    The predictions are deliberately complete so that when a metric is skipped
    it is skipped because the *ground-truth label* is absent, not because a
    prediction is missing.
    """
    preds: Dict[str, Dict[str, Any]] = {}
    for i, cve in enumerate(cves):
        preds[cve] = {
            "ranking_score": float(len(cves) - i),
            "in_kev": True,
            "priority_band": "HIGH",
            "embedding": [float(i), float(i) + 1.0],
        }
    return preds


# ---------------------------------------------------------------------------
# Req 10.6 — empty test split
# ---------------------------------------------------------------------------
def test_empty_test_split_skips_all_metrics_and_records_condition(tmp_path: Path) -> None:
    reporter = CVEEvaluationReporter()

    report = reporter.evaluate(
        test_records=[],
        predictions={},
        output_dir=str(tmp_path),
    )

    # Status flags the empty split and no records were considered (Req 10.6).
    assert report.status == "empty_test_split"
    assert report.num_test_records == 0

    # Every metric is recorded as skipped with a reason.
    assert set(report.skipped_metrics) == {
        "ranking",
        "classification_in_kev",
        "classification_priority_band",
        "embedding_separation",
    }
    assert all(reason for reason in report.skipped_metrics.values())

    # The empty-test-split condition is explicitly recorded in the notes.
    assert "empty_test_split" in report.notes


def test_empty_test_split_writes_report_json(tmp_path: Path) -> None:
    reporter = CVEEvaluationReporter()

    reporter.evaluate(test_records=[], predictions={}, output_dir=str(tmp_path))

    report_path = tmp_path / "evaluation_report.json"
    assert report_path.exists(), "evaluation_report.json must be written even on empty split (Req 10.6/10.7)"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "empty_test_split"
    assert payload["num_test_records"] == 0
    assert set(payload["skipped_metrics"]) == {
        "ranking",
        "classification_in_kev",
        "classification_priority_band",
        "embedding_separation",
    }


# ---------------------------------------------------------------------------
# Req 10.5 — absent target label skips the dependent metric(s)
# ---------------------------------------------------------------------------
def test_absent_priority_score_skips_ranking(tmp_path: Path) -> None:
    # No record carries priority_score -> ranking must be skipped (Req 10.5).
    records: List[Mapping[str, Any]] = [
        _record("CVE-2024-0001", in_kev=True, priority_band="HIGH"),
        _record("CVE-2024-0002", in_kev=False, priority_band="LOW"),
    ]
    reporter = CVEEvaluationReporter()

    report = reporter.evaluate(records, _predictions("CVE-2024-0001", "CVE-2024-0002"), str(tmp_path))

    assert report.status == "ok"
    # Skip is recorded both in the flat map and as an inline marker with a reason.
    assert "ranking" in report.skipped_metrics
    assert "priority_score" in report.skipped_metrics["ranking"]
    assert report.ranking.get("skipped") is True
    assert "priority_score" in report.ranking.get("reason", "")

    # Metrics whose labels ARE present are not skipped.
    assert "classification_in_kev" not in report.skipped_metrics
    assert "classification_priority_band" not in report.skipped_metrics


def test_absent_in_kev_skips_in_kev_classification(tmp_path: Path) -> None:
    # No record carries in_kev -> in_kev classification must be skipped (Req 10.5).
    records: List[Mapping[str, Any]] = [
        _record("CVE-2024-0001", priority_score=90.0, priority_band="HIGH"),
        _record("CVE-2024-0002", priority_score=10.0, priority_band="LOW"),
    ]
    reporter = CVEEvaluationReporter()

    report = reporter.evaluate(records, _predictions("CVE-2024-0001", "CVE-2024-0002"), str(tmp_path))

    assert report.status == "ok"
    assert "classification_in_kev" in report.skipped_metrics
    assert "in_kev" in report.skipped_metrics["classification_in_kev"]
    assert report.classification["in_kev"].get("skipped") is True
    assert "in_kev" in report.classification["in_kev"].get("reason", "")

    # priority_band classification stays enabled since the band label is present.
    assert "classification_priority_band" not in report.skipped_metrics
    assert report.classification["priority_band"].get("skipped") is False


def test_absent_priority_band_skips_band_classification_and_embedding_separation(tmp_path: Path) -> None:
    # No record carries priority_band -> both band classification and embedding
    # separation (which groups embeddings by band) must be skipped (Req 10.5).
    records: List[Mapping[str, Any]] = [
        _record("CVE-2024-0001", priority_score=90.0, in_kev=True),
        _record("CVE-2024-0002", priority_score=10.0, in_kev=False),
    ]
    reporter = CVEEvaluationReporter()

    report = reporter.evaluate(records, _predictions("CVE-2024-0001", "CVE-2024-0002"), str(tmp_path))

    assert report.status == "ok"

    assert "classification_priority_band" in report.skipped_metrics
    assert "priority_band" in report.skipped_metrics["classification_priority_band"]
    assert report.classification["priority_band"].get("skipped") is True

    assert "embedding_separation" in report.skipped_metrics
    assert "priority_band" in report.skipped_metrics["embedding_separation"]
    assert report.embedding_separation.get("skipped") is True

    # Ranking (priority_score present) and in_kev classification are unaffected.
    assert "ranking" not in report.skipped_metrics
    assert "classification_in_kev" not in report.skipped_metrics


def test_all_labels_absent_skips_every_metric(tmp_path: Path) -> None:
    # Records with no supervised labels at all -> every metric skipped, but the
    # split is non-empty so status stays "ok" (distinct from Req 10.6).
    records: List[Mapping[str, Any]] = [_record("CVE-2024-0001"), _record("CVE-2024-0002")]
    reporter = CVEEvaluationReporter()

    report = reporter.evaluate(records, _predictions("CVE-2024-0001", "CVE-2024-0002"), str(tmp_path))

    assert report.status == "ok"
    assert report.num_test_records == 2
    assert set(report.skipped_metrics) == {
        "ranking",
        "classification_in_kev",
        "classification_priority_band",
        "embedding_separation",
    }
    # Each skip carries a non-empty reason (Req 10.5).
    assert all(reason for reason in report.skipped_metrics.values())


def test_absent_label_skip_reason_persisted_to_report_json(tmp_path: Path) -> None:
    # The skip + reason must survive serialization to evaluation_report.json.
    records: List[Mapping[str, Any]] = [
        _record("CVE-2024-0001", in_kev=True, priority_band="HIGH"),
    ]
    reporter = CVEEvaluationReporter()

    reporter.evaluate(records, _predictions("CVE-2024-0001"), str(tmp_path))

    payload = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert "ranking" in payload["skipped_metrics"]
    assert "priority_score" in payload["skipped_metrics"]["ranking"]
    assert payload["ranking"]["skipped"] is True
