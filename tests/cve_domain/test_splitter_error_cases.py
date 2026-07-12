"""Unit tests for CVE Data_Splitter error / edge cases (Task 6.6).

Component under test: ``CVEDataSplitter`` in ``cve_domain/data_splitter.py``.

These focused pytest unit tests cover the splitter's guard rails and special-case
record handling that the property-based tests do not target directly:

* **Invalid proportions (Req 4.9):** proportions that do not sum to 100% within
  ±0.5% cause the splitter to stop before writing any split artifact, emitting only
  ``split_report.json`` with ``status == "invalid_proportions"`` and a reason.
* **Empty-split guard (Req 4.9):** a small dataset that would produce an empty
  train/validation/test split causes the splitter to stop before writing split
  artifacts, emitting only ``split_report.json`` with ``status == "empty_split"``.
* **Undated temporal records (Req 4.7):** the temporal strategy assigns records with
  a missing/unparseable ``nvd_published`` to the train split and increments
  ``temporal_missing_date_reassigned_count``.
* **Missing-band unbanded stratum (Req 4.12):** the stratified strategy assigns
  records with a missing/empty ``priority_band`` to a dedicated unbanded stratum,
  splits that stratum by the configured proportions, and records ``unbanded_count``.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_splitter_error_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cve_domain.data_splitter import SPLIT_NAMES, CVEDataSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(cve: str, band=None, published=None) -> dict:
    """Build a minimal CVE_View_Record with an optional band / published date."""
    cve_labels = {}
    if band is not None:
        cve_labels["priority_band"] = band
    return {"cve": cve, "nvd_published": published, "cve_labels": cve_labels}


def _split_artifacts(output_dir: Path):
    """Return the set of split artifact paths (excluding split_report.json)."""
    return [
        output_dir / "train.jsonl",
        output_dir / "validation.jsonl",
        output_dir / "test.jsonl",
        output_dir / "split_indices.json",
    ]


def _read_split_cves(output_dir: Path) -> dict:
    """Read emitted per-split JSONL files back into {split: [cve, ...]}."""
    result = {}
    for name in SPLIT_NAMES:
        cves = []
        path = output_dir / f"{name}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    cves.append(json.loads(line)["cve"])
        result[name] = cves
    return result


# ---------------------------------------------------------------------------
# Invalid proportions (Req 4.9)
# ---------------------------------------------------------------------------

def test_invalid_proportions_sum_too_low_aborts_before_writing(tmp_path):
    """Proportions summing well below 100% → invalid_proportions, no split artifacts."""
    records = [_record(f"CVE-{i}", band="high") for i in range(30)]
    output_dir = tmp_path / "bad_low"

    splitter = CVEDataSplitter(
        strategy="stratified",
        proportions={"train": 50.0, "validation": 10.0, "test": 10.0},  # sums to 70%
        seed=42,
    )
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "invalid_proportions"
    assert report.reason is not None
    assert "100%" in report.reason

    # Only split_report.json should exist; no split artifacts written.
    assert (output_dir / "split_report.json").exists()
    for artifact in _split_artifacts(output_dir):
        assert not artifact.exists(), f"{artifact.name} must not be written on invalid proportions"

    # The persisted report mirrors the returned report.
    persisted = json.loads((output_dir / "split_report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "invalid_proportions"
    assert persisted["reason"] == report.reason


def test_invalid_proportions_sum_too_high_aborts_before_writing(tmp_path):
    """Proportions summing above the ±0.5% tolerance → invalid_proportions."""
    records = [_record(f"CVE-{i}", band="high") for i in range(30)]
    output_dir = tmp_path / "bad_high"

    splitter = CVEDataSplitter(
        strategy="random",
        proportions={"train": 80.0, "validation": 15.0, "test": 10.0},  # sums to 105%
        seed=7,
    )
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "invalid_proportions"
    for artifact in _split_artifacts(output_dir):
        assert not artifact.exists()


def test_proportions_within_tolerance_are_accepted(tmp_path):
    """Proportions summing to 100% within ±0.5% (e.g. 99.7%) are accepted."""
    records = [_record(f"CVE-{i}", band="high") for i in range(100)]
    output_dir = tmp_path / "within_tol"

    splitter = CVEDataSplitter(
        strategy="random",
        proportions={"train": 79.8, "validation": 9.95, "test": 9.95},  # sums to 99.7%
        seed=42,
    )
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "ok"
    for artifact in _split_artifacts(output_dir):
        assert artifact.exists()


# ---------------------------------------------------------------------------
# Empty-split guard (Req 4.9)
# ---------------------------------------------------------------------------

def test_empty_split_guard_stops_before_writing(tmp_path):
    """A dataset too small to fill all three splits → empty_split, no artifacts."""
    # With 80/10/10 and only 2 records, validation/test cut to 0 → empty split.
    records = [_record("CVE-1", band="high"), _record("CVE-2", band="high")]
    output_dir = tmp_path / "empty_guard"

    splitter = CVEDataSplitter(
        strategy="random",
        proportions={"train": 80.0, "validation": 10.0, "test": 10.0},
        seed=42,
    )
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "empty_split"
    assert report.reason is not None

    # Only split_report.json should exist.
    assert (output_dir / "split_report.json").exists()
    for artifact in _split_artifacts(output_dir):
        assert not artifact.exists(), f"{artifact.name} must not be written on empty split"


def test_empty_split_guard_triggers_on_zero_records(tmp_path):
    """Zero input records → every split empty → empty_split guard fires."""
    output_dir = tmp_path / "empty_input"

    splitter = CVEDataSplitter(strategy="stratified", seed=42)
    report = splitter.split_records([], str(output_dir))

    assert report.status == "empty_split"
    for artifact in _split_artifacts(output_dir):
        assert not artifact.exists()


def test_sufficient_records_do_not_trigger_empty_guard(tmp_path):
    """A dataset large enough for all three splits proceeds to write artifacts."""
    records = [_record(f"CVE-{i}", band="high") for i in range(50)]
    output_dir = tmp_path / "ok_split"

    splitter = CVEDataSplitter(
        strategy="random",
        proportions={"train": 80.0, "validation": 10.0, "test": 10.0},
        seed=42,
    )
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "ok"
    for name in SPLIT_NAMES:
        assert report.per_split_counts[name] > 0


# ---------------------------------------------------------------------------
# Undated temporal records (Req 4.7)
# ---------------------------------------------------------------------------

def test_temporal_undated_records_go_to_train_and_are_counted(tmp_path):
    """Temporal strategy: missing/unparseable nvd_published → train + counted."""
    dated = [
        _record("CVE-D1", band="high", published="2020-01-01T00:00:00"),
        _record("CVE-D2", band="high", published="2021-01-01T00:00:00"),
        _record("CVE-D3", band="high", published="2022-01-01T00:00:00"),
        _record("CVE-D4", band="high", published="2023-01-01T00:00:00"),
        _record("CVE-D5", band="high", published="2024-01-01T00:00:00"),
        _record("CVE-D6", band="high", published="2024-06-01T00:00:00"),
        _record("CVE-D7", band="high", published="2024-07-01T00:00:00"),
        _record("CVE-D8", band="high", published="2024-08-01T00:00:00"),
    ]
    undated = [
        _record("CVE-U1", band="high", published=None),      # missing
        _record("CVE-U2", band="high", published=""),        # blank
        _record("CVE-U3", band="high", published="not-a-date"),  # unparseable
    ]
    records = dated + undated
    output_dir = tmp_path / "temporal_undated"

    splitter = CVEDataSplitter(
        strategy="temporal",
        proportions={"train": 80.0, "validation": 10.0, "test": 10.0},
        seed=42,
    )
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "ok"
    assert report.temporal_missing_date_reassigned_count == len(undated)

    split_cves = _read_split_cves(output_dir)
    train_cves = set(split_cves["train"])
    # Every undated record must land in train.
    for rec in undated:
        assert rec["cve"] in train_cves, f"{rec['cve']} (undated) must be in train"

    # Undated records must NOT appear in validation or test.
    for name in ("validation", "test"):
        for rec in undated:
            assert rec["cve"] not in split_cves[name]

    # Partition integrity: all records assigned exactly once.
    all_assigned = [cve for name in SPLIT_NAMES for cve in split_cves[name]]
    assert sorted(all_assigned) == sorted(r["cve"] for r in records)
    assert len(all_assigned) == len(set(all_assigned))


def test_temporal_no_undated_records_reports_zero(tmp_path):
    """Temporal strategy with all-dated records → reassignment count is zero."""
    records = [
        _record(f"CVE-{i}", band="high", published=f"20{10 + i:02d}-01-01T00:00:00")
        for i in range(20)
    ]
    output_dir = tmp_path / "temporal_all_dated"

    splitter = CVEDataSplitter(strategy="temporal", seed=42)
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "ok"
    assert report.temporal_missing_date_reassigned_count == 0


# ---------------------------------------------------------------------------
# Missing-band unbanded stratum (Req 4.12)
# ---------------------------------------------------------------------------

def test_stratified_missing_band_forms_unbanded_stratum_and_counts(tmp_path):
    """Stratified strategy: missing/empty priority_band → unbanded stratum + count."""
    banded = [_record(f"CVE-B{i}", band="high") for i in range(30)]
    # Missing-band records: some with no priority_band key, some with empty string.
    unbanded = (
        [_record(f"CVE-N{i}", band=None) for i in range(10)]
        + [_record(f"CVE-E{i}", band="") for i in range(10)]
        + [_record(f"CVE-W{i}", band="   ") for i in range(5)]  # whitespace-only
    )
    records = banded + unbanded
    output_dir = tmp_path / "unbanded"

    splitter = CVEDataSplitter(
        strategy="stratified",
        proportions={"train": 80.0, "validation": 10.0, "test": 10.0},
        seed=42,
    )
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "ok"
    # All 25 missing/empty/whitespace-band records form the unbanded stratum.
    assert report.unbanded_count == len(unbanded)

    # The unbanded stratum is split across all three splits by proportions, so it
    # is not dumped wholesale into a single split. Confirm the "unbanded" band label
    # appears in the per-split band distribution for more than one split.
    splits_with_unbanded = [
        name
        for name in SPLIT_NAMES
        if report.per_split_band_distribution.get(name, {}).get("unbanded", 0) > 0
    ]
    assert len(splits_with_unbanded) >= 2, (
        "the unbanded stratum must be allocated across splits by proportions, "
        f"but only appeared in {splits_with_unbanded}"
    )

    # The total count of unbanded records across all splits equals unbanded_count.
    total_unbanded_in_splits = sum(
        report.per_split_band_distribution.get(name, {}).get("unbanded", 0)
        for name in SPLIT_NAMES
    )
    assert total_unbanded_in_splits == len(unbanded)

    # Partition integrity across the whole dataset.
    split_cves = _read_split_cves(output_dir)
    all_assigned = [cve for name in SPLIT_NAMES for cve in split_cves[name]]
    assert sorted(all_assigned) == sorted(r["cve"] for r in records)
    assert len(all_assigned) == len(set(all_assigned))


def test_stratified_no_missing_band_reports_zero_unbanded(tmp_path):
    """Stratified strategy with all records banded → unbanded_count is zero."""
    records = (
        [_record(f"CVE-H{i}", band="high") for i in range(20)]
        + [_record(f"CVE-L{i}", band="low") for i in range(20)]
    )
    output_dir = tmp_path / "all_banded"

    splitter = CVEDataSplitter(strategy="stratified", seed=42)
    report = splitter.split_records(records, str(output_dir))

    assert report.status == "ok"
    assert report.unbanded_count == 0
    # No "unbanded" key should appear in any split's band distribution.
    for name in SPLIT_NAMES:
        assert "unbanded" not in report.per_split_band_distribution.get(name, {})
