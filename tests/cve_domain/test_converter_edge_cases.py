"""Unit tests for CVEDataConverter.convert edge cases (task 4.11).

Component under test: ``CVEDataConverter.convert`` in
``cve_domain/data_converter.py``.

These example-based tests cover the join / skip / exclude edge cases of the
conversion algorithm (Requirement 1):

- Join each CVE_Record to its Ontology_Profile by ``cve`` (Req 1.4).
- Missing-profile join: a row whose ``cve`` has no matching profile is emitted
  as a CSV-only view and the missing-profile condition is recorded (Req 1.5).
- Malformed / unparseable rows and rows with a missing/empty ``cve`` are skipped
  and counted, and conversion continues over the remaining rows (Req 1.8).
- A constructed record that fails the configured Domain_Adapter's ``validate()``
  hook is excluded from the output and counted (Req 1.9).

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_converter_edge_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest

from cve_domain.data_converter import CVEConvertConfig, CVEDataConverter
from contrastive_learning import domain_adapters


# ---------------------------------------------------------------------------
# Fixture helpers: write temp CSV + profiles JSONL, run the converter.
# ---------------------------------------------------------------------------

CSV_HEADER = (
    "cve,priority_score,priority_band,in_kev,known_ransomware_campaign_use,"
    "vendor_project,product,vulnerability_name,cwes,cpes_sample,nvd_published"
)


def _write_csv(tmp_path: Path, rows: List[str]) -> str:
    """Write a CSV with the fixed header + the given raw data lines."""
    csv_path = tmp_path / "cves.csv"
    csv_path.write_text(CSV_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(csv_path)


def _write_profiles(tmp_path: Path, profiles: List[Dict[str, Any]]) -> str:
    """Write an Ontology_Profiles JSONL file (one JSON object per line)."""
    profiles_path = tmp_path / "profiles.jsonl"
    with open(profiles_path, "w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile) + "\n")
    return str(profiles_path)


def _read_output(out_path: str) -> List[Dict[str, Any]]:
    """Read the emitted JSONL back into a list of records (one per line)."""
    records: List[Dict[str, Any]] = []
    with open(out_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_report(out_path: str) -> Dict[str, int]:
    """Read the sibling conversion_report.json for the emitted output."""
    report_path = Path(out_path).parent / "conversion_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


# ===========================================================================
# Req 1.4 — join each CVE_Record to its Ontology_Profile by `cve`
# ===========================================================================

def test_join_by_cve_pulls_ontology_from_matching_profile(tmp_path):
    """Req 1.4: each row is joined to its profile by the `cve` identifier."""
    csv_path = _write_csv(
        tmp_path,
        [
            "CVE-2024-0001,90.0,critical,True,False,acme,widget,First vuln,CWE-79,,2024-01-01",
            "CVE-2024-0002,10.0,low,False,False,globex,gadget,Second vuln,CWE-89,,2024-02-01",
        ],
    )
    profiles_path = _write_profiles(
        tmp_path,
        [
            {
                "cve": "CVE-2024-0001",
                "cwes": ["CWE-79", "CWE-80"],
                "cpes": ["cpe:2.3:a:acme:widget:*:*:*:*:*:*:*:*"],
                "vendors": ["acme"],
            },
            {
                "cve": "CVE-2024-0002",
                "cwes": ["CWE-89"],
                "cpes": ["cpe:2.3:a:globex:gadget:*:*:*:*:*:*:*:*"],
                "vendors": ["globex"],
            },
        ],
    )
    out_path = str(tmp_path / "out.jsonl")

    report = CVEDataConverter().convert(csv_path, profiles_path, out_path)

    records = _read_output(out_path)
    by_cve = {rec["cve"]: rec for rec in records}

    assert report.input_rows == 2
    assert report.emitted_records == 2
    assert report.missing_profile_count == 0

    # The ontology object is joined from the matching profile (not the CSV),
    # so it carries the profile's richer CWE/CPE/vendor lists.
    assert by_cve["CVE-2024-0001"]["ontology"] == {
        "cwes": ["CWE-79", "CWE-80"],
        "cpes": ["cpe:2.3:a:acme:widget:*:*:*:*:*:*:*:*"],
        "vendors": ["acme"],
    }
    assert by_cve["CVE-2024-0002"]["ontology"]["cwes"] == ["CWE-89"]


# ===========================================================================
# Req 1.5 — missing-profile join emits a CSV-only view and records the condition
# ===========================================================================

def test_missing_profile_emits_csv_only_view_and_counts(tmp_path):
    """Req 1.5: a row with no matching profile is emitted from CSV fields only."""
    csv_path = _write_csv(
        tmp_path,
        [
            "CVE-2024-0001,90.0,critical,True,False,acme,widget,Has profile,CWE-79,,2024-01-01",
            # No profile exists for CVE-2024-9999.
            "CVE-2024-9999,50.0,medium,False,False,orphan,thing,No profile,CWE-22;CWE-23,"
            "cpe:2.3:a:orphan:thing:*:*:*:*:*:*:*:*,2024-03-01",
        ],
    )
    profiles_path = _write_profiles(
        tmp_path,
        [
            {
                "cve": "CVE-2024-0001",
                "cwes": ["CWE-79"],
                "cpes": [],
                "vendors": ["acme"],
            }
        ],
    )
    out_path = str(tmp_path / "out.jsonl")

    report = CVEDataConverter().convert(csv_path, profiles_path, out_path)

    records = _read_output(out_path)
    by_cve = {rec["cve"]: rec for rec in records}

    # Both rows are emitted; the orphan is CSV-only and counted once.
    assert report.input_rows == 2
    assert report.emitted_records == 2
    assert report.missing_profile_count == 1

    orphan = by_cve["CVE-2024-9999"]
    # The CSV-only view is non-empty and carries the cve segment.
    assert orphan["encoder_view"].startswith("CVE-2024-9999")
    # The ontology is derived from CSV fields (cwes ";"-split, cpes_sample, vendor_project).
    assert orphan["ontology"] == {
        "cwes": ["CWE-22", "CWE-23"],
        "cpes": ["cpe:2.3:a:orphan:thing:*:*:*:*:*:*:*:*"],
        "vendors": ["orphan"],
    }


# ===========================================================================
# Req 1.8 — malformed / unparseable rows and missing/empty `cve` are skipped
# ===========================================================================

def test_missing_and_empty_cve_rows_are_skipped_and_counted(tmp_path):
    """Req 1.8: rows with a missing/empty/whitespace `cve` are skipped + counted."""
    csv_path = _write_csv(
        tmp_path,
        [
            "CVE-2024-0001,90.0,critical,True,False,acme,widget,Valid,CWE-79,,2024-01-01",
            # Empty cve field -> skipped.
            ",10.0,low,False,False,globex,gadget,Empty cve,CWE-89,,2024-02-01",
            # Whitespace-only cve field -> skipped (trimmed length < 1).
            "   ,20.0,medium,False,False,initech,tps,Whitespace cve,CWE-20,,2024-02-15",
            "CVE-2024-0002,30.0,high,False,False,umbrella,corp,Also valid,CWE-22,,2024-03-01",
        ],
    )
    profiles_path = _write_profiles(tmp_path, [])
    out_path = str(tmp_path / "out.jsonl")

    report = CVEDataConverter().convert(csv_path, profiles_path, out_path)

    records = _read_output(out_path)
    emitted_cves = {rec["cve"] for rec in records}

    # Conversion continues past the skipped rows and emits the two valid ones.
    assert emitted_cves == {"CVE-2024-0001", "CVE-2024-0002"}
    assert report.input_rows == 4
    assert report.emitted_records == 2
    assert report.skipped_row_count == 2
    # Missing/empty cve is the specific reason for both skips.
    assert report.missing_or_empty_field_count == 2


def test_short_row_with_absent_cve_column_is_skipped(tmp_path):
    """Req 1.8: a ragged/short row whose `cve` cell parses to empty is skipped."""
    # A data line with only a couple of fields; DictReader fills the rest with
    # None. Here the first cell (cve) is empty, so the row is skipped.
    csv_path = _write_csv(
        tmp_path,
        [
            "CVE-2024-0001,90.0,critical,True,False,acme,widget,Valid,CWE-79,,2024-01-01",
            ",,",  # malformed short row: empty cve + truncated columns
        ],
    )
    profiles_path = _write_profiles(tmp_path, [])
    out_path = str(tmp_path / "out.jsonl")

    report = CVEDataConverter().convert(csv_path, profiles_path, out_path)

    assert report.input_rows == 2
    assert report.emitted_records == 1
    assert report.skipped_row_count == 1
    assert report.missing_or_empty_field_count == 1
    assert {rec["cve"] for rec in _read_output(out_path)} == {"CVE-2024-0001"}


# ===========================================================================
# Req 1.9 — records failing the Domain_Adapter validate() hook are excluded
# ===========================================================================

class _RejectingCVEAdapter:
    """Test-only Domain_Adapter that rejects a configured set of `cve`s.

    Exposes the record-level ``validate_view_record`` hook the converter routes
    through (``CVEDataConverter._resolve_validator``), so this exercises the real
    Domain_Adapter_Seam path rather than the built-in fallback predicate.
    """

    name = "reject-cve"
    reject_cves: set = set()

    def __init__(self, config: Any) -> None:
        self.config = config

    def validate_view_record(self, record: Mapping[str, Any]) -> bool:
        return record.get("cve") not in self.reject_cves


@pytest.fixture
def rejecting_adapter():
    """Register the rejecting adapter under 'reject-cve' and restore the registry."""
    original = dict(domain_adapters._REGISTRY)
    _RejectingCVEAdapter.reject_cves = set()
    domain_adapters.register_domain_adapter("reject-cve", _RejectingCVEAdapter)
    try:
        yield _RejectingCVEAdapter
    finally:
        domain_adapters._REGISTRY.clear()
        domain_adapters._REGISTRY.update(original)


def test_forced_validation_failure_excludes_record_and_counts(tmp_path, rejecting_adapter):
    """Req 1.9: a record failing validate() is excluded from output and counted."""
    rejecting_adapter.reject_cves = {"CVE-2024-BAD1"}

    csv_path = _write_csv(
        tmp_path,
        [
            "CVE-2024-GOOD,90.0,critical,True,False,acme,widget,Kept,CWE-79,,2024-01-01",
            "CVE-2024-BAD1,50.0,medium,False,False,orphan,thing,Rejected,CWE-22,,2024-03-01",
        ],
    )
    profiles_path = _write_profiles(tmp_path, [])
    out_path = str(tmp_path / "out.jsonl")

    config = CVEConvertConfig(domain_adapter="reject-cve")
    report = CVEDataConverter(config).convert(csv_path, profiles_path, out_path)

    emitted_cves = {rec["cve"] for rec in _read_output(out_path)}

    # The rejected record is excluded; the valid one is still emitted.
    assert emitted_cves == {"CVE-2024-GOOD"}
    assert report.input_rows == 2
    assert report.emitted_records == 1
    assert report.validation_failure_count == 1


def test_conversion_report_accounting_invariant_holds(tmp_path, rejecting_adapter):
    """Req 1.8/1.9: input_rows == emitted + skipped + duplicate + validation_failure.

    Combines a valid row, a duplicate, a skipped (empty cve) row, and a
    validation-rejected row, and confirms the four partitioning counts add up.
    """
    rejecting_adapter.reject_cves = {"CVE-2024-BAD1"}

    csv_path = _write_csv(
        tmp_path,
        [
            "CVE-2024-GOOD,90.0,critical,True,False,acme,widget,Kept,CWE-79,,2024-01-01",
            "CVE-2024-GOOD,91.0,critical,True,False,acme,widget,Dup,CWE-79,,2024-01-02",
            ",10.0,low,False,False,globex,gadget,Empty cve,CWE-89,,2024-02-01",
            "CVE-2024-BAD1,50.0,medium,False,False,orphan,thing,Rejected,CWE-22,,2024-03-01",
        ],
    )
    profiles_path = _write_profiles(tmp_path, [])
    out_path = str(tmp_path / "out.jsonl")

    config = CVEConvertConfig(domain_adapter="reject-cve")
    report = CVEDataConverter(config).convert(csv_path, profiles_path, out_path)

    assert report.input_rows == 4
    assert report.emitted_records == 1
    assert report.duplicate_cve_count == 1
    assert report.skipped_row_count == 1
    assert report.validation_failure_count == 1
    assert report.input_rows == (
        report.emitted_records
        + report.skipped_row_count
        + report.duplicate_cve_count
        + report.validation_failure_count
    )

    # The report is also persisted to conversion_report.json (Req 1.10).
    persisted = _read_report(out_path)
    assert persisted["emitted_records"] == 1
    assert persisted["validation_failure_count"] == 1
