"""Property-based test for the CVE conversion report accounting (Requirement 1).

Component under test: ``CVEDataConverter.convert`` in ``cve_domain/data_converter.py``.

This test drives the full converter over synthetic CSV inputs (written to temp files
via ``tmp_path``) containing a mix of valid rows, duplicate ``cve`` identifiers,
rows with a missing / empty ``cve``, and rows that fail the configured
Domain_Adapter's ``validate()`` hook. It then reads the on-disk
``conversion_report.json`` and asserts Property 2:

    input_rows == emitted_records + skipped_row_count
                  + duplicate_cve_count + validation_failure_count

and that every count enumerated in Requirement 1.10 is present in the report.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_conversion_report_property.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Mapping

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cve_domain.data_converter import (
    CVEConvertConfig,
    CVEDataConverter,
    validate_cve_view_record,
)


# ---------------------------------------------------------------------------
# A converter whose validate() hook additionally rejects any CVE_View_Record
# whose cve carries the "REJECT" marker. This lets the generator drive the
# Req 1.9 validation-failure branch deterministically (the built-in CVE
# predicate alone never fails for a present cve, since the view always carries
# the cve segment). It stands in for a configured Domain_Adapter that fails
# validation, exercising validation_failure_count in the accounting invariant.
# ---------------------------------------------------------------------------
_REJECT_MARKER = "REJECT"


class _RejectMarkedConverter(CVEDataConverter):
    def _resolve_validator(self):
        def validator(record: Mapping) -> bool:
            if not validate_cve_view_record(record):
                return False
            return _REJECT_MARKER not in str(record.get("cve", ""))

        return validator


# ---------------------------------------------------------------------------
# CSV column set (a realistic subset; only ``cve`` gates skip/dedup/validate).
# ---------------------------------------------------------------------------
_COLUMNS = [
    "cve",
    "vulnerability_name",
    "priority_score",
    "priority_band",
    "in_kev",
    "known_ransomware_campaign_use",
    "cwes",
    "nvd_published",
]


# Row-kind directives the generator emits; the test materializes each into a
# concrete CSV row using a per-row unique counter so duplicates are controlled.
_ROW_KINDS = ["valid", "dup", "empty_blank", "empty_ws", "reject"]


def _make_row(kind: str, index: int, last_valid_cve: str | None) -> dict:
    """Build one concrete CSV row for a directive ``kind``.

    - valid: a fresh, unique, present cve
    - reject: a fresh present cve carrying the REJECT marker (fails validation)
    - dup: repeats the most recent successfully-emittable valid cve (or a fresh
      valid cve when none exists yet)
    - empty_blank / empty_ws: a missing / whitespace-only cve (skipped)
    """
    base = {
        "cve": "",
        "vulnerability_name": f"name-{index}",
        "priority_score": "50",
        "priority_band": "high",
        "in_kev": "true",
        "known_ransomware_campaign_use": "false",
        "cwes": "CWE-79;CWE-89",
        "nvd_published": "2024-01-01T00:00:00",
    }
    if kind == "valid":
        base["cve"] = f"CVE-2020-{index}"
    elif kind == "reject":
        base["cve"] = f"CVE-{_REJECT_MARKER}-{index}"
    elif kind == "dup":
        base["cve"] = last_valid_cve if last_valid_cve else f"CVE-2020-{index}"
    elif kind == "empty_blank":
        base["cve"] = ""
    elif kind == "empty_ws":
        base["cve"] = "   "
    return base


def _present(value) -> bool:
    return value is not None and len(str(value).strip()) >= 1


def _expected_counts(rows: List[dict]) -> dict:
    """Independently re-derive the four partitioning counts (mirrors convert()).

    Mirrors the converter's ordering: skip missing/empty cve; discard later
    duplicates of an already-*emitted* cve; validation failures are excluded and
    are NOT recorded as emitted (so their cve is never marked seen).
    """
    emitted = 0
    skipped = 0
    duplicate = 0
    validation_failure = 0
    seen_emitted: set = set()

    for row in rows:
        cve_raw = row.get("cve")
        if not _present(cve_raw):
            skipped += 1
            continue
        cve = str(cve_raw).strip()
        if cve in seen_emitted:
            duplicate += 1
            continue
        # The record built for a present cve always has a non-empty encoder_view
        # (it carries the cve segment), so it fails validation only via the
        # REJECT marker injected by _RejectMarkedConverter.
        if _REJECT_MARKER in cve:
            validation_failure += 1
            continue
        emitted += 1
        seen_emitted.add(cve)

    return {
        "input_rows": len(rows),
        "emitted_records": emitted,
        "skipped_row_count": skipped,
        "duplicate_cve_count": duplicate,
        "validation_failure_count": validation_failure,
    }


# All keys Requirement 1.10 requires the report to enumerate.
_REQUIRED_REPORT_KEYS = {
    "input_rows",
    "emitted_records",
    "missing_or_empty_field_count",
    "missing_profile_count",
    "duplicate_cve_count",
    "skipped_row_count",
    "validation_failure_count",
}


@st.composite
def csv_row_sets(draw) -> List[dict]:
    """Generate a mixed list of CSV rows: valid, duplicate, empty-cve, reject."""
    kinds = draw(st.lists(st.sampled_from(_ROW_KINDS), max_size=40))
    rows: List[dict] = []
    last_valid_cve: str | None = None
    for i, kind in enumerate(kinds):
        row = _make_row(kind, i, last_valid_cve)
        if kind in ("valid",):
            last_valid_cve = row["cve"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_profiles(path: Path, rows: List[dict]) -> None:
    """Write ontology profiles for roughly half the distinct valid cves.

    Exercising both the joined-profile and missing-profile (CSV-only) paths keeps
    ``missing_profile_count`` meaningful, though it is not part of the invariant.
    """
    written: set = set()
    with open(path, "w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            cve = str(row.get("cve", "")).strip()
            if not cve or cve in written or idx % 2 == 1:
                continue
            written.add(cve)
            handle.write(
                json.dumps({"cve": cve, "cwes": ["CWE-79"], "cpes": [], "vendors": ["acme"]})
                + "\n"
            )


# ---------------------------------------------------------------------------
# Property 2
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 2: Conversion report is an exact accounting of every input row
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(rows=csv_row_sets())
def test_conversion_report_is_exact_accounting(rows, tmp_path):
    """Property 2 (Validates: Requirements 1.8, 1.9, 1.10).

    For any set of CSV rows,
    ``input_rows == emitted_records + skipped_row_count + duplicate_cve_count
    + validation_failure_count``, and every count enumerated in Requirement 1.10
    is present in the conversion report.
    """
    csv_path = tmp_path / "input.csv"
    profiles_path = tmp_path / "profiles.jsonl"
    out_path = tmp_path / "out.jsonl"

    _write_csv(csv_path, rows)
    _write_profiles(profiles_path, rows)

    converter = _RejectMarkedConverter(CVEConvertConfig(domain_adapter="cve"))
    converter.convert(str(csv_path), str(profiles_path), str(out_path))

    # Read the on-disk conversion report (Req 1.10 artifact).
    report_path = out_path.parent / "conversion_report.json"
    assert report_path.exists(), "convert() must write conversion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # --- Req 1.10: every required count is present in the report -------------
    missing_keys = _REQUIRED_REPORT_KEYS - set(report)
    assert not missing_keys, f"conversion report missing required keys: {missing_keys}"

    # --- Property 2 invariant: exact partition of every input row ------------
    assert report["input_rows"] == (
        report["emitted_records"]
        + report["skipped_row_count"]
        + report["duplicate_cve_count"]
        + report["validation_failure_count"]
    ), "input rows must partition exactly into emitted/skipped/duplicate/validation-failure"

    # --- Cross-check each partition count against an independent derivation ---
    expected = _expected_counts(rows)
    assert report["input_rows"] == expected["input_rows"]
    assert report["emitted_records"] == expected["emitted_records"]
    assert report["skipped_row_count"] == expected["skipped_row_count"]
    assert report["duplicate_cve_count"] == expected["duplicate_cve_count"]
    assert report["validation_failure_count"] == expected["validation_failure_count"]

    # Emitted records equal the number of lines actually written (Req 1.2).
    emitted_lines = [
        ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(emitted_lines) == report["emitted_records"]
