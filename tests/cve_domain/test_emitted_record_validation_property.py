"""Property-based test for emitted-record adapter validation (Requirement 1.6).

Component under test: ``CVEDataConverter.convert`` in ``cve_domain/data_converter.py``,
together with the CVE validate predicate ``validate_cve_view_record`` (which mirrors
``CVERecordAdapter.validate``: a CVE_View_Record is valid iff it carries a non-empty
``encoder_view`` and a non-empty ``cve``).

Strategy: generate synthetic CSV rows into a temp file (with varying presence/absence
of every field, including rows whose ``cve`` is missing/empty and rows that are
otherwise sparse), run the converter, read back the emitted JSONL, and assert that
*every* emitted record passes the configured Domain_Adapter's ``validate()`` hook
(non-empty ``encoder_view`` + non-empty ``cve``).

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_emitted_record_validation_property.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from cve_domain.data_converter import (
    CVEDataConverter,
    validate_cve_view_record,
)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

# Values that represent an ABSENT field: None becomes "" in a CSV cell, and
# empty / whitespace-only strings all have a trimmed length < 1.
ABSENT_VALUES = ["", "   ", "\t", " \n "]

_SAFE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"

# The CSV columns the converter reads. Every generated row carries exactly these
# keys so ``csv.DictWriter`` produces a uniform table.
COLUMNS = [
    "cve",
    "vulnerability_name",
    "short_description",
    "cwes",
    "cvss_base_score",
    "cvss_base_severity",
    "epss",
    "in_kev",
    "known_ransomware_campaign_use",
    "vendor_project",
    "product",
    "cpes_sample",
    "priority_score",
    "priority_band",
    "nvd_published",
]


def _optional(prefix: str) -> st.SearchStrategy:
    """A field that is either absent (empty/whitespace) or a present token."""
    body = st.text(alphabet=_SAFE_ALPHABET, min_size=1, max_size=12).map(lambda s: prefix + s)
    return st.one_of(st.sampled_from(ABSENT_VALUES), body)


def _cve_field() -> st.SearchStrategy:
    """A ``cve`` cell: present identifier, or an absent (empty/whitespace) value.

    Absent values exercise the skip-row path (Req 1.8) so those rows are never
    emitted, while present identifiers become emitted records that must validate.
    """
    present = st.tuples(
        st.integers(min_value=1990, max_value=2035),
        st.integers(min_value=1, max_value=999999),
    ).map(lambda t: f"CVE-{t[0]}-{t[1]}")
    padded = present.map(lambda s: f"  {s}  ")
    return st.one_of(present, padded, st.sampled_from(ABSENT_VALUES))


@st.composite
def _cve_rows(draw) -> dict:
    """Generate one synthetic CSV row with varying presence/absence of fields."""
    return {
        "cve": draw(_cve_field()),
        "vulnerability_name": draw(_optional("VN")),
        "short_description": draw(_optional("SD")),
        "cwes": draw(_optional("CWE-")),
        "cvss_base_score": draw(_optional("SCORE")),
        "cvss_base_severity": draw(_optional("SEV")),
        "epss": draw(_optional("EP")),
        "in_kev": draw(st.sampled_from(["true", "false", "", "unknown", "known"])),
        "known_ransomware_campaign_use": draw(
            st.sampled_from(["true", "false", "", "unknown", "known"])
        ),
        "vendor_project": draw(_optional("VENDOR")),
        "product": draw(_optional("PROD")),
        "cpes_sample": draw(_optional("cpe")),
        "priority_score": draw(st.sampled_from(["", "42.5", "150", "abc"])),
        "priority_band": draw(st.sampled_from(["", "critical", "high", "low"])),
        "nvd_published": draw(st.sampled_from(["", "2024-03-04T18:15:09"])),
    }


def _write_csv(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Property 7
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 7: Every emitted record passes the configured Domain_Adapter's validate() hook (non-empty encoder_view + cve)
@settings(max_examples=150, deadline=None)
@given(rows=st.lists(_cve_rows(), min_size=0, max_size=25))
def test_every_emitted_record_passes_validate(rows, tmp_path_factory):
    """Property 7 (Validates: Requirements 1.6).

    For any set of synthetic CSV rows, every CVE_View_Record the converter emits
    passes the configured Domain_Adapter's ``validate()`` hook: it carries a
    non-empty ``encoder_view`` and a non-empty ``cve`` identifier.
    """
    workdir = tmp_path_factory.mktemp("cve_convert")
    csv_path = workdir / "cves.csv"
    profiles_path = workdir / "profiles.jsonl"
    out_path = workdir / "view_records.jsonl"

    _write_csv(csv_path, rows)
    # No ontology profiles: exercises the CSV-only join path (Req 1.5). An empty
    # file is a valid (zero-line) JSONL profile index.
    profiles_path.write_text("", encoding="utf-8")

    report = CVEDataConverter().convert(str(csv_path), str(profiles_path), str(out_path))

    emitted = []
    with open(out_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            emitted.append(json.loads(line))

    # The output file must contain exactly the reported emitted-record count.
    assert len(emitted) == report.emitted_records

    for record in emitted:
        # Every emitted record passes the adapter's validate() hook (Req 1.6).
        assert validate_cve_view_record(record), (
            f"emitted record failed adapter validation: {record!r}"
        )
        # Restate the hook's contract explicitly for a clear failure signal.
        assert isinstance(record.get("cve"), str) and record["cve"].strip()
        assert isinstance(record.get("encoder_view"), str) and record["encoder_view"].strip()
