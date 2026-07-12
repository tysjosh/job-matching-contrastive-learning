"""Property-based test for the CVE Data_Converter first-wins dedup accounting.

Component under test: ``CVEDataConverter.convert`` in ``cve_domain/data_converter.py``.

This test uses Hypothesis to generate synthetic CSV rows (with duplicate ``cve``
identifiers, leading/trailing-whitespace variants of the same identifier, and rows
whose ``cve`` is missing/empty) plus a profiles JSONL file, runs the converter over
temp files, and reads back the emitted JSONL and ``conversion_report.json`` to assert
Property 1.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_first_wins_dedup_property.py
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from cve_domain.data_converter import CVEDataConverter


# ---------------------------------------------------------------------------
# Generation helpers
#
# We draw a list of CSV rows. Each row's ``cve`` is one of:
#   - a "clean" identifier drawn from a small pool (to force duplicates),
#   - a whitespace-wrapped variant of a pooled identifier (same trimmed id — a
#     duplicate that must collapse to the first occurrence), or
#   - a missing/empty/whitespace-only value (a skip, never a duplicate).
# Each row also carries a UNIQUE ``nvd_published`` sequence marker so the emitted
# record can be traced back to the exact input row it came from, which is how we
# verify the "first occurrence wins" rule.
# ---------------------------------------------------------------------------

# Small pool of base identifiers so duplicates arise frequently.
_CVE_POOL = ["CVE-2024-1", "CVE-2024-2", "CVE-2023-99", "CVE-2020-5", "CVE-1999-7"]

# Whitespace decorations that leave the trimmed identifier unchanged.
_WS_WRAPS = ["  {0}", "{0}  ", " {0} ", "\t{0}", "{0}\n", "  {0}\t "]

# Values that represent an ABSENT cve: trimmed length < 1 (skipped, not a dup).
_ABSENT = ["", "   ", "\t", " \n ", None]


def _cve_value() -> st.SearchStrategy:
    clean = st.sampled_from(_CVE_POOL)
    wrapped = st.tuples(st.sampled_from(_CVE_POOL), st.sampled_from(_WS_WRAPS)).map(
        lambda t: t[1].format(t[0])
    )
    absent = st.sampled_from(_ABSENT)
    # Weight toward present values so most examples exercise real dedup.
    return st.one_of(clean, clean, wrapped, wrapped, absent)


@st.composite
def csv_rows(draw) -> list:
    n = draw(st.integers(min_value=0, max_value=30))
    rows = []
    for i in range(n):
        rows.append(
            {
                "cve": draw(_cve_value()),
                # Present free-text so the encoder_view is non-empty regardless,
                # though the cve segment alone already guarantees that.
                "vulnerability_name": f"vuln-{i}",
                # Unique per-row marker used to trace first-occurrence provenance.
                "nvd_published": str(i),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Independent re-derivation of the spec rules (the "model")
# ---------------------------------------------------------------------------

def _trimmed_nonempty(value) -> bool:
    return value is not None and len(str(value).strip()) >= 1


def _write_inputs(rows: list, csv_path: Path, profiles_path: Path) -> None:
    fieldnames = ["cve", "vulnerability_name", "nvd_published"]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # DictWriter serializes None -> "" which reads back as an empty (absent) cve.
            writer.writerow(row)
    # Property 1 concerns dedup accounting only; an empty profiles file is fine
    # (every row takes the CSV-only join path). The file must still exist.
    profiles_path.write_text("", encoding="utf-8")


def _read_emitted(out_path: Path) -> list:
    records = []
    with open(out_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Property 1
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 1: Converter emits exactly the unique valid CVEs (first-wins)
@settings(max_examples=200, deadline=None)
@given(rows=csv_rows())
def test_converter_emits_unique_valid_cves_first_wins(rows):
    """Property 1 (Validates: Requirements 1.1, 1.3, 1.7).

    For any set of CSV rows, the converter emits exactly one record per unique
    whitespace-trimmed non-empty ``cve`` identifier, keeping the first occurrence and
    discarding later duplicates, and the reported ``duplicate_cve_count`` equals the
    number of discarded later occurrences.
    """
    # --- Model: what the converter SHOULD do --------------------------------
    valid = [
        (i, str(r["cve"]).strip())
        for i, r in enumerate(rows)
        if _trimmed_nonempty(r["cve"])
    ]
    first_occurrence: dict = {}
    for row_index, cve in valid:
        if cve not in first_occurrence:
            first_occurrence[cve] = row_index  # first-wins provenance
    expected_unique = set(first_occurrence.keys())
    expected_duplicates = len(valid) - len(first_occurrence)

    # --- Run the converter over temp files ----------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "cves.csv"
        profiles_path = tmp_path / "profiles.jsonl"
        out_path = tmp_path / "view_records.jsonl"

        _write_inputs(rows, csv_path, profiles_path)

        report = CVEDataConverter().convert(
            str(csv_path), str(profiles_path), str(out_path)
        )

        emitted = _read_emitted(out_path)
        report_json = json.loads(
            (tmp_path / "conversion_report.json").read_text(encoding="utf-8")
        )

    # --- Assertions ----------------------------------------------------------
    emitted_cves = [rec["cve"] for rec in emitted]

    # Req 1.1 / 1.3: exactly one emitted record per unique trimmed non-empty cve,
    # each carrying its (trimmed) cve identifier.
    assert len(emitted_cves) == len(set(emitted_cves)), "no duplicate cve in the output"
    assert set(emitted_cves) == expected_unique, (
        "emitted cves must be exactly the unique trimmed non-empty identifiers"
    )
    assert len(emitted) == len(expected_unique)

    # Req 1.7 (first-wins): each emitted record must come from the FIRST occurrence
    # of its cve, verified via the unique per-row nvd_published provenance marker.
    by_cve = {rec["cve"]: rec for rec in emitted}
    for cve, row_index in first_occurrence.items():
        assert by_cve[cve]["nvd_published"] == str(row_index), (
            f"record for {cve} must originate from its first occurrence (row {row_index})"
        )

    # Report accounting agrees with the emitted output and the model.
    assert report.emitted_records == len(expected_unique)
    assert report_json["emitted_records"] == len(expected_unique)

    # Req 1.7: duplicate_cve_count == number of discarded LATER occurrences.
    assert report.duplicate_cve_count == expected_duplicates
    assert report_json["duplicate_cve_count"] == expected_duplicates
