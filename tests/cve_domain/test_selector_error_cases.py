"""Unit tests for CVE negative-selection error / edge cases (Task 7.7).

Components under test:

* ``CVENegativeSelector`` in ``cve_domain/negative_selector.py`` — the
  skip-and-count behavior for a referenced negative id that is not present among
  the converted ``CVE_View_Records`` (Req 5.6).
* ``CVEOntologyAdapter`` in ``cve_domain/ontology_adapter.py`` — the hard-stop
  (``CVEOntologyLoadError``) when ``cve_denominator_pools.jsonl`` cannot be read
  (missing file) or one of its lines cannot be parsed as a Denominator_Pool
  (malformed JSON / non-object / missing ``cve``), with the error naming the
  offending file or line (Req 5.7).

These focused pytest unit tests complement the property-based tests
(Property 11 / Property 12) by targeting the specific error paths and the exact
report/exception accounting.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_selector_error_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cve_domain.negative_selector import CVENegativeSelector
from cve_domain.ontology_adapter import CVEOntologyAdapter, CVEOntologyLoadError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pool_lines(path: Path, lines) -> None:
    """Write raw JSONL ``lines`` (already-serialized strings) to ``path``."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pool(path: Path, anchor: str, hard=None, medium=None, easy=None) -> None:
    """Write a one-line, well-formed ``cve_denominator_pools.jsonl`` for ``anchor``."""
    line = {
        "cve": anchor,
        "hard_negatives": hard or [],
        "medium_negatives": medium or [],
        "easy_negatives": easy or [],
    }
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")


# ===========================================================================
# Req 5.6 — Missing referenced id is skipped and counted
# ===========================================================================

def test_missing_referenced_id_is_skipped_and_counted(tmp_path):
    """A pooled negative id absent from the present universe is skipped + counted.

    The anchor's hard tier references three ids, only one of which is present
    among the converted records. The two absent ids must be skipped (never
    selected) and counted in ``report.skipped_missing_id_count`` (Req 5.6).
    """
    anchor = "CVE-2024-0001"
    present_id = "CVE-2024-0002"
    missing_a = "CVE-2024-9998"
    missing_b = "CVE-2024-9999"

    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool(pools_path, anchor, hard=[present_id, missing_a, missing_b])

    adapter = CVEOntologyAdapter(str(pools_path))
    selector = CVENegativeSelector(
        ontology_adapter=adapter,
        max_negatives_per_anchor=10,
        seed=42,
    )

    # Present universe contains the anchor and the single present negative only.
    present_ids = {anchor, present_id}
    selected = selector.select_negatives(
        anchor, split_cve_ids=[anchor, present_id], present_ids=present_ids
    )

    # The two absent ids must never be selected.
    assert missing_a not in selected
    assert missing_b not in selected
    # The one present id is selectable.
    assert selected == [present_id]
    # Exactly the two absent ids were skipped and counted (Req 5.6).
    assert selector.report.skipped_missing_id_count == 2


def test_missing_ids_counted_across_all_tiers(tmp_path):
    """Missing ids are skipped + counted regardless of which tier references them."""
    anchor = "CVE-2024-1000"
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool(
        pools_path,
        anchor,
        hard=["CVE-MISS-1"],
        medium=["CVE-MISS-2", "CVE-OK-1"],
        easy=["CVE-MISS-3"],
    )

    adapter = CVEOntologyAdapter(str(pools_path))
    selector = CVENegativeSelector(
        ontology_adapter=adapter, max_negatives_per_anchor=10, seed=7
    )

    present_ids = {anchor, "CVE-OK-1"}
    selected = selector.select_negatives(
        anchor, split_cve_ids=[anchor, "CVE-OK-1"], present_ids=present_ids
    )

    assert selected == ["CVE-OK-1"]
    # Three absent ids across the three tiers were skipped and counted.
    assert selector.report.skipped_missing_id_count == 3


def test_duplicate_missing_id_counted_once_per_occurrence_first_tier(tmp_path):
    """An absent id de-duplicated within a tier is counted once (not per repeat).

    ``_filter_tier`` counts a missing id at the moment it fails the present-ids
    check, which happens before intra-tier de-duplication. A within-tier repeat of
    a *present* id is de-duplicated (not double counted); an absent id repeated in
    the tier is counted each time it is seen. This test pins that behavior for a
    single missing id appearing once per tier.
    """
    anchor = "CVE-2024-2000"
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    # Same missing id referenced once in each of hard/medium/easy.
    _write_pool(
        pools_path,
        anchor,
        hard=["CVE-GONE"],
        medium=["CVE-GONE"],
        easy=["CVE-GONE"],
    )

    adapter = CVEOntologyAdapter(str(pools_path))
    selector = CVENegativeSelector(
        ontology_adapter=adapter, max_negatives_per_anchor=5, seed=1
    )

    selected = selector.select_negatives(
        anchor, split_cve_ids=[anchor], present_ids={anchor}
    )

    assert selected == []
    # The missing id is counted once per tier occurrence (3 total).
    assert selector.report.skipped_missing_id_count == 3


def test_no_missing_ids_reports_zero_skips(tmp_path):
    """When every pooled id is present, no skips are recorded."""
    anchor = "CVE-2024-3000"
    present = ["CVE-A", "CVE-B", "CVE-C"]
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool(pools_path, anchor, hard=present)

    adapter = CVEOntologyAdapter(str(pools_path))
    selector = CVENegativeSelector(
        ontology_adapter=adapter, max_negatives_per_anchor=10, seed=42
    )

    present_ids = {anchor, *present}
    selected = selector.select_negatives(
        anchor, split_cve_ids=[anchor, *present], present_ids=present_ids
    )

    assert sorted(selected) == sorted(present)
    assert selector.report.skipped_missing_id_count == 0


def test_skipped_missing_id_count_accumulates_over_split(tmp_path):
    """Over a whole split, skip counts accumulate across anchors (Req 5.6)."""
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool_lines(
        pools_path,
        [
            json.dumps({"cve": "CVE-A", "hard_negatives": ["CVE-MISS-A", "CVE-B"]}),
            json.dumps({"cve": "CVE-B", "hard_negatives": ["CVE-MISS-B", "CVE-A"]}),
        ],
    )

    adapter = CVEOntologyAdapter(str(pools_path))
    selector = CVENegativeSelector(
        ontology_adapter=adapter, max_negatives_per_anchor=10, seed=42
    )

    records = [{"cve": "CVE-A"}, {"cve": "CVE-B"}]
    selections = selector.select_for_split(records, output_dir=str(tmp_path / "out"))

    # CVE-A and CVE-B each reference one missing id (CVE-MISS-A / CVE-MISS-B).
    assert selector.report.skipped_missing_id_count == 2
    assert selections["CVE-A"] == ["CVE-B"]
    assert selections["CVE-B"] == ["CVE-A"]

    # The skip count is persisted to the report artifact (Req 6.7 accounting).
    report_path = tmp_path / "out" / "negative_selection_report.json"
    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["skipped_missing_id_count"] == 2


# ===========================================================================
# Req 5.7 — Unreadable / unparseable pools cause a hard stop
# ===========================================================================

def test_missing_pools_file_raises_load_error_naming_file(tmp_path):
    """A missing pools file hard-stops with CVEOntologyLoadError naming the file."""
    missing_path = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(CVEOntologyLoadError) as exc_info:
        CVEOntologyAdapter(str(missing_path))

    err = exc_info.value
    # The error names the offending file and carries no line number (whole-file read).
    assert err.file_path == str(missing_path)
    assert err.line_number is None
    assert str(missing_path) in str(err)


def test_unreadable_directory_path_raises_load_error(tmp_path):
    """Pointing the adapter at a directory (not a file) hard-stops with the error."""
    dir_path = tmp_path / "a_directory"
    dir_path.mkdir()

    with pytest.raises(CVEOntologyLoadError) as exc_info:
        CVEOntologyAdapter(str(dir_path))

    assert exc_info.value.file_path == str(dir_path)
    assert exc_info.value.line_number is None


def test_malformed_json_line_raises_load_error_with_line_number(tmp_path):
    """A line that is not valid JSON hard-stops, naming file + 1-based line number."""
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool_lines(
        pools_path,
        [
            json.dumps({"cve": "CVE-A", "hard_negatives": ["CVE-B"]}),  # line 1 ok
            "{ this is not valid json",                                  # line 2 bad
            json.dumps({"cve": "CVE-C"}),                               # line 3 ok
        ],
    )

    with pytest.raises(CVEOntologyLoadError) as exc_info:
        CVEOntologyAdapter(str(pools_path))

    err = exc_info.value
    assert err.file_path == str(pools_path)
    assert err.line_number == 2
    # The message points at the offending file:line.
    assert f"{pools_path}:2" in str(err)


def test_non_object_line_raises_load_error_with_line_number(tmp_path):
    """A JSON line that is not an object (e.g. a list) hard-stops with line number."""
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool_lines(
        pools_path,
        [
            json.dumps({"cve": "CVE-A"}),          # line 1 ok
            json.dumps(["not", "an", "object"]),   # line 2 bad (JSON array)
        ],
    )

    with pytest.raises(CVEOntologyLoadError) as exc_info:
        CVEOntologyAdapter(str(pools_path))

    err = exc_info.value
    assert err.line_number == 2
    assert "not a JSON object" in err.reason


def test_missing_cve_field_raises_load_error_with_line_number(tmp_path):
    """A pool line missing the ``cve`` id hard-stops, naming the offending line."""
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool_lines(
        pools_path,
        [
            json.dumps({"cve": "CVE-A", "hard_negatives": []}),   # line 1 ok
            json.dumps({"cve": "CVE-B"}),                         # line 2 ok
            json.dumps({"hard_negatives": ["CVE-X"]}),            # line 3 bad: no cve
        ],
    )

    with pytest.raises(CVEOntologyLoadError) as exc_info:
        CVEOntologyAdapter(str(pools_path))

    err = exc_info.value
    assert err.line_number == 3
    assert "cve" in err.reason.lower()


def test_empty_cve_field_raises_load_error(tmp_path):
    """A pool line whose ``cve`` is blank/whitespace hard-stops (missing/empty cve)."""
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool_lines(
        pools_path,
        [
            json.dumps({"cve": "   ", "hard_negatives": []}),  # line 1 bad: empty cve
        ],
    )

    with pytest.raises(CVEOntologyLoadError) as exc_info:
        CVEOntologyAdapter(str(pools_path))

    assert exc_info.value.line_number == 1
    assert "cve" in exc_info.value.reason.lower()


def test_non_list_tier_value_raises_load_error(tmp_path):
    """A tier value that is not a list of strings hard-stops with the line number."""
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool_lines(
        pools_path,
        [
            json.dumps({"cve": "CVE-A", "hard_negatives": "CVE-B"}),  # str, not list
        ],
    )

    with pytest.raises(CVEOntologyLoadError) as exc_info:
        CVEOntologyAdapter(str(pools_path))

    err = exc_info.value
    assert err.line_number == 1
    assert "hard_negatives" in err.reason


def test_valid_pools_file_loads_without_error(tmp_path):
    """A well-formed pools file loads cleanly and indexes one pool per unique cve."""
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool_lines(
        pools_path,
        [
            json.dumps({"cve": "CVE-A", "hard_negatives": ["CVE-B"]}),
            json.dumps({"cve": "CVE-B", "medium_negatives": ["CVE-A"]}),
            "",  # tolerated blank separator line
        ],
    )

    adapter = CVEOntologyAdapter(str(pools_path))
    assert adapter.pool_count() == 2
    assert adapter.has_pool("CVE-A")
    assert adapter.has_pool("CVE-B")
