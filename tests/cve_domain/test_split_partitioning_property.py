"""Property-based test for CVE Data_Splitter partitioning (Property 8).

Component under test: ``CVEDataSplitter.split_records`` in
``cve_domain/data_splitter.py``.

This test uses Hypothesis to generate lists of synthetic ``CVE_View_Record``
dicts (each carrying ``cve``, ``nvd_published``, and ``cve_labels.priority_band``),
runs the splitter over a ``tmp_path`` output directory across all three strategies
(``stratified``, ``temporal``, ``random``), and asserts Property 8:

    For any input records, the train/validation/test splits are
      * disjoint (no record appears in more than one split),
      * covering (every input record is assigned to exactly one split, none dropped
        and none duplicated), and
      * correctly sized (split sizes match the configured proportions within integer
        rounding).

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_split_partitioning_property.py
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cve_domain.data_splitter import SPLIT_NAMES, CVEDataSplitter


# ---------------------------------------------------------------------------
# Generation helpers
#
# Each synthetic CVE_View_Record carries a UNIQUE ``cve`` identifier (so records
# can be counted and traced unambiguously across splits), an ``nvd_published``
# value that is sometimes a valid ISO-8601 timestamp and sometimes missing/blank
# (to exercise the temporal undated→train path), and a ``cve_labels.priority_band``
# that is sometimes one of a small band pool and sometimes missing/empty (to
# exercise the stratified unbanded stratum).
# ---------------------------------------------------------------------------

_BANDS = ["critical", "high", "medium", "low", None, ""]

_PUBLISHED = [
    "2024-03-04T18:15:09.377",
    "2023-01-01T00:00:00",
    "2020-12-31T23:59:59.999",
    "1999-07-15T12:00:00",
    "2024-03-04T18:15:09.377Z",
    "",       # blank → undated
    None,     # missing → undated
    "not-a-date",  # unparseable → undated
]


@st.composite
def cve_records(draw) -> list:
    """Draw a list of unique-``cve`` CVE_View_Record dicts."""
    n = draw(st.integers(min_value=0, max_value=60))
    records = []
    for i in range(n):
        band = draw(st.sampled_from(_BANDS))
        cve_labels = {}
        if band is not None:
            cve_labels["priority_band"] = band
        records.append(
            {
                "cve": f"CVE-TEST-{i}",  # unique per record
                "nvd_published": draw(st.sampled_from(_PUBLISHED)),
                "cve_labels": cve_labels,
            }
        )
    return records


def _expected_sizes(n: int, proportions: dict) -> dict:
    """Re-derive the expected per-split sizes using the same cumulative-rounding
    rule the splitter uses in ``_cut_by_proportions`` (the final split absorbs the
    remainder)."""
    total = sum(proportions.values()) or 1.0
    fracs = [proportions[name] / total for name in SPLIT_NAMES]
    sizes = {}
    cumulative_frac = 0.0
    prev_boundary = 0
    for i, name in enumerate(SPLIT_NAMES):
        cumulative_frac += fracs[i]
        if i == len(SPLIT_NAMES) - 1:
            boundary = n
        else:
            boundary = int(round(cumulative_frac * n))
            boundary = max(prev_boundary, min(boundary, n))
        sizes[name] = boundary - prev_boundary
        prev_boundary = boundary
    return sizes


def _read_split_cves(output_dir: Path) -> dict:
    """Read the emitted per-split JSONL files back into {split: [cve, ...]}."""
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
# Property 8
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 8: Splits partition the input (disjoint, covering, correctly sized)
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(records=cve_records(), strategy=st.sampled_from(["stratified", "temporal", "random"]))
def test_splits_partition_the_input(records, strategy, tmp_path):
    """Property 8 (Validates: Requirements 4.1, 4.4).

    For any input records, the train/validation/test splits are disjoint, cover all
    input records exactly once, and are sized according to the configured proportions
    within integer rounding.
    """
    proportions = {"train": 80.0, "validation": 10.0, "test": 10.0}
    output_dir = tmp_path / f"splits_{strategy}"

    splitter = CVEDataSplitter(strategy=strategy, proportions=proportions, seed=42)
    report = splitter.split_records(records, str(output_dir))

    n = len(records)
    input_cves = [r["cve"] for r in records]

    # Small inputs can legitimately trigger the empty-split guard (Req 4.9): with
    # 80/10/10 proportions any split can be empty when n is small. In that case the
    # splitter stops before writing split artifacts, so partitioning does not apply.
    if report.status != "ok":
        assert report.status == "empty_split"
        return

    split_cves = _read_split_cves(output_dir)

    # --- Covering: every input cve assigned exactly once (Req 4.4) -----------
    all_assigned = [cve for name in SPLIT_NAMES for cve in split_cves[name]]
    assert len(all_assigned) == n, "every input record must be assigned to a split (none dropped)"
    assert sorted(all_assigned) == sorted(input_cves), (
        "the union of the splits must equal the exact input set of records"
    )

    # --- Disjoint: no record appears in more than one split ------------------
    assert len(all_assigned) == len(set(all_assigned)), "no record may appear in more than one split"

    # Cross-check the report's per-split counts against the written artifacts.
    for name in SPLIT_NAMES:
        assert report.per_split_counts[name] == len(split_cves[name])

    # --- Correctly sized within integer rounding (Req 4.1) -------------------
    # The stratified strategy rounds PER band group, so the aggregate split sizes
    # can differ from the single-cut rounding by at most one record per band group.
    if strategy in ("random", "temporal"):
        # Temporal places undated records into train on top of the proportion cut,
        # so account for those explicitly.
        undated = sum(1 for r in records if _is_undated(r))
        if strategy == "temporal":
            dated_sizes = _expected_sizes(n - undated, proportions)
            expected = dict(dated_sizes)
            expected["train"] += undated
        else:
            expected = _expected_sizes(n, proportions)
        for name in SPLIT_NAMES:
            assert report.per_split_counts[name] == expected[name], (
                f"{strategy} split {name}: expected {expected[name]}, "
                f"got {report.per_split_counts[name]}"
            )
    else:  # stratified — allow per-band rounding slack
        # Count how many distinct band groups exist (each rounds independently).
        band_keys = set()
        for r in records:
            labels = r.get("cve_labels") or {}
            band = labels.get("priority_band")
            band = str(band).strip() if band is not None else ""
            band_keys.add(band if band else "__unbanded__")
        num_groups = max(len(band_keys), 1)

        ideal = {name: proportions[name] / 100.0 * n for name in SPLIT_NAMES}
        for name in SPLIT_NAMES:
            # Each band group contributes at most ~1 record of rounding error.
            assert abs(report.per_split_counts[name] - ideal[name]) <= num_groups + 1, (
                f"stratified split {name}: {report.per_split_counts[name]} deviates from "
                f"ideal {ideal[name]:.2f} by more than the per-band rounding tolerance"
            )


def _is_undated(record) -> bool:
    """Mirror the splitter's temporal date-parse: undated when missing/blank/unparseable."""
    from cve_domain.data_splitter import CVEDataSplitter as _S

    return _S._parse_published(record.get("nvd_published")) is None
