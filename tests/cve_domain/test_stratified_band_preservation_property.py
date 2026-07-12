"""Property-based test for CVE Data_Splitter stratified band preservation (Property 15).

Component under test: ``CVEDataSplitter`` (stratified strategy — the default) in
``cve_domain/data_splitter.py``.

This test uses Hypothesis to generate lists of synthetic ``CVE_View_Record`` dicts
(each carrying a unique ``cve`` and a ``cve_labels.priority_band`` drawn from a small
band pool plus missing/empty), runs the stratified splitter over a ``tmp_path``
output directory, and asserts Property 15:

    With the stratified strategy, each split's ``priority_band`` distribution matches
    the overall distribution within integer rounding: every band group (including the
    dedicated "unbanded" stratum for records with a missing/empty band) is allocated
    across train/validation/test by the configured proportions, and the reported
    ``unbanded_count`` equals the number of records with a missing/empty band.

Because stratification cuts each band group independently by the same proportions,
the per-split count of a given band is fully determined by the group size and the
proportions (``_cut_by_proportions``). This test verifies that deterministic
allocation exactly, and additionally checks that each per-band per-split count stays
within one record of the ideal proportional share (integer-rounding tolerance).

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_stratified_band_preservation_property.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cve_domain.data_splitter import (
    SPLIT_NAMES,
    CVEDataSplitter,
)


# ---------------------------------------------------------------------------
# Generation helpers
#
# Each synthetic CVE_View_Record carries a UNIQUE ``cve`` identifier and a
# ``cve_labels.priority_band`` drawn from a small pool of real bands plus the two
# "missing/empty" forms (``None`` and ``""``) that must route to the dedicated
# unbanded stratum (Req 4.12). Keeping the band pool small ensures band groups are
# populated enough that the stratified allocation is meaningfully exercised.
# ---------------------------------------------------------------------------

_BANDS = ["critical", "high", "medium", "low", None, ""]

# The proportions under test (the data_splits_v7-style 80/10/10 default, Req 4.1).
_PROPORTIONS = {"train": 80.0, "validation": 10.0, "test": 10.0}


@st.composite
def cve_records(draw) -> list:
    """Draw a list of unique-``cve`` CVE_View_Record dicts with varied bands."""
    n = draw(st.integers(min_value=0, max_value=80))
    records = []
    for i in range(n):
        band = draw(st.sampled_from(_BANDS))
        cve_labels = {}
        # Mirror the converter contract: an omitted priority_band means the key is
        # absent; an explicit empty string is present-but-empty. Both are unbanded.
        if band is not None:
            cve_labels["priority_band"] = band
        records.append(
            {
                "cve": f"CVE-TEST-{i}",  # unique per record
                "cve_labels": cve_labels,
            }
        )
    return records


def _normalized_band(record) -> str:
    """Return the stratum key for a record: the trimmed band, or ``"unbanded"``."""
    labels = record.get("cve_labels") or {}
    band = labels.get("priority_band")
    band = str(band).strip() if band is not None else ""
    return band if band else "unbanded"


def _read_split_records(output_dir: Path) -> dict:
    """Read the emitted per-split JSONL files back into {split: [record, ...]}."""
    result = {}
    for name in SPLIT_NAMES:
        recs = []
        path = output_dir / f"{name}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        result[name] = recs
    return result


# ---------------------------------------------------------------------------
# Property 15
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 15: Stratified split preserves the Priority_Band distribution across splits
@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(records=cve_records())
def test_stratified_split_preserves_band_distribution(records, tmp_path):
    """Property 15 (Validates: Requirements 4.10, 4.11, 4.12).

    Under the stratified strategy each ``priority_band`` group (and the dedicated
    unbanded stratum) is allocated across train/validation/test by the configured
    proportions, so every split's band distribution matches the overall distribution
    within integer rounding, and the reported ``unbanded_count`` is exact.
    """
    output_dir = tmp_path / "splits_stratified"

    # strategy=None → the splitter must default to "stratified" (Req 4.10).
    splitter = CVEDataSplitter(strategy=None, proportions=_PROPORTIONS, seed=42)
    assert splitter.strategy == "stratified", "default strategy must be stratified (Req 4.10)"

    report = splitter.split_records(records, str(output_dir))

    # Small inputs can legitimately trip the empty-split guard (Req 4.9): with
    # 80/10/10 proportions and small band groups, validation/test can round to zero.
    # In that case the splitter stops before writing split artifacts, so the
    # distribution property does not apply.
    if report.status != "ok":
        assert report.status == "empty_split"
        return

    # --- Req 4.12: unbanded_count equals the number of missing/empty-band records.
    expected_unbanded = sum(1 for r in records if _normalized_band(r) == "unbanded")
    assert report.unbanded_count == expected_unbanded, (
        f"unbanded_count {report.unbanded_count} != expected {expected_unbanded}"
    )

    # Overall band-group sizes (each group, including "unbanded", is stratified).
    overall_counts = Counter(_normalized_band(r) for r in records)

    # Re-derive the deterministic per-group allocation the splitter uses. Because
    # the stratified strategy cuts each band group by ``_cut_by_proportions`` after a
    # per-band deterministic shuffle, the *count* placed in each split for a given
    # band is fully determined by the group size and the proportions.
    expected_band_dist = {name: Counter() for name in SPLIT_NAMES}
    for band, group_size in overall_counts.items():
        cuts = splitter._cut_by_proportions(group_size)
        for name in SPLIT_NAMES:
            start, end = cuts[name]
            count = end - start
            if count:
                expected_band_dist[name][band] = count

    # The report's per-split band distribution must match the deterministic cut.
    for name in SPLIT_NAMES:
        reported = Counter(report.per_split_band_distribution.get(name, {}))
        assert reported == expected_band_dist[name], (
            f"split {name}: reported band distribution {dict(reported)} != "
            f"expected {dict(expected_band_dist[name])}"
        )

    # Cross-check the written artifacts against the report's distribution (Req 4.8).
    written = _read_split_records(output_dir)
    for name in SPLIT_NAMES:
        written_dist = Counter(_normalized_band(r) for r in written[name])
        assert written_dist == expected_band_dist[name], (
            f"split {name}: written band distribution {dict(written_dist)} != "
            f"expected {dict(expected_band_dist[name])}"
        )

    # --- Req 4.11: each per-band per-split count is within integer rounding of the
    # ideal proportional share (i.e. the split preserves the overall distribution).
    total = sum(_PROPORTIONS.values()) or 1.0
    for band, group_size in overall_counts.items():
        for name in SPLIT_NAMES:
            ideal = _PROPORTIONS[name] / total * group_size
            actual = expected_band_dist[name].get(band, 0)
            assert abs(actual - ideal) <= 1.0 + 1e-9, (
                f"band {band!r} in split {name}: count {actual} deviates from ideal "
                f"{ideal:.3f} by more than integer-rounding tolerance"
            )

    # --- Every input record is accounted for exactly once across the splits, and the
    # per-band group totals are conserved (nothing dropped or duplicated by strata).
    reassembled = Counter()
    for name in SPLIT_NAMES:
        reassembled.update(expected_band_dist[name])
    assert reassembled == overall_counts, (
        f"reassembled band totals {dict(reassembled)} != overall {dict(overall_counts)}"
    )
