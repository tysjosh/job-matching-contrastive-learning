"""Tests for cross-band negative biasing (Stage 1 realignment).

Component under test: ``CVENegativeSelector`` in
``cve_domain/negative_selector.py`` with ``cross_band_negatives=True`` plus a
band lookup set via ``set_band_lookup``.

When enabled, the selector prefers negatives whose ``priority_band`` differs from
the anchor's — within every ontology tier and in the random-sample fallback —
keeping same-band candidates only as fill so pools never run dry. This mirrors
supervised-contrastive practice (negatives should be other classes) and
complements the ``priority_band`` positive signal. This file verifies:

* different-band negatives are preferred when enough exist;
* same-band candidates still fill the remainder (no deficit introduced);
* the validity invariants (unique, anchor-excluded, bounded) are preserved;
* biasing is off by default and a no-op without a band lookup;
* the report's cross/same-band counters reconcile with the selection.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_negative_cross_band_bias.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cve_domain.negative_selector import CVENegativeSelector
from cve_domain.ontology_adapter import CVEOntologyAdapter


def _write_pool(path: Path, anchor: str, hard, medium, easy) -> None:
    path.write_text(
        json.dumps(
            {
                "cve": anchor,
                "hard_negatives": list(hard),
                "medium_negatives": list(medium),
                "easy_negatives": list(easy),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _make_selector(pools_path, max_n, cross_band, band_by_id=None, seed=42):
    adapter = CVEOntologyAdapter(str(pools_path))
    selector = CVENegativeSelector(
        ontology_adapter=adapter,
        max_negatives_per_anchor=max_n,
        tier_ratios={"hard": 0.34, "medium": 0.33, "easy": 0.33},
        seed=seed,
        cross_band_negatives=cross_band,
    )
    if band_by_id is not None:
        selector.set_band_lookup(band_by_id)
    return selector


def test_cross_band_negatives_are_preferred(tmp_path):
    """Different-band negatives are chosen before same-band ones when enough exist."""
    anchor = "CVE-A"
    # 4 same-band (critical) + 4 different-band (watch) candidates in the pool.
    same_band = [f"CVE-S{i}" for i in range(4)]
    diff_band = [f"CVE-D{i}" for i in range(4)]
    pool_ids = same_band + diff_band
    pools = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool(pools, anchor, hard=pool_ids, medium=[], easy=[])

    band_by_id = {anchor: "critical"}
    band_by_id.update({c: "critical" for c in same_band})
    band_by_id.update({c: "watch" for c in diff_band})

    present = set(pool_ids) | {anchor}
    selector = _make_selector(pools, max_n=4, cross_band=True, band_by_id=band_by_id)
    selected = selector.select_negatives(anchor, list(present), present_ids=present)

    # All 4 slots should be filled by the different-band candidates.
    assert len(selected) == 4
    assert set(selected) == set(diff_band), (
        f"expected all different-band negatives, got {selected}"
    )
    assert selector.report.cross_band_selected_count == 4
    assert selector.report.same_band_selected_count == 0


def test_same_band_fills_remainder_when_diff_band_scarce(tmp_path):
    """Same-band candidates fill in when there aren't enough different-band ones."""
    anchor = "CVE-A"
    same_band = [f"CVE-S{i}" for i in range(5)]
    diff_band = ["CVE-D0", "CVE-D1"]  # only 2 different-band available
    pool_ids = same_band + diff_band
    pools = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool(pools, anchor, hard=pool_ids, medium=[], easy=[])

    band_by_id = {anchor: "critical"}
    band_by_id.update({c: "critical" for c in same_band})
    band_by_id.update({c: "watch" for c in diff_band})

    present = set(pool_ids) | {anchor}
    selector = _make_selector(pools, max_n=4, cross_band=True, band_by_id=band_by_id)
    selected = selector.select_negatives(anchor, list(present), present_ids=present)

    assert len(selected) == 4  # no deficit — same-band fills the rest
    # Both different-band candidates must be included (preferred).
    assert set(diff_band).issubset(set(selected))
    # Remainder are same-band.
    assert selector.report.cross_band_selected_count == 2
    assert selector.report.same_band_selected_count == 2


def test_off_by_default_ignores_band(tmp_path):
    """Without cross_band, selection is unchanged and band counters stay zero."""
    anchor = "CVE-A"
    pool_ids = [f"CVE-{i}" for i in range(6)]
    pools = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool(pools, anchor, hard=pool_ids, medium=[], easy=[])
    present = set(pool_ids) | {anchor}

    # Even if a band map is provided, cross_band=False must ignore it.
    band_by_id = {anchor: "critical", **{c: "watch" for c in pool_ids}}
    selector = _make_selector(pools, max_n=4, cross_band=False, band_by_id=band_by_id)
    selected = selector.select_negatives(anchor, list(present), present_ids=present)

    assert len(selected) == 4
    assert selector.report.cross_band_selected_count == 0
    assert selector.report.same_band_selected_count == 0


def test_no_band_lookup_is_noop(tmp_path):
    """cross_band=True but no band lookup behaves like the default selection."""
    anchor = "CVE-A"
    pool_ids = [f"CVE-{i}" for i in range(6)]
    pools = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool(pools, anchor, hard=pool_ids, medium=[], easy=[])
    present = set(pool_ids) | {anchor}

    selector = _make_selector(pools, max_n=4, cross_band=True, band_by_id=None)
    selected = selector.select_negatives(anchor, list(present), present_ids=present)

    # Valid selection, but nothing counted since there is no band info.
    assert len(selected) == 4
    assert anchor not in selected
    assert selector.report.cross_band_selected_count == 0
    assert selector.report.same_band_selected_count == 0


def test_invariants_preserved_with_biasing(tmp_path):
    """Unique / anchor-excluded / bounded hold with biasing on."""
    anchor = "CVE-A"
    pool_ids = [f"CVE-{i}" for i in range(20)] + [anchor]  # anchor in pool -> excluded
    pools = tmp_path / "cve_denominator_pools.jsonl"
    _write_pool(pools, anchor, hard=pool_ids[:10], medium=pool_ids[10:15], easy=pool_ids[15:])
    present = set(pool_ids)

    # Alternate bands across the candidates.
    band_by_id = {anchor: "critical"}
    for i, c in enumerate(pool_ids):
        band_by_id[c] = "watch" if i % 2 == 0 else "critical"

    selector = _make_selector(pools, max_n=7, cross_band=True, band_by_id=band_by_id)
    selected = selector.select_negatives(anchor, list(present), present_ids=present)

    assert len(selected) <= 7
    assert len(selected) == len(set(selected))
    assert anchor not in selected
    # Counters reconcile with the number of banded selections.
    counted = (
        selector.report.cross_band_selected_count
        + selector.report.same_band_selected_count
    )
    assert counted == len(selected)


def test_from_config_reads_cross_band_flag():
    class _Cfg:
        max_negatives_per_anchor = 7
        negative_tier_ratios = {"hard": 0.34, "medium": 0.33, "easy": 0.33}
        split_seed = 42
        cve_negative_cross_band = True

    class _FakeAdapter:
        pass

    sel = CVENegativeSelector.from_config(_FakeAdapter(), _Cfg())
    assert sel.cross_band_negatives is True

    class _Bare:
        max_negatives_per_anchor = 7
        negative_tier_ratios = {"hard": 0.34, "medium": 0.33, "easy": 0.33}
        split_seed = 42

    assert CVENegativeSelector.from_config(_FakeAdapter(), _Bare()).cross_band_negatives is False
