"""Tests for priority-aware positive selection (Stage 1 realignment).

Component under test: ``CVEPositiveSelector`` in
``cve_domain/positive_selector.py``, specifically the ``cve_positive_signal``
modes ``"priority_band"`` and ``"priority_band_and_ontology"`` that realign the
Stage 1 contrastive objective onto the downstream priority target instead of the
task-orthogonal ontology-overlap signal.

These modes select a positive that shares the anchor's ``priority_band`` label
(read from ``record["cve_labels"]["priority_band"]``), so the InfoNCE objective
clusters same-band CVEs (supervised-contrastive / SupCon style). This file
verifies:

* ``priority_band``: the positive shares the band, is valid (same split, not the
  anchor, not an excluded negative); anchors with no band label or no same-band
  sibling are excluded with no fabrication; selection is reproducible.
* ``priority_band_and_ontology``: a same-band positive that also shares an
  ontology token is preferred, falling back to same-band-only.
* the default ``"ontology"`` mode never consults the band index (regression guard
  for the unchanged hot path), and an invalid signal is rejected.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_positive_selection_priority_signal.py
"""

from __future__ import annotations

from typing import Dict, List, Set

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cve_domain.positive_selector import (
    CVEPositiveSelector,
    POSITIVE_SIGNAL_ONTOLOGY,
    POSITIVE_SIGNAL_PRIORITY_BAND,
    POSITIVE_SIGNAL_PRIORITY_BAND_AND_ONTOLOGY,
)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
_BAND_POOL = ["critical", "high", "medium", "watch"]
_CWE_POOL = [f"CWE-{n}" for n in range(1, 5)]
_VENDOR_POOL = [f"vendor{n}" for n in range(1, 4)]
_ID_UNIVERSE = [f"CVE-2024-{n:04d}" for n in range(30)]


@st.composite
def priority_split_inputs(draw) -> dict:
    """Draw CVE_View_Records with priority_band labels + per-anchor negatives."""
    cve_ids: List[str] = draw(
        st.lists(st.sampled_from(_ID_UNIVERSE), min_size=1, max_size=12, unique=True)
    )

    records: List[dict] = []
    for cve in cve_ids:
        cwes = draw(st.lists(st.sampled_from(_CWE_POOL), max_size=2, unique=True))
        vendors = draw(st.lists(st.sampled_from(_VENDOR_POOL), max_size=2, unique=True))
        # priority_band is present-key-only: sometimes omitted so the
        # "no band label -> excluded" branch is exercised.
        band = draw(st.one_of(st.none(), st.sampled_from(_BAND_POOL)))
        labels: Dict[str, object] = {"in_kev": False, "ransomware": False}
        if band is not None:
            labels["priority_band"] = band
        records.append(
            {
                "cve": cve,
                "encoder_view": f"{cve}. synthetic profile.",
                "ontology": {"cwes": cwes, "cpes": [], "vendors": vendors},
                "cve_labels": labels,
            }
        )

    negatives_by_anchor: Dict[str, List[str]] = {}
    for cve in cve_ids:
        others = [c for c in cve_ids if c != cve]
        if others:
            negs = draw(st.lists(st.sampled_from(others), max_size=3, unique=True))
            if negs:
                negatives_by_anchor[cve] = negs

    return {
        "records": records,
        "negatives_by_anchor": negatives_by_anchor,
        "seed": draw(st.integers(min_value=0, max_value=10_000)),
    }


def _band_of(rec: dict) -> str:
    return str(rec.get("cve_labels", {}).get("priority_band", "")).strip()


def _onto_tokens(rec: dict) -> Set[str]:
    onto = rec.get("ontology", {})
    return set(onto.get("cwes", [])) | set(onto.get("cpes", [])) | set(onto.get("vendors", []))


# ---------------------------------------------------------------------------
# priority_band mode
# ---------------------------------------------------------------------------
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(inputs=priority_split_inputs())
def test_priority_band_positive_is_same_band_and_valid(inputs):
    """A priority_band positive shares the anchor's band and is valid.

    Anchors with no band label or no other same-band member are excluded with no
    fabrication; selection is reproducible under a fixed seed.
    """
    records = inputs["records"]
    negatives_by_anchor = inputs["negatives_by_anchor"]
    seed = inputs["seed"]

    by_id = {r["cve"]: r for r in records}
    split_ids = set(by_id)

    selector = CVEPositiveSelector(
        seed=seed, positive_signal=POSITIVE_SIGNAL_PRIORITY_BAND
    )
    selections = selector.select_for_split(
        records, negatives_by_anchor=negatives_by_anchor
    )

    for anchor in split_ids:
        anchor_band = _band_of(by_id[anchor])
        anchor_negs = set(negatives_by_anchor.get(anchor, []))
        # Independent reference: other same-band members minus self/negatives.
        expected = {
            c for c in split_ids
            if c != anchor and c not in anchor_negs and _band_of(by_id[c]) == anchor_band
        } if anchor_band else set()

        if not expected:
            assert anchor not in selections, (
                f"anchor {anchor} (band={anchor_band!r}) has no same-band sibling "
                f"but a positive was fabricated: {selections.get(anchor)!r}"
            )
            continue

        assert anchor in selections, (
            f"anchor {anchor} has same-band siblings {expected} but was not resolved"
        )
        positive = selections[anchor]
        assert positive != anchor
        assert positive in split_ids
        assert positive not in anchor_negs
        assert _band_of(by_id[positive]) == anchor_band, (
            f"positive {positive} band {_band_of(by_id[positive])!r} != anchor "
            f"band {anchor_band!r}"
        )
        assert positive in expected

    # Reproducibility under a fixed seed.
    again = CVEPositiveSelector(
        seed=seed, positive_signal=POSITIVE_SIGNAL_PRIORITY_BAND
    )
    assert selections == again.select_for_split(
        records, negatives_by_anchor=negatives_by_anchor
    )


# ---------------------------------------------------------------------------
# priority_band_and_ontology mode
# ---------------------------------------------------------------------------
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(inputs=priority_split_inputs())
def test_band_and_ontology_prefers_shared_token_then_falls_back(inputs):
    """The positive shares the band; a band+ontology sibling is preferred."""
    records = inputs["records"]
    negatives_by_anchor = inputs["negatives_by_anchor"]
    seed = inputs["seed"]

    by_id = {r["cve"]: r for r in records}
    split_ids = set(by_id)

    selector = CVEPositiveSelector(
        seed=seed, positive_signal=POSITIVE_SIGNAL_PRIORITY_BAND_AND_ONTOLOGY
    )
    selections = selector.select_for_split(
        records, negatives_by_anchor=negatives_by_anchor
    )

    for anchor in split_ids:
        anchor_band = _band_of(by_id[anchor])
        anchor_negs = set(negatives_by_anchor.get(anchor, []))
        band_candidates = {
            c for c in split_ids
            if c != anchor and c not in anchor_negs and _band_of(by_id[c]) == anchor_band
        } if anchor_band else set()

        if not band_candidates:
            assert anchor not in selections
            continue

        assert anchor in selections
        positive = selections[anchor]
        # Always in the same band regardless of the ontology preference.
        assert _band_of(by_id[positive]) == anchor_band
        assert positive in band_candidates

        # If any same-band candidate also shares an ontology token, the chosen
        # positive must be one of those (the preferred sub-case).
        anchor_tokens = _onto_tokens(by_id[anchor])
        band_and_onto = {
            c for c in band_candidates if _onto_tokens(by_id[c]) & anchor_tokens
        }
        if band_and_onto:
            assert positive in band_and_onto, (
                f"anchor {anchor} had band+ontology candidates {band_and_onto} but "
                f"selected {positive} which only shares the band"
            )


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------
def test_ontology_mode_ignores_band_index():
    """Default ontology mode does not build/consult the priority-band index."""
    records = [
        {"cve": "CVE-1", "ontology": {"cwes": ["CWE-1"], "cpes": [], "vendors": []},
         "cve_labels": {"priority_band": "critical"}},
        {"cve": "CVE-2", "ontology": {"cwes": ["CWE-1"], "cpes": [], "vendors": []},
         "cve_labels": {"priority_band": "watch"}},
    ]
    selector = CVEPositiveSelector(seed=42, positive_signal=POSITIVE_SIGNAL_ONTOLOGY)
    selections = selector.select_for_split(records)
    # Pairs by shared CWE despite different bands (ontology signal, unchanged).
    assert selections == {"CVE-1": "CVE-2", "CVE-2": "CVE-1"}
    assert selector._band_index == {}
    assert selector._anchor_band == {}
    assert selector.report.resolved_by_priority_band == 0


def test_priority_band_pairs_across_ontology_boundaries():
    """priority_band mode pairs same-band CVEs even with no shared ontology."""
    records = [
        {"cve": "CVE-1", "ontology": {"cwes": ["CWE-1"], "cpes": [], "vendors": []},
         "cve_labels": {"priority_band": "critical"}},
        {"cve": "CVE-2", "ontology": {"cwes": ["CWE-9"], "cpes": [], "vendors": []},
         "cve_labels": {"priority_band": "critical"}},
        {"cve": "CVE-3", "ontology": {"cwes": ["CWE-1"], "cpes": [], "vendors": []},
         "cve_labels": {"priority_band": "watch"}},
    ]
    selector = CVEPositiveSelector(
        seed=42, positive_signal=POSITIVE_SIGNAL_PRIORITY_BAND
    )
    selections = selector.select_for_split(records)
    # CVE-1 and CVE-2 are same band (critical) despite disjoint CWEs.
    assert selections["CVE-1"] == "CVE-2"
    assert selections["CVE-2"] == "CVE-1"
    # CVE-3 is the only "watch" -> no same-band sibling -> excluded.
    assert "CVE-3" not in selections


def test_invalid_positive_signal_rejected():
    with pytest.raises(ValueError):
        CVEPositiveSelector(seed=42, positive_signal="not-a-signal")


def test_from_config_reads_positive_signal():
    class _Cfg:
        split_seed = 7
        cve_positive_signal = POSITIVE_SIGNAL_PRIORITY_BAND

    selector = CVEPositiveSelector.from_config(_Cfg())
    assert selector.positive_signal == POSITIVE_SIGNAL_PRIORITY_BAND
    assert selector.seed == 7

    # Default when the flag is absent (career-domain / legacy configs).
    class _Bare:
        split_seed = 1

    assert CVEPositiveSelector.from_config(_Bare()).positive_signal == (
        POSITIVE_SIGNAL_ONTOLOGY
    )
