"""Property-based test for CVEPositiveSelector validity (Property 16).

Component under test: ``CVEPositiveSelector`` in
``cve_domain/positive_selector.py``.

This test uses Hypothesis to generate synthetic ``CVE_View_Records`` — each a
``{cve, ontology: {cwes, cpes, vendors}}`` object drawn from small shared token
pools so that ontology siblings actually occur — plus an optional per-anchor
``negatives_by_anchor`` exclusion map. It builds the per-split inverted index via
``select_for_split`` and asserts Property 16 for every anchor:

    For any anchor CVE with at least one in-split ontology sibling, the selected
    positive
      * shares the required ontology structure per the priority cascade
        (level 1: shares a CWE AND a CPE/vendor; level 2: shares a CWE;
         level 3: shares a vendor),
      * is in the same split as the anchor,
      * is not the anchor itself,
      * is not among the anchor's selected negatives, and
      * is identical across repeated runs under a fixed seed.
    Any anchor with no in-split ontology sibling across all cascade levels is
    excluded (returns None / omitted from the selection map) and no positive is
    fabricated.

The expected cascade level and its candidate set are recomputed independently
here (a small reference implementation over the generated records) so the test is
a genuine oracle rather than a mirror of the production code paths.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_positive_selection_validity_property16.py
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cve_domain.positive_selector import CVEPositiveSelector


# ---------------------------------------------------------------------------
# Generation helpers
#
# Small shared token pools guarantee that ontology siblings (shared CWE / CPE /
# vendor) actually occur across generated records, so every cascade level is
# exercised rather than trivially empty. Records carry unique cve ids (first-wins
# dedup is a separate concern); ontology token lists may be empty so the
# "no in-split sibling -> excluded, no fabrication" branch (Req 13.5) is hit too.
# ---------------------------------------------------------------------------

_CWE_POOL = [f"CWE-{n}" for n in range(1, 6)]        # CWE-1 .. CWE-5
_CPE_POOL = [f"cpe:{n}" for n in range(1, 5)]        # cpe:1 .. cpe:4
_VENDOR_POOL = [f"vendor{n}" for n in range(1, 5)]   # vendor1 .. vendor4

_ID_UNIVERSE = [f"CVE-2024-{n:04d}" for n in range(30)]


@st.composite
def split_inputs(draw) -> dict:
    """Draw a set of CVE_View_Records + a per-anchor negatives map + a seed."""
    # Unique anchor ids for this split.
    cve_ids: List[str] = draw(
        st.lists(st.sampled_from(_ID_UNIVERSE), min_size=1, max_size=12, unique=True)
    )

    records: List[dict] = []
    for cve in cve_ids:
        cwes = draw(st.lists(st.sampled_from(_CWE_POOL), max_size=3, unique=True))
        cpes = draw(st.lists(st.sampled_from(_CPE_POOL), max_size=3, unique=True))
        vendors = draw(st.lists(st.sampled_from(_VENDOR_POOL), max_size=3, unique=True))
        records.append(
            {
                "cve": cve,
                "encoder_view": f"{cve}. synthetic profile.",
                "ontology": {"cwes": cwes, "cpes": cpes, "vendors": vendors},
            }
        )

    # Optional negatives-by-anchor: for each anchor, exclude a subset of the other
    # ids (Req 13.4). Drawn from the same universe so exclusions are meaningful.
    negatives_by_anchor: Dict[str, List[str]] = {}
    for cve in cve_ids:
        others = [c for c in cve_ids if c != cve]
        if others:
            negs = draw(st.lists(st.sampled_from(others), max_size=4, unique=True))
            if negs:
                negatives_by_anchor[cve] = negs

    seed = draw(st.integers(min_value=0, max_value=10_000))

    return {
        "records": records,
        "negatives_by_anchor": negatives_by_anchor,
        "seed": seed,
    }


def _ontology_sets(records: List[dict]) -> Dict[str, Tuple[Set[str], Set[str], Set[str]]]:
    """Map each cve -> (cwes, cpes, vendors) as sets (independent reference)."""
    out: Dict[str, Tuple[Set[str], Set[str], Set[str]]] = {}
    for rec in records:
        onto = rec["ontology"]
        out[rec["cve"]] = (
            set(onto["cwes"]),
            set(onto["cpes"]),
            set(onto["vendors"]),
        )
    return out


def _expected_level_candidates(
    anchor: str,
    onto: Dict[str, Tuple[Set[str], Set[str], Set[str]]],
    excluded_negatives: List[str],
) -> Set[str]:
    """Recompute the anchor's expected cascade candidate set independently.

    Returns the set the selected positive must be drawn from at the first
    non-empty cascade level, or an empty set when the anchor has no in-split
    ontology sibling across all levels (Req 13.5).
    """
    anchor_cwes, anchor_cpes, anchor_vendors = onto[anchor]

    excluded: Set[str] = {anchor} | set(excluded_negatives)

    cwe_matches = {
        c for c, (cwes, _, _) in onto.items() if c not in excluded and (cwes & anchor_cwes)
    }
    cpe_matches = {
        c for c, (_, cpes, _) in onto.items() if c not in excluded and (cpes & anchor_cpes)
    }
    vendor_matches = {
        c for c, (_, _, vends) in onto.items() if c not in excluded and (vends & anchor_vendors)
    }

    # Level 1: shares a CWE AND (a CPE OR a vendor).
    level1 = cwe_matches & (cpe_matches | vendor_matches)
    if level1:
        return level1
    # Level 2: shares a CWE.
    if cwe_matches:
        return cwe_matches
    # Level 3: shares a vendor.
    if vendor_matches:
        return vendor_matches
    # No candidate at any level.
    return set()


# ---------------------------------------------------------------------------
# Property 16
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 16: Positive selection is valid and ontology-related
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(inputs=split_inputs())
def test_positive_selection_is_valid_and_ontology_related(inputs):
    """Property 16 (Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5).

    For any anchor with an in-split ontology sibling, the selected positive is
    valid (same split, not the anchor, not an excluded negative) and
    ontology-related per the priority cascade; anchors with no sibling are
    excluded with no fabrication; and selection is reproducible under a fixed seed.
    """
    records = inputs["records"]
    negatives_by_anchor = inputs["negatives_by_anchor"]
    seed = inputs["seed"]

    split_ids = {rec["cve"] for rec in records}
    onto = _ontology_sets(records)

    selector = CVEPositiveSelector(seed=seed)
    selections = selector.select_for_split(records, negatives_by_anchor=negatives_by_anchor)

    for anchor in split_ids:
        anchor_negatives = list(negatives_by_anchor.get(anchor, []))
        expected_candidates = _expected_level_candidates(anchor, onto, anchor_negatives)

        if not expected_candidates:
            # Req 13.5: no in-split ontology sibling -> excluded, no fabrication.
            assert anchor not in selections, (
                f"anchor {anchor} has no ontology sibling but a positive was "
                f"fabricated: {selections.get(anchor)!r}"
            )
            continue

        # A sibling exists: the anchor must be resolved (Req 13.1).
        assert anchor in selections, (
            f"anchor {anchor} has ontology siblings {expected_candidates} but was "
            f"not resolved"
        )
        positive = selections[anchor]

        # (b) not the anchor itself (Req 13.4).
        assert positive != anchor, "positive must not be the anchor itself"

        # (a) same split (Req 13.2): positive is one of the split's records.
        assert positive in split_ids, (
            f"positive {positive} is not in the anchor's split"
        )

        # (c) not among the anchor's selected negatives (Req 13.4).
        assert positive not in set(anchor_negatives), (
            f"positive {positive} is an excluded negative for anchor {anchor}"
        )

        # (d) ontology-related per the resolved cascade level (Req 13.1): the
        # positive must be drawn from the first non-empty cascade candidate set.
        assert positive in expected_candidates, (
            f"positive {positive} for anchor {anchor} is not in the expected "
            f"cascade candidate set {expected_candidates}"
        )

    # Reproducibility under a fixed seed (Req 13.3): a fresh selector with the same
    # seed and inputs produces an identical selection map.
    selector_again = CVEPositiveSelector(seed=seed)
    selections_again = selector_again.select_for_split(
        records, negatives_by_anchor=negatives_by_anchor
    )
    assert selections == selections_again, (
        "positive selection is not reproducible under a fixed seed"
    )
