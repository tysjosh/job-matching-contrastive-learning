"""Property-based tests for the CVENegativeSelector fallback cascade (Property 12).

# Feature: cve-vulnerability-ranking, Property 12: Fallback cascade fills shortfall in order and stays reproducible

Property 12 (design.md): *For any* anchor CVE whose higher tiers are empty or
insufficient, the shortfall is filled first from ``medium``, then ``easy``, then
from a seeded random sample of other same-split CVE_View_Records (excluding the
anchor and already-selected identifiers), never duplicating an identifier; and for
a fixed seed the full selection is identical across repeated runs. Missing
referenced identifiers are skipped and counted.

**Validates: Requirements 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 11.3**

Requirement 5.6: referenced-but-absent negative ids are skipped and counted.
Requirement 6.1: empty/short hard tier fills from medium then easy.
Requirement 6.2: empty/short medium tier fills from easy.
Requirement 6.3: pooled negatives exhausted before a seeded same-split random draw.
Requirement 6.4: no pooled negatives -> all drawn from seeded same-split random sample.
Requirement 6.5: fewer unique candidates than max -> supply all + record deficit, no crash.
Requirement 11.3: same seed + input -> identical selection.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from hypothesis import given, settings, strategies as st

# Make the repository root importable when pytest is invoked from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cve_domain.negative_selector import (  # noqa: E402
    TIER_NAMES,
    CVENegativeSelector,
)
from cve_domain.ontology_adapter import CVEOntologyAdapter  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers to build a real CVEOntologyAdapter from generated pools.
# --------------------------------------------------------------------------- #


def _write_pools(pools: List[Dict[str, Any]], tmp_root: Path, tag: str) -> str:
    """Write a list of denominator-pool dicts to a JSONL file, return its path."""
    path = tmp_root / f"pools_{tag}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for pool in pools:
            handle.write(json.dumps(pool) + "\n")
    return str(path)


def _build_adapter(pools: List[Dict[str, Any]], tmp_root: Path, tag: str) -> CVEOntologyAdapter:
    return CVEOntologyAdapter(_write_pools(pools, tmp_root, tag))


# --------------------------------------------------------------------------- #
# Generators.
#
# We generate a "scenario": a universe of present CVE ids (the split), a set of
# clearly-absent (missing) ids, and one denominator pool per anchor whose three
# tiers draw from present ids and/or missing ids. Tiers may be empty/sparse. This
# exercises empty/sparse hard/medium tiers, missing referenced ids, and small
# candidate universes (down to a single-record split with no candidates at all).
# --------------------------------------------------------------------------- #

# Present universe ids and missing (absent) ids live in disjoint namespaces so we
# can compute, exactly, which referenced ids are absent.
_PRESENT_PREFIX = "CVE-2024-"
_MISSING_PREFIX = "CVE-MISS-"


def _present_id(i: int) -> str:
    return f"{_PRESENT_PREFIX}{i:05d}"


def _missing_id(i: int) -> str:
    return f"{_MISSING_PREFIX}{i:05d}"


@st.composite
def scenarios(draw):
    """Generate (pools, present_ids, max_n, tier_ratios, seed).

    ``pools`` carries one Denominator_Pool dict per anchor (every present id is an
    anchor). Tiers reference present ids and/or missing ids, may be empty, and may
    repeat ids (to exercise dedup and per-occurrence missing counting).
    """
    n_present = draw(st.integers(min_value=1, max_value=14))
    present_ids = [_present_id(i) for i in range(n_present)]
    present_set = set(present_ids)

    # A modest pool of absent ids that pools may reference.
    n_missing = draw(st.integers(min_value=0, max_value=6))
    missing_ids = [_missing_id(i) for i in range(n_missing)]

    # Candidate ids a tier list may draw from: present ids + missing ids. Empty
    # strings are also injected so the strip/drop-empty path is exercised.
    referable = present_ids + missing_ids + [""]

    def a_tier_list(anchor: str) -> List[str]:
        # 30% chance of an empty tier to exercise sparse/empty fallbacks.
        if draw(st.booleans()) and draw(st.booleans()):
            return []
        if not referable:
            return []
        return draw(
            st.lists(st.sampled_from(referable), min_size=0, max_size=8)
        )

    pools: List[Dict[str, Any]] = []
    for anchor in present_ids:
        pools.append(
            {
                "cve": anchor,
                "hard_negatives": a_tier_list(anchor),
                "medium_negatives": a_tier_list(anchor),
                "easy_negatives": a_tier_list(anchor),
            }
        )

    max_n = draw(st.integers(min_value=1, max_value=12))
    # Sometimes use explicit ratios (non-negative), sometimes the default.
    if draw(st.booleans()):
        tier_ratios = {
            "hard": draw(st.floats(min_value=0.0, max_value=5.0)),
            "medium": draw(st.floats(min_value=0.0, max_value=5.0)),
            "easy": draw(st.floats(min_value=0.0, max_value=5.0)),
        }
    else:
        tier_ratios = None
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    return pools, present_ids, present_set, max_n, tier_ratios, seed


def _pool_universe(
    pool: Dict[str, Any], anchor: str, present_set: Set[str]
) -> Tuple[Set[str], int]:
    """Return (pooled_candidate_set, expected_skipped_occurrences) for one anchor.

    ``pooled_candidate_set``: distinct present, non-anchor ids referenced by the
    three tiers (the ids the pooled cascade may legitimately select).
    ``expected_skipped_occurrences``: number of raw tier entries that are
    non-empty, not the anchor, and NOT present — counted per occurrence, matching
    ``CVENegativeSelector._filter_tier`` (Req 5.6).
    """
    pooled: Set[str] = set()
    skipped = 0
    for tier in TIER_NAMES:
        for raw in pool.get(f"{tier}_negatives", []):
            cid = str(raw).strip()
            if not cid or cid == anchor:
                continue
            if cid not in present_set:
                skipped += 1
                continue
            pooled.add(cid)
    return pooled, skipped


# --------------------------------------------------------------------------- #
# Property 12 — per-anchor cascade validity + aggregate skipped/deficit counts.
# --------------------------------------------------------------------------- #


@settings(max_examples=200, deadline=None)
@given(scenario=scenarios())
def test_fallback_cascade_fills_in_order_and_accounts(scenario):
    pools, present_ids, present_set, max_n, tier_ratios, seed = scenario

    with tempfile.TemporaryDirectory() as tmp:
        adapter = _build_adapter(pools, Path(tmp), "cascade")

    selector = CVENegativeSelector(
        ontology_adapter=adapter,
        max_negatives_per_anchor=max_n,
        tier_ratios=tier_ratios,
        seed=seed,
    )

    pool_by_cve = {p["cve"]: p for p in pools}

    expected_skipped_total = 0
    expected_deficit_anchors = 0

    for anchor in present_ids:
        selected = selector.select_negatives(anchor, present_ids, present_set)

        pooled_set, skipped = _pool_universe(pool_by_cve[anchor], anchor, present_set)
        expected_skipped_total += skipped

        # random candidates: same-split ids that are neither the anchor nor pooled.
        random_set = (present_set - {anchor}) - pooled_set
        total_unique = len(present_set - {anchor})
        if total_unique < max_n:
            expected_deficit_anchors += 1

        # --- Validity (bounds, uniqueness, anchor-exclusion). ---
        assert len(selected) == len(set(selected)), "duplicate negative id selected"
        assert anchor not in selected, "anchor selected as its own negative (Req 6.8)"
        assert len(selected) <= max_n, "exceeded max_negatives_per_anchor"

        # Every selected id is a legitimate candidate (pooled or same-split random).
        assert set(selected) <= (pooled_set | random_set), (
            "selected an id outside the pooled + same-split candidate universe"
        )

        # --- Req 6.5: supply all available unique candidates when short. ---
        assert len(selected) == min(max_n, total_unique), (
            "selector did not supply exactly min(max_n, unique candidates)"
        )

        # --- Req 6.1/6.2/6.3: pooled negatives are drained before any random draw,
        # and pooled ids always precede random ids in the returned order. ---
        pooled_positions = [i for i, s in enumerate(selected) if s in pooled_set]
        random_positions = [i for i, s in enumerate(selected) if s in random_set]
        if pooled_positions and random_positions:
            assert max(pooled_positions) < min(random_positions), (
                "a random-sample negative preceded a pooled negative (Req 6.3)"
            )
        # If a random negative was used at all, every pooled candidate must have
        # already been consumed (pooled exhausted before random — Req 6.3).
        if random_positions:
            assert pooled_set <= set(selected), (
                "random fallback used while pooled candidates remained (Req 6.3)"
            )
        # Pooled selection count equals all pooled candidates that fit under max_n.
        assert len(pooled_positions) == min(max_n, len(pooled_set)), (
            "pooled cascade did not fill the shortfall before random (Req 6.1/6.2/6.3)"
        )

    # --- Req 5.6: skipped-missing-id count matches referenced-but-absent ids. ---
    assert selector.report.skipped_missing_id_count == expected_skipped_total, (
        "skipped_missing_id_count does not match referenced-but-absent ids"
    )
    # --- Req 6.5: deficit recorded (without crashing) when candidates ran out. ---
    assert selector.report.deficit_anchor_count == expected_deficit_anchors, (
        "deficit_anchor_count does not match anchors with too few candidates"
    )


# --------------------------------------------------------------------------- #
# Property 12 — reproducibility under a fixed seed (Req 11.3).
# --------------------------------------------------------------------------- #


@settings(max_examples=150, deadline=None)
@given(scenario=scenarios())
def test_selection_is_reproducible_under_fixed_seed(scenario):
    pools, present_ids, present_set, max_n, tier_ratios, seed = scenario

    def run(tag: str) -> Dict[str, List[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _build_adapter(pools, Path(tmp), tag)
        selector = CVENegativeSelector(
            ontology_adapter=adapter,
            max_negatives_per_anchor=max_n,
            tier_ratios=tier_ratios,
            seed=seed,
        )
        split_records = [{"cve": cid} for cid in present_ids]
        return selector.select_for_split(split_records, present_set)

    first = run("repro_a")
    second = run("repro_b")

    # Identical selection (same ids in the same order) for every anchor.
    assert first == second, "selection differs across identical-seed runs (Req 11.3)"


# --------------------------------------------------------------------------- #
# Targeted example checks for the cascade ORDER (medium-then-easy / easy-only).
# These pin the requirement text (Req 6.1, 6.2) with deterministic scenarios in
# which per-tier targets do not pre-consume the lower tiers, so the pooled
# fallback ordering is directly observable.
# --------------------------------------------------------------------------- #


def _adapter_for(pool: Dict[str, Any]) -> CVEOntologyAdapter:
    tmp = Path(tempfile.mkdtemp())
    return CVEOntologyAdapter(_write_pools([pool], tmp, "example"))


def test_empty_hard_fills_medium_then_easy_in_order():
    """Req 6.1: an empty hard tier fills the shortfall from medium, then easy."""
    anchor = _present_id(0)
    medium = [_present_id(i) for i in range(1, 4)]  # 3 present medium ids
    easy = [_present_id(i) for i in range(4, 10)]  # 6 present easy ids
    present = [anchor] + medium + easy
    pool = {
        "cve": anchor,
        "hard_negatives": [],
        "medium_negatives": medium,
        "easy_negatives": easy,
    }
    # All ratio weight on hard so hard's target == max_n; with hard empty the whole
    # shortfall flows through the pooled cascade (medium then easy), not per-tier.
    selector = CVENegativeSelector(
        ontology_adapter=_adapter_for(pool),
        max_negatives_per_anchor=5,
        tier_ratios={"hard": 1.0, "medium": 0.0, "easy": 0.0},
        seed=7,
    )
    selected = selector.select_negatives(anchor, present, set(present))

    assert len(selected) == 5
    medium_set, easy_set = set(medium), set(easy)
    # All 3 medium ids are used before any easy id (medium preferred over easy).
    assert medium_set <= set(selected), "medium tier not fully used before easy (Req 6.1)"
    med_positions = [i for i, s in enumerate(selected) if s in medium_set]
    easy_positions = [i for i, s in enumerate(selected) if s in easy_set]
    assert max(med_positions) < min(easy_positions), "easy used before medium (Req 6.1)"
    # hard tier flagged as requiring fallback filling.
    assert selector.report.per_tier_fallback_usage["hard"] == 1


def test_empty_medium_fills_from_easy():
    """Req 6.2: an empty medium tier fills the shortfall from easy."""
    anchor = _present_id(0)
    easy = [_present_id(i) for i in range(1, 6)]
    present = [anchor] + easy
    pool = {
        "cve": anchor,
        "hard_negatives": [],
        "medium_negatives": [],
        "easy_negatives": easy,
    }
    selector = CVENegativeSelector(
        ontology_adapter=_adapter_for(pool),
        max_negatives_per_anchor=4,
        tier_ratios={"hard": 0.5, "medium": 0.5, "easy": 0.0},
        seed=13,
    )
    selected = selector.select_negatives(anchor, present, set(present))
    assert len(selected) == 4
    assert set(selected) <= set(easy), "shortfall not filled from easy tier (Req 6.2)"


def test_no_pooled_negatives_uses_same_split_random_sample():
    """Req 6.4: an anchor with no pooled negatives draws from the same-split sample."""
    anchor = _present_id(0)
    others = [_present_id(i) for i in range(1, 8)]
    present = [anchor] + others
    pool = {
        "cve": anchor,
        "hard_negatives": [],
        "medium_negatives": [],
        "easy_negatives": [],
    }
    selector = CVENegativeSelector(
        ontology_adapter=_adapter_for(pool),
        max_negatives_per_anchor=5,
        seed=21,
    )
    selected = selector.select_negatives(anchor, present, set(present))
    assert len(selected) == 5
    assert anchor not in selected
    assert set(selected) <= set(others), "random fallback drew outside the split"
    assert selector.report.random_fallback_anchor_count == 1


def test_candidate_shortage_records_deficit_without_crashing():
    """Req 6.5: fewer unique candidates than max -> supply all + record deficit."""
    anchor = _present_id(0)
    only_other = _present_id(1)
    present = [anchor, only_other]  # only 1 possible negative
    pool = {
        "cve": anchor,
        "hard_negatives": [],
        "medium_negatives": [],
        "easy_negatives": [only_other],
    }
    selector = CVENegativeSelector(
        ontology_adapter=_adapter_for(pool),
        max_negatives_per_anchor=10,
        seed=3,
    )
    selected = selector.select_negatives(anchor, present, set(present))
    assert selected == [only_other]
    assert selector.report.deficit_anchor_count == 1


def test_single_record_split_yields_no_negatives():
    """Req 6.5 edge: a split with only the anchor yields no negatives, no crash."""
    anchor = _present_id(0)
    pool = {"cve": anchor, "hard_negatives": [], "medium_negatives": [], "easy_negatives": []}
    selector = CVENegativeSelector(
        ontology_adapter=_adapter_for(pool),
        max_negatives_per_anchor=5,
        seed=1,
    )
    selected = selector.select_negatives(anchor, [anchor], {anchor})
    assert selected == []
    assert selector.report.deficit_anchor_count == 1
