"""Property-based test for CVENegativeSelector validity (Property 11).

Component under test: ``CVENegativeSelector.select_negatives`` in
``cve_domain/negative_selector.py`` (which consumes a
``cve_domain.ontology_adapter.CVEOntologyAdapter`` indexed over a
``cve_denominator_pools.jsonl`` file).

This test uses Hypothesis to generate synthetic tiered denominator pools
(hard / medium / easy CVE-id lists) for an anchor, a present-ids universe, and a
same-split candidate id list, writes a temporary ``cve_denominator_pools.jsonl``,
builds a ``CVEOntologyAdapter`` over it, runs ``select_negatives``, and asserts
Property 11 — the always-true validity invariants of negative selection:

    For any anchor and pools, the selected negatives are
      * unique  (no duplicate ``cve`` identifiers),
      * anchor-excluded  (the anchor's own id never appears), and
      * bounded  (at most ``max_negatives_per_anchor`` are returned).

Design note (why only the invariants are asserted): Property 11 covers the
invariants that hold for *every* input. The per-tier rounded target (Req 5.5)
caps each tier only in the non-fallback primary draw; the fallback phase (Req 6.3)
may legitimately exceed a tier's target to drain remaining pooled negatives before
random fallback. (Even in a well-populated case, when the rounded per-tier targets
sum to less than ``max_negatives_per_anchor`` the pooled fallback fills the
remainder and exceeds a tier's target.) This test therefore focuses on the
unique / anchor-excluded / bounded invariants, which always hold; the fallback
ordering is covered separately by Property 12.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_negative_selection_validity_property11.py
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cve_domain.negative_selector import CVENegativeSelector
from cve_domain.ontology_adapter import CVEOntologyAdapter


# ---------------------------------------------------------------------------
# Generation helpers
#
# We generate a small universe of CVE ids, pick an anchor from it, and draw the
# anchor's three tier lists from that universe. Tier lists may deliberately:
#   * include the anchor's own id (to exercise anchor-exclusion, Req 6.8),
#   * include ids that are NOT in the present-ids universe (to exercise the
#     skip-missing-id filter, Req 5.6),
#   * repeat ids across / within tiers (to exercise de-duplication, Req 5.2/6.6).
# The present-ids universe is a subset of the id universe, and the same-split id
# list is drawn from the present ids (the random-fallback candidate universe).
# ---------------------------------------------------------------------------

_ID_UNIVERSE = [f"CVE-2024-{n:04d}" for n in range(40)]

_tier_ids = st.lists(st.sampled_from(_ID_UNIVERSE), min_size=0, max_size=25)


@st.composite
def selection_inputs(draw) -> dict:
    """Draw an anchor, its tiered pools, a present-ids universe, and split ids."""
    anchor = draw(st.sampled_from(_ID_UNIVERSE))

    hard = draw(_tier_ids)
    medium = draw(_tier_ids)
    easy = draw(_tier_ids)

    # Present-ids universe: a subset of the whole id universe. Ensures some pooled
    # ids may be missing (filtered + counted), exercising Req 5.6 without affecting
    # the invariants under test.
    present = draw(
        st.lists(st.sampled_from(_ID_UNIVERSE), min_size=0, max_size=40, unique=True)
    )
    present_set = set(present)

    # Same-split candidate universe for the random-sample fallback: drawn from the
    # present ids so fallback negatives are themselves valid present records.
    split_ids = draw(
        st.lists(st.sampled_from(_ID_UNIVERSE), min_size=0, max_size=40, unique=True)
    )
    # The present universe must cover the split candidates for them to be usable.
    present_set |= set(split_ids)

    max_negatives = draw(st.integers(min_value=1, max_value=30))

    # Per-tier ratios: non-negative, at least one positive so normalization is well
    # defined. Occasionally all-equal / skewed to vary the rounded targets.
    ratios = {
        "hard": draw(st.floats(min_value=0.0, max_value=1.0)),
        "medium": draw(st.floats(min_value=0.0, max_value=1.0)),
        "easy": draw(st.floats(min_value=0.0, max_value=1.0)),
    }
    if sum(ratios.values()) <= 0:
        ratios = {"hard": 0.34, "medium": 0.33, "easy": 0.33}

    seed = draw(st.integers(min_value=0, max_value=10_000))

    return {
        "anchor": anchor,
        "hard": hard,
        "medium": medium,
        "easy": easy,
        "present": present_set,
        "split_ids": split_ids,
        "max_negatives": max_negatives,
        "ratios": ratios,
        "seed": seed,
    }


def _write_pools(path: Path, anchor: str, hard, medium, easy) -> None:
    """Write a one-line ``cve_denominator_pools.jsonl`` for the anchor."""
    line = {
        "cve": anchor,
        "hard_negatives": hard,
        "medium_negatives": medium,
        "easy_negatives": easy,
    }
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Property 11
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 11: Negative selection is always valid (unique, anchor-excluded, bounded)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(inputs=selection_inputs())
def test_negative_selection_is_always_valid(inputs, tmp_path):
    """Property 11 (Validates: Requirements 5.1, 5.2, 5.4, 5.5, 6.6, 6.8).

    For any anchor and pools, ``select_negatives`` returns negatives that are
    unique (no duplicate cve ids), exclude the anchor's own id, and number at
    most ``max_negatives_per_anchor``.
    """
    anchor = inputs["anchor"]
    present = inputs["present"]
    split_ids = inputs["split_ids"]
    max_n = inputs["max_negatives"]

    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pools(pools_path, anchor, inputs["hard"], inputs["medium"], inputs["easy"])

    # Req 5.1: the adapter indexes the pool by its cve id.
    adapter = CVEOntologyAdapter(str(pools_path))
    assert adapter.has_pool(anchor)

    selector = CVENegativeSelector(
        ontology_adapter=adapter,
        max_negatives_per_anchor=max_n,          # Req 5.4 (>= 1)
        tier_ratios=inputs["ratios"],            # Req 5.5
        seed=inputs["seed"],
    )

    selected = selector.select_negatives(anchor, split_ids, present_ids=present)

    # --- Bounded (Req 5.4): never more than the configured maximum. ----------
    assert len(selected) <= max_n, (
        f"selected {len(selected)} negatives exceeds max_negatives_per_anchor={max_n}"
    )

    # --- Unique (Req 5.2, 6.6): no duplicate cve identifiers. ----------------
    assert len(selected) == len(set(selected)), (
        f"selected negatives contain duplicates: {selected}"
    )

    # --- Anchor-excluded (Req 6.8): the anchor's own id never appears. -------
    assert anchor not in selected, "the anchor's own id must not appear in its negatives"

    # --- Validity universe: every selected id is either a valid present id or a
    # same-split candidate (pooled negatives are filtered against present_ids;
    # random-fallback negatives come from the split ids). This guards Req 5.6.
    valid_universe = set(present) | set(split_ids)
    assert set(selected).issubset(valid_universe), (
        f"selected ids {set(selected) - valid_universe} are outside the valid universe"
    )
