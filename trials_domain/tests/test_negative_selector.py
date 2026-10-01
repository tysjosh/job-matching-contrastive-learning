"""Tests for TrialsNegativeSelector: curriculum, determinism, and pool handling."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trials_domain.negative_selector import (
    CAREER_DISTANCE_SCALE,
    NEUTRAL_DISTANCE,
    TrialsNegativeSelector,
)


def _views(hard, easy):
    out = {}
    for nct in list(hard) + list(easy):
        out[nct] = {
            "nct_id": nct,
            "encoder_view": f"view {nct}",
            "skill_uris": ["D01"],
            "coarse_uris": ["C01"],
        }
    return out


def _selector(hard, easy, topic="2021_1", **kwargs):
    pools = {topic: {"ineligible": list(hard), "not_relevant": list(easy)}}
    kwargs.setdefault("total_epochs", 10)
    kwargs.setdefault("training_seed", 42)
    return TrialsNegativeSelector(pools, _views(hard, easy), **kwargs)


def _anchor(topic="2021_1", uris=("D02",)):
    return types.SimpleNamespace(
        metadata={"resume_id": topic},
        resume={"topic_id": topic, "skill_uris": list(uris)},
    )


# ---------------------------------------------------------------- curriculum
def test_hard_ratio_ramps_monotonically_and_hits_both_endpoints():
    sel = _selector(["h1"], ["e1"], total_epochs=10)
    ratios = [sel.hard_ratio(e) for e in range(10)]
    assert ratios[0] == pytest.approx(0.2)
    assert ratios[-1] == pytest.approx(0.6)
    assert all(b >= a for a, b in zip(ratios, ratios[1:])), ratios


def test_hard_ratio_clamped_and_single_epoch_uses_endpoint():
    sel = _selector(["h1"], ["e1"], total_epochs=1)
    assert sel.hard_ratio(0) == pytest.approx(0.6)
    # Out-of-range epochs must not escape [0, 1].
    assert 0.0 <= sel.hard_ratio(999) <= 1.0


def test_grade1_share_increases_with_epoch():
    hard = [f"h{i}" for i in range(50)]
    easy = [f"e{i}" for i in range(50)]
    sel = _selector(hard, easy, total_epochs=10)

    sel.set_epoch(0)
    early, _ = sel.select_batch_negatives(_anchor(), [], 10)
    sel.set_epoch(9)
    late, _ = sel.select_batch_negatives(_anchor(), [], 10)

    n_early = sum(1 for n in early if n["grade"] == 1)
    n_late = sum(1 for n in late if n["grade"] == 1)
    assert n_early == 2 and n_late == 6, (n_early, n_late)


# --------------------------------------------------------------- determinism
def test_selection_is_deterministic_across_instances():
    hard = [f"h{i}" for i in range(30)]
    easy = [f"e{i}" for i in range(30)]
    a = _selector(hard, easy, training_seed=7)
    b = _selector(hard, easy, training_seed=7)
    a.set_epoch(4)
    b.set_epoch(4)
    assert [n["nct_id"] for n in a.select_batch_negatives(_anchor(), [], 8)[0]] == [
        n["nct_id"] for n in b.select_batch_negatives(_anchor(), [], 8)[0]
    ]


def test_selection_varies_with_seed_and_epoch():
    hard = [f"h{i}" for i in range(30)]
    easy = [f"e{i}" for i in range(30)]
    base = _selector(hard, easy, training_seed=7)
    base.set_epoch(4)
    ref = [n["nct_id"] for n in base.select_batch_negatives(_anchor(), [], 8)[0]]

    other_seed = _selector(hard, easy, training_seed=99)
    other_seed.set_epoch(4)
    assert ref != [
        n["nct_id"] for n in other_seed.select_batch_negatives(_anchor(), [], 8)[0]
    ]

    base.set_epoch(5)
    assert ref != [n["nct_id"] for n in base.select_batch_negatives(_anchor(), [], 8)[0]]


def test_random_window_is_fixed_but_resampled_each_epoch():
    hard = [f"h{i}" for i in range(30)]
    easy = [f"e{i}" for i in range(30)]
    sel = _selector(hard, easy, tier_sampling="random_window",
                    tier_window_frac=0.34)

    selected = []
    for epoch in range(10):
        sel.set_epoch(epoch)
        selected.append({n["nct_id"] for n in
                         sel.select_batch_negatives(_anchor(), [], 8)[0]})

    union = set().union(*selected)
    assert len(union) < len(hard) + len(easy)
    assert len({tuple(sorted(row)) for row in selected}) > 1


def test_stochastic_mesh_samples_only_from_closest_window():
    hard = [f"h{i}" for i in range(10)]
    easy = [f"e{i}" for i in range(10)]
    views = {}
    for prefix, ids in (("h", hard), ("e", easy)):
        for index, nct in enumerate(ids):
            views[nct] = {
                "nct_id": nct,
                "encoder_view": nct,
                "skill_uris": [f"{prefix}{index:02d}"],
                "coarse_uris": [],
            }

    class OrderedMatcher:
        calls = 0

        def ontology_set_similarity(self, _anchor, candidate):
            self.calls += 1
            return 1.0 - int(candidate[0][1:]) / 100.0

    matcher = OrderedMatcher()
    pools = {"2021_1": {"ineligible": hard, "not_relevant": easy}}
    sel = TrialsNegativeSelector(
        pools, views, matcher=matcher, total_epochs=10, training_seed=42,
        mesh_tiered=True, mesh_score_cap=0, tier_sampling="stochastic",
        tier_window_frac=0.4)

    selections = []
    calls_after_first = None
    for epoch in range(10):
        sel.set_epoch(epoch)
        selections.append(sel.select_batch_negatives(_anchor(), [], 4)[0])
        if epoch == 0:
            calls_after_first = matcher.calls
        elif epoch == 1:
            # The second epoch adds only four emitted-distance calculations;
            # the 20 candidate-ranking calculations are cached.
            assert matcher.calls == calls_after_first + 4

    for negatives in selections:
        assert all(int(n["nct_id"][1:]) < 4 for n in negatives)
    assert len({tuple(sorted(n["nct_id"] for n in row))
                for row in selections}) > 1


def test_random_and_mesh_windows_have_exactly_matched_diversity():
    hard = [f"h{i}" for i in range(13)]
    easy = [f"e{i}" for i in range(19)]
    views = _views(hard, easy)
    for index, nct in enumerate(hard + easy):
        views[nct]["skill_uris"] = [f"D{index:03d}"]

    class Matcher:
        def ontology_set_similarity(self, _anchor, candidate):
            return 1.0 - int(candidate[0][1:]) / 1000.0

    pools = {"2021_1": {"ineligible": hard, "not_relevant": easy}}
    control = TrialsNegativeSelector(
        pools, views, total_epochs=15, training_seed=42,
        tier_sampling="random_window", tier_window_frac=0.34)
    ontology = TrialsNegativeSelector(
        pools, views, matcher=Matcher(), total_epochs=15, training_seed=42,
        mesh_tiered=True, mesh_score_cap=0, tier_sampling="stochastic",
        tier_window_frac=0.34)

    control_seen = {0: set(), 1: set()}
    ontology_seen = {0: set(), 1: set()}
    for epoch in range(15):
        for selector, seen in ((control, control_seen),
                               (ontology, ontology_seen)):
            selector.set_epoch(epoch)
            negatives, _ = selector.select_batch_negatives(_anchor(), [], 7)
            for negative in negatives:
                seen[negative["grade"]].add(negative["nct_id"])

    assert {grade: len(ids) for grade, ids in control_seen.items()} == {
        grade: len(ids) for grade, ids in ontology_seen.items()}


def test_selection_independent_of_call_order():
    """A topic's negatives must not depend on how many anchors preceded it."""
    hard = [f"h{i}" for i in range(20)]
    easy = [f"e{i}" for i in range(20)]
    sel = _selector(hard, easy)
    sel.set_epoch(2)
    first = [n["nct_id"] for n in sel.select_batch_negatives(_anchor(), [], 6)[0]]
    for _ in range(5):
        sel.select_batch_negatives(_anchor(), [], 6)
    assert [n["nct_id"] for n in sel.select_batch_negatives(_anchor(), [], 6)[0]] == first


# ------------------------------------------------------------------ pools
def test_backfills_from_easy_when_hard_pool_is_short():
    """One real topic has a single grade-1 judgment; it must still fill up."""
    sel = _selector(["h1"], [f"e{i}" for i in range(20)], total_epochs=10)
    sel.set_epoch(9)  # wants 60% hard = 6, but only 1 exists
    negs, _ = sel.select_batch_negatives(_anchor(), [], 8)
    assert len(negs) == 8
    assert sum(1 for n in negs if n["grade"] == 1) == 1


def test_backfills_from_hard_when_easy_pool_is_short():
    sel = _selector([f"h{i}" for i in range(20)], ["e1"], total_epochs=10)
    sel.set_epoch(0)  # wants 20% hard = 2, easy only has 1
    negs, _ = sel.select_batch_negatives(_anchor(), [], 8)
    assert len(negs) == 8


def test_returns_none_for_unknown_topic():
    sel = _selector(["h1"], ["e1"])
    assert sel.select_batch_negatives(_anchor(topic="nope"), [], 4) is None


def test_returns_none_when_pool_ids_have_no_views():
    """Pool entries with no indexed view must not yield phantom negatives."""
    pools = {"2021_1": {"ineligible": ["missing"], "not_relevant": ["gone"]}}
    sel = TrialsNegativeSelector(pools, {}, total_epochs=10)
    assert sel.select_batch_negatives(_anchor(), [], 4) is None


def test_grade_and_label_are_carried_on_each_negative():
    sel = _selector(["h1", "h2"], ["e1", "e2"])
    negs, _ = sel.select_batch_negatives(_anchor(), [], 4)
    for n in negs:
        assert n["grade"] in (0, 1)
        expected = "ineligible" if n["grade"] == 1 else "not_relevant"
        assert n["original_label"] == expected


# --------------------------------------------------------------- distances
def test_distances_use_matcher_and_career_scale():
    class M:
        def ontology_set_similarity(self, a, b):
            return 0.25

    sel = _selector(["h1"], ["e1"], matcher=M())
    _, dists = sel.select_batch_negatives(_anchor(), [], 2)
    assert all(d == pytest.approx(0.75 * CAREER_DISTANCE_SCALE) for d in dists)


def test_distance_neutral_without_matcher_and_on_matcher_failure():
    sel = _selector(["h1"], ["e1"], matcher=None)
    _, dists = sel.select_batch_negatives(_anchor(), [], 2)
    assert all(d == pytest.approx(NEUTRAL_DISTANCE * CAREER_DISTANCE_SCALE) for d in dists)

    class Boom:
        def ontology_set_similarity(self, a, b):
            raise RuntimeError("unavailable")

    sel2 = _selector(["h1"], ["e1"], matcher=Boom())
    _, dists2 = sel2.select_batch_negatives(_anchor(), [], 2)
    assert all(d == pytest.approx(NEUTRAL_DISTANCE * CAREER_DISTANCE_SCALE) for d in dists2)


def test_distance_neutral_when_anchor_has_no_uris():
    class M:
        def ontology_set_similarity(self, a, b):
            return 0.9

    sel = _selector(["h1"], ["e1"], matcher=M())
    _, dists = sel.select_batch_negatives(_anchor(uris=()), [], 2)
    assert all(d == pytest.approx(NEUTRAL_DISTANCE * CAREER_DISTANCE_SCALE) for d in dists)
