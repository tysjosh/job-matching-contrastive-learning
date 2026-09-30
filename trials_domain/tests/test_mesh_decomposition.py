"""Disease-only MeSH ranking and tier-specific controls."""

from types import SimpleNamespace

import pytest

from trials_domain.mesh_ontology import MeshIndex, MeshMatcher
from trials_domain.negative_selector import TrialsNegativeSelector


def test_exact_and_hierarchy_use_the_same_bma_aggregator():
    index = MeshIndex(
        {"D1": "disease one", "D2": "disease two"},
        {"D1": ("C01.001",), "D2": ("C01.002",)},
        {"C01.001": "D1", "C01.002": "D2"}, {},
    )
    exact = MeshMatcher(index, similarity_mode="exact")
    hierarchy = MeshMatcher(index, similarity_mode="hierarchy")
    assert exact.ontology_set_similarity(["D1"], ["D2"]) == 0.0
    assert hierarchy.ontology_set_similarity(["D1"], ["D2"]) > 0.0
    assert exact.ontology_set_similarity(["D1", "D2"], ["D1"]) == pytest.approx(.75)
    with pytest.raises(ValueError):
        MeshMatcher(index, similarity_mode="unknown")


class IdentityMatcher:
    def ontology_set_similarity(self, a, b):
        return 1.0 if set(a) & set(b) else 0.0


def selector(scope="both", sampling="stochastic", anchor_disease=("C1",)):
    hard = [f"h{i}" for i in range(20)]
    easy = [f"e{i}" for i in range(20)]
    views = {}
    for pid in hard + easy:
        views[pid] = {
            "nct_id": pid, "encoder_view": pid,
            # The all-MeSH union is intentionally identical for everyone;
            # only the condition facet can distinguish these candidates.
            "skill_uris": ["X"],
            "coarse_uris": ["C1"] if pid.endswith("0") else ["C2"],
        }
    sel = TrialsNegativeSelector(
        {"topic": {"ineligible": hard, "not_relevant": easy}},
        views, matcher=IdentityMatcher(), mesh_tiered=True,
        mesh_facet="disease", mesh_tier_scope=scope,
        tier_sampling=sampling, tier_window_frac=.34,
        training_seed=13, total_epochs=15)
    anchor = SimpleNamespace(metadata={"resume_id": "topic"},
                             resume={"topic_id": "topic",
                                     "skill_uris": ["X"],
                                     "coarse_uris": list(anchor_disease)})
    return sel, anchor


def test_disease_scores_ignore_all_mesh_union():
    sel, _ = selector()
    assert sel._ontology_distance(["C1"], sel.views["h0"]) == 0.0
    assert sel._ontology_distance(["C1"], sel.views["h1"]) == 1.0


def test_tier_scope_guides_only_selected_pool():
    both, anchor = selector("both")
    easy_only, anchor2 = selector("not_relevant")
    hard_only, anchor3 = selector("ineligible")
    for sel, sample in ((both, anchor), (easy_only, anchor2), (hard_only, anchor3)):
        selected, _ = sel.select_batch_negatives(sample, [], 7, 0)
        assert len(selected) == 7
    assert "e0" in easy_only._mesh_order_cache[("topic", "easy")]
    assert ("topic", "hard") not in easy_only._mesh_order_cache
    assert "h0" in hard_only._mesh_order_cache[("topic", "hard")]
    assert ("topic", "easy") not in hard_only._mesh_order_cache
    assert set(both._mesh_order_cache) == {("topic", "hard"), ("topic", "easy")}


def test_missing_disease_falls_back_to_random_window():
    selected, anchor = selector(anchor_disease=())
    control, control_anchor = selector(sampling="random_window", anchor_disease=())
    a, _ = selected.select_batch_negatives(anchor, [], 7, 0)
    b, _ = control.select_batch_negatives(control_anchor, [], 7, 0)
    assert [v["nct_id"] for v in a] == [v["nct_id"] for v in b]
