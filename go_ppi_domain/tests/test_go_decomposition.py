"""GO aspect and structure ablations, including the persisted BP split seam."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from go_ppi_domain.go_ontology import GoIndex, GoMatcher, GoMultiAspectMatcher
from go_ppi_domain.negative_selector import GoPpiNegativeSelector
from go_ppi_domain.run_config import attach_go_ppi_domain


def index(aspect: str, terms: tuple[str, str, str]) -> GoIndex:
    root, left, right = terms
    ancestors = {
        root: frozenset({root}),
        left: frozenset({root, left}),
        right: frozenset({root, right}),
    }
    return GoIndex(
        ancestors, {root: 0, left: 1, right: 1},
        {"A": frozenset({left}), "B": frozenset({right})},
        {"A": ancestors[left], "B": ancestors[right]},
        {"A": frozenset({root}), "B": frozenset({root})},
        {root: 1.0, left: 2.0, right: 3.0},
        {"aspect": aspect, "genes_annotated": 2},
    )


def test_structure_modes_isolate_direct_overlap_ancestry_and_ic():
    go = index("P", ("P0", "P1", "P2"))
    exact = GoMatcher(go, similarity_mode="exact")
    ancestor = GoMatcher(go, similarity_mode="ancestor")
    simgic = GoMatcher(go, similarity_mode="simgic")
    assert exact.ontology_set_similarity(["P1"], ["P2"]) == 0.0
    assert ancestor.ontology_set_similarity(["P1"], ["P2"]) == pytest.approx(1 / 3)
    assert simgic.ontology_set_similarity(["P1"], ["P2"]) == pytest.approx(1 / 6)
    for matcher in (exact, ancestor, simgic):
        assert matcher.ontology_set_similarity(["P1"], ["P1"]) == 1.0
        assert matcher.ontology_set_similarity([], ["P1"]) == 0.0
        assert matcher.ontology_set_similarity(["P1"], ["P2"]) == matcher.ontology_set_similarity(["P2"], ["P1"])
    with pytest.raises(ValueError):
        GoMatcher(go, similarity_mode="unknown")


def test_hybrid_normalizes_by_available_aspects_and_weight():
    p = GoMatcher(index("P", ("P0", "P1", "P2")), similarity_mode="exact")
    f = GoMatcher(index("F", ("F0", "F1", "F2")), similarity_mode="exact")
    c = GoMatcher(index("C", ("C0", "C1", "C2")), similarity_mode="exact")
    # A and B share MF, differ in BP, and B has no CC annotation.
    f.index.direct["B"] = frozenset({"F1"})
    c.index.direct["B"] = frozenset()
    hybrid = GoMultiAspectMatcher({"P": p, "F": f, "C": c},
                                  {"P": 1, "F": 3, "C": 1})
    a = sorted(hybrid.terms_for_gene("A"))
    b = sorted(hybrid.terms_for_gene("B"))
    assert hybrid.aspect_similarities(a, b) == {"P": 0.0, "F": 1.0, "C": None}
    assert hybrid.ontology_set_similarity(a, b) == pytest.approx(0.75)
    assert hybrid.ontology_set_similarity([], b) == 0.0
    assert hybrid.skill_distance("P1", "F1") is None
    with pytest.raises(ValueError):
        GoMultiAspectMatcher({"P": p, "F": f, "C": c}, {"P": 0})


def test_selector_resolves_selected_aspect_from_gene_ids():
    matcher = GoMatcher(index("F", ("F0", "F1", "F2")), similarity_mode="ancestor")
    selector = GoPpiNegativeSelector(
        {"A": {"weak_evidence": ["B"], "no_interaction": []}},
        {"B": {"partner_id": "B", "encoder_view": "protein B",
               "skill_uris": sorted(matcher.terms_for_gene("B")),
               "coarse_uris": sorted(matcher.coarse_for_gene("B"))}},
        matcher=matcher, go_tiered=True, go_score_cap=0,
    )
    sample = SimpleNamespace(
        metadata={"resume_id": "A"},
        resume={"protein_id": "A", "skill_uris": ["P1"], "coarse_uris": ["P0"]},
        job={"partner_id": "B", "skill_uris": ["P2"], "coarse_uris": ["P0"]},
    )
    negatives, distances = selector.select_batch_negatives(sample, [], 1, 0)
    assert sample.resume["skill_uris"] == ["F1"]
    assert sample.job["skill_uris"] == ["F2"]
    assert negatives[0]["skill_uris"] == ["F2"]
    assert distances == pytest.approx([10 * (1 - 1 / 3)])


def test_unannotated_candidates_do_not_look_artificially_close():
    matcher = GoMatcher(index("F", ("F0", "F1", "F2")), similarity_mode="exact")
    selector = GoPpiNegativeSelector({}, {
        "missing": {"skill_uris": []},
        "annotated": {"skill_uris": ["F2"]},
    }, matcher=matcher)
    assert selector._go_ordered(["F1"], ["missing", "annotated"]) == [
        "annotated", "missing"]


@pytest.mark.parametrize("sampling", ["stochastic", "random_window"])
def test_trainer_attachment_forwards_sampling_control(monkeypatch, sampling):
    matcher = GoMatcher(index("P", ("P0", "P1", "P2")))
    captured = {}

    def selector_factory(*args, **kwargs):
        captured.update(kwargs)
        return GoPpiNegativeSelector({}, {}, **kwargs)

    monkeypatch.setattr(GoPpiNegativeSelector, "from_split_dir", selector_factory)

    class Processor:
        def set_ontology_matcher(self, matcher, coarse_distance_fn):
            self.matcher = matcher

        def set_domain_negative_selector(self, selector):
            self.selector = selector

    trainer = SimpleNamespace(batch_processor=Processor())
    config = SimpleNamespace(go_ppi_tier_sampling=sampling,
                             go_ppi_tier_window_frac=0.34,
                             go_ppi_go_tiered_negatives=sampling == "stochastic")
    summary = attach_go_ppi_domain(trainer, config, matcher=matcher)
    assert captured["tier_sampling"] == sampling
    assert captured["tier_window_frac"] == 0.34
    assert summary["tier_sampling"] == sampling
