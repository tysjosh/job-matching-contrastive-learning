"""Verify the BatchProcessor ontology-matcher seam is additive.

Two properties matter, and they pull in opposite directions:

  * with the seam **unset**, every ontology lookup must resolve exactly as it did
    before the seam existed — the career and CVE paths must be untouched;
  * with the seam **set**, both the negative-selection path and ORCA's
    per-negative feature capture must route through the injected matcher, so a
    non-career domain gets a real five-scalar decomposition instead of the
    single blended ``career_distances`` proxy.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contrastive_learning.batch_processor import COARSE_URIS_KEY, BatchProcessor
from contrastive_learning.data_structures import TrainingSample


class _StubMatcher:
    """Records calls so we can prove which matcher was consulted."""

    def __init__(self, similarity: float = 0.75, name: str = "stub"):
        self.similarity = similarity
        self.name = name
        self.calls = []

    def ontology_set_similarity(self, a, b):
        self.calls.append((tuple(a), tuple(b)))
        return self.similarity

    def ot_distance(self, a, b):
        return 1.5


def _make_processor(**attrs) -> BatchProcessor:
    """A BatchProcessor with __init__ bypassed, carrying only what we exercise.

    The real constructor loads ESCO graphs and CSVs; this test is about the seam's
    dispatch, so the object is built directly and given the minimal attribute set
    the two methods under test read.
    """
    proc = BatchProcessor.__new__(BatchProcessor)
    proc.config = types.SimpleNamespace(
        orca_enabled=True, orca_capture_ot_distance=False
    )
    proc.skill_matcher = None
    proc.ontology_matcher_override = None
    proc.coarse_distance_fn = None
    proc.negative_selector = None
    proc.domain_negative_selector = None
    proc.current_epoch = 0
    proc.use_isco_negatives = False
    proc.isco_weight = 0.4
    proc.occ_to_isco = {}
    for key, value in attrs.items():
        setattr(proc, key, value)
    return proc


def _sample(skill_uris, coarse_uris=None) -> TrainingSample:
    resume = {"role": "x", "skills": ["a"], "skill_uris": list(skill_uris)}
    if coarse_uris is not None:
        resume[COARSE_URIS_KEY] = list(coarse_uris)
    return TrainingSample(
        resume=resume,
        job={"title": "t", "description": "d"},
        label="positive",
        sample_id="s1",
        metadata={},
    )


# --------------------------------------------------------------- seam unset
def test_effective_matcher_is_esco_when_override_unset():
    esco = _StubMatcher(name="esco")
    proc = _make_processor(skill_matcher=esco)
    assert proc._effective_skill_matcher() is esco


def test_effective_matcher_is_none_when_nothing_available():
    assert _make_processor()._effective_skill_matcher() is None


def test_feature_capture_uses_esco_and_isco_when_seam_unset():
    """Career path: d_esco from ESCO, d_isco from the neutral ISCO fallback."""
    esco = _StubMatcher(similarity=0.8, name="esco")
    proc = _make_processor(skill_matcher=esco)

    feats = proc._compute_negative_ontology_features(
        _sample(["esco:1"]), [{"skill_uris": ["esco:9"]}]
    )

    assert feats is not None and len(feats) == 1
    assert feats[0]["s_esco"] == pytest.approx(0.8)
    assert feats[0]["d_esco"] == pytest.approx(0.2)
    # ISCO unavailable -> the pre-existing neutral 0.5, not the injected path.
    assert feats[0]["d_isco"] == pytest.approx(0.5)
    assert feats[0]["s_isco"] == pytest.approx(0.5)
    assert esco.calls == [(("esco:1",), ("esco:9",))]


def test_isco_lookup_still_used_when_seam_unset():
    """The ESCO occupation-code table is consulted, unchanged, when no fn is set."""
    esco = _StubMatcher(similarity=0.5)
    proc = _make_processor(
        skill_matcher=esco,
        use_isco_negatives=True,
        occ_to_isco={"occ:a": "2512", "occ:b": "2519"},
    )
    sample = _sample(["esco:1"])
    sample.metadata["resume_occupation_uri"] = "occ:a"

    feats = proc._compute_negative_ontology_features(
        sample, [{"skill_uris": ["esco:2"], "occupation_uri": "occ:b"}]
    )
    # 2512 vs 2519 share the 3-digit prefix -> the ISCO table's 0.2 band.
    assert feats[0]["d_isco"] == pytest.approx(0.2)


# ----------------------------------------------------------------- seam set
def test_override_matcher_takes_precedence():
    esco = _StubMatcher(similarity=0.1, name="esco")
    mesh = _StubMatcher(similarity=0.9, name="mesh")
    proc = _make_processor(skill_matcher=esco)
    proc.set_ontology_matcher(mesh)

    assert proc._effective_skill_matcher() is mesh
    feats = proc._compute_negative_ontology_features(
        _sample(["D01"]), [{"skill_uris": ["D02"]}]
    )
    assert feats[0]["s_esco"] == pytest.approx(0.9)
    assert esco.calls == [], "the ESCO matcher must not be consulted once overridden"
    assert mesh.calls == [(("D01",), ("D02",))]


def test_coarse_distance_fn_replaces_isco_and_receives_lists():
    """d_isco comes from the injected fn, fed both sides' coarse URI lists."""
    mesh = _StubMatcher(similarity=0.5)
    seen = {}

    def coarse(anchor_uris, candidate_uris):
        seen["anchor"] = list(anchor_uris)
        seen["candidate"] = list(candidate_uris)
        return 0.125

    proc = _make_processor()
    proc.set_ontology_matcher(mesh, coarse_distance_fn=coarse)

    feats = proc._compute_negative_ontology_features(
        _sample(["D01", "D02"], coarse_uris=["C18"]),
        [{"skill_uris": ["D03"], COARSE_URIS_KEY: ["C19", "C14"]}],
    )

    assert feats[0]["d_isco"] == pytest.approx(0.125)
    assert feats[0]["s_isco"] == pytest.approx(0.875)
    assert seen["anchor"] == ["C18"]
    assert seen["candidate"] == ["C19", "C14"]


def test_coarse_distance_fn_failure_degrades_to_neutral():
    """A raising coarse fn must not abort feature capture (Req 9.3)."""

    def boom(a, b):
        raise RuntimeError("ontology unavailable")

    proc = _make_processor()
    proc.set_ontology_matcher(_StubMatcher(), coarse_distance_fn=boom)

    feats = proc._compute_negative_ontology_features(
        _sample(["D01"], coarse_uris=["C18"]),
        [{"skill_uris": ["D02"], COARSE_URIS_KEY: ["C19"]}],
    )
    assert feats[0]["d_isco"] == pytest.approx(0.5)


def test_clearing_override_restores_esco():
    esco = _StubMatcher(similarity=0.3, name="esco")
    proc = _make_processor(skill_matcher=esco)
    proc.set_ontology_matcher(_StubMatcher(similarity=0.9))
    proc.set_ontology_matcher(None)

    assert proc._effective_skill_matcher() is esco
    assert proc.coarse_distance_fn is None
    feats = proc._compute_negative_ontology_features(
        _sample(["esco:1"]), [{"skill_uris": ["esco:2"]}]
    )
    assert feats[0]["s_esco"] == pytest.approx(0.3)


def test_negative_with_no_uris_still_neutral_under_override():
    """A candidate carrying no URIs gets s_esco 0.0, same as the career path."""
    proc = _make_processor()
    proc.set_ontology_matcher(_StubMatcher(similarity=0.9))
    feats = proc._compute_negative_ontology_features(
        _sample(["D01"]), [{"skill_uris": []}]
    )
    assert feats[0]["s_esco"] == pytest.approx(0.0)
    assert feats[0]["d_esco"] == pytest.approx(1.0)


# ------------------------------------------- domain selector lifetime (regression)
def _selector_stub(tag):
    """A minimal batch-level selector returning one tagged negative."""

    class S:
        def select_batch_negatives(self, anchor, candidates, max_negatives, epoch=0):
            return [{"nct_id": tag, "encoder_view": tag}], [1.0]

    return S()


def test_domain_selector_survives_orca_seam_being_cleared():
    """Regression: the ORCA orchestrator clears its own seam at Phase 4 start.

    ``OrcaPhaseOrchestrator._configure_phase4_negative_selector`` calls
    ``set_selector(None)`` for every non-adaptive variant, including the
    ORCA-Denominator MVP. A domain selector attached through the ORCA seam was
    therefore wiped exactly when joint training began, and Phase 4 silently fell
    back to random in-batch negatives with the graded grade-1 judgments never
    reaching the loss. The domain slot must be unaffected.
    """
    proc = _make_processor()
    proc.set_domain_negative_selector(_selector_stub("domain"))
    proc.set_negative_selector(_selector_stub("orca"))

    # Simulate what the orchestrator does entering Phase 4 on a non-adaptive variant.
    proc.set_negative_selector(None)

    assert proc.negative_selector is None
    assert proc.domain_negative_selector is not None

    result = proc._select_with_injected_selector(
        _sample(["D01"]), [], 4, selector=proc.domain_negative_selector
    )
    assert result is not None
    negatives, _ = result
    assert negatives[0]["nct_id"] == "domain"


def test_orca_selector_takes_precedence_over_domain_selector():
    """An active adaptive sampler must win: it is the experimental variable."""
    proc = _make_processor()
    proc.set_domain_negative_selector(_selector_stub("domain"))
    proc.set_negative_selector(_selector_stub("orca"))

    orca_result = proc._select_with_injected_selector(_sample(["D01"]), [], 4)
    assert orca_result[0][0]["nct_id"] == "orca"


def test_domain_selector_defaults_to_none_and_clears():
    proc = _make_processor()
    assert proc.domain_negative_selector is None
    proc.set_domain_negative_selector(_selector_stub("d"))
    assert proc.domain_negative_selector is not None
    proc.set_domain_negative_selector(None)
    assert proc.domain_negative_selector is None


def test_incompatible_domain_selector_degrades_to_none():
    """A selector without select_batch_negatives must not break selection."""
    proc = _make_processor()
    proc.set_domain_negative_selector(object())
    assert (
        proc._select_with_injected_selector(
            _sample(["D01"]), [], 4, selector=proc.domain_negative_selector
        )
        is None
    )
