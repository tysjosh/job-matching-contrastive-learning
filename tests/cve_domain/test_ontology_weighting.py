"""Tests for CVE sample-level loss weighting signals.

Feature: cve-vulnerability-ranking

The CVE domain supplies per-sample loss-weight metadata that the shared
``ContrastiveLossEngine._compute_ontology_weight`` consumes (no loss-engine
change). There are two signals, both gated behind ``ontology_weight > 0``:

1. **Label-completeness quality tier (default, ontology-INDEPENDENT).** The
   anchor's ``quality_tier`` is derived from how many completeness labels
   (``priority_score``, ``priority_band``) are present — the direct analogue of
   the career "data quality" tier. This is the principled default weight.

2. **Ontology-overlap signal (opt-in ABLATION).** The weighted-Jaccard overlap of
   the anchor's and its selected positive's CWE/CPE/vendor sets. Because that
   overlap is the same signal that *selected* the positive (self-referential), it
   is OFF by default (neutral 0.5) and only enabled via
   ``config.cve_ontology_overlap_weighting``.

Everything is a no-op when ``ontology_weight == 0`` (the shipped default),
preserving backward compatibility.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_ontology_weighting.py
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.loss_engine import ContrastiveLossEngine
from cve_domain.record_adapter import (
    CVE_NEUTRAL_ONTOLOGY_SIGNAL,
    CVERecordAdapter,
    _label_completeness_tier,
    ontology_overlap_signal,
)


def _onto(cwes=None, cpes=None, vendors=None) -> Dict[str, Any]:
    return {"cwes": cwes or [], "cpes": cpes or [], "vendors": vendors or []}


# --------------------------------------------------------------------------- #
# ontology_overlap_signal (weighted Jaccard over CWE / CPE / vendor)
# --------------------------------------------------------------------------- #
class TestOntologyOverlapSignal:
    def test_identical_ontology_is_one(self):
        onto = _onto(["CWE-79", "CWE-89"], ["cpe:a"], ["acme"])
        assert ontology_overlap_signal(onto, onto) == pytest.approx(1.0)

    def test_disjoint_ontology_is_zero(self):
        a = _onto(["CWE-79"], ["cpe:a"], ["acme"])
        b = _onto(["CWE-22"], ["cpe:b"], ["globex"])
        assert ontology_overlap_signal(a, b) == pytest.approx(0.0)

    def test_partial_overlap_uses_only_populated_facets(self):
        a = _onto(["CWE-79", "CWE-89"])
        b = _onto(["CWE-79"])
        # Only the cwes facet has data -> signal == Jaccard(cwes) == 1/2.
        assert ontology_overlap_signal(a, b) == pytest.approx(0.5)

    def test_empty_facets_excluded_from_average(self):
        a = _onto(["CWE-79"])
        b = _onto(["CWE-79"])
        assert ontology_overlap_signal(a, b) == pytest.approx(1.0)

    def test_no_ontology_is_neutral_half(self):
        assert ontology_overlap_signal(_onto(), _onto()) == pytest.approx(0.5)
        assert ontology_overlap_signal(None, None) == pytest.approx(0.5)

    def test_cwe_weighted_more_than_vendor(self):
        share_cwe = ontology_overlap_signal(
            _onto(cwes=["CWE-79"], vendors=["acme"]),
            _onto(cwes=["CWE-79"], vendors=["globex"]),
        )
        share_vendor = ontology_overlap_signal(
            _onto(cwes=["CWE-79"], vendors=["acme"]),
            _onto(cwes=["CWE-22"], vendors=["acme"]),
        )
        assert share_cwe > share_vendor


# --------------------------------------------------------------------------- #
# Label-completeness quality tier (ontology-independent)
# --------------------------------------------------------------------------- #
class TestLabelCompletenessTier:
    def test_both_labels_present_is_tier_A(self):
        assert _label_completeness_tier({"priority_score": 90.0, "priority_band": "high"}) == "A"

    def test_one_label_present_is_tier_C(self):
        assert _label_completeness_tier({"priority_score": 90.0}) == "C"
        assert _label_completeness_tier({"priority_band": "high"}) == "C"

    def test_no_completeness_labels_is_tier_F(self):
        # in_kev / ransomware are always present but carry no completeness info.
        assert _label_completeness_tier({"in_kev": True, "ransomware": False}) == "F"
        assert _label_completeness_tier({}) == "F"


# --------------------------------------------------------------------------- #
# build_sample metadata injection
# --------------------------------------------------------------------------- #
def _record(anchor_onto, positive_onto, cve_labels) -> Dict[str, Any]:
    return {
        "cve": "CVE-2024-0001",
        "encoder_view": "CVE-2024-0001. anchor view.",
        "cve_labels": cve_labels,
        "ontology": anchor_onto,
        "positive": {
            "cve": "CVE-2024-0002",
            "encoder_view": "CVE-2024-0002. positive view.",
            "ontology": positive_onto,
        },
    }


class TestBuildSampleInjection:
    def test_default_injects_completeness_tier_and_neutral_overlap(self):
        # Default config: overlap weighting OFF -> ontology_similarity neutral,
        # quality_tier from label completeness (both labels present -> A).
        config = TrainingConfig(domain_adapter="cve")
        adapter = CVERecordAdapter(config)
        record = _record(
            _onto(["CWE-79"], ["cpe:a"], ["acme"]),
            _onto(["CWE-79"], ["cpe:a"], ["acme"]),
            {"priority_score": 90.0, "priority_band": "high", "in_kev": True, "ransomware": False},
        )
        sample = adapter.build_sample(record, line_number=0, config=config)
        assert sample is not None
        assert sample.metadata["ontology_similarity"] == pytest.approx(CVE_NEUTRAL_ONTOLOGY_SIGNAL)
        assert sample.metadata["quality_tier"] == "A"

    def test_incomplete_labels_lower_tier(self):
        config = TrainingConfig(domain_adapter="cve")
        adapter = CVERecordAdapter(config)
        # Only in_kev/ransomware (no score/band) -> tier F.
        record = _record(_onto(["CWE-79"]), _onto(["CWE-79"]),
                         {"in_kev": False, "ransomware": False})
        sample = adapter.build_sample(record, line_number=0, config=config)
        assert sample is not None
        assert sample.metadata["quality_tier"] == "F"

    def test_overlap_flag_enables_computed_signal(self):
        # With the ablation flag on, ontology_similarity is the real overlap.
        config = TrainingConfig(domain_adapter="cve", cve_ontology_overlap_weighting=True)
        adapter = CVERecordAdapter(config)
        high = _record(_onto(["CWE-79"], ["cpe:a"], ["acme"]),
                       _onto(["CWE-79"], ["cpe:a"], ["acme"]),
                       {"priority_score": 90.0, "priority_band": "high"})
        low = _record(_onto(["CWE-79"], ["cpe:a"], ["acme"]),
                      _onto(["CWE-22"], ["cpe:b"], ["globex"]),
                      {"priority_score": 90.0, "priority_band": "high"})
        s_high = adapter.build_sample(high, 0, config)
        s_low = adapter.build_sample(low, 0, config)
        assert s_high.metadata["ontology_similarity"] == pytest.approx(1.0)
        assert s_low.metadata["ontology_similarity"] == pytest.approx(0.0)

    def test_preset_quality_tier_not_overwritten(self):
        config = TrainingConfig(domain_adapter="cve")
        adapter = CVERecordAdapter(config)
        record = _record(_onto(["CWE-79"]), _onto(["CWE-79"]),
                         {"priority_score": 90.0, "priority_band": "high"})
        record["metadata"] = {"quality_tier": "B"}
        sample = adapter.build_sample(record, line_number=0, config=config)
        assert sample.metadata["quality_tier"] == "B"


# --------------------------------------------------------------------------- #
# End-to-end effect through the reused ontology weight
# --------------------------------------------------------------------------- #
class TestWeightEffect:
    def _weight(self, *, tier: str, signal: float, ontology_weight: float) -> float:
        engine = ContrastiveLossEngine(
            TrainingConfig(domain_adapter="cve", ontology_weight=ontology_weight,
                           use_ot_distance=False)
        )
        return engine._compute_ontology_weight(
            {"ontology_similarity": signal, "quality_tier": tier})

    def test_completeness_tier_drives_weight_when_overlap_neutral(self):
        # Neutral overlap (0.5) -> weight == tier base. Complete (A=1.0) samples
        # weighted higher than incomplete (F=0.5) ones.
        w_complete = self._weight(tier="A", signal=0.5, ontology_weight=0.3)
        w_incomplete = self._weight(tier="F", signal=0.5, ontology_weight=0.3)
        assert w_complete == pytest.approx(1.0)
        assert w_incomplete == pytest.approx(0.5)
        assert w_complete > w_incomplete

    def test_overlap_modulates_around_tier_base_when_enabled(self):
        # Tier A base 1.0: high overlap -> 1.3, low overlap -> 0.7.
        assert self._weight(tier="A", signal=1.0, ontology_weight=0.3) == pytest.approx(1.3)
        assert self._weight(tier="A", signal=0.0, ontology_weight=0.3) == pytest.approx(0.7)

    def test_disabled_when_ontology_weight_zero(self):
        # Shipped default: no weighting at all regardless of tier/signal.
        assert self._weight(tier="A", signal=1.0, ontology_weight=0.0) == pytest.approx(1.0)
        assert self._weight(tier="F", signal=0.0, ontology_weight=0.0) == pytest.approx(1.0)
