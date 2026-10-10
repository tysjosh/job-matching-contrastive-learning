"""Negative-selection identity for the withholding study (design 5.4, 5.6, 14).

* Unannotated-anchor window flag (5.6) on a synthetic index where anchor ``U``
  has no GO terms and anchor ``A`` is annotated; the flag-false case is compared
  with ``fixtures/selector_flag_false_expected.json``, recorded from the selector
  BEFORE the flag existed.
* Loss-only isolation (5.4): configs that differ only in
  ``go_ontology_loss_weight`` select identical negatives and never build a matcher.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.loss_engine import ContrastiveLossEngine
from go_ppi_domain.negative_selector import (
    CAREER_DISTANCE_SCALE, NEUTRAL_DISTANCE, GoPpiNegativeSelector)
from go_ppi_domain.tests._withholding_fixtures import (
    FLAG_FALSE_EXPECTED, SMALL_EPOCHS, SMALL_SEEDS, make_sample,
    record_neg_only_outputs, select, small_selector, write_identity_fixture,
    write_small_fixture)
from go_ppi_domain.withholding_diagnostics import SelectionRecorder
from scripts.run_learning_curve_point import attach_go_ppi, attach_go_relation_loss


@pytest.fixture
def small(tmp_path):
    return write_small_fixture(tmp_path)


def _pairs(out):
    return list(zip(out["partner_ids"], out["grades"]))


# ------------------------------------------------------------- flag (5.6)
def test_flag_true_unannotated_anchor_matches_baseline(small):
    for seed in SMALL_SEEDS:
        neg = small_selector(small, "neg_only", seed, unannotated_anchor_window=True)
        base = small_selector(small, "baseline", seed, unannotated_anchor_window=True)
        assert neg.matcher is not None and not neg.matcher.terms_for_gene("U")
        for e in SMALL_EPOCHS:
            got, ref = select(neg, "U", e), select(base, "U", e)
            assert _pairs(got) == _pairs(ref), (seed, e)
            assert got["distances"] == [CAREER_DISTANCE_SCALE * NEUTRAL_DISTANCE] * len(
                got["distances"])


def test_flag_true_annotated_anchor_still_go_ordered(small):
    for seed in SMALL_SEEDS:
        neg = small_selector(small, "neg_only", seed, unannotated_anchor_window=True)
        base = small_selector(small, "baseline", seed, unannotated_anchor_window=True)
        assert any(_pairs(select(neg, "A", e)) != _pairs(select(base, "A", e))
                   for e in SMALL_EPOCHS)


def test_flag_false_reproduces_frozen_pre_edit_output(small):
    expected = json.loads(FLAG_FALSE_EXPECTED.read_text(encoding="utf-8"))["outputs"]
    assert record_neg_only_outputs(small) == expected
    assert record_neg_only_outputs(small, unannotated_anchor_window=False) == expected


def test_from_config_default_is_false(small):
    cfg = SimpleNamespace(go_ppi_split_dir=str(small["split_dir"]),
                          go_ppi_converted_dir=str(small["converted_dir"]))
    assert not hasattr(cfg, "go_ppi_unannotated_anchor_window")
    assert GoPpiNegativeSelector.from_config(cfg).unannotated_anchor_window is False
    tc = TrainingConfig(domain_adapter="go_ppi", go_ppi_split_dir=str(small["split_dir"]),
                        go_ppi_converted_dir=str(small["converted_dir"]))
    assert tc.go_ppi_unannotated_anchor_window is False
    assert GoPpiNegativeSelector.from_config(tc).unannotated_anchor_window is False
    tc.go_ppi_unannotated_anchor_window = True
    assert GoPpiNegativeSelector.from_config(tc).unannotated_anchor_window is True


def test_recorder_counts_unannotated_anchor_calls(small):
    selector = small_selector(small, "neg_only", 13, unannotated_anchor_window=True)
    rec = SelectionRecorder(selector, unannotated_anchors={"U"})
    for e in (0, 1):
        for anchor in ("U", "A", "U"):
            assert selector.select_batch_negatives(make_sample(anchor), [], 6, e) is not None
        rec.on_epoch_end(None, e, {})
    assert rec.unannotated_anchor_calls_by_epoch == {"0": 2, "1": 2}
    assert rec.calls_by_epoch == {"0": 3, "1": 3}
    assert rec.calls == 6 and rec.none_returns == rec.exceptions == 0


# ------------------------------------------------- loss-only isolation (5.4)
class RecordingProcessor:
    """Batch-processor stub recording every seam call."""

    def __init__(self):
        self.matcher_calls = []
        self.ontology_matcher_override = None
        self.domain_negative_selector = None

    def set_ontology_matcher(self, matcher, coarse_distance_fn=None):
        self.matcher_calls.append(matcher)
        self.ontology_matcher_override = matcher

    def set_domain_negative_selector(self, selector):
        self.domain_negative_selector = selector


def _wire(fx, tiered: bool, weight: float, seed: int = 13):
    cfg = TrainingConfig(
        domain_adapter="go_ppi", loss_type="go_ordinal", ontology_weight=0.0,
        num_epochs=15, training_seed=seed,
        go_ppi_split_dir=str(fx["split_dir"]), go_ppi_converted_dir=str(fx["converted_dir"]),
        go_index_cache=str(fx["index"]), go_aspect="P",
        go_ppi_go_tiered_negatives=tiered,
        go_ppi_tier_sampling="stochastic" if tiered else "random_window",
        go_ppi_tier_window_frac=0.34, go_ppi_unannotated_anchor_window=True,
        go_ontology_loss_weight=weight)
    trainer = SimpleNamespace(batch_processor=RecordingProcessor(),
                              loss_engine=ContrastiveLossEngine(cfg))
    attach = attach_go_ppi(trainer, cfg)
    relation = attach_go_relation_loss(trainer, cfg, fx["train_file"])
    return trainer, attach, relation


def _draws(trainer, fx, with_distances=False):
    selector = trainer.batch_processor.domain_negative_selector
    out = {}
    for anchor in fx["anchors"]:
        for e in range(15):
            r = selector.select_batch_negatives(
                make_sample(anchor, fx["partners"][anchor]), [], 7, e)
            assert r is not None
            negs, dist = r
            key = [(n["partner_id"], n["grade"]) for n in negs]
            out[(anchor, e)] = (key, list(dist)) if with_distances else key
    return out


def test_loss_only_selects_identical_negatives(tmp_path):
    fx = write_identity_fixture(tmp_path)
    base, _, rel0 = _wire(fx, tiered=False, weight=0.0)
    loss, _, rel1 = _wire(fx, tiered=False, weight=0.3)
    for t in (base, loss):
        assert t.batch_processor.matcher_calls == []
        assert t.batch_processor.ontology_matcher_override is None
        assert t.batch_processor.domain_negative_selector.matcher is None
        assert t.batch_processor.domain_negative_selector.unannotated_anchor_window is True
    assert rel0 == {"weight": 0.0, "table": None}
    assert base.loss_engine.go_relation_table is None
    assert loss.loss_engine.go_relation_table is not None
    assert rel1["weight"] == 0.3 and rel1["table_proteins"] == 2 * len(fx["anchors"])
    for name in ("ontology_set_similarity", "terms_for_gene", "branch_distance"):
        assert not hasattr(loss.loss_engine.go_relation_table, name)
    assert _draws(base, fx) == _draws(loss, fx)


def test_neg_only_and_both_select_identical_negatives(tmp_path):
    fx = write_identity_fixture(tmp_path)
    neg, _, _ = _wire(fx, tiered=True, weight=0.0)
    both, _, rel = _wire(fx, tiered=True, weight=0.3)
    for t in (neg, both):
        assert len(t.batch_processor.matcher_calls) == 1
        assert t.batch_processor.domain_negative_selector.matcher is not None
    assert neg.loss_engine.go_relation_table is None
    assert both.loss_engine.go_relation_table is not None and rel["weight"] == 0.3
    assert _draws(neg, fx, with_distances=True) == _draws(both, fx, with_distances=True)
    # The negative factor itself changes the draws (the fixture is not inert).
    base, _, _ = _wire(fx, tiered=False, weight=0.0)
    assert _draws(base, fx) != _draws(neg, fx)
    # Unannotated anchors get the control's draws in the tiered cells (5.6).
    bd, nd = _draws(base, fx), _draws(neg, fx)
    assert fx["unannotated"]
    for a in fx["unannotated"]:
        assert all(bd[(a, e)] == nd[(a, e)] for e in range(15))


def test_attach_go_relation_loss_rejects_wrong_domain(tmp_path):
    cfg = SimpleNamespace(go_ontology_loss_weight=0.3, domain_adapter="career")
    with pytest.raises(ValueError, match="go_ppi"):
        attach_go_relation_loss(SimpleNamespace(), cfg, tmp_path / "x.jsonl")
