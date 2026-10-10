"""L_GO wiring in ContrastiveLossEngine (design 5.3, 14).

The weight-0 tests are the proof that the default reproduces current behaviour:
same loss tensor (``torch.equal``) and identical gradients as (a) an engine whose
config does not carry the field at all and (b) the pre-edit loss engine loaded
from the task snapshot, for go_ordinal and InfoNCE, with and without a table.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from contrastive_learning.data_structures import ContrastiveTriplet, TrainingConfig
from contrastive_learning.loss_engine import ContrastiveLossEngine
from go_ppi_domain.go_relation import GoRelationTable
from go_ppi_domain.tests._withholding_fixtures import build_index
# Registers the "go_ppi" adapter at collection time. test_go_ordinal_loss.py's
# trainer test needs it and otherwise depends on another module importing it.
import go_ppi_domain.record_adapter  # noqa: F401,E402

PRE_EDIT_ENGINE = (Path(__file__).resolve().parents[2] / ".agents" / "tasks"
                   / "ontology-withholding-feasibility" / "task-go-ppi-withholding"
                   / "pre_edit" / "loss_engine.py")

DIRECT = {"pA": ["T5"], "pB": ["T1"], "pC": ["T2", "T3"], "pD": ["T6"],
          "pE": ["T4"], "pF": ["T3"]}


def _cfg(loss_type="go_ordinal", **kw):
    return TrainingConfig(domain_adapter="go_ppi", loss_type=loss_type,
                          ontology_weight=0.0, **kw)


def _cfg_without_field(loss_type):
    """A config object on which ``go_ontology_loss_weight`` does not exist."""
    d = dataclasses.asdict(_cfg(loss_type))
    del d["go_ontology_loss_weight"]
    ns = SimpleNamespace(**d)
    assert not hasattr(ns, "go_ontology_loss_weight")
    return ns


def _table(proteins=("pA", "pB", "pC", "pD", "pE", "pF", "pU")):
    return GoRelationTable.from_index(build_index(DIRECT), proteins)


def _slot(pid, id_key, grade=None):
    s = {id_key: pid, "encoder_view": f"{pid} view"}
    if grade is not None:
        s["grade"] = grade
    return s


def _batch(engine, seed=0, positives=("pB", "pD")):
    """Two triplets (anchors pA, pC) with graded negatives pE (1) and pF (0)."""
    anchors = [_slot("pA", "protein_id"), _slot("pC", "protein_id")]
    pos = [_slot(p, "partner_id") for p in positives]
    negs = [_slot("pE", "partner_id", 1), _slot("pF", "partner_id", 0)]
    triplets = [ContrastiveTriplet(a, p, [dict(n) for n in negs], [5.0, 5.0],
                                   {"positive_original_label": "good_fit"})
                for a, p in zip(anchors, pos)]
    g = torch.Generator().manual_seed(seed)
    embeddings = {}
    for s in anchors + pos + negs:
        embeddings[engine._get_content_key(s)] = torch.nn.functional.normalize(
            torch.randn(8, generator=g), dim=0).requires_grad_(True)
    return triplets, embeddings


def _loss_and_grads(engine, triplets, embeddings):
    leaves = list(embeddings.values())
    loss = engine.compute_loss(triplets, embeddings)
    grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    return loss, grads


def _load_pre_edit_engine_class():
    if not PRE_EDIT_ENGINE.exists():
        pytest.skip("pre-edit loss_engine snapshot not present")
    spec = importlib.util.spec_from_file_location(
        "contrastive_learning._pre_edit_loss_engine", PRE_EDIT_ENGINE,
        submodule_search_locations=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "contrastive_learning"
    spec.loader.exec_module(module)
    return module.ContrastiveLossEngine


def _assert_identical(a, b):
    (la, ga), (lb, gb) = a, b
    assert torch.equal(la, lb)
    assert len(ga) == len(gb)
    for x, y in zip(ga, gb):
        assert (x is None and y is None) or torch.equal(x, y)


@pytest.mark.parametrize("loss_type", ["go_ordinal", "infonce"])
@pytest.mark.parametrize("with_table", [False, True])
def test_weight_zero_is_identical_to_field_absent(loss_type, with_table):
    new = ContrastiveLossEngine(_cfg(loss_type))
    assert new.go_ontology_loss_weight == 0.0
    if with_table:
        new.set_go_relation_table(_table())
    absent = ContrastiveLossEngine(_cfg_without_field(loss_type))
    triplets, embeddings = _batch(new)
    _assert_identical(_loss_and_grads(new, triplets, embeddings),
                      _loss_and_grads(absent, triplets, embeddings))
    assert new.go_relation_applied_batches == new.go_relation_skipped_batches == 0


@pytest.mark.parametrize("loss_type", ["go_ordinal", "infonce"])
@pytest.mark.parametrize("with_table", [False, True])
def test_weight_zero_is_identical_to_pre_edit_engine(loss_type, with_table):
    Old = _load_pre_edit_engine_class()
    new = ContrastiveLossEngine(_cfg(loss_type))
    if with_table:
        new.set_go_relation_table(_table())
    old = Old(_cfg(loss_type))
    assert not hasattr(old, "go_ontology_loss_weight")
    triplets, embeddings = _batch(new)
    _assert_identical(_loss_and_grads(new, triplets, embeddings),
                      _loss_and_grads(old, triplets, embeddings))
    # The single-triplet InfoNCE return path too.
    _assert_identical(_loss_and_grads(new, triplets[:1], embeddings),
                      _loss_and_grads(old, triplets[:1], embeddings))


def _hand_lgo(engine, embeddings, table, ids):
    keys = [engine._get_content_key({"encoder_view": f"{p} view"}) for p in ids]
    E = np.stack([embeddings[k].detach().numpy().astype(np.float64) for k in keys])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    C = E @ E.T
    S = table.submatrix(ids).astype(np.float64)
    vals = [(C[i, j] - S[i, j]) ** 2 for i in range(len(ids)) for j in range(i + 1, len(ids))]
    return float(np.mean(vals))


@pytest.mark.parametrize("loss_type", ["go_ordinal", "infonce"])
def test_lgo_matches_hand_computation_for_three_proteins(loss_type):
    table = _table()
    w = 0.3
    base = ContrastiveLossEngine(_cfg(loss_type))
    eng = ContrastiveLossEngine(_cfg(loss_type, go_ontology_loss_weight=w))
    eng.set_go_relation_table(table)
    # anchors pA, pC; positives pB and pU (unannotated, excluded) -> 3 proteins.
    triplets, embeddings = _batch(eng, positives=("pB", "pU"))
    total = eng.compute_loss(triplets, embeddings)
    expected = base.compute_loss(triplets, embeddings).item() + w * _hand_lgo(
        eng, embeddings, table, ["pA", "pB", "pC"])
    assert total.item() == pytest.approx(expected, abs=1e-6)
    assert eng.go_relation_applied_batches == 1
    assert eng.go_relation_seen_ids == {"pA", "pB", "pC", "pU"}


def test_negatives_receive_no_gradient_from_lgo():
    eng = ContrastiveLossEngine(_cfg(go_ontology_loss_weight=0.5))
    eng.set_go_relation_table(_table())
    triplets, embeddings = _batch(eng)
    term = eng._go_relation_loss(triplets, embeddings)
    neg_keys = [eng._get_content_key(n) for n in triplets[0].negatives]
    grads = torch.autograd.grad(term, [embeddings[k] for k in neg_keys], allow_unused=True)
    assert all(g is None for g in grads)
    pos_grad = torch.autograd.grad(
        eng._go_relation_loss(triplets, embeddings),
        [embeddings[eng._get_content_key(triplets[0].positive)]])[0]
    assert torch.count_nonzero(pos_grad) > 0


def test_unannotated_and_missing_proteins_are_excluded():
    eng = ContrastiveLossEngine(_cfg(go_ontology_loss_weight=0.5))
    table = _table(("pA", "pB", "pC", "pU"))   # pD missing from the table
    eng.set_go_relation_table(table)
    triplets, embeddings = _batch(eng, positives=("pB", "pD"))
    term = eng._go_relation_loss(triplets, embeddings)
    assert term.item() == pytest.approx(_hand_lgo(eng, embeddings, table,
                                                  ["pA", "pB", "pC"]), abs=1e-6)
    assert eng.go_relation_missing_ids == 1
    # Only one annotated protein -> skipped, base loss returned unchanged.
    eng2 = ContrastiveLossEngine(_cfg(go_ontology_loss_weight=0.5))
    eng2.set_go_relation_table(_table(("pA", "pU")))
    base = ContrastiveLossEngine(_cfg())
    assert torch.equal(eng2.compute_loss(triplets, embeddings),
                       base.compute_loss(triplets, embeddings))
    assert eng2.go_relation_skipped_batches == 1 and eng2.go_relation_applied_batches == 0


def test_weight_positive_without_table_raises():
    eng = ContrastiveLossEngine(_cfg(go_ontology_loss_weight=0.2))
    triplets, embeddings = _batch(eng)
    with pytest.raises(RuntimeError, match="no GoRelationTable attached"):
        eng.compute_loss(triplets, embeddings)


@pytest.mark.parametrize("weight", [-0.1, math.inf, math.nan])
def test_bad_weight_raises(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ContrastiveLossEngine(_cfg(go_ontology_loss_weight=weight))


def test_wrong_domain_or_loss_type_raises():
    with pytest.raises(ValueError, match="go_ppi"):
        ContrastiveLossEngine(TrainingConfig(loss_type="infonce", go_ontology_loss_weight=0.2))
    with pytest.raises(ValueError, match="go_ordinal or infonce"):
        ContrastiveLossEngine(_cfg("ordinal", go_ontology_loss_weight=0.2))


def test_components_base_total_equals_go_ordinal_loss():
    eng = ContrastiveLossEngine(_cfg())
    triplets, embeddings = _batch(eng)
    comps = eng.go_ordinal_components(triplets, embeddings, relation_table=_table())
    ref = eng._compute_go_ordinal_loss(triplets, embeddings)
    assert abs(comps["base_total"].item() - ref.item()) <= 1e-6
    assert comps["infonce"].item() + comps["ranking"].item() == pytest.approx(
        comps["base_total"].item(), abs=1e-6)
    assert comps["go_relation"] is not None
    assert eng.go_ordinal_components(triplets, embeddings)["go_relation"] is None


def _counters(eng):
    return (set(eng.go_relation_seen_ids), eng.go_relation_missing_ids,
            eng.go_relation_skipped_batches, eng.go_relation_applied_batches)


def test_counters_update_only_with_grad_and_not_diagnostic():
    eng = ContrastiveLossEngine(_cfg(go_ontology_loss_weight=0.3))
    eng.set_go_relation_table(_table())
    triplets, embeddings = _batch(eng)
    with torch.no_grad():
        eng.compute_loss(triplets, embeddings)
    assert _counters(eng) == (set(), 0, 0, 0)
    eng._diagnostic_mode = True
    eng.compute_loss(triplets, embeddings)
    assert _counters(eng) == (set(), 0, 0, 0)
    eng._diagnostic_mode = False
    eng.compute_loss(triplets, embeddings)
    assert _counters(eng) == ({"pA", "pB", "pC", "pD"}, 0, 0, 1)


def test_components_with_other_table_leave_counters_unchanged():
    eng = ContrastiveLossEngine(_cfg(go_ontology_loss_weight=0.3))
    eng.set_go_relation_table(_table())
    triplets, embeddings = _batch(eng)
    eng.compute_loss(triplets, embeddings)
    before = _counters(eng)
    t2 = _table(("pA", "pC"))
    comps = eng.go_ordinal_components(triplets, embeddings, relation_table=t2)
    assert comps["go_relation"] is not None
    assert _counters(eng) == before
