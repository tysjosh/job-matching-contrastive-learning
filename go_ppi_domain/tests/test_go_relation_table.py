"""GoRelationTable and the pure go_relation_loss (design 5.2, 14)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from contrastive_learning.data_structures import ContrastiveTriplet
from go_ppi_domain.go_ontology import GoMatcher
from go_ppi_domain.go_relation import GoRelationTable, go_relation_loss
from go_ppi_domain.tests._withholding_fixtures import (
    build_index, write_identity_fixture)


def _index():
    direct = {
        "g1": ["T5"], "g2": ["T1"], "g3": ["T2", "T3"], "g4": ["T6"],
        "g5": ["T4", "T5"], "g6": ["T3"], "g7": ["T6", "T1"],
    }
    return build_index(direct)


def test_dense_table_equals_matcher_simgic_on_closures():
    index = _index()
    proteins = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "none1", "none2"]
    table = GoRelationTable.from_index(index, proteins)
    matcher = GoMatcher(index, similarity_mode="simgic")
    assert table.ids == sorted(proteins)
    for a in table.ids:
        for b in table.ids:
            expected = matcher.ontology_set_similarity(
                sorted(index.closure_for(a)), sorted(index.closure_for(b)))
            got = float(table.sim[table.row[a], table.row[b]])
            assert got == pytest.approx(expected, abs=1e-6), (a, b)
    assert table.annotated.tolist() == [p not in {"none1", "none2"} for p in table.ids]
    assert table.sim.dtype == np.float32
    assert table.source_meta["aspect"] == "P"


def test_submatrix_and_sha256():
    index = _index()
    t1 = GoRelationTable.from_index(index, ["g3", "g1", "g2"])
    t2 = GoRelationTable.from_index(index, ["g1", "g2", "g3"])
    assert t1.sha256() == t2.sha256()
    sub = t1.submatrix(["g2", "g1"])
    assert sub.shape == (2, 2)
    assert sub[0, 1] == t1.sim[t1.row["g2"], t1.row["g1"]]
    assert GoRelationTable.from_index(index, ["g1", "g2"]).sha256() != t1.sha256()


def test_from_train_file_protein_set_equals_train_file_proteins(tmp_path):
    fx = write_identity_fixture(tmp_path)
    table = GoRelationTable.from_train_file(fx["index"], fx["train_file"])
    expected = set()
    for line in Path(fx["train_file"]).read_text().splitlines():
        rec = json.loads(line)
        expected |= {rec["resume"]["protein_id"], rec["job"]["partner_id"]}
    assert table.ids == sorted(expected)
    assert not any(table.annotated[table.row[a]] for a in fx["unannotated"])


def test_table_has_no_matcher_api():
    table = GoRelationTable.from_index(_index(), ["g1", "g2"])
    for name in ("ontology_set_similarity", "terms_for_gene", "branch_distance",
                 "coarse_for_gene", "index"):
        assert not hasattr(table, name), name


def _triplets_and_embeddings():
    def slot(pid, key):
        return {key: pid, "encoder_view": f"view {pid}"}
    anchors = [slot("g1", "protein_id"), slot("g3", "protein_id"), slot("g1", "protein_id")]
    positives = [slot("g2", "partner_id"), slot("g5", "partner_id"), slot("none1", "partner_id")]
    negatives = [slot("g4", "partner_id"), slot("g6", "partner_id")]
    triplets = [ContrastiveTriplet(a, p, negatives, [5.0, 5.0], {})
                for a, p in zip(anchors, positives)]
    torch.manual_seed(0)
    key = lambda s: s["encoder_view"]  # noqa: E731
    embeddings = {key(s): torch.randn(4, requires_grad=True)
                  for s in anchors + positives + negatives}
    return triplets, embeddings, key


def test_go_relation_loss_is_pure_and_excludes_negatives():
    table = GoRelationTable.from_index(_index(), ["g1", "g2", "g3", "g4", "g5", "g6",
                                                  "none1"])
    triplets, embeddings, key = _triplets_and_embeddings()
    snap_trip = copy.deepcopy([(t.anchor, t.positive, t.negatives) for t in triplets])
    snap_emb = {k: v.detach().clone() for k, v in embeddings.items()}
    snap_sim = table.sim.copy()
    loss1, ids1, miss1 = go_relation_loss(table, triplets, embeddings, key)
    loss2, ids2, miss2 = go_relation_loss(table, triplets, embeddings, key)
    assert torch.equal(loss1, loss2) and ids1 == ids2 and miss1 == miss2 == 0
    assert ids1 == ["g1", "g2", "g3", "g5", "none1"]
    assert [(t.anchor, t.positive, t.negatives) for t in triplets] == snap_trip
    assert all(torch.equal(embeddings[k], v) for k, v in snap_emb.items())
    assert np.array_equal(table.sim, snap_sim)
    loss1.backward()
    for pid in ("g4", "g6"):   # negatives get no gradient from L_GO
        assert embeddings[f"view {pid}"].grad is None


def test_go_relation_loss_missing_and_too_few():
    table = GoRelationTable.from_index(_index(), ["g1", "none1"])
    triplets, embeddings, key = _triplets_and_embeddings()
    loss, ids, missing = go_relation_loss(table, triplets, embeddings, key)
    assert loss is None
    assert ids == ["g1", "none1"]
    assert missing == 3   # g2, g3, g5 not in the table
