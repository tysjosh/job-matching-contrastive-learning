"""Text-similarity covariate (design 10): stub encoder vs a hand computation."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from go_ppi_domain import withholding_constants as wc
from scripts.go_withholding import text_similarity_covariate as tc

#: Fixed 3-d vectors per protein (not unit length; the script normalises).
VEC = {
    "q1": [1, 0, 0], "q2": [0, 1, 0],                    # test anchors (P_test)
    "p1": [1, 1, 0], "p2": [2, 0, 1],                    # P_train
    "x1": [0, 0, 1],                                     # pool-only annotated (T_train only)
    "n1": [1, 2, 0], "n2": [0, 1, 3],                    # NP_train_sib
    "m1": [3, 1, 1],                                     # NP_train_2 extra
    "r1": [1, 0, 2], "r2": [0, 3, 1], "r3": [2, 2, 2], "r4": [1, -1, 0],
}


def _text(p):
    return f"{p} — view of {p}"


class StubEncoder:
    calls = []

    def encode(self, texts, batch_size=64, convert_to_numpy=True, **_):
        StubEncoder.calls.append(list(texts))
        return np.asarray([VEC[t.split(" ")[0]] for t in texts], dtype=np.float32)


def _rec(a, b):
    return {"resume": {"protein_id": a, "encoder_view": _text(a)},
            "job": {"partner_id": b, "encoder_view": _text(b)},
            "metadata": {"protein_id": a, "partner_id": b, "go_grade": 2}}


@pytest.fixture()
def lay(tmp_path, monkeypatch):
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    monkeypatch.setattr(tc, "N_REFERENCE", 3)
    monkeypatch.setattr(tc, "load_encoder", lambda *a, **k: StubEncoder())
    StubEncoder.calls.clear()
    splits, conv = tmp_path / "splits", tmp_path / "converted"
    splits.mkdir()
    conv.mkdir()
    train = [_rec("p1", "n1"), _rec("p2", "r1"), _rec("n2", "r2"), _rec("m1", "r3")]
    (splits / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    pools = [{"anchor_id": "x1", "split": "train", "weak_evidence": ["r4"], "no_interaction": []},
             {"anchor_id": "q1", "split": "test", "weak_evidence": ["q2"], "no_interaction": []}]
    (splits / "negative_pools.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pools))
    (conv / "proteins.jsonl").write_text("".join(
        json.dumps({"protein_id": p, "encoder_view": _text(p)}) + "\n" for p in VEC))
    table = {"targets": [{"go_id": "GO:1", "pack_id": "p01", "test_anchors": ["q1", "q2"],
                          "train_proteins": ["p1", "p2"],
                          "train_annotated_proteins": ["p1", "p2", "x1"],
                          "nbr_train_proteins_sib": ["n1", "n2"],
                          "nbr_train_proteins_2": ["n1", "m1"]}]}
    tpath = tmp_path / "target_table.json"
    tpath.write_text(json.dumps(table))
    cache = tmp_path / "test_x.npz"
    np.savez(cache, protein_ids=np.asarray(["q1", "q2"]),
             emb=np.asarray([VEC["q1"], VEC["q2"]], dtype=np.float32))
    return {"root": tmp_path, "splits": splits, "converted": conv, "table": tpath,
            "cache": cache}


def _cos(a, b):
    a, b = np.asarray(VEC[a], float), np.asarray(VEC[b], float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _hand_excess(P_test, NP, ref):
    per_i = []
    for i in P_test:
        nbr = sum(_cos(i, j) for j in NP) / len(NP)
        rr = sum(_cos(i, j) for j in ref) / len(ref)
        per_i.append(nbr - rr)
    return sum(per_i) / len(per_i)


def _hand_train(P, NP):
    vals = [_cos(a, b) for a in P for b in NP]
    return sum(vals) / len(vals)


def test_covariate_matches_hand_computation(lay):
    out = lay["root"] / "targets" / "text_covariate.json"
    rc = tc.main(["--target-table", str(lay["table"]), "--cache", str(lay["cache"]),
                  "--out", str(out), "--splits", str(lay["splits"]),
                  "--converted", str(lay["converted"])])
    assert rc == 0
    doc = json.loads(out.read_text())
    row = doc["targets"]["GO:1"]
    partition = ["m1", "n1", "n2", "p1", "p2", "r1", "r2", "r3", "r4", "x1"]
    # reference: 3 train-partition proteins outside T_train | NP, Random(0) on the sorted list
    ref_sib = random.Random(0).sample(sorted(set(partition) - {"p1", "p2", "x1", "n1", "n2"}), 3)
    ref_2 = random.Random(0).sample(sorted(set(partition) - {"p1", "p2", "x1", "n1", "m1"}), 3)
    assert row["tsim_excess"] == pytest.approx(_hand_excess(["q1", "q2"], ["n1", "n2"], ref_sib))
    assert row["tsim_train"] == pytest.approx(_hand_train(["p1", "p2"], ["n1", "n2"]))
    assert row["tsim_excess_2"] == pytest.approx(_hand_excess(["q1", "q2"], ["n1", "m1"], ref_2))
    assert row["tsim_train_2"] == pytest.approx(_hand_train(["p1", "p2"], ["n1", "m1"]))
    assert row["n_reference"] == 3 and "x1" not in ref_sib
    assert doc["n_from_cache"] == 0 and doc["n_encoded"] == doc["n_train_needed"]
    pins = wc.read_pins()
    assert pins[-1]["gate"] == "G2-covariate"
    assert pins[-1]["text_covariate_sha256"] == wc.sha256_file(out)
    # refuses to overwrite
    assert tc.main(["--target-table", str(lay["table"]), "--cache", str(lay["cache"]),
                    "--out", str(out), "--splits", str(lay["splits"]),
                    "--converted", str(lay["converted"])]) == 2


def test_trainer_cache_entries_are_used_before_encoding(lay):
    from contrastive_learning.embedding_cache import EmbeddingCache

    keyer = EmbeddingCache(max_cache_size=0, device=torch.device("cpu"), enable_stats=False)
    fake = torch.tensor([0.0, 0.0, 5.0])                  # differs from VEC["n1"]
    cache_file = lay["root"] / "train_emb.pt"
    torch.save({"cache": {keyer.get_content_key({"encoder_view": _text("n1")}): fake},
                "metadata": {}}, cache_file)
    views = tc.load_views(lay["converted"])
    res = tc.train_embeddings(["n1", "n2"], views, cache_file,
                              lambda texts: StubEncoder().encode(texts))
    assert res["n_from_cache"] == 1 and res["n_encoded"] == 1
    np.testing.assert_allclose(res["vectors"]["n1"], [0, 0, 5])
    np.testing.assert_allclose(res["vectors"]["n2"], VEC["n2"])
    assert StubEncoder.calls[-1] == [_text("n2")]


def test_excess_is_none_without_neighbours():
    assert tc.excess(np.ones((2, 3)), np.zeros((0, 3)), np.ones((1, 3))) is None
    assert tc._mean_cos(np.ones((1, 3)), np.zeros((0, 3))) is None
