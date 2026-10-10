"""Evaluator tests (design 9, 14): per-target AUC, collateral selection, checkpoint
resolution, the 512-token assertion and the G6 encoder cross-check. No MPNet."""
from __future__ import annotations

import functools
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.go_ontology import GoIndex
from scripts.go_withholding import evaluate_withholding as ev

DIM = 768


def vec(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.strip().encode()).digest()[:4], "big")
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


TARGETS = [
    {"go_id": "GO:0000001", "pack_id": "p01", "test_anchors": ["a0", "a1", "a2"],
     "nbr_test_proteins_sib": ["b0", "b1"]},
    {"go_id": "GO:0000002", "pack_id": "p02", "test_anchors": ["a3", "a4"],
     "nbr_test_proteins_sib": ["a5", "b2"]},
]
TABLE = {"targets": TARGETS, "n_packs": 2, "n_targets": 2,
         "packs": [{"pack_id": "p01", "targets": ["GO:0000001"]},
                   {"pack_id": "p02", "targets": ["GO:0000002"]}]}


def _views():
    v = {p: f"{p} — protein {p} text" for p in [f"a{i}" for i in range(6)] +
         [f"b{i}" for i in range(8)]}
    v["b3"] = v["b3"] + " "               # trailing space, like 3,494 real views
    return v


def make_layout(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    views = _views()
    recs = []
    for i in range(6):
        a = f"a{i}"
        for j, g in enumerate((2, 2, 1, 1, 0)):
            b = f"b{(i + j) % 8}"
            recs.append({"resume": {"protein_id": a, "encoder_view": views[a]},
                         "job": {"partner_id": b, "encoder_view": views[b], "title": b,
                                 "grade": g},
                         "metadata": {"resume_id": a, "protein_id": a, "partner_id": b,
                                      "go_grade": g}})
    splits, conv = root / "splits", root / "converted"
    splits.mkdir()
    conv.mkdir()
    test = splits / "test.jsonl"
    test.write_text("".join(json.dumps(r) + "\n" for r in recs))
    (splits / "frozen_hashes.json").write_text(json.dumps({"test_sha256": wc.sha256_file(test)}))
    (conv / "proteins.jsonl").write_text("".join(
        json.dumps({"protein_id": p, "encoder_view": t}) + "\n" for p, t in views.items()))
    terms = ["GO:1", "GO:2", "GO:3"]
    closure = {p: frozenset(terms[: (k % 3) + 1]) for k, p in enumerate(sorted(views))}
    GoIndex({t: frozenset({t}) for t in terms}, {t: 0 for t in terms}, closure, closure, {},
            {"GO:1": 0.5, "GO:2": 1.0, "GO:3": 2.0}, {"aspect": "P"}).save(conv / "go_index_P.pkl")
    tpath = root / "target_table.json"
    tpath.write_text(json.dumps(TABLE))
    return {"root": root, "test": test, "splits": splits, "converted": conv, "views": views,
            "records": recs, "table": tpath}


def write_cache(lay: dict) -> Path:
    data = ev.read_test_records(lay["test"])
    out = lay["root"] / "eval_cache"
    out.mkdir()
    npz, meta = ev.cache_paths(out, wc.sha256_file(lay["test"]))
    np.savez(npz, protein_ids=np.asarray(data["ids"], dtype=str),
             emb=np.stack([vec(t) for t in data["texts"]]), rec_anchor=data["rec_anchor"],
             rec_partner=data["rec_partner"], grade=data["grade"], query=data["query"])
    meta.write_text(json.dumps({"test_sha256": wc.sha256_file(lay["test"])}))
    return npz


def make_model(seed: int):
    from run_phase1_embedding_evaluation import CareerAwareContrastiveModel

    torch.manual_seed(seed)
    return CareerAwareContrastiveModel(input_dim=DIM, projection_dim=128, dropout=0.3,
                                       use_structured_features=False)


def make_run(root: Path, epochs: int = 2, cond: str = "p01_k0") -> Path:
    run = root / "runs" / "p01__k0__baseline__s13"
    run.mkdir(parents=True)
    for e in range(epochs):
        torch.save({"epoch": e, "model_state_dict": make_model(100 + e).state_dict()},
                   run / f"checkpoint_epoch_{e}.pt")
    name = f"checkpoint_epoch_{epochs - 1}.pt"
    lc = {"epochs": epochs, "withholding": {
        "cond_id": cond, "selected_checkpoint": name,
        "selected_checkpoint_sha256": wc.sha256_file(run / name)}}
    (run / "lc_manifest.json").write_text(json.dumps(lc))
    return run


@pytest.fixture()
def lay(tmp_path, monkeypatch):
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    lay = make_layout(tmp_path / "L")
    lay["cache"] = write_cache(lay)
    return lay


# ------------------------------------------------------------------ metrics
def test_per_target_hard_auc_equals_sklearn(lay):
    cache = ev.load_cache(lay["cache"])
    rng = np.random.default_rng(0)
    scores = np.round(rng.standard_normal(len(cache["grade"])), 1)   # ties included
    per = ev.per_target_metrics(scores, cache["grade"], cache["query"], TABLE)
    for t in TARGETS:
        m = np.isin(cache["query"], t["test_anchors"]) & (cache["grade"] > 0)
        want = roc_auc_score(cache["grade"][m] == 2, scores[m])
        assert per[t["go_id"]]["hard_auc"] == pytest.approx(want, abs=1e-12)
        assert per[t["go_id"]]["n_pos"] == int(((cache["grade"] == 2) & m).sum())
        assert per[t["go_id"]]["queries"] == len(t["test_anchors"])
    g = ev.global_metrics(scores, cache["grade"], cache["query"])
    m = cache["grade"] > 0
    assert g["hard_auc"] == pytest.approx(roc_auc_score(cache["grade"][m] == 2, scores[m]))


def test_target_without_both_grades_is_fatal(lay):
    cache = ev.load_cache(lay["cache"])
    grades = cache["grade"].copy()
    grades[np.isin(cache["query"], ["a3", "a4"]) & (grades == 1)] = 0
    with pytest.raises(ev.EvalError, match="GO:0000002"):
        ev.per_target_metrics(np.zeros(len(grades)), grades, cache["query"], TABLE)


def test_collateral_query_selection(lay):
    cache = ev.load_cache(lay["cache"])
    ids = cache["protein_ids"]
    anchors = [ids[i] for i in cache["rec_anchor"]]
    scores = np.random.default_rng(1).standard_normal(len(anchors))
    pack = ev.collateral_metrics(scores, cache["grade"], cache["query"], anchors, TABLE, "p02_k5")
    assert list(pack) == ["p02"]
    keep = [a not in {"a3", "a4", "a5", "b2"} for a in anchors]
    assert pack["p02"]["queries"] == len({a for a, k in zip(anchors, keep) if k}) == 3
    m = np.asarray(keep)
    g = cache["grade"]
    assert pack["p02"]["hard_auc"] == pytest.approx(
        ev.rank_auc(scores[m & (g == 2)], scores[m & (g == 1)]))
    allrun = ev.collateral_metrics(scores, g, cache["query"], anchors, TABLE, "all")
    assert sorted(allrun) == ["p01", "p02"]
    assert allrun["p01"]["queries"] == 3          # a3, a4, a5 (a0-a2 excluded)


# ------------------------------------------------------------------ score
def test_score_uses_final_checkpoint_from_lc_manifest(lay):
    run = make_run(lay["root"], epochs=2)
    rc = ev.main(["score", "--run-dir", str(run), "--cache", str(lay["cache"]),
                  "--target-table", str(lay["table"]), "--relation",
                  "--converted", str(lay["converted"])])
    assert rc == 0, (run / "eval" / "eval_error.json").read_text() \
        if (run / "eval" / "eval_error.json").exists() else ""
    m = json.loads((run / "eval" / "metrics.json").read_text())
    assert m["checkpoint"] == "checkpoint_epoch_1.pt"
    assert m["checkpoint_sha256"] == wc.sha256_file(run / "checkpoint_epoch_1.pt")
    assert m["epochs"] == 2 and m["cond_id"] == "p01_k0"
    assert sorted(m["per_target"]) == ["GO:0000001", "GO:0000002"]
    assert list(m["collateral"]) == ["p01"]
    # scores equal the epoch-1 projection, not epoch 0
    cache = ev.load_cache(lay["cache"])
    model = make_model(101).eval()
    with torch.no_grad():
        proj = model(torch.tensor(cache["emb"])).numpy()
    want = np.sum(proj[cache["rec_anchor"]] * proj[cache["rec_partner"]], axis=1)
    np.testing.assert_allclose(np.load(run / "eval" / "scores.npy"), want, rtol=1e-5, atol=1e-6)
    assert m["scores_sha256"] == wc.sha256_file(run / "eval" / "scores.npy")
    rel = json.loads((run / "eval" / "relation.json").read_text())
    assert rel["global"]["n_pairs"] == ev.N_GLOBAL_PAIRS
    r1 = rel["per_target"]["GO:0000001"]
    assert r1["excess"] == pytest.approx(r1["target_nbr_mean_cos"] - r1["random_ref_mean_cos"])
    assert str(wc.ROOT) not in (run / "eval" / "metrics.json").read_text()


def test_fusion_reference_is_reported_with_the_fixed_weight(lay):
    run = make_run(lay["root"], epochs=1)
    fw = lay["root"] / "fusion_weight.json"
    fw.write_text(json.dumps({"w": 0.5}))
    m = ev.score(run, lay["cache"], lay["table"], fusion_weight_file=fw,
                 converted=lay["converted"])
    fr = m["fusion_reference"]
    assert fr["w"] == 0.5 and fr["exploratory"] is True
    assert 0.0 <= fr["global_hard_auc"] <= 1.0
    assert sorted(fr["per_target_hard_auc"]) == ["GO:0000001", "GO:0000002"]


@pytest.mark.parametrize("doctor", ["sha", "epoch", "name", "missing"])
def test_score_checkpoint_mismatch_is_fatal(lay, doctor):
    run = make_run(lay["root"], epochs=2)
    lc = json.loads((run / "lc_manifest.json").read_text())
    if doctor == "sha":
        lc["withholding"]["selected_checkpoint_sha256"] = "0" * 64
    elif doctor == "epoch":
        torch.save({"epoch": 0, "model_state_dict": make_model(1).state_dict()},
                   run / "checkpoint_epoch_1.pt")
        lc["withholding"]["selected_checkpoint_sha256"] = wc.sha256_file(
            run / "checkpoint_epoch_1.pt")
    elif doctor == "name":
        lc["withholding"]["selected_checkpoint"] = "checkpoint_epoch_0.pt"
    else:
        (run / "checkpoint_epoch_1.pt").unlink()
    (run / "lc_manifest.json").write_text(json.dumps(lc))
    rc = ev.main(["score", "--run-dir", str(run), "--cache", str(lay["cache"]),
                  "--target-table", str(lay["table"])])
    assert rc == 2
    assert (run / "eval" / "eval_error.json").is_file()
    assert not (run / "eval" / "metrics.json").exists()


def test_nan_scores_are_fatal(lay):
    run = make_run(lay["root"], epochs=1)
    state = make_model(5).state_dict()
    state["projection_head.0.weight"][0, 0] = float("nan")
    torch.save({"epoch": 0, "model_state_dict": state}, run / "checkpoint_epoch_0.pt")
    lc = json.loads((run / "lc_manifest.json").read_text())
    lc["withholding"]["selected_checkpoint_sha256"] = wc.sha256_file(run / "checkpoint_epoch_0.pt")
    (run / "lc_manifest.json").write_text(json.dumps(lc))
    assert ev.main(["score", "--run-dir", str(run), "--cache", str(lay["cache"]),
                    "--target-table", str(lay["table"])]) == 2
    assert "non-finite" in (run / "eval" / "eval_error.json").read_text()


# ------------------------------------------------------------------ build-cache
class StubTokenizer:
    def __init__(self, split_on_space=False, long_text=None):
        self.split_on_space = split_on_space
        self.long_text = long_text

    def __call__(self, text, add_special_tokens=True, **_):
        if self.long_text is not None and text == self.long_text:
            return {"input_ids": list(range(512))}
        words = text.split(" ") if self.split_on_space else text.split()
        return {"input_ids": [0] + [int(hashlib.sha256(w.encode()).hexdigest()[:6], 16)
                                    for w in words] + [2]}


class StubEncoder:
    instances = []

    def __init__(self, tokenizer=None):
        self.max_seq_length = 384
        self.tokenizer = tokenizer or StubTokenizer()
        StubEncoder.instances.append(self)

    def encode(self, texts, batch_size=64, convert_to_numpy=True, **_):
        return np.stack([vec(t) for t in texts])


def test_build_cache_sets_512_writes_npz_meta_and_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    lay = make_layout(tmp_path / "L")
    StubEncoder.instances.clear()
    monkeypatch.setattr(ev, "load_encoder", lambda name, device=None: StubEncoder())
    out = tmp_path / "eval_cache"
    rc = ev.main(["build-cache", "--test", str(lay["test"]), "--frozen-hashes",
                  str(lay["splits"] / "frozen_hashes.json"), "--out", str(out),
                  "--converted", str(lay["converted"])])
    assert rc == 0
    assert StubEncoder.instances[-1].max_seq_length == 512
    sha = wc.sha256_file(lay["test"])
    npz, meta = ev.cache_paths(out, sha)
    cache = ev.load_cache(npz)
    assert cache["protein_ids"][:2] == ["a0", "b0"]           # record order
    assert cache["emb"].shape == (14, DIM) and len(cache["grade"]) == 30
    m = json.loads(meta.read_text())
    assert m["max_seq_length"] == 512 and m["max_tok"] < 512 and m["test_sha256"] == sha
    assert m["frozen_global_spearman"] is not None
    pins = wc.read_pins()
    assert [p["gate"] for p in pins] == ["G2-cache"]
    assert pins[0]["cache_npz_sha256"] == wc.sha256_file(npz)
    assert not any(isinstance(v, str) and "/" in v for v in pins[0].values())
    # refuses to overwrite
    assert ev.main(["build-cache", "--test", str(lay["test"]), "--frozen-hashes",
                    str(lay["splits"] / "frozen_hashes.json"), "--out", str(out)]) == 2


def test_build_cache_exits_2_on_a_512_token_text(tmp_path, monkeypatch):
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    lay = make_layout(tmp_path / "L")
    long_text = lay["views"]["a2"].strip()
    monkeypatch.setattr(ev, "load_encoder", lambda name, device=None:
                        StubEncoder(StubTokenizer(long_text=long_text)))
    out = tmp_path / "eval_cache"
    rc = ev.main(["build-cache", "--test", str(lay["test"]), "--frozen-hashes",
                  str(lay["splits"] / "frozen_hashes.json"), "--out", str(out)])
    assert rc == 2
    assert not list(out.glob("*.npz")) if out.exists() else True
    assert wc.read_pins() == []


def test_build_cache_test_sha_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    lay = make_layout(tmp_path / "L")
    (lay["splits"] / "frozen_hashes.json").write_text(json.dumps({"test_sha256": "0" * 64}))
    monkeypatch.setattr(ev, "load_encoder", lambda name, device=None: StubEncoder())
    assert ev.main(["build-cache", "--test", str(lay["test"]), "--frozen-hashes",
                    str(lay["splits"] / "frozen_hashes.json"),
                    "--out", str(tmp_path / "c")]) == 2


# ------------------------------------------------------------------ cross-check
def stub_trainer(tokenizer, max_seq_length=512):
    from contrastive_learning.embedding_cache import BatchEfficientEncoder

    enc = StubEncoder(tokenizer)
    enc.max_seq_length = max_seq_length
    return SimpleNamespace(
        text_encoder=enc,
        batch_encoder=SimpleNamespace(
            _content_to_text=functools.partial(BatchEfficientEncoder._content_to_text, None)))


def test_crosscheck_trailing_space_differs_in_text_not_tokens(lay):
    run = make_run(lay["root"], epochs=2)
    res = ev.score(run, lay["cache"], lay["table"], crosscheck_config=Path("unused.json"),
                   converted=lay["converted"],
                   trainer_factory=lambda cfg, d, dev: stub_trainer(StubTokenizer()))
    cc = json.loads((run / "eval" / "encoder_crosscheck.json").read_text())
    assert cc["n_text_differs"] == 1 and cc["n_token_mismatch"] == 0
    assert cc["n_token_checked"] == 14 and cc["n"] == 14 and cc["min_cos"] >= 0.9999
    assert cc["max_seq_length"] == 512
    assert res["checkpoint"] == "checkpoint_epoch_1.pt"


def test_crosscheck_token_mismatch_exits_2(lay, monkeypatch):
    run = make_run(lay["root"], epochs=2)
    monkeypatch.setattr(ev, "build_crosscheck_trainer",
                        lambda cfg, d, dev: stub_trainer(StubTokenizer(split_on_space=True)))
    rc = ev.main(["score", "--run-dir", str(run), "--cache", str(lay["cache"]),
                  "--target-table", str(lay["table"]), "--encoder-crosscheck",
                  "--config", str(lay["root"] / "baseline.json"),
                  "--converted", str(lay["converted"])])
    assert rc == 2
    err = json.loads((run / "eval" / "eval_error.json").read_text())
    assert "token ids" in err["error"] and err["ids"] == ["b3"]
    assert not (run / "eval" / "metrics.json").exists()


def test_crosscheck_requires_512_and_config(lay):
    run = make_run(lay["root"], epochs=2)
    with pytest.raises(ev.EvalError, match="max_seq_length"):
        ev.score(run, lay["cache"], lay["table"], crosscheck_config=Path("x.json"),
                 converted=lay["converted"],
                 trainer_factory=lambda c, d, dev: stub_trainer(StubTokenizer(), 384))
    with pytest.raises(SystemExit) as e:
        ev.main(["score", "--run-dir", str(run), "--cache", str(lay["cache"]),
                 "--target-table", str(lay["table"]), "--encoder-crosscheck"])
    assert e.value.code == 2


def test_simgic_matches_relation_table(lay):
    from go_ppi_domain.go_relation import GoRelationTable

    idx = GoIndex.load(lay["converted"] / "go_index_P.pkl")
    ids = sorted(lay["views"])
    tab = GoRelationTable.from_index(idx, ids)
    sim = ev.SimGic(idx)
    for a, b in random.Random(0).sample([(x, y) for x in ids for y in ids], 40):
        assert sim(a, b) == pytest.approx(float(tab.submatrix([a, b])[0, 1] if a != b
                                                else tab.submatrix([a])[0, 0]), abs=1e-6)
