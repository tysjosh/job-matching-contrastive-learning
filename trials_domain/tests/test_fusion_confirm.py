"""Gate-1 unit tests for the TREC-CT MeSH fusion confirmation (protocol_trials_fusion.md §5.1).

Synthetic fixtures only, except: the protocol-parser test reads the protocol file and the
gate-2 classifier test reads the key structure of the published fusion JSON (both are
part of the transfer, §6.2). Nothing here trains, encodes or scores a real model.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from trials_domain import fusion_confirm as fc

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "trials_fusion_confirm"
PROTOCOL = ROOT / fc.PROTOCOL_REL
PUBLISHED = ROOT / fc.PUBLISHED_FUSION_REL


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"tfc_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def restore_root():
    old_root, old_cwd = fc.ROOT, os.getcwd()
    yield
    fc.ROOT = old_root
    os.chdir(old_cwd)


# ----------------------------------------------------------------- fusion and weight selection
def test_fusion_arithmetic():
    t, p = np.array([1.0, -1.0, 0.5]), np.array([0.0, 2.0, -2.0])
    assert np.allclose(fc.fuse(t, p, 0.2), 0.8 * t + 0.2 * p)
    assert np.array_equal(fc.fuse(t, p, 0.0), t)


def test_weight_tie_goes_to_smaller_w(monkeypatch):
    values = {0.0: 0.50, 0.1: 0.55, 0.2: 0.60, 0.3: 0.60, 0.4: 0.58, 0.5: 0.40}
    calls = iter(fc.WEIGHT_GRID)
    monkeypatch.setattr(fc, "validation_ndcg", lambda s, g, q: values[next(calls)])
    w, curve = fc.select_weight(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    assert w == 0.2 and curve["0.2"] == curve["0.3"]


def test_constant_phi_selects_zero():
    rng = np.random.default_rng(0)
    q = np.repeat(["a", "b"], 20)
    g = rng.integers(0, 3, 40)
    t = rng.normal(size=40)
    w, curve = fc.select_weight(t, np.zeros(40), g, q)
    assert w == 0.0 and len(set(curve.values())) == 1


def test_within_patient_z_sd0_is_zero():
    v = np.array([1.0, 1.0, 1.0, 2.0, 4.0])
    q = np.array(["a", "a", "a", "b", "b"])
    z = fc.within_patient_z(v, q)
    assert np.array_equal(z[:3], np.zeros(3))
    assert np.allclose(z[3:], [-1.0, 1.0])


# ----------------------------------------------------------------- derangement
def _groups(components):
    return fc.group_lookup(components)


def test_group_derangement_rules():
    topics = [f"2021_{i}" for i in range(1, 11)]
    comps = [["2021_1", "2021_2"], ["2021_3", "2021_4", "2021_5"]]
    g = _groups(comps)
    mapping, draws = fc.group_derangement(topics, g, seed=11)
    assert set(mapping) == set(topics) and draws >= 1
    for recv, src in mapping.items():
        assert recv != src
        assert g.get(recv, ("s", recv)) != g.get(src, ("s", src))
    assert fc.group_derangement(topics, g, seed=11) == (mapping, draws)


def test_derangement_singletons_equal_cyclic_shift():
    topics = [f"2022_{i}" for i in (3, 1, 2, 10, 7)]
    mapping, draws = fc.group_derangement(topics, {}, seed=11)
    assert draws == 1
    ts = fc.sort_topics(topics)
    order = np.random.default_rng(11).permutation(len(ts))
    assert mapping == {ts[order[i]]: ts[order[(i + 1) % len(ts)]] for i in range(len(ts))}


def test_derangement_aborts_when_infeasible():
    topics = ["2021_1", "2021_2", "2021_3", "2021_4"]
    g = _groups([["2021_1", "2021_2", "2021_3"]])
    with pytest.raises(fc.ProtocolError, match="10000"):
        fc.group_derangement(topics, g, seed=11, max_draws=10000)


def test_shuffled_topic_uris_uses_mapping():
    q = ["a", "b", "a"]
    uris = [["x"], ["y"], ["x"]]
    assert fc.shuffled_topic_uris(q, uris, {"a": "b", "b": "a"}) == [["y"], ["x"], ["y"]]


# ----------------------------------------------------------------- loader and digest
class _Keyer:
    def get_content_key(self, content):
        return hashlib.sha256(content["encoder_view"].encode()).hexdigest()[:16]


def _toy_split(tmp_path, n=4):
    rows = []
    for i in range(n):
        rows.append({"resume": {"encoder_view": "topic A", "skill_uris": ["D1"]},
                     "job": {"encoder_view": f"trial {i}", "title": "t", "grade": i % 3, "skill_uris": ["D2"]},
                     "metadata": {"resume_id": "2022_1", "nct_id": f"NCT{i}", "condition_signal_present": True}})
    p = tmp_path / "test.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p, rows


def _ctt(content, kind):
    return content["encoder_view"]


def test_loader_asserts_embedding_dim(tmp_path, restore_root):
    fc.ROOT = tmp_path
    p, rows = _toy_split(tmp_path)
    k = _Keyer()
    cache = {k.get_content_key(r["resume"]): np.ones(768) for r in rows}
    cache.update({k.get_content_key(r["job"]): np.ones(128) for r in rows})
    with pytest.raises(fc.ProtocolError, match="768"):
        fc.load_split(p, cache, published_chunk_dir=tmp_path / "none", keyer=k, content_to_text=_ctt, log=lambda s: None)
    cache.update({k.get_content_key(r["job"]): np.ones(768) for r in rows})
    prep = fc.load_split(p, cache, published_chunk_dir=tmp_path / "none", keyer=k, content_to_text=_ctt, log=lambda s: None)
    assert prep["embedding_dim"] == 768 and prep["base"].shape == (5, 768)
    assert list(prep["grades"]) == [0, 1, 2, 0] and prep["embedding_sources"]["frozen_cache"] == 5


def test_loader_2022_missing_text_aborts_without_encoder(tmp_path, restore_root):
    fc.ROOT = tmp_path
    p, rows = _toy_split(tmp_path)
    k = _Keyer()
    cache = {k.get_content_key(r["resume"]): np.ones(768) for r in rows}

    def boom():
        raise AssertionError("no encoder may be constructed for 2022 files")
    with pytest.raises(fc.ProtocolError, match="never re-encoded"):
        fc.load_split(p, cache, published_chunk_dir=tmp_path / "none", encoder_factory=boom, keyer=k,
                      content_to_text=_ctt, log=lambda s: None)


def test_loader_resolves_published_chunks_by_text_hash(tmp_path, restore_root):
    fc.ROOT = tmp_path
    p, rows = _toy_split(tmp_path)
    k = _Keyer()
    cache = {k.get_content_key(rows[0]["resume"]): np.ones(768)}
    d = tmp_path / "chunks"
    d.mkdir()
    keys = np.asarray([fc.text_sha256(f"trial {i}") for i in range(4)])
    np.savez(d / "chunk_0.npz", keys=keys, embeddings=np.full((4, 768), 2.0, np.float32))
    prep = fc.load_split(p, cache, published_chunk_dir=d, keyer=k, content_to_text=_ctt, log=lambda s: None)
    assert prep["embedding_sources"] == {"frozen_cache": 1, "chunks": 4, "fresh_encoded": 0}


def test_loader_2021_fresh_encode_writes_epf_chunks_and_resumes(tmp_path, restore_root):
    fc.ROOT = tmp_path
    p, rows = _toy_split(tmp_path)
    k = _Keyer()
    cache = {k.get_content_key(rows[0]["resume"]): np.ones(768)}

    class Enc:
        calls = 0

        def encode(self, texts, batch_size, convert_to_numpy, show_progress_bar):
            Enc.calls += 1
            assert batch_size == 64 and convert_to_numpy
            return np.full((len(texts), 768), 3.0, np.float32)

    prep = fc.load_split(p, cache, fresh_encode_root=tmp_path / "fresh", encoder_factory=Enc, keyer=k,
                         content_to_text=_ctt, log=lambda s: None)
    assert prep["embedding_sources"] == {"frozen_cache": 1, "chunks": 0, "fresh_encoded": 4} and Enc.calls == 1
    d = tmp_path / "fresh" / fc.prepared_dir_name(p)
    chunks = sorted(d.glob("chunk_*.npz"))
    assert len(chunks) == 1
    with np.load(chunks[0]) as z:
        assert sorted(z["keys"].tolist()) == sorted(fc.text_sha256(f"trial {i}") for i in range(4))
    again = fc.load_split(p, cache, fresh_encode_root=tmp_path / "fresh", encoder_factory=None, keyer=k,
                          content_to_text=_ctt, log=lambda s: None)
    assert again["embedding_sources"] == {"frozen_cache": 1, "chunks": 4, "fresh_encoded": 0}
    assert np.array_equal(again["base"], prep["base"])


def test_manifest_digest_basenames_cwd_independent(tmp_path, monkeypatch):
    d = tmp_path / "chunks"
    d.mkdir()
    for name, data in (("chunk_b.npz", b"bbb"), ("chunk_a.npz", b"a")):
        (d / name).write_bytes(data)
    expected = hashlib.sha256("".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in (("chunk_a.npz", b"a"), ("chunk_b.npz", b"bbb"))).encode()).hexdigest()
    monkeypatch.chdir(tmp_path)
    one = fc.manifest_digest(d)
    monkeypatch.chdir(d)
    assert fc.manifest_digest(d) == one == expected
    assert fc.manifest_digest_from_hashes({"/x/y/chunk_a.npz": hashlib.sha256(b"a").hexdigest(),
                                           "chunk_b.npz": hashlib.sha256(b"bbb").hexdigest()}) == expected


# ----------------------------------------------------------------- paraphrase, closure, folds
def test_paraphrase_rule_on_toy_matrix():
    base = np.eye(4)
    vecs = {"2021_1": base[0], "2021_2": base[0] * 0.9 + base[1] * 0.1, "2021_3": base[2],
            "2021_4": base[2] * 0.5 + base[3] * 0.5}
    res = fc.cosine_edges(vecs, 0.80)
    assert [e[:2] for e in res["edges"]] == [["2021_1", "2021_2"]]
    assert [e[:2] for e in res["audit"]] == [["2021_3", "2021_4"]]   # cos 0.707
    comps = fc.connected_components(fc.sort_topics(vecs), [e[:2] for e in res["edges"]] + [["2021_3", "2021_1"]])
    assert comps == [["2021_1", "2021_2", "2021_3"], ["2021_4"]]
    near = fc.cosine_edges({"2021_1": np.array([1.0, 0.0]), "2021_2": np.array([0.8, 0.6])}, 0.80)
    assert near["near"]


def test_exposure_closure_is_union_of_touched_components():
    comps = [["2021_1", "2021_2"], ["2021_3", "2021_4", "2021_5"], ["2021_6"], ["2021_7", "2021_8"]]
    assert fc.exposure_closure(["2021_2", "2021_5", "2021_6"], comps) == \
        ["2021_1", "2021_2", "2021_3", "2021_4", "2021_5", "2021_6"]


def _toy_corpus(n_topics=12, missing=("NCT_MISSING_0",)):
    topics = {f"2021_{i}": {"topic_id": f"2021_{i}", "year": "2021", "encoder_view": f"topic {i}",
                            "mesh_uris": ["D1"], "condition_uris": ["C1"], "condition_signal_present": True}
              for i in range(1, n_topics + 1)}
    trials, graded = {}, {}
    for t in topics:
        graded[t] = {2: [], 1: [], 0: []}
        n0 = 70 if t == "2021_1" else 5
        for grade, n in ((2, 3), (1, 2), (0, n0)):
            for j in range(n):
                nct = f"NCT_{t}_{grade}_{j:03d}"
                graded[t][grade].append(nct)
                trials[nct] = {"nct_id": nct, "encoder_view": f"trial {nct}", "title": nct,
                               "mesh_uris": ["D2"], "condition_uris": ["C2"]}
    for m in missing:
        graded["2021_1"][0].append(m)
    graded["2021_2"][0].append("NCT_MISSING_1")
    return topics, trials, graded


def _build_all(tmp_path, tag):
    topics, trials, graded = _toy_corpus()
    comps = fc.connected_components(fc.sort_topics(topics), [("2021_1", "2021_2"), ("2021_3", "2021_4"),
                                                              ("2021_4", "2021_5")])
    folds = fc.assign_folds(comps, n_folds=3, seed=2021)
    mans = [fc.build_fold_files(topics, trials, graded, folds, k, tmp_path / tag / f"fold_{k}",
                                components=[c for c in comps if len(c) > 1]) for k in range(3)]
    return folds, comps, mans


def test_fold_builder_groups_disjoint_and_deterministic(tmp_path):
    folds, comps, mans = _build_all(tmp_path, "a")
    _, _, mans2 = _build_all(tmp_path, "b")
    assert [m["sha256"] for m in mans] == [m["sha256"] for m in mans2]
    fold_of = {t: i for i, f in enumerate(folds) for t in f}
    for c in comps:
        assert len({fold_of[t] for t in c}) == 1
    for m in mans:
        s = {k: set(v) for k, v in m["topics"].items()}
        assert not (s["train"] & s["validation"] or s["train"] & s["test"] or s["validation"] & s["test"])
    with pytest.raises(fc.ProtocolError, match="split across folds"):
        topics, trials, graded = _toy_corpus()
        bad = [["2021_1"] + [t for t in fc.sort_topics(topics) if t not in ("2021_1", "2021_2")][:3],
               ["2021_2"] + [t for t in fc.sort_topics(topics) if t not in ("2021_1", "2021_2")][3:6],
               [t for t in fc.sort_topics(topics) if t not in ("2021_1", "2021_2")][6:]]
        fc.build_fold_files(topics, trials, graded, bad, 0, tmp_path / "bad", components=[["2021_1", "2021_2"]])


def test_grade0_cap_and_missing_after_sampling(tmp_path):
    topics, trials, graded = _toy_corpus()
    folds = [["2021_3", "2021_4", "2021_5", "2021_6"], ["2021_1", "2021_2", "2021_7", "2021_8"],
             ["2021_9", "2021_10", "2021_11", "2021_12"]]
    man = fc.build_fold_files(topics, trials, graded, folds, 0, tmp_path / "f0")   # validation = fold 1
    rows = fc.read_jsonl(tmp_path / "f0" / "validation.jsonl")
    g0 = sorted(r["metadata"]["nct_id"] for r in rows if r["metadata"]["resume_id"] == "2021_1" and r["job"]["grade"] == 0)
    ids = sorted(graded["2021_1"][0])
    sample = sorted(random.Random("2021|validation|2021_1|0").sample(ids, 60))
    assert g0 == [x for x in sample if x in trials]                                   # >60: 60 sampled, then skip
    small = [r for r in rows if r["metadata"]["resume_id"] == "2021_2" and r["job"]["grade"] == 0]
    assert len(small) == 5                                                            # <=60 keeps all (minus missing)
    assert man["skipped_missing_trials"]["validation"] == (1 if "NCT_MISSING_0" in sample else 0) + 1
    vp = fc.read_jsonl(tmp_path / "f0" / "validation_positive.jsonl")
    assert vp == [r for r in rows if r["job"]["grade"] == 2]


# ----------------------------------------------------------------- bootstrap and decisions
def test_bootstrap_shapes_and_determinism():
    D = np.random.default_rng(1).normal(0.01, 0.02, size=(5, 30))
    D[:, 3] = np.nan
    C, P = fc.draw_indices(np.random.default_rng([31415, 2022]), 5, 30, 200)
    C2, P2 = fc.draw_indices(np.random.default_rng([31415, 2022]), 5, 30, 200)
    assert C.shape == (200, 5) and P.shape == (200, 30)
    assert np.array_equal(C, C2) and np.array_equal(P, P2)
    s = fc.summarize(D, C, P)
    assert s["n_defined"] == 29 and s["_T"].shape == (200,)
    assert np.isclose(s["point"], np.nanmean(D.mean(axis=0)))
    assert 0 < s["p"] <= 1 and s["ci95"][0] <= s["point"] <= s["ci95"][1]
    assert fc.boot_p(np.full(2000, 1.0)) == pytest.approx(2 / 2001)


def test_holm_tie_negative_and_steps():
    floor = 2 / 2001
    h = fc.holm(floor, floor, 0.01, 0.02)
    assert h["order"] == ["H1", "H2"] and h["H1"]["companion_level"] == 0.975 and h["H2"]["companion_level"] == 0.95
    assert h["H1"]["positive"] and h["H2"]["positive"]
    h = fc.holm(0.001, 0.04, -0.01, 0.01)
    assert h["H1"]["harm"] and not h["H1"]["positive"] and h["H2"]["positive"]
    h = fc.holm(0.03, 0.04, 0.01, 0.01)                    # first p > 0.025: nothing rejected
    assert not h["H1"]["rejected"] and not h["H2"]["rejected"] and h["H2"]["companion_level"] == 0.975
    h = fc.holm(0.2, 0.01, 0.01, 0.01, names=("H1z", "H2z"))
    assert h["order"] == ["H2z", "H1z"] and h["H2z"]["rejected"] and not h["H1z"]["rejected"]


def test_c2_states_ordered_precedence():
    assert fc.c2_state(0.001, 0.004, 0.005, 0.013, 0.002) == "fail"       # protocol worked example
    assert fc.c2_state(-0.004, 0.003, 0.004, 0.013, -0.001) == "pass"
    assert fc.c2_state(-0.004, 0.006, 0.004, 0.013, 0.001) == "inconclusive"
    assert fc.c2_state(-0.004, 0.003, -0.001, 0.001, 0.002) == "fail"     # diff LB <= 0 and hier <= C2
    assert fc.c2_state(-0.004, 0.003, -0.001, 0.003, 0.002) == "inconclusive"


def test_s4_equivalence_states():
    assert fc.equivalence_state(-0.005, 0.008) == "equivalent"
    assert fc.equivalence_state(0.01, 0.03) == "not equivalent"
    assert fc.equivalence_state(-0.04, -0.01) == "not equivalent"
    assert fc.equivalence_state(-0.012, 0.004) == "inconclusive"


def test_final_label_rule_and_headline():
    L = lambda **k: fc.final_label(**{**dict(positive_2022=True, harm_2022=False, positive_2021=True,  # noqa: E731
                                            harm_2021=False, c2_2022="pass", c2_2021="pass"), **k})
    a = {h: L(c2_2022="fail") for h in ("H1", "H2")}                                   # (a)
    assert all(v["label_id"] == 2 for v in a.values()) and "replicated" not in fc.headline(a)
    assert fc.headline(a) == ("H1: not attributable to patient-specific MeSH information (2022); "
                              "H2: not attributable to patient-specific MeSH information (2022)")
    b = {h: L() for h in ("H1", "H2")}                                                  # (b)
    assert all(v["label_id"] == 5 for v in b.values()) and fc.headline(b).startswith(
        "MeSH fusion improves within-patient ranking, replicated")
    c = {h: L(c2_2021="fail") for h in ("H1", "H2")}                                   # (c)
    assert all(v["label_id"] == 4 for v in c.values()) and "replicated" not in fc.headline(c).replace(
        "replication", "")
    d = {"H1": L(), "H2": L(positive_2021=False)}                                       # (d)
    assert fc.headline(d).startswith("MeSH fusion improves within-patient ranking, partially replicated")
    assert d["H2"]["label"] == "confirmed, not replicated"
    e = L(positive_2022=False, c2_2022="inconclusive")                                  # (e)
    assert e == {"label_id": 1, "label": "not confirmed"}
    assert L(positive_2022=False, harm_2022=True)["label"] == "significant harm"
    assert L(positive_2021=False, harm_2021=True)["label"] == "confirmed, not replicated; significant harm in 2021"
    assert L(c2_2022="inconclusive", c2_2021="inconclusive")["label"] == \
        "replicated (shuffled control inconclusive: 2022) (shuffled control inconclusive: 2021)"
    assert L(positive_2021=False, c2_2021="inconclusive")["label"] == "confirmed, not replicated"


# ----------------------------------------------------------------- gate 2 classifier / comparator
@pytest.fixture(scope="module")
def published_run():
    return json.loads(PUBLISHED.read_text())["runs"][0]


def test_leaf_classifier_on_real_key_tuples(published_run):
    cases = {("test", "fusion", "contrasts", "global_auc", "pooled"): "rank",
             ("test", "delta", "ranking_macro", "ndcg@10"): "rank",
             ("test", "fusion", "ranking", "n_queries"): "exact",
             ("test", "text", "ordinal", "tier_stats", "2", "n"): "exact",
             ("test", "text", "ordinal", "tier_stats", "2", "median"): "continuous",
             ("validation_selection_curve", "0.1"): "rank",
             ("validation_stats", "text", "mean"): "continuous",
             ("selected_weight",): "exact",
             ("test", "fusion", "contrasts", "queries_with_both_classes", "pooled"): "exact",
             ("test", "text", "ranking", "per_query", "2022_1", "average_precision"): "rank_per_query",
             ("test", "text", "ordinal", "kendall_tau_b"): "rank",
             ("test", "text", "ordinal", "cohens_d", "hard"): "continuous",
             ("test", "text", "binary_classification", "confusion_matrix", "tn"): "skipped"}
    leaves = dict(fc.iter_leaves(published_run))
    for P, cls in cases.items():
        assert P in leaves, P
        assert fc.leaf_class(P) == cls, P
    for P in leaves:   # no rank leaf may fall into the 1e-6 continuous class (round-6 stall)
        if any(k in P for k in ("ranking", "global_auc", "macro_query_auc", "per_query", "ranking_macro",
                                "validation_selection_curve")) and fc.leaf_class(P) != "skipped":
            assert fc.leaf_class(P) in ("rank", "rank_per_query", "exact"), P


def _perturb(run, path, delta):
    r = copy.deepcopy(run)
    node = r
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = node[path[-1]] + delta
    return r


def test_gate2_comparator_tolerances(published_run):
    run = published_run
    assert fc.compare_gate2_run(run, copy.deepcopy(run))["passed"]
    for path in (("validation_selection_curve", "0.1"), ("test", "fusion", "ranking", "macro", "ndcg@10")):
        assert fc.compare_gate2_run(run, _perturb(run, path, 5e-5))["passed"], path
        assert not fc.compare_gate2_run(run, _perturb(run, path, 2e-4))["passed"], path
    for p in (("validation_stats", "text", "mean"), ("test", "text", "ordinal", "tier_stats", "2", "median")):
        assert fc.compare_gate2_run(run, _perturb(run, p, 5e-6))["passed"], p      # §9 A2: 1e-5
        assert not fc.compare_gate2_run(run, _perturb(run, p, 2e-5))["passed"], p
    pq = run["test"]["text"]["ranking"]["per_query"]
    def per_query_delta(d):
        r = copy.deepcopy(run)
        r["test"]["text"]["ranking"]["per_query"][0]["average_precision"] = pq[0]["average_precision"] + d
        return r
    assert fc.compare_gate2_run(run, per_query_delta(5e-4))["passed"]          # §9 A2: per-patient 1e-3
    assert not fc.compare_gate2_run(run, per_query_delta(2e-3))["passed"]
    bad = copy.deepcopy(run)
    bad["selected_weight"] = run["selected_weight"] + 0.1
    assert not fc.compare_gate2_run(run, bad)["passed"]
    n = _perturb(run, ("test", "fusion", "ranking", "n_queries"), 1)
    assert not fc.compare_gate2_run(run, n)["passed"]


def test_gate2_coverage_and_per_query_pairing(published_run):
    run = published_run
    drop = copy.deepcopy(run)
    del drop["test"]["fusion"]["ordinal"]["cohens_d"]["hard"]
    res = fc.compare_gate2_run(run, drop)
    assert not res["passed"] and ["test", "fusion", "ordinal", "cohens_d", "hard"] in res["missing"]
    skip = copy.deepcopy(run)
    del skip["test"]["fusion"]["binary_classification"]
    assert fc.compare_gate2_run(run, skip)["passed"]
    rev = copy.deepcopy(run)
    rev["test"]["text"]["ranking"]["per_query"] = list(reversed(rev["test"]["text"]["ranking"]["per_query"]))
    assert fc.compare_gate2_run(run, rev)["passed"]                       # paired by "query", not index
    one = copy.deepcopy(run)
    one["test"]["text"]["ranking"]["per_query"] = one["test"]["text"]["ranking"]["per_query"][1:]
    assert not fc.compare_gate2_run(run, one)["passed"]                  # query present on one side only


# ----------------------------------------------------------------- gate 3 device check
def test_gate3_device_check():
    ok = {"requested": "cuda", "resolved": "cuda", "cuda_device_name": "NVIDIA A100-SXM4-80GB"}
    assert fc.check_device_block(ok, "NVIDIA A100-SXM4-80GB") == []
    assert fc.check_device_block(dict(ok, cuda_device_name="NVIDIA L40S"), "NVIDIA A100-SXM4-80GB")
    assert fc.check_device_block(dict(ok, requested="auto"), "NVIDIA A100-SXM4-80GB")
    assert fc.check_device_block(dict(ok, resolved="cpu"), "NVIDIA A100-SXM4-80GB")
    assert fc.check_device_block(None, "x")


def test_gate3_failures_on_toy_output(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "best_checkpoint.pt").write_bytes(b"x")
    m = {"single_factor_audit": {"unexpected_fields": ["embedding_cache_path"]},
         "attach": {"cache_check": {"dims": {"(768,)": 5}, "expected_text_dim": 768}},
         "device": {"requested": "cuda", "resolved": "cuda", "cuda_device_name": "G"}}
    (out / "lc_manifest.json").write_text(json.dumps(m))
    assert fc.gate3_failures(out, 0, ["embedding_cache_path"], "G", True) == []
    assert fc.gate3_failures(out, 0, ["embedding_cache_path", "trials_split_dir"], "G", True)
    assert fc.gate3_failures(out, 1, ["embedding_cache_path"], "G", True)
    assert fc.gate3_failures(out, 0, ["embedding_cache_path"], "G", False)
    (out / "run_error.json").write_text("{}")
    assert fc.gate3_failures(out, 0, ["embedding_cache_path"], "G", True)


# ----------------------------------------------------------------- protocol parser
def test_protocol_parser_rows_and_truncation():
    text = PROTOCOL.read_text(encoding="utf-8")
    parsed = fc.parse_protocol(text)
    assert len(parsed["checkpoints"]) == 15 and len(parsed["inputs"]) == 21
    assert parsed["manifest_digest"].startswith("3d015e60") and parsed["n_test_chunks"] == 36
    assert parsed["mpnet_revision"] == "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    ck_lines = {ln for ln in text.splitlines() if fc.CKPT_ROW.match(ln)}
    in_lines = {ln for ln in text.splitlines() if fc.INPUT_ROW.match(ln)}
    assert ck_lines and in_lines and not (ck_lines & in_lines)
    assert "preprocess/trec_ct_lc/frac_100/train.jsonl" in parsed["inputs"]      # annotation ignored
    lines = text.splitlines()
    cut = [ln for ln in lines if not (fc.CKPT_ROW.match(ln) and "| 100 | 21 |" in ln)]
    with pytest.raises(fc.ProtocolError, match="checkpoint rows"):
        fc.parse_protocol("\n".join(cut))
    cut = [ln for ln in lines if not (fc.INPUT_ROW.match(ln) and "trials.jsonl" in ln)]
    with pytest.raises(fc.ProtocolError, match="input rows"):
        fc.parse_protocol("\n".join(cut))


# ----------------------------------------------------------------- pins chain
def _pins(tmp_path, **over):
    base = {"schema": "s", "revision": 1, "supersedes": None, "amendment": None, "created_utc": "t0",
            "protocol": {"path": "p.md", "sha256": "a"}, "git": {"head": "h1"}, "environment": {"host": "x"},
            "inputs": {"f": "1"}, "code": {"c.py": "1"}, "configs": {"base": {}, "derived": {}},
            "analysis": {"B": 2000, "gate2_tolerance_abs": {"rank": 1e-4}}, "cohort2022": {"new_runs": []}}
    base.update(over)
    return base


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return fc.file_sha256(path)


def _revision(prev, prev_path, prev_sha, changed=(), seen=False, **over):
    new = copy.deepcopy(prev)
    new.update(over)
    new.update({"revision": prev["revision"] + 1, "created_utc": "t1",
                "supersedes": {"path": prev_path, "sha256": prev_sha},
                "amendment": {"id": "A1", "confirmatory_output_seen": seen, "changed_fields": list(changed),
                              "previous": {"git": prev["git"], "environment": prev["environment"]}}})
    return new


def test_pins_chain_whitelist_and_declared(tmp_path, restore_root):
    fc.ROOT = tmp_path
    p1 = _pins(tmp_path)
    sha1 = _write(tmp_path / "pins.json", p1)
    r2 = _revision(p1, "pins.json", sha1, code={"c.py": "2"}, protocol={"path": "p.md", "sha256": "b"},
                   git={"head": "h2"}, environment={"host": "y"})
    fc.verify_revision(p1, r2, "pins.json", sha1)                                     # whitelisted only
    _write(tmp_path / "pins_r2.json", r2)
    assert fc.load_verified_pins(tmp_path / "pins_r2.json")["revision"] == 2
    with pytest.raises(fc.ProtocolError, match="latest"):
        fc.load_verified_pins(tmp_path / "pins.json")                                 # non-latest
    bad_prev = copy.deepcopy(r2)
    bad_prev["amendment"]["previous"]["git"] = {"head": "zz"}
    with pytest.raises(fc.ProtocolError, match="previous"):
        fc.verify_revision(p1, bad_prev, "pins.json", sha1)
    tol = {"B": 2000, "gate2_tolerance_abs": {"rank": 2e-4}}
    ok = _revision(p1, "pins.json", sha1, changed=["analysis.gate2_tolerance_abs"], analysis=tol)
    fc.verify_revision(p1, ok, "pins.json", sha1)                                     # declared branch
    with pytest.raises(fc.ProtocolError, match="declared"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, analysis=tol), "pins.json", sha1)   # undeclared
    with pytest.raises(fc.ProtocolError, match="declared"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, changed=["analysis.B"], analysis=tol),
                           "pins.json", sha1)                                          # inexact set
    with pytest.raises(fc.ProtocolError, match="declared"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, changed=["analysis.B"]), "pins.json", sha1)
    with pytest.raises(fc.ProtocolError, match="confirmatory_output_seen"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, changed=["analysis.gate2_tolerance_abs"],
                                         seen=True, analysis=tol), "pins.json", sha1)
    with pytest.raises(fc.ProtocolError, match="supersedes"):
        fc.verify_revision(p1, r2, "pins.json", "0" * 64)                              # broken sha
    with pytest.raises(fc.ProtocolError, match="configs"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, configs={"base": {"x": 1}}), "pins.json", sha1)


def test_pins_chain_dev_root_only_for_non_confirmatory_head(tmp_path, restore_root):
    """Review finding 7: a dev_pins.json root is accepted only when the head is non-confirmatory."""
    fc.ROOT = tmp_path
    dev = _pins(tmp_path, confirmatory=False)
    sha = _write(tmp_path / "dev_pins.json", dev)
    assert fc.load_verified_pins(tmp_path / "dev_pins.json")["revision"] == 1
    r2 = _revision(dev, "dev_pins.json", sha, code={"c.py": "2"})
    _write(tmp_path / "pins_r2.json", r2)
    assert fc.load_verified_pins(tmp_path / "pins_r2.json")["revision"] == 2        # dev chain ok
    r2c = _revision(dev, "dev_pins.json", sha, changed=["confirmatory"], confirmatory=True)
    _write(tmp_path / "pins_r2.json", r2c)
    with pytest.raises(fc.ProtocolError, match="revision 1 pins.json"):
        fc.load_verified_pins(tmp_path / "pins_r2.json")
    (tmp_path / "pins_r2.json").unlink()
    _write(tmp_path / "dev_pins.json", _pins(tmp_path, confirmatory=True))
    with pytest.raises(fc.ProtocolError, match="revision 1 pins.json"):
        fc.load_verified_pins(tmp_path / "dev_pins.json")


def test_compare_code_hashes_missing_extra_differing():
    """Review finding 4: the transferred code tree must equal the Mac manifest exactly."""
    mac = {"a.py": "1", "b.py": "2", "c.py": "3"}
    assert fc.compare_code_hashes(mac, dict(mac)) == []
    diffs = fc.compare_code_hashes(mac, {"a.py": "1", "b.py": "X", "d.py": "4"})
    assert {(d["path"], d["reason"]) for d in diffs} == {
        ("b.py", "sha256 differs"), ("c.py", "missing on remote"), ("d.py", "not in the Mac manifest")}


def test_dev_local_restrict_covers_validation_and_test():
    """Review findings 1/3: the dev-local 2021 smoke restricts BOTH files to exposed_closed
    patients and aborts before scoring when a restricted file cannot be deranged."""
    sc = load_script("score")

    def prep(topics):
        q = [t for t in topics for _ in range(3)]
        n = len(q)
        return {"queries": np.asarray(q), "left_index": np.arange(n), "right_index": np.arange(n),
                "grades": np.zeros(n, dtype=int), "nct_ids": np.asarray([f"N{i}" for i in range(n)]),
                "condition_signal": np.ones(n, dtype=bool), "topic_uris": [["u"]] * n, "trial_uris": [["v"]] * n}

    closed = ["2021_1", "2021_2", "2021_3", "2021_5", "2021_6", "2021_7", "2021_9"]
    g = _groups([["2021_1", "2021_2"]])
    val = prep(["2021_1", "2021_2", "2021_3", "2021_4", "2021_9"])     # 2021_4 unexposed
    test = prep(["2021_5", "2021_6", "2021_7", "2021_8"])    # 2021_8 unexposed
    v, t = sc.dev_local_restrict(val, test, closed, g, 11, 10000)
    assert set(v["queries"]) == {"2021_1", "2021_2", "2021_3", "2021_9"} and len(v["topic_uris"]) == 12
    assert set(t["queries"]) == {"2021_5", "2021_6", "2021_7"}
    with pytest.raises(fc.ProtocolError, match="cannot be deranged"):    # only one paraphrase pair left
        sc.dev_local_restrict(prep(["2021_1", "2021_2", "2021_4"]), test, closed, g, 11, 10000)
    with pytest.raises(fc.ProtocolError, match="no exposed_topics_closed"):
        sc.dev_local_restrict(val, prep(["2021_8"]), closed, g, 11, 10000)


def test_dev_smoke_exposed_topics_are_closed_unexposed():
    """§9 A0: the 13 fold-1 topics are disjoint from the protocol's 23-topic exposed_closed."""
    closed = {"2021_6", "2021_15", "2021_16", "2021_17", "2021_21", "2021_23", "2021_25", "2021_26", "2021_27",
              "2021_33", "2021_37", "2021_40", "2021_42", "2021_45", "2021_49", "2021_51", "2021_57", "2021_59",
              "2021_61", "2021_63", "2021_64", "2021_69", "2021_74"}
    assert len(fc.DEV_SMOKE_EXPOSED_2021) == 13 and not (set(fc.DEV_SMOKE_EXPOSED_2021) & closed)
    assert 75 - len(closed) - len(fc.DEV_SMOKE_EXPOSED_2021) == 39


def _toy_root(tmp_path):
    """A minimal repo root on which make_pins.py can run (synthetic hashes)."""
    root = tmp_path / "root"
    files = {}

    def put(rel, data: bytes):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[rel] = hashlib.sha256(data).hexdigest()

    put(fc.BASE_CONFIG_REL, json.dumps({"embedding_cache_path": fc.ORIGINAL_CACHE_REL,
                                        "trials_split_dir": "preprocess/trec_ct_splits"}).encode())
    put(fc.ORIGINAL_CACHE_REL, b"cache")
    for i in range(19):   # 19 toy + config + cache = 21 input rows (parser minimum, §9 A1)
        put(f"preprocess/toy/input_{i}.jsonl", f"{i}".encode())
    ck_rows = []
    for F in (10, 25, 50, 75, 100):
        for S in (42, 13, 21):
            rel = fc.PUBLISHED_CKPT_PATTERN.format(F=F, S=S)
            put(rel, f"ck{F}{S}".encode())
            ck_rows.append(f"| {F} | {S} | 1 | {files[rel]} |")
            del files[rel]
    tdir = "results/pub/prepared_embeddings/trials/test_0123456789abcdef_m"
    put(f"{tdir}/chunk_a.npz", b"chunk")
    del files[f"{tdir}/chunk_a.npz"]
    (root / "results/pub/prepared_embeddings/trials/validation_x").mkdir(parents=True)
    digest = fc.manifest_digest_from_hashes({"chunk_a.npz": hashlib.sha256(b"chunk").hexdigest()})
    proto = "\n".join(
        ["# toy protocol", "The `test_0123456789abcdef_…` directory has 1 chunks.",
         f"revision `{'e' * 40}`", f"§2.1 manifest digest `{digest}`",
         f'"test_dir": "{tdir}",', '"validation_dir": "results/pub/prepared_embeddings/trials/validation_x",',
         "| F | S | train records | sha |", "|---|---|---|---|"] + ck_rows +
        ["", "| file | sha256 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in files.items()]) + "\n"
    put(fc.PROTOCOL_REL, proto.encode())
    cv = root / fc.CV2021_REL
    cv.mkdir(parents=True)
    (cv / "cv2021_manifest.json").write_text(json.dumps({"fold_files": {}, "n_folds": 1, "seeds": [13],
                                                         "components": [], "fold_dir": fc.CV2021_REL}))
    res = root / fc.RESULTS_REL
    (res / "cache").mkdir(parents=True)
    (res / "cache" / "trials_text_embeddings.pt").write_bytes(b"cache")
    put("trials_domain/fusion_confirm.py", b"code")          # toy code tree + Mac manifest
    del files["trials_domain/fusion_confirm.py"]
    manifest = json.dumps({"code": {"trials_domain/fusion_confirm.py": hashlib.sha256(b"code").hexdigest()}}).encode()
    put("code_hashes.mac.json", manifest)
    del files["code_hashes.mac.json"]
    (res / "transfer_check.json").write_text(json.dumps({
        "passed": True, "parsed_protocol_sha256": hashlib.sha256(proto.encode()).hexdigest(),
        "code": {"ok": True, "manifest": "code_hashes.mac.json",
                 "manifest_sha256": hashlib.sha256(manifest).hexdigest()}}))
    return root


def test_make_pins_revise_writes_no_config_and_checks_hashes(tmp_path, monkeypatch, restore_root):
    mp = load_script("make_pins")
    root = _toy_root(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(fc, "resolve_mpnet_snapshot", lambda expected: (expected, "/snap"))
    monkeypatch.setattr(fc, "environment_record", lambda *a, **k: {"gpu_name": "GPU-X", "host": "toy"})
    monkeypatch.setattr(fc, "git_record", lambda: {"head": "h"})
    args = ["--root", str(root), "--protocol", fc.PROTOCOL_REL]
    assert mp.main(args + ["--out", f"{fc.RESULTS_REL}/pins.json"]) == 0
    cfg_dir = root / fc.RESULTS_REL / "configs"
    before = {p.name: fc.file_sha256(p) for p in cfg_dir.iterdir()}
    assert len(before) == 2
    rev = ["--revise", "--previous", f"{fc.RESULTS_REL}/pins.json", "--amendment", "A1",
           "--changed-fields", "", "--confirmatory-output-seen", "false"]
    assert mp.main(args + rev + ["--out", f"{fc.RESULTS_REL}/pins_r2.json"]) == 0
    assert {p.name: fc.file_sha256(p) for p in cfg_dir.iterdir()} == before          # no config written
    r2 = json.loads((root / fc.RESULTS_REL / "pins_r2.json").read_text())
    assert r2["revision"] == 2 and r2["configs"] == json.loads((root / fc.RESULTS_REL / "pins.json").read_text())["configs"]
    (cfg_dir / "lc_trials_baseline_cache.json").write_text("{}")
    rev[2] = f"{fc.RESULTS_REL}/pins_r2.json"
    with pytest.raises(fc.ProtocolError, match="config hashes differ"):
        mp.main(args + rev + ["--out", f"{fc.RESULTS_REL}/pins_r3.json"])
    assert not (root / fc.RESULTS_REL / "pins_r3.json").exists()


# ----------------------------------------------------------------- ledger / resume
def test_ledger_resume_semantics(tmp_path, restore_root):
    fc.ROOT = tmp_path
    led = fc.Ledger(tmp_path / "run_ledger.json")
    (tmp_path / "run_ledger.json").write_text(fc.Ledger._render([]))
    led.append({"run": "A", "status": "passed", "order_index": 0})
    calls = []

    def attempt_factory(results):
        def attempt(run, n):
            calls.append((run["run"], n))
            out = fc.absp(run["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            return {"run": run["run"], "status": results.pop(0), "attempt": n}
        return attempt

    assert fc.run_with_resume(led, {"run": "A", "output_dir": "outA"}, attempt_factory([])) == "skipped-passed"
    assert calls == []
    before = (tmp_path / "run_ledger.json").read_text()
    (tmp_path / "outB").mkdir()                                     # interrupted attempt, no passing entry
    status = fc.run_with_resume(led, {"run": "B", "output_dir": "outB", "order_index": 1},
                                attempt_factory(["failed"]))
    assert status == "excluded" and calls == [("B", 2)]
    assert (tmp_path / "outB.failed1").is_dir() and (tmp_path / "outB.failed2").is_dir()
    after = (tmp_path / "run_ledger.json").read_text()
    assert after.startswith(before.rstrip("\n]").rstrip())          # earlier entries byte-identical
    sts = [e["status"] for e in led.entries() if e["run"] == "B"]
    assert sts == ["failed", "failed", "excluded"]
    assert fc.run_with_resume(led, {"run": "B", "output_dir": "outB"}, attempt_factory([])) == "skipped-excluded"
    assert fc.run_with_resume(led, {"run": "C", "output_dir": "outC"}, attempt_factory(["failed", "passed"])) == "passed"
    assert [e["status"] for e in led.entries() if e["run"] == "C"] == ["failed", "passed"]
    lines = (tmp_path / "run_ledger.json").read_text().splitlines()
    assert lines[1].rstrip(",") == json.dumps({"order_index": 0, "run": "A", "status": "passed"}, sort_keys=True)


# ----------------------------------------------------------------- PYTHONHASHSEED guard
@pytest.mark.parametrize("name", ["verify_transfer", "make_pins", "build_cv2021", "train", "score", "analyze",
                                  "dump_code_hashes"])
def test_hashseed_guard(name, monkeypatch):
    mod = load_script(name)
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with pytest.raises(fc.ProtocolError, match="PYTHONHASHSEED"):
        mod.main(["--help"])
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(fc.ProtocolError, match="PYTHONHASHSEED"):
        mod.main(["--help"])


def test_a1_cohort2022_population_and_plan():
    """§9 A1: all 19 cohort-2022 cells are retrained, in the fixed order, with the right
    train file and --fraction; they are exactly the §3b table cells plus (25|100, 87|123)."""
    runs = list(fc.COHORT2022_RUNS)
    assert len(runs) == 19 and len(set(runs)) == 19
    table = {(r["fraction"], r["seed"]) for r in fc.parse_protocol(PROTOCOL.read_text(encoding="utf-8"))["checkpoints"]}
    assert set(runs) == table | {(25, 87), (25, 123), (100, 87), (100, 123)}
    assert runs[:4] == [(10, 42), (10, 13), (10, 21), (25, 42)] and runs[-2:] == [(100, 87), (100, 123)]
    order = fc.ANALYSIS_PARAMS["training_order"]
    assert order[:19] == [f"2022:f{F}_s{S}" for F, S in runs] and len(order) == 34
    tr = load_script("train")
    cfg = "results/trials_fusion_confirm/configs/lc_trials_baseline_cache.json"
    pins = {"configs": {"derived": {cfg: {"expected_unexpected_fields": ["embedding_cache_path"]}}},
            "analysis": {"training_order": order},
            "cohort2022": {"new_runs": [{"fraction": F, "seed": S, "device": "cuda",
                                         "output_dir": f"o/f{F}_s{S}"} for F, S in runs]}}
    plan = tr.plan(pins, "2022", "cuda")
    assert [p["run"] for p in plan] == order[:19]
    for p in plan:
        cmd = p["command"]
        F = p["fraction"]
        assert cmd[cmd.index("--train-file") + 1] == f"preprocess/trec_ct_lc/frac_{F}/train.jsonl"
        assert float(cmd[cmd.index("--fraction") + 1]) == F / 100
        assert cmd[cmd.index("--device") + 1] == "cuda"
    inputs = fc.parse_protocol(PROTOCOL.read_text(encoding="utf-8"))["inputs"]
    assert all(f"preprocess/trec_ct_lc/frac_{F}/train.jsonl" in inputs for F in (10, 25, 50, 75, 100))


def test_verify_transfer_requires_code_manifest(tmp_path, monkeypatch):
    """Review pass 2, issue 4: step -1 fails without --code-manifest (argparse) and on a
    missing manifest file, instead of passing with a warning."""
    vt = load_script("verify_transfer")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    proto = fc.absp(fc.PROTOCOL_REL)
    with pytest.raises(SystemExit):
        vt.main(["--protocol", str(proto), "--out", str(tmp_path / "tc.json")])
    with pytest.raises(fc.ProtocolError, match="code-manifest"):
        vt.main(["--protocol", str(proto), "--out", str(tmp_path / "tc.json"),
                 "--code-manifest", str(tmp_path / "missing.json")])
    assert not (tmp_path / "tc.json").exists()
