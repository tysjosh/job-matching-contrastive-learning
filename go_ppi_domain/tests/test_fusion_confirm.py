"""Gate-1 unit tests for the GO/PPI fusion confirmation (protocol_go_ppi_fusion.md §5.1).

Synthetic fixtures only, except: the protocol-parser test reads the protocol file and the
gate-2 classifier / near-tie tests read the published fusion JSONs (both are part of the
transfer, §6.2). Nothing here trains, encodes or scores a real model.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from go_ppi_domain import fusion_confirm as fc
from go_ppi_domain.tests import _leak_fixtures as lf

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "go_ppi_fusion_confirm"
PROTOCOL = ROOT / fc.PROTOCOL_REL
PUBLISHED_NDCG = ROOT / fc.PUBLISHED_FUSION_REL["ndcg@10"]
SCRIPT_NAMES = ("dump_code_hashes", "verify_transfer", "build_inputs", "encode", "make_pins", "train", "score", "analyze")


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"gpfc_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def restore_root():
    old_root, old_cwd = fc.ROOT, os.getcwd()
    yield
    fc.ROOT = old_root
    os.chdir(old_cwd)


def _view(text, phis, queries, grades=None):
    q = np.asarray(queries, dtype=object)
    g = np.zeros(len(q), dtype=int) if grades is None else np.asarray(grades)
    return {"text": np.asarray(text, float), "phi": {k: np.asarray(v, float) for k, v in phis.items()},
            "grades": g, "queries": q, "ai": fc.AnchorIndex(q, g)}


# ----------------------------------------------------------------- fusion arithmetic and selection
def test_fusion_arithmetic_az_and_gz():
    q = ["a", "a", "a", "b", "b"]
    t, x = np.array([1.0, 2.0, 4.0, 0.5, 0.5]), np.array([0.0, 1.0, 0.0, 3.0, 1.0])
    v = _view(t, {"X": x, "P": x}, q)
    az = fc.apply_arm("AZ-X", {"w": 0.3}, v)
    assert np.allclose(az, 0.7 * fc.anchor_z(t, v["ai"]) + 0.3 * fc.anchor_z(x, v["ai"]))
    st = {"w": 0.4, "text_stats": [1.0, 2.0], "phi_stats": [[0.5, 0.25]]}
    gz = fc.apply_arm("GZ-P", st, v)
    assert np.allclose(gz, 0.6 * (t - 1.0) / 2.0 + 0.4 * (x - 0.5) / 0.25)
    assert np.allclose(fc.fuse2(t, x, x, 0.2, 0.3), 0.5 * t + 0.2 * x + 0.3 * x)
    assert np.array_equal(fc.fuse(t, x, 0.0), t)


def test_weight_tie_goes_to_smaller_w(monkeypatch):
    values = iter([0.5, 0.55, 0.6, 0.6, 0.58, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
    monkeypatch.setattr(fc, "selection_value", lambda s, g, q, rule: next(values))
    sel = fc.select_1d(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), "global_pooled_auc")
    assert sel["w"] == 0.2 and sel["curve"]["0.2"] == sel["curve"]["0.3"] and sel["top2_gap"] == 0.0


def test_c5_grid_and_tie_order(monkeypatch):
    assert len(fc.C5_GRID) == 66 and all(a + b <= 1 + 1e-12 for a, b in fc.C5_GRID)
    best = {(0.2, 0.3), (0.2, 0.5), (0.3, 0.0)}
    calls = iter(sorted(fc.C5_GRID))
    monkeypatch.setattr(fc, "selection_value", lambda s, g, q, rule: 1.0 if next(calls) in best else 0.5)
    sel = fc.select_2d(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    assert sel["w"] == [0.2, 0.3]                       # smaller w_g, then smaller w_b
    calls = iter(sorted(fc.C5_GRID))
    monkeypatch.setattr(fc, "selection_value", lambda s, g, q, rule: (next(calls), 0.7)[1])
    assert fc.select_2d(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))["w"] == [0.0, 0.0]


def test_within_anchor_z_rules_and_equality():
    from go_ppi_fusion.fusion import within_query_z
    from go_ppi_fusion.metrics import QueryIndex
    rng = np.random.default_rng(0)
    q = np.asarray(rng.choice(list("abcdefg"), 60), dtype=object)
    x = rng.normal(size=60)
    x[q == "c"] = 7.0                                   # constant anchor
    ai = fc.AnchorIndex(q, np.zeros(60, int))
    z = fc.anchor_z(x, ai)
    assert np.array_equal(z, within_query_z(x, QueryIndex(q, np.ones(60, int))))
    assert np.all(z[q == "c"] == 0.0)
    for a in set(q):
        m = q == a
        assert abs(z[m].mean()) < 1e-12
        sd = x[m].std()
        ref = (x[m] - x[m].mean()) / (sd if sd > 0 else 1.0)
        assert np.allclose(z[m], ref)


def test_global_pooled_auc_equals_published_rule():
    E = fc.epf()
    from go_ppi_fusion.metrics import global_auc
    rng = np.random.default_rng(1)
    s = np.round(rng.normal(size=200), 1)               # ties
    g = rng.integers(0, 3, size=200)
    prep = {"grades": g, "queries": np.asarray(["q"] * 200, dtype=object)}
    assert fc.global_pooled_auc(s, g) == pytest.approx(E.rank_auc(s[g == 2], s[g != 2]), abs=1e-15)
    assert fc.global_pooled_auc(s, g) == pytest.approx(E.validation_selection_value(s, prep, "pooled_auc"), abs=1e-15)
    assert fc.global_pooled_auc(s, g) == global_auc(s, g)


def test_anchor_metrics_match_vectorised_reference():
    from go_ppi_fusion import metrics as M
    rng = np.random.default_rng(2)
    q = np.asarray(rng.choice([f"a{i}" for i in range(12)], 150), dtype=object)
    g = rng.integers(0, 3, 150)
    s = np.round(rng.normal(size=150), 1)
    ai = fc.AnchorIndex(q, g)
    mine, ref = fc.anchor_metrics(s, ai), M.per_query(s, M.QueryIndex(q, g))
    for a, b in (("pooled", "auc_pooled"), ("hard", "auc_hard"), ("easy", "auc_easy"), ("g1g0", "auc_middle_vs_easy"),
                 ("ap", "average_precision"), ("rr", "reciprocal_rank"), ("ndcg10", "ndcg@10"), ("p10", "precision@10")):
        assert np.allclose(mine[a], ref[b], equal_nan=True), a


# ----------------------------------------------------------------- stratified derangement (§2.3)
class _StubMatcher:
    """closure = the set itself; similarity = Jaccard (stands in for simGIC_P in rule (e))."""

    def _closure(self, s):
        return frozenset(s)

    def ontology_set_similarity(self, a, b):
        a, b = frozenset(a), frozenset(b)
        return len(a & b) / len(a | b) if a and b else 0.0


def _shuffle_file(n_anchors=40, extra=()):
    rows = []
    sets = {f"A{i:02d}": [f"T{i}a", f"T{i}b"] + ([f"T{i}c"] if i % 3 == 0 else []) for i in range(n_anchors)}
    sets["A02"] = list(sets["A03"])                     # rule (c): A02 and A03 share a set
    sets["A06"], sets["A07"] = ["S1", "S2", "S3"], ["S1", "S2", "S4"]   # rule (e): Jaccard 0.5
    sets["E0"], sets["E1"] = [], []                      # empty-set anchors
    for a, s in sets.items():
        rows.append((a, f"P{a}", s, [f"PT{a}"]))
    rows.append(("A00", "A01", sets["A00"], sets["A01"]))   # rule (b): A01 is a candidate of A00
    rows.append(("A04", "Pz", sets["A04"], sets["A05"]))    # rule (d): partner set == A05's set
    rows += list(extra)
    q, a, b, ua, ub = zip(*[(r[0], r[0], r[1], r[2], r[3]) for r in rows])
    return list(q), list(a), list(b), [list(x) for x in ua], [list(x) for x in ub]


def test_derangement_rules_determinism_and_empty_sets():
    q, a, b, ua, ub = _shuffle_file()
    m = _StubMatcher()
    out = fc.shuffle_map(q, a, b, ua, ub, m, seed=11)
    again = fc.shuffle_map(q, a, b, ua, ub, m, seed=11)
    assert out["mapping"] == again["mapping"]
    own = {x: frozenset(s) for x, s in zip(q, ua)}
    cands = {x: {p for y, p in zip(q, b) if y == x} for x in own}
    psets = {x: {frozenset(s) for y, s in zip(q, ub) if y == x} for x in own}
    assert "E0" not in out["mapping"] and "E1" not in out["mapping"] and out["n_kept_empty"] == 2
    for r, d in out["mapping"].items():
        assert d != r and d not in cands[r] and own[d] != own[r] and own[d] not in psets[r]
        assert m.ontology_set_similarity(own[r], own[d]) < 0.5
    assert sorted(out["mapping"].values()) == sorted(out["mapping"])      # a permutation within strata
    assert len(out["strata"]) == 10 and all(s["draws"] >= 1 for s in out["strata"])
    shuf = fc.apply_shuffle(q, ua, out["mapping"])
    for x, before, after in zip(q, ua, shuf):
        if x in ("E0", "E1"):
            assert after == before == []
        else:
            assert frozenset(after) == own[out["mapping"][x]]


def test_derangement_aborts_after_10000_draws():
    n = 20
    anchors = [f"A{i:02d}" for i in range(n)]
    rows = [(x, y, [f"T{x}"], [f"T{y}"]) for x in anchors for y in anchors if x != y]   # all candidates of all
    q, b, ua, ub = zip(*rows)
    with pytest.raises(fc.ProtocolError, match="10000 draws"):
        fc.shuffle_map(list(q), list(q), list(b), [list(u) for u in ua], [list(u) for u in ub], _StubMatcher())


# ----------------------------------------------------------------- typed parser, n_PMID
def test_typed_parser_matches_parse_obo_and_drops_part_of(tmp_path):
    from go_ppi_domain.go_ontology import parse_obo
    obo = lf.write_obo(tmp_path / "go.obo")
    ref = parse_obo(obo)
    full = fc.parse_obo_typed(obo, ("is_a", "part_of"))
    assert dict(full[0]) == dict(ref[0]) and full[1:] == ref[1:]
    isa = fc.parse_obo_typed(obo, ("is_a",))
    assert "GO:0000003" in ref[0]["GO:0000004"] and isa[0]["GO:0000004"] == set()
    assert {t: p for t, p in isa[0].items() if t != "GO:0000004"} == {t: p for t, p in ref[0].items() if t != "GO:0000004"}
    with pytest.raises(ValueError):
        fc.parse_obo_typed(obo, ("part_of",))


def test_typed_check_reproduces_index_and_isa_drops_part_of(tmp_path):
    from go_ppi_domain.go_ontology import GoIndex
    obo = lf.write_obo(tmp_path / "go.obo")
    gaf = lf.write_gaf(tmp_path / "g.gaf", [lf.gaf_line("G1", "P1", "GO:0000004"), lf.gaf_line("G2", "P2", "GO:0000002"),
                                            lf.gaf_line("G3", "P3", "GO:0000005"), lf.gaf_line("G1", "P1", "GO:0000002")])
    base = GoIndex.build(obo, gaf, "P")
    assert fc.typed_check(base, obo)["passed"]
    isa = fc.build_typed_index(base, obo, ("is_a",))
    assert "GO:0000003" in base.ancestors["GO:0000004"] and "GO:0000003" not in isa.ancestors["GO:0000004"]
    assert isa.direct == base.direct and set(isa.closure) == set(base.closure)
    assert fc.make_matcher(isa, "simgic").ontology_set_similarity(["GO:0000004"], ["GO:0000002"]) >= 0.0


def test_n_pmid_parser(tmp_path):
    def line(sym, qual, ref):
        return "\t".join(["UniProtKB", "X", sym, qual, "GO:0000001", ref, "IDA", "", "P", "", "", "protein",
                          "taxon:9606", "2024", "UniProt", "", ""]) + "\n"
    p = tmp_path / "g.gaf.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("!gaf-version: 2.2\n")
        fh.writelines([line("A", "involved_in", "PMID:1|PMID:2|GO_REF:0000001"), line("A", "enables", "PMID:2"),
                       line("A", "NOT|involved_in", "PMID:3"), line("B", "involved_in", "PMID:5|GO_REF:2"),
                       line("C", "involved_in", "GO_REF:0000033")])
    n = fc.parse_n_pmid(p)
    assert n == {"A": 2, "B": 1, "C": 0}
    assert np.allclose(fc.bz_scores(["A", "B", "Z"], n), np.log1p([2, 1, 0]))


# ----------------------------------------------------------------- fold builder, HuRI builder
def _toy_proteins(n=40):
    out = {}
    for i in range(n):
        body = f"body {i}" if i not in (5, 6, 7) else "shared body"
        if i in (11, 12):
            body = "second shared"
        out[f"P{i:02d}"] = {"protein_id": f"P{i:02d}", "gene": f"P{i:02d}", "encoder_view": f"P{i:02d} — {body}",
                            "go_uris": [f"GO:{i:07d}"], "coarse_uris": []}
    return out


def _toy_pairs(proteins, seed=3):
    rng = random.Random(seed)
    ids = sorted(proteins)
    pairs = []
    for a in ids:
        for b in rng.sample(ids, 25):
            if b != a:
                pairs.append({"anchor_id": a, "partner_id": b, "grade": rng.choice([0, 0, 0, 1, 2])})
    return pairs


def test_fold_builder_groups_disjoint_and_identical_builds(tmp_path, restore_root):
    fc.ROOT = tmp_path
    prot = _toy_proteins()
    pairs = _toy_pairs(prot)
    groups = fc.view_groups(prot)
    assert groups == [["P05", "P06", "P07"], ["P11", "P12"]]
    folds = fc.assign_protein_folds(prot, groups, 5, 2027)
    assert folds == fc.assign_protein_folds(prot, groups, 5, 2027)
    assert sorted(p for f in folds for p in f) == sorted(prot)
    for g in groups:
        assert len({i for i, f in enumerate(folds) for p in g if p in f}) == 1
    for k in range(5):
        files, m = fc.build_fold(prot, pairs, folds, k, 2027, val_cap=3, groups=groups)
        _, m2 = fc.build_fold(prot, pairs, folds, k, 2027, val_cap=3, groups=groups)
        assert m["rows_sha256"] == m2["rows_sha256"]
        roles = {s: {r["resume"]["protein_id"] for r in files[s]} | {r["job"]["partner_id"] for r in files[s]}
                 for s in ("train", "validation", "test")}
        assert not (roles["train"] & roles["validation"]) and not (roles["train"] & roles["test"])
        assert not (roles["validation"] & roles["test"])
        assert all(r["job"]["grade"] == 2 for r in files["train"])
        per = {}
        for r in files["validation"]:
            if r["job"]["grade"] == 0:
                per[r["resume"]["protein_id"]] = per.get(r["resume"]["protein_id"], 0) + 1
        assert max(per.values(), default=0) <= 3
        assert {p["split"] for p in files["negative_pools"]} <= {"train", "validation"}
        public = fc.write_fold(files, m, tmp_path / f"fold_{k}")
        assert public["sha256"] == m["rows_sha256"]
    bad = [list(folds[0][:1]) + list(folds[1][:1])]
    with pytest.raises(fc.ProtocolError, match="split across folds"):
        fc.build_fold(prot, pairs, folds, 0, 2027, groups=bad)


def _toy_huri():
    prot = _toy_proteins(30)
    part = {p for p in prot if int(p[1:]) < 24}
    edges = [("P00", "P01"), ("P00", "P02"), ("P01", "P03"), ("P04", "P05"), ("P06", "P07"), ("P08", "P09"),
             ("P10", "P11"), ("P12", "P13"), ("P02", "P25"), ("P14", "P15"), ("P16", "P17")]
    pairs = [{"anchor_id": "P00", "partner_id": "P03", "grade": 1},     # HuRI-observed weak partner
             {"anchor_id": "P00", "partner_id": "P18", "grade": 1},     # never observed in HuRI
             {"anchor_id": "P00", "partner_id": "P01", "grade": 1},     # HuRI partner of the anchor -> not grade 1
             {"anchor_id": "P00", "partner_id": "P01", "grade": 2},
             {"anchor_id": "P04", "partner_id": "P19", "grade": 1}]
    nbrs = {"P00": {"P04"}, "P04": {"P06"}}
    closure = {p: (int(p[1:]) % 4) + 1 for p in prot}
    return prot, part, edges, pairs, nbrs, closure


def test_huri_builder_rules_and_determinism():
    prot, part, edges, pairs, nbrs, closure = _toy_huri()
    h1 = fc.build_huri_rows(prot, pairs, part, edges, nbrs, closure, n_neg=5)
    h2 = fc.build_huri_rows(prot, pairs, part, edges, nbrs, closure, n_neg=5)
    assert fc.rows_sha256(h1["rows"]) == fc.rows_sha256(h2["rows"])
    assert "P25" not in h1["anchors"] and "P00" in h1["anchors"]
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for r in h1["rows"]:
        a, b, g = r["resume"]["protein_id"], r["job"]["partner_id"], r["job"]["grade"]
        assert b in part
        if g == 2:
            assert b in adj[a]
        if g == 1:
            assert b in adj and b not in adj[a]                  # HuRI-observed only, no HuRI edge
        if g == 0:
            assert b != a and b not in adj[a] and b not in nbrs.get(a, set()) and b in adj
    g1 = {(r["resume"]["protein_id"], r["job"]["partner_id"]) for r in h1["rows"] if r["job"]["grade"] == 1}
    assert g1 == {("P00", "P03")}
    assert h1["string_grade2_positives"] == [["P00", "P01"]]


def test_huri_unrestricted_differs_only_in_grade1():
    prot, part, edges, pairs, nbrs, closure = _toy_huri()
    h = fc.build_huri_rows(prot, pairs, part, edges, nbrs, closure, n_neg=5)

    def by_grade(rows, g):
        return sorted((r["resume"]["protein_id"], r["job"]["partner_id"]) for r in rows if r["job"]["grade"] == g)

    for g in (2, 0):
        assert by_grade(h["rows"], g) == by_grade(h["rows_grade1_unrestricted"], g)
    assert by_grade(h["rows_grade1_unrestricted"], 1) == [("P00", "P03"), ("P00", "P18"), ("P04", "P19")]
    assert h["counts"]["grade1_counts"] == {"huri_observed_only": 1, "unrestricted": 3}


# ----------------------------------------------------------------- read-only loader and digest
def _write_chunk(d, texts, dim=fc.EMBEDDING_DIM):
    d.mkdir(parents=True, exist_ok=True)
    keys = np.asarray([fc.text_sha256(t) for t in texts])
    np.savez(d / "chunk_0001.npz", keys=keys, embeddings=np.ones((len(texts), dim), np.float32))


def _toy_records(path):
    rows = []
    for i, g in enumerate((2, 1, 0)):
        rows.append({"resume": {"protein_id": "A", "encoder_view": "A — anchor", "skill_uris": ["GO:1"]},
                     "job": {"partner_id": f"B{i}", "encoder_view": f"B{i} — p", "skill_uris": [], "grade": g},
                     "metadata": {"resume_id": "A", "partner_id": f"B{i}"}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ["A — anchor"] + [f"B{i} — p" for i in range(3)]


def test_loader_width_and_missing_text_abort(tmp_path, restore_root):
    fc.ROOT = tmp_path
    ctt = lambda slot, kind: slot["encoder_view"]  # noqa: E731
    path = tmp_path / "f.jsonl"
    texts = _toy_records(path)
    _write_chunk(tmp_path / "ok", texts)
    p = fc.load_prepared(path, chunk_dir=tmp_path / "ok", content_to_text=ctt, log=lambda *_: None)
    assert p["embedding_dim"] == 768 and p["base"].shape == (4, 768) and list(p["grades"]) == [2, 1, 0]
    assert p["uris_a"][0] == ["GO:1"] and p["partners"] == ["B0", "B1", "B2"]
    _write_chunk(tmp_path / "bad", texts, dim=384)
    with pytest.raises(fc.ProtocolError, match="width"):
        fc.load_prepared(path, chunk_dir=tmp_path / "bad", content_to_text=ctt, log=lambda *_: None)
    _write_chunk(tmp_path / "few", texts[:2])
    with pytest.raises(fc.ProtocolError, match="never encodes"):
        fc.load_prepared(path, chunk_dir=tmp_path / "few", content_to_text=ctt, log=lambda *_: None)
    assert not (tmp_path / "missing_dir").exists()
    with pytest.raises(fc.ProtocolError):
        fc.load_prepared(path, chunk_dir=tmp_path / "missing_dir", content_to_text=ctt, log=lambda *_: None)
    assert not (tmp_path / "missing_dir").exists()                     # no mkdir


def test_manifest_digest_cwd_independent(tmp_path, monkeypatch):
    d = tmp_path / "chunks"
    d.mkdir()
    for name in ("chunk_b.npz", "chunk_a.npz"):
        (d / name).write_bytes(name.encode())
    expected = hashlib.sha256("".join(f"{hashlib.sha256(n.encode()).hexdigest()}  {n}\n"
                                      for n in ("chunk_a.npz", "chunk_b.npz")).encode()).hexdigest()
    monkeypatch.chdir(tmp_path)
    one = fc.manifest_digest(d)
    monkeypatch.chdir(d)
    assert fc.manifest_digest(d) == one == expected


# ----------------------------------------------------------------- bootstrap, DE slope
def test_bootstrap_shapes_determinism_and_reference_formula():
    rng = np.random.default_rng(5)
    D = rng.normal(0.01, 0.02, size=(4, 30))
    D[:, 3] = np.nan                                     # undefined anchor (same for every model)
    d1, d2 = fc.Draws([31415, 1], 4, 30, B=200), fc.Draws([31415, 1], 4, 30, B=200)
    assert np.array_equal(d1.C, d2.C) and np.array_equal(d1.P, d2.P) and d1.C.shape == (200, 4)
    s = fc.summarize(D, d1)
    assert s["_T"].shape == (200,) and s["n_defined"] == 29
    assert s["point"] == pytest.approx(np.delete(D, 3, axis=1).mean(axis=0).mean())
    M = D[d1.C].mean(axis=1)                             # trials formula
    T = np.nanmean(np.take_along_axis(M, d1.P, axis=1), axis=1)
    assert np.allclose(s["_T"], T)
    assert np.array_equal(fc.summarize(D, d2)["_T"], s["_T"])
    z = d1.zero_shot()
    assert fc.summarize(D[:1], z)["n_defined"] == 29
    bad = D.copy()
    bad[0, 5] = np.nan
    with pytest.raises(fc.ProtocolError, match="undefined-anchor"):
        fc.summarize(bad, d1)
    assert fc.boot_p(np.ones(99)) == pytest.approx(2 / 100)


def test_de_slope_known_slope():
    fractions = [10, 10, 10, 25, 25, 25, 50, 50, 50, 75, 75, 75, 100, 100, 100]
    x = np.log10(np.asarray(fractions) / 100)
    D = np.repeat((0.02 - 0.01 * x)[:, None], 20, axis=1)
    d = fc.Draws([31415, 1], 15, 20, B=100, strata=fractions)
    s = fc.de_slope(D, fractions, d)
    assert s["point"] == pytest.approx(-0.01) and np.allclose(s["_T"], -0.01)
    assert fc.de_state(s) == "supported"
    flat = fc.de_slope(np.full((15, 20), 0.01), fractions, d)
    assert fc.de_state(flat) == "flat"


# ----------------------------------------------------------------- Holm(3)
def test_holm3_steps_ties_and_harm():
    pts = {"H1": 0.02, "H2": 0.03, "H3": 0.01}
    h = fc.holm3({"H1": 0.001, "H2": 0.02, "H3": 0.04}, pts)
    assert all(h[n]["rejected"] and h[n]["positive"] for n in ("H1", "H2", "H3"))
    assert [h[n]["companion_level"] for n in h["order"]] == pytest.approx([1 - 0.05 / 3, 0.975, 0.95])
    h = fc.holm3({"H1": 0.01, "H2": 0.01, "H3": 0.01}, pts)                 # ties -> H1, H2, H3
    assert h["order"] == ["H1", "H2", "H3"] and all(h[n]["rejected"] for n in ("H1", "H2", "H3"))
    h = fc.holm3({"H1": 0.02, "H2": 0.001, "H3": 0.5}, pts)                 # step 2 fails (0.02 <= 0.025 ok)
    assert h["order"] == ["H2", "H1", "H3"] and h["H1"]["rejected"] and not h["H3"]["rejected"]
    h = fc.holm3({"H1": 0.03, "H2": 0.001, "H3": 0.001}, pts)                # third step: 0.03 <= 0.05
    assert h["order"] == ["H2", "H3", "H1"] and h["H1"]["rejected"]
    h = fc.holm3({"H1": 0.06, "H2": 0.001, "H3": 0.001}, pts)
    assert not h["H1"]["rejected"] and h["H1"]["companion_level"] == pytest.approx(0.95)
    h = fc.holm3({"H1": 0.02, "H2": 0.5, "H3": 0.5}, pts)                   # 0.02 > 0.0167: nothing rejected
    assert not any(h[n]["rejected"] for n in ("H1", "H2", "H3"))
    assert all(h[n]["companion_level"] == pytest.approx(1 - 0.05 / 3) for n in ("H1", "H2", "H3"))
    h = fc.holm3({"H1": 0.001, "H2": 0.5, "H3": 0.5}, {"H1": -0.02, "H2": 0.0, "H3": 0.0})
    assert h["H1"]["rejected"] and h["H1"]["harm"] and not h["H1"]["positive"]
    assert h["H2"]["companion_level"] == pytest.approx(0.975) and h["H3"]["companion_level"] == pytest.approx(0.975)


# ----------------------------------------------------------------- §4.3 example tables
#: C2 examples: (Δ_X point, Δ_S point, Δ_S lo, Δ_S hi, X−S lo, X−S hi, state, trials state).
#: "—" cells use Δ_X = Δ_S + (X − S); "point > point(Δ_S)" uses Δ_S + 0.010.
C2_EXAMPLES = [
    (0.018, 0.006, 0.003, 0.009, 0.007, 0.017, "partial", "fail"),
    (0.017, 0.002, -0.001, 0.004, 0.008, 0.022, "pass", "pass"),
    (0.010, 0.008, 0.004, 0.012, -0.002, 0.0045, "fail", "fail"),
    (0.014, 0.004, -0.001, 0.009, -0.001, 0.020, "inconclusive", "inconclusive"),
    (0.004, 0.0003, -0.0010, 0.0015, 0.0020, 0.0048, "pass", "pass"),
    (0.006, 0.003, -0.001, 0.007, 0.001, 0.0045, "inconclusive", "inconclusive"),
    (0.020, 0.018, 0.014, 0.022, -0.003, 0.007, "fail", "fail"),
    (0.020, 0.0, 0.0, 0.0, 0.010, 0.030, "pass", "pass"),
]


@pytest.mark.parametrize("row", C2_EXAMPLES)
def test_c2_example_table(row):
    dx, ds, ds_lo, ds_hi, d_lo, d_hi, state, trials = row
    assert fc.c2_state(dx, ds, ds_lo, ds_hi, d_lo, d_hi) == state
    assert fc.c2_trials_state(dx, ds, ds_lo, ds_hi, d_lo) == trials


def test_c2_branches_and_fail_over_partial():
    # branch 2 alone (LB(X−S) > 0, UB(X−S) < 0.005, LB(Δ_S) > 0): fail beats partial
    assert fc.c2_state(0.006, 0.003, 0.001, 0.005, 0.0005, 0.0045) == "fail"
    assert fc.c2_state(0.006, 0.003, 0.001, 0.005, 0.0005, 0.0060) == "partial"
    # branch 1 via point(Δ_X) <= point(Δ_S) with LB(Δ_S) <= 0
    assert fc.c2_state(0.002, 0.003, -0.001, 0.007, -0.002, 0.004) == "fail"
    s = {"point": 0.01, "ci95": [0.004, 0.012]}
    st = fc.c2_states_of({"point": 0.01}, {"point": 0.008, "ci95": [0.004, 0.012]}, {"ci95": [-0.002, 0.0045]})
    assert st == {"state": "fail", "trials_state": "fail"} and s


#: C5 examples: (Δ_X, Δ_C5 point, lo, hi, LB95(Δ_X − Δ_C5), state).
C5_EXAMPLES = [
    (0.012, 0.004, 0.001, 0.0045, 0.002, "fail"),
    (0.0045, 0.004, 0.001, 0.0045, -0.001, "pass"),
    (None, -0.001, -0.004, 0.002, None, "fail"),
    (0.003, 0.001, -0.002, 0.004, -0.002, "inconclusive"),
    (0.020, 0.002, -0.002, 0.006, 0.004, "fail"),
]


@pytest.mark.parametrize("row", C5_EXAMPLES)
def test_c5_example_table(row):
    _dx, pt, lo, hi, absorbed, state = row
    for ab in ((absorbed,) if absorbed is not None else (-0.01, 0.01)):     # "—": either value
        assert fc.c5_state(pt, lo, hi, ab) == state


def test_tost_and_four_state():
    assert fc.tost_state(-0.005, 0.008) == "equivalent"
    assert fc.tost_state(0.011, 0.02) == "not equivalent" and fc.tost_state(-0.03, -0.01) == "not equivalent"
    assert fc.tost_state(-0.005, 0.012) == "inconclusive"
    assert fc.four_state(0.001, 0.02, 0.002, 0.018) == "positive"
    assert fc.four_state(-0.02, -0.001, -0.018, -0.002) == "negative"
    assert fc.four_state(-0.004, 0.006, -0.003, 0.005) == "equivalent"
    assert fc.four_state(-0.004, 0.016, -0.003, 0.012) == "inconclusive"


# ----------------------------------------------------------------- final labels and headline
def _li(**over):
    base = {"positive": True, "harm": False, "tost": "not equivalent", "c2": "pass", "c5": "pass",
            "ds": (0.0, -0.001, 0.001), "xs": (0.02, 0.01, 0.03)}
    base.update(over)
    return base


def test_final_label_cases():
    a = fc.final_label(_li(), _li(), "positive")
    assert a == {"label_id": 7, "label": "replicated [HuRI: positive]"}
    labels = {h: a for h in ("H1", "H2", "H3")}
    assert fc.headline(labels).startswith(fc.HEADLINE_BASE + ", replicated: H1: replicated")
    b = fc.final_label(_li(c2="fail"), _li(), "positive")                       # (b) C2_A(H3) fail
    assert b["label_id"] == 2 and b["label"].startswith("not attributable to anchor-specific GO annotations (A)")
    c = fc.final_label(_li(), _li(c5="fail"), "positive")                       # (c) C5_B(H1) fail
    assert c["label_id"] == 6 and c["label"].startswith("confirmed; replication not separable from study bias")
    d = fc.final_label(_li(positive=False, tost="inconclusive", c2="inconclusive"), _li(), "positive")   # (d)
    assert d == {"label_id": 1, "label": "not confirmed (inconclusive)"}
    assert fc.final_label(_li(positive=False, harm=True), _li(), "positive")["label"] == "significant harm (A)"
    e = fc.final_label(_li(), _li(), "equivalent")                              # (e)
    assert e["label"].endswith("[HuRI: equivalent] (not supported on literature-independent HuRI labels)")
    f = fc.final_label(_li(c2="partial", ds=(0.006, 0.003, 0.009), xs=(0.012, 0.007, 0.017)), _li(), "positive")
    assert f["label_id"] == 7 and ("(partly carried by annotation density: Δ_S = +0.0060 [+0.0030, +0.0090]; "
                                   "X − S = +0.0120 [+0.0070, +0.0170]; A)") in f["label"]
    g = fc.final_label(_li(), _li(positive=False, tost="inconclusive"), "positive")   # (g) all_B never enters
    assert g["label_id"] == 4 and g["label"].startswith("confirmed, not replicated (inconclusive)")
    assert "significant harm in B" in fc.final_label(_li(), _li(positive=False, harm=True), "positive")["label"]
    inc = fc.final_label(_li(c2="inconclusive", c5="inconclusive"), _li(c2="inconclusive"), "positive")
    assert "(shuffled control inconclusive: A, B)" in inc["label"] and "(study-bias control inconclusive: A)" in inc["label"]
    three = fc.final_label(_li(c5="fail", c2="inconclusive"), _li(), "positive")
    assert three["label_id"] == 3 and "(shuffled control inconclusive: A)" in three["label"]
    labels = {"H1": a, "H2": a, "H3": fc.final_label(_li(c5="fail"), _li(), "positive")}
    hl = fc.headline(labels)
    assert hl.startswith(fc.HEADLINE_BASE + ", partially replicated (H1, H2): H3: not separable from study bias (A)")
    none = {h: d for h in ("H1", "H2", "H3")}
    assert fc.headline(none) == "H1: not confirmed (inconclusive); H2: not confirmed (inconclusive); " \
                                "H3: not confirmed (inconclusive)"


def test_exposure_warning_sign_disagreement():
    rep = {"label_id": 7, "label": "replicated"}
    labels = {"H1": rep, "H2": rep, "H3": {"label_id": 4, "label": "x"}}
    w = fc.exposure_warnings(labels, {"H1": 0.01, "H2": 0.01, "H3": 0.01}, {"H1": -0.002, "H2": 0.02, "H3": -0.01})
    assert w == ["exposed anchors disagree with the clean replication (H1)"]


# ----------------------------------------------------------------- gate-2 classifier / comparator
@pytest.fixture(scope="module")
def published_run():
    return json.loads(PUBLISHED_NDCG.read_text())["runs"][0]


def test_leaf_classifier_on_real_key_tuples(published_run):
    q0 = published_run["test"]["text"]["ranking"]["per_query"][0]["query"]
    cases = {("test", "fusion", "contrasts", "global_auc", "pooled"): "rank",
             ("test", "delta", "ranking_macro", "ndcg@10"): "rank",
             ("test", "fusion", "ranking", "n_queries"): "exact",
             ("test", "text", "ordinal", "tier_stats", "2", "n"): "exact",
             ("test", "text", "ordinal", "tier_stats", "2", "median"): "continuous",
             ("validation_selection_curve", "0.1"): "rank",
             ("validation_stats", "text", "mean"): "continuous",
             ("selected_weight",): "exact", ("checkpoint",): "identifier",
             ("test", "fusion", "contrasts", "queries_with_both_classes", "pooled"): "exact",
             ("test", "text", "ranking", "per_query", q0, "average_precision"): "rank_per_query",
             ("test", "text", "ordinal", "kendall_tau_b"): "rank",
             ("test", "text", "ordinal", "cohens_d", "hard"): "continuous",
             ("test", "text", "binary_classification", "confusion_matrix", "tn"): "skipped"}
    leaves = dict(fc.iter_leaves(published_run))
    for P, cls in cases.items():
        assert P in leaves, P
        assert fc.leaf_class(P) == cls, P


def _perturb(run, path, delta):
    r = copy.deepcopy(run)
    node = r
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = node[path[-1]] + delta
    return r


def test_gate2_comparator_tiers_and_checkpoint_identifier(published_run):
    run = published_run
    mine = copy.deepcopy(run)
    mine["checkpoint"] = "results/publication_phase1_fixedval/checkpoints/" + "/".join(Path(run["checkpoint"]).parts[-2:])
    assert fc.compare_gate2_run(run, mine)["passed"]
    other = dict(mine, checkpoint="results/x/go_ppi_baseline_f99_s1/best_checkpoint.pt")
    assert not fc.compare_gate2_run(run, other)["passed"]
    for path in (("validation_selection_curve", "0.1"), ("test", "fusion", "ranking", "macro", "ndcg@10")):
        assert fc.compare_gate2_run(run, _perturb(mine, path, 5e-5))["passed"], path
        assert not fc.compare_gate2_run(run, _perturb(mine, path, 2e-4))["passed"], path
    for p in (("validation_stats", "text", "mean"), ("test", "text", "ordinal", "tier_stats", "2", "median")):
        assert fc.compare_gate2_run(run, _perturb(mine, p, 5e-6))["passed"], p
        assert not fc.compare_gate2_run(run, _perturb(mine, p, 2e-5))["passed"], p
    pq = copy.deepcopy(mine)
    pq["test"]["text"]["ranking"]["per_query"][0]["average_precision"] += 5e-4
    assert fc.compare_gate2_run(run, pq)["passed"]
    pq["test"]["text"]["ranking"]["per_query"][0]["average_precision"] += 2e-3
    assert not fc.compare_gate2_run(run, pq)["passed"]
    assert not fc.compare_gate2_run(run, _perturb(mine, ("test", "fusion", "ranking", "n_queries"), 1))["passed"]
    drop = copy.deepcopy(mine)
    del drop["test"]["fusion"]["ordinal"]["cohens_d"]["hard"]
    assert not fc.compare_gate2_run(run, drop)["passed"]
    skip = copy.deepcopy(mine)
    del skip["test"]["fusion"]["binary_classification"]
    assert fc.compare_gate2_run(run, skip)["passed"]


def test_gate2_near_tie_exception(published_run):
    run = published_run
    mine = copy.deepcopy(run)
    mine["selected_weight"] = run["selected_weight"] + 0.1
    assert not fc.compare_gate2_run(run, mine)["passed"]
    assert fc.compare_gate2_run(run, mine, weight_exception=True)["passed"]
    assert fc.top_two_weights({"0.0": 0.5, "0.1": 0.7, "0.2": 0.7, "0.3": 0.6}) == [0.1, 0.2]
    pub = {r: json.loads((ROOT / p).read_text()) for r, p in fc.PUBLISHED_FUSION_REL.items()}
    cells = fc.near_tie_cells(pub)
    assert [tuple(c[:3]) for c in cells] == sorted(fc.GATE2_NEAR_TIE_CELLS)
    assert all(c[3] < 1e-4 for c in cells)


# ----------------------------------------------------------------- pins chain
def _pins(**over):
    base = {"schema": "s", "revision": 1, "supersedes": None, "amendment": None, "created_utc": "t0",
            "protocol": {"path": "p.md", "sha256": "a"}, "git": {"head": "h1"}, "environment": {"host": "x"},
            "inputs": {"f": "1"}, "code": {"c.py": "1"}, "configs": {"base": {}, "derived": {}},
            "analysis": {"B": 2000, "gate2_tolerance_abs": {"rank": 1e-4}}, "confirmatory": True}
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


def test_pins_chain_rules(tmp_path, restore_root):
    fc.ROOT = tmp_path
    p1 = _pins()
    sha1 = _write(tmp_path / "pins.json", p1)
    r2 = _revision(p1, "pins.json", sha1, code={"c.py": "2"}, protocol={"path": "p.md", "sha256": "b"},
                   git={"head": "h2"}, environment={"host": "y"})
    fc.verify_revision(p1, r2, "pins.json", sha1)                                      # whitelist only
    _write(tmp_path / "pins_r2.json", r2)
    assert fc.load_verified_pins(tmp_path / "pins_r2.json")["revision"] == 2
    with pytest.raises(fc.ProtocolError, match="latest"):
        fc.load_verified_pins(tmp_path / "pins.json")                                  # non-latest
    tol = {"B": 2000, "gate2_tolerance_abs": {"rank": 2e-4}}
    fc.verify_revision(p1, _revision(p1, "pins.json", sha1, changed=["analysis.gate2_tolerance_abs"], analysis=tol),
                       "pins.json", sha1)                                              # declared change
    with pytest.raises(fc.ProtocolError, match="declared"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, analysis=tol), "pins.json", sha1)
    with pytest.raises(fc.ProtocolError, match="declared"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, changed=["analysis.B"], analysis=tol), "pins.json", sha1)
    with pytest.raises(fc.ProtocolError, match="confirmatory_output_seen"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, changed=["analysis.gate2_tolerance_abs"], seen=True,
                                         analysis=tol), "pins.json", sha1)
    with pytest.raises(fc.ProtocolError, match="supersedes"):
        fc.verify_revision(p1, r2, "pins.json", "0" * 64)                               # broken link
    with pytest.raises(fc.ProtocolError, match="configs"):
        fc.verify_revision(p1, _revision(p1, "pins.json", sha1, configs={"base": {"x": 1}}), "pins.json", sha1)
    bad_prev = copy.deepcopy(r2)
    bad_prev["amendment"]["previous"]["git"] = {"head": "zz"}
    with pytest.raises(fc.ProtocolError, match="previous"):
        fc.verify_revision(p1, bad_prev, "pins.json", sha1)
    _write(tmp_path / "pins_r2.json", dict(r2, supersedes={"path": "pins.json", "sha256": "f" * 64}))
    with pytest.raises(fc.ProtocolError, match="supersedes"):
        fc.load_verified_pins(tmp_path / "pins_r2.json")


def test_revise_never_rewrites_configs(tmp_path, restore_root):
    mp = load_script("make_pins")
    fc.ROOT = tmp_path
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")
    base = tmp_path / "b.json"
    base.write_text("{}")
    configs = {"base": {"path": "b.json", "sha256": fc.file_sha256(base)},
               "derived": {"c.json": {"sha256": fc.file_sha256(cfg),
                                      "expected_unexpected_fields": list(fc.EXPECTED_UNEXPECTED_FIELDS)}}}
    assert mp.check_configs(configs) is configs
    cfg.write_text('{"x": 1}')
    with pytest.raises(fc.ProtocolError, match="config hashes differ"):
        mp.check_configs(configs)
    assert cfg.read_text() == '{"x": 1}'                                             # never rewritten


def test_compare_code_hashes():
    mac = {"a.py": "1", "b.py": "2", "c.py": "3"}
    assert fc.compare_code_hashes(mac, dict(mac)) == []
    diffs = fc.compare_code_hashes(mac, {"a.py": "1", "b.py": "X", "d.py": "4"})
    assert {(d["path"], d["reason"]) for d in diffs} == {
        ("b.py", "sha256 differs"), ("c.py", "missing on remote"), ("d.py", "not in the Mac manifest")}


# ----------------------------------------------------------------- gate 3
def test_gate3_device_check():
    ok = {"requested": "cuda", "resolved": "cuda", "cuda_device_name": "NVIDIA L40S"}
    assert fc.check_device_block(ok, "NVIDIA L40S") == []
    assert fc.check_device_block(dict(ok, cuda_device_name="NVIDIA A100"), "NVIDIA L40S")
    assert fc.check_device_block(dict(ok, requested="auto"), "NVIDIA L40S")
    assert fc.check_device_block(dict(ok, resolved="cpu"), "NVIDIA L40S")
    assert fc.check_device_block(None, "NVIDIA L40S")


def test_gate3_failures_on_toy_output(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "best_checkpoint.pt").write_bytes(b"x")
    m = {"single_factor_audit": {"unexpected_fields": list(fc.EXPECTED_UNEXPECTED_FIELDS)},
         "attach": {"matcher": None, "cache_check": {"dims": {"(768,)": 5}, "expected_text_dim": 768}},
         "device": {"requested": "cuda", "resolved": "cuda", "cuda_device_name": "NVIDIA L40S"}}
    (out / "lc_manifest.json").write_text(json.dumps(m))
    exp = fc.EXPECTED_UNEXPECTED_FIELDS
    assert fc.gate3_failures(out, 0, exp, "NVIDIA L40S", True) == []
    assert fc.gate3_failures(out, 0, ["embedding_cache_path"], "NVIDIA L40S", True)
    assert fc.gate3_failures(out, 1, exp, "NVIDIA L40S", True)
    assert fc.gate3_failures(out, 0, exp, "NVIDIA L40S", False)
    m["attach"]["matcher"] = "GoMatcher"
    (out / "lc_manifest.json").write_text(json.dumps(m))
    assert any("matcher" in f for f in fc.gate3_failures(out, 0, exp, "NVIDIA L40S", True))
    (out / "run_error.json").write_text("{}")
    assert fc.gate3_failures(out, 0, exp, "NVIDIA L40S", True)


# ----------------------------------------------------------------- protocol parser (verify_transfer.py)
def test_protocol_parser_rows_and_truncation():
    text = PROTOCOL.read_text(encoding="utf-8")
    parsed = fc.parse_protocol(text)
    assert len(parsed["checkpoints"]) == 15 and len(parsed["inputs"]) >= 30
    assert parsed["mpnet_revision"] == "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    assert parsed["prepared"]["test"]["n_chunks"] == 5 and parsed["prepared"]["validation"]["n_chunks"] == 2
    assert parsed["prepared"]["test"]["digest"].startswith("42abdb8c")
    assert parsed["prepared"]["validation"]["digest"].startswith("cefa1e04")
    assert "results/go_ppi_fusion_best/test_results.json" in parsed["inputs"]
    lines = text.splitlines()
    cut = [ln for ln in lines if not (fc.CKPT_ROW.match(ln) and "| 100 | 21 |" in ln)]
    with pytest.raises(fc.ProtocolError, match="checkpoint rows"):
        fc.parse_protocol("\n".join(cut))
    keep = [ln for ln in lines if not fc.INPUT_ROW.match(ln)] + [ln for ln in lines if fc.INPUT_ROW.match(ln)][:29]
    with pytest.raises(fc.ProtocolError, match="input rows"):
        fc.parse_protocol("\n".join(keep))
    with pytest.raises(fc.ProtocolError):
        fc.parse_protocol("\n".join(lines[: len(lines) // 2]))


def test_verify_transfer_requires_code_manifest(tmp_path, monkeypatch):
    vt = load_script("verify_transfer")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(SystemExit):
        vt.main(["--protocol", str(PROTOCOL), "--out", str(tmp_path / "tc.json")])
    with pytest.raises(fc.ProtocolError, match="code-manifest"):
        vt.main(["--protocol", str(PROTOCOL), "--out", str(tmp_path / "tc.json"),
                 "--code-manifest", str(tmp_path / "missing.json")])
    assert not (tmp_path / "tc.json").exists()


# ----------------------------------------------------------------- train.py plan, ledger / resume
def _toy_train_pins():
    data, res = fc.DATA_REL, fc.RESULTS_REL
    derived = {f"{res}/configs/lc_go_ppi_baseline_masked.json": {"expected_unexpected_fields": fc.EXPECTED_UNEXPECTED_FIELDS}}
    for k in range(5):
        derived[f"{res}/configs/lc_go_ppi_baseline_cv_fold{k}.json"] = {"expected_unexpected_fields": fc.EXPECTED_UNEXPECTED_FIELDS}
    return {"configs": {"derived": derived}, "analysis": {"training_order": list(fc.TRAINING_ORDER)},
            "cohortA": {"new_runs": [{"run": f"A:f{F}_s{S}", "fraction": F, "seed": S,
                                      "config": f"{res}/configs/lc_go_ppi_baseline_masked.json",
                                      "train_file": f"{data}/cohortA/frac_{F}/train.jsonl",
                                      "validation_file": f"{data}/cohortA/validation_positive.jsonl",
                                      "fraction_arg": fc.FRACTION_ARG[F], "output_dir": f"o/f{F}_s{S}"}
                                     for F, S in reversed(fc.COHORT_A_RUNS)]},
            "cohortB": {"runs": [{"run": f"B:k{k}_s{S}", "k": k, "seed": S,
                                  "config": f"{res}/configs/lc_go_ppi_baseline_cv_fold{k}.json",
                                  "train_file": f"{data}/cv/fold_{k}/train.jsonl",
                                  "validation_file": f"{data}/cv/fold_{k}/validation_positive.jsonl",
                                  "fraction_arg": "1.0", "output_dir": f"o/k{k}_s{S}"}
                                 for k in range(5) for S in fc.COHORT_B_SEEDS]}}


def test_train_plan_order_and_commands():
    tr = load_script("train")
    pins = _toy_train_pins()
    a, b = tr.plan(pins, "A"), tr.plan(pins, "B")
    assert len(a) == 19 and len(b) == 15 and len(fc.TRAINING_ORDER) == 34
    assert [r["run"] for r in a + b] == list(fc.TRAINING_ORDER)
    assert [r["run"] for r in a][:4] == ["A:f10_s42", "A:f10_s13", "A:f10_s21", "A:f25_s42"]
    assert [r["run"] for r in a][-2:] == ["A:f100_s87", "A:f100_s123"]
    for r in a + b:
        cmd = r["command"]
        assert cmd[cmd.index("--device") + 1] == "cuda" and cmd[cmd.index("--domain") + 1] == "go_ppi"
        assert cmd[cmd.index("--compare-config") + 1] == fc.BASE_CONFIG_REL
        assert not any(str(c).startswith("/") for c in cmd[1:])
    for r in a:
        cmd = r["command"]
        assert float(cmd[cmd.index("--fraction") + 1]) == r["fraction"] / 100


def test_ledger_resume_semantics(tmp_path, restore_root):
    fc.ROOT = tmp_path
    led = fc.Ledger(tmp_path / "run_ledger.json")
    (tmp_path / "run_ledger.json").write_text(fc.Ledger._render([]))
    led.append({"run": "A", "status": "passed", "order_index": 0})
    calls = []

    def factory(results):
        def attempt(run, n):
            calls.append((run["run"], n))
            fc.absp(run["output_dir"]).mkdir(parents=True, exist_ok=True)
            return {"run": run["run"], "status": results.pop(0), "attempt": n}
        return attempt

    assert fc.run_with_resume(led, {"run": "A", "output_dir": "outA"}, factory([])) == "skipped-passed"
    before = (tmp_path / "run_ledger.json").read_text()
    (tmp_path / "outB").mkdir()
    assert fc.run_with_resume(led, {"run": "B", "output_dir": "outB", "order_index": 1}, factory(["failed"])) == "excluded"
    assert calls == [("B", 2)]
    assert (tmp_path / "outB.failed1").is_dir() and (tmp_path / "outB.failed2").is_dir()
    assert (tmp_path / "run_ledger.json").read_text().startswith(before.rstrip("\n]").rstrip())
    assert [e["status"] for e in led.entries() if e["run"] == "B"] == ["failed", "failed", "excluded"]
    assert fc.run_with_resume(led, {"run": "B", "output_dir": "outB"}, factory([])) == "skipped-excluded"
    assert fc.run_with_resume(led, {"run": "C", "output_dir": "outC"}, factory(["failed", "passed"])) == "passed"
    assert [e["status"] for e in led.entries() if e["run"] == "C"] == ["failed", "passed"]


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_hashseed_guard(name, monkeypatch):
    mod = load_script(name)
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with pytest.raises(fc.ProtocolError, match="PYTHONHASHSEED"):
        mod.main(["--help"])
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(fc.ProtocolError, match="PYTHONHASHSEED"):
        mod.main(["--help"])
