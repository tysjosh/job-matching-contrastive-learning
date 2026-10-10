"""Frozen target table: graph neighbourhoods, roles, matching, singleton packs (design 2, AM1)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding_constants as wc  # noqa: E402
from go_ppi_domain import withholding_targets as wt  # noqa: E402
from go_ppi_domain.go_ontology import GoIndex  # noqa: E402
from go_ppi_domain.tests import _leak_fixtures as fx  # noqa: E402
from scripts.go_withholding import build_target_table as btt  # noqa: E402


# ------------------------------------------------------------------ graph
def _graph(parents):
    anc = {}
    for t in parents:
        acc, stack = {t}, [t]
        while stack:
            for p in parents[stack.pop()]:
                if p not in acc:
                    acc.add(p)
                    stack.append(p)
        anc[t] = frozenset(acc)
    children = {t: frozenset(c for c in parents if t in parents[c]) for t in parents}
    return wt.BpGraph(parents={t: frozenset(p) for t, p in parents.items()},
                      children=children, ancestors=anc, name={t: t for t in parents},
                      depth={t: len(anc[t]) - 1 for t in parents})


def test_n_sib_and_n2_exclude_ancestors_and_descendants():
    g = _graph({"R": (), "A": ("R",), "B": ("R",), "t": ("A",), "s": ("A",),
                "c": ("t",), "d": ("t", "A"), "s2": ("s",), "b1": ("B",)})
    desc = wt.descendants(g)
    assert desc["t"] == {"c", "d"}
    assert wt.n_sib("t", g, desc) == {"s"}            # d is a child of A but a descendant of t
    assert wt.n_2("t", g, desc) == {"s"}              # A, R ancestors; c, d descendants
    assert wt.n_2("s2", g, desc) == frozenset()         # s and A are ancestors
    assert "b1" not in wt.n_2("t", g, desc)           # 4 hops away
    assert wt.n_sib("b1", g, desc) == frozenset()


def test_regulates_and_obsolete_ignored(tmp_path):
    bulk = fx.write_bulk_dir(tmp_path / "bulk", n=12)
    index = GoIndex.build(bulk["obo"], bulk["gaf"], "P")
    g = wt.bp_graph(bulk["obo"], index)
    assert g.parents["GO:0000003"] == {"GO:0008150"}
    assert "GO:0000099" not in g.parents
    assert "GO:0000001" not in g.ancestors["GO:0000004"]   # only part_of -> GO:0000003


# ------------------------------------------------------------------ roles
def _rec(a, b, grade=2):
    return {"resume": {"protein_id": a}, "job": {"partner_id": b}, "label": int(grade == 2),
            "metadata": {"resume_id": a, "protein_id": a, "partner_id": b, "go_grade": grade}}


def test_roles_T_train_vs_P_train():
    from go_ppi_domain.tests._withholding_fixtures import build_index

    index = build_index({"a": ["T5"], "b": ["T1"], "pool_only": ["T5"], "anchor_only": ["T5"],
                         "c": ["T2"]})
    records = [_rec("a", "b"), _rec("b", "c")]
    pools = [{"split": "train", "anchor_id": "a", "weak_evidence": ["pool_only"],
              "no_interaction": ["c"]},
             {"split": "train", "anchor_id": "anchor_only", "weak_evidence": [],
              "no_interaction": ["b"]}]
    roles = wt.partition_roles(records, pools)
    assert roles["positive"] == {"a", "b", "c"}
    assert roles["any"] == {"a", "b", "c", "pool_only", "anchor_only"}
    T = wt.term_proteins(index, roles["any"])
    P = {t: s & roles["positive"] for t, s in T.items()}
    assert T["T5"] == {"a", "pool_only", "anchor_only"}
    assert P["T5"] == {"a"}
    assert T["T1"] == {"a", "b", "pool_only", "anchor_only"}        # propagated closure
    assert P["R"] == {"a", "b", "c"}


def test_support_S_arithmetic():
    pairs = wt.undirected_positive_pairs([_rec("a", "b"), _rec("b", "a"), _rec("a", "c"),
                                          _rec("d", "e")])
    assert pairs == {("a", "b"), ("a", "c"), ("d", "e")}
    deg = wt.degree(pairs)
    assert deg == Counter({"a": 2, "b": 1, "c": 1, "d": 1, "e": 1})
    P = {"n1": {"a", "b"}, "n2": {"b", "d"}, "t": {"c"}}
    out = wt.support_S("t", ["n1", "n2"], P, deg, T_train_t={"c", "b"})
    assert out == {"S": 2 + 1, "proteins": 2}                       # NP = {a, d}


def test_retention_eligible_rule():
    P = {"t": {"p1", "p2", "p3", "p4"}}
    by_protein = {"p1": [("p1", "q1")], "p2": [("p2", "w1")], "p3": [("q3", "p3")],
                  "p4": [("p4", "q4")]}
    pools = {"p1": frozenset({"m1"}), "p2": frozenset({"m1"}), "q3": frozenset({"w2"}),
             "p4": frozenset({"ghost"})}
    views = {"m1", "w2", "q1", "q3", "q4", "p1", "p2", "p3", "p4"}
    W = {"w1", "w2", "p1", "p2", "p3", "p4"}
    # p1: ok; p2: partner withheld; p3: anchor's pool only W; p4: pool member not in views
    assert wt.retention_eligible("t", P, W, by_protein, pools, views) == {"p1"}


# --------------------------------------------------------------- matching
def _row(t, n, sup, s):
    return {"go_id": t, "n_train_proteins": n, "test_support": sup, "S_sib": s}


def test_match_strata_deterministic_and_jaccard_queue():
    low = [_row(f"GO:L{i}", 40 + 5 * i, 8 + i, 1) for i in range(4)]
    high = [_row(f"GO:H{i}", 41 + 5 * i, 8 + i, 9) for i in range(4)]
    P = {r["go_id"]: {f"{r['go_id']}_{j}" for j in range(10)} for r in low + high}
    # GO:H2 overlaps GO:L0 heavily. Cost order is (L3,H3), (L2,H2), (L1,H1), (L0,H0):
    # m01, m02 accepted, m03 queued, (L0,H0) skipped by the Jaccard check (not numbered).
    P["GO:H2"] = set(list(P["GO:L0"])[:8]) | {"x1", "x2"}
    a = wt.match_strata(low, high, P, n_pairs=2)
    b = wt.match_strata(list(reversed(low)), list(reversed(high)), P, n_pairs=2)
    assert a == b
    ids = [p["pair_id"] for p in a["accepted"] + a["queue"]]
    assert ids == ["m01", "m02", "m03"]
    assert [(p["low"], p["high"]) for p in a["accepted"] + a["queue"]] == [
        ("GO:L3", "GO:H3"), ("GO:L2", "GO:H2"), ("GO:L1", "GO:H1")]
    costs = [p["cost"] for p in a["accepted"] + a["queue"]]
    assert costs == sorted(costs)


def _pairs(n, start=1):
    return [{"pair_id": f"m{i:02d}", "low": f"L{i:02d}", "high": f"H{i:02d}",
             "cost": float(i)} for i in range(start, start + n)]


def test_size_rule():
    acc, q, short = wt.apply_size_rule(_pairs(12), _pairs(1, 13))
    assert [p["pair_id"] for p in acc] == [f"m{i:02d}" for i in range(1, 13)] and not short
    assert [p["pair_id"] for p in q] == ["m13"]
    acc, q, short = wt.apply_size_rule(_pairs(11), [])
    assert len(acc) == 10 and short and [p["pair_id"] for p in q] == ["m11"]
    acc, q, short = wt.apply_size_rule(_pairs(10), [])
    assert len(acc) == 10 and short and q == []
    with pytest.raises(wt.TargetTableStop):
        wt.apply_size_rule(_pairs(9), [])
    # 13 accepted by match_strata cannot happen (it stops at n_pairs) but the rule is total:
    acc, q, short = wt.apply_size_rule(_pairs(13), [])
    assert len(acc) == 12 and [p["pair_id"] for p in q] == ["m13"]


# ---------------------------------------------------------------- packing
def test_singleton_packs_one_target_per_pack_in_order():
    """AM1: no pack-mates; each target alone must pass the per-target checks."""
    seen = []

    def pack_ok(pk):
        seen.append(pk)
        return True

    targets = ["L01", "H01", "L02", "H02"]
    assert wt.singleton_packs(targets, pack_ok) == [["L01"], ["H01"], ["L02"], ["H02"]]
    assert seen == [frozenset([t]) for t in targets]          # never a multi-target pack


def test_singleton_packs_stop_names_counts_only():
    with pytest.raises(wt.TargetTableStop) as exc:
        wt.singleton_packs(["GO:0000101", "GO:0000102", "GO:0000103"],
                           lambda pk: pk != frozenset(["GO:0000102"]))
    msg = str(exc.value)
    assert "1 of 3" in msg and "GO:" not in msg               # protocol 4: no ids


def test_default_params_are_singleton():
    assert wt.DEFAULT_PARAMS["pack_size"] == 1
    assert wt.DEFAULT_PARAMS["pack_removal_cap"] == 0.08
    for gone in ("pack_nbr_overlap_cap", "pack_search_seed"):
        assert gone not in wt.DEFAULT_PARAMS
    for gone in ("pack_shape", "pack_targets", "pack_search", "DEFAULT_RESTARTS"):
        assert not hasattr(wt, gone)


# ------------------------------------------------------------- end to end
LEAVES_LOW = [f"GO:00001{i:02d}" for i in range(1, 5)]
LEAVES_HIGH = [f"GO:00002{i:02d}" for i in range(1, 5)]

E2E_PARAMS = {"min_train_proteins": 4, "min_test_support": 1, "max_removal_frac": 0.3,
              "max_protein_frac": 0.3, "min_retention_eligible": 4, "max_jaccard": 0.2,
              "n_pairs": 2, "min_pairs": 1, "pack_removal_cap": 0.5}


def _obo() -> str:
    terms = [("GO:0008150", "biological_process", [])]
    terms += [("GO:0000100", "group low", ["GO:0008150"]),
              ("GO:0000200", "group high", ["GO:0008150"])]
    terms += [(t, f"low leaf {i}", ["GO:0000100"]) for i, t in enumerate(LEAVES_LOW)]
    terms += [(t, f"high leaf {i}", ["GO:0000200"]) for i, t in enumerate(LEAVES_HIGH)]
    out = ["format-version: 1.2\n"]
    for t, name, parents in terms:
        out.append(f"\n[Term]\nid: {t}\nname: {name}\nnamespace: biological_process\n")
        out.extend(f"is_a: {p} ! x\n" for p in parents)
    out.append("relationship: regulates GO:0000200 ! x\n")     # ignored
    return "".join(out)


def write_target_fixture(root: Path) -> dict:
    """Converted + splits dirs: 4 low-support and 4 high-support sibling leaves."""
    root = Path(root)
    obo = root / "go-basic.obo"
    obo.write_text(_obo(), encoding="utf-8")
    gaf_lines, train, pools, test, val = [], [], [], [], []
    leaves = LEAVES_LOW + LEAVES_HIGH
    proteins = set()

    def annotate(p, t):
        gaf_lines.append(fx.gaf_line(p, f"A{p}", t))
        proteins.add(p)

    hubs = [f"HUB{k}" for k in range(10)]
    members = [f"MEM{k}" for k in range(6)]
    proteins.update(hubs + members)
    for i, t in enumerate(leaves):
        high = t in LEAVES_HIGH
        for j in range(6):
            p = f"P{i}_{j}"
            annotate(p, t)
            for d in range(3 if high else 1):
                train.append(_rec(p, hubs[(j + d + i) % 10]))
            pools.append({"anchor_id": p, "split": "train",
                          "weak_evidence": [f"X{i}"], "no_interaction": members[:3]})
        annotate(f"X{i}", t)                                   # pool-only, annotated
        for j in range(i + 1):
            a = f"T{i}_{j}"
            annotate(a, t)
            test.append(_rec(a, f"TP{i}_{j}a", 2))
            test.append(_rec(a, f"TP{i}_{j}b", 1))
            proteins.update({f"TP{i}_{j}a", f"TP{i}_{j}b"})
        v = f"V{i}"
        annotate(v, t)
        val.append(_rec(v, f"VP{i}", 2))
        proteins.add(f"VP{i}")
    pools.append({"anchor_id": "T0_0", "split": "test", "weak_evidence": ["TP0_0b"],
                  "no_interaction": []})
    pools.append({"anchor_id": "V0", "split": "validation", "weak_evidence": ["VP1"],
                  "no_interaction": []})
    gaf = fx.write_gaf(root / "goa_human.gaf.gz", gaf_lines)
    conv, splits = root / "converted", root / "splits"
    conv.mkdir()
    splits.mkdir()
    index = GoIndex.build(obo, gaf, "P", leak_pruning={"with_from_human_protein": False,
                                                        "excluded_term_roots": []})
    index.save(conv / "go_index_P.pkl")
    with open(conv / "proteins.jsonl", "w", encoding="utf-8") as fh:
        for p in sorted(proteins):
            fh.write(json.dumps({"protein_id": p, "gene": p, "encoder_view": f"{p} — text",
                                 "go_uris": sorted(index.terms_for(p)),
                                 "coarse_uris": [], "n_terms": 0}) + "\n")
    (conv / "pairs.jsonl").write_text("", encoding="utf-8")
    for name, rows in (("train.jsonl", train), ("test.jsonl", test),
                       ("validation.jsonl", val), ("negative_pools.jsonl", pools)):
        with open(splits / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    frozen = {k: wc.sha256_file((conv if where == "converted" else splits) / n)
              for k, (where, n) in btt.FROZEN_FILES.items()}
    (splits / "frozen_hashes.json").write_text(json.dumps(frozen), encoding="utf-8")
    return {"obo": obo, "converted": conv, "splits": splits}


def _inputs():
    return {"train_sha256": "t", "test_sha256": "x"}


def test_end_to_end_table_deterministic(tmp_path):
    f = write_target_fixture(tmp_path)
    ctx = wt.load_context(f["converted"], f["splits"], f["obo"])
    assert ctx.T_train[LEAVES_LOW[0]] == {f"P0_{j}" for j in range(6)} | {"X0"}
    assert ctx.P_train[LEAVES_LOW[0]] == {f"P0_{j}" for j in range(6)}
    table, rows = wt.build_target_table(ctx, _inputs(), params=E2E_PARAMS)
    ctx2 = wt.load_context(f["converted"], f["splits"], f["obo"])
    table2, _ = wt.build_target_table(ctx2, _inputs(), params=E2E_PARAMS)
    assert wt.table_json_bytes(table) == wt.table_json_bytes(table2)

    assert table["schema_version"] == 2
    assert table["n_candidates"] == 8 and table["n_non_nested"] == 8
    assert table["strata_cuts"]["low_max_S_sib"] == 18.0
    assert table["strata_cuts"]["high_min_S_sib"] == 54.0
    # AM1: one target per pack, so n_packs == n_targets and 5 * 4 + 1 conditions.
    assert table["n_targets"] == 4 and table["n_packs"] == 4 and not table["shortfall"]
    assert table["n_conditions"] == 21 and table["n_runs"] == 4 * 5 * 21
    assert table["params"]["pack_size"] == 1
    assert [p["pair_id"] for p in table["pairs"]] == ["m01", "m02"]
    assert table["replacement_queue"] == ["m03", "m04"]
    assert "pack_search_log" not in table
    assert table["train_totals"]["pool_only_proteins"] == 8 + 3       # X_i + MEM0..2
    assert [p["pack_id"] for p in table["packs"]] == ["p01", "p02", "p03", "p04"]
    assert [p["targets"] for p in table["packs"]] == [[t["go_id"]] for t in table["targets"]]
    assert [t["pack_id"] for t in table["targets"]] == ["p01", "p02", "p03", "p04"]
    assert [(t["pair_id"], t["stratum"]) for t in table["targets"]] == [
        ("m01", "low"), ("m01", "high"), ("m02", "low"), ("m02", "high")]
    for pk, t in zip(table["packs"], table["targets"]):
        assert set(pk) == {"pack_id", "targets", "removal_frac_k0", "withheld_k0",
                           "min_retention_eligible"}
        assert pk["withheld_k0"] == t["n_train_annotated"]
        assert pk["removal_frac_k0"] == pytest.approx(t["removal_frac"])
        assert pk["min_retention_eligible"] == len(t["retention_eligible"])
    for t in table["targets"]:
        assert isinstance(t["retention_order"], list)
        assert sorted(t["retention_order"]) == t["retention_eligible"]
        assert len(t["retention_order"]) == 6
        assert set(t["train_annotated_proteins"]) - set(t["train_proteins"]) == \
            {f"X{(LEAVES_LOW + LEAVES_HIGH).index(t['go_id'])}"}
        assert t["test_pos_records"] == t["test_support"] == t["test_weak_records"]
        assert len(t["validation_proteins"]) == 1
        assert t["S_sib"] == (18 if t["stratum"] == "low" else 54)
        assert len(t["N_sib"]) == 3
        assert not set(t["nbr_train_proteins_sib"]) & set(t["train_annotated_proteins"])
    assert {t["stratum"] for t in table["targets"]} == {"low", "high"}
    rows_by = {r["go_id"]: r for r in rows}
    assert rows_by["GO:0000100"]["rule4_protein_frac"] is False      # group term too big


def test_script_preflight_refusal_and_stop(tmp_path, monkeypatch):
    f = write_target_fixture(tmp_path)
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    params = dict(wt.DEFAULT_PARAMS)
    params.update(E2E_PARAMS)
    monkeypatch.setattr(wt, "DEFAULT_PARAMS", params)
    out = tmp_path / "targets" / "target_table.json"
    args = ["--converted", str(f["converted"]), "--splits", str(f["splits"]),
            "--obo", str(f["obo"]), "--out", str(out)]
    assert btt.main(args) == 2                                  # no G1 pin yet
    assert not out.exists()
    wc.append_pin("G1", {"test_sha256": "x"})
    assert btt.main(args) == 0
    digest = (out.parent / "target_table.json.sha256").read_text().split()[0]
    assert digest == wc.sha256_file(out)
    assert json.loads((out.parent / "candidates.json").read_text())["rows"]
    pin = wc.read_pins()[-1]
    assert pin["gate"] == "G2-targets" and pin["target_table_sha256"] == digest
    assert pin["n_conditions"] == 21 and pin["n_packs"] == 4 and pin["n_runs"] == 420
    assert btt.main(args) == 2                                  # refuses to overwrite
    # < min_pairs matched pairs -> protocol stop, exit 3
    params2 = dict(params)
    params2.update({"n_pairs": 12, "min_pairs": 10})
    monkeypatch.setattr(wt, "DEFAULT_PARAMS", params2)
    out2 = tmp_path / "t2" / "target_table.json"
    assert btt.main(args[:-1] + [str(out2)]) == 3
    assert not out2.exists()
    # a target above the single-target removal cap -> protocol stop, exit 3 (AM1)
    params3 = dict(params)
    params3["pack_removal_cap"] = 0.0
    monkeypatch.setattr(wt, "DEFAULT_PARAMS", params3)
    out4 = tmp_path / "t4" / "target_table.json"
    assert btt.main(args[:-1] + [str(out4)]) == 3
    assert not out4.exists()
    monkeypatch.setattr(wt, "DEFAULT_PARAMS", params2)
    # tampered frozen file -> exit 2
    (f["splits"] / "frozen_hashes.json").write_text(json.dumps({"train_sha256": "0"}))
    assert btt.main(args[:-1] + [str(tmp_path / "t3" / "target_table.json")]) == 2
