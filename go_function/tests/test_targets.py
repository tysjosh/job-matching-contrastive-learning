import importlib.util
import json

import numpy as np
import pytest
import scipy.sparse as sp

from go_function import labels as L
from go_function import targets as T
from go_function.ontology import ancestors, parse_obo
from go_function.pins import canonical_json
from go_function.runtime import ROOT, ProtocolStop

BP_VOCAB = ["GO:0000010", "GO:0000011", "GO:0000012", "GO:0000013", "GO:0000014",
            "GO:0000015", "GO:0000016", "GO:0000099"]


def _script():
    spec = importlib.util.spec_from_file_location(
        "build_targets_script", ROOT / "scripts" / "go_function" / "build_targets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ strata and flags
def test_tertiles_and_stratum():
    assert T.tertile_cuts(list(range(10))) == pytest.approx((3.0, 6.0))
    cuts = (3.0, 6.0)
    assert [T.stratum(s, cuts) for s in (0, 3, 4, 5, 6, 9)] == \
        ["low", "low", None, None, "high", "high"]


def test_bands():
    bands = T.TargetParams().bands
    assert [T.band_of(n, bands) for n in (29, 30, 99, 100, 249, 250, 4000)] == \
        [None, "B1", "B1", "B2", "B2", "B3", "B3"]


def test_ld_flag(mini_obo):
    ont = parse_obo(mini_obo)
    assert T.ld_flag(ont, "GO:0000010", {}, 30) == ("no", [])
    flag, fill = T.ld_flag(ont, "GO:0000016", {"GO:0000011": 40}, 30)
    assert flag == "yes" and fill == [["positively_regulates", "GO:0000011", 40]]
    assert T.ld_flag(ont, "GO:0000016", {"GO:0000011": 29}, 30) == (None, [])


def test_filler_support_counts_non_column_terms(mini_obo):
    ont = parse_obo(mini_obo)
    anc = ancestors(ont)
    vocab = ["GO:0000010"]
    direct = sp.csr_matrix(np.array([[1], [0], [0]], dtype=np.uint8))
    extra = [(), ("GO:0000011",), ("GO:0000012",)]
    n = T.tp_closure_counts(direct, extra, vocab, anc, "GO:0008150")
    assert n["GO:0000011"] == 1            # not a column, still counted from the closure
    assert n["GO:0000010"] == 3 and n["GO:0000013"] == 1
    assert "GO:0008150" not in n


def test_alt_id_exclusion(mini_obo):
    ont = parse_obo(mini_obo)
    with_alt = L.build_vocab(BP_VOCAB, ont)
    assert T.alt_excluded("GO:0000011", ont, with_alt, frozenset())
    no_alt = L.build_vocab([g for g in BP_VOCAB if g != "GO:0000099"], ont)
    assert not T.alt_excluded("GO:0000011", ont, no_alt, frozenset())
    assert T.alt_excluded("GO:0000011", ont, no_alt, frozenset({"GO:0000099"}))
    assert not T.alt_excluded("GO:0000012", ont, with_alt, frozenset({"GO:0000099"}))


def test_seed_formula_and_retention():
    import hashlib
    assert T.seed_for(20261011, "GO:1") == int(
        hashlib.sha256(b"20261011:GO:1").hexdigest()[:16], 16)
    names = [f"P{i:02d}" for i in range(20)]
    rows = list(range(20))[::-1]
    a = T.retention_order(rows, names, "GO:1", 20261011)
    assert a == T.retention_order(sorted(rows), names, "GO:1", 20261011)   # input order free
    assert sorted(a) == list(range(20))
    assert a != T.retention_order(rows, names, "GO:1", 20261012)


# ------------------------------------------------------------------ acceptance
def _P(**kw):
    return {k: frozenset(v) for k, v in kw.items()}


ANC0 = {t: frozenset({t}) for t in ("a1", "a2", "a3", "a4", "b1", "b2")}


def test_round_robin_visit_order_one_acceptance_per_visit():
    slots = {("A", "B1"): ["a1", "a2"], ("A", "B2"): ["a3"], ("B", "B1"): ["b1"]}
    quotas = {("A", "B1"): 2, ("A", "B2"): 1, ("B", "B1"): 1}
    P = _P(a1={1}, a2={2}, a3={3}, b1={4})
    visits = []
    acc = T.accept_round_robin(slots, quotas, P, ANC0, 0.2, visits=visits)
    assert acc == ["a1", "a3", "b1", "a2"]
    assert visits == [(("A", "B1"), "a1"), (("A", "B2"), "a3"), (("B", "B1"), "b1"),
                      (("A", "B1"), "a2")]


def test_borrowing_moves_quota_to_next_band():
    slots = {("A", "B1"): ["a1"], ("A", "B2"): ["a2", "a3"], ("A", "B3"): []}
    quotas = {("A", "B1"): 2, ("A", "B2"): 1, ("A", "B3"): 0}
    acc = T.accept_round_robin(slots, quotas, _P(a1={1}, a2={2}, a3={3}), ANC0, 0.2)
    assert sorted(acc) == ["a1", "a2", "a3"]


def test_borrowing_wraps_b3_to_b1():
    slots = {("A", "B1"): ["a1", "a2"], ("A", "B2"): [], ("A", "B3"): ["a3"]}
    quotas = {("A", "B1"): 1, ("A", "B2"): 0, ("A", "B3"): 2}
    acc = T.accept_round_robin(slots, quotas, _P(a1={1}, a2={2}, a3={3}), ANC0, 0.2)
    assert sorted(acc) == ["a1", "a2", "a3"]


def test_ancestor_and_jaccard_rejection():
    anc = dict(ANC0)
    anc["a2"] = frozenset({"a2", "a1"})            # a1 is an ancestor of a2
    slots = {("A", "B1"): ["a1", "a2", "a3", "a4"]}
    P = _P(a1={1, 2}, a2={3}, a3={1, 2, 9}, a4={7})  # J(a1, a3) = 2/3 > 0.2
    acc = T.accept_round_robin(slots, {("A", "B1"): 2}, P, anc, 0.2)
    assert acc == ["a1", "a4"]


def test_shortfall_is_protocol_stop():
    slots = {("A", "B1"): ["a1"], ("A", "B2"): [], ("A", "B3"): []}
    with pytest.raises(T.Shortfall) as e:
        T.accept_round_robin(slots, {("A", "B1"): 2, ("A", "B2"): 0, ("A", "B3"): 0},
                             _P(a1={1}), ANC0, 0.2)
    assert isinstance(e.value, ProtocolStop) and e.value.exit_code == 3
    assert "1 of 2" in str(e.value)


# ------------------------------------------------------------------ packing (A1)
def _pack(order, P, dtp=None, ddg=None, th=None, cells=None, n_packs=2, size=2, share=1.0,
          n_total=100):
    E = frozenset()
    dtp = dtp or {t: frozenset({t}) for t in order}
    ddg = ddg or {t: frozenset({t}) for t in order}
    th = th or {t: E for t in order}
    cells = cells or {t: "c" for t in order}
    return T.pack_targets(order, cells, th, dtp, ddg, P, n_total, n_packs=n_packs,
                          pack_size=size, max_union_share=share)


def test_placement_order_is_decreasing_n_train_then_id():
    P = _P(b={1, 2}, a={3, 4}, c={5, 6, 7}, d={8})
    assert T.placement_order(["d", "b", "a", "c"], P) == ["c", "a", "b", "d"]


def test_pack_preference_uses_union_after_adding():
    # placement c (3), a (2), b (1). c -> p01 (tie, pack id). a: same-cell count 1 in
    # p01 vs 0 in p02 -> p02. b: same-cell 1 in both; union after adding is 4 in p01
    # and 3 in p02 -> p02 (the v3 rule would pick p01 by pack size).
    P = _P(c={1, 2, 3}, a={1, 2}, b={9})
    packs = _pack(["a", "b", "c"], P, size=3)
    assert packs == {"p01": ["c"], "p02": ["a", "b"]}
    assert packs == _pack(["c", "b", "a"], P)           # deterministic, input order free
    assert sorted(t for ts in packs.values() for t in ts) == ["a", "b", "c"]


def test_packing_rejects_planted_non_column_term_implying_two_candidates():
    P = _P(t1={1}, t2={2})
    ddg = {"t1": frozenset({"t1", "GO:X"}), "t2": frozenset({"t2", "GO:X"})}
    with pytest.raises(ProtocolStop):
        _pack(["t1", "t2"], P, ddg=ddg, n_packs=1, size=2)
    packs = _pack(["t1", "t2"], P, ddg=ddg, n_packs=2, size=2)
    assert sorted(len(v) for v in packs.values()) == [1, 1]


def test_packing_two_hop_and_union_limits():
    P = _P(t1={1, 2}, t2={3})
    th = {"t1": frozenset({"t2"}), "t2": frozenset()}
    with pytest.raises(ProtocolStop):
        _pack(["t1", "t2"], P, th=th, n_packs=1, size=2)
    with pytest.raises(ProtocolStop):
        _pack(["t1", "t2"], P, n_packs=1, size=2, share=0.02, n_total=100)   # 3 > 2
    assert _pack(["t1", "t2"], P, n_packs=1, size=2, share=0.03, n_total=100) == \
        {"p01": ["t1", "t2"]}


def test_main_exit_3_on_protocol_stop(tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_inputs", lambda aspect: object())

    def stop(inp):
        raise T.Shortfall("target shortfall in cell low: 3 of 24 accepted, 3 slots exhausted")
    monkeypatch.setattr(mod.T, "build_table", stop)
    assert mod.main(["--aspect", "mf"]) == 3
    assert not (tmp_path / "preprocess" / "go_function" / "mf" / "targets" /
                "target_table.json").exists()


# ------------------------------------------------------------------ real data (A1)
def _have_labels(aspect):
    return all((ROOT / "preprocess" / "go_function" / aspect / f"labels_{r}.json").exists()
               for r in ("TP", "DGZ"))


@pytest.mark.needs_dgz
@pytest.mark.parametrize("aspect", ["bp", "mf"])
def test_real_packing_8x6_and_target_set_unchanged(aspect):
    if not _have_labels(aspect):
        pytest.skip("G1 label files absent")
    mod = _script()
    inp = mod.load_inputs(aspect)
    tab = T.build_table(inp)
    packed = [t for p in tab["packs"].values() for t in p["targets"]]
    assert len(tab["packs"]) == 8 and all(len(p["targets"]) == 6 for p in tab["packs"].values())
    assert sorted(packed) == sorted(tab["acceptance_order"]) and len(set(packed)) == 48
    assert tab["target_set_sha256_before_packing"] == tab["target_set_sha256_after_packing"]
    assert sorted(tab["targets"]) == sorted(tab["acceptance_order"])
    lim = 0.10 * tab["parameters"]["n_train_total"]
    assert all(p["union_P_TP"] <= lim for p in tab["packs"].values())
    assert tab["filter_counts"]["alt_id_excluded_before_n_test"] > 0
    assert canonical_json(tab) == canonical_json(T.build_table(inp))         # deterministic
    on_disk = ROOT / "preprocess" / "go_function" / aspect / "targets" / "target_table.json"
    if on_disk.exists():
        assert on_disk.read_bytes() == canonical_json(tab)
    assert b"/Users/" not in canonical_json(tab)
    json.loads(canonical_json(T.summary_counts(tab)))
