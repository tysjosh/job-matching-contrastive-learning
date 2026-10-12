"""Analysis (plan item 16): planted effect recovered, null -> practically null, the
C3 decomposition identity, channel reads against hand AUCs, empty channel sets
excluded and counted, F and DS hand examples, DONE/run.json hash disagreement."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from go_function import baselines as BL
from go_function import fusion as FU
from go_function import stats as ST
from go_function.constants import SEEDS
from go_function.pins import canonical_json, sha256_bytes
from go_function.runtime import FatalError

ROOT = Path(__file__).resolve().parents[2]
B = 2000


def _table(desc_sizes=None, x_empty=()):
    packs, targets = {}, {}
    for i in range(8):
        p = f"p{i + 1:02d}"
        ts = [f"GO:{i:03d}{j:04d}" for j in range(6)]
        packs[p] = {"targets": ts}
        for j, t in enumerate(ts):
            nd = (desc_sizes or {}).get(t, 3 if j % 2 else 8)
            targets[t] = {"pack": p, "cell": ("high" if j < 3 else "low") + ("_LDyes" if j % 2 else "_LDno"),
                          "LD": "yes" if j % 2 else "no",
                          "channels": {"X": [] if t in x_empty else [f"X{t}"],
                                       "Desc": [f"D{t}_{n}" for n in range(nd)], "Pa": [f"P{t}"]}}
    return {"packs": packs, "targets": targets}


def _rows(units, arms, value, ks=(0,), seeds=SEEDS, feature="esm2_650m"):
    rows = []
    for (variant, regime, rule) in arms:
        for t in units.targets:
            for k in ks:
                for s in seeds:
                    v = value(variant, regime, rule, t, k, s)
                    if v is None:
                        continue
                    rows.append({"feature": feature, "variant": variant, "regime": regime,
                                 "rule": rule, "draw": 1, "pack": units.pack_of[t] if k != "all" else "all",
                                 "target": t, "k": k, "seed": s, "class_auc": v})
    return rows


def test_planted_effect_recovered_and_null_is_practically_null():
    units = ST.units_from_table(_table())
    rng = np.random.default_rng(0)
    base = {(t, s): 0.7 + 0.05 * rng.standard_normal() for t in units.targets for s in SEEDS}
    noise = {(t, k, s): 0.003 * rng.standard_normal() for t in units.targets for k in (0, 1, 5)
             for s in SEEDS}

    def val(v, regime, rule, t, k, s):
        b = base[(t, s)] + 0.01 * k
        if v == "V4":
            return b + 0.05 + noise[(t, k, s)]
        if v == "V3":
            return b + noise[(t, k, s)]
        if v == "V3c":
            return b - noise[(t, k, s)] * 0.5
        return b
    at = ST.AucTable(_rows(units, [("V0", "TP", "W2"), ("V4", "TP", "W2"), ("V3", "TP", "W2"),
                                   ("V3c", "TP", "W2")], val, ks=(0, 1, 5)))
    c2 = ST.estimate(ST.matrix(units, ST.v_diff(ST.v_auc(at, ("esm2_650m", "V4", "TP", "W2", 1), 0),
                                                ST.v_auc(at, ("esm2_650m", "V0", "TP", "W2", 1), 0))),
                     units, B, 20261011, s1=True)
    small = ST.v_small(lambda k: ST.v_diff(ST.v_auc(at, ("esm2_650m", "V3", "TP", "W2", 1), k),
                                           ST.v_auc(at, ("esm2_650m", "V3c", "TP", "W2", 1), k)))
    c1 = ST.estimate(ST.matrix(units, small), units, B, 20261011, s1=True)
    fam = {"C1": c1, "C2": c2}
    ST.classify_family(fam)
    assert abs(c2["estimate"] - 0.05) < 0.005
    assert c2["ci_low"] <= 0.05 <= c2["ci_high"] and c2["S1"]["ci_low"] <= 0.05 <= c2["S1"]["ci_high"]
    assert c2["classification"] == "supported"
    assert c1["classification"] == "practically null"
    assert c1["n_defined_cells"] == 48 * 5


def test_delta_small_needs_all_three_ks_and_vbest_tie_rule():
    units = ST.units_from_table(_table())
    at = ST.AucTable(_rows(units, [("V1", "TP", "W2"), ("V0", "TP", "W2")],
                           lambda v, r, ru, t, k, s: 0.6 if v == "V1" else 0.5, ks=(0, 1)))
    X = ST.matrix(units, ST.v_small(lambda k: ST.v_diff(
        ST.v_auc(at, ("esm2_650m", "V1", "TP", "W2", 1), k),
        ST.v_auc(at, ("esm2_650m", "V0", "TP", "W2", 1), k))))
    assert np.isnan(X).all()                       # k = 5 missing -> undefined
    assert ST.choose_vbest({"V1": 0.02, "V2": 0.03, "V3": 0.03, "V4": None}) == "V2"
    assert ST.choose_vbest({}) is None


def test_c3_decomposition_identity_and_both_c3_values():
    units = ST.units_from_table(_table())
    rng = np.random.default_rng(1)
    vals = {(arm, t, s): rng.random() for arm in ("WDGZrel", "W2drop", "DGZW2", "TPW2")
            for t in units.targets for s in SEEDS}
    drop = (units.targets[0], 13)

    def val(v, regime, rule, t, k, s):
        key = {"DGZ": {"WDGZrel": "WDGZrel", "W2drop": "W2drop", "W2": "DGZW2"}, "TP": {"W2": "TPW2"}}
        name = key[regime][rule]
        if name == "W2drop" and (t, s) == drop:
            return None
        return vals[(name, t, s)]
    at = ST.AucTable(_rows(units, [("V4", "DGZ", "WDGZrel"), ("V4", "DGZ", "W2drop"),
                                   ("V4", "DGZ", "W2"), ("V4", "TP", "W2")], val))
    d = ST.decomposition(at, units, "esm2_650m", B, 20261011)
    assert d["identity_max_abs_residual"] <= 1e-12
    assert d["n_common_cells"] == 48 * 5 - 1 and d["n_own_cells"] == 48 * 5
    assert d["cell_sets_differ"] is True
    parts = sum(d[k]["estimate"] for k in ("C3_leak", "C3_col", "C3_reg"))
    assert abs(parts - d["C3_common"]["estimate"]) < 1e-12
    a = np.array([vals[("WDGZrel", t, s)] - vals[("TPW2", t, s)] for t in units.targets for s in SEEDS])
    assert abs(d["C3_own"]["estimate"] - a.mean()) < 1e-12


def test_channel_reads_match_hand_aucs_and_empty_sets_are_counted():
    table = _table(x_empty={"GO:0000000"})
    units = ST.units_from_table(table)
    t = units.targets[0]
    t2 = units.targets[1]
    rng = np.random.default_rng(2)
    n = 40
    y = (rng.random(n) < 0.3).astype(int)
    ids = []
    for tt in (t, t2):
        ids += units.channels[tt]["X"] + units.channels[tt]["Desc"] + units.channels[tt]["Pa"]
    a = {"logits": rng.standard_normal((n, len(ids))).astype(np.float32), "column_ids": np.asarray(ids)}
    b = {"logits": rng.standard_normal((n, len(ids))).astype(np.float32), "column_ids": np.asarray(ids)}
    reads = ST.channel_reads(a, b, units.channels, {t: y, t2: y})
    assert reads[t]["X"] is None                                   # empty X(t)
    pos = {g: i for i, g in enumerate(ids)}

    def hand(sc, cols):
        return roc_auc_score(y, sc["logits"][:, [pos[c] for c in cols]].max(axis=1))
    for ch in ("Desc", "Pa"):
        cols = units.channels[t][ch]
        assert abs(reads[t][ch] - (hand(a, cols) - hand(b, cols))) < 1e-12
    cols = units.channels[t2]["X"]
    assert abs(reads[t2]["X"] - (hand(a, cols) - hand(b, cols))) < 1e-12
    cells = {(tt, s): reads[tt] for tt in (t, t2) for s in SEEDS}
    est = ST.channel_estimates(cells, units, B, 20261011)
    assert est["C3_leak_X"]["n_targets_empty"] == 1 and est["C3_leak_X"]["n_targets_nonempty"] == 47
    assert est["C3_leak_X"]["n_defined_cells"] == 5               # only t2 has an X read
    assert abs(est["C3_leak_desc"]["estimate"] - np.mean([reads[t]["Desc"], reads[t2]["Desc"]])) < 1e-12
    # |Desc| strata: target 0 has 8 columns (> 5), target 1 has 3 (<= 5)
    assert abs(est["C3_leak_desc_desc_gt_5"]["estimate"] - reads[t]["Desc"]) < 1e-12
    assert abs(est["C3_leak_desc_desc_le_5"]["estimate"] - reads[t2]["Desc"]) < 1e-12
    assert est["compare_by_ci_sign_only"]["C3_leak_X"] in ("ci > 0", "ci < 0", "ci includes 0")


def test_fusion_hand_example():
    own = np.array([1.0, 2.0, 3.0, 4.0])
    pa = np.array([[0.0, 2.0], [2.0, 2.0], [2.0, 4.0], [4.0, 4.0]])      # mean 1, 2, 3, 4
    sib = np.array([[4.0], [3.0], [2.0], [1.0]])
    s, info = FU.fuse(own, pa, sib)
    z = (own - own.mean()) / own.std()
    assert np.allclose(s, z + z - (-z))
    assert info == {"n_pa": 2, "n_sib": 1, "pa_dropped": False, "sib_dropped": False}
    s2, info2 = FU.fuse(own, np.zeros((4, 0)), None)
    assert np.allclose(s2, z) and info2["pa_dropped"] and info2["sib_dropped"]
    assert np.allclose(FU.zscore(np.ones(4)), 0.0)
    scores = {"logits": np.column_stack([own, pa, sib]).astype(np.float32),
              "columns": np.array([5, 7, 8, 9])}
    counts = np.zeros(10, dtype=int)
    counts[9] = 9                                                   # sibling below 10 positives
    out = FU.fusion_scores(scores, [("t", 5, [7, 8], [9])], counts)
    assert out["t"][1]["sib_dropped"] and out["t"][1]["n_sib_candidates"] == 1
    assert np.allclose(out["t"][0], z + z)


def test_diamond_score_hand_example(tmp_path):
    f = tmp_path / "d.res"
    f.write_text("q1\th1\t10\t50\nq1\th2\t30\t60\nq1\tzz\t99\t1\nq2\th2\t5\t40\n")
    H = BL.diamond_matrix(BL.read_diamond(f), ["q1", "q2", "q3"], ["h1", "h2"])
    y = sp.csr_matrix(np.array([[1, 0], [1, 1]], dtype=np.uint8))
    S = BL.diamond_score(H, y, [0, 1])
    assert np.allclose(S, [[1.0, 0.75], [1.0, 1.0], [0.0, 0.0]])
    S2 = BL.diamond_score(H, y, [0, 1], mask_pairs=np.array([[1, 0]]))   # h2, column 0 masked
    assert np.allclose(S2[:, 0], [0.25, 0.0, 0.0]) and np.allclose(S2[:, 1], [0.75, 1.0, 0.0])
    assert np.allclose(BL.naive_frequency(y), [1.0, 0.5])


def _analyze():
    spec = importlib.util.spec_from_file_location("analyze_t", ROOT / "scripts" / "go_function" / "analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_done_hash_disagreement_is_fatal(tmp_path):
    A = _analyze()
    d = tmp_path / "runs" / "bp" / "interpro" / "V0__TP__W2__p01__k0__s13"
    d.mkdir(parents=True)
    eb = canonical_json({"targets": {}})
    (d / "eval.json").write_bytes(eb)
    rb = canonical_json({"spec": {"aspect": "bp"}, "outputs": {"eval.json": sha256_bytes(eb)}})
    (d / "run.json").write_bytes(rb)
    (d / "DONE.json").write_bytes(canonical_json({"run_json_sha256": sha256_bytes(rb),
                                                  "eval.json_sha256": sha256_bytes(eb)}))
    runs, counts = A.discover(tmp_path / "runs")
    assert counts["done"] == 1
    (d / "eval.json").write_bytes(canonical_json({"targets": {"x": 1}}))
    with pytest.raises(FatalError):
        A.discover(tmp_path / "runs")
    (d / "eval.json").write_bytes(eb)
    (d / "run.json").write_bytes(rb.replace(b"bp", b"mf"))
    with pytest.raises(FatalError):
        A.discover(tmp_path / "runs")
    # attempt dirs are ignored
    (d / "run.json").write_bytes(rb)
    (d.parent / f"{d.name}.attempt1").mkdir()
    (d.parent / f"{d.name}.attempt1" / "FAILED.json").write_text("{}")
    assert A.discover(tmp_path / "runs")[1] == {"done": 1, "failed": 0, "incomplete": 0}


def test_dgz_label_sensitivity_auc_matches_hand_auc(tmp_path):
    """Protocol §7 sensitivity read: target logits scored against DGZ-rule test labels
    on the run's own evaluation rows; a degenerate DGZ column gives None."""
    A = _analyze()
    rng = np.random.default_rng(3)
    n_test, rows = 30, np.arange(2, 26)
    cols = np.array([1, 4, 6])
    logits = rng.standard_normal((len(rows), len(cols))).astype(np.float32)
    np.savez(tmp_path / "scores.npz", logits=logits, columns=cols, eval_rows=rows)
    y = np.zeros((n_test, 8), dtype=np.uint8)
    y[rng.random(n_test) < 0.4, 4] = 1
    y[0, 6] = 1                                         # positive only outside the eval rows
    r = A.RunRec(tmp_path, {"spec": {}}, {"targets": {"GO:a": {}, "GO:b": {}}})
    out = A.dgz_label_aucs(r, sp.csc_matrix(y), {"GO:a": 4, "GO:b": 6})
    assert abs(out["GO:a"] - roc_auc_score(y[rows, 4], logits[:, 1])) < 1e-9
    assert out["GO:b"] is None
    with pytest.raises(FatalError):
        A.dgz_label_aucs(r, sp.csc_matrix(y), {"GO:a": 4, "GO:b": 7})
    row = A.long_rows(A.RunRec(tmp_path, {"spec": {"aspect": "bp", "feature": "interpro",
                                                   "variant": "V0", "regime": "TP", "rule": "W2",
                                                   "pack": "p01", "k": 0, "seed": 13}},
                               {"targets": {"GO:a": {"auc": 0.6}}}), out)[0]
    assert row["class_auc_dgz_labels"] == out["GO:a"] and row["class_auc"] == 0.6


def test_collateral_harm_and_missing_share():
    units = ST.units_from_table(_table())
    vals = {}
    for i, p in enumerate(units.packs):
        for s in SEEDS:
            vals[("esm2_650m", "V0", "TP", "W2", 1, p, 0, s)] = {"fmax": 0.5, "micro_aupr": 0.4}
            vals[("esm2_650m", "V4", "TP", "W2", 1, p, 0, s)] = {"fmax": 0.45 + 0.001 * i,
                                                                "micro_aupr": 0.4}
    c = ST.collateral_contrast(vals, "esm2_650m", "V4", "TP", "W2", 0, units.packs, "fmax", B, 1)
    assert c["classification"] == "harm"
    at = ST.AucTable([])
    m = ST.missing_share(at, units, "esm2_650m")
    assert m["needed"] == 48 * 6 * 5 * 8 and m["share_missing"] == 1.0
