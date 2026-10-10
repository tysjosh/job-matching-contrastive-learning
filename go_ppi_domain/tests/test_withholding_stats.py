"""Statistics tests (design 11, 14; protocol 7-8)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from go_ppi_domain import withholding as wh
from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.tests import _grid_fixtures as gf
from scripts.go_withholding import analyze_withholding as aw

SEEDS = list(wc.SEEDS)


# ------------------------------------------------------------------ bootstrap
def _correlated(U=24, seed=3, mu=0.05, seed_sd=0.02, target_sd=0.01, noise=0.005):
    rng = np.random.default_rng(seed)
    return (mu + rng.normal(0, seed_sd, 5)[None, :] + rng.normal(0, target_sd, U)[:, None]
            + rng.normal(0, noise, (U, 5)))


def test_two_way_bootstrap_is_deterministic_for_a_fixed_seed():
    X = _correlated()
    a = aw.two_way_bootstrap(X, list(range(24)), 24, B=500, seed=20251009)
    b = aw.two_way_bootstrap(X, list(range(24)), 24, B=500, seed=20251009)
    c = aw.two_way_bootstrap(X, list(range(24)), 24, B=500, seed=1)
    assert a == b
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])
    assert a["scheme"] == "two_way_pack_seed" and a["n_discarded_replicates"] == 0


def test_two_way_ci_covers_known_mean_and_is_wider_than_target_then_seed():
    X = _correlated()
    tw = aw.two_way_bootstrap(X, list(range(24)), 24, B=4000, seed=20251009)
    s1 = aw.hierarchical_bootstrap(X, B=4000, seed=20251009)
    assert tw["estimate"] == pytest.approx(float(X.mean()))
    assert tw["ci_low"] < 0.05 < tw["ci_high"]
    assert tw["ci_low"] < X.mean() < tw["ci_high"]
    assert (tw["ci_high"] - tw["ci_low"]) > 1.5 * (s1["ci_high"] - s1["ci_low"])


def test_pack_clusters_resample_targets_together():
    # two targets per pack, identical within a pack: the pack is the unit
    X = np.repeat(_correlated(U=6, seed_sd=0.0), 2, axis=0)
    per_target = aw.two_way_bootstrap(X, list(range(12)), 12, B=2000, seed=5)
    per_pack = aw.two_way_bootstrap(X, [i // 2 for i in range(12)], 6, B=2000, seed=5)
    assert (per_pack["ci_high"] - per_pack["ci_low"]) > (per_target["ci_high"] - per_target["ci_low"])


def test_discarded_replicates_are_counted_and_unstable_above_5_percent():
    X = np.full((24, 5), np.nan)
    X[0, 0] = 0.1
    r = aw.two_way_bootstrap(X, list(range(24)), 24, B=1000, seed=20251009)
    Cp, Cs = aw.two_way_draws(24, 5, 1000, 20251009)
    expected = int(((Cp[:, 0] == 0) | (Cs[:, 0] == 0)).sum())
    assert r["n_discarded_replicates"] == expected > 50
    assert r["unstable"] is True
    assert aw.classify_contrast({**r, "p_holm": 0.0}) == "inconclusive"
    full = aw.two_way_bootstrap(np.full((24, 5), 0.1), list(range(24)), 24, B=1000)
    assert full["n_discarded_replicates"] == 0 and full["unstable"] is False
    empty = aw.two_way_bootstrap(np.full((3, 5), np.nan), [0, 1, 2], 3, B=100)
    assert empty["estimate"] is None and empty["ci_low"] is None and empty["unstable"]


def test_moderation_is_high_minus_low_and_discards_one_sided_replicates():
    X = np.zeros((4, 5))
    X[[1, 3]] = 0.05
    strata = [0, 1, 0, 1]
    r = aw.two_way_bootstrap(X, [0, 1, 2, 3], 4, B=1000, seed=7, strata=strata)
    assert r["estimate"] == pytest.approx(0.05)
    Cp, _ = aw.two_way_draws(4, 5, 1000, 7)
    one_sided = int(((Cp[:, [1, 3]].sum(1) == 0) | (Cp[:, [0, 2]].sum(1) == 0)).sum())
    assert r["n_discarded_replicates"] == one_sided
    assert r["ci_low"] == pytest.approx(0.05) and r["ci_high"] == pytest.approx(0.05)


def test_s4_recovers_the_high_coefficient():
    high = [0, 1] * 6
    tsim = list(np.linspace(-1, 1, 12))
    y = np.asarray([0.01 + 0.03 * h + 0.02 * t for h, t in zip(high, tsim)])
    X = np.repeat(y[:, None], 5, axis=1)
    r = aw.s4_adjusted(X, high, tsim, list(range(12)), 12, B=200, seed=1)
    assert r["estimate"] == pytest.approx(0.03)
    assert r["ci_low"] == pytest.approx(0.03) and r["ci_high"] == pytest.approx(0.03)


# ------------------------------------------------------------------ Holm, rules
def test_holm_ordering_and_monotonicity():
    adj = aw.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    adj2 = aw.holm({"a": 0.2, "b": 0.001})
    assert adj2 == pytest.approx({"b": 0.002, "a": 0.2})
    adj3 = aw.holm({"a": None, "b": 0.02, "c": 0.5})
    assert adj3["a"] is None and adj3["b"] == pytest.approx(0.06) and adj3["c"] == 1.0


def _c(est, lo, hi, p=0.001, unstable=False):
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "p_holm": p, "unstable": unstable}


@pytest.mark.parametrize("c,expected", [
    (_c(-0.02, -0.03, -0.001), "harm"),
    (_c(-0.01, -0.03, -0.001), "harm"),                     # boundary estimate == -0.01
    (_c(-0.009, -0.03, -0.001), "inconclusive"),            # CI < 0 but above the floor
    (_c(0.01, 0.002, 0.03, p=0.01), "supported"),           # boundary estimate == +0.01
    (_c(0.03, 0.002, 0.05, p=0.0499), "supported"),
    (_c(0.03, 0.002, 0.05, p=0.05), "inconclusive"),        # review finding 8 boundary
    (_c(0.03, 0.0, 0.05, p=0.001), "inconclusive"),         # ci_low == 0 is not > 0
    (_c(0.0, -0.005, 0.009), "practically null"),
    (_c(0.0, -0.01, 0.009), "inconclusive"),                # strict: -0.01 < ci_low
    (_c(0.005, 0.001, 0.0099), "practically null"),         # rule 3 precedes rule 4
    (_c(0.008, 0.001, 0.02, p=0.001), "below floor"),
    (_c(0.009, 0.001, 0.02, p=0.9), "below floor"),
    (_c(-0.05, -0.06, -0.04, unstable=True), "inconclusive"),  # unstable checked first
    (_c(None, None, None), "inconclusive"),
])
def test_ordered_decision_rules(c, expected):
    assert aw.classify_contrast(c) == expected


# ------------------------------------------------------------------ definedness
def _rows(values):
    """values: {(go, k, cell, seed): (hard_auc, collateral)} -> long-table rows."""
    return [{"go_id": g, "k": k, "cell": c, "seed": s, "pack_id": "p" + g[-2:],
             "hard_auc": v[0], "collateral_hard_auc": v[1], "relation_excess": None}
            for (g, k, c, s), v in values.items()]


def _full_values(gos=("GO:01", "GO:02"), seeds=(13, 21)):
    vals = {}
    for g in gos:
        for k in aw.K_LEVELS:
            for s in seeds:
                for i, c in enumerate(wc.CELLS):
                    base = 0.6 + (0.01 * (k if k != "all" else 50)) / 10
                    vals[(g, k, c, s)] = (base + 0.01 * i, 0.7 + 0.002 * i)
    return vals


def test_delta_small_and_D_definedness():
    vals = _full_values()
    for c in wc.CELLS:
        del vals[("GO:01", 1, c, 13)]                 # one small k missing for (GO:01, 13)
        del vals[("GO:02", "all", c, 21)]             # `all` missing for (GO:02, 21)
    d = aw.paired_deltas(_rows(vals))
    assert ("GO:01", 13) not in d["small"]["loss_only"]
    assert ("GO:01", 13) not in d["D"]["loss_only"]
    assert ("GO:01", 21) in d["small"]["loss_only"]
    assert ("GO:02", 21) in d["small"]["both"] and ("GO:02", 21) not in d["D"]["both"]
    assert d["small"]["loss_only"][("GO:01", 21)] == pytest.approx(0.01)
    assert d["small"]["both"][("GO:01", 21)] == pytest.approx(0.03)
    assert d["D"]["both"][("GO:01", 21)] == pytest.approx(0.0)


def test_E_c_equals_hand_built_fixture():
    vals = {("GO:01", 0, "baseline", 13): (0.60, 0.70), ("GO:01", 0, "loss_only", 13): (0.66, 0.72),
            ("GO:01", 0, "neg_only", 13): (0.61, 0.70), ("GO:01", 0, "both", 13): (0.70, 0.75)}
    d = aw.paired_deltas(_rows(vals))
    assert d["E"]["loss_only"][("GO:01", 13)] == pytest.approx((0.66 - 0.60) - (0.72 - 0.70))
    assert d["E"]["both"][("GO:01", 13)] == pytest.approx((0.70 - 0.60) - (0.75 - 0.70))
    assert d["coll"]["both"][0][("p01", 13)] == pytest.approx(0.05)
    vals2 = dict(vals)
    vals2[("GO:01", 0, "loss_only", 13)] = (0.66, None)       # collateral undefined
    assert ("GO:01", 13) not in aw.paired_deltas(_rows(vals2))["E"]["loss_only"]


# ------------------------------------------------------------------ completeness / grid
def _value(go, pack, k, cell, seed):
    eff = {"baseline": 0.0, "loss_only": 0.03, "neg_only": 0.0, "both": 0.02}[cell]
    small = 0.02 if k in (0, 1, 5) and cell in ("loss_only", "both") else 0.0
    return 0.6 + eff + small + 0.001 * (seed % 7) + 0.002 * int(go[-1])


def _grid(tmp_path, table, skip=()):
    runs = tmp_path / "runs"
    for cond in wh.condition_ids(table):
        for s in SEEDS:
            if (cond, s) in skip:
                continue
            gf.write_group(runs, table, cond, s, value=_value,
                           relation_excess=lambda g, c, cell, s: 0.05 if cell != "baseline"
                           else 0.0)
    return runs


def test_completeness_denominator_is_n_targets_x_6_x_5():
    table = gf.make_table(3)
    groups = {(c, s): "checked" for c in wh.condition_ids(table) for s in SEEDS}
    groups[("p01_k0", 13)] = "incomplete"
    groups[("all", 21)] = "failed_member"
    ts = aw.tuple_status(table, groups)
    assert ts["denominator"] == 3 * 6 * 5 == 90
    # p01_k0,s13 drops one tuple (one target); all,s21 drops (t, all, 21) for every target
    assert len(ts["missing"]) == 1 + 3
    assert {m["reason"] for m in ts["missing"]} == {"incomplete", "failed_member"}
    assert ts["missing_share"] == pytest.approx(4 / 90)
    t24 = gf.make_table(24)
    assert aw.tuple_status(t24, {})["denominator"] == 720


def test_analysis_end_to_end_and_incomplete_rule(tmp_path):
    table = gf.make_table(2)
    tpath = gf.write_table(table, tmp_path / "targets")
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"targets": {t["go_id"]: {"tsim_excess": 0.1 * i}
                                           for i, t in enumerate(table["targets"])}}))
    runs = _grid(tmp_path, table)
    out = tmp_path / "analysis"
    rc = aw.main(["--runs", str(runs), "--target-table", str(tpath), "--covariate", str(cov),
                  "--out", str(out), "--B", "300"])
    assert rc == 0
    for name in ("per_target.csv", "contrasts.json", "missing.json", "sensitivity.json",
                 "decisions.json", "diagnostics_summary.json"):
        assert (out / name).is_file()
    con = json.loads((out / "contrasts.json").read_text())
    fam = con["families"]
    assert sorted(fam) == ["H1", "H2", "H3", "descriptive"]
    assert sorted(fam["H1"]) == ["both", "loss_only", "neg_only"]
    assert sorted(fam["H3"]) == ["both", "loss_only"]
    assert fam["H1"]["loss_only"]["estimate"] == pytest.approx(0.05)
    assert fam["H1"]["neg_only"]["estimate"] == pytest.approx(0.0)
    assert fam["H2"]["loss_only"]["estimate"] == pytest.approx(0.02)
    assert fam["H1"]["loss_only"]["n_defined_cells"] == 2 * 5
    assert fam["H1"]["loss_only"]["p_holm"] >= fam["H1"]["loss_only"]["p_boot"]
    assert con["exploratory"] is False
    dec = json.loads((out / "decisions.json").read_text())
    assert dec["H1"]["neg_only"]["classification"] == "practically null"
    assert set(dec) >= {"H1", "H2", "H3", "A1", "mechanism", "collateral"}
    assert dec["mechanism"]["a_relation_ci_low_gt_0"] is True
    miss = json.loads((out / "missing.json").read_text())
    assert miss["denominator"] == 60 and miss["n_missing"] == 0
    rows = (out / "per_target.csv").read_text().splitlines()
    assert len(rows) == 1 + 2 * 6 * 5 * 4
    sens = json.loads((out / "sensitivity.json").read_text())
    assert set(sens) >= {"S1", "S2", "S3", "S4", "S5"}

    # > 10% missing -> exit 3, unless --allow-incomplete (exploratory)
    import shutil
    for cond in ("p01_k0", "p02_k0"):
        for s in SEEDS:
            for rid in wc.group_run_ids(cond, s):
                shutil.rmtree(runs / rid)
    out2 = tmp_path / "analysis2"
    args = ["--runs", str(runs), "--target-table", str(tpath), "--out", str(out2), "--B", "100"]
    assert aw.main(args) == 3
    assert aw.main(args + ["--allow-incomplete"]) == 0
    assert json.loads((out2 / "decisions.json").read_text())["exploratory"] is True
    assert json.loads((out2 / "missing.json").read_text())["n_missing"] == 10


def test_classify_groups_statuses(tmp_path):
    table = gf.make_table(1)
    runs = tmp_path / "runs"
    gf.write_group(runs, table, "all", 13, value=_value)                    # checked
    gf.write_group(runs, table, "all", 21, value=_value, checked=False)     # unchecked
    d = gf.write_group(runs, table, "p01_k0", 13, value=_value)
    (d[1] / "FAILED.json").write_text("{}")                                  # failed member
    d = gf.write_group(runs, table, "p01_k1", 13, value=_value, checked=False)
    (d[2] / "GROUP_FAILED.json").write_text("{}")                            # group failed
    g = aw.classify_groups(runs, table)
    assert g[("all", 13)] == "checked"
    assert g[("all", 21)] == "unchecked"
    assert g[("p01_k0", 13)] == "failed_member"
    assert g[("p01_k1", 13)] == "group_failed"
    assert g[("p01_k5", 13)] == "incomplete"
    with pytest.raises(aw.AnalysisStop) as e:
        aw.preflight(g)
    assert e.value.code == 2


def test_no_statsmodels_and_scipy_only_for_spearman():
    src = Path(aw.__file__).read_text(encoding="utf-8")
    assert "import statsmodels" not in src and "from statsmodels" not in src
    code = ("import sys; sys.path.insert(0, %r); import scripts.go_withholding.analyze_withholding;"
            "print('statsmodels' in sys.modules)" % str(wc.ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    assert "default_rng" in src and "scipy.stats.bootstrap" not in src
