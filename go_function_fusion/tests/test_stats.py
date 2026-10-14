"""Bootstraps (fixed draw order), Holm, classification, null labels, snapshot rule."""
import os

import numpy as np
import pytest

from go_function_fusion import stats as ST


def test_term_draws_fixed_order_U_S_R():
    band = np.array([0, 1, 1, 2, 4, 4, 5, 1])
    reg = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
    dr = ST.term_draws(8, band, reg, B=5, seed=7)
    rng = np.random.default_rng(7)
    assert np.array_equal(dr.U, rng.integers(0, 8, size=(5, 8)))
    for name, n in (("0", 1), ("1-10", 3), ("11-30", 1), ("101-1000", 2), (">1000", 1)):
        assert np.array_equal(dr.S[name], rng.integers(0, n, size=(5, n)))
    assert "31-100" not in dr.S                         # empty band: no draw block
    assert np.array_equal(dr.R, rng.integers(0, 3, size=(5, 3)))
    assert dr.reg_members.tolist() == [0, 2, 5]


def test_contrasts_point_estimates():
    band = np.array([0, 0, 1, 4, 5])
    d = np.array([0.2, 0.4, 0.1, 0.0, 0.05])
    dr = ST.term_draws(5, band, np.array([1, 0, 0, 0, 1], bool), B=50, seed=1)
    assert ST.contrast_all(d, dr)["estimate"] == pytest.approx(d.mean())
    sc0 = ST.contrast_scarcity(d, dr, "0")
    assert sc0["estimate"] == pytest.approx(0.3 - 0.025)
    assert ST.contrast_reg(d, dr)["estimate"] == pytest.approx(0.125)
    assert not ST.contrast_scarcity(d, dr, "11-30")["computed"]


def test_protein_draw_order():
    f, p = ST.protein_draws(6, B=4, B_P=3, seed=9)
    rng = np.random.default_rng(9)
    assert np.array_equal(f, rng.integers(0, 6, size=(4, 6)))
    assert np.array_equal(p, rng.integers(0, 6, size=(3, 6)))


def test_holm_hand_family_with_tie_and_p1_slot():
    names = ["a", "b", "c", "d"]
    adj = ST.holm(names, {"a": 0.01, "b": 0.01, "c": 1.0, "d": 0.03})
    # sorted: a(0.01), b(0.01, table order), d(0.03), c(1) ; m = 4
    assert adj["a"] == pytest.approx(0.04)
    assert adj["b"] == pytest.approx(0.04)          # max(0.04, 3 * 0.01)
    assert adj["d"] == pytest.approx(0.06)
    assert adj["c"] == 1.0
    adj2 = ST.holm(names, {"a": 0.01, "b": None, "c": 0.02, "d": None})   # missing: m = 2
    assert adj2["a"] == pytest.approx(0.02) and adj2["c"] == pytest.approx(0.02) and adj2["b"] is None


def test_classification_rules():
    c = lambda e, lo, hi: {"estimate": e, "ci95": [lo, hi], "computed": True}  # noqa: E731
    assert ST.classify(c(-0.02, -0.03, -0.011), 0.01, 0.01) == "harm"
    assert ST.classify(c(0.02, 0.01, 0.03), 0.01, 0.01) == "supported"
    assert ST.classify(c(0.02, 0.01, 0.03), 0.01, 0.2) == "inconclusive"
    assert ST.classify(c(0.001, -0.005, 0.008), 0.01, 0.5) == "practically null"
    assert ST.classify(c(0.005, 0.001, 0.009), 0.01, 0.01) == "practically null"   # rule 3 first
    assert ST.classify(c(0.008, 0.001, 0.015), 0.01, 0.01) == "below floor"
    assert ST.classify(ST.null_contrast(), 0.01, 1.0) == ST.NULL_LABEL


def test_hf_zero_exactly_when_alpha_star_zero_and_typed_null():
    hf = ST.hf_or_null(0.0, {}, {}, np.zeros((2, 2), int))
    assert hf["estimate"] == 0.0 and hf["p"] == 1.0 and hf["label"] == ST.NULL_LABEL
    assert hf["reading"] == "not tested on test"
    dr = ST.term_draws(3, np.zeros(3, int), np.ones(3, bool), B=10, seed=0)
    dt = ST.d_typed(0.0, np.ones(3), np.zeros(3), dr)
    assert dt["estimate"] == 0.0 and dt["label"] == ST.NULL_LABEL
    assert ST.weighted_or_null(0.0, np.ones(3), dr)["label"] == ST.NULL_LABEL


def test_d_struct_and_d_typed_use_frozen_scorings():
    rng = np.random.default_rng(2)
    a, s = rng.random(40) + 0.1, rng.random(40)
    dr = ST.term_draws(40, np.zeros(40, int), np.r_[np.ones(10, bool), np.zeros(30, bool)], B=200, seed=3)
    ds = ST.d_struct(a, s, dr)
    assert ds["estimate"] == pytest.approx((a - s).mean())
    dt = ST.d_typed(0.5, a, s, dr)
    assert dt["estimate"] == pytest.approx((a - s)[:10].mean())


def test_fmax_contrast_protein_bootstrap():
    from go_function_fusion.metrics import protein_tables
    import scipy.sparse as sp
    rng = np.random.default_rng(4)
    Y = sp.csr_matrix(rng.random((30, 6)) < 0.4)
    P = rng.random((30, 6))
    t1, t2 = protein_tables(P, Y, np.ones(6)), protein_tables(P, Y, np.ones(6))
    D, _ = ST.protein_draws(t1["recall"].shape[1], B=20, B_P=2)
    c = ST.fmax_contrast(t1, t2, D)
    assert c["estimate"] == 0.0 and c["ci95"] == [0.0, 0.0]


def test_snapshot_rule(tmp_path):
    runs = tmp_path / "runs"
    assert ST.grid_snapshot(runs) == {"started_run_dirs": 0, "claim_files": 0}
    ok, why = ST.s2_remote_eligible({"started_run_dirs": 0, "claim_files": 0}, runs, 1e9)
    assert ok and why is None                                  # eligible fixture
    (runs / "bp" / "r1").mkdir(parents=True)
    (runs / "bp" / "r1" / "epochs.jsonl").write_text("{}\n")
    (runs / "bp" / "x.CLAIM").write_text("")
    assert ST.grid_snapshot(runs) == {"started_run_dirs": 1, "claim_files": 1}
    assert ST.s2_remote_eligible({"started_run_dirs": 1, "claim_files": 0}, runs, 0) == \
        (False, "grid_started_before_GF2")
    mt = os.path.getmtime(runs / "bp" / "r1" / "epochs.jsonl")
    assert ST.s2_remote_eligible({"started_run_dirs": 0, "claim_files": 0}, runs, mt + 10) == \
        (False, "mtime_check_failed")
    assert ST.s2_remote_eligible(None, runs, None) == (False, "no_GF2_snapshot")


def test_two_level_imports_trials_bootstrap():
    D = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]])
    s = ST.two_level(D, B=50, seed=1)
    assert s["point"] == pytest.approx(D.mean()) and "under-covers" in s["note"]
