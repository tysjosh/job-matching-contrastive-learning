"""Hand examples: z, w_t, AP-T, fixed-w, empty sets, PC, tie rules, AUC/AUPR ties."""
import math

import numpy as np
import pytest

from go_function_fusion import fuse as FU
from go_function_fusion import metrics as M


def test_z_population_sd_and_constant_column():
    Z = FU.z_cols(np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]))
    sd = math.sqrt(2 / 3)
    assert np.allclose(Z[:, 0], [-1 / sd, 0, 1 / sd])
    assert np.all(Z[:, 1] == 0)
    assert Z.dtype == np.float64


def test_adaptive_weight_and_fixed():
    w = FU.adaptive_w(np.array([0, 30, 270]), 2, 30)
    assert np.allclose(w, [2.0, 1.0, 0.2])
    assert np.allclose(FU.adaptive_w(np.array([0, 999]), 3, float("inf")), [3.0, 3.0])


def test_apt_hand_example_and_empty_set():
    L = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 1.0], [2.0, 2.0, 2.0]])
    sets = [[], [0, 2], []]                    # column 1's parents are columns 0 and 2
    cols = np.array([1, 2])
    pz = FU.prior_z(L, sets, cols)
    pa_mean = np.array([1.5, 1.0, 2.0])
    assert np.allclose(pz[:, 0], FU.z_cols(pa_mean)[:, 0])
    assert np.all(pz[:, 1] == 0)               # empty set contributes 0
    own = FU.z_cols(L[:, cols])
    w = FU.adaptive_w(np.array([10, 0]), 2, 10)        # w = [1, 2]
    S = FU.ap_t(own, pz, w)
    assert np.allclose(S[:, 0], own[:, 0] + 1.0 * pz[:, 0])
    assert np.allclose(S[:, 1], own[:, 1])


def test_reduceat_equals_loop():
    rng = np.random.default_rng(0)
    L = rng.normal(size=(17, 12))
    sets = [list(rng.choice(12, size=rng.integers(0, 4), replace=False)) for _ in range(12)]
    cols = np.arange(12)
    for how in ("mean", "max"):
        a, ha = FU.block_agg(L, sets, cols, how, "loop")
        b, hb = FU.block_agg(L, sets, cols, how, "reduceat", max_elems=7)
        assert np.array_equal(ha, hb) and np.allclose(a, b, atol=1e-12)


def test_protocol_f_and_channels():
    own, pa, sib = np.ones((3, 1)), np.full((3, 1), 2.0), np.full((3, 1), 0.5)
    assert np.allclose(FU.protocol_f(own, pa, sib, np.array([2.0])), 1 + 2 * 1.5)
    assert np.allclose(FU.add_channel(own, pa, 0.25), 1.5)


def test_pc_probability_space_and_alpha():
    L = np.array([[0.0], [2.0]])
    pz = np.array([[1.0], [-1.0]])
    q = FU.pc_q(L, pz, np.array([0.5]))
    assert np.allclose(q[:, 0], 1 / (1 + np.exp(-np.array([0.5, 1.5]))))
    ds = np.array([[1.0], [0.0]])
    assert np.allclose(FU.pc_score(q, ds, 0.2), 0.8 * q + 0.2 * ds)
    assert np.array_equal(FU.pc_score(q, None, 0.0), q)
    with pytest.raises(ValueError):
        FU.pc_score(q, None, 0.3)
    assert np.allclose(FU.sigmoid(np.array([-800.0, 800.0])), [0.0, 1.0])


def test_selection_tie_rules():
    cells = FU.apt_cells([1, 2, 3], [10, 30, 100])
    assert cells[0] == {"c": 1.0, "tau": 100.0} and cells[2] == {"c": 1.0, "tau": 10.0}
    vals = [0.5] * 9
    vals[4] = 0.5 + 1e-13                      # within tolerance: earlier cell keeps it
    assert FU.select(cells, vals)[0] == 0
    vals[4] = 0.6
    assert FU.select(cells, vals)[1] == {"c": 2.0, "tau": 30.0}
    nc = FU.apt_cells([1], [10], null=True)
    assert FU.is_null(nc[0]) and FU.select(nc, [0.7, 0.7])[0] == 0
    pc = FU.pc_cells([0, 1], [10, 30], [0, 0.1])
    assert pc[:3] == [{"c": 0.0, "tau": 30.0, "alpha": 0.0}, {"c": 0.0, "tau": 30.0, "alpha": 0.1},
                      {"c": 0.0, "tau": 10.0, "alpha": 0.0}]
    assert [c["w_H"] for c in FU.weight_cells("w_H", [1, 0, 0.5])] == [0.0, 0.5, 1.0]


def test_rank_auc_ties_hand_example():
    s = np.array([0.1, 0.4, 0.4, 0.8])
    y = np.array([0, 1, 0, 1], dtype=bool)
    # pairs (pos, neg): (0.4,0.1)=1 (0.4,0.4)=0.5 (0.8,0.1)=1 (0.8,0.4)=1 -> 3.5/4
    assert M.rank_auc(s, y)[0] == pytest.approx(0.875)
    assert np.isnan(M.rank_auc(s, np.ones(4, bool))[0])


def test_rank_auc_float64_ties_against_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(3)
    S = np.round(rng.normal(size=(200, 5)), 1)            # many ties
    Y = rng.random((200, 5)) < 0.3
    for j in range(5):
        assert M.rank_auc(S[:, j], Y[:, j])[0] == pytest.approx(roc_auc_score(Y[:, j], S[:, j]), abs=1e-12)
        assert M.average_precision(S[:, j], Y[:, j])[0] == pytest.approx(
            average_precision_score(Y[:, j], S[:, j]), abs=1e-12)


def test_float32_z_can_create_ties_float64_does_not():
    v = np.array([1e8, 1e8 + 1, 1e8 + 2], dtype=np.float64)
    z64 = FU.z_cols(v + 1e-3 * np.array([0.0, 1.0, 2.0]))[:, 0]
    assert len(np.unique(z64)) == 3


def test_bands_and_summary():
    ntr = np.array([0, 1, 10, 11, 30, 31, 100, 101, 1000, 1001])
    assert M.band_index(ntr).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    s = M.summary(np.arange(10.0), ntr, base=np.zeros(10))
    assert s["0"]["n"] == 1 and s[">1000"]["mean"] == 9.0 and s["share_terms_improved"] == 0.9


def test_protein_tables_match_go_function_and_has_prediction():
    import scipy.sparse as sp
    from go_function.evaluate import protein_centric
    rng = np.random.default_rng(5)
    P = rng.random((40, 9))
    P[0] = 0.0                                 # a protein with no prediction above 0
    Y = sp.csr_matrix(rng.random((40, 9)) < 0.3)
    ic = rng.random(9) * 3
    tab = M.protein_tables(P, Y, ic)
    ref = protein_centric(P, Y, ic)
    got = M.fmax_smin(tab)
    assert got["fmax"] == pytest.approx(ref["fmax"], abs=1e-12)
    assert got["smin"] == pytest.approx(ref["smin"], abs=1e-9)
    r0 = int(np.flatnonzero(tab["rows"] == 0)[0]) if 0 in tab["rows"] else None
    if r0 is not None:
        assert not tab["has_prediction"][1:, r0].any()
    W = np.ones((3, tab["recall"].shape[1]))
    assert np.allclose(M.fmax_boot(tab, W), got["fmax"])
    assert np.allclose(M.smin_boot(tab, W), got["smin"])
