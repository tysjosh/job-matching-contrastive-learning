"""Neighbour sets (confirmatory vs pilot-compat), Sib, CoA / CoA⁺, permutation
(shuffled graph / shuffled Reg) control, DS parity with go_function.baselines."""
import numpy as np
import scipy.sparse as sp

from go_function_fusion import fuse as FU
from go_function_fusion import priors as PR

# columns: 0 root, 1 A(10), 2 B(11), 3 alt of B(99), 4 C(12), 5 D(13), 6 regB(14), 7 reg(15), 8 posregB(16)


def tables(m, mode):
    return PR.neighbour_tables(m["ont"], m["vocab"], m["anc"], m["desc"], m["ntr"], m["root"], mode)


def test_sets_hand_ontology(mini):
    t = tables(mini, "confirmatory")
    assert t["pa_tp"][4] == [1, 5]            # C: is_a A, part_of D
    assert t["pa_isa"][4] == [1]
    assert t["pa_tp"][6] == [7]               # regB: is_a biological regulation
    assert t["pa_any"][6] == [2, 7]           # + regulates B
    assert t["reg"][6] == [2] and t["reg"][8] == [2]
    assert t["pa_tp"][1] == []                # root removed
    assert t["anc"][2] == [1]
    assert t["desc"][1] == [2, 4]             # duplicate column 3 removed


def test_confirmatory_vs_pilot_compat(mini):
    c, p = tables(mini, "confirmatory"), tables(mini, "pilot")
    assert p["desc"][1] == [2, 3, 4] and c["desc"][1] == [2, 4]
    assert p["reg"][6] == [2, 3] and c["reg"][6] == [2]
    # own columns: the alt column 3 (primary B) never lists B's columns in confirmatory mode
    assert all(2 not in c[k][3] and 3 not in c[k][3] for k in c)
    ch = PR.changed_terms(c, p, range(9), ("desc", "reg"))
    assert ch["desc"] >= 1 and ch["reg"] >= 2


def test_sib_equals_isa_siblings(mini):
    from go_function.ontology import isa_siblings
    t = tables(mini, "confirmatory")
    sib = isa_siblings(mini["ont"], "GO:0000012", mini["anc"], mini["desc"])
    assert sib == frozenset({"GO:0000011"})
    assert t["sib"][4] == [2]                  # duplicate column 3 removed
    m2 = dict(mini, ntr=np.zeros(9))
    assert tables(m2, "confirmatory")["sib"][4] == []      # < MIN_SIB_POS train positives


def test_coannotation_excludes_relatives_coaplus_may_contain_parent(mini):
    # positives (rows): A {0,1,2,3}, B {0,1,2}, dup B same, C {3,4}, D {3,4,5}, bio-reg {0,1}
    Y = np.zeros((6, 9), dtype=np.uint8)
    Y[:, 0] = 1
    for c, rows in {1: [0, 1, 2, 3], 2: [0, 1, 2], 3: [0, 1, 2], 4: [3, 4], 5: [3, 4, 5],
                    7: [0, 1]}.items():
        Y[rows, c] = 1
    Y = sp.csr_matrix(Y)
    t = tables(mini, "confirmatory")
    args = (Y, [2], t["pa_tp"], mini["ont"], mini["vocab"], mini["anc"], mini["desc"], 0,
            "biological_process")
    coap, share = PR.coannotation_sets(*args, plus=True)
    coa, _ = PR.coannotation_sets(*args, plus=False)
    assert coap[2] == [1] and share[2] == 1.0         # top Jaccard is the parent A (3/4)
    assert coa[2] == [7]                              # next non-relative (2/3)
    Y0 = sp.csr_matrix(Y.toarray() * (np.arange(9) != 2)[None, :])
    coa0, _ = PR.coannotation_sets(Y0, [2], *args[2:], plus=False)
    assert coa0[2] == []                              # n_train(t) = 0 -> empty


def test_shuffled_graph_permutation_reproducible_and_within_deciles():
    ntr = np.r_[np.zeros(30), np.arange(70) * 10]
    p1, p2 = PR.decile_permutation(ntr), PR.decile_permutation(ntr)
    assert np.array_equal(p1, p2) and sorted(p1) == list(range(100))
    dec = np.searchsorted(np.quantile(ntr, np.linspace(0, 1, 11)[1:-1]), ntr)
    assert np.array_equal(dec[p1], dec)
    assert not np.array_equal(p1, np.arange(100))
    assert not np.array_equal(p1, PR.decile_permutation(ntr, seed=1))
    sets = [[1, 2], [], [99]]
    assert PR.permute_sets(sets, p1) == [[int(p1[1]), int(p1[2])], [], [int(p1[99])]]


def test_shuffled_graph_control_breaks_a_real_parent_signal():
    rng = np.random.default_rng(1)
    n, V = 400, 60
    parent = rng.normal(size=n)
    L = rng.normal(size=(n, V))
    L[:, :30] = parent[:, None] + 0.3 * rng.normal(size=(n, 30))
    Y = (parent + 0.5 * rng.normal(size=n))[:, None] > 0.5
    L[:, 30] = rng.normal(size=n)                      # the child: uninformative logit
    sets = [[] for _ in range(V)]
    sets[30] = [0, 1, 2]
    perm = np.arange(V)
    perm[:3] = [40, 41, 42]                            # parents mapped to noise columns
    shuf = PR.permute_sets(sets, perm)
    from go_function_fusion.metrics import rank_auc
    C = np.array([30])
    w = np.array([2.0])
    real = rank_auc(FU.ap_t(FU.z_cols(L[:, C]), FU.prior_z(L, sets, C), w), Y)[0]
    fake = rank_auc(FU.ap_t(FU.z_cols(L[:, C]), FU.prior_z(L, shuf, C), w), Y)[0]
    base = rank_auc(L[:, C], Y)[0]
    assert real > base + 0.1 and abs(fake - base) < 0.1 and real > fake


def test_permutation_report(mini):
    perm = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8])
    sets = [[], [0], [1], [], [3], [], [], [], []]
    r = PR.permutation_report(sets, perm, [1, 2, 4], mini["vocab"], 0)
    assert r == {"pairs": 3, "perm_is_root": 1, "perm_is_t": 1, "perm_is_alt_id": 1}


def test_ds_parity_with_go_function_baselines(tmp_path):
    from go_function import baselines as BL
    f = tmp_path / "d.res"
    f.write_text("q1\th1\t10\t50\nq1\th2\t30\t60\nq1\tzz\t99\t1\nq2\th2\t5\t40\n")
    y = sp.csr_matrix(np.array([[1, 0], [1, 1]], dtype=np.uint8))
    H = PR.diamond_matrix(f, ["q1", "q2", "q3"], ["h1", "h2"])
    ref = BL.diamond_score(BL.diamond_matrix(BL.read_diamond(f), ["q1", "q2", "q3"], ["h1", "h2"]),
                           y, [0, 1])
    got = PR.ds_scores(H, y, [0, 1])
    assert np.array_equal(got, ref) and np.allclose(got, [[1.0, 0.75], [1.0, 1.0], [0.0, 0.0]])
    perm = PR.shuffled_train_rows(2)
    assert np.array_equal(perm, np.random.default_rng(20261017).permutation(2))
    assert np.array_equal(PR.ds_scores(H, y, [0, 1], shuffled=True),
                          BL.diamond_score(H, sp.csr_matrix(y)[perm], [0, 1]))


def test_train_only_checks(tmp_path):
    import pytest
    f = tmp_path / "v.res"
    f.write_text("q1\th1\t10\t50\n")
    assert PR.check_train_only(f, ["q1"], ["h1"]) == {"n_hits": 1, "n_queries": 1}
    with pytest.raises(ValueError):
        PR.check_train_only(f, ["q1"], ["h2"])
    with pytest.raises(ValueError):
        PR.check_train_only(f, ["h1"], ["h1"])


def test_text_prior_is_out_of_scope():
    import pytest
    with pytest.raises(NotImplementedError):
        PR.text_neighbour_sets()
