"""Evaluator tests (design §8, plan item 5)."""
import math

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

from go_function import evaluate as E
from go_function import labels as L
from go_function.ontology import DgzRule, ancestors, parse_obo
from go_function.runtime import FatalError, ProtocolStop

TERMS = ["A", "B", "C"]
IC = {"A": 1.0, "B": 2.0, "C": 0.5}
PREDS = np.array([[0.9, 0.4, 0.1],
                  [0.2, 0.0, 0.7]])
LABELS = [{"A", "B"}, {"B", "C"}]


def _dgz_reference(preds, terms, labels, go_set, ic):
    """Literal port of DGZ evaluate.py main loop + evaluate_annotations (Python sets)."""
    precisions, recalls = [], []
    fmax, smin = 0.0, 1000000.0
    labels = [set(g for g in x if g in go_set) for x in labels]
    for t in range(0, 101):
        thr = t / 100.0
        pred = [set(g for j, g in enumerate(terms) if float(preds[i][j]) >= thr and g in go_set)
                for i in range(len(labels))]
        total = 0; p = r = ru = mi = 0.0; p_total = 0
        for i in range(len(labels)):
            if len(labels[i]) == 0:
                continue
            tp = labels[i] & pred[i]
            fp = pred[i] - tp
            fn = labels[i] - tp
            mi += sum(ic.get(g, 0.0) for g in fp)
            ru += sum(ic.get(g, 0.0) for g in fn)
            total += 1
            r += len(tp) / (len(tp) + len(fn))
            if len(pred[i]) > 0:
                p_total += 1
                p += len(tp) / (len(tp) + len(fp))
        ru /= total; mi /= total; r /= total
        if p_total > 0:
            p /= p_total
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        precisions.append(p); recalls.append(r)
        fmax = max(fmax, f)
        smin = min(smin, math.sqrt(ru * ru + mi * mi))
    precisions, recalls = np.array(precisions), np.array(recalls)
    idx = np.argsort(recalls)
    return fmax, smin, float(np.trapezoid(precisions[idx], recalls[idx]))


# ------------------------------------------------------------------ class-centric
def test_class_metrics_equal_sklearn():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    y = rng.random(200) < 0.2
    m = E.class_metrics(s, y)
    assert m["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert m["aupr"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert m["n_pos"] == int(y.sum()) and m["n"] == 200


@pytest.mark.parametrize("y", [np.zeros(5), np.ones(5)])
def test_class_metrics_degenerate_is_protocol_stop(y):
    with pytest.raises(ProtocolStop):
        E.class_metrics(np.arange(5.0), y)


# ------------------------------------------------------------------ protein-centric toy
def test_dgz_fmax_smin_hand_toy():
    m = E.dgz_protein_metrics(PREDS, TERMS, LABELS, set(TERMS), IC)
    # hand computed: best F at t=0.21 (P=1, R=0.75); Smin at t=0 (ru=0, mi=0.75)
    assert m["fmax"] == pytest.approx(2 * 0.75 / 1.75, abs=1e-12)
    assert m["tmax"] == pytest.approx(0.21)
    assert m["smin"] == pytest.approx(0.75, abs=1e-12)
    f, s, a = _dgz_reference(PREDS, TERMS, LABELS, set(TERMS), IC)
    assert (m["fmax"], m["smin"], m["aupr"]) == pytest.approx((f, s, a), abs=1e-12)


def test_dgz_mode_matches_reference_random_with_outside_labels():
    rng = np.random.default_rng(1)
    terms = [f"T{i}" for i in range(12)]
    go_set = set(terms[:10]) | {"X1", "X2"}          # T10, T11 outside the namespace
    preds = rng.random((30, 12)).round(3)
    labels = [set(rng.choice(terms + ["X1", "X2", "Y"], size=rng.integers(0, 5), replace=False))
              for _ in range(30)]
    ic = {g: float(rng.random() * 3) for g in terms + ["X1", "X2"]}
    m = E.dgz_protein_metrics(preds, terms, labels, go_set, ic)
    f, s, a = _dgz_reference(preds, terms, labels, go_set, ic)
    assert (m["fmax"], m["smin"], m["aupr"]) == pytest.approx((f, s, a), abs=1e-10)
    assert m["n_skipped_no_label"] == sum(1 for x in labels if not (x & go_set))


def test_study_protein_centric_matches_dgz_mode_without_extras():
    y = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8))
    probs = np.vstack([PREDS, [[0.5, 0.5, 0.5]]])
    ic = np.array([IC[t] for t in TERMS])
    m = E.protein_centric(probs, y, ic)
    ref = E.dgz_protein_metrics(PREDS, TERMS, LABELS, set(TERMS), IC)
    assert m["fmax"] == pytest.approx(ref["fmax"]) and m["smin"] == pytest.approx(ref["smin"])
    assert m["aupr_protein"] == pytest.approx(ref["aupr"])
    assert m["n_eval"] == 2 and m["n_skipped_no_label"] == 1
    # micro AUPR: at t=0 micro precision 4/6, micro recall 1
    assert 0 < m["aupr_micro"] <= 1


def test_max_propagation():
    # columns: 0 parent of 1 and 2; 1 parent of 2 (transitive pairs given explicitly)
    src = np.array([1, 2, 2]); dst = np.array([0, 0, 1])
    probs = np.array([[0.1, 0.3, 0.8], [0.9, 0.2, 0.1]], dtype=np.float32)
    out = E.propagate_max(probs, src, dst).numpy()
    np.testing.assert_allclose(out, [[0.8, 0.8, 0.8], [0.9, 0.2, 0.1]])
    out2 = E.propagate_max(probs, src, dst, max_elems=3).numpy()   # one row per chunk
    np.testing.assert_allclose(out, out2)


def test_tp_ancestor_pairs_and_propagation_in_protein_centric(mini_obo):
    ont = parse_obo(mini_obo)
    ctx = L.Context.build("bp", ont, DgzRule(mini_obo), ["GO:0000010", "GO:0000011",
                                                          "GO:0000012", "GO:0000014"])
    terms = [0, 1, 2, 3]
    src, dst = E.tp_ancestor_pairs(terms, ctx.vocab, ancestors(ont))
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert pairs == {(1, 0), (2, 0)}        # B, C is_a A; regulation of B not propagated
    probs = np.array([[0.1, 0.6, 0.2, 0.3]])
    y = sp.csr_matrix(np.array([[1, 1, 0, 0]], dtype=np.uint8))
    m = E.protein_centric(probs, y, np.ones(4), propagate=(src, dst))
    assert m["fmax"] == pytest.approx(1.0)   # A lifted to 0.6 -> {A, B} at t in (0.3, 0.6]


def test_ic_from_labels_dgz_formula():
    # col 0 root-like (no parents), col 1 child of 0, col 2 child of 1
    y = sp.csr_matrix(np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=np.uint8))
    ic = E.ic_from_labels(y, [(), (0,), (1,)])
    np.testing.assert_allclose(ic, [0.0, math.log2(4 / 2), math.log2(2 / 1)])


def test_dgz_ic_and_namespace_terms(mini_obo):
    dgz = DgzRule(mini_obo)
    annots = [dgz.prop(["GO:0000012"]), dgz.prop(["GO:0000011"])]
    ic = E.dgz_ic(dgz, annots)
    # GO:0000012 parents (with_rels): GO:0000010 (count 2), GO:0000013 (count 1)
    assert ic["GO:0000012"] == pytest.approx(0.0)
    assert ic["GO:0000010"] == pytest.approx(math.log2(2 / 2))
    gs = E.dgz_namespace_terms(dgz, "bp")
    assert "GO:0008150" not in gs and "GO:0000099" in gs and "GO:0005515" not in gs


def test_dgz_class_auc_equals_mean_sklearn():
    rng = np.random.default_rng(2)
    terms = ["A", "B", "C", "D"]
    preds = rng.random((40, 4))
    prop = [set(rng.choice(["A", "B", "C", "Z"], size=rng.integers(0, 3), replace=False))
            for _ in range(40)]
    prop[0].add("A")
    res = E.dgz_class_auc(preds, terms, prop)
    y = np.array([[g in x for g in terms] for x in prop])
    expect = [roc_auc_score(y[:, j], preds[:, j].astype(np.float32)) for j in range(4)
              if 0 < y[:, j].sum() < 40]
    assert res["n_terms"] == len(expect)            # D never positive -> skipped
    assert res["auc"] == pytest.approx(np.mean(expect), abs=1e-12)


# ------------------------------------------------------------------ regime mapping
def test_eval_label_source_mapping():
    assert E.eval_label_source("TPnoIPI", "mf").label_file.endswith("mf/labels_TPnoIPI.npz")
    for r in ("TP", "DGZ"):
        for a in ("bp", "mf"):
            src = E.eval_label_source(r, a)
            assert src.label_file == f"preprocess/go_function/{a}/labels_TP.npz"
            assert src.eval_set == "eval_test"


def test_noipi_never_gets_tp_labels():
    src = E.eval_label_source("TPnoIPI", "mf")
    assert "labels_TP.npz" not in src.label_file and "TPnoIPI" in src.label_file
    with pytest.raises(FatalError):
        E.eval_label_source("TPnoIPI", "bp")
    with pytest.raises(FatalError):
        E.eval_label_source("noIPI", "mf")


def test_collateral_terms(mini_obo):
    ont = parse_obo(mini_obo)
    ids = ["GO:0000010", "GO:0000011", "GO:0000012", "GO:0000099", "GO:0008150", "GO:0000013"]
    vocab = L.build_vocab(ids, ont)
    train = sp.csr_matrix(np.array([[1, 1, 1, 1, 1, 0]], dtype=np.uint8))
    got = E.collateral_terms(vocab, train, excluded_cols=[2], root="GO:0008150")
    # 3 is a duplicate column, 4 the root, 5 has no train positive, 2 excluded
    assert got.tolist() == [0, 1]
