import numpy as np
import pytest

from go_function import labels as L
from go_function.ontology import DgzRule, ancestors, parse_obo
from go_function.pins import canonical_json, npz_bytes

BP_VOCAB = ["GO:0000010", "GO:0000011", "GO:0000012", "GO:0000013", "GO:0000014",
            "GO:0000015", "GO:0000016", "GO:0000099"]
MF_VOCAB = ["GO:0005488", "GO:0005515", "GO:0000020"]


@pytest.fixture
def bp_ctx(mini_obo):
    return L.Context.build("bp", parse_obo(mini_obo), DgzRule(mini_obo), BP_VOCAB)


@pytest.fixture
def mf_ctx(mini_obo):
    return L.Context.build("mf", parse_obo(mini_obo), DgzRule(mini_obo), MF_VOCAB)


def _split(ctx, rows, regimes, prop=None):
    """rows: (name, exp, annotations); prop defaults to DGZ's own closure."""
    names = [r[0] for r in rows]
    exp = [r[1] for r in rows]
    ann = [r[2] for r in rows]
    if prop is None:
        prop = [sorted(ctx.dgz.prop(e)) for e in exp]
    return L.build_split(ctx, "test", names, exp, ann, prop, regimes)


BP_ROWS = [
    ("P1", ["GO:0000016"], ["GO:0000016|IDA", "GO:0000013|IEA"]),
    ("P2", ["GO:0000099"], ["GO:0000099|IMP"]),
    ("P3", ["GO:0000020", "GO:0000012"], ["GO:0000020|IDA", "GO:0000012|IPI"]),
]


def _cols(ctx, ids):
    return {ctx.vocab.index[g] for g in ids}


def test_vocab_duplicate_columns(bp_ctx):
    v = bp_ctx.vocab
    assert v.alt_cols == {7} and v.duplicate_cols == {7}
    assert v.primary[7] == "GO:0000011"
    assert v.primary_to_cols["GO:0000011"] == (1, 7)


def test_bp_regimes_and_checks_pass(bp_ctx):
    res = _split(bp_ctx, BP_ROWS, ("TP", "DGZ"))
    assert not any(res.failures.values())
    tp = res.regimes["TP"].labels.toarray()
    dgz = res.regimes["DGZ"].labels.toarray()
    idx = bp_ctx.vocab.index
    # P1: regulates not propagated under TP, propagated under DGZ
    assert {j for j in np.flatnonzero(tp[0])} == _cols(bp_ctx, ["GO:0000016", "GO:0000014",
                                                                "GO:0000015"])
    assert dgz[0, idx["GO:0000011"]] == 1 and dgz[0, idx["GO:0000010"]] == 1
    # P2: alt annotation -> TP has the primary and the duplicate column; DGZ lacks the primary
    assert tp[1, idx["GO:0000011"]] == 1 and tp[1, idx["GO:0000099"]] == 1
    assert dgz[1, idx["GO:0000011"]] == 0 and dgz[1, idx["GO:0000099"]] == 1
    # P3: the MF annotation is not in A_TP but is kept in A_DGZ (non-column extra)
    assert res.regimes["TP"].direct_extra[2] == ()
    assert res.regimes["DGZ"].direct_extra[2] == ("GO:0000020",)
    assert res.regimes["DGZ"].direct.toarray()[2, idx["GO:0000012"]] == 1
    cc = res.check_counts
    assert cc["dgz_excess"] == 2            # P1: GO:0000011, GO:0000010
    assert cc["tp_only_primalt"] == 1       # P2: GO:0000011
    assert cc["tp_only_altcols"] == 0


def test_check1_and_check2_trip(bp_ctx):
    bad_prop = [["GO:0000016"], sorted(bp_ctx.dgz.prop(["GO:0000099"])),
                sorted(bp_ctx.dgz.prop(["GO:0000020", "GO:0000012"]))]
    res = _split(bp_ctx, BP_ROWS, ("TP", "DGZ"), prop=bad_prop)
    assert res.failures["1_dgz_eq_prop"] == ["P1"]
    rows = [("P1", ["GO:0000016"], ["GO:0000016|IEA"])] + BP_ROWS[1:]
    res = _split(bp_ctx, rows, ("TP",))
    assert res.failures["2_exp_eq_codes"] == ["P1"]
    with pytest.raises(L.LabelCheckError):
        L.raise_on_failures({"test": res})


def _protein_sets(ctx, exp):
    a_tp = L.direct_tp(exp, ctx.ont, ctx.namespace)
    clo = L.tp_closure(a_tp, ctx.anc, ctx.root)
    row = L.tp_cols(clo, ctx.vocab)
    cols_b = L.tp_cols(L._bfs_closure_tp(ctx.ont, a_tp, ctx.namespace, ctx.root), ctx.vocab)
    clo_d = ctx.dgz.prop(exp)
    cols_d = L.dgz_cols(clo_d, ctx.vocab)
    prim_alt = {ctx.vocab.index[ctx.ont.alt_to_primary[a]] for a in exp
                if a in ctx.ont.alt_to_primary}
    return row, cols_b, cols_d, prim_alt, clo, clo_d


def test_check3c_trips_on_regulates_label_and_passes_on_alt(bp_ctx):
    idx = bp_ctx.vocab.index
    row, cols_b, cols_d, _, _, _ = _protein_sets(bp_ctx, ["GO:0000016"])
    assert L.check_3c(row, cols_d, cols_b) == set()
    planted = row | {idx["GO:0000011"]}           # reachable only through positively_regulates
    assert L.check_3c(planted, cols_d, cols_b) == {idx["GO:0000011"]}
    row2, cols_b2, cols_d2, _, clo2, clo_d2 = _protein_sets(bp_ctx, ["GO:0000099"])
    assert clo2 != clo_d2 - {"GO:0008150"}          # raw strings differ ...
    assert L.check_3c(row2, cols_d2, cols_b2) == set()  # ... but the column-level check passes


def test_check3d_trips_on_non_alt_tp_only_label(bp_ctx):
    idx = bp_ctx.vocab.index
    row, cols_b, cols_d, prim_alt, _, _ = _protein_sets(bp_ctx, ["GO:0000099"])
    assert L.check_3d(row, cols_b, cols_d, bp_ctx.vocab.alt_cols, prim_alt) == set()
    row1, cols_b1, cols_d1, prim_alt1, _, _ = _protein_sets(bp_ctx, ["GO:0000016"])
    planted = row1 | {idx["GO:0000012"]}
    assert L.check_3d(planted, cols_b1, cols_d1, bp_ctx.vocab.alt_cols, prim_alt1) == \
        {idx["GO:0000012"]}


def test_check3a(bp_ctx):
    idx = bp_ctx.vocab.index
    row, *_ = _protein_sets(bp_ctx, ["GO:0000012"])
    assert L.check_3a(row, bp_ctx.parent_cols) == set()
    assert L.check_3a(row - {idx["GO:0000013"]}, bp_ctx.parent_cols) == {idx["GO:0000013"]}


def test_bfs_closure_matches_ancestors(bp_ctx):
    ont = bp_ctx.ont
    anc = ancestors(ont)
    for t, term in ont.terms.items():
        ns = term.namespace
        root = "GO:0008150" if ns == "biological_process" else "GO:0003674"
        assert L._bfs_closure_tp(ont, [t], ns, root) == set(anc[t]) - {root}


MF_ROWS = [
    ("Q1", ["GO:0005515"], ["GO:0005515|IPI"]),
    ("Q2", ["GO:0000020"], ["GO:0000020|IDA"]),
    ("Q3", ["GO:0000020", "GO:0005515"], ["GO:0000020|IPI", "GO:0005515|IDA"]),
]


def test_mf_protein_binding_only_and_noipi_eval_set(mf_ctx):
    res = _split(mf_ctx, MF_ROWS, ("TP", "DGZ", "TPnoIPI"))
    assert not any(res.failures.values())
    tp, noipi = res.regimes["TP"], res.regimes["TPnoIPI"]
    assert mf_ctx.pb_only == {"GO:0005515", "GO:0005488"}
    assert tp.eval_rows.tolist() == [1, 2] and tp.protein_binding_only == 1
    # Q1 is IPI-only (empty noIPI closure); Q3 is protein-binding-only without IPI
    assert noipi.eval_rows.tolist() == [1] and noipi.protein_binding_only == 1
    assert noipi.closure_empty == 1
    assert noipi.labels.toarray()[0].sum() == 0
    assert res.regimes["DGZ"].eval_rows is None


def test_storage_roundtrip_and_manifest(bp_ctx, tmp_path):
    results = {s: _split(bp_ctx, BP_ROWS, ("TP", "DGZ")) for s in L.SPLITS}
    for s in L.SPLITS:
        results[s].split = s
    for r in ("TP", "DGZ"):
        arrays = L.regime_arrays(bp_ctx.vocab, results, r)
        blob = npz_bytes(arrays)
        assert blob == npz_bytes(L.regime_arrays(bp_ctx.vocab, results, r))
        p = tmp_path / f"labels_{r}.npz"
        p.write_bytes(blob)
        back = L.load_labels(p)
        assert back["vocab"] == BP_VOCAB
        assert (back["train"] != results["train"].regimes[r].labels).nnz == 0
        assert back["direct_extra_test"] == results["test"].regimes[r].direct_extra
        assert back["proteins_valid"] == ["P1", "P2", "P3"]
        assert ("eval_test" in back) == (r == "TP")
        man = L.regime_manifest(bp_ctx, results, r, "preprocess/go_function/bp/x.npz", "0",
                                {"dataset/go_function/dgz/go.obo": "1"})
        data = canonical_json(man)
        assert b"/Users/" not in data
        assert man["vocab"]["duplicate_columns"] == ["GO:0000099"]


@pytest.mark.slow
@pytest.mark.needs_dgz
@pytest.mark.parametrize("aspect", ["bp", "mf"])
def test_dgz_rule_matches_prop_annotations_real(aspect):
    import pandas as pd
    from go_function.runtime import ROOT
    dgz_dir = ROOT / "dataset" / "go_function" / "dgz"
    rule = DgzRule(dgz_dir / "go.obo")
    df = pd.read_pickle(dgz_dir / aspect / "train_data.pkl")
    rows = np.random.default_rng(20261011).choice(len(df), size=200, replace=False)
    for i in sorted(rows.tolist()):
        r = df.iloc[i]
        assert rule.prop(r["exp_annotations"]) == set(r["prop_annotations"]), r["proteins"]
