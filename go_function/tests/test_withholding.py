"""Closed-world withholding (design §5 v4) on a synthetic 30-protein BP dataset.

Targets: t1 = GO:0000016 "positive regulation of process B" (X = GO:0000011 is its
DGZ-only filler) and t2 = GO:0000013 "process D" (GO:0000012 part_of it; the MF term
GO:0000020 is part_of it across namespaces). GO:0000012 is not a vocabulary column,
so its annotations are non-column direct annotations implying t2.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from go_function import labels as L
from go_function import targets as T
from go_function import withholding as WH
from go_function.ontology import DgzRule, ancestors, parse_obo
from go_function.pins import csr_sha256, write_or_verify

VOCAB = ["GO:0000010", "GO:0000011", "GO:0000013", "GO:0000014", "GO:0000015",
         "GO:0000016", "GO:0000099"]
T1, T2, X = "GO:0000016", "GO:0000013", "GO:0000011"

GROUPS = [(4, ["GO:0000016", "GO:0000012"]),   # P00-P03: t1 and t2 (multiple inheritance)
          (6, ["GO:0000016"]),                 # P04-P09: t1
          (4, ["GO:0000012"]),                 # P10-P13: t2 via a non-column annotation
          (2, ["GO:0000013"]),                 # P14-P15: t2 directly
          (2, ["GO:0000020", "GO:0000011"]),   # P16-P17: t2 under DGZ only (MF part_of)
          (4, ["GO:0000011"]),
          (2, ["GO:0000099"]),                 # alt id of GO:0000011
          (4, ["GO:0000010"]),
          (2, ["GO:0000015"])]


class Fixture:
    def __init__(self, obo):
        self.ont = parse_obo(obo)
        self.dgz = DgzRule(obo)
        self.anc = ancestors(self.ont)
        self.ctx = L.Context.build("bp", self.ont, self.dgz, VOCAB)
        self.vocab = self.ctx.vocab
        exp = [ids for n, ids in GROUPS for _ in range(n)]
        self.names = [f"P{i:02d}" for i in range(len(exp))]
        ann = [[f"{g}|IDA" for g in e] for e in exp]
        prop = [sorted(self.dgz.prop(e)) for e in exp]
        res = L.build_split(self.ctx, "train", self.names, exp, ann, prop, ("TP", "DGZ"))
        assert not any(res.failures.values())
        self.rd = {r: WH.RegimeData(regime=r, vocab=self.vocab, train=res.regimes[r].labels,
                                    direct=res.regimes[r].direct,
                                    extra=res.regimes[r].direct_extra, proteins=self.names)
                   for r in ("TP", "DGZ")}
        self.prop = {"TP": WH.Propagator("TP", self.vocab, "bp", self.anc),
                     "DGZ": WH.Propagator("DGZ", self.vocab, "bp", self.anc, self.dgz)}
        self.col = self.vocab.index

    def info(self, t):
        rows = WH.column_positive_rows(self.rd["TP"].train, self.col[t])
        return WH.TargetInfo(id=t, column=self.col[t],
                             draw1=tuple(T.retention_order(rows, self.names, t, 20261011)),
                             draw2=tuple(T.retention_order(rows, self.names, t, 20261012)))

    def P(self, regime, ts):
        return {t: WH.column_positive_rows(self.rd[regime].train, self.col[t]) for t in ts}

    def cond(self, regime, rule, k, targets=(T1, T2), draw=1):
        spec = WH.Spec("bp", regime, rule, None if rule == "all" else "p01",
                       "all" if rule == "all" else k, draw)
        return WH.make_condition(spec, self.rd[regime], self.prop[regime],
                                 [self.info(t) for t in targets],
                                 P_tp=self.P("TP", targets), P_dgz=self.P("DGZ", targets))

    def check(self, c, w2_sha=None):
        return WH.run_assertions(c.check, self.vocab, base_labels_sha=csr_sha256(
            self.rd[c.spec.regime].train), w2_labels_sha=w2_sha)


@pytest.fixture
def fx(mini_obo):
    return Fixture(mini_obo)


def _dense(m):
    return np.asarray(sp.csr_matrix(m).todense())


def _set(m, r, c, v):
    lil = sp.lil_matrix(m)
    lil[r, c] = v
    return sp.csr_matrix(lil, dtype=np.uint8)


def test_fixture_sets(fx):
    assert len(fx.names) == 30
    P_tp, P_dgz = fx.P("TP", [T2]), fx.P("DGZ", [T2])
    assert set(P_tp[T2].tolist()) < set(P_dgz[T2].tolist())          # MF part_of adds P16-17
    assert fx.rd["TP"].extra[10] == ("GO:0000012",)
    assert "GO:0000020" in fx.rd["DGZ"].extra[16]


@pytest.mark.parametrize("regime,rule,k", [
    ("TP", "W2", 0), ("TP", "W2", 1), ("TP", "W2", 5), ("DGZ", "W2", 0), ("DGZ", "W2", 5),
    ("TP", "W1", 0), ("TP", "W1", 1), ("TP", "W1", 5), ("DGZ", "WDGZrel", 0),
    ("TP", "all", None), ("DGZ", "all", None)])
def test_every_assertion_holds(fx, regime, rule, k):
    c = fx.cond(regime, rule, k)
    done = fx.check(c)
    assert 10 in done and 5 in done
    if rule in ("W1", "W2"):
        assert {1, 2, 8, 9}.issubset(done)
        assert len(c.mask) == 0


def test_w2drop_assertions_and_mask(fx):
    w2 = fx.cond("DGZ", "W2", 0)
    c = fx.cond("DGZ", "W2drop", 0)
    assert 7 in fx.check(c, w2_sha=csr_sha256(w2.labels))
    assert csr_sha256(c.labels) == csr_sha256(w2.labels)
    assert len(c.mask) == 30 * 2 and set(c.mask[:, 1].tolist()) == {fx.col[T1], fx.col[T2]}


def test_closed_world_w2_mask_empty_and_only_erased_entries_differ(fx):
    for regime in ("TP", "DGZ"):
        c = fx.cond(regime, "W2", 0)
        assert c.mask.shape == (0, 2)
        base = fx.rd[regime].train
        diff = sp.csr_matrix(base, dtype=np.int16) - sp.csr_matrix(c.labels, dtype=np.int16)
        assert (diff.data == -1).sum() == 0                  # nothing added
        coo = sp.coo_matrix(diff)
        W_all = set().union(*(set(w.tolist()) for w in c.check.W.values()))
        assert set(coo.row.tolist()) <= W_all                 # only withheld proteins differ
        # every removed entry is exactly an entry lost by the erasure + re-propagation
        for p in sorted(W_all):
            ann = fx.rd[regime].annotations(p)
            ts = [t for t in (T1, T2) if p in set(c.check.W[t].tolist())]
            kept = [a for a in ann if not any(fx.prop[regime].implies(a, t) for t in ts)]
            erased = fx.rd[regime].row(p) - fx.prop[regime].cols(kept)
            assert set(coo.col[coo.row == p].tolist()) == erased


def test_wdgzrel_keeps_filler_and_w2_erases_it(fx):
    x = fx.col[X]
    rel = fx.cond("DGZ", "WDGZrel", 0)
    w2 = fx.cond("DGZ", "W2", 0)
    t1_pos = fx.P("DGZ", [T1])[T1]
    assert _dense(rel.labels)[t1_pos, x].all()               # the leak channel is present
    assert not _dense(w2.labels)[t1_pos, x].any()            # erased by re-propagation ...
    mk = set(map(tuple, w2.mask.tolist()))
    assert not any((int(p), x) in mk for p in t1_pos)        # ... as unmasked 0 labels
    # WDGZrel mask = target columns on every train row
    assert len(rel.mask) == 60


def test_leaked_condition_trips_assertion_1(fx):
    c = fx.cond("TP", "W2", 0)
    p = int(c.check.W[T1][0])
    c.check.labels = _set(c.check.labels, p, fx.col[T1], 1)             # left one t1 positive
    with pytest.raises(WH.WithholdingAssertion, match="assertion 1"):
        WH.assertion_1(c.check)
    with pytest.raises(WH.WithholdingAssertion, match="assertion 2"):
        WH.assertion_2(c.check)


def test_row_selective_mask_trips_assertion_10(fx):
    c = fx.cond("TP", "W2", 0)
    rows = c.check.W[T1]
    c.check.mask = np.stack([rows, np.full(len(rows), fx.col[T1])], axis=1).astype(np.int32)
    with pytest.raises(WH.WithholdingAssertion, match="assertion 10"):
        WH.assertion_10(c.check)
    d = fx.cond("DGZ", "W2drop", 0)
    d.check.mask = d.check.mask[1:]                          # one train row unmasked
    with pytest.raises(WH.WithholdingAssertion, match="assertion 10"):
        WH.assertion_10(d.check)


def test_cross_namespace_part_of_erased_under_dgz(fx):
    c = fx.cond("DGZ", "W2", 0, targets=(T2,))
    t2 = fx.col[T2]
    assert not _dense(c.labels)[[16, 17], t2].any()
    fx.check(c)
    # planted aspect-filtered erasure: only BP annotations are tested and erased
    new = {}
    for p in c.check.W[T2].tolist():
        ann = fx.rd["DGZ"].annotations(p)
        kept = [a for a in ann if not (fx.ont.terms[a].namespace == "biological_process"
                                       and fx.prop["DGZ"].implies(a, T2))]
        new[p] = fx.prop["DGZ"].cols(kept)
    c.check.labels = WH.replace_rows(fx.rd["DGZ"].train, new)
    with pytest.raises(WH.WithholdingAssertion, match="assertion [12]"):
        WH.assertion_1(c.check)
        WH.assertion_2(c.check)


def test_alt_id_target_trips_assertion_8(fx):
    d = "GO:0000011"                                         # has alt column GO:0000099
    assert "GO:0000011" not in fx.dgz.closure("GO:0000099")
    assert "GO:0000010" in fx.dgz.closure("GO:0000099")
    c = fx.cond("DGZ", "WDGZrel", 0, targets=(d,))
    c.check.mask = WH.full_column_mask(30, [fx.col[d]])      # only column t masked
    with pytest.raises(WH.WithholdingAssertion, match="assertion 8"):
        WH.assertion_8(c.check, fx.vocab)


def test_alt_id_descendant_erased_under_dgz_w2(fx):
    t = "GO:0000010"
    assert fx.prop["DGZ"].implies("GO:0000099", t)           # d' in D*^GO_DGZ(t)
    c = fx.cond("DGZ", "W2", 0, targets=(t,))
    m = _dense(c.labels)
    assert not m[[22, 23], fx.col["GO:0000099"]].any() and not m[[22, 23], fx.col[t]].any()


def test_non_column_annotation_implying_t_is_erased(fx):
    c = fx.cond("TP", "W2", 0, targets=(T2,))
    m = _dense(c.labels)
    assert not m[10:14].any()                                # GO:0000012 erased: rows empty
    # P00-P03 keep t1's labels; GO:0000010 was implied only through GO:0000012 (Lost)
    assert m[0, fx.col[T1]] == 1 and m[0, fx.col["GO:0000010"]] == 0
    assert len(c.mask) == 0


def test_lost_entries_multiple_inheritance_are_unmasked_zeros(fx):
    c = fx.cond("TP", "W2", 0, targets=(T1,))
    m = _dense(c.labels)
    for col in ("GO:0000014", "GO:0000015", T1):
        assert m[0:10, fx.col[col]].sum() == 0
    assert m[0:4, fx.col["GO:0000010"]].all() and m[0:4, fx.col[T2]].all()   # via GO:0000012
    assert c.stats["n_lost_entries_unmasked"] > 0 and len(c.mask) == 0


def test_multi_target_erasure_before_one_repropagation(fx):
    c = fx.cond("TP", "W2", 0)
    assert not _dense(c.labels)[0:4].any()                   # both annotations erased


def test_w1_sets_strict_ancestors_without_mask(fx):
    c = fx.cond("TP", "W1", 0, targets=(T1,))
    m = _dense(c.labels)
    assert m[4:10, fx.col["GO:0000014"]].all() and m[4:10, fx.col["GO:0000015"]].all()
    assert not m[4:10, fx.col[T1]].any() and len(c.mask) == 0


def test_draw2_differs_and_is_nested(fx):
    i = fx.info(T1)
    assert i.draw1 != i.draw2 and sorted(i.draw1) == sorted(i.draw2)
    c1 = fx.cond("TP", "W2", 1, draw=2)
    c5 = fx.cond("TP", "W2", 5, draw=2)
    assert set(c1.check.R[T1]) < set(c5.check.R[T1])
    assert c5.check.R[T1] == list(i.draw2[:5])
    fx.check(c5)


def test_spec_validation():
    with pytest.raises(WH.FatalError):
        WH.Spec("bp", "TP", "WDGZrel", "p01", 0)
    with pytest.raises(WH.FatalError):
        WH.Spec("bp", "DGZ", "W2drop", "p01", 1)
    with pytest.raises(WH.FatalError):
        WH.Spec("bp", "TP", "W1", "p01", 1, draw=2)
    assert WH.Spec("bp", "TP", "W2", "p03", 5, 2).name == "TP/W2/p03_k5_d2"
    assert WH.Spec("bp", "DGZ", "all").name == "DGZ/all"


def test_files_roundtrip_and_assertion_4(fx, tmp_path):
    c = fx.cond("DGZ", "W2drop", 0)
    base = fx.rd["DGZ"]
    base.valid_sha256, base.test_sha256 = "v" * 64, "t" * 64
    files = WH.condition_files(c, target_table_sha="0" * 64, base=base,
                               validation_sha="1" * 64, assertions=[1, 2])
    d = tmp_path / "DGZ" / "W2drop" / "p01_k0"
    assert write_or_verify(d, files) == "written"
    assert write_or_verify(d, files) == "unchanged"
    labels, mask, man = WH.load_condition_files(d)
    assert csr_sha256(labels) == csr_sha256(c.labels) and man["mask_count"] == 60
    assert not man["closed_world"]
    ci = WH.check_input_from_files(d, c.spec, base, fx.prop["DGZ"], [fx.info(T1), fx.info(T2)],
                                   P_tp=fx.P("TP", [T1, T2]), P_dgz=fx.P("DGZ", [T1, T2]))
    WH.run_assertions(ci, fx.vocab, base_labels_sha=csr_sha256(base.train),
                      w2_labels_sha=csr_sha256(c.labels))
    good = {"splits": {"valid": {"labels_sha256": "v" * 64}, "test": {"labels_sha256": "t" * 64},
                       "train": {"labels_sha256": csr_sha256(base.train)}}}
    WH.assertion_4(man, good)
    bad = {"splits": dict(good["splits"], test={"labels_sha256": "x" * 64})}
    with pytest.raises(WH.WithholdingAssertion, match="assertion 4"):
        WH.assertion_4(man, bad)


def test_validation_mask_files():
    f = WH.validation_files("TP", "bp", [5, 2, 2], "0" * 64, "1" * 64)
    assert set(f) == {"validation_mask.npz", "validation_manifest.json"}
    assert b'"n_columns_masked": 2' in f["validation_manifest.json"]
