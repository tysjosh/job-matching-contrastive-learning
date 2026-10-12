"""Label regimes, storage and the G1 checks (design §2).

Regimes (design §2.2):

- TP: A_TP(p) = exp_annotations mapped by ``primary()``, obsolete/unknown dropped,
  filtered to the aspect namespace; closure over is_a + same-namespace part_of,
  root removed; y[p, c] = 1 iff primary(c) is in the closure.
- TPnoIPI (MF only): as TP, from ``annotations`` entries with code in
  EXP_CODES minus IPI.
- DGZ: A_DGZ(p) = exp_annotations as stored (all namespaces, raw strings);
  closure = ``DgzRule.prop``; y[p, c] = 1 iff the column id is in the closure.

Checks 1, 2, 3a-3d and 5 of design §2.4 run on every protein of every split;
any failure is fatal (``LabelCheckError``, exit 2) and nothing is written.
"""
from __future__ import annotations

import gc
import logging
from array import array
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from go_function.constants import (BINDING, EXP_CODES, EXP_CODES_NO_IPI, NAMESPACES,
                                   PROTEIN_BINDING, ROOTS)
from go_function.ontology import DgzRule, Ontology, ancestors, parents
from go_function.pins import csr_arrays, csr_sha256, load_csr, sha256_bytes
from go_function.runtime import FatalError

logger = logging.getLogger("go_function.labels")

SPLITS = ("train", "valid", "test")
CHECK_NAMES = ("1_dgz_eq_prop", "2_exp_eq_codes", "3a_ancestor_implied", "3b_bfs_closure",
               "3c_no_nontp_label", "3d_tp_only_via_alt")


class LabelCheckError(FatalError):
    pass


# ------------------------------------------------------------------ vocabulary
@dataclass(frozen=True)
class Vocab:
    ids: Tuple[str, ...]
    index: Mapping[str, int]
    primary: Tuple[Optional[str], ...]          # primary(c) per column (None: obsolete/unknown)
    alt_cols: FrozenSet[int]                    # columns whose id is an alt id
    duplicate_cols: FrozenSet[int]              # alt columns whose primary is also a column
    primary_to_cols: Mapping[str, Tuple[int, ...]]

    def __len__(self) -> int:
        return len(self.ids)


def read_terms(terms_pkl: Path) -> List[str]:
    import pandas as pd
    df = pd.read_pickle(terms_pkl)
    return [str(g) for g in df["gos"].tolist()]


def build_vocab(ids: Sequence[str], ont: Ontology) -> Vocab:
    ids = tuple(ids)
    if len(set(ids)) != len(ids):
        raise LabelCheckError("terms.pkl has duplicate ids")
    index = {g: i for i, g in enumerate(ids)}
    prim = tuple(ont.primary(g) for g in ids)
    alt = frozenset(i for i, g in enumerate(ids) if g in ont.alt_to_primary)
    dup = frozenset(i for i in alt if prim[i] in index)
    p2c: Dict[str, List[int]] = {}
    for i, p in enumerate(prim):
        if p is not None:
            p2c.setdefault(p, []).append(i)
    return Vocab(ids=ids, index=index, primary=prim, alt_cols=alt, duplicate_cols=dup,
                 primary_to_cols={k: tuple(v) for k, v in p2c.items()})


# ------------------------------------------------------------------ direct sets
def direct_tp(raw_ids: Iterable[str], ont: Ontology, namespace: str,
              counts: Optional[Dict[str, int]] = None) -> FrozenSet[str]:
    """A_TP: primary-mapped, obsolete/unknown dropped, aspect-filtered."""
    out: Set[str] = set()
    for a in raw_ids:
        p = ont.primary(a)
        if p is None:
            if counts is not None:
                key = "obsolete_dropped" if a in ont.obsolete else "unknown_dropped"
                counts[key] = counts.get(key, 0) + 1
            continue
        if ont.terms[p].namespace != namespace:
            continue
        if counts is not None and p != a:
            counts["alt_mapped"] = counts.get("alt_mapped", 0) + 1
        out.add(p)
    return frozenset(out)


def exp_from_annotations(annotations: Iterable[str], codes: FrozenSet[str]) -> Set[str]:
    """GO ids of ``'GO:x|CODE'`` entries whose code is in ``codes``."""
    out: Set[str] = set()
    for entry in annotations:
        go, _, code = entry.partition("|")
        if code in codes:
            out.add(go)
    return out


def tp_closure(direct: Iterable[str], anc: Mapping[str, FrozenSet[str]], root: str) -> FrozenSet[str]:
    out: Set[str] = set()
    for a in direct:
        out |= anc[a]
    out.discard(root)
    return frozenset(out)


def _bfs_closure_tp(ont: Ontology, direct: Iterable[str], namespace: str, root: str) -> Set[str]:
    """Independent TP closure for check 3b: plain BFS over raw ``Term.edges``
    (is_a, and part_of within ``namespace``), no memoisation, no shared helper."""
    seen: Set[str] = set()
    q = deque(direct)
    while q:
        t = q.popleft()
        if t in seen:
            continue
        seen.add(t)
        for rel, parent in ont.terms[t].edges:
            if rel == "is_a" or (rel == "part_of" and ont.terms[parent].namespace == namespace):
                if parent not in seen:
                    q.append(parent)
    seen.discard(root)
    return seen


def tp_cols(closure: Iterable[str], vocab: Vocab) -> Set[int]:
    out: Set[int] = set()
    for g in closure:
        out.update(vocab.primary_to_cols.get(g, ()))
    return out


def dgz_cols(closure: Iterable[str], vocab: Vocab) -> Set[int]:
    idx = vocab.index
    return {idx[g] for g in closure if g in idx}


# ------------------------------------------------------------------ per-protein checks
def check_3a(row: Set[int], parent_cols: Sequence[Tuple[int, ...]]) -> Set[int]:
    """Columns missing from ``row`` although a positive child column implies them."""
    missing: Set[int] = set()
    for c in row:
        for q in parent_cols[c]:
            if q not in row:
                missing.add(q)
    return missing


def check_3c(row: Set[int], cols_dgz: Set[int], cols_bfs: Set[int]) -> Set[int]:
    """TP positives that only a non-TP relation reaches (column level)."""
    return row & (cols_dgz - cols_bfs)


def check_3d(row: Set[int], cols_bfs: Set[int], cols_dgz: Set[int], alt_cols: FrozenSet[int],
             prim_alt_cols: Set[int]) -> Set[int]:
    """TP-only columns not explained by an alt-id column or an alt-id annotation."""
    return ((row | cols_bfs) - cols_dgz) - alt_cols - prim_alt_cols


def protein_binding_only_set(anc: Mapping[str, FrozenSet[str]], root: str) -> FrozenSet[str]:
    return frozenset(anc[PROTEIN_BINDING] - {root})


# ------------------------------------------------------------------ split build
@dataclass
class _CSRBuilder:
    n_cols: int
    indptr: array = field(default_factory=lambda: array("q", [0]))
    indices: array = field(default_factory=lambda: array("i"))

    def add(self, cols: Iterable[int]) -> None:
        self.indices.extend(sorted(cols))
        self.indptr.append(len(self.indices))

    def matrix(self) -> sp.csr_matrix:
        ind = np.frombuffer(self.indices, dtype=np.int32).copy() if len(self.indices) else \
            np.zeros(0, dtype=np.int32)
        ptr = np.frombuffer(self.indptr, dtype=np.int64).copy()
        return sp.csr_matrix((np.ones(len(ind), dtype=np.uint8), ind, ptr),
                             shape=(len(ptr) - 1, self.n_cols))


@dataclass
class RegimeSplit:
    labels: sp.csr_matrix
    direct: sp.csr_matrix
    direct_extra: List[Tuple[str, ...]]
    closure_empty: int
    eval_rows: Optional[np.ndarray] = None
    protein_binding_only: int = 0


@dataclass
class SplitResult:
    split: str
    proteins: List[str]
    regimes: Dict[str, RegimeSplit]
    direct_counts: Dict[str, Dict[str, int]]
    check_counts: Dict[str, int]
    failures: Dict[str, List[str]]


@dataclass
class Context:
    aspect: str
    ont: Ontology
    dgz: DgzRule
    vocab: Vocab
    anc: Dict[str, FrozenSet[str]]
    parent_cols: List[Tuple[int, ...]]
    namespace: str
    root: str
    pb_only: FrozenSet[str]

    @classmethod
    def build(cls, aspect: str, ont: Ontology, dgz: DgzRule, vocab_ids: Sequence[str]) -> "Context":
        vocab = build_vocab(vocab_ids, ont)
        anc = ancestors(ont)
        par = parents(ont)
        pcols: List[Tuple[int, ...]] = []
        for c, p in enumerate(vocab.primary):
            if p is None:
                pcols.append(())
            else:
                pcols.append(tuple(sorted(vocab.index[q] for q in par[p] if q in vocab.index)))
        root = ROOTS[aspect]
        return cls(aspect=aspect, ont=ont, dgz=dgz, vocab=vocab, anc=anc, parent_cols=pcols,
                   namespace=NAMESPACES[aspect], root=root,
                   pb_only=protein_binding_only_set(anc, root))


def _first(xs: List[str], n: int = 10) -> List[str]:
    return xs[:n]


def build_split(ctx: Context, split: str, proteins: Sequence[str],
                exp_annotations: Sequence[Sequence[str]], annotations: Sequence[Sequence[str]],
                prop_annotations: Sequence[Sequence[str]], regimes: Sequence[str]) -> SplitResult:
    """Labels of every requested regime for one split, with every §2.4 check.

    The TP family checks (3a-3d) run for TP and TPnoIPI; checks 1-2 always run.
    """
    v = ctx.vocab
    want_noipi = "TPnoIPI" in regimes
    tp_family = [r for r in ("TP", "TPnoIPI") if r in regimes]
    builders = {r: (_CSRBuilder(len(v)), _CSRBuilder(len(v))) for r in regimes}
    extras: Dict[str, List[Tuple[str, ...]]] = {r: [] for r in regimes}
    empty = {r: 0 for r in regimes}
    pb_only = {r: 0 for r in tp_family}
    eval_rows: Dict[str, List[int]] = {r: [] for r in tp_family}
    dcounts: Dict[str, Dict[str, int]] = {r: {} for r in regimes}
    fails: Dict[str, List[str]] = {k: [] for k in CHECK_NAMES}
    cc = {"dgz_excess": 0, "tp_only_altcols": 0, "tp_only_primalt": 0,
          "noipi_dgz_excess": 0, "noipi_tp_only_altcols": 0, "noipi_tp_only_primalt": 0}

    for i, name in enumerate(proteins):
        exp = list(exp_annotations[i])
        exp_set = set(exp)
        # check 2: exp_annotations == EXP-coded annotations
        if exp_set != exp_from_annotations(annotations[i], EXP_CODES):
            fails["2_exp_eq_codes"].append(name)
        # DGZ regime
        a_dgz = frozenset(exp_set)
        clo_dgz = ctx.dgz.prop(a_dgz)
        cols_d = dgz_cols(clo_dgz, v)
        if cols_d != dgz_cols(prop_annotations[i], v):
            fails["1_dgz_eq_prop"].append(name)
        prim_alt = {ctx.ont.alt_to_primary[a] for a in a_dgz if a in ctx.ont.alt_to_primary}
        prim_alt_cols = {v.index[g] for g in prim_alt if g in v.index}
        if "DGZ" in regimes:
            lab, dirb = builders["DGZ"]
            lab.add(cols_d)
            dirb.add(v.index[a] for a in a_dgz if a in v.index)
            extras["DGZ"].append(tuple(sorted(a for a in a_dgz if a not in v.index)))
            if not cols_d:
                empty["DGZ"] += 1
        # TP family
        for r in tp_family:
            if r == "TP":
                a_tp = direct_tp(exp, ctx.ont, ctx.namespace, dcounts["TP"])
            else:
                a_tp = direct_tp(sorted(exp_from_annotations(annotations[i], EXP_CODES_NO_IPI)),
                                 ctx.ont, ctx.namespace, dcounts["TPnoIPI"])
            clo = tp_closure(a_tp, ctx.anc, ctx.root)
            row = tp_cols(clo, v)
            bfs = _bfs_closure_tp(ctx.ont, a_tp, ctx.namespace, ctx.root)
            cols_b = tp_cols(bfs, v)
            if bfs != clo or cols_b != row:
                fails["3b_bfs_closure"].append(f"{r}:{name}")
            if check_3a(row, ctx.parent_cols):
                fails["3a_ancestor_implied"].append(f"{r}:{name}")
            if check_3c(row, cols_d, cols_b):
                fails["3c_no_nontp_label"].append(f"{r}:{name}")
            if check_3d(row, cols_b, cols_d, v.alt_cols, prim_alt_cols):
                fails["3d_tp_only_via_alt"].append(f"{r}:{name}")
            pre = "" if r == "TP" else "noipi_"
            cc[pre + "dgz_excess"] += len(cols_d - cols_b)
            tp_only = cols_b - cols_d
            cc[pre + "tp_only_altcols"] += len(tp_only & v.alt_cols)
            cc[pre + "tp_only_primalt"] += len(tp_only & prim_alt_cols)
            lab, dirb = builders[r]
            lab.add(row)
            dirb.add(v.index[a] for a in a_tp if a in v.index)
            extras[r].append(tuple(sorted(a for a in a_tp if a not in v.index)))
            if not clo:
                empty[r] += 1
            # evaluation set (design §2.5): MF excludes protein-binding-only;
            # TPnoIPI also requires a non-empty closure.
            is_pb = ctx.aspect == "mf" and clo == ctx.pb_only
            if is_pb:
                pb_only[r] += 1
            if not is_pb and (r == "TP" or clo):
                eval_rows[r].append(i)

    out: Dict[str, RegimeSplit] = {}
    for r in regimes:
        lab, dirb = builders[r]
        rs = RegimeSplit(labels=lab.matrix(), direct=dirb.matrix(), direct_extra=extras[r],
                         closure_empty=empty[r])
        if r in tp_family:
            rs.eval_rows = np.asarray(eval_rows[r], dtype=np.int64)
            rs.protein_binding_only = pb_only[r]
        out[r] = rs
    if not want_noipi:
        for k in [k for k in cc if k.startswith("noipi_")]:
            del cc[k]
    return SplitResult(split=split, proteins=list(proteins), regimes=out, direct_counts=dcounts,
                       check_counts=cc, failures={k: v_ for k, v_ in fails.items()})


def load_split_columns(pkl: Path) -> Dict[str, list]:
    """The four columns the label builder reads, with the DataFrame freed."""
    import pandas as pd
    df = pd.read_pickle(pkl)
    cols = {"proteins": [str(x) for x in df["proteins"].tolist()],
            "exp_annotations": [list(x) for x in df["exp_annotations"].tolist()],
            "annotations": [list(x) for x in df["annotations"].tolist()],
            "prop_annotations": [list(x) for x in df["prop_annotations"].tolist()]}
    del df
    gc.collect()
    return cols


# ------------------------------------------------------------------ storage
def _str_array(xs: Sequence[str]) -> np.ndarray:
    return np.asarray(list(xs), dtype=str) if len(xs) else np.zeros(0, dtype="<U1")


def regime_arrays(vocab: Vocab, results: Mapping[str, SplitResult], regime: str) -> Dict[str, np.ndarray]:
    """npz members of ``labels_<regime>.npz`` (design §2.3). ``direct_extra_<split>``
    is stored as ``_indptr`` + flat ``_ids`` (no pickled objects)."""
    arrays: Dict[str, np.ndarray] = {"vocab": _str_array(vocab.ids)}
    for split in SPLITS:
        res = results[split]
        rs = res.regimes[regime]
        arrays.update(csr_arrays(split, rs.labels))
        arrays.update(csr_arrays(f"direct_{split}", rs.direct))
        ptr = np.zeros(len(rs.direct_extra) + 1, dtype=np.int64)
        np.cumsum([len(t) for t in rs.direct_extra], out=ptr[1:])
        arrays[f"direct_extra_{split}_indptr"] = ptr
        arrays[f"direct_extra_{split}_ids"] = _str_array([g for t in rs.direct_extra for g in t])
        arrays[f"proteins_{split}"] = _str_array(res.proteins)
    if results["test"].regimes[regime].eval_rows is not None:
        arrays["eval_test"] = results["test"].regimes[regime].eval_rows
    return arrays


def load_labels(path: Path) -> Dict[str, Any]:
    """Read ``labels_<regime>.npz`` back: CSR matrices, direct_extra as lists of
    tuples, protein and vocabulary lists, and ``eval_test`` when present."""
    with np.load(path, allow_pickle=False) as z:
        out: Dict[str, Any] = {"vocab": [str(x) for x in z["vocab"]]}
        for split in SPLITS:
            out[split] = load_csr(z, split)
            out[f"direct_{split}"] = load_csr(z, f"direct_{split}")
            ptr = z[f"direct_extra_{split}_indptr"]
            ids = [str(x) for x in z[f"direct_extra_{split}_ids"]]
            out[f"direct_extra_{split}"] = [tuple(ids[ptr[i]:ptr[i + 1]])
                                            for i in range(len(ptr) - 1)]
            out[f"proteins_{split}"] = [str(x) for x in z[f"proteins_{split}"]]
        if "eval_test" in z.files:
            out["eval_test"] = z["eval_test"].copy()
    return out


def regime_manifest(ctx: Context, results: Mapping[str, SplitResult], regime: str,
                    npz_rel: str, npz_sha: str, inputs: Mapping[str, str]) -> Dict[str, Any]:
    v = ctx.vocab
    splits: Dict[str, Any] = {}
    zero_train: List[str] = []
    for split in SPLITS:
        rs = results[split].regimes[regime]
        m = rs.labels
        col_pos = np.asarray(m.sum(axis=0)).ravel()
        zero = np.flatnonzero(col_pos == 0)
        if split == "train":
            zero_train = [v.ids[j] for j in zero]
        dc = results[split].direct_counts.get(regime, {})
        entry = {"n_rows": int(m.shape[0]), "n_labels": int(m.nnz),
                 "n_zero_positive_columns": int(len(zero)),
                 "n_rows_empty_closure": int(rs.closure_empty),
                 "labels_sha256": csr_sha256(m), "direct_sha256": csr_sha256(rs.direct),
                 "n_direct_column_annotations": int(rs.direct.nnz),
                 "n_direct_extra_annotations": int(sum(len(t) for t in rs.direct_extra)),
                 "obsolete_dropped": int(dc.get("obsolete_dropped", 0)),
                 "unknown_dropped": int(dc.get("unknown_dropped", 0)),
                 "alt_mapped": int(dc.get("alt_mapped", 0)),
                 "proteins_sha256": sha256_bytes("\n".join(results[split].proteins).encode())}
        if rs.eval_rows is not None and split == "test":
            entry["n_eval_rows"] = int(len(rs.eval_rows))
            entry["n_protein_binding_only"] = int(rs.protein_binding_only)
        splits[split] = entry
    checks = {split: dict(results[split].check_counts) for split in SPLITS}
    return {
        "schema": 1,
        "aspect": ctx.aspect,
        "regime": regime,
        "npz": npz_rel,
        "npz_sha256": npz_sha,
        "inputs": dict(sorted(inputs.items())),
        "ontology_parse_report": dict(sorted(ctx.ont.report.items())),
        "vocab": {"n_columns": len(v), "vocab_sha256": sha256_bytes("\n".join(v.ids).encode()),
                  "n_alt_columns": len(v.alt_cols),
                  "alt_columns": sorted(v.ids[j] for j in v.alt_cols),
                  "n_duplicate_columns": len(v.duplicate_cols),
                  "duplicate_columns": sorted(v.ids[j] for j in v.duplicate_cols),
                  "n_unmapped_columns": sum(1 for p in v.primary if p is None)},
        "splits": splits,
        "zero_positive_columns_train": zero_train,
        "checks_passed": list(CHECK_NAMES),
        "check_counts": checks,
        "evaluation_set": ({"rule": "mf: test rows minus protein-binding-only"
                            + ("; non-empty TPnoIPI closure" if regime == "TPnoIPI" else ""),
                            "protein_binding_only_closure": sorted(ctx.pb_only)}
                           if ctx.aspect == "mf" and regime != "DGZ" else
                           {"rule": "all test rows"} if regime != "DGZ" else None),
    }


def raise_on_failures(results: Mapping[str, SplitResult]) -> None:
    bad = []
    for split, res in results.items():
        for check, ids in res.failures.items():
            if ids:
                bad.append(f"check {check} failed on {len(ids)} {split} proteins; first: "
                           f"{_first(ids)}")
    if bad:
        raise LabelCheckError("; ".join(bad))
