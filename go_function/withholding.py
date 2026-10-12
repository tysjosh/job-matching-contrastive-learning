"""Closed-world withholding generator and its assertions (design §5 v4, protocol §5; gap 8).

Rules (protocol D19, round-3 orchestrator decision):

- W2 (closed world): for each pack target t and p in W(t) = P(t) \\ R_k, erase from
  A_rho(p) every annotation a with t in closure_rho(a) (term level D*^GO_rho(t)),
  then re-propagate once (all targets of p first). No mask: erased D*_rho(t) and
  Lost(p) entries are ordinary 0 labels.
- W1 (closed world): W2, then the strict ancestors of t are set positive. No mask.
- WDGZrel (DGZ, k = 0): base DGZ labels unchanged; T_cols(t) masked on every train row.
- W2drop (DGZ, k = 0): closed-world W2 labels + T_cols(t) masked on every train row.
- ``all``: base labels, no mask (one condition per regime).

Assertions 1-10 of design §5.5 are plain functions over a ``CheckInput`` so that
``run.py`` preflight can re-run them on files (``check_input_from_files``). Every
failure raises ``WithholdingAssertion`` (exit 2) naming the assertion and counts
only: target ids are referred to by their position in the pack.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

from go_function import labels as L
from go_function.constants import ROOTS
from go_function.labels import Vocab
from go_function.ontology import DgzRule
from go_function.pins import canonical_json, csr_arrays, csr_sha256, load_csr, npz_bytes, sha256_bytes
from go_function.runtime import FatalError

K = Union[int, str]
SCHEMA = 1
NUMERIC_KS = (0, 1, 5, 10, 25)


class WithholdingAssertion(FatalError):
    pass


def _fail(n: int, msg: str) -> None:
    raise WithholdingAssertion(f"assertion {n} failed: {msg}")


# ------------------------------------------------------------------ condition identity
@dataclass(frozen=True)
class Spec:
    aspect: str
    regime: str               # TP | DGZ | TPnoIPI
    rule: str                 # W2 | W1 | WDGZrel | W2drop | all
    pack: Optional[str] = None
    k: K = "all"
    draw: int = 1

    def __post_init__(self):
        if self.rule == "all":
            if self.pack is not None or self.k != "all" or self.draw != 1:
                raise FatalError("the 'all' condition has no pack, k or draw")
            return
        if self.rule not in ("W2", "W1", "WDGZrel", "W2drop"):
            raise FatalError(f"unknown rule {self.rule!r}")
        if self.k not in NUMERIC_KS:
            raise FatalError(f"k must be one of {NUMERIC_KS} for a pack condition")
        if self.rule in ("WDGZrel", "W2drop") and (self.regime != "DGZ" or self.k != 0):
            raise FatalError(f"{self.rule} exists only for regime DGZ at k = 0")
        if self.draw not in (1, 2):
            raise FatalError("draw must be 1 or 2")
        if self.draw == 2 and not (self.rule == "W2" and self.regime == "TP"):
            raise FatalError("draw 2 exists only for TP/W2")

    @property
    def name(self) -> str:
        if self.rule == "all":
            return f"{self.regime}/all"
        return f"{self.regime}/{self.rule}/{self.pack}_k{self.k}" + ("_d2" if self.draw == 2 else "")


# ------------------------------------------------------------------ regime data and rules
@dataclass
class RegimeData:
    """Base train labels and the stored direct set A_rho of one regime (design §2.3)."""
    regime: str
    vocab: Vocab
    train: sp.csr_matrix
    direct: sp.csr_matrix
    extra: Sequence[Tuple[str, ...]]
    proteins: Sequence[str]
    valid_sha256: str = ""
    test_sha256: str = ""
    base_manifest_sha256: str = ""

    def annotations(self, p: int) -> List[str]:
        d = self.direct
        ids = self.vocab.ids
        out = [ids[c] for c in d.indices[d.indptr[p]:d.indptr[p + 1]]]
        out.extend(self.extra[p])
        return out

    def row(self, p: int) -> Set[int]:
        m = self.train
        return set(int(c) for c in m.indices[m.indptr[p]:m.indptr[p + 1]])


class Propagator:
    """closure_rho, the erasure test (t in closure_rho(a)) and the column sets of one
    regime. TP and TPnoIPI share the TP closure (design §5.1)."""

    def __init__(self, regime: str, vocab: Vocab, aspect: str,
                 anc: Mapping[str, FrozenSet[str]], dgz: Optional[DgzRule] = None):
        self.regime, self.vocab, self.anc, self.dgz = regime, vocab, anc, dgz
        self.root = ROOTS[aspect]
        self.is_dgz = regime == "DGZ"
        if self.is_dgz and dgz is None:
            raise FatalError("the DGZ regime needs a DgzRule")
        self._dstar: Dict[str, np.ndarray] = {}

    def implies(self, a: str, t: str) -> bool:
        if self.is_dgz:
            return t in self.dgz.closure(a)
        anc = self.anc.get(a)
        return anc is not None and t in anc

    def cols(self, annotations: Iterable[str]) -> Set[int]:
        if self.is_dgz:
            return L.dgz_cols(self.dgz.prop(annotations), self.vocab)
        return L.tp_cols(L.tp_closure(annotations, self.anc, self.root), self.vocab)

    def t_cols(self, t: str) -> List[int]:
        return sorted(self.vocab.primary_to_cols.get(t, ()))

    def dstar_cols(self, t: str) -> np.ndarray:
        """Column-level D*_rho(t) (design §5.1); for DGZ including T_cols(t)."""
        hit = self._dstar.get(t)
        if hit is not None:
            return hit
        v = self.vocab
        if self.is_dgz:
            cs = {c for c in range(len(v)) if t in self.dgz.closure(v.ids[c])}
            cs.update(self.t_cols(t))
        else:
            cs = {c for c in range(len(v)) if v.primary[c] is not None
                  and t in self.anc.get(v.primary[c], ())}
        out = np.asarray(sorted(cs), dtype=np.int64)
        self._dstar[t] = out
        return out

    def strict_ancestor_cols(self, t: str) -> np.ndarray:
        if self.is_dgz:
            cs = L.dgz_cols(self.dgz.closure(t) - {t}, self.vocab)
        else:
            cs = L.tp_cols(self.anc[t] - {t, self.root}, self.vocab)
        return np.asarray(sorted(cs), dtype=np.int64)


@dataclass(frozen=True)
class TargetInfo:
    id: str
    column: int
    draw1: Tuple[int, ...]
    draw2: Tuple[int, ...]
    noipi_order: Optional[Tuple[int, ...]] = None
    noipi_kept: bool = True


def pack_target_infos(table: Mapping[str, Any], pack: str) -> List[TargetInfo]:
    out = []
    for t in table["packs"][pack]["targets"]:
        e = table["targets"][t]
        ni = e.get("noipi")
        out.append(TargetInfo(id=t, column=int(e["column"]),
                              draw1=tuple(e["retention"]["draw1"]),
                              draw2=tuple(e["retention"]["draw2"]),
                              noipi_order=tuple(ni["order"]) if ni else None,
                              noipi_kept=bool(ni["kept"]) if ni else True))
    return out


def all_target_ids(table: Mapping[str, Any]) -> List[str]:
    return sorted(table["targets"])


# ------------------------------------------------------------------ helpers
def column_positive_rows(m: sp.csr_matrix, c: int) -> np.ndarray:
    col = sp.csc_matrix(m[:, c])
    return np.sort(col.indices.astype(np.int64))


def replace_rows(base: sp.csr_matrix, new_rows: Mapping[int, Iterable[int]]) -> sp.csr_matrix:
    """``base`` with the given rows replaced (vectorised; no dense matrix)."""
    base = sp.csr_matrix(base)
    if not new_rows:
        out = base.copy()
        out.sort_indices()
        return out
    n_rows = base.shape[0]
    row_of = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(base.indptr))
    changed = np.fromiter(sorted(new_rows), dtype=np.int64)
    keep = ~np.isin(row_of, changed)
    nr: List[int] = []
    nc: List[int] = []
    for p in changed.tolist():
        cs = sorted(set(int(c) for c in new_rows[p]))
        nr.extend([p] * len(cs))
        nc.extend(cs)
    r = np.concatenate([row_of[keep], np.asarray(nr, dtype=np.int64)])
    c = np.concatenate([base.indices[keep].astype(np.int64), np.asarray(nc, dtype=np.int64)])
    out = sp.csr_matrix((np.ones(len(r), dtype=np.uint8), (r, c)), shape=base.shape)
    out.sort_indices()
    out.data[:] = 1
    return out


def full_column_mask(n_rows: int, cols: Iterable[int]) -> np.ndarray:
    """(row, col) int32 pairs of whole train columns, sorted by (row, col)."""
    cols = np.asarray(sorted(set(int(c) for c in cols)), dtype=np.int32)
    if not len(cols):
        return np.zeros((0, 2), dtype=np.int32)
    rows = np.repeat(np.arange(n_rows, dtype=np.int32), len(cols))
    return np.stack([rows, np.tile(cols, n_rows)], axis=1).astype(np.int32)


def mask_sha256(mask: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(mask, dtype=np.int32).tobytes())


def _keys(pairs: np.ndarray, n_cols: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return pairs[:, 0] * n_cols + pairs[:, 1]


# ------------------------------------------------------------------ generation
@dataclass
class CheckInput:
    """Everything assertions 1-10 read for one condition."""
    spec: Spec
    labels: sp.csr_matrix
    mask: np.ndarray                         # (n, 2) int32
    base: sp.csr_matrix                      # base train labels of the regime
    targets: List[str]
    t_cols: Dict[str, List[int]]
    dstar: Dict[str, np.ndarray]
    strict_anc: Dict[str, np.ndarray]
    P: Dict[str, np.ndarray]                 # P_rho(t), base train rows
    R: Dict[str, List[int]]                  # retained rows (draw / noIPI order)
    W: Dict[str, np.ndarray]
    P_tp: Optional[Dict[str, np.ndarray]] = None
    P_dgz: Optional[Dict[str, np.ndarray]] = None
    target_column: Dict[str, int] = field(default_factory=dict)


@dataclass
class Condition:
    spec: Spec
    labels: sp.csr_matrix
    mask: np.ndarray
    check: CheckInput
    stats: Dict[str, Any]


def _retained(info: TargetInfo, spec: Spec) -> List[int]:
    if spec.regime == "TPnoIPI":
        if info.noipi_order is None:
            raise FatalError("the target table has no noIPI retention order")
        order = info.noipi_order
    else:
        order = info.draw2 if spec.draw == 2 else info.draw1
    if spec.k == "all":
        return list(order)
    return list(order[:int(spec.k)])


def make_condition(spec: Spec, rd: RegimeData, prop: Propagator, infos: Sequence[TargetInfo],
                   *, P_tp: Optional[Mapping[str, np.ndarray]] = None,
                   P_dgz: Optional[Mapping[str, np.ndarray]] = None) -> Condition:
    """Build one condition (labels, mask, the CheckInput and descriptive stats)."""
    if rd.regime != spec.regime or prop.regime != spec.regime:
        raise FatalError("regime data / propagator / spec regime mismatch")
    base = rd.train
    n_rows = base.shape[0]
    if spec.regime == "TPnoIPI":
        infos = [i for i in infos if i.noipi_kept]
    if spec.rule == "all":
        infos = []
    targets = [i.id for i in infos]
    t_cols = {i.id: prop.t_cols(i.id) for i in infos}
    dstar = {i.id: prop.dstar_cols(i.id) for i in infos}
    sanc = {i.id: prop.strict_ancestor_cols(i.id) for i in infos}
    P: Dict[str, np.ndarray] = {}
    R: Dict[str, List[int]] = {}
    W: Dict[str, np.ndarray] = {}
    for i in infos:          # T_cols(t) = {t} is checked by assertion 8
        P[i.id] = column_positive_rows(base, i.column)
        R[i.id] = _retained(i, spec)
        W[i.id] = np.setdiff1d(P[i.id], np.asarray(R[i.id], dtype=np.int64))
    stats: Dict[str, Any] = {"n_targets": len(infos)}
    mask = np.zeros((0, 2), dtype=np.int32)

    if spec.rule in ("all", "WDGZrel"):
        labels = sp.csr_matrix(base, copy=True)
        labels.sort_indices()
        if spec.rule == "WDGZrel":
            mask = full_column_mask(n_rows, [c for t in targets for c in t_cols[t]])
    else:
        per_row: Dict[int, List[str]] = {}
        for t in targets:
            for p in W[t].tolist():
                per_row.setdefault(int(p), []).append(t)
        new_rows: Dict[int, Set[int]] = {}
        n_erased = n_lost = n_set_w1 = 0
        for p in sorted(per_row):
            ts = per_row[p]
            ann = rd.annotations(p)
            before = rd.row(p)
            if prop.cols(ann) != before:
                raise FatalError(f"re-propagation of the stored direct set does not reproduce "
                                 f"base train row {p} ({spec.name})")
            kept = [a for a in ann if not any(prop.implies(a, t) for t in ts)]
            n_erased += len(ann) - len(kept)
            after = prop.cols(kept)
            dst: Set[int] = set()
            for t in ts:
                dst.update(int(c) for c in dstar[t])
            n_lost += len((before - after) - dst)
            if spec.rule == "W1":
                for t in ts:
                    add = set(int(c) for c in sanc[t])
                    n_set_w1 += len(add - after)
                    after |= add
            new_rows[p] = after
        labels = replace_rows(base, new_rows)
        if spec.rule == "W2drop":
            mask = full_column_mask(n_rows, [c for t in targets for c in t_cols[t]])
        diff = sp.csr_matrix(base, dtype=np.int16) - sp.csr_matrix(labels, dtype=np.int16)
        stats.update({"n_withheld_rows": len(per_row), "n_erased_annotations": n_erased,
                      "n_lost_entries_unmasked": n_lost,
                      "n_entries_removed": int((diff.data == 1).sum()),
                      "n_entries_added": int((diff.data == -1).sum())})
        if spec.rule == "W1":
            stats["n_w1_ancestor_entries_set"] = n_set_w1
    ci = CheckInput(spec=spec, labels=labels, mask=mask, base=base, targets=targets,
                    t_cols=t_cols, dstar=dstar, strict_anc=sanc, P=P, R=R, W=W,
                    P_tp=dict(P_tp) if P_tp is not None else None,
                    P_dgz=dict(P_dgz) if P_dgz is not None else None,
                    target_column={i.id: i.column for i in infos})
    stats["per_target"] = [{"W": int(len(W[t])), "R_k": int(len(R[t])), "P": int(len(P[t])),
                            "n_dstar_cols": int(len(dstar[t]))} for t in targets]
    stats["mask_count"] = int(len(mask))
    return Condition(spec=spec, labels=labels, mask=mask, check=ci, stats=stats)


# ------------------------------------------------------------------ assertions (design §5.5)
def _masked_keys(ci: CheckInput) -> np.ndarray:
    return np.unique(_keys(ci.mask, ci.labels.shape[1]))


def assertion_1(ci: CheckInput) -> None:
    """W1/W2/W2drop: for every t and p in W(t), y[p, c] = 0 and (p, c) not masked for
    all c in D*_rho(t) (under W2drop the T_cols(t) are masked by design: y = 0 only)."""
    if ci.spec.rule not in ("W1", "W2", "W2drop"):
        return
    n_cols = ci.labels.shape[1]
    mk = _masked_keys(ci)
    for j, t in enumerate(ci.targets):
        rows, cols = ci.W[t], ci.dstar[t]
        if not len(rows) or not len(cols):
            continue
        sub = ci.labels[rows][:, cols]
        if sub.nnz:
            _fail(1, f"{int(sub.nnz)} positive D* entries on withheld rows of pack target #{j}")
        check_cols = cols
        if ci.spec.rule == "W2drop":
            check_cols = np.setdiff1d(cols, np.asarray(ci.t_cols[t], dtype=np.int64))
        if len(check_cols) and len(mk):
            keys = (rows[:, None] * n_cols + check_cols[None, :]).ravel()
            hit = int(np.isin(keys, mk).sum())
            if hit:
                _fail(1, f"{hit} masked D* entries on withheld rows of pack target #{j}")


def assertion_2(ci: CheckInput) -> None:
    """Numeric k, W1/W2/W2drop: exactly k train positives on column t, equal to R_k."""
    if ci.spec.rule not in ("W1", "W2", "W2drop") or ci.spec.k == "all":
        return
    for j, t in enumerate(ci.targets):
        pos = column_positive_rows(ci.labels, ci.target_column[t])
        if len(pos) != int(ci.spec.k) or set(pos.tolist()) != set(ci.R[t]):
            _fail(2, f"pack target #{j} has {len(pos)} train positives, expected k={ci.spec.k} "
                     "equal to R_k")


def assertion_3(ci: CheckInput) -> None:
    """W1: every strict ancestor column of t is positive for every p in W(t)."""
    if ci.spec.rule != "W1":
        return
    for j, t in enumerate(ci.targets):
        rows, cols = ci.W[t], ci.strict_anc[t]
        if not len(rows) or not len(cols):
            continue
        sub = ci.labels[rows][:, cols]
        missing = len(rows) * len(cols) - int(sub.nnz)
        if missing:
            _fail(3, f"{missing} strict-ancestor entries not positive for pack target #{j}")


def assertion_4(cond_manifest: Mapping[str, Any], base_manifest: Mapping[str, Any]) -> None:
    """Test and validation labels and their hashes equal the base label manifest."""
    b = cond_manifest.get("base", {})
    for split in ("valid", "test"):
        want = base_manifest["splits"][split]["labels_sha256"]
        if b.get(f"{split}_labels_sha256") != want:
            _fail(4, f"{split} label hash differs from the base label manifest")
    if b.get("train_labels_sha256") != base_manifest["splits"]["train"]["labels_sha256"]:
        _fail(4, "base train label hash differs from the base label manifest")


def assertion_5(ci: CheckInput) -> None:
    """No protein outside the union of P_rho(t) changed any label."""
    diff = (ci.labels != ci.base)
    rows = np.unique(sp.coo_matrix(diff).row) if diff.nnz else np.zeros(0, dtype=np.int64)
    allowed = set()
    for t in ci.targets:
        allowed.update(ci.P[t].tolist())
    bad = sum(1 for r in rows.tolist() if r not in allowed)
    if bad:
        _fail(5, f"{bad} rows outside the union of P(t) changed")


def assertion_6(ci: CheckInput, base_labels_sha: str) -> None:
    """WDGZrel: labels = base DGZ; mask = T_cols of the pack targets on every train row."""
    if ci.spec.rule != "WDGZrel":
        return
    if csr_sha256(ci.labels) != base_labels_sha:
        _fail(6, "WDGZrel train labels differ from the base DGZ labels")
    want = full_column_mask(ci.labels.shape[0], [c for t in ci.targets for c in ci.t_cols[t]])
    if not np.array_equal(_sorted_mask(ci.mask), want):
        _fail(6, "WDGZrel mask is not exactly the target columns on every train row")


def assertion_7(ci: CheckInput, w2_labels_sha: str) -> None:
    """W2drop: labels = the W2/DGZ/k0 condition of the pack; mask = T_cols on all rows."""
    if ci.spec.rule != "W2drop":
        return
    if csr_sha256(ci.labels) != w2_labels_sha:
        _fail(7, "W2drop train labels differ from the W2/DGZ/k0 condition of the same pack")
    want = full_column_mask(ci.labels.shape[0], [c for t in ci.targets for c in ci.t_cols[t]])
    if not np.array_equal(_sorted_mask(ci.mask), want):
        _fail(7, "W2drop mask is not exactly the target columns on every train row")


def assertion_8(ci: CheckInput, vocab: Vocab) -> None:
    """No column with primary(c) = t has an unmasked train positive outside R_k; under
    WDGZrel/W2drop (k = 0 only) no such column has any unmasked train entry; and
    T_cols(t) = {t} for every target."""
    n_rows, n_cols = ci.labels.shape
    mk = _masked_keys(ci)
    for j, t in enumerate(ci.targets):
        cols = list(vocab.primary_to_cols.get(t, ()))
        if cols != [ci.target_column[t]]:
            _fail(8, f"pack target #{j} has {len(cols)} columns with its primary id")
        for c in cols:
            masked_rows = (mk[mk % n_cols == c] // n_cols) if len(mk) else np.zeros(0, np.int64)
            if ci.spec.rule in ("WDGZrel", "W2drop"):
                if len(masked_rows) != n_rows:
                    _fail(8, f"{n_rows - len(masked_rows)} unmasked train entries on a column "
                             f"of pack target #{j}")
                continue
            pos = column_positive_rows(ci.labels, c)
            unmasked = np.setdiff1d(pos, masked_rows)
            outside = np.setdiff1d(unmasked, np.asarray(ci.R[t], dtype=np.int64))
            if len(outside):
                _fail(8, f"{len(outside)} unmasked train positives outside R_k on a column of "
                         f"pack target #{j}")


def assertion_9(ci: CheckInput) -> None:
    """R_k subset of P_TP(t) subset of P_DGZ(t) (TPnoIPI: R_k subset of P_TPnoIPI(t))."""
    for j, t in enumerate(ci.targets):
        R = set(ci.R[t])
        if ci.spec.regime == "TPnoIPI":
            if not R <= set(ci.P[t].tolist()):
                _fail(9, f"R_k is not inside P_TPnoIPI for pack target #{j}")
            continue
        if ci.P_tp is None or ci.P_dgz is None:
            raise FatalError("assertion 9 needs P_TP and P_DGZ")
        ptp, pdg = set(ci.P_tp[t].tolist()), set(ci.P_dgz[t].tolist())
        if not (R <= ptp <= pdg):
            _fail(9, f"R_k <= P_TP <= P_DGZ does not hold for pack target #{j}")


def assertion_10(ci: CheckInput) -> None:
    """Closed world: W1/W2/all have an empty train mask; WDGZrel/W2drop mask whole
    train columns only (so no mask is row-selective on any column)."""
    if ci.spec.rule in ("W1", "W2", "all"):
        if len(ci.mask):
            _fail(10, f"{len(ci.mask)} masked entries in a closed-world {ci.spec.rule} condition")
        return
    n_rows = ci.labels.shape[0]
    m = np.asarray(ci.mask, dtype=np.int64).reshape(-1, 2)
    if len(np.unique(_keys(m, ci.labels.shape[1]))) != len(m):
        _fail(10, "duplicate masked entries")
    cols, counts = np.unique(m[:, 1], return_counts=True)
    partial = int((counts != n_rows).sum())
    if partial:
        _fail(10, f"{partial} masked columns are row-selective (not masked on all train rows)")


def _sorted_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.int32).reshape(-1, 2)
    if not len(m):
        return m
    order = np.lexsort((m[:, 1], m[:, 0]))
    return m[order]


def run_assertions(ci: CheckInput, vocab: Vocab, *, base_labels_sha: str,
                   w2_labels_sha: Optional[str] = None,
                   cond_manifest: Optional[Mapping[str, Any]] = None,
                   base_manifest: Optional[Mapping[str, Any]] = None) -> List[int]:
    """Run every assertion that applies to the condition's rule; returns their numbers.
    Used by ``make_conditions.py`` and by ``run.py`` preflight."""
    done: List[int] = []
    rule = ci.spec.rule
    if rule in ("W1", "W2", "W2drop"):
        assertion_1(ci); done.append(1)
        if ci.spec.k != "all":
            assertion_2(ci); done.append(2)
    if rule == "W1":
        assertion_3(ci); done.append(3)
    if cond_manifest is not None and base_manifest is not None:
        assertion_4(cond_manifest, base_manifest); done.append(4)
    assertion_5(ci); done.append(5)
    if rule == "all" and csr_sha256(ci.labels) != base_labels_sha:
        _fail(5, "the 'all' condition differs from the base train labels")
    if rule == "WDGZrel":
        assertion_6(ci, base_labels_sha); done.append(6)
    if rule == "W2drop":
        if w2_labels_sha is None:
            raise FatalError("assertion 7 needs the W2/DGZ/k0 label hash")
        assertion_7(ci, w2_labels_sha); done.append(7)
    if rule != "all":
        assertion_8(ci, vocab); done.append(8)
        assertion_9(ci); done.append(9)
    assertion_10(ci); done.append(10)
    return done


# ------------------------------------------------------------------ files (design §5.4)
def condition_files(cond: Condition, *, target_table_sha: str, base: RegimeData,
                    validation_sha: str, assertions: Sequence[int],
                    extra: Optional[Mapping[str, Any]] = None) -> Dict[str, bytes]:
    lab_bytes = npz_bytes(csr_arrays("train", cond.labels))
    mask = _sorted_mask(cond.mask)
    mask_bytes = npz_bytes({"pairs": mask.astype(np.int32),
                            "shape": np.asarray(cond.labels.shape, dtype=np.int64)})
    ci = cond.check
    man: Dict[str, Any] = {
        "schema": SCHEMA, "aspect": cond.spec.aspect, "regime": cond.spec.regime,
        "rule": cond.spec.rule, "pack": cond.spec.pack, "k": cond.spec.k,
        "draw": cond.spec.draw, "closed_world": cond.spec.rule in ("W1", "W2", "all"),
        "targets": list(ci.targets),
        "retained_rows": {t: list(ci.R[t]) for t in ci.targets},
        "n_withheld": {t: int(len(ci.W[t])) for t in ci.targets},
        "n_retained": {t: int(len(ci.R[t])) for t in ci.targets},
        "mask_count": int(len(mask)),
        "train_labels_sha256": csr_sha256(cond.labels),
        "train_labels_npz_sha256": sha256_bytes(lab_bytes),
        "train_mask_sha256": mask_sha256(mask),
        "train_mask_npz_sha256": sha256_bytes(mask_bytes),
        "base": {"label_manifest_sha256": base.base_manifest_sha256,
                 "train_labels_sha256": csr_sha256(base.train),
                 "valid_labels_sha256": base.valid_sha256,
                 "test_labels_sha256": base.test_sha256},
        "validation_mask_sha256": validation_sha,
        "target_table_sha256": target_table_sha,
        "stats": {k: v for k, v in cond.stats.items() if k != "per_target"},
        "assertions_passed": sorted(assertions),
    }
    if extra:
        man.update(extra)
    return {"train_labels.npz": lab_bytes, "train_mask.npz": mask_bytes,
            "withholding_manifest.json": canonical_json(man)}


def load_condition_files(cond_dir: Path) -> Tuple[sp.csr_matrix, np.ndarray, Dict[str, Any]]:
    """(train labels, mask pairs, manifest) of a written condition directory."""
    with np.load(cond_dir / "train_labels.npz", allow_pickle=False) as z:
        labels = load_csr(z, "train")
    with np.load(cond_dir / "train_mask.npz", allow_pickle=False) as z:
        mask = z["pairs"].copy()
    man = json.loads((cond_dir / "withholding_manifest.json").read_text())
    return labels, mask, man


def check_input_from_files(cond_dir: Path, spec: Spec, rd: RegimeData, prop: Propagator,
                           infos: Sequence[TargetInfo], *,
                           P_tp: Optional[Mapping[str, np.ndarray]] = None,
                           P_dgz: Optional[Mapping[str, np.ndarray]] = None) -> CheckInput:
    """A CheckInput over the files of ``cond_dir`` (for run.py preflight): the
    target-derived sets are recomputed from the base data, labels/mask come from disk."""
    labels, mask, man = load_condition_files(cond_dir)
    ref = make_condition(spec, rd, prop, infos, P_tp=P_tp, P_dgz=P_dgz).check
    if man["targets"] != ref.targets:
        raise FatalError(f"{spec.name}: manifest targets differ from the target table")
    ref.labels, ref.mask = labels, mask
    return ref


# ------------------------------------------------------------------ validation (§5.3)
def validation_files(regime: str, aspect: str, dstar_union: Sequence[int],
                     target_table_sha: str, valid_sha: str) -> Dict[str, bytes]:
    cols = np.asarray(sorted(set(int(c) for c in dstar_union)), dtype=np.int32)
    blob = npz_bytes({"columns": cols})
    man = {"schema": SCHEMA, "aspect": aspect, "regime": regime,
           "rule": "target-free validation: columns of the union of D*_rho(t) over all "
                   "targets excluded from every validation loss",
           "n_columns_masked": int(len(cols)), "columns_sha256": sha256_bytes(cols.tobytes()),
           "npz_sha256": sha256_bytes(blob), "valid_labels_sha256": valid_sha,
           "target_table_sha256": target_table_sha}
    return {"validation_mask.npz": blob, "validation_manifest.json": canonical_json(man)}
