"""Target selection, strata, packing and retention orders (design §4, protocol §4; gap 7).

The algorithm, in order (design §4 steps 1-10):

1. base filter (primary column, not root / protein binding / binding, 30 <= n_train
   <= 5% N_train, n_test >= 10 on the evaluation set) with the alt-id exclusion;
2. S_sib(t) = |union of P_train(u) over is_a siblings u in V, minus P_train(t)|;
3. BP: the LD flag, filler support n_train(F) = |{p in train : F in closure_TP(p)}|
   computed from the closure whether or not F is a column (v4 NIT);
4. tertile strata of S_sib, middle excluded;
5. n bands B1 30-99, B2 100-249, B3 250+ with per-cell quotas (BP 4, MF 8);
6.-7. ``accept_round_robin`` with borrowing B1->B2->B3->B1; shortfall = exit 3;
8. SMD balance report;
9. packing 8 x 6 (two-hop, term-level D*^GO_TP and D*^GO_DGZ disjointness,
   pack union <= 10% N_train); no feasible pack = exit 3;
10. retention orders (draw 1, draw 2, and the MF noIPI order).

Only ``jaccard`` is vendored (go_function.vendored). Target names are written to
the table but never logged; every log line carries counts only.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from go_function.constants import (BINDING, NAMESPACES, PROTEIN_BINDING, RETENTION_SEED,
                                   RETENTION_SEED_2, ROOTS)
from go_function.labels import Vocab
from go_function.ontology import (DgzRule, Ontology, ancestors, descendants, isa_siblings,
                                  logical_definition, parents, two_hop)
from go_function.runtime import FatalError, ProtocolStop
from go_function.vendored import jaccard

logger = logging.getLogger("go_function.targets")

SCHEMA = 1
REG_FAMILY = ("regulates", "positively_regulates", "negatively_regulates")
Slot = Tuple[str, str]          # (cell, band)


@dataclass(frozen=True)
class TargetParams:
    """Protocol §4 values; tests override them for small fixtures."""
    min_train: int = 30
    max_train_share: float = 0.05
    min_test: int = 10
    min_filler: int = 30
    bands: Tuple[Tuple[str, int, Optional[int]], ...] = (("B1", 30, 99), ("B2", 100, 249),
                                                          ("B3", 250, None))
    quota: Optional[int] = None          # per cell per band; None -> BP 4, MF 8
    n_packs: int = 8
    pack_size: int = 6
    max_jaccard: float = 0.2
    max_pack_union_share: float = 0.10
    retention_seed: int = RETENTION_SEED
    retention_seed_2: int = RETENTION_SEED_2

    def quota_for(self, aspect: str) -> int:
        if self.quota is not None:
            return self.quota
        return {"bp": 4, "mf": 8}[aspect]


@dataclass
class TargetInputs:
    """Everything the builder reads (loaded by build_targets.py, or built in tests)."""
    aspect: str
    ont: Ontology
    dgz: DgzRule
    vocab: Vocab
    tp_train: sp.csr_matrix
    tp_test: sp.csr_matrix
    eval_test: np.ndarray
    tp_direct_train: sp.csr_matrix
    tp_extra_train: Sequence[Tuple[str, ...]]
    dgz_train: sp.csr_matrix
    dgz_raw_ids: FrozenSet[str]          # every raw id of A_DGZ(p) over all splits
    proteins_train: Sequence[str]
    noipi_train: Optional[sp.csr_matrix] = None
    noipi_test: Optional[sp.csr_matrix] = None
    noipi_eval_test: Optional[np.ndarray] = None
    inputs: Mapping[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------ small helpers
def seed_for(seed: int, t: str) -> int:
    """seed_t = int(sha256(f"{seed}:{t}")[:16], 16) (design §4.10)."""
    return int(hashlib.sha256(f"{seed}:{t}".encode()).hexdigest()[:16], 16)


def retention_order(rows: Iterable[int], proteins: Sequence[str], t: str, seed: int) -> List[int]:
    """P_train(t) sorted by protein name, permuted by default_rng(seed_t)."""
    srt = sorted((int(r) for r in rows), key=lambda r: (proteins[r], r))
    perm = np.random.default_rng(seed_for(seed, t)).permutation(len(srt))
    return [srt[i] for i in perm]


def column_rows(m: sp.spmatrix) -> List[np.ndarray]:
    """Positive row indices of every column (sorted)."""
    csc = sp.csc_matrix(m)
    csc.sort_indices()
    return [csc.indices[csc.indptr[j]:csc.indptr[j + 1]].astype(np.int64)
            for j in range(csc.shape[1])]


def band_of(n: int, bands: Sequence[Tuple[str, int, Optional[int]]]) -> Optional[str]:
    for name, lo, hi in bands:
        if n >= lo and (hi is None or n <= hi):
            return name
    return None


def tertile_cuts(values: Sequence[float]) -> Tuple[float, float]:
    """33.33rd and 66.67th percentiles (numpy linear interpolation)."""
    q = np.percentile(np.asarray(values, dtype=np.float64), [100.0 / 3.0, 200.0 / 3.0])
    return float(q[0]), float(q[1])


def stratum(s: float, cuts: Tuple[float, float]) -> Optional[str]:
    """low = S <= q1, high = S >= q2 (low wins a tie q1 == q2), middle -> None."""
    if s <= cuts[0]:
        return "low"
    if s >= cuts[1]:
        return "high"
    return None


def smd(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Standardised mean difference (mean_a - mean_b) / sqrt((var_a + var_b) / 2)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def tp_closure_counts(direct: sp.csr_matrix, extra: Sequence[Tuple[str, ...]],
                      vocab_ids: Sequence[str], anc: Mapping[str, FrozenSet[str]],
                      root: str) -> Counter:
    """n_train(F) for every GO id F: proteins whose TP closure contains F (root removed)."""
    direct = sp.csr_matrix(direct)
    cnt: Counter = Counter()
    for i in range(direct.shape[0]):
        a = [vocab_ids[c] for c in direct.indices[direct.indptr[i]:direct.indptr[i + 1]]]
        a.extend(extra[i])
        clo: Set[str] = set()
        for g in a:
            clo |= anc[g]
        clo.discard(root)
        cnt.update(clo)
    return cnt


def primary_noduplicate_cols(vocab: Vocab) -> List[int]:
    """Columns with a live primary that are not duplicate columns (evaluate.py rule)."""
    return [c for c in range(len(vocab))
            if vocab.primary[c] is not None and c not in vocab.duplicate_cols]


def dgz_descendant_sets(dgz: DgzRule, universe: Iterable[str],
                        targets: Iterable[str]) -> Dict[str, FrozenSet[str]]:
    """Term-level D*^GO_DGZ(t) = {a in universe : t in DgzRule.closure(a)} (design §5.1)."""
    tset = set(targets)
    out: Dict[str, Set[str]] = {t: set() for t in tset}
    for a in sorted(set(universe)):
        for t in tset.intersection(dgz.closure(a)):
            out[t].add(a)
    return {t: frozenset(s) for t, s in out.items()}


def alt_excluded(t: str, ont: Ontology, vocab: Vocab, raw_ids: FrozenSet[str]) -> bool:
    """§4.1 alt-id exclusion: t has an alt-id column (any extra column with primary t)
    or an alt id of t occurs in any protein's raw ``exp_annotations``."""
    if len(vocab.primary_to_cols.get(t, ())) != 1:
        return True
    return any(a in vocab.index or a in raw_ids for a in ont.terms[t].alt_ids)


def ld_flag(ont: Ontology, t: str, n_clo: Mapping[str, int], min_filler: int
            ) -> Tuple[Optional[str], List[List[Any]]]:
    """§4 step 3: ("no", []) without intersection_of; ("yes", supported fillers) with a
    genus and >= 1 differentia filler F with n_train(F) >= min_filler (closure count,
    columns or not); (None, []) otherwise (excluded)."""
    genus, diff = logical_definition(ont, t)
    if genus is None and not diff:
        return "no", []
    sup = sorted([rel, f, int(n_clo.get(f, 0))] for rel, f in diff
                 if n_clo.get(f, 0) >= min_filler)
    if genus is not None and sup:
        return "yes", sup
    return None, []


# ------------------------------------------------------------------ acceptance (§4.6-4.7)
class Shortfall(ProtocolStop):
    pass


def accept_round_robin(slots: Mapping[Slot, Sequence[str]], quotas: Mapping[Slot, int],
                       P_train: Mapping[str, FrozenSet[int]],
                       ancestors: Mapping[str, FrozenSet[str]], max_jaccard: float,
                       *, visits: Optional[List[Tuple[Slot, Optional[str]]]] = None) -> List[str]:
    """Greedy round-robin acceptance (design §4.6-4.7).

    ``slots`` maps (cell, band) to its candidates in preference order; its
    insertion order is the visit order (cells as listed, then bands). Each visit
    accepts at most one candidate. A candidate is accepted if, against every
    accepted u, neither is an ancestor of the other and Jaccard(P_train) <=
    ``max_jaccard``. An exhausted slot passes its open quota to the next band of
    the same cell (B1->B2->B3->B1, skipping exhausted slots); if none is left the
    cell is short and ``Shortfall`` (exit 3) is raised with counts only.
    ``visits`` (optional) records (slot, accepted id or None) for tests.
    """
    order = list(slots)
    cells: Dict[str, List[Slot]] = {}
    for s in order:
        cells.setdefault(s[0], []).append(s)
    remaining = {s: int(quotas.get(s, 0)) for s in order}
    ptr = {s: 0 for s in order}
    exhausted: Set[Slot] = set()
    accepted: List[str] = []

    def ok(c: str) -> bool:
        for u in accepted:
            if u in ancestors[c] or c in ancestors[u]:
                return False
            if jaccard(set(P_train[c]), set(P_train[u])) > max_jaccard:
                return False
        return True

    while any(remaining[s] > 0 for s in order):
        for s in order:
            if remaining[s] <= 0:
                continue
            cands = slots[s]
            found: Optional[str] = None
            while ptr[s] < len(cands):
                c = cands[ptr[s]]
                ptr[s] += 1
                if ok(c):
                    found = c
                    break
            if found is not None:
                accepted.append(found)
                remaining[s] -= 1
                if visits is not None:
                    visits.append((s, found))
                continue
            if visits is not None:
                visits.append((s, None))
            exhausted.add(s)
            open_q, remaining[s] = remaining[s], 0
            ring = cells[s[0]]
            i = ring.index(s)
            nxt = next((ring[(i + j) % len(ring)] for j in range(1, len(ring))
                        if ring[(i + j) % len(ring)] not in exhausted), None)
            if nxt is None:
                total = sum(int(quotas.get(x, 0)) for x in ring)
                got = sum(1 for a in accepted if any(a in slots[x] for x in ring))
                raise Shortfall(f"target shortfall in cell {s[0]}: {got} of {total} accepted, "
                                f"{len(ring)} slots exhausted")
            remaining[nxt] += open_q
    return accepted


# ------------------------------------------------------------------ packing (§4.9)
def placement_order(targets: Iterable[str], P_tp: Mapping[str, FrozenSet[int]]) -> List[str]:
    """Amendment A1: decreasing n_train(t) = |P_TP(t)|, ties by GO id (train counts only)."""
    return sorted(targets, key=lambda t: (-len(P_tp[t]), t))


def target_set_sha256(targets: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(targets)).encode()).hexdigest()


def pack_targets(order: Sequence[str], cell_of: Mapping[str, str],
                 two_hop_sets: Mapping[str, FrozenSet[str]],
                 dstar_tp: Mapping[str, FrozenSet[str]], dstar_dgz: Mapping[str, FrozenSet[str]],
                 P_tp: Mapping[str, FrozenSet[int]], n_train_total: int, *, n_packs: int,
                 pack_size: int, max_union_share: float) -> Dict[str, List[str]]:
    """Deterministic greedy packing (design §4.9 as amended by A1): targets are placed
    in ``placement_order``; candidate packs are ranked by (same-cell count, |union
    P_TP| after adding t, pack id); the first pack meeting every constraint is used.
    ``order`` is the selected target set (its order is not used). ``ProtocolStop``
    (exit 3) if a target fits no pack."""
    names = [f"p{i + 1:02d}" for i in range(n_packs)]
    packs: Dict[str, List[str]] = {n: [] for n in names}
    unions: Dict[str, Set[int]] = {n: set() for n in names}
    limit = max_union_share * n_train_total
    order = placement_order(order, P_tp)
    for t in order:
        cands = sorted(names, key=lambda n: (sum(1 for u in packs[n] if cell_of[u] == cell_of[t]),
                                             len(unions[n] | P_tp[t]), n))
        placed = False
        for n in cands:
            mates = packs[n]
            if len(mates) >= pack_size:
                continue
            if any(t in two_hop_sets[u] or u in two_hop_sets[t] for u in mates):
                continue
            if any(dstar_tp[t] & dstar_tp[u] or dstar_dgz[t] & dstar_dgz[u] for u in mates):
                continue
            if len(unions[n] | P_tp[t]) > limit:
                continue
            mates.append(t)
            unions[n] |= P_tp[t]
            placed = True
            break
        if not placed:
            done = sum(len(v) for v in packs.values())
            raise ProtocolStop(f"no feasible pack: {done} of {len(order)} targets packed")
    return packs


# ------------------------------------------------------------------ channel sets (§7.3)
def channel_sets(ont: Ontology, dgz: DgzRule, vocab: Vocab, t: str,
                 anc: Mapping[str, FrozenSet[str]], desc: Mapping[str, FrozenSet[str]],
                 par: Mapping[str, FrozenSet[str]], root: str,
                 eligible: FrozenSet[int]) -> Dict[str, List[str]]:
    """X(t), X_reg(t), Desc(t), Pa(t) and the is_a siblings as column ids (primary,
    non-duplicate columns != t)."""
    ids = vocab.ids

    def cols_of(go_ids: Iterable[str]) -> Set[int]:
        out: Set[int] = set()
        for g in go_ids:
            out.update(c for c in vocab.primary_to_cols.get(g, ()) if c in eligible)
        return out

    t_cols = set(vocab.primary_to_cols.get(t, ()))
    clo_dgz = dgz.closure(t)
    x = {c for c in eligible if ids[c] in clo_dgz and vocab.primary[c] not in anc[t]} - t_cols
    reg_reach: Set[str] = set()
    for rel, q in ont.terms[t].edges:
        if rel in REG_FAMILY:
            reg_reach |= dgz.closure(q)
    _, diff = logical_definition(ont, t)
    reg_fill = {cls for rel, cls in diff if rel in REG_FAMILY}
    x_reg = {c for c in x if ids[c] in reg_reach or vocab.primary[c] in reg_fill}
    dsc = cols_of(desc[t] - {t}) - t_cols
    pa = cols_of(par[t] - {root}) - t_cols
    sib = cols_of(isa_siblings(ont, t, anc, desc)) - t_cols

    def srt(cs: Set[int]) -> List[str]:
        return sorted(ids[c] for c in cs)
    return {"X": srt(x), "X_reg": srt(x_reg), "Desc": srt(dsc), "Pa": srt(pa),
            "siblings": srt(sib)}


# ------------------------------------------------------------------ the builder
def build_table(inp: TargetInputs, params: TargetParams = TargetParams()) -> Dict[str, Any]:
    """The canonical target table (design §4 output, schema 1)."""
    aspect = inp.aspect
    ont, vocab, dgz = inp.ont, inp.vocab, inp.dgz
    namespace, root = NAMESPACES[aspect], ROOTS[aspect]
    n_train_total = int(inp.tp_train.shape[0])
    anc = ancestors(ont)
    desc = descendants(anc)
    par = parents(ont)
    pos_rows = column_rows(inp.tp_train)
    test_eval = sp.csr_matrix(inp.tp_test)[np.asarray(inp.eval_test, dtype=np.int64)]
    n_test_col = np.asarray(test_eval.sum(axis=0)).ravel().astype(np.int64)
    n_train_col = np.asarray([len(r) for r in pos_rows], dtype=np.int64)
    max_train = params.max_train_share * n_train_total
    raw_ids = inp.dgz_raw_ids

    # ---- step 1: base filter, with the alt-id exclusion
    counts: Dict[str, int] = {}
    primary_cols = [c for c in range(len(vocab)) if vocab.ids[c] in ont.terms
                    and ont.terms[vocab.ids[c]].namespace == namespace
                    and vocab.ids[c] not in (root, PROTEIN_BINDING, BINDING)]
    counts["primary_columns"] = len(primary_cols)
    in_range = [c for c in primary_cols if params.min_train <= n_train_col[c] <= max_train]
    counts["n_train_in_range"] = len(in_range)

    def has_alt(c: int) -> bool:
        return alt_excluded(vocab.ids[c], ont, vocab, raw_ids)
    alt_excl = [c for c in in_range if has_alt(c)]
    counts["alt_id_excluded_before_n_test"] = len(alt_excl)
    test_ok = [c for c in in_range if n_test_col[c] >= params.min_test]
    counts["n_test_ok"] = len(test_ok)
    counts["alt_id_excluded_after_n_test"] = sum(1 for c in test_ok if has_alt(c))
    base = [vocab.ids[c] for c in test_ok if not has_alt(c)]
    counts["base_filter"] = len(base)
    col_of = {t: vocab.index[t] for t in base}
    P = {t: frozenset(int(r) for r in pos_rows[col_of[t]]) for t in base}

    # ---- step 2: support
    def sib_support(t: str) -> int:
        rows: List[np.ndarray] = []
        for u in isa_siblings(ont, t, anc, desc):
            cs = vocab.primary_to_cols.get(u, ())
            if cs:
                rows.append(pos_rows[cs[0]])
        if not rows:
            return 0
        u_rows = np.unique(np.concatenate(rows))
        return int(len(np.setdiff1d(u_rows, pos_rows[col_of[t]], assume_unique=True)))
    s_sib = {t: sib_support(t) for t in base}

    # ---- step 3: logical definition (BP only)
    ld: Dict[str, Optional[str]] = {}
    fillers: Dict[str, List[List[Any]]] = {}
    if aspect == "bp":
        n_clo = tp_closure_counts(inp.tp_direct_train, inp.tp_extra_train, vocab.ids, anc, root)
        for t in base:
            ld[t], fillers[t] = ld_flag(ont, t, n_clo, params.min_filler)
        counts["ld_excluded"] = sum(1 for t in base if ld[t] is None)
        cands = [t for t in base if ld[t] is not None]
    else:
        cands = list(base)
    counts["after_ld"] = len(cands)

    # ---- step 4: strata
    cuts = tertile_cuts([s_sib[t] for t in cands]) if cands else (0.0, 0.0)
    strat = {t: stratum(s_sib[t], cuts) for t in cands}
    counts["middle_tertile_excluded"] = sum(1 for t in cands if strat[t] is None)
    cands = [t for t in cands if strat[t] is not None]

    def cell(t: str) -> str:
        return f"{strat[t]}_LD{ld[t]}" if aspect == "bp" else strat[t]
    cell_names = (["low_LDyes", "low_LDno", "high_LDyes", "high_LDno"] if aspect == "bp"
                  else ["low", "high"])

    # ---- step 5: bands
    band = {t: band_of(int(n_train_col[col_of[t]]), params.bands) for t in cands}
    band_names = [b[0] for b in params.bands]
    slots: Dict[Slot, List[str]] = {}
    for cn in cell_names:
        for bn in band_names:
            members = [t for t in cands if cell(t) == cn and band[t] == bn]
            members.sort(key=lambda t: (-int(n_test_col[col_of[t]]), t))
            slots[(cn, bn)] = members
    candidate_counts = {f"{cn}/{bn}": len(v) for (cn, bn), v in slots.items()}
    quota = params.quota_for(aspect)
    quotas = {s: quota for s in slots}
    logger.info("%s: base %d, candidates %d, per slot %s", aspect, len(base), len(cands),
                candidate_counts)

    # ---- steps 6-7: acceptance (Shortfall -> exit 3)
    accepted = accept_round_robin(slots, quotas, P, anc, params.max_jaccard)
    n_expected = quota * len(slots)
    if len(accepted) != n_expected:
        raise FatalError(f"accepted {len(accepted)} targets, expected {n_expected}")

    # ---- P_TP subset P_DGZ for every target (design §5.1; fatal otherwise)
    dgz_rows = column_rows(inp.dgz_train)
    P_dgz = {t: frozenset(int(r) for r in dgz_rows[col_of[t]]) for t in accepted}
    bad = sum(1 for t in accepted if not P[t] <= P_dgz[t])
    if bad:
        raise FatalError(f"P_TP(t) is not a subset of P_DGZ(t) for {bad} targets")

    # ---- step 8: balance report
    def logs(ts: Sequence[str], arr: np.ndarray) -> List[float]:
        return [math.log(int(arr[col_of[t]])) for t in ts]
    hi = [t for t in accepted if strat[t] == "high"]
    lo = [t for t in accepted if strat[t] == "low"]
    balance: Dict[str, Any] = {
        "S_high_vs_low": {"smd_log_n_train": smd(logs(hi, n_train_col), logs(lo, n_train_col)),
                          "smd_log_n_test": smd(logs(hi, n_test_col), logs(lo, n_test_col)),
                          "n_high": len(hi), "n_low": len(lo)}}
    if aspect == "bp":
        ly = [t for t in accepted if ld[t] == "yes"]
        ln = [t for t in accepted if ld[t] == "no"]
        balance["LD_yes_vs_no"] = {
            "smd_log_n_train": smd(logs(ly, n_train_col), logs(ln, n_train_col)),
            "smd_log_n_test": smd(logs(ly, n_test_col), logs(ln, n_test_col)),
            "n_yes": len(ly), "n_no": len(ln)}

    # ---- step 9: packing
    universe = set(dgz.known_ids()) | set(raw_ids)
    dstar_dgz = dgz_descendant_sets(dgz, universe, accepted)
    dstar_tp = {t: desc[t] for t in accepted}
    th = {t: two_hop(ont, t, anc, desc) for t in accepted}
    cell_of = {t: cell(t) for t in accepted}
    sha_before = target_set_sha256(accepted)
    packs = pack_targets(accepted, cell_of, th, dstar_tp, dstar_dgz, P, n_train_total,
                         n_packs=params.n_packs, pack_size=params.pack_size,
                         max_union_share=params.max_pack_union_share)
    packed = [t for ts in packs.values() for t in ts]
    sha_after = target_set_sha256(packed)
    # A1: packing changes placement only, never the selected set (fatal otherwise).
    if (sorted(packed) != sorted(accepted) or len(set(packed)) != len(packed)
            or sha_before != sha_after
            or any(len(ts) != params.pack_size for ts in packs.values())):
        raise FatalError("packing changed the selected target set or left a pack incomplete")
    pack_of = {t: n for n, ts in packs.items() for t in ts}
    pack_info = {n: {"targets": list(ts),
                     "union_P_TP": len(set().union(*(P[t] for t in ts))) if ts else 0,
                     "union_P_DGZ": len(set().union(*(P_dgz[t] for t in ts))) if ts else 0,
                     "cells": dict(sorted(Counter(cell_of[t] for t in ts).items())),
                     "bands": dict(sorted(Counter(band[t] for t in ts).items()))}
                 for n, ts in packs.items()}

    # ---- step 10: retention orders, noIPI support; channel sets
    noipi_rows = column_rows(inp.noipi_train) if inp.noipi_train is not None else None
    noipi_test = None
    if inp.noipi_test is not None:
        nt = sp.csr_matrix(inp.noipi_test)[np.asarray(inp.noipi_eval_test, dtype=np.int64)]
        noipi_test = np.asarray(nt.sum(axis=0)).ravel().astype(np.int64)
    eligible = frozenset(primary_noduplicate_cols(vocab))
    targets: Dict[str, Any] = {}
    for t in accepted:
        c = col_of[t]
        ch = channel_sets(ont, dgz, vocab, t, anc, desc, par, root, eligible)
        entry: Dict[str, Any] = {
            "id": t, "name": ont.terms[t].name, "column": c, "cell": cell_of[t],
            "stratum": strat[t], "band": band[t], "pack": pack_of[t],
            "n_train": int(n_train_col[c]), "n_test": int(n_test_col[c]),
            "n_train_dgz": len(P_dgz[t]), "S_sib": s_sib[t],
            "LD": ld.get(t), "LD_fillers": fillers.get(t, []),
            "n_dstar_go_tp": len(dstar_tp[t]), "n_dstar_go_dgz": len(dstar_dgz[t]),
            "channels": ch, "channel_sizes": {k: len(v) for k, v in ch.items()},
            "retention": {
                "draw1": retention_order(P[t], inp.proteins_train, t, params.retention_seed),
                "draw2": retention_order(P[t], inp.proteins_train, t, params.retention_seed_2)},
        }
        if noipi_rows is not None:
            rows = noipi_rows[c]
            n_tr, n_te = int(len(rows)), int(noipi_test[c])
            kept = n_tr >= params.min_train and n_te >= params.min_test
            entry["noipi"] = {"n_train": n_tr, "n_test": n_te, "kept": kept,
                              "order": retention_order(rows, inp.proteins_train, t,
                                                       params.retention_seed)}
        targets[t] = entry
    noipi_summary = None
    if noipi_rows is not None:
        noipi_summary = {"n_kept": sum(1 for e in targets.values() if e["noipi"]["kept"]),
                         "n_dropped": sum(1 for e in targets.values() if not e["noipi"]["kept"]),
                         "rule": "drop if n_train < min_train or n_test < min_test on TPnoIPI "
                                 "train labels and the TPnoIPI evaluation set"}
    return {
        "schema": SCHEMA,
        "aspect": aspect,
        "parameters": {"min_train": params.min_train, "max_train_share": params.max_train_share,
                       "min_test": params.min_test, "min_filler": params.min_filler,
                       "bands": [list(b) for b in params.bands], "quota_per_cell_band": quota,
                       "n_packs": params.n_packs, "pack_size": params.pack_size,
                       "max_jaccard": params.max_jaccard,
                       "max_pack_union_share": params.max_pack_union_share,
                       "retention_seed": params.retention_seed,
                       "retention_seed_2": params.retention_seed_2,
                       "seed_formula": "int(sha256(f'{seed}:{t}')[:16], 16)",
                       "n_train_total": n_train_total,
                       "n_eval_test": int(len(inp.eval_test))},
        "inputs": dict(sorted(inp.inputs.items())),
        "filter_counts": counts,
        "S_sib_cuts": {"q33": cuts[0], "q67": cuts[1]},
        "cells": cell_names,
        "candidate_counts": candidate_counts,
        "acceptance_order": accepted,
        "packing_rule": "A1: targets by (-n_train, id); packs by (same-cell count, "
                        "|union P_TP| after adding, pack id)",
        "target_set_sha256_before_packing": sha_before,
        "target_set_sha256_after_packing": sha_after,
        "balance": balance,
        "packs": pack_info,
        "targets": targets,
        "noipi": noipi_summary,
    }


def summary_counts(table: Mapping[str, Any]) -> Dict[str, Any]:
    """Counts only (safe to log): no target id or name."""
    per_cell = Counter(e["cell"] for e in table["targets"].values())
    per_band = Counter(e["band"] for e in table["targets"].values())
    out = {"n_targets": len(table["targets"]), "n_packs": len(table["packs"]),
           "pack_sizes": sorted(len(p["targets"]) for p in table["packs"].values()),
           "per_cell": dict(sorted(per_cell.items())), "per_band": dict(sorted(per_band.items())),
           "filter_counts": table["filter_counts"],
           "pack_composition": {n: {"cells": p["cells"], "bands": p["bands"],
                                    "union_P_TP": p["union_P_TP"],
                                    "union_P_DGZ": p["union_P_DGZ"]}
                                for n, p in sorted(table["packs"].items())},
           "target_set_unchanged_by_packing": (table["target_set_sha256_before_packing"]
                                               == table["target_set_sha256_after_packing"]),
           "max_pack_union_P_TP": max(p["union_P_TP"] for p in table["packs"].values()),
           "max_pack_union_P_DGZ": max(p["union_P_DGZ"] for p in table["packs"].values())}
    if table.get("noipi"):
        out["noipi"] = {k: table["noipi"][k] for k in ("n_kept", "n_dropped")}
    return out
