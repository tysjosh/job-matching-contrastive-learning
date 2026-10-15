"""Neighbour tables and priors (protocol §3, §3.5, §4.3).

Neighbour sets are lists of vocabulary columns per vocabulary column, built from
the pinned ``go.obo`` with ``go_function.ontology`` and ``go_function.labels.Vocab``:

=========  ==========================================================================
key        set
=========  ==========================================================================
pa_isa     columns of the is_a parents (same namespace)
pa_tp      Pa_TP: is_a + part_of parents (same namespace)
pa_any     Pa_any: parents by any ``relationship:`` or is_a edge (same namespace)
reg        Reg: targets of regulates / positively_ / negatively_regulates edges
sib        Sib: is_a siblings with >= ``go_function.fusion.MIN_SIB_POS`` TP train
           positives (confirmatory: ``go_function.ontology.isa_siblings``)
ch         direct TP children (pilot ``ch_max``)
anc        strict TP ancestors
desc       Desc: strict TP descendants
=========  ==========================================================================

Modes (§3): ``pilot`` removes only the root column (exactly the pilot
``c_struct.py``; its Sib does not remove descendants); ``confirmatory`` also removes
alt-id duplicate columns and every column of t's own primary id, matching
``go_function.targets.channel_sets``. A column without a primary id has every set
empty.

Also here: the shuffled-graph permutation (n_train deciles, seed 20261013), the
co-annotation sets CoA / CoA⁺, DiamondScore via ``go_function.baselines`` with the
shuffled-hits control (seed 20261017), the Mac-only identity proxy H_sim, and the
text-neighbour prior stub (out of scope for S1, §3.6).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from go_function_fusion import guard

TRUE_PATH = ("is_a", "part_of")
REG_FAMILY = ("regulates", "positively_regulates", "negatively_regulates")
KEYS = ("pa_isa", "pa_tp", "pa_any", "reg", "sib", "ch", "anc", "desc")
SHUFFLE_GRAPH_SEED = 20261013
SHUFFLE_HITS_SEED = 20261017

Tables = Dict[str, List[List[int]]]


# ------------------------------------------------------------------ neighbour tables
def neighbour_tables(ont, vocab, anc: Mapping[str, frozenset], desc: Mapping[str, frozenset],
                     ntr: np.ndarray, root: int, mode: str,
                     keys: Sequence[str] = KEYS) -> Tables:
    """Neighbour sets for every vocabulary column (see module docstring)."""
    from go_function.fusion import MIN_SIB_POS
    from go_function.ontology import _isa_children, isa_siblings
    if mode not in ("confirmatory", "pilot"):
        raise ValueError("mode must be 'confirmatory' or 'pilot'")
    conf = mode == "confirmatory"
    dup = vocab.duplicate_cols
    children = _isa_children(ont)
    tp_children: Dict[str, Set[str]] = {}
    if "ch" in keys:
        for tid, t in ont.terms.items():
            for r, p in t.edges:
                if r in TRUE_PATH and ont.terms[p].namespace == t.namespace:
                    tp_children.setdefault(p, set()).add(tid)
    out: Tables = {k: [] for k in keys}
    for c in range(len(vocab)):
        g = vocab.primary[c]
        if g is None:
            for k in keys:
                out[k].append([])
            continue
        own = set(vocab.primary_to_cols.get(g, ())) if conf else set()

        def cols_of(ids: Iterable[str]) -> List[int]:
            s = {j for x in ids for j in vocab.primary_to_cols.get(x, ()) if j != root}
            if conf:
                s = {j for j in s if j not in dup and j not in own}
            return sorted(s)

        t = ont.terms[g]

        def same(p: str) -> bool:
            return ont.terms[p].namespace == t.namespace

        for k in keys:
            if k == "pa_isa":
                v = cols_of(p for r, p in t.edges if r == "is_a" and same(p))
            elif k == "pa_tp":
                v = cols_of(p for r, p in t.edges if r in TRUE_PATH and same(p))
            elif k == "pa_any":
                v = cols_of(p for r, p in t.edges if same(p))
            elif k == "reg":
                v = cols_of(p for r, p in t.edges if r in REG_FAMILY and same(p))
            elif k == "sib":
                if conf:
                    sib = isa_siblings(ont, g, anc, desc)
                else:                       # pilot c_struct.py: no descendant removal
                    sib = set()
                    for r, p in t.edges:
                        if r == "is_a":
                            sib |= children[p]
                    sib -= {g} | set(anc[g])
                v = [j for j in cols_of(sib) if int(ntr[j]) >= MIN_SIB_POS]
            elif k == "ch":
                v = cols_of(tp_children.get(g, ()))
            elif k == "anc":
                v = cols_of(anc[g] - {g})
            elif k == "desc":
                v = cols_of(desc.get(g, frozenset({g})) - {g})
            else:
                raise KeyError(k)
            out[k].append(v)
    return out


def changed_terms(a: Tables, b: Tables, cols: Sequence[int], keys: Sequence[str]) -> Dict[str, int]:
    """Number of ``cols`` whose set differs between two table builds, per key (G-R(c))."""
    return {k: int(sum(a[k][int(c)] != b[k][int(c)] for c in cols)) for k in keys
            if k in a and k in b}


# ------------------------------------------------------------------ shuffled graph
def decile_permutation(ntr: np.ndarray, seed: int = SHUFFLE_GRAPH_SEED,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Column identities permuted within n_train deciles over all vocabulary
    columns; one ``rng.permutation`` per decile in increasing decile order (§3.5).
    ``rng`` (pilot-compat only) continues a caller's generator."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    ntr = np.asarray(ntr)
    dec = np.searchsorted(np.quantile(ntr, np.linspace(0, 1, 11)[1:-1]), ntr)
    perm = np.arange(len(ntr))
    for b in np.unique(dec):
        idx = np.where(dec == b)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def permute_sets(sets: Sequence[Sequence[int]], perm: np.ndarray) -> List[List[int]]:
    """perm[S(t)] for every t (order kept, duplicates kept as the pilot)."""
    return [[int(perm[j]) for j in v] for v in sets]


def permutation_report(sets: Sequence[Sequence[int]], perm: np.ndarray, cols: Sequence[int],
                       vocab, root: int) -> Dict[str, int]:
    """Counts of (t, pi) pairs with perm[pi] in {root, t} and with perm[pi] an
    alt-id column, over the evaluated ``cols`` (§3.5)."""
    n_pairs = n_root = n_self = n_alt = 0
    for t in cols:
        for p in sets[int(t)]:
            q = int(perm[p])
            n_pairs += 1
            n_root += q == root
            n_self += q == int(t)
            n_alt += q in vocab.alt_cols
    return {"pairs": n_pairs, "perm_is_root": n_root, "perm_is_t": n_self,
            "perm_is_alt_id": n_alt}


# ------------------------------------------------------------------ co-annotation (CoA, CoA⁺)
def relatives_cols(g: str, vocab, anc, desc) -> Set[int]:
    """cols(anc_TP(t)) ∪ cols(desc_TP(t)) (reflexive, so t's own columns included)."""
    out: Set[int] = set()
    for x in set(anc.get(g, ())) | set(desc.get(g, ())):
        out.update(vocab.primary_to_cols.get(x, ()))
    return out


def coannotation_sets(y_train: sp.spmatrix, cols: Sequence[int], pa_tp: Sequence[Sequence[int]],
                      ont, vocab, anc, desc, root: int, namespace: str, plus: bool = False,
                      chunk: int = 256) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """CoA(t) (``plus=False``) or CoA⁺(t) for every t in ``cols`` (§3.5).

    m(t) = max(1, |Pa_TP(t)|) columns c with the largest train Jaccard
    J(t,c) = |P(t) ∩ P(c)| / |P(t) ∪ P(c)| over candidates that are primary,
    non-duplicate, same-namespace columns, c ∉ cols(t) ∪ {root} (∪ cols(anc_TP(t))
    ∪ cols(desc_TP(t)) for CoA), with J > 0; ties by column index; ∅ if
    n_train(t) = 0. Returns (sets, overlap share |S ∩ relatives| / m(t))."""
    Y = sp.csc_matrix(y_train, dtype=np.float64)
    Y.data[:] = 1.0
    n = np.asarray(Y.sum(axis=0)).ravel()
    V = Y.shape[1]
    eligible = np.zeros(V, dtype=bool)
    for c in range(V):
        g = vocab.primary[c]
        eligible[c] = (g is not None and c not in vocab.duplicate_cols and c != root
                       and ont.terms[g].namespace == namespace)
    sets: Dict[int, List[int]] = {}
    share: Dict[int, float] = {}
    cols = [int(c) for c in cols]
    YT = sp.csr_matrix(Y.T)
    for a in range(0, len(cols), chunk):
        blk = cols[a:a + chunk]
        inter = sp.csr_matrix(YT[blk] @ Y)
        for i, t in enumerate(blk):
            g = vocab.primary[t]
            m = max(1, len(pa_tp[t]))
            if g is None or n[t] == 0:
                sets[t], share[t] = [], 0.0
                continue
            row = inter.getrow(i)
            cand, iv = row.indices, row.data
            own = set(vocab.primary_to_cols.get(g, ()))
            rel = relatives_cols(g, vocab, anc, desc)
            excl = own if plus else (own | rel)
            ok = np.array([eligible[c] and c not in excl for c in cand], dtype=bool) \
                if len(cand) else np.zeros(0, dtype=bool)
            cand, iv = cand[ok], iv[ok]
            J = iv / (n[t] + n[cand] - iv)
            pos = J > 0
            cand, J = cand[pos], J[pos]
            order = np.lexsort((cand, -J))[:m]
            chosen = sorted(int(c) for c in cand[order])
            sets[t] = chosen
            share[t] = len(set(chosen) & rel) / m
    return sets, share


# ------------------------------------------------------------------ DiamondScore
def diamond_matrix(path, query_names: Sequence[str], train_names: Sequence[str]) -> sp.csr_matrix:
    """Row-normalised bitscore matrix (queries x train), ``go_function.baselines``."""
    from go_function.baselines import diamond_matrix as dm, read_diamond
    guard.check_path(path)
    return dm(read_diamond(path), query_names, train_names)


def check_train_only(path, query_names: Sequence[str], train_names: Sequence[str]) -> Dict[str, int]:
    """G-D fatal checks: every hit id is a train protein; no query id is a train id."""
    from go_function.baselines import read_diamond
    guard.check_path(path)
    tr = set(train_names)
    q_in_train = sorted(set(query_names) & tr)
    if q_in_train:
        raise ValueError(f"{len(q_in_train)} query ids are train ids, e.g. {q_in_train[:3]}")
    n = bad = 0
    for q, h, _ in read_diamond(path):
        n += 1
        if h not in tr:
            bad += 1
    if bad:
        raise ValueError(f"{bad} of {n} hits are not train proteins")
    return {"n_hits": n, "n_queries": len(query_names)}


def shuffled_train_rows(n_train_rows: int, seed: int = SHUFFLE_HITS_SEED) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_train_rows)


def ds_scores(H: sp.csr_matrix, y_train: sp.spmatrix, cols: Sequence[int],
              shuffled: bool = False, seed: int = SHUFFLE_HITS_SEED) -> np.ndarray:
    """DS (n_query x len(cols), float64) by ``go_function.baselines.diamond_score``;
    ``shuffled``: train label rows permuted (shuffled-hits control)."""
    from go_function.baselines import diamond_score
    y = sp.csr_matrix(y_train)
    if shuffled:
        y = y[shuffled_train_rows(y.shape[0], seed)]
    return diamond_score(H, y, cols)


def sim_matrix(path, query_names: Sequence[str], train_names: Sequence[str]) -> sp.csr_matrix:
    """H_sim (Mac exploratory only, §4.3): identity-weighted kNN weights from
    ``swissprot_exp.sim`` (query, hit, pident), row-normalised; repeated pairs are
    summed (the pilot's ``csr_matrix`` semantics)."""
    guard.check_path(path)
    qi = {n: i for i, n in enumerate(query_names)}
    ti = {n: i for i, n in enumerate(train_names)}
    r_, c_, w_ = [], [], []
    with open(path) as fh:
        for line in fh:
            a, b, p = line.rstrip("\n").split("\t")
            if a in qi and b in ti:
                r_.append(qi[a])
                c_.append(ti[b])
                w_.append(float(p))
    W = sp.csr_matrix((w_, (r_, c_)), shape=(len(qi), len(ti)))
    tot = np.asarray(W.sum(1)).ravel()
    return sp.csr_matrix(sp.diags(np.divide(1, tot, out=np.zeros_like(tot), where=tot > 0)) @ W)


def dgz_blast_preds(query_names: Sequence[str], res_path, source_names: Sequence[str],
                    source_annots: Sequence[Iterable[str]], terms: Sequence[str], dgz_rule
                    ) -> np.ndarray:
    """DGZ ``predict_diamond.py`` blast_preds (DGZ-faithful check, §6.3): transfer
    source = train + valid ``prop_annotations``; score(go) = sum of bitscores of hits
    annotated with go / total bitscore; then DGZ-rule ancestor max-propagation
    (``DgzRule.closure``); a repeated (query, hit) keeps the last bitscore."""
    guard.check_path(res_path)
    scores: Dict[str, Dict[str, float]] = {}
    with open(res_path) as fh:
        for line in fh:
            it = line.strip().split()
            scores.setdefault(it[0], {})[it[1]] = float(it[2])
    idx = {n: i for i, n in enumerate(source_names)}
    ann = [set(a) for a in source_annots]
    tindex = {g: i for i, g in enumerate(terms)}
    out = np.zeros((len(query_names), len(terms)), dtype=np.float32)
    for r, q in enumerate(query_names):
        sims = scores.get(q)
        if not sims:
            continue
        total = sum(sims.values())
        annots: Dict[str, float] = {}
        allgos = sorted(set().union(*(ann[idx[p]] for p in sims)))
        for go in allgos:
            annots[go] = np.float32(sum(s for p, s in sims.items() if go in ann[idx[p]]) / total)
        prop = dict(annots)
        for go, s in annots.items():
            for sup in dgz_rule.closure(go):
                prop[sup] = max(prop[sup], s) if sup in prop else s
        for go, s in prop.items():
            j = tindex.get(go)
            if j is not None:
                out[r, j] = s
    return out


# ------------------------------------------------------------------ all confirmatory sets
def build_sets(ctx, cols: Sequence[int], namespace: str) -> Tuple[Dict[str, List[List[int]]], Dict]:
    """Every confirmatory-mode set used by :class:`go_function_fusion.fuse.Space`
    (``ctx`` = ``data.AspectContext``) and the §7 / §3.5 counts over ``cols``."""
    keys = ("pa_tp", "pa_any", "reg", "sib", "desc")
    t = neighbour_tables(ctx.ont, ctx.vocab, ctx.anc, ctx.desc, ctx.ntr, ctx.root,
                         "confirmatory", keys)
    perm = decile_permutation(ctx.ntr)
    sets: Dict[str, List[List[int]]] = dict(t)
    sets["pa_shuf"] = permute_sets(t["pa_tp"], perm)
    sets["reg_shuf"] = permute_sets(t["reg"], perm)
    V = len(ctx.vocab)
    coa, _ = coannotation_sets(ctx.y_train, cols, t["pa_tp"], ctx.ont, ctx.vocab, ctx.anc,
                               ctx.desc, ctx.root, namespace, plus=False)
    coap, share = coannotation_sets(ctx.y_train, cols, t["pa_tp"], ctx.ont, ctx.vocab, ctx.anc,
                                    ctx.desc, ctx.root, namespace, plus=True)
    sets["coa"] = [list(coa.get(c, ())) for c in range(V)]
    sets["coap"] = [list(coap.get(c, ())) for c in range(V)]
    cols = [int(c) for c in cols]
    info = {"n_empty": {k: int(sum(not sets[k][c] for c in cols))
                        for k in ("pa_tp", "reg", "coa", "coap", "sib", "desc", "pa_any")},
            "perm_report_pa": permutation_report(t["pa_tp"], perm, cols, ctx.vocab, ctx.root),
            "perm_report_reg": permutation_report(t["reg"], perm, cols, ctx.vocab, ctx.root),
            "coap_overlap_share": np.array([share.get(c, 0.0) for c in cols])}
    return sets, info


# ------------------------------------------------------------------ text neighbours (stub)
def text_neighbour_sets(*_args, **_kwargs):
    """Term-text (MPNet) neighbour prior: out of scope for S1 (§3.6, spec §1(d));
    needs GF G5 term embeddings. Not implemented by design."""
    raise NotImplementedError("text-neighbour prior is out of scope for S1 (protocol §3.6)")
