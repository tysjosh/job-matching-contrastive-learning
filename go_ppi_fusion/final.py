"""Report-step helpers: per-anchor deltas, the two-level bootstrap, the shuffled-GO
derangement and the protein-disjoint CV sensitivity.

* The bootstrap is the trials estimator (``trials_domain.fusion_confirm``,
  imported read-only): one replicate draws the 15 checkpoints and the anchors
  independently with replacement, the statistic is the mean over drawn anchors
  of the mean over drawn checkpoints of Delta[c, a]. The draws are made once
  and reused for every variant and contrast (common random numbers).
* The shuffled-GO control is the closure-size-decile, candidate-excluding
  anchor-side derangement of protocol_go_ppi_fusion.md section 2.3 (the trials
  whole-file derangement is infeasible on GO). The receiver takes the donor
  anchor's whole GO profile (pruned BP record set, and MF/CC through the donor
  gene); partner sides are unchanged.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from go_ppi_fusion import metrics as M
from go_ppi_fusion import search as S
from go_ppi_fusion.data import Split
from go_ppi_fusion.fusion import Inputs, make_family

B = 2000
BOOT_SEED = 20260101
SHUFFLE_SEED = 11
N_STRATA = 10
MAX_DRAWS = 10_000
SIM_FORBID = 0.5
METRICS = ("auc_hard", "auc_pooled", "auc_easy", "average_precision", "ndcg@10",
           "precision@10", "reciprocal_rank")


def _fc():
    from trials_domain import fusion_confirm as fc
    return fc


# ------------------------------------------------------------------ per-anchor deltas
def per_anchor(scores: Sequence[np.ndarray], qi: M.QueryIndex) -> Dict[str, np.ndarray]:
    """``{metric: array (n_checkpoints, n_anchors)}``; NaN where undefined."""
    rows = [M.per_query(s, qi) for s in scores]
    return {k: np.stack([r[k] for r in rows]) for k in METRICS}


def deltas(variant: Dict[str, np.ndarray], base: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: variant[k] - base[k] for k in METRICS}


def draws(n_ck: int, n_anchors: int, seed: int = BOOT_SEED, b: int = B):
    return _fc().draw_indices(np.random.default_rng(seed), n_ck, n_anchors, b)


def summarize(D: np.ndarray, C: np.ndarray, P: np.ndarray) -> Dict[str, Any]:
    """Point, 95% CI, two-sided bootstrap p, defined anchors, checkpoints with Delta > 0."""
    s = _fc().summarize(D, C, P)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_ck = np.nanmean(D, axis=1)
    return {"point": s["point"], "ci95": list(s["ci95"]), "p": s["p"],
            "n_anchors": s["n_defined"], "n_ck_positive": int((per_ck > 0).sum()),
            "per_checkpoint": [float(x) for x in per_ck]}


def summarize_all(Ds: Dict[str, np.ndarray], C, P, cols: Optional[np.ndarray] = None,
                  metrics: Sequence[str] = METRICS) -> Dict[str, Any]:
    out = {}
    for k in metrics:
        D = Ds[k] if cols is None else Ds[k][:, cols]
        out[k] = summarize(D, C, P)
    return out


# ------------------------------------------------------------------ shuffled-GO derangement
def _anchor_profile(split: Split):
    """Per query (np.unique order): anchor set, candidates, candidate partner sets."""
    queries = np.unique(np.asarray(split.queries, dtype=object))
    pos = {q: i for i, q in enumerate(queries)}
    own: List[Optional[frozenset]] = [None] * len(queries)
    cands: List[set] = [set() for _ in queries]
    psets: List[set] = [set() for _ in queries]
    for q, a, p, ua, ub in zip(split.queries, split.anchors, split.partners,
                               split.pruned_uris_a, split.pruned_uris_b):
        i = pos[q]
        if a != q:
            raise AssertionError(f"anchor protein {a} != query id {q}")
        fs = frozenset(ua)
        if own[i] is None:
            own[i] = fs
        elif own[i] != fs:
            raise AssertionError(f"anchor-side set differs across records of {q}")
        cands[i].add(p)
        psets[i].add(frozenset(ub))
    return queries, own, cands, psets


def derange_anchors(split: Split, matcher, seed: int = SHUFFLE_SEED) -> Dict[str, Any]:
    """Closure-size-decile, candidate-excluding derangement of anchor-side GO sets.

    ``matcher`` is a simGIC ``GoMatcher`` on the pruned BP index (closures and
    the similarity rule (e)). Returns ``{"mapping": {receiver: donor}, ...}``.
    """
    queries, own, cands, psets = _anchor_profile(split)
    units = [i for i in range(len(queries)) if own[i]]
    size = {i: len(matcher._closure(own[i])) for i in units}
    units.sort(key=lambda i: (size[i], queries[i]))
    rng = np.random.default_rng(seed)
    mapping: Dict[str, str] = {}
    log = []
    near = 0
    for stratum, members in enumerate(np.array_split(np.asarray(units, dtype=int), N_STRATA)):
        n = len(members)
        forbid = np.zeros((n, n), dtype=bool)
        for r in range(n):
            a = members[r]
            for d in range(n):
                b = members[d]
                forbid[r, d] = (
                    a == b                                     # (a) itself
                    or queries[b] in cands[a]                  # (b) a candidate of a
                    or own[b] == own[a]                        # (c) same set
                    or own[b] in psets[a]                      # (d) a candidate's partner set
                    or matcher.ontology_set_similarity(own[a], own[b]) >= SIM_FORBID)  # (e)
        for k in range(1, MAX_DRAWS + 1):
            order = rng.permutation(n)
            donors = np.roll(order, -1)
            if not forbid[order, donors].any():
                break
        else:
            raise RuntimeError(f"derangement aborted: stratum {stratum} > {MAX_DRAWS} draws")
        for r, d in zip(order, donors):
            mapping[str(queries[members[r]])] = str(queries[members[d]])
            near += sum(matcher.ontology_set_similarity(own[members[d]], ps) >= SIM_FORBID
                        for ps in psets[members[r]] if ps)
        log.append({"stratum": stratum, "size": int(n), "draws": int(k),
                    "closure_size_range": [int(size[members[0]]), int(size[members[-1]])]})
    return {"mapping": mapping, "strata": log, "n_units": len(units),
            "n_kept_empty": int(len(queries) - len(units)),
            "n_receiver_partner_sets_with_donor_simgic_ge_0.5": int(near), "seed": seed}


def deranged_split(split: Split, mapping: Dict[str, str]) -> Split:
    """Copy of ``split`` whose receivers carry the donor anchor's GO profile.

    The pruned and unpruned BP record sets are taken from the donor's records and
    ``anchors`` is set to the donor gene (MF/CC phi look up ``index.direct`` by it).
    Queries, partners, grades and text are unchanged.
    """
    first: Dict[str, int] = {}
    for i, q in enumerate(split.queries):
        first.setdefault(q, i)
    anchors, ua, pua = list(split.anchors), list(split.uris_a), list(split.pruned_uris_a)
    for i, q in enumerate(split.queries):
        d = mapping.get(q)
        if d is None:
            continue
        j = first[d]
        anchors[i] = split.anchors[j]
        ua[i] = list(split.uris_a[j])
        pua[i] = list(split.pruned_uris_a[j])
    return dataclasses.replace(split, anchors=anchors, uris_a=ua, pruned_uris_a=pua,
                               meta={**split.meta, "deranged": True})


# ------------------------------------------------------------------ protein-disjoint CV
def protein_disjoint_masks(split: Split, qi: M.QueryIndex, folds: np.ndarray) -> List[np.ndarray]:
    """FIT mask per held-out fold: other folds' records that share no protein with it."""
    rec_fold = folds[qi.inverse]
    anchors = np.asarray(split.anchors, dtype=object)
    partners = np.asarray(split.partners, dtype=object)
    out = []
    for s in range(S.N_FOLDS):
        held = rec_fold == s
        prot = set(anchors[held]) | set(partners[held])
        touch = np.isin(anchors, list(prot)) | np.isin(partners, list(prot))
        out.append((rec_fold != s) & ~touch)
    return out


def cv_with_fit_masks(spec: Dict[str, Any], texts: List[np.ndarray], inp: Inputs,
                      folds: np.ndarray, fit_masks: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Cross-fitted validation Delta with a caller-given FIT record mask per fold.

    Same procedure as ``search.evaluate_variant`` (checkpoint scope only): prepare
    and select theta on the FIT records / other-fold anchors, score the held-out
    fold's anchors. With ``fit_masks[s] = rec_fold != s`` it equals the logged CV.
    """
    if inp.split != "validation":
        raise PermissionError("CV runs on validation only")
    if spec.get("scope", "checkpoint") != "checkpoint":
        raise ValueError("checkpoint scope only")
    fam = make_family(spec)
    thetas = fam.thetas()
    metric = spec.get("select_on", "ndcg@10")
    qi = inp.qi
    per_ck = []
    for text in texts:
        oof = None
        for s in range(S.N_FOLDS):
            fit_r, fit_q = fit_masks[s], folds != s
            state = fam.prepare(text, inp, fit_r)
            vals, pqs = [], []
            for th in thetas:
                sc = fam.score(state, th, text, inp)
                pq = M.per_query(sc, qi)
                pqs.append(pq)
                vals.append(S.selection_value(pq, fit_q, metric, sc, inp.grades, fit_r))
            pq = pqs[S._argmax_first(vals)]
            if oof is None:
                oof = {k: np.full(qi.Q, np.nan) for k in pq}
            m = folds == s
            for k in oof:
                oof[k][m] = pq[k][m]
        tm, om = M.macro(M.per_query(text, qi)), M.macro(oof)
        per_ck.append({k: om[k] - tm[k] for k in S.REPORT})
    summ = {k: float(np.mean([r[k] for r in per_ck])) for k in S.REPORT}
    summ["composite"] = float(np.mean([M.composite(r) for r in per_ck]))
    return {"cv_delta": summ, "per_checkpoint": per_ck}
