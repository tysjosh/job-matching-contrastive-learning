"""Validation-only search with anchor-grouped cross-fitting.

Each variant is scored two ways on the published validation split, per checkpoint:

* ``insample``: tune on all of validation and read the same validation (the
  published procedure; optimistic, logged for reference).
* ``cv`` (the selection criterion): 5 anchor-disjoint folds. For each fold the
  variant is prepared (z stats, stacker fit) and its parameter selected on the
  other 4 folds, then scored on the held-out fold's anchors. Every anchor gets
  exactly one out-of-fold score list, so flexible variants (2-D grids,
  stackers) pay for their tuning and are comparable with 1-parameter ones.

Delta = variant - text on the same checkpoint and the same anchors. Text is
fold-independent (z is monotone), so its per-anchor metrics are computed once.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from go_ppi_fusion import guard
from go_ppi_fusion import metrics as M
from go_ppi_fusion.fusion import Inputs, make_family

N_FOLDS = 5
FOLD_SEED = 20261012
REPORT = ("auc_hard", "auc_pooled", "auc_easy", "average_precision", "ndcg@10",
          "precision@10", "reciprocal_rank")


def anchor_folds(qi: M.QueryIndex, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Fold id per query (np.unique order), from a fixed permutation of the anchors."""
    perm = np.random.default_rng(seed).permutation(qi.Q)
    folds = np.empty(qi.Q, dtype=int)
    folds[perm] = np.arange(qi.Q) % n_folds
    return folds


def selection_value(pq: Dict[str, np.ndarray], qmask: np.ndarray, metric: str,
                    scores: np.ndarray, grades: np.ndarray, rmask: np.ndarray) -> float:
    """Inner selection metric over the FIT anchors (qmask) / FIT records (rmask)."""
    if metric == "global_pooled_auc":
        return M.global_auc(scores[rmask], grades[rmask])
    m = M.macro(pq, qmask)
    if metric == "composite":
        return M.composite(m)
    return m[metric]


def _argmax_first(values: Sequence[float]) -> int:
    """First maximiser (ties -> earliest theta = smaller weight), NaN never wins."""
    best, arg = -np.inf, 0
    for i, v in enumerate(values):
        if v is not None and np.isfinite(v) and v > best:
            best, arg = v, i
    return arg


def evaluate_variant(spec: Dict[str, Any], texts: List[np.ndarray], inp: Inputs,
                     groups: Optional[List[List[int]]] = None,
                     folds: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Cross-fitted and in-sample validation metrics of one variant over all checkpoints.

    ``groups`` lists checkpoint indices that share one parameter choice
    (spec["scope"]): per checkpoint (default), per fraction, or global.
    """
    guard.check_split(inp.split)
    if inp.split != "validation":
        raise guard.TestAccessError("search runs on validation only")
    fam = make_family(spec)
    thetas = fam.thetas()
    qi, grades = inp.qi, inp.grades
    folds = anchor_folds(qi) if folds is None else folds
    rec_fold = folds[qi.inverse]
    n_ck = len(texts)
    groups = groups or [[i] for i in range(n_ck)]
    metric = spec.get("select_on", "ndcg@10")
    all_q = np.ones(qi.Q, bool)
    all_r = np.ones(inp.n, bool)

    # per checkpoint: [setting][theta] -> per-query metrics; setting = fold k or "all"
    settings = list(range(N_FOLDS)) + ["all"]
    sel = np.zeros((n_ck, len(settings), len(thetas)))
    states: Dict[tuple, Any] = {}
    need_auc = metric in ("composite", "auc_pooled", "auc_hard")
    for c, text in enumerate(texts):
        for si, s in enumerate(settings):
            fit_r = all_r if s == "all" else rec_fold != s
            fit_q = all_q if s == "all" else folds != s
            state = fam.prepare(text, inp, fit_r)
            states[(c, si)] = state
            for ti, th in enumerate(thetas):
                scores = fam.score(state, th, text, inp)
                pq = M.per_query(scores, qi, contrasts=need_auc)
                sel[c, si, ti] = selection_value(pq, fit_q, metric, scores, grades, fit_r)

    pq_cache: Dict[tuple, Dict[str, np.ndarray]] = {}

    def pqs_get(key):
        c, si, ti = key
        if key not in pq_cache:
            scores = fam.score(states[(c, si)], thetas[ti], texts[c], inp)
            pq_cache[key] = M.per_query(scores, qi)
        return pq_cache[key]

    text_pq = [M.per_query(t, qi) for t in texts]
    rows = []
    for g in groups:
        for si, s in enumerate(settings):
            ti = _argmax_first(sel[g, si, :].mean(0))
            for c in g:
                rows.append((c, si, ti))
    chosen = {(c, si): ti for c, si, ti in rows}

    per_ck = []
    for c in range(n_ck):
        oof = {k: np.full(qi.Q, np.nan) for k in text_pq[c]}
        for si, s in enumerate(settings[:-1]):
            pq = pqs_get((c, si, chosen[(c, si)]))
            m = folds == s
            for k in oof:
                oof[k][m] = pq[k][m]
        ins = pqs_get((c, len(settings) - 1, chosen[(c, len(settings) - 1)]))
        tm, om, im = M.macro(text_pq[c]), M.macro(oof), M.macro(ins)
        per_ck.append({
            "theta_all": _json_theta(thetas[chosen[(c, len(settings) - 1)]]),
            "theta_folds": [_json_theta(thetas[chosen[(c, si)]]) for si in range(N_FOLDS)],
            "text": {k: tm[k] for k in REPORT},
            "cv_delta": {k: om[k] - tm[k] for k in REPORT},
            "insample_delta": {k: im[k] - tm[k] for k in REPORT},
        })
    summary = {}
    for kind in ("cv_delta", "insample_delta"):
        summary[kind] = {k: float(np.mean([r[kind][k] for r in per_ck])) for k in REPORT}
        summary[kind]["composite"] = float(np.mean(
            [M.composite({k: r[kind][k] for k in M.COMPOSITE}) for r in per_ck]))
        summary[kind + "_n_positive_composite"] = int(sum(
            M.composite({k: r[kind][k] for k in M.COMPOSITE}) > 0 for r in per_ck))
    summary["text"] = {k: float(np.mean([r["text"][k] for r in per_ck])) for k in REPORT}
    return {"summary": summary, "per_checkpoint": per_ck, "n_thetas": len(thetas)}


def final_fit(spec: Dict[str, Any], texts: List[np.ndarray], inp: Inputs,
              groups: Optional[List[List[int]]] = None) -> List[Dict[str, Any]]:
    """Freeze a variant on ALL of validation (for the later test-scoring step).

    Returns one {state, theta} per checkpoint: z stats / stacker fit on all
    validation records, theta = first maximiser of the inner metric over all
    validation anchors (averaged within each scope group). Same selection rule
    as the ``all`` setting of :func:`evaluate_variant`.
    """
    if inp.split != "validation":
        raise guard.TestAccessError("variants are frozen on validation only")
    fam = make_family(spec)
    thetas = fam.thetas()
    metric = spec.get("select_on", "ndcg@10")
    all_q, all_r = np.ones(inp.qi.Q, bool), np.ones(inp.n, bool)
    groups = groups or [[i] for i in range(len(texts))]
    states, sel = [], np.zeros((len(texts), len(thetas)))
    for c, text in enumerate(texts):
        st = fam.prepare(text, inp, all_r)
        states.append(st)
        for ti, th in enumerate(thetas):
            s = fam.score(st, th, text, inp)
            sel[c, ti] = selection_value(M.per_query(s, inp.qi), all_q, metric, s,
                                         inp.grades, all_r)
    out: List[Dict[str, Any]] = [None] * len(texts)  # type: ignore[list-item]
    for g in groups:
        ti = _argmax_first(sel[g, :].mean(0))
        for c in g:
            out[c] = {"state": states[c], "theta": thetas[ti]}
    return out


def final_score(spec: Dict[str, Any], fitted: Dict[str, Any], text: np.ndarray,
                inp: Inputs) -> np.ndarray:
    """Score any split (e.g. test in the report step) with a frozen variant."""
    return make_family(spec).score(fitted["state"], fitted["theta"], text, inp)


def _json_theta(th):
    if th is None:
        return None
    if isinstance(th, tuple):
        return [float(x) for x in th]
    return float(th)


def scope_groups(scope: str, grid: Sequence[Dict[str, Any]]) -> List[List[int]]:
    if scope == "checkpoint":
        return [[i] for i in range(len(grid))]
    if scope == "fraction":
        fr = sorted({r["fraction"] for r in grid})
        return [[i for i, r in enumerate(grid) if r["fraction"] == f] for f in fr]
    if scope == "global":
        return [list(range(len(grid)))]
    raise ValueError(scope)
