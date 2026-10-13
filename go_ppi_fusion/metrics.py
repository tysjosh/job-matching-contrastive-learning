"""Vectorised within-anchor metrics, numerically equal to the published code.

Definitions follow ``evaluate_publication_fusion.contrast_metrics`` and
``ranking_metrics`` exactly: queries in ``np.unique`` order, ties broken by file
order (stable sort on ``-score``), gains ``2**g - 1``, strict relevance = grade 2,
tie-aware AUC, macro = mean over queries where the metric is finite. The unit
tests check every field against the published functions.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

K = 10
CONTRASTS = {"hard": ((2,), (1,)), "easy": ((2,), (0,)),
             "middle_vs_easy": ((1,), (0,)), "pooled": ((2,), (0, 1))}
RANKING = ("ndcg@10", "precision@10", "recall@10", "average_precision", "reciprocal_rank")
#: The four metrics averaged by the composite criterion (equal weight).
COMPOSITE = ("auc_hard", "auc_pooled", "average_precision", "ndcg@10")


class QueryIndex:
    """Padded (queries x max_len) record index, queries in ``np.unique`` order."""

    def __init__(self, queries: np.ndarray, grades: np.ndarray):
        queries = np.asarray(queries, dtype=object)
        self.queries, inverse = np.unique(queries, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(self.queries))
        self.L = int(counts.max())
        self.Q = len(self.queries)
        idx = np.full((self.Q, self.L), -1, dtype=int)
        fill = np.zeros(self.Q, dtype=int)
        for i, q in enumerate(inverse):          # file order within each query
            idx[q, fill[q]] = i
            fill[q] += 1
        self.idx = idx
        self.valid = idx >= 0
        self.inverse = inverse
        g = np.asarray(grades, dtype=int)
        self.G = np.where(self.valid, g[np.where(self.valid, idx, 0)], -1)
        self.n_strict = (self.G == 2).sum(1)
        ideal = -np.sort(-np.where(self.valid, self.G, -1), axis=1)[:, :K]
        ideal_gain = np.where(ideal >= 0, np.power(2.0, ideal) - 1.0, 0.0)
        disc = np.log2(np.arange(2, ideal.shape[1] + 2))
        self.idcg = (ideal_gain / disc).sum(1)
        self.disc = disc

    def padded(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        return np.where(self.valid, s[np.where(self.valid, self.idx, 0)], -np.inf)


def per_query(scores: np.ndarray, qi: QueryIndex, contrasts: bool = True) -> Dict[str, np.ndarray]:
    """Per-query metric arrays (NaN where undefined)."""
    S = qi.padded(scores)
    order = np.argsort(-S, axis=1, kind="stable")
    R = np.take_along_axis(qi.G, order, axis=1)
    top = R[:, :K]
    gains = np.where(top >= 0, np.power(2.0, top) - 1.0, 0.0)
    dcg = (gains / qi.disc[: top.shape[1]]).sum(1)
    out: Dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ndcg@10"] = np.where(qi.idcg > 0, dcg / np.where(qi.idcg > 0, qi.idcg, 1), np.nan)
        hit = R == 2
        top_hits = hit[:, :K].sum(1)
        out["precision@10"] = top_hits / K
        has = qi.n_strict > 0
        out["recall@10"] = np.where(has, top_hits / np.maximum(qi.n_strict, 1), np.nan)
        cum = np.cumsum(hit, axis=1)
        pos = np.arange(1, R.shape[1] + 1)
        ap = (np.where(hit, cum / pos, 0.0)).sum(1) / np.maximum(qi.n_strict, 1)
        out["average_precision"] = np.where(has, ap, np.nan)
        first = np.argmax(hit, axis=1)
        rr = np.where(hit.any(1), 1.0 / (first + 1), 0.0)
        out["reciprocal_rank"] = np.where(has, rr, np.nan)
    if contrasts:
        nums = {name: np.zeros(qi.Q) for name in CONTRASTS}
        block = max(1, 4_000_000 // (qi.L * qi.L))   # bound the Q x L x L buffer
        for b0 in range(0, qi.Q, block):
            Sb, Gb = S[b0:b0 + block], qi.G[b0:b0 + block]
            win = ((Sb[:, :, None] > Sb[:, None, :]).astype(float)
                   + 0.5 * (Sb[:, :, None] == Sb[:, None, :]))
            for name, (pg, ng) in CONTRASTS.items():
                pm = np.isin(Gb, pg).astype(float)
                nm = np.isin(Gb, ng).astype(float)
                nums[name][b0:b0 + block] = np.einsum("qi,qij,qj->q", pm, win, nm)
        for name, (pg, ng) in CONTRASTS.items():
            den = np.isin(qi.G, pg).sum(1) * np.isin(qi.G, ng).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"auc_{name}"] = np.where(den > 0, nums[name] / np.maximum(den, 1), np.nan)
    return out


def macro(pq: Dict[str, np.ndarray], mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Mean over queries with a finite value (optionally restricted to ``mask``)."""
    out = {}
    for k, v in pq.items():
        vv = v if mask is None else v[mask]
        vv = vv[np.isfinite(vv)]
        out[k] = float(vv.mean()) if len(vv) else float("nan")
    return out


def composite(m: Dict[str, float]) -> float:
    return float(np.mean([m[k] for k in COMPOSITE]))


def global_auc(scores: np.ndarray, grades: np.ndarray, pos=(2,), neg=(0, 1)) -> float:
    """Tie-aware rank AUC across all records (the published ``pooled_auc`` rule)."""
    from scipy.stats import rankdata
    s = np.asarray(scores, dtype=float)
    g = np.asarray(grades)
    pm, nm = np.isin(g, pos), np.isin(g, neg)
    sel = pm | nm
    r = rankdata(s[sel], method="average")
    npos, nneg = int(pm.sum()), int(nm.sum())
    return float((r[pm[sel]].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
