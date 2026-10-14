"""Metrics (protocol §7).

* class-centric rank AUC per column, float64, average ranks for ties
  (``scipy.stats.rankdata``); undefined (n_pos in {0, n}) -> NaN;
* class-centric AUPR (average precision, ``sklearn`` semantics: one step per
  distinct score);
* n_train bands {0, 1-10, 11-30, 31-100, 101-1000, >1000} and summaries;
* protein-centric: a wrapper over ``go_function.evaluate.protein_centric`` and the
  per-protein threshold table (precision, recall, ru, mi, has-prediction per
  protein and threshold) from which Fmax and Smin are re-maximised for any
  resample of proteins (§8);
* the pilot's own Fmax (pilot-compat reproduction only).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

BANDS = (("0", 0, 0), ("1-10", 1, 10), ("11-30", 11, 30), ("31-100", 31, 100),
         ("101-1000", 101, 1000), (">1000", 1001, None))
BAND_NAMES = tuple(b[0] for b in BANDS)
PILOT_BAND_KEYS = ("0-0", "1-10", "11-30", "31-100", "101-1000", "1001-1000000000")
THRESHOLDS = tuple(t / 100.0 for t in range(0, 101))


# ------------------------------------------------------------------ class-centric
def rank_auc(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-column AUC with average-rank ties, float64."""
    S = np.asarray(S, dtype=np.float64)
    Y = np.asarray(Y, dtype=bool)
    if S.ndim == 1:
        S, Y = S[:, None], Y[:, None]
    rk = rankdata(S, axis=0)
    npos = Y.sum(axis=0).astype(np.float64)
    nneg = Y.shape[0] - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((rk * Y).sum(axis=0) - npos * (npos + 1) / 2) / (npos * nneg)
    auc[(npos == 0) | (nneg == 0)] = np.nan
    return auc


def rank_auc_pilot(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The pilot's AUC: identical formula, ranks of the input dtype (pilot-compat)."""
    rk = rankdata(S, axis=0)
    npp = Y.sum(0)
    nn = Y.shape[0] - npp
    return ((rk * Y).sum(0) - npp * (npp + 1) / 2) / (npp * nn)


def average_precision(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-column average precision (= ``sklearn.metrics.average_precision_score``)."""
    S = np.asarray(S, dtype=np.float64)
    Y = np.asarray(Y, dtype=bool)
    if S.ndim == 1:
        S, Y = S[:, None], Y[:, None]
    out = np.full(S.shape[1], np.nan)
    n = S.shape[0]
    for j in range(S.shape[1]):
        y = Y[:, j]
        npos = int(y.sum())
        if npos == 0 or npos == n:
            continue
        o = np.argsort(-S[:, j], kind="mergesort")
        s, ys = S[o, j], y[o]
        ends = np.r_[np.flatnonzero(np.diff(s) != 0), n - 1]
        tps = np.cumsum(ys)[ends].astype(np.float64)
        prec = tps / (ends + 1)
        rec = tps / npos
        out[j] = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return out


# ------------------------------------------------------------------ bands
def band_index(ntr: np.ndarray) -> np.ndarray:
    """Band position 0..5 of each n_train value."""
    ntr = np.asarray(ntr)
    out = np.empty(len(ntr), dtype=np.int64)
    for i, (_, lo, hi) in enumerate(BANDS):
        m = (ntr >= lo) & ((ntr <= hi) if hi is not None else True)
        out[m] = i
    return out


def summary(values: np.ndarray, ntr_cols: np.ndarray, reg_mask: Optional[np.ndarray] = None,
            base: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Mean over terms, per band (mean, n), reg subset, share improved vs ``base``."""
    v = np.asarray(values, dtype=np.float64)
    bi = band_index(ntr_cols)
    out: Dict[str, Any] = {"mean": float(np.nanmean(v)) if len(v) else None, "n": int(len(v))}
    for i, name in enumerate(BAND_NAMES):
        m = bi == i
        out[name] = {"mean": float(np.nanmean(v[m])) if m.any() else None, "n": int(m.sum())}
    if reg_mask is not None:
        out["reg_subset"] = {"mean": float(np.nanmean(v[reg_mask])) if reg_mask.any() else None,
                             "n": int(reg_mask.sum())}
    if base is not None:
        d = v - np.asarray(base, dtype=np.float64)
        out["delta_vs_base"] = float(np.nanmean(d))
        out["share_terms_improved"] = float(np.mean(d > 0))
    return out


# ------------------------------------------------------------------ protein-centric
def protein_centric(probs, y: sp.spmatrix, ic: np.ndarray, device: str = "cpu",
                    propagate=None) -> Dict[str, Any]:
    """``go_function.evaluate.protein_centric`` (Fmax, Smin, protein/micro AUPR)."""
    from go_function.evaluate import protein_centric as pc
    return pc(probs, y, ic, device=device, propagate=propagate)


def protein_tables(probs: np.ndarray, y: sp.spmatrix, ic: np.ndarray,
                   row_chunk: int = 256) -> Dict[str, np.ndarray]:
    """Per-protein, per-threshold statistics (T x n) over proteins with >= 1 label
    in the term set: precision (0 where no prediction), recall, ru, mi and the
    has-prediction indicator (>= 1 score >= threshold), as DGZ's Fmax averages
    precision only over proteins with a prediction."""
    y = sp.csr_matrix(y)
    lab_n = np.asarray(y.sum(axis=1)).ravel().astype(np.float64)
    keep = np.flatnonzero(lab_n > 0)
    icv = np.asarray(ic, dtype=np.float64)
    th = np.asarray(THRESHOLDS)
    T, n = len(th), len(keep)
    prec = np.zeros((T, n)); rec = np.zeros((T, n)); ru = np.zeros((T, n)); mi = np.zeros((T, n))
    has = np.zeros((T, n), dtype=bool)
    for a in range(0, n, row_chunk):
        r = keep[a:a + row_chunk]
        s = np.asarray(probs[r], dtype=np.float64)
        yy = y[r].toarray().astype(bool)
        ln, lic = lab_n[r], yy.astype(np.float64) @ icv
        for k in range(T):
            P = s >= th[k]
            tp = (P & yy).sum(1).astype(np.float64)
            pn = P.sum(1).astype(np.float64)
            tic = (P & yy).astype(np.float64) @ icv
            pic = P.astype(np.float64) @ icv
            h = pn > 0
            sl = slice(a, a + len(r))
            prec[k, sl] = np.divide(tp, pn, out=np.zeros_like(tp), where=h)
            rec[k, sl] = tp / ln
            ru[k, sl] = lic - tic
            mi[k, sl] = pic - tic
            has[k, sl] = h
    return {"precision": prec, "recall": rec, "ru": ru, "mi": mi, "has_prediction": has,
            "rows": keep}


def _summarise(prec: np.ndarray, rec: np.ndarray, ru: np.ndarray, mi: np.ndarray) -> Dict[str, float]:
    """Fmax / Smin over thresholds (``go_function.evaluate._summarise_curve`` rule)."""
    fmax, tmax, smin = 0.0, 0.0, 1000000.0
    for k, t in enumerate(THRESHOLDS):
        p, r = prec[k], rec[k]
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if fmax < f:
            fmax, tmax = f, t
        s = math.sqrt(ru[k] ** 2 + mi[k] ** 2)
        if smin > s:
            smin = s
    return {"fmax": float(fmax), "tmax": float(tmax), "smin": float(smin)}


def fmax_smin(tab: Mapping[str, np.ndarray], weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Fmax and Smin from a protein table; ``weights`` = per-protein multiplicities
    of a bootstrap resample (None: every protein once)."""
    n = tab["recall"].shape[1]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    W = w.sum()
    rec = tab["recall"] @ w / W
    ru = tab["ru"] @ w / W
    mi = tab["mi"] @ w / W
    hw = tab["has_prediction"] @ w
    prec = np.divide((tab["precision"] * tab["has_prediction"]) @ w, hw,
                     out=np.zeros(len(hw)), where=hw > 0)
    return _summarise(prec, rec, ru, mi)


def fmax_boot(tab: Mapping[str, np.ndarray], W: np.ndarray) -> np.ndarray:
    """Fmax for every replicate of a (B x n) multiplicity matrix, vectorised."""
    Wt = np.asarray(W, dtype=np.float64)
    tot = Wt.sum(axis=1, keepdims=True)
    rec = Wt @ tab["recall"].T / tot
    hw = Wt @ tab["has_prediction"].T.astype(np.float64)
    pw = Wt @ (tab["precision"] * tab["has_prediction"]).T
    prec = np.divide(pw, hw, out=np.zeros_like(pw), where=hw > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return f.max(axis=1)


def smin_boot(tab: Mapping[str, np.ndarray], W: np.ndarray) -> np.ndarray:
    Wt = np.asarray(W, dtype=np.float64)
    tot = Wt.sum(axis=1, keepdims=True)
    ru = Wt @ tab["ru"].T / tot
    mi = Wt @ tab["mi"].T / tot
    return np.sqrt(ru ** 2 + mi ** 2).min(axis=1)


def pilot_fmax(S: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """The pilot ``b_valid_pilot.py`` Fmax (thresholds 0.01..0.99), pilot-compat only."""
    best = (0.0, 0.0)
    ny = Y.sum(1)
    for t in np.arange(0.01, 1.0, 0.01):
        pr = S >= t
        tp = (pr & Y).sum(1)
        npr = pr.sum(1)
        has = npr > 0
        if not has.any():
            continue
        p = (tp[has] / npr[has]).mean()
        r = (tp / ny).mean()
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        if f > best[0]:
            best = (float(f), float(t))
    return {"fmax": round(best[0], 4), "tmax": round(best[1], 2)}
