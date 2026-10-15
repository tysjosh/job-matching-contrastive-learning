"""Fusion families: how text and GO scores are combined at scoring time.

A family has a parameter grid (``thetas``), a ``prepare`` step that fits
everything learned from data (z statistics, stacker coefficients) on a set of
FIT records, and a ``score`` step that maps any split to fused scores with the
prepared state frozen. ``prepare`` only ever receives validation inputs: the
stacker refuses anything else, and in search mode the test split cannot even be
loaded (``guard``).

Families:
    linear      (1-w) z_text + w z_phi            (published form when missing="zero")
    linear2     (1-w1-w2) z_text + w1 z_phi1 + w2 z_phi2
    coverage    linear with w scaled by min(1, min(n_a, n_b) / c)
    rank        (1-w) r_text + w r_phi, r = within-anchor percentile rank
    rrf         (1-w)/(k + rank_text) + w/(k + rank_phi), within-anchor ranks
    anchor_z    (1-w) z_text + w z_phi with z computed within each anchor
    stacker     logistic regression / gradient boosting / pairwise logistic on
                [z_text, z_phi..., missing flags, log counts, z_text*z_phi]

``missing`` handling for phi: "zero" keeps phi=0 for an empty side (published);
"impute" fits z on annotated FIT records and sets z_phi=0 (the mean) when missing;
"text_only" additionally scores missing records by z_text alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from go_ppi_fusion.metrics import QueryIndex


@dataclass
class Inputs:
    """One split's label-free scoring inputs plus (validation only) its grades."""

    split: str
    queries: np.ndarray
    phis: Dict[str, np.ndarray]
    qi: QueryIndex
    grades: Optional[np.ndarray] = None   # only needed by stackers and selection

    @property
    def n(self) -> int:
        return len(self.queries)


def z_fit(x: np.ndarray, mask: Optional[np.ndarray] = None):
    v = x if mask is None else x[mask]
    if len(v) == 0:
        return 0.0, 1.0
    mean, sd = float(np.mean(v)), float(np.std(v))
    return mean, sd if sd > 0 else 1.0


def z_apply(x, stats):
    return (x - stats[0]) / stats[1]


def _miss_key(phi_name: str) -> str:
    """Name of the missing flag that matches a phi array."""
    parts = phi_name.split("_")
    version = parts[-1]
    if "mean" in parts or "max" in parts:
        return f"miss_all_{version}"
    return f"miss_{parts[-2]}_{version}"


def _count_keys(phi_name: str):
    parts = phi_name.split("_")
    version = parts[-1]
    aspect = "P" if ("mean" in parts or "max" in parts) else parts[-2]
    return f"na_{aspect}_{version}", f"nb_{aspect}_{version}"


def within_query_rank(x: np.ndarray, qi: QueryIndex, pct: bool = True) -> np.ndarray:
    """Average rank within each anchor, ascending (higher score -> higher rank).

    pct=True returns rank / n_in_query; pct=False returns the descending rank
    position (1 = best, average ties) used by RRF.
    """
    from scipy.stats import rankdata
    S = qi.padded(x)
    R = np.where(qi.valid, rankdata(np.where(qi.valid, S, np.inf), axis=1, method="average"), 0.0)
    n = qi.valid.sum(1, keepdims=True)
    out = np.empty(len(x))
    if pct:
        vals = R / n
    else:
        vals = n + 1 - R
    out[qi.idx[qi.valid]] = vals[qi.valid]
    return out


def within_query_z(x: np.ndarray, qi: QueryIndex) -> np.ndarray:
    S = qi.padded(x)
    S = np.where(qi.valid, S, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(S, 1, keepdims=True)
        sd = np.nanstd(S, 1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (S - mu) / sd
    out = np.empty(len(x))
    out[qi.idx[qi.valid]] = Z[qi.valid]
    return out


class Family:
    name = "base"
    n_params = 1

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec

    def thetas(self) -> List[Any]:
        raise NotImplementedError

    def prepare(self, text: np.ndarray, inp: Inputs, fit: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError

    def score(self, state, theta, text: np.ndarray, inp: Inputs) -> np.ndarray:
        raise NotImplementedError


def _grid(spec, default_step=0.1):
    if "grid" in spec:
        return [float(w) for w in spec["grid"]]
    step = spec.get("step", default_step)
    n = int(round(1.0 / step))
    return [round(i * step, 10) for i in range(n + 1)]


class _ZPhi:
    """Shared z(phi) handling for the linear families."""

    @staticmethod
    def stats(inp: Inputs, phi: str, missing: str, fit: np.ndarray):
        x = inp.phis[phi]
        if missing == "zero":
            return z_fit(x, fit)
        miss = inp.phis[_miss_key(phi)] > 0
        return z_fit(x, fit & ~miss)

    @staticmethod
    def apply(inp: Inputs, phi: str, missing: str, stats):
        z = z_apply(inp.phis[phi], stats)
        if missing != "zero":
            z = np.where(inp.phis[_miss_key(phi)] > 0, 0.0, z)
        return z


class Linear(Family):
    name = "linear"

    def thetas(self):
        return _grid(self.spec)

    def prepare(self, text, inp, fit):
        phi, missing = self.spec["phi"], self.spec.get("missing", "zero")
        return {"t": z_fit(text, fit), "p": _ZPhi.stats(inp, phi, missing, fit)}

    def score(self, state, w, text, inp):
        phi, missing = self.spec["phi"], self.spec.get("missing", "zero")
        zt = z_apply(text, state["t"])
        zp = _ZPhi.apply(inp, phi, missing, state["p"])
        fused = (1 - w) * zt + w * zp
        if missing == "text_only":
            fused = np.where(inp.phis[_miss_key(phi)] > 0, zt, fused)
        return fused


class Linear2(Family):
    name = "linear2"
    n_params = 2

    def thetas(self):
        g = _grid(self.spec)
        return [(a, b) for a in g for b in g if a + b <= 1.0 + 1e-9]

    def prepare(self, text, inp, fit):
        missing = self.spec.get("missing", "zero")
        return {"t": z_fit(text, fit),
                "p": [_ZPhi.stats(inp, p, missing, fit) for p in self.spec["phi"]]}

    def score(self, state, theta, text, inp):
        missing = self.spec.get("missing", "zero")
        zt = z_apply(text, state["t"])
        z1, z2 = (_ZPhi.apply(inp, p, missing, s) for p, s in zip(self.spec["phi"], state["p"]))
        a, b = theta
        return (1 - a - b) * zt + a * z1 + b * z2


class Coverage(Linear):
    name = "coverage"
    n_params = 2

    def thetas(self):
        return [(w, c) for w in _grid(self.spec) for c in self.spec.get("c_grid", (2, 5, 10))]

    def score(self, state, theta, text, inp):
        w, c = theta
        phi, missing = self.spec["phi"], self.spec.get("missing", "zero")
        ka, kb = _count_keys(phi)
        cov = np.minimum(1.0, np.minimum(inp.phis[ka], inp.phis[kb]) / c)
        weff = w * cov
        zt = z_apply(text, state["t"])
        zp = _ZPhi.apply(inp, phi, missing, state["p"])
        return (1 - weff) * zt + weff * zp


class Rank(Family):
    name = "rank"

    def thetas(self):
        return _grid(self.spec)

    def prepare(self, text, inp, fit):
        return {}

    def score(self, state, w, text, inp):
        rt = within_query_rank(text, inp.qi)
        rp = within_query_rank(inp.phis[self.spec["phi"]], inp.qi)
        return (1 - w) * rt + w * rp


class RRF(Family):
    name = "rrf"

    def thetas(self):
        return _grid(self.spec)

    def prepare(self, text, inp, fit):
        return {}

    def score(self, state, w, text, inp):
        k = float(self.spec.get("k", 60))
        rt = within_query_rank(text, inp.qi, pct=False)
        rp = within_query_rank(inp.phis[self.spec["phi"]], inp.qi, pct=False)
        return (1 - w) / (k + rt) + w / (k + rp)


class AnchorZ(Family):
    name = "anchor_z"

    def thetas(self):
        return _grid(self.spec)

    def prepare(self, text, inp, fit):
        return {}

    def score(self, state, w, text, inp):
        return ((1 - w) * within_query_z(text, inp.qi)
                + w * within_query_z(inp.phis[self.spec["phi"]], inp.qi))


class Stacker(Family):
    """Learned combiner, fit on validation FIT records only."""

    name = "stacker"
    n_params = 0

    def thetas(self):
        return [None]

    def features(self, state, text, inp) -> np.ndarray:
        zt = z_apply(text, state["t"])
        cols = [zt]
        for p, s in zip(self.spec["phi"], state["p"]):
            zp = _ZPhi.apply(inp, p, "impute", s)
            cols.append(zp)
            if self.spec.get("interactions", True):
                cols.append(zt * zp)
        for k in self.spec.get("flags", []):
            cols.append(inp.phis[k])
        for k in self.spec.get("counts", []):
            cols.append(np.log1p(inp.phis[k]))
        return np.column_stack(cols)

    def prepare(self, text, inp, fit):
        if inp.split != "validation":
            raise PermissionError(f"stacker may only be fit on validation, got {inp.split!r}")
        if inp.grades is None:
            raise ValueError("stacker fit needs validation grades")
        state = {"t": z_fit(text, fit),
                 "p": [_ZPhi.stats(inp, p, "impute", fit) for p in self.spec["phi"]]}
        X = self.features(state, text, inp)
        mu, sd = X[fit].mean(0), X[fit].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        state["mu"], state["sd"] = mu, sd
        Xs = (X - mu) / sd
        model = self.spec["model"]
        y = inp.grades
        if model == "pairwise":
            state["clf"] = _fit_pairwise(Xs[fit], y[fit], inp.queries[fit],
                                         C=self.spec.get("C", 1.0))
        elif model == "lr":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=self.spec.get("C", 1.0), max_iter=2000)
            clf.fit(Xs[fit], (y[fit] == 2).astype(int))
            state["clf"] = clf
        elif model == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(
                max_iter=self.spec.get("max_iter", 150), learning_rate=0.05,
                max_depth=self.spec.get("max_depth", 3), min_samples_leaf=40,
                l2_regularization=1.0, random_state=0)
            clf.fit(Xs[fit], (y[fit] == 2).astype(int))
            state["clf"] = clf
        else:
            raise ValueError(model)
        return state

    def score(self, state, theta, text, inp):
        Xs = (self.features(state, text, inp) - state["mu"]) / state["sd"]
        clf = state["clf"]
        if isinstance(clf, np.ndarray):
            return Xs @ clf
        if hasattr(clf, "decision_function"):
            return clf.decision_function(Xs)
        p = clf.predict_proba(Xs)[:, 1]
        return np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(1 - p, 1e-12, 1))


def _fit_pairwise(X, y, q, C=1.0) -> np.ndarray:
    """Within-anchor pairwise logistic (RankNet-style, linear, no intercept).

    One difference vector per (higher grade, lower grade) pair inside an anchor,
    both orientations, so the fit is symmetric. Returns the weight vector.
    """
    from sklearn.linear_model import LogisticRegression
    diffs = []
    order = np.argsort(q, kind="stable")
    qs = q[order]
    bounds = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        idx = order[s:e]
        g = y[idx]
        hi, lo = np.nonzero(g[:, None] > g[None, :])
        if len(hi):
            diffs.append(X[idx[hi]] - X[idx[lo]])
    D = np.vstack(diffs)
    Xp = np.vstack([D, -D])
    yp = np.r_[np.ones(len(D)), np.zeros(len(D))]
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
    clf.fit(Xp, yp)
    return clf.coef_.ravel()


FAMILIES = {c.name: c for c in (Linear, Linear2, Coverage, Rank, RRF, AnchorZ, Stacker)}


def make_family(spec: Dict[str, Any]) -> Family:
    return FAMILIES[spec["family"]](spec)
