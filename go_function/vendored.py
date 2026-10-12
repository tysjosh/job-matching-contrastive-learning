"""Pure helpers vendored from the GO-PPI withholding study (design §13.2).

Copied with attribution, not imported, so this study's code pin covers only its
own directories. Only go_function/tests/test_vendored_parity.py imports the
originals.

Sources (sha256 of the source file at copy time):

- ``jaccard``: go_ppi_domain/withholding_targets.py, function ``jaccard``
  (sha256 567794b5153043ddedc5d6a2205fa91dced7afa67cf4a0fa5a6a03fb9382ade8).
- ``_counts``, ``two_way_draws``, ``_summarise``, ``two_way_bootstrap``,
  ``hierarchical_bootstrap``, ``holm``, ``classify_contrast``, ``ci_excludes_0``:
  scripts/go_withholding/analyze_withholding.py, functions of the same names
  (sha256 0fec71ded3e68a16b03dac29b65f6cbf93c8e8df464ec0a5c33a38d1f0a581d8).

Change from the originals: the module constants they read (``ANALYSIS_SEED``,
``FLOOR``, ``ALPHA``, ``UNSTABLE_SHARE``) are required keyword arguments with no
defaults; go_function.constants holds this study's values. Logic is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np

SCHEME = "two_way_pack_seed"


def jaccard(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _counts(draws: np.ndarray, n: int) -> np.ndarray:
    B = draws.shape[0]
    C = np.zeros((B, n))
    np.add.at(C, (np.repeat(np.arange(B), draws.shape[1]), draws.ravel()), 1.0)
    return C


def two_way_draws(n_packs: int, n_seeds: int, B: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per replicate: n_packs pack indices and one seed multiset (shared by all packs)."""
    rng = np.random.default_rng(seed)
    pk = rng.integers(0, n_packs, size=(B, n_packs))
    sd = rng.integers(0, n_seeds, size=(B, n_seeds))
    return _counts(pk, n_packs), _counts(sd, n_seeds)


def _summarise(T: np.ndarray, keep: np.ndarray, B: int, estimate: Optional[float],
               n_defined: int, *, unstable_share: float) -> Dict[str, Any]:
    kept = T[keep]
    n_disc = int(B - keep.sum())
    res = {"estimate": estimate, "ci_low": None, "ci_high": None, "p_boot": None,
           "n_defined_cells": int(n_defined), "n_discarded_replicates": n_disc,
           "n_replicates": int(B), "unstable": bool(n_disc / B > unstable_share) if B else True}
    if len(kept):
        lo, hi = np.percentile(kept, [2.5, 97.5])
        res["ci_low"], res["ci_high"] = float(lo), float(hi)
        res["p_boot"] = float(min(1.0, 2.0 * min(np.mean(kept <= 0), np.mean(kept >= 0))))
    return res


def two_way_bootstrap(X: np.ndarray, unit_pack: Sequence[int], n_packs: int, B: int, *,
                      seed: int, unstable_share: float,
                      strata: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Two-way (pack x seed) cluster bootstrap of the mean over defined cells.

    ``X[u, s]`` (NaN = undefined), ``unit_pack[u]`` the pack index of unit u.
    With ``strata`` (1 high, 0 low, -1 excluded) the statistic is the mean over
    the defined high cells minus the mean over the defined low cells (M_c); a
    replicate without a defined high or low cell is discarded.
    """
    X = np.asarray(X, dtype=float)
    D = (~np.isnan(X)).astype(float)
    Xf = np.nan_to_num(X)
    up = np.asarray(unit_pack, dtype=int)
    Cp, Cs = two_way_draws(n_packs, X.shape[1], B, seed)
    Wu = Cp[:, up]

    def mean_over(mask: np.ndarray):
        Wm = Wu * mask[None, :]
        num = np.einsum("bu,us,bs->b", Wm, Xf, Cs)
        den = np.einsum("bu,us,bs->b", Wm, D, Cs)
        return num, den

    if strata is None:
        num, den = mean_over(np.ones(len(up)))
        keep = den > 0
        T = np.where(keep, num / np.where(keep, den, 1.0), np.nan)
        est = float(np.nanmean(X)) if D.sum() else None
        return {**_summarise(T, keep, B, est, int(D.sum()), unstable_share=unstable_share),
                "scheme": SCHEME}
    st = np.asarray(strata, dtype=int)
    hi, lo = (st == 1).astype(float), (st == 0).astype(float)
    nh, dh = mean_over(hi)
    nl, dl = mean_over(lo)
    keep = (dh > 0) & (dl > 0)
    T = np.where(keep, nh / np.where(dh > 0, dh, 1.0) - nl / np.where(dl > 0, dl, 1.0), np.nan)
    Xh, Xl = X[st == 1], X[st == 0]
    est = (float(np.nanmean(Xh) - np.nanmean(Xl))
           if (~np.isnan(Xh)).any() and (~np.isnan(Xl)).any() else None)
    n_def = int((~np.isnan(Xh)).sum() + (~np.isnan(Xl)).sum())
    return {**_summarise(T, keep, B, est, n_def, unstable_share=unstable_share),
            "scheme": SCHEME}


def hierarchical_bootstrap(X: np.ndarray, B: int, *, seed: int, unstable_share: float,
                           strata: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """S1 (v1 scheme): resample targets, then seeds within each drawn target."""
    X = np.asarray(X, dtype=float)
    U, S = X.shape
    rng = np.random.default_rng(seed)
    ui = rng.integers(0, U, size=(B, U))
    sd = rng.integers(0, S, size=(B, U, S))
    C = np.zeros((B, U, S))
    np.add.at(C, (np.repeat(np.arange(B), U * S), np.tile(np.repeat(np.arange(U), S), B),
                  sd.ravel()), 1.0)
    Xd = X[ui]                                  # B x U x S
    Dd = (~np.isnan(Xd)).astype(float)
    Xf = np.nan_to_num(Xd)

    def mean_over(mask_units: Optional[np.ndarray]):
        w = C if mask_units is None else C * mask_units[ui][:, :, None]
        return (w * Xf).sum(axis=(1, 2)), (w * Dd).sum(axis=(1, 2))

    if strata is None:
        num, den = mean_over(None)
        keep = den > 0
        T = np.where(keep, num / np.where(keep, den, 1.0), np.nan)
        est = float(np.nanmean(X)) if (~np.isnan(X)).any() else None
        return {**_summarise(T, keep, B, est, int((~np.isnan(X)).sum()),
                             unstable_share=unstable_share),
                "scheme": "hierarchical_target_seed"}
    st = np.asarray(strata, dtype=int)
    nh, dh = mean_over((st == 1).astype(float))
    nl, dl = mean_over((st == 0).astype(float))
    keep = (dh > 0) & (dl > 0)
    T = np.where(keep, nh / np.where(dh > 0, dh, 1.0) - nl / np.where(dl > 0, dl, 1.0), np.nan)
    Xh, Xl = X[st == 1], X[st == 0]
    est = (float(np.nanmean(Xh) - np.nanmean(Xl))
           if (~np.isnan(Xh)).any() and (~np.isnan(Xl)).any() else None)
    return {**_summarise(T, keep, B, est, int((~np.isnan(Xh)).sum() + (~np.isnan(Xl)).sum()),
                         unstable_share=unstable_share),
            "scheme": "hierarchical_target_seed"}


def holm(pvals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Holm step-down adjustment within one family; ``None`` p stays ``None``
    (counted in the family size, ordered last)."""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (pvals[k] is None, pvals[k] if pvals[k] is not None
                                         else 1.0, k))
    out: Dict[str, Optional[float]] = {}
    running = 0.0
    for i, k in enumerate(order):
        p = pvals[k]
        if p is None:
            out[k] = None
            continue
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def classify_contrast(c: Dict[str, Any], *, floor: float, alpha: float) -> str:
    """Protocol ordered first-match rules; the unstable check comes first."""
    if c.get("unstable"):
        return "inconclusive"
    est, lo, hi, p = c.get("estimate"), c.get("ci_low"), c.get("ci_high"), c.get("p_holm")
    if est is None or lo is None or hi is None:
        return "inconclusive"
    if hi < 0 and est <= -floor:
        return "harm"
    if p is not None and p < alpha and lo > 0 and est >= floor:
        return "supported"
    if -floor < lo and hi < floor:
        return "practically null"
    if lo > 0 and est < floor:
        return "below floor"
    return "inconclusive"


def ci_excludes_0(c: Dict[str, Any]) -> bool:
    lo, hi = c.get("ci_low"), c.get("ci_high")
    return lo is not None and hi is not None and (lo > 0 or hi < 0)
