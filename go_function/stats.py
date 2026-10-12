"""Contrast engine of ``scripts/go_function/analyze.py`` (design §12.2, protocol §1,
§8, §9).

The unit of analysis is a (t, s) cell: target t (clustered by its pack) x study
seed s. A contrast is a value function (t, pack, s) -> float | None evaluated on
every cell into a matrix X[u, s] (NaN = undefined) and summarised with the
vendored two-way (pack x seed) cluster bootstrap; preregistered contrasts also get
the S1 hierarchical (target-then-seed) bootstrap. Families are Holm-adjusted per
aspect and classified with the vendored ordered rules (protocol §9.1).

Conventions:
- Delta^small(t, s) = mean over k in {0, 1, 5} of Delta(t, k, s), defined only when
  all three are present (protocol §8).
- Vbest (D11): the arm in {V1, ..., V6} with the largest point estimate of
  Delta^small_{X-V0}; a tie keeps the earlier-listed arm (the first in V1..V6).
- C3 decomposition: on the common cells where all four V4 runs are complete,
  C3 = C3_leak + C3_col + C3_reg is asserted to 1e-12; the Holm-tested C3 uses its
  own two-run cell set and both values are reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from go_function import vendored as V
from go_function.constants import ALPHA, FLOOR, SEEDS, UNSTABLE_SHARE
from go_function.runtime import FatalError

SMALL_KS = (0, 1, 5)
ALL_KS = (0, 1, 5, 10, 25, "all")
ARMS_BEST = ("V1", "V2", "V3", "V4", "V5", "V6")
MAIN_ARMS = ("V0", "V1", "V2", "V3", "V3c", "V4", "V5", "V6")
IDENTITY_TOL = 1e-12
DESC_SMALL = 5

#: (feature, variant, regime, rule, draw)
Arm = Tuple[str, str, str, str, int]
ValueFn = Callable[[str, str, Optional[int]], Optional[float]]


# ------------------------------------------------------------------ units
@dataclass
class Units:
    targets: List[str]
    pack_of: Dict[str, str]
    packs: List[str]
    unit_pack: np.ndarray
    S: np.ndarray                       # 1 high, 0 low, -1 none
    LD: np.ndarray                      # 1 yes, 0 no, -1 none
    desc_n: np.ndarray
    channels: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    noipi_kept: Dict[str, bool] = field(default_factory=dict)

    @property
    def n_packs(self) -> int:
        return len(self.packs)


def units_from_table(table: Mapping[str, Any]) -> Units:
    packs = sorted(table["packs"])
    targets, up = [], []
    for i, p in enumerate(packs):
        for t in table["packs"][p]["targets"]:
            targets.append(t)
            up.append(i)
    info = table["targets"]
    S = np.asarray([1 if str(info[t]["cell"]).split("_")[0] == "high" else
                    0 if str(info[t]["cell"]).split("_")[0] == "low" else -1 for t in targets])
    LD = np.asarray([1 if info[t].get("LD") == "yes" else 0 if info[t].get("LD") == "no" else -1
                     for t in targets])
    desc = np.asarray([len(info[t]["channels"]["Desc"]) for t in targets])
    return Units(targets=targets, pack_of={t: info[t]["pack"] for t in targets}, packs=packs,
                 unit_pack=np.asarray(up, dtype=int), S=S, LD=LD, desc_n=desc,
                 channels={t: {c: list(info[t]["channels"][c]) for c in ("X", "Desc", "Pa")}
                           for t in targets},
                 noipi_kept={t: bool((info[t].get("noipi") or {}).get("kept", True))
                             for t in targets})


# ------------------------------------------------------------------ AUC lookup
def norm_k(k):
    return "all" if str(k) == "all" else int(k)


class AucTable:
    """Long-table metric lookup keyed by (arm, pack, target, k, seed)."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], metric: str = "class_auc"):
        self.v: Dict[tuple, float] = {}
        for r in rows:
            val = r.get(metric)
            if val is None or val == "":
                continue
            seed = r.get("seed")
            seed = None if seed in (None, "") else int(seed)
            key = (r["feature"], r["variant"], r["regime"], r["rule"], int(r["draw"]),
                   r["pack"], r["target"], norm_k(r["k"]), seed)
            self.v[key] = float(val)

    def get(self, arm: Arm, t: str, pack: str, k, s: Optional[int]) -> Optional[float]:
        k = norm_k(k)
        return self.v.get((*arm, "all" if k == "all" else pack, t, k, s))


def v_auc(at: AucTable, arm: Arm, k, seedless: bool = False) -> ValueFn:
    return lambda t, p, s: at.get(arm, t, p, k, None if seedless else s)


def v_diff(f: ValueFn, g: ValueFn) -> ValueFn:
    def h(t, p, s):
        a, b = f(t, p, s), g(t, p, s)
        return None if a is None or b is None else a - b
    return h


def v_small(fn_k: Callable[[Any], ValueFn]) -> ValueFn:
    """Delta^small: mean over k in {0, 1, 5}; undefined unless all three exist."""
    fns = [fn_k(k) for k in SMALL_KS]

    def h(t, p, s):
        vals = [f(t, p, s) for f in fns]
        return None if any(v is None for v in vals) else float(np.mean(vals))
    return h


def matrix(units: Units, fn: ValueFn, seeds: Sequence[int] = SEEDS,
           only: Optional[Callable[[str], bool]] = None) -> np.ndarray:
    X = np.full((len(units.targets), len(seeds)), np.nan)
    for u, t in enumerate(units.targets):
        if only is not None and not only(t):
            continue
        for j, s in enumerate(seeds):
            v = fn(t, units.pack_of[t], s)
            if v is not None:
                X[u, j] = v
    return X


# ------------------------------------------------------------------ estimation
def estimate(X: np.ndarray, units: Units, B: int, seed: int,
             strata: Optional[np.ndarray] = None, s1: bool = False) -> Dict[str, Any]:
    res = V.two_way_bootstrap(X, units.unit_pack, units.n_packs, B, seed=seed,
                              unstable_share=UNSTABLE_SHARE,
                              strata=None if strata is None else list(strata))
    if s1:
        res["S1"] = V.hierarchical_bootstrap(X, B, seed=seed, unstable_share=UNSTABLE_SHARE,
                                             strata=None if strata is None else list(strata))
    res["n_targets_defined"] = int((~np.isnan(X)).any(axis=1).sum())
    return res


def point(X: np.ndarray) -> Optional[float]:
    return float(np.nanmean(X)) if (~np.isnan(X)).any() else None


def classify_family(contrasts: Dict[str, Dict[str, Any]], *, floor: float = FLOOR,
                    alpha: float = ALPHA) -> None:
    """Holm within the family, then the ordered classification (in place)."""
    adj = V.holm({k: c.get("p_boot") for k, c in contrasts.items()})
    for k, c in contrasts.items():
        c["p_holm"] = adj[k]
        c["classification"] = V.classify_contrast(c, floor=floor, alpha=alpha)
        c["ci_excludes_0"] = V.ci_excludes_0(c)


def classify_descriptive(c: Dict[str, Any]) -> Dict[str, Any]:
    """Unadjusted reads: no p_holm, so 'supported' is impossible; harm and the null
    readings remain (used for collateral harm)."""
    c["classification"] = V.classify_contrast({**c, "p_holm": None}, floor=FLOOR, alpha=ALPHA)
    c["ci_excludes_0"] = V.ci_excludes_0(c)
    return c


def choose_vbest(points: Mapping[str, Optional[float]]) -> Optional[str]:
    best, best_v = None, None
    for a in ARMS_BEST:
        v = points.get(a)
        if v is None:
            continue
        if best is None or v > best_v:
            best, best_v = a, v
    return best


# ------------------------------------------------------------------ C3 decomposition
def decomposition(at: AucTable, units: Units, feature: str, B: int, seed: int) -> Dict[str, Any]:
    a = matrix(units, v_auc(at, (feature, "V4", "DGZ", "WDGZrel", 1), 0))
    b = matrix(units, v_auc(at, (feature, "V4", "DGZ", "W2drop", 1), 0))
    c = matrix(units, v_auc(at, (feature, "V4", "DGZ", "W2", 1), 0))
    d = matrix(units, v_auc(at, (feature, "V4", "TP", "W2", 1), 0))
    common = ~(np.isnan(a) | np.isnan(b) | np.isnan(c) | np.isnan(d))

    def on_common(M):
        out = M.copy()
        out[~common] = np.nan
        return out
    c3, leak, col, reg = on_common(a - d), on_common(a - b), on_common(b - c), on_common(c - d)
    resid = np.abs(c3 - (leak + col + reg))[common]
    max_resid = float(resid.max()) if resid.size else 0.0
    if max_resid > IDENTITY_TOL:
        raise FatalError(f"C3 decomposition identity violated: max residual {max_resid:.3e}")
    own = a - d
    own_cells = ~np.isnan(own)
    return {"n_common_cells": int(common.sum()), "n_own_cells": int(own_cells.sum()),
            "identity_max_abs_residual": max_resid,
            "cell_sets_differ": bool((own_cells != common).any()),
            "C3_common": estimate(c3, units, B, seed, s1=True),
            "C3_own": estimate(own, units, B, seed, s1=True),
            "C3_leak": estimate(leak, units, B, seed),
            "C3_col": estimate(col, units, B, seed),
            "C3_reg": estimate(reg, units, B, seed)}


# ------------------------------------------------------------------ channel reads
CHANNELS = ("X", "Desc", "Pa")


def channel_auc(block: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Class AUC for t of the max logit over a channel's columns; None if empty."""
    from go_function.evaluate import class_metrics
    block = np.asarray(block)
    if block.ndim != 2 or block.shape[1] == 0:
        return None
    return class_metrics(block.max(axis=1), y)["auc"]


def channel_reads(wdgz: Mapping[str, np.ndarray], w2drop: Mapping[str, np.ndarray],
                  channels: Mapping[str, Mapping[str, Sequence[str]]],
                  y_by_t: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, Optional[float]]]:
    """Per target and channel: AUC_WDGZrel - AUC_W2drop of the channel max (both V4,
    k = 0, same seed and pack). ``wdgz``/``w2drop`` are scores.npz contents
    (``logits``, ``column_ids``); a target with an empty channel set gives None."""
    def block(sc, ids):
        pos = {str(g): i for i, g in enumerate(np.asarray(sc["column_ids"]).tolist())}
        miss = [g for g in ids if g not in pos]
        if miss:
            raise FatalError(f"scores.npz lacks channel columns {miss[:5]}")
        return np.asarray(sc["logits"])[:, [pos[g] for g in ids]]

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for t, y in y_by_t.items():
        out[t] = {}
        for ch in CHANNELS:
            ids = list(channels[t][ch])
            if not ids:
                out[t][ch] = None
                continue
            a = channel_auc(block(wdgz, ids), y)
            b = channel_auc(block(w2drop, ids), y)
            out[t][ch] = a - b
    return out


def channel_estimates(reads: Mapping[Tuple[str, int], Mapping[str, Optional[float]]],
                      units: Units, B: int, seed: int) -> Dict[str, Any]:
    """C3_leak_X / _desc / _anc over the (t, s) cells with reads; targets whose
    channel set is empty are excluded and counted; _desc also by |Desc| <= 5 / > 5."""
    names = {"X": "C3_leak_X", "Desc": "C3_leak_desc", "Pa": "C3_leak_anc"}
    out: Dict[str, Any] = {}
    for ch, name in names.items():
        fn = lambda t, p, s, ch=ch: (reads.get((t, s)) or {}).get(ch)  # noqa: E731
        X = matrix(units, fn)
        res = estimate(X, units, B, seed)
        res["n_targets_nonempty"] = int(sum(1 for t in units.targets if units.channels[t][ch]))
        res["n_targets_empty"] = int(len(units.targets) - res["n_targets_nonempty"])
        out[name] = res
    small = units.desc_n <= DESC_SMALL
    for label, sel in (("desc_le_5", small), ("desc_gt_5", ~small)):
        fn = lambda t, p, s: (reads.get((t, s)) or {}).get("Desc")  # noqa: E731
        X = matrix(units, fn)
        X[~sel] = np.nan
        res = estimate(X, units, B, seed)
        res["n_targets_in_stratum"] = int(sel.sum())
        out[f"C3_leak_desc_{label}"] = res
    out["compare_by_ci_sign_only"] = {n: ci_sign(out[n]) for n in names.values()}
    return out


def ci_sign(c: Mapping[str, Any]) -> str:
    lo, hi = c.get("ci_low"), c.get("ci_high")
    if lo is None or hi is None:
        return "undefined"
    return "ci > 0" if lo > 0 else "ci < 0" if hi < 0 else "ci includes 0"


# ------------------------------------------------------------------ other reads
def collateral_contrast(values: Mapping[tuple, Mapping[str, float]], feature: str, arm: str,
                        regime: str, rule: str, k, packs: Sequence[str], metric: str, B: int,
                        seed: int) -> Dict[str, Any]:
    """Run-level (pack x seed) collateral difference arm - V0 in the same condition,
    with the two-way bootstrap over packs (one unit per pack); classified
    descriptively (harm is the reading of interest)."""
    X = np.full((len(packs), len(SEEDS)), np.nan)
    for i, p in enumerate(packs):
        for j, s in enumerate(SEEDS):
            a = values.get((feature, arm, regime, rule, 1, p, norm_k(k), s), {}).get(metric)
            b = values.get((feature, "V0", regime, rule, 1, p, norm_k(k), s), {}).get(metric)
            if a is not None and b is not None:
                X[i, j] = a - b
    res = V.two_way_bootstrap(X, list(range(len(packs))), len(packs), B, seed=seed,
                              unstable_share=UNSTABLE_SHARE)
    return classify_descriptive(res)


def redraw(at: AucTable, units: Units, feature: str, k: int, B: int, seed: int) -> Dict[str, Any]:
    """Delta_{V4-V0} at seed 13, draw 2 minus draw 1, per target; target-level
    bootstrap (S1 with one seed column)."""
    def d(draw):
        return v_diff(v_auc(at, (feature, "V4", "TP", "W2", draw), k),
                      v_auc(at, (feature, "V0", "TP", "W2", draw), k))
    X = matrix(units, v_diff(d(2), d(1)), seeds=(13,))
    res = V.hierarchical_bootstrap(X, B, seed=seed, unstable_share=UNSTABLE_SHARE)
    res["n_targets_defined"] = int((~np.isnan(X)).sum())
    return res


def missing_share(at: AucTable, units: Units, feature: str) -> Dict[str, Any]:
    """Share of the main block's (t, k, s, arm) AUC values that are missing."""
    need = have = 0
    for v in MAIN_ARMS:
        arm = (feature, v, "TP", "W2", 1)
        for t in units.targets:
            for k in ALL_KS:
                for s in SEEDS:
                    need += 1
                    have += at.get(arm, t, units.pack_of[t], k, s) is not None
    return {"needed": need, "present": have, "share_missing": (need - have) / need if need else 1.0}
