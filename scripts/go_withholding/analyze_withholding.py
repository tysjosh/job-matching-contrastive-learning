"""Statistics for the GO/PPI withholding study (design 11, protocol 7-8).

Steps:

1. Group-check preflight (fatal, exit 2). Run dirs are the direct children of
   ``--runs`` named exactly ``<pack>__k<k>__<cell>__s<seed>``; renamed
   ``*.attempt*`` / ``*.groupfailed*`` (and ``*.interrupted*``) dirs are never a
   run id, so they are ignored. Each (cond, seed) group is ``checked`` (four
   ``DONE.json`` and a valid ``GROUP_OK.json`` everywhere, via
   ``withholding_constants.group_ok_valid``), ``failed_member`` (a
   ``FAILED.json``; missing), ``unchecked`` (four ``DONE.json``, not all
   ``GROUP_OK.json`` valid; FATAL), ``group_failed`` (a ``GROUP_FAILED.json``;
   FATAL) or ``incomplete`` (missing). ``--allow-incomplete`` never relaxes
   the fatal cases.
2. Completeness: a (t, k, s) tuple is present iff its group is checked; the
   denominator is ``n_targets * 6 * 5`` (six k levels, five seeds; 720 at 24
   targets, unchanged by AM1). More than 10% missing: exit 3 unless
   ``--allow-incomplete`` (then every output is labelled exploratory).
3. Long table ``per_target.csv`` and the paired differences with the protocol 7
   definedness rules (no partial averaging, no imputation).
4. Primary two-way cluster bootstrap over packs x seeds,
   ``numpy.random.default_rng(seed)`` (20251009), B replicates; replicates with
   zero defined cells are discarded per contrast and counted; > 5% discarded is
   ``unstable``. Holm within H1 / H2 / H3.
5. Sensitivity S1-S5 (S4 via ``numpy.linalg.lstsq``), labelled, never verdicts.
6. ``decisions.json``: the protocol 8 ordered rules, the A1 reading, the
   mechanism rule and the collateral cost.

scipy is used for ``spearmanr`` only (imported lazily); statsmodels is never used.

Library functions for in-process use (G6 smoke): ``classify_groups``,
``build_long_table``, ``paired_deltas``, ``two_way_bootstrap``, ``holm``,
``classify_contrast``, ``decide``.

Usage (protocol section 10)::

    "$PY" "$S/analyze_withholding.py" --runs "$R/results/go_withholding/runs" \\
      --target-table "$R/preprocess/go_withholding/targets/target_table.json" \\
      --covariate "$R/preprocess/go_withholding/targets/text_covariate.json" \\
      --out "$R/results/go_withholding/analysis" --seed 20251009 --B 10000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding as wh  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("analyze_withholding")

K_LEVELS: Tuple[Any, ...] = (*wc.KS, "all")
SMALL_KS = (0, 1, 5)
TREAT_CELLS = ("loss_only", "neg_only", "both")
LOSS_CELLS = ("loss_only", "both")
FLOOR = 0.01
ALPHA = 0.05
MISSING_CAP = 0.10
UNSTABLE_SHARE = 0.05
MECH_RATIO_RANGE = (0.05, 0.5)
MECH_RUN_SHARE = 0.90
SCHEME = "two_way_pack_seed"


class AnalysisStop(RuntimeError):
    def __init__(self, msg: str, code: int):
        super().__init__(msg)
        self.code = code


# ============================================================ preflight
def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def classify_groups(runs_dir: Path, table: Dict[str, Any],
                    seeds: Sequence[int] = wc.SEEDS) -> Dict[Tuple[str, int], str]:
    """Status of every (cond, seed) group of the table's grid (design 11 step 5)."""
    runs_dir = Path(runs_dir)
    out: Dict[Tuple[str, int], str] = {}
    for cond in wh.condition_ids(table):
        for seed in seeds:
            dirs = [runs_dir / r for r in wc.group_run_ids(cond, seed)]
            if any((d / "GROUP_FAILED.json").is_file() for d in dirs):
                out[(cond, seed)] = "group_failed"
            elif any((d / "FAILED.json").is_file() for d in dirs):
                out[(cond, seed)] = "failed_member"
            elif all((d / "DONE.json").is_file() for d in dirs):
                out[(cond, seed)] = ("checked" if all(wc.group_ok_valid(dirs))
                                     else "unchecked")
            else:
                out[(cond, seed)] = "incomplete"
    return out


def preflight(groups: Dict[Tuple[str, int], str]) -> None:
    gf = sorted(f"{c},s{s}" for (c, s), v in groups.items() if v == "group_failed")
    if gf:
        raise AnalysisStop(f"{len(gf)} group(s) with GROUP_FAILED.json {gf[:10]}: the grid is "
                           f"stopped; fix under an amendment (protocol 13)", 2)
    un = sorted(f"{c},s{s}" for (c, s), v in groups.items() if v == "unchecked")
    if un:
        raise AnalysisStop(f"{len(un)} unchecked group(s) {un[:10]}: run `sweep.py status` "
                           f"to check groups", 2)


def tuple_status(table: Dict[str, Any], groups: Dict[Tuple[str, int], str],
                 seeds: Sequence[int] = wc.SEEDS) -> Dict[str, Any]:
    """Present / missing (t, k, s) tuples over ``n_targets * 6 * 5``."""
    present, missing = [], []
    for t in table["targets"]:
        for k in K_LEVELS:
            cond = "all" if k == "all" else f"{t['pack_id']}_k{k}"
            for s in seeds:
                st = groups.get((cond, s), "incomplete")
                if st == "checked":
                    present.append((t["go_id"], k, s))
                else:
                    missing.append({"go_id": t["go_id"], "k": k, "seed": s, "cond_id": cond,
                                    "reason": st})
    denom = int(table["n_targets"]) * len(K_LEVELS) * len(seeds)
    if len(present) + len(missing) != denom:
        raise AnalysisStop(f"tuple count {len(present) + len(missing)} != n_targets x 6 x 5 "
                           f"= {denom}", 2)
    return {"present": present, "missing": missing, "denominator": denom,
            "missing_share": len(missing) / denom if denom else 0.0}


# ============================================================ long table
def _cond(t: Dict[str, Any], k: Any) -> str:
    return "all" if k == "all" else f"{t['pack_id']}_k{k}"


def build_long_table(runs_dir: Path, table: Dict[str, Any],
                     groups: Dict[Tuple[str, int], str],
                     covariate: Optional[Dict[str, Any]] = None,
                     seeds: Sequence[int] = wc.SEEDS) -> List[Dict[str, Any]]:
    """One row per present (target, k, cell, seed) (design 11 step 1)."""
    runs_dir = Path(runs_dir)
    cov = (covariate or {}).get("targets", {})
    cache: Dict[str, Tuple[dict, Optional[dict]]] = {}

    def load(run_id: str) -> Tuple[dict, Optional[dict]]:
        if run_id not in cache:
            m = _read_json(runs_dir / run_id / "eval" / "metrics.json")
            if m is None:
                raise AnalysisStop(f"{run_id}: eval/metrics.json missing in a checked group", 2)
            cache[run_id] = (m, _read_json(runs_dir / run_id / "eval" / "relation.json"))
        return cache[run_id]

    rows = []
    for t in table["targets"]:
        go = t["go_id"]
        for k in K_LEVELS:
            cond = _cond(t, k)
            for s in seeds:
                if groups.get((cond, s)) != "checked":
                    continue
                for cell in wc.CELLS:
                    rid = wc.run_id_for(cond, cell, s)
                    m, rel = load(rid)
                    pt = m["per_target"][go]
                    coll = (m.get("collateral") or {}).get(t["pack_id"]) or {}
                    relt = ((rel or {}).get("per_target") or {}).get(go) or {}
                    rows.append({
                        "go_id": go, "k": k, "cell": cell, "seed": s, "run_id": rid,
                        "hard_auc": pt.get("hard_auc"),
                        "macro_query_hard_auc": pt.get("macro_query_hard_auc"),
                        "macro_mrr": pt.get("macro_mrr"),
                        "collateral_hard_auc": coll.get("hard_auc"),
                        "relation_excess": relt.get("excess"),
                        "stratum": t.get("stratum"), "pack_id": t["pack_id"],
                        "S_sib": t.get("S_sib"), "S_2": t.get("S_2"),
                        "tsim_excess": (cov.get(go) or {}).get("tsim_excess")})
    return rows


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def paired_deltas(rows: Iterable[Dict[str, Any]], metric: str = "hard_auc") -> Dict[str, Any]:
    """Paired differences vs baseline with the protocol 7 definedness rules.

    Returns ``{"delta": {c: {k: {(t, s): v}}}, "small": {c: {(t, s): v}},
    "D": {...}, "coll": {c: {k: {(pack, s): v}}}, "E": {c: {(t, s): v}},
    "rel_small": {c: {(t, s): v}}}``; only defined cells are present.
    """
    idx: Dict[Tuple[str, Any, str, int], Dict[str, Any]] = {}
    pack_of: Dict[str, str] = {}
    for r in rows:
        idx[(r["go_id"], r["k"], r["cell"], int(r["seed"]))] = r
        pack_of[r["go_id"]] = r["pack_id"]
    ts = sorted({(g, s) for g, _k, _c, s in idx})
    out: Dict[str, Any] = {name: {c: {} for c in TREAT_CELLS}
                           for name in ("small", "D", "E", "rel_small")}
    out["delta"] = {c: {k: {} for k in K_LEVELS} for c in TREAT_CELLS}
    out["coll"] = {c: {k: {} for k in K_LEVELS} for c in TREAT_CELLS}
    for c in TREAT_CELLS:
        rel: Dict[Any, Dict[Tuple[str, int], float]] = {k: {} for k in K_LEVELS}
        for (g, s) in ts:
            for k in K_LEVELS:
                a, b = idx.get((g, k, c, s)), idx.get((g, k, "baseline", s))
                if a is None or b is None:
                    continue
                va, vb = _num(a.get(metric)), _num(b.get(metric))
                if va is not None and vb is not None:
                    out["delta"][c][k][(g, s)] = va - vb
                ca, cb = _num(a.get("collateral_hard_auc")), _num(b.get("collateral_hard_auc"))
                if ca is not None and cb is not None:
                    out["coll"][c][k][(pack_of[g], s)] = ca - cb
                ra, rb = _num(a.get("relation_excess")), _num(b.get("relation_excess"))
                if ra is not None and rb is not None:
                    rel[k][(g, s)] = ra - rb
            d = out["delta"][c]
            if all((g, s) in d[k] for k in SMALL_KS):
                small = float(np.mean([d[k][(g, s)] for k in SMALL_KS]))
                out["small"][c][(g, s)] = small
                if (g, s) in d["all"]:
                    out["D"][c][(g, s)] = small - d["all"][(g, s)]
            if (g, s) in d[0] and (pack_of[g], s) in out["coll"][c][0]:
                out["E"][c][(g, s)] = d[0][(g, s)] - out["coll"][c][0][(pack_of[g], s)]
            if all((g, s) in rel[k] for k in SMALL_KS):
                out["rel_small"][c][(g, s)] = float(np.mean([rel[k][(g, s)] for k in SMALL_KS]))
    return out


# ============================================================ bootstrap
def to_matrix(values: Dict[Tuple[str, int], float], units: Sequence[str],
              seeds: Sequence[int] = wc.SEEDS) -> np.ndarray:
    X = np.full((len(units), len(seeds)), np.nan)
    ui = {u: i for i, u in enumerate(units)}
    si = {s: j for j, s in enumerate(seeds)}
    for (u, s), v in values.items():
        if u in ui and s in si:
            X[ui[u], si[s]] = v
    return X


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
               n_defined: int) -> Dict[str, Any]:
    kept = T[keep]
    n_disc = int(B - keep.sum())
    res = {"estimate": estimate, "ci_low": None, "ci_high": None, "p_boot": None,
           "n_defined_cells": int(n_defined), "n_discarded_replicates": n_disc,
           "n_replicates": int(B), "unstable": bool(n_disc / B > UNSTABLE_SHARE) if B else True}
    if len(kept):
        lo, hi = np.percentile(kept, [2.5, 97.5])
        res["ci_low"], res["ci_high"] = float(lo), float(hi)
        res["p_boot"] = float(min(1.0, 2.0 * min(np.mean(kept <= 0), np.mean(kept >= 0))))
    return res


def two_way_bootstrap(X: np.ndarray, unit_pack: Sequence[int], n_packs: int, B: int,
                      seed: int = wc.ANALYSIS_SEED, strata: Optional[Sequence[int]] = None
                      ) -> Dict[str, Any]:
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
        return {**_summarise(T, keep, B, est, int(D.sum())), "scheme": SCHEME}
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
    return {**_summarise(T, keep, B, est, n_def), "scheme": SCHEME}


def hierarchical_bootstrap(X: np.ndarray, B: int, seed: int = wc.ANALYSIS_SEED,
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
        return {**_summarise(T, keep, B, est, int((~np.isnan(X)).sum())),
                "scheme": "hierarchical_target_seed"}
    st = np.asarray(strata, dtype=int)
    nh, dh = mean_over((st == 1).astype(float))
    nl, dl = mean_over((st == 0).astype(float))
    keep = (dh > 0) & (dl > 0)
    T = np.where(keep, nh / np.where(dh > 0, dh, 1.0) - nl / np.where(dl > 0, dl, 1.0), np.nan)
    Xh, Xl = X[st == 1], X[st == 0]
    est = (float(np.nanmean(Xh) - np.nanmean(Xl))
           if (~np.isnan(Xh)).any() and (~np.isnan(Xl)).any() else None)
    return {**_summarise(T, keep, B, est, int((~np.isnan(Xh)).sum() + (~np.isnan(Xl)).sum())),
            "scheme": "hierarchical_target_seed"}


def s4_adjusted(X: np.ndarray, high: Sequence[int], tsim: Sequence[Optional[float]],
                unit_pack: Sequence[int], n_packs: int, B: int,
                seed: int = wc.ANALYSIS_SEED) -> Dict[str, Any]:
    """S4: OLS of per-target mean Delta_small on [1, high, tsim_excess]; ``high`` coefficient.

    Point estimate on the full data; CI from the same two-way draws (rows = targets
    of the drawn packs with multiplicity, per-target means over the drawn seeds).
    """
    X = np.asarray(X, dtype=float)
    hv = np.asarray(high, dtype=float)
    tv = np.asarray([np.nan if v is None else float(v) for v in tsim])
    up = np.asarray(unit_pack, dtype=int)
    D = (~np.isnan(X)).astype(float)
    Xf = np.nan_to_num(X)

    def fit(y: np.ndarray, w: np.ndarray) -> Optional[float]:
        ok = (w > 0) & np.isfinite(y) & np.isfinite(tv)
        if ok.sum() < 3:
            return None
        A = np.column_stack([np.ones(ok.sum()), hv[ok], tv[ok]])
        sw = np.sqrt(w[ok])
        Aw, yw = A * sw[:, None], y[ok] * sw
        if np.linalg.matrix_rank(Aw) < 3:
            return None
        coef = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        return float(coef[1])

    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = np.where(D.sum(1) > 0, Xf.sum(1) / np.maximum(D.sum(1), 1), np.nan)
    est = fit(ybar, np.ones(len(ybar)))
    Cp, Cs = two_way_draws(n_packs, X.shape[1], B, seed)
    T = np.full(B, np.nan)
    for b in range(B):
        den = D @ Cs[b]
        y = np.where(den > 0, (Xf @ Cs[b]) / np.where(den > 0, den, 1.0), np.nan)
        v = fit(y, Cp[b][up])
        if v is not None:
            T[b] = v
    keep = np.isfinite(T)
    return {**_summarise(T, keep, B, est, int(D.sum())), "scheme": SCHEME,
            "model": "ols[1, high, tsim_excess]", "coefficient": "high"}


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


def classify_contrast(c: Dict[str, Any]) -> str:
    """Protocol 8 ordered first-match rules; the unstable check comes first."""
    if c.get("unstable"):
        return "inconclusive"
    est, lo, hi, p = c.get("estimate"), c.get("ci_low"), c.get("ci_high"), c.get("p_holm")
    if est is None or lo is None or hi is None:
        return "inconclusive"
    if hi < 0 and est <= -FLOOR:
        return "harm"
    if p is not None and p < ALPHA and lo > 0 and est >= FLOOR:
        return "supported"
    if -FLOOR < lo and hi < FLOOR:
        return "practically null"
    if lo > 0 and est < FLOOR:
        return "below floor"
    return "inconclusive"


def ci_excludes_0(c: Dict[str, Any]) -> bool:
    lo, hi = c.get("ci_low"), c.get("ci_high")
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


# ============================================================ contrasts
def compute_contrasts(deltas: Dict[str, Any], table: Dict[str, Any], B: int, seed: int,
                      strata: Optional[Dict[str, int]] = None) -> Dict[str, Dict[str, Any]]:
    """All primary contrasts (families H1, H2, H3, descriptive), Holm-adjusted."""
    targets = [t["go_id"] for t in table["targets"]]
    packs = [p["pack_id"] for p in table["packs"]]
    pidx = {p: i for i, p in enumerate(packs)}
    pack_of = {t["go_id"]: t["pack_id"] for t in table["targets"]}
    unit_pack = [pidx[pack_of[g]] for g in targets]
    n_packs = int(table["n_packs"])
    if strata is None:
        strata = {t["go_id"]: (1 if t.get("stratum") == "high" else
                               0 if t.get("stratum") == "low" else -1) for t in table["targets"]}
    st = [strata.get(g, -1) for g in targets]

    def tgt(values, **kw):
        r = two_way_bootstrap(to_matrix(values, targets), unit_pack, n_packs, B, seed, **kw)
        r.update({"n_targets": len(targets), "n_packs": n_packs, "n_seeds": len(wc.SEEDS)})
        return r

    def pck(values):
        r = two_way_bootstrap(to_matrix(values, packs), list(range(len(packs))), n_packs, B,
                              seed)
        r.update({"n_targets": len(targets), "n_packs": n_packs, "n_seeds": len(wc.SEEDS)})
        return r

    fam: Dict[str, Dict[str, Any]] = {"H1": {}, "H2": {}, "H3": {}, "descriptive": {}}
    for c in TREAT_CELLS:
        fam["H1"][c] = tgt(deltas["small"][c])
        fam["H2"][c] = tgt(deltas["D"][c])
    for c in LOSS_CELLS:
        fam["H3"][c] = tgt(deltas["small"][c], strata=st)
    for c in TREAT_CELLS:
        for k in K_LEVELS:
            fam["descriptive"][f"delta_k{k}_{c}"] = tgt(deltas["delta"][c][k])
            fam["descriptive"][f"collateral_k{k}_{c}"] = pck(deltas["coll"][c][k])
    for c in LOSS_CELLS:
        fam["descriptive"][f"E_{c}"] = tgt(deltas["E"][c])
        fam["descriptive"][f"mechanism_a_{c}"] = tgt(deltas["rel_small"][c])
    fam["descriptive"]["A1_neg_only_k0"] = tgt(deltas["delta"]["neg_only"][0])
    for name in ("H1", "H2", "H3"):
        adj = holm({k: v["p_boot"] for k, v in fam[name].items()})
        for k, v in fam[name].items():
            v["p_holm"] = adj[k]
    for v in fam["descriptive"].values():
        v["p_holm"] = v["p_boot"]          # protocol 8: unadjusted for descriptive contrasts
    for f in fam.values():
        for v in f.values():
            v["classification"] = classify_contrast(v)
            v["ci_excludes_0"] = ci_excludes_0(v)
    return fam


def run_ratio_medians(runs_dir: Path, run_ids: Iterable[str]) -> Dict[str, Optional[float]]:
    """Per run: median over epochs of ``ratio_weighted_go_to_infonce.median``."""
    out: Dict[str, Optional[float]] = {}
    for rid in run_ids:
        vals = []
        path = Path(runs_dir) / rid / "go_diagnostics.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                v = (row.get("ratio_weighted_go_to_infonce") or {}).get("median")
                if v is not None and math.isfinite(float(v)):
                    vals.append(float(v))
        out[rid] = float(statistics.median(vals)) if vals else None
    return out


def mechanism_b(ratios: Dict[str, Optional[float]]) -> Dict[str, Any]:
    vals = list(ratios.values())
    n_in = sum(1 for v in vals if v is not None and MECH_RATIO_RANGE[0] <= v <= MECH_RATIO_RANGE[1])
    share = n_in / len(vals) if vals else None
    return {"n_runs": len(vals), "n_in_range": n_in, "share": share,
            "range": list(MECH_RATIO_RANGE), "holds": bool(share is not None
                                                           and share >= MECH_RUN_SHARE)}


def decide(contrasts: Dict[str, Dict[str, Any]], sensitivity: Optional[Dict[str, Any]] = None,
           mech_b: Optional[Dict[str, Any]] = None, exploratory: bool = False) -> Dict[str, Any]:
    """Protocol 8 readings from classified contrasts (mechanical)."""
    sens = sensitivity or {}
    cls = {f: {k: v["classification"] for k, v in d.items()} for f, d in contrasts.items()}
    out: Dict[str, Any] = {"exploratory": bool(exploratory), "rules": "protocol section 8"}
    for fam in ("H1", "H2", "H3"):
        out[fam] = {c: {"classification": v["classification"],
                        "ci_excludes_0": v["ci_excludes_0"]}
                    for c, v in contrasts[fam].items()}
    for c in TREAT_CELLS:
        h1, h2 = cls["H1"][c], cls["H2"][c]
        out["H1"][c]["reading"] = ("improves performance on withheld targets"
                                   if h1 == "supported" else "not supported")
        if h2 == "supported":
            out["H2"][c]["reading"] = "helps more with fewer examples"
        elif h1 == "supported" and h2 in ("practically null", "inconclusive"):
            out["H2"][c]["reading"] = "uniform benefit, not data efficiency"
        else:
            out["H2"][c]["reading"] = "not supported"
    s2 = (sens.get("S2") or {}).get("H3", {})
    s4 = sens.get("S4") or {}
    for c in LOSS_CELLS:
        row = out["H3"][c]
        s2c = (s2.get(c) or {}).get("classification")
        s4c = s4.get(c) or {}
        s4_pos = s4c.get("ci_low") is not None and s4c["ci_low"] > 0
        s4_incl0 = (s4c.get("ci_low") is None or s4c.get("ci_high") is None
                    or (s4c["ci_low"] <= 0 <= s4c["ci_high"]))
        if cls["H3"][c] == "supported":
            row["reading"] = ("neighbour support moderates the benefit; possibly text-mediated"
                              if s4_incl0 else "neighbour support moderates the benefit")
        elif s2c == "supported" or s4_pos:
            row["reading"] = "exploratory (holds only under " + \
                ("S2 N_2 strata" if s2c == "supported" else "S4 adjustment") + ")"
        else:
            row["reading"] = "not supported"
    d = contrasts["descriptive"]
    a1 = d["A1_neg_only_k0"]
    out["A1"] = {"neg_only_k0": {"classification": a1["classification"],
                                 "ci_excludes_0": a1["ci_excludes_0"]},
                 "global_reshaping_present": bool(a1["ci_low"] is not None and a1["ci_low"] > 0),
                 "E": {}}
    for c in LOSS_CELLS:
        e = d[f"E_{c}"]
        out["A1"]["E"][c] = {"classification": e["classification"],
                             "ci_excludes_0": e["ci_excludes_0"],
                             "target_specific": bool(e["ci_low"] is not None and e["ci_low"] > 0)}
    supported_lo = [f for f in ("H1", "H2", "H3") if cls[f].get("loss_only") == "supported"]
    ma = d["mechanism_a_loss_only"]
    a_ok = bool(ma["ci_low"] is not None and ma["ci_low"] > 0)
    b_ok = bool((mech_b or {}).get("holds"))
    out["mechanism"] = {"applies_to": supported_lo, "a_relation_ci_low_gt_0": a_ok,
                        "b_ratio_in_range": mech_b, "confirmed":
                        (a_ok and b_ok) if supported_lo else None,
                        "reading": (None if not supported_lo else
                                    "mechanism confirmed" if (a_ok and b_ok)
                                    else "mechanism not confirmed")}
    out["collateral"] = {}
    for c in TREAT_CELLS:
        harm = [str(k) for k in K_LEVELS if d[f"collateral_k{k}_{c}"]["classification"] == "harm"]
        out["collateral"][c] = {"harm_ks": harm,
                                "reported_as_cost": bool(harm and cls["H1"][c] == "supported")}
    return out


# ============================================================ sensitivity
def s2_strata(table: Dict[str, Any]) -> Dict[str, int]:
    vals = np.asarray([float(t.get("S_2") or 0.0) for t in table["targets"]])
    lo, hi = np.percentile(vals, [33, 67]) if len(vals) else (0.0, 0.0)
    return {t["go_id"]: (0 if v <= lo else 1 if v >= hi else -1)
            for t, v in zip(table["targets"], vals)}


def seed_spread(values: Dict[Tuple[str, int], float]) -> Dict[str, Any]:
    per_seed = defaultdict(list)
    for (_g, s), v in values.items():
        per_seed[s].append(v)
    means = {str(s): float(np.mean(v)) for s, v in sorted(per_seed.items())}
    sd = float(np.std(list(means.values()), ddof=1)) if len(means) >= 2 else None
    return {"per_seed_mean": means, "sd_across_seeds": sd}


def compute_sensitivity(rows: List[Dict[str, Any]], deltas: Dict[str, Any],
                        table: Dict[str, Any], B: int, seed: int) -> Dict[str, Any]:
    targets = [t["go_id"] for t in table["targets"]]
    st_sib = [1 if t.get("stratum") == "high" else 0 if t.get("stratum") == "low" else -1
              for t in table["targets"]]
    pidx = {p["pack_id"]: i for i, p in enumerate(table["packs"])}
    unit_pack = [pidx[t["pack_id"]] for t in table["targets"]]
    n_packs = int(table["n_packs"])
    out: Dict[str, Any] = {"label": "sensitivity analyses; never change verdicts"}
    s1: Dict[str, Any] = {"H1": {}, "H2": {}, "H3": {}}
    for c in TREAT_CELLS:
        s1["H1"][c] = hierarchical_bootstrap(to_matrix(deltas["small"][c], targets), B, seed)
        s1["H2"][c] = hierarchical_bootstrap(to_matrix(deltas["D"][c], targets), B, seed)
    for c in LOSS_CELLS:
        s1["H3"][c] = hierarchical_bootstrap(to_matrix(deltas["small"][c], targets), B, seed,
                                             strata=st_sib)
    out["S1"] = s1
    s2 = compute_contrasts(deltas, table, B, seed, strata=s2_strata(table))["H3"]
    for v in s2.values():
        v["p_holm"] = v["p_boot"]
        v["classification"] = classify_contrast(v)
    out["S2"] = {"H3": s2, "strata": "S_2 tertiles over the selected targets"}
    out["S3"] = {f: {c: {k: v[k] for k in ("estimate", "ci_low", "ci_high", "p_boot",
                                            "classification")}
                     for c, v in d.items()}
                 for f, d in compute_contrasts(paired_deltas(rows, "macro_query_hard_auc"),
                                               table, B, seed).items() if f != "descriptive"}
    by_go = {r["go_id"]: r.get("tsim_excess") for r in rows}
    tsim = [by_go.get(g) for g in targets]
    out["S4"] = {c: s4_adjusted(to_matrix(deltas["small"][c], targets),
                                [1 if s == 1 else 0 for s in st_sib], tsim, unit_pack,
                                n_packs, B, seed) for c in LOSS_CELLS}
    out["S5"] = {c: {"small": seed_spread(deltas["small"][c]),
                     **{f"k{k}": seed_spread(deltas["delta"][c][k]) for k in K_LEVELS}}
                 for c in TREAT_CELLS}
    return out


# ============================================================ diagnostics
def diagnostics_summary(runs_dir: Path, rows: List[Dict[str, Any]],
                        audits_dir: Optional[Path]) -> Dict[str, Any]:
    run_ids = sorted({r["run_id"] for r in rows})
    by_cell: Dict[str, Dict[str, Any]] = {}
    ratios = run_ratio_medians(runs_dir, run_ids)
    drift = []
    for rid in run_ids:
        path = Path(runs_dir) / rid / "go_diagnostics.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("drift_flag"):
                    drift.append(rid)
                    break
    for cell in wc.CELLS:
        rids = [r for r in run_ids if f"__{cell}__" in r]
        vals = [ratios[r] for r in rids if ratios.get(r) is not None]
        rel = [_read_json(Path(runs_dir) / r / "eval" / "relation.json") or {} for r in rids]
        sp = [x["global"]["spearman"] for x in rel
              if (x.get("global") or {}).get("spearman") is not None]
        ex = [r["relation_excess"] for r in rows
              if r["cell"] == cell and r.get("relation_excess") is not None]
        by_cell[cell] = {"n_runs": len(rids),
                         "ratio_weighted_go_to_infonce_median_of_run_medians":
                             float(statistics.median(vals)) if vals else None,
                         "relation_global_spearman_mean": float(np.mean(sp)) if sp else None,
                         "relation_excess_mean": float(np.mean(ex)) if ex else None}
    audits: Dict[str, Any] = {}
    if audits_dir is not None and Path(audits_dir).is_dir():
        acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for p in sorted(Path(audits_dir).glob("neg__*.json")):
            a = _read_json(p) or {}
            for ep in (a.get("by_epoch") or {}).values():
                for key, v in ep.items():
                    if key.startswith("share_") and v is not None:
                        acc[a.get("cell", "?")][key].append(float(v))
        audits = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in acc.items()}
    return {"by_cell": by_cell, "drift_flag_runs": drift, "audit_share_means": audits}


# ============================================================ main
def _jsonable_keys(d: Dict[Tuple[str, int], str]) -> Dict[str, str]:
    return {f"{c},s{s}": v for (c, s), v in sorted(d.items())}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["go_id", "k", "cell", "seed", "run_id", "hard_auc", "macro_query_hard_auc",
            "macro_mrr", "collateral_hard_auc", "relation_excess", "stratum", "pack_id",
            "S_sib", "S_2", "tsim_excess"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _dump(path: Path, obj: Any) -> None:
    def safe(v):
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        if isinstance(v, (np.floating, float)):
            return float(v) if math.isfinite(float(v)) else None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    path.write_text(json.dumps(wc.relativize_json(safe(obj)), indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def analyze(runs_dir: Path, table_path: Path, covariate_path: Optional[Path], out: Path,
            seed: int, B: int, allow_incomplete: bool,
            audits_dir: Optional[Path] = None) -> Dict[str, Any]:
    table = json.loads(Path(table_path).read_text(encoding="utf-8"))
    groups = classify_groups(runs_dir, table)
    preflight(groups)
    ts = tuple_status(table, groups)
    exploratory = False
    if ts["missing_share"] > MISSING_CAP:
        if not allow_incomplete:
            raise AnalysisStop(f"incomplete grid: {len(ts['missing'])} of {ts['denominator']} "
                               f"tuples missing ({ts['missing_share']:.1%} > 10%)", 3)
        exploratory = True
    covariate = (json.loads(Path(covariate_path).read_text(encoding="utf-8"))
                 if covariate_path and Path(covariate_path).is_file() else None)
    rows = build_long_table(runs_dir, table, groups, covariate)
    deltas = paired_deltas(rows)
    contrasts = compute_contrasts(deltas, table, B, seed)
    sens = compute_sensitivity(rows, deltas, table, B, seed)
    lo_runs = sorted({r["run_id"] for r in rows if r["cell"] == "loss_only"})
    mech_b = mechanism_b(run_ratio_medians(runs_dir, lo_runs))
    decisions = decide(contrasts, sens, mech_b, exploratory)
    label = {"exploratory": exploratory, "seed": seed, "B": B,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path(out)
    wc.assert_writable_target(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_target.csv", rows)
    _dump(out / "contrasts.json", {**label, "families": contrasts})
    _dump(out / "missing.json", {**label, "denominator": ts["denominator"],
                                 "n_missing": len(ts["missing"]),
                                 "missing_share": ts["missing_share"],
                                 "groups": _jsonable_keys(groups), "missing": ts["missing"]})
    _dump(out / "sensitivity.json", {**label, **sens})
    _dump(out / "decisions.json", {**label, **decisions})
    _dump(out / "diagnostics_summary.json",
          {**label, **diagnostics_summary(runs_dir, rows, audits_dir)})
    return {"exploratory": exploratory, "n_rows": len(rows), "n_missing": len(ts["missing"]),
            "denominator": ts["denominator"],
            "H1": {c: v["classification"] for c, v in contrasts["H1"].items()},
            "H2": {c: v["classification"] for c, v in contrasts["H2"].items()},
            "H3": {c: v["classification"] for c, v in contrasts["H3"].items()}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--target-table", type=Path, default=None)
    ap.add_argument("--covariate", type=Path, default=None)
    ap.add_argument("--audits", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=wc.ANALYSIS_SEED)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)
    if args.B < 1:
        ap.error("--B must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    res = Path(wc.STUDY_RES)
    try:
        summary = analyze(Path(args.runs or res / "runs"),
                          Path(args.target_table or Path(wc.TARGETS) / "target_table.json"),
                          args.covariate or Path(wc.TARGETS) / "text_covariate.json",
                          Path(args.out or res / "analysis"), args.seed, args.B,
                          args.allow_incomplete, args.audits or res / "audits")
    except AnalysisStop as exc:
        logger.error("%s", exc)
        return exc.code
    except Exception as exc:  # noqa: BLE001
        logger.error("analysis failed: %s: %s", type(exc).__name__, exc)
        return 2
    summary["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
