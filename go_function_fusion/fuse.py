"""Fusion families (protocol §3), float64 throughout.

Notation: ``L`` is a base model's logit matrix over the evaluated proteins of one
split (n x |V|); ``cols`` the scored vocabulary columns; z_t standardises each
column over the rows (population sd; a constant column gives zeros).

* AP-T:      z(l_t) + w_t z(mean_{Pa_TP(t)} l),  w_t = c tau / (tau + n_train(t))
* fixed-w:   tau = inf (w_t = c); parent-only: w_t = 1
* AP-T+H:    AP-T + w_H z(DS)
* AP-T+R:    AP-T + w_R z(mean_{Reg(t)} l)
* protocol F: z(l_t) + w_t [z(mean Pa_TP) - z(mean Sib)]  (w_t = 1: GF arm F)
* PC:        (1 - alpha) prop_TP(sigma(l_t + w_t z(Pa mean))) + alpha DS
* controls:  the same forms with a permuted / substituted neighbour set
  (shuffled graph, shuffled Reg, CoA, CoA⁺, Pa_any, Desc-max) or DS on permuted
  train rows (shuffled hits); see :mod:`go_function_fusion.priors`.

Empty neighbour sets contribute 0. Selection helpers implement the §3 tie rules.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

TIE_TOL = 1e-12


# ------------------------------------------------------------------ z and priors
def z_cols(M: np.ndarray) -> np.ndarray:
    """Per-column z over rows, float64; population sd; constant column -> 0."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        M = M[:, None]
    m = M.mean(axis=0, keepdims=True)
    s = M.std(axis=0, keepdims=True)
    bad = ~np.isfinite(s) | (s == 0)
    s = np.where(bad, 1.0, s)
    out = (M - m) / s
    out[:, bad.ravel()] = 0.0
    return out


def z_cols_pilot32(S: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The pilot ``b_valid_pilot.py`` ``zcol`` (pilot-compat only): statistics over
    ``rows`` in the input dtype, sd 0 -> 1, cast to float32."""
    m = S[rows].mean(0, keepdims=True)
    s = S[rows].std(0, keepdims=True)
    s[s == 0] = 1
    return ((S - m) / s).astype(np.float32)


def block_agg(L: np.ndarray, sets: Sequence[Sequence[int]], cols: Sequence[int],
              how: str = "mean", method: str = "reduceat", max_elems: int = 1 << 25
              ) -> Tuple[np.ndarray, np.ndarray]:
    """(A, has): A[:, i] = mean (or max) of L over sets[cols[i]]; has = non-empty.

    ``method="loop"`` is the pilot's column loop (pilot-compat); ``"reduceat"``
    gathers all pairs of the chunk once per row block (bounded by ``max_elems``)."""
    n = L.shape[0]
    A = np.zeros((n, len(cols)), dtype=np.float64)
    has = np.array([len(sets[int(c)]) > 0 for c in cols], dtype=bool)
    if method == "loop":
        for i, c in enumerate(cols):
            v = sets[int(c)]
            if len(v):
                blk = L[:, list(v)]
                A[:, i] = blk.max(axis=1) if how == "max" else blk.mean(axis=1)
        return A, has
    idx = np.flatnonzero(has)
    if not len(idx):
        return A, has
    src = np.concatenate([np.asarray(sets[int(cols[i])], dtype=np.int64) for i in idx])
    sizes = np.array([len(sets[int(cols[i])]) for i in idx], dtype=np.int64)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    step = max(1, max_elems // max(1, len(src)))
    for a in range(0, n, step):
        G = np.asarray(L[a:a + step][:, src], dtype=np.float64)
        if how == "max":
            A[a:a + step, idx] = np.maximum.reduceat(G, starts, axis=1)
        else:
            A[a:a + step, idx] = np.add.reduceat(G, starts, axis=1) / sizes[None, :]
    return A, has


def prior_z(L: np.ndarray, sets: Sequence[Sequence[int]], cols: Sequence[int],
            how: str = "mean", method: str = "reduceat") -> np.ndarray:
    """z_t of the neighbour aggregate; 0 for an empty set."""
    A, has = block_agg(L, sets, cols, how, method)
    return z_cols(A) * has[None, :]


def adaptive_w(ntr: np.ndarray, c: float, tau: float) -> np.ndarray:
    """w_t = c tau / (tau + n_train(t)); tau = inf gives the fixed weight c."""
    ntr = np.asarray(ntr, dtype=np.float64)
    if math.isinf(tau):
        return np.full(ntr.shape, float(c))
    return float(c) * float(tau) / (float(tau) + ntr)


# ------------------------------------------------------------------ families
def ap_t(own_z: np.ndarray, pz: np.ndarray, w: np.ndarray) -> np.ndarray:
    return own_z + np.asarray(w, dtype=np.float64)[None, :] * pz


def add_channel(score: np.ndarray, ch_z: np.ndarray, weight: float) -> np.ndarray:
    """AP-T+H (ch = z(DS)) and AP-T+R (ch = z(mean Reg))."""
    return score + float(weight) * ch_z


def protocol_f(own_z: np.ndarray, pa_z: np.ndarray, sib_z: np.ndarray, w: np.ndarray) -> np.ndarray:
    return own_z + np.asarray(w, dtype=np.float64)[None, :] * (pa_z - sib_z)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def pc_q(L_cols: np.ndarray, pz: np.ndarray, w: np.ndarray) -> np.ndarray:
    """q = sigma(l_t + w_t z_t(prior)) (probability space, §3.3)."""
    return sigmoid(np.asarray(L_cols, dtype=np.float64) + np.asarray(w)[None, :] * pz)


def prop_pairs(cols: Sequence[int], vocab, anc) -> Tuple[np.ndarray, np.ndarray]:
    """``go_function.evaluate.tp_ancestor_pairs`` over positions of ``cols``."""
    from go_function.evaluate import tp_ancestor_pairs
    return tp_ancestor_pairs([int(c) for c in cols], vocab, anc)


def prop_tp(P: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray], device: str = "cpu") -> np.ndarray:
    """True-path max-propagation (``go_function.evaluate.propagate_max``)."""
    from go_function.evaluate import propagate_max
    import torch
    return propagate_max(torch.as_tensor(np.asarray(P, dtype=np.float64)), *pairs,
                         device=device).cpu().numpy()


def pc_score(q_prop: np.ndarray, ds: Optional[np.ndarray], alpha: float) -> np.ndarray:
    """PC = (1 - alpha) prop_TP(q) + alpha DS (DS not propagated)."""
    if alpha == 0 or ds is None:
        if alpha != 0:
            raise ValueError("alpha > 0 needs DS")
        return np.asarray(q_prop, dtype=np.float64)
    return (1.0 - alpha) * q_prop + alpha * ds


def convex(base: np.ndarray, other: np.ndarray, alpha: float) -> np.ndarray:
    """X+DS comparator / pilot convex: (1 - alpha) base + alpha other."""
    return (1.0 - alpha) * base + alpha * other


# ------------------------------------------------------------------ grids and selection
def apt_cells(cs: Sequence[float], taus: Sequence[float], null: bool = False) -> List[Dict[str, float]]:
    """AP-T cells in tie-preference order: smaller c, then larger tau (null first)."""
    out = [{"c": 0.0, "tau": float("inf")}] if null else []
    for c in sorted(cs):
        for t in sorted(taus, reverse=True):
            out.append({"c": float(c), "tau": float(t)})
    return out


def fixed_cells(cs: Sequence[float], null: bool = False) -> List[Dict[str, float]]:
    out = [{"c": 0.0, "tau": float("inf")}] if null else []
    return out + [{"c": float(c), "tau": float("inf")} for c in sorted(cs)]


def weight_cells(name: str, ws: Sequence[float]) -> List[Dict[str, float]]:
    """w_H / w_R cells, smaller first (0 = the null cell)."""
    return [{name: float(w)} for w in sorted(ws)]


def pc_cells(cs: Sequence[float], taus: Sequence[float], alphas: Sequence[float]
             ) -> List[Dict[str, float]]:
    """PC cells: smaller c, then larger tau, then smaller alpha."""
    out = []
    for c in sorted(cs):
        for t in sorted(taus, reverse=True):
            for a in sorted(alphas):
                out.append({"c": float(c), "tau": float(t), "alpha": float(a)})
    return out


def cell_key(cell: Mapping[str, float]) -> str:
    return ",".join(f"{k}={'inf' if isinstance(v, float) and math.isinf(v) else repr(float(v))}"
                    for k, v in sorted(cell.items()))


def select(cells: Sequence[Mapping[str, float]], values: Sequence[float],
           tol: float = TIE_TOL) -> Tuple[int, Mapping[str, float]]:
    """Largest value; a value within ``tol`` of the best keeps the earlier cell
    (cells are in tie-preference order). NaN never wins."""
    best_i, best_v = None, -np.inf
    for i, v in enumerate(values):
        if v is None or not np.isfinite(v):
            continue
        if best_i is None or v > best_v + tol:
            best_i, best_v = i, float(v)
    if best_i is None:
        raise ValueError("no finite selection value")
    return best_i, cells[best_i]


def is_null(cell: Mapping[str, float]) -> bool:
    """The null cell of a family: c = 0, w_H = 0 or w_R = 0 (alpha* = 0 is handled by
    the Hf reading, :func:`go_function_fusion.stats.hf_or_null`)."""
    for k in ("c", "w_H", "w_R"):
        if k in cell and float(cell[k]) == 0.0:
            return True
    return False


# ------------------------------------------------------------------ family space
#: family -> (neighbour-set name, aggregation) for the AP-T-form families
SET_OF = {"apt": ("pa_tp", "mean"), "fixed": ("pa_tp", "mean"), "parent_only": ("pa_tp", "mean"),
          "shuf_graph": ("pa_shuf", "mean"), "coa": ("coa", "mean"), "coap": ("coap", "mean"),
          "pa_any": ("pa_any", "mean"), "desc_max": ("desc", "max")}
FAMILIES = tuple(SET_OF) + ("base", "protocol_F", "F_weightfree", "apt_r", "apt_r_shuf",
                            "apt_h", "apt_h_shufhits", "own_hsim", "apt_hsim")


class Space:
    """Every class-centric family of §3 over one (split, base).

    ``L``: logits of the evaluated proteins (n x |V|, float64); ``ntr``: TP train
    counts (|V|); ``sets``: neighbour lists per vocabulary column with keys pa_tp,
    pa_shuf, coa, coap, pa_any, desc, sib, reg, reg_shuf; ``ds``/``ds_shuf``/``hsim``:
    callables C -> (n x |C|) channel scores (or None)."""

    def __init__(self, L: np.ndarray, ntr: np.ndarray, sets: Mapping[str, Sequence[Sequence[int]]],
                 ds: Optional[Callable] = None, ds_shuf: Optional[Callable] = None,
                 hsim: Optional[Callable] = None):
        self.L, self.ntr, self.sets = L, np.asarray(ntr), sets
        self.ds, self.ds_shuf, self.hsim = ds, ds_shuf, hsim

    def ctx(self, C: np.ndarray) -> LazyCtx:
        L, s = self.L, self.sets
        b: Dict[str, Callable[[], Any]] = {"own": lambda: z_cols(L[:, C])}
        for name in ("pa_tp", "pa_shuf", "coa", "coap", "pa_any", "sib", "reg", "reg_shuf"):
            if name in s:
                b[name] = (lambda n=name: prior_z(L, s[n], C))
        if "desc" in s:
            b["desc"] = lambda: prior_z(L, s["desc"], C, how="max")
        for name, fn in (("ds", self.ds), ("ds_shuf", self.ds_shuf), ("hsim", self.hsim)):
            if fn is not None:
                b[name] = (lambda f=fn: z_cols(f(C)))
        return LazyCtx(b)

    def w(self, C: np.ndarray, cell: Mapping[str, float]) -> np.ndarray:
        return adaptive_w(self.ntr[C], cell["c"], cell["tau"])

    def score(self, family: str, cell: Mapping[str, float], C: np.ndarray, x) -> np.ndarray:
        own = x["own"]
        if family == "base":
            return own
        if family in SET_OF:
            if family == "parent_only":
                return own + x["pa_tp"]
            return ap_t(own, x[SET_OF[family][0]], self.w(C, cell))
        if family == "protocol_F":
            return protocol_f(own, x["pa_tp"], x["sib"], self.w(C, cell))
        if family == "F_weightfree":
            return protocol_f(own, x["pa_tp"], x["sib"], np.ones(len(C)))
        apt = ap_t(own, x["pa_tp"], self.w(C, cell)) if "c" in cell else None
        if family == "apt_r":
            return add_channel(apt, x["reg"], cell["w_R"])
        if family == "apt_r_shuf":
            return add_channel(apt, x["reg_shuf"], cell["w_R"])
        if family == "apt_h":
            return add_channel(apt, x["ds"], cell["w_H"])
        if family == "apt_h_shufhits":
            return add_channel(apt, x["ds_shuf"], cell["w_H"])
        if family == "own_hsim":
            return add_channel(own, x["hsim"], cell["w"])
        if family == "apt_hsim":
            return add_channel(apt, x["hsim"], cell["w"])
        raise KeyError(family)


def sets_list(d: Mapping[int, Sequence[int]], n_cols: int) -> List[List[int]]:
    """Dict {column: set} -> list over all columns ([] where absent)."""
    return [list(d.get(c, ())) for c in range(n_cols)]


# ------------------------------------------------------------------ chunked scoring
VariantFn = Callable[[np.ndarray, Dict[str, Any]], np.ndarray]


def score_variants(n_rows: int, cols: np.ndarray, variants: Mapping[str, VariantFn],
                   ctx_fn: Callable[[np.ndarray], Dict[str, Any]], y_fn: Callable[[np.ndarray], np.ndarray],
                   metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
                   chunk: int = 1500, log: Optional[Callable[[str], None]] = None
                   ) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-term metrics for every variant, in column chunks (no dense score matrix
    is kept). ``ctx_fn(C)`` builds the chunk's shared blocks (lazy dict allowed),
    ``y_fn(C)`` the n x |C| bool labels; returns {metric: {variant: per-term}}."""
    out = {m: {v: np.full(len(cols), np.nan) for v in variants} for m in metric_fns}
    for a in range(0, len(cols), chunk):
        C = cols[a:a + chunk]
        ctx = ctx_fn(C)
        y = y_fn(C)
        for name, f in variants.items():
            S = f(C, ctx)
            if S.shape != (n_rows, len(C)):
                raise ValueError(f"variant {name}: shape {S.shape}")
            for m, fn in metric_fns.items():
                out[m][name][a:a + len(C)] = fn(S, y)
        if log:
            log(f"chunk {a}/{len(cols)}")
    return out


class LazyCtx(dict):
    """dict whose missing keys are computed once by ``builders[key]()``."""

    def __init__(self, builders: Mapping[str, Callable[[], Any]]):
        super().__init__()
        self._b = builders

    def __missing__(self, key):
        v = self._b[key]()
        self[key] = v
        return v
