"""Weight-free inference-time fusion F (design §10, protocol D15).

From a V0 run's ``scores.npz`` (evaluation-set test logits of t, Pa(t) and the
is_a siblings of every pack target):

    s_F(p, t) = z(l_t) + z(mean_{pi in Pa(t)} l_pi) - z(mean_{sigma in Sib(t)} l_sigma)

z(.) standardises a score vector over the evaluation-set test proteins. Pa(t) are
the TP parents of t that are primary columns (root excluded); Sib(t) are the is_a
siblings that are primary columns with at least ``MIN_SIB_POS`` positives in the
condition's train labels. An empty set drops its term (recorded). No weight is
tuned; F is descriptive.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

MIN_SIB_POS = 10


def zscore(v: np.ndarray) -> np.ndarray:
    """(v - mean) / std over the evaluation rows; a constant vector gives zeros."""
    v = np.asarray(v, dtype=np.float64)
    sd = float(v.std())
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(v)
    return (v - float(v.mean())) / sd


def fuse(own: np.ndarray, parents: Optional[np.ndarray], siblings: Optional[np.ndarray]
         ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """F score of one target. ``parents``/``siblings`` are n x m logit blocks (m may
    be 0 or the block None: that term is dropped)."""
    s = zscore(own)
    info = {"n_pa": 0, "n_sib": 0, "pa_dropped": True, "sib_dropped": True}
    if parents is not None and np.asarray(parents).ndim == 2 and np.asarray(parents).shape[1]:
        s = s + zscore(np.asarray(parents, dtype=np.float64).mean(axis=1))
        info.update(n_pa=int(np.asarray(parents).shape[1]), pa_dropped=False)
    if siblings is not None and np.asarray(siblings).ndim == 2 and np.asarray(siblings).shape[1]:
        s = s - zscore(np.asarray(siblings, dtype=np.float64).mean(axis=1))
        info.update(n_sib=int(np.asarray(siblings).shape[1]), sib_dropped=False)
    return s, info


def sibling_columns(sib_cols: Sequence[int], train_col_counts: np.ndarray,
                    min_pos: int = MIN_SIB_POS) -> list:
    """Siblings with >= ``min_pos`` positives in the condition's train labels."""
    return [int(c) for c in sib_cols if int(train_col_counts[int(c)]) >= min_pos]


def fusion_scores(scores: Mapping[str, np.ndarray],
                  targets: Sequence[Tuple[str, int, Sequence[int], Sequence[int]]],
                  train_col_counts: np.ndarray, min_pos: int = MIN_SIB_POS
                  ) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """F for every target of a V0 run. ``scores`` has ``logits`` (n x m) and
    ``columns`` (m vocabulary indices); ``targets`` lists (id, column, Pa columns,
    sibling columns). A Pa/sibling column absent from ``columns`` is an error."""
    pos = {int(c): i for i, c in enumerate(np.asarray(scores["columns"]).tolist())}
    L = np.asarray(scores["logits"], dtype=np.float64)

    def block(cols):
        missing = [c for c in cols if c not in pos]
        if missing:
            raise KeyError(f"scores.npz lacks columns {missing[:5]}")
        return L[:, [pos[c] for c in cols]] if cols else None

    out = {}
    for tid, col, pa, sib in targets:
        sib_kept = sibling_columns(sib, train_col_counts, min_pos)
        s, info = fuse(L[:, pos[int(col)]], block([int(c) for c in pa]), block(sib_kept))
        info["n_sib_candidates"] = len(sib)
        out[tid] = (s, info)
    return out
