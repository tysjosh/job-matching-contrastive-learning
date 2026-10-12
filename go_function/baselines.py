"""Baselines without training (design §11).

- Naive: term frequency in the condition's train labels (constant per term; class
  AUC 0.5 by definition, protein-centric context only).
- DiamondScore (DS): for test protein p and term c,

      DS(p, c) = sum_{h in hits(p) & train} bits(p, h) * y_cond[h, c] / sum_{h in hits(p) & train} bits(p, h)

  from DGZ's ``test_diamond.res`` (query, hit, bitscore, ...), with y_cond the
  condition's train labels; masked entries count as 0. A protein without a train
  hit scores 0. Computed per condition, no seed; a descriptive reference curve.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


def read_diamond(path: Path) -> List[Tuple[str, str, float]]:
    """(query, hit, bitscore) per line of a DIAMOND tabular file."""
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            out.append((parts[0], parts[1], float(parts[2])))
    return out


def diamond_matrix(hits: Iterable[Tuple[str, str, float]], test_names: Sequence[str],
                   train_names: Sequence[str]) -> sp.csr_matrix:
    """Row-normalised bitscore matrix H (n_test x n_train): H[p, h] =
    bits(p, h) / sum over p's train hits. Hits outside ``train_names`` are ignored;
    a repeated (query, hit) pair keeps its first bitscore (DGZ's dict semantics keep
    the last; DIAMOND writes each pair once)."""
    ti = {n: i for i, n in enumerate(test_names)}
    hi = {n: i for i, n in enumerate(train_names)}
    seen = set()
    rows, cols, vals = [], [], []
    for q, h, b in hits:
        if q in ti and h in hi and (q, h) not in seen:
            seen.add((q, h))
            rows.append(ti[q])
            cols.append(hi[h])
            vals.append(b)
    H = sp.csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)),
                      shape=(len(test_names), len(train_names)))
    tot = np.asarray(H.sum(axis=1)).ravel()
    inv = np.divide(1.0, tot, out=np.zeros_like(tot), where=tot > 0)
    return sp.csr_matrix(sp.diags(inv) @ H)


def masked_columns(y_train: sp.spmatrix, cols: Sequence[int],
                   mask_pairs: Optional[np.ndarray] = None) -> sp.csr_matrix:
    """y_cond[:, cols] with masked (row, col) entries set to 0."""
    cols = [int(c) for c in cols]
    Y = sp.csr_matrix(y_train)[:, cols].astype(np.float64)
    if mask_pairs is not None and len(mask_pairs):
        pairs = np.asarray(mask_pairs).reshape(-1, 2)
        lut = np.full(int(max(cols + [int(pairs[:, 1].max())])) + 1, -1, dtype=np.int64)
        lut[cols] = np.arange(len(cols))
        j = lut[pairs[:, 1]]
        keep = j >= 0
        if keep.any():
            M = sp.csr_matrix((np.ones(int(keep.sum())), (pairs[keep, 0], j[keep])),
                              shape=Y.shape)
            M.data[:] = 1.0
            Y = sp.csr_matrix(Y - Y.multiply(M))
            Y.eliminate_zeros()
    return Y


def diamond_score(H: sp.csr_matrix, y_train: sp.spmatrix, cols: Sequence[int],
                  mask_pairs: Optional[np.ndarray] = None) -> np.ndarray:
    """DS scores (n_test x len(cols))."""
    Y = masked_columns(y_train, cols, mask_pairs)
    return np.asarray((H @ Y).todense(), dtype=np.float64)


def naive_frequency(y_train: sp.spmatrix) -> np.ndarray:
    """Per-column positive frequency in the condition's train labels."""
    y = sp.csr_matrix(y_train)
    return np.asarray(y.sum(axis=0)).ravel() / max(1, y.shape[0])
