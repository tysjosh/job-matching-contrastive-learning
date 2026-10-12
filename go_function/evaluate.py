"""Evaluator (design §8): class-centric metrics, protein-centric Fmax/Smin/AUPR,
the regime -> evaluation-label mapping, the collateral term set and the
DGZ-faithful mode used by the gates.

Formulas are ported from the public DeepGOZero source, read with web_fetch:

- https://raw.githubusercontent.com/bio-ontology-research-group/deepgozero/main/evaluate.py
  (``main`` threshold loop, ``evaluate_annotations``, AUPR = ``np.trapz`` over the
  101 (recall, precision) points sorted by recall);
- https://raw.githubusercontent.com/bio-ontology-research-group/deepgozero/main/evaluate_terms.py
  (class-centric average AUC over terms with 0 < n_pos < n, ``roc_curve`` + ``auc``);
- https://raw.githubusercontent.com/bio-ontology-research-group/deepgozero/main/utils.py
  (``Ontology.calculate_ic``, ``get_ic``, ``get_parents``, ``get_namespace_terms``).

Differences between DGZ's code and design §8.2 (recorded in FEAT-002 findings):

1. IC: DGZ's ``evaluate.py`` computes IC from train + valid + **test**
   ``prop_annotations`` (``calculate_ic(annotations + test_annotations)``); design
   §8.2 says train+valid TP labels. DGZ-faithful mode follows DGZ; study mode
   (``ic_from_labels``) follows the design (train+valid).
2. AUPR: DGZ's AUPR is the trapezoid area under the **protein-centric** PR curve of
   the 101 thresholds, not a micro AUPR over flattened (protein, term) pairs.
   DGZ-faithful mode reports DGZ's AUPR; study mode reports both ``aupr_micro``
   (micro precision/recall per threshold, same trapezoid rule) and
   ``aupr_protein`` (DGZ's definition).
3. Proteins with no true label in the evaluated term set are skipped by DGZ's
   ``evaluate_annotations`` (recall undefined); both modes skip them in the
   protein-centric metrics and report the count. Class-centric metrics keep every
   evaluation-set protein (negatives included).
4. DGZ's protein-centric label set is every GO id of the namespace (alt ids
   included, root removed) in ``prop_annotations``, not only the ``terms.pkl``
   columns, so recall counts labels the model cannot predict. DGZ-faithful mode
   keeps this; study mode evaluates on the term set T only.
5. Thresholds compare in float64 (``score >= t/100``), which is what DGZ's numpy-1
   scalar comparison did. Class AUC uses the scores cast to float32, as
   ``evaluate_terms.py`` stores them in a float32 matrix.

Fmax uses DGZ's strict ``fmax < f`` update (first maximum), precision averaged
over proteins with >= 1 predicted term, recall over all evaluated proteins.
Smin = min over thresholds of sqrt(ru^2 + mi^2), ru and mi averaged over the
evaluated proteins with IC (not normalised IC).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from go_function.constants import NAMESPACES, REGIMES, ROOTS
from go_function.runtime import FatalError, ProtocolStop

THRESHOLDS = tuple(t / 100.0 for t in range(0, 101))


# ------------------------------------------------------------------ class-centric (§8.1)
def class_metrics(logits_col, y_col) -> Dict[str, float]:
    """AUC and AUPR of one column on raw logits (no propagation). Asserts
    0 < n_pos < n (a degenerate target is a protocol stop, design §8.1)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    s = np.asarray(logits_col, dtype=np.float64).ravel()
    y = np.asarray(y_col).ravel().astype(bool)
    n, n_pos = int(len(y)), int(y.sum())
    if len(s) != n:
        raise FatalError(f"class_metrics: {len(s)} scores for {n} labels")
    if not 0 < n_pos < n:
        raise ProtocolStop(f"class_metrics: n_pos={n_pos} of n={n} (need 0 < n_pos < n)")
    return {"auc": float(roc_auc_score(y, s)), "aupr": float(average_precision_score(y, s)),
            "n_pos": n_pos, "n": n}


# ------------------------------------------------------------------ evaluation labels
@dataclass(frozen=True)
class EvalSource:
    regime: str
    label_file: str      # repo-relative npz
    eval_set: str        # npz member holding the evaluation-set test rows


def eval_label_source(regime: str, aspect: str) -> EvalSource:
    """The single regime -> (test-label file, evaluation set) mapping (design §2.2,
    §15): ``labels_TPnoIPI.npz`` iff ``regime == "TPnoIPI"``, else ``labels_TP.npz``.
    DGZ-rule test labels are never returned (gate scripts read them directly)."""
    if regime not in REGIMES:
        raise FatalError(f"unknown regime {regime!r}; allowed {REGIMES}")
    if aspect not in NAMESPACES or aspect == "cc":
        raise FatalError(f"unknown aspect {aspect!r}")
    if regime == "TPnoIPI" and aspect != "mf":
        raise FatalError("regime TPnoIPI applies to MF only")
    name = "labels_TPnoIPI.npz" if regime == "TPnoIPI" else "labels_TP.npz"
    return EvalSource(regime=regime, label_file=f"preprocess/go_function/{aspect}/{name}",
                      eval_set="eval_test")


def load_eval_labels(regime: str, aspect: str) -> Dict[str, Any]:
    """Evaluation-set test labels of a regime: ``y`` (CSR rows = evaluation set),
    ``rows`` (test row indices), ``proteins``, ``vocab``, ``train``/``valid`` (base
    labels of the same file, for IC and the collateral set)."""
    from go_function.labels import load_labels
    from go_function.runtime import repo_path
    src = eval_label_source(regime, aspect)
    lab = load_labels(repo_path(src.label_file))
    if src.eval_set not in lab:
        raise FatalError(f"{src.label_file} has no {src.eval_set} member")
    rows = lab[src.eval_set]
    return {"source": src, "y": lab["test"][rows], "rows": rows,
            "proteins": [lab["proteins_test"][i] for i in rows], "vocab": lab["vocab"],
            "train": lab["train"], "valid": lab["valid"]}


# ------------------------------------------------------------------ term sets
def collateral_terms(vocab, train_labels: sp.spmatrix, excluded_cols: Iterable[int],
                     root: str) -> np.ndarray:
    """Collateral term set (design §8.2): primary, non-duplicate columns with >= 1
    train positive in the base labels, minus the pack's D*_TP columns
    (``excluded_cols``), root excluded. ``vocab`` is a ``labels.Vocab``."""
    pos = np.asarray(sp.csr_matrix(train_labels).sum(axis=0)).ravel()
    excl = set(int(c) for c in excluded_cols)
    keep = [c for c in range(len(vocab))
            if vocab.primary[c] is not None and c not in vocab.duplicate_cols
            and pos[c] > 0 and c not in excl and vocab.primary[c] != root]
    return np.asarray(keep, dtype=np.int64)


def ic_from_labels(labels: sp.spmatrix, parent_cols: Sequence[Tuple[int, ...]]) -> np.ndarray:
    """IC per column, DGZ ``calculate_ic`` on a label matrix (study mode: train+valid
    TP labels stacked): IC(c) = log2(min over parent columns of count / count(c));
    no parent column -> 0; count 0 -> 0 (DGZ ``get_ic`` default)."""
    cnt = np.asarray(sp.csr_matrix(labels).sum(axis=0)).ravel().astype(np.float64)
    ic = np.zeros(len(cnt), dtype=np.float64)
    for c, n in enumerate(cnt):
        if n <= 0:
            continue
        par = parent_cols[c]
        min_n = n if not par else min(cnt[q] for q in par)
        if min_n <= 0:
            raise FatalError(f"IC: column {c} has a parent with zero count (labels not closed)")
        ic[c] = math.log(min_n / n, 2)
    return ic


def tp_ancestor_pairs(terms: Sequence[int], vocab, anc: Mapping[str, frozenset]) -> Tuple[np.ndarray, np.ndarray]:
    """(src, dst) positions in ``terms`` with primary(dst) a TP ancestor of
    primary(src), src != dst, for max-propagation within T."""
    pos: Dict[str, List[int]] = {}
    for i, c in enumerate(terms):
        pos.setdefault(vocab.primary[c], []).append(i)
    src, dst = [], []
    for i, c in enumerate(terms):
        for a in anc[vocab.primary[c]]:
            for j in pos.get(a, ()):
                if j != i:
                    src.append(i)
                    dst.append(j)
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def propagate_max(probs, src: np.ndarray, dst: np.ndarray, device: str = "cpu",
                  max_elems: int = 1 << 25):
    """pred(c) = max(prob(c), max over (d -> c) pairs of prob(d)); torch, chunked by
    rows so a chunk gathers at most ``max_elems`` values."""
    import torch
    p = torch.as_tensor(probs, device=device)
    if len(src) == 0:
        return p.clone()
    s = torch.as_tensor(src, device=device)
    d = torch.as_tensor(dst, device=device)
    out = p.clone()
    rows = max(1, max_elems // len(src))
    for a in range(0, p.shape[0], rows):
        blk = out[a:a + rows]
        blk.scatter_reduce_(1, d.expand(blk.shape[0], -1), p[a:a + rows][:, s], "amax",
                            include_self=True)
    return out


# ------------------------------------------------------------------ protein-centric core
def _curve(scores, y: np.ndarray, ic: np.ndarray, lab_n: np.ndarray, lab_ic: np.ndarray,
           device: str, row_chunk: int = 512, rows: Optional[np.ndarray] = None,
           cols: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-threshold protein-centric statistics.

    ``scores[rows][:, cols]`` (n x m, any float dtype; ``rows``/``cols`` default to
    all, and are applied per chunk so no full copy is made) and ``y`` (n x m bool)
    over the predictable columns; ``lab_n``/``lab_ic`` (n) count every true label of
    a protein, including labels outside the m columns (DGZ-faithful mode). Rows with
    lab_n == 0 must already be removed. Torch float64, chunked by rows, all 101
    thresholds per chunk.
    """
    import torch
    n = y.shape[0]
    if rows is None:
        rows = np.arange(n)
    if len(rows) != n:
        raise FatalError(f"_curve: {len(rows)} score rows for {n} label rows")
    th = torch.tensor(THRESHOLDS, dtype=torch.float64, device=device)
    icv = torch.as_tensor(np.asarray(ic, dtype=np.float64), device=device)
    T = len(THRESHOLDS)
    tp_n = np.zeros((T, n)); pred_n = np.zeros((T, n)); tp_ic = np.zeros((T, n))
    pred_ic = np.zeros((T, n))
    for a in range(0, n, row_chunk):
        blk = np.asarray(scores[rows[a:a + row_chunk]])
        if cols is not None:
            blk = blk[:, cols]
        s = torch.as_tensor(blk, device=device).to(torch.float64)
        yy = torch.as_tensor(np.asarray(y[a:a + row_chunk], dtype=bool), device=device)
        yf = yy.to(torch.float64)
        for k in range(T):
            P = (s >= th[k]).to(torch.float64)
            tpm = P * yf
            tp_n[k, a:a + row_chunk] = tpm.sum(1).cpu().numpy()
            pred_n[k, a:a + row_chunk] = P.sum(1).cpu().numpy()
            tp_ic[k, a:a + row_chunk] = (tpm @ icv).cpu().numpy()
            pred_ic[k, a:a + row_chunk] = (P @ icv).cpu().numpy()
    lab_n = np.asarray(lab_n, dtype=np.float64)
    lab_ic = np.asarray(lab_ic, dtype=np.float64)
    prec = np.zeros(T); rec = np.zeros(T); ru = np.zeros(T); mi = np.zeros(T)
    micro_p = np.zeros(T); micro_r = np.zeros(T)
    for k in range(T):
        rec[k] = float(np.mean(tp_n[k] / lab_n))
        has = pred_n[k] > 0
        prec[k] = float(np.mean(tp_n[k][has] / pred_n[k][has])) if has.any() else 0.0
        ru[k] = float(np.mean(lab_ic - tp_ic[k]))
        mi[k] = float(np.mean(pred_ic[k] - tp_ic[k]))
        tot_pred = pred_n[k].sum()
        micro_p[k] = tp_n[k].sum() / tot_pred if tot_pred > 0 else 0.0
        micro_r[k] = tp_n[k].sum() / lab_n.sum()
    return {"precision": prec, "recall": rec, "ru": ru, "mi": mi,
            "micro_precision": micro_p, "micro_recall": micro_r}


def _trapz_by_recall(precisions: np.ndarray, recalls: np.ndarray) -> float:
    # DGZ: sorted_index = np.argsort(recalls); np.trapz(precisions, recalls)
    idx = np.argsort(recalls)
    return float(np.trapezoid(precisions[idx], recalls[idx]))


def _summarise_curve(c: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    fmax, tmax = 0.0, 0.0
    smin = 1000000.0
    for k, t in enumerate(THRESHOLDS):
        p, r = c["precision"][k], c["recall"][k]
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if fmax < f:
            fmax, tmax = f, t
        s = math.sqrt(c["ru"][k] ** 2 + c["mi"][k] ** 2)
        if smin > s:
            smin = s
    return {"fmax": float(fmax), "tmax": float(tmax), "smin": float(smin),
            "aupr_protein": _trapz_by_recall(c["precision"], c["recall"]),
            "aupr_micro": _trapz_by_recall(c["micro_precision"], c["micro_recall"])}


def protein_centric(probs, y: sp.spmatrix, ic: np.ndarray, device: str = "cpu",
                    propagate: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
    """Study-mode protein-centric metrics over a term set T (design §8.2).

    ``probs``: n x |T| sigmoid scores; ``y``: n x |T| TP labels over T; ``ic``: IC
    of the T columns; ``propagate``: ``tp_ancestor_pairs`` output for TP
    max-propagation within T (None: scores used as given). Proteins without a
    label in T are skipped (count reported)."""
    import torch
    y = sp.csr_matrix(y)
    if propagate is not None:
        probs = propagate_max(torch.as_tensor(np.asarray(probs)), *propagate, device=device)
        probs = probs.cpu().numpy()
    probs = np.asarray(probs)
    lab_n = np.asarray(y.sum(axis=1)).ravel()
    keep = np.flatnonzero(lab_n > 0)
    yk = y[keep]
    lab_ic = np.asarray(yk @ np.asarray(ic, dtype=np.float64)).ravel()
    curve = _curve(probs, yk.toarray().astype(bool), ic, lab_n[keep], lab_ic, device, rows=keep)
    out = _summarise_curve(curve)
    out.update({"n_eval": int(len(keep)), "n_skipped_no_label": int(len(lab_n) - len(keep)),
                "n_terms": int(y.shape[1])})
    return out


# ------------------------------------------------------------------ DGZ-faithful mode (§8.3)
def dgz_namespace_terms(dgz_rule, aspect: str) -> Set[str]:
    """DGZ ``go_rels.get_namespace_terms(NAMESPACES[ont])`` minus the root
    (alt-id aliases included, as DGZ's dict holds them)."""
    ns = NAMESPACES[aspect]
    out = {g for g, obj in dgz_rule.ont.items() if obj.get("namespace") == ns}
    out.discard(ROOTS[aspect])
    return out


def dgz_ic(dgz_rule, annots: Iterable[Iterable[str]]) -> Dict[str, float]:
    """DGZ ``Ontology.calculate_ic``: parents are the with_rels ``is_a`` targets
    present in the dict; a GO id absent from the counts has IC 0 (``get_ic``)."""
    cnt: Counter = Counter()
    for x in annots:
        cnt.update(set(x))
    ic: Dict[str, float] = {}
    for go_id, n in cnt.items():
        obj = dgz_rule.ont.get(go_id)
        parents = set() if obj is None else {p for p in obj["is_a"] if p in dgz_rule.ont}
        min_n = n if len(parents) == 0 else min(cnt[x] for x in parents)
        ic[go_id] = math.log(min_n / n, 2)
    return ic


def dgz_protein_metrics(preds: np.ndarray, terms: Sequence[str],
                        test_prop: Sequence[Iterable[str]], go_set: Set[str],
                        ic: Mapping[str, float], device: str = "cpu") -> Dict[str, Any]:
    """DGZ ``evaluate.py``: Fmax, Smin and AUPR of released predictions as given
    (no propagation), labels = ``prop_annotations`` filtered to ``go_set``."""
    cols = [j for j, g in enumerate(terms) if g in go_set]
    col_terms = [terms[j] for j in cols]
    col_index = {g: i for i, g in enumerate(col_terms)}
    labels = [set(g for g in x if g in go_set) for x in test_prop]
    keep = [i for i, lab in enumerate(labels) if len(lab) > 0]
    y = np.zeros((len(keep), len(cols)), dtype=bool)
    lab_n = np.zeros(len(keep)); lab_ic = np.zeros(len(keep))
    for r, i in enumerate(keep):
        for g in labels[i]:
            j = col_index.get(g)
            if j is not None:
                y[r, j] = True
        lab_n[r] = len(labels[i])
        lab_ic[r] = sum(ic.get(g, 0.0) for g in labels[i])
    icv = np.asarray([ic.get(g, 0.0) for g in col_terms], dtype=np.float64)
    curve = _curve(preds, y, icv, lab_n, lab_ic, device, rows=np.asarray(keep, dtype=np.int64),
                   cols=np.asarray(cols, dtype=np.int64))
    s = _summarise_curve(curve)
    return {"fmax": s["fmax"], "tmax": s["tmax"], "smin": s["smin"], "aupr": s["aupr_protein"],
            "n_eval": len(keep), "n_skipped_no_label": len(labels) - len(keep),
            "n_pred_columns": len(cols)}


def dgz_class_auc(preds: np.ndarray, terms: Sequence[str],
                  test_prop: Sequence[Iterable[str]]) -> Dict[str, Any]:
    """DGZ ``evaluate_terms.py``: mean ROC AUC (``roc_curve`` + ``auc``) over the
    ``terms.pkl`` columns with 0 < n_pos < n, labels = ``prop_annotations``."""
    from sklearn.metrics import auc, roc_curve
    index = {g: i for i, g in enumerate(terms)}
    n = len(test_prop)
    labels = np.zeros((n, len(terms)), dtype=np.float32)
    for i, x in enumerate(test_prop):
        for g in x:
            j = index.get(g)
            if j is not None:
                labels[i, j] = 1
    p32 = np.asarray(preds, dtype=np.float32)
    pos = labels.sum(axis=0)
    total, n_terms = 0.0, 0
    for j in range(len(terms)):
        if 0 < pos[j] < n:
            fpr, tpr, _ = roc_curve(labels[:, j], p32[:, j])
            total += auc(fpr, tpr)
            n_terms += 1
    if n_terms == 0:
        raise FatalError("dgz_class_auc: no term with 0 < n_pos < n")
    return {"auc": float(total / n_terms), "n_terms": n_terms}
