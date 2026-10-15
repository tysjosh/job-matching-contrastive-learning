"""Uncertainty, multiplicity and readings (protocol §8, §11.2, §12).

* Term bootstrap: one ``default_rng(20261014)`` per aspect, drawn in the fixed
  order U (unstratified, all terms), S (one block per band, band order), R
  (reg-subset terms). Common random numbers across contrasts.
* Protein bootstrap: ``default_rng(20261015)`` per aspect; the Fmax draws
  (B x n_prot) first, the P_b draws (B_P x n_prot) second.
* Two-level bases x terms: ``trials_domain.fusion_confirm.draw_indices``,
  ``boot_stats``, ``summarize`` (imported), ``default_rng(20261016)``.
* p values: ``trials_domain.fusion_confirm.boot_p`` (imported).
* Holm within a family (fixed m; ties by table order; step-down max).
* Classification rules 1-5 of §12 and the "null selected on validation" label.
* The §11.2 snapshot rule.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from go_function_fusion.metrics import BAND_NAMES
from trials_domain.fusion_confirm import boot_p, boot_stats, draw_indices, summarize

TERM_SEED = 20261014
PROTEIN_SEED = 20261015
TWO_LEVEL_SEED = 20261016
S2_SEED = 20261011
B_DEFAULT = 2000
B_P_DEFAULT = 200
NULL_LABEL = "null selected on validation"
HIGH_BANDS = ("101-1000", ">1000")
FLOORS = {"P": 0.01, "Sc": 0.01, "Ty": 0.01, "Hc": 0.002, "Hf": 0.01}


# ------------------------------------------------------------------ term bootstrap
@dataclass
class TermDraws:
    U: np.ndarray
    S: Dict[str, np.ndarray]                    # band -> (B, n_b) positions within the band
    band_members: Dict[str, np.ndarray]         # band -> term positions
    R: Optional[np.ndarray]
    reg_members: np.ndarray
    B: int


def term_draws(n_terms: int, band_idx: np.ndarray, reg_mask: Optional[np.ndarray],
               B: int = B_DEFAULT, seed: int = TERM_SEED) -> TermDraws:
    """Draw sets U, S, R in the fixed §8 order from one generator. A band with 0
    terms yields no draw block (and consumes no random numbers)."""
    rng = np.random.default_rng(seed)
    U = rng.integers(0, n_terms, size=(B, n_terms))
    S, members = {}, {}
    for i, name in enumerate(BAND_NAMES):
        mem = np.flatnonzero(np.asarray(band_idx) == i)
        members[name] = mem
        if len(mem):
            S[name] = rng.integers(0, len(mem), size=(B, len(mem)))
    reg = np.flatnonzero(reg_mask) if reg_mask is not None else np.zeros(0, dtype=np.int64)
    R = rng.integers(0, len(reg), size=(B, len(reg))) if len(reg) else None
    return TermDraws(U=U, S=S, band_members=members, R=R, reg_members=reg, B=B)


def _summ(point: Optional[float], T: Optional[np.ndarray]) -> Dict[str, Any]:
    if point is None or T is None:
        return {"estimate": None, "ci95": None, "p": None, "computed": False}
    T = np.asarray(T, dtype=np.float64)
    T = T[np.isfinite(T)]
    return {"estimate": float(point), "ci95": [float(np.percentile(T, 2.5)),
                                               float(np.percentile(T, 97.5))],
            "p": boot_p(T), "computed": True, "B": int(len(T))}


def contrast_all(d: np.ndarray, dr: TermDraws) -> Dict[str, Any]:
    """Δ[all] with draw set U."""
    d = np.asarray(d, dtype=np.float64)
    return _summ(float(np.mean(d)), d[dr.U].mean(axis=1))


def contrast_band(d: np.ndarray, dr: TermDraws, band: str) -> Dict[str, Any]:
    """Δ[band] with that band's block of draw set S."""
    mem = dr.band_members[band]
    if band not in dr.S:
        return _summ(None, None)
    v = np.asarray(d, dtype=np.float64)[mem]
    return _summ(float(v.mean()), v[dr.S[band]].mean(axis=1))


def _high(d: np.ndarray, dr: TermDraws) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Δ[>100]: pooled mean over the 101-1000 and >1000 terms (weights = band sizes)."""
    parts = [b for b in HIGH_BANDS if b in dr.S]
    if not parts:
        return None, None
    d = np.asarray(d, dtype=np.float64)
    sums = sum(d[dr.band_members[b]][dr.S[b]].sum(axis=1) for b in parts)
    n = sum(len(dr.band_members[b]) for b in parts)
    point = float(np.concatenate([d[dr.band_members[b]] for b in parts]).mean())
    return point, sums / n


def contrast_scarcity(d: np.ndarray, dr: TermDraws, low: str) -> Dict[str, Any]:
    """Sc = Δ[low] − Δ[>100] (low = '0' for Sc0, '1-10' for Sc1), draw set S."""
    if low not in dr.S:
        return _summ(None, None)
    hp, hT = _high(d, dr)
    if hp is None:
        return _summ(None, None)
    v = np.asarray(d, dtype=np.float64)[dr.band_members[low]]
    return _summ(float(v.mean()) - hp, v[dr.S[low]].mean(axis=1) - hT)


def contrast_reg(d: np.ndarray, dr: TermDraws) -> Dict[str, Any]:
    """Δ[reg-subset] with draw set R (``d`` over all evaluated terms)."""
    if dr.R is None:
        return _summ(None, None)
    v = np.asarray(d, dtype=np.float64)[dr.reg_members]
    return _summ(float(v.mean()), v[dr.R].mean(axis=1))


# ------------------------------------------------------------------ protein bootstrap
def protein_draws(n_prot: int, B: int = B_DEFAULT, B_P: int = B_P_DEFAULT,
                  seed: int = PROTEIN_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """(Fmax draws, P_b draws), drawn in that order from one generator."""
    rng = np.random.default_rng(seed)
    D_f = rng.integers(0, n_prot, size=(B, n_prot))
    D_p = rng.integers(0, n_prot, size=(B_P, n_prot))
    return D_f, D_p


def multiplicities(D: np.ndarray, n: int) -> np.ndarray:
    """(B, n) counts of each protein in each replicate."""
    W = np.zeros((D.shape[0], n), dtype=np.float64)
    for b in range(D.shape[0]):
        W[b] = np.bincount(D[b], minlength=n)
    return W


def fmax_contrast(tab_a: Mapping[str, np.ndarray], tab_b: Mapping[str, np.ndarray],
                  D: np.ndarray) -> Dict[str, Any]:
    """Fmax(a) − Fmax(b) with the protein bootstrap (shared draws)."""
    from go_function_fusion.metrics import fmax_boot, fmax_smin
    n = tab_a["recall"].shape[1]
    if tab_b["recall"].shape[1] != n:
        raise ValueError("protein tables differ in rows")
    W = multiplicities(D, n)
    point = fmax_smin(tab_a)["fmax"] - fmax_smin(tab_b)["fmax"]
    return _summ(point, fmax_boot(tab_a, W) - fmax_boot(tab_b, W))


def protein_boot_P(S_fused: np.ndarray, S_base: np.ndarray, Y: np.ndarray, D_P: np.ndarray
                   ) -> Dict[str, Any]:
    """Descriptive protein-bootstrap CI for P_b: per replicate, every term's AUC on
    the resampled proteins (multiplicity kept), undefined terms dropped."""
    from go_function_fusion.metrics import rank_auc
    T = np.empty(D_P.shape[0])
    for b, idx in enumerate(D_P):
        d = rank_auc(S_fused[idx], Y[idx]) - rank_auc(S_base[idx], Y[idx])
        T[b] = np.nanmean(d)
    point = float(np.nanmean(rank_auc(S_fused, Y) - rank_auc(S_base, Y)))
    return _summ(point, T)


# ------------------------------------------------------------------ two-level (bases x terms)
def two_level(D: np.ndarray, B: int = B_DEFAULT, seed: int = TWO_LEVEL_SEED) -> Dict[str, Any]:
    """Pooled estimate over bases (rows) x terms (columns); descriptive. With 2-3
    bases the base level under-covers (stated beside every pooled number)."""
    rng = np.random.default_rng(seed)
    C, P = draw_indices(rng, D.shape[0], D.shape[1], B)
    s = summarize(D, C, P)
    s.pop("_T", None)
    s["note"] = "2-3 bases: the base level under-covers"
    return s


# ------------------------------------------------------------------ Holm and classification
def holm(names: Sequence[str], pvals: Mapping[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Holm step-down within a family. ``names`` is the §6.1 table order (fixed m;
    a null-selected contrast keeps its slot with p = 1; a missing contrast (p None)
    is dropped and m shrinks). Ties in p are broken by table order."""
    present = [n for n in names if pvals.get(n) is not None]
    m = len(present)
    order = sorted(range(m), key=lambda i: (float(pvals[present[i]]), i))
    out: Dict[str, Optional[float]] = {n: None for n in names}
    run = 0.0
    for j, i in enumerate(order, start=1):
        run = max(run, min(1.0, (m - j + 1) * float(pvals[present[i]])))
        out[present[i]] = run
    return out


def classify(c: Mapping[str, Any], floor: float, p_adj: Optional[float]) -> str:
    """§12 rules 1-5; ``c`` has estimate and ci95."""
    if c.get(NULL_LABEL) or c.get("label") == NULL_LABEL:
        return NULL_LABEL
    if not c.get("computed"):
        return "not computed"
    est, (lo, hi) = c["estimate"], c["ci95"]
    if hi < 0 and est <= -floor:
        return "harm"
    if p_adj is not None and p_adj < 0.05 and lo > 0 and est >= floor:
        return "supported"
    if -floor < lo and hi < floor:
        return "practically null"
    if lo > 0 and est < floor:
        return "below floor"
    return "inconclusive"


def null_contrast() -> Dict[str, Any]:
    """A confirmatory contrast whose frozen weight is its null cell (§12)."""
    return {"estimate": 0.0, "ci95": None, "p": 1.0, "computed": True, "label": NULL_LABEL,
            "reading": "not tested on test"}


def hf_or_null(alpha_star: float, tab_alpha: Mapping[str, np.ndarray],
               tab_alpha0: Mapping[str, np.ndarray], D: np.ndarray) -> Dict[str, Any]:
    """Hf_b = Fmax(PC(alpha*)) − Fmax(PC(alpha = 0)) at frozen (c_PC*, tau_PC*);
    exactly 0 with the null label when alpha* = 0 (§12)."""
    if float(alpha_star) == 0.0:
        return null_contrast()
    return fmax_contrast(tab_alpha, tab_alpha0, D)


def weighted_or_null(w_star: float, d: np.ndarray, dr: TermDraws, reg: bool = False
                     ) -> Dict[str, Any]:
    """Hc (draw set U) or Ty (draw set R): the null label when the frozen weight is 0."""
    if float(w_star) == 0.0:
        return null_contrast()
    return contrast_reg(d, dr) if reg else contrast_all(d, dr)


def d_struct(auc_fused: np.ndarray, auc_shuffled: np.ndarray, dr: TermDraws) -> Dict[str, Any]:
    """D_struct_b = Δ_{AP-T(c*,tau*) − AP-Tshuf(c*,tau*)}[all] (frozen scorings, U)."""
    return contrast_all(np.asarray(auc_fused) - np.asarray(auc_shuffled), dr)


def d_typed(w_R_star: float, auc_r: np.ndarray, auc_r_shuf: np.ndarray, dr: TermDraws
            ) -> Dict[str, Any]:
    """D_typed_b = Δ_{AP-T+R(w_R*) − AP-T+Rshuf(w_R*)}[reg-subset] (frozen, R); ≡ 0
    with the null label when w_R* = 0."""
    if float(w_R_star) == 0.0:
        out = null_contrast()
        out["reading"] = "null selected on validation"
        return out
    return contrast_reg(np.asarray(auc_r) - np.asarray(auc_r_shuf), dr)


def ci_low_positive(c: Mapping[str, Any]) -> bool:
    return bool(c.get("computed") and c.get("ci95") and c["ci95"][0] > 0)


# ------------------------------------------------------------------ §11.2 snapshot rule
RUN_MARKERS = ("epochs.jsonl", "eval.json", "scores.npz", "run.json", "DONE.json")


def grid_snapshot(runs_dir: Path) -> Dict[str, int]:
    """Directories under ``runs_dir`` containing any run marker, and ``*.CLAIM``
    files (listing only; no file is opened)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return {"started_run_dirs": 0, "claim_files": 0}
    started, claims = set(), 0
    for dirpath, _dirs, files in os.walk(runs_dir):
        fs = set(files)
        if fs & set(RUN_MARKERS):
            started.add(dirpath)
        claims += sum(1 for f in files if f.endswith(".CLAIM"))
    return {"started_run_dirs": len(started), "claim_files": int(claims)}


def s2_remote_eligible(snapshot: Optional[Mapping[str, Any]], runs_dir: Path,
                       gf2_epoch_seconds: Optional[float]) -> Tuple[bool, Optional[str]]:
    """§11.2: remote parameters are eligible for S2 only if both snapshot counts
    were 0 at G-F2 and no run's ``epochs.jsonl`` is older than the G-F2 timestamp."""
    if snapshot is None or gf2_epoch_seconds is None:
        return False, "no_GF2_snapshot"
    if int(snapshot.get("started_run_dirs", 1)) != 0 or int(snapshot.get("claim_files", 1)) != 0:
        return False, "grid_started_before_GF2"
    for dirpath, _d, files in os.walk(Path(runs_dir)):
        if "epochs.jsonl" in files:
            if os.path.getmtime(os.path.join(dirpath, "epochs.jsonl")) < gf2_epoch_seconds:
                return False, "mtime_check_failed"
    return True, None
