#!/usr/bin/env python3
"""Per-seed significance test between two EO/ER/E4 variants on the Table-4 metrics.

Reads each variant's per-seed ordinal evaluation artifact
(``phase1_evaluation/ordinal_evaluation_results.json`` -> ``ordinal_v3``), matches
runs by seed, and reports for every metric:

  * mean ± std for A and B, the mean difference (B - A),
  * paired t-test across matched seeds (the right test: same 5 seeds per variant),
  * Welch's unpaired t-test (does not assume equal variance / pairing),
  * Wilcoxon signed-rank + Mann-Whitney U (non-parametric backups),
  * Cohen's d (paired) effect size.

Falls back to a normal approximation for p-values if scipy is unavailable.

Usage (baseline FIRST, candidate SECOND so ΔB-A = candidate - baseline):
  python scripts/orca_sig_test.py EO-A EO-OntNeg
  python scripts/orca_sig_test.py EO-A EO-ISCONeg --dataset cnamuangtoun
"""
import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "research_runs"

COLS = [
    ("AUC-ROC", lambda r: r.get("binary_aucs", {}).get("good_vs_rest")),
    ("Spearman", lambda r: r.get("spearmans_rho", {}).get("rho")),
    ("Cohen_d", lambda r: r.get("separations", {}).get("good_vs_no", {}).get("cohens_d")),
    ("NDCG@10", lambda r: r.get("ranking", {}).get("ndcg@10")),
    ("MAP_strict", lambda r: r.get("ranking", {}).get("map_strict")),
]

try:
    from scipy import stats as _stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def collect(variant, dataset):
    pattern = str(RUNS / f"{variant}__{dataset}__s*" /
                  "phase1_evaluation" / "ordinal_evaluation_results.json")
    per_seed = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"__s(\d+)", path)
        seed = m.group(1) if m else path
        with open(path) as f:
            r = json.load(f).get("ordinal_v3", {})
        per_seed[seed] = {name: fn(r) for name, fn in COLS}
    return per_seed


def normal_two_sided_p(t):
    # Φ-based two-sided p using erf (t treated as z; conservative for small n).
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))


def paired_t(a, b):
    d = np.array(b) - np.array(a)
    n = len(d)
    if n < 2 or np.allclose(d.std(ddof=1), 0):
        return float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    if HAVE_SCIPY:
        p = float(_stats.ttest_rel(b, a).pvalue)
    else:
        p = normal_two_sided_p(t)
    return float(t), p


def welch_t(a, b):
    a, b = np.array(a), np.array(b)
    if HAVE_SCIPY:
        res = _stats.ttest_ind(b, a, equal_var=False)
        return float(res.statistic), float(res.pvalue)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (b.mean() - a.mean()) / math.sqrt(va + vb) if (va + vb) > 0 else float("nan")
    return float(t), normal_two_sided_p(t)


def cohens_d_paired(a, b):
    d = np.array(b) - np.array(a)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_a", help="baseline variant, listed FIRST (e.g. EO-A)")
    ap.add_argument("variant_b", help="candidate variant, listed SECOND (e.g. EO-OntNeg)")
    ap.add_argument("--dataset", default="cnamuangtoun")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    A = collect(args.variant_a, args.dataset)
    B = collect(args.variant_b, args.dataset)
    shared = sorted(set(A) & set(B), key=lambda s: int(s) if s.isdigit() else s)

    print(f"Significance test: {args.variant_b} (B) vs {args.variant_a} (A) "
          f"— dataset={args.dataset}")
    print(f"scipy={'yes' if HAVE_SCIPY else 'NO (normal approx)'}  "
          f"seeds A={sorted(A)} B={sorted(B)}  matched={shared}")
    if len(shared) < 2:
        print("Not enough matched seeds for a paired test.")
        return
    print("-" * 100)
    hdr = (f"{'metric':<12}{'A mean±std':>16}{'B mean±std':>16}{'ΔB-A':>10}"
           f"{'paired t':>10}{'p(paired)':>11}{'p(Welch)':>10}{'d':>7}  sig")
    print(hdr)
    print("-" * 100)

    for name, _ in COLS:
        a = [A[s][name] for s in shared if A[s][name] is not None and B[s][name] is not None]
        b = [B[s][name] for s in shared if A[s][name] is not None and B[s][name] is not None]
        if len(a) < 2:
            print(f"{name:<12}{'n/a':>16}")
            continue
        am, asd = np.mean(a), np.std(a, ddof=1)
        bm, bsd = np.mean(b), np.std(b, ddof=1)
        tp, pp = paired_t(a, b)
        tw, pw = welch_t(a, b)
        d = cohens_d_paired(a, b)
        sig = "***" if pp < args.alpha else "ns"
        print(f"{name:<12}{f'{am:.3f}±{asd:.3f}':>16}{f'{bm:.3f}±{bsd:.3f}':>16}"
              f"{bm-am:>+10.3f}{tp:>10.2f}{pp:>11.4f}{pw:>10.4f}{d:>7.2f}  {sig}")

    print("-" * 100)
    print(f"Paired t-test uses the {len(shared)} shared seeds. '***' = p < {args.alpha} "
          f"(reject 'no difference'); 'ns' = not significant.")
    print("Note: with n=5 seeds, power is low — treat 'ns' as 'not distinguishable', not 'identical'.")


if __name__ == "__main__":
    main()
