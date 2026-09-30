#!/usr/bin/env python3
"""Test the two claims an equivalence-framed abstract actually makes.

1. DATA REPLACEMENT. "How much labeled data can ontology guidance replace" is a
   HORIZONTAL claim: ontology at X% should match the baseline at some fraction
   larger than X. That is not the same comparison as the vertical
   ontology-vs-baseline-at-equal-data delta we measured, so it is tested directly
   here by locating each arm's AUC on the baseline's own curve.

2. EQUIVALENCE (TOST). A null p-value does not establish parity. Parity requires
   the confidence interval on the difference to fall inside a stated margin. This
   reports the tightest margin the current data can support, which is the honest
   version of "we establish parity".
"""

from __future__ import annotations

import json
import math
import statistics as st

RES = "results/lc_single_factor"
FRACTIONS = (5, 10, 15, 20)
SEEDS = (42, 13, 21)
ARMS = ("ontneg", "ontweight", "ontfull", "ontfull_abs")

#: t critical value, one-sided 0.95, df=11 (n=12 paired observations).
T_CRIT_DF11 = 1.796


def auc(arm: str, frac: int, seed: int):
    path = f"{RES}/career_{arm}_f{frac}_s{seed}/eval/phase1_evaluation_results.json"
    try:
        return json.load(open(path))["metrics"]["auc_roc"]
    except Exception:
        return None


def mean_auc(arm: str, frac: int):
    vals = [v for v in (auc(arm, frac, s) for s in SEEDS) if v is not None]
    return st.fmean(vals) if vals else None


def equivalent_fraction(value: float, curve: dict):
    """Where ``value`` sits on the baseline curve, by linear interpolation."""
    fr = sorted(curve)
    if value <= curve[fr[0]]:
        return None, f"<={fr[0]}%"
    if value >= curve[fr[-1]]:
        return None, f">={fr[-1]}%"
    for lo, hi in zip(fr, fr[1:]):
        if curve[lo] <= value <= curve[hi]:
            span = curve[hi] - curve[lo]
            pos = lo + (hi - lo) * ((value - curve[lo]) / span if span else 0)
            return pos, f"{pos:.1f}%"
    return None, "?"


def main() -> int:
    base = {f: mean_auc("baseline", f) for f in FRACTIONS}

    print("=" * 86)
    print("1) DATA-REPLACEMENT TEST")
    print("   The abstract's claim is horizontal: ontology at X% should equal")
    print("   baseline at some fraction > X. Testing that directly.")
    print("=" * 86)
    print("baseline curve:  " + "   ".join(f"{f}% = {base[f]:.4f}" for f in FRACTIONS))
    print()
    print(f"{'arm':13s} {'at':>4s} {'AUC':>8s} {'== baseline at':>16s} {'multiplier':>12s}")
    print("-" * 86)
    for arm in ARMS:
        for f in FRACTIONS:
            a = mean_auc(arm, f)
            pos, label = equivalent_fraction(a, base)
            mult = f"{pos/f:.2f}x" if pos else "n/a"
            print(f"{arm:13s} {f:3d}% {a:8.4f} {label:>16s} {mult:>12s}")
        print()

    print("=" * 86)
    print("2) EQUIVALENCE (TOST) — tightest margin at which parity can be claimed")
    print("   90% CI on the paired delta (n=12). Equivalence holds within the")
    print("   larger absolute bound; a claim tighter than that is unsupported.")
    print("=" * 86)
    print(f"{'arm':13s} {'mean':>9s} {'sd':>8s} {'n':>3s} {'90% CI':>22s} {'margin':>9s}")
    print("-" * 86)
    margins = {}
    for arm in ARMS:
        d = [auc(arm, f, s) - auc("baseline", f, s)
             for f in FRACTIONS for s in SEEDS
             if auc(arm, f, s) is not None and auc("baseline", f, s) is not None]
        m, sd, n = st.fmean(d), st.stdev(d), len(d)
        se = sd / math.sqrt(n)
        lo, hi = m - T_CRIT_DF11 * se, m + T_CRIT_DF11 * se
        margin = max(abs(lo), abs(hi))
        margins[arm] = margin
        print(f"{arm:13s} {m:+9.4f} {sd:8.4f} {n:3d} "
              f"[{lo:+.4f}, {hi:+.4f}] {margin:9.4f}")

    print("\ninterpretation")
    print("-" * 86)
    best = min(margins, key=margins.get)
    print(f"  Tightest supportable parity claim: {best} within "
          f"+/-{margins[best]:.4f} AUC.")
    print(f"  Loosest: {max(margins, key=margins.get)} within "
          f"+/-{max(margins.values()):.4f} AUC.")
    print("  With 3 seeds and per-cell sd of 0.007-0.025, margins below ~0.01 AUC")
    print("  are not reachable; claiming parity at a tighter bound would overstate")
    print("  the evidence. More seeds narrow this as 1/sqrt(n).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
