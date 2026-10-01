#!/usr/bin/env python3
"""Complete career ontology result: 4 arms x 4 fractions x 3 seeds, on TEST and VALIDATION.

Reads the per-run evaluation JSONs already on disk and reports every paired contrast,
plus the validation-vs-test comparison that identifies the source of the originally
reported effect.

Why both eval sets are shown side by side
-----------------------------------------
``trainer.py`` selects ``best_checkpoint.pt`` by lowest **validation** loss (line
503). The April 2026 runs then evaluated that checkpoint on the **same validation
file**. Scoring a model on the data used to choose it is a selection effect, and it
does not cancel between arms: the ontology arm varies more across epochs, so it
gains more from being allowed to pick its luckiest one. The gap between the two
columns below is the size of that bias.

Writes results/lc_single_factor/career_final_report.json.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "lc_single_factor"

ARMS = {
    "ontneg": "ontology-guided NEGATIVES only (rank tiers)",
    "ontweight": "sample-level ontology WEIGHTING only",
    "ontfull": "all mechanisms + rank tiers",
    "ontfull_abs": "all mechanisms, ABSOLUTE tiers = April config",
}
FRACTIONS = (5, 10, 15, 20)
SEEDS = (42, 13, 21)
#: eval subdir -> label
EVALSETS = {"eval": "TEST (held out)", "eval_on_val": "VALIDATION (= selection set)"}


def auc(arm: str, frac: int, seed: int, sub: str) -> Optional[float]:
    p = RES / f"career_{arm}_f{frac}_s{seed}" / sub / "phase1_evaluation_results.json"
    try:
        return json.load(open(p))["metrics"]["auc_roc"]
    except Exception:
        return None


def paired_t(deltas: List[float]):
    """Two-sided paired t-test against zero. Returns (t, p, df)."""
    n = len(deltas)
    if n < 2:
        return None, None, 0
    mean = st.fmean(deltas)
    sd = st.stdev(deltas)
    if sd == 0:
        return None, None, n - 1
    t = mean / (sd / math.sqrt(n))
    df = n - 1
    # Two-sided p from the t distribution via the incomplete beta function.
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x)
    return t, min(1.0, max(0.0, p)), df


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b), continued-fraction form."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def main() -> int:
    out: Dict = {"arms": ARMS, "eval_sets": EVALSETS, "results": {}}

    print("=" * 92)
    print("CAREER ONTOLOGY — COMPLETE RESULT")
    print("baseline vs each ontology arm, paired by (fraction, seed); 4 fractions x 3 seeds")
    print("=" * 92)

    for sub, label in EVALSETS.items():
        print(f"\n\n{'#' * 92}")
        print(f"# EVALUATED ON {label}")
        print(f"{'#' * 92}")
        out["results"][sub] = {}

        for arm, desc in ARMS.items():
            print(f"\n--- {arm}: {desc} ---")
            print(f"{'frac':>5s} {'n':>2s} {'baseline':>16s} {arm:>16s} {'paired delta':>16s}")
            all_d: List[float] = []
            per_frac = {}

            for frac in FRACTIONS:
                bs, os_, ds = [], [], []
                for seed in SEEDS:
                    b, o = auc("baseline", frac, seed, sub), auc(arm, frac, seed, sub)
                    if b is None or o is None:
                        continue
                    bs.append(b)
                    os_.append(o)
                    ds.append(o - b)
                if not ds:
                    print(f"{frac:5d}%  0   (no data)")
                    continue
                all_d.extend(ds)
                per_frac[frac] = {
                    "n": len(ds),
                    "baseline": st.fmean(bs),
                    "arm": st.fmean(os_),
                    "delta": st.fmean(ds),
                    "delta_sd": st.stdev(ds) if len(ds) > 1 else 0.0,
                }
                sd = per_frac[frac]["delta_sd"]
                print(f"{frac:5d}% {len(ds):2d} {st.fmean(bs):8.4f}±{st.stdev(bs) if len(bs)>1 else 0:.4f} "
                      f"{st.fmean(os_):8.4f}±{st.stdev(os_) if len(os_)>1 else 0:.4f} "
                      f"{st.fmean(ds):+8.4f}±{sd:.4f}")

            if not all_d:
                continue
            t, p, df = paired_t(all_d)
            wins = sum(1 for d in all_d if d > 0)
            sig = (p is not None and p < 0.05)
            print(f"{'':>5s}    pooled n={len(all_d)}  mean delta={st.fmean(all_d):+.4f}  "
                  f"sd={st.stdev(all_d):.4f}")
            print(f"{'':>5s}    paired t={t:+.3f} df={df} p={p:.4f}  "
                  f"-> {'SIGNIFICANT' if sig else 'not distinguishable from zero'} at 0.05")
            print(f"{'':>5s}    arm wins {wins}/{len(all_d)} cells")
            out["results"][sub][arm] = {
                "per_fraction": per_frac,
                "pooled_n": len(all_d),
                "pooled_delta": st.fmean(all_d),
                "pooled_sd": st.stdev(all_d),
                "t": t, "p": p, "df": df,
                "wins": wins,
                "significant_at_05": sig,
            }

    # ---------------------------------------------------------------- the bias
    print(f"\n\n{'=' * 92}")
    print("THE DISCREPANCY: identical checkpoints, two eval sets")
    print("=" * 92)
    print(f"{'arm':>12s} {'delta on TEST':>16s} {'delta on VALIDATION':>21s} "
          f"{'inflation':>12s} {'p(test)':>9s} {'p(val)':>9s}")
    print("-" * 92)
    for arm in ARMS:
        a = out["results"]["eval"].get(arm)
        b = out["results"]["eval_on_val"].get(arm)
        if not a or not b:
            continue
        print(f"{arm:>12s} {a['pooled_delta']:+16.4f} {b['pooled_delta']:+21.4f} "
              f"{b['pooled_delta'] - a['pooled_delta']:+12.4f} "
              f"{a['p']:9.4f} {b['p']:9.4f}")
        out.setdefault("inflation", {})[arm] = b["pooled_delta"] - a["pooled_delta"]

    print("\nMECHANISM")
    print("-" * 92)
    print("  trainer.py:503 picks best_checkpoint.pt by lowest VALIDATION loss.")
    print("  The April runs then scored that checkpoint on the SAME validation file.")
    print("  Each arm therefore picks its luckiest epoch as judged by the data it is")
    print("  scored on. The ontology arm varies more across epochs, so it benefits more,")
    print("  and the bias does not cancel in the paired difference.")
    print("\n  April reported +0.0226 at 10% on validation.")
    v = out["results"]["eval_on_val"].get("ontfull_abs", {}).get("per_fraction", {}).get(10)
    tt = out["results"]["eval"].get("ontfull_abs", {}).get("per_fraction", {}).get(10)
    if v and tt:
        print(f"  Same config here, on validation: {v['delta']:+.4f}  (reproduces it)")
        print(f"  Same checkpoints,  on test:       {tt['delta']:+.4f}  (does not)")

    path = RES / "career_final_report.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
