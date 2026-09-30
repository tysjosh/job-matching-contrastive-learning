#!/usr/bin/env python3
"""Evaluate a learning-curve sweep in ONE process and report the paired contrast.

One process because ``run_phase1_embedding_evaluation`` caches frozen text-encoder
outputs in a module-level dict that dies with the interpreter. A shell loop over 30
checkpoints re-encodes the same evaluation texts 30 times; a single process encodes
them once.

The contrast is reported **paired by (fraction, seed)**, not as a difference of
arm means. Seed variance on this data is +/-0.013 AUC, comparable to the effect
being measured, so unpaired means can invert the sign of a real effect or invent
one. Pairing removes the seed as a source of variance because both arms of a cell
saw the same data, the same seed and the same frozen embeddings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_SCRIPT = ROOT / "run_phase1_embedding_evaluation.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("_p1eval", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_one(ev, checkpoint: Path, config: Path, dataset: Path,
                 out_dir: Path) -> Optional[Dict[str, Any]]:
    argv = [
        "run_phase1_embedding_evaluation.py",
        "--dataset", str(dataset),
        "--checkpoint", str(checkpoint),
        "--config", str(config),
        "--output-dir", str(out_dir),
    ]
    saved, sys.argv = sys.argv, argv
    try:
        ev.main()
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return None
    finally:
        sys.argv = saved

    path = out_dir / "phase1_evaluation_results.json"
    return json.load(open(path)) if path.exists() else None


def paired_t(diffs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Paired t statistic and two-sided p for small n, via a t-distribution CDF.

    Returns ``(None, None)`` for n < 2 rather than a misleading number.
    """
    n = len(diffs)
    if n < 2:
        return None, None
    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    if sd == 0:
        return (math.inf if mean else 0.0), (0.0 if mean else 1.0)
    t = mean / (sd / math.sqrt(n))
    df = n - 1
    # Two-sided p from the regularized incomplete beta function.
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x)
    return t, min(max(p, 0.0), 1.0)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 200,
            eps: float = 3e-16) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", default="career")
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT / "results" / "lc_single_factor")
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "preprocess" / "learning_curve_v7" / "frac_100" / "test.jsonl")
    ap.add_argument("--fractions", nargs="+", type=int,
                    default=[10, 25, 50, 75, 100])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 21])
    ap.add_argument("--arms", nargs="+", default=["baseline", "ontneg"])
    ap.add_argument("--eval-subdir", default="eval")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    ev = _load_eval_module()
    # results[(pct, seed)][arm] = evaluation dict
    cells: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}

    for pct in args.fractions:
        for seed in args.seeds:
            for arm in args.arms:
                run = args.results_dir / f"{args.domain}_{arm}_f{pct}_s{seed}"
                ckpt = run / "best_checkpoint.pt"
                cfg = run / "training_config.json"
                out = run / args.eval_subdir
                existing = out / "phase1_evaluation_results.json"

                if not ckpt.exists():
                    print(f"[skip] {arm} f{pct} s{seed}: no checkpoint")
                    continue
                if not cfg.exists():
                    print(f"[skip] {arm} f{pct} s{seed}: no training_config.json")
                    continue
                if existing.exists() and not args.force:
                    print(f"[cached] {arm} f{pct} s{seed}")
                    cells.setdefault((pct, seed), {})[arm] = json.load(open(existing))
                    continue

                print(f"[eval] {arm} f{pct} s{seed} ...")
                res = evaluate_one(ev, ckpt, cfg, args.dataset, out)
                if res:
                    cells.setdefault((pct, seed), {})[arm] = res

    def auc(r):
        return r["metrics"]["auc_roc"]

    base_arm, ont_arm = args.arms[0], args.arms[-1]

    print("\n" + "=" * 78)
    print(f"LEARNING CURVE — {args.domain}, single factor "
          f"({base_arm} vs {ont_arm}), paired by (fraction, seed)")
    print(f"eval set: {args.dataset}")
    print("=" * 78)
    print(f"{'frac':>6s} {'n':>3s} {base_arm:>16s} {ont_arm:>16s} "
          f"{'paired delta':>16s}")
    print("-" * 78)

    curve = []
    all_diffs: List[float] = []
    for pct in args.fractions:
        pairs = []
        for seed in args.seeds:
            cell = cells.get((pct, seed), {})
            if base_arm in cell and ont_arm in cell:
                pairs.append((auc(cell[base_arm]), auc(cell[ont_arm])))
        if not pairs:
            print(f"{pct:5d}% {'0':>3s}      (no complete pairs)")
            continue
        b = [p[0] for p in pairs]
        o = [p[1] for p in pairs]
        d = [y - x for x, y in pairs]
        all_diffs.extend(d)
        sd = st.stdev(d) if len(d) > 1 else 0.0
        bs = st.stdev(b) if len(b) > 1 else 0.0
        os_ = st.stdev(o) if len(o) > 1 else 0.0
        print(f"{pct:5d}% {len(pairs):3d} {st.mean(b):.4f}±{bs:.4f} "
              f"{st.mean(o):.4f}±{os_:.4f} {st.mean(d):+.4f}±{sd:.4f}")
        curve.append({"fraction": pct, "n_pairs": len(pairs),
                      base_arm: st.mean(b), ont_arm: st.mean(o),
                      "delta": st.mean(d), "delta_sd": sd})

    if all_diffs:
        t, p = paired_t(all_diffs)
        print("-" * 78)
        print(f"pooled over all fractions: n={len(all_diffs)} pairs, "
              f"mean delta={st.mean(all_diffs):+.4f}")
        if t is not None:
            print(f"paired t={t:+.3f}, two-sided p={p:.4f}")
            verdict = ("distinguishable from zero" if p < 0.05
                       else "NOT distinguishable from zero")
            print(f"  -> the ontology effect is {verdict} at alpha=0.05.")
        wins = sum(1 for d in all_diffs if d > 0)
        print(f"cells where the ontology arm wins: {wins}/{len(all_diffs)}")

    # Monotonicity is the substantive claim for a data-efficiency argument: the
    # ontology should help MOST when data is scarce. A flat or non-monotonic
    # profile does not support that claim even if the pooled mean is positive.
    if len(curve) > 1:
        deltas = [c["delta"] for c in curve]
        decreasing = all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))
        profile = ", ".join(
            "{}%={:+.4f}".format(c["fraction"], c["delta"]) for c in curve)
        print(f"\ndelta by fraction: {profile}")
        print("monotonically decreasing with more data: "
              f"{'YES' if decreasing else 'NO'}"
              + ("" if decreasing else
                 "  <- does not support 'ontology substitutes for data'"))

    # Keyed by the arms compared, not just the domain: several single-factor arms
    # are evaluated against the same baseline, and a domain-only name meant each
    # sweep silently overwrote the previous one's curve (the ontneg result was
    # lost this way and had to be recomputed from the per-run evals).
    summary = args.results_dir / f"summary_{args.domain}_{'_vs_'.join(args.arms)}.json"
    summary.write_text(json.dumps({
        "domain": args.domain,
        "dataset": str(args.dataset),
        "arms": [base_arm, ont_arm],
        "curve": curve,
        "n_pairs_total": len(all_diffs),
        "pooled_delta": st.mean(all_diffs) if all_diffs else None,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
