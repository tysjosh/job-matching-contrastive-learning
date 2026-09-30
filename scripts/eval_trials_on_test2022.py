#!/usr/bin/env python3
"""Evaluate trials label-budget checkpoints on the held-out 2022 TEST split.

Every trials number so far came from ``trec_ct_splits/validation.jsonl`` (2021,
n=3,108) — which is also the ``validation_path`` used to pick ``best_checkpoint.pt``
by lowest validation loss. Scoring a model on the set that selected it is a
selection effect, and it was measured on the career side to inflate
negative-selection arms by +0.012 to +0.021 AUC — the same order as the +0.0408
claimed here for MeSH-vs-random negatives.

``test.jsonl`` is the genuine temporal holdout: 35,394 records, all 2022 topics,
never used for training or checkpoint selection. It is 11.4x validation
(70,788 texts vs 6,216), so the first arm pays the encoding cost and the rest reuse
the module-level text cache in ``run_phase1_embedding_evaluation`` — which is why
all arms are evaluated in ONE process here.

Output goes to ``<run>/eval_test2022/`` alongside the existing ``eval_fixed/``
(validation) results so the two can be compared directly rather than overwriting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_SCRIPT = ROOT / "run_phase1_embedding_evaluation.py"
DEFAULT_TEST = ROOT / "preprocess" / "trec_ct_splits" / "test.jsonl"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("_p1eval", EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+",
                    default=["full", "low_ontology", "low_random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "label_budget")
    ap.add_argument("--eval-subdir", default="eval_test2022")
    args = ap.parse_args(argv)

    ev = load_eval_module()
    collected = {}

    for seed in args.seeds:
        for arm in args.arms:
            run = args.results_dir / f"{arm}_s{seed}"
            ckpt = run / "best_checkpoint.pt"
            cfg = run / "training_config.json"
            out = run / args.eval_subdir
            if not ckpt.exists():
                print(f"[skip] {arm} s{seed}: no checkpoint", flush=True)
                continue

            print(f"[eval] {arm} s{seed} on {args.dataset.name} ...", flush=True)
            saved, sys.argv = sys.argv, [
                "run_phase1_embedding_evaluation.py",
                "--dataset", str(args.dataset),
                "--checkpoint", str(ckpt),
                "--config", str(cfg),
                "--output-dir", str(out),
            ]
            t0 = time.time()
            try:
                ev.main()
            except SystemExit:
                pass
            except Exception as exc:
                print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
                sys.argv = saved
                continue
            finally:
                sys.argv = saved

            path = out / "phase1_evaluation_results.json"
            if path.exists():
                r = json.load(open(path))
                collected[(arm, seed)] = r
                print(f"### {arm} s{seed}  {(time.time()-t0)/60:.1f} min  "
                      f"AUC={r['metrics']['auc_roc']:.4f}", flush=True)

    # ------------------------------------------------------------- comparison
    print("\n" + "=" * 78)
    print("TRIALS ON HELD-OUT 2022 TEST vs VALIDATION (the selection set)")
    print("=" * 78)
    print(f"{'arm':16s} {'seed':>4s} {'TEST 2022':>10s} {'VALIDATION':>11s} {'shift':>8s}")
    print("-" * 78)
    rows = {}
    for (arm, seed), r in collected.items():
        v = args.results_dir / f"{arm}_s{seed}" / "eval_fixed" / "phase1_evaluation_results.json"
        val = json.load(open(v))["metrics"]["auc_roc"] if v.exists() else None
        t = r["metrics"]["auc_roc"]
        rows[(arm, seed)] = (t, val)
        print(f"{arm:16s} {seed:4d} {t:10.4f} "
              f"{(f'{val:.4f}' if val else 'n/a'):>11s} "
              f"{(f'{t-val:+.4f}' if val else 'n/a'):>8s}")

    for seed in args.seeds:
        o = rows.get(("low_ontology", seed))
        rn = rows.get(("low_random", seed))
        f = rows.get(("full", seed))
        if o and rn:
            print(f"\nseed {seed} — ontology effect at equal label budget "
                  f"(low_ontology - low_random):")
            print(f"   on TEST 2022  {o[0]-rn[0]:+.4f}")
            if o[1] and rn[1]:
                print(f"   on validation {o[1]-rn[1]:+.4f}   <- the previously reported figure")
        if f and o:
            print(f"seed {seed} — cost of the low budget (full - low_ontology):")
            print(f"   on TEST 2022  {f[0]-o[0]:+.4f}")

    # per-grade breakdown, which the evaluation script now emits
    print("\n" + "=" * 78)
    print("PER-GRADE ON TEST 2022 (additive to pooled AUC)")
    print("=" * 78)
    print(f"{'arm':16s} {'g2 vs g0':>10s} {'g2 vs g1':>10s} {'g1 vs g0':>10s}")
    print("-" * 78)
    for (arm, seed), r in collected.items():
        pg = (r.get("per_grade") or {}).get("pairwise_auc", {})
        def g(k):
            v = pg.get(k)
            return f"{v:.4f}" if v is not None else "n/a"
        print(f"{arm:16s} {g('eligible_vs_not_relevant'):>10s} "
              f"{g('eligible_vs_ineligible'):>10s} "
              f"{g('ineligible_vs_not_relevant'):>10s}")

    summary = args.results_dir / "trials_test2022_summary.json"
    json.dump({f"{a}_s{s}": {"test2022": t, "validation": v}
               for (a, s), (t, v) in rows.items()},
              open(summary, "w"), indent=2)
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
