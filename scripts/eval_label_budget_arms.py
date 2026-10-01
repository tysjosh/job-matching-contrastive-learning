#!/usr/bin/env python3
"""Evaluate several label-budget checkpoints in ONE process, then summarize.

Why one process: ``run_phase1_embedding_evaluation`` caches frozen text-encoder
outputs in a module-level dict keyed by the exact text. That cache dies with the
process, so a shell loop over N checkpoints re-encodes the same 6,216 validation
texts N times (~8 min each). Evaluating in a single process encodes them once and
reuses them for every subsequent checkpoint.

Reports pooled AUC-ROC **and** the graded breakdown side by side. Both are needed:
pooled AUC is the comparable headline, while the per-grade contrasts show whether
a gain comes from topical relevance or from the eligibility judgement — which on
this dataset diverge sharply, and which the pooled figure averages together in a
ratio determined by the split's grade composition rather than by the model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Run the evaluation script's main() for one checkpoint via argv."""
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
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return None
    finally:
        sys.argv = saved

    path = out_dir / "phase1_evaluation_results.json"
    return json.load(open(path)) if path.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT / "results" / "label_budget")
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "preprocess" / "trec_ct_splits" / "validation.jsonl")
    ap.add_argument("--arms", nargs="+",
                    default=["full", "low_ontology", "low_random"])
    ap.add_argument("--seeds", nargs="+", default=["42", "13", "21"])
    ap.add_argument("--eval-subdir", default="eval_fixed")
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate even when a results file already exists")
    args = ap.parse_args(argv)

    ev = _load_eval_module()
    collected: Dict[str, Dict[str, Dict[str, Any]]] = {a: {} for a in args.arms}

    for seed in args.seeds:
        for arm in args.arms:
            run_dir = args.results_dir / f"{arm}_s{seed}"
            ckpt = run_dir / "best_checkpoint.pt"
            cfg = run_dir / "training_config.json"
            out = run_dir / args.eval_subdir
            existing = out / "phase1_evaluation_results.json"

            if not ckpt.exists():
                print(f"[skip] {arm} s{seed}: no checkpoint yet")
                continue
            if existing.exists() and not args.force:
                print(f"[cached] {arm} s{seed}")
                collected[arm][seed] = json.load(open(existing))
                continue

            print(f"[eval] {arm} s{seed} ...")
            result = evaluate_one(ev, ckpt, cfg, args.dataset, out)
            if result:
                collected[arm][seed] = result

    # ---------------------------------------------------------------- summary
    def auc(r):
        return r["metrics"]["auc_roc"]

    def pg(r, key):
        return (r.get("per_grade") or {}).get("pairwise_auc", {}).get(key)

    print("\n" + "=" * 86)
    print("LABEL-BUDGET SUMMARY  (pooled AUC-ROC, plus graded contrasts)")
    print("=" * 86)
    header = (f"{'arm':14s} {'n':>3s} {'pooled AUC':>14s} "
              f"{'g2vs g0 (topical)':>18s} {'g2vs g1 (elig.)':>17s}")
    print(header)
    print("-" * 86)

    means: Dict[str, Dict[str, float]] = {}
    for arm in args.arms:
        runs = collected[arm]
        if not runs:
            print(f"{arm:14s} {'0':>3s}   (no results)")
            continue

        def agg(fn):
            vals = [v for v in (fn(r) for r in runs.values()) if v is not None]
            if not vals:
                return None, None
            return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

        a_m, a_s = agg(auc)
        t_m, t_s = agg(lambda r: pg(r, "eligible_vs_not_relevant"))
        e_m, e_s = agg(lambda r: pg(r, "eligible_vs_ineligible"))
        means[arm] = {"pooled": a_m, "topical": t_m, "eligibility": e_m}

        def cell(m, s):
            return "     n/a     " if m is None else f"{m:.4f}±{s:.4f}"

        print(f"{arm:14s} {len(runs):3d} {cell(a_m, a_s):>14s} "
              f"{cell(t_m, t_s):>18s} {cell(e_m, e_s):>17s}")
        print(f"{'':14s}     seeds={sorted(runs)}")

    if "low_ontology" in means and "low_random" in means:
        o, r = means["low_ontology"], means["low_random"]
        print("-" * 86)
        print("ontology effect (low_ontology - low_random), equal label budget:")
        for key, label in (("pooled", "pooled AUC"),
                           ("topical", "topical relevance (g2 vs g0)"),
                           ("eligibility", "ELIGIBILITY (g2 vs g1)")):
            if o.get(key) is not None and r.get(key) is not None:
                print(f"   {label:32s} {o[key] - r[key]:+.4f}")

    if "full" in means and "low_ontology" in means:
        f, o = means["full"], means["low_ontology"]
        print("\ncost of the low budget (full - low_ontology), 28,433 vs 435 judgments:")
        for key, label in (("pooled", "pooled AUC"),
                           ("topical", "topical relevance"),
                           ("eligibility", "ELIGIBILITY")):
            if f.get(key) is not None and o.get(key) is not None:
                print(f"   {label:32s} {f[key] - o[key]:+.4f}")
        if f.get("pooled") and o.get("pooled") and f["pooled"] > 0.5:
            ret = (o["pooled"] - 0.5) / (f["pooled"] - 0.5)
            print(f"\n   above-chance retention (pooled): {ret:.1%}")
            print("   NOTE: pooled retention depends on the split's grade mix "
                  "(grade-0 is capped at 60/topic in validation), so quote the "
                  "per-grade contrasts alongside it.")

    n_seeds = {len(collected[a]) for a in args.arms if collected[a]}
    if n_seeds and max(n_seeds) < 3:
        print(f"\n   WARNING: at most {max(n_seeds)} seed(s) per arm — treat "
              f"differences as directional, not significant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
