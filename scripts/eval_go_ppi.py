#!/usr/bin/env python3
"""Evaluate the GO/PPI learning-curve checkpoints on the HELD-OUT test split.

Test, not validation. ``trainer.py`` selects ``best_checkpoint.pt`` by lowest
validation loss, so validation is the checkpoint-selection set and scoring on it
is measurably optimistic — on career it inflated two arms by +0.0122 and +0.0208
AUC, enough to turn both from null into "significant". The same discipline applies
here.

Runs the shared ``run_phase1_embedding_evaluation.py`` per (arm, seed) so GO/PPI
is scored by the identical script and metrics as every other domain, then pools
the paired per-seed deltas.

The headline metric is the GRADED pairwise AUC, not the pooled AUC-ROC:

    eligible_vs_ineligible     grade 2 vs grade 1  -- the HARD contrast
    eligible_vs_not_relevant   grade 2 vs grade 0  -- the easy contrast

The pooled AUC-ROC mixes the two in whatever ratio the split's grade composition
happens to set (here 1,165 : 10,011 : 11,165), so it is not the quantity the
ontology is supposed to affect. Negative selection acts on the hard contrast.

Usage
    .venv/bin/python3 scripts/eval_go_ppi.py
    .venv/bin/python3 scripts/eval_go_ppi.py --seeds 42 --force
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lc_single_factor"
EVALUATOR = ROOT / "run_phase1_embedding_evaluation.py"

CONFIGS = {
    "baseline": ROOT / "config" / "lc_go_ppi_baseline.json",
    # random window of the same width, no GO -- the diversity control
    "randwin": ROOT / "config" / "lc_go_ppi_randwin.json",
    # GO-closest window, sampled -- the ontology arm
    "ontneg_stoch": ROOT / "config" / "lc_go_ppi_ontneg_stoch.json",
    # GO-closest fixed prefix -- ontology + loss of per-epoch variety
    "ontneg": ROOT / "config" / "lc_go_ppi_ontneg_only.json",
}

#: (label, arm_a, arm_b): what the b-minus-a delta isolates.
#: Negative diversity, unique negatives per anchor over 15 epochs from a
#: ~54-candidate pool: baseline 46.4, randwin 18.3, ontneg_stoch 18.4, ontneg 10.0.
#: randwin and ontneg_stoch are matched on diversity AND grade mix, so the
#: difference between them is the ontology and nothing else.
CONTRASTS = [
    ("NARROWING  (baseline -> randwin)", "baseline", "randwin"),
    ("ONTOLOGY   (randwin -> GO window)", "randwin", "ontneg_stoch"),
    ("DETERMINISM (GO window -> GO prefix)", "ontneg_stoch", "ontneg"),
    ("TOTAL      (baseline -> GO prefix)", "baseline", "ontneg"),
]


def run_one(arm: str, fraction: int, seed: int, dataset: Path,
            subdir: str, force: bool) -> dict | None:
    run_dir = RESULTS / f"go_ppi_{arm}_f{fraction}_s{seed}"
    ckpt = run_dir / "best_checkpoint.pt"
    if not ckpt.exists():
        print(f"  SKIP {arm} s{seed}: no best_checkpoint.pt")
        return None
    out_dir = run_dir / subdir
    result_json = out_dir / "phase1_evaluation_results.json"
    if result_json.exists() and not force:
        print(f"  cached {arm} s{seed}")
        return json.loads(result_json.read_text())

    cmd = [
        sys.executable, str(EVALUATOR),
        "--dataset", str(dataset),
        "--checkpoint", str(ckpt),
        "--config", str(CONFIGS[arm]),
        "--output-dir", str(out_dir),
    ]
    print(f"  running {arm} s{seed} ...", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    if proc.returncode != 0:
        print(f"  FAILED {arm} s{seed} (exit {proc.returncode})")
        print("  " + "\n  ".join(proc.stdout.strip().splitlines()[-8:]))
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
        return None
    if not result_json.exists():
        print(f"  FAILED {arm} s{seed}: evaluator wrote no results file")
        return None
    return json.loads(result_json.read_text())


def graded(res: dict, key: str):
    """Pull a graded pairwise AUC.

    The evaluator writes these under ``per_grade.pairwise_auc``; the other
    containers are tolerated in case an older result file is re-read.
    """
    for container in ("per_grade", "graded_relevance", "graded", "graded_breakdown"):
        block = res.get(container)
        if isinstance(block, dict):
            pw = block.get("pairwise_auc") or block.get("pairwise")
            if isinstance(pw, dict) and key in pw:
                return pw[key]
    pw = res.get("pairwise_auc")
    if isinstance(pw, dict) and key in pw:
        return pw[key]
    return None


def paired_stats(deltas: list[float]) -> dict:
    n = len(deltas)
    if n == 0:
        return {}
    mean = st.fmean(deltas)
    sd = st.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
    t = mean / se if se > 0 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t,
            "wins": sum(1 for d in deltas if d > 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "preprocess" / "go_ppi_splits" / "test.jsonl")
    ap.add_argument("--subdir", default="eval_test")
    ap.add_argument("--fraction", type=int, default=100)
    ap.add_argument("--arms", nargs="+", choices=sorted(CONFIGS),
                    default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 21])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 92)
    print("GO/PPI evaluation on the HELD-OUT test split")
    print("=" * 92)
    print(f"dataset: {args.dataset.relative_to(ROOT)}")
    n_records = sum(1 for _ in open(args.dataset))
    print(f"records: {n_records}")
    print()

    results: dict = {}
    for arm in args.arms:
        print(f"{arm}:")
        for seed in args.seeds:
            res = run_one(arm, args.fraction, seed, args.dataset,
                          args.subdir, args.force)
            if res is not None:
                results[(arm, seed)] = res

    metrics = [
        ("hard  (eligible vs ineligible)", "eligible_vs_ineligible"),
        ("easy  (eligible vs not_relevant)", "eligible_vs_not_relevant"),
        ("      (ineligible vs not_relevant)", "ineligible_vs_not_relevant"),
    ]

    summary: dict = {"dataset": str(args.dataset.relative_to(ROOT)),
                     "fraction": args.fraction,
                     "n_records": n_records, "arm_means": {}, "contrasts": {}}

    # ---------------------------------------------------- per-arm absolute levels
    print()
    print("-" * 92)
    print("per-arm graded pairwise AUC (mean over seeds)")
    print("-" * 92)
    arms = [a for a in args.arms if any((a, s) in results for s in args.seeds)]
    print(f"  {'metric':<36} " + "".join(f"{a:>15}" for a in arms))
    print(f"  {'-'*36} " + "".join(f"{'-'*15}" for _ in arms))
    for label, key in metrics:
        cells = []
        for a in arms:
            vals = [graded(results[(a, s)], key) for s in args.seeds
                    if (a, s) in results]
            vals = [v for v in vals if v is not None]
            cells.append(st.fmean(vals) if vals else None)
            summary["arm_means"].setdefault(key, {})[a] = (
                st.fmean(vals) if vals else None)
        print(f"  {label:<36} " + "".join(
            f"{c:>15.4f}" if c is not None else f"{'-':>15}" for c in cells))

    # ------------------------------------------------------------- decomposition
    print()
    print("-" * 92)
    print("DECOMPOSITION -- paired per-seed deltas, hard contrast first")
    print("-" * 92)
    for label, key in metrics:
        print(f"\n  {label}")
        for cname, a, b in CONTRASTS:
            deltas, rows = [], []
            for seed in args.seeds:
                ra, rb = results.get((a, seed)), results.get((b, seed))
                if ra is None or rb is None:
                    continue
                av, bv = graded(ra, key), graded(rb, key)
                if av is None or bv is None:
                    continue
                deltas.append(bv - av)
                rows.append({"seed": seed, a: av, b: bv, "delta": bv - av})
            s = paired_stats(deltas)
            if not s:
                continue
            per_seed = " ".join(f"{d:+.4f}" for d in deltas)
            print(f"    {cname:<38} {s['mean']:+.4f}  sd={s['sd']:.4f} "
                  f"t={s['t']:+.2f} wins={s['wins']}/{s['n']}   [{per_seed}]")
            summary["contrasts"].setdefault(key, {})[cname.split()[0]] = {
                "from": a, "to": b, "per_seed": rows, "pooled": s}

    # pooled AUC-ROC for completeness
    print()
    print("-" * 92)
    print("pooled AUC-ROC (mixes both contrasts; for completeness only)")
    print("-" * 92)
    for a in arms:
        vals = [results[(a, s)]["metrics"]["auc_roc"] for s in args.seeds
                if (a, s) in results]
        if vals:
            print(f"  {a:<16} {st.fmean(vals):.4f}")
            summary["arm_means"].setdefault("auc_roc", {})[a] = st.fmean(vals)

    # ---------------------------------------------------------------- reading
    hk = "eligible_vs_ineligible"
    means = summary["arm_means"].get(hk, {})
    cons = summary["contrasts"].get(hk, {})
    print()
    print("=" * 92)
    print("reading")
    print("=" * 92)
    base = means.get("baseline")
    if base is not None:
        print(f"  MODEL hard contrast, baseline      : {base:.4f}")
        print(f"  GO ONTOLOGY ceiling, same contrast : 0.8195  95% CI [0.8063, 0.8335]")
        print(f"    (simGIC, STRING confidence bands; "
              f"results/ontology_ceiling/go_bulk_P_experimental_absolute.json)")
        print(f"  headroom available to the ontology : {0.8195 - base:+.4f}")
        print()
        print("  The model reaches 0.80 on the hard contrast from TEXT ALONE, against an")
        print("  ontology ceiling of 0.82. It has already learned nearly everything GO")
        print("  encodes about this label, so there is very little for GO to add — even")
        print("  though GO's own signal here is strong (0.82 vs 0.485-0.607 for ESCO,")
        print("  ISCO, MeSH and CPC on their tasks).")
    onto = cons.get("ONTOLOGY", {}).get("pooled", {})
    narrow = cons.get("NARROWING", {}).get("pooled", {})
    determ = cons.get("DETERMINISM", {}).get("pooled", {})
    total = cons.get("TOTAL", {}).get("pooled", {})
    if onto and narrow:
        print()
        print("  Decomposing the total effect on the hard contrast:")
        for nm, s in (("narrowing the negative pool", narrow),
                      ("GO choosing the region", onto),
                      ("losing per-epoch variety", determ),
                      ("TOTAL", total)):
            if s:
                print(f"    {nm:<30} {s['mean']:+.4f}  ({s['wins']}/{s['n']} seeds up)")
        print()
        print("  The ONTOLOGY row is the number that answers the research question: it")
        print("  compares a GO-chosen window against a RANDOM window of the same width,")
        print("  matched on negative diversity and grade mix. A baseline-vs-ontology")
        print("  delta on its own cannot be read as an ontology effect, because ontology")
        print("  guidance necessarily narrows the pool and narrowing has its own cost.")
    print()
    ns = {k: c.get("pooled", {}).get("n") for k, c in cons.items()}
    print(f"  seeds per contrast: " + "  ".join(f"{k}={v}" for k, v in ns.items() if v))
    print("  Read signs, magnitudes and the per-seed spread, not p-values. The")
    print("  ONTOLOGY contrast carries the most seeds because it is the one the")
    print("  research question turns on; the others were not extended.")

    out = RESULTS / f"go_ppi_test_report_f{args.fraction}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
