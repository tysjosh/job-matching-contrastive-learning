#!/usr/bin/env python3
"""
Aggregate ORCA (ER-*) Phase-1 embedding-evaluation results into mean ± std per
variant across seeds, and report the delta vs the InfoNCE and OSCAR baselines.

Reads results/research_runs/{ER-*, E4-InfoNCE, E4-OSCAR-Skill}__<dataset>__s*/
phase1_evaluation/phase1_evaluation_results.json (the artifact produced by
run_phase1_embedding_evaluation.py) and summarizes the ranking/classification
metrics so the ORCA family is compared apples-to-apples with the baselines.

Usage:
    python scripts/aggregate_orca_results.py [--dataset cnamuangtoun]
"""
import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Metric label -> accessor into the phase1_evaluation_results.json structure.
METRICS = {
    "accuracy": lambda d: d["metrics"]["accuracy"],
    "f1": lambda d: d["metrics"]["f1_score"],
    "auc_roc": lambda d: d["metrics"]["auc_roc"],
    "separation": lambda d: d["similarity_stats"]["separation"],
}

VARIANT_NAME = {
    "E4-InfoNCE": "InfoNCE (baseline: all negatives trusted)",
    "E4-OSCAR-Skill": "OSCAR-Skill (baseline: fixed ontology, no reliability)",
    "ER-DEN": "ORCA-Denominator (MVP)",
    "ER-EXT": "ORCA-ExternalWeight",
    "ER-NOONT": "ORCA-NoOntology (MLP input features removed)",
    "ER-NOALIGN": "ORCA-NoAlign",
    "ER-FULL": "ORCA-Full",
    "ER-NOONTW": "ORCA-NoOntologyWeakTarget (lambda_ont=0; features kept)",
    "ER-NOONTALL": "ORCA-NoOntologyAtAll (lambda_ont=0 AND features removed)",
}

# Column order: baselines first, then the ORCA family.
ORDER = ["E4-InfoNCE", "E4-OSCAR-Skill",
         "ER-DEN", "ER-EXT", "ER-NOONT", "ER-NOALIGN", "ER-FULL",
         "ER-NOONTW", "ER-NOONTALL"]

# Baselines the ORCA delta is reported against.
BASELINES = ["E4-OSCAR-Skill", "E4-InfoNCE"]

_VARIANT_RE = "|".join(re.escape(v) for v in ORDER)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate ORCA (ER-*) results.")
    ap.add_argument("--dataset", default="cnamuangtoun",
                    help="Dataset label used in the run-id (default: cnamuangtoun).")
    args = ap.parse_args()
    ds = args.dataset

    pattern = str(
        ROOT / f"results/research_runs/*__{ds}__s*/"
        "phase1_evaluation/phase1_evaluation_results.json"
    )
    files = sorted(glob.glob(pattern))

    # variant -> metric -> [values]; variant -> [seeds]
    agg = defaultdict(lambda: defaultdict(list))
    seeds_seen = defaultdict(list)

    run_re = re.compile(rf"({_VARIANT_RE})__{re.escape(ds)}__s(\d+)")
    for fp in files:
        m = run_re.search(fp)
        if not m:
            continue
        variant, seed = m.group(1), m.group(2)
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception as exc:
            print(f"skip {fp}: {exc}")
            continue
        seeds_seen[variant].append(seed)
        for label, fn in METRICS.items():
            try:
                val = fn(d)
                if val is not None:
                    agg[variant][label].append(float(val))
            except Exception:
                pass

    if not agg:
        print(f"No results found for dataset '{ds}'. Expected files matching:\n  {pattern}")
        print("Run the ER-* grid first (scripts/generate_orca_experiments.py "
              "--execute-list run_orca.sh, then bash run_orca.sh).")
        return

    variants = [v for v in ORDER if v in agg] + \
               [v for v in sorted(agg) if v not in ORDER]

    def mean_std(vals):
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    # Precompute baseline means for deltas.
    baseline_means = {}
    for b in BASELINES:
        baseline_means[b] = {
            label: mean_std(agg.get(b, {}).get(label, []))[0]
            for label in METRICS
        }

    print(f"ORCA vs baselines — {ds}, mean ± std across seeds")
    print("Δ columns: gain over each baseline (variant − baseline)\n")

    metric_labels = list(METRICS.keys())
    header = f"{'variant':<12} " + "  ".join(f"{m:>16}" for m in metric_labels)
    print(header)
    print("-" * len(header))
    for v in variants:
        cells = []
        for label in metric_labels:
            mean, sd = mean_std(agg[v].get(label, []))
            cells.append("n/a".rjust(16) if mean is None else f"{mean:.3f}±{sd:.3f}".rjust(16))
        n = len(seeds_seen[v])
        print(f"{v:<12} " + "  ".join(cells) + f"   [n={n}]")

    # Delta tables vs each baseline.
    for b in BASELINES:
        if b not in agg:
            continue
        print(f"\nΔ vs {b} ({VARIANT_NAME.get(b, b)}):")
        for v in variants:
            if v == b or v in BASELINES:
                continue
            parts = []
            for label in metric_labels:
                mean, _ = mean_std(agg[v].get(label, []))
                base = baseline_means[b].get(label)
                if mean is None or base is None:
                    parts.append(f"{label}=n/a")
                else:
                    parts.append(f"{label}={mean - base:+.3f}")
            print(f"  {v:<12} " + "  ".join(parts))

    print(f"\nVariants: " + ", ".join(f"{v}={VARIANT_NAME.get(v, v)}" for v in variants))


if __name__ == "__main__":
    main()
