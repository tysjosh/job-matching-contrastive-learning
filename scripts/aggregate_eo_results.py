#!/usr/bin/env python3
"""
Aggregate EO ordinal-evaluation results into mean ± std per variant.

Reads results/research_runs/EO-*__cnamuangtoun__s*/phase1_evaluation/
ordinal_evaluation_results.json and summarizes the key ordinal metrics
across seeds for each EO variant.

Run: python3 scripts/aggregate_eo_results.py
"""
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# metric label -> extractor from the ordinal_v3 result dict
METRICS = {
    "CohenD good/no":  lambda d: d["separations"]["good_vs_no"]["cohens_d"],
    "CohenD good/pot": lambda d: d["separations"]["good_vs_potential"]["cohens_d"],
    "CohenD pot/no":   lambda d: d["separations"]["potential_vs_no"]["cohens_d"],
    "TripletAcc":      lambda d: d["ordinal_triplet_accuracy"],
    "Kendall tau":     lambda d: d["kendalls_tau"]["tau"],
    "NDCG@10":         lambda d: d["ranking"]["ndcg@10"],
    "NDCG full":       lambda d: d["ranking"]["ndcg_full"],
    "MAP strict":      lambda d: d["ranking"]["map_strict"],
    "3cls AUC ovr":    lambda d: d["three_class_auc_ovr"],
}

VARIANT_NAME = {
    "E4-InfoNCE": "InfoNCE (frozen baseline)",
    "EO-A": "Ordinal-Base (frozen)",
    "EO-B": "Ordinal+OSCAR-Skill",
    "EO-C": "Ordinal-NoCurriculum",
    "EO-D": "Ordinal-FixedMargin",
    "EO-E": "Ordinal-NoGrouping",
    "UF-InfoNCE": "InfoNCE (unfrozen)",
    "UF-Ordinal": "Ordinal (unfrozen)",
    "UFG-InfoNCE": "InfoNCE (unfrozen, gentle)",
    "UFG-Ordinal": "Ordinal (unfrozen, gentle)",
}

# InfoNCE reference baseline (same v7 dataset, same metrics) always included.
BASELINE = "E4-InfoNCE"

# Column order: frozen baseline + EO variants, then unfrozen probes.
ORDER = ["E4-InfoNCE", "EO-A", "EO-B", "EO-C", "EO-D", "EO-E",
         "UF-InfoNCE", "UF-Ordinal", "UFG-InfoNCE", "UFG-Ordinal"]


def load_ordinal(path):
    with open(path) as f:
        data = json.load(f)
    # results are nested under "ordinal_v3"
    return data.get("ordinal_v3", data)


def main():
    # Include EO variants, the E4-InfoNCE frozen baseline, and the UF unfrozen probe.
    patterns = [
        str(ROOT / "results/research_runs/EO-*__cnamuangtoun__s*/"
                   "phase1_evaluation/ordinal_evaluation_results.json"),
        str(ROOT / "results/research_runs/E4-InfoNCE__cnamuangtoun__s*/"
                   "phase1_evaluation/ordinal_evaluation_results.json"),
        str(ROOT / "results/research_runs/UF-*__cnamuangtoun__s*/"
                   "phase1_evaluation/ordinal_evaluation_results.json"),
        str(ROOT / "results/research_runs/UFG-*__cnamuangtoun__s*/"
                   "phase1_evaluation/ordinal_evaluation_results.json"),
    ]
    files = sorted(f for p in patterns for f in glob.glob(p))

    # variant -> metric -> list of values
    agg = defaultdict(lambda: defaultdict(list))
    seeds_seen = defaultdict(list)

    for fp in files:
        m = re.search(r"(EO-[A-E]|E4-InfoNCE|UFG-Ordinal|UFG-InfoNCE|UF-Ordinal|UF-InfoNCE)__cnamuangtoun__s(\d+)", fp)
        if not m:
            continue
        variant, seed = m.group(1), m.group(2)
        try:
            d = load_ordinal(fp)
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        seeds_seen[variant].append(seed)
        for label, fn in METRICS.items():
            try:
                agg[variant][label].append(float(fn(d)))
            except Exception:
                pass

    if not agg:
        print("No EO results found yet.")
        return

    # Order columns: baseline first, then EO variants; only those present.
    variants = [v for v in ORDER if v in agg] + \
               [v for v in sorted(agg) if v not in ORDER]

    print("EO Ordinal Ablation — v7 (cnamuangtoun), mean ± std across seeds")
    print("(Δ = variant − InfoNCE baseline)\n")
    for v in variants:
        tag = " [BASELINE]" if v == BASELINE else ""
        print(f"{v}  ({VARIANT_NAME.get(v, v)}){tag}  "
              f"[n={len(seeds_seen[v])} seeds: {','.join(sorted(seeds_seen[v]))}]")

    have_baseline = BASELINE in agg

    def cell(mean, sd):
        return f"{mean:.3f} ± {sd:.3f}"

    col_w = 22
    header = "Metric".ljust(16) + "".join(v.ljust(col_w) for v in variants)
    print("\n" + header)
    print("-" * len(header))
    for label in METRICS:
        row = label.ljust(16)
        base_mean = None
        if have_baseline and agg[BASELINE][label]:
            base_mean = statistics.mean(agg[BASELINE][label])
        for v in variants:
            vals = agg[v][label]
            if not vals:
                row += "-".ljust(col_w)
                continue
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            if v == BASELINE or base_mean is None:
                row += cell(mean, sd).ljust(col_w)
            else:
                row += f"{cell(mean, sd)} ({mean - base_mean:+.3f})".ljust(col_w)
        print(row)

    # Also dump machine-readable summary
    out = {v: {label: {"mean": statistics.mean(vals),
                       "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                       "n": len(vals), "values": vals}
               for label, vals in agg[v].items()}
           for v in variants}
    out_path = ROOT / "results/eo_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMachine-readable summary written to {out_path}")


if __name__ == "__main__":
    main()
