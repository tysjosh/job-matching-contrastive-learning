#!/usr/bin/env python3
"""
Statistical significance tests for E4 experiment results.

Compares each OSCAR variant against the InfoNCE baseline using:
- Paired t-test (parametric)
- Wilcoxon signed-rank test (non-parametric)
- Effect size (Cohen's d between paired seed results)

Usage:
    python scripts/statistical_significance.py
"""
import json
import glob
import numpy as np
from scipy import stats
from itertools import combinations


def collect_results(variant, dataset):
    """Collect per-seed metric values for a variant+dataset."""
    pattern = f"results/research_runs/E4-{variant}__{dataset}__*/phase1_evaluation/ordinal_evaluation_results.json"
    seed_results = {}
    for path in sorted(glob.glob(pattern)):
        seed = path.split("__")[2].split("/")[0]  # e.g. "s42"
        with open(path) as f:
            r = json.load(f).get("ordinal_v3", {})
        seed_results[seed] = {
            "AUC-ROC": r.get("binary_aucs", {}).get("good_vs_rest"),
            "Spearman": r.get("spearmans_rho", {}).get("rho"),
            "Cohen_d": r.get("separations", {}).get("good_vs_no", {}).get("cohens_d"),
            "NDCG@10": r.get("ranking", {}).get("ndcg@10"),
            "MAP_strict": r.get("ranking", {}).get("map_strict"),
            "Triplet": r.get("ordinal_triplet_accuracy"),
        }
    return seed_results


def paired_test(baseline_vals, variant_vals):
    """Run paired t-test and Wilcoxon test on matched seed pairs."""
    if len(baseline_vals) < 3 or len(variant_vals) < 3:
        return None

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(variant_vals, baseline_vals)

    # Wilcoxon signed-rank (non-parametric)
    diffs = np.array(variant_vals) - np.array(baseline_vals)
    if np.all(diffs == 0):
        w_stat, w_pval = 0, 1.0
    else:
        try:
            w_stat, w_pval = stats.wilcoxon(diffs)
        except ValueError:
            w_stat, w_pval = 0, 1.0

    # Effect size (Cohen's d for paired samples)
    diff_mean = np.mean(diffs)
    diff_std = np.std(diffs, ddof=1)
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0

    return {
        "mean_diff": float(diff_mean),
        "t_stat": float(t_stat),
        "t_pval": float(t_pval),
        "w_stat": float(w_stat),
        "w_pval": float(w_pval),
        "cohens_d_paired": float(cohens_d),
    }


def significance_symbol(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "†"
    else:
        return "n.s."


def main():
    variants = ["OSCAR-Skill", "OSCAR-Hybrid", "OSCAR-ISCO"]
    baseline = "InfoNCE"
    datasets = [
        ("cnamuangtoun", "Cnamuangtoun (v7)"),
        ("cnamuangtou_sparse_v6", "Cnamuangtou (v6)"),
        ("indian", "Indian"),
    ]
    metrics = ["AUC-ROC", "Spearman", "Cohen_d", "NDCG@10", "MAP_strict"]

    all_results = {}

    for ds, ds_label in datasets:
        print(f"\n{'='*90}")
        print(f"  {ds_label}")
        print(f"{'='*90}")

        base_data = collect_results(baseline, ds)
        seeds = sorted(base_data.keys())

        if len(seeds) < 3:
            print(f"  Skipping — only {len(seeds)} seeds available")
            continue

        for variant in variants:
            var_data = collect_results(variant, ds)
            common_seeds = sorted(set(seeds) & set(var_data.keys()))

            if len(common_seeds) < 3:
                print(f"\n  {baseline} vs {variant}: insufficient seeds ({len(common_seeds)})")
                continue

            print(f"\n  {baseline} vs {variant} ({len(common_seeds)} paired seeds)")
            print(f"  {'Metric':<14} {'Δ mean':>8} {'t-test p':>10} {'Wilcoxon p':>12} {'Cohen d':>9} {'Sig':>6}")
            print(f"  {'-'*62}")

            for metric in metrics:
                b_vals = [base_data[s][metric] for s in common_seeds if base_data[s][metric] is not None]
                v_vals = [var_data[s][metric] for s in common_seeds if var_data[s][metric] is not None]

                if len(b_vals) != len(v_vals) or len(b_vals) < 3:
                    print(f"  {metric:<14} {'—':>8}")
                    continue

                result = paired_test(b_vals, v_vals)
                if result is None:
                    print(f"  {metric:<14} {'—':>8}")
                    continue

                sig = significance_symbol(result["t_pval"])
                direction = "▲" if result["mean_diff"] > 0 else "▼"

                print(
                    f"  {metric:<14} {direction}{abs(result['mean_diff']):>7.4f} "
                    f"{result['t_pval']:>10.4f} {result['w_pval']:>12.4f} "
                    f"{result['cohens_d_paired']:>9.3f} {sig:>6}"
                )

                key = f"{ds_label}|{variant}|{metric}"
                all_results[key] = result

    # Summary: which comparisons are significant?
    print(f"\n\n{'='*90}")
    print("  SUMMARY: Significant improvements over InfoNCE (p < 0.05)")
    print(f"{'='*90}")

    sig_count = 0
    total_count = 0
    for key, result in sorted(all_results.items()):
        total_count += 1
        if result["t_pval"] < 0.05 and result["mean_diff"] > 0:
            sig_count += 1
            ds_label, variant, metric = key.split("|")
            print(f"  ✓ {ds_label}: {variant} > InfoNCE on {metric} (p={result['t_pval']:.4f}, d={result['cohens_d_paired']:.2f})")

    print(f"\n  {sig_count}/{total_count} comparisons significantly better (p < 0.05)")

    # Save full results
    output_path = "results/research_runs/statistical_significance.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
