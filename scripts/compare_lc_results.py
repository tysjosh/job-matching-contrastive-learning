#!/usr/bin/env python3
"""Compare learning curve results across all fractions and methods."""
import json, os

configs = []
for frac in [10, 25, 50, 75, 100]:
    for method in ["vanilla", "ontology", "isco", "isco_only"]:
        if frac == 100:
            path = f"results/results_v7_{method}/test_evaluation/ordinal_evaluation_results.json"
        else:
            path = f"results/results_v7_lc_{frac}pct_{method}/test_evaluation/ordinal_evaluation_results.json"
        configs.append((frac, method, path))

print(f"{'Frac':<6} {'Method':<12} {'AUC':>7} {'Spearman':>9} {'MAP_strict':>10} {'NDCG':>7} {'NDCG@10':>8}")
print("-" * 65)
for frac, method, path in configs:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        r = json.load(f)["ordinal_v3"]
    auc = r.get("binary_aucs", {}).get("good_vs_rest", 0)
    spearman = r.get("spearmans_rho", {}).get("rho", 0)
    rnk = r.get("ranking", {})
    map_s = rnk.get("map_strict", 0)
    ndcg = rnk.get("ndcg_full", 0)
    ndcg10 = rnk.get("ndcg@10", 0)
    print(f"{frac}%    {method:<12} {auc:>7.3f} {spearman:>9.3f} {map_s:>10.3f} {ndcg:>7.3f} {ndcg10:>8.3f}")
