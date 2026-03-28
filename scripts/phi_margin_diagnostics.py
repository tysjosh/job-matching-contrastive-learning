#!/usr/bin/env python3
"""Diagnostics requested by reviewer: φ distribution and margin analysis.

Computes:
1. φ_p (potential_fit) and φ_n (no_fit) distribution stats
2. Resulting margins m₁ = α·(1 − φ_p) distribution
3. Comparison with winning fixed margin (m₁ = m₂ = 0.3)
4. On-the-fly φ(resume_p, job_α) vs precomputed φ(resume_p, job_p) comparison
"""
import json
import numpy as np
from collections import defaultdict

# Load v6 training data
data = []
for split in ["train", "validation", "test"]:
    with open(f"preprocess/data_splits_v6/{split}.jsonl") as f:
        for line in f:
            data.append(json.loads(line))

print(f"Total samples: {len(data)}")

# Group φ by label
phi_by_label = defaultdict(list)
for d in data:
    label = d.get("metadata", {}).get("original_label", "unknown")
    phi = d.get("metadata", {}).get("phi")
    if phi is not None:
        phi_by_label[label].append(phi)

# ── Diagnostic 1: φ distributions by label ──
print("\n" + "=" * 60)
print("DIAGNOSTIC 1: φ distributions by label")
print("=" * 60)

for label in ["good_fit", "potential_fit", "no_fit"]:
    vals = np.array(phi_by_label[label])
    print(f"\n  {label} (n={len(vals)}):")
    print(f"    mean: {vals.mean():.3f}, std: {vals.std():.3f}")
    print(f"    min:  {vals.min():.3f}, max: {vals.max():.3f}")
    print(f"    25th: {np.percentile(vals, 25):.3f}, median: {np.median(vals):.3f}, 75th: {np.percentile(vals, 75):.3f}")

phi_p = np.array(phi_by_label["potential_fit"])
phi_n = np.array(phi_by_label["no_fit"])
phi_g = np.array(phi_by_label["good_fit"])

# ── Diagnostic 2: Resulting margins ──
print("\n" + "=" * 60)
print("DIAGNOSTIC 2: Margin distributions (m₁ = α·(1 − φ_p))")
print("=" * 60)

alpha = 0.5  # from config
margins_p = alpha * (1 - phi_p)
margins_n = alpha * (1 - phi_n)
margins_g = alpha * (1 - phi_g)

print(f"\n  α = {alpha}")
print(f"\n  potential_fit margins (what L₂ actually uses):")
print(f"    mean: {margins_p.mean():.3f}, std: {margins_p.std():.3f}")
print(f"    min:  {margins_p.min():.3f}, max: {margins_p.max():.3f}")
print(f"    25th: {np.percentile(margins_p, 25):.3f}, median: {np.median(margins_p):.3f}, 75th: {np.percentile(margins_p, 75):.3f}")

print(f"\n  no_fit margins (if m₂ were also φ-guided):")
print(f"    mean: {margins_n.mean():.3f}, std: {margins_n.std():.3f}")
print(f"    min:  {margins_n.min():.3f}, max: {margins_n.max():.3f}")

print(f"\n  good_fit margins (reference):")
print(f"    mean: {margins_g.mean():.3f}, std: {margins_g.std():.3f}")
print(f"    min:  {margins_g.min():.3f}, max: {margins_g.max():.3f}")

# ── Diagnostic 3: Comparison with fixed margin ──
print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Fixed margin comparison")
print("=" * 60)

fixed_m1 = 0.3  # ordinal_m2 used as fixed m₁
fixed_m2 = 0.3  # ordinal_m2

print(f"\n  Winning fixed margins: m₁ = {fixed_m1}, m₂ = {fixed_m2}")
print(f"  φ-derived margin mean: {margins_p.mean():.3f}")
print(f"  Difference (φ mean - fixed): {margins_p.mean() - fixed_m1:+.3f}")
print(f"  φ margins > fixed m₁: {(margins_p > fixed_m1).sum()} / {len(margins_p)} ({100*(margins_p > fixed_m1).mean():.1f}%)")
print(f"  φ margins < fixed m₁: {(margins_p < fixed_m1).sum()} / {len(margins_p)} ({100*(margins_p < fixed_m1).mean():.1f}%)")
print(f"  φ margins = 0 (φ=1.0): {(margins_p < 0.001).sum()} ({100*(margins_p < 0.001).mean():.1f}%)")
print(f"  φ margins = α={alpha} (φ=0.0): {(margins_p > alpha - 0.001).sum()} ({100*(margins_p > alpha - 0.001).mean():.1f}%)")

# ── Diagnostic 4: Margin variance analysis ──
print("\n" + "=" * 60)
print("DIAGNOSTIC 4: Is φ distribution collapsed?")
print("=" * 60)

cv_phi_p = phi_p.std() / phi_p.mean() if phi_p.mean() > 0 else 0
cv_margins = margins_p.std() / margins_p.mean() if margins_p.mean() > 0 else 0

print(f"\n  φ_p coefficient of variation: {cv_phi_p:.3f}")
print(f"  Margin coefficient of variation: {cv_margins:.3f}")
print(f"  φ_p IQR: {np.percentile(phi_p, 75) - np.percentile(phi_p, 25):.3f}")
print(f"  Margin IQR: {np.percentile(margins_p, 75) - np.percentile(margins_p, 25):.3f}")

# Bucket analysis
print(f"\n  φ_p distribution buckets:")
for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
    count = ((phi_p >= lo) & (phi_p < hi)).sum()
    pct = 100 * count / len(phi_p)
    print(f"    [{lo:.1f}, {hi:.1f}): {count:4d} ({pct:5.1f}%)")
count_1 = (phi_p >= 1.0).sum()
print(f"    [1.0, 1.0]: {count_1:4d} ({100*count_1/len(phi_p):5.1f}%)")
