#!/usr/bin/env python3
"""Compare v5 (old) vs v6 (new, with fixes) training data quality."""
import json
import statistics
from collections import Counter, defaultdict
import numpy as np

def load_splits(base_dir):
    data = []
    for split in ["train", "validation", "test"]:
        with open(f"{base_dir}/{split}.jsonl") as f:
            for line in f:
                data.append(json.loads(line))
    return data

v5 = load_splits("preprocess/data_splits_v5")
v6 = load_splits("preprocess/data_splits_v6")
print(f"v5: {len(v5)} samples, v6: {len(v6)} samples")

# Extract skill URIs from training format (resume.skill_uris, job.skill_uris)
def get_stats(data):
    r_uris = [len(d.get("resume", {}).get("skill_uris", [])) for d in data]
    j_uris = [len(d.get("job", {}).get("skill_uris", [])) for d in data]
    phi = [d.get("metadata", {}).get("phi") for d in data]
    phi_valid = [p for p in phi if p is not None]
    ont = [d.get("metadata", {}).get("ontology_similarity") for d in data]
    ont_valid = [o for o in ont if o is not None]
    return r_uris, j_uris, phi_valid, ont_valid

v5_r, v5_j, v5_phi, v5_ont = get_stats(v5)
v6_r, v6_j, v6_phi, v6_ont = get_stats(v6)

print(f"\n{'Metric':40s} {'v5 (old)':>10s} {'v6 (new)':>10s} {'Diff':>10s}")
print("-" * 72)

def row(label, old, new):
    print(f"{label:40s} {old:>10.2f} {new:>10.2f} {new-old:>+10.2f}")

def row_int(label, old, new):
    print(f"{label:40s} {old:>10d} {new:>10d} {new-old:>+10d}")

row("Resume skill URIs (mean)", statistics.mean(v5_r), statistics.mean(v6_r))
row_int("Resume zero-URI samples", sum(1 for u in v5_r if u==0), sum(1 for u in v6_r if u==0))
row("Job skill URIs (mean)", statistics.mean(v5_j), statistics.mean(v6_j))
row_int("Job zero-URI samples", sum(1 for u in v5_j if u==0), sum(1 for u in v6_j if u==0))

# Combined
v5_both_zero = sum(1 for r, j in zip(v5_r, v5_j) if r == 0 and j == 0)
v6_both_zero = sum(1 for r, j in zip(v6_r, v6_j) if r == 0 and j == 0)
row_int("Both sides zero-URI", v5_both_zero, v6_both_zero)

# Phi
print()
row("phi mean", statistics.mean(v5_phi), statistics.mean(v6_phi))
row("phi median", statistics.median(v5_phi), statistics.median(v6_phi))

# Ontology similarity
if v5_ont and v6_ont:
    print()
    row("ontology_similarity mean", statistics.mean(v5_ont), statistics.mean(v6_ont))
    row_int("ontology_similarity valid count", len(v5_ont), len(v6_ont))

# Phi by label
print("\n\n=== PHI BY LABEL ===")
for version, data, label_str in [(v5, v5, "v5"), (v6, v6, "v6")]:
    phi_by_label = defaultdict(list)
    for d in data:
        lbl = d.get("metadata", {}).get("original_label", "unknown")
        p = d.get("metadata", {}).get("phi")
        if p is not None:
            phi_by_label[lbl].append(p)
    
    print(f"\n{label_str}:")
    for lbl in ["good_fit", "potential_fit", "no_fit"]:
        vals = phi_by_label.get(lbl, [])
        if vals:
            arr = np.array(vals)
            print(f"  {lbl:15s}: mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, n={len(vals)}")

# Cohen's d comparison
print("\n=== COHEN'S D (phi discrimination) ===")
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0

for version, data, label_str in [(v5, v5, "v5"), (v6, v6, "v6")]:
    phi_by_label = defaultdict(list)
    for d in data:
        lbl = d.get("metadata", {}).get("original_label", "unknown")
        p = d.get("metadata", {}).get("phi")
        if p is not None:
            phi_by_label[lbl].append(p)
    
    gf = np.array(phi_by_label.get("good_fit", []))
    pf = np.array(phi_by_label.get("potential_fit", []))
    nf = np.array(phi_by_label.get("no_fit", []))
    if len(gf) > 0 and len(pf) > 0 and len(nf) > 0:
        print(f"  {label_str}: d(gf vs pf)={cohens_d(gf,pf):.3f}, d(pf vs nf)={cohens_d(pf,nf):.3f}, d(gf vs nf)={cohens_d(gf,nf):.3f}")
