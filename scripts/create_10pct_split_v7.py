#!/usr/bin/env python3
"""Create a 10% nested stratified split from the v7 25% split."""
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
SOURCE = Path("preprocess/learning_curve_v7/frac_25/train.jsonl")
OUTPUT_DIR = Path("preprocess/learning_curve_v7/frac_10")

# Load 25% split
with open(SOURCE) as f:
    samples = [json.loads(line) for line in f]
print(f"25% split: {len(samples)} samples")

# Stratified subsample to 10% (640 / 6400)
# 25% has 1600, we need 640 = 40% of the 25% split
rng = random.Random(SEED)
by_label = defaultdict(list)
for s in samples:
    label = s.get("metadata", {}).get("original_label", "unknown")
    by_label[label].append(s)

target = 640
fraction = target / len(samples)  # 0.4

subsampled = []
for label, items in sorted(by_label.items()):
    k = max(1, int(len(items) * fraction))
    subsampled.extend(rng.sample(items, k))
    print(f"  {label}: {len(items)} -> {k}")

rng.shuffle(subsampled)
print(f"10% split: {len(subsampled)} samples")

# Write
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / "train.jsonl", "w") as f:
    for s in subsampled:
        f.write(json.dumps(s) + "\n")

# Symlink val and test from the main v7 splits
import os
for name in ["validation.jsonl", "test.jsonl"]:
    src = Path("preprocess/data_splits_v7") / name
    dst = OUTPUT_DIR / name
    if dst.exists():
        dst.unlink()
    # Use relative path for symlink
    os.symlink(os.path.relpath(src, OUTPUT_DIR), dst)
    print(f"Linked {dst} -> {src}")

print("Done!")
