#!/usr/bin/env python3
"""Split data into train and validation sets."""

import json
import random
from pathlib import Path

# Paths
input_path = Path(__file__).parent.parent / "preprocess" / "train_final_cleaned_with_uri.jsonl"
train_path = Path(__file__).parent.parent / "preprocess" / "train_split.jsonl"
val_path = Path(__file__).parent.parent / "preprocess" / "val_split.jsonl"

# Load data
with open(input_path) as f:
    data = [json.loads(line) for line in f]

print(f"Total records: {len(data)}")

# Shuffle with seed for reproducibility
random.seed(42)
random.shuffle(data)

# Split 80/20
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
print(f"Val: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")

# Check label distribution
train_pos = sum(1 for d in train_data if d.get('label') == 1)
val_pos = sum(1 for d in val_data if d.get('label') == 1)
print(f"\nTrain positives: {train_pos} ({train_pos/len(train_data)*100:.1f}%)")
print(f"Val positives: {val_pos} ({val_pos/len(val_data)*100:.1f}%)")

# Save
with open(train_path, 'w') as f:
    for d in train_data:
        f.write(json.dumps(d) + '\n')

with open(val_path, 'w') as f:
    for d in val_data:
        f.write(json.dumps(d) + '\n')

print(f"\nSaved:")
print(f"  - {train_path}")
print(f"  - {val_path}")
