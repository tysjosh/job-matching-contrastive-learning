#!/usr/bin/env python3
"""Export complete dataset as CSVs for external model evaluation."""

import json
import pandas as pd
from itertools import product
from collections import defaultdict
from pathlib import Path
import hashlib

SPLITS = [
    "preprocess/data_splits_v6/train.jsonl",
    "preprocess/data_splits_v6/validation.jsonl",
    "preprocess/data_splits_v6/test.jsonl",
]
OUTPUT_DIR = Path("exports")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load all data
samples = []
for split_path in SPLITS:
    with open(split_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            m = d.get("metadata", {})
            label = m.get("original_label", "unknown")
            if label not in ("good_fit", "potential_fit", "no_fit"):
                continue
            samples.append({
                "pair_id": d.get("job_applicant_id", f"idx_{i}"),
                "label": label,
                "job": d["job"],
                "resume": d["resume"],
            })

print(f"Loaded {len(samples)} samples from {len(SPLITS)} splits")

# === Export 1: test_labels.csv ===
rows = []
for s in samples:
    rows.append({
        "pair_id": s["pair_id"],
        "job_title": s["job"]["title"],
        "resume_role": s["resume"]["role"],
        "label": s["label"],
    })
df_labels = pd.DataFrame(rows)
df_labels.to_csv(OUTPUT_DIR / "test_labels.csv", index=False)
print(f"\nExported {len(df_labels)} pairs to {OUTPUT_DIR}/test_labels.csv")
print(f"  Distribution: {df_labels['label'].value_counts().to_dict()}")

# === Export 2: test_tuples.csv ===
def job_key(job):
    desc = job.get("description", "")
    if not isinstance(desc, str):
        desc = json.dumps(desc)[:200]
    else:
        desc = desc[:200]
    text = job["title"] + "|" + desc
    return hashlib.md5(text.encode()).hexdigest()[:12]

job_groups = defaultdict(lambda: {"good_fit": [], "potential_fit": [], "no_fit": []})
for s in samples:
    jk = job_key(s["job"])
    job_groups[jk][s["label"]].append(s)

tuples = []
jobs_with_all = 0
for jk, tiers in job_groups.items():
    if not tiers["good_fit"] or not tiers["potential_fit"] or not tiers["no_fit"]:
        continue
    jobs_with_all += 1
    for gf, pf, nf in product(tiers["good_fit"], tiers["potential_fit"], tiers["no_fit"]):
        tuples.append({
            "job_title": gf["job"]["title"],
            "good_fit_pair_id": gf["pair_id"],
            "good_fit_resume_role": gf["resume"]["role"],
            "potential_fit_pair_id": pf["pair_id"],
            "potential_fit_resume_role": pf["resume"]["role"],
            "no_fit_pair_id": nf["pair_id"],
            "no_fit_resume_role": nf["resume"]["role"],
        })

df_tuples = pd.DataFrame(tuples)
df_tuples.to_csv(OUTPUT_DIR / "test_tuples.csv", index=False)
print(f"\nExported {len(df_tuples)} tuples to {OUTPUT_DIR}/test_tuples.csv")
print(f"  Jobs with all 3 tiers: {jobs_with_all}")
print(f"  Jobs missing a tier: {len(job_groups) - jobs_with_all}")

# === Export 3: full JSONL with all splits combined ===
with open(OUTPUT_DIR / "complete_dataset.jsonl", "w") as out:
    for split_path in SPLITS:
        with open(split_path) as f:
            for line in f:
                out.write(line)
print(f"\nCombined all splits to {OUTPUT_DIR}/complete_dataset.jsonl")
