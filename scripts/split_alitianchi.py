#!/usr/bin/env python3
"""
Create stratified train/val/test splits for AliTianChi data.
Adds quality_tier based on skill URI availability.

Usage:
    python3 scripts/split_alitianchi.py \
        --input dataset/alitianchi_train_scored_phi.jsonl \
        --output-dir preprocess/alitianchi_splits
"""
import argparse, json, random, logging
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def assign_quality_tier(rec):
    r_uris = rec["resume"].get("skill_uris", [])
    j_uris = rec["job"].get("skill_uris", [])
    if r_uris and j_uris:
        return "A"
    elif r_uris or j_uris:
        return "B"
    return "C"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="preprocess/alitianchi_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        samples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(samples)} samples")

    # Add quality tier + ensure metadata
    for s in samples:
        s.setdefault("metadata", {})
        s["metadata"]["quality_tier"] = assign_quality_tier(s)
        if "job_applicant_id" not in s["metadata"]:
            s["metadata"]["job_applicant_id"] = s.get("job_applicant_id", "")

    # Stratified split by label
    rng = random.Random(args.seed)
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    train, val, test = [], [], []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    for name, data in [("train", train), ("validation", val), ("test", test)]:
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        labels = Counter(s["label"] for s in data)
        tiers = Counter(s["metadata"]["quality_tier"] for s in data)
        logger.info(f"{name}: {len(data)} samples, labels={dict(labels)}, tiers={dict(tiers)}")

    logger.info(f"Output: {out}")

if __name__ == "__main__":
    main()
