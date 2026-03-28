#!/usr/bin/env python3
"""
Prepare v5 training data with stratified splitting following the design document.

Key changes from v3/v4 preparation:
  - Stratified split: each label group (good_fit/potential_fit/no_fit) is split
    independently at 80/10/10, then recombined. This guarantees identical label
    proportions across train/validation/test.
  - Saves split indices for reproducibility.
  - Includes soft_label in metadata (good_fit=1.0, potential_fit=0.4, no_fit=0.0).
  - Runs tuple availability analysis and φ coverage check.

Input:  Combined_Structured_V1.2_esco_enriched_v4_scored.jsonl
Output: preprocess/data_splits_v5/{train,validation,test}.jsonl
        preprocess/data_splits_v5/split_indices.json

Usage:
    python3 scripts/prepare_training_data_v5.py \
        --input preprocess/Combined_Structured_V1.2_esco_enriched_v4_scored.jsonl \
        --output-dir preprocess/data_splits_v5 \
        --min-tier C \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}

# Binary label for the pipeline (good_fit=positive, rest=negative)
BINARY_LABEL_MAP = {"good_fit": 1, "potential_fit": 0, "no_fit": 0}

# Soft labels preserving ordinal structure
SOFT_LABEL_MAP = {"good_fit": 1.0, "potential_fit": 0.4, "no_fit": 0.0}


def convert_record(rec: dict) -> dict | None:
    """Convert enriched record to training format.

    Returns None for records that would fail DataLoader validation.
    """
    raw = rec.get("_raw", {})
    raw_resume = raw.get("resume", {})
    raw_job = raw.get("job", {})
    enrichment = rec.get("esco_enrichment_v3", {})

    label_str = rec.get("label", "")
    binary_label = BINARY_LABEL_MAP.get(label_str)
    if binary_label is None:
        return None

    # Job must have a non-empty title
    job_title = raw_job.get("title", "")
    if not isinstance(job_title, str) or not job_title.strip():
        return None

    # Job must have a non-empty description
    job_desc = raw_job.get("description", {})
    if isinstance(job_desc, dict):
        if not any(isinstance(v, str) and len(v.strip()) > 20 for v in job_desc.values()):
            return None
    elif isinstance(job_desc, str):
        if not job_desc.strip():
            return None
    else:
        return None

    # Resume must have experience or skills
    if not raw_resume.get("experience") and not raw_resume.get("skills"):
        return None

    resume = {
        "role": raw_resume.get("role", ""),
        "experience": raw_resume.get("experience", []),
        "experience_level": raw_resume.get("experience_level", ""),
        "skills": raw_resume.get("skills", []),
        "keywords": raw_resume.get("keywords", []),
        "skill_uris": enrichment.get("resume_skill_uris", []),
    }

    occ = enrichment.get("occupation", {})
    resolved_uri = occ.get("resolved_uri")

    job = {
        "title": raw_job.get("title", ""),
        "description": raw_job.get("description", {}),
        "skills": raw_job.get("skills", []),
        "experience_level": raw_job.get("experience_level", ""),
        "occupation_uri": resolved_uri or raw_job.get("occupation_uri", ""),
        "skill_uris": enrichment.get("job_skill_uris", []),
        "original_label": label_str,
    }

    scores = enrichment.get("scores", {})
    metadata = {
        "job_applicant_id": rec.get("job_applicant_id"),
        "original_label": label_str,
        "soft_label": SOFT_LABEL_MAP.get(label_str, 0.0),
        "quality_tier": enrichment.get("quality_tier", "F"),
        "ontology_similarity": scores.get("ontology_similarity"),
        "ot_distance": scores.get("ot_distance"),
        "essential_coverage": scores.get("essential_coverage"),
        "optional_coverage": scores.get("optional_coverage"),
        "skill_overlap": scores.get("skill_overlap"),
        "occupation_match_mode": occ.get("match_mode"),
        "occupation_match_score": occ.get("match_score"),
        "resume_skill_uri_count": len(enrichment.get("resume_skill_uris", [])),
        "job_skill_uri_count": len(enrichment.get("job_skill_uris", [])),
    }

    return {
        "resume": resume,
        "job": job,
        "label": binary_label,
        "job_applicant_id": rec.get("job_applicant_id"),
        "metadata": metadata,
    }


def stratified_split(samples: list[dict], seed: int = 42,
                     train_ratio: float = 0.8, val_ratio: float = 0.1
                     ) -> tuple[list, list, list]:
    """Split samples stratified by original_label, preserving proportions exactly."""
    rng = random.Random(seed)

    # Group by label
    by_label: dict[str, list] = defaultdict(list)
    for s in samples:
        label = s["metadata"]["original_label"]
        by_label[label].append(s)

    train, val, test = [], [], []

    for label in ["good_fit", "potential_fit", "no_fit"]:
        group = by_label.get(label, [])
        rng.shuffle(group)

        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    # Shuffle each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def tuple_availability_analysis(samples: list[dict]) -> None:
    """Check how many jobs have candidates from all 3 tiers."""
    job_labels: dict[str, set] = defaultdict(set)
    for s in samples:
        job_id = s.get("job_applicant_id", s["job"].get("title", "unknown"))
        label = s["metadata"]["original_label"]
        job_labels[str(job_id)].append(label) if False else job_labels[str(job_id)].add(label)

    full = sum(1 for tiers in job_labels.values()
               if {"good_fit", "potential_fit", "no_fit"}.issubset(tiers))
    partial = sum(1 for tiers in job_labels.values() if len(tiers) == 2)
    single = len(job_labels) - full - partial

    logger.info("Tuple availability analysis:")
    logger.info(f"  Total unique jobs: {len(job_labels)}")
    logger.info(f"  Jobs with all 3 tiers: {full} ({100*full/len(job_labels):.1f}%)")
    logger.info(f"  Jobs with 2 tiers: {partial} ({100*partial/len(job_labels):.1f}%)")
    logger.info(f"  Jobs with 1 tier: {single} ({100*single/len(job_labels):.1f}%)")

    if full / len(job_labels) < 0.3:
        logger.info("  → In-batch tuple construction (Strategy B) recommended")
    else:
        logger.info("  → Natural tuple construction (Strategy A) viable")


def phi_coverage_check(samples: list[dict]) -> None:
    """Verify URI coverage for φ computation."""
    tiers = Counter()
    for s in samples:
        has_resume = len(s["resume"].get("skill_uris", [])) > 0
        has_job = len(s["job"].get("skill_uris", [])) > 0
        if has_resume and has_job:
            tiers["A"] += 1
        elif has_resume or has_job:
            tiers["C"] += 1
        else:
            tiers["D"] += 1

    total = sum(tiers.values())
    logger.info("φ computability (URI coverage):")
    for tier in ["A", "C", "D"]:
        count = tiers.get(tier, 0)
        logger.info(f"  Tier {tier}: {count} ({100*count/total:.1f}%)")
    logger.info(f"  Tier A samples get real φ; C/D get fallback φ=0.5")


def main():
    parser = argparse.ArgumentParser(description="Prepare v5 training data (stratified)")
    parser.add_argument("--input", required=True,
                        help="Path to enriched+scored JSONL")
    parser.add_argument("--output-dir", default="preprocess/data_splits_v5")
    parser.add_argument("--min-tier", default="C",
                        help="Minimum quality tier to include (A/B/C/D/F)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    min_tier_rank = TIER_ORDER.get(args.min_tier, 2)

    # ── Load and convert ──
    samples = []
    tier_counts = Counter()
    label_counts = Counter()
    skipped_tier = 0
    skipped_convert = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            enrichment = rec.get("esco_enrichment_v3", {})
            tier = enrichment.get("quality_tier", "F")
            tier_counts[tier] += 1

            if TIER_ORDER.get(tier, 4) > min_tier_rank:
                skipped_tier += 1
                continue

            converted = convert_record(rec)
            if converted is None:
                skipped_convert += 1
                continue

            samples.append(converted)
            label_counts[converted["metadata"]["original_label"]] += 1

    logger.info(f"Loaded {len(samples)} samples (skipped {skipped_tier} tier, {skipped_convert} convert)")
    logger.info(f"Source tier distribution: {dict(tier_counts)}")
    logger.info(f"Label distribution: {dict(label_counts)}")

    # ── Stratified split ──
    train, val, test = stratified_split(samples, seed=args.seed)

    # ── Write splits ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_indices = {}
    for split_name, split_data in [("train", train), ("validation", val), ("test", test)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in split_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        split_labels = Counter(r["metadata"]["original_label"] for r in split_data)
        total = len(split_data)
        logger.info(
            f"  {split_name:12s}: {total} samples — "
            f"good_fit={split_labels['good_fit']} ({100*split_labels['good_fit']/total:.1f}%), "
            f"potential_fit={split_labels['potential_fit']} ({100*split_labels['potential_fit']/total:.1f}%), "
            f"no_fit={split_labels['no_fit']} ({100*split_labels['no_fit']/total:.1f}%)"
        )

        # Save indices (job_applicant_id) for reproducibility
        split_indices[split_name] = [r.get("job_applicant_id") for r in split_data]

    # Save split indices
    indices_path = out_dir / "split_indices.json"
    with open(indices_path, "w") as f:
        json.dump(split_indices, f, indent=2)
    logger.info(f"Split indices saved to {indices_path}")

    # ── Tuple availability analysis ──
    tuple_availability_analysis(train)

    # ── φ coverage check ──
    phi_coverage_check(train)

    logger.info(f"\nOutput: {out_dir}")
    logger.info("Next step: run compute_phi_scores.py on the new splits")


if __name__ == "__main__":
    main()
