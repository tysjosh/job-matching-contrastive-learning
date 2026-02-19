#!/usr/bin/env python3
"""
Convert v3 scored enriched data into the training format expected by the pipeline.

Input:  Combined_Structured_V1.2_esco_enriched_v3_scored.jsonl
Output: {resume, job, label, job_applicant_id, metadata} JSONL

Key transformations:
  - Reshapes _raw.resume / _raw.job to top-level resume/job
  - Injects resolved UUID occupation_uri into job
  - Converts label (good_fit/potential_fit/no_fit) → 1/0
  - Adds ontology scores + quality tier to metadata
  - Filters by quality tier (configurable)
  - Splits into train/validation/test

Usage:
    python3 scripts/prepare_training_data_v3.py \
        --input preprocess/Combined_Structured_V1.2_esco_enriched_v3_scored.jsonl \
        --output-dir preprocess/data_splits_v3 \
        --min-tier C \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}

LABEL_MAP = {
    "good_fit": 1,
    "potential_fit": 0,
    "no_fit": 0,
}


def convert_record(rec: dict) -> dict | None:
    """Convert enriched record to training format.
    
    Returns None for records that would fail DataLoader validation:
    - No job title
    - Empty job description
    - No resume experience and no resume skills
    """
    raw = rec.get("_raw", {})
    raw_resume = raw.get("resume", {})
    raw_job = raw.get("job", {})
    enrichment = rec.get("esco_enrichment_v3", {})

    # Label
    label_str = rec.get("label", "")
    label = LABEL_MAP.get(label_str)
    if label is None:
        return None

    # ── Reject records that would fail DataLoader validation ──
    # Job must have a non-empty title
    job_title = raw_job.get("title", "")
    if not isinstance(job_title, str) or not job_title.strip():
        return None

    # Job must have a non-empty description
    job_desc = raw_job.get("description", {})
    if isinstance(job_desc, dict):
        has_desc = any(
            isinstance(v, str) and len(v.strip()) > 20
            for v in job_desc.values()
        )
        if not has_desc:
            return None
    elif isinstance(job_desc, str):
        if not job_desc.strip():
            return None
    else:
        return None

    # Resume must have experience or skills
    has_exp = bool(raw_resume.get("experience"))
    has_skills = bool(raw_resume.get("skills"))
    if not has_exp and not has_skills:
        return None

    # Resume — inject skill URIs for ontology-aware negative selection
    resume = {
        "role": raw_resume.get("role", ""),
        "experience": raw_resume.get("experience", []),
        "experience_level": raw_resume.get("experience_level", ""),
        "skills": raw_resume.get("skills", []),
        "keywords": raw_resume.get("keywords", []),
        "skill_uris": enrichment.get("resume_skill_uris", []),
    }

    # Job — inject resolved occupation URI + skill URIs
    occ = enrichment.get("occupation", {})
    resolved_uri = occ.get("resolved_uri")

    job = {
        "title": raw_job.get("title", ""),
        "description": raw_job.get("description", {}),
        "skills": raw_job.get("skills", []),
        "experience_level": raw_job.get("experience_level", ""),
        "occupation_uri": resolved_uri or raw_job.get("occupation_uri", ""),
        "skill_uris": enrichment.get("job_skill_uris", []),
    }

    # Metadata with ontology scores
    scores = enrichment.get("scores", {})
    metadata = {
        "job_applicant_id": rec.get("job_applicant_id"),
        "original_label": label_str,
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
        "label": label,
        "job_applicant_id": rec.get("job_applicant_id"),
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare v3 training data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-tier", default="C",
                        help="Minimum quality tier to include (A/B/C/D/F)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    min_tier_rank = TIER_ORDER.get(args.min_tier, 2)
    random.seed(args.seed)

    # Load and convert
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
            label_counts[converted["label"]] += 1

    logger.info(f"Loaded {len(samples)} samples (skipped {skipped_tier} by tier, {skipped_convert} by conversion)")
    logger.info(f"Tier distribution: {dict(tier_counts)}")
    logger.info(f"Label distribution: {dict(label_counts)}")

    # Shuffle and split
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    # Write splits
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("validation", val), ("test", test)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in split_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        split_labels = Counter(r["label"] for r in split_data)
        logger.info(f"  {split_name}: {len(split_data)} records (pos={split_labels.get(1,0)}, neg={split_labels.get(0,0)})")

    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
