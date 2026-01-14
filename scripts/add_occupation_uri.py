"""
Small preprocessing script to add `occupation_uri` to job records by matching
job title/description to ESCO occupations CSV data.

This script is non-invasive and does not modify repository source files.
It reads an input JSONL, attempts to identify the best-matching ESCO occupation
URI for each job, writes a new JSONL with `occupation_uri` added when found.

Usage:
  python3 scripts/add_occupation_uri.py \
    --input training_small.jsonl \
    --output training_small_with_uri.jsonl \
    --esco-csv-dir dataset/esco/

Note: Requires ESCO CSV files (occupations_en.csv) in the `--esco-csv-dir`.
If those files are not present, the script will fail with a clear error.
"""

import json
import argparse
import logging
import os
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_occupations(esco_csv_dir: str) -> Dict[str, Dict]:
    """Load ESCO occupations CSV into a mapping uri -> data.
    Uses a lightweight CSV reader to avoid importing heavy modules.
    """
    import csv
    occupations_file = os.path.join(esco_csv_dir, 'occupations_en.csv')
    if not os.path.exists(occupations_file):
        raise FileNotFoundError(
            f"ESCO occupations CSV not found: {occupations_file}")

    occupations = {}
    with open(occupations_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = row.get('conceptUri') or row.get('occupationUri')
            if not uri:
                continue
            occupations[uri] = {
                'preferred_label': row.get('preferredLabel', ''),
                'alt_labels': row.get('altLabels', '').split('\n') if row.get('altLabels') else [],
                'description': row.get('description', '')
            }
    logger.info(f"Loaded {len(occupations)} occupations from ESCO CSV")
    return occupations


def score_job_to_occupation(job_text: str, occupation: Dict) -> int:
    """Simple scoring: reward preferred label and alt label substring matches and description overlaps."""
    score = 0
    pref = occupation.get('preferred_label', '').lower()
    if pref and pref in job_text:
        score += len(pref) * 3

    for alt in occupation.get('alt_labels', []):
        alt = alt.lower().strip()
        if alt and alt in job_text:
            score += len(alt)

    desc = occupation.get('description', '').lower()
    if desc:
        # Count shared words
        job_words = set(job_text.split())
        desc_words = set(desc.split())
        common = job_words & desc_words
        score += len(common)

    return score


def job_text_from_job(job: Dict) -> str:
    parts = []
    if 'title' in job and job['title']:
        parts.append(str(job['title']))
    desc = job.get('description', {})
    if isinstance(desc, dict):
        parts.append(str(desc.get('original', '')))
        kws = desc.get('keywords', [])
        if isinstance(kws, list):
            parts.extend([str(k) for k in kws])
    elif isinstance(desc, str):
        parts.append(desc)
    return ' '.join(parts).lower()


def annotate_file(input_path: str, output_path: str, esco_csv_dir: str, threshold: int = 10):
    occupations = load_occupations(esco_csv_dir)

    # Precompute a list of (uri, data, lower_pref) for faster matching
    occupation_items = [(uri, data, (data.get('preferred_label') or '').lower())
                        for uri, data in occupations.items()]

    with open(input_path, 'r', encoding='utf-8') as inf, open(output_path, 'w', encoding='utf-8') as outf:
        total = 0
        annotated = 0
        for line in inf:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line")
                continue

            job = record.get('job', {})
            job_text = job_text_from_job(job)
            best_score = 0
            best_uri = None

            for uri, data, pref_lower in occupation_items:
                s = score_job_to_occupation(job_text, data)
                if s > best_score:
                    best_score = s
                    best_uri = uri

            if best_score >= threshold and best_uri:
                # Annotate job with occupation_uri
                job['occupation_uri'] = best_uri
                annotated += 1

            # Write the (potentially annotated) record
            json.dump(record, outf, ensure_ascii=False)
            outf.write('\n')

    logger.info(
        f"Wrote {total} records to {output_path}, annotated {annotated} with occupation_uri")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--esco-csv-dir', default='dataset/esco/')
    parser.add_argument('--threshold', type=int, default=10,
                        help='Minimum score to accept occupation match')
    args = parser.parse_args()

    try:
        annotate_file(args.input, args.output, args.esco_csv_dir,
                      threshold=args.threshold)
    except Exception as e:
        logger.error(f"Failed to annotate file: {e}")
        raise