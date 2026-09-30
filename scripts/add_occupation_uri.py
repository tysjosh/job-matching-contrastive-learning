"""
Small preprocessing script to add `occupation_uri` to job AND resume records by
matching job title/description or resume role/skills to ESCO occupations CSV data.

This script is non-invasive and does not modify repository source files.
It reads an input JSONL, attempts to identify the best-matching ESCO occupation
URI for each job and each resume, writes a new JSONL with `occupation_uri` added
when found on either side.

Why the resume side matters
----------------------------
The upstream enrichment pipeline (scripts/prepare_training_data_v5.py) resolves
occupation_uri for jobs only -- resume.occupation_uri is 0/8000 populated in
every career dataset file checked (source, splits, learning-curve fractions).
That's not a data-quality ceiling, it's a step that was never run on the resume
side: resumes carry a free-text `role` field ("Software Engineer Front End")
that is exactly the kind of text this scorer already matches against ESCO
occupation labels for jobs. ISCO negative selection / sample weighting
(contrastive_learning/batch_processor.py) needs an anchor-side occupation to
compute ISCO group distance; without resume.occupation_uri it falls back to
using the PAIRED POSITIVE JOB's occupation as a proxy for the resume's true
occupation (batch_processor.py:911), which is a plausible stand-in but not the
resume's own signal, and misses entirely on the ~966/6400 resumes with no
good_fit record to proxy from (verified in scripts/isco_signal_diagnosis.py).

Usage:
  python3 scripts/add_occupation_uri.py \
    --input training_small.jsonl \
    --output training_small_with_uri.jsonl \
    --esco-csv-dir dataset/esco/

  # Job side only (previous behavior):
  python3 scripts/add_occupation_uri.py --input in.jsonl --output out.jsonl \
    --esco-csv-dir dataset/esco/ --no-resume

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


def resume_text_from_resume(resume: Dict) -> str:
    """Mirrors job_text_from_job: role is the closest analogue of a job title,
    and skills carry the same kind of matchable vocabulary as a job's keywords.
    Experience descriptions are deliberately excluded -- they are long free text
    dominated by employer names and achievement narrative, which would dilute
    the label/description word-overlap scoring rather than sharpen it (the same
    reason job_text_from_job uses description.keywords, not the full narrative).
    """
    parts = []
    role = resume.get('role')
    if role:
        parts.append(str(role))
    skills = resume.get('skills', [])
    if isinstance(skills, list):
        parts.extend([str(s) for s in skills])
    elif isinstance(skills, str):
        parts.append(skills)
    return ' '.join(parts).lower()


def best_occupation_match(text: str, occupation_items, threshold: int):
    """Shared scoring loop used for both job and resume text."""
    best_score = 0
    best_uri = None
    for uri, data, _pref_lower in occupation_items:
        s = score_job_to_occupation(text, data)
        if s > best_score:
            best_score = s
            best_uri = uri
    if best_score >= threshold and best_uri:
        return best_uri
    return None


def annotate_file(input_path: str, output_path: str, esco_csv_dir: str,
                  threshold: int = 10, annotate_resume: bool = True):
    occupations = load_occupations(esco_csv_dir)

    # Precompute a list of (uri, data, lower_pref) for faster matching
    occupation_items = [(uri, data, (data.get('preferred_label') or '').lower())
                        for uri, data in occupations.items()]

    with open(input_path, 'r', encoding='utf-8') as inf, open(output_path, 'w', encoding='utf-8') as outf:
        total = 0
        job_annotated = 0
        resume_annotated = 0
        resume_skipped_existing = 0
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
            job_uri = best_occupation_match(job_text, occupation_items, threshold)
            if job_uri:
                job['occupation_uri'] = job_uri
                job_annotated += 1

            if annotate_resume:
                resume = record.get('resume', {})
                # Never overwrite an occupation_uri a record already carries --
                # this script is additive, matching the job-side behavior of
                # only setting the field when a match clears the threshold.
                if resume.get('occupation_uri'):
                    resume_skipped_existing += 1
                else:
                    resume_text = resume_text_from_resume(resume)
                    resume_uri = best_occupation_match(resume_text, occupation_items, threshold)
                    if resume_uri:
                        resume['occupation_uri'] = resume_uri
                        resume_annotated += 1

            # Write the (potentially annotated) record
            json.dump(record, outf, ensure_ascii=False)
            outf.write('\n')

    logger.info(f"Wrote {total} records to {output_path}")
    logger.info(f"  job.occupation_uri annotated: {job_annotated}/{total}")
    if annotate_resume:
        logger.info(f"  resume.occupation_uri annotated: {resume_annotated}/{total} "
                    f"({resume_skipped_existing} already had one)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--esco-csv-dir', default='dataset/esco/')
    parser.add_argument('--threshold', type=int, default=10,
                        help='Minimum score to accept occupation match')
    parser.add_argument('--no-resume', action='store_true',
                        help='Skip resume-side annotation (previous, job-only behavior)')
    args = parser.parse_args()

    try:
        annotate_file(args.input, args.output, args.esco_csv_dir,
                      threshold=args.threshold, annotate_resume=not args.no_resume)
    except Exception as e:
        logger.error(f"Failed to annotate file: {e}")
        raise