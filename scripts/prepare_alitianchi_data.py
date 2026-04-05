#!/usr/bin/env python3
"""
Convert AliTianChi translated English CSVs + original label file into
training JSONL format compatible with the contrastive learning pipeline.

Usage:
    python3 scripts/prepare_alitianchi_data.py \
        --jd dataset/translated_files/translated_eng_jd.csv \
        --resume dataset/translated_files/translated_eng_resume.csv \
        --labels dataset/AliTianChi_release_1203/train_labeled_data.jsonl \
        --output dataset/alitianchi_train.jsonl
"""
import argparse, csv, json, logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_jds(path):
    jds = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jds[row["jd_no"]] = row
    logger.info(f"Loaded {len(jds)} job descriptions")
    return jds


def load_resumes(path):
    resumes = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resumes[row["user_id"]] = row
    logger.info(f"Loaded {len(resumes)} resumes")
    return resumes


def parse_resume_skills(experience_str):
    """Parse pipe-separated experience/skills into a list."""
    if not experience_str or experience_str.strip() == "":
        return []
    return [s.strip() for s in experience_str.split("|") if s.strip()]


def build_resume(row):
    """Convert resume CSV row to training format."""
    skills = parse_resume_skills(row.get("experience_en", ""))
    return {
        "role": row.get("cur_jd_type_en", ""),
        "experience": row.get("experience_en", ""),
        "experience_level": row.get("cur_degree_id_en", ""),
        "skills": skills,
        "keywords": skills,
    }


def build_job(row):
    """Convert JD CSV row to training format."""
    return {
        "title": row.get("jd_title_en", ""),
        "description": row.get("job_description_en", ""),
        "skills": [],  # Will be extracted later by ESCO matcher
        "experience_level": row.get("min_edu_level_en", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare AliTianChi data for training")
    parser.add_argument("--jd", default="dataset/translated_files/translated_eng_jd.csv")
    parser.add_argument("--resume", default="dataset/translated_files/translated_eng_resume.csv")
    parser.add_argument("--labels", default="dataset/AliTianChi_release_1203/train_labeled_data.jsonl")
    parser.add_argument("--output", default="dataset/alitianchi_train.jsonl")
    args = parser.parse_args()

    jds = load_jds(args.jd)
    resumes = load_resumes(args.resume)

    total, matched, skipped = 0, 0, 0
    label_counts = Counter()

    with open(args.labels) as fin, open(args.output, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            total += 1

            jd_row = jds.get(rec["jd_no"])
            resume_row = resumes.get(rec["user_id"])

            if not jd_row or not resume_row:
                skipped += 1
                continue

            satisfied = rec["satisfied"]
            label_counts[satisfied] += 1

            sample = {
                "resume": build_resume(resume_row),
                "job": build_job(jd_row),
                "label": satisfied,
                "job_applicant_id": f"{rec['user_id']}_{rec['jd_no']}",
                "metadata": {
                    "job_applicant_id": f"{rec['user_id']}_{rec['jd_no']}",
                    "original_label": "satisfied" if satisfied == 1 else "not_satisfied",
                    "user_id": rec["user_id"],
                    "jd_no": rec["jd_no"],
                },
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            matched += 1

    logger.info(f"Total: {total}, Matched: {matched}, Skipped: {skipped}")
    logger.info(f"Labels: {dict(label_counts)}")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
