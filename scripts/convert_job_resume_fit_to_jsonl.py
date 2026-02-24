#!/usr/bin/env python3
"""
Convert dataset/job_resume_fit.csv → JSONL format compatible with reenrich_esco_v3.py

Maps:
  resume_skill_list    → resume_skills
  job_required_skills  → job_skills
  first line of job_text → job_title
  resume_text, job_text, category, scores → preserved
"""
import ast
import json
import pandas as pd

INPUT  = "dataset/job_resume_fit.csv"
OUTPUT = "dataset/job_resume_fit.jsonl"


def safe_parse(val):
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        parsed = ast.literal_eval(val)
        return [str(s).strip() for s in parsed if str(s).strip()] if isinstance(parsed, list) else []
    except Exception:
        return []


def extract_job_title(job_text):
    if pd.isna(job_text) or not isinstance(job_text, str):
        return ""
    first_line = job_text.strip().split("\n")[0].strip()
    # Remove separator lines
    first_line = first_line.replace("-", "").strip()
    return first_line if len(first_line) > 2 else ""


def main():
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} rows from {INPUT}")

    records = []
    for _, row in df.iterrows():
        rec = {
            "resume_text": str(row["resume_text"]) if pd.notna(row["resume_text"]) else "",
            "job_text": str(row["job_text"]) if pd.notna(row["job_text"]) else "",
            "job_title": extract_job_title(row["job_text"]),
            "category": str(row["category"]) if pd.notna(row["category"]) else "",
            "resume_skills": safe_parse(row["resume_skill_list"]),
            "job_skills": safe_parse(row["job_required_skills"]),
            "ai_matched_skills": safe_parse(row["ai_matched_skills"]),
            "ai_match_score": float(row["ai_match_score"]) if pd.notna(row["ai_match_score"]) else None,
            "skill_string_match_score": float(row["skill_string_match_score"]) if pd.notna(row["skill_string_match_score"]) else None,
            "fuzzy_match_score": float(row["fuzzy_match_score"]) if pd.notna(row["fuzzy_match_score"]) else None,
            "ID": int(row["ID"]) if pd.notna(row["ID"]) else None,
            "_raw": {"job": {}, "resume": {}},
        }
        records.append(rec)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {OUTPUT}")

    # Quick sanity check
    sample = records[0]
    print(f"\nSample record:")
    print(f"  job_title: {sample['job_title']}")
    print(f"  category: {sample['category']}")
    print(f"  resume_skills ({len(sample['resume_skills'])}): {sample['resume_skills'][:5]}...")
    print(f"  job_skills ({len(sample['job_skills'])}): {sample['job_skills'][:5]}...")


if __name__ == "__main__":
    main()
