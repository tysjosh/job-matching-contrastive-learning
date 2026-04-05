#!/usr/bin/env python3
"""
ESCO skill enrichment for AliTianChi dataset.

Matches resume skills (from pipe-separated keywords) and job skills
(extracted from description text) to ESCO URIs using the semantic matcher.

Usage:
    python3 scripts/enrich_alitianchi_esco.py \
        --input dataset/alitianchi_train.jsonl \
        --output dataset/alitianchi_train_enriched.jsonl
"""
import argparse, json, logging, time, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Import from existing scripts
sys.path.insert(0, str(Path(__file__).parent))
from semantic_esco_matcher import (
    ESCOSemanticIndex, extract_candidate_skills_from_description
)
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/alitianchi_train.jsonl")
    parser.add_argument("--output", default="dataset/alitianchi_train_enriched.jsonl")
    parser.add_argument("--esco-dir", default="dataset/esco")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--desc-threshold", type=float, default=0.75)
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model}")
    st_model = SentenceTransformer(args.model)

    logger.info("Building ESCO index...")
    esco_index = ESCOSemanticIndex(
        str(Path(args.esco_dir) / "skills_en.csv"), st_model)

    total = 0
    resume_matched = 0
    job_matched = 0
    start = time.time()

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            total += 1

            # --- Resume skill matching ---
            resume_skills = rec["resume"].get("skills", [])
            resume_uris = set()
            if resume_skills:
                matches = esco_index.match_batch(
                    resume_skills, st_model, threshold=args.threshold, top_k=1)
                for skill_matches in matches:
                    if skill_matches:
                        for m in skill_matches:
                            resume_uris.add(m["uri"])
            if resume_uris:
                resume_matched += 1
            rec["resume"]["skill_uris"] = sorted(resume_uris)

            # --- Job skill extraction from description ---
            job_desc = rec["job"].get("description", "")
            job_uris = set()
            if job_desc:
                desc_matches = extract_candidate_skills_from_description(
                    job_desc, esco_index, st_model,
                    threshold=args.desc_threshold, max_candidates=15)
                for m in desc_matches:
                    job_uris.add(m["uri"])
            if job_uris:
                job_matched += 1
            rec["job"]["skill_uris"] = sorted(job_uris)
            rec["job"]["skills"] = [m["preferred_label"] for m in desc_matches] if job_desc else []

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if total % 1000 == 0:
                elapsed = time.time() - start
                logger.info(f"  {total} records ({elapsed:.1f}s) — "
                            f"resume: {resume_matched}, job: {job_matched}")

    elapsed = time.time() - start
    logger.info(f"Done: {total} records in {elapsed:.1f}s")
    logger.info(f"  Resume matched: {resume_matched}/{total} ({100*resume_matched/total:.1f}%)")
    logger.info(f"  Job matched: {job_matched}/{total} ({100*job_matched/total:.1f}%)")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
