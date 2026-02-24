#!/usr/bin/env python3
"""
Semantic ESCO Skill Matcher
============================
Uses sentence-transformers embeddings to match raw skill strings
to ESCO skill URIs via cosine similarity. Falls back to extracting
skills from job_description text when raw job_skills are empty.

Replaces/augments the fuzzy string matching in reenrich_esco_v3.py
for the job side, where vocabulary mismatch is the main problem
(e.g. "AWS" → "cloud computing", "Docker" → "containerisation").

Usage:
    python3 scripts/semantic_esco_matcher.py \
        --input preprocess/Combined_Structured_V1.2_esco_enriched_v3.jsonl \
        --output preprocess/Combined_Structured_V1.2_esco_enriched_v4.jsonl \
        --esco-dir dataset/esco \
        --threshold 0.55
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# ESCO Embedding Index
# ──────────────────────────────────────────────────────────────

class ESCOSemanticIndex:
    """Pre-computed embedding index over all ESCO skill labels."""

    def __init__(self, skills_csv: str, model: SentenceTransformer,
                 batch_size: int = 256):
        logger.info("Building ESCO semantic index...")
        df = pd.read_csv(skills_csv)

        # Collect all (label, uri) pairs — preferred + alt labels
        self.labels: List[str] = []
        self.uris: List[str] = []
        self.preferred: List[str] = []  # preferred label for each entry

        for _, row in df.iterrows():
            uri = str(row["conceptUri"]).strip()
            pref = str(row["preferredLabel"]).strip()

            # Add preferred label
            self.labels.append(pref.lower())
            self.uris.append(uri)
            self.preferred.append(pref)

            # Add alt labels
            alt = row.get("altLabels")
            if pd.notna(alt):
                for a in str(alt).replace("\r", "\n").split("\n"):
                    a = a.strip()
                    if a and len(a) >= 2:
                        self.labels.append(a.lower())
                        self.uris.append(uri)
                        self.preferred.append(pref)

        logger.info(f"  {len(self.labels)} labels from {df.shape[0]} skills")

        # Encode all labels
        logger.info("  Encoding ESCO labels (this takes ~30s on CPU)...")
        self.embeddings = model.encode(
            self.labels, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.info(f"  Embedding matrix: {self.embeddings.shape}")

        # Build exact-match lookup for fast bypass
        self.exact_lookup: Dict[str, Tuple[str, str]] = {}
        for label, uri, pref in zip(self.labels, self.uris, self.preferred):
            if label not in self.exact_lookup:
                self.exact_lookup[label] = (uri, pref)

    def match(self, query: str, model: SentenceTransformer,
              threshold: float = 0.55, top_k: int = 3
              ) -> List[Dict]:
        """Find top ESCO matches for a query string.

        Returns list of {uri, preferred_label, matched_label, score}.
        """
        q = query.lower().strip()
        if not q:
            return []

        # Exact match shortcut
        if q in self.exact_lookup:
            uri, pref = self.exact_lookup[q]
            return [{"uri": uri, "preferred_label": pref,
                     "matched_label": q, "score": 1.0, "mode": "exact"}]

        # Semantic match
        q_emb = model.encode([q], normalize_embeddings=True)
        sims = q_emb @ self.embeddings.T  # (1, N)
        sims = sims[0]

        top_indices = np.argsort(sims)[-top_k:][::-1]
        results = []
        seen_uris = set()
        for idx in top_indices:
            score = float(sims[idx])
            if score < threshold:
                break
            uri = self.uris[idx]
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
            results.append({
                "uri": uri,
                "preferred_label": self.preferred[idx],
                "matched_label": self.labels[idx],
                "score": round(score, 4),
                "mode": "semantic",
            })
        return results

    def match_batch(self, queries: List[str], model: SentenceTransformer,
                    threshold: float = 0.55, top_k: int = 3
                    ) -> List[List[Dict]]:
        """Batch match multiple queries at once (much faster)."""
        if not queries:
            return []

        # Separate exact matches from semantic
        results = [None] * len(queries)
        semantic_indices = []
        semantic_queries = []

        for i, q in enumerate(queries):
            q_lower = q.lower().strip()
            if q_lower in self.exact_lookup:
                uri, pref = self.exact_lookup[q_lower]
                results[i] = [{"uri": uri, "preferred_label": pref,
                               "matched_label": q_lower, "score": 1.0,
                               "mode": "exact"}]
            else:
                semantic_indices.append(i)
                semantic_queries.append(q_lower)

        if semantic_queries:
            q_embs = model.encode(semantic_queries, normalize_embeddings=True,
                                  batch_size=64)
            sims = q_embs @ self.embeddings.T  # (M, N)

            for j, orig_idx in enumerate(semantic_indices):
                row = sims[j]
                top_idx = np.argsort(row)[-top_k:][::-1]
                matches = []
                seen_uris = set()
                for idx in top_idx:
                    score = float(row[idx])
                    if score < threshold:
                        break
                    uri = self.uris[idx]
                    if uri in seen_uris:
                        continue
                    seen_uris.add(uri)
                    matches.append({
                        "uri": uri,
                        "preferred_label": self.preferred[idx],
                        "matched_label": self.labels[idx],
                        "score": round(score, 4),
                        "mode": "semantic",
                    })
                results[orig_idx] = matches

        return results


# ──────────────────────────────────────────────────────────────
# Skill extraction from job description text
# ──────────────────────────────────────────────────────────────

# Common generic phrases to skip
_SKIP_PHRASES = {
    "the", "and", "for", "with", "our", "you", "your", "this",
    "that", "will", "are", "have", "has", "been", "from", "they",
    "their", "about", "into", "also", "such", "than", "other",
    "more", "most", "some", "any", "all", "each", "every",
    "both", "few", "many", "much", "own", "same", "able",
    "just", "over", "only", "very", "well", "back", "even",
    "still", "also", "here", "there", "when", "where", "how",
    "what", "which", "who", "whom", "why",
}


def extract_candidate_skills_from_description(
    description: str,
    esco_index: ESCOSemanticIndex,
    model: SentenceTransformer,
    threshold: float = 0.75,  # high threshold — only confident matches
    max_candidates: int = 10,
) -> List[Dict]:
    """
    Extract skill candidates from job description text by:
    1. Splitting into noun-phrase-like chunks (2-3 words)
    2. Matching each chunk against ESCO semantically
    3. Returning deduplicated matches above threshold

    Uses a high threshold (0.75) to avoid noisy matches from
    generic description prose.
    """
    if not description or not description.strip():
        return []

    # Extract 2-3 word phrases (most skill names are this length)
    bigrams = re.findall(
        r"\b([a-zA-Z\+\#\.]{2,}(?:\s+[a-zA-Z\+\#\.]{2,}){1,2})\b",
        description)
    # Also extract known-format technical terms (single words, 3+ chars)
    singles = re.findall(r"\b([A-Z][a-zA-Z\+\#\.]{2,})\b", description)
    # And ALL-CAPS acronyms (SQL, AWS, etc.)
    acronyms = re.findall(r"\b([A-Z]{2,6})\b", description)

    candidates = set()
    for phrase in bigrams:
        phrase = phrase.strip().lower()
        tokens = phrase.split()
        if all(t in _SKIP_PHRASES for t in tokens):
            continue
        if len(phrase) >= 4:
            candidates.add(phrase)
    for w in singles + acronyms:
        w_lower = w.lower()
        if w_lower not in _SKIP_PHRASES and len(w_lower) >= 2:
            candidates.add(w_lower)

    if not candidates:
        return []

    # Cap candidates
    candidates = list(candidates)[:150]

    # Batch match with high threshold
    all_matches = esco_index.match_batch(candidates, model,
                                         threshold=threshold, top_k=1)

    # Deduplicate by URI
    seen_uris: Set[str] = set()
    results = []
    for matches in all_matches:
        if not matches:
            continue
        for m in matches:
            if m["uri"] not in seen_uris:
                seen_uris.add(m["uri"])
                results.append(m)

    # Sort by score, take top N
    results.sort(key=lambda x: -x["score"])
    return results[:max_candidates]


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Semantic ESCO skill matching")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--esco-dir", default="dataset/esco")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Cosine similarity threshold for semantic match")
    parser.add_argument("--desc-threshold", type=float, default=0.75,
                        help="Higher threshold for description extraction")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--side", default="job", choices=["job", "resume", "both"],
                        help="Which side to re-match")
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading sentence transformer: {args.model}")
    st_model = SentenceTransformer(args.model)

    # Build ESCO index
    esco_index = ESCOSemanticIndex(
        str(Path(args.esco_dir) / "skills_en.csv"), st_model)

    # Process records
    total = 0
    improved = 0
    from_desc = 0
    start = time.time()

    sides = ["job"] if args.side == "job" else ["resume"] if args.side == "resume" else ["job", "resume"]

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            rec = json.loads(line)
            total += 1
            e = rec.get("esco_enrichment_v3", {})

            for side in sides:
                # Get existing data
                raw_skills = e.get(f"{side}_skills_cleaned",
                                   rec.get(f"{side}_skills", []))
                old_uris = set(e.get(f"{side}_skill_uris", []))
                old_unmapped = e.get(f"{side}_unmapped_skills", [])

                new_uris = set(old_uris)
                new_details = []
                still_unmapped = []

                # Re-match previously unmapped skills semantically
                if old_unmapped:
                    matches = esco_index.match_batch(
                        old_unmapped, st_model, threshold=args.threshold)
                    for skill, skill_matches in zip(old_unmapped, matches):
                        if skill_matches:
                            for m in skill_matches:
                                new_uris.add(m["uri"])
                            new_details.append({
                                "input": skill,
                                "mode": "semantic",
                                "matches": skill_matches,
                            })
                        else:
                            still_unmapped.append(skill)

                # If still few URIs and we have description text, extract from it
                desc_key = "job_description" if side == "job" else "resume_text"
                description = rec.get(desc_key, "")
                if len(new_uris) < 3 and description:
                    desc_matches = extract_candidate_skills_from_description(
                        description, esco_index, st_model,
                        threshold=args.desc_threshold)
                    for m in desc_matches:
                        if m["uri"] not in new_uris:
                            new_uris.add(m["uri"])
                            new_details.append({
                                "input": f"[from_description]",
                                "mode": "description_extraction",
                                "matches": [m],
                            })
                    if desc_matches:
                        from_desc += 1

                # Update enrichment
                if len(new_uris) > len(old_uris):
                    improved += 1

                e[f"{side}_skill_uris"] = sorted(new_uris)
                e[f"{side}_unmapped_skills"] = still_unmapped

                # Store semantic match details
                if new_details:
                    e.setdefault("semantic_matches", {})[side] = new_details

            # Recompute skill overlap
            r_uris = set(e.get("resume_skill_uris", []))
            j_uris = set(e.get("job_skill_uris", []))
            if j_uris:
                e.setdefault("scores", {})["skill_overlap"] = round(
                    len(r_uris & j_uris) / len(j_uris), 4)

            rec["esco_enrichment_v3"] = e
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if total % 500 == 0:
                elapsed = time.time() - start
                logger.info(f"  {total} records ({elapsed:.1f}s), "
                            f"improved: {improved}, from_desc: {from_desc}")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"Done: {total} records in {elapsed:.1f}s")
    logger.info(f"  Improved (more URIs): {improved} ({improved/total*100:.1f}%)")
    logger.info(f"  Used description extraction: {from_desc}")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
