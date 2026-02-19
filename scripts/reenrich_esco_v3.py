#!/usr/bin/env python3
"""
Re-enrich Combined_Structured_V1.2 with ESCO data, fixing v2 issues:

1. Slug→UUID occupation URI resolution via fuzzy label matching
2. Occupation-guided essential/optional skill coverage scoring
3. Skill string cleaning (CamelCase splitting, alias normalization, degluing)
4. Improved fuzzy skill linking with cleaned inputs
5. Records with empty skills flagged (esco_quality_tier)
6. No reliance on ancestor expansion (use graph distance downstream instead)

Usage:
    python3 scripts/reenrich_esco_v3.py \
        --input preprocess/Combined_Structured_V1.2_esco_enriched_v2.jsonl \
        --output preprocess/Combined_Structured_V1.2_esco_enriched_v3.jsonl \
        --esco-dir dataset/esco
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Skill string cleaning  (Fix #5)
# ──────────────────────────────────────────────────────────────

# Common aliases → canonical form
SKILL_ALIASES = {
    "j2ee": "java ee",
    "j2se": "java se",
    "ms excel": "excel",
    "ms word": "word",
    "ms office": "microsoft office",
    "ms sql": "microsoft sql server",
    "mssql": "microsoft sql server",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "k8s": "kubernetes",
    "tf": "terraform",
    "gcp": "google cloud platform",
    "js": "javascript",
    "ts": "typescript",
    "node": "node.js",
    "nodejs": "node.js",
    "react.js": "react",
    "reactjs": "react",
    "vue.js": "vue",
    "vuejs": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "c sharp": "c#",
    "csharp": "c#",
    "dotnet": ".net",
    "dot net": ".net",
    "ci cd": "continuous integration",
    "ci/cd": "continuous integration",
    "devops": "development operations",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "oop": "object-oriented programming",
    "rest api": "rest",
    "restful": "rest",
}

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_PROFICIENCY_RE = re.compile(r"\s*\((?:expert|intermediate|beginner|basic|advanced|"
                             r"good knowledge|experienced|proficient|master|"
                             r"entry|junior|mid|senior|lead)\)\s*$", re.I)


def _split_camel(s: str) -> str:
    """JavaJ2EE → Java J2EE, TestDrivenDesignDevelopment → Test Driven Design Development"""
    return _CAMEL_RE.sub(" ", s)


def clean_skill_string(raw: str) -> List[str]:
    """Clean a single raw skill string, possibly yielding multiple skills.

    Handles:
    - Proficiency suffixes: "Python (expert)" → "python"
    - CamelCase gluing: "JavaJ2EE" → "java", "java ee"
    - Slash/comma splitting: "HTML/CSS" → "html", "css"
    - Alias normalization
    """
    s = raw.strip()
    if not s:
        return []

    # Strip proficiency parenthetical
    s = _PROFICIENCY_RE.sub("", s).strip()

    # Split CamelCase
    s = _split_camel(s)

    # Lowercase
    s = s.lower().strip()

    # Split on / or , if they look like skill separators (not inside a word like "CI/CD")
    # Only split if both sides are ≥2 chars
    parts = [s]
    new_parts = []
    for p in parts:
        if "/" in p:
            candidates = [x.strip() for x in p.split("/")]
            if all(len(c) >= 2 for c in candidates):
                new_parts.extend(candidates)
            else:
                new_parts.append(p)
        else:
            new_parts.append(p)
    parts = new_parts

    # Apply aliases
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        canonical = SKILL_ALIASES.get(p, p)
        # Clean remaining noise
        canonical = re.sub(r"[^a-z0-9\s\-\+\.\#]", " ", canonical)
        canonical = re.sub(r"\s+", " ", canonical).strip()
        if canonical and len(canonical) >= 2:
            result.append(canonical)
    return result


def clean_skill_list(raw_skills: List[str]) -> List[str]:
    """Clean and deduplicate a list of raw skill strings."""
    seen = set()
    cleaned = []
    for raw in raw_skills:
        for s in clean_skill_string(raw):
            if s not in seen:
                seen.add(s)
                cleaned.append(s)
    return cleaned


# ──────────────────────────────────────────────────────────────
# ESCO label index for skill linking  (same approach as v2 but
# with cleaned inputs)
# ──────────────────────────────────────────────────────────────

def _normalize_label(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = re.sub(r"[^a-z0-9\s\-\+\.\#]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_labels(x) -> List[str]:
    if pd.isna(x):
        return []
    return [p.strip() for p in str(x).replace("\r", "\n").split("\n") if p.strip()]


class SkillLinker:
    """Fast skill→ESCO linker using token inverted index for candidate pre-filtering."""

    def __init__(self, skills_csv: str, score_cutoff: int = 86, limit: int = 3,
                 max_candidates: int = 500):
        self.score_cutoff = score_cutoff
        self.limit = limit
        self.max_candidates = max_candidates

        # Build label → URI index
        df = pd.read_csv(skills_csv)
        self.label_index: Dict[str, Set[str]] = defaultdict(set)
        for _, row in df.iterrows():
            uri = row.get("conceptUri")
            if pd.isna(uri):
                continue
            uri = str(uri).strip()
            pref = row.get("preferredLabel")
            if not pd.isna(pref):
                self.label_index[_normalize_label(str(pref))].add(uri)
            for alt in _split_labels(row.get("altLabels")):
                self.label_index[_normalize_label(alt)].add(uri)

        self.label_keys = list(self.label_index.keys())

        # Build token inverted index for fast candidate retrieval
        self.token_to_labels: Dict[str, Set[int]] = defaultdict(set)
        for idx, label in enumerate(self.label_keys):
            for token in label.split():
                if len(token) >= 2:
                    self.token_to_labels[token].add(idx)

        logger.info(f"SkillLinker: {len(self.label_keys)} labels, "
                     f"{len(set().union(*self.label_index.values()))} URIs, "
                     f"{len(self.token_to_labels)} index tokens")

    def _get_candidates(self, query: str) -> List[str]:
        """Get candidate labels using token overlap (fast pre-filter)."""
        tokens = [t for t in query.split() if len(t) >= 2]
        if not tokens:
            return self.label_keys[:self.max_candidates]

        # Collect candidate indices with any token overlap
        candidate_indices: Set[int] = set()
        for token in tokens:
            if token in self.token_to_labels:
                candidate_indices |= self.token_to_labels[token]
            # Also try prefix matching for short queries
            if len(token) >= 3:
                for idx_token, idx_set in self.token_to_labels.items():
                    if idx_token.startswith(token[:3]):
                        candidate_indices |= idx_set

        if not candidate_indices:
            # Fallback: return a manageable subset
            return self.label_keys[:self.max_candidates]

        # Cap candidates
        indices = sorted(candidate_indices)[:self.max_candidates]
        return [self.label_keys[i] for i in indices]

    def link(self, skill_strings: List[str]) -> Tuple[Set[str], List[dict], List[str]]:
        """Link cleaned skill strings to ESCO URIs (exact then fuzzy with pre-filtering)."""
        linked_uris: Set[str] = set()
        details: List[dict] = []
        unmapped: List[str] = []

        for raw in skill_strings:
            key = _normalize_label(raw)
            if not key:
                continue

            # Exact
            if key in self.label_index:
                uris = set(self.label_index[key])
                linked_uris |= uris
                details.append({"input": raw, "mode": "exact",
                                "matches": [{"label": key, "score": 100, "uris": list(uris)}]})
                continue

            # Fuzzy with pre-filtered candidates
            candidates = self._get_candidates(key)
            hits = process.extract(key, candidates, scorer=fuzz.WRatio,
                                   limit=self.limit, score_cutoff=self.score_cutoff)
            if not hits:
                unmapped.append(raw)
                details.append({"input": raw, "mode": "none", "matches": []})
                continue

            match_objs = []
            for match_label, score, _ in hits:
                uris = list(self.label_index.get(match_label, []))
                if uris:
                    linked_uris |= set(uris)
                    match_objs.append({"label": match_label, "score": float(score), "uris": uris})
            if match_objs:
                details.append({"input": raw, "mode": "fuzzy", "matches": match_objs})
            else:
                unmapped.append(raw)
                details.append({"input": raw, "mode": "none", "matches": []})

        return linked_uris, details, unmapped


# ──────────────────────────────────────────────────────────────
# Occupation URI resolution: slug/plain-text → UUID  (Fix #1)
# ──────────────────────────────────────────────────────────────

def build_occupation_index(occupations_csv: str) -> Tuple[Dict[str, str], List[str]]:
    """Build normalized_label → UUID URI index for occupations.

    Returns:
        occ_label_to_uri: normalized_label → UUID URI
        occ_label_keys: list of normalized labels for fuzzy search
    """
    df = pd.read_csv(occupations_csv)
    occ_index: Dict[str, str] = {}
    for _, row in df.iterrows():
        uri = row.get("conceptUri")
        if pd.isna(uri):
            continue
        uri = str(uri).strip()
        pref = row.get("preferredLabel")
        if not pd.isna(pref):
            occ_index[_normalize_label(str(pref))] = uri
        for alt in _split_labels(row.get("altLabels")):
            key = _normalize_label(alt)
            if key not in occ_index:  # prefer preferredLabel
                occ_index[key] = uri
    logger.info(f"Occupation index: {len(occ_index)} labels → "
                f"{len(set(occ_index.values()))} unique URIs")
    return occ_index, list(occ_index.keys())


def _extract_title_from_slug(slug_uri: str) -> Optional[str]:
    """http://data.europa.eu/esco/occupation/senior-software-engineer → senior software engineer"""
    prefix = "http://data.europa.eu/esco/occupation/"
    if not slug_uri.startswith(prefix):
        return None
    tail = slug_uri[len(prefix):]
    # Skip if it looks like a UUID already
    if re.match(r"^[0-9a-f]{8}-", tail):
        return None
    # Convert slug to title
    title = tail.replace("-", " ").replace("_", " ").strip()
    # Skip garbage
    if not title or title in ("...", "") or len(title) < 3:
        return None
    return title.lower()


def resolve_occupation_uri(
    raw_uri: str,
    job_title: str,
    occ_index: Dict[str, str],
    occ_keys: List[str],
    score_cutoff: int = 82,
) -> Tuple[Optional[str], str, float]:
    """Resolve a slug/plain-text occupation URI to a real ESCO UUID URI.

    Returns: (resolved_uuid_uri, match_mode, match_score)
    """
    if not raw_uri:
        # Fall back to job_title
        raw_uri = ""

    # 1) Already a UUID URI?
    if re.match(r"http://data\.europa\.eu/esco/occupation/[0-9a-f]{8}-", raw_uri):
        return raw_uri, "uuid_direct", 100.0

    # 2) Extract searchable title from slug or use raw text
    search_title = _extract_title_from_slug(raw_uri)
    if not search_title:
        # Plain text like "Data Engineer" or "N/A"
        search_title = raw_uri.lower().strip()
        # Remove the URI prefix if present but not a valid slug
        prefix = "http://data.europa.eu/esco/occupation/"
        if search_title.startswith(prefix):
            search_title = search_title[len(prefix):]

    # Skip garbage values
    if not search_title or search_title in ("n/a", "not provided", "...", "unknown"):
        search_title = None

    # Try slug-derived title first, then fall back to job_title
    for candidate, source in [(search_title, "slug"), (_normalize_label(job_title), "job_title")]:
        if not candidate or len(candidate) < 3:
            continue

        # Exact match
        if candidate in occ_index:
            return occ_index[candidate], f"exact_{source}", 100.0

        # Fuzzy match
        hits = process.extract(candidate, occ_keys, scorer=fuzz.WRatio,
                               limit=1, score_cutoff=score_cutoff)
        if hits:
            label, score, _ = hits[0]
            return occ_index[label], f"fuzzy_{source}", float(score)

    return None, "unresolved", 0.0


# ──────────────────────────────────────────────────────────────
# Occupation-guided skill coverage  (Fix #1 continued)
# ──────────────────────────────────────────────────────────────

def build_occupation_skill_map(
    relations_csv: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """Build occupation_uri → {essential: set(skill_uri), optional: set(skill_uri)}."""
    df = pd.read_csv(relations_csv)
    occ_skills: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: {"essential": set(), "optional": set()}
    )
    for _, row in df.iterrows():
        occ_uri = row.get("occupationUri")
        skill_uri = row.get("skillUri")
        rel_type = str(row.get("relationType", "")).lower().strip()
        if pd.isna(occ_uri) or pd.isna(skill_uri):
            continue
        bucket = "essential" if rel_type == "essential" else "optional"
        occ_skills[str(occ_uri).strip()][bucket].add(str(skill_uri).strip())
    logger.info(f"Occupation-skill map: {len(occ_skills)} occupations")
    return dict(occ_skills)


def compute_occupation_coverage(
    linked_skill_uris: Set[str],
    occupation_uuid: Optional[str],
    occ_skill_map: Dict[str, Dict[str, Set[str]]],
) -> Dict:
    """Compute essential/optional coverage for a resolved occupation."""
    if not occupation_uuid or occupation_uuid not in occ_skill_map:
        return {
            "occupation_uri": occupation_uuid,
            "essential_coverage": None,
            "optional_coverage": None,
            "essential_matched": 0,
            "essential_total": 0,
            "optional_matched": 0,
            "optional_total": 0,
        }

    profile = occ_skill_map[occupation_uuid]
    essential = profile["essential"]
    optional = profile["optional"]

    ess_matched = linked_skill_uris & essential
    opt_matched = linked_skill_uris & optional

    return {
        "occupation_uri": occupation_uuid,
        "essential_coverage": len(ess_matched) / len(essential) if essential else None,
        "optional_coverage": len(opt_matched) / len(optional) if optional else None,
        "essential_matched": len(ess_matched),
        "essential_total": len(essential),
        "optional_matched": len(opt_matched),
        "optional_total": len(optional),
    }


# ──────────────────────────────────────────────────────────────
# Quality tier assignment  (Fix #6)
# ──────────────────────────────────────────────────────────────

def assign_quality_tier(
    resume_skill_uris: Set[str],
    job_skill_uris: Set[str],
    resume_skills_raw: List[str],
    job_skills_raw: List[str],
    occ_coverage: Dict,
) -> str:
    """Assign a quality tier for downstream filtering/weighting.

    Tiers:
        A - Both sides have linked URIs AND occupation coverage available
        B - Both sides have linked URIs but no occupation coverage
        C - One side has linked URIs
        D - Neither side has linked URIs (but raw skills exist)
        F - Missing raw skills entirely
    """
    has_resume_uris = len(resume_skill_uris) > 0
    has_job_uris = len(job_skill_uris) > 0
    has_occ = occ_coverage.get("essential_coverage") is not None

    if not resume_skills_raw and not job_skills_raw:
        return "F"
    if has_resume_uris and has_job_uris:
        return "A" if has_occ else "B"
    if has_resume_uris or has_job_uris:
        return "C"
    return "D"


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Re-enrich ESCO v3")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--esco-dir", default="dataset/esco", help="ESCO CSV directory")
    parser.add_argument("--fuzzy-cutoff", type=int, default=86, help="Skill fuzzy score cutoff")
    parser.add_argument("--occ-fuzzy-cutoff", type=int, default=82, help="Occupation fuzzy score cutoff")
    args = parser.parse_args()

    esco = Path(args.esco_dir)

    # Build indices
    logger.info("Building skill linker (with token inverted index)...")
    skill_linker = SkillLinker(str(esco / "skills_en.csv"), score_cutoff=args.fuzzy_cutoff)

    logger.info("Building occupation label index...")
    occ_index, occ_keys = build_occupation_index(str(esco / "occupations_en.csv"))

    logger.info("Building occupation-skill relation map...")
    occ_skill_map = build_occupation_skill_map(str(esco / "occupationSkillRelations_en.csv"))

    # Process records
    input_path = Path(args.input)
    output_path = Path(args.output)

    stats = defaultdict(int)
    occ_resolve_stats = defaultdict(int)

    logger.info(f"Processing {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            rec = json.loads(line)

            # ── Extract raw data ──
            raw_job = rec.get("_raw", {}).get("job", {})
            raw_resume = rec.get("_raw", {}).get("resume", {})
            resume_skills_raw = rec.get("resume_skills", []) or []
            job_skills_raw = rec.get("job_skills", []) or []
            job_title = rec.get("job_title", "") or ""
            raw_occ_uri = raw_job.get("occupation_uri", "") or ""

            # ── Fix #5: Clean skills ──
            resume_skills_cleaned = clean_skill_list(resume_skills_raw)
            job_skills_cleaned = clean_skill_list(job_skills_raw)

            # ── Link skills to ESCO ──
            r_uris, r_details, r_unmapped = skill_linker.link(resume_skills_cleaned)
            j_uris, j_details, j_unmapped = skill_linker.link(job_skills_cleaned)

            # ── Fix #1: Resolve occupation URI ──
            resolved_occ, occ_mode, occ_score = resolve_occupation_uri(
                raw_occ_uri, job_title, occ_index, occ_keys,
                score_cutoff=args.occ_fuzzy_cutoff)
            occ_resolve_stats[occ_mode] += 1

            # ── Fix #1 continued: Occupation-guided coverage ──
            all_linked = r_uris | j_uris
            occ_coverage = compute_occupation_coverage(all_linked, resolved_occ, occ_skill_map)

            # ── Simple skill overlap coverage (as before) ──
            if j_uris:
                skill_overlap = len(r_uris & j_uris) / len(j_uris)
            else:
                skill_overlap = 0.0

            # ── Fix #6: Quality tier ──
            tier = assign_quality_tier(r_uris, j_uris, resume_skills_raw, job_skills_raw, occ_coverage)
            stats[f"tier_{tier}"] += 1

            # ── Build enrichment block ──
            enrichment = {
                "version": "v3",
                "params": {
                    "fuzzy_score_cutoff": args.fuzzy_cutoff,
                    "occ_fuzzy_cutoff": args.occ_fuzzy_cutoff,
                },
                "occupation": {
                    "raw_uri": raw_occ_uri or None,
                    "resolved_uri": resolved_occ,
                    "match_mode": occ_mode,
                    "match_score": occ_score,
                    **occ_coverage,
                },
                "resume_skill_uris": sorted(r_uris),
                "job_skill_uris": sorted(j_uris),
                "resume_skills_cleaned": resume_skills_cleaned,
                "job_skills_cleaned": job_skills_cleaned,
                "resume_unmapped_skills": r_unmapped,
                "job_unmapped_skills": j_unmapped,
                "scores": {
                    "skill_overlap": round(skill_overlap, 4),
                    "essential_coverage": occ_coverage["essential_coverage"],
                    "optional_coverage": occ_coverage["optional_coverage"],
                },
                "quality_tier": tier,
            }

            rec["esco_enrichment_v3"] = enrichment
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if line_num % 1000 == 0:
                logger.info(f"  processed {line_num} records...")

            stats["total"] += 1

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info(f"Total records: {stats['total']}")
    logger.info(f"Quality tiers: A={stats.get('tier_A',0)} B={stats.get('tier_B',0)} "
                f"C={stats.get('tier_C',0)} D={stats.get('tier_D',0)} F={stats.get('tier_F',0)}")
    logger.info("Occupation resolution breakdown:")
    for mode, count in sorted(occ_resolve_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {mode}: {count}")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
