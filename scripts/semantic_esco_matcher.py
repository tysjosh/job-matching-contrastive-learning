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
# Vendor / product name → ESCO URI direct mapping
# ──────────────────────────────────────────────────────────────
# These are common tech skills that have no ESCO preferred/alt label.
# Each maps to the closest ESCO concept URI.

VENDOR_SKILL_ALIASES: Dict[str, Tuple[str, str]] = {
    # Cloud platforms → cloud technologies
    "aws":                ("http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0", "cloud technologies"),
    "amazon web services":("http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0", "cloud technologies"),
    "azure":              ("http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0", "cloud technologies"),
    "gcp":                ("http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0", "cloud technologies"),
    "google cloud":       ("http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0", "cloud technologies"),
    "google cloud platform":("http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0", "cloud technologies"),
    # Cloud services
    "aws lambda":         ("http://data.europa.eu/esco/skill/6b643893-0a1f-4f6c-83a1-e7eef75849b9", "develop with cloud services"),
    "s3":                 ("http://data.europa.eu/esco/skill/d3286405-49f8-4e8a-8046-a4376b4e7963", "manage cloud data and storage"),
    "ec2":                ("http://data.europa.eu/esco/skill/6195c5f7-a4fb-425d-a3dd-c4467c4471a3", "deploy cloud resource"),
    "dynamo db":          ("http://data.europa.eu/esco/skill/d3286405-49f8-4e8a-8046-a4376b4e7963", "manage cloud data and storage"),
    "dynamodb":           ("http://data.europa.eu/esco/skill/d3286405-49f8-4e8a-8046-a4376b4e7963", "manage cloud data and storage"),
    "sqs":                ("http://data.europa.eu/esco/skill/6b643893-0a1f-4f6c-83a1-e7eef75849b9", "develop with cloud services"),
    "azure data factory": ("http://data.europa.eu/esco/skill/6b643893-0a1f-4f6c-83a1-e7eef75849b9", "develop with cloud services"),
    # Containers / virtualisation
    "docker":             ("http://data.europa.eu/esco/skill/ae4f0cc6-e0b9-47f5-bdca-2fc2e6316dce", "manage ICT virtualisation environments"),
    "kubernetes":         ("http://data.europa.eu/esco/skill/ae4f0cc6-e0b9-47f5-bdca-2fc2e6316dce", "manage ICT virtualisation environments"),
    "k8s":                ("http://data.europa.eu/esco/skill/ae4f0cc6-e0b9-47f5-bdca-2fc2e6316dce", "manage ICT virtualisation environments"),
    # CI/CD / DevOps tools
    "terraform":          ("http://data.europa.eu/esco/skill/ce8ae6ca-61d8-4174-b457-641de96cbff4", "automate cloud tasks"),
    "ansible":            ("http://data.europa.eu/esco/skill/ce8ae6ca-61d8-4174-b457-641de96cbff4", "automate cloud tasks"),
    "jenkins":            ("http://data.europa.eu/esco/skill/f47a1998-0beb-43be-9f46-380aa4d183da", "Jenkins (tools for software configuration management)"),
    "jira":               ("http://data.europa.eu/esco/skill/bec4359e-cb92-468f-a997-8fb28e32fba9", "ICT project management methodologies"),
    "git":                ("http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f", "tools for software configuration management"),
    "github":             ("http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f", "tools for software configuration management"),
    "gitlab":             ("http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f", "tools for software configuration management"),
    "bitbucket":          ("http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f", "tools for software configuration management"),
    # Agile / Scrum
    "scrum":              ("http://data.europa.eu/esco/skill/0a9acb6b-1139-4be9-b431-3a80a959f2f4", "Agile project management"),
    "kanban":             ("http://data.europa.eu/esco/skill/0a9acb6b-1139-4be9-b431-3a80a959f2f4", "Agile project management"),
    "agile":              ("http://data.europa.eu/esco/skill/dba46f87-0831-49cd-a1c7-340a653c0221", "Agile development"),
    # Data / BI tools
    "tableau":            ("http://data.europa.eu/esco/skill/65e58886-bd1e-4c5b-8ca5-8d9b353c8aa1", "data visualisation software"),
    "power bi":           ("http://data.europa.eu/esco/skill/143769cb-b61e-47d8-a61e-eedfbec1016c", "business intelligence"),
    "snowflake":          ("http://data.europa.eu/esco/skill/ab1e97ed-2319-4293-a8b7-072d2648822f", "database management systems"),
    "redis":              ("http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c", "NoSQL"),
    "mongodb":            ("http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c", "NoSQL"),
    "mongo db":           ("http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c", "NoSQL"),
    "elasticsearch":      ("http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c", "NoSQL"),
    "cassandra":          ("http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c", "NoSQL"),
    "couchdb":            ("http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c", "NoSQL"),
    # Web frameworks (no ESCO label) → JavaScript Framework
    "react":              ("http://data.europa.eu/esco/skill/9b9de2a4-d8af-4a7b-933a-a8334ae60067", "JavaScript Framework"),
    "vue":                ("http://data.europa.eu/esco/skill/9b9de2a4-d8af-4a7b-933a-a8334ae60067", "JavaScript Framework"),
    "vue.js":             ("http://data.europa.eu/esco/skill/9b9de2a4-d8af-4a7b-933a-a8334ae60067", "JavaScript Framework"),
    "jquery":             ("http://data.europa.eu/esco/skill/9b9de2a4-d8af-4a7b-933a-a8334ae60067", "JavaScript Framework"),
    "j query":            ("http://data.europa.eu/esco/skill/9b9de2a4-d8af-4a7b-933a-a8334ae60067", "JavaScript Framework"),
    "redux":              ("http://data.europa.eu/esco/skill/9b9de2a4-d8af-4a7b-933a-a8334ae60067", "JavaScript Framework"),
    "next.js":            ("http://data.europa.eu/esco/skill/9b9de2a4-d8af-4a7b-933a-a8334ae60067", "JavaScript Framework"),
    "express.js":         ("http://data.europa.eu/esco/skill/9b9de2a4-d8af-4a7b-933a-a8334ae60067", "JavaScript Framework"),
    "node.js":            ("http://data.europa.eu/esco/skill/3cd569a2-4f88-4c1e-9995-8dce8c5e51a7", "JavaScript"),
    # Java ecosystem
    "hibernate":          ("http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389", "Java (computer programming)"),
    "spring boot":        ("http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389", "Java (computer programming)"),
    "spring framework":   ("http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389", "Java (computer programming)"),
    "spring":             ("http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389", "Java (computer programming)"),
    "maven":              ("http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389", "Java (computer programming)"),
    "gradle":             ("http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389", "Java (computer programming)"),
    # Python ecosystem
    "django":             ("http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d", "Python (computer programming)"),
    "flask":              ("http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d", "Python (computer programming)"),
    "pandas":             ("http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d", "Python (computer programming)"),
    "numpy":              ("http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d", "Python (computer programming)"),
    "scikit-learn":       ("http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d", "Python (computer programming)"),
    "pytorch":            ("http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d", "Python (computer programming)"),
    "tensorflow":         ("http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d", "Python (computer programming)"),
    # .NET / C#
    "c#":                 ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    ".net":               ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    "asp.net":            ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    "vb.net":             ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    "kotlin":             ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    "go":                 ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    "golang":             ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    "rust":               ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    # Markup / data formats
    "html5":              ("http://data.europa.eu/esco/skill/0af062de-eb43-41e9-9b96-249e2cd22d26", "use markup languages"),
    "html":               ("http://data.europa.eu/esco/skill/0af062de-eb43-41e9-9b96-249e2cd22d26", "use markup languages"),
    "xml":                ("http://data.europa.eu/esco/skill/0af062de-eb43-41e9-9b96-249e2cd22d26", "use markup languages"),
    "json":               ("http://data.europa.eu/esco/skill/0af062de-eb43-41e9-9b96-249e2cd22d26", "use markup languages"),
    "css3":               ("http://data.europa.eu/esco/skill/c2999f0c-eb37-4cdf-b9b0-82107b628794", "style sheet languages"),
    "sass":               ("http://data.europa.eu/esco/skill/c2999f0c-eb37-4cdf-b9b0-82107b628794", "style sheet languages"),
    "less":               ("http://data.europa.eu/esco/skill/c2999f0c-eb37-4cdf-b9b0-82107b628794", "style sheet languages"),
    # Database
    "t-sql":              ("http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742", "SQL"),
    "pl/sql":             ("http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742", "SQL"),
    "mysql":              ("http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742", "SQL"),
    "my sql":             ("http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742", "SQL"),
    "postgresql":         ("http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742", "SQL"),
    "oracle":             ("http://data.europa.eu/esco/skill/ab1e97ed-2319-4293-a8b7-072d2648822f", "database management systems"),
    "ssrs":               ("http://data.europa.eu/esco/skill/143769cb-b61e-47d8-a61e-eedfbec1016c", "business intelligence"),
    "ssis":               ("http://data.europa.eu/esco/skill/ab1e97ed-2319-4293-a8b7-072d2648822f", "database management systems"),
    # OS / systems
    "unix":               ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    "linux":              ("http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97", "computer programming"),
    # Office tools
    "excel":              ("http://data.europa.eu/esco/skill/1973c966-f236-40c9-b2d4-5d71a89019be", "use spreadsheets software"),
    "word":               ("http://data.europa.eu/esco/skill/81633a44-f1db-4a01-a940-804c6905e330", "use word processing software"),
    "powerpoint":         ("http://data.europa.eu/esco/skill/234aeb8d-56c3-4531-9193-1c5e6a8d16cb", "use presentation software"),
    "power point":        ("http://data.europa.eu/esco/skill/234aeb8d-56c3-4531-9193-1c5e6a8d16cb", "use presentation software"),
    "outlook":            ("http://data.europa.eu/esco/skill/fdf15b35-9028-4acb-bfd2-43f00a015cd9", "use personal organization software"),
    "sharepoint":         ("http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f", "tools for software configuration management"),
    "share point":        ("http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f", "tools for software configuration management"),
    "microsoft office":   ("http://data.europa.eu/esco/skill/1973c966-f236-40c9-b2d4-5d71a89019be", "use spreadsheets software"),
    "microsoft office suite":("http://data.europa.eu/esco/skill/1973c966-f236-40c9-b2d4-5d71a89019be", "use spreadsheets software"),
    # Testing
    "selenium":           ("http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f", "tools for software configuration management"),
    "selenium web driver":("http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f", "tools for software configuration management"),
    "cucumber bdd":       ("http://data.europa.eu/esco/skill/dba46f87-0831-49cd-a1c7-340a653c0221", "Agile development"),
    # Accounting / business
    "quick books":        ("http://data.europa.eu/esco/skill/1973c966-f236-40c9-b2d4-5d71a89019be", "use spreadsheets software"),
    "quickbooks":         ("http://data.europa.eu/esco/skill/1973c966-f236-40c9-b2d4-5d71a89019be", "use spreadsheets software"),
    "sap":                ("http://data.europa.eu/esco/skill/ab1e97ed-2319-4293-a8b7-072d2648822f", "database management systems"),
    "gaap":               ("http://data.europa.eu/esco/skill/52e53619-fa77-4f72-b237-5e4aae784dc2", "financial management"),
    # Statistical tools
    "spss":               ("http://data.europa.eu/esco/skill/04f1b938-d4d4-4cb1-a863-982af76b9d93", "SAS language"),
    "stata":              ("http://data.europa.eu/esco/skill/04f1b938-d4d4-4cb1-a863-982af76b9d93", "SAS language"),
    # Data entry / admin
    "data entry":         ("http://data.europa.eu/esco/skill/88564d2d-2dc9-4789-864a-f9cb73f389ec", "maintain data entry requirements"),
    # Project management
    "project management": ("http://data.europa.eu/esco/skill/237db40b-4600-47c0-837f-4a2c4f3014ab", "project management principles"),
}


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

        # Vendor alias shortcut
        if q in VENDOR_SKILL_ALIASES:
            uri, pref = VENDOR_SKILL_ALIASES[q]
            return [{"uri": uri, "preferred_label": pref,
                     "matched_label": q, "score": 1.0, "mode": "vendor_alias"}]

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
            elif q_lower in VENDOR_SKILL_ALIASES:
                uri, pref = VENDOR_SKILL_ALIASES[q_lower]
                results[i] = [{"uri": uri, "preferred_label": pref,
                               "matched_label": q_lower, "score": 1.0,
                               "mode": "vendor_alias"}]
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

                # Re-match previously unmapped skills:
                # 1. Try vendor alias map first (instant, no model needed)
                # 2. Fall back to semantic matching for the rest
                if old_unmapped:
                    alias_remaining = []
                    for skill in old_unmapped:
                        skill_lower = skill.lower().strip()
                        if skill_lower in VENDOR_SKILL_ALIASES:
                            uri, pref = VENDOR_SKILL_ALIASES[skill_lower]
                            new_uris.add(uri)
                            new_details.append({
                                "input": skill,
                                "mode": "vendor_alias",
                                "matches": [{"uri": uri, "preferred_label": pref,
                                             "matched_label": skill_lower,
                                             "score": 1.0, "mode": "vendor_alias"}],
                            })
                        else:
                            alias_remaining.append(skill)

                    # Semantic matching for skills not in alias map
                    if alias_remaining:
                        matches = esco_index.match_batch(
                            alias_remaining, st_model, threshold=args.threshold)
                        for skill, skill_matches in zip(alias_remaining, matches):
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
