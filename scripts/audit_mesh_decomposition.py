#!/usr/bin/env python3
"""Audit source-specific MeSH facets on TREC 2021 validation topics.

This is a read-only diagnostic apart from its JSON report. It does not train,
change negative pools, or use the 2022 test split for experiment selection.
Scores are computed only where both topic and trial have the given facet;
coverage is reported separately so selective missingness remains visible.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    ordered = sorted([(x, 1) for x in positive] + [(x, 0) for x in negative])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = (i + 1 + j) / 2
        rank_sum += rank * sum(label for _, label in ordered[i:j])
        i = j
    n, m = len(positive), len(negative)
    return (rank_sum - n * (n + 1) / 2) / (n * m)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def exact_bma(a: set[str], b: set[str]) -> float:
    overlap = len(a & b)
    return 0.5 * (overlap / len(a) + overlap / len(b))


def audit(split: Path, topics_path: Path, trials_path: Path,
          mesh_cache: Path, seed: int, bootstrap: int) -> dict:
    from trials_domain.mesh_ontology import MeshIndex, MeshMatcher

    index = MeshIndex.load(mesh_cache)
    matcher = MeshMatcher(index)
    topic_rows, trial_rows = list(rows(topics_path)), list(rows(trials_path))
    topics = {row["topic_id"]: row for row in topic_rows}
    trials = {row["nct_id"]: row for row in trial_rows}
    if len(topics) != len(topic_rows) or len(trials) != len(trial_rows):
        raise ValueError("duplicate topic or trial identifier in converted sources")
    totals = Counter()
    coverage = Counter()
    missing = Counter()
    seen = set()
    by_facet = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    @lru_cache(None)
    def ancestor_paths(ui: str) -> frozenset[str]:
        paths = set()
        for tree in index.trees(ui):
            components = tree.split(".")
            # Exclude the letter-only root, shared by virtually every disease.
            paths.update(".".join(components[:n])
                         for n in range(2, len(components) + 1))
        return frozenset(paths)

    for row in rows(split):
        topic_id = row["resume"]["topic_id"]
        nct_id = row["job"]["nct_id"]
        grade = int(row["job"]["grade"])
        pair = (topic_id, nct_id)
        if pair in seen:
            raise ValueError(f"duplicate topic-trial pair: {pair}")
        seen.add(pair)
        totals[grade] += 1
        topic, trial = topics.get(topic_id), trials.get(nct_id)
        if topic is None or trial is None:
            missing["topic" if topic is None else "trial"] += 1
            continue
        for facet, field in (("disease", "condition_uris"),
                             ("intervention", "intervention_uris"),
                             ("all", "mesh_uris")):
            a, b = set(topic[field]), set(trial[field])
            if not a or not b:
                continue
            coverage[(facet, grade)] += 1
            ap = set().union(*(ancestor_paths(ui) for ui in a))
            bp = set().union(*(ancestor_paths(ui) for ui in b))
            scores = {
                "exact_jaccard": jaccard(a, b),
                "exact_bma": exact_bma(a, b),
                "ancestor_jaccard": jaccard(ap, bp),
                "hierarchy_bma": matcher.ontology_set_similarity(sorted(a), sorted(b)),
            }
            for mode, score in scores.items():
                by_facet[(facet, mode)][topic_id][grade].append(score)

    result = {"split": str(split), "source_topics": str(topics_path),
              "source_trials": str(trials_path), "mesh_cache": str(mesh_cache),
              "totals_by_grade": {str(g): totals[g] for g in (2, 1, 0)},
              "missing_joins": dict(missing),
              "source_counts": {"topics": len(topics), "trials": len(trials)},
              "source_anatomy_coverage": {
                  "topics": sum(any("A" in index.branches(ui)
                                    for ui in row["mesh_uris"])
                                for row in topic_rows),
                  "trials": sum(any("A" in index.branches(ui)
                                    for ui in row["mesh_uris"])
                                for row in trial_rows),
              }, "facets": {}}
    rng = random.Random(seed)
    for facet in ("disease", "intervention", "all"):
        out = {"paired_coverage": {
            str(g): {"count": coverage[(facet, g)],
                     "fraction": coverage[(facet, g)] / totals[g]}
            for g in (2, 1, 0)}, "scores": {}}
        for mode in ("exact_jaccard", "exact_bma", "ancestor_jaccard",
                     "hierarchy_bma"):
            by_topic = by_facet[(facet, mode)]
            contrasts = {}
            for higher, lower in ((2, 1), (2, 0), (1, 0)):
                values = [auc(grades[higher], grades[lower])
                          for grades in by_topic.values()
                          if grades[higher] and grades[lower]]
                values = [v for v in values if v is not None]
                if not values:
                    continue
                estimates = sorted(statistics.mean(rng.choices(values, k=len(values)))
                                   for _ in range(bootstrap)) if bootstrap else []
                contrasts[f"{higher}>{lower}"] = {
                    "topics": len(values),
                    "macro_auc": statistics.mean(values),
                    "topic_bootstrap_ci95": (
                        [estimates[int(0.025 * bootstrap)],
                         estimates[min(bootstrap - 1, int(0.975 * bootstrap))]]
                        if estimates else None),
                }
            out["scores"][mode] = contrasts
        paired_deltas = {}
        for higher, lower in ((2, 1), (2, 0), (1, 0)):
            differences = []
            exact_topics = by_facet[(facet, "exact_bma")]
            hierarchy_topics = by_facet[(facet, "hierarchy_bma")]
            for topic_id in exact_topics:
                a, b = exact_topics[topic_id], hierarchy_topics[topic_id]
                if a[higher] and a[lower] and b[higher] and b[lower]:
                    differences.append(auc(b[higher], b[lower]) -
                                       auc(a[higher], a[lower]))
            if not differences:
                continue
            estimates = sorted(statistics.mean(rng.choices(differences,
                                                            k=len(differences)))
                               for _ in range(bootstrap)) if bootstrap else []
            paired_deltas[f"{higher}>{lower}"] = {
                "topics": len(differences),
                "mean_auc_delta": statistics.mean(differences),
                "topic_bootstrap_ci95": (
                    [estimates[int(0.025 * bootstrap)],
                     estimates[min(bootstrap - 1, int(0.975 * bootstrap))]]
                    if estimates else None),
            }
        out["hierarchy_minus_exact_bma"] = paired_deltas
        result["facets"][facet] = out
    result["method_note"] = (
        "All scores and topic-bootstrap intervals use only pairs with both facet "
        "sides annotated. These descriptive AUCs are not trained-model results. "
        "Exact BMA and hierarchy BMA share the same set aggregator, isolating "
        "term identity versus MeSH tree-distance similarity."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", type=Path,
                        default=ROOT / "preprocess/trec_ct_splits/validation.jsonl")
    parser.add_argument("--topics", type=Path,
                        default=ROOT / "preprocess/trec_ct/topics.jsonl")
    parser.add_argument("--trials", type=Path,
                        default=ROOT / "preprocess/trec_ct/trials.jsonl")
    parser.add_argument("--mesh-cache", type=Path,
                        default=ROOT / "embedding_cache/mesh2021_index.pkl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/ontology_ceiling/mesh_decomposition_audit_validation.json")
    parser.add_argument("--seed", type=int, default=412)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = audit(args.split, args.topics, args.trials, args.mesh_cache,
                   args.seed, args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for facet, data in result["facets"].items():
        print(facet)
        print("  coverage:", {g: round(v["fraction"], 3)
                               for g, v in data["paired_coverage"].items()})
        for mode, contrasts in data["scores"].items():
            print(" ", mode, {name: round(row["macro_auc"], 3)
                               for name, row in contrasts.items()})
    print("wrote", args.output)


if __name__ == "__main__":
    main()
