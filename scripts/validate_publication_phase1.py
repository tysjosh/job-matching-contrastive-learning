#!/usr/bin/env python3
"""Independent QA for the fixed-validation Phase-1 publication experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {
    "trials": {
        "split_dir": ROOT / "preprocess" / "trec_ct_splits",
        "lc_dir": ROOT / "preprocess" / "trec_ct_lc",
        "pair_fields": ("topic_id", "nct_id"),
        "expected_train": 4408,
        "expected_val_positive": 1162,
    },
    "go_ppi": {
        "split_dir": ROOT / "preprocess" / "go_ppi_splits",
        "lc_dir": ROOT / "preprocess" / "go_ppi_lc",
        "pair_fields": ("protein_id", "partner_id"),
        "expected_train": 15708,
        "expected_val_positive": 321,
    },
}
FRACTIONS = [10, 25, 50, 75, 100]
SEEDS = [42, 13, 21]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lines(path: Path) -> List[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def finite(values: Iterable[Any]) -> List[float]:
    output = []
    for value in values:
        if value is not None and math.isfinite(float(value)):
            output.append(float(value))
    return output


def profile_jsonl(path: Path, pair_fields: Tuple[str, str]) -> Dict[str, Any]:
    query_field, candidate_field = pair_fields
    queries = set()
    pairs = Counter()
    grades = Counter()
    invalid = 0
    label_mismatches = 0
    expected_labels = {
        2: {"good_fit", "established"},
        1: {"potential_fit", "weak_evidence"},
        0: {"no_fit", "no_interaction", "not_relevant"},
    }
    for line in read_lines(path):
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        query = metadata.get(query_field) or metadata.get("resume_id")
        candidate = metadata.get(candidate_field)
        if candidate is None:
            candidate = (row.get("job") or {}).get(candidate_field)
        grade = (row.get("job") or {}).get("grade")
        label = metadata.get("original_label")
        if query is None or candidate is None or grade not in (0, 1, 2):
            invalid += 1
            continue
        if label is not None and label not in expected_labels[int(grade)]:
            label_mismatches += 1
        queries.add(str(query))
        pairs[(str(query), str(candidate))] += 1
        grades[int(grade)] += 1
    duplicate_rows = sum(count - 1 for count in pairs.values() if count > 1)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": sum(grades.values()) + invalid,
        "valid_rows": sum(grades.values()),
        "invalid_rows": invalid,
        "label_mismatches": label_mismatches,
        "grade_counts": {str(key): grades[key] for key in (0, 1, 2)},
        "queries": len(queries),
        "pair_duplicates": duplicate_rows,
        "query_ids": queries,
        "pair_ids": set(pairs),
    }


def check(condition: bool, message: str, failures: List[str]) -> None:
    if not condition:
        failures.append(message)


def audit_data(domain: str, spec: Dict[str, Any], failures: List[str]) -> Dict[str, Any]:
    profiles = {
        split: profile_jsonl(spec["split_dir"] / f"{split}.jsonl",
                             spec["pair_fields"])
        for split in ("train", "validation", "test")
    }
    for split, profile in profiles.items():
        check(profile["invalid_rows"] == 0,
              f"{domain} {split}: invalid rows={profile['invalid_rows']}", failures)
        check(profile["label_mismatches"] == 0,
              f"{domain} {split}: label mismatches={profile['label_mismatches']}",
              failures)
        check(profile["pair_duplicates"] == 0,
              f"{domain} {split}: duplicate pairs={profile['pair_duplicates']}",
              failures)
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        query_overlap = profiles[left]["query_ids"] & profiles[right]["query_ids"]
        pair_overlap = profiles[left]["pair_ids"] & profiles[right]["pair_ids"]
        check(not query_overlap,
              f"{domain}: {left}/{right} query overlap={len(query_overlap)}", failures)
        check(not pair_overlap,
              f"{domain}: {left}/{right} pair overlap={len(pair_overlap)}", failures)

    manifest = json.loads((spec["lc_dir"] / "fraction_manifest.json").read_text())
    source_lines = read_lines(spec["split_dir"] / "train.jsonl")
    check(len(source_lines) == spec["expected_train"],
          f"{domain}: unexpected source train count={len(source_lines)}", failures)
    check(manifest["source_train_sha256"] == sha256(spec["split_dir"] / "train.jsonl"),
          f"{domain}: source train hash differs from fraction manifest", failures)
    previous: List[str] = []
    fraction_counts = {}
    for fraction in FRACTIONS:
        lines = read_lines(spec["lc_dir"] / f"frac_{fraction}" / "train.jsonl")
        fraction_counts[str(fraction)] = len(lines)
        check(all((json.loads(line).get("job") or {}).get("grade") == 2
                  for line in lines),
              f"{domain} f{fraction}: non-grade-2 training record", failures)
        if previous:
            check(lines[:len(previous)] == previous,
                  f"{domain} f{fraction}: fraction is not a nested prefix", failures)
        previous = lines
        check(len(lines) == int(manifest["fractions"][str(fraction)]),
              f"{domain} f{fraction}: count differs from manifest", failures)
    check(previous == source_lines,
          f"{domain}: 100% fraction differs from canonical train split", failures)

    val_positive = read_lines(spec["lc_dir"] / "validation_positive.jsonl")
    check(len(val_positive) == spec["expected_val_positive"],
          f"{domain}: unexpected positive validation count={len(val_positive)}",
          failures)
    check(all((json.loads(line).get("job") or {}).get("grade") == 2
              for line in val_positive),
          f"{domain}: training-objective validation includes non-grade-2 rows",
          failures)

    cleaned = {}
    for split, profile in profiles.items():
        cleaned[split] = {key: value for key, value in profile.items()
                          if key not in ("query_ids", "pair_ids")}
    return {
        "splits": cleaned,
        "cross_split_query_overlap": 0,
        "cross_split_pair_overlap": 0,
        "fraction_counts": fraction_counts,
        "fractions_nested": True,
        "positive_validation_rows": len(val_positive),
    }


def audit_results(domain: str, results_dir: Path,
                  failures: List[str]) -> Dict[str, Any]:
    path = results_dir / f"{domain}_fusion_learning_curve.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    check(payload["max_records_per_split"] == 0,
          f"{domain}: result is not full-split evaluation", failures)
    check(not payload["missing_checkpoints"],
          f"{domain}: missing checkpoints recorded", failures)
    expected = {(fraction, seed) for fraction in FRACTIONS for seed in SEEDS}
    observed = {(int(run["fraction"]), int(run["seed"]))
                for run in payload["runs"]}
    check(observed == expected,
          f"{domain}: incomplete or unexpected result grid", failures)
    check(len(payload["runs"]) == len(observed),
          f"{domain}: duplicate result runs", failures)

    max_selection_error = 0.0
    max_delta_error = 0.0
    max_macro_error = 0.0
    ontology_stats = set()
    positive_delta_counts = Counter()
    for run in payload["runs"]:
        checkpoint = Path(run["checkpoint"])
        check(checkpoint.exists(), f"{domain}: missing {checkpoint}", failures)
        curve = {float(key): float(value)
                 for key, value in run["validation_selection_curve"].items()}
        selected = float(run["selected_weight"])
        max_value = max(curve.values())
        max_selection_error = max(max_selection_error,
                                  abs(curve[selected] - max_value))
        ontology = run["validation_stats"]["ontology"]
        ontology_stats.add((round(float(ontology["mean"]), 12),
                            round(float(ontology["sd"]), 12)))
        check(float(run["validation_stats"]["text"]["sd"]) > 0,
              f"{domain}: non-positive validation text SD", failures)
        check(float(ontology["sd"]) > 0,
              f"{domain}: non-positive validation ontology SD", failures)

        text_result = run["test"]["text"]
        fusion_result = run["test"]["fusion"]
        deltas = run["test"]["delta"]
        for metric in ("hard", "easy", "middle_vs_easy", "pooled"):
            text_value = float(text_result["contrasts"]["global_auc"][metric])
            fusion_value = float(fusion_result["contrasts"]["global_auc"][metric])
            stored = float(deltas["global_auc"][metric])
            max_delta_error = max(max_delta_error,
                                  abs(stored - (fusion_value - text_value)))
            check(0 <= text_value <= 1 and 0 <= fusion_value <= 1,
                  f"{domain}: AUC outside [0,1]", failures)
            if fusion_value > text_value:
                positive_delta_counts[f"global_auc_{metric}"] += 1
        for metric in ("ndcg@10", "precision@10", "recall@10",
                       "average_precision", "reciprocal_rank"):
            text_value = float(text_result["ranking"]["macro"][metric])
            fusion_value = float(fusion_result["ranking"]["macro"][metric])
            stored = float(deltas["ranking_macro"][metric])
            max_delta_error = max(max_delta_error,
                                  abs(stored - (fusion_value - text_value)))
            check(0 <= text_value <= 1 and 0 <= fusion_value <= 1,
                  f"{domain}: ranking metric outside [0,1]", failures)
            if fusion_value > text_value:
                positive_delta_counts[metric] += 1
            for result, stored_macro in (
                    (text_result, text_value), (fusion_result, fusion_value)):
                values = finite(row[metric]
                                for row in result["ranking"]["per_query"])
                recomputed = float(np.mean(values))
                max_macro_error = max(max_macro_error,
                                      abs(recomputed - stored_macro))

    check(max_selection_error < 1e-12,
          f"{domain}: selected fusion weight is not validation-optimal", failures)
    check(max_delta_error < 1e-12,
          f"{domain}: stored delta arithmetic mismatch", failures)
    check(max_macro_error < 1e-12,
          f"{domain}: per-query macro recomputation mismatch", failures)
    check(len(ontology_stats) == 1,
          f"{domain}: ontology normalization stats vary by checkpoint", failures)

    # Independently reconcile every summary CSV row against the seed-level JSON.
    summary_path = results_dir / "tables" / f"{domain}_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    raw_path = results_dir / "tables" / f"{domain}_seed_metrics.csv"
    with raw_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    max_summary_error = 0.0
    for row in summary_rows:
        fraction, mode, metric = int(row["fraction"]), row["mode"], row["metric"]
        if mode in ("text", "fusion"):
            values = [float(raw["value"]) for raw in raw_rows
                      if int(raw["fraction"]) == fraction
                      and raw["mode"] == mode and raw["metric"] == metric]
        else:
            by_seed = {}
            for raw in raw_rows:
                if int(raw["fraction"]) == fraction and raw["metric"] == metric:
                    by_seed.setdefault(int(raw["seed"]), {})[raw["mode"]] = float(raw["value"])
            values = [value["fusion"] - value["text"]
                      for value in by_seed.values()
                      if "fusion" in value and "text" in value]
        if not values:
            continue
        expected_mean = float(np.mean(values))
        expected_sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
        max_summary_error = max(max_summary_error,
                                abs(float(row["mean"]) - expected_mean))
        if expected_sd is not None:
            max_summary_error = max(max_summary_error,
                                    abs(float(row["sd"]) - expected_sd))
    check(max_summary_error < 1e-12,
          f"{domain}: summary mean/SD reconciliation mismatch", failures)
    return {
        "result_path": str(path),
        "result_sha256": sha256(path),
        "runs": len(payload["runs"]),
        "selected_weight_max_error": max_selection_error,
        "delta_max_error": max_delta_error,
        "macro_max_error": max_macro_error,
        "summary_max_error": max_summary_error,
        "positive_delta_run_counts": dict(positive_delta_counts),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results" / "publication_phase1_fixedval")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    failures: List[str] = []
    data = {domain: audit_data(domain, spec, failures)
            for domain, spec in DOMAINS.items()}
    results = {domain: audit_results(domain, results_dir, failures)
               for domain in DOMAINS}
    report = {
        "assessment": "ready_to_share" if not failures else "needs_revision",
        "failures": failures,
        "checks": {
            "data": data,
            "results": results,
        },
        "required_caveats": [
            "Only three training seeds are available; mean±SD and paired t intervals quantify training-seed variability with low degrees of freedom.",
            "The same fixed test split is reused across seeds, so seeds are not independent samples of test-population uncertainty.",
            "Fusion weights optimize validation pooled AUC; ranking metrics are secondary outcomes at that AUC-selected operating point.",
            "Observed differences support predictive complementarity, not a causal claim about ontology information.",
        ],
    }
    output = args.output or (results_dir / "validation_report.json")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"assessment": report["assessment"],
                      "failures": failures,
                      "output": str(output)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
