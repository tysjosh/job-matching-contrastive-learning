#!/usr/bin/env python3
"""Evaluate paired negative-selection learning curves on untouched test splits.

The two arms at each (fraction, seed) use byte-identical training data and differ
only in how graded negatives are selected.  Frozen sentence embeddings are
prepared once per split and reused for every checkpoint, making a full paired
grid much faster than invoking the legacy evaluator once per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.evaluate_publication_fusion as publication_eval
from contrastive_learning.data_structures import TrainingConfig


DOMAINS: Dict[str, Dict[str, Any]] = {
    "go_ppi": {
        "control": "randwin",
        "treatment": "ontneg_stoch",
        "config": ROOT / "config" / "lc_go_ppi_randwin.json",
        "validation": ROOT / "preprocess" / "go_ppi_splits" / "validation.jsonl",
        "test": ROOT / "preprocess" / "go_ppi_splits" / "test.jsonl",
        "hard_name": "established_vs_weak_evidence",
        "easy_name": "established_vs_no_interaction",
    },
    "trials": {
        "control": "baseline",
        "treatment": "ontneg",
        "config": ROOT / "config" / "lc_trials_baseline.json",
        "validation": ROOT / "preprocess" / "trec_ct_splits" / "validation.jsonl",
        "test": ROOT / "preprocess" / "trec_ct_splits" / "test.jsonl",
        "hard_name": "eligible_vs_ineligible",
        "easy_name": "eligible_vs_not_relevant",
    },
}


class NullMatcher:
    """Skip ontology-score computation; this experiment evaluates text models."""

    @staticmethod
    def ontology_set_similarity(_left, _right) -> float:
        return 0.0


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def metric_value(result: Dict[str, Any], metric: str) -> Optional[float]:
    if metric.startswith("global_auc."):
        key = metric.split(".", 1)[1]
        return result["contrasts"]["global_auc"].get(key)
    if metric.startswith("macro_query_auc."):
        key = metric.split(".", 1)[1]
        return result["contrasts"]["macro_query_auc"].get(key)
    if metric.startswith("ranking."):
        key = metric.split(".", 1)[1]
        return result["ranking"]["macro"].get(key)
    if metric.startswith("ordinal."):
        key = metric.split(".", 1)[1]
        return result["ordinal"].get(key)
    if metric.startswith("binary."):
        key = metric.split(".", 1)[1]
        return result["binary_classification"].get(key)
    if metric.startswith("ordinal_classification."):
        key = metric.split(".", 1)[1]
        return result["ordinal_classification"].get(key)
    raise ValueError(metric)


METRICS = (
    "global_auc.hard",
    "global_auc.easy",
    "global_auc.middle_vs_easy",
    "global_auc.pooled",
    "macro_query_auc.hard",
    "macro_query_auc.easy",
    "macro_query_auc.pooled",
    "ranking.ndcg@10",
    "ranking.precision@10",
    "ranking.recall@10",
    "ranking.average_precision",
    "ranking.reciprocal_rank",
    "ordinal.kendall_tau_b",
    "ordinal.spearman_rho",
    "binary.f1",
    "ordinal_classification.macro_f1",
)


def finite(values: Iterable[Optional[float]]) -> list[float]:
    return [float(value) for value in values
            if value is not None and math.isfinite(float(value))]


def summarize(runs: list[Dict[str, Any]], fractions: list[int],
              seeds: list[int], control: str, treatment: str) -> Dict[str, Any]:
    by_key = {(row["fraction"], row["seed"], row["arm"]): row for row in runs}
    summary: Dict[str, Any] = {}
    for fraction in fractions:
        fraction_summary: Dict[str, Any] = {}
        for metric in METRICS:
            control_values = []
            treatment_values = []
            deltas = []
            paired_seeds = []
            for seed in seeds:
                c = by_key.get((fraction, seed, control))
                t = by_key.get((fraction, seed, treatment))
                if c is None or t is None:
                    continue
                cv = metric_value(c["test"], metric)
                tv = metric_value(t["test"], metric)
                if cv is None or tv is None:
                    continue
                control_values.append(float(cv))
                treatment_values.append(float(tv))
                deltas.append(float(tv) - float(cv))
                paired_seeds.append(seed)
            cvals, tvals, dvals = (finite(control_values), finite(treatment_values),
                                    finite(deltas))
            fraction_summary[metric] = {
                "paired_seeds": paired_seeds,
                "n": len(dvals),
                "control_mean": float(np.mean(cvals)) if cvals else None,
                "control_sd": float(np.std(cvals, ddof=1)) if len(cvals) > 1 else None,
                "treatment_mean": float(np.mean(tvals)) if tvals else None,
                "treatment_sd": float(np.std(tvals, ddof=1)) if len(tvals) > 1 else None,
                "delta_mean": float(np.mean(dvals)) if dvals else None,
                "delta_sd": float(np.std(dvals, ddof=1)) if len(dvals) > 1 else None,
                "wins": int(sum(value > 0 for value in dvals)),
                "deltas_by_seed": dict(zip(map(str, paired_seeds), dvals)),
            }
        summary[str(fraction)] = fraction_summary
    return summary


def render_markdown(output: Dict[str, Any]) -> str:
    lines = [
        f"# {output['domain']} negative-selection evaluation",
        "",
        (f"Treatment is `{output['treatment_arm']}`; control is "
         f"`{output['control_arm']}`. Values are mean ± sample SD across paired "
         f"seeds. Delta = treatment − control."),
        "",
        "| Train fraction | Hard AUC control | Hard AUC ontology | Δ hard AUC | "
        "Pooled AUC control | Pooled AUC ontology | Δ pooled AUC | Hard wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(mean, sd) -> str:
        if mean is None:
            return "—"
        return f"{mean:.4f} ± {sd:.4f}" if sd is not None else f"{mean:.4f}"

    for fraction in output["fractions"]:
        hard = output["summary"][str(fraction)]["global_auc.hard"]
        pooled = output["summary"][str(fraction)]["global_auc.pooled"]
        lines.append(
            f"| {fraction}% | {cell(hard['control_mean'], hard['control_sd'])} | "
            f"{cell(hard['treatment_mean'], hard['treatment_sd'])} | "
            f"{cell(hard['delta_mean'], hard['delta_sd'])} | "
            f"{cell(pooled['control_mean'], pooled['control_sd'])} | "
            f"{cell(pooled['treatment_mean'], pooled['treatment_sd'])} | "
            f"{cell(pooled['delta_mean'], pooled['delta_sd'])} | "
            f"{hard['wins']}/{hard['n']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results" / "negative_selection_fixedval")
    parser.add_argument("--control-arm", default=None,
                        help="override the domain's default control arm")
    parser.add_argument("--treatment-arm", default=None,
                        help="override the domain's default ontology arm")
    parser.add_argument("--fractions", nargs="+", type=int,
                        default=[5, 10, 15, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 21])
    parser.add_argument("--prepared-cache-dir", type=Path,
                        default=ROOT / "results" / "publication_phase1_fixedval"
                                     / "prepared_embeddings")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    spec = DOMAINS[args.domain]
    control = args.control_arm or spec["control"]
    treatment = args.treatment_arm or spec["treatment"]
    arms = (control, treatment)
    grid = []
    for fraction in args.fractions:
        for seed in args.seeds:
            for arm in arms:
                run_dir = args.results_dir / f"{args.domain}_{arm}_f{fraction}_s{seed}"
                grid.append({
                    "fraction": fraction,
                    "seed": seed,
                    "arm": arm,
                    "run_dir": run_dir,
                    "checkpoint": run_dir / "best_checkpoint.pt",
                })
    missing = [row for row in grid if not row["checkpoint"].exists()]
    if args.list:
        for row in grid:
            state = "ready" if row["checkpoint"].exists() else "missing"
            print(f"{args.domain} {row['arm']} f{row['fraction']} s{row['seed']}: {state}")
        print(f"ready={len(grid) - len(missing)} missing={len(missing)} total={len(grid)}")
        return 0
    if missing and not args.allow_missing:
        raise SystemExit("missing checkpoints:\n" + "\n".join(
            str(row["checkpoint"]) for row in missing))
    grid = [row for row in grid if row["checkpoint"].exists()]
    if not grid:
        raise SystemExit("no checkpoints available")

    config = TrainingConfig.from_json(str(spec["config"]))
    cache_dir = args.prepared_cache_dir / args.domain
    frozen_cache = publication_eval.load_frozen_embedding_cache(config)
    validation = publication_eval.prepare_split(
        spec["validation"], config, NullMatcher(), "resume_id",
        max_records=args.max_records, frozen_cache=frozen_cache,
        prepared_cache_dir=cache_dir)
    test = publication_eval.prepare_split(
        spec["test"], config, NullMatcher(), "resume_id",
        max_records=args.max_records, frozen_cache=frozen_cache,
        prepared_cache_dir=cache_dir)

    output: Dict[str, Any] = {
        "domain": args.domain,
        "control_arm": control,
        "treatment_arm": treatment,
        "contrast": "ontology-guided negative selection minus matched control",
        "fractions": args.fractions,
        "seeds": args.seeds,
        "hard_contrast": spec["hard_name"],
        "easy_contrast": spec["easy_name"],
        "validation_negative_policy": {
            "fixed": True,
            "epoch": 14,
            "seed": 1729,
        },
        "splits": {
            "validation": {
                "path": str(spec["validation"]),
                "sha256": publication_eval.file_sha256(spec["validation"]),
                "records": validation["records_scored"],
            },
            "test": {
                "path": str(spec["test"]),
                "sha256": publication_eval.file_sha256(spec["test"]),
                "records": test["records_scored"],
            },
        },
        "missing_checkpoints": [str(row["checkpoint"]) for row in missing],
        "runs": [],
    }

    for index, row in enumerate(grid, 1):
        print(f"[{index}/{len(grid)}] {row['arm']} f{row['fraction']} s{row['seed']}",
              flush=True)
        validation_scores = publication_eval.text_scores(
            validation, row["checkpoint"], config)
        test_scores = publication_eval.text_scores(test, row["checkpoint"], config)
        binary_threshold = publication_eval.fit_binary_threshold(
            validation_scores, validation["grades"])
        ordinal_thresholds = publication_eval.fit_ordinal_thresholds(
            validation_scores, validation["grades"])
        test_result = publication_eval.evaluate_score(
            test_scores, test, binary_threshold, ordinal_thresholds)
        output["runs"].append({
            "fraction": row["fraction"],
            "seed": row["seed"],
            "arm": row["arm"],
            "checkpoint": str(row["checkpoint"]),
            "test": test_result,
        })
        auc = test_result["contrasts"]["global_auc"]
        print(f"  hard={auc['hard']:.4f} pooled={auc['pooled']:.4f}", flush=True)

    output["summary"] = summarize(
        output["runs"], args.fractions, args.seeds,
        control, treatment)
    output_path = args.output or (args.results_dir / f"{args.domain}_test_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(output), indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(render_markdown(json_safe(output)), encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
