#!/usr/bin/env python3
"""Aggregate publication Phase-1 fusion evaluations across training seeds.

The input is produced by ``evaluate_publication_fusion.py``.  The script writes
one tidy row per fraction/seed/scoring mode/metric, one summary row per
fraction/scoring mode/metric, and a compact Markdown table of the primary
metrics.  Uncertainty is deliberately computed across training seeds: the same
held-out test examples are shared by all seeds and are not independent
experimental replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


METRICS: Dict[str, Tuple[str, ...]] = {
    "global_auc_hard": ("contrasts", "global_auc", "hard"),
    "global_auc_easy": ("contrasts", "global_auc", "easy"),
    "global_auc_middle_vs_easy": (
        "contrasts", "global_auc", "middle_vs_easy"),
    "global_auc_pooled": ("contrasts", "global_auc", "pooled"),
    "macro_query_auc_hard": ("contrasts", "macro_query_auc", "hard"),
    "macro_query_auc_easy": ("contrasts", "macro_query_auc", "easy"),
    "macro_query_auc_middle_vs_easy": (
        "contrasts", "macro_query_auc", "middle_vs_easy"),
    "macro_query_auc_pooled": ("contrasts", "macro_query_auc", "pooled"),
    "ndcg@10": ("ranking", "macro", "ndcg@10"),
    "precision@10": ("ranking", "macro", "precision@10"),
    "recall@10": ("ranking", "macro", "recall@10"),
    "map": ("ranking", "macro", "average_precision"),
    "mrr": ("ranking", "macro", "reciprocal_rank"),
    "kendall_tau_b": ("ordinal", "kendall_tau_b"),
    "spearman_rho": ("ordinal", "spearman_rho"),
    "binary_f1": ("binary_classification", "f1"),
    "ordinal_macro_f1": ("ordinal_classification", "macro_f1"),
    "ordinal_mae": ("ordinal_classification", "ordinal_mae"),
}

PRIMARY = [
    "global_auc_hard",
    "global_auc_pooled",
    "ndcg@10",
    "precision@10",
    "recall@10",
    "map",
    "mrr",
]


def nested_get(value: Dict[str, Any], path: Sequence[str]) -> Any:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray([float(value) for value in values if finite(value)], dtype=float)
    n = len(array)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None,
                "ci95_high": None}
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if n > 1 else None
    if n > 1 and sd is not None:
        half_width = float(stats.t.ppf(0.975, df=n - 1) * sd / math.sqrt(n))
        low, high = mean - half_width, mean + half_width
    else:
        low = high = None
    return {"n": n, "mean": mean, "sd": sd,
            "ci95_low": low, "ci95_high": high}


def fmt(summary: Dict[str, Any], signed: bool = False) -> str:
    if summary["mean"] is None:
        return "—"
    prefix = "+" if signed else ""
    if summary["sd"] is None:
        return f"{summary['mean']:{prefix}.4f}"
    return f"{summary['mean']:{prefix}.4f} ± {summary['sd']:.4f}"


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(payload: Dict[str, Any], source: Path, output_dir: Path) -> None:
    domain = str(payload["domain"])
    raw_rows: List[Dict[str, Any]] = []
    values: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)

    seen = set()
    for run in payload.get("runs", []):
        fraction, seed = int(run["fraction"]), int(run["seed"])
        key = (fraction, seed)
        if key in seen:
            raise ValueError(f"duplicate run for fraction={fraction}, seed={seed}")
        seen.add(key)
        for mode in ("text", "fusion"):
            result = run["test"][mode]
            for metric, path in METRICS.items():
                value = nested_get(result, path)
                if not finite(value):
                    continue
                number = float(value)
                raw_rows.append({
                    "domain": domain, "fraction": fraction, "seed": seed,
                    "mode": mode, "metric": metric, "value": number,
                    "selected_weight": run.get("selected_weight"),
                })
                values[(fraction, mode, metric)].append(number)

    fractions = sorted(int(value) for value in payload.get(
        "fractions", {key[0] for key in seen}))
    seeds = sorted(int(value) for value in payload.get(
        "seeds", {key[1] for key in seen}))
    expected = {(fraction, seed) for fraction in fractions for seed in seeds}
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"incomplete fraction/seed grid: {missing}")
    unexpected = sorted(seen - expected)
    if unexpected:
        raise ValueError(f"runs outside declared fraction/seed grid: {unexpected}")

    summary_rows: List[Dict[str, Any]] = []
    summary_index: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for fraction in fractions:
        for metric in METRICS:
            for mode in ("text", "fusion"):
                result = summarize(values[(fraction, mode, metric)])
                row = {"domain": domain, "fraction": fraction, "mode": mode,
                       "metric": metric, **result}
                summary_rows.append(row)
                summary_index[(fraction, mode, metric)] = result
            paired = []
            for seed in seeds:
                text_value = next((row["value"] for row in raw_rows
                                   if row["fraction"] == fraction
                                   and row["seed"] == seed
                                   and row["mode"] == "text"
                                   and row["metric"] == metric), None)
                fusion_value = next((row["value"] for row in raw_rows
                                     if row["fraction"] == fraction
                                     and row["seed"] == seed
                                     and row["mode"] == "fusion"
                                     and row["metric"] == metric), None)
                if text_value is not None and fusion_value is not None:
                    paired.append(fusion_value - text_value)
            result = summarize(paired)
            row = {"domain": domain, "fraction": fraction, "mode": "delta",
                   "metric": metric, **result}
            summary_rows.append(row)
            summary_index[(fraction, "delta", metric)] = result

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{domain}_seed_metrics.csv"
    summary_path = output_dir / f"{domain}_summary.csv"
    markdown_path = output_dir / f"{domain}_primary_table.md"
    write_csv(raw_path, raw_rows,
              ["domain", "fraction", "seed", "mode", "metric", "value",
               "selected_weight"])
    write_csv(summary_path, summary_rows,
              ["domain", "fraction", "mode", "metric", "n", "mean", "sd",
               "ci95_low", "ci95_high"])

    lines = [
        f"# {domain} Phase-1 score-fusion learning curve",
        "",
        f"Source: `{source}`",
        "",
        "Values are mean ± sample SD across training seeds. Delta is fusion − text.",
        "The CSV also includes a paired 95% t interval across seeds (n=3).",
        "",
        "| Fraction | Metric | Text | Fusion | Delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for fraction in fractions:
        for metric in PRIMARY:
            lines.append(
                f"| {fraction}% | {metric} | "
                f"{fmt(summary_index[(fraction, 'text', metric)])} | "
                f"{fmt(summary_index[(fraction, 'fusion', metric)])} | "
                f"{fmt(summary_index[(fraction, 'delta', metric)], signed=True)} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {markdown_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/publication_phase1/tables"))
    args = parser.parse_args(argv)
    for source in args.inputs:
        payload = json.loads(source.read_text(encoding="utf-8"))
        aggregate(payload, source.resolve(), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
