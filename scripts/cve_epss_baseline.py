#!/usr/bin/env python3
"""EPSS / percentile ranking baseline for the CVE priority task.

Why this exists
---------------
``priority_score`` is a deterministic formula dominated by EPSS
(``0.42*percentile + 0.28*sqrt(epss) + 0.20*kev + ...``; see
``cybersecurity-vulnerability-ranking/scripts/build_ranking.py``). Ranking CVEs by
their raw EPSS (or EPSS percentile) is therefore NOT an independent competitor — it
is a near-oracle *upper reference* / ceiling for the ranking metrics, because it
ranks by the label's own dominant input.

In the no-leakage setup (EPSS/CVSS/KEV tokens stripped from the encoder input), this
baseline is exactly the informative reference the paper needs: it quantifies how much
of the priority ordering is recoverable from the raw EPSS signal, so the learned
model's NDCG/MAP can be read as "fraction of the EPSS ceiling recovered from the
semantic description alone."

What it does
------------
Loads the shared *test* split (the same records the models were evaluated on), looks
up each CVE's ``epss`` / ``percentile`` from the source CSV, and runs the SAME
``CVEEvaluationReporter`` used for the models — so the NDCG / MAP / Spearman numbers
are computed by identical code on the identical split and are directly comparable.

Note: ``priority_band_from_score`` and ``embedding_separation`` are not meaningful for
a raw-EPSS baseline (no priority_score-scale prediction, no embedding) and should be
ignored in the baseline row; only the ranking block is comparable.

Examples
--------
    .venv/bin/python scripts/cve_epss_baseline.py \
        --test-split cve_domain/runs/experiments/_shared/split_stratified_seed42/test.jsonl \
        --csv cybersecurity-vulnerability-ranking/data/processed/vulnerability_priority_ranking_nvd_all.csv \
        --rank-by both
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cve_domain.evaluation_reporter import CVEEvaluationReporter

_DEFAULT_TEST = "cve_domain/runs/experiments/_shared/split_stratified_seed42/test.jsonl"
_DEFAULT_CSV = "cybersecurity-vulnerability-ranking/data/processed/vulnerability_priority_ranking_nvd_all.csv"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_epss(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Map upper-cased CVE id -> {'epss': float, 'percentile': float}."""
    out: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cve = str(row.get("cve", "")).strip().upper()
            if not cve:
                continue
            try:
                epss = float(row.get("epss") or 0.0)
            except ValueError:
                epss = 0.0
            try:
                percentile = float(row.get("percentile") or 0.0)
            except ValueError:
                percentile = 0.0
            out[cve] = {"epss": epss, "percentile": percentile}
    return out


def _build_predictions(records: List[Dict[str, Any]], epss_by_cve: Dict[str, Dict[str, float]],
                       key: str) -> Dict[str, Dict[str, Any]]:
    """Predictions dict: ranking_score = the chosen raw signal (epss|percentile)."""
    preds: Dict[str, Dict[str, Any]] = {}
    missing = 0
    for rec in records:
        cve = str(rec.get("cve", ""))
        row = epss_by_cve.get(cve.upper())
        if row is None:
            missing += 1
            continue
        preds[cve] = {"ranking_score": row[key]}
    if missing:
        print(f"  [warn] {missing} test CVEs had no EPSS row (ranked implicitly last).")
    return preds


def _print_ranking(label: str, report_dict: Dict[str, Any]) -> None:
    rk = report_dict.get("ranking", {})
    if rk.get("skipped"):
        print(f"\n== {label} ==  ranking SKIPPED: {rk.get('reason')}")
        return
    mab = rk.get("map_at_band_boundaries", {}) or {}
    print(f"\n== {label} ==")
    print(f"  n_ranked            : {rk.get('num_ranked')}")
    print(f"  NDCG (full)         : {rk.get('ndcg'):.4f}")
    ndcg_at = rk.get("ndcg_at_k", {})
    for k, v in ndcg_at.items():
        print(f"  NDCG@{k:<15}: {v:.4f}")
    print(f"  MAP (median thr={rk.get('map_relevance_threshold'):.2f}): {rk.get('map'):.4f}")
    print(f"  Spearman rho        : {rk.get('spearman_rho')}")
    for name in ("medium_plus_45", "high_plus_70", "critical_plus_85"):
        blk = mab.get(name)
        if blk:
            print(f"  MAP@{name:<16}: {blk['map']:.4f}  (n_rel={blk['num_relevant']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="EPSS/percentile ranking baseline (upper reference).")
    ap.add_argument("--test-split", default=_DEFAULT_TEST, help="Shared test split JSONL.")
    ap.add_argument("--csv", default=_DEFAULT_CSV, help="Source CSV with epss/percentile columns.")
    ap.add_argument("--rank-by", choices=["epss", "percentile", "both"], default="both")
    ap.add_argument("--output-dir", default="cve_domain/runs/experiments/_epss_baseline",
                    help="Where to write the baseline evaluation report(s).")
    args = ap.parse_args()

    test_path = Path(args.test_split)
    csv_path = Path(args.csv)
    if not test_path.exists():
        raise SystemExit(f"Test split not found: {test_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    records = _read_jsonl(test_path)
    epss_by_cve = _load_epss(csv_path)
    print(f"Loaded {len(records)} test records; {len(epss_by_cve)} CVEs with EPSS.")

    reporter = CVEEvaluationReporter()
    keys = ["epss", "percentile"] if args.rank_by == "both" else [args.rank_by]
    for key in keys:
        preds = _build_predictions(records, epss_by_cve, key)
        out_dir = Path(args.output_dir) / f"rank_by_{key}"
        report = reporter.evaluate(records, preds, output_dir=str(out_dir))
        _print_ranking(f"EPSS baseline: rank by {key}", report.to_dict())
        print(f"  (report written to {out_dir}/evaluation_report.json)")

    print("\nNOTE: band_from_score / embedding_separation are N/A for a raw-EPSS "
          "baseline; only the ranking block above is comparable to the model rows.")


if __name__ == "__main__":
    main()
