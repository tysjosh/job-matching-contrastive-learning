#!/usr/bin/env python3
"""End-to-end CVE experiment runner (feature: cve-vulnerability-ranking).

Runs the experiment matrix in ``experiments.json`` **once**, resumably:

    convert (shared) -> split (shared per strategy) -> [per experiment]
        Stage 1 contrastive pretrain (skipped for the E0 base-embeddings baseline)
        -> Stage 2 supervised heads -> predict on test -> evaluate

and writes a per-experiment ``evaluation_report.json`` plus a cross-experiment
``summary.json`` / ``summary.md`` comparing the real-label metrics (NDCG, MAP,
in_kev accuracy/macro-F1, priority_band accuracy/macro-F1, band-separation ratio).

Design notes
------------
* **Shared, cached preprocessing.** The CSV+profiles are converted to
  ``CVE_View_Records`` once; each split strategy is materialized once and reused
  by every experiment that declares it. Re-running skips any step whose output
  already exists (use ``--force`` to redo).
* **Isolation.** All artifacts live under ``--output-root`` (default
  ``cve_domain/runs/experiments``), separate from career-domain outputs.
* **Honest evaluation.** Metrics are computed by ``CVEEvaluationReporter`` against
  the ground-truth supervised labels only (no circular ontology metric).

Examples
--------
Run everything (heavy — full data, encoder, all Stage 1/2 training)::

    .venv/bin/python -m cve_domain.experiments.run_experiments

Quick wiring smoke on a small slice, few epochs::

    .venv/bin/python -m cve_domain.experiments.run_experiments \
        --limit 2000 --epochs-stage1 1 --epochs-stage2 1 --experiments E0,E1

Just prepare data + print the resolved plan (no training)::

    .venv/bin/python -m cve_domain.experiments.run_experiments --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Repo root importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cve_domain.data_converter import CVEConvertConfig, CVEDataConverter
from cve_domain.data_splitter import CVEDataSplitter
from cve_domain.evaluation_reporter import CVEEvaluationReporter
from cve_domain.run_config import CVERunConfig

logger = logging.getLogger("cve.experiments")

_MANIFEST_PATH = Path(__file__).resolve().parent / "experiments.json"

# Split proportions shared by all experiments (mirrors the example configs).
_SPLIT_PROPORTIONS = {"train": 80, "validation": 10, "test": 10}


# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #
def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_config(base: Dict[str, Any], overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged


# --------------------------------------------------------------------------- #
# Shared preprocessing (convert + split), cached
# --------------------------------------------------------------------------- #
def prepare_view_records(base_stage1: Dict[str, Any], shared_dir: Path,
                         limit: Optional[int], force: bool) -> Path:
    """Convert the CSV + profiles to CVE_View_Records once (cached)."""
    out_path = shared_dir / "view_records.jsonl"
    if out_path.exists() and not force and limit is None:
        logger.info("Reusing existing view records: %s", out_path)
        return out_path

    converter = CVEDataConverter(CVEConvertConfig(domain_adapter="cve"))
    full_path = shared_dir / "view_records_full.jsonl"
    if not full_path.exists() or force:
        logger.info("Converting CSV + profiles -> %s", full_path)
        report = converter.convert(
            base_stage1["cve_csv_path"],
            base_stage1["cve_profiles_path"],
            str(full_path),
        )
        logger.info("Conversion: %d emitted / %d input rows", report.emitted_records, report.input_rows)

    if limit is not None:
        logger.info("Truncating to first %d view records (smoke mode)", limit)
        records = []
        with open(full_path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= limit:
                    break
                if line.strip():
                    records.append(line.rstrip("\n"))
        out_path.write_text("\n".join(records) + "\n", encoding="utf-8")
        return out_path

    return full_path


def prepare_split(view_records_path: Path, strategy: str, seed: int,
                  shared_dir: Path, force: bool) -> Dict[str, Path]:
    """Materialize one split (train/validation/test) for a strategy, cached."""
    split_dir = shared_dir / f"split_{strategy}_seed{seed}"
    train_path = split_dir / "train.jsonl"
    if train_path.exists() and not force:
        logger.info("Reusing existing %s split: %s", strategy, split_dir)
    else:
        logger.info("Splitting (%s, seed=%d) -> %s", strategy, seed, split_dir)
        splitter = CVEDataSplitter(strategy=strategy, proportions=_SPLIT_PROPORTIONS, seed=seed)
        report = splitter.split_dataset(str(view_records_path), str(split_dir))
        if report.status != "ok":
            raise RuntimeError(f"Split failed ({strategy}): {report.status} — {report.reason}")
        logger.info("Split counts: %s", report.per_split_counts)
    return {
        "train": split_dir / "train.jsonl",
        "validation": split_dir / "validation.jsonl",
        "test": split_dir / "test.jsonl",
    }


# --------------------------------------------------------------------------- #
# Per-experiment execution
# --------------------------------------------------------------------------- #
def run_stage1(config_dict: Dict[str, Any], exp_dir: Path, split: Dict[str, Path],
               view_records_path: Path, epochs: Optional[int], force: bool) -> Path:
    """Run Stage 1 contrastive pretraining; return the best-checkpoint path."""
    from cve_domain.stage1 import Stage1ContrastivePretrainer  # torch-heavy, lazy

    stage1_dir = exp_dir / "stage1"
    ckpt = stage1_dir / "best_checkpoint.pt"
    marker = stage1_dir / ".stage1_complete"
    # A checkpoint alone is NOT proof of completion: the trainer writes
    # best_checkpoint.pt every time validation improves, so an interrupted run
    # (e.g. killed at epoch 4/10) also leaves one. Only the completion marker,
    # written after all epochs finish, means "done". Checkpoint-but-no-marker =
    # a partial run, which we retrain from scratch rather than silently proceed on.
    if ckpt.exists() and marker.exists() and not force:
        logger.info("[%s] Reusing completed Stage 1 checkpoint: %s", exp_dir.name, ckpt)
        return ckpt
    if ckpt.exists() and not marker.exists() and not force:
        logger.warning(
            "[%s] Found a Stage 1 checkpoint with no completion marker (partial/"
            "interrupted run) — retraining Stage 1 from scratch.", exp_dir.name)

    cfg = dict(config_dict)
    if epochs is not None:
        cfg["num_epochs"] = epochs
    # Isolate the preloaded-embedding cache per experiment, inside the run root,
    # so experiments never share/overwrite the base config's cache file.
    cfg["embedding_cache_path"] = str(stage1_dir / "embedding_cache" / "text_embeddings.pt")
    config = CVERunConfig.from_dict(cfg)

    pretrainer = Stage1ContrastivePretrainer(
        config=config,
        output_dir=str(stage1_dir),
        denominator_pools_path=cfg["cve_denominator_pools_path"],
        cyber_kg_path=cfg.get("cyber_kg_path"),
    )
    pretrainer.run(
        train_path=str(split["train"]),
        validation_path=str(split["validation"]),
        full_records_path=str(view_records_path),
    )
    if not ckpt.exists():
        raise RuntimeError(f"[{exp_dir.name}] Stage 1 did not produce {ckpt}")
    marker.write_text("ok\n", encoding="utf-8")  # mark Stage 1 fully complete
    return ckpt


def run_stage2(config_dict: Dict[str, Any], exp_dir: Path, split: Dict[str, Path],
               stage1_ckpt: Optional[Path], base_embeddings: bool,
               epochs: Optional[int], force: bool):
    """Run (or resume) Stage 2, returning a ready-to-predict trainer."""
    from cve_domain.stage2 import CVEStage2Trainer, STAGE2_BEST_CHECKPOINT  # lazy

    stage2_dir = exp_dir / "stage2"
    cfg = dict(config_dict)
    if epochs is not None:
        cfg["num_epochs"] = epochs
    # Stage 2 reuses TrainingConfig fields; build via CVERunConfig for validation.
    if base_embeddings:
        cfg.setdefault("cve_denominator_pools_path", "unused")  # not needed for base heads
    config = CVERunConfig.from_dict(cfg)

    trainer = CVEStage2Trainer(
        config=config,
        output_dir=str(stage2_dir),
        stage1_checkpoint_path=None if base_embeddings else str(stage1_ckpt),
        base_embeddings=base_embeddings,
    )

    ckpt = stage2_dir / STAGE2_BEST_CHECKPOINT
    marker = stage2_dir / ".stage2_complete"
    # Same completion-marker logic as Stage 1: a checkpoint without the marker is
    # a partial/interrupted run, so retrain rather than resume on it.
    if ckpt.exists() and marker.exists() and not force:
        logger.info("[%s] Reusing completed Stage 2 checkpoint; loading heads for inference.",
                    exp_dir.name)
        trainer.load_heads_from_checkpoint(ckpt)
    else:
        if ckpt.exists() and not marker.exists() and not force:
            logger.warning(
                "[%s] Found a Stage 2 checkpoint with no completion marker (partial/"
                "interrupted run) — retraining Stage 2 from scratch.", exp_dir.name)
        trainer.train(train_path=str(split["train"]), validation_path=str(split["validation"]))
        marker.write_text("ok\n", encoding="utf-8")  # mark Stage 2 fully complete
    return trainer


def evaluate_experiment(trainer, exp_dir: Path, test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Predict on the test split and compute the evaluation report."""
    predictions = trainer.predict_records(test_records)
    reporter = CVEEvaluationReporter()
    report = reporter.evaluate(test_records, predictions, output_dir=str(exp_dir / "eval"))
    return report.to_dict()


def _summary_row(exp_id: str, name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the headline metrics from an evaluation report for the summary."""
    ranking = report.get("ranking", {})
    cls = report.get("classification", {})
    kev = cls.get("in_kev", {})
    band = cls.get("priority_band", {})
    band_score = cls.get("priority_band_from_score", {})
    sep = report.get("embedding_separation", {})
    return {
        "id": exp_id,
        "name": name,
        "status": report.get("status"),
        "ndcg": None if ranking.get("skipped") else ranking.get("ndcg"),
        "map": None if ranking.get("skipped") else ranking.get("map"),
        "in_kev_acc": None if kev.get("skipped") else kev.get("accuracy"),
        "in_kev_macro_f1": None if kev.get("skipped") else kev.get("macro_f1"),
        "band_acc": None if band.get("skipped") else band.get("accuracy"),
        "band_macro_f1": None if band.get("skipped") else band.get("macro_f1"),
        # Ordinal band derived from the predicted priority_score (non-degenerate).
        "band_from_score_macro_f1": None if band_score.get("skipped") else band_score.get("macro_f1"),
        "band_from_score_acc": None if band_score.get("skipped") else band_score.get("accuracy"),
        "band_separation_ratio": None if sep.get("skipped") else sep.get("separation_ratio"),
        "skipped_metrics": list(report.get("skipped_metrics", {}).keys()),
    }


def _write_summary(rows: List[Dict[str, Any]], output_root: Path) -> None:
    _write_json(output_root / "summary.json", rows)

    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    headers = ["id", "name", "ndcg", "map", "in_kev_acc", "in_kev_macro_f1",
               "band_acc", "band_macro_f1", "band_from_score_macro_f1",
               "band_separation_ratio"]
    lines = ["# CVE experiment summary", "",
             "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    lines.append("")
    lines.append("Metrics are computed against ground-truth supervised labels only "
                 "(no circular ontology evaluation). '-' = metric skipped (see each "
                 "experiment's eval/evaluation_report.json for the skip reason).")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote summary -> %s", output_root / "summary.md")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CVE experiment matrix end-to-end.")
    parser.add_argument("--manifest", default=str(_MANIFEST_PATH), help="Path to experiments.json")
    parser.add_argument("--output-root", default="cve_domain/runs/experiments",
                        help="Root dir for all experiment artifacts")
    parser.add_argument("--experiments", default="", help="Comma list of experiment ids (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N view records (quick smoke)")
    parser.add_argument("--epochs-stage1", type=int, default=None, help="Override Stage 1 epochs")
    parser.add_argument("--epochs-stage2", type=int, default=None, help="Override Stage 2 epochs")
    parser.add_argument("--force", action="store_true", help="Recompute steps even if outputs exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare data + print the resolved plan; do not train")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    manifest = _load_json(args.manifest)
    base_stage1 = _load_json(_REPO_ROOT / manifest["base"]["stage1_config"])
    base_stage2 = _load_json(_REPO_ROOT / manifest["base"]["stage2_config"])

    output_root = Path(args.output_root)
    shared_dir = output_root / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    selected = {e.strip() for e in args.experiments.split(",") if e.strip()}
    experiments = [e for e in manifest["experiments"]
                   if not selected or e["id"] in selected]
    if not experiments:
        raise SystemExit(f"No experiments matched {selected}")

    seed = int(base_stage1.get("split_seed", 42))

    # 1) Shared conversion.
    view_records_path = prepare_view_records(base_stage1, shared_dir, args.limit, args.force)

    # 2) Materialize each needed split once.
    needed_splits = sorted({e["split"] for e in experiments})
    splits: Dict[str, Dict[str, Path]] = {
        strategy: prepare_split(view_records_path, strategy, seed, shared_dir, args.force)
        for strategy in needed_splits
    }

    # Print the resolved plan.
    logger.info("Planned experiments:")
    for e in experiments:
        s1 = _resolve_config(base_stage1, e.get("stage1_overrides")) if not e.get("base_embeddings") else None
        logger.info("  %s (%s) split=%s base_embeddings=%s stage1_overrides=%s",
                    e["id"], e.get("name"), e["split"], bool(e.get("base_embeddings")),
                    e.get("stage1_overrides", {}))

    if args.dry_run:
        logger.info("--dry-run: data prepared, plan printed; skipping training.")
        return

    # 3) Per-experiment run + evaluate.
    summary_rows: List[Dict[str, Any]] = []
    for e in experiments:
        exp_id = e["id"]
        exp_dir = output_root / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        split = splits[e["split"]]
        base_emb = bool(e.get("base_embeddings"))
        logger.info("=== Experiment %s (%s) ===", exp_id, e.get("name"))

        # Fully-completed experiment: reuse its evaluation report and skip all
        # compute, so a resumed run jumps straight to the first unfinished one.
        eval_report_path = exp_dir / "eval" / "evaluation_report.json"
        if eval_report_path.exists() and not args.force:
            logger.info("[%s] Reusing existing evaluation report (experiment complete).", exp_id)
            report = _load_json(eval_report_path)
            summary_rows.append(_summary_row(exp_id, e.get("name", ""), report))
            continue

        try:
            stage1_ckpt: Optional[Path] = None
            if not base_emb:
                s1_cfg = _resolve_config(base_stage1, e.get("stage1_overrides"))
                _write_json(exp_dir / "stage1_config.resolved.json", s1_cfg)
                stage1_ckpt = run_stage1(s1_cfg, exp_dir, split, view_records_path,
                                         args.epochs_stage1, args.force)

            s2_cfg = _resolve_config(base_stage2, e.get("stage2_overrides"))
            _write_json(exp_dir / "stage2_config.resolved.json", s2_cfg)
            trainer = run_stage2(s2_cfg, exp_dir, split, stage1_ckpt, base_emb,
                                 args.epochs_stage2, args.force)

            test_records = _read_jsonl(split["test"])
            report = evaluate_experiment(trainer, exp_dir, test_records)
            summary_rows.append(_summary_row(exp_id, e.get("name", ""), report))
            logger.info("[%s] done: %s", exp_id, _summary_row(exp_id, e.get("name", ""), report))
        except Exception as exc:  # keep going so one failure doesn't sink the batch
            logger.exception("[%s] FAILED: %s", exp_id, exc)
            summary_rows.append({"id": exp_id, "name": e.get("name", ""), "status": f"failed: {exc}"})

    _write_summary(summary_rows, output_root)
    logger.info("All experiments complete. Summary at %s", output_root / "summary.md")


if __name__ == "__main__":
    main()
