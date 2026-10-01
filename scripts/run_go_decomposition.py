#!/usr/bin/env python3
"""Run the fixed-split GO aspect x similarity experiment.

All ontology arms use stochastic sampling from the closest 34% of each graded
pool. The paired random-window control has the same pool width and epoch
variety. Protein splits, labels, text embeddings, seeds and training settings
are shared. Configs are materialized for reproducibility before execution.

Examples:
  .venv/bin/python scripts/run_go_decomposition.py
  .venv/bin/python scripts/run_go_decomposition.py --prepare
  .venv/bin/python scripts/run_go_decomposition.py --execute
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config" / "lc_go_ppi_ontneg_stoch.json"
CONTROL = ROOT / "config" / "lc_go_ppi_randwin.json"
CONFIG_DIR = ROOT / "config" / "go_decomposition"
RESULTS = ROOT / "results" / "go_decomposition"
ASPECTS = ("P", "F", "C", "A")
MODES = ("exact", "ancestor", "simgic")
GO_ORDINAL_DEFAULTS = {
    "go_ordinal_lambda_high_weak": 1.0,
    "go_ordinal_lambda_weak_unobserved": 0.1,
    "go_ordinal_margin_high_weak": 0.1,
    "go_ordinal_margin_weak_unobserved": 0.05,
    "go_ordinal_rank_temperature": 0.1,
}


def arm_name(aspect: str, mode: str) -> str:
    return f"go_{aspect}_{mode}"


def arm_config(aspect: str, mode: str, loss_type: str = "infonce") -> dict:
    if aspect not in ASPECTS or mode not in MODES:
        raise ValueError((aspect, mode))
    cfg = json.loads(BASE.read_text(encoding="utf-8"))
    # Remove prose tied to the historical BP-only ceiling number.
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg["go_aspect"] = aspect
    cfg["go_similarity_mode"] = mode
    cfg["go_index_cache"] = "preprocess/go_ppi/go_index_P.pkl"
    cfg["go_aspect_weights"] = {"P": 1.0, "F": 1.0, "C": 1.0}
    cfg["go_ppi_go_tiered_negatives"] = True
    cfg["go_ppi_tier_sampling"] = "stochastic"
    cfg["go_ppi_tier_window_frac"] = 0.34
    cfg["ontology_weight"] = 0.0
    cfg["use_ot_distance"] = False
    cfg["loss_type"] = loss_type
    if loss_type == "go_ordinal":
        cfg.update(GO_ORDINAL_DEFAULTS)
    return cfg


def config_dir(loss_type: str) -> Path:
    return CONFIG_DIR if loss_type == "infonce" else ROOT / "config" / "go_decomposition_go_ordinal"


def results_dir(loss_type: str) -> Path:
    return RESULTS if loss_type == "infonce" else ROOT / "results" / "go_decomposition_go_ordinal"


def config_path(aspect: str, mode: str, loss_type: str = "infonce") -> Path:
    return config_dir(loss_type) / f"{arm_name(aspect, mode)}.json"


def materialize(loss_type: str = "infonce") -> None:
    config_dir(loss_type).mkdir(parents=True, exist_ok=True)
    for aspect in ASPECTS:
        for mode in MODES:
            config_path(aspect, mode, loss_type).write_text(
                json.dumps(arm_config(aspect, mode, loss_type), indent=2) + "\n",
                encoding="utf-8")
    if loss_type == "go_ordinal":
        control = {k: v for k, v in json.loads(CONTROL.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}
        control["loss_type"] = loss_type
        control.update(GO_ORDINAL_DEFAULTS)
        (config_dir(loss_type) / "randwin.json").write_text(
            json.dumps(control, indent=2) + "\n", encoding="utf-8")


def run_command(config: Path, arm: str, fraction: int, seed: int,
                loss_type: str = "infonce") -> list[str]:
    lc = ROOT / "preprocess" / "go_ppi_lc"
    train = lc / f"frac_{fraction}" / "train.jsonl"
    validation = lc / "validation_positive.jsonl"
    if not train.is_file() or not validation.is_file():
        raise FileNotFoundError(f"missing GO split: {train} or {validation}")
    return [sys.executable, str(ROOT / "scripts" / "run_learning_curve_point.py"),
            "--domain", "go_ppi", "--config", str(config),
            "--train-file", str(train), "--validation-file", str(validation),
            "--output-dir", str(results_dir(loss_type) / f"go_ppi_{arm}_f{fraction}_s{seed}"),
            "--fraction", str(fraction / 100), "--seed", str(seed)]


def evaluate(fractions: list[int], seeds: list[int], loss_type: str = "infonce") -> dict:
    """Evaluate the same held-out GO/PPI test set for every completed run."""
    from eval_learning_curve import _load_eval_module, evaluate_one

    ev = _load_eval_module()
    test = ROOT / "preprocess" / "go_ppi_splits" / "test.jsonl"
    arms = ["randwin"] + [arm_name(a, m) for a in ASPECTS for m in MODES]
    scores = {}
    result_root = results_dir(loss_type)
    for fraction in fractions:
        for seed in seeds:
            for arm in arms:
                run = result_root / f"go_ppi_{arm}_f{fraction}_s{seed}"
                ckpt, cfg = run / "best_checkpoint.pt", run / "training_config.json"
                if not ckpt.is_file() or not cfg.is_file():
                    continue
                path = run / "eval_test" / "phase1_evaluation_results.json"
                result = (json.loads(path.read_text(encoding="utf-8"))
                          if path.is_file() else evaluate_one(ev, ckpt, cfg, test,
                                                               run / "eval_test"))
                if result:
                    scores[(fraction, seed, arm)] = float(result["metrics"]["auc_roc"])
    summary = {"dataset": str(test), "control": "randwin",
               "loss_type": loss_type, "arms": {}}
    for arm in arms:
        paired = [
            {"fraction": fraction, "seed": seed,
             "auc": scores[(fraction, seed, arm)],
             "control_auc": scores[(fraction, seed, "randwin")],
             "delta": scores[(fraction, seed, arm)] - scores[(fraction, seed, "randwin")]}
            for fraction in fractions for seed in seeds
            if (fraction, seed, arm) in scores
            and (fraction, seed, "randwin") in scores
        ]
        deltas = [row["delta"] for row in paired]
        summary["arms"][arm] = {
            "n_pairs": len(paired), "paired": paired,
            "mean_auc_delta": statistics.mean(deltas) if deltas else None,
            "delta_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
        }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "test_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fractions", nargs="+", type=int, default=[100])
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42, 87, 123])
    ap.add_argument("--loss-type", choices=("infonce", "go_ordinal"),
                    default="go_ordinal", help="training objective for all arms and control")
    ap.add_argument("--prepare", action="store_true", help="write the 12 arm configs")
    ap.add_argument("--execute", action="store_true", help="train all planned points")
    ap.add_argument("--evaluate", action="store_true",
                    help="score completed checkpoints on the held-out test split")
    args = ap.parse_args(argv)
    if args.prepare or args.execute:
        materialize(args.loss_type)
    plan = []
    for fraction in args.fractions:
        for seed in args.seeds:
            control = (CONTROL if args.loss_type == "infonce" else
                       config_dir(args.loss_type) / "randwin.json")
            plan.append(run_command(control, "randwin", fraction, seed,
                                    args.loss_type))
            for aspect in ASPECTS:
                for mode in MODES:
                    plan.append(run_command(config_path(aspect, mode, args.loss_type),
                                            arm_name(aspect, mode), fraction, seed,
                                            args.loss_type))
    print(f"GO decomposition [{args.loss_type}]: {len(plan)} runs ({len(args.fractions)} fractions, "
          f"{len(args.seeds)} seeds, 12 ontology arms + random-window control)")
    if not args.execute and not args.evaluate:
        for cmd in plan:
            print(" ".join(cmd))
        return 0
    if args.execute:
        for cmd in plan:
            output = Path(cmd[cmd.index("--output-dir") + 1])
            if (output / "lc_manifest.json").is_file():
                print(f"[skip] {output.name}: complete", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            print(f"[train] {output.name}", flush=True)
            with (output / "go_decomposition.log").open("w", encoding="utf-8") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True)
    if args.evaluate:
        summary = evaluate(args.fractions, args.seeds, args.loss_type)
        for arm, row in summary["arms"].items():
            print(f"{arm:18s} n={row['n_pairs']:2d} "
                  f"delta={row['mean_auc_delta']}")
        print(f"wrote {results_dir(args.loss_type) / 'test_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
