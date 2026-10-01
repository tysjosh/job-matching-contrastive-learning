#!/usr/bin/env python3
"""Run the five-seed disease-MeSH exact-versus-hierarchy study.

Seven arms: one equal-width random-window control, plus exact/hierarchy
similarity applied to both negative tiers, ineligible only, or not-relevant
only. The 2021 train/validation split is used for model selection; 2022 test
evaluation is a separate explicit step.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/lc_trials_ontneg_stoch.json"
CONTROL = ROOT / "config/lc_trials_randwin.json"
CONFIG_DIR = ROOT / "config/mesh_disease_decomposition"
RESULTS = ROOT / "results/mesh_disease_decomposition"
MODES = ("exact", "hierarchy")
SCOPES = ("both", "ineligible", "not_relevant")


def arm_name(mode: str, scope: str) -> str:
    return f"disease_{mode}_{scope}"


def arm_config(mode: str, scope: str) -> dict:
    if mode not in MODES or scope not in SCOPES:
        raise ValueError((mode, scope))
    config = {key: value for key, value in json.loads(BASE.read_text()).items()
              if not key.startswith("_")}
    config.update({
        "mesh_similarity_mode": mode,
        "trials_mesh_facet": "disease",
        "trials_mesh_tier_scope": scope,
        "trials_mesh_tiered_negatives": True,
        "trials_tier_sampling": "stochastic",
        "trials_tier_window_frac": .34,
        "loss_type": "infonce",
        "ontology_weight": 0.0,
        "use_ot_distance": False,
    })
    return config


def config_path(mode: str, scope: str) -> Path:
    return CONFIG_DIR / f"{arm_name(mode, scope)}.json"


def prepare() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for scope in SCOPES:
            config_path(mode, scope).write_text(
                json.dumps(arm_config(mode, scope), indent=2) + "\n")
    control = {key: value for key, value in json.loads(CONTROL.read_text()).items()
               if not key.startswith("_")}
    control.update({"mesh_similarity_mode": "hierarchy",
                    "trials_mesh_facet": "disease",
                    "trials_mesh_tier_scope": "both",
                    "loss_type": "infonce"})
    (CONFIG_DIR / "randwin.json").write_text(json.dumps(control, indent=2) + "\n")


def command(config: Path, arm: str, fraction: int, seed: int,
            epochs: int | None = None) -> list[str]:
    lc = ROOT / "preprocess/trec_ct_lc"
    train = lc / f"frac_{fraction}" / "train.jsonl"
    validation = lc / "validation_positive.jsonl"
    if not train.is_file() or not validation.is_file():
        raise FileNotFoundError(f"missing Trials split: {train} or {validation}")
    cmd = [sys.executable, str(ROOT / "scripts/run_learning_curve_point.py"),
           "--domain", "trials", "--config", str(config),
           "--train-file", str(train), "--validation-file", str(validation),
           "--output-dir", str(RESULTS / f"trials_{arm}_f{fraction}_s{seed}"),
           "--fraction", str(fraction / 100), "--seed", str(seed)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    return cmd


def evaluate(fractions: list[int], seeds: list[int], *, test: bool = False,
             selected_arms: list[str] | None = None) -> dict:
    # Validation is for arm selection. The 2022 test is available only through
    # an explicit final-evaluation flag and a declared, fixed arm list.
    from eval_learning_curve import _load_eval_module, evaluate_one

    ev = _load_eval_module()
    dataset = ROOT / "preprocess/trec_ct_splits" / (
        "test.jsonl" if test else "validation.jsonl")
    arms = selected_arms if test else ["randwin"] + [
        arm_name(m, s) for m in MODES for s in SCOPES]
    scores = {}
    for fraction in fractions:
        for seed in seeds:
            for arm in arms:
                run = RESULTS / f"trials_{arm}_f{fraction}_s{seed}"
                checkpoint = run / "best_checkpoint.pt"
                config = run / "training_config.json"
                if not checkpoint.is_file() or not config.is_file():
                    continue
                out = run / ("eval_test2022" if test else "eval_validation2021")
                path = out / "phase1_evaluation_results.json"
                result = (json.loads(path.read_text()) if path.is_file() else
                          evaluate_one(ev, checkpoint, config, dataset, out))
                if result:
                    scores[(fraction, seed, arm)] = result
    summary = {"dataset": str(dataset), "control": "randwin", "arms": {}}
    def metrics(result: dict) -> dict:
        return {"pooled_auc": result["metrics"]["auc_roc"],
                **(result.get("per_grade") or {}).get("pairwise_auc", {})}
    for arm in arms:
        paired = [{"fraction": fraction, "seed": seed,
                   "metrics": metrics(scores[(fraction, seed, arm)]),
                   "control_metrics": metrics(scores[(fraction, seed, "randwin")])}
                  for fraction in fractions for seed in seeds
                  if (fraction, seed, arm) in scores
                  and (fraction, seed, "randwin") in scores]
        for row in paired:
            row["deltas"] = {key: value - row["control_metrics"][key]
                             for key, value in row["metrics"].items()
                             if key in row["control_metrics"]}
        keys = sorted({key for row in paired for key in row["deltas"]})
        summary["arms"][arm] = {
            "n_pairs": len(paired), "paired": paired,
            "mean_deltas": {key: statistics.mean(
                row["deltas"][key] for row in paired if key in row["deltas"])
                for key in keys}}
    summary["exact_vs_hierarchy"] = {}
    for scope in SCOPES:
        exact, hierarchy = arm_name("exact", scope), arm_name("hierarchy", scope)
        contrasts = []
        for fraction in fractions:
            for seed in seeds:
                if (fraction, seed, exact) not in scores or (
                        fraction, seed, hierarchy) not in scores:
                    continue
                a, b = metrics(scores[(fraction, seed, exact)]), metrics(
                    scores[(fraction, seed, hierarchy)])
                contrasts.append({"fraction": fraction, "seed": seed,
                                  "deltas": {key: b[key] - a[key]
                                             for key in a.keys() & b.keys()}})
        keys = sorted({key for row in contrasts for key in row["deltas"]})
        summary["exact_vs_hierarchy"][scope] = {
            "n_pairs": len(contrasts), "paired": contrasts,
            "mean_deltas": {key: statistics.mean(
                row["deltas"][key] for row in contrasts if key in row["deltas"])
                for key in keys}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    filename = "test2022_summary.json" if test else "validation2021_summary.json"
    (RESULTS / filename).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fractions", nargs="+", type=int, default=[100])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[13, 21, 42, 87, 123])
    parser.add_argument("--epochs", type=int, default=None,
                        help="optional training-epoch override for smoke tests")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate finished checkpoints on 2021 validation")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="final evaluation on untouched 2022 test")
    parser.add_argument("--selected-arms", nargs="+",
                        help="required with --evaluate-test: arms fixed on validation")
    args = parser.parse_args(argv)
    all_arms = {"randwin"} | {arm_name(m, s) for m in MODES for s in SCOPES}
    if args.evaluate_test and (not args.selected_arms or
                               "randwin" not in args.selected_arms or
                               set(args.selected_arms) - all_arms):
        parser.error("--evaluate-test requires --selected-arms including randwin "
                     "and only named disease arms, fixed before opening 2022 test")
    if args.prepare or args.execute:
        prepare()
    plan = []
    for fraction in args.fractions:
        for seed in args.seeds:
            plan.append(command(CONFIG_DIR / "randwin.json", "randwin",
                                fraction, seed, args.epochs))
            for mode in MODES:
                for scope in SCOPES:
                    plan.append(command(config_path(mode, scope),
                                        arm_name(mode, scope), fraction, seed,
                                        args.epochs))
    print(f"MeSH disease decomposition: {len(plan)} runs "
          f"({len(args.fractions)} fractions, {len(args.seeds)} seeds, "
          "6 disease arms + random-window control)")
    if not args.execute and not args.evaluate and not args.evaluate_test:
        for cmd in plan:
            print(" ".join(cmd))
        return 0
    if args.execute:
        for cmd in plan:
            out = Path(cmd[cmd.index("--output-dir") + 1])
            if (out / "lc_manifest.json").is_file():
                print(f"[skip] {out.name}: complete", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            print(f"[train] {out.name}", flush=True)
            with (out / "mesh_decomposition.log").open("w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
    if args.evaluate:
        summary = evaluate(args.fractions, args.seeds)
        for arm, row in summary["arms"].items():
            print(arm, row["n_pairs"], row["mean_deltas"])
    if args.evaluate_test:
        summary = evaluate(args.fractions, args.seeds, test=True,
                           selected_arms=args.selected_arms)
        for arm, row in summary["arms"].items():
            print(arm, row["n_pairs"], row["mean_deltas"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
