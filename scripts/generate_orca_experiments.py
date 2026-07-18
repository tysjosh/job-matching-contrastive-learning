#!/usr/bin/env python3
"""
Generate the ER (Experiment-Reliability / ORCA) family, mirroring the E-series
layout so ORCA runs sit alongside — and are directly comparable to — the
existing OSCAR / InfoNCE baselines (E4-InfoNCE, E4-OSCAR-Skill).

Each ORCA experiment = base ORCA config (config/orca_denominator_config.json)
+ the variant's ``orca_overrides`` from orca/experiments/experiments.json.
For every (variant x dataset x seed) it writes:

    results/research_runs/{ER-*}__<dataset>__s<seed>/training_config.json

and emits a runnable bash list of train + eval commands:

  * train: run_orca_training.py (the 4-phase OrcaPhaseOrchestrator), which saves
           phase1_pretraining/best_checkpoint.pt (the trained projection head);
  * eval:  run_phase1_embedding_evaluation.py against that checkpoint, producing
           phase1_evaluation/phase1_evaluation_results.json — the SAME artifact
           the OSCAR/InfoNCE baselines produce, so scripts/aggregate_orca_results.py
           can compare them.

Usage:
    python3 scripts/generate_orca_experiments.py [--execute-list run_orca.sh]
                                                 [--experiments ER-DEN,ER-EXT]
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "research_runs"
MANIFEST = ROOT / "orca" / "experiments" / "experiments.json"


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the ORCA (ER-*) experiment grid.")
    ap.add_argument("--manifest", default=str(MANIFEST),
                    help="Path to orca/experiments/experiments.json")
    ap.add_argument("--execute-list", default=None,
                    help="Write a bash script with train+eval commands for all runs")
    ap.add_argument("--experiments", default="",
                    help="Comma list of experiment ids to generate (default: all)")
    ap.add_argument("--require-ontology", action="store_true", default=True,
                    help="Pass --require-ontology to the ORCA runner (default on).")
    args = ap.parse_args()

    manifest = _load_json(Path(args.manifest))
    base_config_path = ROOT / manifest["base_config"]
    base_config = _load_json(base_config_path)
    datasets = manifest["datasets"]
    seeds = manifest["seeds"]

    wanted = {e.strip() for e in args.experiments.split(",") if e.strip()}
    experiments = [
        e for e in manifest["experiments"]
        if not wanted or e["id"] in wanted
    ]

    written = 0
    cmds = []
    for exp in experiments:
        exp_id = exp["id"]
        overrides = exp.get("orca_overrides", {})
        for ds_label, ds in datasets.items():
            for seed in seeds:
                cfg = copy.deepcopy(base_config)
                cfg.update(overrides)          # variant-specific ORCA overrides
                cfg["orca_enabled"] = True     # ER-* are always ORCA runs
                cfg["validation_path"] = ds["validation"]
                cfg["training_seed"] = seed

                run_id = f"{exp_id}__{ds_label}__s{seed}"
                run_dir = OUT_ROOT / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg_path = run_dir / "training_config.json"
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                written += 1

                out_dir = f"results/research_runs/{run_id}"
                require = " --require-ontology" if args.require_ontology else ""
                train = (
                    f"python run_orca_training.py "
                    f"--train-file {ds['train']} "
                    f"--validation-file {ds['validation']} "
                    f"--config {out_dir}/training_config.json "
                    f"--output-dir {out_dir}/phase1_pretraining "
                    f"--variant {cfg['orca_variant']} --seed {seed}{require}"
                )
                evl = (
                    f"python run_phase1_embedding_evaluation.py "
                    f"--dataset {ds['test']} "
                    f"--checkpoint {out_dir}/phase1_pretraining/best_checkpoint.pt "
                    f"--config {out_dir}/training_config.json "
                    f"--output-dir {out_dir}/phase1_evaluation"
                )
                cmds.append((run_id, train, evl))

    print(f"Wrote {written} ORCA configs to {OUT_ROOT}")
    print(f"  {len(experiments)} variants x {len(datasets)} datasets x {len(seeds)} seeds")

    if args.execute_list:
        # Resumable + continue-on-error runner: no ``set -e`` so a single failed
        # run never aborts the whole grid, and each run is skipped when its
        # phase1_evaluation_results.json already exists so re-invoking the script
        # picks up only the outstanding runs. A per-run failure is logged and the
        # loop moves on; a summary of failures is printed at the end.
        lines = [
            "#!/usr/bin/env bash",
            "# Auto-generated by scripts/generate_orca_experiments.py.",
            "# Resumable: re-run to execute only the runs missing a completed",
            "# phase1_evaluation/phase1_evaluation_results.json. Continues past",
            "# individual failures instead of aborting the whole grid.",
            "set -uo pipefail",
            "",
            "FAILED=()",
            "SKIPPED=()",
            "",
        ]
        for run_id, train, evl in cmds:
            out_dir = f"results/research_runs/{run_id}"
            eval_result = f"{out_dir}/phase1_evaluation/phase1_evaluation_results.json"
            block = [
                f'echo "=== {run_id} ==="',
                f'if [ -f "{eval_result}" ]; then',
                f'  echo "  [skip] {run_id}: {eval_result} already exists"',
                f'  SKIPPED+=("{run_id}")',
                "else",
                f"  if {train} && {evl}; then",
                f'    echo "  [ok]   {run_id}"',
                "  else",
                f'    echo "  [FAIL] {run_id}"',
                f'    FAILED+=("{run_id}")',
                "  fi",
                "fi",
                "",
            ]
            lines += block
        lines += [
            'echo ""',
            'echo "=== ORCA grid summary ==="',
            'echo "Skipped (already complete): ${#SKIPPED[@]}"',
            'echo "Failed: ${#FAILED[@]}"',
            'if [ ${#FAILED[@]} -gt 0 ]; then',
            '  printf "  - %s\\n" "${FAILED[@]}"',
            "  exit 1",
            "fi",
            "",
        ]
        script = ROOT / args.execute_list
        script.write_text("\n".join(lines))
        print(f"Wrote run script: {script}")


if __name__ == "__main__":
    main()
