#!/usr/bin/env python3
"""
Generate the EO (Experiment-Ordinal) family: ablations of the query-anchored
ordinal contrastive loss, mirroring the E-series layout on disk.

Variants (each isolates one design decision of the ordinal loss):
  EO-A  Ordinal-Base        : loss_type=ordinal, φ-guided margins, curriculum on,
                              grouped batching (auto), NO OSCAR loss weighting.
  EO-B  Ordinal+OSCAR-Skill : EO-A + skill-based sample weighting (ontology_weight).
  EO-C  Ordinal-NoCurriculum: EO-A with ordinal_curriculum_switch=0.0 (L2+L3 from start).
  EO-D  Ordinal-FixedMargin : EO-A with ordinal_fixed_m1=true (no φ-guided m1).
  EO-E  Ordinal-NoGrouping  : EO-A with group_by_resume=false (siblings not co-batched).
  EO-RandNeg Ordinal-RandNeg : EO-A with use_pathway_negatives=false (random negatives;
                              isolates ontology-tiered negative selection).

Reference baseline for comparison: existing E4-InfoNCE runs.

Output: results/research_runs/EO-X__<dataset>__s<seed>/training_config.json
Run:    python3 scripts/generate_ordinal_experiments.py [--execute-list run_eo.sh]
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "research_runs"

SEEDS = [13, 21, 42, 87, 123]

# Ordinal experiments run on v7 only (graded good/potential/no_fit labels).
DATASETS = {
    "cnamuangtoun": {
        "train": "preprocess/data_splits_v7/train.jsonl",
        "validation": "preprocess/data_splits_v7/validation.jsonl",
        "test": "preprocess/data_splits_v7/test.jsonl",
        "graded": True,
    },
}

# Base ordinal config (matches results_ordinal_v6_phi_guided template).
BASE_ORDINAL = {
    "batch_size": 64,
    "learning_rate": 8.5e-05,
    "num_epochs": 15,
    "temperature": 0.07,
    "negative_sampling_ratio": 0.7,
    "pathway_weight": 0.8,
    "use_pathway_negatives": True,
    "use_view_augmentation": True,
    "checkpoint_frequency": 500,
    "log_frequency": 10,
    "shuffle_data": True,
    "text_encoder_model": "sentence-transformers/all-mpnet-base-v2",
    "text_encoder_device": None,
    "embedding_cache_size": 10000,
    "enable_embedding_preload": True,
    "clear_cache_between_epochs": False,
    "max_resume_views": 5,
    "max_job_views": 5,
    "fallback_on_augmentation_failure": True,
    "hard_negative_max_distance": 2.0,
    "medium_negative_max_distance": 4.0,
    "max_negatives_per_anchor": 7,
    "esco_graph_path": "training_output/career_graph_bridged_complete.gexf",
    "esco_kg_path": "dataset/esco/esco_kg.gexf",
    "global_negative_sampling": True,
    "global_negative_pool_size": 1000,
    "freeze_text_encoder": True,
    "projection_dim": 128,
    "projection_dropout": 0.3,
    "weight_decay": 0.0,
    "training_phase": "self_supervised",
    "use_augmentation_labels_only": False,
    "augmentation_positive_ratio": 1.0,
    "pretrained_model_path": None,
    "freeze_contrastive_layers": False,
    "classification_dropout": 0.0,
    "validate_every_n_epochs": 1,
    # ── ordinal core ──
    "loss_type": "ordinal",
    "ontology_weight": 0.0,          # EO-A isolates ordinal (no OSCAR weighting)
    "ot_distance_scale": 10.0,
    "use_ot_distance": False,
    "ws2_weight": 0.0,
    "ordinal_alpha": 0.5,
    "ordinal_lambda1": 1.0,
    "ordinal_lambda2": 1.0,
    "ordinal_m2": 0.3,
    "ordinal_fixed_m1": False,
    "ordinal_curriculum_switch": 0.3,
    "phi_gate_threshold": 1.0,
    "group_by_resume": None,          # auto -> on for ordinal
    "negative_curriculum": False,
    "negative_hard_ratio": 0.33,
    "negative_medium_ratio": 0.34,
    "negative_easy_ratio": 0.33,
}


def overlay(variant: str) -> dict:
    if variant == "EO-A":      # Ordinal-Base
        return {}
    if variant == "EO-B":      # + OSCAR-Skill sample weighting
        return {"ontology_weight": 0.3, "use_ot_distance": True}
    if variant == "EO-C":      # no curriculum
        return {"ordinal_curriculum_switch": 0.0}
    if variant == "EO-D":      # fixed margin (no phi-guided m1)
        return {"ordinal_fixed_m1": True}
    if variant == "EO-E":      # no resume grouping
        return {"group_by_resume": False}
    if variant == "EO-RandNeg":  # random (non-ontology) negatives
        return {"use_pathway_negatives": False}
    raise ValueError(variant)


VARIANTS = ["EO-A", "EO-B", "EO-C", "EO-D", "EO-E", "EO-RandNeg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-list", default=None,
                    help="Write a bash script with train+eval commands for all runs")
    args = ap.parse_args()

    written = 0
    cmds = []
    for variant in VARIANTS:
        for ds_label, ds in DATASETS.items():
            for seed in SEEDS:
                cfg = copy.deepcopy(BASE_ORDINAL)
                cfg.update(overlay(variant))
                cfg["validation_path"] = ds["validation"]
                cfg["training_seed"] = seed

                run_id = f"{variant}__{ds_label}__s{seed}"
                run_dir = OUT_ROOT / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg_path = run_dir / "training_config.json"
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                written += 1

                out_dir = f"results/research_runs/{run_id}"
                train = (f"python -m contrastive_learning train {ds['train']} "
                         f"--config {out_dir}/training_config.json "
                         f"--output-dir {out_dir}/phase1_pretraining --seed {seed}")
                evl = (f"python run_ordinal_evaluation.py "
                       f"--ordinal-checkpoint {out_dir}/phase1_pretraining/best_checkpoint.pt "
                       f"--ordinal-config {out_dir}/training_config.json "
                       f"--dataset {ds['test']} "
                       f"--output-dir {out_dir}/phase1_evaluation")
                cmds.append((run_id, train, evl))

    print(f"Wrote {written} configs to {OUT_ROOT}")
    print(f"  {len(VARIANTS)} variants x {len(DATASETS)} datasets x {len(SEEDS)} seeds")

    if args.execute_list:
        lines = ["#!/usr/bin/env bash", "set -e", ""]
        for run_id, train, evl in cmds:
            lines += [f'echo "=== {run_id} ==="', train, evl, ""]
        script = ROOT / args.execute_list
        script.write_text("\n".join(lines))
        print(f"Wrote run script: {script}")


if __name__ == "__main__":
    main()
