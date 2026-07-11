#!/usr/bin/env python3
"""
Manifest Adapter: Translates run_manifests YAML into CDCL pipeline JSON configs + CLI commands.

Supports the manifest fields that map to existing pipeline capabilities:
  - E1-A (fixed negative ratios)
  - E3-A (φ weighting), E3-D (no weighting)
  - E4 variants (InfoNCE, OSCAR-Skill, OSCAR-Hybrid, OSCAR-ISCO)
  - Configurable seed, projection_dim, dataset paths

Usage:
  python manifest_adapter.py <manifest.yaml> --workdir /path/to/CDCL [--execute] [--output-dir results/research_runs]
  python manifest_adapter.py --batch <manifest_index.json> --workdir /path/to/CDCL [--experiment E4-InfoNCE] [--dataset indian]
"""
import argparse
import json
import copy
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# ── Base config templates (matching existing CDCL configs) ──────────────────

BASE_PHASE1 = {
    "batch_size": 64,
    "learning_rate": 0.000085,
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
    "esco_graph_path": "training_output/career_graph_bridged_complete.gexf",
    "esco_kg_path": "dataset/esco/esco_kg.gexf",
    "global_negative_sampling": True,
    "global_negative_pool_size": 1000,
    "freeze_text_encoder": True,
    "projection_dim": 128,
    "projection_dropout": 0.3,
    "training_phase": "self_supervised",
    "use_augmentation_labels_only": False,
    "augmentation_positive_ratio": 1.0,
    "pretrained_model_path": None,
    "freeze_contrastive_layers": False,
    "classification_dropout": 0.0,
    "max_negatives_per_anchor": 20,
    "validate_every_n_epochs": 1,
    "ontology_weight": 0.0,
    "ot_distance_scale": 10.0,
    "use_ot_distance": False,
    "loss_type": "infonce",
    "ws2_weight": 0.0,
    "use_reuse_weighting": False,
    "negative_curriculum": False,
    "negative_hard_ratio": 0.33,
    "negative_medium_ratio": 0.34,
    "negative_easy_ratio": 0.33,
    "training_seed": 42,
}

# ── Dataset path mapping ────────────────────────────────────────────────────

DATASET_PATHS = {
    "cnamuangtoun": {
        "train": "preprocess/data_splits_v7/train.jsonl",
        "validation": "preprocess/data_splits_v7/validation.jsonl",
        "test": "preprocess/data_splits_v7/test.jsonl",
        "skill_distances": "embedding_cache/skill_distances_v7_backup.pkl",
    },
    "cnamuangtou_sparse_v6": {
        "train": "preprocess/data_splits_v6/train.jsonl",
        "validation": "preprocess/data_splits_v6/validation.jsonl",
        "test": "preprocess/data_splits_v6/test.jsonl",
        "skill_distances": "embedding_cache/skill_distances_v6.pkl",
    },
    "indian": {
        "train": "preprocess/indian_splits/train.jsonl",
        "validation": "preprocess/indian_splits/validation.jsonl",
        "test": "preprocess/indian_splits/test.jsonl",
        "skill_distances": "embedding_cache/skill_distances_indian_backup.pkl",
    },
}

# ── Variant → config overlay mapping ────────────────────────────────────────

def _overlay_infonce():
    """InfoNCE baseline: random negatives, no ontology weighting."""
    return {
        "use_pathway_negatives": False,
        "ontology_weight": 0.0,
        "use_ot_distance": False,
        "negative_curriculum": False,
        "negative_hard_ratio": 0.33,
        "negative_medium_ratio": 0.34,
        "negative_easy_ratio": 0.33,
    }


def _overlay_oscar_skill():
    """OSCAR-Skill: skill-graph negatives + skill-based loss weighting."""
    return {
        "use_pathway_negatives": True,
        "ontology_weight": 0.3,
        "use_ot_distance": True,
        "negative_curriculum": False,
        "negative_hard_ratio": 0.33,
        "negative_medium_ratio": 0.34,
        "negative_easy_ratio": 0.33,
    }


def _overlay_oscar_hybrid():
    """OSCAR-Hybrid: skill + ISCO blended negatives + blended loss weighting."""
    return {
        "use_pathway_negatives": True,
        "ontology_weight": 0.3,
        "use_ot_distance": True,
        "use_isco_negatives": True,
        "isco_weight": 0.4,
        "isco_loss_weight": True,
        "esco_occupations_path": "dataset/esco/occupations_en.csv",
        "negative_curriculum": False,
        "negative_hard_ratio": 0.33,
        "negative_medium_ratio": 0.34,
        "negative_easy_ratio": 0.33,
    }


def _overlay_oscar_isco():
    """OSCAR-ISCO: pure ISCO group negatives + ISCO-only loss weighting."""
    return {
        "use_pathway_negatives": True,
        "ontology_weight": 0.3,
        "use_ot_distance": False,
        "use_isco_negatives": True,
        "isco_weight": 1.0,
        "isco_loss_weight": True,
        "isco_only_weight": True,
        "esco_occupations_path": "dataset/esco/occupations_en.csv",
        "negative_curriculum": False,
        "negative_hard_ratio": 0.33,
        "negative_medium_ratio": 0.34,
        "negative_easy_ratio": 0.33,
    }


def _overlay_e3e_oscar_isco_no_weighting():
    """E3-E: OSCAR-ISCO negative selection WITHOUT loss weighting."""
    return {
        "use_pathway_negatives": True,
        "ontology_weight": 0.0,
        "use_ot_distance": False,
        "use_isco_negatives": True,
        "isco_weight": 1.0,
        "isco_loss_weight": False,
        "isco_only_weight": True,
        "esco_occupations_path": "dataset/esco/occupations_en.csv",
        "negative_curriculum": False,
        "negative_hard_ratio": 0.33,
        "negative_medium_ratio": 0.34,
        "negative_easy_ratio": 0.33,
    }


def _overlay_e3f_oscar_hybrid_no_weighting():
    """E3-F: OSCAR-Hybrid negative selection WITHOUT loss weighting."""
    return {
        "use_pathway_negatives": True,
        "ontology_weight": 0.0,
        "use_ot_distance": False,
        "use_isco_negatives": True,
        "isco_weight": 0.4,
        "isco_loss_weight": False,
        "esco_occupations_path": "dataset/esco/occupations_en.csv",
        "negative_curriculum": False,
        "negative_hard_ratio": 0.33,
        "negative_medium_ratio": 0.34,
        "negative_easy_ratio": 0.33,
    }


# Map experiment_id or (family + ontology/variant) to overlay function
VARIANT_MAP = {
    # E4 robustness variants (directly named)
    "E4-InfoNCE": _overlay_infonce,
    "E4-OSCAR-Skill": _overlay_oscar_skill,
    "E4-OSCAR-Hybrid": _overlay_oscar_hybrid,
    "E4-OSCAR-ISCO": _overlay_oscar_isco,
    # E3 ablation variants (negative selection without loss weighting)
    "E3-E": _overlay_e3e_oscar_isco_no_weighting,
    "E3-F": _overlay_e3f_oscar_hybrid_no_weighting,
    # EO ablation variants (ordinal contrastive loss)
    "EO-A": lambda: _overlay_ordinal_base(),
    "EO-B": lambda: _overlay_ordinal_oscar_skill(),
    "EO-C": lambda: _overlay_ordinal_no_curriculum(),
    "EO-D": lambda: _overlay_ordinal_fixed_margin(),
    "EO-E": lambda: _overlay_ordinal_no_grouping(),
}


# ── Ordinal (EO) variant overlays ───────────────────────────────────────────

def _overlay_ordinal_base():
    """EO-A: query-anchored ordinal loss, φ-guided margins, curriculum, grouped
    batching (auto), NO OSCAR sample weighting."""
    return {
        "loss_type": "ordinal",
        "use_pathway_negatives": True,
        "max_negatives_per_anchor": 7,
        "ontology_weight": 0.0,
        "use_ot_distance": False,
        "ordinal_alpha": 0.5,
        "ordinal_lambda1": 1.0,
        "ordinal_lambda2": 1.0,
        "ordinal_m2": 0.3,
        "ordinal_fixed_m1": False,
        "ordinal_curriculum_switch": 0.3,
        "group_by_resume": None,
    }


def _overlay_ordinal_oscar_skill():
    """EO-B: EO-A + skill-based OSCAR sample weighting."""
    o = _overlay_ordinal_base()
    o.update({"ontology_weight": 0.3, "use_ot_distance": True})
    return o


def _overlay_ordinal_no_curriculum():
    """EO-C: EO-A without curriculum (L2+L3 active from epoch 0)."""
    o = _overlay_ordinal_base()
    o.update({"ordinal_curriculum_switch": 0.0})
    return o


def _overlay_ordinal_fixed_margin():
    """EO-D: EO-A with fixed m1 instead of φ-guided margin."""
    o = _overlay_ordinal_base()
    o.update({"ordinal_fixed_m1": True})
    return o


def _overlay_ordinal_no_grouping():
    """EO-E: EO-A without resume-grouped batching (graded siblings not co-batched)."""
    o = _overlay_ordinal_base()
    o.update({"group_by_resume": False})
    return o


def _resolve_variant(manifest: dict) -> dict:
    """Resolve manifest to config overlay based on experiment_id or ontology fields."""
    exp_id = manifest.get("experiment_id", "")

    # Direct E4 variant match
    if exp_id in VARIANT_MAP:
        return VARIANT_MAP[exp_id]()

    # E1/E3 family: resolve from ontology fields
    ontology = manifest.get("ontology", {})
    weighting = ontology.get("weighting", "none")
    scheduler = ontology.get("negative_scheduler", "fixed_033_034_033")

    overlay = {}

    # Weighting strategy
    if weighting == "none":
        overlay.update({
            "ontology_weight": 0.0,
            "use_ot_distance": False,
            "use_pathway_negatives": False,
        })
    elif weighting == "phi":
        overlay.update({
            "ontology_weight": 0.3,
            "use_ot_distance": True,
            "use_pathway_negatives": True,
        })

    # Negative scheduler
    if scheduler == "fixed_033_034_033":
        overlay.update({
            "negative_curriculum": False,
            "negative_scheduler": "fixed",
            "negative_hard_ratio": 0.33,
            "negative_medium_ratio": 0.34,
            "negative_easy_ratio": 0.33,
        })
    elif scheduler == "linear_easy_to_hard":
        overlay.update({
            "negative_curriculum": True,
            "negative_scheduler": "linear_easy_to_hard",
        })
    elif scheduler == "adaptive_val_dgp":
        overlay.update({
            "negative_curriculum": True,
            "negative_scheduler": "adaptive_val_dgp",
        })
    elif scheduler == "performance_gated_triplet":
        overlay.update({
            "negative_curriculum": True,
            "negative_scheduler": "performance_gated_triplet",
        })
    elif scheduler == "random":
        overlay.update({
            "use_pathway_negatives": False,
            "negative_curriculum": False,
            "negative_scheduler": "fixed",
        })

    return overlay


def translate_manifest(manifest: dict, workdir: str = ".") -> dict:
    """Translate a manifest YAML dict into a CDCL-compatible JSON config dict."""
    config = copy.deepcopy(BASE_PHASE1)

    # Seed
    config["training_seed"] = manifest.get("seed", 42)

    # Model overrides
    model = manifest.get("model", {})
    if model.get("encoder"):
        config["text_encoder_model"] = f"sentence-transformers/{model['encoder']}"
    if model.get("freeze_encoder") is not None:
        config["freeze_text_encoder"] = model["freeze_encoder"]
    if model.get("dropout") is not None:
        config["projection_dropout"] = model["dropout"]

    # Projection dim from last element of projection_dims
    proj_dims = model.get("projection_dims", [])
    if proj_dims:
        config["projection_dim"] = proj_dims[-1]

    # Training overrides
    training = manifest.get("training", {})
    if training.get("batch_size"):
        config["batch_size"] = training["batch_size"]
    if training.get("lr"):
        config["learning_rate"] = training["lr"]
    if training.get("epochs"):
        config["num_epochs"] = training["epochs"]
    if training.get("temperature"):
        config["temperature"] = training["temperature"]
    if training.get("max_negatives"):
        # Manifest specifies max_negatives but we override to 20 for fair comparison
        # across all E4 variants (InfoNCE was accidentally trained with 20)
        pass  # Use BASE_PHASE1 value (20) instead of manifest value (7)

    # Dataset paths
    dataset_label = manifest.get("dataset", "cnamuangtoun")
    ds = DATASET_PATHS.get(dataset_label, DATASET_PATHS["cnamuangtoun"])
    config["validation_path"] = ds["validation"]

    # Apply variant overlay
    overlay = _resolve_variant(manifest)
    config.update(overlay)

    return config


def build_commands(manifest: dict, config_path: str, workdir: str, output_dir: str) -> list:
    """Build the CLI commands to execute for this manifest."""
    dataset_label = manifest.get("dataset", "cnamuangtoun")
    ds = DATASET_PATHS.get(dataset_label, DATASET_PATHS["cnamuangtoun"])
    seed = manifest.get("seed", 42)
    run_id = manifest.get("run_id", "unknown")

    commands = []

    # Phase 1 training
    train_cmd = (
        f"python -m contrastive_learning train "
        f"{ds['train']} "
        f"--config {config_path} "
        f"--output-dir {output_dir}/phase1_pretraining "
        f"--seed {seed}"
    )
    commands.append(("phase1_train", train_cmd))

    # Phase 1 ordinal evaluation (on test set)
    eval_cmd = (
        f"python run_ordinal_evaluation.py "
        f"--ordinal-checkpoint {output_dir}/phase1_pretraining/best_checkpoint.pt "
        f"--ordinal-config {config_path} "
        f"--dataset {ds['test']} "
        f"--output-dir {output_dir}/phase1_evaluation"
    )
    commands.append(("phase1_eval", eval_cmd))

    return commands


def process_manifest(manifest_path: str, workdir: str, output_root: str,
                     execute: bool = False, dry_run: bool = True,
                     skip_completed: bool = False) -> dict:
    """Process a single manifest file: translate, write config, optionally execute."""
    manifest_path = Path(manifest_path)

    if yaml is None:
        raise ImportError("PyYAML required: pip install PyYAML")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    run_id = manifest.get("run_id", manifest_path.stem)
    output_dir = f"{output_root}/{run_id}"

    # Translate to config
    config = translate_manifest(manifest, workdir)

    # Write config JSON
    config_dir = Path(workdir) / output_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = str(config_dir / "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Build commands
    commands = build_commands(manifest, config_path, workdir, output_dir)

    result = {
        "run_id": run_id,
        "config_path": config_path,
        "output_dir": output_dir,
        "commands": commands,
        "supported": True,
    }

    # Check for unsupported features
    unsupported = []
    warnings = []
    exp_id = manifest.get("experiment_id", "")
    ontology = manifest.get("ontology", {})
    scheduler = ontology.get("negative_scheduler", "")
    weighting = ontology.get("weighting", "")

    if scheduler in ("__none_unsupported__",):  # All schedulers now supported
        unsupported.append(f"negative_scheduler={scheduler}")
    if weighting in ("phi_plus_link_conf", "phi_plus_link_conf_plus_ot_gate"):
        unsupported.append(f"weighting={weighting}")
    if manifest.get("retrieval"):
        unsupported.append("retrieval/reranker pipeline")
    if manifest.get("robustness_tests"):
        warnings.append("robustness_tests (skipped, training still works)")
    if manifest.get("index"):
        idx_type = manifest["index"].get("type", "")
        if idx_type in ("ivf_pq", "hnsw"):
            unsupported.append(f"index={idx_type}")
        if manifest["index"].get("emb_quant", "none") != "none":
            unsupported.append(f"emb_quant={manifest['index']['emb_quant']}")

    if unsupported:
        result["supported"] = False
        result["unsupported_features"] = unsupported
    if warnings:
        result["warnings"] = warnings

    # Completion markers for skip logic
    workdir_path = Path(workdir)
    checkpoint_exists = (workdir_path / output_dir / "phase1_pretraining" / "best_checkpoint.pt").exists()
    eval_exists = (workdir_path / output_dir / "phase1_evaluation" / "ordinal_evaluation_results.json").exists()

    # Print or execute
    for step_name, cmd in commands:
        if dry_run:
            print(f"  [{step_name}] {cmd}")
        elif execute:
            # Skip logic
            if skip_completed:
                if step_name == "phase1_train" and checkpoint_exists:
                    print(f"  ✓ [{step_name}] SKIPPED (best_checkpoint.pt exists)")
                    continue
                if step_name == "phase1_eval" and eval_exists:
                    print(f"  ✓ [{step_name}] SKIPPED (eval results exist)")
                    continue

            print(f"  Executing [{step_name}]: {cmd}")
            rc = subprocess.call(cmd, shell=True, cwd=workdir)
            if rc != 0:
                print(f"  ERROR: {step_name} failed with exit code {rc}")
                result["error"] = f"{step_name} failed (rc={rc})"
                break
            if rc != 0:
                print(f"  ERROR: {step_name} failed with exit code {rc}")
                result["error"] = f"{step_name} failed (rc={rc})"
                break

    if warnings:
        print(f"  ℹ Warnings: {', '.join(warnings)}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Translate run manifests to CDCL pipeline configs")
    ap.add_argument("manifest", nargs="?", help="Path to a single manifest YAML file")
    ap.add_argument("--batch", help="Path to manifest_index.json for batch processing")
    ap.add_argument("--workdir", default=".", help="CDCL project root directory")
    ap.add_argument("--output-root", default="results/research_runs",
                    help="Root directory for output artifacts")
    ap.add_argument("--execute", action="store_true", help="Actually run the commands")
    ap.add_argument("--experiment", action="append", help="Filter by experiment_id (repeatable)")
    ap.add_argument("--dataset", action="append",
                    choices=["cnamuangtoun", "cnamuangtou_sparse_v6", "indian"],
                    help="Filter by dataset (repeatable)")
    ap.add_argument("--seed", type=int, action="append", help="Filter by seed (repeatable)")
    ap.add_argument("--supported-only", action="store_true",
                    help="Only process manifests with fully supported features")
    ap.add_argument("--skip-completed", action="store_true",
                    help="Skip runs where training/eval already completed (checks for best_checkpoint.pt and eval results)")
    args = ap.parse_args()

    if args.manifest:
        print(f"Processing: {args.manifest}")
        result = process_manifest(
            args.manifest, args.workdir, args.output_root,
            execute=args.execute, dry_run=not args.execute,
            skip_completed=getattr(args, 'skip_completed', False)
        )
        if not result["supported"]:
            print(f"  ⚠ Unsupported features: {', '.join(result['unsupported_features'])}")
        print()

    elif args.batch:
        with open(args.batch) as f:
            index = json.load(f)

        # Apply filters
        filtered = []
        for entry in index:
            if args.experiment and entry["experiment_id"] not in args.experiment:
                continue
            if args.dataset and entry["dataset"] not in args.dataset:
                continue
            if args.seed and entry["seed"] not in args.seed:
                continue
            filtered.append(entry)

        print(f"Processing {len(filtered)} manifests (of {len(index)} total)")

        supported_count = 0
        skipped_count = 0
        for i, entry in enumerate(filtered, 1):
            yaml_path = entry.get("yaml", "")
            # Try local path first
            local_path = Path(args.workdir) / "run_manifests" / "yaml" / Path(yaml_path).name
            if local_path.exists():
                yaml_path = str(local_path)
            elif not Path(yaml_path).exists():
                print(f"[{i}/{len(filtered)}] SKIP {entry['run_id']} — manifest not found")
                skipped_count += 1
                continue

            print(f"[{i}/{len(filtered)}] {entry['run_id']}")
            result = process_manifest(
                yaml_path, args.workdir, args.output_root,
                execute=args.execute, dry_run=not args.execute,
                skip_completed=getattr(args, 'skip_completed', False)
            )

            if not result["supported"]:
                if args.supported_only:
                    print(f"  ⚠ SKIPPED (unsupported: {', '.join(result['unsupported_features'])})")
                    skipped_count += 1
                    continue
                else:
                    print(f"  ⚠ Unsupported features: {', '.join(result['unsupported_features'])}")
            else:
                supported_count += 1

        print(f"\nSummary: {supported_count} supported, {skipped_count} skipped, {len(filtered)} total")

    else:
        ap.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
