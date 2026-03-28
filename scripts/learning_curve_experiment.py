#!/usr/bin/env python3
"""Learning curve experiment: subsample training data at different fractions.

Creates stratified subsamples of the v6 training set at 100%, 50%, 25%, 10%,
preserving label ratios (good_fit / potential_fit / no_fit). Writes each
subsample into its own splits directory alongside symlinked val/test files,
then prints the 12 training commands (4 fractions × 3 configs).

Usage:
    python3 scripts/learning_curve_experiment.py [--seed 42] [--dry-run]
"""
import json
import os
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ── Configuration ──
SOURCE_DIR = Path("preprocess/data_splits_v6")
OUTPUT_BASE = Path("preprocess/learning_curve")
FRACTIONS = [1.0, 0.5, 0.25, 0.1]

CONFIGS = {
    "fixed_margin": "CDCL/config/phase1_ordinal_fixed_margin_config.json",
    "phi_corrected": "CDCL/config/phase1_ordinal_config.json",
}

DATASET_PATH = "preprocess/data_splits_v6/train.jsonl"


def stratified_subsample(samples, fraction, seed):
    """Subsample preserving label distribution."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for s in samples:
        label = s.get("metadata", {}).get("original_label", "unknown")
        by_label[label].append(s)

    subsampled = []
    for label, items in sorted(by_label.items()):
        k = max(1, int(len(items) * fraction))
        subsampled.extend(rng.sample(items, k))

    rng.shuffle(subsampled)
    return subsampled


def main():
    parser = argparse.ArgumentParser(description="Learning curve data preparation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Only print commands, don't create files")
    args = parser.parse_args()

    # Load full training set
    train_path = SOURCE_DIR / "train.jsonl"
    with open(train_path) as f:
        all_samples = [json.loads(line) for line in f]
    print(f"Full training set: {len(all_samples)} samples")

    # Label distribution
    label_counts = defaultdict(int)
    for s in all_samples:
        label_counts[s.get("metadata", {}).get("original_label", "unknown")] += 1
    print(f"Label distribution: {dict(label_counts)}")

    if not args.dry_run:
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Create subsampled splits
    for frac in FRACTIONS:
        pct = int(frac * 100)
        split_dir = OUTPUT_BASE / f"frac_{pct}"

        subset = stratified_subsample(all_samples, frac, args.seed)

        # Count labels in subset
        sub_labels = defaultdict(int)
        for s in subset:
            sub_labels[s.get("metadata", {}).get("original_label", "unknown")] += 1

        print(f"\n{'='*60}")
        print(f"Fraction {pct}%: {len(subset)} samples")
        print(f"  Labels: {dict(sub_labels)}")

        if not args.dry_run:
            split_dir.mkdir(parents=True, exist_ok=True)

            # Write subsampled train.jsonl
            train_out = split_dir / "train.jsonl"
            with open(train_out, "w") as f:
                for s in subset:
                    f.write(json.dumps(s) + "\n")

            # Symlink validation and test (same for all fractions)
            for name in ["validation.jsonl", "test.jsonl"]:
                target = (SOURCE_DIR / name).resolve()
                link = split_dir / name
                if link.exists() or link.is_symlink():
                    link.unlink()
                os.symlink(target, link)

            print(f"  Written to: {split_dir}/")

    # Print all training commands
    total_runs = len(FRACTIONS) * len(CONFIGS)
    print(f"\n{'='*60}")
    print(f"TRAINING COMMANDS ({total_runs} runs)")
    print(f"{'='*60}")

    for frac in FRACTIONS:
        pct = int(frac * 100)
        split_dir = OUTPUT_BASE / f"frac_{pct}"
        for config_name, config_path in CONFIGS.items():
            results_dir = f"results_lc_{pct}pct_{config_name}"
            cmd = (
                f"python3 run_training_with_split.py "
                f"--dataset {DATASET_PATH} "
                f"--phase1-config {config_path} "
                f"--use-existing-splits --splits-dir {split_dir} "
                f"--skip-phase2 --skip-phase2-eval"
            )
            print(f"\n# {pct}% data — {config_name}")
            print(f"# -> save results to {results_dir}/")
            print(cmd)

    # Print ordinal evaluation commands
    print(f"\n{'='*60}")
    print("ORDINAL EVALUATION COMMANDS (after each training run)")
    print(f"{'='*60}")
    for frac in FRACTIONS:
        pct = int(frac * 100)
        for config_name, config_path in CONFIGS.items():
            print(f"\n# {pct}% data — {config_name}")
            print(
                f"python3 run_ordinal_evaluation.py "
                f"--dataset preprocess/data_splits_v6/validation.jsonl "
                f"--ordinal-checkpoint phase1_pretraining/best_checkpoint.pt "
                f"--ordinal-config {config_path}"
            )

    print(f"\n{'='*60}")
    print("IMPORTANT: After each training run, save results before starting the next:")
    print("  mv phase1_pretraining results_lc_<pct>pct_<config>/phase1_pretraining")
    print("  mv phase1_evaluation results_lc_<pct>pct_<config>/phase1_evaluation")
    print("  mv ordinal_evaluation results_lc_<pct>pct_<config>/ordinal_evaluation")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
