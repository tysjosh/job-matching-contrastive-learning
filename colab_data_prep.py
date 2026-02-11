#!/usr/bin/env python3
"""
Colab Data Preparation for Two-Phase Training Pipeline.

Creates a compressed package and Jupyter notebook for running the
two-phase contrastive learning pipeline on Google Colab with GPU.
"""

import json
import os
import shutil
import tarfile
from pathlib import Path


def verify_required_files() -> dict:
    """Check that all files needed for the Colab package exist."""
    required = {
        # Data splits
        "preprocess/data_splits_new_data/train.jsonl": "Training split",
        "preprocess/data_splits_new_data/validation.jsonl": "Validation split",
        "preprocess/data_splits_new_data/test.jsonl": "Test split",
        # Career graph
        "training_output/career_graph_bridged_complete.gexf": "ESCO career graph",
        # Configs
        "config/colab_phase1_config.json": "Phase 1 config (Colab/GPU)",
        "config/colab_phase2_config.json": "Phase 2 config (Colab/GPU)",
        # Training scripts
        "run_training_with_split.py": "Two-phase training pipeline",
        "run_phase1_embedding_evaluation.py": "Phase 1 evaluation",
        "run_phase2_classification_evaluation.py": "Phase 2 evaluation",
        # Requirements
        "requirements.txt": "Python dependencies",
    }

    # Directories
    required_dirs = {
        "contrastive_learning": "Core ML pipeline",
        "augmentation": "Career-aware augmentation",
        "diagnostic": "Training diagnostics",
        "config": "Configuration files",
    }

    results = {}
    print("=== Verifying required files ===")
    all_ok = True

    for path, desc in required.items():
        exists = os.path.exists(path)
        size = ""
        if exists:
            s = os.path.getsize(path)
            size = f" ({s / 1024 / 1024:.1f}MB)" if s > 1024 * 1024 else f" ({s / 1024:.1f}KB)"
        status = "✓" if exists else "✗"
        print(f"  {status} {desc}: {path}{size}")
        results[path] = exists
        if not exists:
            all_ok = False

    for path, desc in required_dirs.items():
        exists = os.path.isdir(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {desc}: {path}/")
        results[path] = exists
        if not exists:
            all_ok = False

    print(f"\n{'✅ All files present' if all_ok else '❌ Some files missing'}")
    return results


def create_colab_package(output_dir: str = "colab_package") -> str:
    """
    Create a compressed package with everything needed for Colab two-phase training.

    Returns:
        Path to the created tar.gz file.
    """
    package_dir = Path(output_dir)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    # Data splits → packaged as data_splits/ (flat)
    splits_dest = package_dir / "data_splits"
    splits_dest.mkdir()
    for split_name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        src = Path("preprocess/data_splits_new_data") / split_name
        if src.exists():
            shutil.copy2(src, splits_dest / split_name)
            print(f"✓ Data split: {split_name}")
        else:
            print(f"✗ Missing: {src}")

    # Career graph
    graph_src = Path("training_output/career_graph_bridged_complete.gexf")
    graph_dest = package_dir / "training_output"
    graph_dest.mkdir(parents=True)
    if graph_src.exists():
        shutil.copy2(graph_src, graph_dest / graph_src.name)
        print(f"✓ Career graph: {graph_src.name}")

    # Config files
    config_dest = package_dir / "config"
    config_dest.mkdir()
    config_files = [
        "config/colab_phase1_config.json",
        "config/colab_phase2_config.json",
    ]
    # Also copy metadata mappings and transformation rules if they exist
    for config_subdir in ["metadata_mappings", "transformation_rules", "synonyms"]:
        src_dir = Path("config") / config_subdir
        if src_dir.exists():
            shutil.copytree(src_dir, config_dest / config_subdir, dirs_exist_ok=True)
            print(f"✓ Config dir: config/{config_subdir}/")

    for cf in config_files:
        if Path(cf).exists():
            shutil.copy2(cf, config_dest / Path(cf).name)
            print(f"✓ Config: {Path(cf).name}")

    # Training scripts
    scripts = [
        "run_training_with_split.py",
        "run_phase1_embedding_evaluation.py",
        "run_phase2_classification_evaluation.py",
        "requirements.txt",
    ]
    for script in scripts:
        if Path(script).exists():
            shutil.copy2(script, package_dir / script)
            print(f"✓ Script: {script}")

    # Python packages (directories)
    dirs_to_copy = [
        "contrastive_learning",
        "augmentation",
        "diagnostic",
    ]
    for d in dirs_to_copy:
        if Path(d).exists():
            dest = package_dir / d
            shutil.copytree(d, dest, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
            print(f"✓ Package: {d}/")

    # ESCO data (optional but useful)
    esco_dir = Path("dataset/esco")
    if esco_dir.exists():
        dest = package_dir / "dataset" / "esco"
        dest.mkdir(parents=True)
        shutil.copytree(esco_dir, dest, dirs_exist_ok=True)
        print("✓ ESCO data: dataset/esco/")

    # Create archive
    archive_path = f"{output_dir}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_dir, arcname=".")

    size_mb = Path(archive_path).stat().st_size / (1024 * 1024)
    print(f"\n📦 Package created: {archive_path} ({size_mb:.1f} MB)")
    return archive_path


def create_colab_notebook(notebook_path: str = "CDCL_Two_Phase_Training.ipynb") -> None:
    """Create a Jupyter notebook for two-phase training on Google Colab."""

    cells = []

    # --- Title ---
    cells.append(_md_cell([
        "# CDCL: Two-Phase Career-Aware Contrastive Learning\n",
        "\n",
        "This notebook runs the full two-phase training pipeline on Google Colab with GPU:\n",
        "\n",
        "1. **Setup** — Install deps, mount Drive, extract data\n",
        "2. **Phase 1** — Contrastive pretraining (self-supervised)\n",
        "3. **Phase 1 Eval** — Embedding quality on validation set\n",
        "4. **Phase 2** — Classification fine-tuning (supervised)\n",
        "5. **Phase 2 Eval** — Final test set evaluation\n",
        "\n",
        "**Run cells in order. Ensure GPU runtime is enabled.**"
    ]))

    # --- Cell 1: Setup ---
    cells.append(_md_cell(["## 1. Environment Setup"]))
    cells.append(_code_cell(_setup_cell()))

    # --- Cell 2: Extract data ---
    cells.append(_md_cell(["## 2. Mount Drive & Extract Data"]))
    cells.append(_code_cell(_extract_data_cell()))

    # --- Cell 3: Verify ---
    cells.append(_md_cell(["## 3. Verify Installation"]))
    cells.append(_code_cell(_verify_cell()))

    # --- Cell 4: Run training ---
    cells.append(_md_cell([
        "## 4. Run Two-Phase Training\n",
        "\n",
        "This runs `run_training_with_split.py` with pre-split data.\n",
        "Phase 1 → Phase 1 Eval → Phase 2 → Phase 2 Eval."
    ]))
    cells.append(_code_cell(_training_cell()))

    # --- Cell 5: Results ---
    cells.append(_md_cell(["## 5. Analyze Results"]))
    cells.append(_code_cell(_results_cell()))

    # --- Cell 6: Download ---
    cells.append(_md_cell(["## 6. Download Results"]))
    cells.append(_code_cell(_download_cell()))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU"
        },
        "cells": cells
    }

    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=2)

    print(f"✓ Notebook created: {notebook_path}")


def _md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }


def _code_cell(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source
    }


def _setup_cell() -> str:
    return '''import subprocess, sys, os, torch

print("1. GPU Check")
if torch.cuda.is_available():
    print(f"  ✓ {torch.cuda.get_device_name()} — {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("  ✗ No GPU — enable GPU runtime in Runtime > Change runtime type")

print("\\n2. Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "sentence-transformers", "transformers", "networkx",
    "scikit-learn", "matplotlib", "seaborn", "tqdm", "faiss-cpu",
    "nltk", "PyYAML", "pandas"], check=True)

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("\\n✅ Setup complete")'''


def _extract_data_cell() -> str:
    return '''import os, tarfile

# Mount Google Drive
from google.colab import drive
drive.mount("/content/drive")

# Extract package — adjust path if you placed it elsewhere
package_path = "/content/drive/MyDrive/colab_package.tar.gz"

if not os.path.exists(package_path):
    print(f"✗ Package not found at {package_path}")
    print("  Upload colab_package.tar.gz to your Google Drive root")
else:
    os.chdir("/content")
    with tarfile.open(package_path, "r:gz") as tar:
        tar.extractall("/content/cdcl")
    os.chdir("/content/cdcl")
    print(f"✓ Extracted to {os.getcwd()}")
    print(f"  Contents: {os.listdir('.')}")'''


def _verify_cell() -> str:
    return '''import os, sys
sys.path.insert(0, "/content/cdcl")
os.chdir("/content/cdcl")

required = [
    "data_splits/train.jsonl",
    "data_splits/validation.jsonl",
    "data_splits/test.jsonl",
    "training_output/career_graph_bridged_complete.gexf",
    "config/colab_phase1_config.json",
    "config/colab_phase2_config.json",
    "run_training_with_split.py",
    "run_phase1_embedding_evaluation.py",
    "run_phase2_classification_evaluation.py",
]

all_ok = True
for f in required:
    exists = os.path.exists(f)
    size = ""
    if exists:
        s = os.path.getsize(f)
        size = f" ({s/1024/1024:.1f}MB)" if s > 1e6 else f" ({s/1024:.1f}KB)"
    print(f"  {'✓' if exists else '✗'} {f}{size}")
    if not exists:
        all_ok = False

# Count samples
for split in ["train", "validation", "test"]:
    path = f"data_splits/{split}.jsonl"
    if os.path.exists(path):
        with open(path) as fh:
            count = sum(1 for _ in fh)
        print(f"  📊 {split}: {count:,} samples")

# Test imports
try:
    from contrastive_learning.trainer import ContrastiveLearningTrainer
    from contrastive_learning.fine_tuning_trainer import FineTuningTrainer
    from contrastive_learning.data_splitter import DataSplitter
    print("\\n  ✓ All pipeline imports OK")
except ImportError as e:
    print(f"\\n  ✗ Import error: {e}")
    all_ok = False

import torch
print(f"  ✓ Device: {'cuda — ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")
print(f"\\n{'✅ Ready for training' if all_ok else '❌ Fix issues above'}")'''


def _training_cell() -> str:
    return '''import subprocess, sys, os, time

os.chdir("/content/cdcl")

# The dataset arg is required but won't be re-split (--use-existing-splits)
cmd = [
    sys.executable, "run_training_with_split.py",
    "--dataset", "data_splits/train.jsonl",
    "--use-existing-splits",
    "--splits-dir", "data_splits",
    "--phase1-config", "config/colab_phase1_config.json",
    "--phase2-config", "config/colab_phase2_config.json",
]

print(f"Running: {' '.join(cmd)}")
print("=" * 60)

start = time.time()
process = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1
)

for line in iter(process.stdout.readline, ""):
    print(line.rstrip())

return_code = process.wait()
elapsed = time.time() - start

print("=" * 60)
if return_code == 0:
    print(f"✅ Pipeline complete in {elapsed/60:.1f} minutes")
else:
    print(f"❌ Pipeline failed (exit code {return_code})")'''


def _results_cell() -> str:
    return '''import json, os
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/content/cdcl")

# Phase 1 results
p1_results_path = "phase1_pretraining/training_results.json"
if os.path.exists(p1_results_path):
    with open(p1_results_path) as f:
        p1 = json.load(f)

    print("=== Phase 1: Contrastive Pretraining ===")
    print(f"  Final loss: {p1['final_loss']:.6f}")
    print(f"  Training time: {p1['training_time']/60:.1f} min")
    print(f"  Epochs: {len(p1['epoch_losses'])}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(p1["epoch_losses"]) + 1)
    axes[0].plot(epochs, p1["epoch_losses"], "b-o", label="Train")
    if p1.get("validation_losses"):
        axes[0].plot(epochs, p1["validation_losses"], "r-s", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Phase 1: Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

# Phase 1 eval
p1_eval_path = "phase1_evaluation/phase1_evaluation_results.json"
if os.path.exists(p1_eval_path):
    with open(p1_eval_path) as f:
        p1_eval = json.load(f)

    print("\\n=== Phase 1 Evaluation (Validation Set) ===")
    for k, v in p1_eval["metrics"].items():
        print(f"  {k}: {v:.4f}")
    print(f"  Optimal threshold: {p1_eval['optimal_threshold']:.4f}")
    print(f"  Positive/Negative similarity gap: {p1_eval['similarity_stats']['separation']:.4f}")

# Phase 2 results
p2_results_path = "phase2_finetuning/training_results.json"
if os.path.exists(p2_results_path):
    with open(p2_results_path) as f:
        p2 = json.load(f)

    print("\\n=== Phase 2: Classification Fine-tuning ===")
    print(f"  Final accuracy: {p2.get('final_accuracy', 'N/A')}")
    print(f"  Training time: {p2.get('training_time', 0)/60:.1f} min")

    if p2.get("epoch_losses"):
        epochs2 = range(1, len(p2["epoch_losses"]) + 1)
        axes[1].plot(epochs2, p2["epoch_losses"], "b-o", label="Train")
        if p2.get("validation_losses"):
            axes[1].plot(epochs2, p2["validation_losses"], "r-s", label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Phase 2: Loss Curves")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# Phase 2 eval (test set)
test_eval_path = "test_evaluation/phase2_evaluation_results.json"
if os.path.exists(test_eval_path):
    with open(test_eval_path) as f:
        test_eval = json.load(f)

    print("\\n=== Phase 2 Evaluation (Test Set) ===")
    if "metrics" in test_eval:
        for k, v in test_eval["metrics"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    if "confusion_matrix" in test_eval:
        cm = test_eval["confusion_matrix"]
        print(f"  TP={cm.get('true_positives')}, TN={cm.get('true_negatives')}, "
              f"FP={cm.get('false_positives')}, FN={cm.get('false_negatives')}")
else:
    print("\\nPhase 2 evaluation results not found yet.")'''


def _download_cell() -> str:
    return '''import tarfile, os
from pathlib import Path

os.chdir("/content/cdcl")

archive = "/content/cdcl_results.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for d in ["phase1_pretraining", "phase1_evaluation", "phase2_finetuning", "test_evaluation"]:
        if os.path.exists(d):
            tar.add(d)
            print(f"  ✓ Packed: {d}/")
    if os.path.exists("training_curves.png"):
        tar.add("training_curves.png")

size = os.path.getsize(archive) / (1024 * 1024)
print(f"\\n📦 Results archive: {size:.1f} MB")

try:
    from google.colab import files
    files.download(archive)
    print("✅ Download started")
except ImportError:
    print(f"Archive at: {archive}")'''


def main():
    """Main entry point — verify, package, and create notebook."""
    print("=" * 60)
    print("CDCL Colab Package Builder (Two-Phase Pipeline)")
    print("=" * 60)

    verify_required_files()

    print("\n" + "=" * 60)
    print("Creating Colab package...")
    archive = create_colab_package()

    print("\n" + "=" * 60)
    print("Creating Colab notebook...")
    create_colab_notebook()

    print("\n" + "=" * 60)
    print("✅ Done! Next steps:")
    print(f"  1. Upload {archive} to Google Drive root")
    print("  2. Open CDCL_Two_Phase_Training.ipynb in Colab")
    print("  3. Enable GPU runtime (Runtime > Change runtime type)")
    print("  4. Run all cells")


if __name__ == "__main__":
    main()
