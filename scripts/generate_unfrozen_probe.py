#!/usr/bin/env python3
"""
Generate a small UNFROZEN probe to test the capacity-ceiling hypothesis:
does the ordinal loss beat InfoNCE when the encoder is allowed to fine-tune?

Creates head-to-head pairs (same setup as the frozen comparison, only the
encoder is unfrozen + lower LR), so any delta is attributable to unfreezing:

  UF-Ordinal  : from EO-A config      (loss_type=ordinal)
  UF-InfoNCE  : from E4-InfoNCE config (loss_type=infonce, random negatives)

Overrides applied to both:
  freeze_text_encoder = False        -> encoder is fine-tuned (grad-preserving path)
  learning_rate       = 2e-5         -> safe for pretrained-encoder fine-tuning
  enable_embedding_preload = False   -> cache bypassed anyway when unfrozen
  num_epochs          = EPOCHS       -> reduced; unfrozen is much slower on CPU/MPS

NOTE: single shared LR for head+encoder (probe simplicity). This is a PROBE,
not a full sweep — results are indicative.

Run: python3 scripts/generate_unfrozen_probe.py [--epochs 6] [--seeds 13 42]
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "research_runs"

TRAIN = "preprocess/data_splits_v7/train.jsonl"
TEST = "preprocess/data_splits_v7/test.jsonl"


# If the frozen base configs are absent (e.g. results/ was removed from the
# clone), fall back to the already-generated UF configs, which carry the same
# loss/negative settings. lr/epochs/batch/seq are overridden anyway.
_FALLBACK = {"EO-A": "UF-Ordinal", "E4-InfoNCE": "UF-InfoNCE"}


def load_base(variant_dir_prefix, seed):
    """Load a base config to derive from (frozen preferred, UF as fallback)."""
    candidates = [variant_dir_prefix]
    if variant_dir_prefix in _FALLBACK:
        candidates.append(_FALLBACK[variant_dir_prefix])
    for prefix in candidates:
        p = RUNS / f"{prefix}__cnamuangtoun__s{seed}" / "training_config.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No base config found for seed {seed}. Tried: "
        + ", ".join(f"{c}__cnamuangtoun__s{seed}" for c in candidates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch size. Default 64 to match the frozen EO/E4 runs "
                         "(use smaller, e.g. 16, only on memory-constrained CPUs).")
    ap.add_argument("--max-seq-length", type=int, default=256,
                    help="Encoder seq-length cap while fine-tuning (memory scales "
                         "with seq_len^2). 256 fits ~44GB GPUs; use 512 on an "
                         "A100 80GB for parity with the frozen runs.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 42])
    ap.add_argument("--variant-prefix", default="UF",
                    help="Run-id prefix. Use 'UF' for the main probe, or e.g. "
                         "'UFG' for a gentler variant (lower lr / fewer epochs) "
                         "so it doesn't overwrite the main UF runs.")
    ap.add_argument("--execute-list", default="run_unfrozen_probe.sh")
    args = ap.parse_args()

    # (probe variant name, source frozen config prefix)
    p = args.variant_prefix
    variants = [(f"{p}-Ordinal", "EO-A"), (f"{p}-InfoNCE", "E4-InfoNCE")]

    cmds = []
    written = 0
    for probe_name, src_prefix in variants:
        for seed in args.seeds:
            cfg = copy.deepcopy(load_base(src_prefix, seed))
            cfg["freeze_text_encoder"] = False
            cfg["learning_rate"] = args.lr
            cfg["num_epochs"] = args.epochs
            cfg["batch_size"] = args.batch_size
            cfg["unfrozen_max_seq_length"] = args.max_seq_length
            cfg["enable_embedding_preload"] = False
            cfg["training_seed"] = seed
            cfg["validation_path"] = "preprocess/data_splits_v7/validation.jsonl"

            run_id = f"{probe_name}__cnamuangtoun__s{seed}"
            run_dir = RUNS / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "training_config.json", "w") as f:
                json.dump(cfg, f, indent=2)
            written += 1

            out = f"results/research_runs/{run_id}"
            train = (f"python -m contrastive_learning train {TRAIN} "
                     f"--config {out}/training_config.json "
                     f"--output-dir {out}/phase1_pretraining --seed {seed}")
            evl = (f"python run_ordinal_evaluation.py "
                   f"--ordinal-checkpoint {out}/phase1_pretraining/best_checkpoint.pt "
                   f"--ordinal-config {out}/training_config.json "
                   f"--dataset {TEST} "
                   f"--output-dir {out}/phase1_evaluation")
            cmds.append((run_id, train, evl))

    print(f"Wrote {written} unfrozen probe configs "
          f"({len(variants)} variants x {len(args.seeds)} seeds), "
          f"epochs={args.epochs}, lr={args.lr}")

    lines = ["#!/usr/bin/env bash", "set -e", ""]
    for run_id, train, evl in cmds:
        lines += [f'echo "=== {run_id} ==="', train, evl, ""]
    script = ROOT / args.execute_list
    script.write_text("\n".join(lines))
    print(f"Wrote run script: {script}")
    print("\nNOTE: unfrozen training is slow (full encoder fwd+bwd per batch, no cache).")
    print("Run AFTER the frozen sweep finishes to avoid CPU contention:")
    print(f"  bash {args.execute_list}")


if __name__ == "__main__":
    main()
