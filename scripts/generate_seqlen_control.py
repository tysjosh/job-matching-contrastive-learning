#!/usr/bin/env python3
"""
Generate a FROZEN-encoder @ seq_length=256 control to disentangle the two
things the unfrozen probe changed at once:

  (a) the encoder was unfrozen (fine-tuned), and
  (b) the encoder input was truncated 512 -> 256 to fit GPU memory.

The unfrozen runs collapsed on good_vs_potential. This control keeps the
encoder FROZEN (the known-good design) but truncates inputs to 256, exactly
matching the unfrozen runs' truncation. Interpretation:

  * frozen-256 keeps good_vs_potential ~= 1.0  -> truncation is exonerated;
    unfreezing (catastrophic forgetting) is the cause of the collapse.
  * frozen-256 ALSO collapses                  -> the 512->256 truncation is
    (at least partly) the culprit, not unfreezing.

Derives two controls from the existing UF configs (which carry the correct
loss/negative settings), flipping the encoder back to frozen and pinning the
seq cap:

  CTRL256-Ordinal : from UF-Ordinal  (loss_type=ordinal)
  CTRL256-InfoNCE : from UF-InfoNCE  (loss_type=infonce, random negatives)

Overrides applied to both:
  freeze_text_encoder      = True     -> known-good frozen design
  encoder_max_seq_length   = 256      -> unified cap, applies while frozen
  learning_rate            = 8.5e-5   -> frozen-run LR (head only)
  num_epochs               = 15       -> match the frozen sweep
  batch_size               = 64       -> match the frozen sweep
  enable_embedding_preload = False    -> IMPORTANT: cache key does NOT include
                                         seq_length, so preload would reuse
                                         stale 512-token embeddings. Keep off.

Run: python3 scripts/generate_seqlen_control.py [--seeds 13 21 42 87 123]
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "research_runs"

TRAIN = "preprocess/data_splits_v7/train.jsonl"
TEST = "preprocess/data_splits_v7/test.jsonl"

# Prefer deriving from the frozen bases; fall back to the UF configs (same
# loss/negative settings) since the GPU box may only carry the UF configs.
_BASE_CANDIDATES = {
    "CTRL256-Ordinal": ["EO-A", "UF-Ordinal"],
    "CTRL256-InfoNCE": ["E4-InfoNCE", "UF-InfoNCE"],
}


def load_base(probe_name, seed):
    for prefix in _BASE_CANDIDATES[probe_name]:
        p = RUNS / f"{prefix}__cnamuangtoun__s{seed}" / "training_config.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No base config found for {probe_name} seed {seed}. Tried: "
        + ", ".join(f"{c}__cnamuangtoun__s{seed}"
                    for c in _BASE_CANDIDATES[probe_name]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-length", type=int, default=256,
                    help="Frozen-encoder seq cap for the control (match the "
                         "unfrozen runs' truncation; default 256).")
    ap.add_argument("--lr", type=float, default=8.5e-5,
                    help="Head LR (frozen-run default).")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[13, 21, 42, 87, 123])
    ap.add_argument("--execute-list", default="run_seqlen_control.sh")
    args = ap.parse_args()

    cmds = []
    written = 0
    for probe_name in _BASE_CANDIDATES:
        for seed in args.seeds:
            cfg = copy.deepcopy(load_base(probe_name, seed))
            cfg["freeze_text_encoder"] = True
            cfg["encoder_max_seq_length"] = args.seq_length
            cfg["learning_rate"] = args.lr
            cfg["num_epochs"] = args.epochs
            cfg["batch_size"] = args.batch_size
            # Cache key ignores seq_length -> preload would serve stale
            # 512-token embeddings. Must stay off for a valid control.
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

    print(f"Wrote {written} seq-length control configs "
          f"(2 variants x {len(args.seeds)} seeds), "
          f"frozen encoder @ seq_length={args.seq_length}, "
          f"lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}")

    lines = ["#!/usr/bin/env bash", "set -e", ""]
    for run_id, train, evl in cmds:
        lines += [f'echo "=== {run_id} ==="', train, evl, ""]
    script = ROOT / args.execute_list
    script.write_text("\n".join(lines))
    print(f"Wrote run script: {script}")
    print("\nThis control is FAST (frozen encoder, cache bypassed but no "
          "encoder backward). Run it, then:")
    print("  python3 scripts/aggregate_eo_results.py")
    print("Compare CTRL256-* good_vs_potential against frozen (~1.0) and "
          "unfrozen (~0.06).")


if __name__ == "__main__":
    main()
