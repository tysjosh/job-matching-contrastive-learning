# Unfrozen Encoder Probe — GPU Run Guide

Tests the capacity-ceiling hypothesis: does the ordinal loss beat InfoNCE when the
SentenceTransformer encoder is **fine-tuned** (not frozen)?

## What runs

10 runs (2 variants × 5 seeds), v7 dataset:

| Variant | loss | encoder | negatives |
|---------|------|---------|-----------|
| UF-Ordinal  | ordinal | fine-tuned | pathway, 7 |
| UF-InfoNCE  | infonce | fine-tuned | random, 7 |

Settings: `freeze_text_encoder=false`, `lr=2e-5`, `batch_size=32`, `epochs=15`,
gradient checkpointing on, embedding cache bypassed (gradients flow to the encoder).

Compare against the already-completed **frozen** runs:
`EO-A` (frozen ordinal) and `E4-InfoNCE` (frozen InfoNCE), both 5 seeds.

## Prerequisites on the GPU box

- CUDA GPU (≥12 GB recommended; batch 32 + mpnet + checkpointing fits comfortably).
- Same repo + data (`preprocess/data_splits_v7/`), ESCO graph
  (`dataset/esco/esco_kg.gexf`, `training_output/career_graph_bridged_complete.gexf`).
- Python deps installed (torch w/ CUDA, sentence-transformers, networkx, numpy, scipy).

The code auto-selects CUDA (`torch.cuda.is_available()`); no config change needed.

## Run

```bash
# from repo root
bash run_unfrozen_probe_gpu.sh
```

This trains + evaluates all 10 runs sequentially, writing to
`results/research_runs/UF-*__cnamuangtoun__s*/`.

On a modern GPU expect a few minutes per epoch → roughly 15–40 min per run.

## Tips

- Larger batch: regenerate with `--batch-size 64` if VRAM allows (faster).
- If you hit OOM, lower `--batch-size` (e.g., 16) — checkpointing is already on.
- LR: `2e-5` is conservative for fine-tuning. `best_checkpoint.pt` is selected by
  validation loss, so mild over-training is handled by checkpoint selection.
- To regenerate configs:
  ```bash
  python3 scripts/generate_unfrozen_probe.py --epochs 15 --lr 2e-5 \
      --batch-size 32 --seeds 13 21 42 87 123 --execute-list run_unfrozen_probe_gpu.sh
  ```

## Aggregate + compare

After the runs finish, pull the numbers with the same aggregator used for EO:

```bash
python3 scripts/aggregate_eo_results.py        # EO (frozen) table + InfoNCE baseline
```

For the frozen-vs-unfrozen comparison, aggregate the UF runs the same way (the
eval JSONs are at `results/research_runs/UF-*/phase1_evaluation/ordinal_evaluation_results.json`).
Key contrasts:
- **UF-Ordinal vs EO-A**  → effect of unfreezing on the ordinal model
- **UF-InfoNCE vs E4-InfoNCE** → effect of unfreezing on InfoNCE
- **UF-Ordinal vs UF-InfoNCE** → does ordinal beat InfoNCE once capacity is unlocked?

## Caveats to report

- Unfrozen runs use `lr=2e-5` (vs `8.5e-5` frozen) — appropriate for encoder
  fine-tuning; note the LR difference when comparing.
- Everything else (loss, negatives, epochs=15, grouped batching, φ-guided margins)
  matches the frozen counterparts, so deltas are attributable to unfreezing.
- Verified: with `freeze_text_encoder=false`, 109.7M params train (encoder + head);
  gradient flow to the encoder confirmed via `scripts/confirm_encoder_finetune.py`.
