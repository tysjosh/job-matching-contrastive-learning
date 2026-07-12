# CVE vulnerability-ranking experiments

An end-to-end, resumable harness that runs a matrix of Stage 1 (contrastive) +
Stage 2 (supervised heads) experiments and produces a single comparison table.

Everything is defined in [`experiments.json`](experiments.json) and executed by
[`run_experiments.py`](run_experiments.py).

## The matrix

| Exp | What it isolates | Key knobs |
|-----|------------------|-----------|
| **E0** | **No Stage 1** — supervised heads over the frozen base sentence-transformer. The honest reference: any lift of E1–E5 over E0 is the value of ontology-supervised contrastive pretraining. | `base_embeddings=true` |
| **E1** | Baseline contrastive: ontology-tiered negatives, **no** sample weighting. | `ontology_weight=0.0`, tiers 0.34/0.33/0.33 |
| **E2** | Principled, ontology-**independent** sample weighting by label-completeness quality tier. | `ontology_weight=0.3`, overlap off |
| **E3** | Ablation: the self-referential ontology-overlap sample weight (overlap that also *selected* the positive). | `ontology_weight=0.3`, `cve_ontology_overlap_weighting=true` |
| **E4** | Harder negative mix. | tiers 0.7/0.2/0.1 |
| **E5** | Temporal generalization (train older → test newer CVEs). | `split_strategy=temporal` |

All experiments share `split_seed=42`. E1–E4 reuse one stratified split; E5 uses its own temporal split. E0 reuses the stratified split.

## Pipeline per experiment

```
convert (shared, cached)
  └─ split per strategy (shared, cached)
       └─ Stage 1 contrastive pretrain      (skipped for E0)
            └─ Stage 2 supervised heads
                 └─ predict on test split
                      └─ evaluate (NDCG/MAP + in_kev/band accuracy & macro-F1 + band separation)
```

Metrics are computed by `CVEEvaluationReporter` **against the ground-truth
supervised labels only** (no circular ontology metric).

## Run it once

Full run (heavy — full data, encoder, all Stage 1/2 training; use a GPU):

```bash
.venv/bin/python -m cve_domain.experiments.run_experiments
```

Quick wiring smoke (tiny slice, 1 epoch each, two experiments):

```bash
.venv/bin/python -m cve_domain.experiments.run_experiments \
    --limit 2000 --epochs-stage1 1 --epochs-stage2 1 --experiments E0,E1
```

Prepare data + print the resolved plan without training:

```bash
.venv/bin/python -m cve_domain.experiments.run_experiments --dry-run
```

Subset / overrides:

```bash
# only E2 and E3
... run_experiments --experiments E2,E3
# recompute everything even if outputs exist
... run_experiments --force
```

### Useful flags

| Flag | Meaning |
|------|---------|
| `--experiments E1,E2` | Run a subset (default: all) |
| `--limit N` | Use only the first N converted view records (smoke) |
| `--epochs-stage1 N` / `--epochs-stage2 N` | Override training epochs |
| `--output-root DIR` | Where artifacts go (default `cve_domain/runs/experiments`) |
| `--force` | Recompute even if outputs exist |
| `--dry-run` | Prepare data, print plan, no training |

## Outputs

```
cve_domain/runs/experiments/
├── _shared/
│   ├── view_records_full.jsonl          # conversion output (cached)
│   ├── conversion_report.json
│   ├── split_stratified_seed42/         # train/validation/test.jsonl + reports
│   └── split_temporal_seed42/
├── E0/ ... E5/
│   ├── stage1_config.resolved.json      # (not for E0)
│   ├── stage1/best_checkpoint.pt        # (not for E0)
│   ├── stage2_config.resolved.json
│   ├── stage2/stage2_best_checkpoint.pt
│   └── eval/evaluation_report.json
├── summary.json
└── summary.md                           # the comparison table
```

## Resumability

Every step is skipped if its output already exists (conversion, splits, Stage 1
checkpoint, Stage 2 checkpoint). Re-running continues where it stopped; pass
`--force` to redo. If Stage 2 is already trained, the runner reloads the heads
from the checkpoint (`load_heads_from_checkpoint`) and jumps straight to
prediction/evaluation.

## Reading the results

`summary.md` is the headline table. Interpretation guide:
- **E1 vs E0** — does ontology-supervised Stage 1 help the real-label metrics at all?
- **E2 vs E1** — does the (independent) label-completeness weighting help?
- **E3 vs E2** — does the self-referential ontology-overlap weight add anything, or does it just amplify ontology bias? (Expected to be marginal or negative; that's a legitimate finding.)
- **E4 vs E1** — does a harder negative mix sharpen the embedding?
- **E5** — how much does performance drop under a temporal (future-CVE) split vs the stratified holdout?
