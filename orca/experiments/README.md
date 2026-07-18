# ORCA experiments (ER-* family)

This directory defines the ORCA experiment matrix and how it slots into the
existing career research-run grid so ORCA is compared apples-to-apples with the
OSCAR / InfoNCE baselines.

## Files
- `experiments.json` — the matrix: base ORCA config + per-variant overrides,
  datasets, seeds, and the baselines ORCA is compared against.
- `../../scripts/generate_orca_experiments.py` — materializes one
  `training_config.json` per (variant × dataset × seed) under
  `results/research_runs/` and emits a runnable train+eval bash list.
- `../../scripts/aggregate_orca_results.py` — summarizes mean ± std per variant
  across seeds and the delta vs the InfoNCE / OSCAR baselines.

## The matrix
| ID | Variant | What it isolates |
|----|---------|------------------|
| ER-DEN | denominator (MVP) | reliability inside the InfoNCE denominator |
| ER-EXT | external_weight | headline: reliability on final loss vs denominator (Property 6) |
| ER-NOONT | no_ontology | value of the ontology features in the ReliabilityMLP |
| ER-NOALIGN | no_align | denominator + adaptive, no alignment term |
| ER-FULL | full | denominator + adaptive sampling + ontology alignment |

Baselines (already in `results/research_runs/`): `E4-InfoNCE` (all negatives
trusted) and `E4-OSCAR-Skill` (fixed ontology, no reliability).

## Run it
```bash
# 1. Materialize configs + a runnable command list
python3 scripts/generate_orca_experiments.py --execute-list run_orca.sh

# 2. Execute (each run: 4-phase ORCA train -> Phase-1 embedding eval)
bash run_orca.sh

# 3. Aggregate and compare against the baselines
python scripts/aggregate_orca_results.py --dataset cnamuangtoun
```

Notes
- ORCA runs require the prepared v6/v7 splits (with `skill_uris`); the runner
  passes `--require-ontology` and errors on raw data.
- Each ORCA run trains via `run_orca_training.py` (the 4-phase
  `OrcaPhaseOrchestrator`) and saves `phase1_pretraining/best_checkpoint.pt`
  (the trained projection head), which the shared
  `run_phase1_embedding_evaluation.py` loads — identical to how the OSCAR /
  InfoNCE baselines are evaluated.
- `ER-NOFUT` (history ablation) is omitted because it only applies to temporal
  datasets; add it once a temporal split exists.
