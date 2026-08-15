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
| ER-NOONTW | denominator, `λ_ont=0` | value of the ontology *weak-target supervision* |
| ER-NOONTALL | denominator, `λ_ont=0` + no features | total value of the ontology (both sites) |

### Reading the ablations: pick the matched control
The variants form a lattice over four factors (reliability application site,
adaptive sampling, alignment, ontology). Each factor is only isolable against
the control that matches on the other three — see `VARIANT_TABLE` in
`orca/config.py`:

| factor | contrast | note |
|---|---|---|
| reliability | ER-DEN − E4-OSCAR-Skill | |
| application site | ER-EXT − ER-DEN | both have adaptive sampling off |
| adaptive sampling | ER-NOALIGN − ER-DEN | both denominator, alignment off |
| alignment | ER-FULL − ER-NOALIGN | both have adaptive sampling on |
| ontology MLP features | ER-NOALIGN − ER-NOONT | **not** vs ER-DEN: `no_ontology` has `adaptive_sampling: True` while `denominator` has it off, so ER-NOONT − ER-DEN confounds two factors |
| ontology supervision | ER-NOONTW − ER-DEN | |
| ontology, total | ER-NOONTALL − ER-DEN | |

Note that ER-NOONT alone understates the ontology's role: it removes only the
ReliabilityMLP's input scalars, while ontology still reaches the model through
`r_ont = 1-exp(-β·d_ont)` in the weak targets (`orca/weak_targets.py`), which
supervises reliability via the BCE term. ER-NOONTW / ER-NOONTALL close that gap.

NDCG@10 carries std 0.06–0.16 across every variant and baseline, so it is too
noisy at n=5 to support a claim on its own.

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

## Artifact retention (read before uploading)
A run is only fully analyzable if THREE artifacts survive:

| artifact | produced by | consumed by |
|---|---|---|
| `phase1_pretraining/best_checkpoint.pt` | `run_orca_training.py` | both evals + `probe_reliability*.py` |
| `phase1_evaluation/phase1_evaluation_results.json` | `run_phase1_embedding_evaluation.py` | `aggregate_orca_results.py` |
| `phase1_evaluation/ordinal_evaluation_results.json` | `run_ordinal_evaluation.py` | `orca_table4.py`, `orca_sig_test.py` |

Two traps that have already cost re-runs:

1. **The generated runner does not call `run_ordinal_evaluation.py`.** It emits
   train + `run_phase1_embedding_evaluation.py` only, so four of the five Table-4
   metrics are missing unless you run the ordinal pass yourself. Only AUC is
   recoverable from the phase-1 artifact (there `metrics.auc_roc` is numerically
   identical to ordinal `binary_aucs.good_vs_rest`).
2. **Do not upload with `--include '*.json'`.** That silently drops the
   checkpoint, and without it the ordinal eval and the reliability probes cannot
   be run later — the run must be retrained from scratch. Upload the whole run
   directory:

```bash
r=ER-DEN__cnamuangtoun__s13
hf upload olukotunjosh/cdcl-orca-results "results/research_runs/$r" "orca/$r" --repo-type=dataset
```

Pull them back (checkpoints are opt-in, since they dominate the transfer):
```bash
python3 scripts/hf_pull_results.py --repo-id olukotunjosh/cdcl-orca-results \
    --include-checkpoints --runs 'ER-*' --overwrite
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
