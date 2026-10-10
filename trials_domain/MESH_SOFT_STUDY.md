# MeSH V2 development study

The four arms are `uniform`, `text`, `mesh`, and `hybrid`. All draw without
replacement from the full judged candidate pool. Guided arms mix 50% uniform
probability with 50% guided probability, using softmax temperature 0.2. The
hybrid guided component averages text and MeSH probability distributions.
Only the not-relevant tier is guided; ineligible sampling, grade allocation,
InfoNCE loss, and model settings are shared. Uniform fallback applies when a
signal is unavailable or constant. Partially missing scores are imputed with
the mean observed score. Equal scores receive equal probability.

Frozen text mining uses only the training topics and their judged not-relevant
trials. Validation topics always use the same uniform sampler, fixed negative
seed, and grade allocation across arms for checkpoint selection. This remains
checkpoint selection by a common sampled validation loss; the patient ranking
metrics below are computed after training. It does not select checkpoints by
MAP or evaluate full rankings every epoch.

Run the isolated four-arm, one-epoch, 64-training-record smoke test:

```bash
python scripts/run_mesh_soft_sampling.py --smoke --execute
```

Run three patient-disjoint 2021 development folds across five seeds:

```bash
python scripts/run_mesh_soft_sampling.py --folds 3 --execute --evaluate
```

This is 60 runs (4 arms × 3 folds × 5 seeds: 13, 21, 42, 87, 123).
Completed runs with matching protocol fingerprints are skipped. Smoke outputs
use a different root and cannot be evaluated as full study runs. A single-fold
run uses the historical 2021 training/validation split; the three- and five-fold
options reconstruct uncapped 2021 judgments, holding each patient out once.
They are development CV, not nested CV or an independent confirmation set.
There is no 2022 evaluation option in this runner.

The declared development primary metric is patient-macro eligible-only average
precision, stored as `eligible_map`. Each patient contributes equally; only
grade 2 is positive. The evaluator also reports patient-macro pairwise AUC and
graded nDCG@10 with gains 0, 1, 3 for grades 0, 1, 2, while retaining pooled AUC.
It uses scikit-learn average precision and tie-aware nDCG. Per-patient metrics
and the number of patients supporting each contrast are included.

`results/mesh_soft_sampling/folds_3/validation2021_summary.json` contains all
run-level metrics, matched arm-minus-uniform deltas, and descriptive bootstrap
intervals over patients after averaging seeds within each patient. These
intervals do not adjust for model selection, overlapping fold training sets,
or all training randomness. Five seeds are repeated training runs, not five
independent cohorts. Preserve the original 2022 result; further development
requires a fresh independent test for a confirmatory performance claim.
