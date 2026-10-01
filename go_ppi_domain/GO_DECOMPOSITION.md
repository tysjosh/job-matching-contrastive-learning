# GO decomposition experiment

This experiment separates Gene Ontology *content* from *structure* on the fixed,
protein-disjoint GO/PPI split. Content has four arms: Biological Process (P),
Molecular Function (F), Cellular Component (C), and a combined P+F+C arm (A).
Structure has three arms per content choice:

| Mode | Similarity between two proteins |
| --- | --- |
| `exact` | Jaccard of directly annotated GO terms |
| `ancestor` | Jaccard after `is_a`/`part_of` ancestor propagation |
| `simgic` | Information-content-weighted Jaccard after propagation |

The combined arm computes each aspect's similarity with its own GO index and IC
distribution. It averages the available aspect scores with configurable positive
weights (equal by default). An aspect missing from either protein is omitted from
that pair's denominator; no overlap in a present aspect scores zero. Cross-aspect
term distance and transport are not defined.

All arms use the same fixed positives, graded negative pools, training settings,
and seeds. The persisted split stores BP terms, but the selector replaces those
terms for anchors, positive partners, and negative candidates from the selected
GO index before scoring. The source GAF excludes IPI, IEA, ND, and the binding
terms `GO:0005515` and `GO:0005488`. This avoids direct physical-interaction
evidence in the ontology feature. MF annotation is not universal: a protein with
no selected-aspect annotation receives neutral distance, and missing candidates
are placed after scorable ones in GO ordering. Report aspect coverage alongside
performance rather than treating missing annotations as biological dissimilarity.

Each ontology arm samples per epoch from the GO-closest 34% of each negative
grade tier. The paired `randwin` control samples from a fixed random 34% window,
isolating GO-based ordering from the effect of narrowing the candidate pool.
Evaluate held-out test AUC as a within-(fraction, seed) delta from this control.

The loss is independently switchable. The default `--loss-type go_ordinal`
adds same-anchor softplus ranking
penalties for high-evidence > weak-evidence and weak-evidence > unobserved,
with the second penalty downweighted by default. The generated control always
uses the *same* loss as the ontology arms. The ordinal-specific weights,
margins, and ranking temperature are declared in `TrainingConfig`; set
`go_ordinal_lambda_weak_unobserved` to zero for the conservative partial-order
ablation. An unobserved STRING edge is not a proven negative. The career-domain
`loss_type="ordinal"` is a different objective and is not used here. Standard
InfoNCE remains available with `--loss-type infonce` but is not part of the
current GO-only run plan.

Prepare the twelve reproducible configs and inspect the run plan:

```bash
.venv/bin/python scripts/run_go_decomposition.py --prepare
```

Run the default 100%-data, five-seed matrix (65 training runs per loss), then score all
completed checkpoints on the fixed test set:

```bash
.venv/bin/python scripts/run_go_decomposition.py --execute
.venv/bin/python scripts/run_go_decomposition.py --evaluate
.venv/bin/python scripts/run_go_decomposition.py --execute --evaluate --loss-type go_ordinal
```

`--fractions` can select existing learning-curve fractions (5, 10, 15, 20, 25,
50, 75, 100). The summary is written to
`results/go_decomposition_go_ordinal/test_summary.json` for the default GO
ordinal objective.
An experiment is not a research
result until its checkpoints have been trained and evaluated; passing unit and
integration tests alone does not establish an accuracy gain.
