# MeSH disease signal study

The first clinical-trials decomposition compares **exact condition-descriptor
overlap** with **MeSH hierarchy similarity**. Both use the same symmetric
best-match-average set aggregator. The only scorer difference is whether
different descriptors can be similar through the 2021 MeSH tree.

The disease facet is the converted `condition_uris` field on both topics and
trials. It is not the full trial MeSH union (`mesh_uris`), which would mingle
diseases with interventions and other concepts. Missing condition annotations
fall back to the same fixed random window as the control; unannotated trial
candidates rank after annotated ones. The existing grade-1 ineligible and
grade-0 not-relevant pools stay separate. Each arm keeps the same grade mix,
34% candidate window, stochastic sampling within that window, InfoNCE loss,
and fixed encoder/training settings.

Seven arms are planned: a matched random-window control, and exact/hierarchy
for both negative tiers, ineligible only, and not-relevant only. In a
single-tier arm, the other tier uses the matched random window. This separates
the value of the disease scorer from the pool in which it acts. The five seeds
are 13, 21, 42, 87, and 123, on the same 2021 training fraction (100%).

Run `python scripts/run_mesh_disease_decomposition.py --prepare` to generate
configs and print the plan. Add `--execute` to train all 35 models; this has
not been done. Add `--evaluate` **after training** to score the 2021 validation
set. Its report includes pooled AUC and eligible-vs-ineligible,
eligible-vs-not-relevant, and ineligible-vs-not-relevant AUC, paired by seed.
Pairwise exact-versus-hierarchy comparisons are also reported at each tier
scope. Pick any final arm using only those validation results, then run
`--evaluate-test --selected-arms randwin <chosen-arm>` for the 2022 temporal
holdout. Do not use 2022 results to add, change, or choose arms.

The source-data audit is in
`results/ontology_ceiling/mesh_decomposition_audit_validation.json`. Disease
concepts have high paired coverage in validation; intervention has much lower
coverage and should remain a coverage-qualified pilot. Anatomy and biomarker
studies need additional extraction before comparable arms can be built. The
2022 test split was not used in that audit or in arm selection.

The initial smoke check used 5% of the 2021 training rows, one epoch, and the
2021 validation positives; it tests wiring, not research performance.
