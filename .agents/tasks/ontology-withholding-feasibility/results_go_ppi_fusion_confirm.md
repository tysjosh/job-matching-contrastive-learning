# GO/PPI ontology fusion: confirmation and replication results

**Bottom line.** Fusing the text score with GO annotation overlap at inference time gives a consistent ranking gain, and the gain **replicates on new proteins with newly trained models**. It needs each pair's own annotations, and it survives a study-bias control and removal of both known leaks. It does **not** appear on literature-independent interaction labels (HuRI). So the defensible claim is: *GO fusion improves ranking of literature-supported interactions*. It is not evidence that GO adds biology the text misses.

All results below are preregistered (`protocol_go_ppi_fusion.md` v4). Each was computed once, on one remote L40S host, after the code passed a reproduction gate. The full machine report is `results/go_ppi_fusion_confirm/report.md`.

## 1. What was tested

**Method (fixed before any new data was seen).**

> score = (1 − w)·z_a(text) + w·z_a(φ)

- φ = Jaccard overlap of the two proteins' direct GO biological-process terms, using leak-pruned annotations.
- z_a = z-score within each anchor protein's candidate list.
- w ∈ {0, 0.1, …, 1}, chosen per model on validation.
- **This method was chosen by an earlier validation-only search** of 208 fusion variants. The published z(text) + z(simGIC) form is the pre-declared secondary.

**Design.** Inference-time fusion only: the text model never sees GO. All 34 text models were retrained from scratch on **masked text**, with partner gene names removed.

| cohort | models | test anchors | role |
|---|---|---|---|
| A | 19 (5 data fractions × 3–5 seeds) | 1,396 (729 with a hard contrast) | confirmation on the familiar split |
| **Bclean** | 15 (5-fold protein-disjoint CV × 3 seeds) | 4,942 never-analysed anchors (2,669 hard) | **replication** |
| HuRI | cohort-A models | 768 | literature-independent labels (Y2H screen) |

**Co-primary hypotheses** (Holm-corrected, within-anchor):
- H1: pooled AUC (established partner vs everything else);
- H2: average precision;
- H3: hard AUC (established vs weak-evidence partner).

## 2. Main result: replicated on all three co-primaries

Δ = fused − text on the same model, with two-level bootstrap 95% CIs (models × anchors, B = 2000). Every Holm test is rejected at p ≤ 0.001.

| | Cohort A | **Bclean (new anchors)** | HuRI |
|---|---|---|---|
| H1 pooled AUC | +0.031 [+0.024, +0.039] | **+0.022 [+0.018, +0.026]** | +0.001 [−0.002, +0.005], equivalent to 0 |
| H2 AP | +0.050 [+0.036, +0.066] | **+0.040 [+0.031, +0.049]** | −0.001 [−0.007, +0.005], equivalent |
| H3 hard AUC | +0.036 [+0.026, +0.047] | **+0.025 [+0.019, +0.031]** | −0.001 [−0.010, +0.009], equivalent |
| nDCG@10 (secondary) | +0.017 [+0.013, +0.021] | +0.016 [+0.014, +0.018] | +0.002, equivalent |

- **Final label for H1, H2 and H3:** "replicated", with the mandatory qualifier "not supported on literature-independent HuRI labels".
- **Zero-shot MPNet** (no trained head) shows the same gains: Bclean H1 +0.022, H2 +0.035, H3 +0.025. So the effect isn't an artifact of a weak trained model.
- **Pooled A + Bclean** (descriptive): H1 +0.024, H2 +0.042, H3 +0.028.

## 3. Controls: what the gain is and isn't

| control | question | result |
|---|---|---|
| Shuffled GO (anchor annotations deranged within closure-size deciles) | Does the gain need each pair's own annotations? | **Pass, all cohorts.** On shuffled GO, validation chooses w = 0 (Δ ≡ 0). The whole gain is pair-specific (A: X − S = +0.036 on H3). |
| Study bias (C5): text + partner publication count, with GO added on top | Is GO just a proxy for "well-studied protein"? | **Pass in A and Bclean.** GO still adds H3 +0.032 (A) and +0.021 (Bclean) beyond the publication count. On HuRI it adds nothing. |
| With/from annotation leak: unpruned − pruned GO | Do annotations that name the partner inflate it? | **No.** Equivalent to 0 (A: H3 −0.001 [−0.005, +0.003]). |
| Text leak: retrained masked-text models vs the old unmasked checkpoints | Did partner names in the text carry it? | **No.** The gain is *larger* with masked text (H3 +0.007 [+0.003, +0.012]). |
| GO hierarchy: simGIC − exact terms | Does the ontology structure help? | **No; it hurts.** A: H3 −0.015, AP −0.021, with nDCG@10 equal. The is_a-only closure is also slightly worse. |
| Partner-name mention sensitivity | | Dropping mention-flagged positives leaves H3 +0.039. |

## 4. The HuRI result, and why it matters

- **HuRI cohort:** labels come from a systematic yeast-two-hybrid screen that doesn't depend on the literature. On those labels, the fusion gain is **equivalent to zero** on all three hypotheses. This matches the earlier finding that GO separates HuRI interactions much less well (simGIC AUC 0.548) than literature-curated Lit-BM ones (0.674).
- **Within the STRING-labelled test sets,** anchors whose positives are *also* HuRI-supported still gain (Bclean, 209 anchors: H3 +0.028 [+0.008, +0.048]). That is descriptive only.
- **Reading:** GO annotations and "established" STRING evidence share a source, the literature. The fusion reliably exploits that shared signal, and it's specific to the pair rather than a protein-popularity effect (controls C2 and C5). But it doesn't predict interactions found without the literature.

## 5. Secondary findings

**Primary vs secondary fusion (Bclean, paired).** AZ-X is better than the published form on the whole list but worse at the top. Pick by use case:

| metric | AZ-X − GZ-P |
|---|---|
| hard AUC | +0.006 [+0.000, +0.011] |
| AP | +0.015 [+0.007, +0.022] |
| MRR | +0.017 |
| nDCG@10 | −0.008 [−0.010, −0.005] |

**Data efficiency (cohort A).** The hard-AUC gain falls from +0.038 at 10% training data to +0.033 at 100%. The preregistered slope test is **inconclusive** (β = −0.005 per decade, p = 0.11). The slopes for pooled AUC and AP are significantly negative (secondary).

**Gain by anchor degree (cohort A, descriptive).** The gain is larger for low-degree (less-connected) anchors: H3 +0.067 in the bottom STRING-degree tertile vs +0.031 in the top. The difference is significant for H1 (−0.036 [−0.069, −0.009]) but not for H3. Gain by anchor publication count is flat (top − bottom H3 +0.004, n.s.).

## 6. Integrity and deviations (all disclosed in the protocol, §9)

- **Reproduction gate.** Before any new model was scored, the confirmation code reproduced the published fusion and the exploration's test numbers on the old checkpoints.
- **Gate amendments A1 and A2.** The gate needed two amendments for float32 tie-breaking between the Mac and the L40S. Scores 1e-7 apart swapped order, which changed 5 per-anchor nDCG@10 values. Every macro value and selected weight matched throughout, and the tolerances were not loosened.
- **Deviation.** Training started before gate 2 had passed, on the user's instruction. Training doesn't use the gate or scoring code, and no checkpoint was scored until gate 2 passed.
- **Training checks.** 34/34 models passed their per-run checks, with no exclusions.
- **Exposure.** Cohort A's test anchors had been used before, including to choose the method, so cohort A is a confirmation, not an independent test. **Bclean** is the replication: anchors never scored under any fusion before this run.

## 7. Suggested framing for a paper

> Inference-time fusion of a text-similarity model with exact GO biological-process term overlap improves within-anchor ranking of literature-supported protein interactions (replicated on 2,669 new anchors with newly trained models: hard AUC +0.025, AP +0.040). The gain is pair-specific, survives a publication-count control, and does not depend on annotation or text leakage, but it is absent on systematic yeast-two-hybrid interactions, consistent with GO and STRING evidence sharing a literature source.

## Files

- `results/go_ppi_fusion_confirm/report.md`: the full preregistered report: every table, sensitivity and near-tie log.
- `results/go_ppi_fusion_confirm/analysis.json`: all estimates.
- `results/go_ppi_fusion_confirm/pins*.json`, `gate_reproduction*.json`, `run_ledger.json`: the audit trail.
- `.agents/tasks/ontology-withholding-feasibility/protocol_go_ppi_fusion.md`: the preregistration and amendments A1 and A2.
- `.agents/tasks/ontology-withholding-feasibility/results_go_ppi_fusion_best.md`: the exploration that chose the method.
