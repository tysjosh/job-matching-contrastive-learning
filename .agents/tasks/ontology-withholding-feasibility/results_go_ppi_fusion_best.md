# GO/PPI fusion: final test scoring of the frozen finalists

Inference-time fusion only. The 15 published GO/PPI text checkpoints (5 fractions × 3 seeds) are scored read-only and nothing is retrained. Test is the published split: 22,341 records and 1,396 anchors. Every Δ is fused − text on the same checkpoint. The 95% CIs come from the trials two-level bootstrap over checkpoints × anchors (`trials_domain.fusion_confirm`, B = 2000, seed 20260101, common random numbers). Test was scored once, by `scripts/go_ppi_fusion_best/score_test.py`, after the finalists (`finalists.json`, 22:02), the plan (`test_plan.json`, 22:21) and its amendment (`test_plan_amendment.json`, 22:24) were frozen.

## Bottom line

- **No leak-free method beats the current best on every metric.** The recommended leak-free method is per-anchor z fusion of exact BP-term overlap on the pruned annotations (F3_P). Against the current best (published z(text)+z(simGIC), nDCG@10-selected) it gives:
  - higher hard AUC (+0.0287 vs +0.0222), but the paired difference is not significant: +0.0065 [−0.0015, +0.0145];
  - higher AP: +0.0390 vs +0.0256, paired +0.0135 [+0.0001, +0.0260];
  - **lower nDCG@10**: +0.0155 vs +0.0259, paired −0.0104 [−0.0144, −0.0067].
- **If top-10 quality is what matters, the best leak-free choice is the published form on pruned φ (PUB_ndcg_P).** It matches the current best on every metric (all paired CIs include 0) and removes the with/from leak.
- **The with/from annotation leak does not carry the gain.** Pruned − unpruned is within ±0.005 on every metric for the published form, and pruned is better for F1. The published number is not inflated by that leak.
- **The gain needs each anchor's own annotations.**
  - On deranged GO (closure-size-decile shuffle), validation picks w = 0 for all 15 checkpoints, so Δ = 0.
  - At the winner's own weights, the shuffled GO hurts: nDCG@10 −0.0117, AP −0.037.
- **The search's CV favourites did not transfer.** The learned HGB stackers that topped the validation CV (F1, F2) owed much of that lead to partners shared across folds.
  - The F2 stacker loses nearly all of its top-10 gain on test: nDCG@10 +0.0036 [−0.0021, +0.0089].
  - The F1 stacker on pruned inputs did well on test, but it was not the pre-declared winner. Treat it as a post-hoc observation (see "Post-hoc").

## 1. Best leak-free method (declared winner F3_P)

Search-log id `239fcb00f573`, the pruned counterpart of finalist F3.

```
score(a, b) = (1 − w) · z_a(text(a, b)) + w · z_a(φ(a, b))

φ(a, b) = |A ∩ B| / |A ∪ B|     A, B = direct BP terms of anchor a and partner b,
                                after with/from pruning (pruned record skill_uris);
                                φ = 0 if either side is empty
z_a(x)  = (x − mean over a's candidates) / SD over a's candidates   (SD 0 → 1)
```

- **Fitted parameters.** w is the only one. Within-anchor z needs no fitted statistics.
- **Weight rule.** w is chosen per checkpoint on the published validation split. It maximises global pooled AUC (grade 2 vs {0, 1}, all validation records) over w ∈ {0.0, 0.1, …, 1.0}, and ties go to the smaller w.
- **Selected w.** 0.4 for 12 checkpoints, 0.3 for 2 (f75_s13, f75_s21), 0.5 for 1 (f10_s13).
- **Anchors with no pruned BP terms.** φ is constant within the anchor, so z_a(φ) = 0 and the ranking is text only. That covers 42 of 1,396 test anchors.
- **Implementation.** `go_ppi_fusion.fusion.AnchorZ` with `phi = "exact_P_pruned"`. φ comes from the protected `GoMatcher(similarity_mode="exact")` on the withholding study's pruned BP index.
- **How it was chosen (before test).** It is the leak-pruned planned variant with the highest protein-disjoint validation CV (§5). The plan originally named F2. The amendment's reason is recorded in `test_plan_amendment.json` and was written before any test scoring.

## 2. Comparison with the current best (test, 15 checkpoints)

Text alone has macro hard AUC 0.7969, pooled 0.8463, easy 0.8887, AP 0.6641, nDCG@10 0.7595 and P@10 0.0766. The anchors with each metric defined are: hard 729, pooled/easy/AP 735, nDCG@10 1,387 and P@10 1,396.

| Δ vs text | current best: PUB_ndcg_U (published, nDCG rule) | **F3_P (winner, leak-free)** | F3_P − current best (paired) |
|---|---|---|---|
| within-anchor hard AUC | +0.0222 [+0.0118, +0.0331], 15/15 | **+0.0287 [+0.0194, +0.0384]**, 15/15 (range +0.021..+0.041) | +0.0065 [−0.0015, +0.0145] |
| within-anchor pooled AUC | +0.0253 [+0.0179, +0.0334] | +0.0271 [+0.0202, +0.0344] | +0.0017 [−0.0041, +0.0077] |
| within-anchor easy AUC | +0.0275 [+0.0205, +0.0354] | +0.0251 [+0.0191, +0.0315] | −0.0024 [−0.0083, +0.0030] |
| AP | +0.0256 [+0.0116, +0.0404] | +0.0390 [+0.0251, +0.0533] | +0.0135 [+0.0001, +0.0260] |
| nDCG@10 | +0.0259 [+0.0217, +0.0304] | +0.0155 [+0.0121, +0.0189] | **−0.0104 [−0.0144, −0.0067]** |
| P@10 | +0.0019 [+0.0012, +0.0026] | +0.0016 [+0.0010, +0.0022] | −0.0003 [−0.0008, +0.0001] |
| MRR | +0.0201 [+0.0049, +0.0354] | +0.0334 [+0.0180, +0.0486] | n/a |
| global hard AUC (point) | +0.0225 | +0.0289 | n/a |

- **The current best is reproduced exactly.** The selected weights equal the published ones under both rules (asserted in the script). The point values equal the published means (hard +0.0222, pooled +0.0253, easy +0.0275, AP +0.0256, nDCG@10 +0.0259, P@10 +0.0019). The CIs match the earlier ones to within 0.0004.
- **Pooled-AUC published rule (PUB_pooled_U).** Hard +0.0195 [+0.0122, +0.0274], nDCG@10 +0.0177 [+0.0148, +0.0211], AP +0.0256, as published.
- **Every finalist and counterpart beats text on hard AUC and AP with CIs above 0, but none beats the current best on nDCG@10.** The full table is in §3.

## 3. Leak-free vs published-comparable (every finalist, both φ versions)

U = unpruned φ, comparable to the published number. P = with/from-pruned φ, leak-free. Each cell is Δ vs text [95% CI] and checkpoints with Δ > 0.

| variant | hard AUC | pooled AUC | AP | nDCG@10 | P@10 |
|---|---|---|---|---|---|
| PUB_ndcg_U (published, current best) | +0.0222 [+0.0118, +0.0331] 15/15 | +0.0253 | +0.0256 | +0.0259 [+0.0217, +0.0304] | +0.0019 |
| PUB_ndcg_P | +0.0226 [+0.0124, +0.0327] 15/15 | +0.0249 | +0.0307 | +0.0244 [+0.0204, +0.0285] | +0.0018 |
| PUB_pooled_U (published) | +0.0195 [+0.0122, +0.0274] 15/15 | +0.0198 | +0.0256 | +0.0177 [+0.0148, +0.0211] | +0.0013 |
| PUB_pooled_P | +0.0209 [+0.0131, +0.0290] 15/15 | +0.0213 | +0.0299 | +0.0185 [+0.0157, +0.0215] | +0.0014 |
| F1_U (HGB stacker, 6 φ + flags + counts) | +0.0239 [+0.0128, +0.0355] 15/15 | +0.0259 | +0.0400 | +0.0219 [+0.0155, +0.0281] 14/15 | +0.0016 |
| F1_P | +0.0310 [+0.0205, +0.0424] 15/15 | +0.0310 | +0.0479 | +0.0205 [+0.0150, +0.0258] | +0.0021 |
| F2_U | +0.0136 [+0.0046, +0.0238] 13/15 | +0.0138 | +0.0200 | +0.0080 [+0.0020, +0.0136] 14/15 | +0.0009 |
| F2_P (HGB stacker, BP simGIC + flags + counts) | +0.0171 [+0.0080, +0.0272] 15/15 | +0.0148 | +0.0245 | +0.0036 [−0.0021, +0.0089] 11/15 | +0.0008 |
| F3_U (anchor-z exact) | +0.0290 [+0.0196, +0.0386] 15/15 | +0.0280 | +0.0404 | +0.0177 [+0.0142, +0.0213] | +0.0017 |
| **F3_P (winner)** | **+0.0287 [+0.0194, +0.0384] 15/15** | +0.0271 | +0.0390 | +0.0155 [+0.0121, +0.0189] | +0.0016 |

**Annotation-leak contribution, U − P (paired):**

| pair | hard | pooled | AP | nDCG@10 |
|---|---|---|---|---|
| PUB_ndcg | −0.0004 [−0.0040, +0.0033] | +0.0004 | −0.0052 [−0.0111, +0.0009] | +0.0015 [−0.0005, +0.0038] |
| PUB_pooled | −0.0014 [−0.0042, +0.0013] | −0.0015 | −0.0043 [−0.0086, −0.0003] | −0.0008 |
| F1 | −0.0070 [−0.0133, −0.0002] | −0.0051 | −0.0078 [−0.0151, −0.0006] | +0.0014 |
| F2 | −0.0035 [−0.0091, +0.0022] | −0.0010 | −0.0045 | +0.0044 [+0.0016, +0.0073] |
| F3 | +0.0002 [−0.0042, +0.0046] | +0.0009 | +0.0014 | +0.0022 [+0.0005, +0.0040] |

The with/from leak (5.9% of test positives have a kept annotation naming the partner) adds at most about +0.002–0.004 nDCG@10. On the AUCs and AP, pruning is neutral or slightly better. So the published number is comparable to a leak-free number, and the pruned published form (PUB_ndcg_P) is a drop-in leak-free replacement.

## 4. Diagnostics (winner only)

**Shuffled GO.** This is the closure-size-decile, candidate-excluding anchor-side derangement of protocol_go_ppi_fusion.md §2.3. Rules (a)–(e) apply: no own set, no candidate's set, no identical set, no candidate partner's set, and simGIC < 0.5. The seed is 11 per file, with a 10,000-draw cap. Each receiver takes the donor's pruned BP set, and its MF/CC follow the donor gene. Partners are unchanged.

| file | strata (size) | accepted draws per stratum | units shuffled / empty kept |
|---|---|---|---|
| validation | 10 × 68 | 7, 2, 9, 4, 1, 4, 4, 1, 1, 5 | 680 / 17 |
| test | 10 × 135–136 | 69, 1, 2, 9, 1, 4, 3, 1, 1, 1 | 1,354 / 42 |

The derangement changed φ_exact on 12.5% of test records; the rest are 0 under both. The number of receiver–candidate pairs where the donor set is close to the candidate's partner set (simGIC ≥ 0.5, descriptive) is 16 on test and 11 on validation.

| arm | hard | pooled | AP | nDCG@10 |
|---|---|---|---|---|
| shuffled, own validation w | w = 0 for 15/15, so Δ ≡ 0 | 0 | 0 | 0 |
| shuffled, at the winner's w | −0.0050 [−0.0105, +0.0011] | −0.0064 [−0.0107, −0.0017] | −0.0369 [−0.0481, −0.0256] | −0.0117 [−0.0153, −0.0083] |
| real − shuffled (own w) | +0.0287 [+0.0194, +0.0384] | +0.0271 | +0.0390 | +0.0155 [+0.0121, +0.0189] |
| real − shuffled (winner's w) | +0.0337 [+0.0237, +0.0442] | +0.0335 | +0.0760 | +0.0273 [+0.0225, +0.0320] |

The strata hold the anchor's annotation density fixed, and partner annotations are untouched. So what the shuffle removes is pair-specific: which terms the anchor and partner share. All of the winner's gain is in that component. Partner annotation density alone, inside this form, gives nothing; with a wrong anchor set it hurts.

**Hierarchy vs exact terms.** The winner already uses exact terms, so the comparison arm is the same anchor-z form with simGIC on the pruned index (`2ef31536a048`), with its own validation w (0.3 ×10, 0.2 ×5).

| | hard | pooled | easy | AP | nDCG@10 |
|---|---|---|---|---|---|
| simGIC arm, Δ vs text | +0.0203 [+0.0134, +0.0278] | +0.0205 | +0.0209 | +0.0275 | +0.0193 [+0.0163, +0.0225] |
| simGIC − exact | −0.0084 [−0.0160, −0.0007] | −0.0065 [−0.0121, −0.0013] | −0.0042 | −0.0115 [−0.0234, +0.0004] | **+0.0038 [+0.0006, +0.0071]** |

The GO hierarchy (simGIC's closure plus IC) does not help the whole-list contrasts. Shared direct terms do better on hard and pooled AUC. The hierarchy helps a little at the top of the ranking (nDCG@10 +0.004). The search's validation CV pointed the same way: exact and IC-direct φ beat simGIC and ancestor φ under every fusion form.

## 5. Fold sensitivity and why the winner changed before test

The review flagged a problem with the CV folds (finding 1). The folds were anchor-grouped, but partners were shared across them: 92.8% of validation records have a partner that also appears in another fold. Test shares no protein with validation. Before scoring test, the same cross-fitting was re-run three ways (validation only; `test_plan.json` → `fold_sensitivity`, and `fold_sensitivity_sizematched.json`):

| variant | anchor-grouped CV composite (search) | size-matched random FIT (same size as protein-disjoint) | protein-disjoint FIT | test composite* |
|---|---|---|---|---|
| F1_U | +0.0282 | +0.0029 | −0.0287 | +0.0279 |
| F2_P | +0.0254 | −0.0032 | −0.0209 | +0.0150 |
| F3_U | +0.0184 | +0.0168 | +0.0157 | +0.0288 |
| F3_P | +0.0158 | +0.0152 | +0.0135 | +0.0276 |
| PUB_ndcg_U | +0.0034 | +0.0021 | +0.0020 | +0.0248 |
| PUB_ndcg_P | +0.0057 | +0.0048 | +0.0063 | +0.0257 |

\* The test composite is the mean of Δ hard, pooled, AP and nDCG@10. It is shown here for comparison only and played no role in any choice.

- **Protein-disjoint FIT sets are small.** They keep only 15–20% of the anchor-grouped FIT records.
- **Two causes of the stacker collapse.** Fit-set size accounts for part of it: F2 goes from +0.025 to −0.003. Partner overlap accounts for about −0.018 more. The non-learned anchor-z forms barely move.
- **The winner was re-declared on this basis.** The new choice is F3_P, the best leak-pruned planned variant under protein-disjoint CV. F2 is still reported.
- **On test, the change holds up.** F2_P lost its top-10 gain. F1, fit on all 5,587 validation records, transferred better than its protein-disjoint CV predicted.
- **The validation CV does not predict test levels.** For the published form it was near 0 (validation caps grade 0 at 20 per anchor; test is uncapped). It is useful for ranking forms, not for estimating gains.

## 6. Gain by training fraction and by anchor annotation count

**By training fraction.** Mean Δ over 3 seeds at F = 10 / 25 / 50 / 75 / 100%.

| variant | hard AUC | nDCG@10 | AP |
|---|---|---|---|
| current best (PUB_ndcg_U) | +0.027 / +0.025 / +0.025 / +0.019 / +0.015 | +0.025 / +0.030 / +0.024 / +0.026 / +0.025 | +0.033 / +0.027 / +0.031 / +0.024 / +0.013 |
| PUB_ndcg_P | +0.028 / +0.024 / +0.025 / +0.020 / +0.016 | +0.025 / +0.025 / +0.024 / +0.025 / +0.025 | +0.039 / +0.033 / +0.037 / +0.025 / +0.020 |
| **F3_P (winner)** | +0.034 / +0.031 / +0.029 / +0.026 / +0.024 | +0.016 / +0.017 / +0.015 / +0.015 / +0.014 | +0.045 / +0.042 / +0.039 / +0.038 / +0.031 |
| F1_P | +0.033 / +0.038 / +0.036 / +0.028 / +0.021 | +0.019 / +0.028 / +0.019 / +0.024 / +0.012 | +0.052 / +0.055 / +0.051 / +0.045 / +0.037 |
| F2_P | +0.020 / +0.021 / +0.019 / +0.015 / +0.010 | +0.006 / +0.012 / −0.001 / +0.005 / −0.003 | +0.029 / +0.028 / +0.025 / +0.030 / +0.011 |

- Every form gains more at low data on hard AUC and AP.
- The winner keeps more of its gain at 100% than the current best: hard +0.024 vs +0.015, AP +0.031 vs +0.013.
- nDCG@10 gains are flat across fractions for the linear and anchor-z forms.
- There are 3 seeds per fraction (f100_s21 stopped at epoch 4), so this is descriptive.

**By anchor annotation count.** The count is the anchor's direct BP terms after pruning; test quartiles are 2 / 5 / 9. Each cell is Δ [95% CI] with the bootstrap restricted to the bin's anchors. The n columns are anchors with hard AUC defined / all anchors in the bin.

| count (n hard / n all) | winner hard | winner nDCG@10 | current best hard | current best nDCG@10 |
|---|---|---|---|---|
| 0 (20 / 42) | 0 (text only) | 0 | +0.023 [−0.026, +0.090] | +0.008 [−0.022, +0.037] |
| 1–2 (166 / 332) | +0.020 [−0.001, +0.041] | +0.011 [+0.004, +0.017] | +0.009 [−0.019, +0.039] | +0.017 [+0.007, +0.028] |
| 3–5 (210 / 419) | +0.032 [+0.015, +0.051] | +0.015 [+0.009, +0.021] | +0.025 [+0.004, +0.047] | +0.026 [+0.018, +0.033] |
| 6–10 (181 / 331) | +0.031 [+0.011, +0.051] | +0.018 [+0.010, +0.026] | +0.023 [+0.006, +0.041] | +0.026 [+0.019, +0.034] |
| 11–25 (113 / 207) | +0.039 [+0.018, +0.064] | +0.019 [+0.010, +0.028] | +0.027 [+0.010, +0.046] | +0.037 [+0.028, +0.046] |
| > 25 (39 / 65) | +0.023 [−0.011, +0.057] | +0.033 [+0.015, +0.054] | +0.049 [+0.018, +0.082] | +0.050 [+0.035, +0.065] |

- The current best's gain rises steadily with annotation count. For nDCG@10 it goes from +0.017 to +0.050, and for hard AUC from +0.009 to +0.049. This is what literature/annotation bias would produce.
- The winner's hard-AUC gain is flatter, about +0.03 from 3 to 25 terms. Its nDCG@10 gain still rises with count.
- On the 42 anchors with no pruned BP terms, the current best's φ still varies within the anchor, because their unpruned terms are with/from-leaked or protein-binding-only.
- The F1_P stacker is the only form with a clear gain at 1–2 terms: hard +0.038 [+0.011, +0.067].

## 7. What didn't help

These conclusions come from the validation search (anchor-grouped CV composite unless stated) and the test results above.

- **MF and CC similarity.** Alone, under the published form, they were negative or about 0 (simGIC_F −0.008, simGIC_C +0.002). BP+MF/CC means and maxes added nothing over BP.
- **The GO hierarchy for whole-list ranking.** Ancestor Jaccard (+0.002) ≈ simGIC (+0.003) < exact (+0.014) ≈ IC-direct (+0.015). The is_a-only closure (+0.002) and Resnik BMA (+0.008, its gain all in nDCG) did not help either. On test, simGIC − exact in the same form gave hard −0.008 and nDCG@10 +0.004.
- **Missing-value handling and weight scope.** Zero, impute and text-only handling, and per-checkpoint, per-fraction or global weights, changed the composite by ≤ 0.002.
- **Rank fusion, RRF and coverage-gated weights.** Each was at or below the matching linear or anchor-z form.
- **Two-φ linear mixes.** The best was +0.016, no better than one exact φ.
- **Linear stackers.** Logistic and pairwise RankNet-style stackers peaked at +0.018 and +0.014.
- **HGB stackers.** They were best in CV but did not survive protein-disjoint fitting (§5). On test, F2 lost its top-10 gain: nDCG@10 +0.004 vs the current best's +0.026.
- **Changing the weight rule alone.** On test, the pooled-AUC rule falls below the current best on nDCG@10 under both φ versions: U −0.0082 [−0.0105, −0.0060], P −0.0074 [−0.0101, −0.0049]. It is also lower on pooled and easy AUC, and level on hard AUC and AP.

## 8. Search summary

- **Search size.** 208 variants were scored on the published validation split only: 697 anchors, 5,587 records, with anchor-grouped 5-fold cross-fitting (fold seed 20261012). The 15 checkpoints were held fixed.
- **Stages:**
  - (0) the published form;
  - (1) a 31-φ screen: BP/MF/CC × simGIC/ancestor/exact/IC-direct, Resnik, is_a-only, multi-aspect, each U and P;
  - (2) the inner selection metric and grid step (global pooled AUC won);
  - (3) missing handling × weight scope;
  - (4) rank, RRF, anchor-z and coverage forms;
  - (5) two-φ linear;
  - (6) LR, pairwise and HGB stackers.
- **Finalists.** Three were frozen by a fixed rule (F1: best overall, F2: best leak-pruned, F3: best ≤1-parameter non-learned). See `finalists.json`.
- **Optimism.** Finalists' CV composites are best-of-208 values. The test Δ above is the estimate.
- **Reproduction gate.** It passed before the search: 15/15 weights equal under both rules, and 251,820 published test leaves are bit-identical (`reproduction.json`). The unit tests pass (16/16, `pytest_summary.txt`).
- **Plan and amendment.** Both are pre-test and validation-only. They fixed the 10 scored variants (3 finalists, their φ-version counterparts, and the published form under both rules × U/P) and the winner, and then re-declared the winner from protein-disjoint CV (§5).

## 9. Post-hoc

- **F1_P was the strongest on test across the whole list, but it was not the declared winner.**
  - Δ vs text: hard +0.0310, pooled +0.0310, AP +0.0479, nDCG@10 +0.0205.
  - vs the current best: hard +0.0088 [+0.0009, +0.0166], AP +0.0223 [+0.0114, +0.0329], nDCG@10 −0.0054 [−0.0106, −0.0007].
- **Its CV did not predict this.** Its protein-disjoint CV was the worst of all (−0.030), and picking it now would be selection on test.
- **A learned stacker could pay off, but needs new test data.** It would have to be confirmed on new anchors, for example the protocol's cohort B or the HuRI cohort.

## 10. Limitations

- **Test exposure.** The 1,396 test anchors were already used in the earlier post-hoc fusion analysis (findings_go_ppi_fusion.md §5, Tables G1–G3) and the reproduction gate. Their published fusion results were known when the search was designed, though the search itself never read test. Today's test numbers are a second look at already-analysed anchors, not a replication. 10 variants were scored on test.
- **Text-side partner-symbol leak.** It remains in these checkpoints: 11.1% of test positives have the partner's symbol in the anchor's encoder text. The text baseline, and so every Δ, comes from models trained on that text. Removing it needs retraining on the masked `encoder_view`, which is out of scope here.
- **Literature/study bias.**
  - simGIC separates interacting pairs from matched non-pairs at AUC 0.674 on Lit-BM (literature-curated) but only 0.548 on HuRI (systematic screen).
  - The shuffled control rules out anchor-level and partner-level annotation density as the source of the winner's gain. It cannot rule out pair-level co-study: two proteins studied in the same papers share terms and get "established" STRING evidence.
  - The current best's gain rises with annotation count (§6).
  - No HuRI-only evaluation was run. The HuRI stratum of this test set has 58 anchors, too few for inference.
- **The validation CV is a weak guide.** Validation caps grade 0 at 20 per anchor while test is uncapped, and the search's anchor-grouped CV shared partners across folds. Validation-CV magnitudes do not predict test levels.
- **No preregistration or replication.** Everything here is exploratory, though test was scored once under a frozen plan.

## 11. Files

Created in this phase:

- `go_ppi_fusion/final.py`: per-anchor Δ, the bootstrap wrapper (imports `trials_domain.fusion_confirm`), the shuffled-GO derangement, and protein-disjoint CV.
- `scripts/go_ppi_fusion_best/freeze_test_plan.py`: pre-test plan freeze and protein-disjoint fold sensitivity (search mode).
- `scripts/go_ppi_fusion_best/fold_sensitivity_sizematched.py`: the size-matched FIT control (search mode).
- `scripts/go_ppi_fusion_best/score_test.py`: the one-time test scoring and winner diagnostics. Its `--dry-run` mode runs on validation in search mode.
- `results/go_ppi_fusion_best/test_plan.json`, `fold_sensitivity_sizematched.json`, `test_plan_amendment.json`: frozen before test.
- `results/go_ppi_fusion_best/test_results.json`: all test summaries, contrasts, by-fraction and by-annotation results, and the shuffle manifest.
- `results/go_ppi_fusion_best/score_test.log`: stdout of the test run.
- `results/go_ppi_fusion_best/pytest_summary.txt`: 16 passed.
- `/tmp/goppi_fusion_best/scored_deltas.npz`: per-checkpoint × per-anchor Δ arrays. Scratch, not in the repo.

Edited (new code only): `go_ppi_fusion/phi.py` docstring (review finding 5) and the `go_ppi_fusion/__init__.py` module list. No protected path was modified and nothing was committed.

Commands (CPU only, `CUDA_VISIBLE_DEVICES=""`):

```
.venv/bin/python scripts/go_ppi_fusion_best/freeze_test_plan.py              # 93 s, validation only
.venv/bin/python scripts/go_ppi_fusion_best/fold_sensitivity_sizematched.py  # validation only
.venv/bin/python scripts/go_ppi_fusion_best/score_test.py --dry-run          # validation stands in for test
.venv/bin/python scripts/go_ppi_fusion_best/score_test.py                    # the single test run, 63 s
.venv/bin/python -m pytest go_ppi_fusion/tests -q                            # 16 passed
```
