# Where MeSH helps in TREC-CT clinical-trial matching: findings

This was a read-only investigation. Repo `/Users/olukotunjosh/Downloads/CDCL`, HEAD `c911188`. I modified no repo file and trained nothing. I re-scored 93 existing checkpoints on CPU, using the frozen MPNet embeddings in a `/tmp` copy of `results/publication_phase1_fixedval/prepared_embeddings/trials`. All scratch work is in `/tmp/trials_helps/`:
- `score.py`: per-pair scores, written to `scores.npz`
- `lib.py`: metrics and bootstraps
- `a2_fusion.py`: fusion variants
- `a3_matched.py`: negative-selection and label-budget contrasts
- `a4_strata.py`, `a4b_textnov.py`: patient strata
- `a5_hard.py`: eligibility decomposition
- `a6_twolevel.py`: two-level bootstrap
- `a7_controls.py`, `a7b.py`: exact-match and shuffled controls

Before analysing anything, I checked the re-scoring against the stored results. It reproduces them exactly: pooled and hard AUC match `trials_fusion_learning_curve.json` and `trials_test_summary.json` to four decimals [ran].

Evidence tags: **[read]** from code or files, **[ran]** computed here, **[inferred]** reasoned but not verified.

"Macro" means a metric computed within each of the 50 test patients (2022 topics) and then averaged. Every CI is a 95% patient-level paired bootstrap (5,000 resamples) of per-patient deltas, averaged over seeds and checkpoints first. Where stated, a two-level bootstrap also resamples checkpoints. "frac>0" is the share of patients who improve.

## Summary answer

1. **MeSH helps as an inference-time relevance prior. This is supported.** Fusing z(text) with z(φ_MeSH) on the same checkpoint, with the weight chosen on 2021 validation by nDCG@10, gives these within-patient gains over 15 baseline checkpoints (5 fractions × 3 seeds):
   - pooled AUC **+0.013 [+0.006, +0.020]**;
   - AP **+0.017 [+0.006, +0.029]**;
   - ineligible-vs-not-relevant AUC **+0.014 [+0.007, +0.022]**.

   Those are the two-level CIs. The controls point the same way:
   - Shuffling topic MeSH sets across patients removes the gain and turns it negative (pooled −0.002, nDCG −0.014).
   - The MeSH *hierarchy* adds a small amount on top of exact descriptor overlap: +0.005 [+0.002, +0.008] pooled, +0.006 [+0.001, +0.011] AP, at w = 0.2.
   - The gain is about the same at every training fraction.
2. **MeSH does not help eligibility (eligible vs ineligible within a patient). This is not supported.** The earlier "fusion helps hard AUC (+0.02 to +0.044)" result is a cross-patient artifact:
   - Only **2.4%** of global eligible × ineligible pairs belong to the same patient.
   - Within-patient hard AUC is 0.511 for φ alone and −0.001 to −0.010 under fusion. The grouped-CV within-patient increment of φ is −0.0005 [−0.022, +0.020] over zero-shot text and −0.007 [−0.033, +0.017] over trained text.
   - GO/PPI is different. There, the fusion hard gain survives within anchor (macro +0.015 to +0.022).
3. **MeSH-guided negatives do not help, and slightly hurt topical ranking. The null is now well-bounded.** Across all 24 matched (MeSH-window − random-window) pairs (v1 + v2, 5–20%):
   - macro hard **+0.002 [−0.005, +0.009]**;
   - macro pooled **−0.0075 [−0.013, −0.002]**.

   The v1/v2 sign flips come from **training non-determinism**. v1 and v2 use byte-identical configs and seeds, yet their test scores correlate only r ≈ 0.66–0.73. The sd of the hard-AUC difference between reruns (0.033) equals the sd of the arm difference (0.032).
4. **Training itself is the bigger problem, and it limits every training-time ontology claim.** Every trained projection (all fractions) underperforms the zero-shot frozen MPNet cosine on 2022 test:
   - **nDCG@10 by −0.07 to −0.13** (CIs exclude 0);
   - pooled per-patient AUC by −0.01 to −0.02.

   The text learning curve is flat or declining (macro nDCG 0.454 / 0.468 / 0.470 / 0.467 / 0.420 at 10/25/50/75/100%). So there is no data-efficiency curve to shorten, and an "examples-to-threshold" figure cannot be estimated.
5. **Not supported:**
   - a novel-condition moderation (stratifier directions disagree and no effect survives multiplicity);
   - the label-budget "ontology" advantage (low_ontology − low_random macro pooled +0.007 [−0.007, +0.021]);
   - MeSH fusion improving nDCG@10 over the zero-shot encoder (−0.008 [−0.031, +0.016]);
   - criterion-level MeSH fusion (`trials_criterion_nested`: macro-F1 +0.007 [−0.002, +0.018]).

**One-line framing:** in clinical-trial matching, MeSH works as a *topical relevance prior at inference time*. It is not an eligibility signal and not a useful training-time negative sampler.

---

## A. Results inventory

| path | design | ckpts / seeds | on frozen 2022 test? | status / quality |
|---|---|---|---|---|
| `results/publication_phase1_fixedval/checkpoints/trials_baseline_f{10,25,50,75,100}_s{42,13,21}` + `trials_fusion_learning_curve.json` (select pooled AUC), `_ndcg_selected.json` | uniform-negative InfoNCE learning curve; post-hoc z-fusion text + φ (w chosen on validation) | 15 / 3 | yes (35,394 rows, sha 4030cc85…) | **Main usable asset.** Complete. Fixed validation negatives. The summary table reports only global AUC and seed sd [read]. |
| `results/trials_negative_selection_matched/` (v1) | randwin vs ontneg_stoch (MeSH 34% window), fractions 5/10/15/20 | 24 / 3 | yes | Single-factor audit passes (`lc_manifest.single_factor_audit`) [read]. |
| `results/trials_negative_selection_matched_v2/` | **identical configs to v1** (diff of `training_config.json` is only the `train_result` string) [ran] | 24 / 3 | yes | A rerun. Its value is as a non-determinism probe (§B4). |
| `results/negative_selection_fixedval/trials_{baseline,ontneg}_f{5..20}` | uniform vs MeSH deterministic, cap 100 | 24 / 3 | yes | **Confounded** with diversity (86% of negatives replayed across epochs) and the NCT-sorted cap (prior audit). |
| `results/label_budget/{full,low_ontology,low_random}_s{42,13}` | unjudged-corpus negatives, low-label arms | 6 / 2 | yes (`eval_test2022`) | **Confounded**: unjudged ≠ negative, older code, 2 seeds. The summary json has s42 only. |
| `results/ontology_ceiling/fusion_probe_trials.json` | fusion on the label_budget checkpoints, **select_on = hard**, 6,000-row test subsample | 6 / 2 | subsample | The selection rule picked **w = 1.0 (MeSH only) in 5 of 6** checkpoints [ran]. That alone explains easy −0.068 / pooled −0.056. |
| `…/phase1_single_stage_fusion_trials.json` | same checkpoints, select_on = pooled | 6 | subsample | hard +0.055, easy +0.021, pooled +0.024 (global) [read]. |
| `…/two_stage_trials.json` | rerank top-k1 with φ, selected on validation nDCG | 6 | yes | nDCG two-stage − text +0.003, P@10 +0.003 [read]. Null. |
| `…/fusion_reselect.json` | re-selection summary | 6 | subsample | trials, pooled-select: hard +0.045, pooled +0.020, 5/6 up [read]. |
| `…/text_vs_ontology.json` (trials) | raw-MPNet cosine vs φ, 3,000 sampled test pairs, CV increment | — | sample | Hard increment +0.122 [0.041, 0.174] [read]. This is **global**, so it is the same cross-patient artifact (§B2). |
| `…/mesh_hier_vs_set_test.json`, `mesh_decomposition_audit_validation.json` | φ separation by measure | — | 2,668-pair sample | Global hard AUC 0.61–0.63 [read]. Cross-patient. |
| `results/trials_criterion_nested/` (+ `_pilot`) | TrialGPT criterion annotations, nested patient-grouped CV, text+NLI vs MeSH fusions | 5 systems, seeds 11/22/33 | different dataset | Patient bootstrap present: MeSH-hierarchy fusion macro-F1 −0.000 [−0.007, +0.008]; gated +0.007 [−0.002, +0.018]; shuffled control −0.002 [read]. Null. |
| `results/mesh_soft_sampling{,_smoke}/` | V2 soft sampling, 2021 CV folds | only `uniform_s42` per fold (+ smoke arms) | no | Not executed beyond smoke. `validation2021_summary.json` is absent [ran]. |
| `results/research_runs/ER-DEN__trec_ct__s*` | ORCA denominator | — | — | `training_config.json` only. Never ran. |
| `results/lc_single_factor/trials_*` | — | none | — | Absent [ran]. |
| `trials_domain/MESH_SOFT_STUDY.md`, `MESH_DISEASE_STUDY.md` | protocols | — | — | No results [read]. |

## B. Re-analysed effects with patient-level CIs

### B1. Fusion on the 15 baseline checkpoints [ran, `a2_fusion.py`, `a6_twolevel.py`]

Δ = fused − text on the *same* checkpoint, macro over 50 patients:

| variant (weight rule) | weights chosen | pooled AUC | hard (2 vs 1) | g1 vs g0 | AP | nDCG@10 | P@10 |
|---|---|---|---|---|---|---|---|
| validation pooled-AUC (published rule) | 0.2–0.7 | **+0.017 [+0.002, +0.030]** | −0.007 [−0.030, +0.011] | **+0.021 [+0.009, +0.035]** | +0.014 [−0.007, +0.036] | −0.011 [−0.057, +0.035] | −0.014 [−0.046, +0.018] |
| validation hard-AUC (old probe rule) | 0.1–1.0 | **−0.027 [−0.052, −0.003]** | −0.022 | −0.022 | −0.021 | **−0.073 [−0.116, −0.032]** | −0.068 |
| **validation nDCG@10 (recommended)** | 0.0–0.4 | **+0.013 [+0.006, +0.020]** | −0.001 [−0.013, +0.009] | **+0.014 [+0.007, +0.022]** | **+0.017 [+0.006, +0.029]** | +0.014 [−0.008, +0.038] | +0.011 [−0.006, +0.028] |
| within-patient rank fusion (validation macro pooled) | 0.2–0.4 | **+0.013 [+0.003, +0.021]** | −0.003 | **+0.014 [+0.004, +0.024]** | +0.013 [−0.004, +0.030] | −0.005 | −0.005 |
| fixed w = 0.2 (post hoc) | 0.2 | **+0.016 [+0.009, +0.022]** | +0.000 | **+0.017 [+0.011, +0.024]** | **+0.023 [+0.012, +0.034]** | **+0.028 [+0.009, +0.048]** | **+0.026 [+0.007, +0.044]** |
| hard filter: push φ = 0 trials down | — | **−0.026** | +0.001 | −0.035 | −0.010 | −0.003 | −0.003 |

The first three rows use two-level CIs (checkpoints × patients); the rest use patient-level CIs.

- **The weight rule decides the sign.** Validation has grade-0 capped at 60 per topic, so validation pooled/hard AUC is dominated by grade-1 negatives, where φ is uninformative. Those rules therefore choose large w (MeSH-heavy), which hurts top-10. nDCG-based selection chooses w ≤ 0.4 and is the only rule aligned with test [ran, inferred].
- **No variant improves within-patient hard AUC.**
- **The gain is flat across fractions.** Under the published rule, macro pooled Δ is +0.012 / +0.016 / +0.018 / +0.018 / +0.019 at 10–100%. nDCG@10 Δ is only positive at 100% (+0.037 [+0.008, +0.071]), where the text model is worst. Fusion does *not* concentrate at low data [ran].

**Controls** [ran, `a7_controls.py`, `a7b.py`]:

| φ variant | φ alone, macro pooled | fusion Δ pooled (nDCG-selected) | fusion Δ nDCG@10 (w = 0.2) |
|---|---|---|---|
| hierarchy (production) | 0.696 | +0.013 [+0.006, +0.018] | +0.028 [+0.009, +0.048] |
| exact descriptor overlap | 0.638 | +0.009 [+0.002, +0.016] | +0.024 [+0.004, +0.044] |
| topic URIs shuffled across patients | 0.503 | **−0.002 [−0.004, −0.001]** | **−0.065 [−0.095, −0.037]** |

- Paired hierarchy − exact at w = 0.2: pooled +0.005 [+0.002, +0.008], AP +0.006 [+0.001, +0.011], nDCG +0.005 [−0.003, +0.012].

**Zero-shot reference.** The frozen MPNet cosine, with no training, scores macro pooled 0.795, hard 0.553, nDCG@10 0.551 and P@10 0.466. Its global pooled AUC is 0.788.
- Trained text minus zero-shot, per fraction: nDCG@10 −0.097, −0.083, −0.081, −0.084, −0.130 (all CIs exclude 0) and macro hard +0.010 to +0.029 (significant only at 75%).
- Zero-shot + φ, w = 0.2 (also the nDCG-selected weight): pooled **+0.012 [+0.005, +0.018]**, AP **+0.011 [+0.002, +0.020]**, nDCG@10 −0.008 [−0.031, +0.016].
- Zero-shot + φ with the pooled-AUC-selected weight (0.6): nDCG −0.130.

### B2. Eligibility boundary: where the "hard gain" comes from [ran, `a5_hard.py`]

The global hard AUC splits into within-patient and between-patient pairs (mean of 15 checkpoints):

| score | within-patient (2.4% of pairs) | between-patient (97.6%) |
|---|---|---|
| text | 0.549 | 0.554 |
| fusion w = 0.2 | 0.554 | 0.570 |
| fusion w = 0.4 | 0.553 | 0.585 |
| φ alone | **0.511** | 0.579 |
| zero-shot cosine | 0.552 | 0.526 |

- **Mechanism.** φ ranks one patient's eligible trials above *another* patient's ineligible trials. It does this because patients with high pool-mean φ tend to have more eligible trials (Spearman 0.24, p = 0.09).
- **Within patient,** mean φ of eligible minus ineligible trials is +0.022 [−0.003, +0.045]. That is tiny next to the φ sd of about 0.2.
- **Grouped-CV logistic within-patient increment** (patient-centred features, 5 patient folds):
  - hard: +0.004 / −0.007 / −0.0005 over trained f25 / trained f100 / zero-shot (all CIs span 0);
  - pooled: **+0.022 [+0.010, +0.033]** over trained f100 and **+0.014 [+0.005, +0.023]** over zero-shot.
- **Is the hard gain a by-product of pushing off-topic trials down?** Yes. All of the within-patient benefit is in grade ≥1 vs grade 0 (g1 vs g0 +0.014 to +0.025). None of it is in 2 vs 1.
- **Cross-domain check.** In the GO fusion file the hard gain is +0.015 to +0.022 both globally *and* per anchor. Trials: global +0.020 to +0.039, per patient −0.006 to −0.010 [ran over `publication_phase1_fixedval/*_fusion_learning_curve.json`].

### B3. Negative selection, patient-level [ran, `a3_matched.py`]

Δ = MeSH stochastic window − random window. Macro CIs are patient-level; global CIs are patient-cluster bootstraps (400 resamples).

| fraction | runs | macro hard | macro pooled | macro nDCG@10 | global hard |
|---|---|---|---|---|---|
| 5% | v1+v2 (6 pairs) | +0.000 [−0.012, +0.012] | **−0.013 [−0.023, −0.004]** | −0.005 [−0.025, +0.013] | +0.004 [−0.020, +0.028] |
| 10% | v1 / v2 | **+0.020 [+0.004, +0.036]** / −0.004 [−0.022, +0.014] | +0.007 / −0.009 | +0.006 / −0.003 | **+0.030 [+0.004, +0.062]** / −0.025 [−0.054, +0.007] |
| 15% | v1 / v2 | +0.001 / +0.002 | −0.009 / −0.006 | −0.002 / +0.014 | −0.043 [−0.092, +0.007] / **+0.040 [+0.006, +0.081]** |
| 20% | v1+v2 | −0.001 [−0.011, +0.009] | −0.008 [−0.016, +0.001] | −0.012 [−0.032, +0.007] | −0.012 [−0.032, +0.013] |
| **all** | 24 pairs | **+0.002 [−0.005, +0.009]** | **−0.0075 [−0.013, −0.002]** | −0.002 [−0.012, +0.008] | −0.002 (point) |
| deterministic MeSH vs uniform (`negative_selection_fixedval`, confounded) | 12 pairs | −0.015 [−0.030, +0.000] | **−0.026 [−0.035, −0.016]** | **−0.026 [−0.049, −0.004]** | +0.007 |

Two "significant" single-version cells (v1 10% and v2 15% global hard) have opposite-sign twins in the identical-config rerun. That is exactly what non-determinism predicts.

**Label budget** (2 seeds; confounded) [ran]:
- low_ontology − low_random: macro pooled +0.007 [−0.007, +0.021], hard +0.005 [−0.014, +0.025], nDCG +0.008 [−0.028, +0.045].
- Global hard −0.001 [−0.050, +0.034].
- Absolute values on all 2 × 2 checkpoints: global pooled full 0.733 / low_ontology 0.754 / low_random 0.752. The 0.705 / 0.749 / 0.766 figures in `trials_test2022_summary.json` are s42-only.
- Both low arms beat "full" on pooled because "full" is the older code path. They lose on hard: low − full −0.019 / −0.024, CIs exclude 0.

### B4. Why v1 and v2 disagree [ran, read]
- The configs differ only in the `train_result` string. The device is `cpu`. `training_seed` and `split_seed` are equal.
- Yet the epoch-0 training loss already differs (1.897 vs 1.955 for randwin f10 s13).
- Per-pair test scores for the same seed and arm correlate at only **r = 0.66–0.73** between v1 and v2.
- Across all 24 rerun pairs, sd(v2 − v1 global hard) = **0.033**. The sd of the arm difference is 0.032.
- So the seed does not fix the run. Rerun noise is as large as any arm effect measured so far.
- The source is not identified. Candidates [inferred]: hash-randomised set or dict iteration (`PYTHONHASHSEED`) in pool or batch construction, or unseeded DataLoader or selector RNG paths.

### B5. Learning-curve view [ran]

Macro pooled / macro hard / nDCG@10, seed means:

| fraction | uniform baseline | randwin (v1+v2) | MeSH window (v1+v2) |
|---|---|---|---|
| 5% | — | 0.777 / 0.557 / 0.466 | 0.764 / 0.557 / 0.460 |
| 10% | 0.772 / 0.563 / 0.454 | 0.772 / 0.562 / 0.455 | 0.771 / 0.570 / 0.457 |
| 20–25% | 0.783 / 0.574 / 0.468 | 0.781 / 0.574 / 0.465 | 0.773 / 0.573 / 0.453 |
| 50% | 0.784 / 0.572 / 0.470 | — | — |
| 100% | 0.776 / 0.578 / 0.420 | — | — |
| zero-shot | 0.795 / 0.553 / 0.551 | | |

- The curve has no slope to compress: 5% of positives (220) performs like 100% (4,408).
- Neither MeSH arm, nor fusion, closes a gap that does not exist.

### B6. Patient strata (closest analogue of low-n(o)) [ran, `a4_strata.py`, `a4b_textnov.py`]

All strata are defined from train data and label-free test inputs only:
- `mesh_nov` = 1 − max φ(test topic conditions, any train topic conditions), excluding the 5 patients with no extracted conditions;
- `desc_in_train_pos_trials` = share of a patient's condition descriptors that appear on any train-positive trial;
- `text_nov` = 1 − max MPNet cosine to the 60 train topics;
- `n_conds`;
- `pool_phi_pos`.

The outcomes are per-patient fusion Δ (w = 0.2), MeSH-negatives Δ and trained − zero-shot.

| stratifier | outcome | novel vs familiar tercile (diff CI) | Spearman ρ [CI] |
|---|---|---|---|
| mesh_nov (non-empty) | fusion Δ pooled | +0.022 vs +0.012 ([−0.006, +0.028]) | +0.18 [−0.12, +0.43] |
| mesh_nov (non-empty) | fusion Δ nDCG | +0.052 vs +0.034 ([−0.040, +0.077]) | +0.17 [−0.13, +0.47] |
| desc_in_train_pos_trials (low = novel) | fusion Δ pooled | novel +0.025 vs familiar +0.008 ([−0.034, −0.003]) | −0.27 [−0.50, +0.02] |
| text_nov | fusion Δ nDCG | **novel +0.018 vs familiar +0.057** ([−0.076, −0.000]), the *opposite* direction | −0.28 [−0.53, +0.01] |
| mesh_nov (non-empty) | MeSH-negatives Δ pooled | −0.001 vs −0.014 ([−0.000, +0.028]) | +0.20 [−0.09, +0.47] |
| text_nov | trained − zero-shot pooled | novel +0.006 vs familiar −0.052 ([+0.028, +0.089]) | +0.45 [+0.19, +0.67] |

- **MeSH-based novelty** leans toward "fusion helps more for MeSH-novel patients" (one CI excludes 0).
- **Text-based novelty** points the other way. Across 5 stratifiers × 11 outcomes, nothing survives multiplicity.
- **Verdict:** no credible novel-condition moderation.
- **Side finding:** training hurts most for patients who are *textually close* to the 2021 train topics (ρ = +0.45). That fits over-fitting to 2021 paraphrase pairs [inferred].

### B7. Realised negative composition and the mechanism of the null

**Composition** [read: prior audit in `findings_trials.md` §D; ran: losses]:
- Both arms have the same grade-1 share (0.556).
- MeSH-window negatives share a condition descriptor with the positive more often (0.35 vs 0.28) and have higher φ(topic, negative) (0.31 vs 0.19).
- They are all judged negatives, so there are no unjudged false negatives.

**Losses** barely change:

| | randwin | MeSH window |
|---|---|---|
| final train InfoNCE, v1 | 1.650 | 1.664 |
| final train InfoNCE, v2 | 1.659 | 1.686 |
| best validation loss, v1 | 2.218 | 2.211 |
| best validation loss, v2 | 2.191 | 2.196 |

Chance level is ln 8 = 2.08.

**Mechanism** [inferred from these plus B2]:
- MeSH-close negatives are barely harder for the projection.
- The only extra signal they carry is "push down topically similar grade-0 and grade-1 trials". That is the opposite of what helps at test, where topicality is the main within-patient signal. Hence a small pooled-AUC loss (−0.0075).
- They carry no eligibility information, because φ is at chance for 2 vs 1 within patient. Hence the null on hard.
- I did not re-measure gradient scale (no ontology loss component exists to compare against).

## C. Ranked claims

**Supported**

1. **MeSH φ adds topical-relevance information beyond text, at inference time.**
   - Evidence: validation-nDCG-selected fusion gives macro pooled +0.013 [+0.006, +0.020], AP +0.017 [+0.006, +0.029] and g1-vs-g0 +0.014 [+0.007, +0.022] (two-level, 15 checkpoints × 50 patients). The grouped-CV within-patient increment is +0.022 over trained text and +0.014 over zero-shot. Shuffling the topic URIs abolishes it.
   - Caveat: the effect is small, and nDCG@10 and P@10 are not significant under validation selection. There is only one test collection (50 synthetic patients), and the topic-side MeSH extraction is noisy (about 60% of primary conditions captured in 2022).
2. **MeSH does not inform the eligibility boundary.**
   - Evidence: within-patient hard AUC of φ is 0.511, the fusion within-patient hard Δ is ≈0 under every rule, and the grouped-CV hard increment is ≈0. The published "+0.02–0.044 hard AUC" is between-patient (97.6% of pairs).
   - Caveat: this is a negative claim, and well powered. Ineligibility in TREC is about criteria (age, labs, exclusions), not about the condition.
3. **MeSH-guided negatives do not improve the learned model.**
   - Evidence: macro hard +0.002 [−0.005, +0.009] and macro pooled −0.0075 [−0.013, −0.002] over 24 matched pairs.
   - Caveat: training non-determinism (B4) inflates variance. The arm effect is bounded to |Δhard| ≲ 0.01, not to zero.
4. **The prior trials sign flips are noise, not effects.**
   - Evidence: identical-config reruns, score r ≈ 0.7, rerun sd equal to arm sd.

**Suggestive**

5. **The hierarchy matters, not just shared descriptors.** Hierarchy − exact fusion at w = 0.2: pooled +0.005 [+0.002, +0.008], AP +0.006 [+0.001, +0.011]. Caveat: w was not preregistered and the effect is tiny. With the nDCG-selected weight the difference shrinks (+0.013 vs +0.009, unpaired).
6. **Low-weight fusion improves top-10.** w = 0.2 gives nDCG@10 +0.028 [+0.009, +0.048] and P@10 +0.026 [+0.007, +0.044]. Caveats: post hoc weight; the validation-selected version gives +0.014 [−0.008, +0.038]; and it is relative to a trained model that is itself 0.08–0.13 nDCG worse than zero-shot. Over zero-shot it is −0.008.
7. **Fusion helps more for MeSH-novel patients.** One stratifier's CI excludes 0, but another stratifier reverses the direction. Treat as hypothesis-generating only.

**Not supported**

8. Ontology helps eligibility / the "hard" contrast (B2).
9. Ontology helps data efficiency or low-label regimes. The curve is flat, the fusion gain is constant across fractions, and the label-budget difference is null and confounded.
10. Ontology-guided negatives or any training-time MeSH use helps (B3; label budget; ORCA never ran; no ontology-loss arm exists).
11. Ontology beats or improves the zero-shot encoder on ranking (nDCG −0.008, P@10 −0.024 [−0.052, +0.004]).
12. Criterion-level MeSH fusion helps (`trials_criterion_nested`, null with a shuffled control).
13. Per-target withholding (n × S) conclusions on trials (prior report: about 8 targets with 1–4 patients).

## D. Cross-domain framing

| | career (v7) | GO/PPI | clinical trials |
|---|---|---|---|
| φ alone, hard contrast | about 0.55 (weak) | 0.82 (strong) | 0.511 within patient, 0.58 between patients (eligibility-blind) |
| inference-time fusion | small (+0.057 CV increment on train-split pairs, global) | **+0.015–0.023 hard, holds within anchor** | **+0.013–0.017 pooled/AP within patient, relevance only**; hard gain is a cross-patient artifact |
| guided negatives | null / confounded | null on hard; lowers GO relation preservation | null on hard, −0.0075 pooled |
| ontology-in-loss | ordinal φ-margin (career-only) | none yet | none |

Suggested wording:
- "Across three domains, the ontology helps when used as an inference-time prior on the score. Its benefit is graded by how much the ontology encodes the target relation: strong for PPI (GO co-function ≈ interaction), relevance-only for trial matching (MeSH ≈ topic, not eligibility), weak for careers."
- "Used only to choose training negatives, it does not help in any domain."
- Do **not** claim "ontology helps novel or low-data conditions" for trials.
- Do not reuse global hard AUC as the trials eligibility metric. Any GO, trials or career claim based on *global* hard AUC should be re-checked within query; GO passes this check.

## E. Minimal new experiments (cheap, A100 or even CPU)

**E1. Preregistered inference-time fusion confirmation** (turns claims 5 and 6 into supported; no training).
- Reuse the 15 baseline checkpoints plus zero-shot, and add seeds 87 and 123 at fractions 25 and 100. That is 4 new runs of `config/lc_trials_baseline.json` via `scripts/run_learning_curve_point.py`, about 2 min each.
- Freeze before looking:
  - weight rule = validation nDCG@10 over {0, 0.1, …, 0.5};
  - primary metrics = macro pooled AUC and AP;
  - secondary = nDCG@10 and P@10;
  - controls = exact-mode φ and shuffled-topic φ (seed 11);
  - CI = two-level bootstrap;
  - success = lower CI > 0 on the primary metrics, and hierarchy − exact > 0.
- **To implement:**
  - a `--phi-mode {hierarchy,exact,shuffled}` and `--export-per-pair` option in a *copy* of `scripts/evaluate_publication_fusion.py`;
  - a macro-per-patient bootstrap aggregator (port `/tmp/trials_helps/lib.py`);
  - a `--zero-shot` scorer (identity projection).
- **Replication on 2021:** a 5-fold patient-disjoint CV over the 75 2021 topics, with paraphrase pairs kept in the same fold. Fold assignment reuses `trials_domain/data_splitter` logic. Training is 5 folds × 3 seeds (13 / 21 / 42) at full data. Fusion is evaluated on each held-out fold. That gives a second, independent cohort of 75 patients.

**E2. Determinism fix, then a definitive negative-selection bound** (turns claim 3 into "|Δ| < 0.005").
- Set `PYTHONHASHSEED=0`, `torch.use_deterministic_algorithms(True)`, seed the selector and DataLoader explicitly, and verify that two reruns give identical `epoch_losses`.
- Then rerun randwin vs ontneg_stoch at f = 10 and 100 with seeds 13 / 21 / 42 / 87 / 123 (20 runs, about 20 min).
- Primary metric: macro pooled and hard with patient × seed bootstrap.
- To implement: determinism flags in the trainer entry point (`run_learning_curve_point.py` is under concurrent edit, so coordinate).

**E3. Fix the training regression before any training-time ontology claim.**
- Add the zero-shot encoder (identity / initialised-to-identity projection) as a preregistered baseline.
- Try early stopping on validation nDCG rather than InfoNCE loss.
- Hold out the six train↔validation paraphrase pairs.
- If trained < zero-shot still holds, report the trials ontology story as inference-time only.

**E4. (Optional) Eligibility-relevant ontology.** MeSH can only help 2 vs 1 if it touches criteria.
- The cheapest test reuses `results/trials_criterion_nested`: add exclusion-criterion MeSH matching (patient condition ∈ exclusion descriptors) as a feature, evaluated within patient.
- Needs criterion-section concept extraction (`embedding_cache/trials_criteria_concepts*.json` already exists [read; contents not inspected]).

## F. Not verified

- **Source of the training non-determinism.** Seeding code was not inspected; candidates are inferred.
- **Gradient scale / loss-component logging.** Not measured; there is no ontology loss to compare.
- **Negative composition** was not recomputed (`/tmp/trialsinv/` is gone). Those numbers are cited from `findings_trials.md` §D.
- **Weights 0.2 / 0.4 are post hoc.** They are reported as such; the validation-selected rules are the primary evidence.
- **Seed variance in the stratum analyses.** The bootstraps resample patients only, with seed-averaged deltas. Only B1 uses a two-level bootstrap.
- **`trials_criterion_pilot` / `nested`.** Only the summary and bootstrap files were read. Their protocol was not audited.
- **`embedding_cache/trials_criteria_concepts*.json`.** Not inspected.
- **Pooling bias.** TREC judged pools come from participant systems. If those systems used MeSH/condition matching, φ's relevance AUC could be partly a pooling artefact [inferred, untested]. E1's 2021 replication shares this caveat.
- **Topic-side MeSH extraction quality.** Estimated by eye in the prior report (about 80% / 60%). Not re-checked.
- **External validity.** All test patients are synthetic TREC vignettes. No real-cohort check exists.
