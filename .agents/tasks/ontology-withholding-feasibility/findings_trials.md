# Per-target withholding on the clinical-trial matching domain (TREC-CT 2021/2022): findings

Read-only investigation. Repo `/Users/olukotunjosh/Downloads/CDCL`, HEAD `699e0cb`, with uncommitted edits in `contrastive_learning/{batch_processor,loss_engine,trainer}.py`. Nothing in the repo was modified and nothing was trained. Scratch scripts are in `/tmp/trialsinv/` (`stats.py`, `dups.py`, `targets.py`, `tgt2.py`, `pooled.py`, `qside.py`, `leak.py`, `negaudit.py`, `phi.py`).

Evidence tags:
- **[read]**: from code.
- **[ran]**: from a read-only Python snippet over the data, configs or result files.
- **[inferred]**: reasoned, not directly verified.

`findings_go_ppi.md` did not exist when this report was written, so there is no GO/PPI cross-reference.

## Summary answer

1. **The split is clean on the query side, but that is the problem.** Train and validation are 2021 topics (60 and 15). Test is all 50 topics from 2022. The split is topic-disjoint by construction (`data_splitter.build_splits`, with an assert). So the career failure (the same resume and same target in train and test) cannot happen here.
   - But the 2021 and 2022 topic sets cover largely **different diseases**. For example, 2022 has no neoplasm and no cardiovascular primary topics.
   - Every disease-level target is therefore supported by a handful of patients. In train they are often near-duplicate paraphrases (2021 topics come in pairs such as 41/75, 36/71, 38/72). In test it is usually **a single patient**.
2. **The right target unit is a trial-side MeSH condition descriptor, and only a pilot exists.** The unit is taken from the NLM `condition_browse` annotation on the trial. Under the current split, 13 specific (non-C23, non-generic) descriptors have ≥25 train eligible pairs and ≥5 test eligible pairs from ≥2 test topics. Seven have ≥10 test pairs from ≥3 topics.
   - These collapse to about 8 independent targets, because COPD ⊂ Lung Diseases, Obstructive, Obesity ≈ Overweight, and CAD ≈ Myocardial Ischemia.
   - Train support is concentrated. Parkinson Disease has 255 train pairs, but 254 of them come from 2 paraphrased topics.
   - Test support is concentrated too. 31 of the 35 Parkinson test positives come from one 2022 patient with essential tremor, and 22 of 26 Diabetes positives from one patient.
   - For 9 of the 13 targets, **no test patient whose positives fall in o actually has o** (within 2 MeSH hops). The positives are comorbidity, control-arm or healthy-volunteer eligibility.
   - Pooled over all 125 topics, no descriptor has more than 4 "on-topic" patients with ≥5 eligible o-trials.
   - **Paired uncertainty across targets × seeds is not achievable.** The effective number of independent test patients per target is 1–4.
3. **The trials path uses InfoNCE, not the ordinal loss, and it has no ontology-in-loss term.**
   - Every trials config sets `loss_type="infonce"`. `run_learning_curve_point.py` forces `orca_enabled=False`.
   - The career `ordinal` loss is not portable. Trials train records are grade-2 only, so no `potential_fit` or `no_fit` siblings exist in a batch, and on-the-fly φ needs the ESCO KG.
   - `ontology_weight>0` would apply a constant 0.5 multiplier. Trials records carry no `ontology_similarity` or `quality_tier` (0/4,408 [ran]), so `_compute_ontology_weight` returns the tier-F base.
   - The only MeSH use is **negative ordering within each topic's own judged pool** (`TrialsNegativeSelector`).
   - So in the 2×2 only the guided-negatives factor exists: control `lc_trials_randwin.json` vs MeSH `lc_trials_ontneg_stoch.json`. **The loss-only and both cells need new code.**
4. **No term relates target o to neighbour j.** MeSH enters only as topic↔trial set similarity in negative selection. That is the same structural gap as career, so a k=0 effect can only come from global reshaping of the shared projection and from text similarity. MeSH φ also separates topical relevance well but is blind to eligibility:
   - pooled AUC for φ(topic, trial), eligible vs not-relevant: 0.73;
   - pooled AUC for eligible vs ineligible: 0.50 (hierarchy/all facet) [ran].
5. **Prior trials results are at noise level.** Matched MeSH-window vs random-window, 3 seeds, 5–20% fractions: Δ hard AUC (eligible vs ineligible) moves between −0.043 and +0.040 with flipping sign. Δ pooled AUC stays within ±0.02.
6. **Compute is very cheap.** Training is about 18–20 s wall per run at 20% (882 positives). Full data is about 1.5–2 min [inferred]. Evaluation is seconds with a cached frozen-embedding evaluator.
7. **Verdict.** TREC-CT **cannot support** the preregistered per-target n(o) × S(o) withholding study. At most it supports a small mechanism pilot (about 8 trial-condition targets, single-patient test support, no ontology path to the target). That is weaker than career v7, which after a resume-disjoint re-split had about 4–6 ESCO targets (or about 15–19 title clusters) and tens of independent test resumes per target.

---

## A. Task and data

### Query, candidate, text [read]
- `sample.resume` holds the **patient topic** (query). Its `encoder_view` is the raw narrative (`data_converter.build_topic_record`).
- `sample.job` holds the **trial** (candidate). Its `encoder_view` is "Trial: title / Conditions / Interventions / Eligibility bounds / Summary (≤800 chars) / Criteria (≤1,200 chars)" (`data_converter.build_trial_view`).
- Both slots also carry `skill_uris` (all MeSH descriptors: condition + intervention) and `coarse_uris` (condition descriptors) (`data_splitter._ontology_slot`; `record_adapter._slot`).

### Source and sizes [read README, ran]
- Data: TREC CT 2021 (75 topics, 35,832 qrels) and 2022 (50 topics, 35,394 qrels), against the 2021-04-27 ClinicalTrials.gov snapshot (375,580 trials).
- Ontology: MeSH 2021 descriptors (`desc2021.gz`, 29,917 descriptors).
- Converted data: `preprocess/trec_ct/{topics,trials,qrels}.jsonl`, with 48,714 judged trials and 71,226 judgments (`conversion_manifest.json`), plus `unjudged_pool.jsonl` (12,000 sampled unjudged trials).

### Label scheme [read `data_splitter.py` L36–103]

| TREC grade | trec_label | original_label | label |
|---|---|---|---|
| 2 | eligible | good_fit | 1 |
| 1 | ineligible (has the condition, fails criteria) | potential_fit | 0 |
| 0 | not_relevant | no_fit | 0 |

### Splits [ran, `preprocess/trec_ct_splits/split_manifest.json`, seed 42]

| split | topics | records | grades |
|---|---|---|---|
| train | 60 (2021) | 4,408 | grade 2 only (positives) |
| validation | 15 (2021) | 3,108 | 1,162 / 1,046 / 900, with grade 0 capped at 60 per topic |
| test | 50 (2022) | 35,394 | 3,939 / 3,036 / 28,419, uncapped |

- Train negatives come at batch time from `negative_pools.jsonl` (each topic's own grade-1 and grade-0 lists).
- Learning-curve fractions (`preprocess/trec_ct_lc/frac_*`) are nested prefixes of the seed-42-shuffled train positives. The graded negative pools stay fixed at full size (`fraction_manifest.json`).
- Per-topic counts: train eligible median 58.5 (min 6, max 203), ineligible median 63.5, not-relevant median 305. Test eligible median 59.5 [ran].

### Overlap and leakage [ran, `/tmp/trialsinv/stats.py`, `dups.py`, `leak.py`]
- **Patients:** train ∩ validation ∩ test = ∅ (asserted in the splitter and again in `run_learning_curve_point.attach_trials`).
- **Near-duplicate topics.** 2021 contains paraphrased topic pairs. Measured by judged-pool Jaccard and narrative similarity, six pairs straddle train and validation:

  | train topic | validation topic | judged Jaccard | eligible Jaccard |
  |---|---|---|---|
  | 2021_40 | 2021_74 | 0.37 | 0.83 |
  | 2021_27 | 2021_64 | 0.33 | — |
  | 2021_25 | 2021_61 | — | — |
  | 2021_33 | 2021_69 | — | — |
  | 2021_57 | 2021_63 | — | — |
  | 2021_23 | 2021_59 | — | — |

  - About 10 more pairs sit inside train (39/73, 41/75, 36/71, 34/70, 38/72, 21/57, 24/60, 30/66, 31/67, 22/58). The 60 train topics therefore hold roughly 45–50 independent scenarios.
  - 2021↔2022 near-duplicates are negligible: the largest judged Jaccard is 0.13 (2021_55 Wilson disease vs 2022_38 tremor).
- **Trials:**
  - 3,409 trials are judged for both train and test topics.
  - Of the 3,807 distinct test-eligible trials, 374 are judged in train and **56 are eligible in train** (1.5%). This is the closest analogue of "same pair seen in train".
  - Per target it can be large: 25 of the 35 Parkinson test-eligible trials are judged in train, and 9 are train positives.
- **"Same patient + same target in train":** impossible by construction, because patients are disjoint.

## B. Model and representation [read; architecture as in `findings.md` B]
- Frozen `sentence-transformers/all-mpnet-base-v2` (`freeze_text_encoder: true`) with one shared projection MLP to 128-d (`projection_dim: 128`, `projection_dropout: 0.3`) for both topics and trials. Scoring is a dot product of the normalised projections.
- The frozen text cache is `embedding_cache/trials_text_embeddings.pt` (27,631 entries [ran]). It is label-independent.
- A zero-example target is represented as projection(MPNet(trial `encoder_view`)). There is no concept or ID embedding and **no ontology path**: MeSH URIs reach only the negative selector (and ORCA features, which are disabled).

## C. Ontology in the loss
- **The single-phase trials path has none [read, ran].**
  - `loss_engine.compute_loss` → `_compute_triplet_loss` → `_infonce_loss` uses only anchor, positive and negative embeddings. The selector's `career_distances` (10 × (1 − MeSH sim)) never enter InfoNCE.
  - `_compute_ontology_weight` (L368–427) reads `view_metadata['ontology_similarity'|'ot_distance'|'quality_tier']`, which `BatchProcessor` copies from record metadata (~L518). Trials records carry none of these (0/4,408 [ran]).
  - So with `ontology_weight>0` (e.g. `orca_trials_denominator_config.json`: 0.3) the weight is the constant tier-None base of 0.5. That is a loss rescale, not ontology.
  - All LC and matched configs use `ontology_weight: 0.0`.
- **ORCA-Denominator** (`orca_use_ontology_features: true`) is the only design in which MeSH features (d_esco / d_isco from `MeshMatcher` + `branch_distance`) affect a loss, through learned negative reliability weights. It is confounded by the 3-phase schedule, `run_learning_curve_point.py` L307 forces it off, and `results/research_runs/ER-DEN__trec_ct__s*` contain only `training_config.json` (never completed).
- **Ordinal loss on trials [read/inferred]:**
  - The career `_compute_ordinal_loss` takes level 1 from in-batch same-query `potential_fit` records and level 0 from `no_fit` records (uncommitted `batch_processor` edit, L446–475). Trials train has neither, so L2/L3 would be inert.
  - `ordinal_phi_source="on_the_fly"` raises because there is no ESCO KG (uncommitted `trainer._loss_skill_matcher`).
  - `loss_type="go_ordinal"` (InfoNCE + softplus high>weak and weak>unobserved over the negatives' `grade` field) is the right structural analogue: `TrialsNegativeSelector` already stamps `grade` 1/0 on every negative. But `loss_engine.py` L74–76 rejects any domain other than `go_ppi`.
- **Strength of a candidate MeSH-in-loss signal [ran, `/tmp/trialsinv/phi.py`, train topics × their judged trials]:**

  | facet/mode | grade 2 φ mean (sd) | grade 1 | grade 0 | AUC 2 vs 1 | AUC 2 vs 0 | AUC 1 vs 0 | missing side |
  |---|---|---|---|---|---|---|---|
  | all / hierarchy | 0.304 (0.209), CV 0.69 | 0.310 (0.214) | 0.130 (0.155) | **0.495** | 0.732 | 0.742 | 7% |
  | disease / exact | 0.298 (0.277), 38% zero | 0.288 | 0.081, 81% zero | 0.511 | 0.724 | 0.718 | 13% |

  - Unlike career φ (sd 0.05), MeSH φ varies a lot, so a φ-adaptive margin would be a large perturbation, not a jitter. m = 0.5(1 − φ) on grade-1 pairs has mean 0.345, sd 0.107, 5–95% range 0.18–0.50.
  - But φ carries **no information about eligibility** (grade 2 vs grade 1). It only separates topical from off-topic trials.
- **Component and gradient logging:** none. With no ontology loss component there was nothing to probe. The InfoNCE-only gradient scale was not measured.
- **o↔j relation:** none anywhere in the trials path.

## D. Ontology-guided negatives

**Logic [read, `negative_selector.py`].**
- Per anchor, `select_batch_negatives` draws `max_negatives_per_anchor=7` from the topic's own ineligible (hard) and not-relevant (easy) lists.
- The grade-1 share ramps linearly from 0.2 to 0.6 over epochs, with backfill between tiers.
- The RNG is seeded by (seed, epoch, topic).
- Modes:
  - `uniform`: random within each tier.
  - `random_window`: a fixed random 34% subset, resampled per epoch.
  - MeSH `stochastic`: rank by 1 − `ontology_set_similarity(topic URIs, trial URIs)`, keep the closest 34% window, resample per epoch.
  - MeSH deterministic: `tier_sampling` left at `uniform` with `mesh_tiered` on gives the closest-first prefix.
  - `mesh_facet` all/disease, `mesh_tier_scope` both/ineligible/not_relevant, `mesh_similarity_mode` exact/hierarchy, and the V2 `soft_guidance` uniform/text/mesh/hybrid.
- Attachment is through the domain slot (`BatchProcessor._select_negatives` ~L855) and bypasses `_select_ontology_negatives`. The `ontology_rank_tie_break` edit is therefore irrelevant here.

**False-negative guard.**
- Pools are judged-only, so no unjudged trial becomes a negative on the main path.
- Grade-1 negatives are topically relevant by definition. That is intended, but it is the eligibility boundary that MeSH cannot see.
- The corpus selectors (`corpus_negative_selector.py`, label-budget study) draw **unjudged** trials and exclude only the topic's judged ones. TREC states that unjudged ≠ negative, so that arm has unguarded false negatives (documented in its docstring).

**Negative audit [ran, `/tmp/trialsinv/negaudit.py`, 400 train anchors, production selector, epoch 14]:**

| arm (config) | grade-1 share | negatives sharing a specific condition descriptor with the positive | mean MeSH φ(topic, negative) | epoch 7 vs 14 overlap |
|---|---|---|---|---|
| uniform (`lc_trials_baseline`) | 0.556 | 0.278 | 0.187 | 0.06 |
| random window (`lc_trials_randwin`) | 0.556 | 0.284 | 0.193 | 0.09 |
| MeSH deterministic, cap 100 (`lc_trials_ontneg_only`) | 0.556 | 0.354 | 0.416 | **0.86** |
| MeSH stochastic all/hierarchy (`lc_trials_ontneg_stoch`) | 0.556 | 0.346 | 0.307 | 0.09 |
| disease/exact stochastic (`mesh_disease_decomposition/disease_exact_both`) | 0.556 | 0.383 | 0.296 | 0.09 |

- `lc_trials_ontneg_only` replays about 86% of the same negatives across epochs. It also ranks only the first 100 NCT-sorted candidates (`trials_mesh_score_cap: 100`), while the median not-relevant pool is 305. **It is confounded with diversity and should not be a 2×2 cell.**
- The stochastic pair is matched in width and diversity, and the run log confirms the single-factor audit ("differences are confined to the intended factor").

**2×2 mapping with existing configs:**

| | random window | MeSH-guided window |
|---|---|---|
| no ontology in loss | `lc_trials_randwin.json` | `lc_trials_ontneg_stoch.json` |
| ontology in loss | **missing (no term exists)** | **missing** |

Confounded and to be excluded: `lc_trials_baseline` vs `lc_trials_ontneg_only` (diversity and cap), the ORCA config (phases, constant 0.5 weight), the V2 soft arms (they guide only the not-relevant tier and use `trials_common_validation`), and the label-budget arms (unjudged corpus negatives).

## E. Ontology graph, mapping quality, neighbourhoods

**Graph [read, `mesh_ontology.py`].**
- MeSH 2021 tree numbers (polyhierarchy).
- `skill_distance` = minimum LCA hops over tree pairs, capped at 12. `skill_sim` = exp(−0.5 · hops).
- Set similarity is a symmetric best-match average. `branch_distance` is the minimum Wu-Palmer distance over condition trees.

**Trial-side mapping [ran].**
- Condition terms come from NLM's algorithmic `condition_browse` and resolve 100% (`conversion_manifest.json`). Interventions resolve 82%, because supplementary concepts (e.g. gemcitabine) are missing.
- 5,504 of 48,714 judged trials (11%) have no condition descriptor.
- Browse annotations include ancestors and generic descriptors: Syndrome, Disease, Infections, Emergencies, Neoplasms, and many C23 "signs/pathologic processes".
- Of the 9,509 eligible pairs, 735 have no condition and 494 have generic-only conditions, which leaves 8,280 with a specific descriptor.

**Topic-side mapping [ran listing; judgement inferred by single-reviewer eyeballing of all 125].**
- Dictionary n-gram extraction (`concept_extractor.extract_concepts`, C + F03) has no abbreviation or negation handling.
- Example errors:
  - 2021_34 and 2021_70 (MDD) → "Colonic Neoplasms", taken from the husband's history;
  - 2021_8 (CLL) → only "Splenomegaly; Lymphadenopathy";
  - 2022_15 (DMD) → nothing;
  - 2021_18 (BPH) → "Urethral Stricture";
  - 7 topics are empty (`degraded_topics`).
- Every topic also carries 2–14 incidental symptom or comorbidity concepts.
- Rough estimate: the primary condition is captured in about 80% of 2021 topics and about 60% of 2022 topics, which are mostly diagnostic vignettes.
- This corrupts the MeSH φ anchor and any query-side target unit.

**Target unit.**
- Query-side disease groups fail on support. Using the first specific topic descriptor at MeSH depth 1, only 8–9 chapters have ≥3 topics from 2021 and ≥2 from 2022. 2022 has 0 C04 (neoplasm) and 0 C14 (cardiovascular) primary topics [ran, `qside.py`].
- The candidate-side unit (trial condition descriptor) is the only one with numbers. In that framing, n(o) = train eligible pairs whose trial carries o.

**Viable targets** (train ≥25 eligible pairs, test ≥5 from ≥2 topics) [ran, `tgt2.py`]. Column notes:
- "on-topic" = topics whose own condition URIs lie within 2 MeSH hops of o;
- S_h2 / S_h4 = train eligible pairs, excluding o-pairs, whose trial carries a descriptor within 2 / 4 hops of o. Brackets give (number of neighbour descriptors, distinct topics).

| target | n(o) pairs | train topics (on-topic) | val | test pairs / topics (on-topic) | test non-eligible | o-trials in train neg pools | S_h2 | S_h4 |
|---|---|---|---|---|---|---|---|---|
| Parkinson Disease | 255 | 3 (2) | 1 | 35 / 4 (0) | 201 | 403 | 11 (5, 4 topics) | 190 |
| COPD | 125 | 4 (2) | 27 | 10 / 6 (0) | 435 | 273 | 188 | 297 |
| Diabetes Mellitus | 110 | 11 (2) | 0 | 26 / 5 (2) | 389 | 576 | 145 | 628 |
| Obesity | 107 | 7 (2) | 0 | 13 / 5 (2) | 201 | 214 | 55 | 86 |
| Lung Diseases, Obstructive | 93 | 4 (3) | 16 | 10 / 6 (2) | 349 | 211 | 302 | 490 |
| Influenza, Human | 75 | 3 (0) | 31 | 6 / 4 (0) | 157 | 74 | 80 | 692 |
| Sleep Apnea Syndromes | 54 | 9 (3) | 3 | 5 / 4 (0) | 158 | 214 | 15 | 49 |
| Coronary Artery Disease | 51 | 8 (2) | 0 | 6 / 6 (0) | 143 | 324 | 52 | 325 |
| Hypertension | 38 | 11 (5) | 24 | 37 / 10 (4) | 482 | 338 | 118 | 531 |
| Weight Loss | 37 | 5 (1) | 0 | 8 / 4 (0) | 167 | 162 | 13 | 126 |
| Overweight | 35 | 5 (1) | 0 | 16 / 7 (2) | 323 | 156 | 124 | 238 |
| Myocardial Ischemia | 29 | 7 (5) | 0 | 5 / 5 (0) | 121 | 239 | 255 | 748 |
| Congenital Abnormalities | 25 | 6 (3) | 1 | 7 / 6 (1) | 115 | 101 | 30 | 538 |

- Concentration [ran]:
  - Parkinson: train 2021_75 = 129 and 2021_41 = 125 (paraphrases); test 2022_38 (essential tremor) = 31/35.
  - Diabetes: test 2022_25 = 22/26.
  - COPD: train 2021_28 = 96.
  - Hypertension: test 2022_9 = 18/37.
- n vs S Spearman across the 13 targets: ρ = 0.03 (S_h2) and −0.26 (S_h4) [ran]. That is uncorrelated, which is good, but n = 13 with nested targets.
- Low-S candidates at S_h2: Parkinson, Sleep Apnea, Weight Loss. High-S: Lung Diseases Obstructive, Myocardial Ischemia, COPD. These high-S values are mostly co-annotated ancestor or sibling descriptors on the *same* trials, so withholding o also thins its neighbours [inferred].
- **Pooled feasibility over all 125 topics [ran, `pooled.py`]:** only 3 descriptors (Hypertension, Lung Diseases Obstructive, Myocardial Ischemia) have ≥4 on-topic patients with ≥5 eligible o-trials. None has ≥6. A patient-disjoint re-split cannot fix test support.

## F. Evaluation [read]
- **`run_phase1_embedding_evaluation.py`.** `content_to_text` honours `encoder_view` (L222–238), so train and eval serialisation are **consistent** for trials, unlike career.
  - Outputs: pooled AUC, `per_grade` pairwise AUCs, and `per_patient` (`trials_domain/patient_metrics.patient_report`: per-topic eligible AP, eligible AUC, graded nDCG@10 with gains 0/1/3, and per-contrast AUCs).
  - It re-encodes texts per process; the first full test pass took about 60 min (`results/label_budget/eval_test2022.log`).
- **`scripts/evaluate_publication_fusion.py`.** Loads the frozen cache plus an optional `--prepared-cache-dir` and applies only the projection (`text_scores`, L295).
  - Outputs: global and macro-per-query AUC for hard/easy/middle/pooled, plus per-query nDCG@10, P@10, R@10, AP and RR.
  - It is the right evaluator to reuse. Per-pair scores exist in memory but are not written out.
- **Candidate pool:** each test topic's judged trials (TREC depth pool), uncapped (about 708 per topic). It is fixed across conditions as long as `test.jsonl` (sha `4030cc85…`) is frozen.
- **Seeds:** existing runs use 42/13/21; the planned V2 study uses 13/21/42/87/123.
- **Checkpoint selection:** `best_checkpoint.pt` = lowest validation InfoNCE loss on grade-2 validation positives, with negatives from each arm's own selector (`fixed_validation_negatives`, epoch 14, seed 1729).
  - Under MeSH arms the validation negatives are MeSH-tiered, so the selection criterion is variant-dependent (as in career). Only the V2 runner has `trials_common_validation`.
  - Validation also contains near-duplicates of train topics (§A).
- **Per-target metrics feasible on the 2022 test:**
  - M1 = per-target judged AUC (o-trial eligible vs non-eligible), with 5–37 positives;
  - M2 = within-topic rank of eligible o-trials among the topic's judged pool.
  - Both are dominated by one test patient per target.

## G. Prior results [ran over result files]

**Matched negative selection**, test 2022, 3 seeds (`results/trials_negative_selection_matched{,_v2}/trials_test_summary.md`). Columns give Δ (MeSH stoch − randwin) as mean ± sd over seeds:

| fraction | v1 Δ hard AUC | v1 Δ pooled | v2 Δ hard | v2 Δ pooled |
|---|---|---|---|---|
| 5% | +0.001 ± 0.023 | −0.004 ± 0.014 | +0.006 ± 0.031 | −0.016 ± 0.023 |
| 10% | +0.030 ± 0.019 | +0.020 ± 0.026 | −0.025 ± 0.031 | +0.009 ± 0.011 |
| 15% | −0.043 ± 0.008 | −0.010 ± 0.021 | +0.040 ± 0.026 | −0.006 ± 0.024 |
| 20% | −0.004 ± 0.021 | +0.006 ± 0.013 | −0.020 ± 0.005 | −0.014 ± 0.014 |

- Absolute hard AUC is 0.52–0.57 and pooled AUC 0.73–0.77. The sign of Δ flips between v1 and v2 at the same fraction, so there is no detectable effect.

**Other studies:**
- **Label budget, unjudged corpus negatives** (`results/label_budget/trials_test2022_summary.json`, s42): test AUC full 0.705, low_ontology 0.749, low_random 0.766. The ordering is reversed on validation. Not comparable.
- **Fusion probe** (`results/ontology_ceiling/fusion_probe_trials.json`): adding the MeSH score gives hard +0.044, easy −0.068 and pooled −0.056.
- **Not executed:** MeSH disease decomposition (configs only), V2 soft sampling (only `requested_config.json` in `results/mesh_soft_sampling/folds_*`), ER-DEN trec_ct, and `lc_single_factor/trials_*`.

## H. Compute [ran]
- **Training:** matched runs at 5/10/15/20% take 9/11/14/16 s of `training_time` and about 12–20 s wall including setup (train.log timestamps). Full data (4,408 positives) is about 80–120 s [inferred, linear]. The old label-budget full run took 1,275 s with older code.
- **Evaluation:** seconds per run with `evaluate_publication_fusion` and a prepared cache. The first full-test encode costs up to about 60 min, once.
- **Grid:** (8 targets × 5 withheld levels + 1 shared "all") × 4 cells × 5 seeds = 820 runs ≈ 25–30 h sequential [inferred]. Packing 2–3 non-neighbouring targets per run (e.g. Parkinson + Diabetes + Influenza) cuts this to about 10 h.

## I. Reusable diagnostics
- `TrialsNegativeSelector.pool_summary` and the attach manifest in `lc_manifest.json`.
- `run_learning_curve_point.report_factor` (single-factor config audit).
- `trials_domain/patient_metrics.py`.
- `scripts/evaluate_publication_fusion.py` (cached, per-query metrics).
- `scripts/mesh_signal_diagnosis.py`, `mesh_hierarchy_vs_set_diagnosis.py`, `audit_mesh_decomposition.py` (φ separation and tie fractions).
- `scripts/trials_diagnose_separation.py`.
- `trials_domain/tests/test_cache_guard.py` plus `label_budget_runner.assert_cache_valid`.
- The scratch audits `/tmp/trialsinv/negaudit.py` (negative composition and diversity) and `phi.py`.

---

## Conclusions and recommendations

### (i) Recommended 2×2

**Shared fields:** all common fields of `config/lc_trials_randwin.json`, i.e. `domain_adapter="trials"`, `freeze_text_encoder=true`, `projection_dim=128`, `num_epochs=15`, `lr=8.5e-5`, `temperature=0.07`, `max_negatives_per_anchor=7`, `trials_start_hard_ratio=0.2`, `trials_end_hard_ratio=0.6`, `trials_tier_window_frac=0.34`, `trials_mesh_score_cap=0`, `mesh_similarity_mode="hierarchy"`, `trials_mesh_facet="all"`, `ontology_weight=0.0`, `use_ot_distance=false`, `fixed_validation_negatives=true`, `validation_negative_seed=1729`, `validation_negative_epoch=14`. Also add `trials_common_validation`-style uniform validation negatives in every cell (needs a small generalisation, see gap 6).

**Base loss for all cells:** the grade-aware ordinal loss (`go_ordinal` semantics) with `go_ordinal_lambda_high_weak=1.0`, `go_ordinal_lambda_weak_unobserved=0.1`, `go_ordinal_rank_temperature=0.1`. This keeps the study on an ordinal contrastive loss, which matches career v7. It needs the `loss_engine.py` L74–76 guard to accept `"trials"`.

| cell | file | margin (eligible > ineligible) | negatives |
|---|---|---|---|
| Baseline | new: randwin + `loss_type="go_ordinal"`, `go_ordinal_margin_high_weak=0.345` (mean-matched) | fixed | `trials_mesh_tiered_negatives=false`, `trials_tier_sampling="random_window"` |
| Ontology loss only | new: baseline + `trials_ordinal_phi_margin=true`, `trials_ordinal_alpha=0.5` | 0.5(1 − φ_MeSH(topic, grade-1 trial)), per pair | random window |
| Guided negatives only | new: `lc_trials_ontneg_stoch.json` + `loss_type="go_ordinal"`, fixed margin 0.345 | fixed | `trials_mesh_tiered_negatives=true`, `trials_tier_sampling="stochastic"` |
| Both | new: guided + `trials_ordinal_phi_margin=true` | φ-adaptive | MeSH stochastic |

- **The margin value.** 0.345 is the train mean of 0.5(1 − φ) on grade-1 pairs [ran]. It removes the career-style mean-margin confound. `go_ordinal_margin_high_weak` currently defaults to 0.1, so this is a pre-registration choice.
- **Fallback if no new code is allowed:** only the guided-negatives factor is testable (InfoNCE, `lc_trials_randwin` vs `lc_trials_ontneg_stoch`). That is a 1×2, not a 2×2.
- **Either way, preregister that φ_MeSH does not separate eligible from ineligible** (AUC 0.50). The loss factor is therefore expected to act through topical, not eligibility, structure.

### (ii) Verdict

- **Not supportable as designed.** The split is patient-disjoint and temporally held out, so there is no query-leakage problem to fix. The real limit is data:
  - 125 synthetic patients, with only about 110 distinct scenarios once paraphrases are merged;
  - 2021 and 2022 diseases barely overlap;
  - each disease's eligible pairs come from 1–4 patients.
- **Usable targets:** about 8 trial-condition descriptors, each with ≤2 on-topic test patients (often 0). Test positives for o mostly come from patients without o.
  - Retention levels {0, 1, 5, 10, 25, all} are only expressible in *pairs*, and those pairs come from 2–3 patients. So k = 5 vs k = 25 changes the number of trials for the same patient, not the number of observed patients.
  - Paired uncertainty across targets is dominated by 8 targets × essentially one test patient each.
  - The S(o) moderation hypothesis has no mechanism, because no term relates o to j.
- **Comparison with the other domains:**
  - **Career v7:** its main flaw (pair-level split) is fixable by re-splitting, after which it supports a 4–6 target (or 15–19 cluster) pilot with tens of independent test resumes per target.
  - **Trials:** has no split flaw but cannot be rescued by re-splitting. Pooling both years under patient-disjoint CV still gives at most 4 on-topic patients per descriptor.
  - **GO/PPI:** report not available.
- **What trials can support:**
  - **(a)** A *mechanism pilot*: withhold the trials of one condition and look for collateral effects and for the k = 0 global-reshaping baseline.
  - **(b)** The existing *global* label-budget learning curve (fractions of positives), where n = 50 test patients gives real paired uncertainty.
  - **(c)** The grade-1-specific question (eligibility boundary) that only this dataset labels.
- **If a pilot is run, the minimum fixes** are:
  - remove every o-trial (all grades) from train positives, all train and validation negative pools and the validation records;
  - drop or hold out validation topics that paraphrase train topics;
  - count retention in pairs and report distinct topics;
  - use variant-independent validation negatives;
  - export per-pair scores.

### (iii) Gap list, trials-specific (ordered)

1. **Freeze the code.** The uncommitted edits do not affect the InfoNCE trials path [read: the ordinal and tie-break branches are gated on `loss_type=='ordinal'` and `ontology_negative_rank_tiers`, and `_loss_skill_matcher` returns the existing `None`]. They would matter once any ordinal loss is used, so tag them.
2. **Target table.** Build a frozen list of specific trial-condition descriptors. Exclude C23-only and generic descriptors (Syndrome, Disease, Infections, Emergencies, Neoplasms, …) and merge nested ones (COPD/LDO, Obesity/Overweight, CAD/MI). Hand-check the 13 candidates.
3. **Validation hygiene.** Move 2021_74, 64, 61, 69, 63 and 59 (or their train twins) so that paraphrase pairs fall in one split, or use a paraphrase-aware fold assignment.
4. **Withholding generator.** For target o:
   - drop train positives whose trial carries o;
   - drop o-trials from every topic's `negative_pools.jsonl` entry (train and validation) and from validation records;
   - retain k ∈ {0, 1, 5, 10, 25} o-pairs, nested and seeded, recording their topics;
   - write to a new split directory with a manifest.
   - The pool step matters: today 74–576 o-trials per target sit in train negative pools, and guided negatives raise the share of negatives that share a condition with the positive from 28% to 35–38% [ran].
5. **Runner assertions.** No o-trial NCT in any train or validation pool or record. Pools must be topic-disjoint (already asserted). Text-mining scores (`soft_sampling.prepare_text_scores`) must be built from the filtered pools.
6. **Variant-independent checkpoint selection.** Generalise `trials_common_validation` (currently V2-only, `run_learning_curve_point` L145) to all arms, or select the final epoch.
7. **Loss code, for the 2×2:**
   - allow `go_ordinal` for `domain_adapter="trials"`;
   - add `trials_ordinal_phi_margin` and `trials_ordinal_alpha` to `TrainingConfig` (default off, so existing configs stay byte-identical);
   - compute φ via the attached `MeshMatcher`, which also needs to be attached in random-negative cells *without* enabling tiering (mirror the career `_loss_skill_matcher` separation).
8. **Optional o↔j path (needed for any S(o) claim).** A concept-alignment term that pulls trial embeddings toward projected MeSH concept-name embeddings, with a hierarchy-consistency term over the label-free MeSH C tree. λ defaults to 0. Preregister that using withheld trials' MeSH annotations (not their qrels) is side information, not leakage.
9. **Evaluation.**
   - Per-pair score export from `evaluate_publication_fusion.py`, joined on topic_id and nct_id.
   - A per-target evaluator for M1/M2, with on-topic vs off-topic test patients reported separately.
   - Freeze the test sha and prepared embedding cache.
10. **Diagnostics.**
    - Per-epoch L_InfoNCE and L_rank with per-component gradient norms (career `/tmp/v7inv/probe.py` recipe).
    - The φ-margin violation rate.
    - The negative audit per cell (`/tmp/trialsinv/negaudit.py`).
    - Relation preservation: projected cosine between o-trials and neighbour-descriptor trials vs MeSH hops.
    - A text-similarity covariate (frozen cosine of o-trials to neighbour trials).
11. **Topic-side MeSH extraction.** Abbreviations (MDD, CLL, DMD, BPH), negation and family-history handling. This affects φ for every arm.
12. **Statistics.** A paired bootstrap over test topics × seeds (as in `run_mesh_soft_sampling.summarize`). statsmodels is not installed (per the career report).

### (iv) Open decisions

- **Whether to run trials at all for this question.** Recommended: treat trials as a mechanism pilot and as the global-budget and eligibility-boundary domain, not as confirmatory evidence for per-target n × S effects.
- **Target unit:** a trial-condition descriptor (recommended) vs a coarser MeSH subtree (depth 2/3 gives 44–49 "viable" units, but they are nested, generic, e.g. C23.550, and off-topic).
- **What withholding removes:** o-trials only (recommended), or also on-topic patients (which turns the study into patient withholding, with ≤4 patients).
- **Retention unit:** pairs (k up to 25 is feasible only for about 13 targets) vs patients (k ∈ {0, 1, 2} at most).
- **Test definition:** all 2022 positives in o (mostly off-topic) vs on-topic patients only (0–4 per target). Alternatively, pooled 2021+2022 patient-disjoint CV with paraphrase-aware folds.
- **Neighbourhood N_k(o):** ≤2 MeSH hops (parent, children, siblings) vs ≤4 hops. Decide whether co-annotated pairs (a trial carrying both o and a neighbour) count, given that withholding o removes them.
- **Loss factor:** the φ-adaptive margin (per-pair, blind to eligibility) vs a concept-alignment term (gives an o↔j path). Use a mean-matched fixed margin either way.
- **Guided-negative family:** all-facet/hierarchy stochastic (recommended; less circular with a condition-descriptor N_k) vs disease-facet/exact.
- **Primary metric:** per-target judged AUC (eligible vs rest, within o-trials) vs within-topic AP. Also pick a minimum test support (suggested ≥10 eligible pairs and ≥3 topics, which leaves 7 targets).
- **Seeds and interpretation:** ≥5 seeds. Preregister that k = 0 deltas are global reshaping and that any S(o) association is read as co-annotation or text similarity unless the concept-alignment term is on.

**Not verified:**
- No training, loss or gradient probes were run, and no ontology-loss component exists to probe.
- Full-data train time is extrapolated.
- Topic-side mapping accuracy is one reviewer's eyeball pass.
- Target counts use trial `condition_browse` descriptors restricted to C/F03 with my generic-descriptor list.
- "On-topic" uses the noisy topic-side extraction, so it may undercount.
- The pooled-CV feasibility count is a simulation, not a built split.
- The ER-DEN and ORCA trials paths were not exercised.
