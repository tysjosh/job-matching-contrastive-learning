# Feasibility of the per-target withholding design: findings

Read-only investigation of `/Users/olukotunjosh/Downloads/CDCL` at HEAD `699e0cb` (branch `orca-experiments-and-runner`). Note that the working tree has uncommitted edits in `contrastive_learning/{batch_processor,loss_engine,trainer}.py` (ordinal-loss candidates, `ontology_rank_tie_break`, a loss-only skill matcher). Nothing was modified and no training was run. Tags used below: **[read]** means verified by reading code, **[ran]** means verified with a read-only Python snippet over the data, **[inferred]** means not directly verified.

## Summary answer

1. **Zero-shot is architecturally possible, but no ESCO path reaches a withheld occupation.** Resumes and jobs share one frozen MPNet encoder and one 768→256→128 MLP projection. There is no occupation ID embedding, so an unseen target's job text can still be scored. No ESCO relation links one occupation to another, though. The "ontology loss" is a per-pair resume↔job skill-similarity sample weight, and the "guided negatives" rank candidate jobs by resume↔job skill distance. If all of a target's records are withheld, no loss term ever involves the target, so the S(o) moderation hypothesis has no mechanism to test unless new code is written.
2. **There is no additive `L_CL + λ·L_ontology`.** `ontology_weight` multiplies each triplet's InfoNCE loss by a clamped factor in [0.5, 1.5]. On v7 this factor has a coefficient of variation of about 6% (mean 0.868). Only λ = 0.3 was ever used. Component losses are not logged, and per-component gradient norms are not logged either.
3. **Guided negatives and the sample-weight "loss" can already be toggled independently** (`ontology_guided_negatives` vs `ontology_weight`). Baseline, loss-only and negatives-only configs exist (`config/lc_career_*`). A clean "both" cell does not: `lc_career_ontfull` also switches on `use_ot_distance`. There is no false-negative guard. The only exclusion is "same job title as the positive".
4. **The current career dataset (v7) is a poor fit for this design.**
   - Only 33 occupations have any positives, and 12 have at least 25 train positives and at least 5 test positives.
   - 379 of 381 test resumes also appear in train.
   - The automated ESCO mapping is badly wrong for several of the largest occupations. For example, "Senior Accountant" maps to *commercial director*, "Software Engineering Manager" to *foundry manager*, and "Full Stack Engineer" to *air traffic safety technician*. Roughly 5 of the 12 viable targets are mislabelled, so N_k(o) and S(o) computed for them would be meaningless.
   - The Indian dataset (`preprocess/indian_splits`) fits much better: 50 occupations with one job text each, 7–127 train positives each, 834 of 885 test resumes independent of train, and 22 targets with at least 10 independent test positives. It still has about 9 of 50 occupations mis-mapped.
5. **Evaluation reports only pooled pairwise AUC-ROC.** It has no per-occupation export, no saved per-pair scores, and no candidate-pool ranking. The career "NDCG@10" in `run_ordinal_evaluation.py` is a global pooled sort, not a per-query metric. The eval script also serializes text differently from the trainer (see E4).
6. **Leakage points under per-target withholding:**
   - The validation set, which drives `best_checkpoint.pt` selection, contains target positives.
   - The global negative pool is built from whatever train file is passed, so it must be condition-specific.
   - The frozen-text embedding cache is label-independent and safe.
   - The data-driven graphs are not consumed by the learning-curve training path.
7. **Compute is cheap per run but large for the grid.** A full-data v7 run trains in about 3.5 min on average (6.5 min worst case) on the CPU of this M2 Mac, which has no CUDA; MPS is never selected. For example, 12 targets × 5 withheld levels × 4 variants × 5 seeds is 1,200 runs, roughly 4–5 days sequential. Withholding several targets that are not each other's neighbours in a single run cuts this by about 5×.

---

## A. Model and representation path

**A1. Architecture.** [read]
- `ContrastiveLearningTrainer.__init__` (`contrastive_learning/trainer.py` ~L118–200) loads `SentenceTransformer(config.text_encoder_model)` (`all-mpnet-base-v2` in every `lc_career_*` config). When `use_structured_features` is false, it builds `CareerAwareContrastiveModel` (`_create_dual_encoder_model`, ~L1465–1540): `Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 128) → L2-normalize`.
- This projection is **shared by resumes and jobs**. Scoring is cosine (dot product) between the two projected vectors.
- With `freeze_text_encoder: true` (all lc/eo configs), encoder params get `requires_grad=False` and Adam optimizes only the projection (~L180–195).
- `freeze_text_encoder: false` fine-tunes both and bypasses the cache (`_generate_embeddings_trainable_encoder`, ~L1172). `run_unfrozen_probe_gpu.sh` runs these UF-Ordinal probes on full v7 with 5 seeds via `python -m contrastive_learning train`.
- `fine_tuning_trainer.py` is the Phase-2 classifier path and is not involved in the learning-curve runs. `sentence_transformer_config.py` is a separate matching-system config that the trainer does not use.

**A2. A zero-example occupation at eval time.** [read]
- `run_phase1_embedding_evaluation.py` encodes each test (resume, job) pair's text with the frozen encoder and passes both through the same projection. The occupation's representation is just "projection(encoder(job text))", which works for any text. Zero-shot is therefore possible.
- Which loss terms can touch the target:
  - InfoNCE anchors are only `label == 'positive'` records (`BatchProcessor.process_batch`, `batch_processor.py` ~L400). In InfoNCE mode, a resume or job from a non-positive record enters the loss only as a negative candidate, via the global job pool.
  - The ontology sample weight (`_compute_ontology_weight`, `loss_engine.py` L368–427) scales the anchor's own pair loss and uses that pair's `metadata.ontology_similarity` and `ot_distance`.
  - Guided negatives (`_select_ontology_negatives`, `batch_processor.py` ~L1165) score candidate jobs against the anchor resume's skill URIs, optionally blended with ISCO distance to the anchor job's occupation.
  - None of these terms relates occupation o to occupation j. With all target records removed, the target job text appears in no loss term. Any Δ at retention 0 can only come from how a variant reshapes the shared projection globally. **No explicit ESCO-neighbour path exists.**
- If the target's `no_fit`/`potential_fit` records are kept, its job texts stay in the negative pool. They then get pushed *away* from other resumes, and under guided negatives they are most likely to be picked as hard negatives for neighbours' resumes.

**A3. Embedding cache.** [read]
- `EmbeddingCache.get_content_key` (`embedding_cache.py` L69–91) is SHA-256 over text-bearing fields only. It caches *frozen* encoder outputs (`trainer._generate_embeddings` docstring ~L1060). The projection runs fresh every batch.
- The cache does not depend on labels or training data, and it is shared across arms on purpose (`_comment_cache` in `config/lc_career_*.json`). No per-condition invalidation is needed while the encoder is frozen.
- `_validate_epoch` (~L915–930) persists validation texts into the same cache. That is text only, with no labels.
- The eval script keeps its own in-process `_TEXT_EMB_CACHE` (`run_phase1_embedding_evaluation.py` ~L120).

## B. What "L_ontology" actually encodes

**B1. Sample-level ontology weight (`ontology_weight`).** [read]
- `_compute_triplet_loss` computes `loss = InfoNCE(anchor, positive, negatives) * _compute_ontology_weight(view_metadata)` (`loss_engine.py` ~L355–360).
- The weight is `base_tier * (1 + λ·(2·signal − 1))`, clamped to [0.5, 1.5].
  - `signal` is the mean of the precomputed `metadata.ontology_similarity` (resume-skill vs job-skill set similarity over the ESCO KG; `scripts/compute_ontology_scores.py`) and, when `use_ot_distance` is on, `1 − ot_distance/10`.
  - `base_tier` comes from `quality_tier`.
- The relation encoded is therefore **pairwise skill-set similarity between the anchor resume and its own positive job**. It does not use typed occupation edges, the ISCO hierarchy between occupations, or graph distance between occupations. It acts only on positive anchors in the batch.
- [ran] Effective spread on v7 positives: mean 0.868, sd 0.051, 5th–95th percentile 0.81–0.95 (CV 0.058). On Indian: mean 0.858, sd 0.092 (CV 0.107). This is close to a uniform rescaling.

**B2. ISCO loss weight (`isco_loss_weight`) is inert.** [read] It reads `view_metadata['resume_occupation_uri']`, which `_create_triplet` fills from `anchor_sample.metadata['resume_occupation_uri']` (~L527). No career file carries that key: the audit reports `resume_occupation_present_rows: 0`, and `train_with_resume_occ.jsonl` puts it under `resume.occupation_uri` instead. `isco_only_weight` is declared in `data_structures.py` L315 and read nowhere.

**B3. Ordinal loss (`loss_type: 'ordinal'`, eo_* configs).** [read]
- `_compute_ordinal_loss` (~L752–978) returns `L1 (InfoNCE) + mean_q[w_q·(λ2·L3 + λ1·L2)]`.
  - L3 is a hinge `relu(m2 − (s_i − s_j))` for any level above the no_fit floor.
  - L2 is good > potential with margin `m1 = α(1 − φ)`, where φ is ESCO skill coverage (precomputed or on the fly).
- This is the closest thing to an additive structured term, but its ontology content is again resume↔job φ.
- `eo_a` vs `eo_d` (`ordinal_fixed_m1`) isolates the φ margin. λ1 = λ2 = 1.0, α = 0.5, m2 = 0.3 in all eo configs.

**B4. λ values, logging and gradients.**
- λ values used [read from configs]: `ontology_weight` is 0.3 in every config that enables it (lc_career_ontweight_only, ontfull, ontfull_abs, CDCL/config/phase1_infonce_ontology*, isco*, confit). Others: `ws2_weight` 0.0; `isco_weight` 0.4 (ontology_isco) or 1.0 (isco_only, eo_isco*); `pathway_weight` 0.8, which is inert (see C2). No λ sweep exists.
- Component logging [read]: only the total loss is logged. `get_loss_components` (`loss_engine.py` L1432) is never called.
- Gradient norm [read]: computed by `clip_grad_norm_` (`trainer.py` ~L1010, `gradient_clip_norm` default 1.0) and logged only when it exceeds 2× the clip. Clipping at 1.0 also changes the effective λ whenever it binds.
- `scripts/investigate_fix6_scale.py` has an offline `grad_norm_wrt(loss, tensors)` that compares L1 vs ordinal-margin gradient norms. It is reusable.

## C. Ontology-guided hard negatives

**C1. Where and how negatives are picked.** [read]
- `BatchProcessor._select_negatives` (`batch_processor.py` ~L814–979) builds candidates from `global_job_pool`. The pool comes from `DataLoader.load_global_job_pool` (`data_loader.py` ~L995): the first job seen per **title** (deduplicated by `job_id` or `title`), up to 1000.
- [ran] That gives 190 titles covering 58 occupations on v7, and 50 titles (one per occupation) on Indian.
- The only exclusion is "same job_id/title as the anchor's positive". Random sampling is used unless `use_pathway_negatives` **or** `ontology_guided_negatives` is set and a skill matcher exists.
- `_select_ontology_negatives` then:
  - computes the distance `1 − OntologySkillMatcher.ontology_set_similarity(resume skill URIs, job skill URIs)`;
  - optionally blends in ISCO distance between the anchor's positive-job occupation and the candidate's occupation (`use_isco_negatives`, weight `isco_weight`);
  - tiers candidates by absolute cut points (≤0.3 / ≤0.6) or by per-anchor terciles (`ontology_negative_rank_tiers`), with an optional tie-break shuffle (uncommitted);
  - samples hard/medium/easy at 0.33/0.34/0.33. eo_*_hard uses 0.6/0.3/0.1.
- `self.career_graph` is built when `use_pathway_negatives` is set but never consulted. `CareerGraph.select_pathway_negatives` has no callers.

**C2. False negatives.** [read + ran] There is no guard. On v7, the share of random-pool candidates per anchor that are labelled `good_fit` for the same resume is 2.4%, `potential_fit` 2.0%, and same-occupation 8.8%. Hard (closest-skill) tiers will be enriched for exactly these. On Indian, every resume has one judged job, so no candidate is a *known* false negative, but neighbouring occupations are unjudged rather than irrelevant. Guided negatives push skill-near occupations apart, which works directly against a loss that pulls ESCO neighbours together. The "both" cell carries this tension.

**C3. Independent toggles and mapping of existing configs to the four variants.** [read]

| Required variant | Existing config | Status |
|---|---|---|
| Baseline | `config/lc_career_baseline.json` (InfoNCE, random negatives, no matcher) | exists |
| Ontology loss only | `config/lc_career_ontweight_only.json` (λ = 0.3 sample weight, OT off) | exists, but it is a sample weight, not an additive L_ontology |
| Guided negatives only | `config/lc_career_ontneg_only.json` (skill-distance rank tiers) | exists |
| Both | none clean. `lc_career_ontfull.json` also sets `use_ot_distance=true` and `use_pathway_negatives=true` (the latter inert). `lc_career_ontfull_abs.json` uses absolute tiers. | missing; config-only fix |
| Ordinal family | `eo_a` (φ margin) vs `eo_d` (fixed margin) isolates the φ margin. `eo_ontneg`/`eo_ontneg_hard` are guided negatives (skill). `eo_isco`/`eo_isco_hard` are ISCO-only guided negatives. `eo_c` changes the curriculum, `eo_e` changes `group_by_resume`. | usable as a second 2×2 |
| `CDCL/config/phase1_infonce_*` | baseline / ontology / isco_only / ontology_isco / low_weight / confit. Ontology arms change several factors at once, and `isco_loss_weight` is inert. | do not use |
| `config/go_decomposition/go_A_*` | GO/PPI domain, not career | not applicable |

`scripts/run_learning_curve_point.py` already accepts any `--train-file` and `--seed`, prints a single-factor config diff (`EXPECTED_FACTOR_FIELDS`), and asserts that the matcher is reachable (`attach_career`). `scripts/verify_career_ontneg_factor.py` checks empirically that negatives differ between arms.

## D. ESCO graph and neighbourhoods

**D1. Graphs and artifacts.** [read]
- **Pure ESCO KG**: `scripts/build_esco_knowledge_graph.py` builds `dataset/esco/esco_kg.gexf` (80 MB) from the ESCO CSVs. Edge types: `has_skill` (with `relation_type` essential/optional and `skill_type`), `in_isco_group`, `skill_parent_of`. `esco_kg_enriched.gexf/.gpickle` add skill–skill relations (`scripts/enrich_kg_with_skill_relations.py`). `OntologySkillMatcher` uses `esco_kg_path`.
- **Raw ESCO CSVs** in `dataset/esco/`: `occupations_en.csv` (`iscoGroup`, `code`), `broaderRelationsOccPillar.csv` (occupation→occupation and occupation→ISCO broader), `occupationSkillRelations_en.csv`, `ISCOGroups_en.csv`, `skillSkillRelations.csv`, `skillsHierarchy_en.csv`.
- **ESCO-derived career graph**: `training_output/career_graph.gexf` (`scripts/build_career_graph.py` → `CareerGraphBuilder`, ESCO CSVs only). `career_graph_bridged*.gexf` adds title-similarity bridge edges (`scripts/add_bridge_edges.py`). The script that produced `career_graph_bridged_complete.gexf` (the `esco_graph_path` in all lc configs) was not found. [unverified provenance]
- **Data-driven graphs**: `scripts/build_data_driven_graph.py` (and a differing root copy, `build_data_driven_graph.py`) reads training jobs (`train_final_cleaned_with_uri.jsonl`) and adds seniority, skill-similarity and title-similarity edges. It writes `training_output/career_graph_data_driven_new.gexf`. The only consumer found is `run_augmentation.py` (`load_career_graph`, default `career_graph_data_driven.gexf`), so these graphs **do not enter the learning-curve training path**. [read] Their risk is indirect: if augmented data were used, it would be derived from them.

**D2. Computing N_k(o).** [ran] All 57 v7 occupation URIs exist in `occupations_en.csv`, so N_k(o) is computable from the CSVs directly (`scripts/add_occupation_uri.py` performs the title→URI mapping). Only 18/57 have an occupation-level broader parent, so radius-1 hierarchy neighbourhoods are almost empty. ISCO-group and essential-skill-overlap neighbourhoods are the usable choices.

**D3. Neighbourhood support on v7** (n = train positives; S over other train occupations). [ran]

| target (ESCO label) | dominant job title | n_pos train | test pos | S_ISCO4 | S_ISCO3 | S_skill(J≥0.2) |
|---|---|---|---|---|---|---|
| software developer | Senior Software Engineer | 399 | 48 | 32 | 295 | 34 |
| business analyst | Business Analyst | 313 | 38 | 0 | 0 | 0 |
| data engineer | Data Engineer | 176 | 23 | 71 | 518 | 0 |
| commercial director ✗ | Senior Accountant | 127 | 14 | 0 | 0 | 0 |
| accountant | Accountant | 76 | 9 | 16 | 21 | 12 |
| foundry manager ✗ | Software Engineering Manager | 72 | 7 | 2 | 2 | 2 |
| air traffic safety technician ✗ | ERP Config / Full Stack Engineer | 44 | 8 | 0 | 0 | 0 |
| electrical engineer | Electrical Engineer | 42 | 8 | 0 | 20 | 20 |
| sales account manager ✗ | Software Development Manager | 35 | 5 | 0 | 0 | 0 |
| cloud engineer | Principal Cloud Software Engineer | 26 | 5 | 405 | 668 | 6 |
| data analyst | — | 26 | 5 | 221 | 668 | 25 |
| AI engineer | — | 25 | 5 | 222 | 669 | 26 |

✗ marks a clearly wrong mapping. Others in train with wrong mappings: university teaching assistant ← "Staff Accountant", footwear designer ← "Angular Developer", theme park technician ← "Software Engineering Supervisor". All have `occupation_match_mode = fuzzy_job_title`.

**D4. Indian dataset** (ISCO-3 neighbourhood). [ran]
- 38 targets have ≥25 train positives and ≥5 independent test positives. Of these, 15 have S_ISCO3 = 0 and 18 lie above the median S.
- 22 targets have ≥10 independent test positives. Of these, 10 have S_ISCO3 = 0 and 10 lie above the median S.
- Independent test rows per target: 11 minimum, median about 19.
- Mapping errors seen: Pilot→maritime instructor, Physician→doctors' surgery assistant, Teacher→politics lecturer, Fitness Coach→bus driver, HR Specialist→hairdresser, Nurse→nanny, AI Specialist→animal artificial insemination technician, Event Planner→wedding planner, Content Writer→advertising copywriter.

## E. Data, splits and evaluation

**E1. Which dataset.** The brief points to `preprocess/learning_curve/`, which is **v6**. The current career learning-curve pipeline uses **`preprocess/learning_curve_v7/`** (`scripts/sweep_learning_curve.py`, `LC_DIRS['career']`).
- v6: built by `scripts/learning_curve_experiment.py` as a global subsample stratified by `original_label`, with val/test symlinked to `data_splits_v6`.
- v7: built by `scripts/create_10pct_split_v7.py` + `scripts/create_lc_v7_fractions.py`. These are nested, global and stratified by label, with seed 42 and val/test symlinked to `data_splits_v7` (identical across fractions).
- Target identification: `job.occupation_uri`. `metadata.original_label` ∈ {good_fit, potential_fit, no_fit}, and top-level `label` = 1 iff good_fit. No resume occupation field, no stable resume/job IDs (`job_applicant_id` is null on v7), and identity is by content hash.

**E2. Counts.** [ran]

| | v7 frac_100 | v6 frac_100 | Indian |
|---|---|---|---|
| train rows (pos) | 6400 (1600) | 4167 (1091) | 7094 (3454) |
| distinct resumes / job texts in train | 639 / 351 | 4167 / ~4124 | 6900 / 50 |
| occupations, any rows / with positives | 58 / 33 | 120 / 51 | 50 / 50 |
| n_pos quantiles (10/25/50/75/90%) | 1.2 / 5 / 20 / 35 / 117 | 1 / 1 / 3 / 15 / 40 | 15 / 62 / 77 / 85 / 95 |
| n_pos ≥ 1/5/10/25/50 | 33/25/23/15/6 | 51/24/17/10/5 | 50/50/50/45/43 (approx.) |
| viable targets (n_pos ≥ 25 & test pos ≥ 5 / ≥ 10) | 12 / 4 | 8 / 3 | 39 / 23 (22–38 with independent test resumes) |
| test resumes also in train | 379/381 | 0/523 | 48/885 |

On v7 every test job text is also in train (272/272). Candidate judgments per v7 test resume are mostly 1–3. On Indian, each resume has exactly one judged job.

**E3. Evaluation.** [read]
- `scripts/eval_learning_curve.py` → `run_phase1_embedding_evaluation.py`. The headline metric is pooled `roc_auc_score(label, cosine)` over all 800 test pairs (`metrics.auc_roc`). Other outputs: accuracy/precision/recall/F1 at the F1-optimal threshold, and similarity statistics. Pairs are reported by (fraction, seed) with a paired t-test.
- The ORCA/eo path uses `run_ordinal_evaluation.py` (`scripts/orca_table4.py`): AUC good_vs_rest, Spearman, Cohen's d, **global** NDCG@10 and MAP_strict.
- "Candidate pool" means only the judged test pairs. There is no per-query ranking, no per-occupation output, and per-pair scores are not saved.
- `contrastive_learning/evaluator.py` has P@K/R@K/NDCG@K helpers, but these are not wired to the learning-curve path.

**E4. Train/eval serialization mismatch.** [read; impact not measured]
- The trainer serializes resumes as `Position: … [SEP] Skills: … [SEP] Profile: <first experience, 800 chars>` and jobs as `Position / Required Skills / Description (800 chars)`, with `max_seq_length = 512` (`trainer.py` ~L1219–1340, `embedding_cache.py` `_content_to_text`).
- `run_phase1_embedding_evaluation.content_to_text` serializes resumes as `Skills | Experience | Education` with no role, and jobs as `Job Title | Description` with no skills and no truncation, at the default encoder `max_seq_length`.
- The mismatch is the same for every arm, so paired deltas are not biased by it. It does depress absolute scores and changes what "zero-shot" text the projection sees. It should be fixed or frozen before preregistration.

**E5. Seeds.** [read]
- `_set_seed(training_seed)` runs in `__init__` (L103) and again in `train()` (L455). That second call overrides `BatchProcessor.__init__`'s hard-coded `random.seed(42)` (`batch_processor.py` L160).
- The runner overrides `training_seed` per run. Prior seeds: 42/13/21 for lc runs, 13/21/42/87/123 for UF probes.
- Batches are consecutive file chunks of 64 records, shuffled *within* a batch only (`DataLoader._prepare_batch`, `data_loader.py` ~L941). Seeds therefore do not change batch membership, while withholding records shifts batch boundaries for everyone.
- Checkpoint = lowest validation loss computed with the variant's own loss and negatives (`trainer.py` ~L545, `_validate_epoch`). Selection criteria therefore differ by variant.

**E6. Compute.** [ran on `results/lc_single_factor/career_*/lc_manifest.json`]
- Mean training wall time per run: 0.4 min (frac_5), 1.0 (frac_20), 2.5 (frac_50), 3.5 (frac_100; maximum 6.5).
- Hardware: Apple M2, 8 cores, 8 GB RAM, CPU only. Torch reports no CUDA, and the trainer never selects MPS.
- Excluded from these times: preload/encoding and ontology-matcher cost (cached at `embedding_cache/skill_distances_v7_backup.pkl`).
- Grid: (#targets × 5 withheld levels + 1 shared "all") × 4 variants × seeds. For example (12 × 5 + 1) × 4 × 5 = 1,220 runs × ~4–5 min ≈ 80–100 h sequential. The sweep runs sequentially by design (`sweep_learning_curve.py`).
- Withholding several targets per run is feasible. Each run takes one arbitrary train file, so a group of targets that are mutually non-neighbouring can share a run at the same retention level. A group of 5 removes ~12% of Indian positives.

## F. Existing diagnostics worth reusing

- `scripts/investigate_fix6_scale.py`: gradient-norm comparison of loss components, offline.
- `scripts/probe_ordinal_margin.py`: measures realized ontology weights, margins and curriculum effects on trained eo checkpoints.
- `scripts/probe_reliability.py`: ORCA reliability spread, a pattern for loading a checkpoint and replaying real batches.
- `scripts/probe_skill_distance_buckets.py`, `scripts/study_ontology_negatives.py`, `scripts/bench_negative_selection.py`, `scripts/evaluate_negative_selection_grid.py`: realized tier distributions of sampled negatives. These are the base for a negative audit; none checks false negatives.
- `scripts/verify_career_ontneg_factor.py`: proves two arms select different negatives.
- `scripts/text_vs_ontology_ceiling.py`, `scripts/mechanism_report.py`, `scripts/ontology_ceiling_report.py`: text-vs-ontology redundancy. Prior result: ESCO ceiling 0.558, ISCO 0.485, near chance. These are the right control for the "text similarity between target and neighbours" confound.
- `scripts/audit_occupation_task_readiness.py` and `results/occupation_task_readiness/`: per-occupation train/val/test support and the resume-overlap audit. Its screen of ≥25 train positive resumes and ≥10 independent test positive resumes passes 0 career v7 targets and 21 Indian targets.
- `scripts/check_unseen_occupation_scoring.py`: checks only that the projection executes. It makes no transfer claim.
- No existing code probes relation preservation in the learned space (embedding similarity vs ESCO distance).

---

## Feasibility verdict per design element

1. **Targets stratified by n(o) and S(o): requires new implementation, plus a data fix.** No S(o)/N_k code exists, though the ESCO CSVs make it straightforward. On v7 the occupation labels are not trustworthy and there are too few targets. Use Indian, or audit and fix the mapping first.
2. **Retain 0/1/5/10/25/all of each target's positives: requires new implementation (split generator).** The runner accepts arbitrary train files, so only the generator is missing. All levels are feasible for targets with n ≥ 25 (v7: 15 occupations; Indian: about 45).
3. **Fixed test set and candidate pool: supported with small changes.** The test files are fixed. A target-wise metric needs per-pair score export and a defined candidate pool, which does not exist yet.
4. **Four variants: supported with small changes.** A "both" config is needed. If "ontology loss" must mean an additive term encoding ESCO occupation relations (the only way S(o) can act), it **requires new implementation**.
5. **Leakage checks and mechanism diagnostics: requires new implementation.** Validation leakage must be fixed, and component-loss/gradient logging, a relation probe and a false-negative audit must be added. The cache is safe and the data-driven graphs are off-path.
6. **Paired uncertainty across targets and seeds: requires new implementation.** Only a pooled paired t-test exists (`eval_learning_curve.paired_t`). There is no mixed model and no bootstrap.
7. **Preregistered criteria: not code.** It depends on settling the open decisions below.

## Gap list (ordered by dependency)

1. **Freeze the code and data.** Commit or stash the uncommitted `batch_processor/loss_engine/trainer` edits and record SHA-256 hashes of the data files (`audit_occupation_task_readiness.py` already prints them).
2. **Occupation-label audit and fix.** Build a hand-checked title→ESCO URI table for every candidate target and its neighbours (v7: 190 titles; Indian: 50) as a new mapping file, plus a script that rewrites `job.occupation_uri` into a new split directory. Do not edit splits in place.
3. **Neighbourhood module.** Add a new `scripts/esco_neighbourhoods.py` that reads only the ESCO CSVs and outputs N_k(o) for a preregistered edge type and k, S(o) from baseline train counts, and the strata. Write it to JSON and hash it.
4. **Per-target withholding split generator.** Add a new `scripts/create_target_withholding_splits.py`. It keeps k ∈ {0, 1, 5, 10, 25} target positives (nested, seeded and stratified) and applies a preregistered rule for the target's non-positive records.
   - It removes target records from **validation** too, or switches to a validation set that excludes all targets, because validation drives `best_checkpoint.pt`.
   - It keeps file order stable for the remaining records, minimizing batch-boundary shifts.
   - It optionally supports groups of mutually non-neighbouring targets per file.
   - It emits a manifest listing the removed record hashes.
5. **Leakage assertion in the runner.** Extend `scripts/run_learning_curve_point.py` to assert that the global job pool (`trainer.global_job_pool`), the train file and the validation file contain no withheld target record or job text beyond the k retained, and to record this in `lc_manifest.json`. Assert that no augmentation or data-driven graph inputs are used.
6. **"Both" config.** Add `config/lc_career_both.json`: `ontology_weight` 0.3 + `ontology_guided_negatives`/rank tiers true, `use_ot_distance` false, pathway false. Add it to `ARMS` in `sweep_learning_curve.py` and to the factor list in `run_learning_curve_point.py`.
7. **If the ontology loss must encode occupation relations, add an additive `L_ontology` to `loss_engine.py`.** For example, a margin or InfoNCE term over occupation entity embeddings built from deployment-available ESCO text (preferredLabel + description) through the same projection: pull N_k neighbours together and rank by graph distance. Make λ a separate config field that defaults to 0 so existing arms stay byte-identical.
8. **Component-loss and gradient logging.** In `trainer._train_batch` (~L960–1015), log L_CL, the ontology term (or mean/sd of the sample weight), the pre-clip total grad norm, and per-component grad norms via `torch.autograd.grad` on a sampled subset of batches (pattern: `investigate_fix6_scale.grad_norm_wrt`). Also log whether clipping bound. Write a per-epoch JSONL to the run directory.
9. **Variant-independent checkpoint selection.** Either use a fixed epoch count or compute validation loss with plain InfoNCE and fixed negatives for all variants (`trainer._validate_epoch`). This removes a selection confound.
10. **Per-pair score export and per-target metrics.** Make `run_phase1_embedding_evaluation.py` write `pair_scores.jsonl` (content hashes, `occupation_uri`, label, original_label, cosine). Add a new `scripts/eval_target_withholding.py` that computes the following per target, plus a global non-target control metric to detect collateral damage:
    - (a) AUC over the target's judged test pairs;
    - (b) rank/MRR/Recall@K of the target's job for its positive test resumes against a fixed candidate pool;
    - (c) the same metrics restricted to resumes absent from train.
11. **Align eval text serialization with training** (`run_phase1_embedding_evaluation.content_to_text` vs `BatchEfficientEncoder._content_to_text`), or document it as a frozen property.
12. **Relation-preservation probe.** Add a new `scripts/probe_relation_preservation.py`: per checkpoint, compute Spearman between projected occupation-embedding cosine and ESCO distance, and neighbourhood recall@k. Report it for target↔N_k(o) separately from the global value.
13. **Negative audit.** Extend `probe_skill_distance_buckets.py`/`study_ontology_negatives.py` into a new `scripts/audit_sampled_negatives.py`. It should replay real batches and report, per tier: the share of same-occupation candidates, same-ISCO-4 candidates, candidates judged good/potential for the anchor resume, and candidates in N_k(anchor occupation). Run it per variant.
14. **Statistics.** Add a new `scripts/analyze_target_withholding.py`: a mixed model of Δ ~ log(k+1)·log(S+1) (and log n) with random intercepts for occupation and seed, using statsmodels if it is pinned, plus a paired bootstrap over targets × seeds. Fix the analysis seed and keep the targets and alpha preregistered.
15. **Text-similarity control.** Compute frozen-encoder cosine between the target's job text and its neighbours' job texts as a preregistered covariate, reusing the `text_vs_ontology_ceiling.py` machinery.

## Risks and open decisions for the protocol

- **Dataset choice.** v7 has too few viable targets, labels that are heavily mis-mapped, and test resumes that are 99% seen in train. Indian has 50 single-job occupations, so "occupation" and "job text" coincide, and resumes are largely independent. Its catch is one judged job per resume, so negatives and ranking candidates are unjudged. The choice changes which leakage rules and metrics apply.
- **Neighbourhood definition.** The edge type must be fixed: ISCO-4, ISCO-3, ESCO broader/narrower (sparse: 18/57 v7 occupations have an occupation-level parent), or essential-skill Jaccard ≥ τ. Fix k, the threshold, and whether S counts positives or all rows, all from baseline train counts. Under ISCO-3, about 40% of Indian targets have S = 0, which gives a natural low-S stratum.
- **What "withholding" removes.** Positives only, or also the target's `no_fit`/`potential_fit` records? Keeping them leaves the target's job text in the negative pool, so it is pushed away and acts as a hard negative for neighbours. Removing them makes retention 0 a true cold item. Decide whether the "all" level is the unmodified full-data run, shared across targets.
- **Zero-shot boundary.** At k = 0 the current mechanisms cannot touch the target. Decide in advance whether a k = 0 Δ is interpreted as an ontology effect at all, or only as a global-regularization effect. The S(o) interaction is only interpretable if an occupation-relation L_ontology exists (gap 7).
- **Conflict in "both".** Guided negatives push skill-similar occupations apart, while a relation loss pulls them together. Preregister which direction is predicted for the interaction.
- **Effective λ.** The current sample weight is near-uniform (CV 6–11%) and gradient clipping at 1.0 may bind. Preregister λ and the clip, and report realized ratios of loss and gradient scale.
- **Rare targets.** Most occupations cannot support k = 25 (v7: n ≥ 25 for 15 occupations). Decide whether rare occupations enter only the low-k levels, or are excluded.
- **Per-target test noise.** There are 4–48 test positives per target, so per-target AUC is coarse. Preregister minimum test support, primary metric, aggregation, and seeds (at least 5; the per-cell seed sd in prior runs was 0.015–0.025 AUC).
- **Batch-composition confound.** Fixed file-order batching means withholding shifts batches for all records. Mitigate with stable ordering or by shuffling the record order per seed in the generator, and preregister which.
- **Multi-target runs.** Grouping targets cuts compute about 5× but removes neighbour data if targets neighbour each other. It needs a non-neighbour constraint and a per-run cap on total withheld positives.
- **Confounds.** Text similarity between a target and its neighbours (encoder redundancy has already been shown on career/GO), title collisions (the pool deduplicates by title), and occupation-mapping errors.
- **Checkpoint selection.** Validation leakage and variant-specific validation loss must be fixed before any run counts.

**Not verified:** the provenance of `career_graph_bridged_complete.gexf` (only inertly loaded); the impact of the eval serialization mismatch; actual runtimes on Indian and for guided-negative arms beyond the v7 manifests; and whether `statsmodels` is installed in `.venv`. The gexf graphs were not loaded; neighbourhoods were computed from the CSVs.
