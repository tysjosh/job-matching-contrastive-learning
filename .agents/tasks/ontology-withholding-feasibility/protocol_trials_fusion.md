# Preregistration: MeSH score fusion in TREC-CT trial matching, confirmation (2022) + replication (2021 CV)

Status: **DRAFT v7, written before any new result**, plus pre-pins amendments §9 A0 (disclosure) and **§9 A1 (cohort 2022 retrained on one device; supersedes the 15 + 4 population wherever the text below still describes it)**. Round 7. It resolves every finding in `design_review_trials_fusion.{json,md}` (round 6, CHANGES_REQUESTED: 0 HIGH, 2 MEDIUM, 3 NIT), on top of the 18 round-1, 8 round-2, 6 round-3, 7 round-4 and 5 round-5 findings. The finding-to-section maps are in §10 (round 1), §11 (round 2), §12 (round 3), §13 (round 4), §14 (round 5) and §15 (round 6).
- Repo: `/Users/olukotunjosh/Downloads/CDCL` (local development checkout), branch `orca-experiments-and-runner`. HEAD at writing: `b524f2a` (its diff from `147b1e7` touches only `.agents/**`). `pins.json` records the HEAD at pin time on the remote.
- **Execution location.** All gates, training, scoring and analysis run on **one** remote NVIDIA GPU host (Linux x86_64): the user's A100, or whichever single host `pins.environment` records. The protocol does not hard-code a GPU model; it pins the one that is used. `pins.json` (revision 1) is written on that host, so `pins.environment` records it (GPU name from `torch.cuda.get_device_name(0)`, driver, CUDA, torch build, Python, BLAS), and gate 3 (§5.3) requires every training run to report that same GPU name. Nothing confirmatory runs on the Mac. The Mac is used only for development, unit tests during development, and a non-confirmatory CPU smoke whose outputs go under `results/trials_fusion_confirm_smoke/` and are never read by any confirmatory script. The transfer list, environment setup and exact remote commands are in the runbook `.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_TRIALS_FUSION.md`, written during implementation; where the runbook and this protocol differ, this protocol wins.
- **Device policy (fixed).** Training of the 19 + 15 new runs (§9 A1) uses `--device cuda`. Scoring (`score.py`, including gate 2), every fresh 2021 encode, the paraphrase cosine computation, and analysis run on the remote **CPU** (`CUDA_VISIBLE_DEVICES=""`, `--device cpu`, `SentenceTransformer(..., device="cpu")`).

## 0. What is already known, and what is new

Exploratory, post hoc results (source: `findings_trials_where_ontology_helps.md` §B1, §C; computed by `/tmp/trials_helps/{score,lib,a2_fusion,a6_twolevel,a7_controls,a7b,a5_hard}.py`):
- Population: 15 baseline checkpoints × 50 test patients (2022).
- Validation-nDCG@10-selected fusion of z(text) and z(φ_MeSH, hierarchy) gave:
  - macro within-patient pooled AUC +0.013 [+0.006, +0.020];
  - AP +0.017 [+0.006, +0.029];
  - ineligible-vs-not-relevant AUC +0.014 [+0.007, +0.022];
  - hard (eligible vs ineligible) −0.001, ns;
  - nDCG@10 +0.014, ns.
- Shuffled-topic φ: −0.002. Hierarchy − exact at w = 0.2: pooled +0.005.
- Zero-shot MPNet + φ (w = 0.2): pooled +0.012 [+0.005, +0.018].
- Trained text is 0.07–0.13 nDCG@10 below zero-shot.

Consequences for interpretation, fixed now:
- **Cohort 2022 is a frozen-pipeline confirmation, not an independent test.** 15 of its 19 checkpoints and all 50 patients have already been analysed. The only new information there is 4 new training seeds plus the restricted, preregistered estimator. *(§9 A1: all 19 cohort-2022 checkpoints are now retrained on the pinned GPU host, so every 2022 model is new; the 50 patients are still already analysed.)*
- **Cohort 2021 (5-fold patient-disjoint CV) is the only replication with new models and mostly new patients.** Every CV model is new, but not every patient is unseen.
  - **Prior exposure, disclosed:** the 15 topics of `preprocess/trec_ct_splits/validation.jsonl` (2021_6, 2021_16, 2021_17, 2021_23, 2021_26, 2021_37, 2021_42, 2021_45, 2021_49, 2021_51, 2021_61, 2021_63, 2021_64, 2021_69, 2021_74; pinned as `pins.cohort2021.exposed_topics`) were the fusion-weight validation set in the exploration. Fused-vs-text validation nDCG@10 curves (`validation_selection_curve` in `trials_fusion_learning_curve_ndcg_selected.json`) and per-pair φ (hierarchy, exact, shuffled; `/tmp/trials_helps/score.py`, `a7_controls.py`) were computed on them with the 15 published checkpoints, and `findings_trials_where_ontology_helps.md` §B1 reasons from their validation behaviour. In the CV each of them becomes a test patient in some fold. The other 60 2021 topics have never themselves been scored under fusion. But 8 of those 60 are §3 paraphrase partners of an exposed topic (2021_15, 21, 25, 27, 33, 40, 57, 59), and near-duplicates share most MeSH descriptors and judged pools, so their outcome was effectively seen. **Closed exposure set:** `exposed_closed` = the union of the §3 paraphrase components that contain at least one `exposed_topics` member. Expected size 23 (the 15 above plus those 8), asserted by `build_cv2021.py` and pinned as `pins.cohort2021.exposed_topics_closed`. **52 2021 patients are closed-unexposed**: neither scored under fusion nor a paraphrase of a scored patient. The preregistered sensitivity uses these 52 (§4.4). *(§9 A0: the Mac dev smoke later scored 13 of them under fusion on the validation side; a descriptive 39-patient subset excludes them.)*
  - Prior exposure, disclosed: the φ separation table in `findings_trials.md` §C used 2021 *train* topics and their judged pools (φ alone, no fusion, no model).
  - Prior exposure, disclosed: the round-1 design review and this revision computed pairwise topic cosines of the 75 2021 topics. These are label-free; no score, grade or qrel was used.
- Nothing about seeds 87/123 or about any 2021 CV model exists yet.

## 1. Hypotheses

All effects are paired: Δ = metric(fused score) − metric(text score) on the **same** text model, computed per patient and macro-averaged over patients.

**Co-primary (Holm across the two, within each cohort, family-wise α = 0.05, two-sided; decision rule in §4.3):**
- **H1**: fusion improves within-patient macro pooled AUC (eligible vs ineligible ∪ not-relevant). Δ_H1 > 0.
- **H2**: fusion improves within-patient macro AP (grade 2 = relevant). Δ_H2 > 0.

The primary text-model population is the **trained baselines** (§3b). Zero-shot (§3a) is a key secondary with its own Holm pair (H1z, H2z). It is reported in the same table but is not part of the replication criterion.

**Secondary (95% CIs, no multiplicity correction, labelled as uncorrected):**
- S1: nDCG@10 (gains 0/1/3). Predicted Δ > 0.
- S2: P@10 (grade 2). Predicted Δ > 0.
- S3: g1-vs-g0 AUC (ineligible vs not-relevant, within patient). Predicted Δ > 0.
- S4: hard AUC (eligible vs ineligible, within patient). **Predicted null.** Equivalence bound ±0.01, using the 90% two-level CI:
  - *equivalent* iff the CI ⊂ (−0.01, +0.01);
  - *not equivalent* iff the CI lies entirely outside [−0.01, +0.01] (LB ≥ +0.01 or UB ≤ −0.01);
  - *inconclusive* otherwise.

**Controls.** All controls use the same checkpoints and the per-variant rule of §2.5.
- **Model-type scope.** C1–C3 and S4 are computed for both the trained population and zero-shot, in each cohort. Only the **trained** C2 state applies the §4.3 overrides. Zero-shot C1–C3 and S4 (including the zero-shot C2 state) are reported descriptively and change no headline.
- **C1, exact-overlap φ.** Predicted Δ_pooled > 0, but smaller than hierarchy.
- **C2, shuffled-topic φ** (seed 11, derangement, §2.3). Predicted no gain. Evaluated **per cohort** on Δ_pooled. The states are evaluated **in order; the first matching state applies** (fail takes precedence, which is the conservative choice):
  1. *fail*: LB95(C2) > 0, **or** [LB95(hier − C2) ≤ 0 **and** point(hier) ≤ point(C2)];
  2. *pass*: UB95(C2) < +0.005 **and** LB95(hier − C2) > 0;
  3. *inconclusive*: otherwise.
  - Example: C2 = +0.002 [+0.001, +0.004] with LB95(hier − C2) > 0 is *fail* (state 1 matches first), even though it also meets the *pass* condition.
- **C3, hierarchy − exact contrast**, paired per patient, in two forms:
  - (i) primary form: each variant with its own validation-selected w;
  - (ii) both variants at fixed w = 0.2, each with its own validation z-stats (replicates the post hoc number).
  - Predicted > 0. Labelled suggestive whatever the outcome.

**Interpretive rules fixed now:**
- The trained C2 state is per cohort and applies to **both** H1 and H2. How it combines with "confirmed"/"replicated" is fixed by the ordered per-hypothesis final-label rule in §4.3, which is the only place a C2 state affects any label or headline.
- No eligibility claim is made from global (cross-patient) hard AUC. Global AUCs are reported for continuity only and are never interpreted.
- Fusion is an inference-time scoring rule. Nothing here supports a claim about using the ontology at training time.

## 2. Scores and fusion rule

### 2.1 Text score and embeddings
- **Trained text score.** Same computation as `epf.text_scores` (`scripts/evaluate_publication_fusion.py` L295–319): project frozen `all-mpnet-base-v2` 768-d embeddings with `best_checkpoint.pt`, then take the dot product of the 128-d projections. Runs on CPU.
- **Zero-shot text score.** Cosine of the frozen 768-d embeddings, no projection (`raw_cos` in `/tmp/trials_helps/score.py`).
- **Serialisation.** `run_phase1_embedding_evaluation.content_to_text` on `encoder_view`, unchanged.
- **Record handling.** Identical to `epf.prepare_split`:
  - grade comes from `job.grade`, with an `original_label` fallback;
  - the query is `metadata.resume_id`;
  - a blank text raises.
- **Embedding source, read-only.** `score.py` has its own loader, a port of `epf.prepare_split`. It never calls `epf.prepare_split` on the published directory, because that function `mkdir`s and can write chunks there. Each text's vector is resolved in this order, the same order `epf` uses:
  1. the frozen cache `embedding_cache/trials_text_embeddings.pt`, looked up by `EmbeddingCache.get_content_key(slot)` (sha256 pinned);
  2. for 2022 files only: the published prepared cache, looked up by the text's sha256 key. Path: `results/publication_phase1_fixedval/prepared_embeddings/trials/{stem}_{sha16}_sentence-transformers_all-mpnet-base-v2/chunk_*.npz`.
     - The `test_4030cc85ff1b4ef3_…` directory has 36 chunks. Each chunk's sha256 is pinned. The manifest digest is defined cwd-independently: the sha256 of the UTF-8 text made of one line `"<sha256>  <basename>\n"` per `chunk_*.npz`, lines sorted by basename. This equals the output of `shasum -a 256 chunk_*.npz | sort -k2 | shasum -a 256` **run with cwd = the test chunk directory**, and is `3d015e60c7d8484a12289299023fa8101a6d12c594af4812995e15e411c03f11`. `make_pins.py` computes it from basenames, never from relative or absolute paths.
     - The `validation_5defb59b858bdeee_…` directory has 0 chunks, so every 2022-validation text resolves from the frozen cache.
     - For 2022 files, a text missing from both sources is a hard abort. No 2022 text is ever re-encoded, and **no `SentenceTransformer` is constructed for 2022 files**.
     - `epf.text_scores` reads `prepared["embedding_dim"]`, which `epf.prepare_split` normally obtains from an encoder. `score.py`'s loader instead sets `embedding_dim = 768` and asserts it equals the width of every loaded vector (frozen-cache and chunk alike); a mismatch is a hard abort.
  3. for 2021 fold files only: a fresh encode on the remote CPU with `SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")`, `batch_size=64`, `convert_to_numpy=True`, no normalisation (as in `epf`). The model is loaded from the transferred Hugging Face snapshot, revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` (the local snapshot; `pins.environment.mpnet_snapshot` records it).
     - **Snapshot check (defined).** Every script that constructs a `SentenceTransformer` (and `verify_transfer.py`, `make_pins.py`) runs with `HF_HUB_OFFLINE=1`. The revision is the snapshot directory name that `huggingface_hub.snapshot_download("sentence-transformers/all-mpnet-base-v2", local_files_only=True)` resolves (i.e. the basename of the returned `.../snapshots/<rev>` path). It must equal `e8c3b32edf5434bc2275fc9bab85f82640a19130`, or the script aborts. The `SentenceTransformer` is then constructed from that resolved local path, so the checked snapshot is the one loaded. Chunks are written in `epf`'s chunk format under `results/trials_fusion_confirm/prepared_embeddings/trials/`.
- The original `embedding_cache/trials_text_embeddings.pt` is opened read-only and its hash is checked before and after scoring.

### 2.2 φ (hierarchy, production)
φ(topic, trial) = `build_mesh_matcher(config).ontology_set_similarity(topic.skill_uris, trial.skill_uris)`.
- `config` = `config/lc_trials_baseline.json`: `mesh_alpha` 0.5, `mesh_max_hops` 12, `similarity_mode` default `"hierarchy"`.
- `skill_uris` = `mesh_uris` (all descriptors, condition + intervention).
- Empty side or exception → 0.0. This is identical to `epf.prepare_split`.

### 2.3 Control φ
- **Exact.** `MeshMatcher(MeshIndex.build_or_load(config.mesh_descriptor_path, config.mesh_index_cache), alpha=0.5, max_hops=12, similarity_mode="exact")`. It uses the same URIs and the same pinned `embedding_cache/mesh2021_index.pkl`.
- **Shuffled (derangement).** Applied separately to each scored file (every validation file, every test file):
  1. Take that file's distinct topics, sorted by key `(year, int(topic_id.split('_')[1]))`; n = their count.
  2. **Group-aware derangement.** `group(t)` = the §3 paraphrase component of topic t (pinned in `pins.cohort2021.components`; every topic not in a multi-topic component, including every 2022 topic, is its own singleton group). `rng = np.random.default_rng(11)` is created **once per file**. Repeat `order = rng.permutation(n)` until no i has `group(order[i]) == group(order[(i+1) % n])`. Abort (hard error, amend §9) after 10,000 draws. The number of draws used is logged per file in `scores/.../selected_w.json` under `shuffle_draws`.
  3. Topic `order[i]` receives the topic-side `skill_uris` of topic `order[(i+1) % n]`. The cyclic shift excludes fixed points, and the group condition excludes paraphrase partners, so no topic receives its own or its paraphrase partner's MeSH set.
  4. Trial URIs are unchanged. Hierarchy matcher.
  - For a 2022 test file all groups are singletons, so the first draw is accepted and the mapping equals the plain cyclic-shift rule.
  - The cohort-2022 validation file contains both 2021_26 and 2021_63 (one component), and each 2021 fold contains whole components; the group condition is what protects those files.
  - This differs from the exploratory a7 version, which used a plain permutation and so allowed fixed points and partner swaps. Either would leak true φ into the control and bias C2 toward *fail*.

### 2.4 Normalisation, as implemented
`epf` does **not** z-score within patient. `z_stats` computes the mean and population SD (`np.std`, ddof 0; SD 0 → 1) over the **whole validation file**, and `z_apply` applies those constants unchanged to test (`main` L632–666). φ stats are fit once per validation file; text stats are fit per text model. The exploratory numbers used the same rule (`lib.z(v, ref)`).
- **Primary: validation-file global z, exactly as implemented.** Reason: this study confirms an existing estimator, and changing the normaliser would test a different one. It uses no test labels or test scores.
- **Sensitivity** (reported, not in any criterion): within-patient z. Mean and SD are taken over each patient's judged pool, for validation and test alike. **SD = 0 → z = 0** for that patient and score (e.g. 2022_31's φ). w is re-selected on validation under this normaliser.

### 2.5 Fusion and weight selection, per φ variant
- score = (1 − w)·z(text) + w·z(φ).
- For each φ variant v ∈ {hierarchy, exact, shuffled}, each text model m (checkpoint, or zero-shot), and each cohort/fold:
  1. Compute v on the validation file and the test file.
  2. Fit `z_stats` for v on the validation file only, and for m's text score on the validation file only.
  3. Select w on validation by macro nDCG@10 (`epf.ranking_metrics(...)["macro"]["ndcg@10"]`, gains 2^g − 1, NaN patients dropped) over W = `[i/10 for i in range(6)]` = {0.0, …, 0.5}.
  4. **Ties** (exact float64 equality) → the smaller w (`max` over ascending W returns the first maximiser, as in `epf.main`).
- w = 0 is allowed. Then Δ = 0 for that model, and it stays in the average.
- C3(ii) skips selection and uses w = 0.2 for both variants, each with its own validation z-stats.
- Grid note: the published grid is 0–1.0, but all 15 published nDCG-selected weights are ≤ 0.4, so the restricted grid changes none of them.

### 2.6 Metrics, per patient
Per patient (`metadata.resume_id`), over that patient's judged trials, all grades, no cap on test:
- **AUCs** (tie-aware rank AUC, `epf.rank_auc`):
  - pooled: 2 vs {0, 1};
  - hard: 2 vs 1;
  - g1-vs-g0: 1 vs 0.
- **AP**: grade 2 relevant, full ranking (`epf.ranking_metrics`).
- **nDCG@10 and P@10**: as in `epf.ranking_metrics`.
- **Undefined metrics.** A patient missing a class is excluded from that metric only. The undefined set is the same for every score, so pairing holds.
- **Topics with weak MeSH signal.** φ ≡ 0 only for **2022_31** (empty `mesh_uris`). It stays in, with Δ = 0 for every φ variant. Sensitivity: exclude the 7 topics with `condition_signal_present == false` (2021_53, 2021_67, 2022_1, 2022_10, 2022_15, 2022_17, 2022_31), whose MeSH sets contain no condition descriptor.

## 3. Text models and checkpoint populations

### 3a. Zero-shot
Frozen MPNet cosine, identity, no training. It is one deterministic function:
- cohort 2022: 1 model;
- cohort 2021: the same function per outer fold, with that fold's validation file used for w.

### 3b. Trained baselines
- Config: `config/lc_trials_baseline.json` (sha256 in §7), run through `scripts/run_learning_curve_point.py`.
- Training: InfoNCE, uniform graded negatives, 15 epochs.
- `best_checkpoint.pt` = lowest validation InfoNCE on grade-2 validation positives, with fixed validation negatives (seed 1729).

**Derived configs.** The trainer writes the cache file at `config.embedding_cache_path` (`trainer.py` L931, after validation if the cache grew, and L1643, after preload). Both writes read the same config attribute, so one redirect covers both. To leave `embedding_cache/trials_text_embeddings.pt` untouched, each new run uses a derived copy of `lc_trials_baseline.json` that differs **only** in:
- `embedding_cache_path` → `results/trials_fusion_confirm/cache/trials_text_embeddings.pt`. This is a byte copy of the original, made in §6.2 step 0 with `cp -n` (never overwrites). It is a mutable performance cache: training may append to it. Its **starting state** is pinned: `make_pins.py` asserts `sha256(copy) == sha256(original) == fa83e10a95a37a378c3d54e110678a64f702a850aef2442b06148d2589e31d4e` and records it as `pins.inputs["results/trials_fusion_confirm/cache/trials_text_embeddings.pt@pin_time"]`; later states are logged per run in the ledger, not pinned. Under `--device cuda`, vectors the trainer appends to the copy are CUDA encodes, used only by later training runs (scoring never reads the copy, §2.1). Because whether a run reads a cached (original-device) vector or a fresh CUDA encode depends on earlier runs, **run order is fixed and part of the condition**: 2022 F 10, 25, 50, 75, 100, within each F seeds 42, 13, 21 and, at F 25 and 100, then 87, 123 (§9 A1); then 2021 k = 0..4, within each k seed 13, 21, 42. `train.py` logs the cache's entry count and sha256 before and after each run in `run_ledger.json` (§6.3).
- for cohort 2021 only, `trials_split_dir` → `preprocess/trials_fusion_confirm/cv2021/fold_{k}`. The negative selector reads `{trials_split_dir}/negative_pools.jsonl`.

These fields change where data and cached frozen embeddings are read from. They do not change the model, loss or schedule. Each run passes `--compare-config config/lc_trials_baseline.json`. The single-factor audit (`report_factor`, stored by `run_learning_curve_point.main` at `lc_manifest.json` → `single_factor_audit.unexpected_fields`) is then required to report:
- `single_factor_audit.unexpected_fields == ["embedding_cache_path"]` for cohort 2022;
- `single_factor_audit.unexpected_fields == ["embedding_cache_path", "trials_split_dir"]` for cohort 2021.

The audit's WARNING line is expected. Any other set fails gate 3 (§5).

### Cohort 2022 population: 19 checkpoints
*(§9 A1: the confirmatory population is the 19 (F, S) cells below (the 15 existing cells plus the 4 new ones), all **retrained** on the pinned GPU host with `--device cuda`, to `results/trials_fusion_confirm/cohort2022/checkpoints/trials_baseline_f{F}_s{S}/`. The 15 published checkpoints in the table are used only by gate 2.)*

**Existing 15.** Path pattern: `results/publication_phase1_fixedval/checkpoints/trials_baseline_f{F}_s{S}/best_checkpoint.pt`. Train file: `preprocess/trec_ct_lc/frac_{F}/train.jsonl`. Validation file (`validation_path` in each `training_config.json`): `preprocess/trec_ct_lc/validation_positive.jsonl`. Training device: `null` (auto).

| F | S | train records | best_checkpoint.pt sha256 |
|---|---|---|---|
| 10 | 42 | 441 | 1cc69264e0dc0452605287ac5a79181cf02ba1022b4a30c94b6068ad16f33195 |
| 10 | 13 | 441 | e8b6a30ecc8b9b5376b0f6c998cc3a8acb2d19e665e8572934f2d72d16a52d6b |
| 10 | 21 | 441 | f35de3be330cfa677052f6a1fa6994224d30195ef3fa0e7321968d857c6bffe8 |
| 25 | 42 | 1102 | 7f0eb96f2a4ca97aaec09d6cb9018bfe34716325f95d9300f2e29c618f42c961 |
| 25 | 13 | 1102 | 8af9aaa0e293505b97773384d0ac626a73261fa4a52cf416ee25e049269a08f0 |
| 25 | 21 | 1102 | c3286f97ae3346cb86ede69d92572e9fb8fa5a463f71718af04c92a06c8aa95d |
| 50 | 42 | 2204 | 4223d683787abfb5660f29f4e1e1a45652860847a66134b30abd618412d1fbfd |
| 50 | 13 | 2204 | a0cd38bbc68965dd09e847b6dd0358e891d4ab9e0db4955047e50e8d04b2bddd |
| 50 | 21 | 2204 | f49070675ef8223cda797ba5dad6df7cca21f68c3a6c52af823047ab3876e8bc |
| 75 | 42 | 3306 | b68d0d694fc8b590cc9a06b7c0eed898e0bf514080272c6233cbdd55632ca9ca |
| 75 | 13 | 3306 | b83b7a51a70d10fc3378261de1cf9ce4032ad2664fad8226fa2a78730be866cd |
| 75 | 21 | 3306 | d180e2e1df1d7609e8841599a5705916be8e5de5eac7550ebbc5e9df590c5978 |
| 100 | 42 | 4408 | ebc411127139613838180fc8dad6e81e9dfc1cd107d341a7cb0c02478f9081be |
| 100 | 13 | 4408 | e0ba6967394b2488ea798ef7a7f6746ae9138b31cd96ddf803bb753b924fa9d0 |
| 100 | 21 | 4408 | bfa32901701db4d446f19774de304161a8a2de414b466e5283aea7043ab52c1f |

**New 4.** F ∈ {25, 100} × S ∈ {87, 123}, trained on the remote GPU host (`pins.environment.gpu_name`) with `--device cuda`. Output: `results/trials_fusion_confirm/cohort2022/checkpoints/trials_baseline_f{F}_s{S}/`.

Every checkpoint gets equal weight, so the population has 5 seeds at F = 25 and F = 100 and 3 seeds at the other fractions. This is accepted: the fraction effect on Δ was flat in exploration, and no per-fraction claim is made. A fraction-stratified bootstrap is reported as a sensitivity (§4.1).

Cohort 2022 files:
- Validation for w: `preprocess/trec_ct_splits/validation.jsonl` (15 2021 topics, grade 0 capped at 60).
- Test: `preprocess/trec_ct_splits/test.jsonl` (50 2022 topics, uncapped).

### Cohort 2021 population: 15 checkpoints (5 outer folds × seeds 13/21/42) + 5 zero-shot

**Topic set.** Topics and qrels are filtered to `year == "2021"` (75 topics). Wherever topics are sorted, the key is `(year, int(topic_id.split('_')[1]))`.

**Paraphrase grouping (label-free for scored outcomes).** Edges between 2021 topics come from two sources.
1. The fixed list from `findings_trials.md` §A (prefix `2021_`): 40–74, 27–64, 25–61, 33–69, 57–63, 23–59, 39–73, 41–75, 36–71, 34–70, 38–72, 21–57, 24–60, 30–66, 31–67, 22–58.
2. Any pair with cosine **≥ 0.80**, computed as follows:
   - vectors are taken from the frozen cache `embedding_cache/trials_text_embeddings.pt`, keyed by `EmbeddingCache.get_content_key(_ontology_slot(topic, "topic_id", topic_id))` (as `data_splitter` L253 builds topic slots);
   - vectors are L2-normalised in float64;
   - a fresh CPU encode (`device="cpu"`, remote) is used only for a topic missing from the cache, and is logged;
   - the cosine values are compared to 0.80 in float64. If any pair falls within 1e-6 of 0.80 on the remote, `build_cv2021.py` stops (amend before building folds); none does locally (nearest values 0.810 and 0.778).

   All 75 topics are currently in the cache.

Groups = connected components of the edge graph.

Expected outcome, computed label-free while writing this revision; `build_cv2021.py` asserts it:
- the cosine rule adds exactly 3 edges outside list (1): 26–63 (0.895), 26–57 (0.826) and 15–21 (0.810);
- the largest component is {2021_15, 2021_21, 2021_26, 2021_57, 2021_63} (5 topics).

Guards, enforced by `build_cv2021.py`:
- If the added-edge set differs from the expected one, or any component has > 6 topics, stop and amend this protocol before building folds.
- `exposed_closed` (§0) must equal exactly {2021_6, 15, 16, 17, 21, 23, 25, 26, 27, 33, 37, 40, 42, 45, 49, 51, 57, 59, 61, 63, 64, 69, 74} (23 topics; 52 closed-unexposed). A mismatch means the components changed: stop and amend before building folds.
- `pins.json` records every component, every edge with its source, and every pair with cosine in [0.70, 0.80) for audit. Pairs below 0.80 (e.g. 45–53 0.778, 13–60 0.758, 5–62 0.730, 7–64 0.715) were read and are not paraphrases.

Disclosure: list (1) was partly found with judged-pool Jaccard (qrels). It is used only to keep topics together, which can only make the split more conservative. It never enters a score.

**Fold assignment (fold seed 2021).**
1. Order the groups by their smallest topic key.
2. Shuffle the groups with `random.Random(2021).shuffle`.
3. Assign each group in turn to the fold with the fewest topics so far (ties → lowest fold index).

The result is 5 folds of about 15 topics each.

**Per outer fold k ∈ {0..4}:**
- test = fold k;
- inner validation = fold (k+1) mod 5;
- train = the other 3 folds (≈ 45 topics, all grade-2 pairs).

The inner validation fold serves `best_checkpoint` selection, z-stats and w selection. The test fold is never read before scoring.

**Deviation from the step brief, with reason.** The brief says "train on 4 folds". That is impossible without leakage. The runner needs a validation set for `best_checkpoint` selection and for w selection, and the only other held-out data is the test fold. 3 train folds + 1 inner validation fold mirrors cohort 2022 (train 60 / validation 15) and uses the same w rule in both cohorts. The learning curve is flat (5% ≈ 100% positives, findings §B5), so losing one fold of training data should not matter.

**Fold files.** Built per outer k in `preprocess/trials_fusion_confirm/cv2021/fold_{k}/`. Records come from `trials_domain.data_splitter._build_record` / `_ontology_slot` (imported, not edited), applied to `preprocess/trec_ct/{topics,trials,qrels}.jsonl`.
- **Missing trials.** A qrel whose `nct_id` is absent from `trials.jsonl` is skipped, and the skip is counted per split in `fold_manifest.json`. For capped validation grade 0, the skip is applied **after** sampling (as in `data_splitter.build_splits`), so a capped topic may end with fewer than 60 grade-0 rows; those skips are counted too.
- **Row order.** Within each file, rows are generated in the order (topic key, grade 2 → 1 → 0, `nct_id` lexicographic). The rows are then shuffled with `random.Random(f"2021|fold{k}|{split}")`, where split ∈ {train, validation, validation_positive, test, negative_pools}. Order matters: ties under the stable mergesort break by file order, and train order feeds the DataLoader.
- **Files:**
  - `train.jsonl`: grade 2 only.
  - `validation.jsonl`: all grades. Grade 0 is capped at 60 per topic. `sorted_nct_ids` = the topic's grade-0 qrel `nct_id`s (before the missing-trial skip), sorted lexicographically. If `len(sorted_nct_ids) > 60`: `ncts = sorted(random.Random(f"2021|validation|{topic_id}|0").sample(sorted_nct_ids, 60))`; otherwise `ncts = sorted_nct_ids` (all). Missing trials are then skipped and counted. This happens before the row ordering above.
  - `validation_positive.jsonl`: the grade-2 rows of `validation.jsonl`, in that file's order.
  - `test.jsonl`: all grades, uncapped.
  - `negative_pools.jsonl`: one row per train and validation topic (`topic_id`, `split` ∈ {train, validation}, full `ineligible` / `not_relevant` lists, each sorted).
  - `fold_manifest.json`: topics per split, record counts, skipped-trial counts, and the sha256 of every file.
- **Asserts:**
  - topic-disjoint across train, validation and test;
  - every paraphrase group falls inside one fold;
  - no train/validation pool overlap.
- **Determinism test.** Two independent builds must give identical sha256 for every file.

Training commands per (k, seed) are in §6.3. Output: `results/trials_fusion_confirm/cohort2021/fold_{k}/trials_baseline_full_s{seed}/`.

## 4. Uncertainty and decision rule

### 4.1 Two-level bootstrap
- B = 2000.
- Analysis RNG: `np.random.default_rng([31415, 2022])` for cohort 2022, `np.random.default_rng([31415, 2021])` for cohort 2021.
- Within a cohort, the same resample indices are reused for every metric, φ variant, contrast and model type (common random numbers), so contrasts are paired.

**Cohort 2022 (trained).**
- D[c, p] = metric_fused − metric_text, for checkpoint c (19, or fewer after §5 exclusions) and patient p (50).
- Each replicate draws c* (n_c with replacement) and p* (50 with replacement), independently.
- Statistic: the mean over defined p ∈ p* of the mean over c* of D[c, p]. The point estimate is the same statistic on the original indices.
- Sensitivity: draw c* stratified by fraction (resample the seeds within each F).

**Cohort 2021 (trained).**
- D[s, p] for seed s ∈ {13, 21, 42} and patient p (75). D[s, p] uses the seed-s checkpoint of p's outer fold.
- Each replicate draws s* (3 with replacement) and p*, which draws n₂₀₂₁ patients with replacement from the patients retained after §5 fold exclusions (75 if none). Per-metric undefined patients are handled in the statistic, as in cohort 2022.
- Patients are resampled across folds, not stratified by fold.

**Zero-shot.** One function per fold, so the bootstrap is patient-level only, using the same p* draws.

**Intervals and p-values.**
- Intervals are percentile intervals.
- Two-sided bootstrap p = min(1, 2·min((1 + #{T* ≤ 0})/(B + 1), (1 + #{T* ≥ 0})/(B + 1))).

### 4.2 Pooled (meta) estimate
- Computed **separately for trained and for zero-shot** (never mixing the two).
- Patient-weighted: (n₂₀₂₂·Δ₂₀₂₂ + n₂₀₂₁·Δ₂₀₂₁)/(n₂₀₂₂ + n₂₀₂₁), where n is, **per metric**, the number of patients with that metric defined in that cohort's estimate for the same model type (e.g. AP needs a grade-2 trial, hard AUC needs grades 2 and 1). For trained, n is taken from the trained estimate (after §5 exclusions); for zero-shot, n is the number of patients with the metric defined in that cohort's zero-shot estimate (zero-shot keeps the patients of incomplete folds, §5).
- Its bootstrap distribution combines replicate b of each cohort (the two RNG streams are independent).
- Reported for every metric; descriptive only.
- Sensitivity: inverse-variance fixed-effect pooling using the bootstrap SDs.

### 4.3 Decision rule and success criterion
**Holm decision (per cohort, per model type).**
1. Order H1 and H2 by bootstrap p. **Ties in p** (e.g. both at the floor 2/(B + 1) ≈ 0.0010) are ordered H1 before H2 (likewise H1z before H2z).
2. Reject the first iff p₍₁₎ ≤ 0.025.
3. Only if the first is rejected: reject the second iff p₍₂₎ ≤ 0.05.

Hi is **positive** in a cohort iff it is Holm-rejected **and** its point estimate is > 0. A Holm-rejected negative effect is reported as "significant harm".

Companion intervals are reported, descriptive only:
- 97.5% for the first-ordered hypothesis;
- 95% for the second if step 3 is reached, otherwise 97.5% for both.

For a positive effect, Holm rejection matches the brief's "Holm-adjusted lower 95% bound > 0" up to the +1 correction. If the interval and the p-decision disagree, the p-decision governs and `report.md` prints the disagreement. The same rule applies to H1z/H2z.

**Criterion, for each Hi ∈ {H1, H2} (trained models):**
- raw_conf(Hi) = positive_2022(Hi);
- raw_rep(Hi) = raw_conf(Hi) **and** positive_2021(Hi) (Holm within 2021).

**Final label per Hi (deterministic, evaluated in order, first match wins).** C2_2022 and C2_2021 are the **trained** C2 states (§1) of each cohort; each applies to **both** H1 and H2.
```
label(Hi) =
  1. if not raw_conf(Hi):   observed 2022 outcome: "not confirmed", or "significant harm"
                            (Holm-rejected with point estimate < 0 in 2022)
  2. elif C2_2022 == fail:  "not attributable to patient-specific MeSH information (2022)"
  3. elif not raw_rep(Hi):  "confirmed, not replicated"  (append "; significant harm in 2021"
                            if Hi is Holm-rejected with point estimate < 0 in 2021)
  4. elif C2_2021 == fail:  "confirmed; 2021 replication not attributable to patient-specific MeSH information"
  5. else:                  "replicated"
then append " (shuffled control inconclusive: <cohort>)" for each cohort whose C2 state is inconclusive,
in the order 2022, 2021, provided that cohort's C2 state was reached by the rule (2022 for labels 2-5,
2021 for labels 4-5; for label 1 nothing is appended).
```

**Headline (printed first in `report.md`):**
- Only label 5 counts as "replicated".
- "MeSH fusion improves within-patient ranking, replicated": iff **both** H1 and H2 have label 5.
- "MeSH fusion improves within-patient ranking, partially replicated": iff exactly one of H1, H2 has label 5; the other's label follows.
- Otherwise the headline is the per-Hi labels joined: `"H1: <label(H1)>; H2: <label(H2)>"`.
- Any inconclusive-control suffix is part of the label and is printed with it.
- Example: H1 and H2 both positive in both cohorts, C2_2022 = fail → both get label 2, and the headline is `"H1: not attributable to patient-specific MeSH information (2022); H2: not attributable to patient-specific MeSH information (2022)"`, with no "replicated".

**Other reports:**
- S4 is reported per cohort as equivalent / not equivalent / inconclusive (§1).
- Zero-shot H1z/H2z are reported with their own Holm decision and raw positive/negative status per cohort; they get no final label and change no headline.
- No other outcome changes the headline.

### 4.4 Prior-exposure sensitivity (cohort 2021)
- **Preregistered sensitivity, `sensitivity.unexposed_closed`.** H1, H2 (trained) and H1z, H2z (zero-shot) in cohort 2021 are recomputed on the **52 closed-unexposed patients only** (the 75 minus `pins.cohort2021.exposed_topics_closed`, §0), intersected with the patients retained after §5 fold exclusions. Same B, same RNG seed `[31415, 2021]` (a fresh generator for this subset), same estimator. Patients are resampled from this subset; checkpoints/seeds as in §4.1. The Holm decision is also reported for this subset.
- It is reported in `analysis.json` and in `report.md` next to the primary 2021 estimate. It is **not part of the success criterion**.
- **Mandatory warning.** For each Hi (trained and zero-shot) that has final label 4 or 5 (§4.3; for zero-shot, positive in 2021) under the primary analysis but whose closed-unexposed (52-patient) point estimate is ≤ 0, `report.md` prints "replication driven by previously exposed patients or their paraphrases" for that Hi, directly under the headline. Only `unexposed_closed` triggers this warning.
- **Descriptive only (no test, no warning):** `sensitivity.unexposed60` (the 75 minus the 15 `exposed_topics`, same estimator), the Δ on the 15 directly exposed patients, and the Δ on the 8 paraphrase partners. Added by §9 A0: `sensitivity.unexposed_closed_minus_dev_smoke` (the 52 minus the 13 `dev_smoke_exposed_topics` = 39 patients).

### 4.5 Power
**Power** (qualitative, no new computation). The exploratory pooled-AUC CI half-width was ≈ 0.007 at 50 patients. With 75 patients, a 2021 effect of the exploratory size should be detectable. An effect half that size likely would not, and that outcome would be reported as "not replicated", not as evidence of absence.

## 5. Gates before the confirmatory analysis (code validation on already-seen data only)

1. **Unit tests** (`trials_domain/tests/test_fusion_confirm.py`):
   - fusion arithmetic and tie → smaller w;
   - the group-aware derangement: no fixed points, no topic receives URIs from its own paraphrase group (toy input with a 2-topic and a 3-topic group), deterministic across two calls, equals the plain cyclic shift when all groups are singletons, and aborts after 10,000 draws on an infeasible input (e.g. one group holding > n/2 topics);
   - the read-only loader asserts `embedding_dim == 768` against vector width;
   - the manifest digest is computed from basenames and is cwd-independent;
   - the pins chain verifier (§7 rule), on toy pins files:
     - accepts a revision that changes only whitelisted fields (`code`, `protocol.sha256`, `git`, `environment`, bookkeeping), and checks that the old `git`/`environment` are in `amendment.previous`;
     - accepts a revision whose changed non-whitelisted fields equal `amendment.changed_fields` exactly with `confirmatory_output_seen == false` (declared-change branch);
     - rejects an undeclared change, a declared field that did not change (inexact set), and any declared change with `confirmatory_output_seen == true`;
     - rejects a broken `supersedes` sha256 and a non-latest `--pins`;
   - `make_pins.py --revise` writes no config, and fails if any `configs/*.json` hash differs from the previous revision;
   - `exposed_closed` closure: on a toy component graph, the closure of the exposed set equals the union of the touched components;
   - within-patient z with SD 0 → 0;
   - the fold builder keeps groups together, is topic-disjoint, and gives identical hashes across two builds;
   - the paraphrase rule reproduces the expected edges and component on a toy matrix;
   - bootstrap shapes and determinism under a fixed seed;
   - Holm decision logic, including the negative-effect case and the tie case (equal p → H1 ordered first, so the 97.5% companion interval goes to H1);
   - the §4.3 final-label rule and headline: (a) H1 and H2 positive in both cohorts, C2_2022 = fail → label 2 for both and a headline without "replicated"; (b) both positive in both cohorts, both C2 pass → both label 5, headline "replicated"; (c) both positive in both cohorts, C2_2021 = fail → label 4 for both, no "replicated"; (d) H1 label 5, H2 not positive in 2021 → "partially replicated"; (e) C2_2022 inconclusive with label 1 → no suffix appended;
   - C2 and S4 three-state labelling, including the ordered-precedence case: C2 = +0.002 [+0.001, +0.004] with LB95(hier − C2) > 0 must give *fail*;
   - the `train.py` ledger/resume logic on a toy ledger and toy output dirs (§6.3): a run with a passing entry is skipped, an existing output dir without one is renamed to `.failed1` and retried, a second failure is recorded as excluded, and existing entries are byte-identical after the call;
   - the grade-0 cap: a topic with ≤ 60 grade-0 qrels keeps all of them without error, a topic with > 60 gets exactly 60 sorted ids, and missing trials are skipped after sampling;
   - the gate-2 leaf classifier and comparator (§5.2): `selected_weight` and count leaves are exact; a `validation_selection_curve` or `ranking.macro.ndcg@10` difference of 5e-5 passes and 2e-4 fails; a `validation_stats.text.mean` difference of 5e-7 passes and 2e-6 fails; a `selected_weight` mismatch fails regardless of other leaves;
   - gate 3's device check (§5.3) on toy `lc_manifest.json` device blocks: passes when `cuda_device_name` equals the pinned `gpu_name`, fails when it differs (e.g. pinned `"NVIDIA A100-SXM4-80GB"`, manifest `"NVIDIA L40S"`), and fails when `requested` or `resolved` is not `"cuda"`;
   - `verify_transfer.py`'s protocol parser (§6.2 step −1) on this protocol file: it finds exactly 15 §3b checkpoint rows and ≥ 21 §7 input rows (18 before §9 A1), and aborts on a truncated copy with fewer;
   - the `PYTHONHASHSEED` guard: each new script's entry check raises when `PYTHONHASHSEED != "0"`.
   - **Gate-1 timing and retry.** Gate 1 (the confirmatory run of these tests) runs on the remote, after the step −1 transfer and **before** `make_pins.py` (§6.2 step 0); passing runs on the Mac during development do not count as gate 1. Revision 1 is therefore written only after the tests pass and a gate-1 code fix needs no pins revision. If a gate-1 test is ever rerun and fails after `pins.json` exists, it follows the gate-2 retry rule below (§9 entry, `make_pins.py --revise`, new revision).
2. **Reproduction gate.** Score the 15 existing checkpoints with W = 0–1.0 (`[i/10 for i in range(11)]`) and nDCG selection, using the read-only embedding sources of §2.1. Compare against `results/publication_phase1_fixedval/trials_fusion_learning_curve_ndcg_selected.json` (sha256 in §7):
   - `selected_weight` must match **exactly**;
   - every numeric leaf of `validation_selection_curve`, `validation_stats`, and `test.{text,fusion,delta}` that the new scorer computes must match under a **tiered absolute tolerance** (preregistered now, before any run; pinned as `pins.analysis.gate2_tolerance_abs = {"continuous": 1e-6, "rank": 1e-4}` and `pins.analysis.gate2_leaf_classes`):
     - **exact** (integer counts): `selected_weight` (above), `n_queries`, `queries_with_both_classes.*`, `tier_stats.*.n`;
     - **rank-based, abs ≤ 1e-4**: every value of `validation_selection_curve`; every leaf under `ranking`, `ranking_macro` and `per_query` (`average_precision`, `ndcg@10`, `precision@10`, `recall@10`, `reciprocal_rank`); every leaf under `global_auc` and `macro_query_auc`; `kendall_tau_b`, `spearman_rho`;
     - **continuous, abs ≤ 1e-6**: `validation_stats.{text,ontology}.{mean,sd}` and every other computed leaf (`tier_stats.*.{mean,sd,median}`, `cohens_d.*`).
     - The leaf class is decided by the leaf's key path with the rules above, in that order; the classifier is unit-tested (§5.1).
     - Reason: rank metrics are discontinuous in the score, so a near-tie flip moves them in steps larger than the score perturbation. Measured on the 15 published checkpoints (round-6 review), relative score noise of 3e-7 moved macro nDCG@10/AP by up to 2.4e-6, and noise of 1e-6 by up to 8.8e-6. 1e-4 is above this measured near-tie rounding effect (≤ 9e-6 at 1e-6 relative score noise) and far below any confirmatory effect size (≥ 5e-3). Continuous leaves are O(1) means and SDs, where float32 BLAS differences (Mac Accelerate vs remote MKL/OpenBLAS, ≈ 1e-7) stay below 1e-6. `selected_weight` exact is safe: the smallest top-two gap in the published validation curves is 1.76e-4.
     - The tolerance changes no confirmatory quantity, because every confirmatory number is computed fresh on the one remote host. The largest absolute difference per leaf class and per leaf group is written to the gate output. Leaves the scorer does not compute (threshold metrics: `binary_classification`, `ordinal_classification`) are listed as skipped.
   - Gate 2 runs on the remote, on CPU (`CUDA_VISIBLE_DEVICES=""`, `--device cpu`), after step −1 and step 0 (§6.2).
   - Zero-shot test macro pooled AUC must equal 0.795 to 3 dp.
   - This gate touches only the already-analysed population. On failure, the code may be fixed and the gate rerun; each attempt is logged in §9 and produces a new pins revision (`pins_r{n}.json`, §7) because the code hashes change. Each rerun writes to a new leaf `gate_reproduction_r{n}.json` (no overwrite). The tolerances above are fixed. A failure that persists after a code fix stops the study and requires a §9 amendment: a declared change of `analysis.gate2_tolerance_abs` (or `analysis.gate2_leaf_classes`) through the declared-change branch of §7, with `confirmatory_output_seen == false`, which always holds at gate 2 because no new run has been scored.
   - **Code freeze.** Once gate 2 passes, `pins.code` is frozen (§7).
3. **Run checks.** For the 19 cohort-2022 runs (§9 A1) and the 15 CV runs, all of:
   - the runner's exit code is 0 (the plain path of `run_learning_curve_point.main` used here raises on failure; `run_error.json` is written only on the withholding-study path, so its absence alone proves nothing);
   - `lc_manifest.json` and `best_checkpoint.pt` both exist in the output dir;
   - no `run_error.json`;
   - `lc_manifest.json` `single_factor_audit.unexpected_fields` is exactly the set stated in §3b;
   - `attach.cache_check` passes;
   - `lc_manifest.json` → `device` (written by `run_learning_curve_point.device_record`, which already raises if the trainer, loss engine, encoder or projection is not on the resolved device) has `requested == "cuda"`, `resolved == "cuda"`, and `cuda_device_name == pins.environment.gpu_name` (exact string equality; `gpu_name` is recorded by `make_pins.py` from `torch.cuda.get_device_name(0)` at pin time on the same host, e.g. `"NVIDIA A100-SXM4-80GB"`);
   - the sha256 of `embedding_cache/trials_text_embeddings.pt` is unchanged.

**Failure handling:**
- **Rerun once.** A failed training run's output dir is renamed to `<dir>.failed1` (logged in the ledger). It is then rerun once with the identical command.
- **Exclusion.** A run that fails twice is excluded and reported.
- **Cohort 2022** proceeds iff ≥ 17 of 19 checkpoints remain. An excluded checkpoint is removed from the c dimension.
- **Cohort 2021** proceeds iff ≥ 4 outer folds have all 3 seeds. The patients of an incomplete fold are dropped from every trained-model estimate and kept in zero-shot; the drop is reported.
- Otherwise stop and amend: log a §9 entry, and produce a new pins revision through the declared-change branch of §7 (`amendment.changed_fields` lists e.g. `analysis.min_checkpoints_2022`, `cohort2022.new_runs` or `cohort2021.folds`; `confirmatory_output_seen` must be false, which holds because scoring of new runs has not started). If confirmatory output has been seen, no such revision is possible: the change is reported as a post hoc deviation in `report.md`.
- **Non-determinism.** Training is non-deterministic at a fixed seed (findings §B4: rerun score r ≈ 0.7). Each checkpoint is scored once, as produced. Its sha256 is written to the ledger before any scoring.

## 6. Files, commands, outputs

**Host and interpreter.** Every confirmatory command runs on the one remote GPU host (header), from the root of the remote checkout (`$R`), with the remote interpreter `.venv/bin/python` (a CUDA build of torch; versions are recorded in `pins.environment`, not fixed here; the local Mac venv has torch 2.8.0, sentence-transformers 5.1.0, numpy 2.3.2, scipy 1.16.1 for reference). Paths below are repo-relative; no machine path enters configs or pins.

**Run conditions:**
- every command runs with `PYTHONHASHSEED=0` (each code block below starts with `export PYTHONHASHSEED=0`). Every new script asserts `os.environ.get("PYTHONHASHSEED") == "0"` at start-up and records the value in its output JSON; `scripts/run_learning_curve_point.py` (not editable) is covered by `train.py`, which sets it in the child environment and records it in the ledger;
- every script that may touch the Hugging Face hub runs with `HF_HUB_OFFLINE=1` (§2.1);
- **training** (§6.3): `--device cuda`, no `CUDA_VISIBLE_DEVICES` override;
- **everything else** (pytest, `build_cv2021.py`, `make_pins.py`, `score.py` including gate 2, `analyze.py`): `CUDA_VISIBLE_DEVICES=""`, `--device cpu` wherever the flag exists, and every `SentenceTransformer` constructed with `device="cpu"`.

### 6.1 New files only
- `trials_domain/fusion_confirm.py`: read-only embedding loader, φ variants, derangement, z / fusion / w selection, per-patient metrics, two-level bootstrap, Holm, TOST labels, C2 states, paraphrase grouping, fold builder.
  - It reuses `epf.rank_auc`, `epf.ranking_metrics`, `epf.z_stats`, `epf.z_apply` and `epf.text_scores`. `scripts/` is not a package, so these are imported by path: `importlib.util.spec_from_file_location("evaluate_publication_fusion", ROOT/"scripts/evaluate_publication_fusion.py")` followed by `sys.path.insert(0, str(ROOT/"scripts"))` for the module's own sibling imports.
  - Port source: `/tmp/trials_helps/{lib,score,a2_fusion,a6_twolevel,a7_controls,a7b,a5_hard}.py` (present at writing). If it is gone, this protocol is the spec.
- `scripts/trials_fusion_confirm/{verify_transfer,make_pins,build_cv2021,train,score,analyze}.py`.
- `.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_TRIALS_FUSION.md`: the runbook (transfer list, remote setup, commands of §6.2–6.4 with the `$R` prefix). Operational only; it cannot change this protocol.
- `trials_domain/tests/test_fusion_confirm.py`.
- Data: `preprocess/trials_fusion_confirm/cv2021/fold_{0..4}/`. Results: `results/trials_fusion_confirm/`.

**No-overwrite rule.**
- No existing file is edited, and no existing result is overwritten.
- Every script refuses to run if the leaf path named by its `--out` / `--output-dir` already exists. Parent directories may exist. There is no `--force`.
- The only sanctioned exceptions are the `.failed1`/`.failed2` renames in §5 and the append-only `run_ledger.json` (§6.3, ledger and resume semantics).

### 6.2 Order and commands
**Step −1 (transfer, before anything else on the remote).** None of the inputs below is git-tracked, so a branch checkout on the remote has none of them. Copy, to the **same repo-relative paths** on the remote (method: `rsync -a --relative` from the local repo root, exact lines in `REMOTE_RUN_TRIALS_FUSION.md`; the method actually used is logged in that runbook's run log):
- the code checkout (working tree including the new, uncommitted study files; never `/300` or `reports/`);
- every file in the §7 inputs table (embedding caches, `preprocess/trec_ct/**`, `preprocess/trec_ct_splits/**`, `preprocess/trec_ct_lc/frac_{10,25,50,75,100}/train.jsonl` (§9 A1) and `validation_positive.jsonl`, `desc2021.gz`, the published fusion JSON);
- the 15 existing checkpoint dirs `results/publication_phase1_fixedval/checkpoints/trials_baseline_f{F}_s{S}/` (at least `best_checkpoint.pt` and `training_config.json`);
- the published prepared-cache dirs `test_4030cc85ff1b4ef3_…` (36 chunks) and `validation_5defb59b858bdeee_…` (0 chunks);
- the Hugging Face snapshot `models--sentence-transformers--all-mpnet-base-v2` (revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`) into the remote HF cache.

Then, on the remote, `scripts/trials_fusion_confirm/verify_transfer.py` (new file, CPU) runs. No pins file exists yet, so **its expected hashes come from this protocol file**, passed as `--protocol`; there are no embedded constants:
- It parses the §3b checkpoint table (rows `| F | S | records | <64-hex> |`, mapped to `results/publication_phase1_fixedval/checkpoints/trials_baseline_f{F}_s{S}/best_checkpoint.pt`) and the §7 inputs table (rows `| <path> | <64-hex> |`, where path = the first whitespace-delimited token of the first cell, so annotations such as `(= frac_100)` are ignored). It aborts if the parse finds fewer than 15 checkpoint rows or fewer than 21 input rows (18 before §9 A1).
- It checks the sha256 of every parsed file.
- It verifies the 36 test chunks through the §2.1 manifest digest `3d015e60c7d8484a12289299023fa8101a6d12c594af4812995e15e411c03f11`, which covers every chunk's sha256 and basename (per-chunk hashes are therefore not listed in this protocol; `make_pins.py` records them later). It also asserts the validation chunk directory holds 0 chunks.
- It confirms the MPNet snapshot revision by the §2.1 rule.
- It records the sha256 of the protocol file it parsed in `transfer_check.json`; `make_pins.py` requires that value to equal the sha256 of the `--protocol` it pins.

Any mismatch or missing file aborts; nothing else runs until it passes. It writes `results/trials_fusion_confirm/transfer_check.json` (leaf must not exist).

```
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1

# (-1) on the remote, after the transfer
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/trials_fusion_confirm/verify_transfer.py \
    --protocol .agents/tasks/ontology-withholding-feasibility/protocol_trials_fusion.md \
    --code-manifest .agents/tasks/ontology-withholding-feasibility/code_hashes_trials_fusion.mac.json \
    --out results/trials_fusion_confirm/transfer_check.json      # --code-manifest: §9 A0

# (0) protocol approved -> unit tests (gate 1), build folds (label-free), copy cache, write pins (all remote, CPU)
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest trials_domain/tests/test_fusion_confirm.py -q   # must pass before anything below
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/trials_fusion_confirm/build_cv2021.py \
    --converted-dir preprocess/trec_ct --out preprocess/trials_fusion_confirm/cv2021 \
    --fold-seed 2021 --n-folds 5 --cosine-threshold 0.80 --max-component 6 --device cpu
mkdir -p results/trials_fusion_confirm/cache
cp -n embedding_cache/trials_text_embeddings.pt results/trials_fusion_confirm/cache/trials_text_embeddings.pt
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/trials_fusion_confirm/make_pins.py \
    --protocol .agents/tasks/ontology-withholding-feasibility/protocol_trials_fusion.md \
    --out results/trials_fusion_confirm/pins.json      # revision 1; also writes results/trials_fusion_confirm/configs/*.json
# make_pins.py aborts unless sha256(cache copy) == sha256(original) == fa83e10a...1d4e, and requires
# transfer_check.json to exist and pass; it records the copy as inputs[".../trials_text_embeddings.pt@pin_time"]

# (1) reproduction gate (already-seen data; remote CPU; selected_weight exact, continuous leaves 1e-6, rank leaves 1e-4)
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/trials_fusion_confirm/score.py --gate-reproduce \
    --pins results/trials_fusion_confirm/pins.json --device cpu \
    --out results/trials_fusion_confirm/gate_reproduction.json

# (1b) only after a gate failure + code fix, or an amendment (§7, §9): new pins revision n = 2, 3, ...
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/trials_fusion_confirm/make_pins.py --revise \
    --previous <latest pins path> --amendment "<§9 entry id>" \
    --protocol .agents/tasks/ontology-withholding-feasibility/protocol_trials_fusion.md \
    --changed-fields "<comma-separated dotted paths, or empty>" --confirmatory-output-seen {false,true} \
    --out results/trials_fusion_confirm/pins_r{n}.json
# --revise never writes configs: it re-hashes the existing results/trials_fusion_confirm/configs/*.json
# and fails unless every hash equals the previous revision's. It always recomputes git and environment.
# then rerun (1) with --pins results/trials_fusion_confirm/pins_r{n}.json --out results/trials_fusion_confirm/gate_reproduction_r{n}.json
```
In every later command, `<PINS>` = the latest pins revision (`pins.json` if none was ever revised, else the highest-numbered `pins_r{n}.json`). Every script verifies the full `supersedes` chain back to `pins.json` and refuses a `--pins` that is not the latest revision present in `results/trials_fusion_confirm/`.

### 6.3 Training (34 runs, §9 A1)
```
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1

# cohort 2022 (§9 A1): (F, FRAC) in (10, 0.1), (25, 0.25), (50, 0.5), (75, 0.75), (100, 1.0);
#   S in 42, 13, 21, plus 87, 123 at F 25 and 100 (19 runs, in this order)
PYTHONHASHSEED=0 .venv/bin/python scripts/run_learning_curve_point.py --domain trials \
  --config results/trials_fusion_confirm/configs/lc_trials_baseline_cache.json \
  --compare-config config/lc_trials_baseline.json \
  --train-file preprocess/trec_ct_lc/frac_${F}/train.jsonl \
  --validation-file preprocess/trec_ct_lc/validation_positive.jsonl \
  --fraction ${FRAC} --seed ${S} --device cuda \
  --output-dir results/trials_fusion_confirm/cohort2022/checkpoints/trials_baseline_f${F}_s${S}

# cohort 2021: k in 0..4, S in {13,21,42}
PYTHONHASHSEED=0 .venv/bin/python scripts/run_learning_curve_point.py --domain trials \
  --config results/trials_fusion_confirm/configs/lc_trials_baseline_cv2021_fold${k}.json \
  --compare-config config/lc_trials_baseline.json \
  --train-file preprocess/trials_fusion_confirm/cv2021/fold_${k}/train.jsonl \
  --validation-file preprocess/trials_fusion_confirm/cv2021/fold_${k}/validation_positive.jsonl \
  --fraction 1.0 --seed ${S} --device cuda \
  --output-dir results/trials_fusion_confirm/cohort2021/fold_${k}/trials_baseline_full_s${S}
```
`scripts/trials_fusion_confirm/train.py --pins <PINS> --cohort {2022,2021}` (run `--cohort 2022` first, then `--cohort 2021`):
- issues exactly these commands in the fixed order of §3b (2022: F 25 then 100, each S 87 then 123; 2021: k 0..4, each S 13, 21, 42);
- verifies gate 3 for each run (exit code 0, `lc_manifest.json` and `best_checkpoint.pt` present, no `run_error.json`, `single_factor_audit.unexpected_fields`, `cache_check`, `device` block = cuda with `cuda_device_name == pins.environment.gpu_name`, original cache hash);
- appends `{run, order_index, command, exit_code, best_checkpoint sha256, lc_manifest sha256, single_factor_audit.unexpected_fields, wall time, failed_attempts, cache_copy: {entries_before, sha256_before, entries_after, sha256_after}}` to `results/trials_fusion_confirm/run_ledger.json`. Entry count = number of keys in the loaded `results/trials_fusion_confirm/cache/trials_text_embeddings.pt`.

**Ledger and resume semantics.**
- `run_ledger.json` is **append-only**. `--cohort 2022` creates it; `--cohort 2021` (and any resumed invocation) appends to it. This is the one sanctioned exception to the leaf-exists refusal of §6.1 for `train.py`; it never modifies or deletes an existing entry. Each append is written atomically (write to a temp file in the same directory, then `os.replace`), and the previous entries must be byte-identical in the new file.
- Every training attempt, passing or failing, gets its own entry with `status` ∈ {`passed`, `failed`, `excluded`}.
- On (re)start, `train.py` walks the fixed order. For each run:
  - a run with a `passed` entry is **skipped** (not retrained);
  - a run with an `excluded` entry is skipped;
  - an existing output dir with no `passed` entry (e.g. an attempt interrupted by a dropped remote session or a crash; `train.py` is run under `tmux`/`nohup` on the remote) is treated as a **failed attempt**: it is renamed to `<dir>.failed1` (or `.failed2` for a second failure, which then triggers exclusion, §5), a `failed` entry is appended, and the run is retried once with the identical command;
  - otherwise the run is trained normally.
- The cache-copy order dependence (§3b) is preserved because resume follows the same fixed order and skips only completed runs.

### 6.4 Scoring and analysis (once)
```
# remote, CPU only
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/trials_fusion_confirm/score.py --pins <PINS> \
    --ledger results/trials_fusion_confirm/run_ledger.json --device cpu \
    --out results/trials_fusion_confirm/scores
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/trials_fusion_confirm/analyze.py --pins <PINS> \
    --scores results/trials_fusion_confirm/scores \
    --out results/trials_fusion_confirm/analysis.json \
    --report results/trials_fusion_confirm/report.md
```

**Outputs:**
- `scores/{cohort}/{fold or all}/{model}.npz`: per pair, the text score, φ_hier, φ_exact and φ_shuf, plus grades, patient ids and nct ids, for validation and test. Alongside it, `selected_w.json` (per model × φ variant × normaliser).
- `analysis.json`: every estimate, CI, p, Holm decision, C2 / S4 state and verdict, plus the sensitivities (within-patient z, `condition_signal_present` exclusion, fraction-stratified c*, inverse-variance pooling, and for cohort 2021 the preregistered `unexposed_closed` plus the descriptive `unexposed60`, exposed-15 and partner-8 subsets, §4.4). It records the pins revision path and sha256 it ran under.
- `report.md`: tables in the order H1, H2, H1z, H2z, S1–S4, C1–C3, sensitivities, pooled. It also lists every p/CI disagreement, every exclusion, and any "replication driven by previously exposed patients or their paraphrases" warning (§4.4).

The published prepared cache is read-only. New encodes go only under `results/trials_fusion_confirm/prepared_embeddings/`.

## 7. Pins JSON (`results/trials_fusion_confirm/pins.json`, revisions `pins_r{n}.json`)

Revision 1 (`pins.json`) is written on the remote GPU host after approval, after the step −1 transfer check and gate 1 pass, and before training (§6.2 step 0). No pins file is ever written on the Mac. The scorer, trainer wrapper and analyzer refuse to run if any pinned hash in the `--pins` revision mismatches.

**Versioning.** No pins file is ever rewritten.
- A code fix after a gate failure, or a §9 amendment, produces a new file `results/trials_fusion_confirm/pins_r{n}.json` (n = 2, 3, …), written by `make_pins.py --revise`.
- Its content is the previous revision plus `"revision": n`, `"supersedes": {"path": <previous path>, "sha256": <previous file sha256>}`, and `"amendment": {"id": <§9 entry id>, "date": ..., "reason": ..., "confirmatory_output_seen": bool, "changed_fields": [<dotted paths>], "previous": {"git": {...}, "environment": {...}}}`, with fresh `created_utc`.
- **Allowed changes.** A revision may change `code`, `protocol.sha256`, `git` and `environment`. `git` and `environment` are always recomputed by `make_pins.py --revise`, and their previous values are copied into `amendment.previous`. It may also change the bookkeeping fields `amendment`, `supersedes`, `revision` and `created_utc`.
- **Declared changes.** Any other field except `configs` (inputs, checkpoints, new runs, folds, analysis parameters, exposed topics) may change only if it is listed in `amendment.changed_fields` **and** `amendment.confirmatory_output_seen == false`. The verifier compares the two revisions leaf by leaf and requires the set of changed non-whitelisted fields (as dotted paths, at the granularity listed) to equal `amendment.changed_fields` exactly. An undeclared change, or a declared field that did not change, is a hard error in `make_pins.py --revise` and in every verifier.
- **After confirmatory output.** If confirmatory output has been seen (new-run checkpoints scored, `scores/`, `analysis.json`), no non-whitelisted field may change. The study then stops, and the change is reported as a post hoc deviation in `report.md`, not as a revision.
- `configs/*.json` are written once, in step 0. `--revise` never writes configs; it re-hashes them and requires equality with the previous revision (a config change is therefore impossible by revision and requires the post hoc route or a new study).
- Scripts take `--pins` pointing at the latest revision, verify the whole `supersedes` chain back to `pins.json` (each link's sha256 and the rules above), and refuse a non-latest revision.
- **`pins.code` is frozen once gate 2 passes.** After that, any code change requires a §9 amendment stating whether any confirmatory output (new-run checkpoints scored, `scores/`, `analysis.json`) had been produced, and a new revision.
- Editing this protocol for an amendment changes its sha256, which is exactly what a new revision records; it does not by itself invalidate earlier outputs.

```json
{
  "schema": "trials_fusion_confirm/pins/v7",
  "revision": 1,
  "supersedes": null,
  "amendment": null,
  "created_utc": "...",
  "protocol": {"path": ".agents/tasks/ontology-withholding-feasibility/protocol_trials_fusion.md", "sha256": "..."},
  "git": {"head": "<HEAD at pin time>", "branch": "orca-experiments-and-runner", "dirty_paths": ["..."]},
  "environment": {"host": "<remote GPU host>", "hostname": "...", "platform": "Linux-...-x86_64",
                   "gpu_name": "<torch.cuda.get_device_name(0)>", "gpu_count": 1, "nvidia_driver": "...", "cuda_runtime": "...",
                   "torch": "...", "torch_cuda_build": "...", "cudnn": "...", "python": "...",
                   "sentence_transformers": "...", "numpy": "...", "scipy": "...",
                   "blas": "<numpy.show_config() BLAS/LAPACK name+version>",
                   "mpnet_snapshot": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
                   "device_policy": {"training": "cuda", "scoring": "cpu", "fresh_encode": "cpu", "analysis": "cpu"},
                   "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1"},
  "transfer_check": {"path": "results/trials_fusion_confirm/transfer_check.json", "sha256": "...", "passed": true,
                     "parsed_protocol_sha256": "<must equal protocol.sha256 in revision 1>"},
  "inputs": {"<relpath>": "sha256",
             "results/trials_fusion_confirm/cache/trials_text_embeddings.pt@pin_time": "fa83e10a95a37a378c3d54e110678a64f702a850aef2442b06148d2589e31d4e"},
  "published_prepared_cache": {
      "test_dir": "results/publication_phase1_fixedval/prepared_embeddings/trials/test_4030cc85ff1b4ef3_sentence-transformers_all-mpnet-base-v2",
      "validation_dir": "results/publication_phase1_fixedval/prepared_embeddings/trials/validation_5defb59b858bdeee_sentence-transformers_all-mpnet-base-v2",
      "chunks": {"<chunk filename>": "sha256"}, "n_chunks": {"test": 36, "validation": 0},
      "manifest_digest": "3d015e60c7d8484a12289299023fa8101a6d12c594af4812995e15e411c03f11"},
  "code": {"<relpath of each new and reused script/module>": "sha256"},
  "configs": {"base": {"path": "config/lc_trials_baseline.json", "sha256": "..."},
              "derived": {"<path>": {"sha256": "...", "diff_vs_base": {"field": ["old", "new"]},
                                     "expected_unexpected_fields": ["embedding_cache_path"]}}},
  "cohort2022": {"existing_checkpoints": [{"fraction": 10, "seed": 42, "path": "...", "sha256": "...", "training_device": null}],
                 "new_runs": [{"fraction": 10, "seed": 42, "output_dir": "...", "device": "cuda"}, "... 19 entries in training order (§9 A1)"],
                 "validation": "preprocess/trec_ct_splits/validation.jsonl",
                 "test": "preprocess/trec_ct_splits/test.jsonl"},
  "cohort2021": {"fold_seed": 2021, "n_folds": 5, "cosine_threshold": 0.80, "max_component": 6,
                 "exposed_topics": ["2021_6", "2021_16", "2021_17", "2021_23", "2021_26", "2021_37", "2021_42", "2021_45",
                                    "2021_49", "2021_51", "2021_61", "2021_63", "2021_64", "2021_69", "2021_74"],
                 "exposed_topics_source": "preprocess/trec_ct_splits/validation.jsonl (metadata.resume_id)",
                 "exposed_topics_closed": ["2021_6", "2021_15", "2021_16", "2021_17", "2021_21", "2021_23", "2021_25", "2021_26",
                                           "2021_27", "2021_33", "2021_37", "2021_40", "2021_42", "2021_45", "2021_49", "2021_51",
                                           "2021_57", "2021_59", "2021_61", "2021_63", "2021_64", "2021_69", "2021_74"],
                 "exposed_topics_closed_rule": "union of paraphrase components containing an exposed_topics member",
                 "cosine_source": "frozen cache, _ontology_slot(topic,'topic_id',id), L2 float64",
                 "freshly_encoded_topics": [],
                 "paraphrase_edges_fixed": [["2021_40", "2021_74"]],
                 "paraphrase_edges_cosine": [["2021_26", "2021_63", 0.895]],
                 "expected_added_edges": [["2021_26","2021_63"],["2021_26","2021_57"],["2021_15","2021_21"]],
                 "pairs_cosine_0_70_to_0_80": [["2021_45", "2021_53", 0.778]],
                 "components": [["2021_15", "2021_21", "2021_26", "2021_57", "2021_63"]],
                 "folds": {"0": ["2021_..."]},
                 "outer": [{"k": 0, "test": 0, "validation": 1, "train": [2, 3, 4]}],
                 "fold_files": {"fold_0/train.jsonl": "sha256"},
                 "skipped_missing_trials": {"fold_0": {"train": 0}},
                 "seeds": [13, 21, 42]},
  "analysis": {"weight_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "gate_weight_grid": "[i/10 for i in range(11)]",
               "select_on": "ndcg@10", "tie_break": "smaller_w",
               "normalisation": "validation_file_global_z", "sensitivity_normalisation": "within_patient_z_sd0_to_0",
               "phi_variants": ["hierarchy", "exact", "shuffled"], "per_variant_w_selection": true,
               "c3_fixed_w": 0.2, "B": 2000, "rng": {"2022": [31415, 2022], "2021": [31415, 2021]},
               "shuffle_seed": 11, "shuffle": "group_aware_cyclic_shift_rejection_per_file", "shuffle_max_draws": 10000,
               "embedding_dim": 768, "manifest_digest_rule": "sha256 of sorted '<sha256>  <basename>\\n' lines",
               "training_order": ["2022:f10_s42", "2022:f10_s13", "2022:f10_s21", "2022:f25_s42", "...",
                                  "2022:f25_s87", "2022:f25_s123", "...", "2022:f100_s123",
                                  "2021:k0_s13", "2021:k0_s21", "2021:k0_s42", "...", "2021:k4_s42"],
               "sensitivity_unexposed_closed": true, "descriptive_unexposed60": true,
               "equivalence_bound": 0.01, "equivalence_ci": 0.90, "alpha": 0.05,
               "holm_family": ["H1", "H2"], "decision": "holm_p_and_positive_point",
               "c2_pass_ub": 0.005, "min_checkpoints_2022": 17, "min_complete_folds_2021": 4,
               "holm_tie_order": ["H1", "H2"], "final_label_rule": "ordered_5_label_v1",
               "gate2_tolerance_abs": {"continuous": 1e-6, "rank": 1e-4}, "gate2_selected_weight": "exact",
               "gate2_leaf_classes": {"exact": ["selected_weight", "n_queries", "queries_with_both_classes.*", "tier_stats.*.n"],
                                      "rank": ["validation_selection_curve.*", "ranking.*", "ranking_macro.*", "per_query.*",
                                               "global_auc.*", "macro_query_auc.*", "kendall_tau_b", "spearman_rho"],
                                      "continuous": ["validation_stats.*", "<every other computed leaf>"],
                                      "rule": "first matching class in the order exact, rank, continuous"},
               "gate3_gpu_name": "pins.environment.gpu_name (exact match)"}
}
```

`inputs` must include at least the following files (current sha256), plus every fold file and every published chunk:

| file | sha256 |
|---|---|
| preprocess/trec_ct_splits/test.jsonl | 4030cc85ff1b4ef389896fa2ce7f3bed827b9640028446dfa96548f00ae24348 |
| preprocess/trec_ct_splits/validation.jsonl | 5defb59b858bdeeed2c04ad737e20def86bc53196dd27ac3828a41d4375cdfe1 |
| preprocess/trec_ct_splits/train.jsonl (= frac_100) | e4620c65c0a2ac8229ae35dd47440accfd7442e25a5d931182012cc519a7bfa8 |
| preprocess/trec_ct_splits/negative_pools.jsonl | f190ab7ca6a0c154c2381d44dfeb25c884565aa0957252c16eda2c39269c6af3 |
| preprocess/trec_ct_splits/split_manifest.json | 32160deaa0f149f902e4379644adb93028ace195c0f1f171a9e367460e9d2d66 |
| preprocess/trec_ct_lc/frac_100/train.jsonl (read by the F = 100 runs, §6.3) | e4620c65c0a2ac8229ae35dd47440accfd7442e25a5d931182012cc519a7bfa8 |
| preprocess/trec_ct_lc/frac_25/train.jsonl | 9fb56dbec881a09c17d714cfd1e03714a7384f40ede9aa41f17f26cdb28a893c |
| preprocess/trec_ct_lc/frac_10/train.jsonl (§9 A1) | e62cdc23f24d33ea4bcc1f0ae47d7c4fbcddcdb558b39fb54162ee971b956b9b |
| preprocess/trec_ct_lc/frac_50/train.jsonl (§9 A1) | 85be44a38742ad4bce65225c759a44673b043acda3877f5b401674de7891f685 |
| preprocess/trec_ct_lc/frac_75/train.jsonl (§9 A1) | 6361db769b4d01c6df6a9d79ed7aa8806ab5a0486303b17962d4e4e21aced115 |
| preprocess/trec_ct_lc/validation_positive.jsonl | 197b9c910dc1b443dd9114740b6ff1ddbc53c82e83f26ad4568aeb1ad80960e7 |
| preprocess/trec_ct/qrels.jsonl | ae4631741963c5fd14c6d62a86ee1c0aaa15bb61771fc2e16f6c2534501ac18a |
| preprocess/trec_ct/topics.jsonl | ab79d0ef3767fb7a1639a431ba90fa5dd2a04a02c918e5f7b42881f0adf70588 |
| preprocess/trec_ct/trials.jsonl | c393a4608cc65c756475d3b856d7fbdb249d04abefb4c327bfaac23fd4ebde72 |
| embedding_cache/trials_text_embeddings.pt | fa83e10a95a37a378c3d54e110678a64f702a850aef2442b06148d2589e31d4e |
| embedding_cache/mesh2021_index.pkl | 780cfce2b1d0ed97bd3086b0aa96b5cf9e0a4d16d6c54a66df3543faa4467610 |
| trec-clinical-trials/raw/ontology/desc2021.gz | 97de49d3263b5c9a03ab0f08ee4c2440603263d90d0b43b9fc7f92bfde565212 |
| config/lc_trials_baseline.json | 708aa7da9669166008f716089373a6af264aaa801aa18cd19869ce954894f333 |
| scripts/evaluate_publication_fusion.py | 606793ef413229dc9fcec9e6facb0f89b363476e402c98b8719b1bd760616947 |
| scripts/run_learning_curve_point.py | 67f1d4887e04b6c478ddcaf9cf81667493b7a341e45ebeaed5dd0da932e22121 |
| results/publication_phase1_fixedval/trials_fusion_learning_curve_ndcg_selected.json | da95c726f7c097577907d08b0cbf7418349014654b0b7ac9c48121b63ec49f86 |

The 2022 qrels are embedded in `test.jsonl` (a grade on each record). The raw TREC qrels for both years are in `preprocess/trec_ct/qrels.jsonl`.

## 8. Limitations, stated in advance

- **Pooling bias.** TREC judged pools come from participant systems. If those systems used condition or MeSH matching, φ's relevance AUC may partly reflect how the pools were built, not patient relevance. Both cohorts share this, so replication does not remove it.
- **External validity.** All 125 patients are synthetic TREC vignettes. There is no check on a real cohort.
- **Cohort 2022 is not independent of the exploration** (§0). Only cohort 2021 tests the claim with new models. 52 of its 75 patients have neither been scored under fusion nor paraphrase an exposed patient; 60 have not themselves been scored. *(§9 A0: after the Mac dev smoke these counts are 39 and 47; the preregistered sensitivity still uses the 52.)* The other 15 were the exploratory fusion-validation set (§0, sensitivity in §4.4). It is also in-domain (same campaign, same annotators), not a new collection.
- **Weak trained base.** Trained text is 0.07–0.13 nDCG@10 below zero-shot, so a gain over trained text does not imply a gain over the best text model. H1z/H2z address this but are not part of the replication criterion.
- **Training non-determinism and device mix.**
  - Checkpoints are not reproducible bit for bit at a fixed seed (§5).
  - *(Superseded by §9 A1.)* All 34 confirmatory checkpoints (19 cohort-2022, 15 CV) are trained on the one remote GPU host with CUDA, so device and host are no longer confounded with seed. The 15 published checkpoints (trained on the Mac) enter only gate 2.
  - All confirmatory scoring runs on one host and one device (remote x86 CPU), so the published numbers are reproduced only to the tiered tolerance of gate 2 (exact `selected_weight`, 1e-6 continuous, 1e-4 rank-based), not bit for bit.
  - The frozen-cache vectors (Mac-encoded) and the fresh remote-CPU encodes used for 2021 fold texts come from different devices; during training, the trainer may also append CUDA encodes to the cache copy. Δ is paired within each text, so this affects text and fused scores alike.
- **Noisy topic-side MeSH extraction.** The primary condition is captured in ≈ 80% of 2021 and ≈ 60% of 2022 topics (eyeball estimate), and 7 topics have no condition descriptor. Effects are conditional on this extractor.
- **Topical relevance only.** Fusion addresses topical relevance. The eligibility contrast (S4) is predicted null, and nothing here supports an eligibility claim.

## 9. Amendments
Any change after `pins.json` is written, and any reproduction-gate retry, is logged here as an entry `A{m}` with the date, the reason, the files changed, whether any confirmatory output had been produced, and the pins revision it creates (`pins_r{n}.json`, §7).

**A0 (2026-10-10, before revision 1; creates no pins revision). Disclosure of the Mac dev smoke, from the implementation code review (`review_trials_fusion.md`, findings 1, 2, 4, 6).** No confirmatory output had been produced: no pins file, no new-seed checkpoint, no confirmatory score.
- **2021 exposure.** The non-confirmatory dev smoke (`score.py --dev-local`, outer fold 0) restricted the 2021 test side to `exposed_topics_closed`, but scored the whole inner-validation file, which is fold 1 (a confirmatory test fold for k = 1). It wrote validation nDCG@10 curves over w ∈ {0..0.5} for hierarchy, exact and shuffled φ, for zero-shot (deterministic, so these equal the confirmatory values for those patients) and a 2-epoch smoke model. Fusion lowered zero-shot validation nDCG@10 there and w = 0 was selected. This exposed **13 of the 52 closed-unexposed patients**: 2021_7, 2021_9, 2021_12, 2021_14, 2021_29, 2021_31, 2021_32, 2021_35, 2021_43, 2021_46, 2021_47, 2021_54, 2021_67 (pinned as `pins.cohort2021.dev_smoke_exposed_topics`). Exposure is by topic id, so this list does not depend on the remote fold build. The §0/§4.4 statement that the 52 were "neither scored under fusion nor a paraphrase of a scored patient" is therefore false for these 13.
  - Resolution: the preregistered `sensitivity.unexposed_closed` (52 patients, §4.4) and its mandatory warning are **unchanged**. A **descriptive** subset `sensitivity.unexposed_closed_minus_dev_smoke` (the 52 minus these 13 = 39 patients, same estimator, fresh generator with seed `[31415, 2021]`, no Holm decision, no warning) is added and reported next to it (`pins.analysis.descriptive_unexposed_closed_minus_dev_smoke`).
  - The dev-local scorer now restricts **both** the inner-validation and the test file to `exposed_topics_closed` and aborts before scoring when a restricted file cannot be deranged. On the real folds that holds for every outer fold, so any further 2021 smoke needs a synthetic fold.
- **2022 C2 preview.** The dev analysis (`analyze.py --dev-local`, B = 200) computed C1–C3 and the C2 state with the group-aware derangement of §2.3 on the 15 published cohort-2022 checkpoints, which are 15 of the 19 C2_2022 inputs. The exploration had only seen a plain-permutation shuffle. Cohort 2022 is already a frozen-pipeline confirmation (§0); this is disclosed, not corrected, and `report.md` prints both A0 disclosures.
- **Code-tree transfer check (finding 4).** Before the step −1 rsync, `scripts/trials_fusion_confirm/dump_code_hashes.py` writes the Mac-side `code_hashes()` manifest. On the remote, `verify_transfer.py` additionally takes `--code-manifest <that file>` and fails on any missing, extra or differing code file. `make_pins.py` (revision 1) requires that passing code check in `transfer_check.json` and re-compares `pins.code` with the manifest.
- **Text pointers.** The A0 pointers in §0, §4.4 and §8 are the only other edits to §0–§8; no table, hypothesis, threshold or decision rule changed. This file is committed before the step −1 transfer, so `pins.git` ties revision 1 to that commit.
- **Smoke directory (finding 6).** The header names `results/trials_fusion_confirm_smoke/`; the implementation writes non-confirmatory outputs to `results/trials_fusion_confirm/dev_local/` (or `/tmp`), as the implementation brief specified. No confirmatory script reads either; the runbook excludes `dev_local/` from the transfer.

**A1 (2026-10-10, before revision 1; creates no pins revision). Cohort 2022 retrained on one device.** User decision, to avoid mixing training platforms within the cohort-2022 population. No confirmatory output had been produced: no pins file, no new checkpoint, no confirmatory score.
- **Population.** The cohort-2022 trained population is the same 19 (F, S) cells as before (F ∈ {10, 25, 50, 75, 100}; seeds 42, 13, 21 at every F, plus 87 and 123 at F 25 and 100), but **all 19 are retrained** on the pinned GPU host with `--device cuda` by the §6.3 command. The 15 published checkpoints (§3b table) no longer enter any confirmatory estimate. They are used only by gate 2, which is unchanged.
- **Why.** Before A1, the 15 published checkpoints were trained on the Mac and the 4 new seeds on CUDA, so device was confounded with seed (old §8). After A1, all 34 confirmatory checkpoints (19 + 15 CV) come from one host and one device.
- **Effect on §0.** Every cohort-2022 model is new. The 50 test patients and the cohort-2022 validation file are unchanged and already analysed, so cohort 2022 remains a confirmation, not an independent test. The A0 shuffled-control preview ran on the published checkpoints, so it no longer covers the trained C2_2022 input; it did cover the zero-shot 2022 controls, which are deterministic and descriptive only.
- **Training order (§3b, §6.3, `pins.analysis.training_order`).** 2022: F 10, 25, 50, 75, 100; within each F seeds 42, 13, 21, then 87, 123 at F 25 and 100. Then 2021 as before. Code constant: `fc.COHORT2022_RUNS`.
- **Inputs (§7).** `preprocess/trec_ct_lc/frac_{10,50,75}/train.jsonl` are added to the inputs table and the transfer list (21 input rows; the parser minimum rises from 18 to 21 so a copy missing any of them aborts).
- **Unchanged.** Hypotheses, metrics, controls, weight grid, normalisation, bootstrap (equal weight per checkpoint, c* over the 19), Holm, labels, the ≥ 17 of 19 stop rule, the fraction-stratified sensitivity, cohort 2021, gate 2 and its tolerances, gate 3 (`device == cuda`, GPU name equals `pins.environment.gpu_name`), and scoring and analysis on CPU.
- **Cost.** 15 more GPU training runs (34 in total).

## 10. Round-1 findings → resolutions (applied in v2)

| # | resolution |
|---|---|
| 1 | Cosine threshold 0.80 computed from frozen-cache vectors; expected added edges and the 5-topic component asserted; [0.70, 0.80) pairs pinned (§3). |
| 2 | Read-only published prepared cache for 2022 with pinned chunks and abort-on-missing; CPU encodes only for 2021; 1e-9 tolerance and exact-w gate kept (§2.1, §5.2). *Tolerance relaxed to 1e-6 for remote execution (§14 row 5), then tiered to 1e-6 continuous / 1e-4 rank-based (§15 row 1).* |
| 3 | φ ≡ 0 only for 2022_31; sensitivity excludes the 7 `condition_signal_present == false` topics (§2.6). |
| 4 | Decision = Holm on bootstrap p plus positive point estimate; CIs descriptive; disagreements printed (§4.3). |
| 5 | C2 has three per-cohort states with stated overrides (§1, §4.3). |
| 6 | Per-variant z-stats and w re-selection; C3(ii) at w = 0.2 with own z-stats; zero-shot follows the same rule (§2.5). |
| 7 | S4 equivalent / not equivalent / inconclusive defined (§1). |
| 8 | 2021 proceeds iff ≥ 4 complete folds, incomplete-fold patients dropped from trained estimates; 2022 proceeds iff ≥ 17/19 (§5). |
| 9 | Year filter, missing-trial skip/count, row order plus seeded shuffle, numeric sort key, two-build hash test (§3, §2.3). |
| 10 | HEAD updated; pins record HEAD at pin time (header, §7). |
| 11 | Gate 3 asserts the exact `unexpected_fields` sets (§3b, §5). |
| 12 | Device mix added to limitations (§8). |
| 13 | epf imported via importlib by path (§6.1). |
| 14 | Within-patient z with SD 0 → z = 0 (§2.4). |
| 15 | Exact matcher built from `MeshIndex.build_or_load` with pinned index (§2.3). |
| 16 | `mkdir -p` added; "exists" = leaf path; `.failed1` rename before the single rerun (§5, §6). |
| 17 | Unbalanced fraction mix accepted; fraction-stratified c* sensitivity added (§3b, §4.1). |
| 18 | Local CPU execution stated; regenerate `pins.environment` if run elsewhere (header). *Superseded by round-5 finding 5 (§14): one remote GPU host is the execution host; made GPU-agnostic but pinned in round 6 (§15 row 2).* |

## 11. Round-2 findings → resolutions (applied in v3)

| # | sev | resolution |
|---|---|---|
| 1 | MEDIUM | §0 now discloses that the 15 `trec_ct_splits/validation.jsonl` topics were exposed under fusion; list pinned as `pins.cohort2021.exposed_topics` (§7); preregistered 60-unexposed-patient sensitivity for H1/H2/H1z/H2z with a mandatory "replication driven by previously exposed patients" warning (§4.4, §6.4); §8 says 60 of 75 unexposed. |
| 2 | MEDIUM | Group-aware derangement: rejection-sample `rng(11).permutation` until no topic receives its own paraphrase group's URIs, abort after 10,000 draws, draws logged; identical to the old rule when all groups are singletons; unit test added (§2.3, §5.1). |
| 3 | MEDIUM | Versioned pins `pins_r{n}.json` with `supersedes` chain; only `code`, `protocol.sha256`, `amendment` (plus bookkeeping `revision`, `supersedes`, `created_utc`) may change; `pins.code` frozen once gate 2 passes; scripts take the latest `--pins <PINS>` and verify the chain; gate reruns write `gate_reproduction_r{n}.json` (§5.2, §6.2–6.4, §7, §9). |
| 4 | NIT | Gate 3 requires exit code 0, `lc_manifest.json` and `best_checkpoint.pt` present, plus the existing checks (§5.3, §6.3). |
| 5 | NIT | Manifest digest defined over sorted `"<sha256>  <basename>\n"` lines (cwd-independent), equal to the shell form run inside the chunk dir (§2.1). |
| 6 | NIT | Read-only loader sets `embedding_dim = 768`, asserted against vector width; no encoder constructed for 2022 files (§2.1). |
| 7 | NIT | Training order fixed; cache-copy entry count and sha256 logged before/after each run in `run_ledger.json` (§3b, §6.3, §7 `training_order`). |
| 8 | NIT | Meta-estimate n is per metric: patients with that metric defined (§4.2). |

(Round-2 row 1 is superseded by round-3 row 1 below: the 60-patient set is now descriptive only.)

## 12. Round-3 findings → resolutions (applied in v4)

| # | sev | resolution |
|---|---|---|
| 1 | MEDIUM | `exposed_closed` = union of §3 components containing an exposed topic (23 topics, asserted by `build_cv2021.py`, pinned as `pins.cohort2021.exposed_topics_closed`); preregistered `sensitivity.unexposed_closed` on the 52 closed-unexposed patients drives the mandatory warning; `unexposed60`, exposed-15 and partner-8 are descriptive; `pins.analysis.sensitivity_unexposed_closed`; §8 updated (§0, §3, §4.4, §6.4, §7, §8). |
| 2 | MEDIUM | §7 rule replaced: `git` and `environment` always recomputed by `--revise`, previous values in `amendment.previous`; other non-config fields change only if listed in `amendment.changed_fields` (exact match) with `confirmatory_output_seen == false`; otherwise stop and report a post hoc deviation. §5 "stop and amend" routed through this branch; unit tests for both branches (§5.1, §5 failure handling, §6.2, §7). |
| 3 | NIT | Gate 3, §3b and the ledger reference `lc_manifest.json` → `single_factor_audit.unexpected_fields` (§3b, §5.3, §6.3). |
| 4 | NIT | `--revise` never writes configs; it re-hashes `configs/*.json` and requires equality with the previous revision; unit-tested (§5.1, §6.2, §7). |
| 5 | NIT | 2021 p* draws n₂₀₂₁ patients with replacement from those retained after §5 fold exclusions; undefined metrics handled in the statistic (§4.1). |
| 6 | NIT | §10 and §11 retitled to match the header (§10, §11). |

## 13. Round-4 findings → resolutions (applied in v5)

| # | sev | resolution |
|---|---|---|
| 1 | MEDIUM | C2 states evaluated in order, first match wins: fail, then pass, then inconclusive; worked example; unit test C2 = +0.002 [+0.001, +0.004] with LB95(hier − C2) > 0 → *fail* (§1, §5.1). |
| 2 | NIT | C1–C3 and S4 computed for trained and zero-shot; only the trained C2 state applies §4.3 overrides; zero-shot controls descriptive (§1). |
| 3 | NIT | pytest (gate 1) now runs first in step 0, before `build_cv2021.py` and `make_pins.py`; a later gate-1 failure after pins exist follows the gate-2 retry rule (§5.1, §6.2). |
| 4 | NIT | `run_ledger.json` append-only (atomic replace, prior entries byte-identical); `train.py` skips `passed`/`excluded` runs, treats an output dir without a passing entry as a failed attempt (`.failed1`/`.failed2`), never edits entries; unit-tested (§5.1, §6.1, §6.3). |
| 5 | NIT | `preprocess/trec_ct_lc/frac_100/train.jsonl` added to the §7 inputs table with its sha256 (verified equal to `trec_ct_splits/train.jsonl`). |
| 6 | NIT | Grade-0 cap samples only if > 60 ids, result sorted; missing trials skipped after sampling as in `data_splitter.build_splits` and counted; unit-tested (§3, §5.1). |
| 7 | NIT | Pooled estimate computed separately for trained and zero-shot; zero-shot n = patients with the metric defined in that cohort's zero-shot estimate (§4.2). |

## 14. Round-5 findings → resolutions (applied in v6)

| # | sev | resolution |
|---|---|---|
| 1 | MEDIUM | Ordered per-Hi final label (1 not confirmed / harm → 2 C2_2022 fail → 3 not replicated → 4 C2_2021 fail → 5 replicated), trained C2 per cohort applies to both H1 and H2, inconclusive suffix rule, headline counts only label 5; §1 now points to §4.3; §4.4 warning keyed to labels 4–5; unit tests (a)–(e) incl. both positive in both cohorts with C2_2022 fail → label 2, no "replicated" (§1, §4.3, §4.4, §5.1). |
| 2 | NIT | Tied p ordered H1 before H2 (H1z before H2z); unit-tested; `pins.analysis.holm_tie_order` (§4.3, §5.1, §7). |
| 3 | NIT | `cp -n`; `make_pins.py` asserts sha256(copy) == sha256(original) == `fa83e10a…1d4e` and records `inputs[".../trials_text_embeddings.pt@pin_time"]` (§3b, §6.2, §7). |
| 4 | NIT | Reworded: the trainer writes the cache file at `config.embedding_cache_path` (`trainer.py` L931, L1643) (§3b). |
| 5 | HIGH | Remote L40S is the execution host; `pins.json` written there; Mac only for development/unit tests/non-confirmatory smoke (header). Step −1 transfer + `verify_transfer.py` sha256 check of all §7 inputs, the 15 checkpoints, 36 chunks, manifest digest and MPNet snapshot, abort on mismatch (§6.2). Device policy: training `--device cuda` (gate 3 checks `lc_manifest.device`), scoring/gate 2/fresh encodes/analysis on remote CPU (header, §2.1, §5.3, §6). Gate 2: `selected_weight` exact, numeric leaves abs ≤ 1e-6, run on the remote (§5.2). "Run elsewhere → amendment" clause removed. `pins.environment` records GPU, driver, CUDA, torch build, BLAS, MPNet snapshot (§7). §8 discloses Mac-vs-L40S checkpoint device mix and the 1e-6 reproduction. Runbook `REMOTE_RUN_TRIALS_FUSION.md` referenced (header, §6.1). *The L40S naming and the flat 1e-6 tolerance are superseded by §15 rows 1–2.* |

## 15. Round-6 findings → resolutions (applied in v7)

| # | sev | resolution |
|---|---|---|
| 1 | MEDIUM | Option A. Gate 2 uses a tiered tolerance: `selected_weight` and count leaves exact; continuous leaves (`validation_stats`, tier means/SDs/medians, Cohen's d) abs ≤ 1e-6; rank-based leaves (every `validation_selection_curve` value, AUCs, AP, nDCG@10, P@10, R@10, RR, Kendall/Spearman) abs ≤ 1e-4, with the stated justification. Leaf classes are listed and pinned (`pins.analysis.gate2_tolerance_abs = {"continuous": 1e-6, "rank": 1e-4}`, `gate2_leaf_classes`) and the classifier is unit-tested. "The tolerances do not change" is replaced by: tolerances fixed; a failure persisting after a code fix stops the study and requires a §9 declared-change amendment with `confirmatory_output_seen == false`, consistent with §7 (§5.1, §5.2, §6.2, §7, §8). |
| 2 | MEDIUM | GPU-agnostic but pinned: header names one remote NVIDIA GPU host (the user's A100, or whichever host `pins.environment` records), hardware sizes dropped. Gate 3 requires `cuda_device_name == pins.environment.gpu_name`, recorded by `make_pins.py` from `torch.cuda.get_device_name(0)` on the same host; §3b, §6, §6.3, §7 JSON example (`"<remote GPU host>"`, `"<torch.cuda.get_device_name(0)>"`) and §8 updated; unit test for match and mismatch (§5.1, §5.3). |
| 3 | NIT | `export PYTHONHASHSEED=0` (and `HF_HUB_OFFLINE=1`) opens every §6.2–§6.4 code block; step 1b prefixed with `CUDA_VISIBLE_DEVICES=""`; every new script asserts `PYTHONHASHSEED == "0"` and records it; `train.py` sets it for the runner child and logs it; unit-tested (§5.1, §6, §6.2–§6.4). |
| 4 | NIT | `verify_transfer.py` parses the §3b checkpoint table and §7 inputs table from `--protocol` (row formats defined), aborts on < 15 / < 18 rows, verifies the 36 chunks through the §2.1 manifest digest (per-chunk hashes are pinned later by `make_pins.py`), and records the parsed protocol sha256 in `transfer_check.json`, which `make_pins.py` checks against the pinned protocol; parser unit-tested (§5.1, §6.2, §7). |
| 5 | NIT | MPNet revision = basename of the snapshot path resolved by `huggingface_hub.snapshot_download(..., local_files_only=True)` under `HF_HUB_OFFLINE=1`; must equal `e8c3b32edf5434bc2275fc9bab85f82640a19130`; the model is loaded from that resolved path (§2.1). |
