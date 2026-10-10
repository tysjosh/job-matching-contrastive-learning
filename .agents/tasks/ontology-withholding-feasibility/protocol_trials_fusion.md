# Preregistration: MeSH score fusion in TREC-CT trial matching, confirmation (2022) + replication (2021 CV)

Status: **DRAFT v1, written before any new result**. Round 1 (no design review existed).
Repo `/Users/olukotunjosh/Downloads/CDCL`, branch `orca-experiments-and-runner`, HEAD `147b1e7`. CPU only (Apple M2).

## 0. What is already known, and what is new

Exploratory, post hoc (source: `findings_trials_where_ontology_helps.md` §B1, §C, computed by `/tmp/trials_helps/{score,lib,a2_fusion,a6_twolevel,a7_controls,a7b,a5_hard}.py`):

- On the 15 baseline checkpoints × 50 test patients (2022), validation-nDCG@10-selected fusion of z(text) and z(φ_MeSH, hierarchy): macro within-patient pooled AUC +0.013 [+0.006, +0.020], AP +0.017 [+0.006, +0.029], ineligible-vs-not-relevant AUC +0.014 [+0.007, +0.022]; hard (eligible vs ineligible) −0.001 ns; nDCG@10 +0.014 ns. Shuffled-topic φ −0.002. Hierarchy − exact at w = 0.2: pooled +0.005. Zero-shot MPNet + φ (w = 0.2): pooled +0.012 [+0.005, +0.018]. Trained text is 0.07–0.13 nDCG@10 below zero-shot.

Consequences for interpretation, fixed now:

- **Cohort 2022 is a frozen-pipeline confirmation, not an independent test.** 15 of its 19 checkpoints and all 50 patients were already analysed. The only new information there is 4 new training seeds and the restricted, preregistered estimator.
- **Cohort 2021 (5-fold patient-disjoint CV) is the only independent replication.** No fusion result has been computed on held-out 2021 patients. The exploratory φ separation table in `findings_trials.md` §C used 2021 *train* topics and their judged pools (φ alone, no fusion, no model); this is disclosed as prior exposure.
- Nothing about seeds 87/123 or about any 2021 CV model exists yet.

## 1. Hypotheses

All effects are paired: Δ = metric(fused score) − metric(text score) on the **same** text model, computed per patient and macro-averaged.

**Co-primary (Holm across the two, within each cohort, family-wise α = 0.05, two-sided):**

- **H1** — fusion improves within-patient macro pooled AUC (eligible vs ineligible ∪ not-relevant). Δ_H1 > 0.
- **H2** — fusion improves within-patient macro AP (grade 2 = relevant). Δ_H2 > 0.

The primary text-model population is the **trained baselines** (§3b). Zero-shot (§3a) is a key secondary with its own Holm pair (H1z, H2z), reported in the same table but not part of the replication criterion.

**Secondary (no multiplicity correction, labelled as such; 95% CIs):**

- S1 nDCG@10 (gains 0/1/3), predicted Δ > 0.
- S2 P@10 (grade 2), predicted Δ > 0.
- S3 g1-vs-g0 AUC (ineligible vs not-relevant within patient), predicted Δ > 0.
- S4 hard AUC (eligible vs ineligible within patient), **predicted null**. Equivalence test (TOST): equivalent iff the 90% two-level CI lies inside (−0.01, +0.01).

**Controls (same weight rule, same checkpoints):**

- C1 exact-overlap φ (`MeshMatcher(..., similarity_mode="exact")`): predicted Δ_pooled > 0 but smaller than hierarchy.
- C2 shuffled-topic φ (seed 11, derangement, §2.3): predicted no gain. Pass iff the upper 95% bound of Δ_pooled(C2) < +0.005 **and** Δ_pooled(hierarchy) − Δ_pooled(C2) has lower 95% bound > 0.
- C3 hierarchy − exact contrast, paired per patient: (i) each with its own validation-selected w (primary form of C3), (ii) both at fixed w = 0.2 (replicates the post hoc number). Predicted > 0; labelled suggestive whatever the outcome.

**Interpretive rules fixed now:**

- If C2 fails (shuffled φ gives a gain with lower bound > 0), H1/H2 gains are reported as "not attributable to patient-specific MeSH information", whatever their CIs.
- No claim about eligibility is made from global (cross-patient) hard AUC. Global AUCs are reported only for continuity and never interpreted.
- Fusion is an inference-time scoring rule. No claim about training-time ontology use follows from this study.

## 2. Scores and fusion rule

### 2.1 Text score
- Trained: `epf.text_scores` — projection of frozen `all-mpnet-base-v2` embeddings by `best_checkpoint.pt`, dot product of 128-d projections (as in `scripts/evaluate_publication_fusion.py` L295–319).
- Zero-shot: cosine of the frozen 768-d MPNet embeddings, no projection (as in `/tmp/trials_helps/score.py`, `raw_cos`).
- Text serialisation: `run_phase1_embedding_evaluation.content_to_text` on `encoder_view` (unchanged).
- Any text not in a cache is encoded with `SentenceTransformer(model, device="cpu")`. The existing evaluator constructs `SentenceTransformer` without a device and would pick MPS on this machine; the new scorer must pass `device="cpu"` explicitly.

### 2.2 φ (hierarchy, production)
φ(topic, trial) = `build_mesh_matcher(config).ontology_set_similarity(topic.skill_uris, trial.skill_uris)` with `config/lc_trials_baseline.json` (`mesh_alpha` 0.5, `mesh_max_hops` 12, `similarity_mode` default `"hierarchy"`); `skill_uris` = all MeSH descriptors (condition + intervention). Empty side or exception → 0.0. Identical to `evaluate_publication_fusion.prepare_split`.

### 2.3 Control φ
- Exact: `MeshMatcher(index, alpha=0.5, max_hops=12, similarity_mode="exact")` on the same URIs.
- Shuffled: separately per split (each validation set, each test set), take topics sorted by id, permute with `np.random.default_rng(11).permutation`, and give topic `order[i]` the `skill_uris` of topic `order[(i+1) % n]` (a cyclic shift of the permuted order, so no topic keeps its own URIs). Trial URIs unchanged; hierarchy matcher. This differs from the exploratory a7 version (a plain permutation, which allows fixed points); the derangement is preregistered because a fixed point leaks the true φ into the control.

### 2.4 Normalisation (as implemented, not within patient)
`scripts/evaluate_publication_fusion.py` does **not** z-score within patient. It fits one mean and one population SD (`np.std`, ddof 0; SD 0 → 1) on the **whole validation split** for each score and applies those constants unchanged to test (`z_stats` / `z_apply`, `main()`). The exploratory numbers used the same rule (`lib.z(v, ref)`).

Preregistered primary: **validation-split global z**, exactly as implemented. Reason: this is a confirmation of an existing estimator; changing the normaliser would test a different estimator. It uses no test-label or test-score information.
Preregistered sensitivity (reported, not part of any criterion): within-patient z (mean/SD over each patient's judged pool, computed for validation and test alike; weight re-selected on validation under that normaliser).

### 2.5 Fusion and weight selection
- score = (1 − w)·z(text) + w·z(φ).
- w is selected **per text model (checkpoint)** on the cohort's validation set by macro nDCG@10 (`epf.ranking_metrics(...)["macro"]["ndcg@10"]`, gains 2^g − 1, NaN patients dropped) over the grid W = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}, built as `[i/10 for i in range(6)]`.
- Ties (exact float64 equality) → the smaller w (Python `max` over ascending W returns the first maximiser, as in `epf.main`).
- w = 0 is allowed; then Δ = 0 for that checkpoint and it stays in the average.
- Grid note: the published grid is 0–1.0; all 15 existing nDCG-selected weights are ≤ 0.4 (`trials_fusion_learning_curve_ndcg_selected.json`), so the restriction does not change them.

### 2.6 Metrics, per patient
Computed per patient (`metadata.resume_id`) over that patient's judged trials, all grades, no cap on test:
- pooled AUC: grade 2 vs grades {0,1}; hard AUC: 2 vs 1; g1-vs-g0: 1 vs 0 (tie-aware rank AUC, `epf.rank_auc`).
- AP: grade 2 relevant, full ranking (`epf.ranking_metrics`).
- nDCG@10 and P@10: as in `epf.ranking_metrics`.
- A patient for whom a metric is undefined (missing class) is excluded from that metric only. The undefined set is the same for every score, so pairing holds.
- Patients with no topic MeSH URIs (2022: 2022_1, 2022_10, 2022_15, 2022_17, 2022_31; 2021: 2021_53, 2021_67) stay in (φ ≡ 0 within patient, so Δ ≈ 0). Sensitivity: excluding them.

## 3. Text models and checkpoint populations

### 3a. Zero-shot
Frozen MPNet cosine, identity, no training. One deterministic model per cohort split (cohort 2022: 1; cohort 2021: one per fold, same function, different validation set for w).

### 3b. Trained baselines
`config/lc_trials_baseline.json` (sha256 `708aa7da…f333`, full hash in §7) via `scripts/run_learning_curve_point.py`, InfoNCE, uniform graded negatives, 15 epochs, `best_checkpoint.pt` = lowest validation InfoNCE on grade-2 validation positives with fixed validation negatives (seed 1729, epoch 14).

**Derived configs.** The trainer rewrites `embedding_cache_path` on every run (`trainer.py` L1643). To leave `embedding_cache/trials_text_embeddings.pt` untouched, every new run uses a derived copy of `lc_trials_baseline.json` that differs **only** in:
- `embedding_cache_path` → `results/trials_fusion_confirm/cache/trials_text_embeddings.pt` (byte copy of `embedding_cache/trials_text_embeddings.pt`, sha256 `fa83e10a…1e4d`), and
- for cohort 2021 only, `trials_split_dir` → `preprocess/trials_fusion_confirm/cv2021/fold_{k}`.

These fields affect where data and cached frozen embeddings are read from, not the model, loss, or schedule. The cache copy is a mutable performance cache: training may append frozen embeddings to it, so only the original's hash is pinned. The scorer reads the original, read-only. Each run passes `--compare-config config/lc_trials_baseline.json` so the single-factor audit records the diff in `lc_manifest.json`.

### Cohort 2022 population: 19 checkpoints
Existing 15 (`results/publication_phase1_fixedval/checkpoints/trials_baseline_f{F}_s{S}/best_checkpoint.pt`). Train file `preprocess/trec_ct_lc/frac_{F}/train.jsonl`; validation-file `preprocess/trec_ct_lc/validation_positive.jsonl` (recorded in each `training_config.json` as `validation_path`).

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

New 4: F ∈ {25, 100} × S ∈ {87, 123}, output `results/trials_fusion_confirm/cohort2022/checkpoints/trials_baseline_f{F}_s{S}/`. Every checkpoint gets equal weight in the average; the population is therefore 5 seeds at F = 25 and 100 and 3 seeds elsewhere. The fraction effect on Δ was flat in exploration; no per-fraction claim is made.

Validation for w (cohort 2022): `preprocess/trec_ct_splits/validation.jsonl` (15 2021 topics, grade 0 capped at 60). Test: `preprocess/trec_ct_splits/test.jsonl` (50 2022 topics, uncapped).

### Cohort 2021 population: 15 checkpoints (5 folds × seeds 13/21/42) + 5 zero-shot

**Paraphrase grouping (label-free for scored outcomes).** Edges between 2021 topics:
1. the fixed list from `findings_trials.md` §A: 40–74, 27–64, 25–61, 33–69, 57–63, 23–59, 39–73, 41–75, 36–71, 34–70, 38–72, 21–57, 24–60, 30–66, 31–67, 22–58 (prefix `2021_`);
2. plus any pair whose frozen-MPNet cosine of topic `encoder_view` is ≥ 0.90 (`SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")`, `normalize_embeddings=True`).

Groups = connected components. Guard: if any component has > 6 topics, stop and amend the protocol before building folds. The pins record every component and every pair with cosine ≥ 0.80 (merged or not).
Disclosure: list (1) was partly found with judged-pool Jaccard (qrels). It is used only to keep topics together, which can only make the split more conservative; it never enters a score.

**Fold assignment (fold seed 2021).** Order groups by smallest numeric topic id, shuffle with `random.Random(2021)`, then assign each group in turn to the fold with the fewest topics so far (ties → lowest fold index). 5 folds, about 15 topics each.

**Per outer fold k ∈ {0..4}:**
- test = fold k;
- inner validation = fold (k+1) mod 5;
- train = the other 3 folds (≈ 45 topics, full data, all grade-2 pairs).

**Deviation from the step brief, with reason.** The brief said "train on 4 folds". That is not possible without leakage, because the runner needs a validation set for `best_checkpoint` selection and w selection, and the only alternative held-out data is the test fold. 3 train folds + 1 inner validation fold mirrors cohort 2022 (train 60 / validation 15) and uses the same w rule in both cohorts. The learning curve is flat (5% ≈ 100% positives, findings §B5), so losing one fold of training data should not matter.

Records are built with `trials_domain.data_splitter._build_record` / `_ontology_slot` (imported, not edited) from `preprocess/trec_ct/{topics,trials,qrels}.jsonl`:
- `train.jsonl`: grade 2 only.
- `validation.jsonl`: all grades, grade 0 capped at 60 per topic via `random.Random(f"2021|validation|{topic_id}|0").sample` then sorted.
- `validation_positive.jsonl`: grade-2 rows of `validation.jsonl`.
- `test.jsonl`: all grades, uncapped.
- `negative_pools.jsonl`: rows for train and validation topics, `split` ∈ {train, validation}, full ineligible / not_relevant lists.
- `fold_manifest.json`: topics per split, record counts, sha256 of every file.
- Asserts: topic-disjoint across train / validation / test; every paraphrase group inside one fold.

Training command per (k, seed): §6.3. Output `results/trials_fusion_confirm/cohort2021/fold_{k}/trials_baseline_full_s{seed}/`.

## 4. Uncertainty and decision rule

### 4.1 Two-level bootstrap
- B = 2000.
- Analysis RNG: `np.random.default_rng([31415, 2022])` for cohort 2022 and `np.random.default_rng([31415, 2021])` for cohort 2021.
- The same resample indices are reused for every metric, control, and contrast within a cohort (common random numbers, so contrasts are paired).

**Cohort 2022 (trained).**
- D[c, p] = metric_fused − metric_text for checkpoint c (19) and patient p (50).
- Each replicate draws c* (19 with replacement) and p* (50 with replacement), independently. The statistic is the mean over defined p in p* of the mean over c* of D[c, p].
- Point estimate = the same statistic on the original indices.

**Cohort 2021 (trained).**
- D[s, p] for seed s ∈ {13, 21, 42} and patient p (75), where D[s, p] uses the checkpoint of p's fold trained with seed s.
- Each replicate draws s* (3 with replacement) and p* (75 with replacement); the statistic is as above.
- Patients are resampled across folds, not stratified.

**Zero-shot.** One model, so patient-level bootstrap only, with the same p* draws.

**Intervals and tests.**
- Percentile intervals.
- Two-sided bootstrap p = min(1, 2·min((1 + #{T* ≤ 0})/(B + 1), (1 + #{T* ≥ 0})/(B + 1))).
- Holm over {H1, H2} within each cohort. The hypothesis with the smaller p is tested at 0.025; if rejected, the other is tested at 0.05.
- The reported Holm-adjusted interval is the percentile interval at the matching level: 97.5% for the first step, 95% for the second step if it is reached. Otherwise 97.5% is reported for both.

### 4.2 Pooled (meta) estimate
- Patient-weighted average of the cohort estimates: (50·Δ₂₀₂₂ + 75·Δ₂₀₂₁)/125.
- Its bootstrap distribution combines replicate b of each cohort (independent streams).
- Reported for every metric; descriptive only.
- Sensitivity: inverse-variance fixed-effect pooling using the bootstrap SDs.

### 4.3 Success criterion
For each Hi ∈ {H1, H2}:
- **confirmed (2022)**: the Holm-adjusted lower bound in cohort 2022 is > 0;
- **replicated**: confirmed in 2022 **and**, in cohort 2021, the point estimate is > 0 and the Holm-adjusted lower bound is > 0 (Holm applied within 2021).

Headline wording:
- "MeSH fusion improves within-patient ranking, replicated" only if **both** H1 and H2 replicate.
- "partially replicated" if exactly one does.
- Otherwise the result is reported as observed: confirmed but not replicated, or not confirmed.

Additional rules:
- A C2 failure overrides the headline (§1).
- S4 equivalence is reported per cohort as "equivalent / not equivalent / inconclusive".
- No other outcome changes the headline.

Power (qualitative, no new computation): the exploratory pooled-AUC CI half-width was about 0.007 at 50 patients. With 75 patients, a 2021 effect of the exploratory size would be detectable. An effect half that size likely would not, and that outcome would be reported as "not replicated", not as evidence of absence.

## 5. Gating checks before the confirmatory analysis (code validation on already-seen data only)

1. Unit tests (`trials_domain/tests/test_fusion_confirm.py`): fusion arithmetic, tie-break to smaller w, derangement has no fixed points, fold builder keeps groups together and is topic-disjoint, bootstrap shapes and determinism under a fixed seed, Holm logic.
2. **Reproduction gate.** On the 15 existing checkpoints, run the new scorer with grid 0–1.0 and nDCG selection. Selected weights and per-run text and fused macro metrics must match `results/publication_phase1_fixedval/trials_fusion_learning_curve_ndcg_selected.json` (sha256 `da95c726…9f86`) within 1e-9. Zero-shot macro pooled must match 0.795 (to 3 dp). This touches only the already-analysed population.
3. The new 4 cohort-2022 runs and the 15 CV runs report no `run_error.json`, the `lc_manifest.json` audit shows only the §3b diffs, and `attach.cache_check` passes.

Failure handling:
- A failed training run is rerun once with the identical command.
- A run that fails twice is excluded and reported; the analysis proceeds if ≥ 17/19 (2022) or ≥ 4 folds × 3 seeds (2021) remain, otherwise stop and amend.
- Training is known to be non-deterministic at a fixed seed (findings §B4: rerun score r ≈ 0.7). Each checkpoint is scored once, as produced. Its hash is recorded in the run ledger before any scoring.

## 6. Files, commands, outputs

Interpreter: `/Users/olukotunjosh/Downloads/CDCL/.venv/bin/python` (torch 2.8.0, sentence-transformers 5.1.0, numpy 2.3.2, scipy 1.16.1). All commands run from the repo root with `PYTHONHASHSEED=0` and `CUDA_VISIBLE_DEVICES=""`. `--device cpu` is passed wherever the flag exists.

### 6.1 New files only
- `trials_domain/fusion_confirm.py`: φ variants, derangement, z / fusion / w selection, per-patient metrics (reusing `epf.rank_auc` / `epf.ranking_metrics`), two-level bootstrap, Holm, TOST, paraphrase grouping, fold builder. Ported from `/tmp/trials_helps/{lib,a2_fusion,a6_twolevel,a7_controls,a7b,a5_hard}.py` (port source; may be gone from `/tmp`, in which case this protocol is the spec).
- `scripts/trials_fusion_confirm/make_pins.py`, `build_cv2021.py`, `train.py` (thin wrapper that shells out to the runner), `score.py`, `analyze.py`.
- `trials_domain/tests/test_fusion_confirm.py`.
- Data: `preprocess/trials_fusion_confirm/cv2021/fold_{0..4}/`. Results: `results/trials_fusion_confirm/`.

No existing file is edited, and no existing result is overwritten. Every script refuses to write to an output path that already exists (no `--force`).

### 6.2 Order and commands
```
# (0) protocol approved -> build folds (label-free), copy cache, write pins
.venv/bin/python scripts/trials_fusion_confirm/build_cv2021.py \
    --converted-dir preprocess/trec_ct --out preprocess/trials_fusion_confirm/cv2021 \
    --fold-seed 2021 --n-folds 5 --cosine-threshold 0.90 --max-component 6 --device cpu
cp embedding_cache/trials_text_embeddings.pt results/trials_fusion_confirm/cache/trials_text_embeddings.pt
.venv/bin/python scripts/trials_fusion_confirm/make_pins.py \
    --protocol .agents/tasks/ontology-withholding-feasibility/protocol_trials_fusion.md \
    --out results/trials_fusion_confirm/pins.json      # writes derived configs too
.venv/bin/python -m pytest trials_domain/tests/test_fusion_confirm.py -q

# (1) reproduction gate (already-seen data)
.venv/bin/python scripts/trials_fusion_confirm/score.py --gate-reproduce \
    --pins results/trials_fusion_confirm/pins.json --device cpu \
    --out results/trials_fusion_confirm/gate_reproduction.json
```

### 6.3 Training (19 runs)
```
# cohort 2022: (F, FRAC) in {(25, 0.25), (100, 1.0)}, S in {87, 123}
PYTHONHASHSEED=0 .venv/bin/python scripts/run_learning_curve_point.py --domain trials \
  --config results/trials_fusion_confirm/configs/lc_trials_baseline_cache.json \
  --compare-config config/lc_trials_baseline.json \
  --train-file preprocess/trec_ct_lc/frac_${F}/train.jsonl \
  --validation-file preprocess/trec_ct_lc/validation_positive.jsonl \
  --fraction ${FRAC} --seed ${S} --device cpu \
  --output-dir results/trials_fusion_confirm/cohort2022/checkpoints/trials_baseline_f${F}_s${S}

# cohort 2021: k in 0..4, S in {13,21,42}
PYTHONHASHSEED=0 .venv/bin/python scripts/run_learning_curve_point.py --domain trials \
  --config results/trials_fusion_confirm/configs/lc_trials_baseline_cv2021_fold${k}.json \
  --compare-config config/lc_trials_baseline.json \
  --train-file preprocess/trials_fusion_confirm/cv2021/fold_${k}/train.jsonl \
  --validation-file preprocess/trials_fusion_confirm/cv2021/fold_${k}/validation_positive.jsonl \
  --fraction 1.0 --seed ${S} --device cpu \
  --output-dir results/trials_fusion_confirm/cohort2021/fold_${k}/trials_baseline_full_s${S}
```
`scripts/trials_fusion_confirm/train.py --pins ... --cohort {2022,2021}` issues exactly these commands and then appends `{run, command, best_checkpoint sha256, lc_manifest sha256, wall time}` to `results/trials_fusion_confirm/run_ledger.json`.

### 6.4 Scoring and analysis (once)
```
.venv/bin/python scripts/trials_fusion_confirm/score.py --pins results/trials_fusion_confirm/pins.json \
    --ledger results/trials_fusion_confirm/run_ledger.json --device cpu \
    --out results/trials_fusion_confirm/scores/
.venv/bin/python scripts/trials_fusion_confirm/analyze.py --pins results/trials_fusion_confirm/pins.json \
    --scores results/trials_fusion_confirm/scores/ \
    --out results/trials_fusion_confirm/analysis.json \
    --report results/trials_fusion_confirm/report.md
```

Outputs:
- `scores/{cohort}/{model}.npz`: per-pair text, φ_hier, φ_exact, φ_shuf, grades, patient ids, nct ids, for validation and test, plus `selected_w.json`.
- `analysis.json`: every estimate, CI, p, Holm result, and verdict, plus the sensitivity analyses.
- `report.md`: tables in the order H1, H2, H1z, H2z, S1–S4, C1–C3, sensitivities, pooled.

Prepared-embedding caches go under `results/trials_fusion_confirm/prepared_embeddings/`, never the published one.

## 7. Pins JSON (`results/trials_fusion_confirm/pins.json`)

Written once after approval, before training (§6.2 step 0). The scorer and analyzer refuse to run if any pinned hash mismatches.

```json
{
  "schema": "trials_fusion_confirm/pins/v1",
  "created_utc": "...",
  "protocol": {"path": ".agents/.../protocol_trials_fusion.md", "sha256": "..."},
  "git": {"head": "147b1e7...", "branch": "orca-experiments-and-runner", "dirty_paths": ["..."]},
  "environment": {"python": "...", "torch": "2.8.0", "sentence_transformers": "5.1.0",
                   "numpy": "2.3.2", "scipy": "1.16.1", "device": "cpu", "PYTHONHASHSEED": "0"},
  "inputs": {"<relpath>": "sha256", "...": "..."},
  "code": {"<relpath of each new and reused script/module>": "sha256"},
  "configs": {"base": {"path": "config/lc_trials_baseline.json", "sha256": "..."},
              "derived": {"<path>": {"sha256": "...", "diff_vs_base": {"field": ["old", "new"]}}}},
  "cohort2022": {"existing_checkpoints": [{"fraction": 10, "seed": 42, "path": "...", "sha256": "..."}],
                 "new_runs": [{"fraction": 25, "seed": 87, "output_dir": "..."}],
                 "validation": "preprocess/trec_ct_splits/validation.jsonl",
                 "test": "preprocess/trec_ct_splits/test.jsonl"},
  "cohort2021": {"fold_seed": 2021, "n_folds": 5, "cosine_threshold": 0.90,
                 "paraphrase_edges_fixed": [["2021_40", "2021_74"]],
                 "paraphrase_edges_cosine": [["...", "...", 0.93]],
                 "pairs_cosine_ge_0_80": [["...", "...", 0.82]],
                 "components": [["2021_21", "2021_57", "2021_63"]],
                 "folds": {"0": ["2021_..."]},
                 "outer": [{"k": 0, "test": 0, "validation": 1, "train": [2, 3, 4]}],
                 "fold_files": {"fold_0/train.jsonl": "sha256"},
                 "seeds": [13, 21, 42]},
  "analysis": {"weight_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "select_on": "ndcg@10",
               "tie_break": "smaller_w", "normalisation": "validation_split_global_z",
               "B": 2000, "rng": {"2022": [31415, 2022], "2021": [31415, 2021]},
               "shuffle_seed": 11, "shuffle": "cyclic_shift_of_permutation",
               "equivalence_bound": 0.01, "alpha": 0.05, "holm_family": ["H1", "H2"]}
}
```

`inputs` must include at least these files (current sha256):

| file | sha256 |
|---|---|
| preprocess/trec_ct_splits/test.jsonl | 4030cc85ff1b4ef389896fa2ce7f3bed827b9640028446dfa96548f00ae24348 |
| preprocess/trec_ct_splits/validation.jsonl | 5defb59b858bdeeed2c04ad737e20def86bc53196dd27ac3828a41d4375cdfe1 |
| preprocess/trec_ct_splits/train.jsonl (= frac_100) | e4620c65c0a2ac8229ae35dd47440accfd7442e25a5d931182012cc519a7bfa8 |
| preprocess/trec_ct_splits/negative_pools.jsonl | f190ab7ca6a0c154c2381d44dfeb25c884565aa0957252c16eda2c39269c6af3 |
| preprocess/trec_ct_splits/split_manifest.json | 32160deaa0f149f902e4379644adb93028ace195c0f1f171a9e367460e9d2d66 |
| preprocess/trec_ct_lc/frac_25/train.jsonl | 9fb56dbec881a09c17d714cfd1e03714a7384f40ede9aa41f17f26cdb28a893c |
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

The 2022 qrels are embedded in `test.jsonl` (grades per record). The raw TREC qrels are in `preprocess/trec_ct/qrels.jsonl` (both years).

## 8. Limitations, stated in advance

- **Pooling bias.** TREC judged pools come from participant systems. If those systems used condition / MeSH matching, φ's relevance AUC may partly reflect how the pools were built, not patient relevance. Both cohorts share this, so replication does not remove it.
- **External validity.** All 125 patients are synthetic TREC vignettes. There is no real-cohort check.
- **Cohort 2022 is not independent of the exploration** (§0). Only cohort 2021 tests the claim out of sample. Cohort 2021 is in-domain (same campaign, same annotators), not a new collection.
- **Weak trained base.** Trained text is 0.07–0.13 nDCG@10 below zero-shot. A gain over trained text does not imply a gain over the best text model; H1z / H2z address this but are not in the replication criterion.
- **Training non-determinism** at a fixed seed (§5). The bootstrap's checkpoint level absorbs it; individual checkpoints are not reproducible bit for bit.
- **Topic-side MeSH extraction is noisy** (primary condition captured in about 80% of 2021 and 60% of 2022 topics, eyeball estimate). Effects are conditional on this extractor.
- Fusion addresses topical relevance only. The eligibility contrast (S4) is predicted null, and nothing here licenses an eligibility claim.

## 9. Amendments
None. Any change after `pins.json` is written is logged here with date, reason, and whether any affected result had been seen.
