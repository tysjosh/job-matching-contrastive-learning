# GO/PPI inference-time fusion: inventory and feasibility for a preregistered confirmation

Read-only investigation. Repo `/Users/olukotunjosh/Downloads/CDCL`, branch `orca-experiments-and-runner`, Apple M2, CPU only (`CUDA_VISIBLE_DEVICES=""`, no `SentenceTransformer` constructed anywhere). No repo file was created or modified apart from this one. Scratch scripts and outputs are in `/tmp/goppi_fusion/`:
- `lib.py`: a read-only port of `epf.prepare_split`, with no encoder and no mkdir;
- `recheck.py`, `all15.py`: the re-score;
- `feasibility.py`, `feas2.py`, `feas3.py`: counts;
- `zs.py`: the zero-shot text level.

Template: `protocol_trials_fusion.md` (v7 + A0–A3) and `results_trials_fusion.md`. Below, `epf` = `scripts/evaluate_publication_fusion.py`.

Tags: **[read]** from code, **[ran]** computed here, **[inferred]** reasoned.

## Summary

1. **The user's exploratory summary is correct.** I re-scored all 15 published checkpoints on CPU from read-only sources. All 30 selected weights match. Every test macro leaf of two checkpoints is bit-identical to the published JSON, and every number in Table G1/G2 reproduces (§5). New here: two-level CIs for the AUC deltas. Hard AUC is **+0.0195 [+0.0114, +0.0272]** (pooled-AUC rule) and **+0.0222 [+0.0118, +0.0330]** (nDCG rule). Only **729 of the 1,396 anchors** carry a hard AUC, and 735 an AP. 661 test anchors have no positive.
2. **The 15 checkpoints were trained on unpruned, unmasked data, but GO never entered their training.** They are the uniform-negative baseline: `attach.matcher = null`. So:
   - The GAF with/from leak enters **only through φ at scoring time**. It can be removed without retraining by switching φ to the pruned index and the pruned record `skill_uris`. These are aligned 1:1 with the published files (verified).
   - The partner-symbol text leak (11.1% of test positives) **is inside the text model**. Removing it needs retraining on the masked `encoder_view`.
3. **Every required control is feasible and cheap.** Scoring is about 2 s per checkpoint per φ variant on CPU. One finding changes how the shuffled control must be built: the trials-style whole-file rejection derangement is **infeasible on GO**. It accepted 0 of 2,000 draws, with a mean of 9.4 violations, because each anchor's candidates are forbidden donors. A derangement within closure-size deciles works in 1–90 draws per stratum (§4.1).
4. **The HuRI stratum on the current test set is too small for inference.** It has 63 HuRI-supported positive records on 58 anchors. Two ways to keep the requirement:
   - a separate **HuRI-labelled cohort** inside the test partition: 769 model-unseen anchors and 2,860 HuRI pairs, 2,782 of which are not existing positives;
   - a descriptive stratum in the CV replication: 375 records on 337 anchors.
5. **One replication option is infeasible:** "unused raw-STRING proteins". The test partition has **0** anchor-eligible proteins that are not already anchors. The 3,735 STRING genes left out of the eligible set have no BP annotation after the guards, so φ ≡ 0 for them.
6. **Recommendation:**
   - Cohort A is a frozen-pipeline confirmation on the current split: 19 checkpoints retrained on masked text, primary φ = pruned simGIC.
   - Cohort B is a 5-fold protein-disjoint CV replication over all 15,155 proteins. It has about 3,850 hard-defined anchor queries, 2,694 of them closed-unexposed.
   - Primary weight rule: nDCG@10 over the full 0–1.0 grid.
   - Controls:
     - C1 exact-term φ;
     - C2 a decile-stratified, candidate-excluding anchor derangement;
     - C4/C5 a study-bias score (partner GAF PMID count), alone and as a competitor;
     - a pruned-vs-unpruned leak contrast;
     - the HuRI cohort.
   - Gate 2 needs a top-two-gap rule for `selected_weight`: the smallest published nDCG validation gap is **7.2e-6**, against 1.76e-4 in trials.

---

## 1. Inventory

### 1.1 Code paths [read]

| path | role | notes |
|---|---|---|
| `scripts/evaluate_publication_fusion.py` (sha `606793ef…`) | the published fusion; `DOMAINS["go_ppi"]` | Protected. `prepare_split` builds `SentenceTransformer(config.text_encoder_model)` with **no device**, so it picks MPS on the Mac. It also `mkdir`s and can write chunks into the published cache dir. Like trials `score.py`, a confirmatory scorer must port it read-only. |
| `go_ppi_domain/run_config.build_go_matcher` → `go_ontology.GoMatcher` | φ | `similarity_mode` ∈ {`simgic` (default), `ancestor`, `exact`}, all already implemented. |
| `go_ppi_domain/go_ontology.GoIndex.load/build_or_load` | index | `build_or_load` rebuilds and **writes** the cache if the aspect mismatches. Use `GoIndex.load` read-only. |
| `scripts/probe_fusion_mechanism.py` | older fusion probe | Used `results/lc_single_factor/go_ppi_baseline_f100_s{42,13,21}` and a 6,000-record subsample, selecting on hard AUC. |
| `scripts/go_bulk_signal_diagnosis.py`, `scripts/go_independent_signal_diagnosis.py` | φ-only ceilings | The independent one maps HuRI/Lit-BM through STRING `Ensembl_gene` aliases and uses degree × GO-size-decile matched non-edges. |
| `trials_domain/…`, `scripts/trials_fusion_confirm/{score,analyze,make_pins,verify_transfer,train,build_cv2021,dump_code_hashes}.py` | the trials pipeline to mirror | Protected. Copy patterns into new files, never edit. |
| `go_ppi_domain/{leak_pruning,data_splitter,data_converter}.py`, `scripts/go_withholding/*` | pruned-feature builder, splitter | Protected. Import only. |

### 1.2 Results files

| file | content |
|---|---|
| `results/publication_phase1_fixedval/go_ppi_fusion_learning_curve.json` (sha `0835241b…`) | selection on `pooled_auc`, W = 0..1.0 |
| `results/publication_phase1_fixedval/go_ppi_fusion_learning_curve_ndcg_selected.json` (sha `07927eb4…`) | selection on `ndcg@10` |
| `results/ontology_ceiling/go_independent_signal.json` | HuRI AUC 0.548 [0.544, 0.552] on 33,661 pairs; Lit-BM 0.674 [0.667, 0.681] on 11,624 pairs |
| `results/ontology_ceiling/{fusion_probe_go_ppi,fusion_probe_go_ppi_leakfixed,phase1_single_stage_fusion_go_ppi,fusion_reselect}.json` | older 3-checkpoint probes (lc_single_factor f100) |

Per run, each published JSON stores: `selected_weight`, `validation_selection_curve`, `validation_stats`, and test `{text,fusion,delta}`. Under test it holds `contrasts.{global_auc,macro_query_auc}` and `ranking.{macro,per_query}`. `per_query` carries AP, nDCG@10, P@10, R@10 and RR per anchor. **Per-anchor AUCs are not stored**, so AUC CIs need a re-score (§5).

### 1.3 The 15 checkpoints [ran]

Path pattern: `results/publication_phase1_fixedval/checkpoints/go_ppi_baseline_f{F}_s{S}/best_checkpoint.pt`.
- Train file: `preprocess/go_ppi_lc/frac_{F}/train.jsonl`. Validation: `preprocess/go_ppi_lc/validation_positive.jsonl` (sha `c610cd6e…`).
- Config: `config/lc_go_ppi_baseline.json` (sha `0a385d97…`).
- `text_encoder_device: null`, so the device was picked automatically on the Mac. `lc_manifest.json` has no device block.
- Best epoch = argmin of the validation loss.

| F | S | train records | best epoch | train s | sha256 |
|---|---|---|---|---|---|
| 10 | 42 | 1,571 | 14 | 21 | f42e599e54f554284bf5cce18eb06540d1b526c92a7938108f62cbeb7ec56f08 |
| 10 | 13 | 1,571 | 10 | 21 | e1dc9a17c5957acd2b6164af1e09e64868ed8a1cd28604296303882dd7cec110 |
| 10 | 21 | 1,571 | 11 | 21 | 46b091f470012581d4181e23db495deec3daae83f7ad75cde721059db188ff5d |
| 25 | 42 | 3,927 | 12 | 51 | 7b1721294a1200684718f9a827919275a933bccbe5df22a58f2039b1821739ac |
| 25 | 13 | 3,927 | 13 | 50 | 49c551d6e6877b249d8adc4f65567c42297297c14b35fe949228c8ad424d927e |
| 25 | 21 | 3,927 | 13 | 48 | bf494d05c4d3a27e8ffabdb5a0343993cdb0d433dcbb579afb6ebbdfdc5d5282 |
| 50 | 42 | 7,854 | 8 | 96 | 59c82d28a342a4604c39faf131a9ef42029ec92b03bbd08cd170660519e14dff |
| 50 | 13 | 7,854 | 9 | 96 | 65dfaefab157353ca97652798e07fd21718646087b7dd037260c2515f8d8e833 |
| 50 | 21 | 7,854 | 7 | 94 | 1449bc54ea503c794fcf4233e5aab2ac810814e82361f49032a0dfdf68506f31 |
| 75 | 42 | 11,781 | 14 | 146 | 9d5cd47647a6d2dc171e516c0b0f8024a1321accaa8a8bf0b58dada731fed121 |
| 75 | 13 | 11,781 | 13 | 147 | f3d328512e45e537322b740ef04d5fac5a2ca7b754588f2e1e1ccccdf779c0bc |
| 75 | 21 | 11,781 | 12 | 148 | 784093247973b9f310ed4408ef78334932d13818d090d9076025e27d2d91f68e |
| 100 | 42 | 15,708 | 12 | 194 | ae1516bfed20880fc38064c595727906235ee2936c376754e683c1b1d24e1d82 |
| 100 | 13 | 15,708 | 12 | 197 | 484e4ea387633af521f4f7079dec1119ccd0624f67bf7c6731a4c4fb1829074d |
| 100 | 21 | 15,708 | **4** | 199 | 70e53518b36551d8f1531d27321f95d4c46ffd2f2dc69b0adf74ad4998856708 |

f100_s21 stopped at epoch 4. Its text-score mean (0.350) is far from the other f100 runs (0.217, 0.226). This is one reason the per-fraction trend is noisy.

### 1.4 Caches and inputs (sha256) [ran]

| input | sha256 | notes |
|---|---|---|
| `preprocess/go_ppi_splits/validation.jsonl` | 07e432fb64f3e2716d74019806b0926aa26b2926a12b541a2fda1b8bf8ac91c6 | 5,587 records, 697 anchors, 321 grade-2 |
| `preprocess/go_ppi_splits/test.jsonl` | c65a1d2ca0e5c16a1c69ecd70d921e670531d325c52625aa15dba88c2e2481e4 | 22,341 records, 1,396 anchors, 1,165 grade-2 |
| `preprocess/go_ppi_splits/train.jsonl` (= `go_ppi_lc/frac_100`) | c2013f8a134d24dbb688a96515dc41c118d8932e211feefde395fe96f9a02efe | |
| `preprocess/go_ppi_lc/frac_{10,25,50,75}/train.jsonl` | 9b353aba… / 0751771d… / 1ae69c62… / 043e385f… | nested prefixes |
| `preprocess/go_ppi/go_index_P.pkl` (unpruned) | 56892eaaf1363d24e22b8bf697dc16c4b68de6672afb019f7263bcdb2968d042 | |
| `embedding_cache/go_ppi_text_embeddings.pt` (frozen, unmasked) | 923412dfa0b8fd7ab60d2011f8fbdbee7fdad7b11d6b77f415f2485fe20f41ba | |
| `prepared_embeddings/go_ppi/test_c65a1d2ca0e5c16a_…` (5 chunks) | manifest digest 42abdb8c3765e6963ab552d019a93cf29a08e87e473b8fb61ae2e056a1fbe5b4 | trials digest rule (`shasum … \| sort -k2 \| shasum`, cwd = chunk dir) |
| `prepared_embeddings/go_ppi/validation_07e432fb64f3e271_…` (2 chunks) | manifest digest cefa1e04cb2be64ba6222b7c7fe2907565422a8bfd258a5250b6e28d9e189063 | validation resolves 1,432 texts from the frozen cache and 25 from chunks; test resolves 0 / 3,014 |
| `preprocess/go_withholding/converted/go_index_P.pkl` (pruned) | a0c8b0b4c062a4f550a3f0f42288fae45714977aca903564822cbf2eca2d590e | read-only |
| `preprocess/go_withholding/splits/{validation,test,train}.jsonl` (masked + pruned) | 90c0b7df… / 4027e9a0a3cc6ec173fe468e90ff48e013efb8f61dc8eb2ce61d5b7a346bbb2a / c38f89c2… | read-only; pins G1 in `results/go_withholding/protocol_pins.json` |
| `results/go_withholding/eval_cache/test_4027e9a0a3cc6ec1.npz` | abf715f9061f5d998b6d21cf7d496bc9859d092f39b0e625cf02de0f2c5f0b28 | masked test, 3,014 proteins, built on CPU |
| `embedding_cache/go_withholding_text_embeddings.pt` | 3aed51487550ec02216b49617310927f48c0e29195dd4e3d1698f6adaa88a150 | 17,620 entries; holds only **740 of 1,457** masked validation views |
| `dataset/go_ppi/independent/HuRI.tsv`, `Lit-BM.tsv` | 4cdf9436… / f39c9c4b… | |
| `dataset/go_ppi/bulk/go-basic.obo`, `goa_human.gaf.gz` | b08d45b2… / db472faf… | |

## 2. How the published fusion computes its numbers [read, confirmed by re-score]

1. **Records.** The grade is `job.grade`, with an `original_label` fallback. The query is `metadata.resume_id`, the anchor protein. The text is `content_to_text` → `encoder_view`, verbatim.
2. **Text score.** Frozen MPNet 768-d vectors are passed through the checkpoint's projection head, then dotted (`epf.text_scores`). Vector resolution: frozen cache by `EmbeddingCache.get_content_key(slot)` first, then prepared chunks by sha256(text), then a fresh encode.
3. **φ.** simGIC = Σ IC(closure_a ∩ closure_b) / Σ IC(closure_a ∪ closure_b).
   - The closures are taken over the **record's** `skill_uris`, which are direct BP terms.
   - The closure follows is_a + part_of; `regulates` is dropped.
   - IC = −log(freq/19,293) over the whole guarded GOA corpus, not training-derived.
   - An empty side gives 0.0. The index enters only through `ancestors` and `ic`, never `index.direct`.
4. **Normalisation.** Global z with **validation-file** mean and population SD (SD 0 → 1), applied unchanged to test. φ stats are fit once (mean 0.0596, SD 0.1121); text stats are fit per checkpoint.
5. **Fusion and selection.** fused = (1 − w)·z_text + w·z_φ over W = {0.0, …, 1.0}. Ties go to the smaller w (first maximiser).
   - **`--select-on pooled_auc` maximises the *global* (cross-anchor) validation pooled AUC, not a within-anchor metric.**
   - `ndcg@10` maximises macro per-anchor nDCG@10 with gains 2^g − 1.
6. **Metrics.**
   - `macro_query_auc.{hard (2 vs 1), easy (2 vs 0), middle_vs_easy, pooled (2 vs {0,1})}`: the mean over anchors that have both classes, using the tie-aware rank AUC.
   - `ranking.macro`: AP and RR over grade 2, nDCG@10, P@10 and R@10, averaged over anchors with the metric defined.
   - Δ = fused − text on the same checkpoint. The user's "within-anchor" numbers are differences of the macro means.
7. **Thresholds** (binary and ordinal) are fit on validation. They are irrelevant here.

## 3. Training data of the 15 checkpoints and what it implies [read, ran]

- **Data.** `lc_manifest.json` / `training_config.json` show:
  - `go_ppi_split_dir = preprocess/go_ppi_splits`, `go_ppi_converted_dir = preprocess/go_ppi`, train `preprocess/go_ppi_lc/frac_*`. These are unpruned and unmasked.
  - `go_ppi_go_tiered_negatives = false`, `attach.matcher = null`: no GO in training.
- **The with/from leak (5.9% of test positives, 0.4% of weak pairs) is a φ-side artefact only.** It sits in the record `skill_uris` and the unpruned IC.
  - The pruned files `preprocess/go_withholding/splits/{validation,test}.jsonl` are **record-for-record aligned** with the published ones: same order, anchor, partner and grade [ran].
  - So the published checkpoints can be scored with pruned φ directly.
  - Label-free effect of the pruning [ran]:

    | | test | validation |
    |---|---|---|
    | anchor-side URI lists changed | 6,908 records | 1,715 records |
    | partner-side URI lists changed | 7,930 records | 1,943 records |
    | φ values changed | 20,876 of 22,341 | 5,229 of 5,587 |
    | Spearman(φ_unpruned, φ_pruned) | 0.836 | 0.846 |

  - Almost every φ value moves because the pruned index recomputes IC on the pruned corpus.
  - On test, 42 anchors (661 records) and 250 partners lose every BP annotation, so φ_pruned = 0 for them.
  - Pruning is **global** (it drops any annotation whose with/from names *any* human protein, plus protein-binding descendants). It is not partner-specific. So it also removes interaction-derived annotations that do not name the partner. That makes it conservative.
- **The partner-symbol leak (11.1% of test positives) is inside the text model**, trained on 11.2% mention positives, and inside the test text.
  - It inflates text on those positives and so changes the room for fusion. The sign of its effect on Δ is not known [inferred].
  - Removing it needs retraining on the masked view. Masked train is aligned 1:1 with the published train (verified), so nested masked fractions reproduce the exact published fraction membership.
- **Residual text leak not removed by masking:** `d_complex_name` is 28.1% / 16.5% / 16.3% (grades 2/1/0, `smoke_report_go_ppi.md` §2). It should be disclosed.

## 4. Feasibility and cost of each required control and cohort

Costs are measured on the Mac CPU unless stated:
- φ for 27,928 records takes about 3 s;
- text scoring of one checkpoint on both files takes about 1 s;
- the full 15 × 2-rule re-score with per-anchor metrics takes 34 s.

### 4.1 Shuffled-GO control (C2) [ran]

- **Rule to mirror** (trials §2.3): anchor-side URIs are deranged and partner URIs are kept.
- **What fails.** Each test anchor's candidates (median 17 forbidden donors including itself) must not be a donor: receiving a candidate's own set makes φ(set_b, b) ≈ 1. Whole-file rejection, repeating `rng.permutation(n)` with a cyclic shift until no violation, gave **0 accepted out of 2,000 draws**, with a mean of 9.4 violations. The trials rule would abort.
- **What works.** Stratify the scored file's anchors by deciles of BP closure size, and run the trials rejection rule (`default_rng(11)`, 10,000-draw cap) **within each decile**. Forbidden donors:
  - the anchor itself;
  - any of its candidates, of any grade;
  - any anchor with an identical URI set (118 test anchors share a set with another);
  - any donor with simGIC(own, donor) ≥ 0.5.

  First accepted draw per decile: test 1–90 (sizes 123–147); validation 1–4 (sizes 66–75).
- **Why decile strata.** They also hold the anchor's annotation density fixed. So C2 keeps the partner-side and anchor-density components of φ and removes only pair-specific information.
  - Partner annotation density is untouched by any anchor-side shuffle. Validation partners' median GAF references are 13 / 20 / 38 for grades 0 / 1 / 2.
  - So **C2 may fail in GO for study-bias reasons**. That is informative, not a bug, and the analysis should say so in advance.

### 4.2 Exact-term vs simGIC (C1, C3) [ran]

- `GoMatcher(similarity_mode="exact")` already exists: Jaccard of direct record terms, with no hierarchy and no IC.
- On test, 88.1% of exact values are 0, against 6.3% for simGIC. Spearman(simGIC, exact) = 0.46.
- `ancestor` mode (closure Jaccard, no IC) has Spearman 0.946 with simGIC. It is the natural middle rung that separates hierarchy from IC.
- Cost: negligible.

### 4.3 Pruned-index reuse [ran]

- Feasible as is: pinned, read-only and aligned (§3).
- For φ, use `GoIndex.load(preprocess/go_withholding/converted/go_index_P.pkl)` together with the **pruned record `skill_uris`**. Use both together, never one without the other.
- The masked `encoder_view` comes from the same files.

### 4.4 HuRI stratum and cohort [ran]

HuRI is mapped to our symbols through the STRING `Ensembl_gene` aliases: 29,301 edges over 6,274 eligible proteins. Lit-BM: 11,302 edges over 5,523 proteins.

| set | grade-2 records with a HuRI edge | anchors with ≥1 (all also have a grade-1) | HuRI edges in grade 1 / grade 0 | both endpoints HuRI-observed, grade 2/1/0 |
|---|---|---|---|---|
| current test | **63** of 1,165 | **58** | 51 / 5 | 347 / 2,173 / 2,214 |
| current validation | 17 of 321 | 16 | 9 / 0 | 77 / 522 / 524 |
| Lit-BM, test | 179 | 151 | 1 / 0 | 589 / 3,646 / 2,266 |
| CV (5 folds, all proteins) | 375 | 337 | — | — |

- **A HuRI stratum of the current test set (58 anchors) has too little power for any decision.** It can only be descriptive.
- **HuRI-labelled cohort (feasible).** Inside the test partition, whose proteins none of the cohort-A models has seen:
  - 769 anchors have ≥1 HuRI partner in the partition (median 2), giving 2,860 directed HuRI pairs;
  - **2,782 of those pairs are not STRING grade-2 records**, so they are new labels;
  - 349 of the anchors are not current test anchors, and 420 are, with different labels;
  - 1,242 test-partition proteins appear in HuRI.
- **Negatives for that cohort.** Use HuRI-observed test-partition proteins with no HuRI edge and no STRING edge of any channel to the anchor. Match them on HuRI-degree decile × BP-closure-size decile, as `go_independent_signal_diagnosis.py` does, with 20 per anchor drawn from a fixed seed.
- **Caveat.** "HuRI-observed" (≥1 edge) is a proxy for the HuRI search space. The ORFeome list is not in the repo, so some negatives were never tested.

### 4.5 Study-bias covariate [ran]

- **Protein-level proxy:** distinct references per gene in `goa_human.gaf.gz`, all aspects, all evidence. Test proteins: median 14, p90 40, no zeros. Validation partners: median 13 / 20 / 38 at grades 0 / 1 / 2.
- **STRING `textmining` is not usable.** It is pair-level and label-adjacent: grade-2 median 505–625, grade-1 about 122, and **grade 0 ≡ 0 by construction**, because grade 0 means no STRING edge at all. It would act as a label leak.

### 4.6 Replication-cohort options (counts from `preprocess/go_ppi/pairs.jsonl`; fold seed 2026 here is illustrative only) [ran]

"Hard-defined" = an anchor with ≥1 grade-2 and ≥1 grade-1 record. "Exposed" = an anchor in the current test partition (scored under fusion) or validation partition (fusion weight-selection curves computed).

| option | anchors (hard-defined) | exposed | train grade-2 (directed) vs current 15,708 | new models | verdict |
|---|---|---|---|---|---|
| (a) new protein-disjoint resplit of the 12,124 non-test proteins (25% test / 12.5% val) | 1,409 (815) | 0 test-exposed; includes validation-partition proteins | 7,996 (51%) | yes | feasible; halves the training data; one test set |
| (b) unused raw-STRING proteins | **0 new anchor-eligible** in the test partition (737 uncapped-eligible = existing anchors) | — | — | — | **infeasible.** 3,735 STRING genes have no BP annotation after the guards (φ ≡ 0) and 809 have short text. The only unused STRING data are 2,540 extra grade-2 partners of *exposed* anchors that the 8-positive cap removed (3,705 uncapped vs 1,165). |
| (c) HuRI-anchored queries | 769 anchors, 2,860 pairs | 420 anchors (new labels) | reuses cohort A | no | feasible as a **secondary cohort** with a different label (§4.4); not a replication of the STRING-grade claim |
| (d) 5-fold protein-disjoint CV over all 15,155 proteins (3 train / 1 val / 1 test per fold, like trials 2021) | about 1,400 per fold, 7,041 total (**3,853**) | 778 test + 381 validation; **2,694 closed-unexposed** (current train partition) | 11,505 (73%) | yes, 15 | **recommended** |
| (d′) 5-fold CV over the non-test proteins only | about 1,130 per fold (2,669) | validation-partition only | 7,706 (49%) | yes, 15 | feasible; smaller and less training data |

Duplicate texts: 14 groups (31 proteins) share an identical masked view body. They should be kept in one fold, the analogue of trials paraphrase groups.

### 4.7 Compute and cost

| item | cost | source |
|---|---|---|
| cohort A retraining | 19 GPU runs (15 cells + seeds 87/123 at F 25/100), each ≤ 3 min on an L40S-class GPU | trials took 62 min for 34 runs; GO f100 took 194 s on the Mac |
| cohort B | 15 GPU runs; data build from `pairs.jsonl` takes seconds | |
| fresh remote-CPU encodes of masked views | ≤ 15,155 proteins, roughly 15–30 min CPU | [inferred]; 717 masked validation views are not in any cache |
| scoring | (34 + 1 zero-shot) models × about 7 φ variants × 2 files: minutes on CPU | |
| bootstrap B = 2000 over 19 × 1,396 | seconds to minutes | |
| disk | well under 1 GB new | the 4 GB `go_withholding/conditions` is not needed |

## 5. Re-check of the exploratory numbers [ran, `/tmp/goppi_fusion/{recheck,all15}.py`]

**Correctness check (f10_s42 and f100_s21, both rules):**
- `selected_weight` matches;
- validation stats differ by 0.0;
- every test `macro_query_auc`, `global_auc` and `ranking.macro` leaf differs by **0.0**;
- per-anchor AP differs by 0.0;
- the validation curve differs by at most 1.2e-6 (pooled rule) and 0.0 (nDCG rule).

All 15 × 2 selected weights match the JSONs.

**All 15, Δ = fused − text, two-level bootstrap (15 checkpoints × anchors, B = 2000, seed 20260101):**

| metric (anchors) | text | pooled-AUC rule: Δ [95% CI], checkpoints > 0 | nDCG@10 rule: Δ [95% CI], checkpoints > 0 |
|---|---|---|---|
| within-anchor hard (729) | 0.7969 | +0.0195 [+0.0114, +0.0272], 15/15, range +0.0139..+0.0270 | +0.0222 [+0.0118, +0.0330], 15/15 |
| within-anchor pooled (735) | 0.8463 | +0.0198 [+0.0144, +0.0260], 15/15 | +0.0253 [+0.0175, +0.0340], 15/15 |
| within-anchor easy (735) | 0.8887 | +0.0198 [+0.0146, +0.0258] | +0.0275 [+0.0201, +0.0357] |
| AP (735) | 0.6641 | +0.0256 [+0.0158, +0.0360] | +0.0256 [+0.0119, +0.0400] |
| nDCG@10 (1,387) | 0.7595 | +0.0177 [+0.0146, +0.0210] | +0.0259 [+0.0216, +0.0305] |
| P@10 (1,396) | 0.0766 | +0.0013 [+0.0008, +0.0019] | +0.0019 [+0.0012, +0.0026] |

- Weights: pooled rule 0.2 ×3, 0.3 ×11, 0.4 ×1; nDCG rule 0.4 ×3, 0.5 ×7, 0.6 ×5.
- Global hard Δ (from the JSONs): +0.0194 and +0.0225.

**By fraction (hard Δ), F = 10/25/50/75/100:**
- pooled rule: +0.022 / +0.022 / +0.021 / +0.017 / +0.015;
- nDCG rule: +0.027 / +0.025 / +0.025 / +0.019 / +0.015.

This matches Table G2. Two notes:
- Under the nDCG rule, nDCG@10 Δ itself is flat (+0.025 / +0.030 / +0.024 / +0.026 / +0.025). The low-data hint appears in the AUCs and AP, not at the top of the ranking.
- There are 3 seeds per fraction, f100_s21 stopped at epoch 4, and there is no CI. This should stay descriptive.

**Zero-shot MPNet cosine, text only:** test macro hard 0.786, pooled 0.838, nDCG@10 0.772; validation 0.849 / 0.892 / 0.875. Unlike trials, trained text is about level with zero-shot (+0.011 hard, −0.013 nDCG@10). So zero-shot is a meaningful key secondary.

**Corrections to the user's summary:**
- Only one is needed. The AUC/AP claims rest on the 729–735 anchors with a positive, not on 1,396. Only nDCG@10 and P@10 use about 1,390 anchors.
- P@10 is near its ceiling: the median is 1 positive per anchor, so P@10 ≤ 0.1. It carries little information.

## 6. Recommendations

**R1. Population and cohorts.**
- **Cohort A, "confirmation" (frozen pipeline on already-analysed anchors).** The current split and its 1,396 test anchors.
  - 19 checkpoints (F ∈ {10, 25, 50, 75, 100} × seeds {42, 13, 21}, plus 87 and 123 at F 25 and 100), **all retrained on one GPU host** (trials A1).
  - Training data: masked text from `preprocess/go_withholding/splits`, with new nested masked fractions built by a *new* script. They are prefixes of the aligned masked `train.jsonl`, so fraction membership equals the published one.
  - Config: a derived copy of `lc_go_ppi_baseline.json` differing only in `go_ppi_split_dir`, `go_ppi_converted_dir`, `validation_path` and `embedding_cache_path`. The single-factor audit must report exactly those fields.
  - The 15 published checkpoints enter gate 2 only.
- **Cohort B, "replication" (new models, mostly new anchors).** Option (d) of §4.6.
  - 5-fold protein-disjoint CV over all 15,155 masked/pruned proteins. Identical-view groups are kept together. The fold seed is fixed in the protocol.
  - Per fold k: test = k, inner validation = k + 1, train = the other 3. Seeds 13/21/42 give 15 models.
  - Grade-0 cap 20 per anchor on validation, uncapped on test, as in `data_splitter`.
  - Preregistered sensitivity: `unexposed_closed` = anchors from the current train partition (about 2,694 hard-defined).
  - Reason: it is the closest mirror of trials 2021. Its unexposed subset alone is about 3.3× the hard-defined size of option (a) (815). It keeps 73% of today's training positives, against 51% for (a).
- **Option (b) is dropped because it is infeasible, not by choice** (§4.6). Option (c) becomes the HuRI cohort (R6).

**R2. Primary weight rule: nDCG@10 (macro over anchors), grid W = {0.0, …, 1.0}, ties → smaller w.**
- The published `pooled_auc` rule optimises a *global* cross-anchor AUC, which is not the within-anchor estimand.
- nDCG@10 is within-anchor, is the trials rule, and is an existing code path.
- Keep the full grid: under nDCG, 5 of 15 exploratory weights are 0.6, so the trials 0–0.5 grid would change them and break gate 2.
- The pooled-AUC rule is a preregistered sensitivity.

**R3. Hypotheses.**
- Co-primary, Holm across three per cohort: H1 within-anchor pooled AUC, H2 AP (both for comparability with trials), and **H3 within-anchor hard AUC**.
- H3 is the GO-distinctive claim and the one circularity threatens. Exploratory half-widths (about 0.008–0.011) leave room for Holm(3).
- Secondary, uncorrected: nDCG@10, easy AUC, 1-vs-0 AUC, MRR. P@10 is descriptive only.
- Zero-shot (H1z–H3z) is a key secondary with its own Holm.

**R4. Primary φ = simGIC on the pruned index with pruned record `skill_uris`** (§4.3).
- Report the unpruned φ (the published variant) as a descriptive arm.
- Report **unpruned − pruned** as the paired "annotation-leak contribution", in the same bootstrap.

**R5. Controls**, each with its own validation-selected w and its own validation z-stats, as in trials:
- **C1 exact-term φ**: `similarity_mode="exact"`, pruned URIs.
- **C1b ancestor φ**: descriptive. It separates hierarchy from IC.
- **C3** hierarchy (simGIC) − exact: (i) each at its own selected w; (ii) both at fixed w = 0.5, the modal nDCG-selected weight. Labelled suggestive.
- **C2 shuffled-GO**: the §4.1 decile-stratified, candidate-excluding derangement, seed 11 per file and per decile, 10,000-draw cap. Draw counts are logged.
  - Keep the trials C2 state machine verbatim for comparability (fail if LB95 > 0 or real ≤ shuffled; pass if UB95 < +0.005 and LB95(real − C2) > 0).
  - Preregister that a C2 fail here is read as "gain carried by annotation density/study bias", and that real − C2 is the pair-specific component.
- **C4 study-bias score alone**: φ_bias(a, b) = log(1 + #distinct PMIDs of partner b in the full GAF), all aspects, NOT lines excluded, `GO_REF` excluded. It is fused with text by the same rule.
- **C5 GO beyond bias** (key control for the literature-bias risk): fused = (1 − w_g − w_b)·z_text + w_g·z_simGIC + w_b·z_bias. Select on a 2-D validation grid (w_g, w_b ∈ {0, 0.1, …, 1}, w_g + w_b ≤ 1). Report Δ against text + bias at its own selected w_b.
  - A within-anchor rank metric cannot take a regression covariate, so a competing score is the right form.
  - STRING textmining is excluded as a label leak (§4.5).
- **Mention sensitivity**: exclude the test positives whose unmasked partner-symbol flag was set (§3), descriptive.

**R6. HuRI.**
- Primary HuRI requirement: the **HuRI cohort** (§4.4), scored with the cohort-A models.
  - Positives: HuRI pairs.
  - Negatives: matched HuRI-observed non-edges with no STRING edge.
  - Secondary "hard-like" negatives: STRING-weak partners without a HuRI edge.
  - H1–H3 analogues are reported with 95% CIs, uncorrected. The prediction is a smaller Δ, given simGIC's 0.548 AUC on HuRI.
- Add a descriptive HuRI-supported-positive stratum in cohorts A (58 anchors) and B (337 anchors).

**R7. Gate 2 (reproduction of the published 15, remote CPU, unpruned and unmasked).**
- Use the A2 tolerances from the start: `{"continuous": 1e-5, "rank": 1e-4, "rank_per_query": 1e-3}`.
- `selected_weight` must match exactly **except** in cells whose published top-two gap is < 1e-4. There, either top-two weight passes and the event is logged. The cells are:
  - nDCG rule f10_s42 (0.5 vs 0.7, gap 7.2e-6);
  - pooled rule f10_s42 (3.6e-5) and f75_s21 (6.4e-5).
- Check that the Mac's local reproduction is bit-identical on test leaves, as found here.

**R8. Typed relations.**
- Keep the true-path closure (is_a + part_of, no regulates). This is standard for simGIC and is what was explored.
- An is_a-only rung would need new parser code outside the protected `go_ontology.py`. It is out of scope; state it as a limitation.

**Nothing found here removes a required control.**
- The shuffled control needs the stratified construction (§4.1).
- The HuRI requirement is met by a cohort rather than a stratum (§4.4).
- "Unused raw-STRING proteins" is infeasible and is replaced by the CV (§4.6).

## 7. Exposure added by this investigation (disclose in the protocol §0)

- Re-scored the 15 published checkpoints with unpruned simGIC fusion on validation and test. These were already exposed. New: per-anchor AUC CIs.
- Zero-shot text-only within-anchor metrics on the published validation and test files. No fusion was run.
- **No fusion outcome was computed for any control or new variant.** That covers pruned, exact, ancestor, shuffled, bias and HuRI. The pruned/exact/ancestor φ statistics in §3–4 are label-free.
- Label-level counts only (no scores): HuRI and Lit-BM support by grade on validation and test; textmining by grade; partner GAF references by grade on validation; CV simulations on `pairs.jsonl` with fold seed 2026. The confirmatory fold seed should differ, or the protocol should state that 2026 was used for counts.
- The derangement feasibility check used candidate membership only, not grades.

## 8. Not verified

- The remote x86 BLAS behaviour of the 7.2e-6 nDCG gap. Only the Mac was run.
- The HuRI search space, because the ORFeome list is absent.
- Whether `run_learning_curve_point` needs `validation_path` or `go_ppi_split_dir` alone for the masked validation. `smoke_report_go_ppi.md` shows derived split dirs working, but the exact audit field set was not run.
- The GPU training time for GO on the remote. It is extrapolated from trials.
