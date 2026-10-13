# Preregistration: GO/PPI inference-time fusion, confirmation (cohort A) + replication (cohort B, protein-disjoint CV) + HuRI cohort

Status: **DRAFT v4 (round 3), written before any confirmatory result; nothing is pinned yet.** v4 changes the **primary fusion** to the method chosen by the best-fusion search (`results_go_ppi_fusion_best.md`): per-anchor z fusion of exact pruned BP-term overlap, w by validation global pooled AUC (**AZ-X**, §2.6). The v3 primary (global-z pruned simGIC, w by validation nDCG@10, **GZ-P**) becomes the pre-declared **secondary fusion**. v4 also resolves the round-2 review (CHANGES_REQUESTED: 0 HIGH, 2 MEDIUM, 2 NIT). It resolves every finding of `design_review_go_ppi_fusion.{json,md}` in round 0 (CHANGES_REQUESTED: 0 HIGH, 4 MEDIUM, 7 NIT), round 1 (CHANGES_REQUESTED: 0 HIGH, 1 MEDIUM, 4 NIT) and round 2; the finding-to-section map is §11, and the v3 → v4 change list is §12. The v3 text is kept at `/tmp/goppi_fusion_best/protocol_go_ppi_fusion.v3.md` (sha256 `29f4617e…`) for the reviewer's diff. Template: `protocol_trials_fusion.md` (v7 + §9 A0–A3). Every feasibility number and open-choice decision comes from `findings_go_ppi_fusion.md` (cited as F§n); deviations from it are listed in §10.

- **Paths.** Every path in this file is relative to the repository root `R` unless written absolutely. `R` = `/Users/olukotunjosh/Downloads/CDCL` on the Mac and the remote checkout root (`$R`) on the remote host. Results root: `R/results/go_ppi_fusion_confirm/` (absolute on the Mac: `/Users/olukotunjosh/Downloads/CDCL/results/go_ppi_fusion_confirm`). New derived data: `R/preprocess/go_ppi_fusion_confirm/`. No existing file under `results/`, `preprocess/`, `dataset/`, `embedding_cache/` or `config/` is ever edited or overwritten (§6.1).
- **Repo.** Branch `orca-experiments-and-runner`, HEAD at writing `e69d26f`. `pins.json` records HEAD at pin time.
- **Execution location (fixed).** All gates, data builds, training, scoring and analysis run on **one remote NVIDIA L40S GPU host** (Linux x86_64). `pins.json` (revision 1) is written on that host; `make_pins.py` is the one step-0 script run **without** the `CUDA_VISIBLE_DEVICES=""` mask (it does no encoding or scoring), aborts unless exactly one CUDA device is visible and `torch.cuda.get_device_name(0)` contains `"L40S"`, and records that exact string as `pins.environment.gpu_name` (plus `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` output as `pins.environment.nvidia_smi`, descriptive). Gate 3 (§5.3) requires every training run to report that exact string. **Nothing confirmatory runs on the Mac.** The Mac is used for development, unit tests during development, and a restricted non-confirmatory dev smoke (§6.5) whose outputs go to `results/go_ppi_fusion_confirm_dev/` and are never read by a confirmatory script. Runbook: `.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_GO_PPI_FUSION.md`, written during implementation; where it and this protocol differ, this protocol wins.
- **Device policy (fixed).** Training of the 34 new runs (19 cohort A + 15 cohort B) uses `--device cuda`. Everything else (unit tests, data builds, fresh encodes, gate 2, scoring, analysis) runs on the remote **CPU**: `CUDA_VISIBLE_DEVICES=""`, `--device cpu`, `SentenceTransformer(..., device="cpu")`. The only exception is `make_pins.py` (above), which reads the GPU name and touches no data on the GPU.

## 0. What is already known, what is new, and prior exposure

### 0.1 Exploratory result (post hoc, not preregistered)
Source: `results/publication_phase1_fixedval/go_ppi_fusion_learning_curve{,_ndcg_selected}.json`, re-scored bit-identically in F§5.
- 15 uniform-negative baseline checkpoints (F ∈ {10, 25, 50, 75, 100} × seeds {42, 13, 21}), trained on **unpruned, unmasked** data with no GO in training (`attach.matcher = null`, F§3). 1,396 test anchors; AUC and AP are defined on only 729–735 of them, nDCG@10 on 1,387.
- Fusion = (1 − w)·z(text) + w·z(simGIC, BP, unpruned index, is_a + part_of closure).
- nDCG@10-selected weights 0.4–0.6: within-anchor hard AUC Δ +0.0222 [+0.0118, +0.0330], pooled +0.0253 [+0.0175, +0.0340], AP +0.0256 [+0.0119, +0.0400], nDCG@10 +0.0259 [+0.0216, +0.0305]; 15/15 checkpoints positive on each.
- Global pooled-AUC-selected weights 0.2–0.4: hard +0.0195 [+0.0114, +0.0272], pooled +0.0198, AP +0.0256, nDCG@10 +0.0177.
- Hard Δ by fraction (nDCG rule) +0.027 / +0.025 / +0.025 / +0.019 / +0.015 at F = 10…100; nDCG@10 Δ itself flat.
- Zero-shot MPNet text level is about equal to trained text (hard 0.786 vs 0.797; nDCG@10 0.772 vs 0.760).
- Known threats (F§3, `findings_go_ppi.md`): GAF with/from leak in 5.9% of test positives (0.4% of weak pairs), φ-side only; partner-symbol mentions in 11.1% of test positives, text-side; simGIC separates HuRI edges from matched non-edges at only 0.548 vs 0.674 on Lit-BM (`results/ontology_ceiling/go_independent_signal.json`), consistent with study bias.

### 0.1b Best-fusion search and its test scoring (post hoc, not preregistered; `results_go_ppi_fusion_best.md`)
Same 15 published checkpoints, inference only.
- **Search.** 208 fusion variants were scored on the published validation split only (697 anchors), with anchor-grouped 5-fold cross-fitting and a code guard that blocked every test file. Three finalists were frozen by a fixed rule (`results/go_ppi_fusion_best/finalists.json`, sha256 `49943320…`). Before any test scoring, a protein-disjoint CV re-declared the leak-free headline as **AZ-X** (`test_plan_amendment.json`, sha256 `efe265f5…`).
- **Test, scored once** (10 variants plus diagnostics; `test_results.json`, sha256 `4cb540ab…`). AZ-X Δ vs text:
  - hard AUC +0.0287 [+0.0194, +0.0384], 15/15;
  - pooled AUC +0.0271 [+0.0202, +0.0344];
  - AP +0.0390 [+0.0251, +0.0533];
  - nDCG@10 +0.0155 [+0.0121, +0.0189].
- **AZ-X vs the published form** (paired, same checkpoints): hard +0.0065 [−0.0015, +0.0145], AP +0.0135 [+0.0001, +0.0260], nDCG@10 −0.0104 [−0.0144, −0.0067].
- **GZ-P** (the published form on pruned φ) Δ vs text: hard +0.0226, pooled +0.0249, AP +0.0307, nDCG@10 +0.0244.
- **Diagnostics on AZ-X.**
  - Shuffled GO (the §2.3 derangement, on the published files): validation picks w = 0 for 15/15 checkpoints.
  - simGIC in the same form: hard −0.0084 [−0.0160, −0.0007] and nDCG@10 +0.0038 [+0.0006, +0.0071] vs exact.
  - Unpruned minus pruned: within ±0.0025 on every metric.
- **By training fraction** (hard Δ): +0.034 / +0.031 / +0.029 / +0.026 / +0.024 at F = 10 … 100.
- **Known weakness of the search.** Its validation CV did not predict test magnitudes. The learned stackers that led the CV did not transfer, which is why no learned combiner is carried forward here.

### 0.2 Consequences, fixed now
- **Cohort A is a frozen-pipeline confirmation, not an independent test.** Its 1,396 test anchors and 697 validation anchors have all been scored under fusion. That includes the primary fusion AZ-X itself, which was **chosen on cohort A's validation file and then scored on cohort A's test file** with the published checkpoints (§0.1b). What is new in A: text models retrained on masked text, the controls on retrained models, and the preregistered estimator. A positive cohort-A result therefore shows that the AZ-X gain survives removal of the text-side partner-symbol leak. It is not evidence that AZ-X generalises.
- **Cohort B (5-fold protein-disjoint CV over all 15,155 proteins) is the only replication with new models and new anchors.** About 2,694 of its ≈ 3,850 hard-defined anchors (F§4.6, fold seed 2026; recounted under seed 2027 at build, §3c) come from the current train partition and have never been scored under fusion. **The replication criterion is evaluated only on these clean anchors (`Bclean`, §3c).** The ≈ 30% exposed cohort-B anchors (≈ 778 test-partition + 381 validation-partition, F§4.6) informed every analyst choice in this protocol, so they enter only the `all_B` sensitivity (§4.5).
- **The HuRI cohort (§3d) tests a different label** (literature-independent Y2H interactions). It is secondary and is not a replication of the STRING-grade claim.

### 0.3 Prior exposure, disclosed (mirrors trials §9 A0; nothing below is corrected, all is printed in `report.md`)
1. **Cohort A test and validation anchors.** All 1,396 test anchors (`preprocess/go_ppi_splits/test.jsonl`) and all 697 validation anchors (`preprocess/go_ppi_splits/validation.jsonl`) were scored under unpruned simGIC fusion with the 15 published checkpoints, under both weight rules, in the publication run and again in F§5 (which added per-anchor AUC CIs). Validation selection curves were inspected.
2. **Older probes on the same files.** `scripts/probe_fusion_mechanism.py` → `results/ontology_ceiling/{fusion_probe_go_ppi,fusion_probe_go_ppi_leakfixed,phase1_single_stage_fusion_go_ppi,fusion_reselect}.json`: 3 `lc_single_factor` f100 checkpoints, 6,000-record subsamples of the same validation/test files, hard-AUC selection, unpruned φ.
3. **Zero-shot.** Text-only zero-shot within-anchor metrics on the published (unmasked) validation and test files (F§5). No zero-shot fusion was computed.
4. **Masked text, no fusion.** The GO-withholding study's dev smoke (`results/go_withholding/smoke/*/eval/metrics.json`) scored 2-epoch smoke text models on the masked test file `preprocess/go_withholding/splits/test.jsonl` (sha `4027e9a0…`). Text only; no φ, no fusion.
5. **φ-only ceilings on the whole interactome.** `go_bulk_signal_diagnosis.py` and `go_independent_signal_diagnosis.py` computed unpruned simGIC AUC (no text, no model) over all STRING/HuRI/Lit-BM edges, which includes every protein in cohorts A, B and the HuRI cohort. The HuRI-cohort prediction (§1.4) is informed by the 0.548 result.
6. **Label-level counts (no scores)** (F§4, F§7): HuRI/Lit-BM support by grade on validation and test; STRING textmining by grade; partner GAF reference counts by grade on validation; CV fold simulations on `pairs.jsonl` with fold seed **2026** (the confirmatory fold seed is **2027**, §3c, so no simulated fold is reused); the derangement feasibility check (candidate membership only, no grades).
7. **Label-free φ statistics** (F§3–4): pruned vs unpruned φ Spearman, exact/ancestor/simGIC distributions. **No fusion outcome was computed for any control or new variant** (pruned, exact, ancestor, is_a-only, shuffled, bias, HuRI).
8. **This protocol's own drafting** read file hashes, the pruned-index metadata (`excluded_evidence`, `leak_pruning`), and counted identical masked-view bodies (14 groups, 31 proteins). No score, no grade.
9. **The best-fusion search and its test scoring (§0.1b).** Every variant in §2.2 (P, U, X, N, T-like is_a-only simGIC, ancestor, Resnik, MF/CC), the anchor-z form, the global pooled-AUC rule, the shuffle construction of §2.3 and the exact-vs-simGIC contrast were scored with the 15 published checkpoints:
   - on cohort A's validation file (cross-fitted search, 208 variants);
   - on cohort A's test file (10 variants plus the AZ-X shuffle and simGIC arms; `results/go_ppi_fusion_best/test_results.json`).
   
   This replaces item 7's "no fusion outcome was computed for any control or new variant" for the published checkpoints on cohort A's files. That statement still holds for every retrained model, every cohort-B file and the HuRI cohort. The search read no cohort-B fold, no HuRI file and no train-partition anchor, so `Bclean` (§3c) is unaffected.
10. **Mac dev smoke (§6.5, 2026-10-13; non-confirmatory, outputs in `results/go_ppi_fusion_confirm_dev/`; report `smoke_report_go_ppi_fusion.md`).** Disclosed as trials §9 A0 did; it creates **no new fusion-outcome exposure**.
    - *Gate 2a + 2f code check* (`score.py --gate-reproduce --dev-local --parts 2a,2f`). It scored the 15 published checkpoints on the published validation and test files with U (grid 0–1.0, both rules), AZ-X and GZ-P, plus zero-shot text on the published test file. These are the quantities of items 1, 3 and 9, already reported in `results/publication_phase1_fixedval/go_ppi_fusion_learning_curve{,_ndcg_selected}.json` and `results/go_ppi_fusion_best/test_results.json` (`F3_P`, `PUB_ndcg_P`). They were reproduced, not extended: no other φ variant, control or rule was scored. Nothing was read from any cohort-B file, the HuRI cohort or any retrained checkpoint.
    - *Cohort-B fold structure* (`build_inputs.py --structure-only`; also run once during implementation). It built the fold-seed-2027 folds in memory and wrote only counts, with no record file and no score. The counts are label-level, of the item-6 kind but under the confirmatory seed 2027: records by grade per split and fold, hard-defined anchors per fold, `Bclean` = 4,942 anchors (2,669 hard-defined, ≥ 2,400 holds), `all_B` hard-defined 3,834, list sha256 `e8911581…`. These counts are deterministic and equal what `build_inputs.py` computes and pins in step 0 (§3c). So the statement above that nothing about any cohort-B fold exists yet holds for scores and outcomes, not for these label-level counts.

Nothing about any retrained checkpoint, any cohort-B fold, or the HuRI cohort file exists yet.

## 1. Hypotheses

All effects are paired: Δ = metric(fused score) − metric(text score) on the **same** text model, computed per anchor and macro-averaged over anchors with the metric defined.

**Primary fusion = AZ-X** (§2.6): within-anchor z of the text score and of φ = **X** (exact-term Jaccard of pruned direct BP terms, §2.2), weight w selected on validation by **global pooled AUC**. Every hypothesis, control, label and headline below refers to AZ-X unless it says otherwise. Δ_X denotes its Δ.

**Secondary fusion = GZ-P** (§2.6): validation-file global z of the text score and of φ = **P** (pruned simGIC), w selected on validation by macro nDCG@10. This is the published form on leak-free φ. It is reported in full (§1.2b) and never enters a label.

### 1.1 Co-primary (per cohort, per model type; Holm across the three; family-wise α = 0.05, two-sided; decision rule §4.3)
- **H1**: AZ-X improves within-anchor macro **pooled AUC** (grade 2 vs {0, 1}). Δ_H1 > 0.
- **H2**: AZ-X improves within-anchor macro **AP** (grade 2 relevant). Δ_H2 > 0.
- **H3**: AZ-X improves within-anchor macro **hard AUC** (grade 2 vs grade 1, established vs weak evidence). Δ_H3 > 0.

Reason for three: H1 and H2 keep comparability with the trials study; H3 is the GO-distinctive claim and the one annotation circularity threatens (F§6 R3). Exploratory AZ-X half-widths on the published checkpoints are ≈ 0.007 (pooled), 0.014 (AP) and 0.010 (hard), against points of 0.027, 0.039 and 0.029 (§0.1b), which leaves room for Holm(3). Those points were measured on unmasked-text models and selected after a search. The confirmatory effect is expected to be smaller, and no magnitude is predicted.

The primary text-model population is the **trained baselines** (§3b, §3c). **Zero-shot** (§3a) is a key secondary with its own Holm triple (H1z, H2z, H3z), reported in the same tables, not part of the replication criterion.

### 1.2 Secondary (95% CIs, uncorrected, labelled as such)
- S1 nDCG@10 (gains 2^g − 1). Predicted Δ > 0.
- S2 easy AUC (2 vs 0). Predicted Δ > 0.
- S3 1-vs-0 AUC (weak vs none). Predicted Δ > 0 (GO separates weak-evidence partners from non-partners).
- S4 MRR (grade 2). Predicted Δ > 0.
- S5 P@10: descriptive only (median 1 positive per anchor, so P@10 ≤ 0.1, F§5).
- Global (cross-anchor) AUCs: reported for continuity only, never interpreted.

### 1.2b Secondary fusion GZ-P (pre-declared; reported, never in a label)
- **H1g, H2g, H3g**: the H1–H3 metrics under GZ-P, with their own Holm(3) per cohort and model type, on the same bootstrap draws. Reported next to H1–H3 and marked "secondary fusion".
- **S1g**: nDCG@10 under GZ-P, with a 95% CI and the four-state label.
- **Head-to-head (paired, four-state label, 95% CI):** AZ-X − GZ-P on every H and S metric. The search predicts AZ-X − GZ-P > 0 on H2 and H3 and < 0 on S1 nDCG@10 (§0.1b). Neither direction is a hypothesis test, and neither changes a label.
- Use: if top-10 quality is the target, GZ-P is the pre-declared method. If whole-list quality is the target, AZ-X is. `report.md` prints both rows under the headline.

### 1.3 Data-efficiency hypothesis (secondary, own test, cohort A trained only)
- **DE**: the AZ-X gain on hard AUC is larger at low training fraction. Estimand: slope β_DE of checkpoint-level Δ_H3 (AZ-X) on log10(F/100) (F ∈ {10, 25, 50, 75, 100}), OLS over checkpoints, estimated inside the two-level bootstrap (§4.1, seeds resampled within fraction). **Predicted β_DE < 0** (exploratory AZ-X ≈ −0.010 per decade, §0.1b; ≈ −0.012 for the published nDCG-rule form). β_DE under GZ-P is a descriptive companion.
- Decision: its own two-sided α = 0.05 test, outside the Holm family; four ordered states in §4.4.
- Descriptive companions: β_DE for Δ_H1 and Δ_H2; the endpoint contrast Δ(F = 10) − Δ(F = 100).

### 1.4 HuRI cohort (secondary; cohort-A models, §3d)
H1-HuRI, H2-HuRI, H3-HuRI: the same three metrics under AZ-X, with HuRI Y2H partners as grade 2. 95% CIs, uncorrected, four-state label (§4.2). GZ-P on HuRI is descriptive. **Predicted: Δ > 0 but smaller than in cohort A** (φ-only simGIC AUC 0.548 on HuRI; no φ-only HuRI AUC is on record for exact-term φ). A HuRI Δ equivalent to zero alongside a positive cohort-A Δ is read as "the STRING-label gain is carried by study bias".

### 1.5 Controls (definitions §2.3; states §4.4; how each enters labels §4.3)
Every control is computed **in the primary fusion form** (anchor-z, global pooled-AUC rule, own validation-selected w), so it differs from AZ-X only in φ or in the added score.

| id | question | φ / construction (all in AZ form) | role |
|---|---|---|---|
| C1 | Does GO hierarchy + IC help beyond shared terms? | P − X (pruned simGIC vs exact-term Jaccard). Search prediction: < 0 on H1–H3, > 0 on nDCG@10 (§0.1b) | report-only state |
| C1a | Ladder: hierarchy vs IC | N (ancestor Jaccard, no IC): P − N (IC), N − X (hierarchy) | descriptive |
| C1t | Does `part_of` matter (typed relations)? | P − T (simGIC over is_a-only closure, IC recomputed) | secondary, suggestive |
| C2 | Does the gain need each anchor's own annotations? | S: X on the decile-stratified, candidate-excluding anchor-side derangement (§2.3) | **gates the final label** |
| C3 | How much of the gain is with/from annotation leak? | UX − X (UX = exact-term Jaccard on the unpruned index and unpruned URIs) | report-only state |
| C4 | How much does study bias alone give? | Bz: log(1 + partner PMID count) | descriptive |
| C5 | Does GO add beyond study bias? | text + X + Bz (2-D grid) vs text + Bz | **mandatory qualifier on the label** |
| C6 | Does the gain hold on literature-independent labels? | HuRI cohort (§1.4) | **mandatory qualifier on the label** |
| ZS | Is the gain specific to a weak trained base? | zero-shot text + X | key secondary (H1z–H3z) |

GZ-P (§1.2b) gets the GZ versions of C2 (S_P: simGIC on the same derangement) and C3 (U − P), descriptive only.

**Model-type scope.** C1–C5 are computed for trained and zero-shot models in cohorts A and B. Only the **trained** C2 and C5 states enter labels (§4.3); zero-shot control states are descriptive.

**Interpretive rules fixed now.**
- **C2 is an anchor-side shuffle.** It keeps the partner side of φ and, through decile stratification, the anchor's annotation density. It removes only pair-specific information.
  - Partner annotation density rises with grade (median partner GAF references 13 / 20 / 38 at grades 0 / 1 / 2, F§4.1), so **Δ_S > 0 is possible on GO**, unlike MeSH.
  - In the exploratory run, AZ-X on the shuffle selected w = 0 for 15/15 published checkpoints (Δ_S ≡ 0, §0.1b). Within-anchor z of an exact-term overlap may therefore leave little density signal, but this is not assumed.
- **The C2 state gates on the pair-specific component Δ_X − Δ_S, not on Δ_S alone (§4.3).**
  - *fail* = no confident pair-specific component, either because the shuffle matches or beats the real gain or because the shuffle keeps a confident share of it. It also fires on a confidently small pair-specific component (UB95(X − S) < 0.005), **but only when** the shuffle has kept a confident share of the gain (LB95(Δ_S) > 0). A small gain that the shuffle removes is never read as non-specific.
  - A *fail* is read as "the gain is carried by annotation density / study bias, not by the anchor's specific annotations", not as a code defect.
  - *partial* = a confident pair-specific component **and** a confident density component. It does not block the label but is printed in it.
  - The trials-verbatim C2 state is still computed and recorded as `c2_trials_state` for cross-domain comparison.
  - X − S is printed with every label.
- C5 is the direct study-bias control. A rank metric within an anchor cannot take a regression covariate, so the covariate enters as a competing fused score (F§6 R5).
- STRING `textmining` is never used as a covariate: grade 0 ≡ 0 by construction, so it is a label leak (F§4.5).
- Fusion is an inference-time scoring rule. Nothing here supports a claim about using GO at training time.

## 2. Scores, φ variants and fusion rule

### 2.1 Text scores and embeddings
- **Trained text score.** As `epf.text_scores` (`scripts/evaluate_publication_fusion.py`, `epf`): frozen `all-mpnet-base-v2` 768-d vector → checkpoint projection head → dot product of 128-d projections. CPU.
- **Zero-shot text score.** Cosine of the frozen 768-d vectors, no projection.
- **Serialisation and records.** As `epf.prepare_split`: text = `run_phase1_embedding_evaluation.content_to_text` on `encoder_view`, verbatim; grade = `job.grade` with `original_label` fallback; query = `metadata.resume_id` (anchor); blank text raises.
- **Read-only loader.** A port of `epf.prepare_split` with no encoder construction for published files and no `mkdir`. `epf.prepare_split` is never called (it builds a device-less `SentenceTransformer`, which would pick MPS on the Mac, and can write chunks into the published cache dir; F§1.1). `embedding_dim` is set to 768 and asserted against every loaded vector width; mismatch aborts.
- **Embedding sources, by use:**
  1. **Gate 2 and the A-pub arm (published unmasked files only).** Order as `epf`: (a) frozen cache `embedding_cache/go_ppi_text_embeddings.pt` by `EmbeddingCache.get_content_key(slot)`; (b) published prepared chunks by sha256(text): `results/publication_phase1_fixedval/prepared_embeddings/go_ppi/test_c65a1d2ca0e5c16a_sentence-transformers_all-mpnet-base-v2/` (5 chunks, manifest digest `42abdb8c3765e6963ab552d019a93cf29a08e87e473b8fb61ae2e056a1fbe5b4`) and `…/validation_07e432fb64f3e271_sentence-transformers_all-mpnet-base-v2/` (2 chunks, digest `cefa1e04cb2be64ba6222b7c7fe2907565422a8bfd258a5250b6e28d9e189063`). A text in neither source is a hard abort; no encoder is constructed. Manifest digest rule as trials §2.1: sha256 of the UTF-8 text of sorted `"<sha256>  <basename>\n"` lines over `chunk_*.npz` (= `shasum -a 256 chunk_*.npz | sort -k2 | shasum -a 256` with cwd = the chunk dir).
  2. **Every confirmatory file (masked views: cohort A, cohort B folds, HuRI cohort).** One fresh encode of every distinct masked text on the remote CPU, in step 0 (§6.2, label-free), by `scripts/go_ppi_fusion_confirm/encode.py`: `SentenceTransformer(<resolved local snapshot path>, device="cpu")`, `batch_size=64`, `convert_to_numpy=True`, no normalisation, written in `epf` chunk format to `results/go_ppi_fusion_confirm/prepared_embeddings/go_ppi_masked/`, keyed by sha256(text). Its manifest digest is pinned. Scoring reads only this directory for masked texts (a missing text aborts). Reason: one device and one source for every confirmatory score; the masked caches on disk are Mac-encoded and incomplete (740 of 1,457 masked validation views, F§1.4).
- **MPNet snapshot.** Revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, resolved with `HF_HUB_OFFLINE=1` by `huggingface_hub.snapshot_download(..., local_files_only=True)`; the directory basename must equal the revision or the script aborts (trials §2.1). The **full** snapshot is transferred (trials §9 A3 lesson); `verify_transfer.py` also performs a CPU load test (construct, encode one string, assert width 768).
- Every original cache file is opened read-only and its sha256 is checked before and after each script that reads it.

### 2.2 φ variants
All variants use BP (aspect P). φ(a, b) = `GoMatcher(index, similarity_mode=…).ontology_set_similarity(anchor.skill_uris, partner.skill_uris)` (`go_ppi_domain/go_ontology.py`, imported, not edited). Empty side → 0.0. Indexes are loaded with **`GoIndex.load`** only (never `build_or_load`, which can rebuild and write the cache, F§1.1).

| id | index (`GoIndex.load`) | record URIs | mode | notes |
|---|---|---|---|---|
| id | index (`GoIndex.load`) | record URIs | mode | notes |
|---|---|---|---|---|
| **X** (primary, AZ-X) | `preprocess/go_withholding/converted/go_index_P.pkl` (pruned) | pruned `skill_uris` | `exact` | \|A ∩ B\| / \|A ∪ B\| over direct BP terms; no hierarchy, no IC. Pruned index: with/from-human-protein lines dropped; IPI, IEA, ND, NOT, `GO:0005515`, `GO:0005488` excluded; `GO:0005515` descendants excluded (0 BP lines affected: it is an MF term, so this exclusion is vacuous in BP and is stated, not claimed as a safeguard); IC over 17,648 genes (used by P, not by X) |
| **P** (secondary, GZ-P) | pruned | pruned | `simgic` | IC-weighted Jaccard over is_a + part_of closures |
| U (published) | `preprocess/go_ppi/go_index_P.pkl` (unpruned) | unpruned `skill_uris` | `simgic` | the published φ; same evidence/term exclusions, no with/from pruning; IC over 19,293 genes |
| UX | unpruned | unpruned | `exact` | C3 for the primary fusion |
| N | pruned | pruned | `ancestor` | Jaccard of is_a + part_of closures, no IC |
| T | pruned-derived is_a-only index (§2.2.1) | pruned | `simgic` | closure over `is_a` only, IC recomputed |
| **S** | pruned | anchor side deranged (§2.3) | `exact` | C2 for the primary fusion |
| S_P | pruned | anchor side deranged (§2.3), same maps as S | `simgic` | C2 for GZ-P, descriptive |
| Bz | — | — | — | log(1 + n_PMID(partner)) (§2.4) |

- **Pruned and unpruned are never mixed.** X/P/N/T/S/S_P use the pruned index *and* pruned record URIs together; U/UX use both unpruned (F§4.3).
- **Record URIs by file.** Cohort A masked files `preprocess/go_withholding/splits/{validation,test}.jsonl` carry pruned URIs. For U on cohort A, the unpruned URIs are taken from the **aligned** published files `preprocess/go_ppi_splits/{validation,test}.jsonl` (record-for-record alignment asserted, §5.2b). For cohort B and the HuRI cohort, record URIs come from `preprocess/go_withholding/converted/proteins.jsonl` (`go_uris`, pruned; asserted equal to the pruned index `direct` for all 15,155 proteins, which holds at writing) and, for U, from `preprocess/go_ppi/proteins.jsonl` (`go_uris`, unpruned).
- **Effect of pruning, label-free (F§3).** φ changes for 20,876 of 22,341 test records (IC is recomputed); Spearman(U, P) = 0.836 on test; 42 test anchors (661 records) and 250 partners lose every BP term, so P = X = 0 for them. The U → P figures are for simGIC. The UX → X change count is not on record and is computed in step 0 (label-free, `build_manifest.json`). Pruning is global (any annotation whose with/from names any human protein), so it is conservative.

#### 2.2.1 Typed variant T (is_a only)
`go_ontology.parse_obo` follows is_a + part_of and is protected. A new function `parse_obo_typed(path, relations)` in `go_ppi_domain/fusion_confirm.py` re-implements the same line rules (`is_a: GO:`, `relationship: part_of GO:`, `alt_id`, `is_obsolete: true`, namespace) with a `relations` argument ∈ {`("is_a",)`, `("is_a", "part_of")`}.
- T's index object is `GoIndex(ancestors_isa, depth_isa, direct, closure_isa, coarse_isa, ic_isa, meta)` built in memory from: the pruned index's `direct` (gene → mapped direct BP terms, unchanged); ancestors over BP terms using is_a only, by the same iterative closure as `GoIndex.build`; closure per gene = union of ancestors of its direct terms; `ic_isa[t] = −log(freq_isa[t]/total)`, with total = number of genes with non-empty closure (= 17,648, unchanged because every mapped term has itself in its closure).
- **Validity check (gate 2c, label-free, step 0).** With `relations = ("is_a", "part_of")` the same code must reproduce the pruned index's `ancestors` for every BP term exactly and its `ic` for every term exactly (float equality; same formula, same counts). If either fails, T is **not computed**, C1t is reported as "not computed (typed-parser check failed)", and nothing else changes; no amendment is needed. If it passes, `ic_isa` and `ancestors_isa` are written once to `results/go_ppi_fusion_confirm/phi/go_index_P_isa_only.pkl` (leaf must not exist) and pinned.
- Reason for including T although F§6 R8 called it out of scope: the user asked for the typed contrast; the only obstacle F§6 R8 names is that parser code must live outside the protected file, which this does, and the validity check makes it safe. It stays secondary and suggestive.

### 2.3 Shuffled-GO control S (C2)
The trials whole-file rejection derangement is **infeasible on GO** (0 of 2,000 draws accepted, F§4.1), so S uses the stratified construction F§4.1 found feasible, with one added forbidden condition (partner-set equality) that makes "no anchor gets its own or its partner's set" literal.

Applied separately to each scored file (every validation and test file of every cohort; the HuRI cohort file too):
1. **Units.** The file's distinct anchors with a **non-empty** pruned anchor-side URI set. Anchors with an empty set are excluded from the shuffle and keep the empty set (their φ is 0 under X, P, S and S_P alike; otherwise all empty-set anchors would be mutually forbidden by rule (c) below).
2. **Strata.** Sort the units by (|pruned BP closure of the anchor set|, anchor_id) and split them into 10 contiguous strata with `np.array_split` (closure-size deciles).
3. **Forbidden donors.** A donor d may not give its set to receiver a if: (a) d = a; (b) d is a candidate of a in this file (any record with anchor a and partner d, any grade); (c) set(d) = set(a); (d) set(d) equals the partner-side URI set of any candidate of a in this file; (e) simGIC_P(set(a), set(d)) ≥ 0.5.
4. **Draws.** One `rng = np.random.default_rng(11)` per file, consumed stratum by stratum in stratum order 0..9. Within a stratum of size n: repeat `order = rng.permutation(n)` until no i has order[(i + 1) % n] forbidden for order[i]; then unit order[i] receives the anchor-side set of unit order[(i + 1) % n]. Cap 10,000 draws per stratum; exceeding it is a hard abort. Draw counts per stratum are written to the shuffle manifest, together with a descriptive count (not a forbidden rule) of receiver–candidate pairs whose donor set has simGIC_P(set(d), partner set of the candidate) ≥ 0.5. Such pairs are random with respect to grade, so they add noise, not bias.
5. Partner-side URIs are unchanged; every record of a receiver gets the receiver's new anchor-side set.
- **The map does not depend on the φ mode.** One map per file defines both S (exact) and S_P (simGIC). Rule (e) uses simGIC_P in both cases.
- **When it is built.** Shuffle maps depend only on file membership and URIs (no grade, no score), so `build_inputs.py` builds every map in step 0, before pins. An abort is therefore resolved by a pre-pins amendment of this section with no exposure. Maps are written to `results/go_ppi_fusion_confirm/phi/shuffle_maps/{file_id}.json` and pinned.
- **Feasibility.** Under the pruned sets with rules (a)–(e), the construction ran on the published cohort-A files in the best-fusion study:
  - validation: 680 units in 10 strata of 68, first accepted draw 1–9 per stratum;
  - test: 1,354 units in strata of 135–136, first accepted draw 1–69 per stratum.
  
  Feasibility on cohort-B folds and the HuRI file is unverified (an abort there is a pre-pins amendment).
- C2 uses its own validation-selected w (§2.6), fit on the deranged validation file. Global-z arms (S_P) also use their own deranged-validation z-stats.

### 2.4 Study-bias score Bz (C4, C5)
- n_PMID(g) = number of distinct `PMID:` tokens in column 6 (`DB:Reference`, `|`-separated) over all lines of `dataset/go_ppi/bulk/goa_human.gaf.gz` whose column 3 symbol is g, **all aspects and all evidence codes**, excluding lines whose qualifier (column 4) contains `NOT`. `GO_REF:` and other non-PMID references are not counted. Symbols are matched to `protein_id` exactly; a protein with no line gets 0.
- Bz(a, b) = log(1 + n_PMID(b)). Partner-only by design: within an anchor, an anchor-side term is constant and cannot change a rank.
- Anchor-level study-bias covariate (for stratification, §4.5): n_PMID(a).
- Written once in step 0 to `results/go_ppi_fusion_confirm/phi/n_pmid.json` (label-free) and pinned.
- Disclosed property: n_PMID counts IPI lines too, so a partner's count can include the paper that reported the very interaction. This is the study-bias signal C5 is meant to absorb; it is protein-level, not pair-level.

### 2.5 Normalisation
- **AZ (within-anchor z; primary form).** For each anchor, the mean and population SD (ddof 0) over that anchor's records in the scored file. z = (x − mean) / SD, with SD 0 → SD 1, so a constant score gives z = 0. This is `go_ppi_fusion.fusion.within_query_z`. Nothing is fitted, so the same rule applies unchanged to validation, test and the HuRI file, and no statistics are carried across files.
- **GZ (validation-file global z; secondary form, as implemented in `epf`).** Mean and population SD (ddof 0; SD 0 → 1) over the whole validation file, applied unchanged to the test file (`epf.z_stats`, `epf.z_apply`). φ stats per variant per validation file; text stats per text model per validation file. The HuRI cohort uses cohort A's validation stats (no HuRI validation set): for S_P the stats of cohort-A's deranged validation file.
- **Sensitivity:** X under GZ with the global pooled-AUC rule (GZ-X), which separates the normaliser from the φ change.

### 2.6 Fusion and weight selection
- **AZ-X (primary):** score = (1 − w)·z_a(text) + w·z_a(X).
- **GZ-P (secondary):** score = (1 − w)·z(text) + w·z(P), i.e. the published form with φ = P.
- **Controls** use the form of the fusion they control (AZ for AZ-X, GZ for GZ-P) and differ only in φ or the added score (§1.5).
- **Weight selection.** For each φ variant v, each text model m, each cohort or fold: compute v on validation and test. Under GZ, fit z-stats on validation only. Select w on validation over **W = [i/10 for i in range(11)] = {0.0, …, 1.0}**:
  - **AZ forms: global pooled AUC.** Tie-aware rank AUC of grade 2 vs {0, 1} over all validation records pooled across anchors. This is `go_ppi_fusion.metrics.global_auc`, the `epf` `--select-on pooled_auc` rule.
  - **GZ forms: macro nDCG@10** (`epf.ranking_metrics(...)["macro"]["ndcg@10"]`, gains 2^g − 1, anchors with NaN dropped).
  - **Ties** (exact float64 equality) → the smaller w (first maximiser over ascending W, as `epf.main`). w = 0 is allowed (Δ = 0, kept in the average).
- **Why each form keeps its own rule.** Each rule is the one the method was selected and test-scored with (§0.1b): AZ-X with global pooled AUC, the published nDCG form with nDCG@10. Changing a rule would turn either method into a new, unscored one.
  - The known objection to the pooled-AUC rule (F§6 R2) is that it maximises a *global*, cross-anchor AUC rather than the within-anchor estimand. Under AZ, every anchor's scores are centred and scaled, so the global AUC is computed on comparable within-anchor scales. It remains a different quantity from the within-anchor estimand, and that is disclosed (§8).
  - **Sensitivity:** each form under the other rule (AZ-X by nDCG@10; GZ-P by global pooled AUC), same grid and tie rule.
- **Why the full 0–1.0 grid.** 5 of 15 exploratory nDCG-selected weights are 0.6, so the trials 0–0.5 grid would change them and break gate 2.
- **C5 (2-D, AZ).** score = (1 − w_g − w_b)·z_a(text) + w_g·z_a(X) + w_b·z_a(Bz) over w_g, w_b ∈ W with w_g + w_b ≤ 1 + 1e-12 (66 points).
  - Selection by validation global pooled AUC; ties → smaller w_g, then smaller w_b.
  - The comparator text + Bz (AZ) uses its own 1-D selected w_b.
  - Δ_C5 = metric(text + X + Bz) − metric(text + Bz).
  - Bz is partner-only, so z_a(Bz) varies within an anchor exactly as Bz does.
- **C1 fixed-weight form (ii).** X and P, both in AZ form at fixed w = 0.4, the modal exploratory AZ-X weight (12 of 15 checkpoints, §0.1b).
- **Near ties.** The near-tie lesson: published nDCG validation top-two gaps go down to 7.2e-6 (F§6 R7). Confirmatory selection runs once on one host, so a near-tie only means a weight is chosen by that host's floats. The selected w and the validation top-two gap are logged per model × variant × rule in `selected_w.json`.

### 2.7 Metrics per anchor
Per anchor (`metadata.resume_id`), over all its records in the file (test uncapped), with tie-aware rank AUC (`epf.rank_auc`):
- pooled AUC: 2 vs {0, 1}; hard AUC: 2 vs 1; easy AUC: 2 vs 0; 1-vs-0 AUC: 1 vs 0;
- AP and RR (grade 2 relevant), nDCG@10 (gains 2^g − 1), P@10: `epf.ranking_metrics`.
- **Undefined metrics.** An anchor missing a class is excluded from that metric only. The undefined set does not depend on the score, so pairing holds.
- Anchors with an empty pruned anchor-side BP set (X ≡ P ≡ 0 for all their records) stay in. Under AZ, z_a(X) ≡ 0, so their fused ranking is the text ranking and Δ = 0. Under GZ, Δ = 0 as well, because a constant shift does not change a within-anchor rank. A sensitivity excludes them (§4.5).

## 3. Text models, cohorts and files

### 3a. Zero-shot
Frozen MPNet cosine on the masked views. One deterministic function: 1 model in cohort A (with cohort A's validation file for w), one per fold in cohort B (with that fold's validation file), and cohort A's in the HuRI cohort.

### 3b. Trained baselines, cohort A (19 checkpoints, all retrained)
- **Base config:** `config/lc_go_ppi_baseline.json` (uniform graded negatives, InfoNCE, 15 epochs, no GO matcher attached), run by `scripts/run_learning_curve_point.py --domain go_ppi`. `best_checkpoint.pt` = lowest validation loss (fixed validation negatives, seed 1729).
- **Cells:** F ∈ {10, 25, 50, 75, 100} × S ∈ {42, 13, 21}, plus S ∈ {87, 123} at F 25 and 100 (19 runs, the trials 15 + 4 design, F§6 R1). All 19 are **trained on the pinned L40S with `--device cuda`** (trials §9 A1 decision applied from the start). The 15 published checkpoints enter **gate 2 only** (and the descriptive A-pub arm, §4.5).
- **Why retrain.** The published 15 were trained on unmasked text that names the partner in 11.1% of test positives (F§3); that leak is inside the text model and is removed only by retraining on the masked `encoder_view`.
- **Training data (masked, aligned).** `preprocess/go_withholding/splits/train.jsonl` is record-for-record aligned with the published train (F§3). `build_inputs.py` (step 0) writes `preprocess/go_ppi_fusion_confirm/cohortA/frac_{F}/train.jsonl`: for each row of the published `preprocess/go_ppi_lc/frac_{F}/train.jsonl`, in that row order, the masked row with the same (`metadata.resume_id`, `metadata.partner_id`) (key uniqueness asserted; row counts 1,571 / 3,927 / 7,854 / 11,781 / 15,708 asserted). So fraction membership and order equal the published ones. `validation_positive.jsonl` = the grade-2 rows of `preprocess/go_withholding/splits/validation.jsonl`, in the order of the published `preprocess/go_ppi_lc/validation_positive.jsonl` (same key matching; 321 rows asserted).
- **Derived config** `results/go_ppi_fusion_confirm/configs/lc_go_ppi_baseline_masked.json`, differing from the base **only** in:
  - `go_ppi_split_dir` → `preprocess/go_withholding/splits` (masked pools; its `negative_pools.jsonl` is byte-identical to the published one, sha `2e4fc52b…`);
  - `go_ppi_converted_dir` → `preprocess/go_withholding/converted` (masked partner views for the negative selector);
  - `embedding_cache_path` → `results/go_ppi_fusion_confirm/cache/go_withholding_text_embeddings.pt`, a `cp -n` byte copy of `embedding_cache/go_withholding_text_embeddings.pt` (sha `3aed5148…`; starting state pinned as `inputs["…@pin_time"]`; a mutable training cache only; scoring never reads it).
- **Single-factor audit.** Each run passes `--compare-config config/lc_go_ppi_baseline.json`; gate 3 requires `single_factor_audit.unexpected_fields == ["embedding_cache_path", "go_ppi_converted_dir", "go_ppi_split_dir"]` (sorted). `validation_path` is set by `--validation-file` at run time, not in the config, so it is not in the audit. F§8 flagged this set as unverified: `build_inputs.py` runs `report_factor` on the derived configs in step 0 (label-free) and aborts if the set differs, which is resolved by a pre-pins amendment.
- **Run order (fixed, part of the condition because the cache copy is appended to):** F 10, 25, 50, 75, 100; within F seeds 42, 13, 21, then 87, 123 at F 25 and 100; then cohort B k = 0..4, seeds 13, 21, 42. Ledger logs cache entry count and sha256 before/after each run.
- **Files:** validation for w = `preprocess/go_withholding/splits/validation.jsonl` (5,587 records, 697 anchors, grade 0 capped at 20); test = `preprocess/go_withholding/splits/test.jsonl` (22,341 records, 1,396 anchors, uncapped).
- Equal weight per checkpoint (5 seeds at F 25 and 100, 3 elsewhere); fraction-stratified bootstrap as sensitivity (§4.1).

**Published 15 (gate 2 and A-pub only).** Path `results/publication_phase1_fixedval/checkpoints/go_ppi_baseline_f{F}_s{S}/best_checkpoint.pt`; config `config/lc_go_ppi_baseline.json`; train `preprocess/go_ppi_lc/frac_{F}/train.jsonl`; validation `preprocess/go_ppi_lc/validation_positive.jsonl`; device auto on the Mac.

| F | S | best_checkpoint.pt sha256 |
|---|---|---|
| 10 | 42 | f42e599e54f554284bf5cce18eb06540d1b526c92a7938108f62cbeb7ec56f08 |
| 10 | 13 | e1dc9a17c5957acd2b6164af1e09e64868ed8a1cd28604296303882dd7cec110 |
| 10 | 21 | 46b091f470012581d4181e23db495deec3daae83f7ad75cde721059db188ff5d |
| 25 | 42 | 7b1721294a1200684718f9a827919275a933bccbe5df22a58f2039b1821739ac |
| 25 | 13 | 49c551d6e6877b249d8adc4f65567c42297297c14b35fe949228c8ad424d927e |
| 25 | 21 | bf494d05c4d3a27e8ffabdb5a0343993cdb0d433dcbb579afb6ebbdfdc5d5282 |
| 50 | 42 | 59c82d28a342a4604c39faf131a9ef42029ec92b03bbd08cd170660519e14dff |
| 50 | 13 | 65dfaefab157353ca97652798e07fd21718646087b7dd037260c2515f8d8e833 |
| 50 | 21 | 1449bc54ea503c794fcf4233e5aab2ac810814e82361f49032a0dfdf68506f31 |
| 75 | 42 | 9d5cd47647a6d2dc171e516c0b0f8024a1321accaa8a8bf0b58dada731fed121 |
| 75 | 13 | f3d328512e45e537322b740ef04d5fac5a2ca7b754588f2e1e1ccccdf779c0bc |
| 75 | 21 | 784093247973b9f310ed4408ef78334932d13818d090d9076025e27d2d91f68e |
| 100 | 42 | ae1516bfed20880fc38064c595727906235ee2936c376754e683c1b1d24e1d82 |
| 100 | 13 | 484e4ea387633af521f4f7079dec1119ccd0624f67bf7c6731a4c4fb1829074d |
| 100 | 21 | 70e53518b36551d8f1531d27321f95d4c46ffd2f2dc69b0adf74ad4998856708 |

### 3c. Cohort B: 5-fold protein-disjoint CV replication (15 checkpoints + 5 zero-shot)
Chosen replication option: F§4.6 option (d). Option (b) "unused raw-STRING proteins" is **infeasible** (0 new anchor-eligible test-partition proteins; the 3,735 unused STRING genes have no BP annotation after the guards, F§4.6), so it is not used. Option (a) is rejected because it halves training data and gives only 815 hard-defined anchors; option (c) is used as the HuRI cohort (§3d).

- **Universe.** All 15,155 proteins of `preprocess/go_withholding/converted/proteins.jsonl` (masked views, pruned `go_uris`) and all 572,637 graded pairs of `preprocess/go_withholding/converted/pairs.jsonl` (byte-identical to `preprocess/go_ppi/pairs.jsonl`; 8 positives per anchor cap already applied by the converter).
- **Identical-view groups.** Two proteins are grouped iff their masked `encoder_view` bodies (text after the first `" — "`) are identical. Expected: 14 groups covering 31 proteins, max size 4 (counted label-free while drafting); `build_inputs.py` asserts this and stops (pre-pins amendment) otherwise. Every other protein is a singleton group.
- **Fold assignment (fold seed 2027; ≠ 2026 used for the F§4.6 counts).** Order groups by their smallest `protein_id`; `random.Random(2027).shuffle(groups)`; assign each group in turn to the fold with the fewest proteins so far (ties → lowest index). 5 folds.
- **Per outer fold k ∈ {0..4}:** test = fold k; inner validation = fold (k + 1) mod 5; train = the other three folds (one partition). A pair is kept only if both endpoints are in the same partition (as `data_splitter` `protein_disjoint`); cross-partition pairs are dropped and counted.
- **Records** via `go_ppi_domain.data_splitter._slot` / `_build_record` (imported, not edited). Grade plan as `data_splitter`: train grade 2 only; validation all grades, grade 0 capped at 20 per anchor (`sorted(random.Random(f"2027|fold{k}|validation|{anchor_id}|0").sample(sorted_ids, 20))` when > 20); test all grades, uncapped. Within each file rows are generated in (anchor_id, grade 2 → 1 → 0, partner_id) order, then shuffled with `random.Random(f"2027|fold{k}|{split}")`.
- **Files** in `preprocess/go_ppi_fusion_confirm/cv/fold_{k}/`: `train.jsonl`, `validation.jsonl`, `validation_positive.jsonl` (grade-2 rows of validation, same order), `test.jsonl`, `negative_pools.jsonl` (train and validation anchors only; `weak_evidence` / `no_interaction` sorted lists, as `data_splitter`), `fold_manifest.json` (proteins, anchors, hard-defined anchors, records by grade per split, dropped pairs, sha256 of every file).
- **Asserts:** protein-disjoint across the three partitions in every role; every view group inside one fold; two independent builds give identical sha256 for every file.
- **Training:** derived config per fold `results/go_ppi_fusion_confirm/configs/lc_go_ppi_baseline_cv_fold{k}.json` = the cohort-A derived config with `go_ppi_split_dir` → `preprocess/go_ppi_fusion_confirm/cv/fold_{k}` (same three unexpected fields). Seeds 13, 21, 42; `--fraction 1.0`. Output `results/go_ppi_fusion_confirm/cohortB/fold_{k}/go_ppi_baseline_full_s{S}/`.
- **Exposure sets (pinned, label-free, by protein id).** `exposed` = every protein appearing in any role in `preprocess/go_ppi_splits/{validation,test}.jsonl` (1,457 + 3,014 proteins; their anchors were scored or used for weight selection under fusion, §0.3). `exposed_closed` = the union of view groups containing an `exposed` protein. **`Bclean`** (= `unexposed_closed` anchors) = cohort-B test anchors (over all five test folds) not in `exposed_closed` (expected ≈ 2,694 hard-defined, F§4.6; the exact count, the anchor list and its sha256 are computed at build and pinned as `derived.bclean`). **Build assert (outcome-free, uses only grade membership as F§4.6 did):** the hard-defined `Bclean` count under fold seed 2027 must be **≥ 2,400** (`bclean_min_hard_defined`); otherwise `build_inputs.py` stops and the population choice (§0.2, §10) is revisited by a pre-pins amendment. No upper bound applies. **`Bclean` is the primary cohort-B population** for H1–H3, H1z–H3z, C2_B and C5_B (§4.3). All of each Bclean anchor's test records are kept, including records whose partner is in `exposed`; the number of such records and anchors is counted at build (`fold_manifest.json`, `bclean_records_with_exposed_partner`) and reported descriptively (§8).
- Training data size: about 11,505 directed grade-2 train records per fold, 73% of the current 15,708 (F§4.6).

### 3d. HuRI cohort (secondary; cohort-A models; test partition only)
- **Universe.** The 3,014 proteins of the current test partition (entities of `preprocess/go_ppi_splits/test.jsonl`), which no cohort-A model has seen in training. Masked views and pruned URIs from `preprocess/go_withholding/converted/proteins.jsonl`.
- **Mapping.** `dataset/go_ppi/independent/HuRI.tsv` Ensembl gene ids → symbols through `dataset/go_ppi/bulk/9606.protein.aliases.v12.0.txt.gz` (`Ensembl_gene` source) and `9606.protein.info.v12.0.txt.gz`, ties resolved as `scripts/go_independent_signal_diagnosis.load_ensembl_gene_map` (richest pruned BP closure, then symbol). Self-edges dropped; undirected.
- **Anchors:** test-partition proteins with ≥ 1 HuRI partner in the partition (expected 769, F§4.4; asserted within ±5, otherwise stop and amend pre-pins).
- **Grade 2:** every HuRI partner of the anchor inside the partition (expected 2,860 directed pairs).
- **Grade 0 (matched non-edges):** "HuRI-observed" = protein with ≥ 1 mapped HuRI edge anywhere. Pool for a positive partner p: HuRI-observed test-partition proteins in the same (HuRI-degree decile, pruned-BP-closure-size decile) cell as p (`decile_bins` rule of `go_independent_signal_diagnosis.py`, computed over all HuRI-observed eligible proteins), excluding the anchor, its HuRI partners, and any protein with a row of any channel to the anchor in `9606.protein.links.detailed.v12.0.txt.gz`. Per anchor, `rng = random.Random(f"2027|huri|{anchor_id}")`; for i = 0..19, p = positives[i mod n_pos] (positives sorted), draw up to 200 times from p's pool without replacement across i; unmatched draws are counted and skipped. Target 20 grade-0 per anchor.
- **Grade 1 ("hard-like"):** STRING weak-evidence partners of the anchor in the partition (grade 1 in `pairs.jsonl`) that are **HuRI-observed** and have no HuRI edge to the anchor. So all three grades are drawn from HuRI-observed proteins, and H3-HuRI contrasts Y2H interaction against weak evidence among screened-proxy proteins, not observed against never-observed proteins.
- **Sensitivities:** `huri_new_labels` = drop HuRI positives that are STRING grade-2 pairs (keeps ≈ 2,782 of 2,860). `huri_grade1_unrestricted` = grade 1 = every STRING weak-evidence partner in the partition with no HuRI edge to the anchor (the v1 definition), written as a second file `preprocess/go_ppi_fusion_confirm/huri/test_grade1_unrestricted.jsonl` with the same grade-2 and grade-0 rows. Both grade-1 record counts and the hard-defined anchor counts under each definition are pinned (`derived.huri.grade1_counts`) and asserted in gate 2d.
- File `preprocess/go_ppi_fusion_confirm/huri/test.jsonl` (records via `_build_record`, rows ordered then shuffled with `random.Random("2027|huri|test")`) plus `huri_manifest.json`. Built in step 0, label-free with respect to every fusion outcome.
- **Scoring:** each cohort-A checkpoint (and zero-shot) with **its cohort-A selected w** for each variant. GZ arms also use their cohort-A validation z-stats. AZ arms need no stats (§2.5). No selection is done on HuRI data. Concretely:
  - **AZ-X and the AZ controls X, P, UX, N, T:** the w fit on cohort-A validation under that variant.
  - **S:** the w fit on cohort-A's **deranged** validation file (§2.3), applied to the HuRI test file's own pinned shuffle map. That map is built by the same §2.3 rules over the HuRI anchors.
  - **C5:** the 2-D (w_g, w_b) selected on cohort-A validation (§2.6). Its comparator text + Bz uses its cohort-A 1-D w_b.
  - **GZ-P, U and S_P:** their cohort-A z-stats and w.
  
  All HuRI control states are descriptive, so none of these choices changes a label.
- **Caveat (§8):** "HuRI-observed" is a proxy for the HuRI search space (the ORFeome list is not in the repo), so some grade-0 proteins were never tested.
- **HuRI-supported-positive strata** (descriptive): in cohort A (58 anchors with a grade-2 record that is also a HuRI edge) and cohort B (≈ 337 anchors), Δ restricted to records of those anchors with grade-2 rows limited to HuRI-supported positives.

## 4. Uncertainty, decision rule and labels

### 4.1 Two-level bootstrap
- **B = 2000.** Analysis RNGs: cohort A `np.random.default_rng([31415, 1])`; cohort B primary (`Bclean`) `np.random.default_rng([31415, 2])`; HuRI cohort `np.random.default_rng([31415, 3])`; cohort B sensitivity `all_B` `np.random.default_rng([31415, 4])`. Within a cohort the same resample indices are reused for every metric, φ variant, contrast and model type (common random numbers), so all contrasts (X − S, P − X, UX − X, C5, AZ-X − GZ-P, …) are paired. The GZ-P Holm triple (§1.2b) uses the same draws.
- **Cohort A (trained):** D[c, a] = metric_fused − metric_text for checkpoint c (19, or fewer after §5 exclusions) and test anchor a (1,396). Each replicate draws c* (n_c with replacement) and a* (1,396 with replacement) independently. Statistic: mean over defined a ∈ a* of the mean over c* of D[c, a]. Point estimate: same statistic on the original indices. Sensitivity: c* stratified by fraction.
- **Cohort B (trained):** D[s, a] for seed s ∈ {13, 21, 42} and test anchor a ∈ `Bclean` (primary) or all test anchors of all five folds (`all_B` sensitivity); D[s, a] uses the seed-s checkpoint of a's fold. Draw s* (3 with replacement) and a* (anchors of the population retained after §5 fold exclusions, with replacement, not stratified by fold). The seed index is an arbitrary pairing label across folds: seed s in fold k and seed s in fold j are unrelated models, and resampling s jointly across folds is a convention (as trials 2021), not a claim that they share anything.
- **HuRI cohort:** as cohort A (c* over cohort-A checkpoints, a* over the 769 HuRI anchors).
- **Zero-shot:** anchor-level only, same a* draws.
- **DE slope:** within each replicate, c* is drawn **stratified by fraction** (seeds resampled within each F) and a* as above; β_DE* = OLS slope of the per-checkpoint means over a* of D[c, a] on log10(F/100).
- **Intervals:** percentile. **Two-sided p** = min(1, 2·min((1 + #{T* ≤ 0})/(B + 1), (1 + #{T* ≥ 0})/(B + 1))).
- **Pooled A + Bclean** (descriptive only, trained and zero-shot separately): anchor-weighted by the per-metric number of defined anchors; replicate b of each cohort combined.

### 4.2 Equivalence (TOST) labels
For a quantity with equivalence bound δ, using the **90%** two-level CI:
- *equivalent* iff CI ⊂ (−δ, +δ);
- *not equivalent* iff CI lies entirely outside [−δ, +δ];
- *inconclusive* otherwise.
δ = 0.01 for every AUC/AP/nDCG Δ and contrast; δ = 0.005 per decade for β_DE. Every co-primary that is not positive (§4.3) also gets its TOST label, so "not confirmed" is split into "equivalent to zero" vs "inconclusive" in `report.md`.

**Four-state label** (secondaries, HuRI, C1, C1a, C1t, C3, C4, S1g and the AZ-X − GZ-P head-to-head; evaluated in order, first match wins): 1 *positive* (LB95 > 0); 2 *negative* (UB95 < 0); 3 *equivalent* (TOST, δ); 4 *inconclusive*.

### 4.3 Holm decision, criterion and final labels
**Notation.** In this section Δ_X is Δ of the primary fusion AZ-X, and Δ_S is Δ of the same AZ form on the shuffled φ S. "X − S" is Δ_X − Δ_S, paired. The same rules are applied descriptively to GZ-P with (Δ_P, Δ_{S_P}); those states never enter a label.

**Holm (per cohort, per model type, family {H1, H2, H3}).**
1. Order by bootstrap p; ties in p ordered H1, H2, H3 (likewise H1z, H2z, H3z and H1g, H2g, H3g).
2. Reject the first iff p₍₁₎ ≤ 0.05/3; only then the second iff p₍₂₎ ≤ 0.05/2; only then the third iff p₍₃₎ ≤ 0.05.
3. Hi is **positive** in a cohort iff Holm-rejected **and** point estimate > 0. A Holm-rejected negative point is "significant harm".
- Companion intervals (descriptive): 98.33% / 97.5% / 95% by Holm step reached; if a step is not reached, the remaining hypotheses get the interval of the last reached step's level. If interval and p disagree, p governs and `report.md` prints the disagreement.

**Control states that enter labels (trained models; per cohort; per Hi metric m).**
Cohort B in this section always means `Bclean` (§3c).

- **C2 state** (`go_pair_specific_v4`; Δ_S = Δ under the shuffled φ S in AZ form; Δ_X − Δ_S paired through common random numbers), first match wins:
  1. *fail* if either branch holds:
     - **branch 1:** LB95(Δ_X − Δ_S) ≤ 0 **and** [point(Δ_X) ≤ point(Δ_S) **or** LB95(Δ_S) > 0]. There is no confident pair-specific component, and either the shuffle matches the real gain or it keeps a confident share of it.
     - **branch 2:** UB95(Δ_X − Δ_S) < +0.005 **and** LB95(Δ_S) > 0. The pair-specific component is confidently small while the shuffle keeps a confident share of the gain. This mirrors the C5 absorption condition.
  2. *pass*: UB95(Δ_S) < +0.005 **and** LB95(Δ_X − Δ_S) > 0;
  3. *partial*: LB95(Δ_S) > 0 **and** LB95(Δ_X − Δ_S) > 0 (a confident pair-specific component plus a confident density component);
  4. *inconclusive*.
  - **Examples (unit-tested):**

    | Δ_X | Δ_S [95% CI] | X − S [95% CI] | state | why |
    |---|---|---|---|---|
    | — | +0.006 [+0.003, +0.009] | +0.012 [+0.007, +0.017] | *partial* | |
    | — | +0.002 [−0.001, +0.004] | +0.015 [+0.008, +0.022] | *pass* | |
    | +0.010 | +0.008 [+0.004, +0.012] | +0.002 [−0.002, +0.0045] | *fail* | both branches: LB95(X − S) ≤ 0 and LB95(Δ_S) > 0; UB95(X − S) < 0.005 and LB95(Δ_S) > 0 |
    | point > point(Δ_S) | +0.004 [−0.001, +0.009] | +0.010 [−0.001, +0.020] | *inconclusive* | branch 1 needs point(Δ_X) ≤ point(Δ_S) or LB95(Δ_S) > 0; neither holds |
    | +0.004 [+0.0015, +0.0065] | +0.0003 [−0.0010, +0.0015] | +0.0037 [+0.0020, +0.0048] | *pass* | a small gain that the shuffle removes almost entirely; branch 2 does not fire because LB95(Δ_S) ≤ 0 |
    | — | +0.003 [−0.001, +0.007] | +0.003 [+0.001, +0.0045] | *inconclusive* | branch 2 needs LB95(Δ_S) > 0; pass needs UB95(Δ_S) < 0.005 |
    | +0.020 | +0.018 [+0.014, +0.022] | +0.002 [−0.003, +0.007] | *fail* | branch 1 via LB95(Δ_S) > 0; round-2 review example; point(Δ_X) > point(Δ_S) and UB95(X − S) ≥ 0.005, so only the new clause fires |
    | > 0, LB95 > 0 | ≡ 0 [0, 0] (w = 0 on the shuffle) | = Δ_X, LB95 > 0 | *pass* | the exploratory AZ-X pattern, §0.1b |
  - **Overlap note.** When fail branch 2 holds, LB95(Δ_X − Δ_S) may also be > 0, so the *partial* condition can hold too. *fail* wins by order: a confidently small pair-specific part beside a confident density part is read as "carried by density".
  - **`c2_trials_state`** (reported only, never in a label): the trials §4.3 rule verbatim.
    1. *fail* iff LB95(Δ_S) > 0 or [LB95(Δ_X − Δ_S) ≤ 0 and point(Δ_X) ≤ point(Δ_S)];
    2. *pass* iff UB95(Δ_S) < +0.005 and LB95(Δ_X − Δ_S) > 0;
    3. *inconclusive*.
    
    Recorded in `analysis.json` per cohort × Hi for cross-domain comparison.
- **C5 state** (`absorbed_share_v3`; Δ_C5 = metric(text + X + Bz) − metric(text + Bz), AZ form; Δ_X − Δ_C5 paired), first match wins:
  1. *fail* if either branch holds:
     - **branch 1:** LB95(Δ_C5) ≤ 0 **and** [point(Δ_C5) ≤ 0 **or** LB95(Δ_X − Δ_C5) > 0]. GO adds nothing confident beyond study bias, and either the residual point is ≤ 0 or Bz has confidently absorbed part of Δ_X.
     - **branch 2:** UB95(Δ_C5) < +0.005 **and** LB95(Δ_X − Δ_C5) > 0. The residual is confidently small after Bz has absorbed a confident share of Δ_X.
  2. *pass*: LB95(Δ_C5) > 0;
  3. *inconclusive*.
  - **Examples (unit-tested):**

    | Δ_X | Δ_C5 [95% CI] | LB95(Δ_X − Δ_C5) | state | why |
    |---|---|---|---|---|
    | +0.012 | +0.004 [+0.001, +0.0045] | > 0 | *fail* | branch 2 |
    | +0.0045 | +0.004 [+0.001, +0.0045] | ≤ 0 | *pass* | a small GO effect that Bz does not absorb is not attributed to study bias |
    | — | −0.001 [−0.004, +0.002] | — | *fail* | branch 1, point ≤ 0 |
    | +0.003 | +0.001 [−0.002, +0.004] | ≤ 0 | *inconclusive* | |
    | +0.020 | +0.002 [−0.002, +0.006] | > 0 | *fail* | branch 1 via absorption; round-2 review example |
- **HuRI qualifier** (cohort A models only; per Hi metric): the four-state label of Hi-HuRI.

**Criterion per Hi ∈ {H1, H2, H3} (trained):** raw_conf(Hi) = positive_A(Hi); raw_rep(Hi) = raw_conf(Hi) and positive_Bclean(Hi), where positive_Bclean is the Holm(3) decision of §4.3 on the `Bclean` test anchors with rng `default_rng([31415, 2])`. All-B anchors never enter the criterion.

**Final label per Hi (deterministic, evaluated in order, first match wins):**
```
label(Hi) =
  1. if not raw_conf(Hi):  "not confirmed (<TOST label in A>)", or "significant harm (A)"
  2. elif C2_A(Hi) == fail: "not attributable to anchor-specific GO annotations (A)"
  3. elif C5_A(Hi) == fail: "not separable from study bias (A)"
  4. elif not raw_rep(Hi): "confirmed, not replicated (<TOST label in B>)"
                           (+ "; significant harm in B" if Holm-rejected with point < 0 in B)
  5. elif C2_B(Hi) == fail: "confirmed; replication not attributable to anchor-specific GO annotations"
  6. elif C5_B(Hi) == fail: "confirmed; replication not separable from study bias"
  7. else:                  "replicated"
suffixes, appended in this order, only for labels 2-7:
  a. " (shuffled control inconclusive: <cohorts>)" for each cohort whose C2 state was reached
     by the rule and is inconclusive (A for labels 3-7, B for labels 6-7), order A, B;
  a2. " (partly carried by annotation density: Δ_S = x [lo, hi]; X − S = y [lo, hi]; <cohort>)"
     for each cohort whose C2 state was reached by the rule and is partial (A for labels 3-7,
     B for labels 6-7), order A, B; numbers are the 95% CIs of that cohort;
  b. " (study-bias control inconclusive: <cohorts>)", same reach rule with C5 (A for 3-7, B for 6-7);
  c. always: " [HuRI: <four-state label of Hi-HuRI>]"; if that state is equivalent or negative,
     additionally " (not supported on literature-independent HuRI labels)".
```
**Headline (first line of `report.md`):**
- "GO fusion (AZ-X, exact pruned BP terms, per-anchor z) improves within-anchor ranking, replicated" iff all three Hi have label 7.
- "… partially replicated (<list of Hi with label 7>)" iff one or two have label 7; the others' labels follow.
- Otherwise `"H1: <label>; H2: <label>; H3: <label>"`. Suffixes are part of labels and always printed.
- Example: all Hi positive in A and Bclean, C2 pass, C5_A(H3) fail → H3 label 3, H1/H2 label 7 → "partially replicated (H1, H2)" with "H3: not separable from study bias (A) …".
- Example: all Hi positive in A and Bclean, C5 pass, C2_A(H3) partial, C2_B(H3) pass → H3 label 7 "replicated (partly carried by annotation density: Δ_S = … ; A) [HuRI: …]".
- C2 is evaluated for A at labels 2-7 and for B at labels 5-7. At label 2 (A) and label 5 (B) the state is *fail* by definition, so suffixes a and a2 can only fire for A at labels 3-7 and for B at labels 6-7; a *partial* or *inconclusive* state never yields labels 2 or 5 (only *fail* does).
- The pair-specific component X − S and the leak component UX − X (with their four-state labels) are printed on the line after the headline.
- **Second line (secondary fusion, never part of the headline).** It has the form `"Secondary GZ-P: H1g <Holm status>, H2g …, H3g …, nDCG@10 <four-state>; AZ-X − GZ-P: H3 <four-state>, nDCG@10 <four-state>"`.

**Reported, not in any label:** `c2_trials_state` per cohort × Hi; zero-shot H1z–H3z (Holm and raw status per cohort; zero-shot control states descriptive); S1–S5; GZ-P H1g–H3g, S1g and its descriptive C2/C3 states; the AZ-X − GZ-P head-to-head; C1/C1a/C1t/C3/C4 four-state labels; DE state; all sensitivities.

### 4.4 Data-efficiency state (cohort A, trained, primary fusion AZ-X)
Evaluated in order:
1. *supported* (p_DE ≤ 0.05 and β_DE < 0);
2. *reversed* (p_DE ≤ 0.05 and β_DE > 0);
3. *flat* (TOST equivalent, δ = 0.005 per decade);
4. *inconclusive*.

The state is computed only if every fraction keeps ≥ 2 checkpoints after §5; otherwise it is "not computed".
- **Descriptive companions:** β_DE for the H1 and H2 metrics; β_DE under GZ-P; Δ by fraction with CIs; the endpoint contrast F10 − F100.
- **Confounder disclosed:** the exploratory gain shrinks with F, but text level also rises with F. β_DE is reported together with the per-fraction text level.

### 4.5 Sensitivities and descriptive arms (none changes a label)
- **Weight rule:** AZ-X with w selected by validation nDCG@10; GZ-P with w by global pooled AUC (§2.6). All three Hi, both cohorts.
- **Normaliser:** GZ-X, i.e. X in the global-z form with the global pooled-AUC rule (§2.5).
- **Exposure (cohort B), preregistered:** `sensitivity.all_B` = H1–H3 (trained, Holm reported) and H1z–H3z, plus C2 and C5 states, on **all** cohort-B test anchors (rng `default_rng([31415, 4])`), reported next to `Bclean`. **Mandatory warning:** for each Hi with label 5–7 whose `all_B` point estimate differs in sign from its `Bclean` point estimate, `report.md` prints "exposed anchors disagree with the clean replication (Hi)" under the headline. Descriptive: Δ on the exposed cohort-B anchors alone.
- **Pruning (C3):** UX in AZ form at its own selected w on the same retrained models, with UX − X paired. For GZ-P, U at its own w, with U − P paired.
- **A-pub arm:** the 15 published (unmasked-text) checkpoints on the published files, same estimator over 15 checkpoints.
  - Arms: AZ-X, AZ-UX, GZ-P and the published U form (= gate-2 numbers). Pruned URIs come from the aligned masked files' `skill_uris`; the text is the published unmasked text.
  - A-pub reproduces the §0.1b test numbers on the retrained-model estimator and shows the masking effect directly: retrained-cohort-A minus A-pub, same anchors, unpaired across models, descriptive.
  - It is descriptive only, because the published models are exposed and Mac-trained.
- **Mention sensitivity (cohort A):** drop test grade-2 records flagged as partner-symbol mentions in the unmasked views (flag computed in step 0 with `go_ppi_domain.leak_pruning.entries_in_text` on the unmasked anchor/partner `encoder_view` against `preprocess/go_withholding/converted/mask_lexicon.json`, partner symbol in anchor view or anchor symbol in partner view). Residual leak `d_complex_name` (28.1% / 16.5% / 16.3% by grade, `smoke_report_go_ppi.md` §2) is disclosed, not corrected.
- **Empty-φ anchors:** exclude anchors whose pruned anchor-side BP set is empty (42 in cohort A test).
- **Study-bias stratification:** Δ for each Hi by tertile of anchor n_PMID(a) (cut points from the cohort's test anchors), and by tertile of anchor STRING degree in `pairs.jsonl` (all grades); descriptive, with a heterogeneity line (top − bottom tertile, 95% CI).
- **HuRI:** `huri_new_labels`; `huri_grade1_unrestricted` (§3d); HuRI-supported-positive strata in A and Bclean (all-B as well, descriptive).
- **Fraction-stratified c*** (cohort A).

### 4.6 Power (qualitative)
Cohort A: 729 hard-defined anchors and 19 checkpoints. The exploratory AZ-X hard Δ is +0.029 with half-width ≈ 0.010 on 15 checkpoints. An effect of that size survives Holm(3); one halved by masking, or by the selection optimism of a post-search estimate, may not. Cohort B (primary population `Bclean`): ≈ 2,694 hard-defined anchors under fold seed 2026, asserted ≥ 2,400 under seed 2027 at build (§3c) (≈ 3.3–3.7× A's 729; all-B ≈ 3,850), 3 seeds; anchor-level noise shrinks, seed noise does not, so Bclean is at least as well powered as A for the same effect. Unlike trials 2021 (52 closed-unexposed patients), the clean subset here is large enough to carry the criterion, which is why it is primary. HuRI: 769 anchors, smaller expected effect; a null there is reported with its TOST label, not as evidence of absence unless equivalent. DE: only 3–5 checkpoints per fraction; likely inconclusive unless the slope is as large as explored.

## 5. Gates before the confirmatory analysis

1. **Unit tests** (`go_ppi_domain/tests/test_fusion_confirm.py`, run on the remote CPU after step −1 and before `make_pins.py`; Mac runs do not count):
   - fusion arithmetic for AZ and GZ, tie → smaller w, 2-D C5 grid tie order;
   - within-anchor z: constant anchor → z ≡ 0; per-anchor mean 0; equality with `go_ppi_fusion.fusion.within_query_z` on toy input;
   - global pooled-AUC selection equals `go_ppi_fusion.metrics.global_auc` / the `epf` `pooled_auc` rule on toy input;
   - stratified derangement on toy input: no receiver gets a forbidden donor under rules (a)–(e); deterministic across calls; per-stratum abort after 10,000 draws on an infeasible stratum; empty-set anchors untouched;
   - typed parser: on a toy OBO, `("is_a", "part_of")` equals `go_ontology.parse_obo` parents and `("is_a",)` drops `part_of` edges;
   - n_PMID parser on a toy GAF (NOT lines excluded, `GO_REF` ignored, `|`-separated references);
   - fold builder: groups kept together, protein-disjoint, identical hashes across two builds; HuRI builder determinism and exclusion rules on toy data;
   - read-only loader asserts width 768; manifest digest is cwd-independent;
   - bootstrap shapes and determinism; DE slope on a toy D with known slope;
   - Holm(3) including ties and the negative-effect case;
   - C2 (`go_pair_specific_v4`) and C5 (`absorbed_share_v3`) ordered states, on every row of both §4.3 example tables. That covers:
     - the *partial* state;
     - both C2 fail branches, including the round-2 branch-1 absorption case;
     - the small-fully-pair-specific pass case and the inconclusive case (the two round-1 examples);
     - the fail-over-partial order;
     - both C5 fail branches, including the round-2 branch-1 absorption case, versus the small-unabsorbed-effect pass case and the inconclusive case with Δ_X = +0.003.
     
     Also `c2_trials_state` on the same inputs (e.g. the *partial* example gives trials *fail*), and the TOST and four-state labels;
   - the §4.3 final-label rule and headline on (a) all positive, all controls pass → "replicated"; (b) C2_A(H3) fail → H3 label 2; (c) C5_B(H1) fail → H1 label 6; (d) C2_A inconclusive with label 1 → no suffix; (e) HuRI equivalent → "not supported" suffix; (f) C2_A(H3) partial → label unchanged, suffix a2 present; (g) positive in A, positive in all_B but not in Bclean → label 4; (h) the all_B/Bclean sign-disagreement warning;
   - HuRI builder: grade-1 rows restricted to HuRI-observed proteins; unrestricted file differs only in grade-1 rows;
   - gate-2 leaf classifier and comparator (tiers below, including the near-tie `selected_weight` exception);
   - pins chain verifier and `--revise` rules as trials §5.1 (whitelist, declared changes, `confirmatory_output_seen`, broken `supersedes`, non-latest pins, configs never rewritten);
   - gate-3 device check (`cuda_device_name` must equal pinned name; `requested`/`resolved` must be `"cuda"`);
   - `verify_transfer.py` protocol parser: finds exactly 15 checkpoint rows (§3b) and ≥ 30 input rows (§7), aborts on a truncated copy;
   - `train.py` ledger/resume semantics as trials §6.3; `PYTHONHASHSEED` guard in every new script.
2. **Reproduction and integrity gate** (remote CPU, after step 0; touches only already-exposed data or label-free data):
   - **2a. Published reproduction.** Score the 15 published checkpoints with U on the published unmasked files, grid 0–1.0, under **both** rules (`ndcg@10` and global `pooled_auc`), and compare with `results/publication_phase1_fixedval/go_ppi_fusion_learning_curve_ndcg_selected.json` and `…/go_ppi_fusion_learning_curve.json`. **Tiered tolerances (preregistered; trials §9 A2 lesson applied from the start):**
     - **exact:** integer counts (`n_queries`, `queries_with_both_classes.*`, `tier_stats.*.n`) and `selected_weight`, **except** cells whose published validation top-two gap is < 1e-4, where either of the two top weights passes and the event is logged. The near-tie cells are derived by the script from the published curves and must equal {ndcg rule f10_s42 (gap 7.2e-6), pooled rule f10_s42 (3.6e-5), pooled rule f75_s21 (6.4e-5)} (F§6 R7), pinned as `pins.analysis.gate2_near_tie_cells`. In a near-tie cell where the other weight is chosen, the cell's test leaves are compared against a re-score at the published weight instead;
     - **rank_per_query, abs ≤ 1e-3:** every leaf with `per_query` in the key path. **Float-tie exception (§9 A1):** a failing per-anchor test ranking leaf passes, and is logged as an event, iff (i) its anchor has at least one *float-tied group* and (ii) some ordering of the tied records reproduces the published value within 1e-3.
       - A float-tied group is a chain of records adjacent in raw float32 text score whose scores differ by < 1e-6; in the fusion arm, the records must also have identical φ.
       - The orderings are enumerated within each group, placing the members at the group's top score minus k·1e-13. At most 5,040 orderings per anchor; above that the exception is refused.
       - The exception never applies to macro, global, curve, stats or count leaves, or to `selected_weight`.
       - Pinned as `pins.analysis.gate2_float_tie`.
     - **rank, abs ≤ 1e-4:** `validation_selection_curve.*`, `ranking.*`, `ranking_macro.*`, `global_auc.*`, `macro_query_auc.*`, `kendall_tau_b`, `spearman_rho`;
     - **continuous, abs ≤ 1e-5:** `validation_stats.*` and every other computed leaf (`tier_stats.*.{mean,sd,median}`, `cohens_d.*`);
     - class order: skipped (threshold metrics not computed) → exact → rank_per_query → rank → continuous. Largest |Δ| per class and per leaf group is written to the gate output.
     - **Zero-shot (unmasked, published files):** test macro hard AUC 0.786, pooled 0.838, nDCG@10 0.772 to 3 dp (F§5).
     - Reason for the tiers: rank metrics step on near-tie flips; per-anchor leaves are undiluted by averaging; float32 BLAS differences between Mac Accelerate and remote x86 BLAS reach ≈ 2e-6 on single-score leaves such as medians (trials §9 A2). Every limit is ≥ 5× below the smallest effect of interest (5e-3); every confirmatory number is computed fresh on the one host, so the tolerance validates code only.
   - **2b. Alignment.** Masked/pruned `preprocess/go_withholding/splits/{train,validation,test}.jsonl` are record-for-record aligned with `preprocess/go_ppi_splits/{train,validation,test}.jsonl` (same order, anchor, partner, grade); the cohort-A masked fractions and `validation_positive.jsonl` key-match their published counterparts with the asserted counts; pruned record URIs equal the pruned index `direct` for every protein.
   - **2c. Typed-parser check** (§2.2.1). Failure only removes T (no stop).
   - **2d. Derived inputs exist and are pinned:** shuffle maps (no abort), n_PMID table, HuRI cohort (anchor count within ±5 of 769; both grade-1 counts equal `derived.huri.grade1_counts`), cohort-B folds (two-build hash identity, group count 14/31, `Bclean` list sha256 equal to `derived.bclean`, hard-defined count ≥ 2,400), masked encodes (manifest digest), config audit sets.
   - **2e. φ path equality (label-free).** On cohort-A validation and test (masked files), the X, P, N, S and S_P arrays produced by `fusion_confirm` must equal direct calls, record by record and with float equality. The direct call is `GoMatcher(GoIndex.load(<pruned index>), similarity_mode=m).ontology_set_similarity(a_uris, b_uris)` with m = `exact` / `simgic` / `ancestor` / `exact` / `simgic`. For S and S_P, `a_uris` is taken from the pinned shuffle map. UX is checked the same way against the unpruned index with `exact`. Uses no grade and no text score. A mismatch is a gate-2 failure under the retry rule.
   - **2f. Best-fusion reproduction (published checkpoints, already exposed).** Score the 15 published checkpoints on the published test file with AZ-X and GZ-P exactly as in §2.6. φ comes from the pruned URIs of the aligned masked files; text is the published unmasked text. Compare with `results/go_ppi_fusion_best/test_results.json` (`variants.F3_P`, `variants.PUB_ndcg_P`):
     - selected w per checkpoint: exact, except for near-tie cells (validation top-two gap < 1e-4, logged), as in 2a;
     - per-checkpoint macro Δ for hard, pooled, easy, AP and nDCG@10: the `rank` tier (abs ≤ 1e-4).
     
     This checks that the confirmatory code implements the method that was chosen. It touches only already-exposed anchors and the already-reported method.
   - **Retry rule** (as trials §5.2): a failure may be fixed in code and the gate rerun; each attempt is a §9 entry and a new pins revision `pins_r{n}.json`; outputs go to new leaves `gate_reproduction_r{n}.json`. Tolerances are fixed; a failure persisting after a code fix stops the study and needs a declared-change amendment of `analysis.gate2_tolerance_abs` / `analysis.gate2_leaf_classes` with `confirmatory_output_seen == false` (always true at gate 2).
   - **Code freeze:** once gate 2 passes, `pins.code` is frozen (§7).
3. **Run checks** (each of the 34 training runs): exit code 0; `lc_manifest.json` and `best_checkpoint.pt` exist; no `run_error.json`; `single_factor_audit.unexpected_fields` exactly as §3b; `attach.cache_check` passes; `attach.matcher` is null (no GO in training); `lc_manifest.json` `device` has `requested == resolved == "cuda"` and `cuda_device_name == pins.environment.gpu_name`; sha256 of `embedding_cache/go_ppi_text_embeddings.pt` and `embedding_cache/go_withholding_text_embeddings.pt` unchanged.

**Failure handling.**
- A failed run's dir is renamed `<dir>.failed1` and rerun once with the identical command; a second failure excludes it. A failure of the environment (as trials §9 A3) is fixed and documented in §9; the ledger may be archived (moved, never deleted) before any checkpoint exists.
- Cohort A proceeds iff ≥ 17 of 19 checkpoints remain; DE additionally needs ≥ 2 per fraction (§4.4).
- Cohort B proceeds iff ≥ 4 folds have all 3 seeds; anchors of an incomplete fold are dropped from trained estimates and kept in zero-shot.
- Otherwise stop and amend via the declared-change branch of §7 (`confirmatory_output_seen` must be false); after confirmatory output, only a post hoc deviation in `report.md` is possible.
- Training is non-deterministic at a fixed seed; each checkpoint is scored once, as produced; its sha256 is ledgered before any scoring.

## 6. Files, commands, outputs

**Host and interpreter.** Every confirmatory command runs on the one remote L40S host from `$R` with `.venv/bin/python` (CUDA build of torch; versions recorded in `pins.environment`). Run conditions as trials §6: `PYTHONHASHSEED=0` (asserted by every new script and recorded in its output), `HF_HUB_OFFLINE=1`; training `--device cuda`; everything else `CUDA_VISIBLE_DEVICES=""`, `--device cpu`, `SentenceTransformer(..., device="cpu")`.

### 6.1 New files only
- `go_ppi_domain/fusion_confirm.py`: read-only loader, φ variants (X, P, U, UX, N, T, S, S_P, Bz), typed parser, stratified derangement, n_PMID, AZ and GZ normalisers / fusion / 1-D and 2-D w selection under both rules (within-anchor z, global pooled AUC and macro nDCG@10 imported from the existing new package `go_ppi_fusion` (`fusion.within_query_z`, `metrics.global_auc`), which is hashed into `pins.code`), per-anchor metrics, two-level bootstrap, DE slope, Holm(3), TOST / four-state / C2 / C5 states, final labels, fold and HuRI builders, gate-2 leaf classifier. Reuses `epf.rank_auc`, `epf.ranking_metrics`, `epf.z_stats`, `epf.z_apply`, `epf.text_scores` via `importlib.util.spec_from_file_location` on `scripts/evaluate_publication_fusion.py` (as trials). Imports `go_ppi_domain.go_ontology`, `data_splitter`, `leak_pruning` (not edited). Generic helpers may be copied from `trials_domain/fusion_confirm.py` into the new file; the trials file is not edited. Port source for the scorer: `/tmp/goppi_fusion/{lib,recheck,all15,feasibility,feas2,feas3,zs}.py` (if gone, this protocol is the spec).
- `scripts/go_ppi_fusion_confirm/{dump_code_hashes,verify_transfer,build_inputs,encode,make_pins,train,score,analyze}.py`.
- `go_ppi_domain/tests/test_fusion_confirm.py`; runbook `.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_GO_PPI_FUSION.md`.
- Data: `preprocess/go_ppi_fusion_confirm/{cohortA,cv,huri}/`. Results: `results/go_ppi_fusion_confirm/`.
- **No-overwrite rule.** No existing file is edited; no existing result or data file is overwritten. Every script refuses to run if the leaf named by `--out` / `--output-dir` exists (parents may exist); no `--force`. Sanctioned exceptions: `.failed1`/`.failed2` renames and the append-only `run_ledger.json` (trials §6.3 semantics, atomic `os.replace`, earlier entries byte-identical).

### 6.2 Order and commands
**Step −1 (transfer).** `dump_code_hashes.py` writes `.agents/tasks/ontology-withholding-feasibility/code_hashes_go_ppi_fusion.mac.json` on the Mac; then `rsync -a --relative` from the Mac repo root to the same repo-relative paths on the remote: the code tree (never `300/` or `reports/`), every §7 input file, the 15 published checkpoint dirs (`best_checkpoint.pt`, `training_config.json`), both published prepared-cache dirs (5 + 2 chunks), and the **full** MPNet snapshot (revision `e8c3b32e…`). On the remote, `verify_transfer.py` parses this protocol (§3b checkpoint rows `| F | S | <64-hex> |`; §7 rows `| <path> | <64-hex> |`, path = first whitespace token of the first cell), aborts on < 15 / < 30 rows, checks every sha256, both chunk manifest digests, the snapshot revision and a CPU load test, the code manifest (missing / extra / differing file fails), and records the parsed protocol's sha256.

```
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
P=.agents/tasks/ontology-withholding-feasibility/protocol_go_ppi_fusion.md
O=results/go_ppi_fusion_confirm

# (-1) remote, CPU
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_confirm/verify_transfer.py --protocol $P \
    --code-manifest .agents/tasks/ontology-withholding-feasibility/code_hashes_go_ppi_fusion.mac.json \
    --out $O/transfer_check.json

# (0) gate 1, label-free builds, fresh masked encodes, cache copy, pins (remote, CPU)
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest go_ppi_domain/tests/test_fusion_confirm.py -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_confirm/build_inputs.py --protocol $P \
    --fold-seed 2027 --n-folds 5 --shuffle-seed 11 --device cpu \
    --out-data preprocess/go_ppi_fusion_confirm --out-phi $O/phi --out-configs $O/configs
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_confirm/encode.py --device cpu \
    --inputs preprocess/go_withholding/splits preprocess/go_ppi_fusion_confirm \
    --out $O/prepared_embeddings/go_ppi_masked
mkdir -p $O/cache
cp -n embedding_cache/go_withholding_text_embeddings.pt $O/cache/go_withholding_text_embeddings.pt
# make_pins.py runs WITHOUT the empty CUDA mask so it can read the GPU name (no data on the GPU)
.venv/bin/python scripts/go_ppi_fusion_confirm/make_pins.py --protocol $P --out $O/pins.json

# (1) gate 2 (remote CPU)
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_confirm/score.py --gate-reproduce \
    --pins $O/pins.json --device cpu --out $O/gate_reproduction.json
# (1b) only after a gate failure + fix, or an amendment: make_pins.py --revise --previous <latest> \
#      --amendment "<A id>" --protocol $P --changed-fields "<paths>" --confirmatory-output-seen {false,true} \
#      --out $O/pins_r{n}.json ; then rerun (1) into $O/gate_reproduction_r{n}.json
```
`build_inputs.py` writes (all label-free with respect to fusion outcomes): cohort-A masked fractions and validation positives, cohort-B folds, HuRI cohort, shuffle maps for every scored file, `n_pmid.json`, the is_a-only index (if 2c passes), the mention flags, exposure sets, the derived configs and their `report_factor` audit, and `build_manifest.json`. `make_pins.py` requires a passing `transfer_check.json` whose parsed protocol sha256 equals the pinned one, asserts the cache copy's sha256 equals the original, asserts exactly one visible CUDA device whose `torch.cuda.get_device_name(0)` contains `L40S` (aborting with a clear message if `CUDA_VISIBLE_DEVICES` is set to empty), and hashes every derived file. A `make_pins.py --revise` call follows the same rule. `<PINS>` below = the latest revision; every script verifies the `supersedes` chain and refuses a non-latest revision.

### 6.3 Training (34 runs, fixed order §3b)
```
# cohort A: (F, FRAC) in (10,0.1) (25,0.25) (50,0.5) (75,0.75) (100,1.0); S as §3b
.venv/bin/python scripts/run_learning_curve_point.py --domain go_ppi \
  --config $O/configs/lc_go_ppi_baseline_masked.json --compare-config config/lc_go_ppi_baseline.json \
  --train-file preprocess/go_ppi_fusion_confirm/cohortA/frac_${F}/train.jsonl \
  --validation-file preprocess/go_ppi_fusion_confirm/cohortA/validation_positive.jsonl \
  --fraction ${FRAC} --seed ${S} --device cuda \
  --output-dir $O/cohortA/checkpoints/go_ppi_baseline_f${F}_s${S}
# cohort B: k in 0..4, S in 13 21 42
.venv/bin/python scripts/run_learning_curve_point.py --domain go_ppi \
  --config $O/configs/lc_go_ppi_baseline_cv_fold${k}.json --compare-config config/lc_go_ppi_baseline.json \
  --train-file preprocess/go_ppi_fusion_confirm/cv/fold_${k}/train.jsonl \
  --validation-file preprocess/go_ppi_fusion_confirm/cv/fold_${k}/validation_positive.jsonl \
  --fraction 1.0 --seed ${S} --device cuda \
  --output-dir $O/cohortB/fold_${k}/go_ppi_baseline_full_s${S}
```
Issued only by `train.py --pins <PINS> --cohort {A,B}` (A first; run under `tmux`/`nohup`), which sets `PYTHONHASHSEED=0` in the child, applies gate 3 to each run, and appends ledger entries `{run, order_index, command, exit_code, status, best_checkpoint sha256, lc_manifest sha256, unexpected_fields, device, wall time, failed_attempts, cache_copy before/after}` to `$O/run_ledger.json`, with trials resume semantics.

### 6.4 Scoring and analysis (once, remote CPU)
```
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_confirm/score.py --pins <PINS> \
    --ledger $O/run_ledger.json --device cpu --out $O/scores
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_confirm/analyze.py --pins <PINS> \
    --scores $O/scores --out $O/analysis.json --report $O/report.md
```
- `scores/{A,B,huri,apub}/{fold or all}/{model}.npz`: per record text score, every φ variant, Bz, grade, anchor id, partner id, for validation and test; `selected_w.json` per model × variant × rule × normaliser, with the validation top-two gap and shuffle draw counts.
- `analysis.json`: every estimate, CI, p, Holm decision, control state, DE state, label, headline, sensitivity, and the pins revision path + sha256 it ran under.
- `report.md`: headline; X − S and UX − X line; the secondary GZ-P / head-to-head line; mandatory exposure warnings; §0.3 disclosures; tables H1–H3, H1z–H3z, S1–S5, controls, DE, HuRI, sensitivities, pooled; every p/CI disagreement, near-tie, exclusion and unmatched HuRI draw count.

### 6.5 Mac dev smoke (non-confirmatory)
Allowed only on:
- toy fixtures;
- the 15 published checkpoints on published files, restricted to the gate-2a and 2f quantities (U under both rules, AZ-X, GZ-P; already exposed and already reported in §0.1b);
- cohort-B **fold structure** without scoring.

Forbidden on the Mac: any score or metric on cohort-B test or validation anchors, the HuRI cohort or any retrained checkpoint. Also forbidden is any further fusion variant or control on the published files beyond those already reported in `results/go_ppi_fusion_best/`. Outputs go to `results/go_ppi_fusion_confirm_dev/` (excluded from transfer). Any breach is disclosed as an A0-style amendment before pins.

## 7. Pins (`results/go_ppi_fusion_confirm/pins.json`, revisions `pins_r{n}.json`)
Versioning, allowed / declared changes, `confirmatory_output_seen`, configs-written-once, chain verification and code freeze follow trials §7 verbatim. Schema `go_ppi_fusion_confirm/pins/v1`. Key `analysis` fields: `weight_grid = [i/10 for i in range(11)]`, `fusion_primary = {"id": "AZ-X", "normaliser": "within_anchor_z", "phi": "X", "select_on": "global_pooled_auc"}`, `fusion_secondary = {"id": "GZ-P", "normaliser": "validation_file_global_z", "phi": "P", "select_on": "ndcg@10"}`, `controls_form = "same_as_controlled_fusion"`, `sensitivity_select_on = {"AZ-X": "ndcg@10", "GZ-P": "global_pooled_auc"}`, `sensitivity_normaliser = "GZ-X"`, `tie_break = "smaller_w"`, `c5_grid = "w_g+w_b<=1, tie smaller w_g then w_b"`, `c5_select_on = "global_pooled_auc"`, `c1_fixed_w = 0.4`, `phi_variants = ["X","P","U","UX","N","T","S","S_P","Bz"]`, `exploration_reference = {"test_results": "results/go_ppi_fusion_best/test_results.json", "finalists": "results/go_ppi_fusion_best/finalists.json", "amendment": "results/go_ppi_fusion_best/test_plan_amendment.json"}` (sha256 in `inputs`), `B = 2000`, `rng = {"A": [31415,1], "Bclean": [31415,2], "huri": [31415,3], "all_B": [31415,4]}`, `cohortB_primary_population = "Bclean"`, `shuffle = {"seed": 11, "strata": 10, "max_draws": 10000, "forbidden": ["self","candidate","same_set","partner_set","simgic>=0.5"]}`, `fold_seed = 2027`, `holm_family = ["H1","H2","H3"]`, `holm_tie_order = ["H1","H2","H3"]`, `alpha = 0.05`, `equivalence = {"bound": 0.01, "bound_de": 0.005, "ci": 0.90}`, `c2_rule = "go_pair_specific_v4"` (states fail / pass / partial / inconclusive, §4.3; fail branch 1 includes the LB95(Δ_S) > 0 absorption clause), `c2_pass_ub = 0.005`, `c2_fail_pairspecific_ub = 0.005`, `c2_fail_requires_absorption = true` (fail branch 2 needs LB95(Δ_S) > 0), `c2_trials_state_reported = true`, `c5_rule = "absorbed_share_v3"` (fail branch 1 includes the LB95(Δ_X − Δ_C5) > 0 absorption clause), `c5_fail_ub = 0.005`, `final_label_rule = "ordered_7_label_v4"` (v3 rule, consumes `c2_rule` v4 and `c5_rule` v3, headline names AZ-X), `bclean_min_hard_defined = 2400`, `huri_grade1 = "huri_observed_only"`, `huri_sensitivities = ["huri_new_labels","huri_grade1_unrestricted"]`, `min_checkpoints_A = 17`, `min_complete_folds_B = 4`, `gate2_tolerance_abs = {"continuous": 1e-5, "rank": 1e-4, "rank_per_query": 1e-3}`, `gate2_selected_weight = "exact_except_near_tie<1e-4"`, `gate2_float_tie = {"gap": 1e-6, "max_orderings": 5040, "offset": 1e-13, "rule": …}` (§9 A1; from pins revision 2), `gate2_near_tie_cells`, `gate2_leaf_classes`, `training_order`, `gpu_name_must_contain = "L40S"`. Also pinned: `environment` (as trials, `device_policy`), `transfer_check`, `inputs`, `published_prepared_cache` (chunks and both digests), `masked_encodes` (digest), `derived` (every `build_inputs` output with sha256, fold membership, view groups, exposure sets and counts, `bclean` anchor list + sha256 + hard-defined count, `bclean_records_with_exposed_partner`, HuRI counts including `grade1_counts` under both definitions, shuffle draw counts and donor–partner-set similarity counts), `code`, `configs` (base, derived with diff and expected unexpected fields), `cohortA.new_runs` (19), `cohortB.runs` (15), `published_checkpoints` (15).

`inputs` must include at least (sha256 at writing):

| file | sha256 |
|---|---|
| preprocess/go_ppi_splits/test.jsonl | c65a1d2ca0e5c16a1c69ecd70d921e670531d325c52625aa15dba88c2e2481e4 |
| preprocess/go_ppi_splits/validation.jsonl | 07e432fb64f3e2716d74019806b0926aa26b2926a12b541a2fda1b8bf8ac91c6 |
| preprocess/go_ppi_splits/train.jsonl | c2013f8a134d24dbb688a96515dc41c118d8932e211feefde395fe96f9a02efe |
| preprocess/go_ppi_splits/negative_pools.jsonl | 2e4fc52b2c98174ef26c04f4ff008771868881c9d1789520bcc15ea398c3a315 |
| preprocess/go_ppi_splits/split_manifest.json | 8b2ae41bec6bc331f531cecfd5b00b76b125329e7692d721c8a8dc24ede4da0d |
| preprocess/go_ppi_lc/frac_10/train.jsonl | 9b353aba70291de284f1652282bec652e84624c1992395c310cb84e19b667176 |
| preprocess/go_ppi_lc/frac_25/train.jsonl | 0751771d38a9b8f5ecb58e6d71c3b33f15a91cb932c2b69e040d07aaf6a97445 |
| preprocess/go_ppi_lc/frac_50/train.jsonl | 1ae69c628988a0ad62b3f109775ad124c3f7293eaa0731aa3256802eb94e442a |
| preprocess/go_ppi_lc/frac_75/train.jsonl | 043e385f8a0656f26832f8684dd56b83d9c40bdd95bcc7e75686b2c21e389efe |
| preprocess/go_ppi_lc/frac_100/train.jsonl | c2013f8a134d24dbb688a96515dc41c118d8932e211feefde395fe96f9a02efe |
| preprocess/go_ppi_lc/validation_positive.jsonl | c610cd6e4552f4cfa128d9d63e7c99e41efa62afc2b888c4165fa3c6ea28ebd7 |
| preprocess/go_ppi_lc/fraction_manifest.json | 502a024de18e1b9fab5686b5c6429ce63524503ef5693976b40fb54a00f9411f |
| preprocess/go_ppi/go_index_P.pkl | 56892eaaf1363d24e22b8bf697dc16c4b68de6672afb019f7263bcdb2968d042 |
| preprocess/go_ppi/proteins.jsonl | a9b51c66fdb9005ada99e6712ab88567bc6691d4a153112441404210fd7fccfa |
| preprocess/go_ppi/pairs.jsonl | f789f237eeee1ecb56f77c30a8a4c5cfd635ede95a70cfecf60c26fee0f47116 |
| preprocess/go_ppi/convert_manifest.json | 3da00a6fb9bd74a91e2544f9b13ba371ed6c0002ddcb085219dc779f0a575f2c |
| preprocess/go_withholding/splits/train.jsonl | c38f89c2dd795121b6e04bc0d8768c6076969cf237ffb18c5e1479d980856d1d |
| preprocess/go_withholding/splits/validation.jsonl | 90c0b7df9851cae23866e3101201981f7339727582276ae459d3d34cbfe31fd7 |
| preprocess/go_withholding/splits/test.jsonl | 4027e9a0a3cc6ec173fe468e90ff48e013efb8f61dc8eb2ce61d5b7a346bbb2a |
| preprocess/go_withholding/splits/negative_pools.jsonl | 2e4fc52b2c98174ef26c04f4ff008771868881c9d1789520bcc15ea398c3a315 |
| preprocess/go_withholding/splits/split_manifest.json | 8b2ae41bec6bc331f531cecfd5b00b76b125329e7692d721c8a8dc24ede4da0d |
| preprocess/go_withholding/converted/go_index_P.pkl | a0c8b0b4c062a4f550a3f0f42288fae45714977aca903564822cbf2eca2d590e |
| preprocess/go_withholding/converted/proteins.jsonl | 4c542c739c06754a4f24508f9c58af007f2f8f629f9b160ff7bd3fda09b711ff |
| preprocess/go_withholding/converted/pairs.jsonl | f789f237eeee1ecb56f77c30a8a4c5cfd635ede95a70cfecf60c26fee0f47116 |
| preprocess/go_withholding/converted/mask_lexicon.json | 108fc21791595a3dc24bb28b128a068890487be7f74eda9000bfe096f7ade098 |
| preprocess/go_withholding/converted/mask_manifest.json | 575db91cff01afbb8d850f154e5f5973517b32dc6881369caf5271d0b9989f09 |
| preprocess/go_withholding/converted/convert_manifest.json | b70a46ad567e9500e71583bb213cf82221cc9567bf5c93e573a8f40188f2c4e0 |
| embedding_cache/go_ppi_text_embeddings.pt | 923412dfa0b8fd7ab60d2011f8fbdbee7fdad7b11d6b77f415f2485fe20f41ba |
| embedding_cache/go_withholding_text_embeddings.pt | 3aed51487550ec02216b49617310927f48c0e29195dd4e3d1698f6adaa88a150 |
| dataset/go_ppi/bulk/go-basic.obo | b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc |
| dataset/go_ppi/bulk/goa_human.gaf.gz | db472faff1785878521693af62646546cea6af4a386609dd01c72c0554a46a30 |
| dataset/go_ppi/bulk/9606.protein.aliases.v12.0.txt.gz | b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680 |
| dataset/go_ppi/bulk/9606.protein.info.v12.0.txt.gz | 144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c |
| dataset/go_ppi/bulk/9606.protein.links.detailed.v12.0.txt.gz | 28a2ea3b22c964e691c0f2f934c5e3e397a4b0a12c833d8787fa02b7847b91b6 |
| dataset/go_ppi/independent/HuRI.tsv | 4cdf94366ff6e23e71ef9ec81922e85b2b3a1fb08e938f7b00d86c027cf1ecd6 |
| dataset/go_ppi/independent/Lit-BM.tsv | f39c9c4b8a372286c3eca77d0ff651d837a643d32235eae866daae082663391d |
| config/lc_go_ppi_baseline.json | 0a385d97f145c3b6deda4295b247580468c97d449185d0c4e1894280231da48a |
| scripts/evaluate_publication_fusion.py | 606793ef413229dc9fcec9e6facb0f89b363476e402c98b8719b1bd760616947 |
| scripts/run_learning_curve_point.py | 67f1d4887e04b6c478ddcaf9cf81667493b7a341e45ebeaed5dd0da932e22121 |
| scripts/go_independent_signal_diagnosis.py | b46afc0b6d81b837de94e6a2fd9fd7d4cc44252369da5b4be38cdfc1db542aef |
| go_ppi_domain/go_ontology.py | e370785fbf117848ffc05f0f0b51d955d1000ec565fb20c5e1ff69bea447ee43 |
| go_ppi_domain/data_splitter.py | e5b0b45d6386e3e8d45806af31dbcaefb3579c7dab022eec70f8c6d31b743c73 |
| go_ppi_domain/leak_pruning.py | c636e68d731124f5de51d01fb726867eaaa4411e2fa4b1bb3c9879e0df123a97 |
| go_ppi_domain/negative_selector.py | 619ae5510610b30b57ece1ace65b67319cf9e9df1ef5eb4ca8ce64c34521fbcc |
| go_ppi_domain/run_config.py | e95a630a492cccf4b75b3c236650ed876f12f9e2ad115544e94edb32ee14fde4 |
| go_ppi_domain/record_adapter.py | a00818d737dc58bb69e24d54909ede8b6534bf8bbaa3a6559e2356a44f466e3b |
| results/publication_phase1_fixedval/go_ppi_fusion_learning_curve.json | 0835241b417a3b010a652b80cafecfa5e88eb893740f58a8eff762ca57ac86c4 |
| results/publication_phase1_fixedval/go_ppi_fusion_learning_curve_ndcg_selected.json | 07927eb4bddb5cd4f903db0ed4ef1ddb80198c869ce782b01e2fa906247f7f7c |
| results/go_ppi_fusion_best/test_results.json | 4cb540abba87dc9dda7157ae7bbc4c711268ec9dffa7afdec9f0132318912fa0 |
| results/go_ppi_fusion_best/finalists.json | 499433200a6edfa6c31e4b7ccabc112fc40823531e54eef248b58832d35b5a44 |
| results/go_ppi_fusion_best/test_plan_amendment.json | efe265f5b656462b7bc026482b201a644cd7821c2acbcb751d6eb874ac423935 |
| results/ontology_ceiling/go_independent_signal.json | (hash recorded by `make_pins.py`; descriptive reference only, not parsed) |

## 8. Limitations, stated in advance
- **Study and literature bias is not fully removable.** STRING "established" evidence and GO annotation density both track how well studied a protein is. C2, C5, the n_PMID strata and the HuRI cohort bound it; none eliminates it. A label of 7 means "GO adds within-anchor ranking information beyond the anchor-specific-annotation and partner-study-bias controls on STRING labels", not "biology the literature misses".
- **C2 is anchor-side.** Partner annotation density is untouched by construction, so C2 alone cannot separate partner-side biology from partner-side study bias; C5 addresses the partner side through a protein-level PMID count, which is a proxy.
- **The primary fusion was chosen after a search on cohort A's files** (§0.1b, §0.3 item 9). On cohort A the AZ-X estimate inherits winner's-curse optimism from a 208-variant search plus a 10-variant test look. Cohort B (`Bclean`) is the only population where AZ-X was fixed before any of its anchors was scored.
- **The primary weight rule is a global cross-anchor AUC**, not the within-anchor estimand. It is kept because it is the rule AZ-X was chosen and test-scored with (§2.6). The nDCG@10 rule is a sensitivity.
- **AZ-X trades top-10 quality for whole-list quality.** In the exploration it was below the published form on nDCG@10 (−0.0104 paired). A confirmed AZ-X label says nothing about top-10 superiority; GZ-P (§1.2b) is the pre-declared top-10 method.
- **Pruning is global and conservative** (it also removes interaction-derived annotations not naming the partner, and changes IC everywhere). UX − X and U − P therefore over-state the leak-only contribution. Annotations derived from the same paper without a with/from entry (e.g. IDA/IMP from an interaction study) remain.
- **Residual text leak.** Masking removes partner symbols; `d_complex_name` co-mentions remain (≈ 28% of grade-2).
- **HuRI search space** is approximated by "HuRI-observed"; HuRI positives are Y2H binary interactions, a different label from STRING experimental ≥ 700. φ-only AUC on HuRI was already seen (0.548).
- **Cohort A is not independent of the exploration** (§0.3). Cohort B's replication population `Bclean` has never been scored under fusion, but (i) its inner validation folds, used for z-stats and w selection, include exposed proteins; (ii) its anchors' test records keep partners that may be exposed proteins (counted as `bclean_records_with_exposed_partner` and printed); (iii) the exposed anchors' training partition still shapes every cohort-B model. `Bclean` is "anchor-unexposed", not "untouched".
- **C2 is a GO-specific rule.** It gates on the pair-specific component X − S rather than on Δ_S (trials rule), because density-preserving anchor shuffles leave partner density, which tracks grade on GO. The trials-verbatim state is reported for comparison; the two can disagree (e.g. *partial* here is *fail* there).
- **C5 2-D grid is more flexible than its comparator** (66 vs 11 points). Validation over-fitting can only lower test Δ_C5, so the C5 state is conservative toward *fail*.
- **HuRI grade 1** is restricted to HuRI-observed proteins to avoid an observed-vs-unobserved contrast; this shrinks the grade-1 set and the hard-defined HuRI anchors (counts pinned). The unrestricted set is a sensitivity.
- **Typed relations:** T (is_a only) is a single rung; `regulates` is not tested; T is skipped if the parser check fails.
- **Data efficiency** rests on 3–5 checkpoints per fraction, and fraction is confounded with text-model quality; one published run (f100_s21) stopped at epoch 4, which illustrates the seed noise.
- **Device and numerics.** All confirmatory training on one CUDA device and all scoring on one x86 CPU; published numbers are reproduced only to the gate-2 tiers; training is not bit-reproducible at a fixed seed. Training reads frozen MPNet vectors from the mutable cache copy (mostly Mac-encoded, appended on CUDA during training), while scoring reads the fresh remote-CPU encodes of the same masked texts. The two differ by ≈ 1e-6 per component, far below any effect of interest; this mismatch is accepted, not corrected.
- **External validity:** human, BP aspect, STRING v12 / GOA snapshot, MPNet text.

## 9. Amendments
Any change after `pins.json` is written, any gate retry, and any pre-pins disclosure is logged here as `A{m}` with date, reason, files changed, whether confirmatory output had been produced, and the pins revision it creates.

**A1 (2026-10-13, after pins revision 1, before any training; creates `pins_r2.json`). Gate-2a per-anchor float-tie exception. First gate-2 attempt failed on float32 tie-breaking only.**

- **Confirmatory output: none.** No new-run checkpoint, no `scores/`, no `analysis.json` (`confirmatory_output_seen = false`).
- **Host:** NVIDIA L40S (`vclvm176-102`), torch 2.14.1+cu130, Python 3.12.3, numpy 2.5.4, scipy-openblas 0.3.34.
- **Attempt 1:** `results/go_ppi_fusion_confirm/gate_reproduction.json` (sha256 `afcd3b0c…`) under `pins.json` (sha256 `b584d708…`).

**What failed.**
- 2b, 2c, 2d, 2e and 2f passed. In 2f, all 30 AZ-X and GZ-P selected weights were exact and the maximum per-checkpoint Δ difference was 1.9e-5.
- 2a passed its zero-shot check and 25 of its 30 cells. In every 2a cell, all selected weights were exact and the macro, global and curve leaves were within their tiers (maximum 2.0e-5).
- The 5 failing cells each failed on exactly one per-anchor nDCG@10 leaf (`rank_per_query`, tolerance 1e-3). All are seed-42 checkpoints:

| rule | checkpoint | arm | anchor | published | computed | difference |
|---|---|---|---|---|---|---|
| pooled AUC | f10_s42 | text | IGLL1 | 0.5612 | 0.5649 | 0.0037 |
| pooled AUC | f50_s42 | text | RACK1 | 0.8792 | 0.8527 | 0.0265 |
| nDCG@10 | f10_s42 | text | IGLL1 | 0.5612 | 0.5649 | 0.0037 |
| nDCG@10 | f25_s42 | fusion | PYCARD | 0.7968 | 0.7924 | 0.0044 |
| nDCG@10 | f50_s42 | text | RACK1 | 0.8792 | 0.8527 | 0.0265 |

**Diagnosis.** A label-free probe of the raw text scores for these anchors, on the published checkpoints and the published test file, compared the Mac with the L40S host:
- IGLL1 f10_s42, ranks 8–9: the two scores are exactly tied on the Mac (gap 0, broken by file order) and 8.9e-8 apart on the L40S. The grade-1 and grade-0 records swap.
- RACK1 f50_s42, ranks 2–3: the two scores are 6e-8 apart on both machines, in opposite order.

These are float32 rounding differences in the projection-head dot product, not code differences: 2e and 2f, which run the same scoring code, are exact. The PYCARD fusion-arm case was not probed separately. It is covered by the same exception only if it meets condition (i) (float-tied text and identical φ). Otherwise gate 2 still fails.

**Change.**
- The float-tie exception in §5.2 2a (`rank_per_query`), with the conditions stated there.
- Code: `fusion_confirm.float_tie_groups`, `float_tie_explains`, a `per_query_tie_check` hook in `compare_gate2_run` (failing per-anchor leaves only), `GATE2_FLOAT_TIE` in `ANALYSIS_PARAMS`, and the 2a caller in `score.py`, which records every exception as an event and prints the count.
- Three new unit tests: grouping with the φ split and chaining; explained vs unexplained vs no-tie vs too-many-orderings; and the comparator hook, which never rescues a macro leaf.
- Tolerance values, leaf classes, the near-tie weight rule and every other gate are unchanged.

**Why it is safe.**
- The exception only accepts a difference that a re-ordering of records whose raw scores differ by < 1e-6 reproduces. A real per-anchor error with no float tie, or one that no tied ordering explains, still fails.
- Every confirmatory number is computed once, on this one host. Cross-host tie-breaking cannot reach a confirmatory result.
- Same class of issue as trials §9 A2.

**Pins revision.** `pins_r2.json` with `--amendment A1 --changed-fields analysis.gate2_float_tie --confirmatory-output-seen false`. Gate 2 is rerun into `gate_reproduction_r2.json`. If 2a still fails, the study stops (§5.2 retry rule).

## 10. Decisions taken from the findings, and deviations
| item | decision | source |
|---|---|---|
| Replication cohort | 5-fold protein-disjoint CV over all 15,155 proteins (option d); option (b) infeasible, (a) rejected (fewer anchors, half the data) | F§4.6, R1 |
| Exposure | all 1,396 test + 697 validation anchors exposed; disclosed in §0.3 | F§4.6, F§7 |
| **Deviation:** replication population | `Bclean` (= F's `unexposed_closed`, ≈ 2,694 hard-defined) is the **primary** cohort-B population for Holm, C2_B and C5_B; all-B is the sensitivity with a sign-disagreement warning. F recommended `unexposed_closed` as a sensitivity; it is promoted because it is large enough to carry the criterion and the task asks for a replication cohort the exploration never touched | review round 0 #3; F§4.6 |
| Fold seed | 2027, not 2026 (2026 used for label counts) | F§7 |
| **Deviation (v4):** primary fusion | AZ-X (within-anchor z, exact pruned BP terms, w by global pooled AUC), the leak-free method declared before test by the best-fusion study; GZ-P (the v3 primary) becomes the pre-declared secondary fusion | user decision after `results_go_ppi_fusion_best.md`; overrides R2 and R4 for the primary |
| **Deviation (v4):** weight rule | AZ forms: global pooled AUC; GZ forms: nDCG@10; each form's other rule as sensitivity; full 0–1.0 grid | the rule each method was test-scored with (§0.1b); R2's objection disclosed (§8) |
| Co-primaries | H1, H2, H3 with Holm(3), now under AZ-X | R3 |
| Primary φ | X (exact) on the pruned index with pruned record URIs; UX reported; UX − X paired. P stays as GZ-P's φ and as C1's comparator | R4 (pruned primary kept; φ mode changed with the fusion) |
| **Deviation (v4):** C1 fixed w | 0.4, the modal exploratory AZ-X weight (was 0.5 for the nDCG form) | §0.1b |
| **Deviation (v4):** gate 2f | the confirmatory code must reproduce the AZ-X and GZ-P test numbers on the published checkpoints | this protocol |
| Shuffle | decile-stratified, candidate-excluding derangement; C2 fail read as density/study bias | F§4.1, R5 |
| **Deviation:** C2 state rule | GO rule gates on Δ_X − Δ_S with a *partial* state; trials-verbatim state reported as `c2_trials_state` only | review round 0 #1; F§4.1 (partner density by grade) |
| **Deviation:** C2 small-component fail branch | branch 2 requires LB95(Δ_S) > 0 (the shuffle kept a confident share), mirroring C5; `c2_rule` v3 | review round 1 #1 |
| **Deviation:** C2 absorption in branch 1 | branch 1 also fails when LB95(Δ_X − Δ_S) ≤ 0 and LB95(Δ_S) > 0; `c2_rule` v4 | review round 2 #1 |
| **Deviation:** C5 fail rule | small-residual fail branch requires LB95(Δ_X − Δ_C5) > 0; branch 1 also fails when LB95(Δ_C5) ≤ 0 and LB95(Δ_X − Δ_C5) > 0; `c5_rule` v3 | review round 0 #2; round 2 #2 |
| `Bclean` size floor | hard-defined `Bclean` ≥ 2,400 under fold seed 2027, else pre-pins amendment | review round 1 #4; F§4.6 |
| **Deviation:** HuRI grade 1 | restricted to HuRI-observed proteins; unrestricted as sensitivity | review round 0 #4 |
| **Deviation:** shuffle forbidden set | adds rule (d) partner-set equality and excludes empty-set anchors, to make "no own or partner set" literal; feasibility re-verified in step 0 | this protocol |
| Exact vs simGIC | C1 = P − X in AZ form, each with own w and both at fixed w = 0.4; N as ladder | R5; v4 |
| **Deviation:** typed variant | F R8 put is_a-only out of scope only because the parser is protected; T added as secondary, implemented outside the protected file with a reproduction check, dropped automatically if the check fails | user request, R8 |
| Study bias | C4 alone, C5 2-D competitor; textmining excluded | F§4.5, R5 |
| **Deviation:** C5 role | made a label gate (labels 3 and 6) rather than report-only, because study bias is the main named threat to the GO claim | this protocol |
| HuRI | separate cohort with matched non-edges; strata descriptive (58 anchors too few) | F§4.4, R6 |
| Gate 2 | A2 tiers from the start; near-tie `selected_weight` exception for 3 cells | R7 |
| Text leak | retrain all 19 on masked text; mention sensitivity | F§3, R1 |
| DE | secondary slope test on log fraction, own α, TOST flat state | user request; F§5 |
| Masked encodes | one fresh remote-CPU encode of every masked text; Mac caches not read for scoring | F§1.4 (cache incomplete) |

## 11. Review findings → resolutions
Rounds 0 and 1 are recorded in v3 notation, where P was the primary φ. In v4, read their "P" / "Δ_P" as the primary fusion's X / Δ_X. Round 2 uses v4 notation.

### Round 0 review (CHANGES_REQUESTED: 0 HIGH, 4 MEDIUM, 7 NIT)
| # | sev | finding | resolution | sections |
|---|---|---|---|---|
| 1 | MEDIUM | trials C2 fail (LB95(Δ_S) > 0) near-predetermined on GO; overrides a confident P − S | **Addressed.** GO C2 rule gates on Δ_P − Δ_S: fail = [LB95(P − S) ≤ 0 and point(P) ≤ point(S)] or UB95(P − S) < 0.005; pass; new *partial* state (does not block, adds suffix a2); trials state kept as reported `c2_trials_state`; worked examples and unit tests; pins `c2_rule`, `c2_fail_pairspecific_ub`, `c2_trials_state_reported`, `final_label_rule` v2 | §1.5, §4.3, §5.1, §7, §8, §10 |
| 2 | MEDIUM | C5 fail fires on a small GO effect that Bz does not absorb | **Addressed.** fail = [LB95(Δ_C5) ≤ 0 and point ≤ 0] or [UB95(Δ_C5) < 0.005 and LB95(Δ_P − Δ_C5) > 0]; the three reviewer examples plus an inconclusive case added and unit-tested; pins `c5_rule` | §4.3, §5.1, §7, §10 |
| 3 | MEDIUM | replication evaluated on all cohort-B anchors (~30% exposed) | **Addressed.** `Bclean` is the primary cohort-B population (Holm with rng [31415, 2], C2_B, C5_B, H1z–H3z); `all_B` sensitivity with rng [31415, 4] and the inverted sign-disagreement warning; power updated; pooled descriptive uses Bclean | §0.2, §3c, §4.1, §4.3, §4.5, §4.6, §5.1, §7, §8, §10 |
| 4 | MEDIUM | HuRI grade 1 not restricted to HuRI-observed proteins | **Addressed.** Grade 1 = HuRI-observed weak-evidence partners with no HuRI edge to the anchor; `huri_grade1_unrestricted` sensitivity file; both counts pinned and asserted in gate 2d | §3d, §4.5, §5.1, §5.2, §7, §8, §10 |
| 5 | NIT | `make_pins.py` reads the GPU name under `CUDA_VISIBLE_DEVICES=""` | **Addressed.** `make_pins.py` (and `--revise`) runs without the mask, asserts one visible device, records torch name and `nvidia-smi` output | header, §6.2 |
| 6 | NIT | training cache vs fresh remote-CPU encodes differ ≈ 1e-6 | **Addressed** (disclosed) | §8 Device and numerics |
| 7 | NIT | Bclean anchors may have exposed partners | **Addressed.** counted at build, pinned, printed; disclosed | §3c, §7, §8 |
| 8 | NIT | rule (e) checks only the anchor's own set | **Addressed** (optional form): donor–partner-set simGIC ≥ 0.5 counts logged in the shuffle manifest and pinned; not a forbidden rule | §2.3, §7 |
| 9 | NIT | gate 2 validates only the U path | **Addressed.** gate 2e: label-free float-equality check of P/X/N/S against direct `GoMatcher` calls | §5.2 |
| 10 | NIT | C5 2-D grid more flexible than comparator | **Addressed** (disclosed as conservative) | §8 |
| 11 | NIT | cohort-B seed index is an arbitrary pairing label | **Addressed** | §4.1 |

Round 0 resolutions change no hypothesis, estimand, weight rule, φ variant or cohort construction; they change decision-rule details (C2, C5), which subset of the already-planned cohort B carries the criterion, and the HuRI grade-1 definition. Row 1's fail branch 2 is superseded by round 1 #1 below.

### Round 1 review (`design_review_go_ppi_fusion.{json,md}`, CHANGES_REQUESTED: 0 HIGH, 1 MEDIUM, 4 NIT)
| # | sev | finding | resolution | sections |
|---|---|---|---|---|
| 1 | MEDIUM | C2 fail branch UB95(Δ_P − Δ_S) < 0.005 fires on a small, fully pair-specific gain | **Addressed** as the reviewer specified. fail = [LB95(P − S) ≤ 0 and point(P) ≤ point(S)] or [UB95(P − S) < 0.005 and LB95(Δ_S) > 0]. The four existing examples keep their states (example 3 now cites both conditions). Two new unit-tested examples: Δ_S +0.0003 [−0.0010, +0.0015], P − S +0.0037 [+0.0020, +0.0048] → *pass*; Δ_S +0.003 [−0.001, +0.007], P − S +0.003 [+0.001, +0.0045] → *inconclusive*. Fail-over-partial order stated. Pins `c2_rule = go_pair_specific_v3`, `c2_fail_requires_absorption = true`, `final_label_rule = ordered_7_label_v3` | §1.5, §4.3, §5.1, §7, §10 |
| 2 | NIT | suffix (a) reach "A for labels 2-7" unreachable at label 2 | **Addressed.** Suffix (a) reach is now A 3-7, B 6-7 (B 5 was unreachable for the same reason); reach note rewritten | §4.3 |
| 3 | NIT | z-stat / w source for S and C5 on HuRI implicit | **Addressed.** S uses cohort-A deranged-validation z-stats and w on the HuRI file's own shuffle map; C5 uses cohort-A 2-D (w_g, w_b) | §2.5, §3d |
| 4 | NIT | no assert band on hard-defined `Bclean` count under seed 2027 | **Addressed.** Build assert ≥ 2,400 (pre-pins amendment otherwise), checked again in gate 2d, pinned `bclean_min_hard_defined` | §3c, §4.6, §5.2, §7, §10 |
| 5 | NIT | §11 status and rule versions stale | **Addressed.** This table; versions bumped consistently in §7 | §7, §11 |

Round 1 resolutions are decision-rule corrections and a build assert. They change no hypothesis, estimand, cohort, φ variant, weight rule or threshold value.

### Round 2 review (`design_review_go_ppi_fusion.{json,md}` as of 20:57, CHANGES_REQUESTED: 0 HIGH, 2 MEDIUM, 2 NIT)
The rows use v4 notation (Δ_X for the primary fusion); the rules are the reviewer's, unchanged in substance.

| # | sev | finding | resolution | sections |
|---|---|---|---|---|
| 1 | MEDIUM | C2: a confident density component with no confident pair-specific component falls to *inconclusive*, so label 7 is reachable | **Addressed** as specified. Fail branch 1 := LB95(Δ_X − Δ_S) ≤ 0 and [point(Δ_X) ≤ point(Δ_S) or LB95(Δ_S) > 0]; branch 2 kept. The reviewer's example (Δ_X +0.020, Δ_S +0.018 [+0.014, +0.022], X − S +0.002 [−0.003, +0.007] → *fail*) is added. All six earlier examples were re-checked and keep their states; example 3 now fires both branches. `c2_rule = go_pair_specific_v4` | §1.5, §4.3, §5.1, §7, §10 |
| 2 | MEDIUM | C5: Bz absorbs a confident share while the residual is not confidently > 0 → *inconclusive*, so label 7 is reachable | **Addressed** as specified. Fail branch 1 := LB95(Δ_C5) ≤ 0 and [point(Δ_C5) ≤ 0 or LB95(Δ_X − Δ_C5) > 0]. The reviewer's example (Δ_X +0.020, Δ_C5 +0.002 [−0.002, +0.006], LB95(Δ_X − Δ_C5) > 0 → *fail*) is added. `c5_rule = absorbed_share_v3`, `final_label_rule = ordered_7_label_v4` | §4.3, §5.1, §7, §10 |
| 3 | NIT | C5 inconclusive example omits Δ_X / LB95(Δ_X − Δ_C5) | **Addressed.** The example states Δ_X = +0.003 and LB95(Δ_X − Δ_C5) ≤ 0 | §4.3 |
| 4 | NIT | interpretive text and round log must follow findings 1–2 | **Addressed.** §1.5 C2 wording rewritten; this table | §1.5, §11 |

## 12. v3 → v4 change list (primary fusion changed; for the round-3 reviewer)
The trigger was the user's decision to adopt the best-fusion study's leak-free winner as the confirmatory method. Nothing is pinned, no confirmatory code exists, and no retrained model, cohort-B fold or HuRI file has been built or scored, so this is a pre-pins revision, not a §9 amendment.

| area | v3 | v4 | sections |
|---|---|---|---|
| Primary fusion | GZ-P: global z, pruned simGIC, w by validation nDCG@10 | **AZ-X**: within-anchor z, exact pruned BP-term Jaccard, w by validation global pooled AUC | header, §1, §2.5, §2.6, §7 |
| Secondary fusion | none | GZ-P with its own Holm triple (H1g–H3g), S1g, AZ-X − GZ-P head-to-head; never in a label | §1.2b, §4.1–§4.3 |
| Primary φ | P (pruned simGIC) | X (pruned exact); P kept for GZ-P and C1 | §2.2 |
| New φ | — | UX (unpruned exact), S as exact on the deranged map, S_P (simGIC on the same map) | §2.2, §2.3 |
| Controls | in GZ form with nDCG rule | in the form of the controlled fusion (AZ, pooled-AUC rule); C1 = P − X, C3 = UX − X, C5 = text + X + Bz | §1.5, §2.6 |
| C1 fixed w | 0.5 | 0.4 | §2.6, §7 |
| Normaliser sensitivity | within-anchor z | GZ-X (global z) | §2.5, §4.5 |
| Weight-rule sensitivity | pooled AUC for P and U | nDCG@10 for AZ-X; pooled AUC for GZ-P | §2.6, §4.5 |
| C2 / C5 rules | v3 / v2 | v4 / v3 (round 2) | §4.3 |
| Gate 2 | 2a–2e | adds 2f: reproduce the AZ-X and GZ-P test numbers on the published checkpoints | §5.2 |
| Exposure | items 1–8 | item 9: the best-fusion search and its one test scoring; AZ-X was chosen on cohort A's validation file | §0.1b, §0.2, §0.3, §8 |
| Inputs pinned | — | `results/go_ppi_fusion_best/{test_results,finalists,test_plan_amendment}.json` | §7 |
| Mac dev smoke | U on published files only | also AZ-X and GZ-P on published files (gate 2f); no further variants | §6.5 |

Unchanged: cohorts A and B, `Bclean` and its floor, the HuRI cohort construction, the shuffle construction, B, the RNGs, Holm(3), TOST bounds, the 7-label order, the training design (34 runs, masked text, L40S) and every file path.

