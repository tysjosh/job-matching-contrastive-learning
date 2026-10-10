# GO/PPI withholding study: smoke report (plan items 24–26, protocol G1–G4 and G6)

Run on 2026-10-10 (machine clock), Apple M2, CPU only, in place at `/Users/olukotunjosh/Downloads/CDCL` (HEAD `c911188`, nothing committed). G1–G4 ran once, with the real pins file. The G6 smoke ran p01_k0 × 4 cells × seed 13 at 2 epochs. Every gate passed, and no protocol stop fired. G5 and the grid (G7) were not run.

Code hash throughout: `code_sha256 = 592125a33d577126fec7b06003d99bfbdd3e1afa8df2edd8909d796880efa258` (219 files). It is the same at G1, in the λ pilot, in every smoke run and at the end.

## 1. Regression gate (G0), before and after

| check | before the gates | after the smoke |
|---|---|---|
| `pytest go_ppi_domain/tests -q` | 335 passed, 3 skipped (22.9 s) | 335 passed, 3 skipped (22.9 s; 25.2 s after the §13 pin mirror) |
| four career files | 28 passed (3.9 s) | 28 passed (4.2 s) |
| `GO_WITHHOLDING_SLOW=1` slow suite | 3 passed (64.0 s) | not re-run (no code changed) |
| `check_additive.py` | exit 0, `check_additive: OK` | exit 0 |
| protected `preprocess/go_ppi` + `go_ppi_splits` (11 files) vs `pre_edit/protected_sha256.txt` | all OK | all OK |

`contrastive_learning/{batch_processor,loss_engine,trainer}.py` were not touched. Tracked modifications are still the same 6 files.

## 2. G1: pruned features and leak audit

`build_pruned_features.py` exited 0 in 11.6 s. GAF lines: 906,445; kept BP rows 126,148; with/from-human drops 12,828. Lexicon: 52,404 entries; token mask rate 3.33%. Splits: train 15,708, validation 5,587, test 22,341, pools 7,041. Pairs sha `f789f237…` (copied verbatim). `measure_leakage.py` exited 0 in 9.9 s, with `pass: true`, `failures: []` and reference sanity matching (a = 0.1107/0.0017/0, c = 0.0592/0.0041/0).

Leak rates before (reference `preprocess/go_ppi*`) and after (pruned `preprocess/go_withholding`). a = symbol mention, b = symbol or alias, c = with/from hit (any aspect), c_P = BP only.

| split, grade | n | a before → after | b before → after | c before → after | c_P before → after |
|---|---|---|---|---|---|
| test, 0 | 11,165 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| test, 1 | 10,011 | 0.17% → 0 | 0.19% → 0 | 0.41% → 0 | 0.20% → 0 |
| test, 2 | 1,165 | **11.07% → 0.17%** (2) | **12.62% → 0.17%** (2) | **5.92% → 0** | **3.95% → 0** |
| train, 2 | 15,708 | 11.17% → 0 | 12.69% → 0.08% (12) | 7.31% → 0 | 4.88% → 0 |

Gate (test, grade 2, a/b/c ≤ 0.5% and ≤ 0.5 pp above grade 1): passed. Residuals (metric d, descriptive, not removed) are unchanged by design. d_complex_name on test is 16.3% / 16.5% / 28.1% (grades 0/1/2). d_complexportal on test is 0.65% → 0.53%, 0.72% → 0.59% and 1.29% → 1.29%.

Pins: `G1`, `G1-leak`.

## 3. G2: targets, eval cache, covariate, target-free validation

**Selection** (`build_target_table.py`, exit 0, 2.5 s): 461 eligible candidates, then 153 non-nested, then strata at the 33rd/67th percentiles of S_sib (low ≤ 1,339.48, high ≥ 3,624.52), giving 51 low and 51 high. 12 matched pairs were accepted, so **24 singleton targets (12 low, 12 high)** in 24 packs, 121 conditions and 2,420 runs. No shortfall; 39 left in the replacement queue. Balance: SMD log n −0.048, log support −0.036. Per-target k = 0 removal is 1.46–4.95% (cap 8%). Retention-eligible is 40–142 (minimum 25).

| pack | GO id | name | stratum | pair | n(t) | S_sib | k0 removal | ret.-elig. | test pos/weak | tsim_excess |
|---|---|---|---|---|---|---|---|---|---|---|
| p01 | GO:0017157 | regulation of exocytosis | low | m01 | 61 | 752 | 2.25% | 57 | 16/58 | 0.0259 |
| p02 | GO:0006109 | regulation of carbohydrate metabolic process | high | m01 | 61 | 8808 | 3.45% | 57 | 21/72 | 0.0227 |
| p03 | GO:1900180 | regulation of protein localization to nucleus | low | m02 | 61 | 1131 | 3.90% | 60 | 22/107 | 0.0033 |
| p04 | GO:0140053 | mitochondrial gene expression | high | m02 | 61 | 6117 | 3.25% | 49 | 24/110 | 0.0766 |
| p05 | GO:0030336 | negative regulation of cell migration | low | m03 | 93 | 811 | 3.80% | 83 | 15/120 | 0.0581 |
| p06 | GO:0019216 | regulation of lipid metabolic process | high | m03 | 91 | 8633 | 4.87% | 88 | 21/102 | 0.0213 |
| p07 | GO:0050804 | modulation of chemical synaptic transmission | low | m04 | 110 | 0 | 4.25% | 94 | 16/89 | n/a |
| p08 | GO:0009611 | response to wounding | high | m04 | 115 | 5607 | 4.31% | 104 | 14/90 | 0.0055 |
| p09 | GO:0006310 | DNA recombination | low | m05 | 110 | 1333 | 4.95% | 93 | 39/128 | 0.1585 |
| p10 | GO:0030111 | regulation of Wnt signaling pathway | high | m05 | 110 | 4603 | 4.87% | 99 | 33/169 | 0.0434 |
| p11 | GO:0006913 | nucleocytoplasmic transport | low | m06 | 104 | 0 | 4.00% | 75 | 28/123 | n/a |
| p12 | GO:0008283 | cell population proliferation | high | m06 | 103 | 20888 | 3.29% | 99 | 28/138 | 0.0170 |
| p13 | GO:0007178 | cell surface receptor protein serine/threonine kinase signaling pathway | low | m07 | 55 | 586 | 1.46% | 40 | 11/53 | 0.0670 |
| p14 | GO:0120161 | regulation of cold-induced thermogenesis | high | m07 | 52 | 4023 | 2.30% | 51 | 13/70 | 0.0224 |
| p15 | GO:0045931 | positive regulation of mitotic cell cycle | low | m08 | 53 | 1332 | 3.02% | 50 | 23/76 | 0.0589 |
| p16 | GO:0048511 | rhythmic process | high | m08 | 50 | 22286 | 3.16% | 47 | 20/76 | 0.0246 |
| p17 | GO:0030036 | actin cytoskeleton organization | low | m09 | 165 | 973 | 4.89% | 142 | 35/199 | 0.0387 |
| p18 | GO:0035295 | tube development | high | m09 | 158 | 4419 | 4.51% | 131 | 31/176 | 0.0224 |
| p19 | GO:0006936 | muscle contraction | low | m10 | 92 | 40 | 4.86% | 75 | 30/130 | −0.0307 |
| p20 | GO:1904951 | positive regulation of establishment of protein localization | high | m10 | 90 | 9943 | 4.34% | 85 | 28/127 | 0.0230 |
| p21 | GO:0007018 | microtubule-based movement | low | m11 | 126 | 847 | 4.60% | 93 | 40/214 | 0.0749 |
| p22 | GO:0002764 | immune response-regulating signaling pathway | high | m11 | 117 | 5403 | 4.66% | 104 | 42/198 | 0.0433 |
| p23 | GO:0070936 | protein K48-linked ubiquitination | low | m12 | 52 | 223 | 2.91% | 47 | 31/108 | 0.2135 |
| p24 | GO:0001503 | ossification | high | m12 | 53 | 5759 | 2.25% | 49 | 21/92 | 0.0182 |

**Eval cache** (`build-cache`): 3,014 test proteins, 22,341 records, max_tok 262 (< 512), frozen global Spearman 0.1702. See deviation D1: the CPU rebuild took 121.1 s and is the pinned cache (`abf715f9…`).

**Covariate** (`text_similarity_covariate.py`, exit 0, 391 s): 24 targets; 10,601 train texts encoded with frozen MPNet (0 from the trainer cache, which did not exist yet).

**Target-free validation** (`make_conditions.py --mode validation`, exit 0, 0.5 s): V = 330 proteins; 3,011 records kept, 2,576 dropped; 181 grade-2 records; 491 pool rows (206 dropped).

Pins: `G2-targets`, `G2-cache` (two entries, see D1), `G2-covariate`, `G2-validation`.

## 4. G3: conditions

`make_conditions.py --mode conditions` exited 0 in 136 s. It wrote all 121 conditions (4.0 GB) with exactly-k and residual scans clean, and 0 anchors dropped by the empty-pool rule in any condition. Pin: `G3` (121 manifest sha256s).

| k | conditions | removal frac min / median / max | withheld proteins min / median / max |
|---|---|---|---|
| 0 | 24 | 1.46% / 3.95% / 4.95% | 63 / 136 / 294 |
| 1 | 24 | 1.45% / 3.94% / 4.94% | 62 / 135 / 293 |
| 5 | 24 | 1.34% / 3.63% / 4.76% | 58 / 131 / 289 |
| 10 | 24 | 1.16% / 3.46% / 4.52% | 53 / 126 / 284 |
| 25 | 24 | 0.62% / 2.74% / 4.17% | 38 / 111 / 269 |

- p01_k0: 103 withheld (61 positive, 42 pool-only), removal 2.25%, 15,316 train records.
- p01_k5: 98 withheld, 5 retained (14 undirected positive pairs), removal 2.13%, 15,338 records.
- `all`: 15,708 records, 0 withheld.

## 5. G4: configs and λ pilot

`materialize_configs.py --lambda none` exited 0 and wrote baseline, neg_only and `loss_pilot_template` (no pin). The baseline-vs-template audit has an empty diff set, as expected at weight 0.

`lambda_pilot.py` exited 0 in 1,004 s. That total includes the one-off ~9.5 min warm-up of the shared frozen-encoder cache `embedding_cache/go_withholding_text_embeddings.pt` (60 MB, 11,619 views) in the first pilot run. Pilot setup: condition `all`, seed 42, 3 epochs, selector epoch 2, 181 grade-2 target-free validation records in 5 batches of 32 (≥ 4).

| role | λ | validation ratio λ·‖∇L_GO‖/‖∇L_InfoNCE‖, per batch | median |
|---|---|---|---|
| pilot | 0.1 | 0.0441, 0.0301, 0.0397, 0.0637, 0.0475 | **r₀ = 0.0441** |
| confirm | 0.5 | 0.1611, 0.0949, 0.1351, 0.2283, 0.1538 | **r* = 0.1538** ∈ [0.10, 0.30] |

λ* = 0.1 × 0.20 / 0.0441 = 0.453, which rounds in log space to **λ = 0.5**. Confirmed on the first try; no re-estimate needed.

Training-diagnostic medians by epoch (0/1/2) were 0.044 / 0.049 / 0.047 at λ 0.1 and 0.163 / 0.178 / 0.166 at λ 0.5.

`lambda_pilot.py` then ran `materialize_configs.py --lambda 0.5` itself and wrote the four cell configs. The factor audit diffs were exact: baseline vs loss_only `{go_ontology_loss_weight}`; baseline vs neg_only and loss_only vs both `{go_ppi_go_tiered_negatives, go_ppi_tier_sampling}`; neg_only vs both `{go_ontology_loss_weight}`. Pins: `G4-lambda`, `G4-configs` (one entry). See D3.

## 6. G6 smoke: p01_k0 × 4 cells × seed 13, 2 epochs

The protocol §10 G6 block was run verbatim into `results/go_withholding/smoke/`. All exit codes were 0.

| cell | train wall | score wall | non-ok assertions | ok_batches (ep 0, 1) | L_GO applied batches | selection sha (ep 0 / ep 1) |
|---|---|---|---|---|---|---|
| baseline | 44 s | 6 s | `9_relation_table: skipped (no table)` (by design, weight 0) | 240, 240 | 0 | `63219af4…` / `591c86c2…` |
| loss_only | 44 s | 5 s | none | 240, 240 | 480 | `63219af4…` / `591c86c2…` |
| neg_only | 48 s | 5 s | `9_relation_table: skipped (no table)` (by design) | 240, 240 | 0 | `76135c77…` / `775db206…` |
| both | 50 s | 5 s | none | 240, 240 | 480 | `76135c77…` / `775db206…` |

- **Runner checks.** Assertions `1_hashes`…`10_callbacks` passed in all four cells, plus every `post_*` check (seen ∩ W = ∅, 0 missing ids, 0 selector None returns and 0 exceptions, every epoch guarded, import coverage ⊆ code pin, `checkpoint_epoch_1.pt` with epoch 1, no emergency checkpoint). Device: CPU for trainer, loss engine, text encoder and projection. Unannotated-anchor window calls: 323 per epoch. withheld_count 103.
- **Recorder identities.** baseline = loss_only and neg_only = both hold per epoch. baseline ≠ neg_only.
- **`check-group`** exited 0 with `failures: []` and wrote `GROUP_OK.json` into all four smoke dirs.
- **Evaluator** scored `checkpoint_epoch_1.pt` in each cell. p01 target hard AUC: baseline 0.8685, loss_only 0.8675, neg_only 0.8588, both 0.8578 (16 positive / 58 weak test records, 10 queries). Global hard AUC: 0.7769 / 0.7769 / 0.7719 / 0.7715. These are 2-epoch smoke values and are not interpreted.
- **Encoder cross-check** (`--encoder-crosscheck`, 9 s) exited 0. The 50 re-encoded texts gave min cos 0.99999999999963 (≥ 0.9999). Token ids: 0 mismatches over all 3,014 test proteins. `n_text_differs` = 700, which is descriptive (whitespace only; the design guessed ≈ 3.5k).
- **Gradient scale** (eval-mode diagnostics, 8 batches per epoch). λ·‖∇L_GO‖/‖∇L_InfoNCE‖ median: loss_only 0.171 / 0.182, both 0.192 / 0.215, baseline and neg_only 0. Per-batch range 0.135–0.232. Loss means in `both`: InfoNCE 0.955 / 1.108, L_GO 0.138 / 0.161. ‖∇InfoNCE‖ median ≈ 4.2, unweighted ‖∇L_GO‖ ≈ 1.4–1.8. max_component_drift is 0 in every cell, and no drift flags were raised. `share_above_clip_eval_diag` = 1.0 in every row (pre-clip total norm ≈ 4.4 > 1.0). This is the known clipping limitation (protocol §12).
- **Relation preservation** (`--relation`): global Spearman(cos, simGIC) was 0.240 / 0.226 / 0.168 / 0.159. p01 neighbour excess was 0.037 / 0.041 / 0.025 / 0.030.
- **Offline audits** (`audit_negatives.py`, all exit 0, 17–18 s each, epochs 0/7/14, seed 13). The W gate holds: `share_withheld_negatives` = 0 in every cell, epoch and condition. Identity holds in all three conditions (baseline == loss_only, neg_only == both).

  | condition | retained share (base / neg) | target-annotated share (base / neg) | neighbour-negative share (base / neg), ep 0 → 14 |
  |---|---|---|---|
  | all | 0 / 0 | 25.6–29.9% / 32.0–35.2% | 44.7 → 53.8% / 54.2 → 62.5% |
  | p01_k0 | 0 / 0 | 0 / 0 | 1.78 → 2.23% / 2.07 → 2.56% |
  | p01_k5 | 0.05–0.06% / 0.07–0.09% | = retained share | 1.78 → 2.22% / 2.06 → 2.54% |

  p01_k5 exercises the retained-protein case: retained proteins are sampled as negatives, while withheld ones never are.
- **Code-pin coverage.** `post_import_coverage` is ok in every run, and the evaluator's coverage check passed.

## 7. Statistics in-process (analyze_withholding library)

The four smoke dirs were symlinked under their grid run ids (`p01__k0__<cell>__s13`) in `/tmp`. Then `build_long_table` → `paired_deltas` → `two_way_bootstrap` were run with B = 1000 and seed 20251009, using `groups = {("p01_k0", 13): "checked"}` (see D6).

- Long table: 4 rows (target GO:0017157, k 0, 4 cells, seed 13). tsim_excess 0.0259, stratum low.
- Δ hard AUC vs baseline at k = 0: loss_only −0.00108, neg_only −0.00970, both −0.01078. E_c (Δ minus collateral Δ): −0.00102 / −0.00515 / −0.00573. Δ_small and D are undefined, as expected with only k = 0 present.
- Two-way bootstrap with 1 unit, 1 pack, 1 seed: CI equals the point estimate, 0 replicates discarded.
- Two-way bootstrap in the full table shape (24 units, 24 packs, only p01 defined): the same estimate, 396/1000 replicates discarded (p01 not drawn; expected ≈ (23/24)²⁴ ≈ 36%), so `unstable: true`. This exercises the discard rule.
- Wall time: 0.05 s.

## 8. Sweep driver: sharding, kill and resume

`sweep.py run` needs the G5 pin, which this workflow does not create (plan decision 7). So the real `sweep.main` was called through a 20-line wrapper that set `wc.STUDY_RES = /tmp/smoke_sweep/res`. That directory held a copy of the real pins file, symlinks to the real eval cache and λ pilot, and a scratch-only `code_pin.json` + G5 entry written by the real `sweep.py pin`: code_sha256 `592125a3…`, git_head `c911188…`, 219 files, import probe covered. See D4. The real `results/go_withholding/` still has no `code_pin.json` and no G5 entry.

- **Sharding** (`plan --shard i/3`): 808 / 808 / 804 runs (sum 2,420). The three shards are pairwise disjoint and their union is the grid. Each group's four cells stay in one shard (202 / 202 / 201 groups). p01_k0 s13 is group 1, so it lands in shard 2/3.
- **Kill.** `run --max-runs 1 --shard 2/3 --seeds 13 --packs p01 --ks 0 --cells baseline` (15 epochs, the sweep's `GRID_EPOCHS`) was SIGKILLed as a whole process group during epoch 3. The run dir was left with `RUNNING.json` (dead pid), and `plan` showed `n_interrupted: 1`.
- **Restart** with the same command: exit 0. The log shows "interrupted run renamed to p01__k0__baseline__s13.interrupted1; restarting". The retrain went cleanly to `DONE.json`: train 292.5 s, eval 5.3 s, global hard AUC 0.7947. `sweep_log.jsonl` has 3 lines: `interrupted_restart`, train attempt 1 exit 0, eval attempt 1 exit 0. The `.interrupted1` dir is kept and never analysed.
- **Idempotent.** The same command again gave exit 0 with `started 0`. The other shard (1/3) with the same filters also gave exit 0, `started 0`. The run dir's files were byte-identical before and after (hash of all file sha256s), and `sweep_log.jsonl` stayed at 3 lines. `status --shard 2/3`: progress 1/808, 0 interrupted, 0 in progress, 0 unchecked, 0 group-failed.

`/tmp/smoke_sweep` was deleted afterwards.

## 9. Leakage assertions fire on a leaked input

`p01_k0` was copied to `/tmp/leak_test`. One record from the base train split whose anchor (`resume.protein_id`) is a W protein was appended. Then the runner was run with the smoke's arguments (baseline, 2 epochs).

- **Required case.** The copy's manifest `files.train.jsonl` sha/lines were updated to match the leaked file, so that check 1 passes. Result: **exit 2** in 8 s, and `run_error.json` showed `stage: condition_assertions`, `error: "RuntimeError: check 2: withheld protein in the train file (first ids: ['CHMP2A'])"`. Training never started.
- **Extra case.** With the manifest left untouched, plain appending is caught one check earlier: exit 2, `check 1: train file sha256 differs from the manifest`.

`/tmp/leak_test` was deleted.

## 10. Wall times (CPU, Apple M2)

| step | wall |
|---|---|
| G0 pytest / career / slow | 22.9 s / 3.9 s / 64.0 s |
| G1 build_pruned_features / measure_leakage | 11.6 s / 9.9 s |
| G2 build_target_table | 2.5 s |
| G2 build-cache (MPS, discarded) / (CPU, pinned) | 48.9 s / 121.1 s |
| G2 text_similarity_covariate | 391 s |
| G2 validation / G3 conditions | 0.5 s / 136 s |
| G4 materialize none / λ pilot (2 runs + measurement + materialize) | 0.04 s / 1,004 s |
| G6 train ×4 (2 epochs) / score ×4 | 44–50 s / 5–6 s |
| G6 crosscheck / check-group / audits ×3 | 9 s / < 1 s / 17–18 s each |
| one grid-shaped run (15 epochs, p01_k0, warm cache) | train 292.5 s + eval 5.3 s |
| leak-assertion run | 8 s |

On CPU a grid run costs about 300 s. Grid CPU estimate: 2,420 × 300 s ≈ 200 h serial, which is inside the design's 136–257 h range.

## 11. Files created or modified

Created (study artifacts; `preprocess/`, `results/` and `embedding_cache/` are gitignored):

- `preprocess/go_withholding/converted/`: `convert_manifest.json`, `go_index_P.pkl`, `mask_lexicon.json`, `mask_manifest.json`, `pairs.jsonl`, `proteins.jsonl`
- `preprocess/go_withholding/splits/`: `frozen_hashes.json`, `split_manifest.json`, `train.jsonl`, `validation.jsonl`, `test.jsonl`, `negative_pools.jsonl` (made read-only by G1)
- `preprocess/go_withholding/leak_audit.json`
- `preprocess/go_withholding/targets/`: `target_table.json`, `target_table.json.sha256`, `candidates.json`, `text_covariate.json`
- `preprocess/go_withholding/validation_targetfree/`: `manifest.json`, `validation.jsonl`, `validation_positive.jsonl`, `negative_pools_validation.jsonl`
- `preprocess/go_withholding/conditions/<121 dirs>/`: `train.jsonl`, `negative_pools.jsonl`, `withholding_manifest.json`
- `results/go_withholding/protocol_pins.json`: G1, G1-leak, G2-targets, G2-cache ×2, G2-covariate, G2-validation, G3, G4-lambda, G4-configs (no G5)
- `results/go_withholding/eval_cache/test_4027e9a0a3cc6ec1.npz`, `.meta.json`
- `results/go_withholding/lambda_pilot/`: `l0.1/`, `l0.5/`, `lambda_choice.json`
- `results/go_withholding/smoke/{baseline,loss_only,neg_only,both}/`: checkpoint, `lc_manifest.json`, `go_diagnostics.jsonl`, `GROUP_OK.json`, `eval/` (baseline's `eval/` also has `encoder_crosscheck.json` and the cross-check trainer's `crosscheck_tmp/logs`)
- `results/go_withholding/audits/neg__{all,p01_k0,p01_k5}__{baseline,loss_only,neg_only,both}__s13.json` (12 files)
- `config/go_withholding/`: `baseline.json`, `loss_only.json`, `neg_only.json`, `both.json`, `loss_pilot_template.json`, `configs.sha256.json` (untracked, not ignored; rsync step 1a ships them)
- `embedding_cache/go_withholding_text_embeddings.pt` (shared frozen-encoder cache, 60 MB)
- this report

Modified:

- `protocol_go_ppi.md` §13 pins table: G1–G4-configs mirrored from `protocol_pins.json` (G5 blank). This is a hash record, not a protocol change (protocol header). `parse_amendment_rows` still returns exactly AM1.
- `plan_go_ppi.md`: items 24–26 ticked; verification-log entry.

No `.py` file was changed. Temporary `/tmp/smoke_logs`, `/tmp/smoke_sweep`, `/tmp/smoke_stats` and `/tmp/leak_test` were deleted.

## 12. Deviations and observations

- **D1. The eval cache was rebuilt on CPU.** The verbatim `build-cache` command (no `--device`) let SentenceTransformer pick **MPS** (the Apple GPU), which conflicts with the CPU-only instruction. Before anything consumed it, that cache was moved aside and rebuilt with `--device cpu`. The pins file is append-only, so it now holds two `G2-cache` entries. The sweep preflight uses the latest one (CPU, `abf715f9…`); the first entry's npz no longer exists. MPS vs CPU embeddings: min cos 0.9999999, same ids. Frozen Spearman 0.1701847 (MPS) vs 0.1701850 (CPU).
- **D2.** `text_similarity_covariate.py` got `--device cpu` appended for the same reason. The trainer, pilot, smoke and audits pick only CUDA or CPU, so they ran on CPU unmodified.
- **D3. The final G4 line was not run separately.** `lambda_pilot.py` runs `materialize_configs.py --lambda <λ>` itself as its step 5, so that run is the protocol's final G4 command. Running it again would have appended a second `G4-configs` pin, against protocol §9 G4 ("the single G4-configs pin") and runbook §4.
- **D4. Sweep tested in a scratch dir.** Sharding and resume were exercised through the real `sweep.main` against a scratch STUDY_RES with a scratch-only G5 pin, because G5 is not run. The resumed run used the grid's 15 epochs.
- **D5. Manifest edited in the leaked copy.** To reach check 2, the copy's manifest train sha was made consistent. Plain appending is caught by check 1 first; both cases are shown.
- **D6. Smoke group marked checked by hand.** `classify_groups` reports the smoke group as `incomplete` because the smoke dirs have no `DONE.json`, which only the sweep writes. The group was set to `checked` by hand, since check-group had passed.
- **Observation: no siblings for two targets.** p07 (GO:0050804) and p11 (GO:0006913) have S_sib = 0 (no sibling neighbours), so `tsim_excess` is null for them. They sit in the low stratum and are valid. S4 (the covariate-adjusted sensitivity) will have 22 of 24 covariate values unless the analysis handles nulls; the analysis code was not changed here.
- **Observation: pin fields.** The `G4-lambda` pin holds λ, r₀, r* and the `lambda_choice.json` sha. n_records (181) and n_batches (5) are in `lambda_choice.json`, which is covered by that sha, not written inline as protocol §11 lists them.
- **Observation: clipping.** `share_above_clip_eval_diag` = 1.0: every diagnostic batch's pre-clip norm exceeds 1.0. This is already a stated limitation (protocol §12), reported, not gated.

## 13. Remote run (A100): exact commands

The full guide is `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_GO_PPI.md`. G0–G4 and the G6 smoke are done locally, so the remote sequence is:

1. **Local, before leaving.** Commit or tag the code, or explicitly accept the content pin (protocol G5). Note the hash:
   ```bash
   .venv/bin/python -c "from go_ppi_domain import withholding_constants as wc; print(wc.code_sha256()[0])"   # 592125a3… today
   ```
2. **Copy.** Runbook §1 rsync blocks 1a–1e, verbatim:
   - 1a: code, `.git`, configs, protocol.
   - 1b: `preprocess/go_withholding/{converted,splits,targets,validation_targetfree}` plus `preprocess/go_ppi`, `preprocess/go_ppi_splits`.
   - 1c: `results/go_withholding/protocol_pins.json`, `eval_cache`, `lambda_pilot`.
   - 1d: `dataset/go_ppi`.
   - 1e: the MPNet snapshot.
3. **Environment and G0.** Runbook §2: venv, `torch==2.8.0` cu128, `requirements-remote.txt`, `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`. Then G0 and the code-hash check, which must equal the local value.
4. **Conditions.** Runbook §3: `"$PY" "$S/regenerate_conditions.py"` (verifies G1/G2/G3; about 4.4 GB, needs ≥ 5 GB free).
5. **G5 and checks.** Runbook §5: `"$PY" "$S/sweep.py" pin`. Optionally, the G6 block once more into `results/go_withholding/smoke_a100/` with `--device cuda` (a device check, not a gate). Then `sweep.py run --max-runs 1 --dry-run` and `sweep.py plan`.
6. **Grid.** Runbook §6: warm up with `"$PY" "$S/sweep.py" run --max-runs 1 --device cuda`, then the sharded or parallel loop. Runbook §7 covers monitoring and resume; §8 covers audits and analysis.

G5 and G7 commands from protocol §10 (shown, **not executed**):

```bash
R=/Users/olukotunjosh/Downloads/CDCL; PY="$R/.venv/bin/python"; S="$R/scripts/go_withholding"; cd "$R"

# G5 (after the user commits/tags, or explicitly accepts the content pin)
"$PY" "$S/sweep.py" pin

# G7 (repeat `run` until `status` shows n_runs/n_runs, i.e. 2420/2420 or 2020/2020 under AM1; seed-major order)
"$PY" "$S/sweep.py" plan
"$PY" "$S/sweep.py" run --max-runs 40
"$PY" "$S/sweep.py" status   # also group-checks any complete, unmarked group; must end with n_groups_unchecked = 0
# offline negative audits for every condition, seed 13 (cheap)
for D in "$R"/preprocess/go_withholding/conditions/*/; do
  "$PY" "$S/audit_negatives.py" --cond "$(basename "$D")" --cells baseline loss_only neg_only both --seed 13 --epochs 0 7 14
done

# Analysis
"$PY" "$S/analyze_withholding.py" --runs "$R/results/go_withholding/runs" \
  --target-table "$R/preprocess/go_withholding/targets/target_table.json" \
  --covariate "$R/preprocess/go_withholding/targets/text_covariate.json" \
  --out "$R/results/go_withholding/analysis" --seed 20251009 --B 10000
```

On the remote, use `R="$PWD"` from the checkout (runbook prefix) instead of the local absolute path.
