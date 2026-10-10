# Preregistered protocol: per-target withholding of GO BP terms in GO/PPI interaction retrieval

Version 6 (revision after the fifth `design_review.json`; changes listed in §14). This protocol is fixed before any study training run. The technical design is `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/design_go_ppi.md`, cited as "design §n". `$R` is the literal absolute path `/Users/olukotunjosh/Downloads/CDCL`; every path below is absolute.

Freezing works in two steps:

- Every criterion, metric, seed, threshold and rule in this document is frozen when it is approved.
- Artifact hashes marked "recorded at gate" are written by the scripts to `$R/results/go_withholding/protocol_pins.json` at the gate named, and mirrored by hand into §13. Recording a hash is not a protocol change. No criterion, metric, seed or threshold may change after gate G5.

## 1. Question and hypotheses

Does ontology knowledge help a text-only protein-interaction retriever on GO processes whose training proteins were withheld, and does that help depend on how much training data exists for *neighbouring* GO terms?

Two ontology mechanisms, crossed 2×2:

- **(L) GO relation-preservation loss.** Pulls training proteins with related BP annotations together in embedding space.
- **(N) GO-guided hard negatives.** Negatives are sampled from the GO-closest window of each anchor's graded pool.

For target t, retention level k and seed s, Δ_c(t,k,s) = per-target hard AUC of cell c minus that of baseline, same condition and seed.

- **H1 (performance).** For each c ∈ {loss_only, neg_only, both}, Δ_c,small = mean over k ∈ {0, 1, 5} of Δ_c is > 0.
- **H2 (data efficiency).** For each c, D_c = Δ_c,small − Δ_c,all > 0, where Δ_c,all is the same target's Δ in the no-withholding `all` condition.
- **H3 (neighbour-support moderation).** For each loss-on c ∈ {loss_only, both}, M_c = mean over high-S targets of Δ_c,small − mean over low-S targets of Δ_c,small > 0.
- **A1 (auxiliary reading, not a test).** At k = 0 no train-partition protein annotated with a pack target appears anywhere in training — not in records, pools or selector views (design §3, verified per run). neg_only also has no attractive GO term. So a k = 0 Δ in neg_only cannot come from target examples or from GO pull towards target neighbours; it is read as **global reshaping** of the projection.

## 2. Data and splits

- **Raw sources.** STRING v12 human, GOA human GAF, go-basic.obo, HuRI, Lit-BM. sha256s in §11, verified by `build_pruned_features.py`.
- **Grades** (unchanged): 2 = STRING experimental ≥ 700; 1 = 0 < experimental < 150; 0 = no STRING edge; 150–699 dropped.
- **Pairs.** `$R/preprocess/go_ppi/pairs.jsonl` is copied byte-for-byte (sha asserted). The converter is not re-run, because its grade-0 sampling depends on `PYTHONHASHSEED` (design §0.1-1). Only `proteins.jsonl` is regenerated, with the same ids in the same order (masked text, pruned GO terms).
- **Split.** The existing deterministic splitter (protein-disjoint 70/10/20, seed 42, `val_no_interaction_cap=20`, `test_no_interaction_cap=0`) is re-run into `$R/preprocess/go_withholding/splits/`. The structural hash, the sorted (anchor, partner, grade) tuples, must equal the reference split's for train, validation and test, and the pool rows must be identical. No re-split. Expected: train 15,708, validation 5,587, test 22,341 records; 7,041 pool rows; protein overlap 0/0/0.
- **Test set and candidate pool** are fixed across all conditions, cells and seeds: `$R/preprocess/go_withholding/splits/test.jsonl`, all records (grades 2/1/0 uncapped), each query's candidates being its own records. sha256 recorded at G1. Its structural sha256 must equal that of the reference test file `c65a1d2ca0e5c16a1c69ecd70d921e670531d325c52625aa15dba88c2e2481e4`.
- **Validation** is target-free: every validation protein annotated with any selected target is removed from records and pools, in any role (design §3.2). It is used only for the λ pilot and the exploratory fusion weight, and never selects checkpoints.

## 3. Leakage control

**Feature pruning** (design §1):

- **Text.** In `encoder_view`, every whole-word mention of another protein's gene symbol or filtered HGNC/UniProt alias becomes `[GENE]`. The protein's own symbol prefix and its own aliases are kept.
- **GO annotations.** Drop annotations whose with/from contains a UniProtKB accession resolving to a different human gene. Drop GO:0005515 descendants (a no-op for BP; the count is reported). Existing guards stay: no NOT, IPI, IEA, ND, GO:0005515, GO:0005488.

**Leak re-measurement (G1, `measure_leakage.py`)**, per grade on train and test, before and after pruning: (a) partner-symbol mention; (b) symbol-or-alias mention; (c) UniProtKB with/from partner reference (an implementation check, ≈0 by construction after pruning); (d) an ungated residual for HGNC/ENSEMBL/ComplexPortal with/from tokens and descriptive complex names. Reference values on test: (a) 11.07% / 0.17% / 0%, (c) 5.92% / 0.41% / 0%.

- **Pass:** on the new test split, (a), (b), (c) at grade 2 are each ≤ 0.5% and exceed grade 1 by ≤ 0.5 percentage points.
- **Fail:** the study stops. Masking rules may be tightened only through an amendment made before G2.

**Withholding leakage (checked in every run; fatal on failure; design §4):**

- No withheld protein in train records (either slot), negative pools (anchor or member), selector views, the GO relation table, or the set of proteins that entered L_GO.
- Validation pools equal the target-free file.
- Exactly k retained proteins per target (numeric k).
- Train, pool, index and test hashes as recorded.
- Cell wiring as specified, including "loss_only has no matcher in the batch processor or selector".
- No failed batches in any epoch; in loss-on cells the L_GO term is applied in ≥ 90% of batches every epoch, in loss-off cells in none.
- Zero selector `None` returns and zero selector exceptions.

**Negative identity (every (condition, seed), from the run-time recorder; design §4.6, §12.3).** Per epoch, the sha256 of the negatives the trainer actually consumed must be equal for baseline and loss_only, equal for neg_only and both, and differ between baseline and neg_only in at least one epoch. The `ok_batches` count per epoch must be equal across all four cells. The check is run for **every** group whose four runs are complete and that lacks a valid pass marker in any of its directories, whether or not the runs finished in the same invocation, and a pass is recorded as a `GROUP_OK.json` marker in all four run directories (design §12.3). Only groups marked in all four directories enter the analysis (§7).

**A mismatch stops the grid.** It writes `GROUP_FAILED.json` into the group's four directories, and from then on every `sweep.py run` refuses to start (exit 4) while any `GROUP_FAILED.json` exists, not only the invocation that found it. The grid resumes only after two steps. First, the cause is fixed and recorded as a row of the §13 amendments table that lists the group in its "groups cleared" cell. Second, the failed group's existing directories are renamed to `<run_dir>.groupfailed<n>` by `sweep.py clear-group` (never deleted). `clear-group` refuses unless exactly one such row exists for its `--amendment` id, with a date, a cause and a fix, and that row names the group (design §12.3). A partially written `GROUP_FAILED.json` (present in only some of the four directories) is cleared the same way. The same rule applies when a completed run's `lc_manifest.json` no longer matches its `DONE.json`: the group is marked `GROUP_FAILED.json` without being checked. The cleared group is then retrained from scratch. Renamed directories are kept for audit and never analysed. The offline audit (`audit_negatives.py`) is descriptive, with one gate: it must show zero **withheld** proteins among sampled negatives in every pack condition at every k. Retained proteins (k ≥ 1) remain pool members by design and may be sampled as negatives; their share, and the share of target-annotated negatives (= withheld + retained), are reported, not gated.

## 4. Target selection (frozen at G2, `build_target_table.py`)

A target is a GO BP term t, using the pruned BP index with the propagated (is_a + part_of) closure. `regulates` edges are excluded.

- **T_train(t)** = train-partition proteins in any role (train record, pool anchor or pool member) whose closure contains t. This is what is withheld.
- **P_train(t)** = proteins in T_train(t) with ≥ 1 positive train record. n(t) = |P_train(t)|. Retention is drawn from P_train(t).
- **Test support P_test(t)** = test anchors whose closure contains t and that have ≥ 1 grade-2 and ≥ 1 grade-1 record.

**Candidate rule:**

- n(t) ≥ 40;
- |P_test(t)| ≥ 8;
- removal of ≤ 5% of undirected train positive pairs;
- P_train(t) ≤ 10% of train proteins with positives;
- ≥ 25 retention-eligible proteins. A protein is eligible if it has a positive record whose other endpoint is not withheld and whose anchor keeps a non-empty pool after withholding;
- greedy non-nested selection, pairwise protein-Jaccard ≤ 0.2, ordered by test support and then GO id.

**Neighbour support:**

- N_sib(t) (primary) = terms sharing an is_a or part_of parent with t, excluding t, its ancestors and its descendants.
- N_2(t) (sensitivity) = terms within ≤ 2 undirected hops, same exclusion.
- S(t) = Σ undirected train positive degree over NP(t) = ∪_{t′∈N(t)} P_train(t′) \ T_train(t), computed on the full post-fix train.

**Strata and matching:**

- low-S = S_sib ≤ 33rd percentile and high-S = S_sib ≥ 67th, over the non-nested candidates. The middle is excluded.
- Optimal assignment on z-scored (log n, log test support), then greedy acceptance under Jaccard ≤ 0.2, gives **12 low/high pairs = 24 targets**, numbered in acceptance order. Size rule: ≥ 12 accepted → the first 12; 10 or 11 accepted → **exactly the first 10** (20 targets, shortfall recorded; 11 is never used); < 10 → the study stops.

**Packing (amended by AM1, §13).** One target per condition: every pack holds exactly one target, so `n_packs = n_targets`, pack ids `p01…` follow the target order (pair number, low before high). There are no pack-mate constraints. Each target's own k = 0 withholding must remove ≤ 8% of pairs and leave it ≥ 25 eligible proteins (both already implied by the candidate rule's 5% cap and its retention rule; a failure stops the study). The §4 size rule is unchanged, so a 20-target shortfall can still come from matching. The pre-AM1 3-target packing (8 × 3 or 6 × 3 + 1 × 2, protein-disjoint pack-mates outside each other's N_2/ancestors/descendants with ≤ 5% neighbour overlap, ≥ 1 low and ≥ 1 high per pack, ≤ 8% pack removal, slot cycling and the 24 → 20 fallback) was infeasible on the pruned data and is superseded.

The table records `n_packs`, `n_conditions = 5·n_packs + 1` and `n_runs = 4·5·n_conditions`: 121 / 2,420 at 24 targets, 101 / 2,020 at 20. Every count below that says 121 or 2,420 means these table fields.

**Retention.** k ∈ {0, 1, 5, 10, 25} proteins per target.

- Retained sets are nested prefixes of one seeded permutation of the target's eligible set under its own withholding (W = T_train(t); AM1) (`retention_seed = 20251009`), shared by all seeds and cells.
- Retained proteins keep only their pairs with non-withheld proteins.
- Pool-only target-annotated proteins are withheld at every numeric k, never retained, and present only in `all`.
- One shared `all` condition with no withholding.

`target_table.json`, `text_covariate.json` and their sha256s are recorded at G2. Target names are not inspected before then. No target may be added, dropped or re-stratified after G2.

## 5. Variants

All four cells share every field of `$R/config/go_decomposition_go_ordinal/randwin.json`:

- `loss_type=go_ordinal` with its ranking weights and margins;
- go_aspect P;
- 15 epochs, batch 64, lr 8.5e-5, temperature 0.07;
- 7 negatives per anchor, with the grade-1 share ramping from 0.2 to 0.6;
- frozen all-mpnet-base-v2 with the projection head;
- `ontology_weight 0`, `orca_enabled false`.

They also share the study overrides of design §12.1 (absolute paths, pruned index, `checkpoint_frequency 1e9`, simGIC, aspect weights). Configs: `$R/config/go_withholding/{baseline,loss_only,neg_only,both}.json`.

| cell | negatives | `go_ontology_loss_weight` |
|---|---|---|
| baseline | random_window (fixed 34% random window, no GO) | 0 |
| loss_only | random_window | λ |
| neg_only | GO-tiered stochastic (closest 34% by simGIC_BP) for anchors with pruned BP terms; random_window for anchors without | 0 |
| both | as neg_only | λ |

**Anchors without pruned BP terms** (created by leak pruning; ~115 train-record anchors estimated, counted per condition at G3) receive the control's fixed 34% random window in **every** cell. All four configs set `go_ppi_unannotated_anchor_window = true` (design §5.6). Without this setting the selector would draw their negatives uniformly from the whole pool in neg_only/both, which changes window narrowing as well as GO ordering. With it, their negatives are identical across all four cells, and the N factor acts only on annotated anchors. Unannotated pool *members* are ordered last in GO-tiered cells, as before.

**The L term.** L_GO = λ · mean over unordered pairs of distinct BP-annotated anchor and positive proteins in the batch of (cos(f_i, f_j) − simGIC_BP(i, j))².

- The relation table covers the condition's training proteins only (proteins in its train records). IC is the pruned index's corpus-wide IC, computed over all annotated human genes in the pruned GAF, including withheld and test proteins. It uses no interaction labels and is identical across cells and conditions.
- Negatives are excluded from the pair set.
- The single-factor audit asserts the **exact** differing-field set per pair (design §12.1):
  - baseline↔loss_only and neg_only↔both: exactly {`go_ontology_loss_weight`};
  - baseline↔neg_only and loss_only↔both: exactly {`go_ppi_go_tiered_negatives`, `go_ppi_tier_sampling`}.

  Configs are not written otherwise.

**λ selection (G4, `lambda_pilot.py`).** Condition `all`, seed 42, 3 epochs, λ₀ = 0.1.

- **Validation batches:** all grade-2 records of the target-free `validation_positive.jsonl`, in file order, in fixed batches of 32, dropping the last partial batch. ≥ 4 batches are required; otherwise stop and ask the user.
- **Ratio:** r₀ = median over batches of ‖∇(λ₀·L_GO)‖ / ‖∇L_InfoNCE‖ on the trainable projection parameters, at the final pilot checkpoint (epoch 2), with the batch processor and loss engine set to epoch 2 (`selector_epoch` recorded) and a relation table over the validation proteins.
- **Choice:** λ* = λ₀·0.2/r₀, rounded in log space to {1, 2, 3, 5}×10ⁿ.
- **Confirmation:** a confirmation run must give r* ∈ [0.10, 0.30]. One re-estimate is allowed; otherwise stop and ask the user.
- **Recording:** λ, r₀, r*, `n_records`, `n_batches` are recorded at G4.

λ is never tuned on test or target metrics.

**Checkpoint.** The final epoch in every cell (`checkpoint_epoch_14.pt`). No validation-based selection; study runs carry no validation path.

**Seeds.** {13, 21, 42, 87, 123}, via `training_seed`. They control initialisation, shuffling and negative draws. The retained-protein draw is fixed.

**Grid.** n_packs × 5 k + 1 `all` = `n_conditions`; × 4 cells × 5 seeds = `n_runs`. At 24 targets (AM1: one target per pack): 24 × 5 + 1 = 121 conditions, **2,420 runs**, ≈ 134 h at 200 s per run (≈ 136–257 h CPU on the pre-AM1 46–87 h scale). Under the 20-target shortfall: 101 conditions, 2,020 runs.

## 6. Metrics

**Primary.** Per-target hard AUC: tie-corrected rank AUC of all grade-2 vs all grade-1 test records of the queries in P_test(t), pooled within the target, from the final checkpoint. Targets in pack p are read from pack p's runs at each numeric k and from the shared `all` runs at k = all.

**Secondary:**

- per-target macro per-query hard AUC;
- per-target macro MRR (grade 2 relevant);
- non-target collateral hard AUC: queries outside the pack's targets and their neighbour proteins;
- global hard, easy and pooled AUC.

**Mechanism diagnostics (descriptive, except as used in the §8 mechanism rule):**

- **Per epoch:** InfoNCE, ranking and L_GO values; their projection-parameter gradient norms; `ratio_weighted_go_to_infonce`; the ranking/InfoNCE ratio; and `share_above_clip_eval_diag`. The last is the eval-mode share of diagnostic batches whose pre-clip gradient norm exceeds 1.0.
- **Negative audit per cell:** simGIC by grade, grade mix, HuRI/Lit-BM false-negative rate, and the shares of target-neighbour, withheld (gated at 0), retained and target-annotated proteins among negatives.
- **Relation preservation:**
  - global Spearman(cos, simGIC_BP) on 20,000 fixed test-protein pairs;
  - per target, `target_nbr_mean_cos − random_ref_mean_cos` (P_test(t) to NP_test_sib(t) vs to fixed random test proteins), plus the target × neighbour Spearman.
- **Text-similarity covariate** tsim_excess(t), from frozen MPNet.

**Exploratory:** the k = 10 and k = 25 curve, the simGIC fusion reference, the N_2 neighbourhood.

## 7. Uncertainty

**Unit and pairing.** (target t, seed s), paired across cells within the same condition and seed.

**Primary method: two-way cluster bootstrap** over packs × seeds. B = 10,000, `numpy.random.default_rng(20251009)`. Each replicate:

1. draws `n_packs` packs with replacement (24, or 20 under a shortfall; one target each under AM1);
2. draws one multiset of 5 seeds with replacement, shared by all packs;
3. averages the contrast over the defined (t, s) cells among (targets of the drawn packs) × (drawn seeds), with multiplicity, each cell weighted equally (definedness rules under "Missing data").

This respects that every target shares the `all` runs and that seeds are crossed with targets. Under AM1 no numeric-k run is shared between targets, so a pack is a target.

- D_c is formed per (t, s) before resampling.
- M_c is high-target mean minus low-target mean within the drawn packs.
- Output: 95% percentile CI and two-sided p = min(1, 2·min(P*(≤0), P*(≥0))).

**Multiplicity.** Holm at α = 0.05 within three families: H1 (3 cells), H2 (3 cells), H3 (2 cells). Per-k CIs are unadjusted and descriptive.

**Sensitivity (reported, never changes verdicts):**

- S1: target-then-seed hierarchical bootstrap;
- S2: N_2 strata;
- S3: per-query metric;
- S4: tsim_excess-adjusted moderation;
- S5: seed spread.

**Missing data.** A run that fails twice (training or evaluation) is missing. The one exception is an evaluation-stage failure later repaired by `sweep.py rescore`, which re-scores the kept checkpoint without retraining (design §12.3); its group must then pass the negative-identity check like any other. Training-stage failures stay missing. A (t, k, s) tuple is present only if all four cells of its run group are complete **and carry a `GROUP_OK.json` marker** from a passed negative-identity check (§3), whose recorded manifest hashes match the runs. Otherwise it is dropped for all four cells. Two cases are not "missing" but stop the analysis (fatal preflight, design §11 step 5), and `--allow-incomplete` does not relax them:

- a group whose four runs are complete but which has no valid `GROUP_OK.json` (unchecked);
- a group with `GROUP_FAILED.json`.

A failed pack run drops (t, k, s) for every target in the pack; a failed `all` run drops (t, all, s) for every target. Fixed rules, no partial averaging and no imputation:

- Δ_c,small(t, s) is defined only if all three k ∈ {0, 1, 5} tuples are present; otherwise (t, s) is missing for H1 and H2.
- D_c(t, s) is defined only if Δ_c,small(t, s) and Δ_c(t, all, s) are both defined.
- E_c(t, s) (§8 A1) is defined only if Δ_c(t, 0, s) and the same pack run's collateral Δ at k = 0 are defined.
- In each bootstrap replicate the statistic is the mean over the defined (t, s) cells among (targets of drawn packs) × (drawn seeds), with multiplicity. A replicate with zero defined cells for a contrast (for M_c: zero defined high cells or zero defined low cells) is discarded for that contrast, and the number discarded is reported. If more than 5% of replicates are discarded, the contrast is flagged unstable and classified inconclusive. The point estimate uses the same rule on the full data.
- The 10% rule counts missing (t, k, s) tuples over all n_targets × 6 × 5 target-condition-seed tuples, where 6 counts the five numeric k plus `all` (720 at 24 targets; 600 at 20; unchanged by AM1). More than 10% missing → no confirmatory analysis; everything is labelled exploratory. A grid stopped by a negative-identity or `ok_batches` mismatch is neither extended nor analysed until the cause is fixed under an amendment and the failed groups are cleared (§3); a cleared group counts as not yet run until it is retrained and checked. Directories named `*.attempt*` or `*.groupfailed*` are ignored by the analysis.

No outlier exclusion. No covariate adjustment in the primary analysis.

## 8. Interpretation criteria (fixed)

**Effect floor:** 0.01 AUC (about the seed sd).

**Classification of a contrast.** Ordered, first match wins. `estimate`, `ci_low`, `ci_high` are from the primary two-way bootstrap; `p_holm` is Holm-adjusted within the contrast's family. For descriptive contrasts without a family (per-k, A1), `p_holm` is read as the unadjusted p.

```text
1. harm:              ci_high < 0 and estimate <= -0.01
2. supported:         p_holm < 0.05 and ci_low > 0 and estimate >= +0.01
3. practically null:  -0.01 < ci_low and ci_high < +0.01
4. below floor:       ci_low > 0 and estimate < +0.01
5. inconclusive:      everything else (including ci_low > 0, estimate >= 0.01, p_holm >= 0.05),
                      and any contrast flagged unstable (> 5% discarded replicates, §7)
```

The unstable check is applied before rules 1–4.

`decisions.json` also reports the raw findings-(iv) flag `ci_excludes_0` for each contrast.

**Readings:**

- **H1.** "Mechanism c improves performance on withheld targets" requires H1 = supported for c. Per-k results (k = 0, 1, 5) are described but do not alter the verdict.
- **H2.** "c helps more with fewer examples" requires H2 = supported for c. If H1 is supported but H2 is null or inconclusive, the result reads as a uniform benefit, not data efficiency.
- **H3.** "Neighbour support moderates the benefit" requires H3 = supported for c ∈ {loss_only, both}.
  - If H3 holds only under N_2 (S2) or only after adjustment (S4), it is exploratory.
  - If H3 is supported but the S4 `high` coefficient's CI includes 0, the moderation is read as possibly text-mediated.
- **A1.** If Δ_neg_only at k = 0 has ci_low > 0 (descriptive), global reshaping is present (the findings-(iv) reading). This is reported alongside the loss-on cells and is **not** subtracted from them, because neg_only's global effect does not measure loss_only's. For each loss-on cell c, the target-specific excess is E_c(t, s) = Δ_c(t, 0, s) − Δ^coll_c(pack(t), 0, s), where Δ^coll_c is the collateral hard-AUC difference (cell − baseline) of the same pack run. E_c is reported with the two-way bootstrap CI. loss_only's k = 0 benefit is called target-specific only if E_loss_only has ci_low > 0 (descriptive, unadjusted).
- **Mechanism (for any supported loss_only result).** Confirmed only if both hold:
  - (a) the two-way bootstrap ci_low of the paired mean over defined (t, s) of [`target_nbr_mean_cos − random_ref_mean_cos`] (loss_only minus baseline, averaged over k ∈ {0, 1, 5} with the Δ_small definedness rule) is > 0;
  - (b) the per-run median over epochs of `ratio_weighted_go_to_infonce` lies in [0.05, 0.5] for ≥ 90% of loss_only runs.

  Otherwise the benefit is reported with "mechanism not confirmed". Global Spearman is descriptive only.
- **Collateral.** A collateral hard-AUC Δ classified as harm (rule 1) is reported as a cost alongside any supported H1.

## 9. Gates (in order; each must pass before the next)

- **G0.** Code implemented; all unit tests pass; regression suites pass (`/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/` and the four career files, design §13). The three user-edited files keep their original uncommitted hunks verbatim.
- **G1.** `build_pruned_features.py` passes: pairs sha and id order equal; structural hashes equal; frozen files made read-only. `measure_leakage.py` passes. Pins: frozen_hashes, leak audit.
- **G2.** `build_target_table.py`, the eval cache (`build-cache`), `text_similarity_covariate.py`, and `make_conditions.py --mode validation`. Pins: target table, covariate, cache, target-free validation.
- **G3.** `make_conditions.py --mode conditions` writes all `n_conditions` conditions (121, or 101 under the shortfall; AM1) with exactly-k and residual scans clean, and with retained proteins still present in the pools. Pin: condition count and manifest hashes.
- **G4.** Configs (`--lambda none`, which writes no pin), λ pilot, configs (`--lambda λ`, which writes the single `G4-configs` pin over the four cell configs). Both materialiser calls assert the exact per-pair diff sets (§5). Pins: λ choice, configs.
- **G5.** The user commits or tags (gap item 1), or explicitly accepts the content pin; then `sweep.py pin`. The pin covers every `.py` file under `contrastive_learning/`, `go_ppi_domain/`, `scripts/`, `trials_domain/` plus `run_phase1_embedding_evaluation.py`, and `pin` refuses if an import probe of the runner and evaluator loads any repo file outside it. Every run and every scoring call re-checks that its imported repo modules are covered. The protocol is now fully frozen.
- **G6.** Smoke: p01_k0 × 4 cells × seed 13, 2 epochs, in `$R/results/go_withholding/smoke/`. Must show:
  - runner assertions and post-training checks pass;
  - the recorder identities hold (baseline = loss_only, neg_only = both, baseline ≠ neg_only);
  - `ok_batches` are equal across cells;
  - `check-group` passes and writes `GROUP_OK.json` into the four smoke dirs;
  - the evaluator scores `checkpoint_epoch_1.pt`;
  - the evaluator's encoder cross-check passes: 50 fixed test texts, built with the trainer's own text function and re-encoded through the trainer's text encoder (`max_seq_length` 512), match the eval cache at cos ≥ 0.9999, and the trainer's and evaluator's text functions give identical token ids for every test protein (design §9.2);
  - offline audits for p01_k0, p01_k5 and `all` pass (zero withheld negatives; p01_k5 exercises the retained-protein case);
  - every run's imported repo modules are covered by the code pin.
- **G7.** The grid, seed-major, through repeated `sweep.py run --max-runs N`; then the offline audits and the analysis.

## 10. Commands

Every block assumes this prefix; all paths are absolute.

```bash
R=/Users/olukotunjosh/Downloads/CDCL; PY="$R/.venv/bin/python"; S="$R/scripts/go_withholding"; cd "$R"
```

```bash
# G0
"$PY" -m pytest "$R/go_ppi_domain/tests" -q
"$PY" -m pytest "$R/tests/test_career_ordinal_candidates.py" "$R/tests/test_mesh_soft_sampling.py" \
  "$R/tests/test_career_adapter_equivalence.py" "$R/tests/test_career_golden_regression.py" -q

# G1
"$PY" "$S/build_pruned_features.py" --bulk-dir "$R/dataset/go_ppi/bulk" \
  --reference-converted "$R/preprocess/go_ppi" --reference-splits "$R/preprocess/go_ppi_splits" \
  --out-converted "$R/preprocess/go_withholding/converted" --out-splits "$R/preprocess/go_withholding/splits"
"$PY" "$S/measure_leakage.py" --splits "$R/preprocess/go_withholding/splits" \
  --converted "$R/preprocess/go_withholding/converted" --reference-splits "$R/preprocess/go_ppi_splits" \
  --reference-converted "$R/preprocess/go_ppi" --out "$R/preprocess/go_withholding/leak_audit.json"

# G2
"$PY" "$S/build_target_table.py" --converted "$R/preprocess/go_withholding/converted" \
  --splits "$R/preprocess/go_withholding/splits" --obo "$R/dataset/go_ppi/bulk/go-basic.obo" \
  --out "$R/preprocess/go_withholding/targets/target_table.json"
"$PY" "$S/evaluate_withholding.py" build-cache --test "$R/preprocess/go_withholding/splits/test.jsonl" \
  --frozen-hashes "$R/preprocess/go_withholding/splits/frozen_hashes.json" --out "$R/results/go_withholding/eval_cache"
CACHE=$(ls "$R"/results/go_withholding/eval_cache/test_*.npz)
"$PY" "$S/text_similarity_covariate.py" --target-table "$R/preprocess/go_withholding/targets/target_table.json" \
  --cache "$CACHE" --train-embeddings "$R/embedding_cache/go_withholding_text_embeddings.pt" \
  --out "$R/preprocess/go_withholding/targets/text_covariate.json"
"$PY" "$S/make_conditions.py" --mode validation \
  --target-table "$R/preprocess/go_withholding/targets/target_table.json" \
  --base-splits "$R/preprocess/go_withholding/splits" --out "$R/preprocess/go_withholding/validation_targetfree"

# G3
"$PY" "$S/make_conditions.py" --mode conditions \
  --target-table "$R/preprocess/go_withholding/targets/target_table.json" \
  --base-splits "$R/preprocess/go_withholding/splits" \
  --targetfree-validation "$R/preprocess/go_withholding/validation_targetfree" \
  --out "$R/preprocess/go_withholding/conditions"

# G4
"$PY" "$S/materialize_configs.py" --lambda none
"$PY" "$S/lambda_pilot.py" --out "$R/results/go_withholding/lambda_pilot"
LAMBDA=$("$PY" -c "import json;print(json.load(open('$R/results/go_withholding/lambda_pilot/lambda_choice.json'))['lambda'])")
"$PY" "$S/materialize_configs.py" --lambda "$LAMBDA"

# G5 (after the user commits/tags, or explicitly accepts the content pin)
"$PY" "$S/sweep.py" pin

# G6 smoke (2 epochs; outside the grid's runs/ directory)
C="$R/preprocess/go_withholding/conditions/p01_k0"
for CELL in baseline loss_only neg_only both; do
  O="$R/results/go_withholding/smoke/$CELL"
  "$PY" "$R/scripts/run_learning_curve_point.py" --domain go_ppi --config "$R/config/go_withholding/$CELL.json" \
    --train-file "$C/train.jsonl" --go-ppi-split-dir "$C" --withholding-manifest "$C/withholding_manifest.json" \
    --study-cell "$CELL" --go-diagnostics "$O/go_diagnostics.jsonl" --output-dir "$O" --seed 13 --epochs 2 \
    --compare-config "$R/config/go_withholding/baseline.json"
  "$PY" "$S/evaluate_withholding.py" score --run-dir "$O" --cache "$CACHE" \
    --target-table "$R/preprocess/go_withholding/targets/target_table.json" --relation
done
"$PY" "$S/evaluate_withholding.py" score --run-dir "$R/results/go_withholding/smoke/baseline" --cache "$CACHE" \
  --target-table "$R/preprocess/go_withholding/targets/target_table.json" \
  --encoder-crosscheck --config "$R/config/go_withholding/baseline.json"
"$PY" "$S/sweep.py" check-group --run-dirs "$R/results/go_withholding/smoke/baseline" \
  "$R/results/go_withholding/smoke/loss_only" "$R/results/go_withholding/smoke/neg_only" "$R/results/go_withholding/smoke/both"
"$PY" "$S/audit_negatives.py" --cond all    --cells baseline loss_only neg_only both --seed 13 --epochs 0 7 14
"$PY" "$S/audit_negatives.py" --cond p01_k0 --cells baseline loss_only neg_only both --seed 13 --epochs 0 7 14
"$PY" "$S/audit_negatives.py" --cond p01_k5 --cells baseline loss_only neg_only both --seed 13 --epochs 0 7 14

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

`sweep.py check-group --run-dirs <4 dirs>` runs the same group checks as the sweep (design §12.3) on any four run directories in cell order; it does not require `epochs == 15`. The full grid is never launched by a single command. Each `sweep.py run` executes at most `--max-runs` (≤ 100) runs. It resumes from `DONE.json` markers, and it refuses to start if any pin in `protocol_pins.json` or the code pin no longer matches. `run` (at its start, after each run, and at its end) and `status` (at its start) group-check every (condition, seed) group that has four `DONE.json`, no `GROUP_FAILED.json`/`FAILED.json`, and at least one directory without a valid `GROUP_OK.json`. So an interrupted driver, an interrupted marker write or a `--cells`-filtered invocation cannot leave a complete group unchecked. If any `GROUP_FAILED.json` exists, `run` exits 4 before training anything, `plan` warns, and `status` reports it; clearing is described in §3 and §13:

```bash
# only after a §13 amendments row with id AM<n> lists "<cond_id>,s<seed>" under "groups cleared"
"$PY" "$S/sweep.py" clear-group --cond <cond_id> --seed <seed> --amendment AM<n>
# only for a run whose FAILED.json has "stage": "eval", after the evaluation cause is fixed
"$PY" "$S/sweep.py" rescore --run-id <run_id>
```

The placeholders in angle brackets are not valid arguments. In particular, `AM<n>` is not a valid amendment id. The re-score command above with `--encoder-crosscheck` is run once at G6 only and overwrites nothing but `eval/` of the smoke baseline run. `rescore` refuses a training-stage failure, runs the same pin and stopped-grid preflights as `run`, and keeps the earlier `eval/` as `eval.attempt<n>` (design §12.3).

## 11. Frozen artifacts and hashes

Raw inputs (from `setup_baseline.md`, verified at G1):

| file | sha256 |
|---|---|
| $R/dataset/go_ppi/bulk/9606.protein.info.v12.0.txt.gz | 144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c |
| $R/dataset/go_ppi/bulk/9606.protein.links.detailed.v12.0.txt.gz | 28a2ea3b22c964e691c0f2f934c5e3e397a4b0a12c833d8787fa02b7847b91b6 |
| $R/dataset/go_ppi/bulk/9606.protein.aliases.v12.0.txt.gz | b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680 |
| $R/dataset/go_ppi/bulk/9606.protein.physical.links.v12.0.txt.gz | 1ca87209a93cf7685e9c2997fbcfbfc69fedc6a803ccc1df917318a651df0823 |
| $R/dataset/go_ppi/bulk/goa_human.gaf.gz | db472faff1785878521693af62646546cea6af4a386609dd01c72c0554a46a30 |
| $R/dataset/go_ppi/bulk/go-basic.obo | b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc |
| $R/dataset/go_ppi/independent/HuRI.tsv | 4cdf94366ff6e23e71ef9ec81922e85b2b3a1fb08e938f7b00d86c027cf1ecd6 |
| $R/dataset/go_ppi/independent/Lit-BM.tsv | f39c9c4b8a372286c3eca77d0ff651d837a643d32235eae866daae082663391d |
| $R/preprocess/go_ppi_splits/test.jsonl (reference; structure only) | c65a1d2ca0e5c16a1c69ecd70d921e670531d325c52625aa15dba88c2e2481e4 |
| $R/preprocess/go_ppi/pairs.jsonl (copied verbatim) | full value recorded at G1 (prefix `f789f237`) |

Recorded at gate, in `$R/results/go_withholding/protocol_pins.json` and mirrored in §13:

| pin key | artifact | gate |
|---|---|---|
| G1 | `$R/preprocess/go_withholding/splits/frozen_hashes.json`: test sha and structural sha, train/validation/pools, pairs, proteins, `go_index_P.pkl`, mask manifest | G1 |
| G1-leak | `$R/preprocess/go_withholding/leak_audit.json` (pass, rates) | G1 |
| G2-targets | `$R/preprocess/go_withholding/targets/target_table.json` sha256 | G2 |
| G2-cache | `$R/results/go_withholding/eval_cache/test_<sha16>.npz` sha256 | G2 |
| G2-covariate | `$R/preprocess/go_withholding/targets/text_covariate.json` sha256 | G2 |
| G2-validation | `$R/preprocess/go_withholding/validation_targetfree/manifest.json` and file hashes | G2 |
| G3 | `n_conditions` and the `withholding_manifest.json` sha256s (121, or 101 under the shortfall; AM1) | G3 |
| G4-lambda | λ, r₀, r*, n_records, n_batches (`lambda_choice.json` sha256) | G4 |
| G4-configs | `$R/config/go_withholding/configs.sha256.json` (exactly the four cell configs; written only by `--lambda <float>`; the sweep uses the last entry) | G4 |
| G5 | git HEAD and `code_sha256` (`$R/results/go_withholding/code_pin.json`) | G5 |

## 12. Known limitations, stated in advance

- **Text-mediated transfer.** Sibling-term proteins have similar function text, so part of an S effect may come from text rather than L. tsim_excess (S4) is the preregistered covariate; A1 is the global-reshaping control.
- **Fixed retained-protein draw.** One retention draw per target means k = 1 and k = 5 depend on which proteins were retained. This variance is not separated from the target effect.
- **Pool-only target proteins** are absent at every numeric k and present at `all`. This is part of the manipulation, so the k = 25 vs `all` contrast differs in more than 25 → n(t) positives.
- **Batch composition.** Withholding shifts batch composition for all proteins. This is inherent and equal across cells within a condition.
- **Residual leaks.** Descriptive complex names, ComplexPortal/HGNC/ENSEMBL with/from references, and GOA evidence sharing source papers with STRING may still carry interaction information. They are reported (leak metric d), not removed.
- **Clipping.** Clipping at 1.0 may rescale λ's effective weight. Only the eval-mode diagnostic share is reported.
- **Cluster bootstrap.** Under AM1 the clusters are 24 single-target packs (8 three-target packs before AM1), so CIs are less coarse than first planned. Targets still share the `all` runs and the seeds; S1 is shown for comparison.

## 13. Pins and amendments

**Pins** (mirrored from `protocol_pins.json`; filled at each gate):

| pin key | value | date |
|---|---|---|
| G1 | test `4027e9a0a3cc6ec173fe468e90ff48e013efb8f61dc8eb2ce61d5b7a346bbb2a`, test structural `16ec161d8b1588e1ee1eb091b59fde643d6f05ad26e398967127d33eb1b45943`, pairs `f789f237eeee1ecb56f77c30a8a4c5cfd635ede95a70cfecf60c26fee0f47116`, train `c38f89c2…`, validation `90c0b7df…`, pools `2e4fc52b…`, proteins `4c542c73…`, go_index_P `a0c8b0b4…`, mask manifest `575db91c…` | 2026-10-10 |
| G1-leak | pass; leak_audit.json `f824b4856a858d127decc8cda26ceee5fba96fe0d1c80a4ad22fd0ee74ab6db5` | 2026-10-10 |
| G2-targets | target_table.json `58153f84f0221265e6d5788d630b8f229bb118d671402a99b5120d5848950e68`; 24 targets, 24 packs, 121 conditions, 2,420 runs, no shortfall | 2026-10-10 |
| G2-cache | `test_4027e9a0a3cc6ec1.npz` `abf715f9061f5d998b6d21cf7d496bc9859d092f39b0e625cf02de0f2c5f0b28` (CPU build, the last G2-cache entry; an earlier MPS-built entry `7fbe1795…` is superseded, see smoke_report_go_ppi.md) | 2026-10-10 |
| G2-covariate | text_covariate.json `5d812f81b095df7704bec3b0775ecbd8694b85d6922ebff42824528db55ef90c` | 2026-10-10 |
| G2-validation | manifest.json `d8c5a4cd9ab7baddd530088c3b8fd5f55c0afe79a020818a338aec99da503502`; V 330, 3,011 records, 181 grade-2 | 2026-10-10 |
| G3 | 121 conditions; manifests in protocol_pins.json (e.g. all `ad9562ea…`, p01_k0 `b0c754ac…`, p01_k5 `35602db3…`) | 2026-10-10 |
| G4-lambda | λ = 0.5, r₀ = 0.0441, r* = 0.1538, 181 records, 5 batches; lambda_choice.json `5845b14f37b670b93f5af6890d3f7b465bd4b20f469dd309b464c7d96890ecb0` | 2026-10-10 |
| G4-configs | configs.sha256.json `afb6dbb78795f7073b4764f833b5e7de2a82b59b80f9f9783730b73da5c2ba49` (λ 0.5) | 2026-10-10 |
| G5 | | |

**Amendments.**

| id | date | gate | groups cleared | cause | fix | re-pin |
|---|---|---|---|---|---|---|
| AM1 | 2026-10-10 | before G2 | — | Packing infeasible at 20 targets (design §2.6 step 4): `build_target_table.py` exited 3, "packing infeasible at 20 targets; queue exhausted", on the plan item-12 scratch run of the pruned data. An exact solver confirmed it: at 24 targets 5 valid 3-packs exist and 8 are needed; none of the 41 slot-cycled 20-target sets packs as 6 × 3 + 1 × 2. Among the 24 accepted targets, 204 of 276 target pairs fail P_train disjointness, 90 the 5% neighbour-overlap cap and 1 the N_2 rule; 86 of 276 exceed the 8% pack removal cap. No parameter was changed before this row. | Adopt option (c), one target per condition, singleton packs (`pack_size` 1). Removed: the pack-mate constraints (P_train disjointness, N_2/ancestor/descendant exclusion, 5% neighbour-overlap cap, ≥ 1 low and ≥ 1 high per pack) and the slot-cycling / 24 → 20 packing fallback. Kept per single target: the 8% k = 0 removal cap and ≥ 25 retention-eligible proteins. The §4 size rule is unchanged. Counts: n_packs = n_targets; 24 targets → 121 conditions (24 × 5 + 1 shared `all`), 2,420 runs; 20 targets → 101 / 2,020. The bootstrap resamples 24 (or 20) single-target packs; the 10% rule denominator n_targets × 6 × 5 (6 = k levels) stays 720 / 600. Consequence: S uncontaminated by pack-mates (S reduction by mates is 0 by construction); 12 low-S / 12 high-S targets, SMD log n −0.048, log support −0.036. Edited: protocol §4, §5, §7, §9, §10, §11, §12, §14; design §22. | no |

Rules for rows:

- **id.** Ids are "AM" followed by a positive integer without leading zeros, assigned in order. They are distinct from the auxiliary reading A1. Example ids are never written in this section outside a real row.
- **date.** `YYYY-MM-DD`.
- **gate.** The gate at which the amendment is made.
- **groups cleared.** The (condition, seed) groups a stopped-grid amendment authorises `sweep.py clear-group` to rename (§3). They are written as `<cond_id>,s<seed>` entries separated by `;`, or `—` for an amendment that clears nothing.
- **cause.** The cause found in `GROUP_FAILED.json`, or for other amendments, the field changed and why.
- **fix.** The fix applied.
- **re-pin.** Whether `sweep.py pin` was re-run (`yes`/`no`).

Cells are separated by `|`; a literal `|` inside a cell is written `\|`.

Every row needs a non-empty cause and fix. `sweep.py clear-group` reads only this table, between the `## 13.` and `## 14.` headings. It refuses unless exactly one row carries its `--amendment` id, that row has a valid date and a non-empty cause and fix, and its "groups cleared" cell lists the group. It also refuses a group already cleared under that row (design §12.3). The full row is copied into `sweep_log.jsonl`.

## 14. Changes

### Plan decision 4b (2026-10-10, implementation, before G1)

Wording only; no criterion, count or rule changes. Not an amendment row (it clears no group and changes no frozen value).

| # | change | where |
|---|---|---|
| 1 | One sentence added under the amendment-row rules: a literal pipe character inside a cell is written as a backslash followed by the pipe. `parse_amendment_rows` and `sweep.py clear-group` already split cells only on unescaped pipes (plan decision 4b, from the approving v6 design review). | §13 |

### Amendment AM1 (2026-10-10, before G2)

Recorded as row AM1 in §13; design §22 has the implementation detail.

| # | change | where |
|---|---|---|
| 1 | One target per condition (singleton packs). Pack-mate constraints, slot cycling and the 24 → 20 packing fallback removed; the 8% k = 0 removal cap and retention feasibility apply per target. | §4 |
| 2 | Counts: 121 conditions and 2,420 runs at 24 targets; 101 / 2,020 at 20. | §4, §5, §9 G3, §10 G7, §11 |
| 3 | The two-way bootstrap resamples 24 (or 20) single-target packs. The 10% rule denominator is unchanged (720 / 600). | §7, §12 |

### From v5 to v6 (fifth design review)

Finding numbers refer to the v5 review; design §21 maps each to its resolution.

| # | change | where |
|---|---|---|
| 1 | Amendments are a table. `clear-group` requires exactly one row with the given id that has a date, a cause and a fix and lists the group being cleared. Each row clears a given group once, and the row is logged. Literal example ids are removed. | §3, §10, §13 |
| 2 | `clear-group` accepts a group with `GROUP_FAILED.json` in any of its directories and renames all existing ones, so an interrupted marker write is recoverable. | §3; design §12.3 |
| 3 | A completed run whose `lc_manifest.json` changed after `DONE.json` stops the grid (`GROUP_FAILED.json`, `lc_manifest_changed_after_done`) instead of leaving its group unchecked forever. | §3; design §12.3 |
| 4 | New `sweep.py rescore` repairs evaluation-stage failures from the kept checkpoint, with the pin and stopped-grid preflights. Training-stage failures stay missing. | §7, §10; design §12.3 |
| 5 | Code fence in §10 fixed. | §10 |

### From v4 to v5 (fourth design review)

Finding numbers refer to the v4 review; design §20 maps each to its resolution.

| # | change | where |
|---|---|---|
| 1 | A `GROUP_FAILED.json` now stops every later `sweep.py run` (exit 4) until the cause is fixed under an amendment and the group is cleared by renaming its directories to `*.groupfailed<n>`; renamed directories are ignored by the analysis. | §3, §7, §10, §13 |
| 2 | A group is re-checked whenever any directory lacks a valid `GROUP_OK.json`, so a partially written or stale marker is repaired, not stranded. | §3, §10 |
| 3 | The G6 encoder cross-check uses the trainer's own text function and requires zero token-id differences between it and the evaluator's. | §9 G6 |
| 4 | Recorder attribute renamed to match the manifest key. | design §4.6, §4.7; no protocol change |

### From v3 to v4 (third design review)

Finding numbers refer to the v3 review; design §19 maps each to its resolution.

| # | change | where |
|---|---|---|
| 1 | Anchors without pruned BP terms get the control's random window in every cell (`go_ppi_unannotated_anchor_window = true` in all four configs); counted per condition. | §5 |
| 2 | Group checks run for every complete group, on resume and with `--cells` filters, and record `GROUP_OK.json`. The analysis requires the marker; unchecked or group-failed groups stop the analysis. | §3, §7, §10 |
| 3 | The single-factor audit asserts exact per-pair diff sets. | §5, §9 G4 |
| 4 | Eval encoder `max_seq_length` 512 with a truncation assertion; encoder cross-check moved to G6. | §9 G6, §10 |
| 5 | Packing slot cycling and the 20-target queue (m11, m12, then the original queue) are specified in the design. | design §2.6; §4 unchanged in substance |
| 6 | `G4-configs` is pinned once, by `--lambda <float>` only; the sweep uses the last entry. | §9 G4, §11 |
| 7 | Wiring check tests the sign of the loss weight; λ equality is a separate check. | design §4.3; no protocol change |
| 8 | A selector exception during the diagnostics pass is intentionally fatal for the run. | design §4.6, §6; no protocol change |
| 9 | G0 test path made absolute. | §9 G0 |

### From v2 to v3 (second design review)

- **Negative-audit gate.** Now "zero withheld proteins among negatives". Retained proteins stay in pools at k ≥ 1; their share is descriptive. Finding 1.
- **Code pin.** Adds `trials_domain/`; imported-module coverage is checked at pin time and per run. Finding 2.
- **Missing data.** Fixed definedness rules for Δ_small, D_c, E_c; replicates average over defined cells; empty replicates are discarded and counted; > 5% discarded → unstable/inconclusive; the 10% rule counts over n_targets × 6 × 5 tuples. Finding 3.
- **Target-set size and packing.** 10–11 pairs → exactly the first 10; a complete packing fallback; condition and run counts read from the target table. Finding 4.
- **A1.** Target-specific excess E_c uses the same run's collateral Δ; neg_only's k = 0 Δ is no longer subtracted. Finding 5.
- **Mechanism (a).** Requires two-way bootstrap ci_low > 0. Finding 10.
- **simGIC IC.** Stated as corpus-wide, label-free and identical across cells. Finding 11.
- **λ pilot.** Selector epoch set to the checkpoint epoch. Finding 15.
- **G6.** Adds the p01_k5 audit.

### From v1 to v2

Each change responds to the first design review:

- **Pairs.** Copied verbatim instead of requiring a byte-identical converter rebuild. Finding 1.
- **Withheld set.** Now T_train(t), covering any role, while n(t) and retention stay on P_train(t). Pack-level retention feasibility is required. Findings 2 and 5.
- **Run-level checks.** Failed-batch and L_GO-applied checks, and the run-time negative-identity and `ok_batches` checks, are now mandatory. Findings 3 and 12.
- **λ pilot.** Uses batches of 32 over all target-free grade-2 validation records, with ≥ 4 batches required. Finding 7.
- **Interpretation.** Ordered first-match rules and an operational mechanism rule. Finding 8.
- **Primary uncertainty.** The two-way pack × seed cluster bootstrap; the v1 scheme becomes S1. Finding 9.
- **Pins.** The content-hash code pin covers untracked files. Pins live in `protocol_pins.json`. Findings 10 and 22.
- **Paths and commands.** All paths and commands are absolute and runnable verbatim. Finding 11.
- **Clip share.** Labelled as an eval-mode diagnostic. Finding 19.
