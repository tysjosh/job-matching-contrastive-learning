# Preregistered protocol: typed GO structure for GO-term prediction under per-term label scarcity

Version 4 (DRAFT, revised after the round-3 `design_review_go_function.md` and the orchestrator decision: closed-world W2/W1, D19; changes in §16, review responses in design §18). Technical design: `design_go_function.md` in this folder, cited "design §n". Spec: `findings_go_function.md`, cited "spec §n". `$R` is `/Users/olukotunjosh/Downloads/CDCL` locally; every path in configs, pins and run specs is repo-relative.

Freezing works in two steps:

1. Every hypothesis, contrast, metric, seed, threshold and rule here is frozen when this protocol is approved. The decisions in §2 may be amended by the user or supervisor until gate G8 (code pin); each amendment is a §14 row.
2. Artifact hashes "recorded at gate" are written by the scripts to `results/go_function/protocol_pins.json` and mirrored by hand into §13. Recording a hash is not a protocol change. After G8 nothing in this document changes except by a §14 amendment made before any grid output (G10) is read.

## 1. Question and hypotheses

Under per-term label scarcity, does *correctly typed* GO structure improve prediction of the scarce term on sequence-novel proteins? Which mechanism helps (train-time axiom loss, text-defined term embedding, inference-time fusion), and is the effect moderated by structural support? Separately: how much of DeepGOZero's (DGZ) reported zero-shot BP performance comes from its label propagation through non-is_a relations?

Notation: AUC_X(t, k, s) is the class-centric test AUC (§7) of target t for arm X at retention k and seed s, TP labels and W2 unless stated. Δ_{X−Y} = AUC_X − AUC_Y, paired within (t, k, s). Δ^small = mean over k ∈ {0, 1, 5} of Δ (defined only if all three are present, §8).

| id | hypothesis | contrast (preregistered, Holm family per aspect) | predicted sign |
|---|---|---|---|
| H1 | separate relation vectors beat one shared vector over part_of + the regulates family (untyped-relation control) | **C1** = Δ^small_{V3−V3c} | > 0 |
| H2 | DGZ-style axioms still give zero-shot benefit under the leak-free protocol | **C2** = Δ_{V4−V0} at k = 0 | > 0 (tested; null is informative) |
| H3 | DGZ's as-released zero-shot setup (DGZ-rule labels, withheld column dropped, no re-propagation) inflates V4's zero-shot AUC relative to true-path labels under leak-free subtree withholding | **C3** = AUC_{V4,DGZ,WDGZrel}(t,0,s) − AUC_{V4,TP,W2}(t,0,s), TP test labels (D17) | > 0 |
| H4 | a text-defined term tower helps scarce terms | **C4** = Δ^small_{V5−V0} | > 0 |
| H5 | axioms add to a text term tower | **C5** = Δ^small_{V6−V5} | > 0 |
| H6a | sibling support moderates the best arm's benefit | **C6** = mean_{high S_sib} Δ^small_{Vbest−V0} − mean_{low} | > 0 |
| H6b | (BP) a supported logical definition moderates it | **C7** = mean_{LD yes} Δ^small_{Vbest−V0} − mean_{LD no} | > 0 |

Vbest (D11): the arm in {V1, V2, V3, V4, V5, V6} with the largest point estimate of Δ^small_{X−V0} in that aspect, chosen by `analyze.py` before C6/C7 are computed; ties go to the lower-listed arm. Strata are balanced (equal targets per stratum), so the selection uses the strata average and C6/C7 test their difference. With balanced strata the selection statistic (stratum mean) is close to orthogonal to the tested statistic (stratum difference), so selection bias in C6/C7 is small but not zero. To show it, C6 and C7 are also reported for every arm in {V1, …, V6} descriptively (unadjusted), next to the Vbest result.

**Decomposition of C3 (descriptive, unadjusted, same bootstrap draws).** On the same (t, s) cells at k = 0, all V4, TP test labels:

- C3_leak = AUC_{DGZ,WDGZrel} − AUC_{DGZ,W2drop}: **the protocol's propagation channel**, covering descendant, ancestor and non-is_a filler labels of t-positives. It compares no erasure/re-propagation with subtree erasure; both drop column t;
- C3_col = AUC_{DGZ,W2drop} − AUC_{DGZ,W2}: dropping column t vs training it on negatives only (risk 6);
- C3_reg = AUC_{DGZ,W2} − AUC_{TP,W2}: DGZ-rule vs true-path labels, both leak-free.

C3 = C3_leak + C3_col + C3_reg exactly, on the cells where all four runs are complete (asserted by `analyze.py`, §8).

**Channel reads of C3_leak (descriptive, unadjusted, same draws).** On the V4 WDGZrel and W2drop runs at k = 0, with TP test labels for t, each read is the class AUC for t of the maximum logit over a channel's columns, WDGZrel − W2drop. The channel sets are fixed in the target table at G6 (design §7.3):

- **C3_leak_X:** X(t) = DGZ-only ancestors of t. These are the fillers reached through regulates-family, has_part, occurs_in or cross-namespace edges, i.e. the relation-conflation channel. It is the controlled analogue of G2b L3.
- **C3_leak_desc:** Desc(t) = strict TP descendants of t.
- **C3_leak_anc:** Pa(t) = TP parents of t.

A read is defined only on targets whose channel set is non-empty; the number of such targets is reported. The reads are not additive. They decide only the attribution wording of §9.2 and §9.3. C3_leak_desc is also reported stratified by |Desc(t)| (≤ 5 vs > 5 columns), because a max over many columns is noisier and biased upward. The channel reads are compared with each other by CI sign only, never by size.

Families: BP (primary) {C1…C7}, 7 contrasts. MF (secondary) {C1…C6}, 6 contrasts (no logical-definition stratum in MF).

## 2. Decisions (amendable until G8)

| id | decision | rationale |
|---|---|---|
| D1 | BP primary, MF secondary. CC out of scope for now. | BP is the only aspect with is_a, part_of and the regulates family plus 673 logically defined candidates (spec §4, §5). |
| D2 | Protein features: ESM-2 650M mean-pooled (primary, computed on the L40S). InterPro binary vectors from the DGZ pickles (secondary), used for exact DGZ comparability and for **all** local smoke and gate runs. | Protein text is near-circular for GO (spec §2); InterPro reproduces DGZ's inputs exactly. |
| D3 | MF labels: DGZ evidence set including IPI (primary); IPI-excluded labels as a sensitivity analysis. | DGZ comparability; IPI is interaction-derived (spec §3). |
| D4 | Withholding W2 primary, W1 secondary, both closed-world (D19). | W2 changes the number of annotated positives; withheld positives become unannotated (closed-world negatives). W1 measures parent-level annotation vs no annotation in the branch (spec §5). |
| D5 | Include the text term tower V5/V6: frozen MPNet on term name + definition, then a projection. Text is used only on the term side, never the protein side. | Zero-shot without axioms; links to the repo's text-encoder work. |
| D6 | No external time split for now; future work. | Scope. |
| D7 | DGZ propagation leak: C3 (V4 under DGZ-rule labels vs true-path labels at k = 0, D17) is a preregistered primary contrast. Gate G2b scores DGZ's released `deepgozero_zero.th` and released zero-shot predictions with our evaluator if they load. The leak is reported as a central result only if G2b confirms it (§9.3). | User decision 7. |
| D8 | V6 uses the V4 axiom set (full go.norm). | "Best typed EL" cannot be chosen on test without circularity; V4 is the complete typed set (design §6.2). |
| D9 | **Withdrawn in v4** (replaced by D19). It masked Lost(p), the labels a withheld protein loses only through the erased annotations. Lost(p) entries are now ordinary 0 labels. | A row-selective mask marks the withheld proteins (round-3 Finding 1). |
| D10 | DGZ's EL formulas are used unchanged, including its NF4 radius indexing (rc = radius of the class whose index equals the relation id). The corrected form is a G4 diagnostic only. V3c shares one relation vector but keeps V3's per-relation radius slots (A5), so the indexing is identical in V3 and V3c. | V4 = DGZ as specified. The formula is shared by every EL arm, and with A5 V3 and V3c read the same radius slots, so C1 contrasts relation-vector typing only. |
| D11 | Vbest rule as in §1. | Makes "support × (Vbest − V0)" fully specified before data. |
| D12 | Epoch budget E* from a target-free pilot (§6.4). Axiom weight λ = 1 (DGZ's value) fixed a priori for every EL arm; the λ sweep is recorded, never used to select. | Validation never sees a target; one setting for all arms avoids variant-specific selection (lesson in `findings.md`). A validation-BCE criterion for λ measures what the EL term competes with and would bias λ toward 0, nulling the ladder by tuning (review finding 2). |
| D13 | The 30%-identity split sensitivity of spec (i) is deferred: no 30% split file is in DGZ's released `data/` listing. Adding it later is an amendment with a re-split protocol. | Re-splitting breaks DGZ comparability and is a new study design. |
| D14 | Primary class-centric metrics use raw logits of the target column (no score propagation). Protein-centric metrics use TP max-propagation. | At k = 0 all D*(t) heads lack positives; propagation would mix descendant heads into t. |
| D15 | Fusion F is weight-free (design §10) and descriptive. | No target-free way to tune a fusion weight for a withheld term. |
| D16 | Vocabulary = DGZ `terms.pkl` in every regime (a deviation from spec (i), "terms with ≥ 1 train annotation after relabelling"). Columns with zero TP train positives are reported, kept in training and excluded from the collateral set. | One column set is needed to pair the TP and DGZ regimes (C3) and to load DGZ checkpoints. |
| D17 | The DGZ side of C3 uses DGZ's as-released protocol (WDGZrel: DGZ labels unchanged, withheld columns dropped from the loss); the TP side uses W2. C3 is a declared two-factor contrast {regime, rule}, decomposed exactly into the three single-factor parts of §1. | The leak needs both DGZ-rule labels and no re-propagation. A regime-only contrast under W2 removes the leak on both sides (review finding 1). This form keeps the user's "DGZ-rule vs true-path labels at k = 0" contrast and makes it able to detect the leak. |
| D18 | A descriptive retention re-draw (second retention seed) for V0 and V4 at k ∈ {1, 5}, seed 13. | Seeds do not sample which proteins are retained (risk 13). |
| D19 | **Closed-world W2 and W1** (a deviation from spec §5 "Masking"; **approved by the user and supervisor on 2026-10-12**, before G8). Withheld proteins are not loss-masked: after erasure and re-propagation, their D*_ρ(t) and Lost(p) entries are ordinary 0 labels in the BCE, exactly as for a protein never annotated in t's subtree. W1 additionally sets the strict ancestors of t positive, with no mask. The only train masks are the WDGZrel/W2drop target columns over all train rows; no mask is row-selective on a target-related column (design §5.5 assertion 10). Spec §5's "the curve changes exactly one thing" reads: the number of annotated positives; withheld positives become unannotated (closed-world negatives). | A mask on exactly W = P(t) \ R_k is the withheld label itself: with a shared encoder it leaves a "hole in the negatives" around W, so k = 0 would not be zero-shot and arm contrasts would be biased by how freely g(t) can use the mask (round-3 Finding 1). Real annotation scarcity has no known unknowns, and DGZ's own BCE is closed-world. The run count is unchanged. |

## 3. Data and labels

- **Source.** DGZ public release, SwissProt 2021_04 experimental set with their 50%-identity split: `bp/` and `mf/` `{train,valid,test}_data.pkl`, `terms.pkl`, `terms_zero.pkl`, `interpros.pkl`, `test_diamond.res`; `go.obo` (releases/2021-11-16), `go.norm`, `definitions_go.txt`, `swissprot_exp.fa`. Hashes in §12 (from `setup_baseline_go_function.md`, verified at G1).
- **Split.** DGZ's split, unchanged. BP 47,733 / 5,552 / 5,444; MF 34,716 / 3,851 / 4,712 proteins. No re-split.
- **Vocabulary.** DGZ `terms.pkl` per aspect (BP 21,381; MF 6,868), identical for every arm and regime. Alt-id columns whose primary is also a column are trained but never targets and never evaluated.
- **Evidence.** EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC, HTP, HDA, HMP, HGI, HEP (DGZ `EXP_CODES`), from `exp_annotations`, filtered to the aspect's namespace; alt ids mapped to primary ids; obsolete ids dropped (counted). MF sensitivity: the same without IPI, rebuilt from `annotations`.
- **Label regimes.**
  - **TP (primary, every arm unless stated):** propagation over is_a + part_of within the namespace; regulates, positively/negatively_regulates, has_part, occurs_in, happens_during, ends_during are never propagated; root removed.
  - **DGZ (control):** DGZ's rule, which treats every `relationship:` as is_a across namespaces and keeps alt ids as raw strings, applied to the unfiltered `exp_annotations`. It reproduces `prop_annotations` exactly (verified at G1 on every protein).
  - **Direct sets per regime (design §2.2):** A_TP and A_TPnoIPI are aspect-filtered and primary-mapped. A_DGZ is the raw `exp_annotations` (all namespaces, alt ids kept). Labels, W2 erasure and checks use the regime's own set.
  - **Alt ids:** under DGZ's rule, a protein annotated only with an alt id of d is not a d-positive; under TP it is. Targets with alt ids are excluded (§4).
  - **TPnoIPI (MF noIPI block only):** TP propagation on direct annotations without IPI.
- **Evaluation labels.** Every block except MF noIPI evaluates on TP test labels and the evaluation set below, including the DGZ-regime runs. The **MF noIPI block trains and evaluates on TPnoIPI labels**. Its evaluation set is the MF test proteins with a non-empty TPnoIPI MF closure that is not protein-binding-only. n_train and n_test for its target drop rule (§6.2) are computed on TPnoIPI train labels and that evaluation set. DGZ-rule test labels are a descriptive sensitivity read only (and the gate label set, §10).
- **MF evaluation set** excludes test proteins whose MF labels are only protein binding (GO:0005515 and its ancestors). They stay in training. *Data note (A2):* on the pinned release GO:0005515 is never a direct MF annotation, so this set is empty (0 proteins under TP and TPnoIPI); the rule is unchanged.
- **Vocabulary columns with zero TP train positives** (D16) are reported and excluded from the collateral term set.
- **Protein features.** ESM-2 650M mean-pooled, sequences truncated to 1,022 residues (primary); InterPro binary (secondary). No protein text anywhere.

## 4. Target selection (frozen at G6, `build_targets.py`)

Per aspect, on TP labels (design §4 for the algorithm):

- **Base filter:** primary vocabulary column; not root, GO:0005515 or GO:0005488; n_train(t) ≥ 30; n_test(t) ≥ 10 on the evaluation set; n_train(t) ≤ 5% of train proteins.
  - **No alt id:** no vocabulary column is an alt id of t, and no train, valid or test protein has an alt id of t in `exp_annotations`. The number excluded is reported; a design-time check gives about 14 BP and 5 MF before the n_test filter.
  - Why: this makes P_TP(t) ⊆ P_DGZ(t) hold for every target, and stops a duplicate column of t leaking t-positives into DGZ-regime conditions (design §4.1).
- **Support:** S_sib(t) = number of distinct train proteins annotated to is_a siblings of t (terms sharing an is_a parent, excluding t, its ancestors and descendants) and not annotated to t. low = S_sib ≤ 33.33rd percentile, high ≥ 66.67th, over base-filtered candidates; middle excluded.
- **Logical definition (BP):** LD yes = t has an `intersection_of` with at least one differentia filler F with n_train(F) ≥ 30, where n_train(F) = |{p ∈ train : F ∈ closure_TP(p)}| is computed from the TP closure whether or not F is a vocabulary column; LD no = no `intersection_of`; otherwise excluded.
- **Cells and quotas:** BP 4 cells (S low/high × LD yes/no) × 12 = 48. MF 2 cells (S low/high) × 24 = 48. n-matching by band quotas: n_train bands 30–99, 100–249, 250+ get 4 (BP) or 8 (MF) targets per cell, with the fixed borrowing order of design §4.7.
- **Global constraints:** no target is an ancestor of another; pairwise protein Jaccard ≤ 0.2.
- **Shortfall:** if any cell cannot reach its quota after borrowing, the study stops (exit 3) and the user decides.
- **Packs:** 8 packs × 6 targets, deterministic assignment (design §4.9; placement order by decreasing n_train(t), ties by GO id, packs ranked by (same-cell count, |∪ P_TP| after adding, pack id), amendment A1 in §14). Pack-mates are outside each other's 2-hop neighbourhood, have disjoint term-level descendant sets D*^GO(t) (all GO terms whose closure contains t, columns or not) under both regimes, and their union of TP-positive train proteins is ≤ 10% of train proteins. No feasible packing → stop (exit 3).
- **Retention:** k ∈ {0, 1, 5, 10, 25} plus `all`. R_k = the first k proteins of a per-target seeded permutation of P_train(t) (`RETENTION_SEED = 20261011`, design §4.10); nested; shared by every arm, regime, rule and seed. Draw 2 (`RETENTION_SEED_2 = 20261012`) is used only by the descriptive re-draw block (D18). The noIPI block applies the same seed formula to P_train(t) under TPnoIPI labels.

The target table and its sha256 are recorded at G6. Target names are not inspected before then. No target may be added, dropped or re-stratified after G6.

## 5. Withholding rules

Per regime ρ (TP, DGZ or TPnoIPI), for target t with term-level descendant set D*^GO_ρ(t) (GO terms whose ρ-closure contains t), its column restriction D*_ρ(t), and withheld set W = P_ρ(t) \ R_k:

- **W2 (primary, closed-world, D19).** For p ∈ W:
  1. Delete from the regime's direct set A_ρ(p) every annotation in D*^GO_ρ(t). Under DGZ this set is the raw `exp_annotations` of all namespaces, so, for example, an MF annotation `part_of` a BP target is deleted.
  2. Re-propagate.
  3. No mask is added. The erased entries (D*_ρ(t), and Lost(p), the labels lost only through the erased annotations) are ordinary 0 labels in the BCE, exactly as for a protein never annotated in t's subtree. Non-P(t) proteins stay negatives at every k, so the curve changes the number of annotated positives; withheld positives become unannotated (closed-world negatives).
- **W1 (secondary, closed-world, D19).** As W2, then the strict ancestors of t are set positive for p ∈ W. No mask.
- **WDGZrel ("DGZ as released"; DGZ regime, k = 0; used by C3).** Train labels are DGZ's propagated labels (`prop_annotations` restricted to the vocabulary), unchanged. There is no erasure and no re-propagation. The columns of every pack target t (every column c with primary(c) = t; after §4 this is t alone) are masked for every train protein. This is equivalent to dropping t from the vocabulary, as DGZ did. Descendants, ancestors and DGZ-only fillers of t keep their t-derived positives. These labels match DGZ's released labels (`prop_annotations ∩ V`). On the withheld column, they equal `zero_annotations` for 6 of 7 BP classes (design §0.2).
- **W2drop (descriptive; DGZ regime, k = 0).** The closed-world W2 DGZ-regime labels at k = 0, with the columns of t masked for every train protein (the only mask). It is used only for the C3 decomposition (§1).
- The v1 rule W_DGZ (remove t's own annotations, re-propagate, mask t) is withdrawn. It was neither DGZ's released protocol nor leak-free.
- **Validation** is target-free: the columns D*_ρ(t) of all 48 targets are excluded from every validation loss. Validation is used only by the pilots (§6.4) and the G4 reproduction, never to select a study checkpoint.
- **Test labels** are never modified.
- **Checks (every condition at build, again in every run's preflight; fatal):** every D*(t) entry of every p ∈ W is 0 and not masked; exactly k train positives for t, equal to R_k; W1 ancestors positive; test and validation label hashes unchanged; no label change outside ∪P(t); WDGZrel labels identical to the base DGZ labels and mask = the pack's target columns for all train rows; W2drop labels identical to the W2 DGZ k = 0 condition and mask = the target columns for all train rows; no column whose primary id is a target has an unmasked train positive outside R_k, and under WDGZrel/W2drop (k = 0 only) no unmasked train entry at all; R_k ⊆ P_TP(t) ⊆ P_DGZ(t) for every target; closed world: the W1/W2 train mask is empty, every WDGZrel/W2drop masked column is masked on all train rows, and no mask is row-selective on a target-related column (design §5.5, assertions 1–10).

## 6. Arms, grid and training

### 6.1 Arms

Model s(p, t) = σ(⟨f(x_p), g(t)⟩ + b_t), BCE over the vocabulary; f = DGZ's MLPBlock + residual MLPBlock (1024-d).

| arm | term tower | ontology use |
|---|---|---|
| V0 | free vector (= DGZ MLP) | none |
| V1 | EL n-balls | NF1 (is_a) only |
| V2 | EL n-balls | + part_of NF3/NF4, typed vector |
| V3 | EL n-balls | + regulates, positively_, negatively_regulates (3 more typed vectors) |
| V3c | EL n-balls | V3's axioms with **one shared** relation vector over part_of and the regulates family (untyped-relation control; still distinct from is_a; DGZ's own conflation is in its labels and is tested by C3) |
| V4 | EL n-balls | full go.norm NF1–NF4, every relation typed (= DGZ) |
| V5 | frozen MPNet(name. definition) → linear projection | none |
| V6 | as V5, ball centres from the projection | + V4 axioms (D8) |
| F | V0's scores at inference | z(own) + z(parent mean) − z(sibling mean) (D15), no retraining |
| DS | DiamondScore on the condition's train labels | none (descriptive reference) |

### 6.2 Grid (per aspect; 8 packs; k ∈ {0,1,5,10,25} → 40 pack conditions + 1 `all`; seeds {13, 21, 42, 87, 123})

| block | feature | arms | regime | rule | conditions | runs |
|---|---|---|---|---|---|---|
| main | ESM | V0, V1, V2, V3, V3c, V4, V5, V6 | TP | W2 | 41 | 1,640 |
| propagation control | ESM | V0, V4 | DGZ | W2 | 41 | 410 |
| W1 | ESM | V0, V4, V5 | TP | W1 | 8 × {0,1,5} = 24 | 360 |
| DGZ protocol (C3) | ESM | V4 | DGZ | WDGZrel | 8 × {0} | 40 |
| C3 decomposition | ESM | V4 | DGZ | W2drop | 8 × {0} | 40 |
| InterPro replicate | InterPro | V0, V4 | TP and DGZ | W2 | 8 × {0} + `all` = 9 | 180 |
| InterPro DGZ protocol | InterPro | V4 | DGZ | WDGZrel | 8 | 40 |
| InterPro C3 decomposition | InterPro | V4 | DGZ | W2drop | 8 | 40 |
| retention re-draw (D18) | ESM | V0, V4 | TP | W2, draw 2 | 8 × {1, 5} = 16, seed 13 only | 32 |
| MF noIPI (MF only) | ESM | V0, V4, V5 | TPnoIPI | W2 | 8 × {0,1,5} + `all` = 25 | 375 |

BP: 2,782 runs. MF: 3,157 runs. Total 5,939. The C3 inputs on the TP side (V4, TP, W2, k = 0) and DGZ-W2 side come from the main and propagation-control blocks. Every block reads the same target table. The noIPI block recomputes P_train(t) and retention under TPnoIPI labels. It drops (and reports) targets with n_train < 30 or n_test < 10, computed on TPnoIPI train labels and the TPnoIPI evaluation set (§3).

### 6.3 Training (fixed)

Adam lr 5e-4, MultiStepLR milestones [5, 20], γ 0.1, batch 37, dropout 0.1, embedding and hidden dimension 1024, EL margin 0.1 (all DGZ's). Loss = masked mean BCE + λ·EL with λ = 1 (EL arms; D12). Epochs = E* for every arm, k, rule and regime within an (aspect, feature); the final-epoch model is evaluated. Seeds control initialisation and shuffling; the retained-protein draw is fixed.

### 6.4 Pilots (G7, target-free; design §9)

- **E\*:** condition `all`, TP, V0 and V4 (λ = 1), seeds 13/21/42, ≤ 64 epochs. With i_best the 0-based index of the epoch with minimum target-free validation BCE in each run: **E* = clip(ceil(median over the six runs of (i_best + 1)), 8, 64).**
- **λ record (descriptive, D12):** V3 and V4, `all`, TP, seeds 13/21, λ ∈ {0.1, 0.3, 1, 3}, E* epochs; final target-free validation BCE and the λ·EL/BCE gradient-norm ratio are reported. λ stays 1 whatever the record shows.
- **Text-arm epoch record (descriptive):** V5 and V6, `all`, TP, seed 13, ≤ 64 epochs, ESM pilot only. For each, i_best + 1 and the validation BCE at E* are recorded. Neither enters E* (risk 14).
- One E* per (aspect, feature). Test data are never read by the pilots: pilot-mode runs load no evaluation labels or test features and write an `eval.json` stub and no `scores.npz` (A4).

## 7. Metrics

- **Primary:** class-centric AUC of target t on the evaluation-set test proteins (positives: TP test label for t), from raw logits of column t (D14). For arm F, its fused score.
- **Secondary:** class-centric AUPR of t.
- **Collateral:** protein-centric Fmax, Smin and micro-AUPR (DGZ's formulas, TP max-propagation) over the non-target terms of the run (primary vocabulary columns with ≥ 1 TP train positive, minus the pack's D*_TP(t); TPnoIPI in the noIPI block).
- **Sensitivity reads:** AUC with DGZ-rule test labels; seed spread; S1 hierarchical (target-then-seed) bootstrap.
- **Mechanism diagnostics (descriptive):** per-epoch BCE and EL per normal form; gradient-norm ratio of λ·EL to BCE on each tower; target-free validation BCE.

## 8. Uncertainty

- **Unit and pairing.** (t, s) cells, paired across arms within the same (pack, k, seed). C3 pairs the V4 DGZ/WDGZrel and V4 TP/W2 runs of the same (pack, k = 0, seed); its components use the four runs DGZ/WDGZrel, DGZ/W2drop, DGZ/W2 and TP/W2 of that cell.

- **The Holm-tested C3** is estimated on its own cell set: every cell where its two runs are complete.
- **The decomposition table** estimates C3, C3_leak, C3_col and C3_reg on the common cell set where all four runs are complete, so the identity holds exactly. The table states that cell count. When the two cell sets differ, both C3 values are printed.
- **The channel reads** (§1) use the WDGZrel/W2drop cells and their own target counts.
- **Primary method:** two-way cluster bootstrap over packs × seeds (vendored `two_way_bootstrap`, design §13.2), B = 10,000, `numpy.random.default_rng(20261011)`. Each replicate draws 8 packs with replacement and one multiset of 5 seeds shared by all packs, and averages the contrast over the defined (t, s) cells of the drawn packs' targets × drawn seeds, with multiplicity. C6/C7: high-mean minus low-mean within the drawn cells. Output: estimate (full-data mean over defined cells), 95% percentile CI, two-sided p = min(1, 2·min(P*(≤0), P*(≥0))).
- **Multiplicity.** Holm at α = 0.05 within each aspect's family (§1). Per-k results, ladder steps, W1 − W2, F, DS, InterPro and noIPI reads, the C3 components and channel reads, C6/C7 for non-Vbest arms and the retention re-draw are descriptive and unadjusted. The re-draw read is Δ_{V4−V0} at k ∈ {1, 5}, seed 13, draw 2 vs draw 1, per target and averaged, with a target-level bootstrap CI.
- **Missing data.** A run that fails twice is missing. A (t, k, s) value of a contrast is defined only if both arms' runs are complete. Δ^small(t, s) requires all three of k ∈ {0, 1, 5}. A replicate with zero defined cells (C6/C7: zero high or zero low) is discarded for that contrast; more than 5% discarded → the contrast is "unstable" and classified inconclusive. If more than 10% of the (t, k, s, arm) values needed by the main block are missing, no confirmatory analysis is made and everything is labelled exploratory. No imputation, no outlier exclusion, no covariate adjustment.

## 9. Interpretation

### 9.1 Classification (ordered, first match; the unstable check first)

Effect floor 0.01 AUC.

```text
1. harm:              ci_high < 0 and estimate <= -0.01
2. supported:         p_holm < 0.05 and ci_low > 0 and estimate >= +0.01
3. practically null:  -0.01 < ci_low and ci_high < +0.01
4. below floor:       ci_low > 0 and estimate < +0.01
5. inconclusive:      everything else, and every unstable contrast
```

### 9.2 Readings

- **C1.** (relation-vector typing only: V3 and V3c share axioms, class tables and NF4 radius slots, A5.) supported → "typing relations in the axiom loss helps scarce terms". practically null → "relation typing in the EL loss does not matter at this scale"; reported as a CI-backed negative.
- **C2.** supported → "DGZ-style axioms give a zero-shot benefit that survives true-path labels and subtree withholding". practically null or harm → "DGZ's zero-shot benefit does not survive the leak-free protocol". Read together with the InterPro replicate (descriptive).
- **C3.** supported → "DGZ's as-released zero-shot setup inflates V4's zero-shot AUC relative to true-path labels under leak-free withholding". Attribution has two levels.
  1. **Protocol propagation leak.** The inflation is called a propagation leak only if C3_leak > 0 with its 95% CI excluding 0. Wording: "the protocol's propagation channel: descendant, ancestor and non-is_a filler labels of withheld positives are kept".
  2. **Typed-relation leak.** The leak is attributed to DGZ's conflation of relation types (the filler channel) only if, in addition, C3_leak_X > 0 with its 95% CI excluding 0 **and** G2b L3 is confirmed. Otherwise the report attributes the inflation to subtree non-erasure in general, names the channel read(s) whose CI excludes 0, and makes no typed-relation claim.

  C3_col and C3_reg are reported as two further shares: dropping column t vs training it on negatives only, and the label rule on non-withheld proteins. If C3 is supported but C3_leak is not, the report says the inflation comes from the protocol difference named by the largest component, not from a leak. Status as a *central* result follows §9.3.
- **C4/C5.** As stated in §1; C5 supported with C4 supported → "axioms add to text"; C5 null with C4 supported → "text alone carries the benefit".
- **C6/C7.** supported → "structural support moderates the benefit of Vbest" (named arm).
- **Collateral.** A collateral Fmax or micro-AUPR difference (arm − V0, same condition) classified harm at any k is reported as a cost next to any supported contrast.
- **MF** results are secondary and do not change BP readings.

### 9.3 DGZ leak: when it is a central result (D7)

G2b (§10) yields one of: **confirmed**, **not confirmed**, **not checkable** (checkpoint fails to load or its released AUCs are not reproduced).

| G2b | C3 | reported as |
|---|---|---|
| confirmed | supported, C3_leak CI > 0, C3_leak_X CI > 0 | central result: "DGZ's BP zero-shot performance is partly a label-propagation leak through conflated relation types. Leak-free withholding with true-path labels lowers it by C3; of this, C3_leak is the propagation channel, and the filler (typed-relation) channel is non-zero" |
| confirmed | supported, C3_leak CI > 0, C3_leak_X CI not > 0 | central result: "DGZ's BP zero-shot performance is partly a protocol leak (withheld positives keep their subtree-implied labels). Leak-free withholding lowers it by C3, of which C3_leak is the propagation channel". No typed-relation attribution; the channel reads are listed |
| confirmed | supported, C3_leak CI not > 0 | central result on DGZ's released model (G2b); the controlled inflation C3 is reported with its decomposition, without attributing it to the leak |
| confirmed | not supported | central result on DGZ's released model; the controlled re-test does not reproduce an inflation (both stated) |
| not confirmed or not checkable | any | secondary methodological note; C3 is reported as a contrast only |

## 10. Gates (in order; each must pass before the next)

- **G0 (local).** All unit tests in `go_function/tests` pass; vendored-parity tests pass; regression suites give exactly the `setup_baseline_go_function.md` results (334 passed / 3 skipped / 1 known failure in `go_ppi_domain/tests`; 108 / 1 in `trials_domain/tests`; 8 career/CVE). No file outside the allowed directories changed.
- **G1 (local).** Every input hash in §12 verified; gate downloads (design §16) hashed into `MANIFEST_gate.json`. Labels built; all design §2.4 checks pass, including DGZ-rule = `prop_annotations` on every protein of every split. Pins: label manifests (including the counts of zero-positive columns, D16), `go.norm` parse counts (NF1 80,942; NF2 11,841; NF3 11,810; NF4 19,594; 9 unparsed role axioms; `owl:Nothing` in exactly 30 lines, all NF2 right-hand sides).
- **G2 (local or remote CPU).** Our evaluator in DGZ-faithful mode on DGZ's released `predictions_mlp.pkl` and `predictions_deepgozero.pkl` (BP, MF) reproduces the published Table 3 values transcribed in `config/go_function/dgz_published.json` within ±0.005 (class-centric AUC, Fmax, Smin, AUPR). A miss → stop and ask (exit 3).
- **G2b (local or remote CPU; does not block G3).** DGZ zero-shot check:
  - L1: `deepgozero_zero.th` (BP, MF) loads strictly into our port with `terms_zero.pkl` and go.norm's zero classes.
  - L2 (DGZ-rule labels, DGZ-faithful evaluator mode, design §8.3): for each withheld class, our zero-shot AUC on each evaluation set DGZ used (test; train+valid+test) matches the AUC from the released `zero_<GO>_auc*.pkl` file of the same set within 0.01. The criterion is ≥ 6 of 7 classes for BP and ≥ 4 of 5 for MF. L1 and L2 are judged per aspect.
  - The §9.3 status (confirmed / not confirmed / not checkable) uses the BP L1, L2 and L3 results only.
  - MF has no L3. Its result is reported as "reproduced" (L1 ∧ L2) or "not reproduced", and it does not change the BP status. Released `predictions_deepgozero_zero.pkl` is scored with our evaluator and reported.
  - L3 (BP, TP test labels): for each withheld class t with a regulates-family filler X, AUC_{X→t} = class AUC of the zero model's trained X logit for t. Confirmed if, for ≥ 5 of 7 classes, AUC_{X→t} ≥ AUC_zero(t) − 0.02 and ≥ 30% of X's DGZ-rule train positives carry t.
  - confirmed = L1 ∧ L2 ∧ L3; not checkable = ¬L1 or ¬L2; otherwise not confirmed. Written to `results/go_function/gates/gate_dgz_zero.json` and pinned. The result decides only §9.3.
- **G3 (local).** `mlp.th` and `deepgozero.th` load strictly into our port (BP, MF); our forward + DGZ-rule score propagation reproduces the released predictions to max abs difference ≤ 1e-4. Fixes the norm layer type in `study.json`.
- **G4 (remote GPU).** Trainer reproduction with InterPro, DGZ-rule train labels, DGZ selection (best validation loss within 256 epochs), λ = 1, DGZ's EL formulas, scored with DGZ-rule test labels in DGZ-faithful evaluator mode (design §8.3): V0 and V4 seed 42 within ±0.01 of the published class AUC and Fmax for BP and MF. If not, seeds 13 and 21 are added and the 3-seed mean must be within ±0.01; otherwise stop and ask. Diagnostic only: V4 with the corrected NF4 radius (D10).
- **G5.** Features: InterPro matrices (local); ESM-2 embeddings with the fp32 cross-check (remote); MPNet term text (either). Pins: feature hashes, ESM and MPNet revisions.
- **G6 (local).** Target table (§4), conditions for every grid block (§5, §6.2), target-free validation masks. All checks pass. Pins: target table, condition manifests, validation masks.
- **G7 (remote).** Pilots: E* per (aspect, feature) and the descriptive λ record. Pins.
- **G8.** `materialize.py` writes every run spec and passes the single-factor audit (design §12.2). The user commits/tags or explicitly accepts the content pin; `sweep.py pin`. D19 was approved by the user and supervisor on 2026-10-12. Decisions D1–D19 are frozen from here.
- **G9 smoke.**
  - (a) local, InterPro, CPU: BP p01 × k ∈ {0, 5} × {V0, V3, V3c} × seed 13 × 2 epochs (outside the grid's `runs/`): preflight assertions, training, evaluation, `scores.npz`, F and DS on these runs, and `analyze.py --smoke` produce every output file.
  - (b) remote, ESM: the same cells at E*, plus V4, V6, and V4 under DGZ with W2, WDGZrel and W2drop at k = 0; GPU memory and seconds per run recorded. Projected grid wall time at the chosen worker count. If the projection exceeds 10 days, stop and ask the user (exit 3); options are listed in §11.
- **G10 (remote).** The grid, seed-major, via repeated `sweep.py run`.
- **G11.** `analyze.py` produces the long table, contrasts, classifications, descriptive reads and `decisions.json`.

## 11. Risks

1. **The leak may partly reflect GO-version drift or an unverified training setup.** G1 rebuilds DGZ labels on DGZ's own 2021 GO (drift removed); G2b checks the released zero model directly; §9.3 limits the claim.
2. **Small typed-relation effect.** A CI-backed null is a valid outcome (§9.2).
3. **EL port mismatch.** G2, G3, G4 in order; the study does not start without them.
4. **ESM homology transfer may flatten scarcity effects.** InterPro replicate (descriptive) and S_sib strata; the 30% split is deferred (D13).
5. **Only 8 packs as bootstrap clusters.** Percentile CIs from 8 clusters can under-cover; S1 (target-level hierarchical bootstrap) is reported beside every primary contrast. A disagreement in sign of the CI bound is stated.
6. **k = 0 under closed-world W2 trains t's column on all-negatives**, unlike DGZ's protocol (and WDGZrel), which drops the column. Every train protein, withheld positives included, is a 0 label for t (D19), which can depress t's zero-shot scores in every arm, and it is part of C3. C3_col = W2drop − W2 measures it, so the C3 decomposition separates it from the propagation channel C3_leak. It is a property of the closed-world W2 design, stated in the report.
7. **Compute.** 5,939 runs; per-run time is unmeasured. G9b stops for a user decision if the projection exceeds 10 days. Suggested reductions for the user, in order: drop the MF noIPI block (−375); W1 to V0 only (−480 over both aspects); MF main block to V0, V1, V4, V5, V6 (−615).
8. **Novelty overlap** (STAR-GO, DeepGO-SE). Positioning rests on the protocol and typed propagation, not on the model (spec (ii)).
9. **Text-tower leakage through term text.** Term names/definitions are curated GO text, not protein text; the protein side never sees text. Stated as a limitation.
10. **Fixed λ = 1** may be far from optimal for some arms or features. This is accepted: it is DGZ's value, it is applied identically to every EL arm, and the descriptive λ record shows how sensitive validation BCE is to it.
11. **C3 is two-factor** (regime and rule). It is the user's "DGZ-rule vs true-path labels" contrast in its leak-sensitive form (D17). Its single-factor parts are descriptive, so the Holm-adjusted claim is about the whole protocol difference, and attribution to the leak rests on the unadjusted C3_leak CI (§9.2).
12. **DGZ's zero model may have used `zero_annotations`** rather than `prop_annotations`. WDGZrel is fixed to `prop_annotations ∩ V`. The two columns differ by about 2.5k BP labels in total; the per-target difference is reported (design §5.2).
13. **The retention draw is fixed across seeds.** At k = 1 and 5 the identity of the retained proteins may dominate variance, and seeds do not sample it; only target-to-target variation does. The CIs therefore describe these specific draws. The descriptive re-draw block (D18) shows how large the draw effect is at seed 13.
14. **E\* is chosen from V0/V4 and applied to the text arms.** V5/V6 train a projection rather than free term vectors and may converge at a different epoch. One shared E* is kept: a per-arm budget would be variant-specific selection. The pilot records the V5/V6 best-validation epoch at `all`, seed 13 (§6.4), and the report states the gap from E*.
15. **Alt-id exclusion narrows the target pool** (§4; about 14 BP and 5 MF candidates before the n_test filter). The excluded terms have duplicate columns, whose labels differ between the regimes for reasons unrelated to relation typing. Excluding them is preferred over a per-regime column remapping, which would break the shared vocabulary (D16).
16. **C3_leak mixes three label channels** (descendant, ancestor, non-is_a filler). Only the filler channel is the typed-relation point of the study. The channel reads (§1) and the two-level attribution (§9.2) keep a leak through descendant heads from being reported as evidence about relation types. The channel AUCs are diagnostics, not a decomposition.

## 12. Inputs (hashes verified at G1)

Raw inputs, from `setup_baseline_go_function.md` (`dataset/go_function/dgz/`, full sha256 in `dataset/go_function/MANIFEST.json`):

| file | sha256 (prefix) |
|---|---|
| go.obo | 0a6b4876004554c7 |
| go.owl | af109c7b8fbec1db |
| go.norm | 870f2caa1abc57b3 |
| definitions_go.txt | 42c3f2df428f4e68 |
| swissprot_exp.fa | ca65b30bc59bc425 |
| bp/train_data.pkl, valid, test | 09bf45263f6c9227, d679ce708fad727d, bbbafafcf347edd3 |
| bp/terms.pkl, terms_zero.pkl | 2c79415a69c9d1ca, 8ea8a9128c2d8b0a |
| mf/train_data.pkl, valid, test | 87c0c5888364b8cd, 22ec90c2d94f02ca, 4481027c1fb6e4e5 |
| mf/terms.pkl, terms_zero.pkl | 5dfa6627d52b1699, 750ba52669d7bebd |
| interpros.pkl (all aspects) | db53f566f2ee54a9 |
| bp/test_diamond.res, mf/test_diamond.res | 01ff07881fc1b79c, 6b5a9915839afb83 |

The full values are copied into `config/go_function/input_hashes.json` at G1 (inside the code pin) and every consumer verifies against it. Gate downloads are added to the same file at G1. ESM-2 and MPNet revisions are pinned at G5 (MPNet: `e8c3b32edf5434bc2275fc9bab85f82640a19130`).

## 13. Pins (filled at each gate)

| pin key | artifact | gate |
|---|---|---|
| G1-labels | `preprocess/go_function/<aspect>/labels_*.json` manifests | G1 |
| G1-norm | go.norm parse counts and relation index | G1 |
| G2 | `gates/gate_dgz_eval.json` | G2 |
| G2b | `gates/gate_dgz_zero.json` (status) | G2b |
| G3 | `gates/gate_checkpoints.json`, norm type | G3 |
| G4 | `gates/gate_trainer.json` | G4 |
| G5 | feature manifests, ESM/MPNet revisions | G5 |
| G6 | target tables, condition manifests, validation masks | G6 |
| G7 | `pilots/<aspect>_<feature>/choice.json` | G7 |
| G8 | spec-set hash, git HEAD (if committed), `code_pin.json` | G8 |

## 14. Amendments

| id | date | gate | change | cause | confirmatory output seen? |
|---|---|---|---|---|---|
| A1 | 2026-10-11 | G6 (before any G6 pin or result) | Packing placement order (design §4.9): targets by decreasing n_train(t), ties by GO id; candidate packs ranked by (same-cell count, \|∪ P_TP over the pack\| after adding t, pack id). The selected 48-target set per aspect, the quotas and every packing constraint (two-hop, D*^GO_TP/D*^GO_DGZ disjointness, pack union ≤ 10% N_train) are unchanged; `build_targets.py` asserts the target set is identical before and after packing and records its hash. Deterministic; uses train counts only. | The v3 greedy (acceptance order; packs by (same-cell count, pack size, pack id)) placed 44/48 BP and 47/48 MF targets (exit 3), although a feasible packing exists under the unchanged constraints. Approved by the orchestrator. | no (no test outcome had been computed) |
| A2 | 2026-10-12 | before G8 (data note, no rule change) | §3 protein-binding-only exclusion: GO:0005515 never occurs in MF `exp_annotations` or `annotations` (0 train, 0 valid, 0 test), so the excluded set is empty under TP and TPnoIPI (`n_protein_binding_only` = 0 in both G1 manifests). Design §0.2's "1,024 proteins" is corrected. | Code review of the implementation (`review_go_function.md`, NIT 6). | no |
| A3 | 2026-10-12 | before G5 term text (no text embedded yet) | Term text of obsolete classes (design §3.3): go.norm names 728 GO classes that are obsolete in the pinned go.obo. Each takes the text of its own obsolete stanza: the name with one leading "obsolete " removed (all 728 names carry it), then the definition exactly as written (it keeps its "OBSOLETE." opening). Alt ids of an obsolete stanza take that stanza's text. No class gets a free vector; a class with no stanza, or an empty text, stays fatal. D8 (V6 = V4 axioms) and the text-only premise are unchanged. `term_text_mpnet.json` records `n_obsolete_text` and the rule. | `embed_terms.py` exited 2 on the 728 classes (FEAT-002; review finding 5). Orchestrator decision: option (a). | no |
| A4 | 2026-10-12 | before G7 (no pilot run exists) | Pilot runs (§6.4) never read test data. A `mode: pilot` run loads no evaluation labels, test features or target columns, makes no test forward pass, writes an `eval.json` stub `{"skipped": "protocol §6.4: pilots never read test data"}` and no `scores.npz`. The label `.npz` files bundle all splits and are still hash-verified as whole files at preflight; no test row is selected, scored or written. E*, the λ record and the text record are unchanged (they read only `epochs.jsonl`). No pilot output had been produced locally; any pilot test output produced before this change would be non-confirmatory and is excluded from every analysis and from the return tarball. | Review finding 2: pilot runs scored the 48 targets on test, against §6.4. Orchestrator decision: change. | no |
| A5 | 2026-10-12 | before G8 (only the 2 G9a smoke runs of V3c, mode smoke, existed; re-run) | NF4 radius slots in V3c (D10, design §6.5): DGZ's formula reads `go_rad[relation id]` as rc. V3c now shares one relation *vector* but keeps V3's per-relation ids for this radius lookup (a separate `nf4_rad` index; V3c and V3 read the same 4 radius classes). Every other arm is unchanged (`nf4_rad` = its relation ids, so V4 still reproduces DGZ). C1 = Δ^small_{V3−V3c} therefore contrasts relation-vector typing only. | Review finding 4: V3c read class 0's radius for all NF4 axioms while V3 read classes 0–3, so C1 was not single-factor. Orchestrator decision: option A. | no (smoke runs only; no target result read) |
| A6 | 2026-10-14 | G4 (V0 cells; the V4 cells and every other gate unchanged) | The G4 V0 cells (DGZ's MLP) are trained with the optimiser settings of DGZ's `deepgopro.py`, the script that produced the published MLP row: Adam lr 1e-3, MultiStepLR milestones [1, 3] gamma 0.1, MLPBlock dropout 0.3 (batch 37, 256 epochs, best validation loss, as before). These values are fixed in `go_function/train.py` `DGPRO_MLP`; a gate-mode V0 spec must carry exactly them and no other spec may (a spec `dropout` field, null everywhere else). The V4 cells keep `deepgozero.py`'s settings (= study.json), and every study, pilot and smoke run, V0 included, keeps study.json, so the grid's arms still differ in one factor only. The two earlier V0 gate runs (study.json settings) are kept as a record, renamed `<run>.studyhp1`, and do not enter G4. | The first BP/MF V0 gate runs selected epoch 0/1 and missed the MLP row (BP Fmax 0.441 vs 0.460, AUC 0.760 vs 0.793; MF Fmax 0.644 vs 0.657, AUC 0.872 vs 0.882): the gate compared our MLP under deepgozero.py's settings with DGZ's MLP under deepgopro.py's. Decided by the user on 2026-10-14. | Yes: the two V0 gate results above (G4 gate runs only; no pilot, grid or test-based study output exists). |
| X1 | 2026-10-14 | before G8 | Cross-reference only: a descriptive fusion add-on to this grid (S2) is preregistered in `protocol_go_function_fusion.md` §11 (version 3 at edit, sha256 `53325461b3f3d3010e432a780e8f6bb225543ed2bb88d06dec8ef2f7af2da9b8`; later changes are recorded in that file's §14). It reads grid outputs after G10 with S1-frozen parameters. **No go_function decision (D1–D19), arm, run, contrast, family, gate or code-pinned file changes; arm F (D15) is unchanged.** | User decision U5 of the fusion study. | no |
| A7 | 2026-10-15 | before G8 (G7 E* stage complete; no grid run, no test-based study output) | **Core A grid.** The grid is cut to the BP ESM-2 main block and the C3 cells: arms V0, V3, V3c, V4, V5, V6 under TP/W2 at k ∈ {0, 1, 5} on all 8 packs plus `all`; V4 under DGZ/W2, DGZ/WDGZrel and DGZ/W2drop at k = 0; study seeds 13, 21, 42 (`constants.SEEDS`, study.json `seeds`). 522 BP ESM-2 runs, plus the GO function fusion study's Tier B bases as ordinary grid cells (V0/V4-InterPro BP and MF, V0-ESM MF; `all`, seed 13; BP V0-ESM is already in the main block): 524 BP + 3 MF = 527 runs (`materialize.py` BLOCKS/TOTALS). Descoped: V1/V2 (the ladder reads become undefined), k = 10 and 25, W1, the re-draw, V0 under DGZ/W2, the InterPro replicate and its C3 cells, MF (except the Tier B cells) and the MF noIPI block. Vbest (D11) is chosen among {V3, V4, V5, V6}. The analysis's >10% missing rule and every seed matrix use the A7 arms, k values and seeds (`stats.MAIN_ARMS`, `ALL_KS`, `SEEDS`). Hypotheses, contrast definitions C1–C7, the Holm family, FLOOR, α, the bootstrap and the shared E* rule are unchanged. E* (G7 E* stage, rule unchanged): bp_esm2_650m 32, bp_interpro 11, mf_esm2_650m 36, mf_interpro 33 (study.json, inside the G8 code pin). The λ record (D12) and text record run for BP ESM-2 only, as descriptive records that do not block G8; they are descoped for the other three pilots, whose partial λ records (MF ESM-2: 3 of 16 runs; MF InterPro: none complete) are kept as is; pin G7 is recorded only when a pilot's records are complete. | GPU cost: the preregistered grid is about 2,800 GPU-hours (about 311 ten-hour reservations; `decision_sheet_go_function_grid.md`), which the available compute cannot run. Core A keeps every BP primary contrast on all 48 targets at about 320 GPU-hours. Decided by the user on 2026-10-15. | Yes: G4 gate results and the 24 G7 E* pilot runs (validation loss only). No grid run and no test result of any study arm exists. |

## 15. Remote-run plan (one L40S, 48 GB)

Local (Mac, CPU): G0, G1, G2, G2b, G3, InterPro features and MPNet term text (G5 part), G6, G9a. Remote (L40S): ESM embeddings (G5), G4, G7, G9b, G10, G11 (CPU on the remote is fine for G11).

1. **Transfer.** The user commits the study code (no step makes git writes) and pushes a branch; the remote clones it. Raw DGZ data are re-downloaded on the remote with `scripts/go_function/download_dgz.py` and `download_dgz_gate.py`, then verified against `config/go_function/input_hashes.json` (a mismatch is fatal). Derived files in `preprocess/go_function/` are rebuilt on the remote with the same scripts and must reproduce the G1/G6 pins exactly; no Mac-built derived file is shipped. Never ship `/300` or `reports/`.
2. **Environment.** `python3.13 -m venv .venv`; `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` (driver < 570: cu126); `pip install -r scripts/go_function/requirements-remote.txt`; assert `torch.cuda.is_available()`. Models: `hf download facebook/esm2_t33_650M_UR50D --revision <G5 pin>`, `hf download sentence-transformers/all-mpnet-base-v2 --revision e8c3b32edf5434bc2275fc9bab85f82640a19130` (full snapshot). Then `export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8`.
3. **Rebuild and verify:** `build_labels.py`, `build_features.py` (InterPro), `embed_terms.py`, `build_targets.py`, `make_conditions.py`, each checked against the Mac pins (`--verify-pins`).
4. **ESM:** `embed_esm.py --device cuda` (≈ 0.5 h estimated) with the fp32 cross-check; pin.
5. **G4:** `run.py` on the four gate specs (BP/MF × V0/V4), seed 42; `gate_trainer.py` compares with `dgz_published.json`.
6. **G7:** `pilot.py --aspect bp --feature esm2_650m`, then MF, then the InterPro pilots; pin.
7. **G8:** `materialize.py`, then `sweep.py pin` after the user accepts the code pin.
8. **G9b:** `sweep.py run --smoke`; read the projection; continue only if within 10 days or after the user's decision.
9. **G10:** in `tmux`, `sweep.py run --workers N` (N from G9b memory; default 4), seed-major; `sweep.py status` for progress; an interrupted run is re-claimed after `status --release-stale`; failed evaluations via `sweep.py rescore`.
10. **G11:** `analyze.py --runs results/go_function/runs --out results/go_function/analysis --seed 20261011 --B 10000`.
11. **Return:** a tarball of `results/go_function/{gates,pilots,specs,analysis,protocol_pins.json,code_pin.json}` and every run's `run.json`, `epochs.jsonl`, `eval.json`, `scores.npz`, `DONE.json`/`FAILED.json` (no checkpoints; the kept `all`/seed-13 checkpoints stay on the remote).

## 16. Changes

Version 1: first authoring.

Version 2 (response to `design_review_go_function.md`; full table in design §18):

- H3/C3 redefined (D17): the DGZ side uses the new WDGZrel rule (DGZ as released). C3 = V4 DGZ/WDGZrel − V4 TP/W2 at k = 0, with an exact three-part descriptive decomposition (C3_leak, C3_col, C3_reg) using the new W2drop rule. The v1 W_DGZ rule is withdrawn. §9.2/§9.3 readings are reworded so the leak attribution requires C3_leak > 0. (Finding 1; the reviewer's within-DGZ C3 is kept as C3_leak + C3_col rather than as the primary, to preserve the user's decision 7 wording.)
- λ fixed at 1; D12 covers E* only; the λ sweep is descriptive (finding 2).
- go.norm `owl:Nothing` (30 NF2 lines) pinned at G1 (finding 3; design §6.4).
- The MF noIPI block trains and evaluates on TPnoIPI labels with its own evaluation set and drop rule (finding 4).
- New D16 (vocabulary) and D18 (retention re-draw); risks 6 and 7 updated; risks 10–13 added; V3c renamed "untyped-relation control"; Vbest selection-bias note; D*^GO in pack constraints; one E* formula; G2b L2 and G4 label/evaluator mode stated (findings 5–15).
- Grid 5,715 → 5,939 runs (+40 W2drop per aspect per feature, +32 re-draw per aspect).

Version 3 (response to the round-2 review; full table in design §18):

- Targets with an alt-id column or alt-id annotation are excluded (§4). Target columns are masked through `primary` in DGZ-regime rules (§5). Generator assertions 8–9 are added, and G1 checks 3c/3d are stated at column level (design §2.4). (Finding 1, HIGH.)
- Per-regime direct sets A_TP, A_TPnoIPI and A_DGZ (raw, all namespaces). W2 erases from A_ρ, so cross-namespace annotations are erased under DGZ (§3, §5). (Finding 2.)
- Channel reads C3_leak_X, C3_leak_desc and C3_leak_anc (§1). C3_leak is renamed "the protocol's propagation channel". There is now a two-level attribution in §9.2, and the §9.3 table is split by C3_leak_X. (Finding 3.)
- G2b L2 MF criterion: ≥ 4 of 5. The §9.3 status uses BP only (finding 4). C3 cell sets are stated (§8; finding 5). Risk 14 and the V5/V6 epoch record are added (§6.4; finding 6). The WDGZrel wording is corrected (§5; finding 7). Risks 15 and 16 are added.
- Grid unchanged (5,939 runs); +2 descriptive pilot runs per aspect.

Version 4 (response to the round-3 review and the orchestrator decision; full table in design §18):

- **Closed-world W2 and W1 (new D19; D9 withdrawn).** Withheld proteins are not loss-masked: erased D*_ρ(t) and Lost(p) entries are ordinary 0 labels. W1 sets the strict ancestors of t positive, with no mask. WDGZrel is unchanged; W2drop = closed-world W2 labels + the target columns masked on every train row. D19 is a deviation from spec §5 "Masking" and needs user/supervisor sign-off before G8. D4 and risk 6 are reworded. (Finding 1, HIGH.)
- §5 checks: "0 and not masked" for D*(t) entries of W proteins; the W2drop mask is the target columns alone; new design assertion 10 (W1/W2 train mask empty; WDGZrel/W2drop masks cover whole train columns; no row-selective mask on a target-related column); assertion 8 states "(k = 0 only)". (Findings 1, 2.)
- LD filler support n_train(F) is computed from the TP closure whether or not F is a column (§4; finding 3). C3_leak_desc is also stratified by |Desc(t)| ≤ 5 vs > 5, and channel reads are compared by CI sign only (§1; finding 4). Wording fixes in §4, §8 and risk 15 (finding 5).
- Grid unchanged (5,939 runs).
- Amendment A1 (§14, 2026-10-11): packing placement order (design §4.9); target set and constraints unchanged.
Version 4, post-review amendments (code review `review_go_function.md`, 2026-10-12; orchestrator decisions):
- A2: data note, the MF protein-binding-only set is empty on the pinned release (§3).
- A3: obsolete go.norm classes take their own stanza's text (design §3.3).
- A4: pilot runs never read test data (§6.4).
- A5: V3c keeps V3's NF4 radius slots; D10 and the C1 reading updated (design §6.5).
- §7 sensitivity read "AUC with DGZ-rule test labels" is computed by `analyze.py` (`class_auc_dgz_labels`, `dgz_test_labels`).
