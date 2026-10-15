# GO protein-function fusion: investigation findings

Read-only investigation. No repo file, config, data or result was changed; nothing was trained or committed. Scratch work is in `/tmp/gofunc_fusion/`. It holds `a_sim_check.py`, `b_valid_pilot.py` and `c_struct.py`, their `*.json`/`*.log`/`*_perterm.npz` outputs, and `swissprot_exp.sim` (31 MB, DGZ all-vs-all identity file, sha256 `a143b389…c3101`). The downloads were CPU-only (`CUDA_VISIBLE_DEVICES=""`). No test data were scored: every number below comes from the DGZ **validation** split.

Evidence tags: [read] code/docs, [ran] computed here, [web] fetched URL, [inferred] reasoning, not verified.

## Summary answer

1. **In function prediction, "fusion" should mean combining a protein → term base score with a prior for the same (protein, term) cell.** The PPI study used the anchor's annotations; here the prior comes from the *same model's scores on GO-related terms*. The class-centric analogue of PPI's per-anchor z is **per-term z over proteins**.
2. **A weight-free parent prior is already a large, genuinely structural gain.** The rule is s(p,t) = z_t(ℓ_t) + w·z_t(mean ℓ over t's true-path parents), measured on validation with DGZ's released checkpoints [ran]:
   - BP MLP class AUC: 0.7745 → 0.7936 (+0.019).
   - MF MLP: 0.8747 → 0.9117 (+0.037).
   - The gain sits in scarce terms. BP n_train 1–10: 0.749 → 0.789. Natural zero-train terms: 0.590 → 0.718 (BP) and 0.630 → 0.856 (MF).
   - **A frequency-matched shuffled-parent control falls *below* the base** (BP 0.735, MF 0.846), so the gain needs the real GO edges.
3. **On scarce terms, an untrained MLP plus the parent prior matches or beats DGZ's train-time axioms.** On validation, MLP + scarcity-adaptive parent prior + homology gives 0.799 (BP) and 0.917 (MF). DGZ alone gives 0.777 / 0.895, and DGZ with the same fusion 0.796 / 0.917 [ran]. This is the most informative result, and it is new relative to DGZ.
4. **Homology (DiamondScore-style) fusion barely moves class-centric AUC but lifts protein-centric Fmax.**
   - Class AUC: +0.002. Fmax, BP MLP: 0.409 → 0.427. Fmax, MF: 0.589 → 0.596.
   - This matches DGZ's own Tables 3–4 (MLP+DS: AUC +0.004, Fmax +0.026 BP) [ran, web].
   - The homology channel is not ontology. It is the mandatory strong baseline/partner, not the contribution.
5. **Two other ontology fusions do not help:**
   - **Score propagation** (true-path max-over-descendants, or DGZ's conflated rule) changes class AUC by about 0 (±0.001). Max over children: −0.003 [ran].
   - **The preregistered go_function arm F** (parents **minus siblings**, D15) is worse than parents alone everywhere: BP MLP 0.777 vs 0.794, MF 0.894 vs 0.912. The sibling subtraction removes most of the gain [ran].
6. **Typed relations:** adding regulates-family targets to the parent prior helps on the 3,518 BP terms that have such edges (0.766 → 0.776). The untyped "any relation" parent set is ≈ TP. **Both reads are confounded on the released checkpoints**, because they were trained on DGZ-rule labels: the regulated-process column X has learned the positives of "regulation of X". A clean typed comparison needs true-path-trained base models (our V0, remote).
7. **Scope.**
   - **S1 first:** a standalone full-data fusion study on DGZ's split, validation-selected and tested once. It can start on the Mac now with released checkpoints. A validation Diamond run and TP-trained base models come from the remote.
   - **S2 second:** fusion as an add-on analysis of the go_function withholding grid. It needs no new runs, because grid `scores.npz` already stores t, Pa(t), Sib(t), X(t) and Desc(t) test logits. It does need frozen weights from S1 and a preregistration before any grid output is read.
   - go_function is pinned only to G6 (no G7/G8). A one-row §14 cross-reference amendment before G8 is the clean route, and it changes no go_function decision.

---

## 1. Candidate fusion definitions

Notation: ℓ_c(p) is a base model's logit for column c. z_t(·) standardises over the evaluated proteins, per term t: class-centric, the analogue of PPI `within_query_z` with the term as the "query". y^train is the condition's train labels.

| | definition score(p,t) = | ontology contribution | genuinely GO? | main leakage risks | fits withholding / scarce? | validation evidence [ran] |
|---|---|---|---|---|---|---|
| (a) homology | (1−α)·B + α·DS(p,t), DS = bitscore-weighted Σ_h y^train_h,t (`go_function/baselines.py` L4–12) | none. It is a homology/kNN prior (transfers labels, which are TP-closed) | **no** | DGZ's `test_diamond.res` DB = train **+ valid** (`run_diamond.sh`: `cat valid_data.fa >> train_data.fa`) and DGZ's `predict_diamond.py` merges valid labels into the source [web]. Our `diamond_matrix` drops non-train hits (L34–52) [read]. The 50% split holds: max test→train identity < 50% for every protein [ran] | at k = 0 under W2, DS(p,t) = 0 for all p (smoke DS AUC = 0.5 exactly, `results/go_function/smoke/analysis/baseline_DS.csv`) [read]. It only acts at k ≥ 1 | class AUC +0.002; Fmax +0.018 BP, +0.007 MF |
| (b) parent / true-path prior | z_t(ℓ_t) + w·z_t(mean_{π∈Pa_TP(t)} ℓ_π) (− z(mean sib) in the protocol's F) | the is_a/part_of graph: borrows the better-trained parent's ranking for the scarce child | **yes** | none from test labels: the parent column is trained on train labels. Under W1 the withheld proteins are parent-positive *by design*. Released checkpoints use DGZ-rule labels (see (c)) | **yes, the zero-shot channel.** At k = 0 it is the only signal besides the model's own column | +0.019 BP / +0.037 MF. Siblings hurt. Shuffled parents −0.04 |
| (b′) score propagation | max(P_t, max_{d∈Desc(t)} P_d) (TP or DGZ rule) | consistency, not information | yes, but no gain | — | no | Δ ≈ 0; children-max −0.003 |
| (c) typed label-propagation prior | (b) + w_r·z_t(mean ℓ over targets of relation r), r ∈ {regulates, positively_regulates, negatively_regulates} | uses typed edges: "regulation of X" gets X's score | **yes, typed** | **DGZ-rule-trained bases leak.** DGZ labels propagate "regulation of X" positives into X (findings_go_function §4: 37–84% of X's positives carry the withheld class), so ℓ_X already encodes t's positives. Use TP-trained bases only. The go_function grid's WDGZrel/W2drop channel reads (protocol §1 L36) measure the same channel | yes; BP only (3,518 BP eval terms with a regulates edge, 82 MF) | reg subset 0.766 → 0.776 (released MLP, confounded) |
| (d) term-text prior | z_t(ℓ_t) + w·z_t(mean ℓ over the k nearest terms by frozen MPNet(name + def) cosine) | text-defined term neighbourhood (no protein text) | partly: definitions are GO content, but neighbours largely coincide with parents/siblings by name [inferred] | none on the protein side, if only term↔term similarity is used. Protein-side text is excluded (near-circular, findings_go_function §2) | yes, k = 0 capable | not run: MPNet term embeddings are G5 (remote) |
| (e) co-annotation prior | z_t(ℓ_t) + w·z_t(Σ_c P̂(t\|c)·σ(ℓ_c)), P̂ from y^train | data-driven label correlation | **no.** It is the key "no-ontology structure" control | P̂ must come from the condition's train labels (W2-erased) | useless at k = 0 (P̂(t\|·) = 0), noisy at small k | not run |
| (e′) frequency prior | B + c_t (per-term constant) | none | no | — | **class-centric AUC is invariant to it**; it matters only protein-centrically (Naive) [inferred, exact] | — |
| (f) model fusion | z_t(ℓ^IPR_t) + w·z_t(ℓ^ESM_t) | none (ensemble) | no | ESM and InterPro both carry homology; the 50% split covers it | yes; best "practical" gain likely [inferred] | not run (ESM is remote) |
| (g) axiom + free fusion | z_t(ℓ^V4_t) + w·z_t(ℓ^V0_t) | train-time axioms, combined at inference | yes (indirectly) | none beyond the base models | yes; available from grid `scores.npz` for free | not run |

The PPI search found that exact terms beat hierarchy similarity. Here the hierarchy *is* the signal. The prior is the parent's *score*, not overlap with annotations, so the PPI result does not transfer as "hierarchy does not help" [inferred].

## 2. What exists already

- **Arm F is implemented** in `go_function/fusion.py` (L1–15, `fuse`, `fusion_scores`). It computes z(own) + z(Pa mean) − z(Sib mean), with Sib restricted to ≥ 10 train positives, weight-free, descriptive (protocol D15 L64, §6.1 L136). It runs over V0 under W2 at every k and W1 at k ∈ {0,1,5} (`scripts/go_function/analyze.py` L18–22) [read].
- **The DS baseline is implemented** in `go_function/baselines.py` (`read_diamond`, `diamond_matrix`, `diamond_score`, `naive_frequency`), per condition with W2 masking (`analyze.py` L258–269, L309–330) [read]. A hand-example test is `go_function/tests/test_analysis.py::test_diamond_score_hand_example` [read].
- **Smoke outputs** (`results/go_function/smoke/analysis/`): `fusion_F.csv` and `baseline_DS.csv` for BP p01 × k ∈ {0,5}, 2-epoch runs (`smoke_report_go_function.md` L33). They are not informative about effect sizes (2 epochs) [read].
- **Gate results on the Mac** [read]:
  - **G2** (`gates/gate_dgz_eval.json`, from released predictions): BP MLP Fmax 0.4596 / AUC 0.7934; DGZ 0.4514 / 0.7982. MF MLP 0.6566 / 0.8819; DGZ 0.6568 / 0.9028. All within tolerance of the paper.
  - **G3** (`gates/gate_checkpoints.json`): released `mlp.th`/`deepgozero.th` load strictly into our port and reproduce released predictions to ≤ 1.4e-6 after DGZ-rule max-propagation. Before propagation the max diff is 0.68 (MLP) / 1.0 (DGZ). **The released `preds` are already DGZ-propagated**: score propagation is baked into every DGZ number.
  - Pins: G1, G2, G2b, G3, G5, G6 (`results/go_function/protocol_pins.json`); **no G7/G8**.
- **On disk** (`dataset/go_function/dgz/`, [ran]):
  - bp/mf `predictions_{mlp,deepgozero,deepgozero_zero}.pkl`: test only, 0.94 GB each BP, 0.27 GB MF.
  - `{mlp,deepgozero,deepgozero_zero}.th`.
  - `train/valid/test_data.pkl`: BP 47,733 / 5,552 / 5,444; MF 34,716 / 3,851 / 4,712. Valid has InterPro for 97.8% / 98.6% of proteins, plus sequences.
  - `test_diamond.res`: BP 61,251 lines, 4,359 queries, 54,702 hits to train, 6,549 to valid, 0 to test. MF 51,917 / 3,811.
  - 78.9% (BP) and 80.1% (MF) of test proteins have ≥ 1 train hit.
  - **No validation Diamond file and no `diamond` binary** on the Mac (`which diamond` → not found).
- **Not on disk, available on bio2vec** [web]:
  - BP: `predictions_{mlp,deepgozero,deepgocnn}_blast.pkl` (1.41 GB each) and `predictions_deepgocnn.pkl` (0.94 GB).
  - MF: `*_blast.pkl` 0.40 GB each.
  - Also `train_data.dmnd` (includes valid), `{train,valid,test}_data.fa` and `swissprot_exp.{sim,dmnd}`.
  - `swissprot_exp.sim` is all-vs-all **percent identity only** (no bitscore). It holds 64% (BP) / 50% (MF) of the `test_diamond.res` test→train pairs, with identical pident. It looks like a less sensitive run [ran].
- **Released predictions alone cannot support an honest search: they cover test only.** The released checkpoints fix that. A forward pass on valid InterPro takes **1–3 s** on the Mac CPU (BP 5,552 × 21,381). A full validation fusion sweep (19 variants × 11,143 BP terms) takes about 90 s and 2.7 GB RSS [ran].

### Validation pilot (released checkpoints, TP valid labels, proteins with ≥ 1 label, terms with 0 < n_pos < n)

**Setup.**
- Eval rows: BP 5,389; MF 3,755. Terms: BP 11,143; MF 2,407.
- n_train is the TP train count. The 0-band holds terms with no train positive under either label rule; these are natural zero-shot terms (BP 366, MF 220).
- H is a proxy homology prior: identity-weighted kNN from `swissprot_exp.sim`, because no valid Diamond run exists.
- `adapt_tau30` uses w_t = 60/(30 + n_train(t)).
- Scripts: `/tmp/gofunc_fusion/c_struct.py` and `b_valid_pilot.py` [ran].

**BP, released MLP (class-centric mean AUC):**

| variant | all | n_tr=0 | 1–10 | 11–30 | 31–100 | 101–1000 | >1000 | regulates-edge subset | share of terms improved |
|---|---|---|---|---|---|---|---|---|---|
| own (base) | 0.7745 | 0.590 | 0.749 | 0.792 | 0.803 | 0.792 | 0.762 | 0.750 | — |
| + parents is_a | 0.7933 | 0.721 | 0.789 | 0.803 | 0.806 | 0.790 | 0.758 | 0.766 | 0.57 |
| + parents TP (w = 1) | 0.7936 | 0.718 | 0.789 | 0.804 | 0.806 | 0.790 | 0.758 | 0.766 | 0.57 |
| + parents TP, adaptive τ = 30 | 0.7970 | 0.742 | 0.795 | 0.804 | 0.807 | 0.793 | 0.762 | 0.768 | 0.59 |
| + parents any relation (untyped) | 0.7954 | 0.720 | 0.791 | 0.806 | 0.808 | 0.791 | 0.758 | 0.771 | 0.59 |
| + parents TP + regulates targets | 0.7968 | 0.727 | 0.794 | 0.807 | 0.808 | 0.791 | 0.758 | 0.776 | 0.59 |
| + all TP ancestors | 0.7899 | 0.739 | 0.787 | 0.798 | 0.800 | 0.784 | 0.755 | 0.759 | 0.52 |
| protocol F (parents − siblings) | 0.7767 | 0.622 | 0.767 | 0.791 | 0.797 | 0.780 | 0.740 | 0.749 | 0.46 |
| + children max | 0.7718 | 0.594 | 0.744 | 0.788 | 0.802 | 0.792 | 0.762 | 0.749 | 0.35 |
| + H only | 0.7764 | 0.590 | 0.750 | 0.794 | 0.805 | 0.795 | 0.765 | 0.750 | 0.22 |
| adaptive τ = 30 + H | **0.7988** | 0.742 | 0.796 | 0.805 | 0.809 | 0.796 | 0.765 | 0.769 | 0.62 |
| **control:** shuffled parents (n_train-decile permutation) | 0.7349 | 0.590 | 0.712 | 0.749 | 0.761 | 0.751 | 0.716 | 0.721 | 0.27 |

**Summary across bases (all / n_tr = 0 / 1–10):**

| aspect, base | own | + parents TP | adaptive τ30 + H | protocol F | shuffled parents |
|---|---|---|---|---|---|
| BP DGZ | 0.7773 / 0.542 / 0.762 | 0.7894 / 0.700 / 0.787 | 0.7962 / 0.733 / 0.793 | 0.7855 | 0.7251 |
| MF MLP | 0.8747 / 0.630 / 0.880 | 0.9117 / 0.856 / 0.921 | 0.9167 / 0.873 / 0.926 | 0.8936 | 0.8459 |
| MF DGZ | 0.8949 / 0.719 / 0.914 | 0.9141 / 0.874 / 0.929 | 0.9172 / 0.887 / 0.930 | 0.9006 | 0.8544 |

**Other reads.**
- **Weight, fixed w:**
  - BP MLP: w = 0.5 gives 0.790, w = 1 gives 0.794, w = 2 gives 0.795.
  - A large w helps scarce terms (1–10: 0.796) and hurts frequent ones (> 1000: 0.753).
  - Hence the scarcity-adaptive form. Its τ is flat over {10, 30, 100}: 0.796–0.797 BP MLP.
- **Score propagation and homology (`b_valid_pilot.py`):**
  - BP MLP: raw 0.7745 / Fmax 0.407. TP-prop 0.7738 / 0.409. DGZ-prop 0.7730 / 0.409.
  - Convex (1 − α)·B_tp + α·H: α = 0.4 gives 0.7755 / Fmax 0.426; α = 0.5 gives Fmax 0.427.
  - Shuffled-hits control: 0.7702.
  - MF MLP: Fmax 0.589 → 0.596 (α = 0.4).
- **Share of terms improved is about 0.5–0.6.** The mean gain is concentrated in scarce terms; the median term barely moves. The bootstrap unit must therefore be the term, and results must be reported by n_train band.

## 3. Honest-selection setup

- **Validation scores:**
  - Released checkpoints give them on the Mac in seconds [ran].
  - For TP-trained bases (V0/V4 InterPro, V0 ESM): the go_function grid keeps the `all`, seed-13 checkpoint of every arm (design §7.2: "No checkpoint is kept … except condition `all`, seed 13") [read]. A forward pass over valid gives validation scores without retraining. These checkpoints are on the remote.
- **Caveat 1:** DGZ early-stopped its released models on this validation split [inferred from DGZ training scripts; not re-read]. Validation is therefore mildly optimistic for every variant equally. That is acceptable for *selecting* a fusion, but not as a reported result.
- **Caveat 2:** this report's pilot has already looked at validation. Any S1 protocol must declare the pilot as exploration. It can freeze a small finalist set now; test must still be untouched (it is).
- **Validation Diamond:** needed for DS. Run `diamond blastp --more-sensitive` with valid queries against a **train-only** DB; DGZ's `train_data.dmnd` contains valid, so rebuild it from `train_data.fa`. Use the same flags as `run_diamond.sh`. It takes minutes on CPU [inferred]. The `.sim` identity file is a usable fallback, but it is not DS.
- **Metrics (consistent with go_function §7):**
  - Primary: class-centric AUC per term, TP labels, valid/test proteins with ≥ 1 label, terms with 0 < n_pos < n, as mean over terms. Also by n_train band {0, 1–10, 11–30, 31–100, 101–1000, > 1000}.
  - Secondary: class AUPR.
  - Protein-centric Fmax/Smin/AUPR via `go_function.evaluate.protein_centric`. This needs a [0, 1] score: per-term z-sums are not calibrated, so protein-centric finalists must come from a probability-space family (convex or σ(ℓ_t + w·z-prior)).
  - Ties in rank AUC: compute in float64. The pilot's float32 z gave tie artefacts of about 0.01 in one band, the same lesson as the PPI gate-2a float ties.
- **In the withholding grid:** apply the S1-frozen rule to each (t, k, seed) from `scores.npz`. Read Δ(fused − V0) vs k, and by S_sib and LD strata, with the existing two-way bootstrap (`go_function/stats.py`, `vendored.two_way_*`).
- **Weights cannot be tuned per withheld term:** validation labels of t are removed (target-free validation, protocol §5), which is D15's own rationale. So w and τ must come frozen from S1, which uses non-target terms on the full-data model.

## 4. Scopes

- **S1, standalone full-data fusion search** (the analogue of `results_go_ppi_fusion_best.md`):
  - Bases: released MLP and DGZ now (Mac); TP-trained V0-IPR, V4-IPR and V0-ESM `all` later (remote checkpoints).
  - Selection on validation; finalists frozen; test scored once.
  - Answers "what fusion gives the best result on DGZ's split", with DGZ, DGZ+DS and MLP+DS as published comparators.
  - Feasible and cheap.
- **S2, add-on to the go_function grid:**
  - Needs no new runs. `scores.npz` holds test logits for t, Pa(t), Sib(t), X(t) (DGZ-only ancestors incl. regulates fillers) and Desc(t) for every run (design §7.3, L363–370). The smoke file has 98 columns for 6 targets [ran].
  - DS per condition is already computed.
  - Every S1 family that uses Pa, X_reg or DS can therefore be scored per (arm, t, k, seed).
  - Families that need columns outside that set cannot: text neighbours, co-annotation, all ancestors.
  - **Protocol state:** go_function decisions are amendable until G8 (protocol L7–8, §10 G8 L243). Only G1–G6 are pinned. After G8, changes require a §14 amendment before any G10 output is read.
  - Because S2 is a *separate* script reading grid outputs, it does not alter any go_function decision, run or code-pinned file. Even so, to stay honest it must be preregistered (protocol file hashed) before G10 outputs are read.
  - **Recommendation:** add a one-row §14 amendment *before G8* that states: "descriptive fusion add-on defined in `protocol_go_function_fusion.md` (hash …); no go_function decision, arm, run or analysis changes". This is a documentation amendment, the same kind as A2.
  - Do **not** change arm F. It stays as preregistered, and the add-on reports parent-only and adaptive variants beside it.
- **Order:** S1 then S2. S1 fixes w and τ honestly; S2 inherits them frozen and asks whether fusion closes the scarce-term gap vs k and stratum, and whether parent-prior fusion of V0 matches V4's axioms at k = 0.

## 5. Controls (analogues of PPI)

| control | construction | status |
|---|---|---|
| shuffled GO graph | permute column identities within n_train deciles, then apply the real parent lists (`c_struct.py`, `perm`) | [ran] 0.735 BP / 0.846 MF, below base. Validation weight selection would choose w = 0 (as PPI derangement did) [inferred] |
| exact vs hierarchy | own column only vs parents vs all ancestors vs children; propagation | [ran] parents > ancestors > own > children |
| typed vs untyped | is_a vs TP vs any-relation parents vs TP + regulates | [ran on DGZ-label bases, confounded]; rerun on TP-trained V0 |
| homology-only vs ontology-only | own + H vs own + parents vs both | [ran] additive, small overlap |
| homology shuffled | H from permuted train rows | [ran] 0.770 < 0.7745 |
| no-ontology structure | co-annotation prior P̂(t\|c) from train labels | not run (new) |
| frequency prior | constant per term: class-centric no-op; protein-centric Naive | analytic |
| sibling term of F | parents − siblings (protocol F) vs parents | [ran] siblings hurt |

## 6. Compute and disk

**Mac now (CPU, no training, about 0.75–1.8 GB free disk):**
- forward passes of the released checkpoints on valid (and later test), each seconds;
- the full validation fusion search above, about 1.5 min per BP base;
- the G2-style reproduction from the released predictions already on disk;
- the S2 add-on dry-run on smoke `scores.npz`.

Do not write dense score matrices (BP valid is 475 MB float32). Compute on the fly and store per-term AUC tables only (≤ 1 MB). Do not download the 1.4 GB BP `*_blast.pkl`. DS can be recomputed from `test_diamond.res`.

**Remote (L40S):**
- validation Diamond (CPU; also feasible on the Mac if the user approves `brew install diamond` [inferred available; not checked]);
- TP-trained base models: forward over valid and test from the grid's kept `all` checkpoints. A standalone S1 can also train V0/V4 `all` itself: 2–30 min each on GPU per REMOTE_RUN L220;
- ESM-2 embeddings and MPNet term embeddings (G5);
- everything S2 (needs grid outputs).

## 7. Prior art and novelty

**Prior art:**
- **DeepGOPlus:** DeepGOCNN + DiamondScore, a weighted sum [web, DGZ paper §2.2.3].
- **DGZ + DiamondScore, MLP + DiamondScore** (DGZ Tables 3–4). DGZ+DS has the best class AUC: 0.906 MF, 0.803 BP [web].
- **TALE+:** transformer + hierarchical loss, plus a similarity combination [web, DGZ paper §2.5].
- **NetGO/GOLabeler:** learning-to-rank over frequency, BLAST-kNN, InterPro, network and text components [web, DGZ paper §2.3].
- **DeepGO-SE:** ESM-2 with axiom-based semantic entailment at inference [findings_go_function §6].
- **Hierarchical post-processing** (true-path reconciliation, Bayesian-network correction, hierarchical shrinkage of rare classes toward parents) is classic [inferred from memory; not re-verified].

**What would be new:**
1. **Scarcity-resolved evidence.** A weight-free or one-parameter parent-score prior on a *no-ontology* model matches or exceeds DGZ's train-time EL axioms on scarce and zero-train terms. Nobody has reported this for DGZ's split, by n_train band, with a shuffled-graph control.
2. **Typed vs untyped relation priors on leak-free TP-trained models.** This separates the regulates channel from DGZ's label-conflation leak.
3. **The same rule evaluated under controlled per-term withholding** (k-curves, S_sib/LD strata) against V4/DGZ at k = 0. This is S2, and it gives the clean "inference-time structure vs train-time axioms" comparison the go_function question lists as a mechanism.
4. **Honest selection** (validation-only, finalists frozen, test once, CIs over terms), which DGZ and DeepGOPlus do not report.

---

## Conclusions and recommendations

### (i) Recommended fusion definitions and scope

**Primary, AP-T** ("adaptive parent prior, true path"):

```
s(p,t) = z_t(ℓ_t(p)) + w_t · z_t( mean_{π ∈ Pa_TP(t)} ℓ_π(p) ) ,   w_t = c·τ/(τ + n_train(t))
```

- Pa_TP(t) = is_a+part_of parents in the same namespace that are vocabulary columns (root excluded); an empty set gives w_t·0.
- z_t is per term over the evaluated proteins (unsupervised and transductive, as in arm F).
- Two global parameters, (c, τ), are selected on validation from small grids: c ∈ {1, 2, 3} and τ ∈ {10, 30, 100}. n_train comes from the condition's own train labels.
- The fixed-w variant (τ = ∞, w = c) is reported beside it.

**Secondary, AP-T+H:** AP-T + w_H·z_t(DS), with DS on train-only labels and a validation Diamond run. w_H ∈ {0, 0.25, 0.5, 1}.

**Protein-centric finalist:** σ(ℓ_t + w_t·z-prior) in probability space, convex with DS. Selected on validation Fmax.

**Typed variant (pre-declared contrast, not a winner candidate):** AP-T + w_reg·z_t(mean ℓ over regulates-family targets), on TP-trained bases only.

**Scope:** S1 (full-data, released bases on the Mac, then TP-trained bases on the remote), then S2 (grid add-on with S1-frozen parameters).

### (ii) Mac now vs remote

**Mac now (CPU, read-only on inputs, outputs under a new results dir):**
1. Gate: re-derive G2 numbers from the released predictions and reproduce this report's validation numbers exactly. The scratch scripts become tested code.
2. Validation search on released MLP and DGZ for BP and MF: families AP-T, fixed-w, any-relation, regulates, ancestors, H-proxy, shuffled controls.
3. Freeze the finalists JSON (hashes).
4. S2 add-on dry-run on smoke `scores.npz`.
5. Optional, needs approval: `brew install diamond`, then a valid Diamond run against a train-only DB.

**Remote:**
1. Validation Diamond, if it was not done on the Mac.
2. Forward valid and test from the grid's kept `all` checkpoints (V0/V4 IPR, V0 ESM, and V3/V3c for the typed reads), or a dedicated V0-TP `all` training if S1 must run before the grid.
3. Test scoring once.
4. S2 on grid outputs after G10.

### (iii) Honest-selection protocol sketch

1. **Pre-registration (before any test read):** the families and grids above; selection metric = validation mean class AUC (primary) and validation Fmax (protein-centric finalist); tie → smaller c, larger τ, w_H = 0. This report's validation pilot is declared exploration.
2. **No per-term fitted weights and no learned stackers.** The PPI lesson: stackers overfit via shared units. Here every term shares the proteins, so per-term fitting on validation would overfit scarce terms. At most 3 global parameters.
3. **Selection per base model** on validation; finalists frozen to `finalists.json` with code and input hashes.
4. **Test scored once:** Δ = fused − base on the same model.
   - CIs: term bootstrap (B = 2000) within aspect; for multiple bases, a two-level bootstrap over bases × terms (`trials_domain.fusion_confirm` pattern, import only).
   - Report all-terms and per n_train band.
5. **Controls, scored with the frozen weights and re-selected on validation:**
   - shuffled-graph (n_train-decile permutation, fixed seed);
   - shuffled-hits DS;
   - co-annotation prior;
   - any-relation parents;
   - protocol F (siblings);
   - propagation-only.
6. **Comparators:** DGZ, DGZ + DS and MLP + DS at the published α. The α rule in DGZ's `evaluate*.py` was not retrieved (GitHub raw fetch failed), so either recover it or report the α-grid on validation.
7. **S2:** apply the frozen AP-T and AP-T+H to every V0 (and V4) run in the grid. Report Δ vs k ∈ {0, 1, 5, 10, 25}, by S_sib × LD stratum, with go_function's two-way bootstrap, all descriptive and unadjusted. Key read: AP-T(V0) − V4 at k = 0.

### (iv) Ordered implementation gap list (new files only)

Protected and untouched: `go_function/`, `go_ppi_fusion/`, `go_ppi_domain/` (existing files). Import only.

1. `.agents/tasks/ontology-withholding-feasibility/protocol_go_function_fusion.md`, plus its design review (`design_review_go_function_fusion.{json,md}`).
2. `config/go_function_fusion/{study.json,input_hashes.json}`: grids, seeds, bands, the `swissprot_exp.sim` hash and the valid Diamond hash once built.
3. `go_function_fusion/__init__.py`, `data.py`:
   - load released or our checkpoints via `go_function.model.DgzCompatModel` / `predict_probs`;
   - TP labels via `go_function.labels.load_labels`;
   - eval rows and terms.
4. `go_function_fusion/priors.py`:
   - neighbour tables (Pa_isa, Pa_TP, Pa_any, regulates targets, siblings, children, ancestors) via `go_function.ontology`;
   - DS via `go_function.baselines`;
   - co-annotation prior;
   - text-neighbour prior (later, after G5).
5. `go_function_fusion/fuse.py`: float64 per-term z, families AP-T/fixed/H/convex/σ-space, and the shuffled-graph and shuffled-hits controls.
6. `go_function_fusion/metrics.py`: rank AUC with ties, AUPR, bands, and a wrapper over `go_function.evaluate.protein_centric`.
7. `go_function_fusion/guard.py`: search log and test-read guard (pattern of `go_ppi_fusion/guard.py`).
8. `scripts/go_function_fusion/`:
   - `gate_repro.py`: G2 values plus the pilot numbers;
   - `diamond_valid.sh`: train-only DB and valid queries;
   - `search_valid.py`, `freeze_finalists.py`, `score_test.py`;
   - `s2_grid_fusion.py`: reads `results/go_function/runs/**/scores.npz`, uses `go_function.fusion` for F parity;
   - `report.py`.
9. `go_function_fusion/tests/test_*.py`: hand examples for z/fusion/AUC ties, a permutation control, DS parity with `go_function.baselines`, a guard test, and an S2 parity test on smoke `scores.npz` (F recomputed equals `fusion_F.csv`).
10. `.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_GO_FUNCTION_FUSION.md`.
11. Optional, pre-G8: the §14 cross-reference row in `protocol_go_function.md`. This is an edit to an existing task doc, not code; it needs user approval.

### (v) Open decisions for the user

1. **Scope order:** S1 → S2 (recommended), S1 only, or S2 only.
2. **Primary metric:** class-centric AUC (scarcity; where structure helps) or protein-centric Fmax (where homology helps; DGZ's headline).
3. **Base models for the confirmatory S1:** released DGZ checkpoints only (Mac, now, but trained on DGZ-rule labels, which confounds typed reads), or also TP-trained V0/V4/ESM from the remote (recommended for any typed-relation claim).
4. **Validation Diamond:** install `diamond` on the Mac via Homebrew (environment change), or run it on the remote.
5. **DS transfer source at test:** train only (consistent with go_function DS; recommended) or train + valid (DGZ-faithful, as a sensitivity read).
6. **go_function protocol:** add the §14 cross-reference row before G8 (recommended), or keep S2 entirely separate with its own pinned preregistration before G10 is read.
7. **Arm F:** leave as preregistered (recommended; the add-on reports parent-only beside it), or amend D15 before G8 to drop siblings. The amendment would be informed by this validation pilot, so it must be recorded as such.
8. **The validation pilot here counts as exploration.** Accept freezing finalists from it, or require a fresh validation half for selection. That would mean splitting valid by DGZ sequence group, which `valid_data.pkl` does not carry; groups would need `swissprot_exp.sim` clustering.

### (vi) Not verified

- DGZ's exact DS-combination weight α and its selection rule: `evaluate.py` could not be fetched. The DeepGOPlus-style α·DS + (1−α)·model form is [inferred].
- Whether DGZ early-stopped on `valid_data.pkl` [inferred].
- The homology prior here is a `.sim` identity proxy covering 50–64% of real Diamond pairs. True DS on validation was not computed.
- No test-split fusion number was computed (deliberately). All gains are validation-only, with released checkpoints trained on DGZ-rule labels. No bootstrap CIs were computed; per-term arrays are in `/tmp/gofunc_fusion/*_perterm.npz`.
- The typed-relation reads (regulates, any-relation) are confounded by DGZ-rule training labels.
- Text-neighbour, co-annotation, ESM and V4-vs-V0 fusions were not run.
- Whether `brew install diamond` is available, and the remote grid's run status (only Mac pins G1–G6 were checked; trainer changes in `go_function/train.py` are uncommitted on the current branch).
- Prior-art claims beyond the DGZ paper are from memory.
