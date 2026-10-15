# Preregistered protocol: inference-time GO-structure fusion for protein-function prediction (S1, with the S2 add-on)

Version 3 (DRAFT, revision after design review round 2, 2026-10-14; v2 2026-10-14 after round 1; v1 2026-10-13). Responses to the review findings are in §17 (round 1) and §18 (round 2). Spec: `findings_go_function_fusion.md` in this folder, cited "spec §n" or "spec (i)–(vi)". Parent study: `protocol_go_function.md` (v4, A1–A5, D19 approved), cited "GF §n". `$R` = `/Users/olukotunjosh/Downloads/CDCL`; every path below is repo-relative unless it starts with `/tmp`.

**Status of this document.** It is a preregistration only. Writing it computed no new result and read no test label: every number quoted here is copied from the spec (validation split only) or from already-pinned go_function gate files. Test data of DGZ's split have not been scored by any fusion rule.

Freezing works in two steps, as in GF:

1. Hypotheses, contrasts, grids, seeds, metrics, tie rules and thresholds are frozen when this protocol is approved. Decisions U1–U7 (§2) are the user's and are not amendable by the implementer.
2. Hashes "recorded at gate" are written by the scripts to `results/go_function_fusion/protocol_pins.json` and mirrored by hand into §13. Recording a hash is not a protocol change. After G-F (finalist freeze) nothing here changes except by a §14 amendment made before the single test scoring (G-T); any change after G-F that would affect a frozen parameter, contrast or family stops the study for a user decision.

## 1. Question and hypotheses

Can a base model's own scores on GO-related terms, combined at inference time through the true-path graph, improve prediction of a term t, and is the gain concentrated where labels are scarce? The fusion borrows the ranking of t's better-trained parents. It needs no retraining and has at most 3 global parameters.

Notation. ℓ_c(p) is a base model's logit for vocabulary column c and protein p. AUC_X(t) is the class-centric AUC of term t for score X on the evaluated proteins (§7). Δ_{X−Y}(t) = AUC_X(t) − AUC_Y(t), paired within term. Δ_{X−Y}[b] is its mean over the terms of n_train band b; Δ_{X−Y}[all] over all evaluated terms.

| id | hypothesis | contrast | predicted sign |
|---|---|---|---|
| H1 (primary) | AP-T improves class-centric AUC over its own base | **P_b** = Δ_{AP-T(b) − b}[all], one per base b (§6.1 families) | > 0 |
| H2 (scarcity) | the AP-T gain is larger for scarce and zero-train terms | **Sc0_b** = Δ_{AP-T(b)−b}[0] − Δ_{AP-T(b)−b}[>100]; **Sc1_b** = Δ_{AP-T(b)−b}[1–10] − Δ_{AP-T(b)−b}[>100] | > 0 |
| H3 (homology, secondary) | a DiamondScore channel adds to the parent-prior fusion | **Hc_b** = Δ_{AP-T+H(b) − AP-T(b)}[all] (class AUC); **Hf_b** = Fmax(PC(b; c_PC*, τ_PC*, α*)) − Fmax(PC(b; c_PC*, τ_PC*, α = 0)) (protein-centric, homology given the frozen prior, §3.3; (c_PC*, τ_PC*, α*) are PC's own Fmax-selected parameters, not AP-T's (c*, τ*)) | > 0 (class gain expected small, around the 0.002 floor; Fmax gain expected ≥ 0.01) |
| H4 (typed contrast) | on true-path-trained bases, adding regulates-family targets to the parent prior helps the terms that have such edges | **Ty_b** = Δ_{AP-T+R(b) − AP-T(b)}[reg-subset], TP-trained bases only | > 0 |

[>100] pools bands 101–1000 and >1000. [reg-subset] = evaluated terms with a non-empty Reg(t) (§3.4). Bands are on TP train counts (§7).

H4 is a pre-declared contrast, **not a winner candidate**: AP-T+R is never a finalist for H1, H3 or S2, whatever its validation score.

## 2. User decisions (recorded verbatim in substance; not amendable by the implementer)

| id | decision |
|---|---|
| U1 | **Scope.** S1 now. S2 later, as a separate preregistered descriptive add-on section (§11) that uses S1-frozen parameters, applied to go_function grid outputs after G10. Implement only an S2 dry-run on the existing smoke `scores.npz` now. |
| U2 | **Primary metric** = class-centric mean AUC per term (term bootstrap), reported by n_train band {0, 1–10, 11–30, 31–100, 101–1000, >1000}. The protein-centric finalist is selected on validation Fmax and reported with Fmax, Smin and AUPR. |
| U3 | **Base models.** DGZ released MLP and DGZ checkpoints, BP and MF, on the Mac now (Tier A). TP-trained V0/V4 InterPro and V0 ESM come from the remote, from the go_function grid's kept `all` checkpoints or a dedicated V0-TP `all` training (Tier B, §5.2). Typed-relation claims rest **only** on true-path-trained bases. Released-checkpoint reads of typed priors are labelled "confounded (DGZ-rule labels)". |
| U4 | **Validation Diamond** runs on the **remote** against a train-only DB. Nothing is installed on the Mac. The Mac H prior is the `.sim` identity proxy, labelled exploratory-only. Confirmatory H uses true Diamond on the remote. |
| U5 | **Cross-reference.** Add a one-row §14 cross-reference to `protocol_go_function.md` pointing to the S2 preregistration (§11), before G8. It is the ONLY edit allowed to an existing go_function file. It changes no go_function decision, is stated as such, and is dated (text in §11.6). |
| U6 | **Arm F** stays as preregistered in go_function (GF D15). S1 and S2 report the parent-only prior beside it. |
| U7 | **Finalists** are frozen from the same validation set as the pilot, with explicit disclosure that the pilot (spec §2) was exploratory and used that validation set (§10). |

Implementer choices made in this protocol (reviewable at design review, frozen at approval): DGZ's α = 0.5 recovered (§6.3); S2 parameters from the Tier A MLP finalist (§11.2), Tier B parameters in S2 only under the mechanical snapshot rule of §11.2; single test scoring after all finalists are frozen (§9 G-T); confirmatory DS from a train-only Diamond run for both valid and test (§4.3); alt-id columns excluded from the confirmatory term set (§5) and from every confirmatory neighbour set (§3); the co-annotation control is the top-m train-Jaccard column set in AP-T form (§3.5), replacing spec §1(e)'s Σ_c P̂(t|c)·σ(ℓ_c) so that it matches AP-T's form and column count, with t's true-path ancestors and descendants excluded from the confirmatory candidate set (CoA) and the unrestricted set kept as descriptive CoA⁺; Tier B source (b) trains through `go_function.train.run(..., out_dir=…)` into a fusion-owned directory (§4.2); V3/V3c typed reads of spec (ii) remote item 2 are not part of S1 (§4.2); seeds in §8.

## 3. Fusion definitions (frozen)

All scores are computed in float64. z_t(v) = (v − mean(v)) / sd(v) over the evaluated proteins of the split being scored (validation for selection, test at G-T), population sd; a constant vector gives the zero vector. z is unsupervised and per term (the class-centric analogue of PPI `within_query_z`). Empty neighbour sets contribute 0. Notation for neighbour sets (built once from the pinned `go.obo` with `go_function.ontology` and `go_function.labels.build_vocab`; "columns of a GO id" = `vocab.primary_to_cols[id]`, root column removed). Two modes:

- **Confirmatory mode** (every selection, freeze, test scoring and S2): from every neighbour set below, also remove `vocab.duplicate_cols` (alt-id duplicate columns) and every column of t's own primary id (`vocab.primary_to_cols[vocab.primary[t]]`). This matches `go_function.targets.channel_sets` (primary, non-duplicate columns, t's columns subtracted), so S1-frozen parameters are applied in S2 to the same set definition they were selected on. The number of evaluated terms whose Pa_TP, Reg, Sib or Desc changed relative to pilot-compat mode is reported at G-R(c).
- **Pilot-compat mode** (G-R(b) only): root removed, nothing else, exactly as the pilot `c_struct.py`.

Sets:

- **Pa_TP(t):** columns of the is_a and part_of parents of t's primary id in the same namespace.
- **Pa_any(t):** columns of parents by any `relationship:` or is_a edge, same namespace.
- **Reg(t):** columns of the targets X of t's `regulates`, `positively_regulates`, `negatively_regulates` edges, same namespace (t = "regulation of X" gets X).
- **Sib(t):** columns of is_a siblings, `go_function.ontology.isa_siblings(ont, t, anc, desc)` (terms sharing an is_a parent with t, minus t, anc_TP(t) and desc_TP(t)), restricted to ≥ 10 TP train positives (GF D15, `go_function.fusion.MIN_SIB_POS`), so protocol F matches GF arm F.
- **anc_TP(t), desc_TP(t):** `go_function.ontology.ancestors(ont)` (TRUE_PATH = is_a, part_of; same namespace; reflexive) and its inverse `descendants(anc)`; "strict" removes t itself.
- **Desc(t):** columns of strict TP descendants of t.
- **CoA(t), CoA⁺(t):** see §3.5.

A column whose primary is None (not in the pinned ontology) has every set empty.

n_train(t) = number of TP train positives of column t (`preprocess/go_function/<aspect>/labels_TP.npz`, train split), for every base, including Tier A bases that were trained on DGZ-rule labels. Pa̅_X(t)(p) = mean over c ∈ X(t) of ℓ_c(p).

### 3.1 Primary: AP-T (adaptive parent prior, true path)

```
AP-T(p,t) = z_t(ℓ_t(p)) + w_t · z_t(Pa̅_TP(t)(p)),     w_t = c·τ/(τ + n_train(t))
```

Grid: c ∈ {1, 2, 3}, τ ∈ {10, 30, 100} (9 cells). Selection: the cell with the largest validation mean class AUC over all evaluated terms (§7), per (aspect, base). Ties (equal to 1e-12) go to the smaller c, then the larger τ. Two global parameters.

**Fixed-w variant (reported beside AP-T, descriptive):** τ = ∞, w_t = c, c ∈ {1, 2, 3}, selected by the same rule (tie → smaller c).

**Parent-only, weight-free (U6, descriptive):** z_t(ℓ_t) + z_t(Pa̅_TP(t)), i.e. arm F without the sibling term.

### 3.2 Secondary: AP-T+H

```
AP-T+H(p,t) = AP-T(p,t) + w_H · z_t(DS(p,t))
```

(c, τ) frozen from §3.1 at G-F. w_H ∈ {0, 0.25, 0.5, 1}, selected on validation mean class AUC; ties → smaller w_H. DS is the bitscore-weighted label transfer of `go_function.baselines.diamond_score` with TP **train-only** labels and hits restricted to train proteins (§4.3). One additional global parameter (3 in total).

### 3.3 Protein-centric finalist PC

```
q(p,t)  = σ( ℓ_t(p) + w_t · z_t(Pa̅_TP(t)(p)) )            (probability space)
PC(p,t) = (1 − α) · prop_TP(q)(p,t) + α · DS(p,t)
```

prop_TP = true-path max-propagation (`go_function.evaluate.tp_ancestor_pairs`, `propagate_max`), as GF D14 does for every protein-centric metric. DS is TP-consistent by construction (a weighted mean of TP-closed rows), so it is not propagated. Grid: c ∈ {0, 1, 2, 3} (c = 0 removes the prior), τ ∈ {10, 30, 100}, α ∈ {0, 0.1, 0.2, …, 0.9} (4 × 3 × 10 = 120 cells; c = 0 cells differ only in α). Selection: largest validation Fmax (§7), per (aspect, base); ties → smaller c, then larger τ, then smaller α. Three global parameters, written (c_PC*, τ_PC*, α*). They are selected jointly on Fmax and are distinct from AP-T's (c*, τ*) of §3.1, which are selected on class AUC; neither set is copied into the other. The reference b_TP = prop_TP(σ(ℓ)) is the base's protein-centric score (= PC with c = 0, α = 0).

PC contrasts (all with the protein bootstrap of §8):

- **Hf_b (confirmatory, F-H)** = Fmax(PC(b; c_PC*, τ_PC*, α*)) − Fmax(PC(b; c_PC*, τ_PC*, α = 0)): the homology channel given the frozen prior. It isolates DS because both terms share the same prior.
- **PCp_b (descriptive)** = Fmax(PC(b; c_PC*, τ_PC*, 0)) − Fmax(b_TP): the prior-only Fmax effect.
- **PCt_b (descriptive)** = Fmax(PC(b; c_PC*, τ_PC*, α*)) − Fmax(b_TP): the total effect.
- **XDS_b (descriptive)** = Fmax(X+DS) − Fmax(b_TP) (§6.3): DGZ-style homology alone on the base.

Smin and micro-AUPR are reported for the same four contrasts, descriptive.

### 3.4 Typed contrast AP-T+R (H4; not a winner candidate)

```
AP-T+R(p,t) = AP-T(p,t) + w_R · z_t(Pa̅_Reg(t)(p))
```

(c, τ) frozen from §3.1 for that base. w_R ∈ {0, 0.25, 0.5, 1}, selected on validation mean class AUC **over the reg-subset**; ties → smaller w_R. Confirmatory on Tier B bases only. On Tier A it is computed and labelled "confounded (DGZ-rule labels)": under DGZ's rule, positives of "regulation of X" are propagated into X, so ℓ_X already encodes t's positives (spec §1(c); GF findings §4).

### 3.5 Controls (each in the form of the fusion it controls)

Every control is scored twice: (i) with the frozen finalist parameters of the fusion it controls, and (ii) re-selected on validation over the same grid **plus the null cell** (c = 0, or w_H = 0, w_R = 0), same tie rule (null cell first). Both scorings are frozen at G-F (Tier A) or G-F2 (Tier B, H, PC) and scored once at G-T.

| control | of | construction | expected direction |
|---|---|---|---|
| shuffled graph | AP-T | Pa_TP replaced by perm[Pa_TP(t)]: column identities permuted within n_train deciles (deciles of n_train over all vocabulary columns, `np.searchsorted(np.quantile(n_train, linspace(0,1,11)[1:-1]), n_train)`), `numpy.random.default_rng(20261013)`, one `rng.permutation` per decile in increasing decile order (the pilot's construction, kept for parity in both modes; perm[π] may be the root, an alt-id column or t itself, and the number of (t, π) pairs with perm[π] ∈ {root, t} and with perm[π] an alt-id column is reported) | frozen: Δ vs base < 0 (pilot: BP MLP 0.7349 vs 0.7745); re-selected: c = 0 chosen, Δ = 0 |
| shuffled graph (PC), descriptive | PC | PC with perm[Pa_TP(t)] (the same permutation) inside σ(·), frozen (c_PC*, τ_PC*, α*); also re-selected over the PC grid | frozen: Fmax ≤ PC(c_PC*, τ_PC*, α*); re-selected: c = 0 |
| shuffled hits | AP-T+H | DS computed with the train label matrix rows permuted by `default_rng(20261017).permutation(n_train_proteins)` | frozen: Hc ≤ 0; re-selected: w_H = 0 |
| shuffled hits (PC) | PC | the same permuted DS inside PC | frozen: Fmax ≤ PC; re-selected: α = 0 |
| co-annotation (confirmatory control) | AP-T | Pa_TP replaced by CoA(t) = the m(t) = max(1, \|Pa_TP(t)\|) columns c with the largest train Jaccard J(t,c) = \|P(t) ∩ P(c)\| / \|P(t) ∪ P(c)\| (P = TP train positive sets), over candidates c that are primary, non-duplicate, same-namespace columns with c ∉ cols(t) ∪ {root} ∪ cols(anc_TP(t)) ∪ cols(desc_TP(t)) and J(t,c) > 0; ties by column index; fewer than m(t) candidates with J > 0 → take those (possibly ∅); CoA(t) = ∅ if n_train(t) = 0. Excluding t's TP relatives matters because TP closure makes J(t,π) = n_t/n_π for an ancestor π and J(t,d) = n_d/n_t for a descendant d, so an unrestricted top-Jaccard set largely re-derives the GO graph through label closure | positive, smaller than AP-T on bands 0 and 1–10 (no prior at n_train = 0) |
| co-annotation⁺, descriptive | AP-T | CoA⁺(t): the same construction without the anc_TP/desc_TP exclusion (cols(t), root and duplicates still excluded). Reported with the overlap share \|CoA⁺(t) ∩ cols(anc_TP(t) ∪ desc_TP(t))\| / m(t), mean per band | ≥ CoA; close to AP-T where the overlap share is high |
| any-relation parents | AP-T | Pa_TP replaced by Pa_any | Tier A: ≈ AP-T (pilot BP MLP 0.7954 vs 0.7936 at w = 1), labelled confounded; Tier B: ≤ AP-T (descriptive) |
| protocol F | AP-T | z_t(ℓ_t) + w_t · [z_t(Pa̅_TP) − z_t(Sib̅)]; also the weight-free GF arm F (w_t ≡ 1) | below AP-T (pilot BP MLP F 0.7767) |
| propagation-only | AP-T | Pa_TP replaced by the max (not mean) over Desc(t) logits; also pure prop_TP(σ(ℓ)) with no weight | Δ vs base ≈ 0, **extrapolated**: the pilot measured pure propagation (±0.001) and a direct-children max at w = 1 (−0.003), not the Desc-max with adaptive w_t used here |
| shuffled Reg | AP-T+R (H4) | Reg(t) replaced by perm[Reg(t)] (the shuffled-graph permutation above), frozen w_R*; also re-selected over the w_R grid | frozen: AP-T+Rshuf − AP-T ≤ 0 on the reg-subset (a positive Ty that the shuffled Reg matches is "any extra column block", not typed structure); re-selected: w_R = 0 |
| no-ontology baseline | — | base alone (c = 0) | reference |

Controls never become finalists. A control whose validation re-selection picks a non-null cell is reported as such.

**Which scoring a reading uses.** Every §12 reading that involves a control (structural, typed structure, co-annotation) uses the **frozen-parameter** scoring (i), because only it asks whether the real edges matter at the selected weight (the PPI derangement logic, spec §5). The re-selected scoring (ii) is expected to pick the null cell, which would make "fusion − control" collapse to the fusion's own contrast; it is therefore **descriptive only and never enters a §12 reading**. The two control contrasts that gate wording are:

```
D_struct_b = Δ_{AP-T(b; c*, τ*) − AP-Tshuf(b; c*, τ*)}[all]          frozen (c*, τ*) of base b, draw set U (§8)
D_typed_b  = Δ_{AP-T+R(b; w_R*) − AP-T+Rshuf(b; w_R*)}[reg-subset]   frozen (c*, τ*, w_R*) of base b, draw set R (§8)
```

Both are descriptive (not Holm-adjusted) and are read by ci_low > 0 (§12). If w_R* = 0, AP-T+R = AP-T+Rshuf = AP-T, D_typed_b ≡ 0 and it is labelled "null selected on validation".

### 3.6 Excluded by design

No per-term fitted weight, no learned stacker, no fusion parameter fitted on test, at most 3 global parameters per fusion (the PPI stacker-overfit lesson, `results_go_ppi_fusion_best.md`). Text-neighbour (MPNet), ESM+InterPro model fusion and V4+V0 axiom fusion are out of scope for S1 (spec §1 (d), (f), (g)).

## 4. Data

### 4.1 Inputs (reused, hash-verified against `config/go_function/input_hashes.json`)

DGZ release under `dataset/go_function/dgz/`: `{bp,mf}/{train,valid,test}_data.pkl`, `terms.pkl`, `interpros.pkl`, `test_diamond.res`, `mlp.th`, `deepgozero.th`, `predictions_{mlp,deepgozero}.pkl`; `go.obo`, `go.norm`. TP labels `preprocess/go_function/{bp,mf}/labels_TP.npz` (pinned at GF G1: `bp/labels_TP.json` 63e1bdeb…, `mf/labels_TP.json` 3eba3e96…). Split: DGZ's 50%-identity split, unchanged (BP 47,733 / 5,552 / 5,444; MF 34,716 / 3,851 / 4,712). Added to `config/go_function_fusion/input_hashes.json` at G-R: `swissprot_exp.sim` (a143b389…c3101, Mac exploratory only) and, at G-D, the remote Diamond outputs.

### 4.2 Base models

- **Tier A (Mac):** DGZ released `mlp.th` and `deepgozero.th` per aspect, loaded strictly with `go_function.model.DgzCompatModel.load` (GF G3 verified), InterPro features via `go_function.features.interpro_csr`; logits from `predict_probs(..., sigmoid=False)`. Valid proteins without InterPro get the zero vector (as in the pilot and DGZ).
- **Tier B (remote):** TP-trained `all` models, seed 13: V0-InterPro, V4-InterPro, V0-ESM (GF §6.2: InterPro replicate block and main block; checkpoints kept by `go_function.train.keep_checkpoint` for `mode == study`, `pack is None`, `seed == 13`). Source rule at G-TP: (a) the grid's kept checkpoints if GF G10 has written them; otherwise (b) the same materialized GF run specs (from `results/go_function/specs/`, read only, byte-identical, sha256 recorded) executed out of order into `results/go_function_fusion/tp_bases/<run_name>/` (needs GF G7 and G8; no new spec, no new hyperparameter). `scripts/go_function/run.py` cannot do this: `go_function.train.validate_spec` requires `out_dir` under `results/go_function/` (verified in `go_function/train.py`, `RESULTS_PREFIX`), and writing there would touch a protected path and pre-empt a GF grid cell. Source (b) therefore uses a fusion-owned wrapper `scripts/go_function_fusion/train_tp_base.py` that calls the imported, unmodified `go_function.train.run(spec_raw, device, log=…, out_dir=<fusion dir>)`. That keyword is documented in `train.run` as "tests only". `train.run` calls `go_function.pins.assert_writable_target(out_dir)`, whose `PROTECTED` tuple is `contrastive_learning`, `go_ppi_domain`, `scripts/go_withholding`, `config/go_withholding`, `trials_domain`, `scripts/trials_fusion_confirm`, `scripts/run_learning_curve_point.py` (verified in `go_function/pins.py`), so `results/go_function_fusion/tp_bases/` passes. `canonical_json` refuses absolute paths, so every path in the provenance file is repo-relative. Note that `train.run` also writes the base's own test `eval.json` and `scores.npz` (GF evaluation of the base, no fusion); the wrapper does not open them, and §10.4 covers this exposure. The run.json it writes keeps the spec's repo-relative paths, so the wrapper also writes `tp_base_provenance.json` (actual directory, spec sha256, code hashes of `go_function/`) and G-TP verifies it. Using this override is disclosed in §16. If neither source is possible, the study pauses for a user decision (Tier B is then dropped or deferred).
- **Not in S1:** the V3/V3c typed reads listed in spec (ii) remote item 2. H4 contrasts AP-T+R with AP-T on V0/V4/ESM bases; V3/V3c are train-time typed arms whose reads belong to GF's own contrasts (C1) and, descriptively, to S2 (§11.3 applies AP-T_S2 to every stored GF run, V3/V3c included).

### 4.3 DiamondScore

- **Confirmatory (remote, G-D):** DIAMOND (exact version and binary sha256 pinned at G-D), `makedb` on the **train** sequences only (from `train_data.pkl`, written with DGZ's `diamond_data.py` format), `blastp --more-sensitive --outfmt 6 qseqid sseqid bitscore pident`, the flags of DGZ's `run_diamond.sh`. Queries: valid and test sequences, in one run each. Checks (fatal): every hit id is a train protein; no query id is a train id; output hashes pinned. DS for valid and test both come from this run, so selection and test use the same transfer source.
- **Sensitivity (descriptive):** test DS from DGZ's released `test_diamond.res`, non-train hits dropped (the GF DS baseline).
- **Mac exploratory only:** H_sim = identity-weighted kNN from `swissprot_exp.sim` (the pilot's H). It covers 50–64% of real Diamond pairs and is not DS. No H_sim number enters a finalist, a confirmatory contrast or S2.

## 5. Evaluation sets

- **Proteins:** the split's proteins with ≥ 1 TP label in the aspect (validation: BP 5,389, MF 3,755, pilot counts; test: determined at G-T).
- **Terms (confirmatory):** primary vocabulary columns (`vocab.primary[c]` not None and c not an alt-id column of another column), root excluded, with 0 < n_pos < n_proteins on that split's evaluated proteins. The pilot used every vocabulary column; G-R reproduces it in pilot-compat mode (§9), and the count of alt-id columns removed is reported.
- **Protein-centric:** `go_function.evaluate.protein_centric` (Fmax, Smin, micro-AUPR; DGZ's formulas), TP labels, all primary columns except root, IC from TP train labels (`ic_from_labels`).
- The test evaluated-term set can differ from the validation set; band membership uses TP n_train only, so it is label-free on test.

## 6. Contrasts, families and comparators

### 6.1 Holm families (α = 0.05; per aspect; BP and MF are separate families)

| family | contrasts | tier |
|---|---|---|
| F-A1 (primary) | P_MLP, P_DGZ | A |
| F-A2 (scarcity) | Sc0_MLP, Sc1_MLP, Sc0_DGZ, Sc1_DGZ | A |
| F-B1 (primary, TP bases) | P_V0IPR, P_V4IPR, P_V0ESM | B |
| F-B2 (scarcity, TP bases) | Sc0, Sc1 for each Tier B base (6) | B |
| F-H (homology) | Hc_b and Hf_b (§3.3: PC(α*) − PC(α = 0) at frozen c_PC*, τ_PC*) for b ∈ {MLP, DGZ} (4) | A, with remote DS |
| F-T (typed, BP only) | Ty_V0IPR, Ty_V4IPR, Ty_V0ESM | B |

MF typed reads (82 reg-subset terms) are descriptive. Every other number (fixed-w, parent-only, controls, comparators, per-band Δ, AP-T(MLP) − DGZ, Tier A typed reads, pooled two-level estimates, DS sensitivity) is descriptive and unadjusted.

### 6.2 Key descriptive reads

- AP-T(MLP) − DGZ and AP-T(MLP) − AP-T(DGZ), by band (spec Summary 3: on validation, MLP + AP-T matched DGZ on scarce terms). "DGZ" here is the raw logit ℓ of the released `deepgozero.th` (the same score as P_DGZ's base); the value with DGZ's released propagated predictions (recomputed from the checkpoint with DGZ's propagation, not read from `predictions_deepgozero.pkl`) is reported beside it as a sensitivity.
- PCp_b, PCt_b and XDS_b (§3.3), with protein-bootstrap CIs.
- D_struct_b (§3.5: AP-T − shuffled graph, both at the frozen (c*, τ*) of base b, draw set U), per base: the "structural" wording of §12 requires ci_low(D_struct_b) > 0. D_typed_b (frozen w_R*, draw set R) likewise gates "typed structure". The re-selected control scorings never enter these reads.
- Controls vs base and vs AP-T, frozen and re-selected (re-selected: descriptive only).

### 6.3 Comparators

DGZ's DS-combination rule was recovered from the DGZ repo: `evaluate.py` at commit `4507529ebb182749bdcc1bbfc2677d2357912af8` (file sha256 `0a89c4c0…af848`, copied to `/tmp/gofunc_fusion/dgz_evaluate.py`; G-R copies it into `config/go_function_fusion/dgz_evaluate_ref.py.txt`) sets a fixed `alpha = 0.5` and scores `preds = blast_preds·α + preds·(1 − α)`, with `preds` DGZ-propagated and `blast_preds` from `predict_diamond.py` (train + valid labels as transfer source, DGZ-rule ancestor propagation, `test_diamond.res`). It is not selected on validation.

- **In our evaluator (TP labels, §5 sets):** DGZ, MLP (bases), and X+DS = 0.5·prop_TP(σ(ℓ_X)) + 0.5·DS for X ∈ {MLP, DGZ}, DS train-only (§4.3). Sensitivity: α re-selected on validation Fmax over {0, 0.1, …, 0.9} (tie → 0.5, then smaller).
- **DGZ-faithful check (G-T, descriptive, does not gate):** blast_preds recomputed from `test_diamond.res` with DGZ's rule (no 1.4 GB `*_blast.pkl` download), α = 0.5, DGZ-faithful evaluator mode (GF design §8.3), compared with the published MLP+DS and DGZ+DS cells of DGZ Tables 3–4. The cells are transcribed into `config/go_function_fusion/dgz_published_ds.json` with cell references at G-R, before G-F. A miss beyond ±0.005 is reported, not fatal.

## 7. Metrics

- **Primary:** class-centric AUC per term, rank-based with average ranks for ties (`scipy.stats.rankdata`), float64, reported as the mean over evaluated terms and per n_train band {0, 1–10, 11–30, 31–100, 101–1000, > 1000}. Band 0 = TP n_train = 0; the subset with DGZ-rule n_train = 0 too ("natural zero-shot under both rules") is reported beside it.
- **Secondary:** class-centric AUPR (average precision) per term, same summaries.
- **Protein-centric:** Fmax, Smin, micro-AUPR (§5); reported for every base, PC, X+DS and the controls of PC.
- **Selection metrics:** validation mean class AUC over all evaluated terms (AP-T, fixed-w, AP-T+H, controls), over the reg-subset (AP-T+R), validation Fmax (PC). Bands never drive selection.
- Also reported: share of terms improved; n terms per band; n terms with empty Pa_TP, Reg, CoA.

## 8. Uncertainty

- **Term bootstrap (all class-centric contrasts):** B = 2,000, within aspect, one generator `numpy.random.default_rng(20261014)` per aspect, drawn in this fixed order: (1) draw set U, B unstratified resamples of all evaluated terms (`rng.integers(0, n_terms, size=(B, n_terms))`), used by P, Hc, every [all]-level control and descriptive read; (2) draw set S, B stratified resamples, one block per band in band order {0, 1–10, 11–30, 31–100, 101–1000, >1000} (`rng.integers(0, n_b, size=(B, n_b))` per band), used by Sc0, Sc1 and every per-band read, with Δ[>100] in a replicate = the pooled mean over the resampled 101–1000 and >1000 terms (weights = band sizes); (3) draw set R, B unstratified resamples of the reg-subset terms, used by Ty and the shuffled-Reg control. Within a draw set the same indices serve every contrast (common random numbers). A band with 0 terms yields no draw block, and a contrast that needs it is reported as not computed. Output: point estimate (full-data mean), 95% percentile CI, two-sided p = min(1, 2·min((1 + #{T ≤ 0})/(B + 1), (1 + #{T ≥ 0})/(B + 1))) (`trials_domain.fusion_confirm.boot_p`, imported).
- **Protein bootstrap (Fmax contrasts Hf, PC reads):** B = 2,000, `default_rng(20261015)` per aspect, resampling evaluated proteins (`rng.integers(0, n_prot, size=(B, n_prot))`, drawn first). Per protein and for each of the 101 thresholds the script stores precision, recall, ru, mi and a has-prediction indicator (≥ 1 score ≥ threshold; DGZ's Fmax averages precision only over these proteins) once; Fmax and Smin are re-maximised over thresholds in each replicate from the resampled rows. The same draws serve every Fmax contrast of the aspect.
- **Protein-bootstrap CI for P_b (descriptive):** the same `default_rng(20261015)` stream, drawn second (`rng.integers(0, n_prot, size=(B_P, n_prot))`, B_P = 200 because each replicate recomputes every term's AUC on the resampled proteins; B_P is pre-set to bound cost). Scores are those of the full evaluated set (z_t is not recomputed per replicate); duplicated proteins enter the AUC with their multiplicity. Terms whose AUC becomes undefined in a replicate are dropped from that replicate's mean. Reported beside the term-bootstrap CI of P_b; it never enters Holm or the §12 classification.
- **Two-level bootstrap over bases × terms (descriptive pooled estimates; Tier A {MLP, DGZ}, Tier B {V0IPR, V4IPR, V0ESM}):** `trials_domain.fusion_confirm.draw_indices` (models = bases, patients = terms), `boot_stats`, `summarize`, imported, B = 2,000, `default_rng(20261016)`. With 2–3 bases the base level under-covers; this is stated beside every pooled number.
- **Multiplicity:** Holm within each family of §6.1 (implemented here; `trials_domain.fusion_confirm.holm` is two-contrast only and is not reused). Procedure: let m be the family size of §6.1 (fixed; a null-selected contrast keeps its slot with p = 1, a missing base's contrasts are dropped and m shrinks accordingly). Sort the family's raw p ascending, ties broken by the contrast's order in the §6.1 table; with p(1) ≤ … ≤ p(m), p_adj(i) = min(1, max_{j ≤ i} (m − j + 1)·p(j)). Each p_adj is mapped back to its contrast; rule 2 of §12 uses p_adj.
- **Missing data:** a term whose AUC is undefined on a split is not evaluated (§5). A base that fails to load or verify is missing; its contrasts are reported as not computed and its family is Holm-adjusted over the remaining contrasts. No imputation, no outlier exclusion.

## 9. Gates (in order; each must pass before the next)

- **G-0 (Mac).** Unit tests in `go_function_fusion/tests` pass: hand examples for z, AP-T, w_t, empty sets, tie-handled AUC, fixed-w, PC, every control construction including shuffled graph (PC) and shuffled Reg (permutation reproducible from seed), confirmatory vs pilot-compat neighbour sets (duplicate and own columns removed only in confirmatory mode), CoA excluding every anc_TP/desc_TP column while CoA⁺ may contain them (a hand ontology where the top-Jaccard column is a parent), Sib equal to `isa_siblings` (desc(t) removed), D_struct and D_typed computed from the frozen-parameter scorings (and D_typed ≡ 0 with the null label when w_R* = 0), Holm on a hand family with tied p and a p = 1 slot, Hf = 0 exactly when α* = 0 and the "null selected on validation" label, Holm with a p = 1 slot, the three bootstraps with the fixed draw order of §8 (U, S, R; protein Fmax then P_b), the has-prediction indicator, the §11.2 snapshot rule (eligible / ineligible fixtures), guard refusal; DS parity with `go_function.baselines`; F parity with `go_function.fusion`. No protected path changed (§15).
- **G-R (Mac; reproduction).**
  - (a) GF G2 numbers: recompute from the released `predictions_{mlp,deepgozero}.pkl` with `go_function.evaluate` in DGZ-faithful mode; every value equals `results/go_function/gates/gate_dgz_eval.json` to 1e-9. This reads test labels **only** for the released models' already-pinned published metrics, in a separate process that never enters search mode and computes no fusion; it adds no test exposure beyond GF G2.
  - (b) Pilot numbers: the new code, in pilot-compat mode (all vocabulary columns; the pilot's float32 `zcol` for `b_valid_pilot.py` variants; H_sim), reproduces every value of the seven pilot JSONs pinned in §13 (`n_rows`, `n_terms`, `n_terms_with_reg_edge`, `band_counts`, `valid_with_sim_hit` exactly; each mean, band mean and fmax to |diff| ≤ 5e-5 and identical when rounded to 4 decimals). BP DGZ has only a `struct_*` file; no `pilot_bp_deepgozero.json` exists and none is required. The pilot snapshot (the 7 JSONs, 7 per-term npz, both scripts) is copied to `results/go_function_fusion/pilot_snapshot/` and hash-checked against §13; if `/tmp/gofunc_fusion` is gone, G-R stops and asks.
  - (c) Confirmatory mode on validation (float64, primary columns) recomputes base and pilot-equivalent variants; differences from (b) are reported, not gated.
  - A miss in (a) or (b) → stop and ask.
- **G-V (Mac; validation search).** `search_valid.py` runs in search mode (§10.3) on Tier A × {BP, MF}: every grid of §3.1–3.5 that needs no DS, plus exploratory H_sim reads. Output: per-term AUC tables (≤ 1 MB each, no dense score matrices), `search_log.jsonl`, selections.
- **G-F (Mac; freeze).** `freeze_finalists.py` writes `finalists_tierA.json`: per (aspect, base) the AP-T (c*, τ*), fixed-w c*, control re-selections, the Tier A typed read's w_R (confounded), the S2 parameters (§11.2), code hashes of `go_function_fusion/` and `scripts/go_function_fusion/`, input hashes. Pinned. From here every confirmatory code file is frozen; G-T verifies the hashes.
- **G-S2dry (Mac).** S2 dry-run on smoke `scores.npz` (§11.5). Not a result.
- **Remote G-D (Diamond).** §4.3. Pins.
- **Remote G-TP (TP bases).** §4.2 source rule; checkpoint sha256, strict load, validation forward deterministic across two calls (max abs diff 0); for source (b), `tp_base_provenance.json` matches (spec sha256 equals the GF materialized spec, `go_function/` code hashes equal GF's code pin). Records the §10.4 exposure facts. Pins.
- **Remote G-V2 / G-F2.** Validation search and freeze, in search mode: w_H (Tier A, frozen c*, τ*); PC (c_PC*, τ_PC*, α*) for Tier A; Tier B AP-T, fixed-w, AP-T+R, controls; X+DS α sensitivity. `finalists_remote.json`, pinned with the same code hashes as G-F (a code change is a §14 amendment and needs user approval).
- **Remote G-T (single test scoring).** `score_test.py` runs once with every frozen finalist, control, comparator and base (Tier A and B) from the pinned finalist files. It refuses if any pin or code hash differs, refuses if `results/go_function_fusion/test/` exists, and writes `test_read.json` (timestamp, hashes) first. Outputs: per-term test AUC/AUPR tables, per-protein precision/recall tables, the DGZ-faithful check. Then `report.py` computes §6–§8 and `decisions.json`. A crash after `test_read.json` is written is reported; the rerun is allowed only for the identical command and code hashes and is logged as such.
- **Remote G-S2 (after GF G10).** §11.

If Tier B is unavailable when G-T is otherwise ready, the study pauses for the user: run G-T for Tier A only (Tier B families then never scored) or wait.

## 10. Exposure and exploration disclosure

1. **The validation pilot (spec §2) was exploratory and used the same DGZ validation split that selects the finalists here (U7).** It computed, for released MLP and DGZ, BP and MF, every family of §3 except co-annotation and PC-with-σ, saw band-level results, and on that basis chose the AP-T form, the adaptive weight w_t = c·τ/(τ + n_train) with c = 2 and τ ∈ {10, 30, 100}, and the grids of this protocol. Validation-selected parameters are therefore not independent of the validation numbers that motivated them; validation numbers are never reported as results, only test numbers are.
2. Pilot headline values (validation, released checkpoints; mean class AUC all / band 0 / band 1–10):

   | aspect, base | own | AP-T τ = 30, c = 2 | shuffled graph | protocol F |
   |---|---|---|---|---|
   | BP MLP | 0.7745 / 0.5896 / 0.7486 | 0.7970 / 0.7422 / 0.7951 | 0.7349 | 0.7767 |
   | BP DGZ | 0.7773 / 0.5419 / 0.7624 | 0.7926 / 0.7331 / 0.7901 | 0.7251 | 0.7855 |
   | MF MLP | 0.8747 / 0.6305 / 0.8798 | 0.9153 / 0.8727 / 0.9250 | 0.8459 | 0.8936 |
   | MF DGZ | 0.8949 / 0.7193 / 0.9142 | 0.9158 / 0.8873 / 0.9290 | 0.8544 | 0.9006 |

3. DGZ's released checkpoints were likely early-stopped on this validation split [inferred]. Validation is mildly optimistic for every variant alike.
4. Test-side contact so far: GF G2 scored the released predictions on test (published-model metrics only); GF smoke runs scored 2-epoch InterPro models on 6 BP test targets (`fusion_F.csv`, `baseline_DS.csv`; non-informative); the spec inspected `test_diamond.res` structure and test→train identity from `swissprot_exp.sim` (ids and identities, no labels). **No fusion score of any kind has been computed on test.** No test label was read in writing this protocol. If Tier B checkpoints come from the GF grid (source (a), §4.2), GF's own `eval.json` and G11 analysis may already report those bases' test metrics (class AUC on GF targets, protein-centric `all` metrics) before G-T. That exposes the base, not any fusion output; G-TP records whether any GF `eval.json` or `results/go_function/analysis/` file existed and whether it was opened by the fusion pipeline (it is not: the guard blocks `results/go_function/**` except the checkpoint files and specs named in G-TP).
5. A fresh validation half (decision 8 of spec (v)) was not chosen (U7).

### 10.3 Test-read guard

`go_function_fusion/guard.py` follows `go_ppi_fusion/guard.py`: a one-way `enter_search_mode()`; every loader calls `check_path`/`check_split`. Blocked in search mode: `dataset/go_function/dgz/*/test_data.pkl`, `predictions_*.pkl`, `test_diamond.res`, the remote test Diamond output, `results/go_function_fusion/test/`, every file under `results/go_function/` except `specs/*.json`, `protocol_pins.json`, `gates/*.json` and the Tier B `checkpoint.pt` files named in G-TP (directory listings for the §11.2 snapshot are allowed; opening is not), and the `test` key of a labels npz (the loader returns train and valid arrays only; the npz is hash-verified as a whole file). G-V, G-V2, G-F, G-F2 run in search mode; a guard violation is fatal (exit 2) and logged.

## 11. S2: descriptive fusion add-on to the go_function grid (separate preregistered section)

### 11.1 Status

S2 is preregistered here, before GF G8 and before any GF grid output (G10) exists. It is descriptive and unadjusted. It changes no go_function decision, arm, run, contrast, Holm family or code-pinned file; it is a separate script reading GF outputs. GF arm F (D15) is unchanged (U6).

### 11.2 Frozen parameters

S2 uses the Tier A **released-MLP** AP-T finalist (c*, τ*) per aspect from `finalists_tierA.json` (G-F). Reason: it is frozen on the Mac before GF G8, so no GF output can influence it, and the released MLP has V0's architecture (GF §6.1).

**Snapshot rule for remote parameters (mechanical; enforced by `s2_grid_fusion.py`).** The boundary is the start of GF G10, not G11. `freeze_finalists.py --remote` (G-F2) records in its `protocol_pins.json` entry a `gf_grid_snapshot`: the pin's UTC timestamp, the number of directories under `results/go_function/runs/` that contain any of `epochs.jsonl`, `eval.json`, `scores.npz`, `run.json` or `DONE.json`, and the number of `*.CLAIM` files there (listing only; no file is opened). A parameter set from `finalists_remote.json` (the Tier B V0 AP-T finalists as a sensitivity, and the Tier A MLP w_H) is eligible for S2 **only if both counts were 0** at G-F2. As a cross-check, `s2_grid_fusion.py` verifies that no study-mode run's `epochs.jsonl` has an mtime earlier than the G-F2 timestamp (`run.json` carries no start time, so the earliest training artefact is used); any violation makes the set ineligible. Otherwise the script uses only `finalists_tierA.json` and logs `tierB_in_s2: false` and `wH_in_s2: false` with the reason (`grid_started_before_GF2` or `mtime_check_failed`). Under source (a) of §4.2 the grid has necessarily started before G-F2, so remote parameters are then never used in S2.

AP-T+H_S2 uses the Tier A MLP w_H from `finalists_remote.json` under this rule, with GF's per-condition DS (0 at k = 0 under W2, so AP-T+H_S2 = AP-T_S2 there); otherwise AP-T+H is not computed in S2. Note (descriptive only): w_H is selected on DS from our train-only remote Diamond run (§4.3), but in S2 it is applied to GF's per-condition DS built from DGZ's `test_diamond.res`. The transfer source differs, so AP-T+H_S2 reads carry this label.

### 11.3 Definitions

For each GF run (arm ∈ {V0, V4}; TP/W2 main block, W1 block; ESM and InterPro) and each pack target t with stored columns (GF design §7.3: t, Pa(t), Sib(t), X(t), Desc(t) in `scores.npz`):

- AP-T_S2(p,t) = z(ℓ_t) + w_t · z(Pa̅(t)), w_t = c*·τ*/(τ* + k), with n_train(t) = k = the condition's retained count (W2 and W1); z over the run's evaluation-set test proteins (`eval_rows`).
- Parent-only (weight-free) beside GF arm F, F recomputed with `go_function.fusion.fusion_scores` (U6).
- AP-T_S2 + R (BP), descriptive: the R block uses the columns X(t) ∩ cols(Reg(t)), where Reg(t) comes from the S1 confirmatory-mode builder (§3) and X(t) is the stored `scores.npz` channel. The stored `X_reg` channel (DGZ closure of the regulates targets plus logical-definition fillers, `go_function.targets.channel_sets`) is **not** used, because it differs from S1's Reg. Reg columns not stored in X(t) are skipped, and the number of skipped (t, column) pairs is reported per run.

### 11.4 Reads (all descriptive, unadjusted)

Δ(AP-T(V0) − V0) and Δ(AP-T(V4) − V4) per k ∈ {0, 1, 5, 10, 25}; **AP-T(V0) − V4 at k = 0** (key read: inference-time structure vs train-time axioms); Δ by S_sib (low/high) and LD (BP) strata; parent-only and F beside AP-T. Uncertainty: GF's two-way pack × seed bootstrap (`go_function.stats`, vendored `two_way_bootstrap`), B = 10,000, `default_rng(20261011)`, imported unchanged; GF's classification of §9.1 is printed with the label "descriptive".

### 11.5 S2 dry-run now (Mac)

`s2_grid_fusion.py --smoke` on `results/go_function/smoke/runs/**/scores.npz` (V0, V3, V3c; BP p01; k ∈ {0, 5}; 2 epochs): (a) F recomputed equals `results/go_function/smoke/analysis/fusion_F.csv` to 1e-9; (b) AP-T_S2 and parent-only computed for every V0 run; (c) the output schema is complete. Outputs go to `results/go_function_fusion/s2_dryrun/`, each file labelled "smoke, 2 epochs, non-informative". It uses placeholder parameters (c, τ) = (2, 30) if G-F has not run, flagged as such; no dry-run number is reported as a result.

### 11.6 Cross-reference row for `protocol_go_function.md` §14 (U5)

Added once, before GF G8, after this protocol is approved; the only edit to an existing go_function file:

| id | date | gate | change | cause | confirmatory output seen? |
|---|---|---|---|---|---|
| X1 | (date of the edit) | before G8 | Cross-reference only: a descriptive fusion add-on to this grid (S2) is preregistered in `protocol_go_function_fusion.md` §11 (version at edit, sha256 `<hash at edit>`; later changes are recorded in that file's §14). It reads grid outputs after G10 with S1-frozen parameters. **No go_function decision (D1–D19), arm, run, contrast, family, gate or code-pinned file changes; arm F (D15) is unchanged.** | User decision U5 of the fusion study. | no |

## 12. Success criteria and readings

Classification per contrast (GF §9.1 logic; effect floor 0.01 AUC for P, Sc, Ty; 0.002 AUC for Hc; 0.01 for Hf):

```text
1. harm:              ci_high < 0 and estimate <= -floor
2. supported:         p_holm < 0.05 and ci_low > 0 and estimate >= +floor
3. practically null:  -floor < ci_low and ci_high < +floor
4. below floor:       ci_low > 0 and estimate < +floor
5. inconclusive:      everything else
```

- **S1 primary success (per aspect):** P_MLP supported. Reading: "an inference-time true-path parent prior improves class-centric AUC of a no-ontology model on DGZ's split". With P_DGZ also supported: "…and adds to DGZ's train-time axioms".
- **Structural wording** ("the gain needs the real GO edges") requires ci_low(D_struct_b) > 0 for that base, where D_struct_b = Δ_{AP-T(b; c*, τ*) − AP-Tshuf(b; c*, τ*)}[all] uses the **frozen** parameters (§3.5, draw set U). Otherwise the report says the gain is not shown to be structural. The re-selected shuffled-graph scoring never enters this reading.
- **Scarcity:** Sc0 and Sc1 supported for a base → "the gain concentrates on scarce and zero-train terms" for that base.
- **MLP + AP-T vs DGZ** (descriptive): stated per band with CIs; "matches" only if the CI of AP-T(MLP) − DGZ lies within ±0.01 in that band, "exceeds" only if ci_low > 0.
- **Null selected on validation.** If the frozen weight of a confirmatory contrast is its null cell (w_H = 0 for Hc, α* = 0 for Hf, w_R = 0 for Ty; P and Sc cannot hit this case because the AP-T grid has c ≥ 1), the contrast is labelled **null selected on validation**. It is not classified by rules 1–5, keeps its Holm slot with p = 1 (family size fixed), and its reading below is "not tested on test". The same label is used, descriptively, for a control whose re-selection picks the null cell (§3.5).
- **Homology:** Hc below floor or supported → "homology adds little class-centric AUC"; Hf supported → "homology lifts protein-centric Fmax beyond the parent prior". PCp, PCt and XDS are descriptive and are cited only as such. The homology channel is reported as a baseline/partner, not as ontology.
- **Typed (H4):** supported on Tier B → "typed regulates-family priors help on true-path-trained models", worded as "typed structure" only if ci_low(D_typed_b) > 0 for that base, where D_typed_b = Δ_{AP-T+R(b; w_R*) − AP-T+Rshuf(b; w_R*)}[reg-subset] uses the **frozen** w_R* (§3.5, draw set R); otherwise "an extra column block helps; typed structure not shown". If w_R* = 0 the reading is "null selected on validation". Tier A typed reads are never cited as evidence.
- **Controls:** a control matching or beating its fusion (frozen-parameter scoring, point estimate) is reported next to the fusion's result and limits the reading. Co-annotation: the confirmatory CoA (TP relatives excluded, §3.5) at AP-T's frozen (c*, τ*) ≥ AP-T → "data-driven label structure suffices". CoA⁺ and its overlap share are reported beside it and never support that reading. Re-selected control scorings are descriptive only.
- **Collateral:** PC is reported with Fmax, Smin, AUPR next to the base and X+DS; a class-centric finalist is never reported with protein-centric metrics (z-sums are uncalibrated).

## 13. Pins

Pilot snapshot (spec §2; recorded at authoring; verified at G-R):

| file (`/tmp/gofunc_fusion/`) | sha256 |
|---|---|
| struct_bp_mlp.json | 80509d22f029e3d9c846c2eae45469410faaf075aa5fe8a4751add23677e5730 |
| struct_bp_deepgozero.json | 9c2a22e15e09384df161fa34c799f39251a4a8ae036ad13a8bc1877d62324c9a |
| struct_mf_mlp.json | 56ec317d63c1904ee48572a103f11497f493b89fc8f0a6ba29a65f3088c58966 |
| struct_mf_deepgozero.json | b8fa2e3823d332e566259e52c05fe21622735cd56666b33bdaca576a46a3d868 |
| pilot_bp_mlp.json | 5fed1e4436bc9dbb3f8ef1c6ea3ab053b87e1acd2fa3dd35064caaef0e411820 |
| pilot_mf_mlp.json | 32f42bc086ce001574193bc7051515a0c7ce89b13df2c2aa05c614a4003a5c70 |
| pilot_mf_deepgozero.json | c1ac81edee47c7c47fd693fc36c56d2904e6d7d770fd4bce403e361d64be1dfa |
| c_struct.py | e11a27d2dafc2de03a31e0b7d7d30981606839bd95985d74fe8ec0f62628d822 |
| b_valid_pilot.py | 660883d74fe59b9aecf83575d6e7b9393dac8ba1b3a76bbb1dba33fe33a62186 |
| swissprot_exp.sim | a143b3896954a4a794d4715205a5db09b6d7ee391d41cea6bd626194576c3101 |
| dgz_evaluate.py (DGZ commit 4507529e) | 0a89c4c013eea4d17f22f7235be6c8351e99ad3b7ac4f8712bbf49f0cb9af848 |

Gate pins (filled by scripts): G-R (`gates/gate_repro.json`), G-F (`finalists_tierA.json`, `code_pin.json`), G-D (DIAMOND version, binary and output hashes), G-TP (checkpoint hashes, source a/b), G-F2 (`finalists_remote.json`), G-T (`test_read.json`).

Recorded on the Mac (CPU), `results/go_function_fusion/`, 2026-10-14 (mirrored from `protocol_pins.json`):

| pin | file | sha256 |
|---|---|---|
| G-R (b, c) | `gates/gate_repro.json` | 1e99cc3272b15cd7072065afbc97a84703cca18434404ab31ddd9afd74420aca |
| G-F (UTC 2026-10-14T14:34:46Z) | `finalists_tierA.json` | 5342268747cc8b9eb7878edfba4cb5e57270a08801052c407c57bb12afcc77f1 |
| G-F | `code_pin.json` (25 files) | c212317fe73c34700c0f0399836c92e39d6ead52b5d88bdcd39b03b040140659 |

## 14. Amendments

| id | date | gate | change | cause | test output seen? |
|---|---|---|---|---|---|
| — | | | | | |

## 15. Constraints, implementation layout and remote plan

**Constraints.** Preregistration only until approved. No modification of protected paths: `contrastive_learning/`, `go_function/`, `scripts/go_function/`, `config/go_function/`, `go_ppi_fusion/`, `go_ppi_domain/` (existing files), `scripts/go_withholding/`, `config/go_withholding/`, `trials_domain/`, `scripts/trials_fusion_confirm/`, `scripts/go_ppi_fusion_confirm/`, `scripts/evaluate_publication_fusion.py`, `scripts/run_learning_curve_point.py`, `results/go_function/`, `results/go_ppi_fusion_best/`. The only later-allowed edit to an existing file is the §11.6 row. Protected packages are imported, never edited. No git commit, stash, reset, checkout, revert or clean by any step. Mac: CPU only (`CUDA_VISIBLE_DEVICES=""`), no MPS, nothing installed, no training; dense score matrices are never written (BP valid is ~0.9 GB in float64 in RAM; outputs are per-term tables); `df -h` checked before writes (≈ 2.8 GB free at v1, 7.5 GB at v2, 5.5 GB at v3). Anything needing a product decision or a protocol change after G-F stops for the user.

**New files.** `config/go_function_fusion/{study.json, input_hashes.json, dgz_published_ds.json, dgz_evaluate_ref.py.txt}`; `go_function_fusion/{__init__, data, priors, fuse, metrics, stats, guard}.py` and `go_function_fusion/tests/`; `scripts/go_function_fusion/{gate_repro, search_valid, freeze_finalists, diamond_valid_test.sh, train_tp_base, verify_tp_bases, score_test, report, s2_grid_fusion, dump_code_hashes}.py`; `results/go_function_fusion/`; `.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_GO_FUNCTION_FUSION.md`.

**Mac (now, after approval):** G-0, G-R, G-V, G-F, G-S2dry.

**Remote (L40S; CPU suffices except an optional dedicated Tier B training):**

1. Clone the pushed branch; re-download DGZ data with `scripts/go_function/download_dgz.py`; verify `input_hashes.json`; verify `finalists_tierA.json` and `code_pin.json` hashes.
2. G-D: install the pinned DIAMOND binary; train-only DB; valid and test queries; pin.
3. G-TP: Tier B checkpoints (source a or b); verify; pin.
4. G-V2/G-F2 in search mode; pin.
5. G-T: `score_test.py` once; `report.py`.
6. After GF G10: G-S2 (`s2_grid_fusion.py` on `results/go_function/runs`).
7. Return: `results/go_function_fusion/` (no checkpoints, no dense matrices).

## 16. Limitations

- Tier A bases were trained on DGZ-rule labels and probably early-stopped on validation; typed reads on them are confounded, and validation is optimistic.
- Finalists are selected on the validation split the pilot already explored (§10).
- z_t standardises over a label-defined protein set (proteins with ≥ 1 annotation in the aspect), as DGZ's evaluation does; it is transductive over test proteins but uses no label of t.
- Class-centric AUC ignores calibration and the frequency prior; protein-centric claims rest on PC only.
- Few bases (2 or 3) make the two-level bootstrap's base level under-cover.
- DS at test uses our train-only Diamond run, which differs from DGZ's train + valid DB; DGZ's own form is a descriptive check.
- Only DGZ's 50%-identity split; no time split (GF D6, D13).
- S2 is descriptive, and its frozen parameters come from a full-data released model, not from TP-trained V0.
- Prior-art positioning beyond the DGZ paper rests on memory (spec (vi)).
- The term bootstrap (spec (iii)4, U2) conditions on the evaluated proteins and treats GO terms as independent, though parent and child terms share proteins; its CIs are likely too narrow. A descriptive protein-bootstrap CI for P_b (§8) is reported beside it.
- Tier B source (b) uses `go_function.train.run`'s `out_dir` override, documented there as "tests only"; provenance is recorded separately (§4.2).
- The §11.2 snapshot rule uses directory listings and file mtimes, which are host-local and could be altered; it is a procedural check, not a cryptographic one.
- The confirmatory co-annotation control excludes t's TP relatives, but its Jaccard is still computed on TP-closed labels. Closure can raise J(t,c) for non-relatives that share deep ancestors, so CoA is "no direct ontology edges" rather than fully ontology-free.
- In S2, w_H is transferred from train-only remote Diamond DS to GF's DS built from `test_diamond.res` (§11.2); AP-T+H_S2 reads are labelled with this source difference.

## 17. Responses to design review round 1 (`design_review_go_function_fusion.{json,md}`)

| # | sev | response | where |
|---|---|---|---|
| 1 | HIGH | **Addressed.** Hf_b = Fmax(PC(c*, τ*, α*)) − Fmax(PC(c*, τ*, α = 0)), confirmatory, F-H size stays 4. PCp, PCt, XDS added as descriptive reads with the protein bootstrap. §12 wording now "beyond the parent prior". | §1 H3, §3.3, §6.1, §6.2, §12 |
| 2 | MEDIUM | **Addressed, with a different mechanism.** Boundary moved to the start of GF G10. The suggested check ("earliest `run.json` start time") is not implementable: `run.json` has no start time (verified in `go_function/train.py` `_finish`; only `wall_seconds`), and `run.json` is written last. Instead G-F2 records a `gf_grid_snapshot` (counts of started run directories and `*.CLAIM` files, listing only), remote parameters enter S2 only if both counts are 0, and `s2_grid_fusion.py` cross-checks `epochs.jsonl` mtimes against the G-F2 timestamp and logs the decision. AP-T+H sentence rewritten. | §11.2, §16 |
| 3 | MEDIUM | **Addressed.** "Null selected on validation" label: not classified by rules 1–5, Holm slot kept with p = 1, reading "not tested on test". | §12, G-0 |
| 4 | NIT | Addressed: confirmatory mode removes `duplicate_cols` and t's own columns (matches `channel_sets`); pilot-compat mode unchanged; changed-term count reported at G-R(c). | §3 |
| 5 | NIT | Addressed: construction kept for parity; counts of perm[π] ∈ {root, t} and alt-id reported. | §3.5 |
| 6 | NIT | Addressed: shuffled graph (PC) control, descriptive. | §3.5 |
| 7 | NIT | Addressed: shuffled-Reg control, descriptive; "typed structure" wording needs Ty − shuffled-Reg Ty CI > 0. | §3.5, §12 |
| 8 | NIT | Addressed: "around the 0.002 floor". | §1 |
| 9 | NIT | Addressed: draw sets U, S (six-band strata, [>100] pooled), R, in fixed order from one generator per aspect. | §8 |
| 10 | NIT | Addressed: limitation added; descriptive protein-bootstrap CI for P_b (B_P = 200 to bound cost). | §8, §16 |
| 11 | NIT | Addressed: per-protein has-prediction indicator at each threshold. | §8 |
| 12 | NIT | Addressed: raw ℓ for DGZ; propagated value as sensitivity. | §6.2 |
| 13 | NIT | Addressed: listed among implementer choices. | §2 |
| 14 | NIT | Addressed: expectation marked extrapolated. | §3.5 |
| 15 | NIT | Addressed: row text names the version and points to this file's §14. | §11.6 |
| 16 | NIT | Addressed: disclosure in §10.4; guard now blocks every `results/go_function/` file except specs, pins, gate JSONs and named checkpoints. | §10.4, §10.3 |
| 17 | NIT | Addressed. V3/V3c drop stated with reason. Feasibility checked: `run.py` rejects an `out_dir` outside `results/go_function/` (`validate_spec`, `RESULTS_PREFIX`); source (b) now uses the fusion-owned `train_tp_base.py` calling `go_function.train.run(..., out_dir=…)` with provenance verified at G-TP. | §4.2, §9, §15, §16 |

## 18. Responses to design review round 2 (`design_review_go_function_fusion.{json,md}`, round 2)

| # | sev | response | where |
|---|---|---|---|
| 1 | MEDIUM | **Addressed.** D_struct_b and D_typed_b are defined on the frozen-parameter scorings (draw sets U and R). The structural and typed-structure readings are ci_low > 0 on them. If w_R* = 0, D_typed is labelled "null selected on validation". Re-selected control scorings are stated to be descriptive only and never enter a §12 reading. | §3.5 (new "Which scoring a reading uses"), §6.2, §12, G-0 |
| 2 | MEDIUM | **Addressed as suggested.** The confirmatory CoA excludes cols(t), root, duplicates, cols(anc_TP(t)) and cols(desc_TP(t)), and requires J > 0. The old construction is kept as descriptive CoA⁺ with the per-band overlap share. The §12 co-annotation reading uses CoA at frozen (c*, τ*) only. A limitation notes that CoA is still computed on TP-closed labels. | §2 implementer choices, §3 sets, §3.5, §12, §16, G-0 |
| 3 | NIT | Addressed: Sib defined via `isa_siblings` (minus t, anc_TP, desc_TP); anc_TP/desc_TP defined. | §3 |
| 4 | NIT | Addressed: Holm written out (fixed m, ties by §6.1 table order, step-down max, p_adj used by rule 2). | §8 |
| 5 | NIT | Addressed: PC parameters written (c_PC*, τ_PC*, α*) and stated distinct from AP-T's (c*, τ*). | §1 H3, §3.3, §3.5, §6.1, §9 G-V2 |
| 6 | NIT | Addressed: S2 R block = X(t) ∩ cols(Reg(t)) from the S1 builder; `X_reg` not used; skipped columns counted. | §11.3 |
| 7 | NIT | Addressed: one-line descriptive note on the DS source difference. | §11.2, §16 |
| unverified | — | Resolved: `go_function.pins.assert_writable_target` checks only `PROTECTED` (which excludes `results/`), so `results/go_function_fusion/tp_bases/` is writable via `train.run(..., out_dir=…)`. Provenance paths are repo-relative because `canonical_json` refuses absolute paths. The other three unverified items (DGZ early stopping, Tables 3–4 DS cells, GF grid remote status) stay disclosed and are resolved at G-R / G-TP as already specified. | §4.2 |
