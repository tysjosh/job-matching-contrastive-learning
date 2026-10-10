# Per-target withholding on the GO/PPI domain: findings

Read-only follow-up to `findings.md` (career, InfoNCE) and `findings_v7_ordinal.md` (career v7, ordinal loss). Repo `/Users/olukotunjosh/Downloads/CDCL`, HEAD `699e0cb`, with uncommitted edits in `contrastive_learning/{batch_processor,loss_engine,trainer}.py`. Nothing in the repo was modified and nothing was trained. Scratch scripts live in `/tmp/goinv/` (`a_split.py`, `b_gaf.py`, `c_targets.py`, `d_refine.py`, `e_negaudit.py`, `f_rawdeg.py`, `g_relation.py`).

Source of truth is the raw data in `dataset/go_ppi/` (STRING v12 `9606.protein.{info,links.detailed,aliases}`, `goa_human.gaf.gz`, `go-basic.obo`, plus HuRI and Lit-BM in `independent/`). It flows `data_converter.py` → `preprocess/go_ppi/{proteins,pairs}.jsonl` (+ `convert_manifest.json`, `go_index_{P,F,C}.pkl`) → `data_splitter.py` → `preprocess/go_ppi_splits/{train,validation,test,negative_pools}.jsonl` → `scripts/create_go_ppi_lc_fractions.py` → `preprocess/go_ppi_lc/frac_*/train.jsonl` + `validation_positive.jsonl`. All counts below are computed from those files.

Evidence tags:
- **[read]**: from code.
- **[ran]**: from a read-only Python snippet over the data or existing checkpoints.
- **[inferred]**: reasoned, not directly verified.

## Summary answer

1. **The split is clean, which is the opposite of career.** `data_splitter.build_splits` partitions *proteins* 70/10/20 and keeps only pairs whose two endpoints fall in the same partition.
   - Train, validation and test share 0 proteins in any role [ran]. That also means 0 shared pairs, and no "same resume + same occupation" analogue.
   - Every test protein is therefore already a zero-shot entity. So "withhold a *protein*" is meaningless on this split.
2. **The right target unit is a GO term t, not a protein.** The analogue of an occupation is a category of proteins.
   - P(t) is the set of proteins whose propagated BP annotation contains t.
   - The examples of t are training positive pairs with an endpoint in P(t).
   - Test support for t is the set of test queries whose anchor is in P(t).
   - The supply is large. Take BP terms with ≥40 train proteins, ≥8 test anchors that have both a grade-2 and a grade-1 partner, and removal of ≤5% of train positive pairs. That gives 469 terms. 158 of them form a non-nested set with pairwise protein-Jaccard ≤0.2 [ran].
   - S(t) over sibling terms ranges from 6 to 22,790 pair-endpoints, and is uncorrelated with n(t) (Spearman −0.04) [ran]. Tertiles give 52 low-S and 52 high-S targets.
   - **This supports a real n × S design, which career could not.**
3. **There is no ontology-in-loss arm for GO today.**
   - `loss_type="go_ordinal"` (`loss_engine._compute_go_ordinal_loss`) is InfoNCE plus evidence-grade ranking: high > weak > unobserved, all from STRING experimental scores. It contains no GO term.
   - `ontology_weight` is inert, because GO records carry no `ontology_similarity`. It only rescales the loss by the 0.5 default tier weight.
   - `career_distances` (10·(1−simGIC)) is computed by the selector, but no loss used on this path reads it [read].
   - GO enters training *only* through negative selection, and that is **repulsive**: GO-close weak or none partners are pushed away. This actually *lowers* GO relation preservation in the learned space (Spearman(cos, simGIC) 0.09–0.15 vs 0.19–0.22 for the controls) [ran].
   - **As in career, nothing lets GO-neighbour information reach a withheld term.** A new loss term is needed for the 2×2. Its specification is in (i).
4. **The leakage to remove is in the features, not the split.**
   - (a) **Text.** The STRING/UniProt `encoder_view` of one protein names its partner's gene symbol in **11.1% of test positives, against 0.17% of weak and 0% of no-interaction pairs**. The train rate is the same, 11.2% [ran].
   - (b) **GO annotation.** A kept GO annotation's with/from column names the partner in 5.9% of test positives, against 0.4% of weak [ran]. 863 MF terms that descend from "protein binding" survive the exclusion of `GO:0005515` itself.
   - Validation chooses `best_checkpoint.pt` by each arm's own loss with its own negatives. For withholding it must also be made target-free.
5. **Prior evidence.** Under InfoNCE, test 22,341 records:
   - GO-guided negatives are null on the hard contrast (+0.0011 ± 0.0076 AUC vs the random-window control, n = 5) and negative on easy/pooled.
   - Post-hoc score fusion with simGIC adds +0.016 to +0.023 hard AUC at every fraction (n = 3).
   - The go_ordinal decomposition (23 runs, 1–2 seeds) has been trained but **never evaluated**.
6. **Verdict.** GO/PPI is a **better confirmatory domain than career v7**: entity-disjoint, about 150 usable targets, n ⟂ S, and train/eval text serialization identical. It still needs:
   - a symbol-masked `encoder_view`;
   - a pruned GAF;
   - a withholding generator that removes target proteins from records, pools and validation;
   - one new GO loss term;
   - a frozen-embedding evaluator with per-target metrics.
   No re-split is needed.

---

## A. Task and data

**Task [read, ran].** This is protein–protein interaction retrieval. The query ("resume" slot, `protein_id`) is an anchor protein. The candidate ("job" slot, `partner_id`) is another protein (`data_splitter._build_record`).
- **Grades** (`data_converter.py` header, `convert_manifest.json`):
  - 2 = STRING experimental ≥700;
  - 1 = 0 < experimental < 150;
  - 0 = no STRING edge;
  - 150–699 is dropped.
  - The labels map onto the career vocabulary good_fit / potential_fit / no_fit (`ORDINAL_LABEL`). The binary label is 1 only for grade 2.
- **Features:** `encoder_view` = "SYMBOL — STRING protein.info annotation" (median 324 characters) on both sides, plus `skill_uris` (direct BP GO terms) and `coarse_uris` (BP ancestors of depth ≤3).
- **Converter caps:** at most 8 positives, 40 weak and 40 none per anchor (`positives_per_anchor=8`, `weak_pool_cap=40`, `none_pool_size=40`). The anchors are 7,041 proteins with ≥1 grade-2 partner and ≥4 grade-1 partners. There are 15,155 eligible proteins (BP-annotated with ≥50 characters of text) and 572,637 graded pairs.

**Splits** (`preprocess/go_ppi_splits/split_manifest.json`, verified [ran]):

| split | records | grade 2 / 1 / 0 | anchors with records | proteins |
|---|---|---|---|---|
| train | 15,708 | 15,708 / 0 / 0 (positives only; negatives come from `negative_pools.jsonl` at batch time) | 4,547 (4,948 have pools) | 4,709 with positives |
| validation | 5,587 (`validation_positive.jsonl`: 321) | 321 / 2,479 / 2,787 (grade 0 capped at 20 per anchor) | 697 | 1,457 |
| test | 22,341 | 1,165 / 10,011 / 11,165 (uncapped) | 1,396 (735 with ≥1 positive) | 3,014 |

- 261,608 pairs that cross partitions were dropped.
- Train records are directed (anchor → partner). Of 15,708, 8,334 also appear in the reverse direction, giving 11,541 undirected positive pairs [ran]. Withholding must remove both directions.
- Undirected training degree per protein: median 3, p90 11, max 99 [ran].
- Test positives per anchor: median 1, max 6 [ran].

**Overlap [ran, `a_split.py`].** Train∩test, train∩validation and validation∩test proteins: **0 / 0 / 0**.
- Pairs shared: 0, and no protein recurs, so there is no analogue of career's 144/200 "same resume + same occupation in train".
- The splitter asserts zero overlap when building (`build_splits`, "leakage audit" block). `run_learning_curve_point.attach_go_ppi` also asserts train/validation pool disjointness.
- **LC fractions** (`create_go_ppi_lc_fractions.py`) are nested prefixes of the shuffled directed train file. Every fraction shares the full train `negative_pools.jsonl`. Validation for training is `validation_positive.jsonl`, the 321 grade-2 records with validation pools.

## B. Model and representation

[read]
- Frozen `all-mpnet-base-v2` (`freeze_text_encoder=true`) over `encoder_view`.
- One `CareerAwareContrastiveModel.projection_head`, shared by both slots: Linear 768→256, ReLU, Dropout 0.3, Linear →128, L2-normalised (`trainer.py` ~L1477–1530). Checkpoint keys `projection_head.{0,3}.*` [ran].
- Score = dot product of the projected embeddings.
- The text path is identical in training and evaluation. `embedding_cache.py` L517–525 and `trainer.py` L1234 return `encoder_view` verbatim, and so does `run_phase1_embedding_evaluation.content_to_text` L225–237. This is unlike career, which had a serialization mismatch.

**Unseen entities.** A protein with zero training pairs (every test protein) is represented only by its text through the shared projection. GO annotations are not model inputs. The only place GO touches an unseen protein's *score* is post-hoc fusion (`scripts/evaluate_publication_fusion.py`: z(text) and z(simGIC), with the weight chosen on validation).

**Paths from GO neighbours to a withheld term:** none during training.
- Negative selection uses only the GO annotations of *training* anchors and their pool partners.
- The only route is indirect and text-mediated: proteins of sibling terms have similar function text, so the projection learned from them generalises [inferred].

## C. Ontology in the loss

**The go_decomposition configs [read, ran].**
- `config/go_decomposition/*.json` is the InfoNCE materialisation, and `config/go_decomposition_go_ordinal/*.json` adds `loss_type=go_ordinal` with `GO_ORDINAL_DEFAULTS`. They are produced by `scripts/run_go_decomposition.py::arm_config` from `lc_go_ppi_ontneg_stoch.json`.
- The 12 arms differ only in `go_aspect` ∈ {P, F, C, A} × `go_similarity_mode` ∈ {exact, ancestor, simgic}. That changes **which similarity orders the negative pool**:
  - exact = Jaccard of direct terms;
  - ancestor = Jaccard after is_a/part_of propagation;
  - simgic = IC-weighted Jaccard;
  - A = mean of the per-aspect scores.
- Every arm has `go_ppi_go_tiered_negatives=true`, `tier_sampling=stochastic` and `window=0.34`. The control is `randwin` (`tier_sampling=random_window`, no matcher).
- **None of the 12 arms changes the loss.** "Aspect × mode" is a negative-selection decomposition.

**What the loss contains [read].**
- `_compute_go_ordinal_loss` (`loss_engine.py` L227–281) = InfoNCE + λ_hw·mean softplus((m_hw − (s_high − s_weak))/τ)·τ + λ_wu·mean softplus((m_wu − (s_weak − s_none))/τ)·τ.
- The defaults are λ_hw=1.0, λ_wu=0.1, m_hw=0.1, m_wu=0.05, τ=0.1. Grades come from the selector's `grade` field.
- **It encodes the STRING evidence partial order, not GO.**
- `_compute_ontology_weight` returns `tier_weights.get(None) = 0.5` whenever `ontology_weight>0`. That is a constant rescale, because GO records have no `ontology_similarity`, `quality_tier` or `ot_distance`.
- `use_ot_distance`, wasserstein/hybrid and WS2 are label-based or off. ORCA (`orca_enabled=false` in every GO config) is the only machinery that would feed GO features (`negative_ontology_features` via `set_ontology_matcher`) into a loss. It was never run on GO, and it is not a clean single factor.
- **Therefore no current term relates protein i to protein j via GO in the loss.** The S(o) mechanism missing in career is missing here too.

**Strength of the only GO signal in training [ran, `e_negaudit.py`].** Over 800 train anchors, simGIC (BP) of each anchor with its pool:

| pool | n | mean | sd | CV | in HuRI ∪ Lit-BM |
|---|---|---|---|---|---|
| positives | 2,706 | 0.232 | 0.254 | 1.09 | 17.3% |
| weak (grade 1) | 20,943 | 0.064 | 0.089 | 1.40 | 0.37% |
| none (grade 0) | 22,548 | 0.027 | 0.045 | 1.70 | 0.01% |

The GO window raises the mean simGIC of sampled negatives from 0.040 to 0.085. That is still far below positives.

Loss components are not logged. `contrastive_training_metrics.jsonl` carries the total only, so the per-component and gradient scale of the go_ordinal ranking term was not measured. That probe was not run because the term carries no ontology. It belongs in gap 7.

**Uncommitted edits [read].** The `batch_processor._create_triplet` `ordinal_candidates`/`ordinal_floor` block and the loss_engine ordinal changes are gated on `loss_type == 'ordinal'`. `trainer._loss_skill_matcher` returns the existing matcher (None at construction) unless `loss_type=='ordinal'`. The `ontology_rank_tie_break` edit is in the career `_select_ontology_negatives`.
- **None of them affects the GO path** (infonce / go_ordinal + domain selector).
- They should still be committed before the study, so the code hash is fixed.

## D. Ontology-guided negatives

[read: `go_ppi_domain/negative_selector.py`]
- `GoPpiNegativeSelector.select_batch_negatives` draws 7 negatives per anchor from **that anchor's own** weak and none pools. In-batch and global pools are off.
- The grade-1 share ramps linearly from 0.2 to 0.6 over the epochs. RNG is seeded per (seed, epoch, anchor).
- Modes:
  - `go_tiered=false`: uniform;
  - `random_window`: a fixed random 34% of each tier, chosen once per anchor;
  - `go_tiered=true, stochastic`: each tier sorted by 1 − simGIC (unannotated last, first 400 scored), then sampled from the closest 34%;
  - `deterministic`: a fixed prefix, which is confounded by loss of per-epoch diversity, as the docstring itself notes.
- **False-negative guard:** none pools exclude *any* STRING edge, including the dropped 150–699 band (`data_converter.build`). Weak pools are by definition unconfirmed.
  - Audit with the production selector at epochs 0 and 14 [ran]: about 0.3–0.5% of sampled grade-1 negatives are HuRI/Lit-BM interactions, and 0–0.04% of grade-0.
  - The GO window does **not** raise that rate (0.28% vs 0.45–0.50%).
- **Independence:** the selector is attached through the domain slot (`run_learning_curve_point.attach_go_ppi`). The matcher is built only when `go_ppi_go_tiered_negatives=true`, and the attach raises if the flag is on without a matcher. This toggle is independent of `loss_type`.

**Mapping onto the 2×2 (go_ordinal loss).** "Loss" stands for the GO loss term (new, see (i)).

| | negatives off (`random_window`) | negatives on (GO window) |
|---|---|---|
| loss off | `config/go_decomposition_go_ordinal/randwin.json` | `config/go_decomposition_go_ordinal/go_P_simgic.json` |
| loss on | **missing**: needs a new loss field + loss-only matcher | **missing**: same field on `go_P_simgic` |

- randwin ↔ go_P_simgic is a clean single-factor pair [ran, diff]. The only differences are `go_ppi_go_tiered_negatives`, `go_ppi_tier_sampling` and two inert fields (`go_aspect_weights`, `go_similarity_mode`).
- **Confounded or excluded:**
  - `lc_go_ppi_ontneg_only.json` (deterministic prefix);
  - `lc_go_ppi_baseline.json` vs `ontneg_stoch` (window narrowing; baseline→randwin is −0.0041 hard, n = 3);
  - F/C/A arms (aspect coverage: missing MF annotations get neutral distance and sort last);
  - `config/go_decomposition/*` (InfoNCE, if the study uses go_ordinal).
- Keep `lc_go_ppi_baseline.json` (uniform) as an optional fifth arm.

## E. Ontology graph and neighbourhoods

**Graph [read: `go_ontology.py`].**
- `parse_obo` follows `is_a` + `part_of` only. `regulates` is dropped, and obsolete terms are dropped.
- `GoIndex` holds the ancestor closure, the depth from the root, direct terms after the guards, and closure IC = −log(frequency/19,293 genes) over the whole GOA corpus. The IC is not derived from training.
- `parse_gaf` drops NOT qualifiers, IEA, IPI, ND, `GO:0005515` and `GO:0005488`. Manifest for BP: 23,974 terms, 19,293 genes, 138,976 kept lines.

**Target unit and neighbourhood (proposed, preregister):**
- **Target:** BP term t. P_train(t) = train proteins with ≥1 positive whose propagated BP closure contains t.
- **N_sib(t):** terms sharing an is_a/part_of parent with t, excluding t's ancestors and descendants.
- **N_2(t):** terms within 2 undirected DAG hops, with the same exclusion.
- **S(t)** = Σ training positive-pair endpoints over proteins in ∪P(t′) for t′ ∈ N(t), minus P(t).
- **n(t)** = |P_train(t)|, counted in proteins, which are the retention unit.

**Distributions [ran, `c_targets.py`, `d_refine.py`].** Base: 4,709 train proteins and 11,541 undirected positives.
- Terms with ≥25 train proteins, ≥5 test positives and ≤10% of train proteins: 1,293.
- With test grade-2 ≥10 *and* grade-1 ≥10, and removal ≤5% of pairs: 676 terms.
  - n_prot quantiles (0/10/25/50/75/90/100%): 25/33/43/58/81/107/180.
  - Removal 0.5–5% (median 2.7%).
- **Recommended pool:** ≥40 train proteins, ≥8 test anchors with both grade 2 and grade 1, removal ≤5%. That gives **469 terms, 158 after non-nesting and protein-Jaccard ≤0.2**.
  - S_sib quartiles: 6 / 861 / 2,068 / 4,954 / 22,790. None is zero.
  - Spearman(n, S_sib) = −0.04 and Spearman(n, S_2) = 0.09.
  - Tertile strata: **low-S ≤1,323 (52 terms), high-S ≥3,662 (52 terms)**.
  - Pairwise protein-Jaccard within the set: median 0.01. Greedy search finds up to 11 mutually protein-disjoint targets.
  - Target proteins make up 0.6–2.3% of train negative-pool entries (median 1.4%). They must be removed from the pools as well.
- Examples (term, n_prot, test pos, S_sib): cell-surface RTK signalling 146/58/217; microtubule-based movement 133/43/933; NF-κB positive regulation 102/43/2,060; DNA replication 76/34/97; regulation of transmembrane transport 128/40/15,684.

**Annotation leakage [ran, `b_gaf.py`].**
- IPI and `GO:0005515` are excluded, but with/from evidence remains. In 5.92% of test positives, a kept annotation of one protein lists the partner in GAF column 8 (BP 46, MF 33, CC 50 hits), against 0.41% of weak and 0% of none. The train rate is 7.3%.
- 863 MF descendants of protein binding remain (4,738 genes carry one). BP is less exposed than MF.
- Evidence such as IDA complex membership and IGI may share source papers with STRING experimental evidence [inferred, not traced].
- The README reports that removing all guards moves the ceiling by only −0.017, so this is a moderate rather than dominant leak. Still, it must be pruned for a confirmatory study.

**Protein as an alternative unit [ran, `f_rawdeg.py`].** In raw STRING, 675 eligible proteins have ≥35 high-confidence partners. Using them as few-shot targets would require re-conversion (cap 8 positives per anchor) and a different split. Not recommended.

## F. Evaluation

[read]
- **`run_phase1_embedding_evaluation.py`** (via `scripts/eval_go_ppi.py` and `eval_learning_curve.evaluate_one`) reports pooled `auc_roc` and graded AUCs: hard = 2 vs 1, easy = 2 vs 0, 1 vs 0.
- **`scripts/evaluate_publication_fusion.py`** reports global and macro-per-query AUC (hard/easy/pooled), NDCG@10, P@10, R@10, MAP and MRR. It writes a `ranking/per_query` list for all 1,396 test anchors [ran], so per-entity results already exist.
- **Candidate pool:** fixed per query, as the test anchor's own grade 2/1/0 records. Test sha256 is `c65a1d2c…`.
- **Per-target metric:** group test queries by anchor ∈ P(t). The primary metric is hard AUC (grade 2 vs grade 1) within the target's queries.
- **Seeds:** prior runs used {13, 21, 42} and {7, 99, 123}. The decomposition plan uses {13, 21, 42, 87, 123}.
- **Checkpoint selection:** `best_checkpoint.pt` = lowest validation loss of the arm's *own* loss with its *own* selector. `fixed_validation_negatives` fixes epoch 14 and seed 1729, but GO arms still validate on GO-window negatives.
  - Best epochs range from 0 to 14 across GO runs [ran].
  - Validation is easier than test (fusion probe: validation hard AUC 0.879 vs test about 0.785 at w=0) [ran].
  - Recommendation: use the final epoch, or a variant-independent validation metric on a target-free validation set.

## G. Prior results

[ran over the result JSONs]

| source | contrast / arm | metric | result |
|---|---|---|---|
| `lc_single_factor/go_ppi_test_report.json` (InfoNCE, f100) | arm means | hard AUC | baseline 0.801, randwin 0.797, ontneg_stoch 0.798, ontneg 0.792 |
| same | arm means | pooled AUC | 0.850 / 0.847 / 0.843 / 0.836 |
| same | ontneg_stoch − randwin (n = 5) | hard AUC | **+0.0011 ± 0.0076** (4/5 wins) |
| same | ontneg_stoch − randwin (n = 5) | easy AUC | −0.0089 ± 0.0015 |
| same | ontneg_stoch − randwin (n = 5) | 1-vs-0 AUC | −0.0199 ± 0.0133 |
| same | baseline → ontneg (n = 3) | hard AUC | −0.0092 |
| same, f5 (n = 3) | ontneg_stoch − randwin | hard AUC | −0.0045 ± 0.0094 |
| `negative_selection_fixedval` (n = 3) | ontneg_stoch − randwin, f5 / f10 / f15 / f20 | hard AUC | −0.010 / 0.000 / −0.002 / +0.003 |
| `publication_phase1_fixedval` (text baseline + simGIC fusion, n = 3) | fusion − text, f10 / 25 / 50 / 75 / 100 | hard AUC | +0.023 / +0.022 / +0.021 / +0.016 / +0.016 (sd 0.001–0.003) |
| `go_decomposition_go_ordinal` | 23 trained runs, 1–2 seeds | — | no `eval_test`, **not evaluated** |

- The README reports the GO ceiling (simGIC alone) at 0.8195 hard AUC.
- Relation preservation [ran, `g_relation.py`, 20k random test-protein pairs], Spearman(cos, simGIC_BP):
  - frozen MPNet 0.223;
  - baseline 0.199–0.217;
  - randwin 0.187–0.235;
  - ontneg_stoch **0.093–0.147**.

## H. Compute

**Measured [ran, `training_results.json`].** f100 train time:
- `publication_phase1_fixedval` baseline: median 96 s;
- `lc_single_factor`: baseline 289 s, randwin 199 s, ontneg_stoch 118 s, with a wide spread of 96–647 s that reflects machine load;
- go_ordinal decomposition: about 270–380 s (one outlier at 2,592 s);
- f5–f20: 11–49 s.

**Evaluation [inferred from file mtimes].** `run_phase1_embedding_evaluation` takes about 8 min per checkpoint on 22,341 records. With frozen embeddings for the 3,014 test proteins precomputed once, as `prepared_embeddings/` already does for fusion, it drops to seconds.

**Grid estimate:**
- Design: 24 targets (12 low-S, 12 high-S, matched on n), retention {0, 1, 5, 10, 25} plus one shared "all" run, 4 cells, 5 seeds.
- Unpacked: (24 × 5 + 1) × 4 × 5 = 2,420 runs. At about 200 s each that is about 134 h.
- Packed, 3 mutually protein-disjoint, non-neighbouring targets per run (combined removal about 8%): (8 × 5 + 1) × 20 = 820 runs, about 46 h of training. Add about 109 h of evaluation unless test embeddings are cached.

## I. Reusable diagnostics

- `go_bulk_signal_diagnosis.py` / `go_independent_signal_diagnosis.py`: GO ceiling with HuRI/Lit-BM validation.
- `text_vs_ontology_ceiling.py`: whether GO adds over text (the cross-validated increment).
- `probe_fusion_mechanism.py` / `evaluate_publication_fusion.py`: score fusion with per-query output.
- `mechanism_report.py` / `ontology_ceiling_report.py`: summary tables.
- `GoPpiNegativeSelector.pool_summary`.
- The `/tmp/goinv/e_negaudit.py` and `g_relation.py` recipes for the negative audit and relation preservation.
- Missing: per-component loss and gradient logging, which `/tmp/v7inv/probe.py` from the career study shows how to do with `torch.autograd.grad`.

---

## Conclusions and recommendations

### (i) Recommended 2×2 (go_ordinal loss, BP, simGIC)

**Shared fields:** all of `config/go_decomposition_go_ordinal/randwin.json`, which includes:
- `domain_adapter=go_ppi`, `loss_type=go_ordinal`;
- `go_ordinal_lambda_high_weak=1.0`, `go_ordinal_lambda_weak_unobserved=0.1`, `go_ordinal_margin_high_weak=0.1`, `go_ordinal_margin_weak_unobserved=0.05`, `go_ordinal_rank_temperature=0.1`;
- `go_aspect=P`, `go_index_cache=preprocess/go_ppi/go_index_P.pkl`, `go_ppi_tier_window_frac=0.34`;
- `go_ppi_start/end_hard_ratio=0.2/0.6`, `max_negatives_per_anchor=7`;
- `num_epochs=15`, `batch_size=64`, `lr=8.5e-5`, `temperature=0.07`;
- `ontology_weight=0.0`, `use_ot_distance=false`, `orca_enabled=false`;
- `fixed_validation_negatives=true`;
- with `go_ppi_split_dir`/`go_ppi_converted_dir` pointing at the withholding condition's directory.

| cell | base file | `go_ppi_go_tiered_negatives` | `go_ppi_tier_sampling` | `go_similarity_mode` | new `go_ontology_loss_weight` |
|---|---|---|---|---|---|
| Baseline | randwin.json | false | random_window | simgic (ignored) | 0.0 |
| Ontology loss only | randwin.json | false | random_window | simgic | **λ** |
| Guided negatives only | go_P_simgic.json | true | stochastic | simgic | 0.0 |
| Both | go_P_simgic.json | true | stochastic | simgic | **λ** |

**The new loss term** (`go_ontology_loss_weight`, default 0 so existing configs stay byte-identical):
- Use GO **relation preservation** over in-batch training proteins: L_GO = λ · mean over in-batch protein pairs (i, j) of (cos(f(i), f(j)) − simGIC_BP(i, j))², or an equivalent rank-hinge.
- Reasoning:
  - it is the only kind of term that pulls proteins with related annotations together;
  - it acts on *all* training proteins, so neighbours of a withheld term shape the region its proteins' text maps to. That gives S(t) a mechanism;
  - it is independent of negative selection, so the two factors are separable.
- The matcher for this term must be **loss-only**. `attach_go_ppi` builds one only when tiering is on, so attaching it to `BatchProcessor` in the loss-only cell would recreate career's silent-confound bug.
- Choose λ once on target-free validation in a pilot, so that the gradient norm of L_GO is about 10–30% of InfoNCE's. Fix it before the study.
- **Alternatives considered:** `ontology_weight` (inert), ORCA (multi-component, not single-factor), a GO-input architecture (a stronger S path, but it changes the representation in every cell; keep it as an open decision). Report simGIC fusion as an *inference-time* reference arm. It uses the target protein's own annotations, so S(t) cannot moderate it by construction.

### (ii) Verdict

- **Yes, with fixes, and it is a clearly better confirmatory domain than career v7.**

| | career v7 | GO/PPI |
|---|---|---|
| split | pair-level, 373/375 test resumes in train | protein-disjoint, 0 overlap |
| usable targets | about 4–6 ESCO (or 15–19 hand-corrected clusters) | about 158 non-nested BP terms (52 low-S / 52 high-S) |
| n vs S | anti-correlated | Spearman about −0.04 |
| ontology signal | weak (ceiling about 0.56) | strong (ceiling 0.82, fusion +0.02) |
| train/eval serialization | mismatched | identical |

- **Shared weakness:** no existing loss path lets ontology neighbours reach a withheld target. Today a k=0 Δ can only come from global reshaping or text similarity, so the new GO loss term is required before the S(t) hypothesis is testable.
- **Re-split:** not needed. Keep `go_ppi_splits` and the frozen test file (sha `c65a1d2c…`). Withholding is a train-side and validation-side filter.
- **Minimum fixes:**
  - mask partner symbols in `encoder_view` and rebuild the splits from the same seed, or preregister excluding the 11% "mention" positives as a sensitivity analysis;
  - prune GAF annotations whose with/from names a protein, plus binding descendants;
  - remove target proteins from train records (both directions), `negative_pools.jsonl` (as anchors and as pool members) and selector views;
  - make validation target-free;
  - use variant-independent checkpoint selection;
  - add the GO loss term with a loss-only matcher;
  - add the per-target evaluator with cached embeddings.

### (iii) Gap list, GO/PPI (ordered by dependency)

1. **Freeze the code.** Commit or tag the uncommitted edits. They do not affect GO, but the hash should be fixed. Record the raw-file sha256s, since GOA and OBO are "current" downloads.
2. **Leakage-pruned features.**
   - A converter option that masks other proteins' gene symbols and aliases in `encoder_view` (the self prefix can stay).
   - A GAF filter that drops annotations whose with/from names a UniProt protein, and terms descending from `GO:0005515`.
   - Rebuild `go_index_P.pkl` into a *new* directory and re-measure the 11.1% and 5.9% leak rates (target: equal across grades).
3. **Target table.** Freeze the BP term list, the P(t) membership rule (propagated), N_sib/N_2, n(t) and S(t) computed on the post-fix train, and the low/high-S strata. Match pairs on n(t) and test support.
4. **Withholding generator.**
   - Per target (or pack), drop every train record with an endpoint in P(t)∖retained, in both directions.
   - Drop those proteins from every train and validation pool and from `validation_positive.jsonl` anchors and partners.
   - Retain k ∈ {0, 1, 5, 10, 25} proteins, nested and seeded, with all their pairs.
   - Write a manifest with removal fraction and hashes.
5. **Runner assertions.**
   - No withheld protein in train records, pools, views, validation or the loss-pair set.
   - No more than k retained P(t) proteins.
   - The test file hash is unchanged.
6. **GO loss term.** `go_ontology_loss_weight` in `TrainingConfig`, computed in `_compute_go_ordinal_loss` (and the InfoNCE path) over in-batch proteins, with a loss-only matcher. Add unit tests in `go_ppi_domain/tests/`. Run a λ pilot on target-free validation.
7. **Diagnostics.**
   - Per-epoch logging of InfoNCE, ranking and L_GO values, plus their gradient norms on projection parameters.
   - Negative audit per cell (`e_negaudit.py`): simGIC, grade mix, HuRI/Lit-BM false-negative rate, and the share of target-neighbour proteins.
   - Relation preservation per cell (`g_relation.py`), plus a target-specific version: Spearman over target × neighbour protein pairs.
8. **Checkpoint selection.** Use the final epoch, or a fixed validation metric (hard AUC with uniform negatives) that is identical across cells.
9. **Evaluator.** A frozen-embedding test cache. Score every checkpoint, export per-record scores, and compute per-target hard AUC (primary), macro per-query AUC/MRR (secondary), and non-target collateral AUC. Optionally add a fusion reference.
10. **Text-similarity covariate.** Frozen-MPNet cosine between target-protein texts and neighbour-protein texts, preregistered as a covariate for the S effect.
11. **Statistics.** A paired bootstrap over targets × seeds with scipy, since `statsmodels` is not installed per the v7 report. Use a mixed model if it is added.

### (iv) Open decisions for the protocol

- **Loss family:** go_ordinal (recommended; graded, matches the ordinal framing) vs InfoNCE (where all existing test evidence lives).
- **Aspect:** BP (recommended; least binding leakage, matches the ceiling) vs A (has MF coverage gaps).
- **GO loss form:** MSE relation preservation vs rank-hinge vs a GO-input architecture. Also the λ and its pilot criterion.
- **Target membership:** propagated vs direct annotation; anchor-only vs either-endpoint test queries; minimum test support (recommended ≥8 anchors with both grade 2 and grade 1).
- **Neighbourhood:** N_sib vs N_2; whether `regulates` edges count (currently dropped); S in protein-pair endpoints vs distinct proteins.
- **Retention unit:** proteins (recommended) vs pairs. Whether retained proteins keep their pairs with withheld proteins (recommended: no).
- **Packing:** pack size (2–3) with a removal cap of about 8% and no neighbouring targets in one pack.
- **Text-leak handling:** mask and re-split vs preregistered exclusion of mention positives.
- **Preregistered criteria:**
  - primary = per-target hard AUC Δ(cell − baseline) at k ∈ {0, 1, 5};
  - "helps with fewer examples" = the Δ at small k exceeds the Δ at "all" with a paired bootstrap 95% CI excluding 0;
  - S moderation = the high-S minus low-S difference in Δ for the loss-on cells;
  - an effect-size floor (for example 0.01 AUC, about the seed sd);
  - k=0 Δ in the negatives-only cell is read as global reshaping.
- **Seeds:** at least 5. Seed sd is 0.002–0.011 hard AUC.

**Not verified:**
- The relation-preservation and negative-audit numbers use 3 seeds and 800 anchors respectively, and no target-specific version was run.
- Loss and gradient scale of go_ordinal components was not measured, because components are not logged and the term carries no GO.
- With/from leakage was not broken down by evidence code. Source-paper overlap between GOA IDA and STRING experimental was not traced.
- Target counts use BP only. F and C were not computed.
- Evaluation wall time is inferred from file mtimes.
- The text-symbol leak's effect on AUC was not measured.
- The proposed GO loss term is a design and is untested.
