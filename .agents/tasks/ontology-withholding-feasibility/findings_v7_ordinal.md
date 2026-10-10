# Per-target withholding on data_splits_v7 with the ordinal loss: findings

Read-only follow-up to `findings.md`. Repo `/Users/olukotunjosh/Downloads/CDCL`, HEAD `699e0cb`, with uncommitted edits in `contrastive_learning/{batch_processor,loss_engine,trainer}.py`. Nothing in the repo was modified and nothing was trained. Scratch scripts and a copy of the text-embedding cache live in `/tmp/v7inv/`.

Evidence tags:
- **[read]**: from code.
- **[ran]**: from a read-only Python snippet over the data or existing checkpoints.
- **[inferred]**: reasoned, not directly verified.

## Summary answer

1. **The test split invalidates per-target withholding as it stands.** v7 is split by (resume, job) pair, not by resume.
   - 373 of 375 test resumes also appear in train (`DataLoader._compute_resume_id`).
   - Not one of the 200 test good_fit pairs has a resume that is absent from train.
   - For 144 of the 200, the *same resume* appears in train with a job of the *same occupation*. In 117 of those it is labelled good_fit there too. Validation looks the same (149/200 and 114/200).
   - So the "all" level and every k ≥ 1 level score largely memorised (resume, occupation) pairs. Removing those test pairs leaves essentially nothing: software developer keeps 1 of 48 test good pairs, business analyst 8 of 38.
   - **A resume-disjoint re-split is required before any per-target result means anything.**
2. **After a resume-disjoint re-split, v7 supports only a pilot.** The ESCO mapping is wrong for 4 of the 12 largest occupations (11 wrong among the 23 with ≥10 good). That leaves 8 correctly mapped targets under the current split, and these counts are still inflated by duplicate (resume, occupation) records.
   - The real unit is distinct resumes, and there are only 627 of them in v7.
   - Over 20 simulated 70/10/20 resume-disjoint splits, only software developer, business analyst and data engineer reliably keep ≥8 test good-resumes. They plus accountant are the only targets that reliably keep ≥25 train good-resumes (accountant has a median of 9 test resumes but a minimum of 5).
   - Electrical engineer, data analyst, cloud engineer, AI engineer and ICT business analyst fall to 1–7 test resumes.
   - That is about 4–6 usable ESCO targets, and S(o) is bimodal among them: BA, accountant and EE are low; SD and DE are high. **This cannot test an n(o) × S(o) moderation with paired uncertainty across targets.**
   - Normalised job-title clusters give about 19 units with ≥23 good resumes. That makes a better pilot unit, with neighbourhoods taken from the cluster's hand-corrected ESCO URI.
3. **Mapping the ordinal configs onto the 2×2.** "Ontology in loss" = the φ-adaptive L2 margin (`ordinal_fixed_m1=false`, `ordinal_phi_source="on_the_fly"`). "Guided negatives" = `ontology_guided_negatives=true` + `ontology_negative_rank_tiers=true`.
   - Baseline = `eo_d`, loss-only = `eo_a`, both = `eo_ontneg`.
   - **Negatives-only is missing.** It needs `eo_ontneg` + `ordinal_fixed_m1: true`.
   - No eo_fixed 2×2 cell changes the curriculum, grouping, OT, epochs or `ontology_weight`. `eo_c` and `eo_e` do, so keep them out.
   - `eo_ontneg_v7_fixed_rerun.json` and `eo_a_v7_frac100.json` predate the judged-floor/φ fields. Under current code their margin is a constant 0.25 and their floor is sampled negatives, so they belong to neither cell.
4. **The ontology-in-loss factor is weak and cannot act on a withheld target.**
   - φ is ESCO skill coverage of a potential_fit job by the resume. Over all 1,441 train L2 pairs, m1 = 0.5(1−φ) has mean 0.262, sd 0.051 and 5–95% range 0.175–0.338. The fixed control uses m1 = 0.30.
   - φ barely separates the labels: mean 0.48 for good jobs, 0.48 for potential, 0.455 for no_fit.
   - L2 is genuinely active under the uncommitted code: 52% of good>potential pairs still violate the margin at the epoch-12 checkpoint.
   - The whole margin term carries 13–18% of L1's gradient norm. Clipping at 1.0 always binds (L1 grad norm 13–21).
   - **Nothing in the loss relates occupation o to occupation j.** Only ISCO-guided negatives (`use_isco_negatives`) compare occupations, and they do it repulsively, in negative selection.
5. **The committed HEAD ordinal loss never exercised L2.** In HEAD, a resume's potential_fit siblings never enter the ordinal group. The uncommitted `ordinal_candidates` edit is what introduces level 1. Every EO-* result produced with committed code therefore has an inert φ-margin.
   - The EOfix-* runs (1 Oct) used the uncommitted code, confirmed by their train logs.
   - Freeze the uncommitted edits before the study.
6. **Compute is cheap. Evaluation dominates.** EOfix training wall time is 37–84 s per run for random negatives and 220–270 s for skill-guided negatives. `run_ordinal_evaluation.py` adds about 240 s per run, almost all of it re-encoding the 800 test pairs with MPNet.
   - A 5-target × 6-level × 4-variant × 5-seed grid on a re-split is 520 runs: about 53 h as is, or about 20 h if the frozen test embeddings are cached once.

---

## 1. Ordinal variant mapping

### Field-by-field diff [ran]

The comparison flattened all 10 configs. Every field below is identical across all ten unless shown:

- **Loss and training:** `loss_type=ordinal`, `ontology_weight=0.0`, `use_ot_distance=false`, `ws2_weight=0`, `num_epochs=15`, `batch_size=64`, `lr=8.5e-5`, `temperature=0.07`.
- **Ordinal parameters:** `ordinal_alpha=0.5`, `ordinal_lambda1=ordinal_lambda2=1.0`, `ordinal_m2=0.3`, `phi_gate_threshold=1.0`.
- **Negatives and model:** `use_pathway_negatives=true`, `use_view_augmentation=true`, `max_negatives_per_anchor=7`, `global_negative_pool_size=1000`, `freeze_text_encoder=true`, `projection_dim=128`.
- **Data and paths:** `validation_path=preprocess/data_splits_v7/validation.jsonl`, `skill_distance_cache_path=embedding_cache/skill_distances_v7_backup.pkl`.

| field | eo_a | eo_c | eo_d | eo_e | eo_ontneg | eo_ontneg_hard | eo_isco | eo_isco_hard | eo_ontneg_v7_fixed_rerun | eo_a_v7_frac100 |
|---|---|---|---|---|---|---|---|---|---|---|
| ordinal_fixed_m1 | false | false | **true** | false | false | false | false | false | false | false |
| ordinal_phi_source | on_the_fly | ← | ← | ← | ← | ← | ← | ← | *absent → "precomputed"* | *absent* |
| ordinal_judged_floor | true | ← | ← | ← | ← | ← | ← | ← | *absent → false* | *absent* |
| ordinal_curriculum_switch | 0.3 | **0.0** | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| group_by_resume | null (auto on) | null | null | **false** | null | null | null | null | null | null |
| ontology_guided_negatives | false | false | false | false | **true** | **true** | **true** | **true** | **true** | false |
| ontology_negative_rank_tiers | false | false | false | false | true | true | true | true | true | false |
| ontology_rank_tie_break | absent | absent | absent | absent | true | true | true | true | absent | absent |
| use_isco_negatives / isco_weight / esco_occupations_path | — | — | — | — | — | — | **true / 1.0 / occupations_en.csv** | same | — | — |
| negative hard/med/easy | .33/.34/.33 | ← | ← | ← | ← | **.6/.3/.1** | .33/.34/.33 | **.6/.3/.1** | .33/.34/.33 | ← |

None of the configs carry a train path; it is passed on the CLI. The generator (`scripts/generate_ordinal_experiments.py` L47) uses `data_splits_v7/train.jsonl`, while the EOfix runs trained on `learning_curve_v7/frac_100/train.jsonl` (per `EOfix-A__cnamuangtoun__s42/train.log`).

### Semantics [read]

**"Ontology in loss" = φ-adaptive L2 margin.** `ContrastiveLossEngine._compute_ordinal_loss` (`loss_engine.py` ~L752–978) returns `L1 + mean_q[w_q·(λ2·L3 + λ1·L2)]`.
- L1 is the per-triplet InfoNCE (`_compute_triplet_loss`, ×`_compute_ontology_weight`, which is 1.0 when `ontology_weight=0`).
- L3 is `relu(m2 − (s_i − s_j))` for any level above a level-0 floor.
- L2 is `relu(m1 − (s_good − s_pot))`, active after `ordinal_curriculum_switch × epochs`.
- How m1 is set:
  - `ordinal_fixed_m1=true` → m1 = `ordinal_m2` = 0.30;
  - `on_the_fly` → m1 = α(1 − φ(resume, potential job)) per candidate, via `_phi_for` → `_compute_phi_on_the_fly`;
  - `precomputed` → `metadata.phi`. v7 has no `phi` field (0/8000 records [ran]), so the code falls back to φ = 0.5, a **constant m1 = 0.25**.
- `w_q` = quality-tier weight (A = 1.0, C = 0.75; v7 is 95% A). It is identical across cells.
- The other in-loss ontology option, `ontology_weight > 0` (EO-B, a sample-weight multiplier on L1), was shown near-uniform in `findings.md` B1. Do not use it as the factor.

**"Ontology-guided negatives"** = `ontology_guided_negatives=true` builds the `OntologySkillMatcher` in `BatchProcessor.__init__` (~L101–160). The `_select_negatives` gate (~L941) then routes to `_select_ontology_negatives` (~L1165), which ranks pool jobs by `1 − ontology_set_similarity(resume skill URIs, job skill URIs)`. With `rank_tiers` it samples per-anchor terciles at hard/med/easy ratios.
- With `use_isco_negatives` and `isco_weight=1.0`, skill matching is skipped (`skip_skill_matching`). Distance becomes the pure `_isco_distance(anchor job occupation, candidate occupation)`, which takes the five values 0/.2/.4/.7/1.0.
- In the random-negative cells, `use_pathway_negatives=true` with no matcher falls back to `random.sample` ("skill_matcher=disabled" in the EOfix-A log).
- The uncommitted `trainer._loss_skill_matcher` builds a *loss-only* matcher for φ, so φ no longer opens the negative gate. This is the fix that keeps `eo_a` a random-negative arm.

### Mapping onto the 2×2

| | random negatives | guided negatives (skill, rank tiers) |
|---|---|---|
| fixed m1 = 0.30 | **eo_d** (baseline) | **missing**: eo_ontneg + `ordinal_fixed_m1: true` (the generator's EO-ON-D; no eo_fixed file exists) |
| φ-adaptive m1 | **eo_a** (loss only) | **eo_ontneg** (both) |

- **Clean single-factor pairs:** eo_d↔eo_a differ only in `ordinal_fixed_m1`. eo_a↔eo_ontneg differ only in the three negative fields. Neither pair touches curriculum, grouping, OT, epochs or LR.
- **Confounded configs:** eo_c (curriculum), eo_e (grouping), the `_hard` configs (ratios), and the two v7 rerun configs (constant 0.25 margin plus sampled-negative floor). Exclude them from the 2×2.
- **Margin size confound [ran]:** the φ-adaptive m1 has mean 0.262, the fixed one is 0.30. Factor A therefore mixes "φ varies per pair" with "the margin is about 13% smaller on average". A mean-matched fixed margin needs a new field, because the code uses m1 = `ordinal_m2`, and changing `ordinal_m2` also moves L3.

## 2. What the ordinal ontology term encodes, and its strength

**φ [read]:**
- Computed on the fly per (good-anchor resume, potential_fit sibling job), memoised in `_phi_cache`.
- Each job skill URI gets credit 1.0 for an exact URI match in the resume, 0.5 if within 2 hops in `dataset/esco/esco_kg.gexf` (skill–skill / hierarchy edges; distances from `skill_distances_v7_backup.pkl`), and 0 otherwise. φ is the mean credit.
- It encodes **resume↔job skill coverage only**.
- Job `skill_uris` are text-extracted, not inherited from the occupation: only 4.8% of a job's skill URIs are ESCO essential skills of its mapped occupation, and 25 of 58 occupations have more than one distinct skill set [ran].

**Distribution on v7 train [ran, `/tmp/v7inv/phi.py`]:**

| φ for | n | mean | sd | 5% | 25% | 50% | 75% | 95% |
|---|---|---|---|---|---|---|---|---|
| (r, potential job) = L2 pairs | 1441 | 0.476 | 0.102 | 0.325 | 0.417 | 0.472 | 0.526 | 0.650 |
| m1 = 0.5(1−φ) | 1441 | 0.262 | 0.051 | 0.175 | 0.237 | 0.264 | 0.292 | 0.338 |
| (r, good job) | 1600 | 0.481 | 0.090 | 0.341 | | 0.479 | | 0.635 |
| (r, no_fit job), ≤3 per resume | 982 | 0.455 | 0.095 | 0.304 | | 0.455 | | 0.600 |

80% of L2 pairs get m1 < 0.30. φ is nearly uninformative about the label, so the "ontology" in this margin is mostly a smaller, slightly jittered margin.

**Activity and scale [ran, `/tmp/v7inv/probe.py`].** The probe replays all 119 batches of one epoch through the production `process_batch` → `_generate_embeddings` → `_compute_ordinal_loss`, at epoch 10/15 (full phase), using the EOfix-A s42 config.

| | random init | best_checkpoint (epoch 12) |
|---|---|---|
| L1 mean | 2.236 | 0.413 |
| margin term (w_q·(L2+L3)) mean | 0.480 | 0.210 |
| margin/L1 | 0.21 | 0.51 |
| ‖∇L1‖ (projection params) | 21.5 | 13.1 |
| ‖∇margin‖ | 2.83 | 2.39 |
| ‖∇margin‖/‖∇L1‖ | 0.13 | 0.18 |
| L2 pairs violating φ-margin / fixed 0.30 | 97% / 99% | 52% / 59% |
| L3 pairs violating m2 | 98% | 65% |

- Per epoch there are 387 query groups, 325 of which have both good and potential jobs. 93% of good anchors carry ≥1 potential sibling and 97% carry a judged no_fit floor.
- `gradient_clip_norm` is not a `TrainingConfig` field, so `trainer.py` ~L1007 always clips at 1.0. Clipping binds on every batch, which rescales both components equally; the margin share of the update direction is therefore about 12–15%.
- The difference between the φ-margin and the fixed margin is a 7-point change in the L2 violation rate within that share. **The φ factor is a small perturbation.**
- Existing `scripts/probe_ordinal_margin.py` has a stale L2-eligibility check: it looks for `negative_original_labels`, not `ordinal_candidates`. Its "L2 possible" count is wrong under current code [read].

**Prior results [ran over `results/research_runs/EOfix-*`, 5 seeds each, test = data_splits_v7/test].** Means; sd ≈ 0.006–0.025.

| arm | AUC good_vs_rest | AUC good_vs_pot | 3-class AUC | Kendall τ |
|---|---|---|---|---|
| EOfix-D (fixed) | 0.816 | 0.758 | 0.695 | 0.421 |
| EOfix-A (φ) | 0.813 | 0.753 | 0.693 | 0.418 |
| EOfix-OntNeg (φ + neg) | 0.809 | 0.749 | 0.691 | 0.413 |
| EOfix-ISCO | 0.814 | 0.757 | 0.694 | 0.421 |

The full-data 2×2 marginals are already at noise level. Any per-target effect will be small.

**Occupation↔occupation path.** Confirmed absent from the ordinal *loss* [read]:
- L2 and L3 compare only jobs judged for the same resume;
- φ is resume↔job;
- `w_q` is a data-quality tier.

The only o↔j relation anywhere in the ordinal path is `_isco_distance` in ISCO-guided negative selection (`batch_processor.py` ~L1148, L1217). It makes same-ISCO jobs *more likely as negatives*, which pushes a resume away from neighbours of its positive's occupation.

So under every 2×2 cell, a target with zero retained records is touched by no loss term. Its scores move only through global reshaping of the shared projection and through text similarity to neighbours. That confirms the prior finding for the ordinal path.

**Negative audit [ran, `/tmp/v7inv/negaudit.py`, 41 batches, ~2,480 negatives per arm].** Columns give the share of sampled negatives that are:

| arm | same occupation as anchor's positive | same ISCO-4 | judged good for this resume | judged potential | judged no_fit |
|---|---|---|---|---|---|
| random (EOfix-A) | 8.6% | 10.4% | 2.0% | 1.0% | 4.3% |
| skill rank-tier (OntNeg) | 7.2% | 9.6% | 1.8% | 1.3% | 3.2% |
| skill hard (.6/.3/.1) | 10.2% | 13.3% | 2.5% | 1.1% | 4.7% |
| ISCO rank-tier | 6.6% | 8.6% | 1.4% | 0.7% | 3.7% |
| ISCO hard | 13.9% | 17.9% | 1.9% | 1.2% | 3.8% |

- At standard ratios, guided negatives barely change the composition of the negatives relative to random sampling.
- About 2% of all negatives are known false negatives (judged good_fit for the same resume). There is no guard against this.
- The uncommitted "drop same-resume potential siblings from negatives" filter compares dicts (`job not in ordinal_candidates`). The pool keeps the *first* record per title, so 0.7–1.3% same-resume potential jobs still slip through.

## 3. How the ordinal loss uses labels, and what "withholding" must mean

[read]
- **Grouping:** `DataLoader._load_grouped_batches` (`data_loader.py` L268–356) groups by `_compute_resume_id` (role + first 300 characters of experience + sorted skills). It packs whole groups into 64-record batches, seeds each batch with a group containing a good_fit, and shuffles groups with the seeded global RNG. This is auto-on for `loss_type="ordinal"` (`group_by_resume=None`).
- **Anchors:** only good_fit records (`label == 'positive'`; `process_batch` ~L400).
- **Siblings:** `_create_triplet` (uncommitted, ~L446–475) attaches the same resume's in-batch potential_fit jobs as `ordinal_candidates` (level 1) and its no_fit jobs as `ordinal_floor` (level 0, when `ordinal_judged_floor`). It also removes those potential jobs from the InfoNCE negatives.
- **Consequence:** a resume with no good_fit record contributes nothing except as a source of pool jobs. Its potential and no_fit records are never attached to any anchor.
- **v7 composition [ran]:** 623 resume groups, median 6 records, max 68. 386 have a good_fit, 324 have both good and potential, and 237 are all-negative.

**If only target positives are withheld:**
- A resume whose only good was on o loses its anchor, so its o-potential and o-no_fit records go silent.
- A resume with goods on other occupations keeps its o-potential and o-no_fit records as level 1/0 *below* those goods. The target job text is then actively pushed down for that resume via L2/L3.
- The target's job texts also stay in the global pool (`load_global_job_pool`, which deduplicates by title over all labels), so they keep acting as InfoNCE negatives, hard ones under guided negatives.
- Retention 0 would then not be a cold target. It would be a target that is actively ranked low.

**Implication: withholding must remove *all* records (all labels) whose `job.occupation_uri` is the target, from train *and* validation.**
- Retain k **resumes** (not records), keeping all of each retained resume's target-occupation records so its graded structure stays intact.
- Count k in resumes, because records are heavily duplicated: there are 1.76 good records per (resume, occupation) on average, 31% of pairs have more than one, and the maximum is 12. Software developer's 399 train good records come from about 70 resumes.
- Job texts map to exactly one occupation (0 of 351 map to more than one [ran]), so dropping the occupation drops its texts from the pool. The runner should still assert this per condition.

## 4. data_splits_v7 facts [ran, `/tmp/v7inv/a.py`, `b.py`, `occ.py`, `leak.py`]

- **Rows:** train 6,400 (good 1,600 / potential 1,600 / no 3,200); validation 800 (200/200/400); test 800 (200/200/400). `label`=1 iff good_fit. There is no `metadata.phi` or `resume_id`, and `quality_tier` is A 95% / C 5%.
- **frac_100 vs train.jsonl:** they are not byte-identical (sha `aeffbfa3…` vs `0208f430…`). They hold the same multiset of 6,400 records in a different order. `train_with_resume_occ.jsonl` has the same pairs plus `resume.occupation_uri`.
  - Under resume grouping the order still changes batch packing, because group insertion order feeds the seeded shuffle.
  - Use one file for every condition.
- **Distinct entities:** 623 train resumes (by `_compute_resume_id`; 639 by full JSON) and 627 across all splits. There are 351 job texts by title+description (727 by full JSON, since skill lists vary). Occupations: 58 with any record, 33 with good_fit, 56 with good+potential. Test has 52 occupations, 26 with good_fit.
- **n(o) over the 58 train occupations:**
  - good_fit quantiles (10/25/50/75/90%) are 0 / 0 / 1.5 / 23.8 / 52.4. Counts with ≥1/5/10/25/50 are 33/25/23/15/6.
  - good+potential quantiles are 1 / 6 / 16 / 32.5 / 105. Counts with ≥1/5/10/25/50 are 56/44/39/22/10.
- **Overlap, test vs train:**

| check | test vs train | validation vs train |
|---|---|---|
| resumes present in train | 373/375 | 352/355 |
| exact (resume, job) pairs | 3/800 (labels agree) | 2/800 |
| pairs whose resume appears in train with the same occupation (any label) | 479/800 | 488/800 |
| good_fit pairs: same resume + same occupation in train (any label / as good_fit) | 144/200 / 117/200 | 149/200 / 114/200 |
| good_fit pairs with a train-unseen resume | **0/200** | — |
| job texts present in train | 407/418 | 400/410 |

- **Per-occupation test counts** are in the table in §5. Test resumes have 1–10 judged jobs (178 have 1, 97 have 2, 45 have 3). 135 test resumes have a good_fit, and 93 of those also have another judged job.

## 5. Occupation mapping quality

[ran, ESCO labels from `dataset/esco/occupations_en.csv`; judgement inferred from the titles.] Columns:
- test g/p/n = test good / potential / no_fit pairs;
- "same-occ leak" = test good pairs whose resume is in train with that occupation (any label / as good);
- "train good outside test/val resumes" = train good records from resumes with no test or validation pair on o.

| ESCO label (ISCO) | train good | train g+p | test g/p/n | same-occ leak | train good outside test/val resumes | dominant titles | match mode | verdict |
|---|---|---|---|---|---|---|---|---|
| software developer (2512) | 399 | 699 | 48/46/71 | 47/47 | 41 | Senior Software Engineer, Software Engineer, Software Developer, Sr Salesforce SE | fuzzy 919 / exact 380 | correct |
| business analyst (2421) | 313 | 577 | 38/29/70 | 38/30 | 107 | Business Analyst, BA III, ServiceNow BA | fuzzy/exact | correct |
| data engineer (2511) | 176 | 336 | 23/19/42 | 23/23 | 30 | Data Engineer, Principal/Sr DE, Sr Mgr Data Architecture | mixed | correct |
| commercial director (1221) | 127 | 258 | 14/16/44 | 10/8 | — | Senior Accountant, Revenue Accountant | fuzzy | **wrong** |
| accountant (2411) | 76 | 157 | 9/10/20 | 7/4 | 41 | Accountant, Technical/Construction/Tax Accountant | mixed | correct |
| foundry manager (1219) | 72 | 103 | 7/3/16 | 2/1 | — | Software Engineering Manager, EE Manager | fuzzy | **wrong** |
| air traffic safety technician (3155) | 44 | 110 | 8/9/10 | 2/0 | — | ERP Configuration Engineer, Full Stack Engineer, Revenue Specialist | fuzzy | **wrong** |
| electrical engineer (2151) | 42 | 102 | 8/6/10 | 7/4 | 21 | Electrical Engineer (Sr/Staff/PE) | exact 123 | correct |
| sales account manager (1420) | 35 | 37 | 5/0/1 | 0/0 | — | Software Development Manager | fuzzy | **wrong** |
| architect (2161) | 31 | 43 | 3/0/2 | 0/0 | — | Salesforce Solutions Architect, Data Architect | fuzzy | **wrong** (ESCO "architect" = buildings) |
| data analyst (2511) | 26 | 63 | 5/5/12 | 4/0 | 17 | Data Analyst | exact 136 | correct |
| cloud engineer (2512) | 26 | 29 | 5/1/2 | 0/0 | 26 | Principal Cloud Software Engineer | fuzzy | correct |
| AI engineer (2511) | 25 | 26 | 5/0/1 | 0/0 | 25 | Machine Learning Engineer | exact | correct |
| footwear designer (2163) | 25 | 31 | 1/1/8 | 0/0 | — | Angular Developer, Jr Java Developer | fuzzy | **wrong** |
| university teaching assistant (2310) | 25 | 92 | 3/5/18 | 1/0 | — | Staff Accountant, Staff Engineer | fuzzy | **wrong** |
| ICT business analyst (2511) | 20 | 36 | 2/4/2 | 2/0 | 16 | Sr IT BA, IT Systems BA | fuzzy | correct |
| electronics engineer (2152) | 20 | 27 | 1/1/4 | 0/0 | — | Sr Digital Electronic Engineer | fuzzy | correct |
| design engineer (2149) | 19 | 23 | 2/1/0 | 0/0 | — | PCB Design Engineer, Product Eng. Intern | fuzzy | doubtful |
| theme park technician (7412) | 18 | 20 | 2/1/3 | — | — | Software Engineering Supervisor (D365 ERP) | fuzzy | **wrong** |
| cost analyst (2411) | 16 | 40 | 1/3/3 | — | — | Cost Accountant | exact | doubtful |
| software tester (2519) | 16 | 20 | 0/1/1 | — | — | Automation Tester | fuzzy | correct |
| research engineer (2149) | 15 | 18 | 3/0/0 | — | — | Engineering Business Analyst | fuzzy | **wrong** |
| accounting assistant (3313) | 11 | 16 | 1/2/2 | — | — | Junior Accountant | exact | doubtful |

Another 13 occupations have 0 train good but ≥10 good+potential, e.g. "animal artificial insemination technician" and "headteacher". Several are also visibly mis-mapped.

**Viable targets after excluding wrong mappings (current split):**
- **(a) train good ≥25 & test good ≥5: 8.** Software developer, business analyst, data engineer, accountant, electrical engineer, data analyst, cloud engineer, AI engineer. Commercial director, foundry manager, ATS technician and sales account manager are excluded as wrong.
- **(b) train good ≥10 & test good ≥5: the same 8.** No other occupation has ≥5 test good.
- **(c) good+potential, train ≥25 & test ≥5: 10.** (a) plus ICT business analyst and cloud DevOps engineer.

Several of these have only 1–2 distinct job texts (AI engineer 1, cloud engineer 2, ICT BA 2). Their test evaluation therefore reduces to scoring one or two texts.

**After a resume-disjoint re-split** [ran, 20 random 70/10/20 resume splits, `/tmp/v7inv/resplit.py`; values are min/median distinct good-resumes]:

| target | test | train |
|---|---|---|
| software developer | 9/17 | 53/61 |
| business analyst | 30/35 | 117/122 |
| data engineer | 9/13 | 42/48 |
| accountant | 5/9 | 29/34 |
| electrical engineer | 3/7 | 24/27 |
| data analyst | 1/7 | 17/20 |
| cloud engineer | 3/5 | 19/24 |
| AI engineer | 3/6 | 19/22 |
| ICT business analyst | 1/6 | 11/15 |

126 of the 332 good-bearing resumes are good for at least two of these targets. That correlates the targets: withholding one removes evidence another target's resumes share.

**Title clusters as the target unit [ran].** Normalising titles (drop seniority words, Roman numerals and parentheses) gives 135 clusters. 25 clusters have ≥23 distinct good resumes. Among them:
- correctly mapped: 'business analyst' 126, 'software engineer' 86, 'data engineer' 68, 'software developer' 44, 'electrical engineer' 38, 'servicenow business analyst' 36, 'construction accountant' 35, 'cloud software engineer' 34, 'machine learning engineer' 32, 'technical accountant' 32, 'data analyst' 31;
- currently mis-mapped: 'accountant' 72 (commercial director), 'software engineering manager' 64 (foundry manager), 'software development manager' 39, 'full stack engineer' 33, 'angular developer' 28, among others.

Clusters roughly double to triple the number of targets and are robust to the URI errors, provided each cluster gets a hand-checked ESCO URI.

Neighbourhoods for a cluster c, from the corrected URI:
- N0 = sibling clusters with the same URI;
- N4 / N3 = clusters whose URI shares ISCO-4 / ISCO-3;
- NJ = clusters whose URI has essential-skill Jaccard ≥ 0.2.

S(c) = Σ train good-resumes over N(c). Sibling clusters give a natural very-high-S stratum, but also the strongest text-similarity confound. Preregister the text-similarity covariate (§iii, gap 11).

## 6. Neighbourhood support on v7 (current train, S = Σ train good_fit over neighbours; number of neighbouring occupations in brackets)

[ran]

| target | n(o) good | S_ISCO4 | S_ISCO3 | S_Jacc≥0.2 |
|---|---|---|---|---|
| software developer | 399 | 32 (3) | 295 (12) | 34 (4) |
| business analyst | 313 | 0 (1) | 0 (1) | 0 (0) |
| data engineer | 176 | 71 (5) | 518 (12) | 0 (0) |
| accountant | 76 | 16 (2) | 21 (5) | 12 (4) |
| electrical engineer | 42 | 0 (0) | 20 (2) | 20 (2) |
| data analyst | 26 | 221 (5) | 668 (12) | 25 (1) |
| cloud engineer | 26 | 405 (3) | 668 (12) | 6 (1) |
| AI engineer | 25 | 222 (5) | 669 (12) | 26 (1) |
| ICT business analyst | 20 | 227 (5) | 674 (12) | 0 (1) |

- These S values are computed with the *mis-mapped* occupations included. For example, "Senior Accountant" sits in ISCO 1221, so it is not counted for accountant.
- **High-S** (ISCO-3 ≥ 295): software developer, data engineer, data analyst, cloud engineer, AI engineer, ICT BA. The whole ISCO-251x software cluster is mutually neighbouring.
- **Low-S** (ISCO-3 ≤ 21): business analyst, accountant, electrical engineer.
- n(o) and S are anti-correlated within the viable set: high-S targets are mostly the small ones. That confounds the n × S design.
- Skill-Jaccard neighbourhoods are nearly empty (0–34) for every target. They cannot carry S on v7.
- After a URI fix (Senior/Staff Accountant → accountant, Software Engineering/Development Manager → an ICT manager URI, Full Stack/Angular/Java → software developer), the low-S group shrinks further and the software cluster grows [inferred].

## 7. Evaluation for the ordinal path

[read]
- **`run_ordinal_evaluation.py` (`main` L622–749):**
  - loads every test record with an `original_label`;
  - encodes resume and job text with the *eval* serializer `content_to_text` (L76–128). This uses Skills | Experience (title at company: desc) | Education and has no role. The trainer uses Position / Skills / Profile, so the serializations do not match (see `findings.md` E4);
  - computes the dot product through the checkpoint's projection (`compute_embeddings` L130–154);
  - reports pooled tier stats, Cohen's d, pairwise ordering, Kendall/Spearman, binary AUCs (good_vs_rest, potential_vs_no, good_vs_potential), 3-class AUC/F1, and a *global* NDCG/MAP over one pooled sort (`_graduated_ranking` L391).
- Per-pair scores exist in memory (`sims_ord` follows file order, `shuffle=False`) but are not saved. Exporting them is a few-line change that joins on line index → `occupation_uri`, resume id, job hash and label.
- `scripts/orca_table4.py` only aggregates five pooled columns across seeds (AUC good_vs_rest, Spearman, Cohen's d good/no, NDCG@10, MAP_strict).
- The text encoder is re-run for every evaluation, which costs about 240 s.

**Per-target metrics that are feasible on a resume-disjoint v7 test set:**
- **(M1, primary) Per-target judged AUC:** over test pairs whose job is in o, AUC(score; good vs potential+no). This is the per-target version of the Table-4 AUC and uses only judged labels.
  - It needs ≥5 good and ≥5 non-good test pairs per target. Today only software developer, business analyst, data engineer, accountant and electrical engineer meet that.
- **(M2) Fixed-pool rank of the target's good job:** for each test resume with a good job in o, rank that job among a fixed candidate pool. Suggested pool: all distinct test job texts, 272 today, built once and frozen. Report MRR and Recall@10.
  - Unjudged candidates count as non-relevant. That bias is the same across conditions, so paired deltas remain valid.
  - Exclude same-resume judged good/potential candidates from the pool for that query.
- **(M3) Per-query graded NDCG over the resume's judged jobs:** restricted to queries that include a target job, about 93 of 135 queries today. It is informative only when a query has ≥2 judged levels.
- **(M4, collateral control):** M1 over all non-target occupations, to show a withholding condition did not degrade everything else.

## 8. Compute [ran over `results/research_runs/EOfix-*`]

| arm | training_time (s) | train wall incl. setup (s) | eval wall (s) |
|---|---|---|---|
| EOfix-D (fixed, random) | 37 | 39 | 237 |
| EOfix-A (φ, random) | 76 (one outlier at 225) | 84 | 236 |
| EOfix-OntNeg (φ, skill negatives) | 213 | 222 | 243 |
| EOfix-OntNeg-hard | 263 | 272 | 250 |
| EOfix-ISCO | 47 | 55 | 242 |

- Hardware: M2 CPU for the projection; the sentence-transformer reports MPS.
- 15 epochs is about 1,770 batches.
- **Grid (re-split, ESCO targets):** (5 targets × 5 withheld levels + 1 shared "all") × 4 cells × 5 seeds = 520 runs.
  - The mean train wall across the 2×2 is about 140 s (two random cells at about 60 s, two skill cells at about 220 s).
  - With the current eval that is about 520 × 380 s ≈ 55 h. Caching the frozen test embeddings once cuts it to about 520 × 150 s ≈ 22 h.
- **Grid (title clusters, about 15 targets):** about 1,520 runs, ≈ 63 h with cached eval. Packing 3 mutually non-neighbouring targets per run cuts this to about 21 h.

---

## Conclusions and recommendations

### (i) Recommended 2×2 for the ordinal loss

**Shared fields:** all common fields of `config/eo_fixed/eo_a.json`, plus:
- `loss_type="ordinal"`, `ontology_weight=0.0`, `use_ot_distance=false`;
- `ordinal_curriculum_switch=0.3`, `group_by_resume=null` (auto on), `num_epochs=15`;
- `ordinal_judged_floor=true`, `ordinal_alpha=0.5`, `ordinal_lambda1=ordinal_lambda2=1.0`, `ordinal_m2=0.3`, `phi_gate_threshold=1.0`;
- `negative_hard/medium/easy_ratio=0.33/0.34/0.33`, `use_isco_negatives` absent/false;
- one train file for all cells, with seeds passed by `--seed`.

| cell | file | ordinal_fixed_m1 | ordinal_phi_source | ontology_guided_negatives | ontology_negative_rank_tiers | ontology_rank_tie_break |
|---|---|---|---|---|---|---|
| Baseline (no ontology) | eo_d.json | **true** | on_the_fly (ignored) | false | false | false |
| Ontology loss only | eo_a.json | false | **on_the_fly** | false | false | false |
| Guided negatives only | **new**: eo_ontneg.json + `ordinal_fixed_m1:true` | **true** | on_the_fly (ignored) | **true** | **true** | true |
| Both | eo_ontneg.json | false | on_the_fly | true | true | true |

**Reasoning:**
- **Skill-guided negatives, not ISCO:**
  - They use the same ESCO skill graph as φ, so the 2×2 varies *where* the ontology enters, not *which* ontology.
  - They stay independent of an ISCO-defined N_k(o). ISCO negatives would make the neighbourhood definition itself the hard-negative rule, which is circular.
  - Keep EO-ISCO as an optional fifth arm if the repulsive-neighbour mechanism is of interest.
- **Margin-size control:** add one field, `ordinal_fixed_m1_value` (default `None` → `ordinal_m2`, so existing configs stay byte-identical). Set it to 0.262, the train mean of α(1−φ), in both fixed-margin cells. Otherwise factor A also changes the mean margin by 13%. If no code change is allowed, use eo_d as is and preregister that confound.
- **Excluded configs:** never use `eo_ontneg_v7_fixed_rerun.json`, `eo_a_v7_frac100.json`, eo_c, eo_e or the `_hard` variants as 2×2 cells.

### (ii) Verdict: can v7 support per-target withholding?

- **Not as split.** Every test good_fit resume is already in train, and 72% (144/200) of test good_fit pairs have the same resume paired with the same occupation in train (58%, 117/200, as good_fit). Validation, which picks `best_checkpoint.pt`, has the same structure.
- **The "all" and k ≥ 1 levels measure recall of seen (resume, occupation) pairs, not occupation learning.** Removing the leaked test pairs leaves 0–8 test good per target. Retention curves on this split are invalid.
- **A re-split is required.** The minimum is a **resume-disjoint** re-split of the 7,961 deduplicated (resume, job) pairs over the 627 resumes (`_compute_resume_id`):
  - stratified by target so each target gets ≥8 test good-resumes;
  - target-free or resume-disjoint validation;
  - test and candidate pool frozen, with hashes.
- **Usable targets after the re-split:**
  - with ESCO URIs and the mis-mappings excluded: about 4 robust (software developer, business analyst, data engineer, accountant) plus 1–2 marginal (electrical engineer, data analyst). That is a **pilot at most**, with no power for an n × S interaction, and S is confounded with n;
  - with hand-corrected title clusters: about 15–19 units with ≥23 good resumes. That is a reasonable pilot, still single-dataset and with strong text-similarity confounds.
- **Other minimum fixes:**
  - freeze the uncommitted ordinal edits;
  - remove all target records from train and validation;
  - count retention in resumes;
  - export per-pair scores;
  - cache the test embeddings;
  - use variant-independent checkpoint selection.
- **Confirmatory test:** the S(o)-moderation hypothesis also lacks a mechanism, since no ordinal term relates o to j. A null at k = 0 is expected by construction.

### (iii) Gap list, v7 + ordinal (ordered by dependency)

1. **Freeze the code.** Commit or tag the uncommitted `batch_processor`/`loss_engine`/`trainer` edits. Without them L2 never fires and φ is inert.
2. **Fix the mapping.** Hand-check the title→ESCO URI table for the 190 pool titles or 135 normalised clusters, and write the result to a new split directory. Do not edit the splits in place.
3. **Resume-disjoint re-split generator.** Group by `_compute_resume_id` and deduplicate pairs. Stratify by target (unit: ESCO URI or title cluster) with ≥8 test good-resumes per target. Use a fixed seed, emit a manifest with record hashes, and make validation target-free.
4. **Withholding generator.** For each target, remove all records (all labels) of o from train and validation. Retain k ∈ {0, 1, 5, 10, 25} **resumes** (nested, seeded) with all their o-records. Use one stable file order. Allow optional multi-target files of mutually non-neighbouring targets. Write a manifest.
5. **Runner leakage assertions.**
   - No withheld job text or title in `trainer.global_job_pool`.
   - No test resume in train.
   - Nothing beyond k retained resumes for o.
   - No job text shared across occupations.
   - Record all of this in the run manifest.
6. **Config cells.** Add `eo_ontneg_fixedm1.json`, plus the optional `ordinal_fixed_m1_value` field (default `None`) in `data_structures.py` and `_compute_ordinal_loss`.
7. **Fix the ordinal_candidates filter.** Compare content keys (`EmbeddingCache.get_content_key` or title+description hash), not dict identity, so same-resume potential jobs never become InfoNCE negatives. Optionally exclude same-resume judged good jobs as well (about 2% of negatives today). Apply the fix in every cell.
8. **Checkpoint selection.** Use the final epoch or a variant-independent validation criterion (`trainer._validate_epoch` currently uses each variant's own loss).
9. **Evaluation.**
   - Per-pair score export in `run_ordinal_evaluation.py`.
   - A new per-target evaluator for M1–M4.
   - A frozen test-embedding cache.
   - Align eval text serialization with `BatchEfficientEncoder._content_to_text`, or freeze the mismatch as a documented property.
10. **Diagnostics.**
    - Per-epoch logging of L1, L2, L3, the L2/L3 violation rates and per-component gradient norms. `/tmp/v7inv/probe.py` shows the exact recipe via `torch.autograd.grad`.
    - Update `scripts/probe_ordinal_margin.py`'s stale L2-eligibility check.
    - A negative audit per cell (`/tmp/v7inv/negaudit.py` pattern).
    - A relation-preservation probe: projected similarity of target vs neighbour job texts against ESCO distance.
11. **Text-similarity covariate.** Frozen-encoder cosine between the target's job texts and its neighbours' job texts.
12. **Statistics.** `statsmodels` is **not installed** in `.venv` [ran], so either pin it or use a paired bootstrap over targets × seeds with scipy.

### (iv) Open decisions for the protocol

- **Target unit:** ESCO URI (about 4–6 targets) or hand-corrected title cluster (about 15–19). Or use the Indian dataset for the confirmatory study and v7 for the pilot.
- **Neighbourhood definition:** ISCO-4 vs ISCO-3 vs sibling clusters. Skill-Jaccard is empty on v7. Preregister k/τ, and compute S from the *re-split baseline* train in resumes, not records.
- **Retention unit and levels:** resumes, {0, 1, 5, 10, 25, all}. Decide whether level 25 is skipped for targets with fewer than 25 train good-resumes, and whether "all" is one shared full-data run.
- **What k ≥ 1 retains:** good resumes only with their full o-graded records (recommended), or also potential-only resumes.
- **Factor A control:** a mean-matched fixed margin (requires the new field) vs eo_d's 0.30.
- **Guided-negative family:** skill (recommended) vs ISCO, and whether a false-negative guard applies to all cells.
- **Primary metric:** M1 per-target judged AUC (recommended), with M2 as secondary. Also set the minimum per-target test support and the candidate pool definition.
- **Interpretation of k = 0:** preregister that no ordinal term can touch a withheld target, so a Δ at k = 0 is read as global reshaping. Any S(o) moderation is attributed to shared resumes and text similarity, not to an ontology path.
- **Seeds and aggregation:** at least 5 seeds (per-cell seed sd is 0.006–0.025 AUC). Choose a paired bootstrap over targets × seeds, or a mixed model if statsmodels is added. Fix the effect-size threshold for "helps rare occupations" in advance.
- **Multi-target packing:** whether to pack, the non-neighbour constraint, and the cap on removed resumes per run (126 of 332 good-bearing resumes are shared across the 9 candidate ESCO targets).

**Not verified:** the gradient and violation numbers come from one seed (s42) and one random-init draw. φ statistics cover train only. The resume-disjoint counts are simulations, not a built split. Title-cluster mapping correctness was not checked cluster by cluster. EO-ISCO and OntNeg checkpoints were not probed for loss scale.
