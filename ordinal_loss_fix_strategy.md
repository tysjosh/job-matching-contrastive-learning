# Ordinal Contrastive Loss — Resolution Strategy

This document defines the strategy to fix the correctness, performance, and design issues in `_compute_ordinal_loss` (`contrastive_learning/loss_engine.py`). The core principle of every fix is: **all ordinal comparisons must be query-anchored** — anchored on a single resume, comparing that resume against jobs whose relevance to *that resume* is known.

---

## Root Cause

The current implementation is **candidate-anchored on the job** and pulls comparison resumes from *other triplets*. Because each triplet is `anchor = resume`, `positive = job`, `negatives = jobs`, the correct query is the **anchor resume**, and the natural graded comparison is the anchor resume against multiple jobs. The current code instead fixes a job and compares foreign resumes, and it reuses each negative's own-pair label to describe a cross-pair it does not belong to.

Everything below realigns the loss to the query-anchored structure the data already provides.

---

## Guiding Invariants (apply to all fixes)

1. **Single query per comparison**: every margin term shares one anchor resume `r`.
2. **Relevance is defined w.r.t. the anchor**: the graded level of a job must describe its relationship to `r`, not the job's origin pair.
3. **Selection metric == loss metric**: if we mine "hardest" candidates, mine them on the exact similarity used in the loss (`sim(r, job)`).
4. **Comparable scales**: margin terms and InfoNCE must be combined on a principled, documented scale.

---

## Strategy by Issue

### Fix 1 + 2 (Critical): Query-anchored graded relevance

Replace the current "collect all items, group globally, cross-pair" scheme with a **per-triplet (per-query) construction**.

For each triplet with anchor resume `r` and positive job `j⁺`:

- `s⁺ = sim(r, j⁺)` — anchor vs its positive job.
- For each negative job `jᵢ⁻`: `sᵢ⁻ = sim(r, jᵢ⁻)` — anchor vs each negative (already the InfoNCE structure).

Assign each candidate a **relevance level w.r.t. `r`** using one of two sources, in priority order:

- **Source A — true graded labels (CONFIRMED FEASIBLE, preferred).**
  The v7 data contains, for the same resume, multiple jobs at different graded levels (see "Data Findings" below). Group candidates by **resume identity** and use those true per-query labels. This is the only source that yields genuine ordinal supervision.

  ⚠️ **Do NOT group by `job_applicant_id`** — it is `null` for 100% of v7 records. Group by a stable **`resume_id`** injected during preprocessing (see Prerequisite).

- **Source B — ontology-distance proxy (fallback for single-level queries).**
  Use the already-computed anchor-to-job ontology distance (`career_distances`, produced by `_select_ontology_negatives`) to assign a graded relevance to each negative *relative to `r`*:
  - distance ≤ 0.3 → `potential_fit`-like (hard, semantically close)
  - 0.3 < distance ≤ 0.6 → weak negative
  - distance > 0.6 → clear `no_fit`

  This proxy is query-anchored by construction (it is `distance(r, job)`), which eliminates the mislabeling bug. Use it for the ~25% of resumes that appear with only one label level.

The positive `j⁺` carries its true label (`positive_original_label`), which *is* correct because it describes `(r, j⁺)`.

**Resulting margins (all anchored on `r`):**
```
L₂ = ReLU( m₁(φ) − (s⁺ − sᵖ) )      # good_fit above potential_fit
L₃ = ReLU( m₂     − (sᵖ − sⁿ) )      # potential_fit above no_fit
```
where `sᵖ` and `sⁿ` are `sim(r, ·)` for the potential- and no-fit jobs of the **same** resume `r`.

### Fix 3 (High): Align selection with the loss metric

Mine hard/clear candidates using `sim(r, job)` — the same quantity used in `L₂`/`L₃` — not job–job similarity:

- Hardest potential-fit: `argmax_p sim(r, jᵖ)` (closest wrong-but-plausible).
- Clearest no-fit: `argmin_n sim(r, jⁿ)` (most obviously irrelevant).

Because selection and loss now use the same score, mining is meaningful.

### Fix 4 (High, performance): Vectorize, remove `.item()`

Within a triplet, stack candidate job embeddings and compute all `sim(r, ·)` in one matmul:
```python
# r: [d], J: [K, d]  (all candidate jobs for this query, L2-normalized)
sims = J @ r            # [K], single kernel, stays on device
```
Do selection with `torch.argmax` / `torch.argmin` on `sims` (no `.item()`), keeping gradients on the selected entries via index (not detached). This removes per-element host↔device synchronization and collapses the nested Python loops.

### Fix 5 (Medium): Make L₃ semantics phase-consistent and documented

Pick one and document it:
- **Option A (recommended)**: keep L₃ defined as `potential_fit > no_fit` in *both* phases; in the easy phase simply skip L₂ (curriculum = "learn the easy ordinal step first"), and only fall back to `good > no_fit` when no potential-fit candidate exists for the query.
- **Option B**: explicitly document the phase-dependent meaning if the current behavior is intentional.

Update the docstring so the code and stated formula agree.

### Fix 6 (Medium): Principled scale between InfoNCE and margins

Two acceptable approaches:
- Compute margin similarities on the **same temperature-scaled** logits as InfoNCE (`sim/τ`), so margins and L₁ live in one space; scale `m₁, m₂` accordingly, **or**
- Keep raw-cosine margins but introduce a single documented weight `β` for the ordinal block: `L = L₁ + β · (λ₁L₂ + λ₂L₃)`, tuned once and reported.

Document the chosen convention and the resulting margin range.

### Fix 7 (Low): Make confidence gating effective

Set a meaningful default for `phi_gate_threshold` (e.g. 0.8) so the "φ too high → fixed margin" branch actually engages, or remove the gate if unused. Record the chosen default in config docs.

---

## Implementation Plan (ordered, low-risk first)

0. **Prerequisite — inject a stable `resume_id` during preprocessing** so triplets can be grouped by query (see Prerequisite section). Backfill existing splits or compute a content hash at load time as an interim measure.
1. **Refactor `_compute_ordinal_loss` to per-triplet, query-anchored form**; build candidate job matrix per query using `sim(r, job)`.
2. **Add per-query relevance assignment**:
   - **Source A (primary)**: within a batch, group jobs seen for the same `resume_id` and use their true labels for the ordinal margins.
   - **Source B (fallback)**: for queries with only one label level, use the `career_distances` proxy.
3. **Vectorize selection** (`J @ r`, `argmax`/`argmin`), remove `.item()` and nested loops (fix 4).
4. **Align docstring and L₃ phase behavior** (fix 5).
5. **Introduce documented ordinal scale/weight `β`** (fix 6).
6. **Set `phi_gate_threshold` default** (fix 7).

---

## Validation Plan

**Unit-level (synthetic):**
- Construct a toy query with hand-set similarities where `s⁺ > sᵖ > sⁿ` already satisfied → margins should be ~0.
- Construct violations (`sᵖ > s⁺`) → `L₂ > 0`; verify gradient sign pushes `s⁺` up / `sᵖ` down.
- Assert no `.item()` on the hot path; assert single query id across a comparison group.

**Numerical/behavioral:**
- Confirm ordinal margin is invariant to which *other* triplets share the batch (isolates the cross-query bug — result must not change when unrelated triplets are shuffled in/out).
- Log realized margin values and their scale vs L₁ each epoch.

**End-to-end (regression):**
- Re-run one small ordinal config (e.g. `results_ordinal_v6_phi_guided` settings) at 10% data, compare **ordinal triplet accuracy**, **Kendall's τ**, and **good_vs_potential / potential_vs_no Cohen's d** before vs after.
- Success criterion: fixed version improves `potential_vs_no` separation and triplet accuracy over the buggy version and over plain InfoNCE, with 5-seed mean ± std.

**Performance:**
- Measure batch time before/after vectorization; expect large reduction from removing `.item()` sync.

---

## Data Findings (v7 train split, 6,400 records)

Verified against `preprocess/data_splits_v7/train.jsonl`:

| Metric | Value |
|--------|-------|
| Records | 6,400 |
| Unique resumes (by content) | 572 |
| Avg jobs per resume | 11.2 (max 80) |
| `job_applicant_id` non-null | **0 (unusable)** |
| Resumes with ≥2 distinct labels | **428 / 572 (75%)** |
| Resumes with all 3 labels | 294 |
| Resumes with good+potential (L₂ pairs) | 305 |
| Resumes with potential+no_fit (L₃ pairs) | 361 |

**Conclusion**: Source A (genuine ordinal supervision) is **feasible** — 75% of resumes appear with jobs at ≥2 graded levels for the same query. The data supports true `good > potential > no_fit` margins; the only blocker is the missing grouping key.

## Prerequisite — inject a stable `resume_id`

The loss must group candidates per query, but `job_applicant_id` is null everywhere. Add a stable resume identifier so grouping is possible:

- **Preferred**: during preprocessing (`resume_preprocessing_new.py` / split generation), assign each unique resume a deterministic `resume_id` (e.g., a hash of normalized resume text) and write it into each record and into `metadata`.
- **Interim (no re-preprocessing)**: compute the same content hash in the data loader (`data_loader.py`) when building `TrainingSample`, and surface it in `view_metadata` so `batch_processor` and `loss_engine` can group by it.

Once `resume_id` is present:
- Resumes with ≥2 label levels → true ordinal margins (Source A).
- Resumes with 1 level → proxy or InfoNCE-only (Source B).

---

## Summary Table

| Issue | Fix | Effort | Risk |
|-------|-----|--------|------|
| 0 No grouping key | Inject stable `resume_id` (preprocess or loader) | Low | Low |
| 1 Mislabeled cross-pairs | Query-anchored relevance (Source A true labels, B fallback) | Medium | Low |
| 2 Cross-query comparison | Per-triplet construction grouped by `resume_id` | Medium | Low |
| 3 Selection≠loss metric | Mine on `sim(r, job)` | Low | Low |
| 4 `.item()` in loops | Vectorized `J @ r`, argmax/argmin | Low | Low |
| 5 L₃ phase semantics | Consistent def + docstring | Low | Low |
| 6 Scale mixing | Temperature-align or weight `β` | Low | Medium |
| 7 Gate inactive | Sensible `phi_gate_threshold` default | Trivial | Low |

Step 0 (`resume_id` injection) unblocks Source A, which the data confirms is feasible for 75% of resumes. Steps 1–3 restore correctness and speed with minimal risk; Source A delivers genuine ordinal supervision, with Source B as fallback for single-level queries.

---

## Implementation Status (applied)

The following changes were implemented and verified with `scripts/confirm_ordinal_bugs.py`.

### Confirmation of bugs (before fix)
- **Issue 2 (cross-query leakage)**: ordinal loss for a fixed query A was `0.319` alone vs `0.019` in a batch with an unrelated triplet B — a `0.30` swing from an unrelated example.
- **Root cause**: `s_n = gf.job_emb · best_nf.resume_emb` equals `s_alpha` for same-triplet negatives (both `0.900`); the negative **job** embedding never entered the L₃ term, making the margin degenerate.
- **Issue 1**: negatives carried their own-pair `original_label` (e.g. `potential_fit`), not their relation to the anchor.

### Changes made
1. **`data_loader.py`** — added `_compute_resume_id()` and inject a stable `resume_id` into each sample's `metadata` (deterministic hash of role + first experience + sorted skills). Fixes the missing grouping key (`job_applicant_id` is null in v7).
2. **`batch_processor.py`** — added matching `_compute_resume_id()` and set `view_metadata['resume_id']` (with content-hash fallback).
3. **`loss_engine.py` `_compute_ordinal_loss`** — rewritten to be **query-anchored**:
   - Group jobs per `resume_id`; anchor every comparison on that resume `r`.
   - Levels from **true labels** (positive jobs of all triplets sharing the resume — Source A) with negatives as the `no_fit` floor.
   - Similarities computed as `sim(r, job)` via a single matmul `J @ r` (fixes the degenerate `s_n`).
   - Vectorized pairwise margins with masks; **no `.item()`** in the hot path (fixes the perf issue).
   - L₃ (vs no_fit) active both phases; L₂ (good>potential) full phase only — consistent with docstring (Fix 5).
   - φ-guided `m₁` uses the query's φ with `phi_gate_threshold` fallback.

### Validation (after fix)
- Cross-query leakage: loss for A alone == loss for A in [A,B] (`0.019045`, diff `0.0`). ✅
- Non-degenerate ordinal signal: a good_fit ranked below a potential_fit for the **same** resume yields loss `0.954`. ✅
- InfoNCE path unaffected (smoke-tested). ✅

### Batch grouping guarantee (applied)
- **`data_loader.py`** — added `_load_grouped_batches()` and a `group_by_resume` flag (auto-enabled when `loss_type == "ordinal"`, also settable via `TrainingConfig.group_by_resume`). It materializes the split, groups records by `resume_id`, and packs whole resume groups into batches (chunking only oversized groups), guaranteeing graded siblings co-occur.
- **`data_structures.py`** — added `group_by_resume: bool = False` to `TrainingConfig`.
- **Verified on v7 train**: 119/119 batches contain a multi-label resume group; 581 multi-record resume groups kept intact. Default InfoNCE path is unchanged (grouping stays off).

### Not yet done (optional follow-ups)
- **Fix 6 (scale `β`)**: InfoNCE and margin terms are still summed with `λ₁/λ₂` only; a documented global ordinal weight `β` was not added to avoid changing existing configs.
- **Source B proxy**: negatives are treated as the no_fit floor rather than graded by `career_distances`; can be added if finer negative grading is desired.
- **Preprocessing backfill**: `resume_id` is injected at load time; optionally persist it into the split files.

---

## Ordinal Experiment Family (EO) — added

Analogous to the InfoNCE E-series, an **EO** ablation family isolates each design
decision of the fixed ordinal loss. 25 configs generated
(`5 variants × 1 dataset (v7) × 5 seeds`) under `results/research_runs/EO-*__cnamuangtoun__s<seed>/`.
Ordinal experiments run on **v7 only** (graded good/potential/no_fit labels); v6 and
the binary indian dataset are intentionally excluded.

| Variant | Isolates | Config delta vs EO-A |
|---------|----------|----------------------|
| **EO-A** Ordinal-Base | the query-anchored ordinal loss itself | φ-guided margins, curriculum on, grouped batching (auto), no OSCAR weighting |
| **EO-B** +OSCAR-Skill | value of sample weighting on top of ordinal | `ontology_weight=0.3`, `use_ot_distance=true` |
| **EO-C** No-Curriculum | value of the easy→full curriculum | `ordinal_curriculum_switch=0.0` |
| **EO-D** Fixed-Margin | value of φ-guided `m₁` | `ordinal_fixed_m1=true` |
| **EO-E** No-Grouping | value of resume-grouped batching (the fix) | `group_by_resume=false` |

Reference baseline for comparison: existing **E4-InfoNCE** runs.

- Dataset: `cnamuangtoun` (v7, graded) only.
- Seeds: 13, 21, 42, 87, 123.
- Generator: `scripts/generate_ordinal_experiments.py` (also writes `run_eo_experiments.sh`).
- Runner integration: EO variants registered in `run_manifests/manifest_adapter.py`
  (`VARIANT_MAP`), and evaluation uses `run_ordinal_evaluation.py`.

**Expected reads**: EO-A vs E4-InfoNCE tests whether ordinal margins help at all;
EO-E vs EO-A is the direct test of the grouped-batching fix (expected to degrade
without grouping, since graded siblings rarely co-occur); EO-C/EO-D test the
curriculum and φ-guided margin; EO-B tests stacking OSCAR weighting on ordinal.
