# Validation-only search for the best GO/PPI inference-time fusion

The change adds a self-contained library (`go_ppi_fusion/`) and two scripts (`scripts/go_ppi_fusion_best/`) that look for a better way to fuse the text score with GO similarity at inference time on the 15 published GO/PPI text checkpoints. Training is untouched. A gate script first reproduces the published z(text) + z(simGIC) fusion bit-for-bit under both published weight rules. The search then scores 208 fusion variants on the published validation split only: φ variants (BP/MF/CC, simGIC/ancestor/exact/IC-direct/Resnik/is_a-only, unpruned and leak-pruned), missing-value handling, weight scope, rank/RRF/per-anchor-z forms, two-signal linear mixes and learned stackers. Each variant is scored with anchor-grouped 5-fold cross-fitting, and three finalists are frozen by a stated rule. No test data reaches selection or stacker fitting, and no test scoring was done in this phase.

Watch for: the CV folds group by anchor but are not protein-disjoint, while validation and test share no proteins. Both learned finalists (F1, F2) are HGB stackers with partner-level features, so the selection criterion likely flatters them relative to test (likely). Separately, the 208-variant adaptive search makes the finalists' CV composites optimistic as point estimates (confirmed).

**Verdict**: APPROVED

## High-level view

The study's integrity depends on the guard, and the guard holds. `guard.enter_search_mode()` runs before any other project import in `run_search.py`. From then on, `load_split` refuses any split other than validation, and every file read in the package goes through `check_path`, which blocks the published and pruned `test.jsonl`, the prepared test-embedding directory and both published fusion JSONs. `evaluate_variant`, `final_fit` and `Stacker.prepare` also refuse non-validation inputs. The search log contains no test field. The guard subprocess test and the stacker label-perturbation test pass (16/16, spot-checked once).

The published result was reproduced before searching. `reproduction.json` (21:32) shows 15/15 selected weights equal under both rules and 251,820/251,820 published test leaves bit-identical. The vectorised metrics match the published `contrast_metrics`/`ranking_metrics` on all 540 real-test leaves. The search's own Linear path also picks exactly the published weight vectors in-sample under both ndcg@10 and global pooled AUC. Metric definitions (tie-aware within-anchor hard/pooled/easy AUC, validation-fit global z applied unchanged, first maximiser = smaller w) match the published code.

Pruned and unpruned inputs are paired correctly. BP pruned φ uses the withholding study's pruned index with the aligned pruned record URIs, and `assert_aligned` checks order, anchor, partner and grade. A spot check found 0/11,174 sides where the record URIs differ from the matching index's `direct` terms, for both versions. Pruned MF/CC indexes are built in `/tmp` with the same leak-pruning dict and defaults as `build_pruned_features.py`. No protected path was modified and no commit was made.

The selection procedure is honest about test but has a validation-internal mismatch. Anchor-grouped folds share partners heavily, while the real target (test) is protein-disjoint. The finalist rule was fixed in code before the only complete run (script edited 21:34, run 21:35–22:02 per the 1,620 s runtime and `/tmp` timestamps). F3 (anchor-z, exact BP, unpruned, one weight) gives a non-learned fallback if the stackers do not transfer.

<details>
<summary>Issues (6)</summary>

1. **Fold design not protein-disjoint (NIT, likely)**: 92.8% of validation records have a partner that also appears in other folds, and 56.9% of partners are also validation anchors, yet validation and test share 0 proteins. HGB stackers can learn partner-level propensities that transfer across folds but not to test. Before test scoring, re-run the top variants (at least the stage-6 stackers and F3) with protein-disjoint folds as a sensitivity check, or state in the report that F1/F2's CV advantage may not transfer.
2. **Adaptive-search optimism (NIT, confirmed)**: stages 2–4 choose the inner metric and the φ shortlist from the same CV results, and finalists are the top of 208 variants. Report the number of variants and treat the finalists' CV composites as optimistic, with only test Δ as the estimate.
3. **Composite hides weak nDCG@10 for F2 (NIT, confirmed)**: F2's CV ΔnDCG@10 is +0.0025, below the published form's +0.0108. The equal-weight composite lets AUC/AP gains carry it. Report per-metric Δ for each finalist, not just the composite.
4. **Published-variant sanity check not asserted (NIT, confirmed)**: `run_search.py` comments that the published variant "must equal" the published curve but never asserts it. The log does match (both weight vectors identical). Add an assert so a future run fails loudly.
5. **Stale docstring (NIT, confirmed)**: `phi.py` says `PhiContext` enforces index/URI pairing. No such class exists. Pairing comes from `_terms` plus `assert_aligned`. Fix the docstring.
6. **Unit-test evidence not persisted (NIT, confirmed)**: no file records the coder's pytest run. A single spot-check run passed 16/16. Save the pytest summary next to `reproduction.json` for the record.

</details>

<details>
<summary>Details</summary>

### Guard coverage limits

The path-based guard does not cover everything a search process could touch. `reproduction.json` holds published test metrics and is not on the block list. The frozen MPNet cache, which holds test-text vectors but no labels, is loaded in search mode. Neither is read by any selection code path, so this is a hardening note only. GO IC comes from the full GAF corpus, test proteins included. That is the published convention, label-free, and does not depend on grades.

### Cross-fitting design and the partner-sharing gap

`search.evaluate_variant` prepares state (z stats, stacker fits) and selects θ on four anchor-disjoint folds, then scores the held-out fold's anchors. Every anchor gets one out-of-fold score, so 2-D grids and stackers pay for their flexibility.

```
validation (697 anchors, 5,587 records)          test (1,396 anchors)
  fold k anchors ──┐                                protein-disjoint from
  other folds  ────┼── share partners (92.8%)       all validation proteins
                   └── partners are anchors elsewhere (56.9%)
```

Anchor grouping alone does not mirror the test condition. A partner seen with its grade pattern in four folds reappears in the fifth. Stackers see partner-level features (`nb_P` counts, MF/CC missing flags, multi-aspect similarities), so HGB can learn per-partner or per-hub propensities that help out-of-fold but cannot help on test, where every protein is new (likely, not traced to feature importances). The large in-sample vs CV gap for F1 (composite +0.073 vs +0.028) shows HGB has the capacity to memorise. Cross-fold memorisation would not show up in that gap. The two learned finalists are exactly these HGB stackers, so the bias favours the variant class most able to exploit it. Test is untouched, so the coming test scoring will measure transfer honestly. The risk is a worse finalist choice, not a contaminated reported number. Fixing the folds afterwards would be selection on test, so any re-fold sensitivity has to happen before test scoring.

Stage-wise adaptivity adds to this. `best_sel` (global_pooled_auc) and the φ shortlist come from the same CV composites the finalists are later ranked by. The top-10 is dominated by stackers that differ only in feature set and C. The finalists' CV composites are best-of-208 values, and the report should not present them as estimates.

### Finalist rule and freeze

`FINALIST_RULE` was fixed in `run_search.py` (last edit 21:34) before the only complete run. That run computed the φ cache at 21:35 (`/tmp/goppi_fusion_best/search.out`) and wrote `finalists.json` at 22:02, a 1,620 s run that clears and rewrites the log at start. The rule is deterministic: F1 is the top composite. F2 is the top composite among variants whose every φ, flag and count is pruned (the log has no mixed-version variant). F3 is the top variant with at most one tuned parameter and no learned model. Ties go to fewer parameters, then log order. Results: F1 is an HGB stacker on six unpruned φ with flags and counts (CV composite +0.0282, 15/15 positive), F2 is an HGB stacker on pruned BP simGIC with pruned flags and counts (+0.0254, 15/15), and F3 is anchor-z fusion of exact-term BP overlap (+0.0184). The published form scores +0.0034 under the same CV. `finalists.json` records `test_scoring: NOT run`, and nothing in `results/go_ppi_fusion_best/` holds test-derived values except the gate's `reproduction.json`.

F2's per-metric profile differs from its composite. CV ΔnDCG@10 is +0.0025 while hard AUC is +0.037 and AP +0.036. F2 reorders mostly below the top 10. F1 (ΔnDCG +0.0148) and F3 (+0.0120) do not show this.

### φ variants and pruned/unpruned pairing

The pairing is hard-wired in `phi._terms`, not enforced by an object: no `PhiContext` exists, despite the `phi.py` docstring. There is also no runtime assert that record URIs equal the matching index's `direct` terms. That equality holds today (0/11,174 mismatched sides, both versions), but it is enforced only by how the inputs were built.

</details>

<details>
<summary>File map</summary>

`go_ppi_fusion/guard.py`: one-way search-mode test-path block. `go_ppi_fusion/data.py`: read-only port of `prepare_split` (cache-only embeddings, pruned alignment assert). `go_ppi_fusion/metrics.py`: vectorised within-anchor metrics equal to the published ones. `go_ppi_fusion/fusion.py`: fusion families (linear, linear2, coverage, rank, RRF, anchor-z, stackers). `go_ppi_fusion/phi.py`: φ variants, is_a-only closure, Resnik BMA, pruned MF/CC builds in `/tmp`. `go_ppi_fusion/search.py`: anchor-grouped cross-fitting, `final_fit`/`final_score` for the report step. `go_ppi_fusion/tests/test_fusion_best.py`: 16 unit tests. `scripts/go_ppi_fusion_best/reproduce_published.py`: reproduction gate. `scripts/go_ppi_fusion_best/run_search.py`: staged search and finalist freeze. Outputs: `results/go_ppi_fusion_best/{reproduction.json,search_log.jsonl,finalists.json}`. Full diff: `git status` shows only untracked additions (`go_ppi_fusion/`, `scripts/go_ppi_fusion_best/`) and no protected-path changes.

</details>
