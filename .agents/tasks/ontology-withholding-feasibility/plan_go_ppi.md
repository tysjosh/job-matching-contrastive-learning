# Implementation Plan: GO/PPI per-target withholding study

Source of truth: `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/design_go_ppi.md` (on disk it is **v6**, APPROVED in `design_review.json`; the step prompts call it "v4") and `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/protocol_go_ppi.md` (v6). Section numbers (§n) refer to the design unless marked "protocol". This plan sequences the design and covers findings gap items 2–11 exactly as the design specifies them. It does not re-decide the design. Anything the plan adds is marked **[plan addition]** with a one-line reason.

## Conventions

- `R=/Users/olukotunjosh/Downloads/CDCL`, `PY=/Users/olukotunjosh/Downloads/CDCL/.venv/bin/python`, `S=/Users/olukotunjosh/Downloads/CDCL/scripts/go_withholding`. Every path in code, configs, commands and notes is absolute. The workspace is the repo itself (no worktree).
- FEAT artifacts live in `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/task-go-ppi-withholding/` (`task.json`, `context.json`, `features/FEAT-00{1..4}.json`). `$T` below means that directory.
- **Regression gate** (G0, run after every item that touches shared code):
  - `"$PY" -m pytest "$R/go_ppi_domain/tests" -q` (baseline: 20 passed, from `setup_baseline.md`; grows with the new tests)
  - `"$PY" -m pytest "$R/tests/test_career_ordinal_candidates.py" "$R/tests/test_mesh_soft_sampling.py" "$R/tests/test_career_adapter_equivalence.py" "$R/tests/test_career_golden_regression.py" -q` (must stay 28 passed)
  - `"$PY" "$T/check_additive.py"` (item 1) exits 0.
- There is no build step: the repo is plain Python with no pyproject or pytest.ini. "Compiles" means `"$PY" -m py_compile <files>`. Tests are the verification.
- Git: never commit, stash, reset, checkout, revert or clean. The uncommitted user edits in `$R/contrastive_learning/{batch_processor,loss_engine,trainer}.py` stay byte-for-byte. New edits there are pure insertions (§13), apart from the three `return`-line wraps in `loss_engine.compute_loss` that the design specifies. `batch_processor.py` is not edited.
- Protected, never written: `$R/preprocess/go_ppi/`, `$R/preprocess/go_ppi_splits/`, `$R/preprocess/go_ppi_lc/`, `$R/dataset/go_ppi/`.
- Environment facts checked while planning:
  - `.venv` has torch 2.8.0, numpy 2.3.2, scipy 1.16.1, sklearn 1.7.1 and sentence-transformers 5.1.0. `statsmodels` is absent and must not be added.
  - MPNet is cached in `~/.cache/huggingface/hub`.
  - The `/tmp/goinv/*.py` port sources exist. `/tmp/v7inv/probe.py` is **gone**, so implement the gradient recipe directly from §6.
  - Free disk is **7.3 GiB** (the design assumed 9.6). That is enough for the ~1.5 GB of conditions, but scratch output must be kept small and deleted.
  - `preprocess/`, `results/` and `embedding_cache/` are gitignored. `config/go_withholding/` and `scripts/go_withholding/` are not.
- Scripts under `$R/scripts/go_withholding/` get no `__init__.py`. Tests import them as `scripts.go_withholding.<name>` (an implicit namespace package; the existing `from scripts.run_go_decomposition import ...` test uses the same mechanism) and call `main(argv)` in-process. Subprocess runs use the absolute file path with `sys.executable` and `cwd=ROOT`.
- Every coder appends what it ran and the results (command, pass/fail counts, wall time) to the **Verification log** at the end of this file, and also to its FEAT file's `findings`. The reviewer judges from that evidence.

## Plan-level decisions [plan additions]

1. **The pins path can be overridden.** `withholding_constants.pins_path()` returns `Path(os.environ["GO_WITHHOLDING_PINS"])` when that variable is set, else `PINS`. `append_pin` and every pin reader go through it. Every other study path is read from `withholding_constants` attributes at call time, never bound at import, so tests can `monkeypatch.setattr` them. Reason: §14 tests assert pin-writing behaviour, and scratch runs must not pollute `$R/results/go_withholding/protocol_pins.json`.
2. **Additive-edit checker.** Item 1 snapshots the edited shared files before any edit. `check_additive.py` then proves that only the allowed edits were made. Reason: it makes the §13 "original hunks verbatim" requirement of G0 mechanical.
3. **Slow tests.** They are marked `@pytest.mark.slow` and skip unless `GO_WITHHOLDING_SLOW=1`. A new `$R/go_ppi_domain/tests/conftest.py` registers the marker. Reason: no pytest config exists, and the regular G0 run must stay fast.
4. **Adopt the three NITs from the approving v6 review** (`design_review.md`):
   - (a) Amendment test (a) uses a frozen fixture copy of protocol §13. A separate test asserts that `parse_amendment_rows(PROTOCOL)` succeeds on the real protocol.
   - (b) `parse_amendment_rows` splits cells on `(?<!\\)\|` and unescapes `\|`. Protocol §13 gains one sentence, "a literal `|` inside a cell is written `\|`", recorded in protocol §14. This is the only protocol edit in this plan.
   - (c) An eval-stage `FAILED.json` records `lc_manifest_sha256`. `rescore` refuses (exit 2, `lc_manifest_changed_after_failure`) on a mismatch.

   All three are fail-closed refinements, not design changes.
5. **Real data is touched only in scratch until review approves.** Implementation items may run real-data checks only into `/tmp/go_withholding_scratch/`, with `GO_WITHHOLDING_PINS=/tmp/go_withholding_scratch/pins.json`, and must delete that directory afterwards. Scratch notes record **counts only, never target names or GO ids**, because protocol §4 forbids inspecting target names before G2. The pinned gates G1–G4 (including the λ pilot) and the G6-style smoke run once, after review approval, in the final smoke step (items 24–26). Reason: G1 makes outputs read-only and pins them, and doing that on code that review may still change would freeze artifacts from unreviewed code.
6. **Protocol stops are user decisions.** If a run hits a protocol stop, do not change a parameter: call `send_message` with severity `warning` and wait. The stops are: leak gate fails, fewer than 10 matched pairs, packing infeasible at 20 targets (since AM1: a selected target failing the single-target removal cap or retention feasibility), fewer than 4 pilot batches, λ not confirmed.
7. **G5 is not run by this workflow.** `sweep.py pin` waits for the user's commit/tag decision (protocol G5). The G6-style smoke runs the runner directly, which does not need `code_pin.json`.

## Dependency overview

| items | depend on | independent of |
|---|---|---|
| 1 (snapshots, frozen selector fixture) | — | must run before any edit |
| 2 (constants) | 1 | — (everything imports it) |
| 3–8 (FEAT-001: config fields, relation table, loss, trainer hook, selector flag, diagnostics, attach) | 2 | 9–13 |
| 9–13 (FEAT-002: GAF filter, leak pruning, rebuild + leak audit, target table, generator) | 2 | 3–8 |
| 14–18 (FEAT-003: runner study path, configs, λ pilot, negative audit, slow integration tests) | FEAT-001 and FEAT-002 | 19–22 |
| 19–22 (FEAT-004: evaluator, covariate, statistics, sweep) | 2, 12 (table schema), 14 (lc_manifest schema) | 15–18 |
| 23 (cross-feature integration) | 1–22 | — |
| 24–26 (real gates G1–G4 and the G6 smoke; the smoke step) | 23 and review APPROVED | — |

FEAT-001 and FEAT-002 are mutually independent; FEAT-001 is ordered first only because FEAT-003 needs both.

---

## FEAT-001: training-path primitives (gap items 6, 7a; parts of 5)

- [x] 1. Snapshot the pre-edit state and freeze the selector's current output.
      - Copy `$R/contrastive_learning/{batch_processor,loss_engine,trainer,data_structures}.py` and `$R/go_ppi_domain/{negative_selector,go_ontology}.py` to `$T/pre_edit/`.
      - Save `git diff HEAD -- <file>` for the three user files as `$T/pre_edit/<file>.diff`, and save `git diff --stat`. Use read-only git commands only.
      - Planning recorded these diff sha256s for reference: batch_processor `15353d4a…`, loss_engine `82d81de9…`, trainer `b7867e14…`.
      - Write `$T/pre_edit/protected_sha256.txt`: `shasum -a 256` of every file under `$R/preprocess/go_ppi/` and `$R/preprocess/go_ppi_splits/`. Item 23 compares against it.
      - Write `$T/check_additive.py`. It compares each pre-edit copy to the current file with `difflib.SequenceMatcher` over lines and passes only if all of these hold:
        - `batch_processor.py` is byte-identical;
        - `trainer.py`, `data_structures.py` and `go_ontology.py` show only `equal`/`insert` opcodes;
        - in `negative_selector.py`, the only `replace` opcodes are the two `if self.tier_sampling == "random_window":` → `if use_random_window:` lines plus the `__init__`/`from_config`/`logger.info` lines that gain the new keyword (§5.6);
        - in `loss_engine.py`, the only `replace` opcodes are exactly three single-line wraps of `compute_loss` return statements (§5.3);
        - every `+` line of each saved `.diff` is still present in the current file, in order.
      - **Before editing the selector**, build the item-6 synthetic fixture in `$R/go_ppi_domain/tests/_withholding_fixtures.py`: a tiny pickled `GoIndex`, a split dir and a converted dir in `tmp_path`, with anchor `U` unannotated and anchor `A` annotated. Record the **unmodified** neg_only selector's `select_batch_negatives` output (partner ids, grades, distances) for seeds 13 and 42 × epochs 0..14 into `$R/go_ppi_domain/tests/fixtures/selector_flag_false_expected.json`.
      Files: `$T/pre_edit/*`, `$T/check_additive.py`, `$R/go_ppi_domain/tests/_withholding_fixtures.py`, `$R/go_ppi_domain/tests/fixtures/selector_flag_false_expected.json`
      Verify: `"$PY" "$T/check_additive.py"` exits 0 on the untouched tree, and the regression gate still gives 20 + 28 passed.

- [x] 2. Create the constants module (§1.3, §13).
      - Constants: `ROOT` (asserted equal to the literal path), `RAW_SHA256` (the 8 files in `setup_baseline.md`), all study paths, `PROTECTED`, the seeds, `KS`, `SEEDS`, `CELLS`, `GRID_EPOCHS`, `RUN_DIR_IGNORE`, `PROTOCOL`, `AMENDMENT_ID_RE`, `AMENDMENT_COLUMNS`, `CODE_PIN_PATHS`.
      - Functions: `assert_writable_target`, `sha256_file`, `structural_sha256`, `code_sha256()` (`git ls-files -co --exclude-standard` with `cwd=ROOT`), `imported_repo_modules()` (skips modules with a relative `__file__` and anything under `.venv`), `pins_path()` and `append_pin` (decision 1), `group_ok_valid`.
      - `parse_amendment_rows`: reads the span between `## 13.` and `## 14.`, matches the table by its header, and splits cells on `(?<!\\)\|` with unescaping (decision 4b).
      - Add `$R/go_ppi_domain/tests/conftest.py` to register `slow` (decision 3).
      - Write `test_code_pin.py` with the §14 cases: an untracked `.py` file under `go_ppi_domain/` changes the hash; a `trials_domain/` file is in the list; `imported_repo_modules` filters correctly. Add parser cases: `\|` inside a cell; the real protocol parses with 7 cells per row.
      Files: `$R/go_ppi_domain/withholding_constants.py`, `$R/go_ppi_domain/tests/conftest.py`, `$R/go_ppi_domain/tests/test_code_pin.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_code_pin.py" -q` passes.

- [x] 3. Add the two `TrainingConfig` fields and the loss-only relation table (§5.1, §5.2).
      - Add `go_ontology_loss_weight: float = 0.0` next to the `go_ordinal_*` fields (`data_structures.py` ~L237–241).
      - Add `go_ppi_unannotated_anchor_window: bool = False` next to `go_ppi_tier_window_frac` (~L605).
      - New `go_relation.py`:
        - `GoRelationTable`: `from_train_file` (loads the index via `GoIndex.load` only), `from_index`, vectorised sparse simGIC, `annotated`, `submatrix`, `sha256`. It deliberately has no `ontology_set_similarity`, `terms_for_gene` or `branch_distance`.
        - Pure `go_relation_loss`: anchor and positive only, deduplicated, missing ids skipped, at least 2 annotated proteins required, returns the upper-triangle mean of `(C − S)²`.
      - Write `test_go_relation_table.py` with the §14 cases: equal to `GoMatcher.ontology_set_similarity` on closures (atol 1e-6), protein set, no matcher API, purity.
      Files: `$R/contrastive_learning/data_structures.py`, `$R/go_ppi_domain/go_relation.py`, `$R/go_ppi_domain/tests/test_go_relation_table.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_go_relation_table.py" "$R/go_ppi_domain/tests/test_go_ppi_wiring.py" -q` passes. The wiring test proves `from_json` still round-trips the existing configs.

- [x] 4. Add L_GO to the loss engine, additively (§5.3). Depends on 3.
      - Insert after the `go_ordinal_*` validation block (`loss_engine.py` ~L86, before the user's hunk at L99): weight parsing and validation, `go_relation_table`, the counters, `_diagnostic_mode`.
      - New methods:
        - `set_go_relation_table`;
        - `_go_relation_loss`, which updates counters only when `torch.is_grad_enabled() and not self._diagnostic_mode`;
        - `_with_go_relation`, which returns the same object when the weight is 0;
        - `go_ordinal_components`, which re-implements the `_compute_go_ordinal_loss` loop and never touches counters.
      - Wrap exactly three `return` lines in `compute_loss`: the go_ordinal dispatch, `return losses[0]`, and the InfoNCE-tail `return torch.stack(losses).mean()`. The zero-tensor early return and `_compute_go_ordinal_loss` stay unchanged.
      - Write `test_go_withholding_loss.py` with every §14 case, reusing the `setup_loss` pattern from `test_go_ordinal_loss.py`. It must include **weight 0 → `torch.equal` loss and identical gradients versus an engine with the field absent**, for go_ordinal and InfoNCE, with and without a table. That test is the required proof that the default reproduces current behaviour.
      Files: `$R/contrastive_learning/loss_engine.py`, `$R/go_ppi_domain/tests/test_go_withholding_loss.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_go_withholding_loss.py" "$R/go_ppi_domain/tests/test_go_ordinal_loss.py" -q` passes, `check_additive.py` passes, and the career suite stays at 28 passed.

- [x] 5. Add the epoch-end callback hook to the trainer (§5.5).
      Insert the 4-line `epoch_stats` dict and the `for callback in getattr(self, 'epoch_end_callbacks', ()):` loop immediately before `return avg_epoch_loss` in `_train_epoch` (`trainer.py` ~L821, after the diagnostic-integration block). Make no `__init__` change.
      Files: `$R/contrastive_learning/trainer.py`
      Verify: `check_additive.py` passes and the regression gate gives 20 + 28 passed. Callback behaviour is tested in item 7.

- [x] 6. Add the unannotated-anchor window flag to the selector (§5.6). Depends on item 1's frozen fixture.
      - `__init__` gets a trailing keyword `unannotated_anchor_window: bool = False`, also shown in the `logger.info` line.
      - `from_config` passes `getattr(config, "go_ppi_unannotated_anchor_window", False)`.
      - In `select_batch_negatives`, compute `use_random_window` once, after `anchor_uris` and after the `if self.matcher:` block, and use it to replace both `if self.tier_sampling == "random_window":` tests.
      - Write the selector cases of `test_go_withholding_negatives.py`:
        - flag true on `U` equals baseline for seeds 13/42 × epochs 0..14, with all distances `NEUTRAL_DISTANCE`;
        - flag true on `A` differs in at least one epoch;
        - flag false on `U` equals `selector_flag_false_expected.json`;
        - the `from_config` default is False.
      Files: `$R/go_ppi_domain/negative_selector.py`, `$R/go_ppi_domain/tests/test_go_withholding_negatives.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_go_withholding_negatives.py" "$R/go_ppi_domain/tests/test_go_decomposition.py" -q` passes, and `check_additive.py` passes.

- [x] 7. Create the diagnostics module (§4.5, §4.6, §6). Depends on 3–6.
      - `preserved_rng`.
      - `projection_params`: trainable parameters only, none from `trainer.text_encoder`, fatal if empty.
      - `StrictEpochGuard`.
      - `SelectionRecorder`: wraps the selector via an instance attribute; has a `paused` context; counts exceptions even while paused. Attribute names are exactly `by_epoch`, `calls_by_epoch`, `unannotated_anchor_calls_by_epoch`, `none_returns`, `exceptions`.
      - `EpochDiagnostics`:
        - uses the fixed first 8×64 train lines via `GoPpiRecordAdapter`;
        - holds its own relation table in every cell;
        - calls `process_batch(batch, None, None)` and `go_ordinal_components`;
        - takes `torch.autograd.grad` per component on the projection params;
        - runs the drift check and writes the §6 JSONL row schema;
        - is non-fatal except for the paused-selector-exception rule;
        - restores everything in `finally`.
      - Write `test_epoch_guard_and_recorder.py` with every §14 case, including a minimal trainer subclass whose `_train_batch` raises inside the batch `try`, so that `train()` re-raises the guard's `RuntimeError`. Also cover recorder hash order-independence, `None`/exception counting, and the absence of an `unannotated_calls_by_epoch` attribute.
      Files: `$R/go_ppi_domain/withholding_diagnostics.py`, `$R/go_ppi_domain/tests/test_epoch_guard_and_recorder.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_epoch_guard_and_recorder.py" -q` passes.

- [x] 8. Add `attach_go_relation_loss`, the factor field, and the §5.4 unit identity test. Depends on 3–7.
      - In `$R/scripts/run_learning_curve_point.py`, add `"go_ontology_loss_weight"` to `EXPECTED_FACTOR_FIELDS["go_ppi"]`.
      - Add `attach_go_relation_loss(trainer, config, train_file)` (§4.3):
        - weight 0 attaches nothing;
        - weight > 0 requires `domain_adapter == "go_ppi"`, builds `GoRelationTable.from_train_file`, and calls only `trainer.loss_engine.set_go_relation_table`;
        - it never calls `set_ontology_matcher`.
      - Make no CLI change yet.
      - Add `test_loss_only_selects_identical_negatives` to `test_go_withholding_negatives.py`:
        - synthetic split dir (60 anchors, mixed-grade pools, synthetic pickled index);
        - two configs that differ only in weight (0 vs 0.3);
        - a stub trainer with a real `ContrastiveLossEngine` and a recording batch-processor stub, wired by `attach_go_ppi` + `attach_go_relation_loss`.
        - Assert `set_ontology_matcher` is never called, `selector.matcher is None`, and the `(partner_id, grade)` lists are identical for seed 13 × epochs 0..14 for every anchor.
        - Assert the same identity for neg_only vs both.
      - Add the `SelectionRecorder(..., unannotated_anchors={"U"})` count case.
      Files: `$R/scripts/run_learning_curve_point.py`, `$R/go_ppi_domain/tests/test_go_withholding_negatives.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_go_withholding_negatives.py" -q` passes, and the full regression gate passes.

## FEAT-002: leak-pruned data, target table, withholding generator (gap items 2, 3, 4)

- [x] 9. Add the additive GAF filter and leak-pruning options to `go_ontology.py` (§1.2).
      - Extend `parse_gaf` with the keyword-only `with_from_symbol_map`, `drop_with_from_human_protein`, `excluded_term_roots` and `descendants_of`, plus the drop counters `drop_with_from_human_protein` and `drop_excluded_root_descendant`. Self-references are kept, and the isoform suffix is stripped.
      - Extend `GoIndex.build` with the keyword-only `leak_pruning` and `with_from_symbol_map`, appended after the existing parameters. It raises `ValueError` when pruning is requested without a map. Descendants come from the full all-namespace `parse_obo` parents. It stores `meta["leak_pruning"]`, and the key is absent when `leak_pruning=None`.
      - Leave `build_or_load` untouched.
      - Start `test_leak_pruning.py` with:
        - `parse_gaf` defaults identical to current output on a fixture GAF;
        - with/from filter, including a kept self-reference;
        - excluded-root descendant drop;
        - the `ValueError`.
      Files: `$R/go_ppi_domain/go_ontology.py`, `$R/go_ppi_domain/tests/test_leak_pruning.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_leak_pruning.py" "$R/go_ppi_domain/tests/test_go_decomposition.py" -q` passes, and `check_additive.py` confirms `go_ontology.py` has only insertions.

- [x] 10. Create the leak-pruning module (§1.1).
      - Constants `ALIAS_SOURCES`, `MASK_TOKEN="[GENE]"`, `VIEW_SEP=" — "`; the `MaskLexicon` dataclass.
      - Functions `load_symbol_aliases`, `build_mask_lexicon` (`min_alias_len` 3, `min_symbol_len` 2, `max_doc_freq` 0.01), `mask_view_text` (two passes, case-sensitive whole-word, the owner keeps its own entries), `human_accession_map`, `other_db_symbol_map`, `rewrite_proteins` (same ids and order, prefix kept and asserted, key order unchanged, `json.dumps(row, ensure_ascii=False) + "\n"`). Port the logic from `/tmp/goinv/a_split.py` and `/tmp/goinv/b_gaf.py`.
      - Complete `test_leak_pruning.py` with the §14 cases:
        - the self symbol is kept and the partner is masked;
        - word boundaries ("TP53-binding" is masked, "TP530" is not);
        - the doc-frequency filter;
        - `rewrite_proteins` preserves ids and order, keeps the prefix, recomputes `go_uris`/`coarse_uris`/`n_terms`, and raises when the prefix is missing.
      Files: `$R/go_ppi_domain/leak_pruning.py`, `$R/go_ppi_domain/tests/test_leak_pruning.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_leak_pruning.py" -q` passes.

- [x] 11. Create `build_pruned_features.py` and `measure_leakage.py` (§1.4, §1.5). Depends on 9, 10.
      - `build_pruned_features.py` follows the §1.4 steps 1–9 exactly:
        - refuses protected or non-empty output dirs (no `--force`);
        - checks the raw shas;
        - builds the in-memory pruned index and saves it;
        - writes the mask lexicon and `mask_manifest.json`;
        - copies `pairs.jsonl` verbatim, then calls `rewrite_proteins` and writes `convert_manifest.json`;
        - asserts the pairs sha and the id order;
        - calls `data_splitter.build_splits(..., seed=42, val_no_interaction_cap=20, test_no_interaction_cap=0)`;
        - asserts structural shas, pool rows and counts (15,708 / 5,587 / 22,341 / 7,041);
        - writes `frozen_hashes.json`, applies `chmod 0o444`, and calls `append_pin("G1", ...)`.
        - Exit 2 on any failure.
      - `measure_leakage.py` computes metrics (a)–(d) per split and grade for the reference and new data, applies the acceptance rule (fatal exit 3), writes `leak_audit.json`, and calls `append_pin("G1-leak", ...)`.
      - Tests: a fixture-sized end-to-end run of both scripts in `tmp_path` with `GO_WITHHOLDING_PINS` redirected and the constants monkeypatched. Cover: a refusal on a non-empty or protected output; a structural mismatch → exit 2; a leak-gate failure → exit 3.
      - **Scratch real-data check** (decision 5): run both scripts with outputs under `/tmp/go_withholding_scratch/` and record wall time.
        - Record the reference leak rates. These must reproduce test (a) 11.07% / 0.17% / 0% and (c) 5.92% / 0.41% / 0%.
        - Record the new-data rates, the dropped GAF rows (expected ≈ 12,871 of 138,976), and the empty-`go_uris` count (≈ 1,187).
        - If the gate fails, stop and ask (decision 6).
        - Delete the scratch dir afterwards. `chmod`ed files delete fine with `rm -rf`.
      Files: `$S/build_pruned_features.py`, `$S/measure_leakage.py`, `$R/go_ppi_domain/tests/test_build_pruned_features.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_build_pruned_features.py" "$R/go_ppi_domain/tests/test_leak_pruning.py" -q` passes; the scratch run exits 0 for both scripts, and its numbers are in the Verification log.

- [x] 12. Create the target-table module and script (§2).
      - **AM1 (2026-10-10, user decision, protocol §13 row AM1, design §22):** packs are singletons (one target per condition). The packing, slot-cycling, 24→20 fallback and `pack_search_log` bullets below are superseded; `singleton_packs` checks the removal cap and retention feasibility per target.
      - Implement every function listed in §2.1, with the definitions of §2.1 and the candidate rule of §2.2 (port `/tmp/goinv/c_targets.py` and `/tmp/goinv/d_refine.py`).
      - Strata as in §2.3.
      - Matched selection as in §2.4, using `linear_sum_assignment`, the exact size rule, and the replacement queue.
      - Neighbour sets as in §2.5.
      - Packing as in §2.6. It is a deterministic search with slot cycling and the 24→20 fallback, and it writes exactly the specified `pack_search_log` entries.
      - Retention order as in §2.7: `random.Random(f"20251009|{go_id}")` on `sorted(eligible)`, kept as a list.
      - `build_target_table.py` (§2.8) does the hash and G1 preflight, refuses to overwrite, writes `target_table.json` (schema_version 2), its `.sha256`, and `candidates.json`, and calls `append_pin("G2-targets", ...)`. Exit 3 on < 10 pairs or on infeasibility at 20.
      - Write `test_target_table.py` with every §14 case, including the exact `pack_search_log` sequence `(24: m12→m13, 24: m13→m14, 20: m10→m11, 20: m11→m12, 20: m12→m13, 20: m13→m14)` and the Jaccard-skip fixture.
      - Scratch real-data check on the item-11 scratch outputs (rebuild them if already deleted). Record **counts only**: `n_candidates`, `n_non_nested`, the strata cuts, `n_targets`/`n_packs`/`shortfall`, the SMDs and the wall time. Never record names or GO ids.
      Files: `$R/go_ppi_domain/withholding_targets.py`, `$S/build_target_table.py`, `$R/go_ppi_domain/tests/test_target_table.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_target_table.py" -q` passes; the scratch run exits 0 and its counts are in the Verification log.

- [x] 13. Create the withholding generator and `make_conditions.py` (§3).
      - **AM1:** the count check uses singleton-pack tables (24 targets → 121 conditions, 20 → 101) instead of the 8-pack / 7-pack tables; "first pack" below is the first single-target pack `p01`.
      - `withholding.py`: `withheld_set`, `filter_train_records` (both directions), `filter_pools`, `drop_empty_pool_anchors`, `build_targetfree_validation`, `build_condition`, and `assert_condition` (all ten §4.3 checks; each raises `RuntimeError` naming the first 5 ids).
      - `make_conditions.py --mode validation|conditions`:
        - writes the §3.2 and §3.3 files and the schema-2 `withholding_manifest.json` with `unannotated_anchors`;
        - applies every §3.4 rule: exactly-k, residual scans, regeneration that is idempotent or fatal, retained proteins still in the pools, ≥ 3 GB free disk, and the unannotated-ids agreement with the pruned index;
        - asserts the written count equals `n_conditions`;
        - calls `append_pin("G2-validation" | "G3", ...)`.
      - Write `test_withholding_generator.py` with every §14 case. It includes the pool-only protein at k=0 and k=25; the case where R_5 stays in the pools while W is gone; and the count check on an 8-pack and a 7-pack (6×3 + 1×2) table.
      - Scratch check: build the target-free validation set plus `all` and the first pack's k ∈ {0, 5} via in-process `build_condition`. Record `removal_frac`, `withheld_count`, `pool_member_share_removed` and the unannotated counts. Delete afterwards.
      Files: `$R/go_ppi_domain/withholding.py`, `$S/make_conditions.py`, `$R/go_ppi_domain/tests/test_withholding_generator.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_withholding_generator.py" -q` passes; the full regression gate passes.

## FEAT-003: runner study path, configs, λ pilot, negative audit (gap items 5, 6 pilot, 7b, 8)

Depends on FEAT-001 and FEAT-002.

- [x] 14. Add the runner's study path (§4.1–§4.4, §4.7, §8).
      - New CLI args `--go-ppi-split-dir`, `--withholding-manifest`, `--study-cell`, `--go-diagnostics`, `--pilot`, with the §4.1 requirement and exclusion rules enforced as argparse errors.
      - The §4.2 preflight runs inside one `try`; on failure it writes `run_error.json` and exits 2. It covers: the absolute-path scope (JSON keys plus CLI overrides only), the index sha and meta, no validation path, the λ equality (skipped only under `--pilot`), and `code_sha256`.
      - Wire the §4.3 order exactly.
      - `assert_post_training` (§4.4) covers seen ids ∩ W, missing ids, recorder counters, import coverage, the final checkpoint and its epoch, and no `*_emergency.pt`.
      - Add the `lc_manifest.json` blocks of §4.7 with the stated sources, including `"pilot"`, `epoch_stats`, `selected_checkpoint` and its sha.
      - With `--withholding-manifest` absent the code path is unchanged.
      - Write `test_runner_withholding_assertions.py` with every §14 case. It uses stub trainers and doctored manifests, and covers: a relative `_path` key in the JSON rejected while absent `TrainingConfig` defaults are ignored; an index sha mismatch rejected before any matcher is built; `--pilot` rules; every §4.3 check firing.
      Files: `$R/scripts/run_learning_curve_point.py`, `$R/go_ppi_domain/tests/test_runner_withholding_assertions.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_runner_withholding_assertions.py" "$R/go_ppi_domain/tests/test_go_ppi_wiring.py" -q` passes; `"$PY" "$R/scripts/run_learning_curve_point.py" --help` lists the new args.

- [x] 15. Create `materialize_configs.py` (§12.1).
      - Writes the four configs to `$R/config/go_withholding/`: baseline and loss_only from `randwin.json`, neg_only and both from `go_P_simgic.json`.
      - Applies the §12.1 overrides with absolute paths, sets `go_ppi_unannotated_anchor_window: true` in all four, and removes `validation_path`.
      - Asserts that every `_path`/`_dir`/`_cache` value is absolute.
      - Runs the `EXPECTED_PAIR_DIFF` audit in `.staging/` before writing anything.
      - `--lambda none` writes baseline, neg_only and `loss_pilot_template.json` only, with no pin and no `configs.sha256.json`.
      - `--lambda <float>` writes all four plus `configs.sha256.json` and appends `G4-configs`.
      - Write `test_materialize_configs.py` with the §14 cases, using a monkeypatched `CONFIG_DIR` and redirected pins.
      Files: `$S/materialize_configs.py`, `$R/go_ppi_domain/tests/test_materialize_configs.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_materialize_configs.py" -q` passes. Do not run it against the real `$R/config/go_withholding/` in this item; that is G4, item 25.

- [x] 16. Create `lambda_pilot.py` (§12.2).
      - Pilot runner subprocess: condition `all`, λ₀ = 0.1, seed 42, 3 epochs, `--pilot`.
      - In-process target-free validation measurement:
        - load `checkpoint_epoch_2.pt` strictly;
        - set the batch processor and loss engine to epoch 2 and assert it;
        - use batches of 32 with the last partial batch dropped; fewer than 4 batches → exit 3;
        - build the relation table over the validation proteins;
        - compute the §6 gradient recipe, and take r₀ as the median ratio.
      - Choose λ* by log-space rounding to {1, 2, 3, 5}×10ⁿ, with one confirmation run and one re-estimate, otherwise exit 3.
      - Write `lambda_choice.json`, call `append_pin("G4-lambda", ...)`, then call `materialize_configs.py --lambda <λ>`.
      - Unit-test the pure parts (rounding, the confirm/re-estimate state machine, the < 4 batches exit 3, the epoch-setting assertion) with the subprocess and the measurement stubbed.
      Files: `$S/lambda_pilot.py`, `$R/go_ppi_domain/tests/test_lambda_pilot.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_lambda_pilot.py" -q` passes. The real pilot runs at G4 (item 25).

- [x] 17. Create `audit_negatives.py` (§7).
      - Index preflight.
      - Per cell: `cfg.training_seed = args.seed`, assert the selector seed, and use the production selector via `from_config`.
      - Separate measurement matcher; HuRI/Lit-BM mapped via STRING `Ensembl_gene` aliases. Port `/tmp/goinv/e_negaudit.py`.
      - Write all the §7 shares, including `share_withheld_negatives` (gated at exactly 0), the share invariant, and the offline identities baseline == loss_only and neg_only == both (exit 2 on any failure).
      - Output `$R/results/go_withholding/audits/neg__<cond>__<cell>__s<seed>.json`.
      - Write `test_negative_audit.py` with the §14 cases: the seed-13 override over a config carrying 42; the gate tripping on a doctored pool; a retained protein at k=5 only raising the descriptive shares.
      Files: `$S/audit_negatives.py`, `$R/go_ppi_domain/tests/test_negative_audit.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_negative_audit.py" -q` passes.

- [x] 18. Write the slow integration tests (§5.4, §6, §14). They use real MPNet and a 256-record slice of a synthetic-but-real-format condition built by `make_conditions` on a fixture.
      - (a) Real `ContrastiveLearningTrainer`, 1 epoch: baseline and loss_only `selection_sha256_by_epoch` are equal; neg_only and both are equal; baseline and neg_only differ; all counters are 0.
      - (b) Diagnostics non-interference: identical epoch loss, `state_dict` and selection hash with and without the callback.
      - (c) A forced L_GO failure (table detached mid-run) → runner exit 2 and `run_error.json`.
      Files: `$R/go_ppi_domain/tests/test_withholding_integration_slow.py`
      Verify: `GO_WITHHOLDING_SLOW=1 "$PY" -m pytest "$R/go_ppi_domain/tests/test_withholding_integration_slow.py" -q` passes (expect a few minutes on CPU). The regular run reports these tests as skipped.

## FEAT-004: evaluator, covariate, statistics, sweep driver (gap items 7c, 9, 10, 11)

Depends on items 2, 12 and 14. It is independent of items 15–18.

- [x] 19. Create `evaluate_withholding.py` (§9).
      - `build-cache`:
        - checks the test sha;
        - builds texts with `content_to_text`;
        - sets `max_seq_length = 512`;
        - asserts `max_tok < 512` (exit 2 otherwise);
        - writes the npz and meta;
        - calls `append_pin("G2-cache", ...)`.
      - `score`:
        - resolves the checkpoint and sha from `lc_manifest.json`;
        - loads `CareerAwareContrastiveModel` strictly;
        - writes `scores.npy` and `metrics.json`: global metrics; per-target `hard_auc` (primary), `macro_query_hard_auc` and `macro_mrr`; collateral per pack;
        - with `--relation`, writes `relation.json` (§9.3);
        - fails fatally on NaN or a mismatch (§9.5);
        - checks import coverage at the end.
      - `--encoder-crosscheck`: exactly §9.2 steps 1–6, using the trainer's `batch_encoder._content_to_text`.
      - `--fusion-weight-file` is the optional §9.4 reference.
      - Import `rank_auc` and `ranking_metrics` from `$R/scripts/evaluate_publication_fusion.py` and `content_to_text` and `CareerAwareContrastiveModel` from `$R/run_phase1_embedding_evaluation.py`.
      - Write `test_withholding_eval_metrics.py` with every §14 case: per-target AUC equals `roc_auc_score`; collateral selection; the 2-epoch fixture → `checkpoint_epoch_1.pt`; the 512 assertion; the trailing-space crosscheck cases with a stub tokenizer.
      Files: `$S/evaluate_withholding.py`, `$R/go_ppi_domain/tests/test_withholding_eval_metrics.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_withholding_eval_metrics.py" -q` passes.

- [x] 20. Create `text_similarity_covariate.py` (§10).
      - Encodes missing train texts with the frozen MPNet and records `n_encoded` and `n_from_cache`.
      - Computes `tsim_excess` (primary), `tsim_train`, and the N_2 analogues.
      - Uses a fixed 2,000-protein reference drawn with `Random(0)`.
      - Calls `append_pin("G2-covariate", ...)`.
      - Add a fixture test with a stub encoder that checks the arithmetic against a hand computation.
      Files: `$S/text_similarity_covariate.py`, `$R/go_ppi_domain/tests/test_text_covariate.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_text_covariate.py" -q` passes.

- [x] 21. Create `analyze_withholding.py` (§11, protocol §7–§8).
      - Expose library functions so the smoke step can call them in-process: `classify_groups`, `build_long_table`, `paired_deltas`, `two_way_bootstrap`, `holm`, `classify_contrast`, `decide`.
      - Implement:
        - the group-check preflight, using the same `group_ok_valid` and ignoring `.attempt*`/`.groupfailed*` dirs;
        - the completeness rule with denominator `n_targets × 6 × 5`;
        - the definedness rules;
        - the two-way pack × seed bootstrap (numpy `default_rng(20251009)`, B from the CLI) with discarded-replicate counting and the `unstable` flag;
        - Holm within H1/H2/H3;
        - sensitivity analyses S1–S5 (S4 via `numpy.linalg.lstsq`);
        - the ordered §8 rules, the mechanism rule and A1/`E_c`.
      - Write all §11 outputs.
      - Use scipy only for `spearmanr`; never import statsmodels.
      - Write `test_withholding_stats.py` with every §14 case.
      Files: `$S/analyze_withholding.py`, `$R/go_ppi_domain/tests/test_withholding_stats.py`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_withholding_stats.py" -q` passes.

- [x] 22. Create `sweep.py` (§12.3).
      - Subcommands `pin`, `plan`, `run --max-runs N` (required, 1–100, plus `--dry-run`), `status`, `check-group`, `clear-group` and `rescore`.
      - Implement exactly as specified: seed-major enumeration; the full pin preflight; the stopped-grid preflight; the argument-list subprocesses; `DONE.json`; the training-failure and eval-failure retry rules; `check_pending_groups()` with validity-based eligibility, the manifest-drift pre-check and atomic `GROUP_OK.json` writes; the mismatch handling; `clear-group` with all amendment-row rules; `rescore`, including decision 4c; and `sweep_log.jsonl`.
      - There is no mode that runs the whole grid.
      - Write `test_sweep_driver.py` with every §14 case:
        - interrupt, `--cells`, mismatch, stopped grid;
        - clearing;
        - amendment rows (a)–(g), with (a) on the frozen §13 fixture per decision 4a;
        - partial `GROUP_FAILED`, manifest drift, `rescore`, partial `GROUP_OK`;
        - a `FAILED.json` member;
        - the analysis preflight;
        - plus decision 4c's `rescore` refusal.
        - Subprocesses are stubbed and `RUNS`/pins are monkeypatched to `tmp_path`.
      - Apply decision 4b's one-sentence edit to protocol §13 and log it in protocol §14.
      - **AM1 note:** protocol §13 now holds a real row `AM1` (clears no groups, `—`). The frozen fixture `protocol_s13_v6.md` for case (a) must be the v6 §13 text with an empty amendments table (strip the AM1 row), as decision 4a intends; the real-protocol parse test must accept the AM1 row.
      Files: `$S/sweep.py`, `$R/go_ppi_domain/tests/test_sweep_driver.py`, `$R/go_ppi_domain/tests/fixtures/protocol_s13_v6.md`, `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/protocol_go_ppi.md`
      Verify: `"$PY" -m pytest "$R/go_ppi_domain/tests/test_sweep_driver.py" -q` passes; `"$PY" "$S/sweep.py" run` without `--max-runs` exits 2 (argparse).

## Cross-feature integration (first fixer iteration)

- [x] 23. Integrate and verify across features.
      - `"$PY" -m py_compile` on every new or modified `.py` file.
      - Run the full regression gate, which includes all new tests.
      - Run the slow suite with `GO_WITHHOLDING_SLOW=1`.
      - Run `check_additive.py`.
      - Confirm that `$R/preprocess/go_ppi/` and `$R/preprocess/go_ppi_splits/` are untouched: re-run the `shasum -a 256` listing and diff it against `$T/pre_edit/protected_sha256.txt`. Also confirm `test.jsonl` still matches `c65a1d2c…` from `setup_baseline.md`.
      - Run an in-process import-coverage probe: import the runner and the evaluator, and assert `imported_repo_modules() ⊆ code_sha256()[1]`.
      - Fix any seams, for example the lc_manifest keys the sweep and analysis read versus what the runner writes.
      - Record everything in the Verification log.
      Files: whatever the seams require.
      Verify: all of the above pass. Record the totals (expected 20 pre-existing + new go_ppi tests, and 28 career).

## Final smoke step (after review APPROVED; executed by the workflow's `smoke` step)

These items run the real gates once, with the real pins file. Follow protocol §10 commands verbatim.

- [x] 24. G1: `build_pruned_features.py` and `measure_leakage.py` into `$R/preprocess/go_withholding/{converted,splits}`.
      Verify: both exit 0, `leak_audit.json` has `"pass": true`, and `protocol_pins.json` has G1 and G1-leak. If the gate fails, stop and ask (decision 6).

- [x] 25. G2–G4.
      - G2: `build_target_table.py`, `evaluate_withholding.py build-cache`, `text_similarity_covariate.py`, `make_conditions.py --mode validation`.
      - G3: `make_conditions.py --mode conditions`. This builds all `n_conditions`, so it covers the "one pack at k=0 and k=5" request.
      - G4: `materialize_configs.py --lambda none`, then `lambda_pilot.py`, then `materialize_configs.py --lambda <λ>`.
      - Record λ, r₀, r* and the per-batch ratios.
      Verify: every command exits 0 and the pins have G2-targets, G2-cache, G2-covariate, G2-validation, G3, G4-lambda and G4-configs. G5 is not run (decision 7).

- [x] 26. G6-style smoke (protocol §10 G6 block, 2 epochs, `p01_k0` × 4 cells × seed 13, in `$R/results/go_withholding/smoke/`).
      - Run scoring for each cell, plus `--encoder-crosscheck` once.
      - Run `check-group`.
      - Run the audits for `all`, `p01_k0` and `p01_k5`.
      - Exercise the statistics in-process on the four smoke `metrics.json` files via `analyze_withholding`'s library functions (`build_long_table` → `paired_deltas` → `two_way_bootstrap`, B=1000). The CLI is covered by tests, and the smoke dirs are not `run_id`-named grid runs.
      - Show that the leakage assertions fire: copy `p01_k0` to `/tmp`, append one train record whose anchor is a W protein, and run the runner on it. Expect exit 2 and a `run_error.json` that names `assert_condition` check 2. Delete the copy afterwards.
      - Write `smoke_report_go_ppi.md` with the contents the smoke prompt requires, including the exact G5/G7 commands from protocol §10 (not executed).
      Verify: all G6 bullets of protocol §9 hold, and the full pytest suite passes again.

---

## Verification log

Coders append below: item, exact command, result (passed/failed counts), wall time, and notes. Scratch notes contain counts only (decision 5).

### FEAT-001 (items 1–8), 2025-10-09

`PY=/Users/olukotunjosh/Downloads/CDCL/.venv/bin/python`, `T=/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/task-go-ppi-withholding`.

- Item 1 (before any code edit):
  - `cp -p` of `contrastive_learning/{batch_processor,loss_engine,trainer,data_structures}.py` and `go_ppi_domain/{negative_selector,go_ontology}.py` to `$T/pre_edit/`; `git diff HEAD -- contrastive_learning/<f>.py > $T/pre_edit/<f>.py.diff` for the three user files; `git diff --stat > $T/pre_edit/git_diff_stat.txt`; `find preprocess/go_ppi preprocess/go_ppi_splits -type f -print0 | sort -z | xargs -0 shasum -a 256 > $T/pre_edit/protected_sha256.txt` (11 files). Diff sha256s match planning: batch_processor `15353d4a…`, loss_engine `82d81de9…`, trainer `b7867e14…`. HEAD `699e0cb`.
  - `"$PY" /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/_withholding_fixtures.py` (selector byte-identical to its snapshot, checked with `cmp` first) → wrote `go_ppi_domain/tests/fixtures/selector_flag_false_expected.json` (anchors U, A × seeds 13, 42 × epochs 0..14).
  - `"$PY" "$T/check_additive.py"` on the untouched tree → exit 0. Gate: `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests -q` → 20 passed (6.7 s); career four files → 28 passed (3.5 s).
- Item 2: `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/test_code_pin.py -q` → 10 passed (0.3 s).
- Item 3: `"$PY" -m pytest .../go_ppi_domain/tests/test_go_relation_table.py .../go_ppi_domain/tests/test_go_ppi_wiring.py -q` → 13 passed (4.6 s).
- Item 4: `"$PY" -m pytest .../go_ppi_domain/tests/test_go_withholding_loss.py .../go_ppi_domain/tests/test_go_ordinal_loss.py -q` → 27 passed (2.2 s). `check_additive.py` → OK, 3 compute_loss return wraps.
- Item 5: `check_additive.py` → OK (trainer.py: 5 inserted lines).
- Item 6: `"$PY" -m pytest .../go_ppi_domain/tests/test_go_withholding_negatives.py .../go_ppi_domain/tests/test_go_decomposition.py -q` → 14 passed (8.6 s). `check_additive.py` → OK (2 random_window tests replaced).
- Item 7: `"$PY" -m pytest .../go_ppi_domain/tests/test_epoch_guard_and_recorder.py -q` → 19 passed (7.5 s).
- Item 8 / final FEAT-001 verification (exact FEAT commands):
  - `"$PY" -m py_compile` of the 8 files listed in FEAT-001 `verification` → OK (the 7 new test/fixture files and `check_additive.py` also compile).
  - `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests -q` → **83 passed** (20 pre-existing + 63 new), 11.3 s (15.4 s wall).
  - `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/tests/test_career_ordinal_candidates.py /Users/olukotunjosh/Downloads/CDCL/tests/test_mesh_soft_sampling.py /Users/olukotunjosh/Downloads/CDCL/tests/test_career_adapter_equivalence.py /Users/olukotunjosh/Downloads/CDCL/tests/test_career_golden_regression.py -q` → **28 passed**, 6.0 s (8.8 s wall).
  - `"$PY" "$T/check_additive.py"` → exit 0: batch_processor byte-identical; loss_engine 3 wraps + insertions; trainer 5 / data_structures 10 inserted lines; negative_selector 2 window-test replacements; go_ontology unchanged; all user-diff `+` lines present in order (56 / 116 / 53).
  - Protected data re-hash vs `$T/pre_edit/protected_sha256.txt` → identical.
- Read-only sizing probe (no writes): `GoRelationTable.from_train_file` on the reference `preprocess/go_ppi/go_index_P.pkl` + `preprocess/go_ppi_splits/train.jsonl` → 4,709 proteins, 4,709 annotated (reference index is unpruned), 84.6 MB float32, 2.5 s, peak RSS ≈ 0.5 GB. No scratch dir was created.
- Note: `test_go_ordinal_loss.py` run ALONE fails `test_trainer_seeds_before_encoder_and_projection_initialization` (KeyError: go_ppi adapter not registered). The cause is pre-existing and independent of these edits: the test relies on another module having imported `go_ppi_domain.record_adapter` (not re-run on the untouched tree). `test_go_withholding_loss.py` now imports it, so the two-file item-4 command passes; the full directory always passed.

### FEAT-002 (items 9–13), 2025-10-09 — STOPPED at a protocol stop (item 12 scratch)

`PY=/Users/olukotunjosh/Downloads/CDCL/.venv/bin/python`, `T=/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/task-go-ppi-withholding`. Port sources `/tmp/goinv/*.py` were **gone** at the start of this step, so items 10–12 were implemented from the design text directly. The scratch runs reproduce the findings numbers: leak rates exactly, and candidate/strata counts within a few terms (see below).

- Item 9: `"$PY" "$T/check_additive.py"` → OK, `go_ontology.py: inserted lines = 106` (insert-only). Defaults identical: `test_parse_gaf_defaults_identical_to_pre_edit` and `test_goindex_build_defaults_identical_to_pre_edit` compare against `$T/pre_edit/go_ontology.py`.
- Items 9–11 tests: `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/test_leak_pruning.py /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/test_build_pruned_features.py -q` → 21 passed (0.7 s).
- Item 11 scratch (real data, counts only). Ran `GO_WITHHOLDING_PINS=/tmp/go_withholding_scratch/pins.json "$PY" /Users/olukotunjosh/Downloads/CDCL/scripts/go_withholding/build_pruned_features.py --out-converted /tmp/go_withholding_scratch/converted --out-splits /tmp/go_withholding_scratch/splits`.
  - Exit 0, 28.3 s wall, peak RSS 0.54 GB.
  - Structural shas for train/validation/test equal the reference, pool rows are identical, and the pairs sha equals the reference.
  - Counts: 15,708 / 5,587 / 22,341 records and 7,041 pool rows.
  - GAF BP rows: 138,976 kept by default guards. 12,828 were dropped by the with/from human-protein filter (v1 measured ≈12,871), leaving 126,148. GO:0005515-descendant drops: 0 (BP no-op).
  - Pruned index: 17,648 genes. Empty pruned `go_uris`: 1,187 of 15,155 proteins (same as v1).
  - Mask lexicon: 52,404 entries (47,691 alnum, 4,713 special); 3 aliases dropped by doc frequency; 0 unmaskable symbols. 25,075 replacements; token mask rate 3.33% (v1 measured 3.1%).
- Item 11 scratch: ran `measure_leakage.py --splits …/splits --converted …/converted --out /tmp/go_withholding_scratch/leak_audit.json` → exit 0, `"pass": true`, 20.5 s, peak RSS 0.18 GB.
  - Reference test rates (grade 2 / 1 / 0): (a) 11.07% / 0.17% / 0%, (c) 5.92% / 0.41% / 0%. Both reproduce exactly (`reference_sanity.match` true). (c) is computed over all GO aspects (BP 46, MF 33, CC 50 hits, as in findings); BP-only `c_P` is 3.95%. Reference (b) 12.62% / 0.19% / 0%. Reference train: (a) 11.17%, (c) 7.32%.
  - New test rates: (a) 0.17% / 0% / 0% (2 of 1,165 grade-2 records), (b) 0.17% / 0% / 0%, (c) 0% / 0% / 0%. New train: (a) 0%, (b) 0.08%, (c) 0%. The gate passes.
  - Ungated (d): `d_other_db` 0 everywhere. `d_complexportal` new test 15 / 59 / 59 records. `d_complex_name` is descriptive.
  - Pins in scratch: G1, G1-leak.
- Item 12 tests: `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/test_target_table.py -q` → 14 passed (0.8 s). This includes the exact `pack_search_log` sequence, the Jaccard-skip fixture, the 24→20 fallback to 6×3+1×2, the size rule, and a build-twice identical sha.
- **Item 12 scratch: PROTOCOL STOP.** Ran `GO_WITHHOLDING_PINS=/tmp/go_withholding_scratch/pins.json "$PY" /Users/olukotunjosh/Downloads/CDCL/scripts/go_withholding/build_target_table.py --converted /tmp/go_withholding_scratch/converted --splits /tmp/go_withholding_scratch/splits --out /tmp/go_withholding_scratch/targets/target_table.json` → **exit 3**, "packing infeasible at 20 targets; queue exhausted", 115.7 s, peak RSS 0.61 GB. Counts only:
  - Terms with n ≥ 40: 1,094. Passing rule 2: 876. Passing rules 1–4: 461. Passing 1–5 (`n_candidates`): 461; findings had 469.
  - `n_non_nested` 153 (findings: 158). S_sib cuts: low ≤ 1,339.5 and high ≥ 3,624.5 (findings: 1,323 / 3,662). Strata sizes 51 low / 51 high.
  - Matched pairs: 12 accepted, replacement queue 39. SMD for the first 24: log n −0.048, log support −0.036. For the first 20: −0.044 / −0.025.
  - Packing constraints among the 24 accepted targets: 64 of 276 target pairs are pack-compatible. Of the incompatible pairs, 204 fail P_train disjointness, 90 fail the neighbour-overlap cap of 5%, and 1 fails N_2 (pairs can fail several). At the pack level, 86 of 276 pairs already exceed the 8% pack removal cap; retention feasibility never binds.
  - Exact enumeration (diagnostic only) confirms infeasibility, so this is not a search weakness:
    - At 24 targets only 5 valid 3-packs exist and 8 are needed.
    - At 20 targets (m01–m10) there are 3 valid 3-packs and 13 valid 2-packs; 6 + 1 are needed.
    - None of the 41 slot-cycled 20-target sets is feasible; the largest has 14 valid 3-packs.
    - Diagnostic only, not proposed: 12×2 packs at 24 targets and 10×2 at 20 are also infeasible.
  - The `pack_search_log` before the stop has 80 entries: 39 replacements in phase 24 and 41 in phase 20.
  - Not ticked; `send_message` warning sent. No parameter was changed.
- Item 13 tests: `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/test_withholding_generator.py -q` → 14 passed (5.2 s). This includes `make_conditions.py` end to end on an 8-pack table (41 conditions) and a 6×3+1×2 table (36 conditions). **Item 13 scratch not run**: it needs a real target table, which the item-12 stop prevents. Not ticked.
- Scratch dir `/tmp/go_withholding_scratch` deleted. Nothing was written under `preprocess/go_withholding/`, `results/go_withholding/` or `config/go_withholding/`.
- Final gate:
  - `"$PY" -m py_compile` on the 13 new or changed `.py` files → OK.
  - FEAT-002's four test files → **49 passed** (6.8 s).
  - `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests -q` → **132 passed** (83 + 49), 16.5 s (19.3 s wall).
  - The four career files → **28 passed**, 6.0 s (8.5 s wall).
  - `"$PY" "$T/check_additive.py"` → exit 0.
  - Protected re-hash vs `$T/pre_edit/protected_sha256.txt` → identical.

### FEAT-002 follow-up: read-only packing feasibility counts (orchestrator request), 2025-10-09

Commands:

- G1 scratch rebuild: `GO_WITHHOLDING_PINS=/tmp/go_withholding_scratch/pins.json "$PY" .../build_pruned_features.py --out-converted /tmp/go_withholding_scratch/converted --out-splits /tmp/go_withholding_scratch/splits` → exit 0, 13.8 s.
- Diagnostics: `"$PY" /tmp/go_withholding_scratch/feas.py` (read-only, under 30 s). It uses EXACT exact-cover packing inside the design 2.6 slot-cycling and fallback procedure, plus a "greedy-by-cost packable subset" over all 51 matched pairs.

No parameter or protocol was changed. Scratch data was deleted afterwards; the script `/tmp/go_withholding_scratch/feas.py` is kept. Counts only.

Conventions:

- runs = (5·packs + 1) × 4 cells × 5 seeds; hours assume 200 s per run.
- disk is an upper bound of 37.3 MB per condition.
- "S reduction by mates" = the share of S_sib(t), i.e. the degree mass over NP_sib(t), that pack-mates' k = 0 withholding removes. It counts withheld neighbour proteins plus pairs to withheld proteins.

**Procedure as designed** (first 24 → 20, slot cycling, 3-packs with at most one 2-pack, exact solver):

| caps | low-high compatible / 144 | valid 2-packs / 3-packs among the first 24 | feasible? | largest feasible (first-N + cycling) |
|---|---|---|---|---|
| protocol | 35 | 24 / 5 | no | 2 targets |
| (a) no disjointness | 93 | 65 / 39 | no | 18 (9×2) |
| (b) removal 10% | 35 | 35 / 19 | no | 2 |
| (b) removal 12% | 35 | 35 / 35 | no | 2 |
| (b) nbr 10% | 38 | 27 / 5 | no | 6 |
| (b) removal 10% + nbr 10% | 38 | 38 / 24 | no | 6 |
| (b) removal 12% + nbr 10% | 38 | 38 / 47 | no | 6 |
| (a) + (b) combined | 129 | 129 / 953 | yes | 24 (8×3): 820 runs, 45.6 h, 1.53 GB; P_train lost to mates median 4.4%, max 18%; S reduction median 8.4%, max 22.3% |
| (e) 16 / 12 targets, protocol caps, 3/2-packs | — | — | no / no | — |

Low-high pair failures among the 144 low-high pairs (protocol caps): P_train overlap 105, nbr overlap > 5% 51, pair removal > 8% 47. Single-target removal fractions: min 1.5%, median 4.0%, max 5.0%.

(c) **One target per condition:** 24 targets (12 low / 12 high), SMD log n −0.048 and log support −0.036, 121 conditions, 2,420 runs, 134.4 h, ≤ 4.51 GB. S reduction by mates is 0 by construction.

(f) **Greedy-by-cost packable subset** (a selection-rule change, not a cap change: add matched pairs in cost order while the set stays exactly packable with any 2/3 mix):

| caps | targets | packs | runs / hours / GB | highest pair index used | SMD log n / log support | P_train lost (median / max) | S reduction (median / max) | high targets pushed below the high cut |
|---|---|---|---|---|---|---|---|---|
| protocol | 24 | 9×2 + 2×3 = 11 | 1,120 / 62.2 / 2.09 | m26 | −0.048 / 0.163 | 0 / 0 | 3.95% / 8.7% | 0 |
| protocol, 16 targets | 16 | 5×2 + 2×3 | 720 / 40.0 / 1.34 | m22 | −0.037 / 0.061 | 0 / 0 | 4.6% / 8.7% | 0 |
| protocol, 12 targets | 12 | 3×2 + 2×3 | 520 / 28.9 / 0.97 | m18 | −0.122 / −0.010 | 0 / 0 | 4.6% / 10.1% | 0 |
| (a) | 24 | 11 | 1,120 / 62.2 | m16 | −0.083 / 0.009 | 2.2% / 14% | 4.1% / 13.9% | 1 |
| (b) removal 10% or 12% | 24 | 11 | 1,120 / 62.2 | m20 | — | — | 5.5% / 15.7% | 1 |
| (b) nbr 10% | 24 | 11 | 1,120 / 62.2 | m23 | — | — | 3.5% / 11.5% | 0 |
| (b) removal 10% or 12% + nbr 10% | 24 | 6×2 + 4×3 = 10 | 1,020 / 56.7 | m20 | — | — | 6.3% / 15.9% | 1 |

Every packed option reduces some targets' S through pack-mates, because the existing 5% neighbour-overlap cap allows it and removed pairs lower neighbours' degree. Only (c) has none.

### FEAT-002 resumed under amendment AM1 (items 12–13), 2026-10-10

User decision A: adopt option (c), one target per condition (singleton packs). `PY=/Users/olukotunjosh/Downloads/CDCL/.venv/bin/python`, `T=/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/task-go-ppi-withholding`. Counts only; no names or GO ids were printed or recorded.

- Amendment record:
  - protocol §13 row `AM1` (2026-10-10, gate "before G2", groups cleared `—`, re-pin `no`); no earlier AM1 row existed. `wc.parse_amendment_rows()` returns 1 row with 7 cells. "(No rows yet.)" removed.
  - Protocol body counts updated in §4 (packing paragraph, retention), §5 (grid), §7 (bootstrap, 10% rule note), §9 G3, §10 G7 comment, §11 G3 pin, §12 (bootstrap limitation), plus a §14 "Amendment AM1" table.
  - Design: status line, §0.1 item 11, §1.3 constants, §2.1, §2.6 (amended rule first, old text marked superseded), §2.7, §2.8 schema, §3.3, §11, §12.3, §14 test lists, §15, and a new §22.
  - The 10% rule denominator `n_targets × 6 × 5` counts the six k levels (five numeric plus `all`), not pack size, so it stays 720 (600 at 20). Only the docs gained a clarifying note.
- Code:
  - `withholding_targets.py`: `DEFAULT_PARAMS` now has `pack_size` 1; `pack_nbr_overlap_cap` and `pack_search_seed` are removed; `pack_removal_cap` 0.08 is kept and applied per target. `pack_shape`, `pack_targets`, `pack_search` and `DEFAULT_RESTARTS` are removed. New `singleton_packs(targets, pack_ok)` stops with exit 3 and a counts-only message if a target fails the per-target cap or retention feasibility. `build_target_table` lost its `restarts` argument. The table drops `pack_search_log` and `packs[*].max_nbr_overlap`. `replacement_queue` is the queue left after the size rule. Pack ids follow the target order.
  - `build_target_table.py`: docstring, no `restarts`, no log count in the summary or in the stop message.
  - `withholding_constants.py`: `PACK_SEARCH_SEED` removed (it had no users).
  - `withholding.py`: one docstring sentence. `make_conditions.py` is unchanged (it is generic and reads `n_conditions` from the table).
- Tests:
  - `test_target_table.py`: the 5 multi-pack tests (pack shape, pack constraints, exact `pack_search_log`, fallback to 6×3+1×2, Jaccard skip) were replaced by 3 tests: singleton order and single-target-only checks; a counts-only stop message; params and removed API.
  - The end-to-end test now expects 4 targets, 4 packs, 21 conditions, 420 runs and no `pack_search_log`. The script test asserts the G2-targets pin `n_conditions` 21 / `n_packs` 4 / `n_runs` 420, plus a new exit-3 case for a target above the per-target removal cap.
  - `test_withholding_generator.py`: the fixture uses singleton packs, and the count check is parametrised as 24 → 121 and 20 → 101.
  - `"$PY" -m pytest .../test_target_table.py -q` → 12 passed. `"$PY" -m pytest .../test_withholding_generator.py -q` → 14 passed (7.7 s; the 121-condition case takes 3.9 s).
- Item 12 scratch (real data):
  - G1 rebuild: `GO_WITHHOLDING_PINS=/tmp/go_withholding_scratch/pins.json "$PY" /Users/olukotunjosh/Downloads/CDCL/scripts/go_withholding/build_pruned_features.py --out-converted /tmp/go_withholding_scratch/converted --out-splits /tmp/go_withholding_scratch/splits` → exit 0, 11.6 s, 0.62 GB RSS.
  - `measure_leakage.py --splits /tmp/go_withholding_scratch/splits --converted /tmp/go_withholding_scratch/converted --out /tmp/go_withholding_scratch/leak_audit.json` (same env) → exit 0, `pass` true, 9.7 s, 0.22 GB.
  - `build_target_table.py --converted /tmp/go_withholding_scratch/converted --splits /tmp/go_withholding_scratch/splits --out /tmp/go_withholding_scratch/targets/target_table.json` (same env) → **exit 0**, 2.8 s wall, 0.65 GB RSS. The pre-AM1 run had stopped with exit 3 after 115.7 s.
  - Selection counts: `n_candidates` 461; `n_non_nested` 153; cuts low ≤ 1,339.48 and high ≥ 3,624.52; strata 51 low / 51 high; 12 pairs accepted. These are identical to the pre-AM1 run, as expected, because only packing changed.
  - Table counts: `n_targets` 24 (12 low / 12 high), `n_packs` 24 (all size 1), `n_conditions` 121, `n_runs` 2,420, `shortfall` false, `replacement_queue` 39. SMD log n −0.048, log support −0.036.
  - Per-target k = 0 removal: min 1.46%, median 3.95%, max 4.95% (cap 8%). Retention-eligible: min 40, median 79, max 142 (≥ 25). Withheld at k = 0: min 63, median 136, max 294 proteins.
  - Scratch pins: G1, G1-leak, G2-targets.
- Item 13 scratch, in-process (`/tmp/go_withholding_scratch/item13.py`, deleted with the dir): `wh.load_base` → `build_targetfree_validation` → `build_condition` for `all`, `p01_k0` and `p01_k5` (`index` = pruned index). Exit 0, 4.5 s total, 0.81 GB RSS.
  - Target-free validation: V 330 proteins; 3,011 records kept, 2,576 dropped; 181 grade-2 records (5 full λ-pilot batches of 32, so ≥ 4 holds); 491 pool rows. `condition_ids(table)` = 121.
  - `all`: withheld_count 0, removal_frac 0, 15,708 records kept, 0 anchors dropped by the empty-pool rule; unannotated pool anchors 140, record anchors 115, records 328; 37.2 MB.
  - `p01_k0`: withheld_count 103 (61 positive + 42 pool-only), removal_frac 2.253% (260 undirected pairs, 392 records dropped, 15,316 kept), 65 pool-anchor rows dropped, 2,842 member entries removed, pool_member_share_removed 1.063%, 0 empty-pool anchors; unannotated 140 / 113 / 323; 36.1 MB.
  - `p01_k5`: withheld_count 98 (56 + 42), retained 5 (14 retained undirected pairs), removal_frac 2.132% (246 pairs, 370 records dropped), 60 anchor rows dropped, 2,707 member entries removed, pool_member_share_removed 1.012%; unannotated 140 / 113 / 323; 36.1 MB.
  - Disk at about 36–37 MB per condition: 121 conditions ≈ 4.4 GB. Note: `make_conditions.MIN_FREE_BYTES` is still the design's 3 GB, which is below that need. It was not changed (see FEAT-002 findings).
- `/tmp/go_withholding_scratch` deleted, including the earlier `feas.py`. Nothing was written under `preprocess/go_withholding/`, `results/go_withholding/` or `config/go_withholding/`.
- Final FEAT-002 gate (the verification block, verbatim):
  - `"$PY" -m py_compile` on the 6 changed `.py` files → OK.
  - The four FEAT-002 test files → **47 passed** (8.2 s; 8.9 s wall). That is 49 − 5 removed + 3 new.
  - `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests -q` → **130 passed** (13.3 s; 14.5 s wall).
  - The four career files → **28 passed** (2.8 s; 4.0 s wall).
  - `"$PY" "$T/check_additive.py"` → exit 0 (unchanged: batch_processor byte-identical; loss_engine 3 wraps; trainer 5 / data_structures 10 / go_ontology 106 inserted lines; negative_selector 2; user `+` lines 56 / 116 / 53 in order).
  - Protected `preprocess/go_ppi` + `preprocess/go_ppi_splits` re-hash vs `$T/pre_edit/protected_sha256.txt` → identical.

### FEAT-003 (items 14–18) + user decision B (remote GPU execution), 2026-10-10

`PY=/Users/olukotunjosh/Downloads/CDCL/.venv/bin/python`, `T=/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/task-go-ppi-withholding`. CPU only (Apple M2); no GPU used. No real study data, config, pin or pilot was written (`config/go_withholding/`, `results/go_withholding/`, `preprocess/go_withholding/`, `embedding_cache/go_withholding_text_embeddings.pt` all absent afterwards). Nothing committed.

- Item 14 (runner, `scripts/run_learning_curve_point.py`):
  - New args `--go-ppi-split-dir`, `--withholding-manifest`, `--study-cell`, `--go-diagnostics`, `--pilot`, `--device`. The §4.1 rules are argparse errors (exit 2): `--pilot`/`--study-cell`/`--go-diagnostics` without the manifest; manifest without `--domain go_ppi`, `--study-cell` or `--go-diagnostics`; manifest with `--validation-file`; `--go-ppi-split-dir` outside go_ppi.
  - `run_study()` runs §4.2 → §4.3 (exact order) → `trainer.train` → `assert_post_training` (§4.4) → `lc_manifest.json` (§4.7) inside one `try`; any failure writes `run_error.json` (`stage` ∈ preflight / wiring / condition_assertions / train / post_training / manifest, `error`, `traceback`, `cell`, `cond_id`, `seed`) and returns 2.
  - `assert_post_training` checks seen ∩ W, missing ids, recorder counters, every epoch guarded, import coverage ⊆ code pin, `checkpoint_epoch_{N-1}.pt` with that `epoch`, no `*emergency*.pt`. Its keys are recorded as `post_*`, the preflight ones as `preflight_*`, next to assert_condition's `1_hashes`…`10_callbacks` in `lc_manifest["withholding"]["assertions"]`.
  - Without `--withholding-manifest` the original path runs; the only additions there are the `--go-ppi-split-dir` override and an opt-in device block (only when `--device` is passed or `training_device != "auto"`). Its `lc_manifest.json` keys are unchanged by default (tested).
  - `$PY $R/scripts/run_learning_curve_point.py --help` → exit 0; it lists `--go-ppi-split-dir`, `--withholding-manifest`, `--study-cell {baseline,loss_only,neg_only,both}`, `--go-diagnostics`, `--pilot` and `--device {auto,cuda,cpu}`.
- Item 15: `scripts/go_withholding/materialize_configs.py` (§12.1, `EXPECTED_PAIR_DIFF` audit in `CONFIG_DIR/.staging/`, deleted afterwards; `--lambda none` = baseline, neg_only, `loss_pilot_template.json`, no pin; `--lambda <float>` = four cells, `configs.sha256.json`, `G4-configs`). Not run against `config/go_withholding/` (that is G4).
- Item 16: `scripts/go_withholding/lambda_pilot.py` (§12.2). Runner subprocess with `--pilot` (argument list, `cwd=ROOT`). Measurement in-process: strict `checkpoint_epoch_2.pt` load, `set_selector_epoch` assertion, batches of 32 with the last partial dropped, < 4 → exit 3, relation table over the validation proteins, ratio median. Log-space rounding to {1,2,3,5}×10ⁿ, one confirm and one re-estimate else exit 3, `lambda_choice.json`, `G4-lambda`, then `materialize_configs --lambda`. A λ equal to one already measured reuses that run (identical settings) instead of retraining into an existing dir; recorded in `attempts`.
- Item 17: `scripts/go_withholding/audit_negatives.py` (§7). `/tmp/goinv/e_negaudit.py` was gone, so implemented from §7: `cfg.training_seed = --seed` asserted on the selector, production `from_config` selector, separate simGIC measurement matcher, HuRI/Lit-BM via STRING `Ensembl_gene`, every §7 share, the W gate (= 0) and the share invariant for pack conditions, offline identity baseline == loss_only and neg_only == both (exit 2).
- Item 18: `go_ppi_domain/tests/test_withholding_integration_slow.py`. Real MPNet on a 9-target fixture whose conditions are written by `make_conditions.py`. It trains on the 240-record `p01_k0` condition (4 batches of 64); the fixture's record count is fixed by the generator, so this is 240, not exactly 256. (a) One clean runner subprocess per cell. (b) In-process with and without diagnostics. (c) In-process forced detach → exit 2.
- User decision B, where each part went:
  1. Device selection, in FEAT-003 item 14:
     - `--device auto|cuda|cpu` in the runner, plus a `TrainingConfig.training_device: str = "auto"` field (insert-only in `data_structures.py`; read only by the runner, and the CLI wins).
     - `resolve_device`: `cuda` without CUDA is fatal (study path: preflight → exit 2; default path: argparse error).
     - `device_scope("cpu")` hides CUDA only while the trainer is built. Every device choice (trainer, loss engine, embedding cache, SentenceTransformer) is made at construction, so the frozen MPNet and the projection head both follow `auto`/`cuda`. It is a no-op on a CPU machine, and the trainer is not edited.
     - `device_record` checks that the trainer, loss engine, text encoder and projection are all on the resolved device, and writes `lc_manifest["device"]` (requested, resolved, cuda_available, torch, cuda_device_name, components).
     - `lambda_pilot.py --device` forwards the flag to the runner and the measurement.
     - `withholding_diagnostics.preserved_rng` also saves and restores initialised CUDA RNG states.
  2. and 3. Not in FEAT-003. They belong in FEAT-004 item 22 (`sweep.py`): forward `--device` to the runner and evaluator subprocesses, plus the remote driver plumbing. The runner and the pilot already accept `--device`.
  4. One-command remote regeneration: new `scripts/go_withholding/regenerate_conditions.py` (+ `go_ppi_domain/tests/test_regenerate_conditions.py`), placed in FEAT-003 next to `make_conditions`.
     - Ship `preprocess/go_withholding/{converted,splits,targets}` + `results/go_withholding/protocol_pins.json` and run `python scripts/go_withholding/regenerate_conditions.py`.
     - It verifies the base files against the latest G1 pin, the table against G2-targets, and the target-free validation against G2-validation (built in a staging dir, moved into place only on a match). For each condition it compares the in-memory manifest sha256 with `G3["manifests"][cond]` **before** writing, then calls `build_condition` (all §3.4 checks, idempotent).
     - It appends no pin. `--only` takes a subset.
  - No absolute paths:
     - `withholding_constants` no longer asserts `ROOT == /Users/...` (nor do the four FEAT-002 scripts).
     - New helpers: `ROOT_TOKEN = "$ROOT"`, `expand_root_token`, `root_token`, `config_path_problems`, `expand_config_dict`, `load_study_config`, `scrub_root`, `relativize_json`, `rel_to_root`, `lambda_choice_path`.
     - Study configs are written with `"$ROOT/<relative>"` values. The runner accepts `$ROOT`-anchored or absolute config values and rejects bare relative ones (the §4.2 step 1 test still holds); CLI overrides must be absolute.
     - `lc_manifest.json`, `run_error.json`, `lambda_choice.json` and the audit JSONs pass through `relativize_json`/`scrub_root`.
     - FEAT-002 emitters fixed: `frozen_hashes["raw"]` keys (and so the G1 pin) are repo-relative; `convert_manifest.json` and the pruned pickle's `meta` `obo_path`/`gaf_path` are repo-relative; `leak_audit.json` `inputs` are repo-relative.
     - Pins hold hashes only.
     - Design text deviation (decision B, design file not edited): §4.2 step 1 and §12.1 said "absolute"; the fail-closed intent (no cwd-relative path) is kept with `$ROOT` anchoring. `test_study_code_bakes_in_no_machine_path` asserts that no study module or script contains `/Users/`.
- Remote caveats for later steps (not changed here):
  - `code_sha256()` shells out to `git ls-files`, so the remote checkout must be a git work tree with the same tracked and untracked `.py` files.
  - `make_conditions.MIN_FREE_BYTES` is still 3 GB against about 4.4 GB of conditions (see FEAT-002).
- Verification (FEAT-003 block, verbatim):
  - `"$PY" -m py_compile` on the 19 new or changed `.py` files → OK.
  - `"$PY" -m pytest .../test_runner_withholding_assertions.py .../test_materialize_configs.py .../test_lambda_pilot.py .../test_negative_audit.py -q` → **92 passed** (38 + 14 + 31 + 9), 7.1 s (8 s wall).
  - `GO_WITHHOLDING_SLOW=1 "$PY" -m pytest .../test_withholding_integration_slow.py -q` → **3 passed**, 61.3 s (63 s wall; an earlier run took 64.7 s).
  - `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests -q` → **230 passed, 3 skipped** (130 + 92 + 8 regeneration/path tests; the 3 slow tests skip), 17.3 s (18 s wall).
  - The four career files → **28 passed**, 2.3 s (3 s wall).
  - `"$PY" "$T/check_additive.py"` → exit 0. batch_processor is byte-identical. loss_engine has 3 wraps. trainer has 5 inserted lines, data_structures 15 (10 + the 5-line `training_device` block), go_ontology 106. negative_selector has 2 replacements. User `+` lines 56 / 116 / 53 are present in order.
  - Protected `preprocess/go_ppi` + `preprocess/go_ppi_splits` re-hash vs `$T/pre_edit/protected_sha256.txt` → identical.
  - Scratch `/tmp/feat3_probe` deleted.

### FEAT-004 (items 19–22) + user decision B items 2, 3, 5, 2026-10-10

`PY=/Users/olukotunjosh/Downloads/CDCL/.venv/bin/python`, `T=/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/task-go-ppi-withholding`. CPU only (Apple M2); no GPU used. No real study artifact, config, pin or condition was written (`results/go_withholding/`, `preprocess/go_withholding/`, `config/go_withholding/` and the study embedding cache are all absent afterwards). Nothing committed.

- Item 19, `scripts/go_withholding/evaluate_withholding.py` (§9):
  - `build-cache` (§9.1): test sha vs `frozen_hashes.json`; unique proteins in record order, texts via `content_to_text`; `max_seq_length = 512`; `max_tok < 512` else exit 2; batch-64 float32 encode; `test_<sha16>.npz` + `.meta.json` (incl. `frozen_global_spearman` over the 20,000 fixed pairs); refuses to overwrite; `G2-cache` pin holds hashes and counts only.
  - `score` (§9.2): checkpoint name, sha and `epoch == epochs-1` from `lc_manifest.json`; strict `CareerAwareContrastiveModel(768, 128, 0.3, False)`; `scores.npy`; `metrics.json` with global (hard/easy/pooled AUC, macro-query hard AUC, macro MRR, n_queries), per-target (`hard_auc` primary, `macro_query_hard_auc`, `macro_mrr`, counts; a target without grade-2 or grade-1 records is fatal) and collateral (own pack; every pack for `all`). NaN/inf fatal. Import coverage checked at the end. Every failure: `eval/eval_error.json` (scrubbed), exit 2.
  - `--relation` (§9.3): global Spearman(cos, simGIC_BP) on `Random(0)` pairs; per target ≤ 5,000 P_test × NP_test_sib pairs, Spearman, `target_nbr_mean_cos`, `random_ref_mean_cos` (5,000 fixed test proteins outside NP ∪ P), `excess`. simGIC is the `GoRelationTable` formula evaluated pairwise (tested equal).
  - `--encoder-crosscheck --config` (§9.2 steps 1–6): trainer's `batch_encoder._content_to_text`, token check over all cache proteins, 50 `Random(0)` proteins re-encoded, cos ≥ 0.9999, `encoder_crosscheck.json`.
  - `--fusion-weight-file` (§9.4): applies a fixed `w`, reported as exploratory. Choosing `w` on validation (writing `fusion_weight.json`) is not implemented; it is optional and outside the grid.
  - `rank_auc`, `ranking_metrics`, `z_stats`, `z_apply`, `json_safe` are imported from `scripts/evaluate_publication_fusion.py`.
  - Read-only parse of the reference test split (`read_test_records`, counts only): 3,014 proteins, 22,341 records, 1,396 queries, grades 0/1/2 = 11,165 / 10,011 / 1,165.
- Item 20, `scripts/go_withholding/text_similarity_covariate.py` (§10): `tsim_excess` (primary), `tsim_train`, and the N_2 analogues; per-target 2,000-protein reference via `Random(0)` on the sorted train partition outside T_train ∪ NP; trainer-cache lookup by content key, else encode with frozen MPNet (`n_from_cache` / `n_encoded`); `G2-covariate` pin; refuses to overwrite.
- Item 21, `scripts/go_withholding/analyze_withholding.py` (§11, protocol §7–§8):
  - Library functions `classify_groups`, `build_long_table`, `paired_deltas`, `two_way_bootstrap`, `holm`, `classify_contrast`, `decide` (plus `tuple_status`, `hierarchical_bootstrap`, `s4_adjusted`, `compute_contrasts`).
  - Preflight via `group_ok_valid`; renamed dirs are never a run id. Completeness denominator `n_targets × 6 × 5` (AM1: 720 at 24 targets); exit 3 above 10% unless `--allow-incomplete`.
  - Two-way pack × seed bootstrap, `default_rng(20251009)`, vectorised. The draws are identical for every contrast. Per-contrast discarded-replicate counts; `unstable` above 5%.
  - Holm within H1 (3 cells) / H2 (3) / H3 (loss_only, both). Ordered §8 rules with the unstable check first. A1/`E_c`, mechanism rule (a)+(b), collateral cost.
  - Sensitivity: S1–S5, with S4 via `numpy.linalg.lstsq`. scipy is used only for `spearmanr` in the evaluator; `analyze_withholding.py` imports no scipy and never statsmodels (tested in a subprocess).
- Item 22, `scripts/go_withholding/sweep.py` (§12.3):
  - Subcommands `pin`, `plan`, `run --max-runs N` (required, 1..100, `--dry-run`), `status`, `check-group`, `clear-group`, `rescore`. No mode runs the grid: `--max-runs` is mandatory and capped at 100, and there is no `--all`.
  - Enumeration is seed-major; run and condition counts are read from the table, with an assertion that the grid size equals `target_table["n_runs"]` (AM1: 121 conditions / 2,420 runs at 24 targets; 101 / 2,020 at 20; tested).
  - Full pin preflight (G1 via `regenerate_conditions.verify_base`, G2-targets, G2-covariate, G2-cache npz + meta, G2-validation manifest + files, G3 count + every manifest, G4-lambda, last G4-configs with exactly four files + `configs.sha256.json`, G5 + `code_sha256()`, ≥ 3 GB free, no pilot dir), then the stopped-grid preflight, then `check_pending_groups()`.
  - Argument-list subprocesses; `DONE.json` with exactly the design keys.
  - Failure handling: training failure → `.attempt<n>` + one retrain → `FAILED.json` (train); eval failure → `eval.attempt<n>` + one re-score → `FAILED.json` (eval, with `lc_manifest_sha256`, plan decision 4c).
  - Group checks: validity-based eligibility, the manifest-drift pre-check, atomic `GROUP_OK.json` in all four dirs, `GROUP_FAILED.json` + `GROUP_OK.superseded.json` on a mismatch. The cell order (`withholding.cell`), cond/seed equality and pilot flag are also checked; this is a fail-closed addition that a correct group always passes.
  - `clear-group` enforces every amendment-row rule. `rescore` enforces 4c (`lc_manifest_changed_after_failure`, exit 2), then the pin preflight, then the stopped-grid check.
  - `sweep_log.jsonl` lines also carry `wall_seconds`, an additive key used for the remote time estimate.
  - `pin` runs an import probe subprocess. Real probe on this tree: 29 repo modules, all covered by `code_sha256()`.
- Plan decision 4b:
  - Protocol §13 gains "Cells are separated by `|`; a literal `|` inside a cell is written `\|`." It was not present before; FEAT-002 had not added it.
  - It is logged in protocol §14 as "Plan decision 4b (2026-10-10, implementation, before G1)" (wording only, not an amendment row).
  - `parse_amendment_rows(PROTOCOL)` still returns exactly the AM1 row with 7 cells.
  - Frozen fixture `go_ppi_domain/tests/fixtures/protocol_s13_v6.md`: the v6 §13 with an empty amendments table (AM1 row and the 4b sentence stripped), used for case (a).
- Shared-file edit (additive, study module): `go_ppi_domain/withholding_constants.py` gains `run_id_for` and `group_run_ids`, used by both the sweep and the analysis. The user files and the other shared files are untouched.
- User decision B, where each part went:
  - (2) Sharding and parallel workers on one GPU, built into `sweep.py` (item 22):
    - `--shard i/N` (1-based) partitions by (cond, seed) group number over the full, unfiltered grid (`g % N + 1`). It is disjoint and exhaustive, keeps groups whole and stays seed-major.
    - `--workers W` (1..16) runs threads, each driving its own subprocesses; group checks stay in the main thread after every completed run. `--max-runs` bounds the total runs started.
    - Per-run claim `RUNNING.json` (pid, host, start; `O_EXCL`). A dir without `DONE.json`/`FAILED.json` whose claim is absent, corrupt or held by a dead pid on this host is renamed `<run_id>.interrupted<n>` and retrained. A live claim, or one from another host, is skipped. `DONE.json` runs are a no-op.
    - `--device` is forwarded to the runner and the evaluator (`rescore` too).
    - Fail-closed guard: `--workers > 1` or `--shard` refuses (exit 2) until the baseline config's `embedding_cache_path` exists. Concurrent runs would otherwise race on writing the shared frozen-encoder cache; the first run (`all__kall__baseline__s13`) warms it.
    - `GROUP_FAILED` / `clear-group` / `DONE.json` semantics are unchanged.
  - (3) No absolute paths: every new emitter writes repo-relative names or passes through `relativize_json` / `scrub_root`, and pins hold hashes and counts only. Paths resolve from the checkout (`ROOT` = the repo the file lives in); subprocess CLI paths are computed at run time. The existing `test_study_code_bakes_in_no_machine_path` still passes over the four new scripts.
  - (5) `.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_GO_PPI.md`: rsync copy list (code + `.git` + `.gitignore` for the code pin, study data, pins, eval cache, raw inputs, the exact MPNet snapshot), pinned env (`scripts/go_withholding/requirements-remote.txt` + CUDA torch 2.8.0 cu128), G0 + code-hash check, `regenerate_conditions.py` against the pins, λ pilot if not finalised, G5/G6, warm-up + sharded/parallel launch, monitoring/resume, audits + statistics, and a time estimate clearly labelled as an estimate (≈ 8–40 h wall at 4–8 concurrent runs, not measured).
- Tests (new): `test_withholding_eval_metrics.py` 17, `test_text_covariate.py` 3, `test_withholding_stats.py` 27, `test_sweep_driver.py` 57; shared fixtures in `go_ppi_domain/tests/_grid_fixtures.py`.
- Verification (FEAT-004 block, verbatim):
  - `"$PY" -m py_compile` on the 10 new or changed `.py` files → OK.
  - `"$PY" -m pytest .../test_withholding_eval_metrics.py .../test_text_covariate.py .../test_withholding_stats.py .../test_sweep_driver.py -q` → **104 passed**, 6.9 s (8 s wall).
  - `"$PY" .../scripts/go_withholding/sweep.py run ; echo exit=$?` → argparse "the following arguments are required: --max-runs", **exit=2**. `run --max-runs 101` → "--max-runs must be in 1..100", **exit=2**.
  - `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests -q` → **334 passed, 3 skipped** (230 + 104; the 3 slow tests skip), 21.3 s (23 s wall).
  - The four career files → **28 passed**, 3.2 s (4 s wall).
  - `"$PY" "$T/check_additive.py"` → **exit 0**, unchanged: batch_processor byte-identical; loss_engine 3 wraps; trainer 5 / data_structures 15 / go_ontology 106 inserted lines; negative_selector 2; user `+` lines 56 / 116 / 53 in order.
  - Protected `preprocess/go_ppi` + `preprocess/go_ppi_splits` re-hash vs `$T/pre_edit/protected_sha256.txt` → identical.

### Item 23: cross-feature integration (first fixer pass), 2026-10-10

`PY=/Users/olukotunjosh/Downloads/CDCL/.venv/bin/python`, `T=/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/task-go-ppi-withholding`. `review_go_ppi.json` was absent, so this is the integration pass. CPU only (Apple M2). Nothing committed; no real study artifact, pin, config or condition written (`preprocess/go_withholding/`, `results/go_withholding/`, `config/go_withholding/` all absent afterwards).

- `"$PY" -m py_compile` on all 49 new or modified `.py` files (`git diff --name-only HEAD` + `git ls-files -o --exclude-standard`, `pre_edit/` snapshots excluded) plus `check_additive.py` → OK.
- Import-coverage probe (in-process): imported the runner, `evaluate_withholding`, `sweep`, `analyze_withholding`, `lambda_pilot`, `audit_negatives`, `text_similarity_covariate`, `materialize_configs`, `make_conditions`, `regenerate_conditions`, `build_target_table`, `build_pruned_features`, `measure_leakage`, `withholding_diagnostics`, `go_relation`. `imported_repo_modules()` = 22 files, `code_sha256()[1]` = 219 files, uncovered = [] → **⊆ holds**.
- Seam audit, writer vs reader (read the code on both sides):
  - Runner `lc_manifest.json` → sweep `group_checks` / `_post_checks_ok` / `training_problem` / `rescore`: `withholding.{cell,cond_id,assertions.post_*,selected_checkpoint,selected_checkpoint_sha256}`, `seed`, `epochs`, `pilot`, `train_records`, `epoch_stats[].{epoch,ok_batches}`, `negative_selection.{selection_sha256_by_epoch,unannotated_anchor_calls_by_epoch,unannotated_anchor_window,selector_none_returns,selector_exceptions}`, `go_relation_loss.applied_batches` — all written with the names read.
  - Runner `lc_manifest.json` → evaluator `resolve_checkpoint` (`epochs`, `withholding.selected_checkpoint` + sha, `cond_id`): match.
  - Evaluator `metrics.json` (`per_target[go].{hard_auc,macro_query_hard_auc,macro_mrr}`, `collateral[pack].hard_auc`) and `relation.json` (`per_target[go].excess`, `global.spearman`) → `analyze_withholding.build_long_table` / `diagnostics_summary`: match (the `_grid_fixtures` stubs use the same keys).
  - `EpochDiagnostics` row `ratio_weighted_go_to_infonce.median`, `drift_flag` → analysis mechanism (b) / drift summary; sweep passes `--go-diagnostics <run>/go_diagnostics.jsonl`, the file analysis reads: match.
  - `audit_negatives` output (`cell`, `by_epoch[e].share_*`, file `neg__*.json` in `results/go_withholding/audits`) → analysis audit summary: match.
  - `text_similarity_covariate` `targets[go].tsim_excess` → analysis S4 / long table: match.
  - Pins written vs sweep `pin_preflight` reads: G1-leak `pass`; G2-covariate `text_covariate_sha256`; G2-cache `test_sha256`/`cache_npz_sha256`/`meta_sha256` with the `test_<sha16>` stem; G2-validation `manifest_sha256` (= sha of the bytes written as `manifest.json`) and `sha256s`; G3 `n_conditions` + `manifests[cond]` (= sha of the written `withholding_manifest.json`); G4-lambda `lambda_choice_sha256` at `wc.lambda_choice_path()` = the pilot's default `--out`; G4-configs `files` keyed `<cell>.json` + `configs_sha256_json`; G5 `code_sha256`: all match. Runner preflight 4 reads `lambda_choice["lambda"]`, which the pilot writes.
  - Singleton packing (AM1) through analysis/sweep: completeness denominator `n_targets × 6 × 5` (six k levels × five seeds, independent of pack size); the bootstrap's `n_packs` and `unit_pack` come from `table["packs"]` / `t["pack_id"]` (one target per pack); sweep asserts grid size = `table["n_runs"]` (121 / 2,420 at 24 targets). `audit_negatives.target_sets` reads `manifest["targets"][*].go_id`, which `build_condition` writes. No mismatch.
  - **No seam defect found; no production code changed.**
- Only edit: `go_ppi_domain/tests/test_withholding_integration_slow.py` test (a) now also feeds the four **real** runner `lc_manifest.json` files into `sweep.group_checks(..., require_epochs=1)` (expects no failures), `sweep._post_checks_ok` (True per cell) and `evaluate_withholding.resolve_checkpoint` (resolves `checkpoint_epoch_0.pt`, sha verified). This closes the gap that the sweep tests only used fixture manifests.
- Gate (final, after the edit):
  - `GO_WITHHOLDING_SLOW=1 "$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/test_withholding_integration_slow.py -q` → **3 passed**, 67.6 s (68.8 s wall). Before the edit: 3 passed, 68.1 s.
  - `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests -q` → **334 passed, 3 skipped** (20 pre-existing + 314 new; the 3 slow tests skip), 20.9 s.
  - The four career files → **28 passed**, 3.1 s.
  - `"$PY" "$T/check_additive.py"` → **exit 0** (batch_processor byte-identical; loss_engine 3 wraps; trainer 5 / data_structures 15 / go_ontology 106 inserted lines; negative_selector 2; user `+` lines 56 / 116 / 53 in order).
  - Protected `preprocess/go_ppi` + `preprocess/go_ppi_splits` (11 files) re-hash vs `$T/pre_edit/protected_sha256.txt` → identical. `preprocess/go_ppi_splits/test.jsonl` sha256 `c65a1d2ca0e5c16a…` = `setup_baseline.md`.
- Carried-over caveats (unchanged, not seams): `make_conditions.MIN_FREE_BYTES` / `sweep.MIN_FREE_BYTES` are the design's 3 GB against ≈ 4.4 GB of conditions; `code_sha256()` needs a git work tree on the remote.

### Review fix pass (review_go_ppi.json R1–R6), 2026-10-10

`review_go_ppi.json` existed (verdict CHANGES_REQUESTED), so this is a later pass. Read it and `review_go_ppi.md`, then fixed all six findings. CPU only (Apple M2). Nothing committed. No study artifact, pin, config or condition written: `preprocess/go_withholding/`, `results/go_withholding/` and `config/go_withholding/` are still absent, so no G5 pin exists yet, and the test-file edits land before it as R1 requires.

- **R1 (blocking), machine-path tests.** `_withholding_fixtures.py`: the `assert ROOT == /Users/...` is removed. `ROOT` stays `Path(__file__).resolve().parents[2]`. `test_code_pin.py` `ROOT`, `test_epoch_guard_and_recorder.py` `RANDWIN` and `test_go_withholding_loss.py` `PRE_EDIT_ENGINE` are now derived from `Path(__file__).resolve().parents[2]`. `test_code_pin.test_imported_repo_modules_filters` used to look torch up with `ROOT/.venv` rglob, and that failed with StopIteration on a relocated checkout. It now uses `importlib.util.find_spec("torch").origin`, the interpreter's own torch. `test_withholding_eval_metrics` checks `str(wc.ROOT)` is absent instead of the literal `/Users/`. `test_study_code_bakes_in_no_machine_path` now also scans `go_ppi_domain/tests/*.py` and checks for both `"/"+"Users/"` (built at runtime so the test does not match itself) and `str(ROOT)`. `grep -rl /Users/` over study modules, tests and scripts finds nothing.
  - **Relocation check.** rsync'd the tree to `/tmp/cdcl_reloc.*/CDCL` with the runbook's 1a excludes, then ran G0 from there with the repo's `.venv`. On the first try, 6 tests failed: the code-pin torch lookup (fixed above) and 5 pre-existing `test_go_ppi_wiring.py` tests. Those 5 read the reference data in `preprocess/go_ppi_splits`, which runbook 1a/1b did not copy. Runbook 1b now also copies `preprocess/go_ppi` and `preprocess/go_ppi_splits` (≈ 170 MB, read-only). After both fixes the relocated copy passed: **335 passed, 3 skipped** (21.3 s), and the slow suite **3 passed** (67.0 s). The temp copy was removed afterwards.
- **R2, broken fence.** Removed the stray ```` ```bash ```` before "Run from the local repository root:". There are 34 fence lines and they alternate open/close.
- **R3, disk guard.** `make_conditions.MIN_FREE_BYTES` 3 → **5 GB** (it also covers `regenerate_conditions`, which reads `mc.MIN_FREE_BYTES`), because 4.4 GB of conditions is more than 3 GB. `sweep.MIN_FREE_BYTES` stays 3 GB, with a comment giving the reason: the guard is checked every invocation, and one invocation writes ≤ 100 runs (≈ 0.32 GB). Design §3.3 and §12.3 each got a one-line post-AM1 implementation note, and the design §16 error-table row reads "disk < 5 GB (3 GB pre-AM1)". Runbook §3 and §7 now say: ≈ 12 GB free before regenerating, 5 GB regeneration guard, 3 GB per-invocation sweep guard, and check `df -h` between invocations. These are operational changes only: no condition, count or gate changed.
- **R4, cross-host claims.** Runbook §7 has a new "Crashed host (multi-machine only)" bullet. It covers the symptom, confirming the host is dead, and renaming each of that host's un-DONE/un-FAILED run dirs to the next free `<run_id>.interrupted<n>` (never deleting). It includes a loop that matches `RUNNING.json` `host`. Runbook §6's multi-machine sentence points to it.
- **R5, rename race.** `sweep.execute_run` now rechecks `run_state(run_dir)` right before renaming an existing dir. If the state is not `interrupted`, it logs a warning and returns `skipped` without renaming or claiming. `cmd_run`'s summary counts `skipped` separately and excludes it from `started`. Runbook §6 now states that concurrent invocations must use disjoint shards. New test `test_sweep_driver.py::test_execute_run_rechecks_claim_before_rename` takes a live same-host claim and checks for `skipped`, dir and claim untouched, no `.interrupted*`, and no trainer call.
- **R6, cookie file.** Runbook 1a rsync now has `--exclude '/300'`, with a comment. The relocation copy confirmed `300` is not shipped.
- Gate (final, in the repo):
  - `"$PY" -m py_compile` on the 9 edited `.py` files → OK.
  - `"$PY" -m pytest /Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests -q` → **335 passed, 3 skipped** (334 before, + the R5 test), 21.2 s.
  - The four career files → **28 passed**, 2.4 s.
  - `GO_WITHHOLDING_SLOW=1 "$PY" -m pytest .../test_withholding_integration_slow.py -q` → **3 passed**, 66.3 s.
  - `"$PY" "$T/check_additive.py"` → **exit 0** ("check_additive: OK"; go_ontology 106 inserted lines, negative_selector 2).
  - Protected 11 files: `shasum -a 256 -c $T/pre_edit/protected_sha256.txt` → all OK. `test.jsonl` sha256 `c65a1d2ca0e5c16a…` = `setup_baseline.md`.
  - Import-coverage probe (runner + `evaluate_withholding` + `sweep`): `imported_repo_modules()` 7 files ⊆ `code_sha256()[1]` 219 files, uncovered [] → holds. The `code_sha256` value changed because test files changed. No G5 pin exists yet.
  - HEAD is still `c911188`. Tracked modifications are the same 6 files as before; `contrastive_learning/{batch_processor,loss_engine,trainer}.py` were not touched in this pass.

### Final smoke step (items 24–26), 2026-10-10

Full details: `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/smoke_report_go_ppi.md`. CPU only (Apple M2). Nothing committed. No `.py` changed. `code_sha256` stayed `592125a3…` throughout.

- G0 before and after: GO suite 335 passed, 3 skipped; career 28 passed; slow suite 3 passed (64 s); `check_additive.py` exit 0; protected re-hash identical.
- Item 24 (G1): `build_pruned_features.py` exit 0 (11.6 s); `measure_leakage.py` exit 0 (9.9 s), `pass: true`. Test grade-2 a/b/c went 11.07/12.62/5.92% → 0.17/0.17/0%. Pins G1, G1-leak.
- Item 25 (G2–G4):
  - `build_target_table.py` exit 0: 461 → 153 → 51/51 → 12 pairs → 24 targets, 121 conditions, 2,420 runs.
  - `build-cache` exit 0. The first build picked MPS, so it was moved aside and rebuilt with `--device cpu`; the second G2-cache pin is the pinned one.
  - `text_similarity_covariate.py --device cpu` exit 0 (391 s).
  - Validation exit 0: V 330, 181 grade-2.
  - G3 exit 0: 121 conditions, 4.0 GB, 136 s.
  - `materialize_configs.py --lambda none` exit 0.
  - `lambda_pilot.py` exit 0 (1,004 s): r₀ 0.0441, then λ 0.5, r* 0.1538. It materialised the four configs itself, giving a single G4-configs pin.
  - Pins G2-targets, G2-cache ×2, G2-covariate, G2-validation, G3, G4-lambda, G4-configs. G5 not run.
- Item 26 (G6 smoke, p01_k0 × 4 cells × s13, 2 epochs):
  - All runner and score calls exit 0 (train 44–50 s each).
  - Identities hold; ok_batches 240/240 in every cell.
  - `check-group` exit 0 with GROUP_OK.json; crosscheck min cos 0.99999999999963 with 0 token mismatches.
  - Audits for all/p01_k0/p01_k5 exit 0, with 0 withheld negatives.
  - Statistics in-process: B 1000.
  - Sweep sharding (808/808/804, disjoint) and kill/resume (`.interrupted1`, retrain 292.5 s + eval 5.3 s, `DONE.json`; re-invocations no-op, byte-identical) ran via a scratch STUDY_RES with a scratch G5.
  - Leaked copy: runner exit 2, `run_error.json` reports `check 2: withheld protein in the train file`.
- Protocol §13 pins table mirrored (G1–G4-configs).
