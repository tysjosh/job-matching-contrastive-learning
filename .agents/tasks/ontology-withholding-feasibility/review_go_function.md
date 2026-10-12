# GO protein-function study: implementation of gap items 2–11 (code review, iteration 3)

The change implements the new GO-function direction end to end. It covers the typed ontology and go.norm loader, the TP / DGZ-rule / noIPI label builder (G1), the DGZ-faithful evaluator (G2), InterPro, ESM and term-text features, the target table and closed-world withholding generator (G6), the bilinear multi-label trainer with free, EL and text towers and checkpoint gates (G3, G2b), pilots, materializer, resumable sweep, analysis, the local G9a smoke and the remote runbook. This pass re-checks the ten findings from iteration 2 against the code and amendments A2–A5. All ten are fixed and tested: full code-pin hashing, pilots without test evaluation, refusals that use no attempt, single-factor C1, own-stanza text for obsolete classes, the DGZ-label sensitivity read, noIPI target filtering, boot-id claims and the G4 time estimate. The slow suite, re-run for this review, gives 239 passed. Two NITs remain.

Watch for:
- **Pilot preflight still loads test features (confirmed, NIT).** The InterPro and ESM loaders build the test split before the pilot branch returns. Amendment A4 and the code comment both say pilots load no test features. No test value reaches any output.
- **G0 left open (confirmed, NIT).** go_ppi_domain/tests has not been re-run on the final code at ≥ 5 GB free, yet the runbook lists G0 as done.

**Verdict**: APPROVED (0 HIGH, 0 MEDIUM, 2 NIT)

## High-level view

**Closed-world fix and gates.** The round-3 decision holds. W1, W2 and `all` carry an empty train mask, and the trainer builds no mask tensor for them. WDGZrel and W2drop mask whole columns only. Assertions 1–10 re-run at preflight. G1, G2, G2b, G3, G5 (InterPro) and G6 are pinned.

**Provenance and sweep.** Study-mode preflight now hashes every file in `code_pin.json`, so a post-pin edit to a module imported after preflight, such as `model.py`, is caught. The check runs again before `run.json`. A refusal now uses no attempt and stops the invocation. Three consecutive preflight failures also stop it, leaving a few runs `failed`, which the runbook explains how to re-queue.

**Pre-registration.** A5 gives V3c a separate `nf4_rad` index equal to V3's relation ids, so C1 contrasts relation-vector typing only. A4 removes test evaluation from pilots, but the preflight still builds the test feature split (NIT 1). A3 resolves all 728 obsolete classes to their own stanza text, which unblocks V5/V6.

**Runbook.** Runbook §0 lists G0 as done, but no passing run on the final code backs that (NIT 2).

<details>
<summary>Issues (2)</summary>

1. **NIT, pilot preflight loads test features** (`go_function/train.py:1013`, `:1025`, `:1070`; protocol §6.4 line 165, A4 line 311; `scripts/go_function/pilot.py:24`). `load_run_data` builds `X["test"]` (the InterPro CSR, or the ESM view) for pilot runs before the pilot branch returns. Nothing derived from it is written, but A4 and the comments say "loads no … test features". Fix: when `spec.mode == "pilot"`, load only the train and valid splits (`splits = ("train", "valid")` in both feature branches), and extend `test_pilot_preflight_never_loads_test_labels` to assert that no test feature rows are built. Or reword A4 and the comments to match the whole-file hash verification already allowed for labels.
2. **NIT, G0 not re-verified on the final code** (`task-go-function/task.json:15`, `REMOTE_RUN_GO_FUNCTION.md:16`). The last go_ppi_domain/tests run (review iteration 2) gave 326/3/3 failed/6 errors at the 5 GB free-disk guard, and only 4 GB are free now. The protected paths are byte-identical (0 lines of diff), so a regression is unlikely. The runbook still lists G0 as a completed local gate. Fix: free ≥ 5.5 GB, re-run `go_ppi_domain/tests` and expect 334 passed, 3 skipped, 1 known failure. Then close the task.json item, or mark G0 as pending in runbook §0.

</details>

<details>
<summary>Details</summary>

### Pilot preflight and the test split

`load_run_data` loads features before the mode branches. On the InterPro path, `X = {s: load_csr(z, s) for s in L.SPLITS}` (train.py:1013) builds the test CSR. On the ESM path, `esm_view` (train.py:1025) builds the test view. The pilot branch at 1068 then drops both. The label `.npz` loading is covered by A4's sentence on whole-file hash verification; the feature loading is not. In practice a pilot holds the test feature matrix in memory for no purpose, against text that says it never loads it.

### Verification evidence

- **Iteration-2 fixes.** All ten are present in code, each with a named test.
  - The study-mode hash of every code-pin entry is at train.py:337–346, and is repeated at :737 and analyze.py:639.
  - The refusal and preflight-stop handling is at sweep.py:412–417 and :476–492.
  - V3c's `nf4_rad` equals V3's relation ids through the shared `untied` order at normforms.py:239–259, and `assert_ladder` enforces it.
  - The real MF table has `noipi.kept` false for 2 of its 48 targets.
  - MF test protein lists and vocabularies are identical across labels_TP, labels_TPnoIPI and labels_DGZ, so the DGZ-label read aligns `eval_rows` for TPnoIPI runs too.

- **Tests.** `GO_FUNCTION_SLOW=1 .venv/bin/python -m pytest go_function/tests -q -p no:cacheprovider`: 239 passed in 28.7 s, which matches FEAT-005.
- **No git writes.** HEAD is still 6a8eda3. The last reflog entry dates from 2026-10-11 09:17, before this task. There are 0 stashes and nothing is staged.
- **Protected paths.** `git status --short --untracked-files=all` and `git diff HEAD --stat` over all seven protected paths print 0 lines. The three modified tracked task docs date from Oct 10–11 and belong to other studies.
- **No MPS.** `DEVICES = ("auto", "cuda", "cpu")`, and `resolve_device` never returns mps.
- **`--device` on every torch CLI.** It is present on run, sweep, pilot, embed_esm, embed_terms, gate_checkpoints, gate_dgz_zero and gate_dgz_eval. `analyze`, `materialize` and `gate_trainer` do no tensor compute.
- **No absolute paths.** `grep -rlaF '/Users/'` over `results/go_function`, `preprocess/go_function`, `config/go_function` and the dataset manifests finds no match.
- **G9a smoke.**
  - Each of the 6 runs has DONE.json, eval.json, scores.npz, run.json and epochs.jsonl.
  - The V3c runs were re-run at 08:50–08:52, after the last code edit at 08:47; the V0 and V3 runs predate the fixes, but their code paths are unchanged by A5.
  - `code_at_finish` is recorded on the re-run V3c runs.
  - The assertions re-run per run are [1, 2, 4, 5, 8, 9, 10], the set that applies to TP/W2.
  - Per-tower grad-norm records are present.
- **Pins.** G1-labels, G1-norm, G2, G2b, G3, G5 and G6 are present.

</details>

<details>
<summary>File map</summary>

- `go_function/{constants,runtime,pins,vendored}.py`: shared constants, no-MPS device resolution, repo-relative paths, pins and write-or-verify, vendored PPI helpers
- `go_function/ontology.py`, `normforms.py`: typed go.obo parser with a DgzRule replica and obsolete-stanza text (A3); go.norm loader, the V1–V6 ladder and V3c `nf4_rad` (A5)
- `go_function/labels.py`: TP/DGZ/TPnoIPI label builder and the G1 checks
- `go_function/evaluate.py`, `features.py`: DGZ-faithful and study evaluators; InterPro, ESM and term-text features (obsolete text, A3)
- `go_function/targets.py`, `withholding.py`: target table and packing; closed-world conditions and assertions 1–10
- `go_function/model.py`, `train.py`: bilinear model, EL losses and DGZ checkpoint loading; spec schema, preflight (full code-pin hashing), trainer, pilot no-test path (A4), noIPI target filter, markers
- `go_function/{fusion,baselines,stats}.py`: F, DS, and the bootstrap/Holm/classification layer
- `scripts/go_function/*.py`: CLIs for downloads, pins, builders, embeddings, gates G2/G2b/G3/G4, run, pilot, materialize, the sweep (refusals, preflight stop, boot-id claims) and analyze (DGZ-label sensitivity)
- `config/go_function/{study,input_hashes,dgz_published}.json`: frozen config, input hashes and published DGZ values
- `go_function/tests/`: 18 test modules and fixtures (239 tests with SLOW)
- `.agents/tasks/ontology-withholding-feasibility/{design,protocol}_go_function.md`: v4 plus amendments A2–A5; `REMOTE_RUN_GO_FUNCTION.md`: runbook
- Full diff: `git status --short` plus the untracked paths above; every study file is new, and no tracked file of this study is modified

</details>
