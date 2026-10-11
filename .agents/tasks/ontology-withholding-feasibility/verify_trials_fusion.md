# Verification: implementation of `protocol_trials_fusion.md` (v7, frozen, plus §9 A0)

This note covers the implementation on the Mac (Apple M2, CPU only). It has two passes:
- **Pass 1** (implementation) built everything below.
- **Pass 2** (fix pass) addressed all 7 findings in `review_trials_fusion.md` (see the first section).

What was not run here:
- confirmatory training;
- a confirmatory `pins.json`;
- any confirmatory 2021-CV or new-seed result.

Local outputs are non-confirmatory and sit under `results/trials_fusion_confirm/dev_local/`. No confirmatory script reads them. Only new files were created (`git status`: untracked study files). The protocol is the one tracked file that changed; its diff is the pre-approval v7 text plus the §9 A0 entry from pass 2. Nothing was committed, per the step's git rules.

## Fix pass: review findings → changes

| # | sev | change |
|---|---|---|
| 1 | MEDIUM | Protocol **§9 A0** names the 13 fold-1 closed-unexposed topics the smoke scored under fusion: 2021_7, 9, 12, 14, 29, 31, 32, 35, 43, 46, 47, 54, 67. The same edit adds pointers in §0 and §4.4. The preregistered 52-patient `unexposed_closed` and its warning are **kept unchanged**. A **descriptive** 39-patient subset `unexposed_closed_minus_dev_smoke` is added. It has no Holm decision and no warning. Code: `fc.DEV_SMOKE_EXPOSED_2021`, pinned by `make_pins.py` as `pins.cohort2021.dev_smoke_exposed_topics`. `make_pins.py` aborts if those topics overlap `exposed_topics_closed`. `analyze.py` computes the 39-patient subset, and `pins.analysis.descriptive_unexposed_closed_minus_dev_smoke` records it. The pass-1 "one value" statement is corrected below. |
| 2 | MEDIUM | §9 A0 discloses the 2022 C2 preview: the derangement C1–C3/C2 path ran on 15 of the 19 C2_2022 checkpoints. `analyze.py` writes both A0 disclosures into `analysis.json` (`disclosures`) and into a "Prior-exposure disclosures (§9)" section of every `report.md`. |
| 3 | LOW | `score.py --dev-local` now uses `dev_local_restrict`. It restricts **both** the inner-validation and the test file to `exposed_topics_closed`. It aborts before any scoring if a restricted file is empty or cannot be deranged. I checked the real folds: under the restriction, every outer fold has at least one infeasible side (k0 val {27, 64}; k1 test {27, 64}; k2 val {16, 25, 61}; k3 both; k4 test {40, 74}). So further 2021 smokes need a synthetic fold. |
| 4 | LOW | New `scripts/trials_fusion_confirm/dump_code_hashes.py` writes the Mac-side `fc.code_hashes()` manifest, `.agents/…/code_hashes_trials_fusion.mac.json`, which rsync 1a ships. `verify_transfer.py --code-manifest` fails on any missing, extra or differing code file (`fc.compare_code_hashes`). `make_pins.py` revision 1 requires a passing code check in `transfer_check.json`, checks that the manifest's sha256 is unchanged, and re-compares the current tree before it records `pins.code`. The command was added to protocol §6.2 and the runbook. |
| 5 | NIT | The runbook now says the planned host is the user's A100. |
| 6 | NIT | §9 A0 records the change of smoke directory, from `results/trials_fusion_confirm_smoke/` to `results/trials_fusion_confirm/dev_local/`. The runbook points to it. |
| 7 | NIT | `load_verified_pins` accepts a `dev_pins.json` root only when both the root and the head are non-confirmatory. `analyze.py` aborts unless `score_manifest.pins == {path, sha256}` of `--pins`. |

New gate-1 tests (pass 2):
- `test_pins_chain_dev_root_only_for_non_confirmatory_head`
- `test_compare_code_hashes_missing_extra_differing`
- `test_dev_local_restrict_covers_validation_and_test`
- `test_dev_smoke_exposed_topics_are_closed_unexposed` (13 topics, disjoint from the 23, 39 remain)
- `dump_code_hashes` added to the `PYTHONHASHSEED` guard test

The `make_pins.py` toy root now includes a passing code check.

Not re-run in pass 2:
- **The real-data dev smoke.** A 2021 dev smoke is now refused by design (finding 3). Rerunning the 2022 dev analysis would repeat the C2 preview for no gain.
- **`analyze.py` end to end.** It cannot run on the old `dev_pins.json` anyway, because the code hashes changed (the code pin works as intended). I checked the changed code with `py_compile` and a `render_report` call that shows the disclosure section.
- **`verify_transfer.py`.** I did run it on the Mac to `/tmp` with a freshly dumped manifest: PASSED, 15 checkpoint rows, 18 input rows (the A0 edit adds none), 36 chunks, code check ok on 56 files. Afterwards I deleted both the manifest and the output. The user must regenerate the manifest right before the rsync.

## Pass 3 (non-blocking items from review pass 2, APPROVED)

- `verify_transfer.py`: `--code-manifest` is now required, and a missing manifest file aborts. New test `test_verify_transfer_requires_code_manifest` (43 tests pass).
- Runbook 1a also excludes three untracked `cve_domain/runs` outputs. The 1a dry run is now 2,546 files and about 520 MB, most of it the `.git` pack.
- Protocol §8 has the A0 pointer (39 and 47). A0 states that the §0/§4.4/§8 pointers were the only other edits to §0–§8. The parser still finds 15 checkpoint rows and 18 input rows.
- The study files are committed before the transfer, so `pins.git` ties revision 1 to that commit.

## Files created

Library:
- `trials_domain/fusion_confirm.py`:
  - guards and repo-root handling;
  - hashing and the manifest digest;
  - the read-only loader;
  - the three φ variants and the group-aware derangement;
  - z, fusion and weight selection;
  - per-patient metrics, the bootstrap, Holm, S4, C2, labels and the headline;
  - paraphrase grouping, closure and folds;
  - the gate-2 classifier and comparator, and gate 3;
  - the protocol parser, the pins chain and the ledger;
  - the environment record;
  - code hashes and their comparison;
  - the §9 A0 constants.

Scripts (`scripts/trials_fusion_confirm/`):
- `verify_transfer.py` (with `--code-manifest`)
- `build_cv2021.py`
- `make_pins.py`, including `--revise`
- `train.py`: resumable driver with `--device auto|cuda|cpu`
- `score.py`: export plus `--gate-reproduce`
- `analyze.py`
- `dump_code_hashes.py` (pass 2)
- `requirements-remote.txt`

Tests:
- `trials_domain/tests/test_fusion_confirm.py`: 42 tests, counting the parametrised guard cases.

Docs:
- Runbook: `REMOTE_RUN_TRIALS_FUSION.md`
- This note.

## Design-review round-7 items (pass 1, unchanged)

**MEDIUM: gate-2 leaf matching.**
- Leaf paths are key tuples relative to `runs[i]`.
- `per_query` rows are paired by `query`. A one-sided or duplicate query fails.
- Classes use any-segment membership; `skipped` is checked first.
- Coverage: a missing non-skipped leaf fails.
- Tested on full key tuples from the real published JSON.

**NIT: anchored row regexes.**
- Inputs rows have exactly 2 cells; checkpoint rows have exactly 4.
- The parser needs 15 checkpoint rows and at least 18 input rows; the two row sets are disjoint.
- A truncated copy aborts.

## Tests run (pass 2)

| suite | result |
|---|---|
| `trials_domain/tests/test_fusion_confirm.py` | 42 passed |
| `trials_domain/tests` (all) | 106 passed, 1 skipped |
| `go_ppi_domain/tests` | 335 passed, 3 skipped |

Command: `PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest …`.

Pass 1 noted 7 failures when both suites run in one process: the `orca/*` import-pin checks. They are a cross-suite import-order effect that predates this work; each suite passes alone.

## Gate-2 code check (pass 1, NON-CONFIRMATORY, Mac CPU)

Output: `dev_local/gate_reproduction_codecheck.json`. Result: **PASS** on all 15 published checkpoints.
- `selected_weight` matched exactly.
- Maximum absolute difference: rank 0, exact 0, continuous 2.6e-18.
- No leaf was missing.
- Zero-shot macro pooled AUC was 0.7952.

Exploratory restricted-grid estimate:

| metric | Δ | 95% CI |
|---|---|---|
| pooled AUC | +0.0126 | [+0.0064, +0.0202] |
| AP | +0.0168 | [+0.0062, +0.0293] |

This reproduces the published +0.013 and +0.017. Gate 2 proper runs on the remote.

## CPU smoke of the driver (pass 1, NON-CONFIRMATORY)

All outputs are under `dev_local/`:
1. **`build_cv2021.py`.** All asserts passed: the 3 added edges, the 5-topic component, the 23-topic `exposed_closed`, and 5×15 folds.
2. **`make_pins.py --dev-local` → `dev_pins.json`.** Marked non-confirmatory.
3. **`train.py --dev-local`** (`2021:k0_s13`, CPU, 2 epochs). Gate 3 passed and resume gave `skipped-passed`. A stale code pin was refused.
4. **`score.py --dev-local`** on 2022 and on 2021 fold 0.
5. **`analyze.py --dev-local`** (B = 200). It ran the full decision path. Its numbers are not results.

**Exposure from this smoke (corrected in pass 2; pass 1 understated it as "one value"):**
- **2021.** The test side was restricted to `exposed_topics_closed`, but the fold-1 inner-validation file was scored whole. `dev_local/scores/2021/fold_0/selected_w.json` holds the full validation nDCG@10 curves over w ∈ {0..0.5} for hierarchy, exact and shuffled φ. They cover zero-shot (deterministic, so they equal the confirmatory values) and the 2-epoch smoke model, on 15 patients. 13 of those patients are closed-unexposed (listed above). Fusion lowered zero-shot validation nDCG@10 there, and w = 0 was selected. These are validation-side outcomes (grade 0 capped at 60), not test-file metrics, but they concern the same patients.
- **2022.** The dev analysis computed C1–C3 and the C2 state, with the new derangement, on the 15 published checkpoints.
- Both are disclosed in §9 A0 and in every `report.md`.

## Implementation choices (protocol silent or ambiguous; pass 1)

- **`validation_positive.jsonl`** follows the `validation.jsonl` order and is not reshuffled. I followed the file rule over the shuffle-seed list.
- **Bootstrap draws.**
  - C/P are drawn once per cohort and reused (common random numbers).
  - Zero-shot reuses P when its patient set is the same.
  - Each subset sensitivity uses a fresh generator with the cohort seed; this includes the new 39-patient subset.
  - The fraction-stratified c* uses `[31415, 2022, 1]`.
- **Inverse-variance pooling** is undefined when a cohort's bootstrap SD is 0.
- **P@10** keeps `epf`'s 0 for a patient with no grade-2 trial.
- **`make_pins.py`** probes the GPU in a child process with CUDA visible, and refuses to run without a GPU.
- **`changed_fields`** granularity is two levels deep.
- **`verify_transfer.py`** writes a failed check to `transfer_check.FAILED.<utc>.json`.
- **`train.py`** refuses `--device cuda` when `CUDA_VISIBLE_DEVICES=""`. It never shards runs; `--max-runs` advances the grid in chunks.
- **Dev-only flags** are refused without `--dev-local`.
- **Device policy.** Training uses CUDA; everything else asserts CPU. Fresh encodes use `SentenceTransformer(<checked snapshot>, device="cpu")`.

## Not verified here

- The CUDA path: `--device cuda` training, gate 3 with a real `cuda_device_name`, and the GPU probe.
- The confirmatory `make_pins.py` on real data, including the new code-check requirement. Only the toy root exercises it.
- Remote-vs-Mac BLAS differences.
- The fresh-encode path on real data.
- Remote wall times.
