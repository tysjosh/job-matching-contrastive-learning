# TREC-CT MeSH fusion confirmation: implementation review, pass 2

This pass reviews the fix round on the implementation of protocol v7. The round added the §9 A0 amendment, the dev-local restriction in `score.py`, and a Mac-side code manifest (`dump_code_hashes.py` plus `verify_transfer.py --code-manifest`, enforced by `make_pins.py`). It also added the 39-patient descriptive subset and disclosure block in `analyze.py`, plus tighter pins-chain and score-manifest checks. All seven pass-1 findings are resolved (confirmed). The estimator, fold rules, derangement and decision logic are unchanged from pass 1, and the 42 unit tests pass (re-run here). No protected file changed, and nothing under `results/` or `preprocess/` is newer than the pass-1 review, so the fix round added no local exposure.
Watch for: the post-approval A0 edits to the protocol are not hash-anchored, and §8 still states the pre-A0 exposure counts (confirmed). Transfer step 1a ships about 1.77 GB, 1.28 GB of it `cve_domain/`, which the runbook neither excludes nor sizes (confirmed). None of these blocks the run.

**Verdict**: APPROVED

## High-level view

Protocol adherence is unchanged from pass 1. That covers the hypotheses, C1–C3, the ordered C2 states, S4 TOST, Holm with ties going to H1, the 5-label rule, B = 2000 with CRN and `[31415, year]`, and the fold rule. Weight selection reads only validation arrays. `analyze.py` aborts on any recomputed-w mismatch.

The A0 disclosure matches the files on disk. The dev-local npz files hold 15 validation topics. A0's 13 are those 15 minus 2021_27 and 2021_64. The 52-patient `unexposed_closed` keeps its Holm decision and warning. The 39-patient subset is descriptive only, and both disclosures print in every `report.md`.

Revision-1 pins now require a passing code-tree check against the Mac manifest. Tracked modules outside `CODE_GLOBS` (for example `go_ppi_domain/withholding_constants.py`) are covered by `pins.git` (HEAD plus dirty paths).

The remaining gaps are bookkeeping. The approved v7 text has no recorded hash, so the A0 edits can't be diffed against it. §8 still says 52 patients are unscored. The 1a rsync ships a 1.28 GB tree that the study never reads.

<details>
<summary>Issues (4)</summary>

1. **Post-approval protocol edits not hash-anchored** (LOW, confirmed). The orchestrator froze v7 by name, not by sha256, and v7 was never committed. HEAD `b524f2a` holds an older draft. The A0 edits to §0, §4.4, §6.2 and §9 can only be audited from prose. Commit the protocol, or record its sha256 in A0, before the 1-0 manifest dump, so `pins.git` ties revision 1 to a commit.
2. **§8 exposure counts predate A0** (NIT, confirmed). §8 still reads "52 of its 75 patients have neither been scored under fusion… 60 have not themselves been scored". Add the A0 pointer used in §0 (39 and 47 after the dev smoke).
3. **1a ships `cve_domain/` (1.28 GB) and `.git` (0.49 GB)** (NIT, confirmed). A dry run of the 1a command lists 2,154 files totalling 1.77 GB, but the runbook gives sizes only for 1b and 1c. Add `--exclude '/cve_domain/'` to 1a and state the expected size.
4. **Code manifest is optional at step −1** (NIT, confirmed). `verify_transfer.py` passes without `--code-manifest` and only warns, and `make_pins.py` refuses the result one step later. Make the flag required outside dev mode so the failure surfaces at step −1.

</details>

<details>
<summary>Details</summary>

## Pass-1 findings: resolution check

For finding 1, `make_pins.py` pins `fc.DEV_SMOKE_EXPOSED_2021` as `cohort2021.dev_smoke_exposed_topics`. It aborts if that set overlaps the rebuilt `exposed_topics_closed`, which catches a remote fold build whose components drift. In `analyze.py` the mandatory exposure warning still reads only `unexposed_closed`, so the 39-patient subset cannot change a label.

For finding 4, the manifest sits under `.agents/`, outside `CODE_GLOBS`, so it does not hash itself. `make_pins.py` re-checks the manifest's sha256 against the value in `transfer_check.json`, which refuses a manifest swapped after step −1. The manifest does not exist yet. Runbook step 1-0 generates it right before the rsync, so any code edit after that point fails the check.

For finding 3, a 2021 dev smoke on the real folds is now impossible. Every outer fold has a side whose restricted topic set can't be deranged, so any further smoke needs a synthetic fold.

## Protocol provenance after approval

The current protocol differs from HEAD by 577/180 lines, because v7 itself was never committed. A0 lists the sections it changed, and the tables the parser reads are unchanged (15 checkpoint rows, 18 input rows). Even so, nobody can check mechanically that these were the only post-approval edits. The edits add disclosures and one descriptive subset and change no hypothesis, threshold or decision rule, hence LOW. Revision 1 pins whatever protocol arrives on the remote. Committing before the transfer makes that state recoverable.

§8's limitation paragraph missed the A0 pointers. It still reports 52 closed-unexposed patients and 60 never scored. After A0 the counts are 39 and 47. The error is only in the prose. The analysis code uses the pinned sets.

## Transfer footprint

The 1a rsync excludes the large data trees but not `cve_domain/` (1.28 GB). With `.git` (0.49 GB), 1a comes to about 1.77 GB, more than three times the 560 MB quoted for 1b. The study never imports `cve_domain`, so the cost is transfer time only. A dry-run listing of 1a shows no cookie, key or `.env` file. `/300`, `reports/` and `results/` (including `dev_local/`) are excluded.

## Test coverage

Pass 2 added five tests: the dev-root chain rule, the code-hash comparison (missing, extra, differing), the dev-local restriction, the A0 topic set, and the `dump_code_hashes` hash-seed guard. Not tested: the revision-1 `make_pins.py` path on real data with the code-check requirement (only a toy root exercises it); `verify_transfer.py` run without `--code-manifest`, followed by `make_pins.py` refusing it; and the CUDA path, which can only run on the remote.

</details>

<details>
<summary>File map</summary>

- `trials_domain/fusion_confirm.py`: A0 constants and disclosures, `compare_code_hashes`, dev-root rule in `load_verified_pins`, the new analysis param.
- `scripts/trials_fusion_confirm/score.py`: `dev_local_restrict` for validation and test.
- `scripts/trials_fusion_confirm/analyze.py`: 39-patient subset, disclosure section, score-manifest pins check.
- `scripts/trials_fusion_confirm/make_pins.py`: code-check requirement, dev-smoke topic pin and overlap guard.
- `scripts/trials_fusion_confirm/verify_transfer.py`: `--code-manifest`.
- `scripts/trials_fusion_confirm/dump_code_hashes.py`: new, Mac-side manifest.
- `trials_domain/tests/test_fusion_confirm.py`: five new tests.
- `protocol_trials_fusion.md`: §9 A0, pointers in §0/§4.4, `--code-manifest` in §6.2.
- `REMOTE_RUN_TRIALS_FUSION.md`: step 1-0, A100, `dev_local/` note, code check at step −1.
- `verify_trials_fusion.md`: fix-pass table, corrected exposure statement.

Full diff: `git status` / `git diff HEAD` in the repo root (study files are untracked).

</details>
