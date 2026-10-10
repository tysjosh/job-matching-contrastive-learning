# GO/PPI per-target withholding study: review fix pass (R1–R6)

This is the second review pass over the GO/PPI withholding study (FEAT-002 to FEAT-004 plus the integration pass). The first pass approved checks 1, 2, 4, 5 and 6 and blocked on check 3: the GO test suite hard-coded the local repo path, so G0 would fail on the A100. The fix pass resolves that blocker and all five non-blocking findings. Only the test files, `sweep.py`, `make_conditions.py`, the runbook and two design notes changed. The coder ran a relocation test: it rsync'd the repo to `/tmp` with the runbook's excludes and ran G0 there (335 passed, 3 skipped; slow suite 3 passed). That directly shows the remote-portability fix works. Nothing was committed (HEAD is still `c911188`), and no study artifact or pin exists yet.

Watch for: no blocking issues. Two small leftovers (both confirmed, both non-blocking). The rename race between overlapping invocations is narrowed, not closed, and the runbook says so. The design header still names the local protocol path, which is fine because it is prose, not a config or pin.

**Verdict**: APPROVED

## High-level view

The R1 machine-path fix is complete. `_withholding_fixtures.py` drops the root assert, and `ROOT`, `RANDWIN` and `PRE_EDIT_ENGINE` all come from `Path(__file__).resolve().parents[2]`. The code-pin test now finds torch with `importlib.util.find_spec` instead of searching `.venv`. `test_study_code_bakes_in_no_machine_path` now also scans `go_ppi_domain/tests/*.py`, for both a runtime-built `/Users/` marker and `str(ROOT)`. Searching `go_ppi_domain`, `scripts` and `contrastive_learning` finds `/Users/` only in the pre-existing `scripts/add_bridge_edges.py`, which is out of scope. The relocation run found one more gap the first review missed: `test_go_ppi_wiring.py` reads `preprocess/go_ppi*`. Runbook step 1b now copies that reference data. The fix lands before G5, so the code pin will hash the corrected tests.

The runbook and the sweep-concurrency fixes are in place. The section 1 fences now pair correctly (30 fence lines). rsync 1a excludes `/300`. Section 7 has a manual recovery for a crashed host: rename its run dirs to `.interrupted<n>` and never delete them. `execute_run` rechecks `run_state` right before the rename and returns `skipped` if the state is no longer `interrupted`. `cmd_run` counts `skipped` separately, and a new test covers a live claim. The disk guards now match the grid. `make_conditions`/`regenerate_conditions` refuse below 5 GB. The sweep's 3 GB guard stays, because one invocation is capped at `MAX_RUNS_CAP = 100` runs (≈ 0.32 GB). Design §12.3 and §16 record this, and the runbook asks for about 12 GB free and a `df -h` check between invocations.

The checks from the first pass still hold. AM1 and the 24 / 121 / 2,420 counts are unchanged. Packing is singleton. `check_additive.py` exits 0, and the fix pass did not touch `batch_processor`, `loss_engine` or `trainer`. The weight-zero bit-identity tests are unchanged. A spot-check confirms the 11 protected files still match `protected_sha256.txt`.

<details>
<summary>Issues (2)</summary>

1. **Residual rename race (non-blocking, confirmed)**: there is still a small window between `execute_run`'s `run_state` recheck and `os.replace`. The runbook acknowledges it and allows only disjoint shards to run concurrently. No action needed unless the user plans overlapping unsharded runs.
2. **Local path in design prose (non-blocking, confirmed)**: line 3 of `design_go_ppi.md` cites the protocol as `/Users/.../protocol_go_ppi.md`. It is a document reference, not a config, manifest or pin, so portability is unaffected. Optionally make it repo-relative the next time the design is edited.

</details>

<details>
<summary>Details</summary>

## Relocation evidence for G0

The first pass flagged the hard-coded paths only by reading the code. The fix pass checked the behaviour: it copied the repo with the runbook's 1a excludes and ran the full GO suite from the copy. The first relocated run failed 6 tests. One was the code-pin torch lookup through `ROOT/.venv`. The other five were pre-existing `test_go_ppi_wiring.py` tests that need `preprocess/go_ppi_splits`, which the copy list didn't include. Both are fixed now: `find_spec("torch")`, plus `preprocess/go_ppi preprocess/go_ppi_splits` in rsync 1b. The relocated copy then passed. Those two directories are protected and read-only. Copying them to the remote doesn't change them locally, and their hashes are still identical.

## Sweep claim recheck

```python
if run_dir.exists():
    state = run_state(run_dir)
    if state != "interrupted":
        logger.warning("%s: %s since selection; not renamed, skipped", rid, state)
        return "skipped"
    dst = next_free(run_dir, INTERRUPTED)
    os.replace(run_dir, dst)
```

A run that another process claimed or finished after selection is now left alone. The recheck and the rename are still two separate steps, so two non-disjoint invocations can in principle both see `interrupted` and race. Runbook §6 now says so and requires disjoint shards, which matches how the shard design is meant to be used.

</details>

<details>
<summary>File map (fix pass)</summary>

- `go_ppi_domain/tests/_withholding_fixtures.py`: root assert removed
- `go_ppi_domain/tests/test_code_pin.py`, `test_epoch_guard_and_recorder.py`, `test_go_withholding_loss.py`: paths derived from `__file__`; torch found through `find_spec`
- `go_ppi_domain/tests/test_regenerate_conditions.py`: the no-machine-path test also covers tests and `str(ROOT)`
- `go_ppi_domain/tests/test_withholding_eval_metrics.py`: checks for `str(wc.ROOT)` instead of the literal path
- `go_ppi_domain/tests/test_sweep_driver.py`: new `test_execute_run_rechecks_claim_before_rename`
- `scripts/go_withholding/sweep.py`: recheck before rename, `skipped` count, guard rationale
- `scripts/go_withholding/make_conditions.py`: `MIN_FREE_BYTES` raised to 5 GB
- `REMOTE_RUN_GO_PPI.md`: fence fixed, `/300` excluded, 1b copies the reference data, crashed-host recovery, disk and disjoint-shard notes
- `design_go_ppi.md`: post-AM1 disk notes in §3.3, §12.3 and §16

Full diff: `git diff HEAD` plus `git ls-files -o --exclude-standard`.

</details>
