# Smoke report: GO protein-function study (final local validation, 2026-10-12)

Local, CPU only (`--device cpu`, MPS never used). No git writes and no remote runs. The code review is APPROVED (`review_go_function.json`: 0 HIGH, 0 MEDIUM, 2 NITs). The previous smoke output was moved to `results/go_function/smoke.prev1` (nothing was deleted). The G9a smoke was then re-run from an empty `results/go_function/smoke/`.

## Gates passed locally

| Gate | Status | Key numbers |
|---|---|---|
| G0 regression | Baseline met earlier today; this run is short only because of the disk guard (see below) | `trials_domain/tests` 108 passed, 1 skipped. Career/CVE files 8 passed. `go_ppi_domain/tests` 326 passed, 3 skipped, 3 failed, 6 errors. Protected paths: 0 diff lines. |
| G1 labels + go.norm | pass | go.norm NF1 80,942 / NF2 11,841 / NF3 11,810 / NF4 19,594, 9 unparsed, owl:Nothing in 30 lines. Vocabulary BP 21,381 / MF 6,868 columns. Train labels: BP TP 1.92M (DGZ +30.4%), MF TP 308k (DGZ +22.3%). Every §2.4 check passes on every split. Re-runs are byte-identical. |
| G2 DGZ evaluator | pass (±0.005) | MF MLP / DGZ AUC 0.882 / 0.903, Fmax 0.657 / 0.657, Smin 9.857 / 9.808. BP MLP / DGZ AUC 0.793 / 0.798, Fmax 0.460 / 0.451, Smin 43.99 / 44.62. Max abs diff to the paper 0.0004. |
| G2b DGZ zero-shot | BP confirmed, MF reproduced (non-blocking) | L1 strict load. L2 BP 7/7, MF 5/5 exact. L3 BP 7/7 (X→t AUC ≥ zero AUC, within slack). |
| G3 checkpoints | pass (≤ 1e-4) | Strict load. Max abs diff: BP MLP 6.6e-7, BP DGZ 1.4e-6, MF MLP 8.3e-7, MF DGZ 3.8e-5. Norm = layernorm, written to `study.json`. |
| G5 InterPro (local part) | pass | 26,406 InterPro vocabulary, 0 unknown ids. Zero-InterPro train rows: BP 2.47%, MF 1.50%. |
| G6 targets + conditions | pass | 48 targets / 8 packs per aspect (amendment A1 packing). BP 138 conditions, MF 163. Every W1/W2 train mask is empty (closed world). Re-runs are "unchanged". |
| G9a smoke | pass | Details below. |

Pins in `results/go_function/protocol_pins.json`: G1-labels, G1-norm, G2, G2b, G3, G5, G6.

### G9a smoke (this run)

The smoke was BP p01 × k ∈ {0, 5} × {V0, V3, V3c}, seed 13, InterPro, TP/W2, 2 epochs.

- `sweep.py run --smoke --device cpu --workers 1 --max-runs 6`: exit 0, 6/6 done, 0 failed, 0 refused.
- Seconds per run (total / train), and peak RSS:
  - V0: 43.2 / 13.9 s and 39.9 / 13.2 s, at 1.7–2.8 GB
  - V3: 148.1 / 118.8 s and 142.6 / 113.4 s, at about 3.7 GB
  - V3c: 146.7 / 116.7 s and 129.2 / 100.1 s, at about 3.7 GB
  - Eval took about 27 s per run. Mean was 108 s per run; max RSS was 3.71 GB.
- Every run has `DONE.json`, `eval.json`, `scores.npz`, `run.json` and `epochs.jsonl`, with a grad-norm record in each epoch.
- No file under `results/go_function/smoke/` contains `/Users/`.
- `analyze.py --runs results/go_function/smoke/runs --out results/go_function/smoke/analysis --smoke`: exit 0.
  - Output: 60 long-table rows (36 run rows, 12 F rows, 12 DS rows), plus `fusion_F.csv`, `baseline_DS.csv`, `contrasts.json` and `decisions.json`.
  - Everything is labelled exploratory: this is a smoke, and 100% of the main block is missing.
  - All five analysis files are byte-identical to `smoke.prev1`.
- `GO_FUNCTION_SLOW=1 pytest go_function/tests`: 239 passed.

### G0 in this run

At this run about 3.9 GB of disk was free. `scripts/go_withholding/make_conditions.py` has its own 5 GB free-disk guard, and every one of the 8 extra non-passes stops at that guard ("only 3.9 GB free (< 5 GB)"):

- `test_make_conditions_counts[24-121]` and `[20-101]`;
- 6 errors in `test_regenerate_conditions.py`.

The known failure `test_real_protocol_amendments_table_parses` still fails, as in the baseline. Earlier today, at 5.01 GB free, the same suite gave the full baseline of 334 passed, 3 skipped and 1 known failure. `git diff HEAD` over the protected paths is empty.

To re-confirm, free at least 5.5 GB and run `.venv/bin/python -m pytest go_ppi_domain/tests -q -p no:cacheprovider`.

## Deviations and pending sign-offs

- **Smoke mode (plan decision 3).** G9a uses `max_train_batches` 40 per epoch for 2 epochs on CPU. Evaluation uses the full evaluation set. G9a tests the plumbing only. Its timings do not project the GPU grid (the 7.4-day "projection" in `smoke_report_interpro.json` is labelled as such). The full-epoch smoke is G9b on the L40S. This needs sign-off before G8.
- **D19: closed-world W2/W1. Awaiting user/supervisor sign-off before G8.**
  - Withheld positives become ordinary 0 labels, and withheld proteins get no loss mask. This is a deviation from spec §5 "Masking", and D9 is withdrawn.
  - The only masks left are the WDGZrel/W2drop whole-column masks and the target-free validation mask.
- **Review NITs** (open, non-blocking):
  - Pilot preflight builds the test feature rows and then discards them; nothing derived from them reaches any output.
  - The G0 re-check above.

## Left for the remote L40S (`REMOTE_RUN_GO_FUNCTION.md`)

1. Clone the committed branch. Rsync `protocol_pins.json` and `gates/`. Re-download, `pin_inputs.py --gate`, and rebuild labels, features, targets and conditions against the pins.
2. G5 remote:
   - `embed_esm.py --device cuda`: ESM-2 650M at a pinned revision, with the fp32 cross-check.
   - `embed_terms.py --device cuda`: MPNet term text; the 728 obsolete classes use their own stanza text, per A3.
3. G4: trainer reproduction (`gate_trainer.py`, 4 gate runs at ±0.01, seed fallback).
4. G7: pilots (E*, λ record, text arms) for bp/mf × esm2_650m/interpro.
5. G8: `materialize.py` (2,782 BP + 3,157 MF = 5,939 specs plus the audit), then `sweep.py pin` after the D19 sign-off and the commit.
6. G9b: full-epoch CUDA smoke, 13 runs. It stops with exit 3 if the grid projection exceeds 10 days.
7. G10: the grid in tmux (`sweep.py run --max-runs … --workers …`, shards, `status --release-stale`, `rescore`).
8. G11: `analyze.py --B 10000`, then the return tarball.

## Files for the user to commit

`results/`, `preprocess/` and `dataset/` are git-ignored and are not committed. `protocol_pins.json` and `gates/` ship by rsync.

- `go_function/`: 17 library files, `tests/` (20 files) and `tests/fixtures/` (`mini.obo`, `mini.norm`).
- `scripts/go_function/`: 19 files (CLIs plus `requirements-remote.txt`; `download_dgz.py` is included).
- `config/go_function/`: `dgz_published.json`, `input_hashes.json`, `study.json`.
- `.agents/tasks/ontology-withholding-feasibility/`:
  - `findings_go_function.md`, `setup_baseline_go_function.md`;
  - `design_go_function.md`, `design_review_go_function.{md,json}`;
  - `protocol_go_function.md`, `plan_go_function.md`;
  - `review_go_function.{md,json}`;
  - `REMOTE_RUN_GO_FUNCTION.md`, `smoke_report_go_function.md`;
  - `task-go-function/{task.json, context.json, features/FEAT-001..005.json}`.

Do not commit `__pycache__/`, `/300` or `reports/`. Three task docs from other studies also show as modified (`REMOTE_RUN_TRIALS_FUSION.md`, `protocol_go_ppi.md`, `protocol_trials_fusion.md`). They predate this study, so commit them separately if at all.
