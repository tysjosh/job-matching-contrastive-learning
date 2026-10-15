# Inference-time GO-structure fusion for protein function (S1 Tier A + S2 dry-run): code and Mac-run review, round 2

The change adds the fusion study package (`go_function_fusion/`, `scripts/go_function_fusion/`, `config/go_function_fusion/`) for protocol v3 (sha256 `53325461…a9b8`, matches the file on disk). It implements AP-T, AP-T+H, the protein-centric finalist PC, the typed contrast AP-T+R, every §3.5 control, the §8 bootstraps, Holm and the §12 readings. A Mac dev run (G-0, G-R(b/c), G-V, G-F, G-S2dry) wrote only to `results/go_function_fusion_dev/`; G-R(a) moved to the remote by user decision. This pass checks the round-1 fixes and re-reads the confirmatory code paths. All three round-1 MEDIUM findings are fixed in code, and no new leakage, selection-on-test or metric problem turned up.

Watch for: only NITs remain. The protein-bootstrap P_b stream is still drawn with n = evaluated test rows while the Fmax draws use n = protein-table rows (descriptive only; confirmed). The runbook calls G-R(a) "optional here" although the protocol gates on it (confirmed). The recorded G-0 count (41) predates the three tests added with the fixes; a spot-check of `test_guard.py` passed 10/10.

**Verdict**: APPROVED

## High-level view

The class-centric core matches §3 and §7. z_t runs per term in float64 over the scored split's evaluated proteins, with population sd and constant columns set to 0. w_t = c·τ/(τ + n_train) uses TP train counts. Rank AUC uses average ranks, AUPR uses sklearn's step rule, and bands and the pooled [>100] scarcity term follow §8. The grid builders list cells in tie-preference order: smaller c then larger τ for AP-T, smaller c for fixed-w, smaller w for w_H/w_R, c→τ→α for PC, 0.5 first for the X+DS sensitivity, and the null cell first for controls. With that order, the 1e-12 tolerance selection applies every protocol tie rule.

Every control is built in the form of the fusion it controls. Shuffled graph, CoA, CoA⁺, Pa_any, Desc-max and protocol F reuse AP-T's form. Shuffled Reg uses AP-T+R's, shuffled hits uses AP-T+H's (and PC's in the PC version), and PC shuffled graph puts the permuted prior inside σ. Each is scored at the frozen cell and re-selected over the grid plus the null cell. Only frozen scorings feed D_struct, D_typed and the co-annotation reading. CoA excludes TP ancestors and descendants; CoA⁺ does not.

The search-mode guard is an audit hook on every `open` of the process. In search mode it blocks test pickles, released predictions, `test_diamond.res`, the train-only test Diamond output, `results/go_function_fusion*/test/`, everything in `tp_bases/` except `checkpoint.pt` and the provenance file, and GF run outputs. The label loader returns only train and valid. The dev search log has no `guard_violation` event, finalists cannot be frozen after a test directory exists, and `score_test.py` refuses on macOS.

The round-1 remote gaps are closed in code. The guard covers Tier B source (b) test outputs, and the checkpoint allow-list call is limited to `results/go_function/`. The `--diamond-valid` rerun builds w_H at the pinned G-F (c*, τ*), and G-F2 refuses otherwise. `verify_tp_bases.py` implements G-TP, including the GF code-pin comparison, the §10.4 listings and the two-call determinism check.

<details>
<summary>Issues (6)</summary>

1. **Protein-bootstrap stream sizes** (NIT, confirmed) — `score_test.py` draws (D_f, D_p) with n = len(rows), `report.py` redraws D_f with n = protein-table rows; draw once from the table n and reuse it, or record both n in `meta_<aspect>.json`.
2. **G-R(a) labelled optional** (NIT, confirmed) — runbook §2 says "optional here", but protocol §9 gates on it and the smoke report schedules it before G-V2; call it required before G-V2.
3. **G-0 evidence predates the fixes** (NIT, confirmed) — the smoke report records 41 tests, the suite now has 44; rerun G-0 when the confirmatory G-F is redone on the Mac (runbook §0) and record the count.
4. **§15 file-list drift** (NIT) — the protocol lists `diamond_valid_test.sh` and `dump_code_hashes.py`; the code has `diamond_valid.sh` and `make_pins.py code`. Align the names in the next protocol touch.
5. **Dev code pin predates the fixes** (NIT) — `results/go_function_fusion_dev/code_pin.json` and its G-F pin do not hash the current code; dev only, and the confirmatory G-F must be written after the last edit.
6. **PC shuffled-hits re-selection scope** (NIT) — re-selected over α only, at the frozen (c_PC*, τ_PC*); state this in the report text (descriptive control).

</details>

<details>
<summary>Details</summary>

### Round-1 fixes, as implemented

`guard.is_test_path` now matches `results/go_function_fusion*/tp_bases/*` (fnmatch `*` crosses `/`) and blocks every file there except `checkpoint.pt` and `tp_base_provenance.json`. `test_tier_b_source_b_outputs_blocked` checks `eval.json`, `scores.npz`, `eval_inputs.npz`, `run.json` and `epochs.jsonl` for both the confirmatory and the dev directory, and also that a real `open` raises in a subprocess. `search_valid.py` calls `allow_checkpoints` only when the checkpoint starts with `results/go_function/`. Source (b) checkpoints pass under the guard's own rule, so G-V2 no longer crashes.

With `--diamond-valid`, `frozen_star` reads (c*, τ*) from `finalists_tierA.json` after checking its hash against the G-F pin. Pass 2 (w_H, shuffled hits, w_R) is built at that cell, and the rerun writes to `search/<aspect>_<base>_dv/`, so the Mac G-F inputs stay as they were. `freeze_finalists.py --remote` refuses unless `pass2_star` equals the G-F cell. It records `rerun_apt_equal_gf` and takes only w_H, PC, the PC controls and the X+DS sensitivity from the `_dv` run; w_R and every AP-T-form control stay with the Mac G-F. `score_test.py` combines the G-F star with that w_H, which is now the cell w_H was selected at.

`verify_tp_bases.py` checks the spec (variant, feature, aspect, study mode, pack None, seed 13) and records the spec and checkpoint sha256. For source (b) it also requires a provenance file with a matching spec sha256, exit code 0, and `go_function/*.py` hashes equal to GF's `code_pin.json` `files` map; that key matches what `scripts/go_function/sweep.py pin` writes. GF's code pin does not exist yet, so source (b) fails closed until GF G8. The §10.4 exposure facts come from directory listings only. Strict load and two validation forwards (max abs diff 0) run after `enter_search_mode()`.

### Remaining protein-bootstrap stream mismatch

§8 specifies one `default_rng(20261015)` stream per aspect: Fmax draws (B × n_prot) first, P_b draws (B_P × n_prot) second. `score_test.py` calls `protein_draws(len(rows), …)`, where rows are test proteins with any TP label. `report.py` calls it again with n = rows of the protein table, which keeps proteins with a label among the primary, non-root columns. A protein labelled only at the root, or only on a column without a primary id, is in the first set and not the second. When the two n differ, the P_b draws are not the second block of the Fmax stream. P_b's protein-bootstrap CI is descriptive and never enters Holm or §12, so this does not affect any decision (confirmed by code reading; whether the counts differ on test is unknown).

### Protected paths and the go_function protocol row

`git status --porcelain` lists modified files under `go_function/` and `scripts/go_function/`, plus other task docs. No file in `go_function/`, `scripts/go_function/`, `config/go_function/`, `results/go_function/`, `go_ppi_fusion/`, `trials_domain/` or `results/go_ppi_fusion_best/` is newer than the first fusion file. The smoke report records identical porcelain and diff-stat before and after the run. Its §6 states that the A6 row in `protocol_go_function.md` is a GF amendment (user decision, tied to the earlier `train.py` change) and that X1 is the only fusion edit. X1 cites the current protocol sha256. `git status` shows nothing changed in `contrastive_learning/`, `go_ppi_domain/`, `scripts/go_withholding/`, `config/go_withholding/`, `trials_domain/`, `scripts/trials_fusion_confirm/`, `go_ppi_fusion/` or `results/go_ppi_fusion_best/`.

### Test coverage

`test_guard.py` (10 tests, spot-checked here: 10 passed) covers the path rules, a subprocess search-mode run that tries to open test labels through the loader, the test pickle, the smoke `scores.npz`, `fusion_F.csv`, released predictions and the test directory, the source (b) outputs, `frozen_star` hash refusal, `verify_tp_bases` spec and code-pin comparison, and the macOS refusals of every remote-only script. Not tested: `score_test.py` and `report.py` end to end on a synthetic fixture, so the first full run of either is the confirmatory one, and `freeze_finalists.py --remote` refusing on a `pass2_star` mismatch.

</details>

<details>
<summary>File map</summary>

- `go_function_fusion/guard.py` — search-mode audit hook; adds the `tp_bases/` rule
- `go_function_fusion/{data,priors,fuse,metrics,stats}.py` — loaders, neighbour sets and controls, fusion families and tie-ordered selection, metrics, bootstraps/Holm/readings
- `go_function_fusion/tests/` — 44 unit tests (3 added with the round-1 fixes)
- `scripts/go_function_fusion/search_valid.py` — `frozen_star`, `_dv` output directory, checkpoint allow-list only for source (a)
- `scripts/go_function_fusion/freeze_finalists.py` — G-F2 refuses on a `pass2_star` mismatch
- `scripts/go_function_fusion/verify_tp_bases.py` — new G-TP gate
- `scripts/go_function_fusion/{gate_repro,score_test,report,s2_grid_fusion,make_pins,train_tp_base}.py`, `diamond_valid.sh` — other gates
- `config/go_function_fusion/` — study grids and seeds, input hashes, published DS cells, DGZ evaluate reference
- `REMOTE_RUN_GO_FUNCTION_FUSION.md` — runbook (G-TP via `verify_tp_bases.py`, `git add -f` for results)
- `protocol_go_function.md` — X1 row (plus the GF-owned A6 row)

Full diff: `git status --porcelain`; the new files are untracked.

</details>
