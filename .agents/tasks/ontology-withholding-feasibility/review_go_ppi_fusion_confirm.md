# GO/PPI fusion confirmation pipeline (protocol v4): code and dev-smoke review

The step-2 code implements the v4 protocol end to end: a library (`go_ppi_domain/fusion_confirm.py`), eight scripts for transfer, builds, encodes, pins, training, scoring and analysis, a 56-test unit suite, and a remote runbook. The step-3 Mac smoke ran only what §6.5 allows. It ran gate 2a and 2f on the 15 published checkpoints, and both reproduced exactly (AZ-X and GZ-P per-checkpoint Δ max |diff| 0 and 1.4e-17, all 30 selected weights exact). It also built the cohort-B fold structure as counts only (`Bclean` 2,669 hard-defined, above the 2,400 floor). Nothing confirmatory ran, and `results/go_ppi_fusion_confirm/` and `preprocess/go_ppi_fusion_confirm/` do not exist. Holm(3), C2 `go_pair_specific_v4`, C5 `absorbed_share_v3`, the 7-label rule and the headline match §4.3. Every row of both §4.3 example tables is unit-tested, as is every other §5.1 item.

Watch for: the generated `report.md` prints only 5 of the 10 §0.3 disclosure items (item 10, the Mac smoke, is missing), and it leaves out the unmatched-HuRI-draw count and cohort-B near ties (confirmed, LOW). Study-bias tertiles are computed for cohort A only (likely, LOW). Both are worth fixing before the code-tree hash is dumped, because the code freezes at gate 2.

**Verdict**: APPROVED

## High-level view

Every control is computed in the form of the fusion it controls. One `ARMS` table declares every fused score with its φ, normaliser and selection rule. AZ-X and all its controls (P, N, T, S, UX, Bz, the 2-D X+Bz C5 grid, the fixed-w 0.4 pair) use per-anchor z and validation global pooled AUC. GZ-P and its descriptive controls (S_P, U) use validation-file global z and nDCG@10. A `CONTRASTS` table writes each paired contrast as a linear combination of per-anchor Δ. That makes X − S, Δ_C5 and Δ_X − Δ_C5 (written as `AZ-X − AZ-X+Bz + AZ-Bz`) paired through common random numbers.

No test data reaches weight selection. Each arm is fit on the validation file of its own cohort or fold, and S is fit on the deranged validation file. HuRI has no fit of its own: `analyze.py` re-fits on cohort A's validation file, asserts the weight equals the recorded one, and reuses cohort-A z-stats for the GZ arms (the deranged ones for S_P), per §3d. The gate-2 near-tie weight substitution exists only in the gate code path.

The shuffle reuses the frozen `go_ppi_fusion.final.derange_anchors`. That function implements rules (a)–(e), closure-size deciles, one `default_rng(11)` per file and the 10,000-draw cap. The map is shared by S and S_P and checked against its file path and sha256 before use.

For leakage, cohort-B folds assert protein disjointness across all three roles, view groups confined to one fold, and two-build hash identity. Fold `negative_pools.jsonl` holds only train and validation anchors, and the negative selector filters by split, so no test-fold protein can enter training as a negative. HuRI rows come only from the published test partition, which is protein-disjoint from cohort-A training (`strategy: protein_disjoint`). That disjointness holds by construction; no explicit assert checks it.

The runbook covers transfer, verification, gate 1, builds and encodes, pins, gate 2 with retry, 34 training runs, scoring, analysis, monitoring and resume, and the return tarball. Protected paths are untouched: only the two new `go_ppi_domain` files appear, and the three modified `.agents` docs predate this workflow. The full `go_ppi_domain/tests` suite was not run on the Mac (2.8 GiB free, below its 5 GB guard). That gap is environmental, since no existing package file changed.

Every finding is on the reporting or robustness side. None touches an estimate, a CI, a Holm decision or a label.

<details>
<summary>Issues (7)</summary>

1. **Incomplete §0.3 disclosures in report.md** (confirmed, LOW): `analyze.DISCLOSURES` has 5 strings, so items 8 and 10 (the Mac dev smoke) are missing even though §0.3 says "all is printed in `report.md`". Print all ten items, for example by embedding the pinned protocol's §0.3 text.
2. **Missing report counts** (confirmed, LOW): §6.4 asks for every near tie and the unmatched HuRI draw count. `analyze.py` scans near ties for cohort A only and never reports `huri_manifest.json` `counts.unmatched_grade0_draws`. Add both.
3. **Study-bias tertiles cohort A only** (likely, LOW): the §4.5 tertiles ("cut points from the cohort's test anchors") are computed only for cohort A. Add Bclean, or state in the report that they are A-only.
4. **TOST "not equivalent" boundary** (confirmed, NIT): `tost_state` uses `>=` / `<=`, but §4.2 says "entirely outside [−δ, +δ]". Use strict `>` / `<`. This only affects label text, and only at an exact float boundary.
5. **Gate-2 pass not enforced by train/score** (confirmed, NIT): neither script requires a passing `gate_reproduction*.json` for the pins revision in use. The order rests on the runbook, as in the trials template. Add a check.
6. **Silent φ exceptions** (confirmed, NIT): `phi_array` maps any matcher exception to 0.0, as `epf` does, and gate 2e's reference uses the same rule, so it cannot see such failures. Count exceptions per file.
7. **Zero-shot Bclean rng fallback** (confirmed, NIT): if a cohort-B fold is incomplete, the Bclean and all_B zero-shot draws switch to `default_rng([31415, 2, 0])` / `[31415, 4, 0]`, not the cohort's a* draws (§4.1). Zero-shot is descriptive. Disclose this if it fires.

</details>

<details>
<summary>Details</summary>

### Reporting gaps in analyze.py (issues 1–3)

```python
DISCLOSURES = (  # analyze.py: 5 entries for §0.3's 10 items; items 8 and 10 absent
    "Cohort A test and validation anchors were all scored under fusion before ...",
    ...
    "The best-fusion search (208 variants, validation only) and its one test scoring; ...",
)
```

The near-tie loop in `analyze.main` iterates `ca["states"]` only. Cohort-B per-fold `states` are computed but never scanned for gaps below 1e-4. `huri_manifest.json` carries `counts.unmatched_grade0_draws`, but `analyze.py` uses the HuRI manifest only indirectly, through `score.py`'s edge set. The study-bias `stratified(...)` call is made only with `DA` / `dA`. None of this changes a number. `pins.code` freezes at gate 2, so fixing these after that point needs a pins revision. Fixing them now, before `dump_code_hashes.py`, costs nothing.

</details>

<details>
<summary>File map</summary>

- `go_ppi_domain/fusion_confirm.py`: loader, φ variants, typed parser, derangement wrapper, n_PMID, AZ/GZ fusion and selection, arms/contrasts, per-anchor metrics, bootstrap, DE, Holm/TOST/C2/C5, labels, fold and HuRI builders, gate-2 comparator, gate 3, pins chain, ledger.
- `scripts/go_ppi_fusion_confirm/dump_code_hashes.py`: Mac code-tree manifest.
- `scripts/go_ppi_fusion_confirm/verify_transfer.py`: step −1 hash, chunk, snapshot and code check.
- `scripts/go_ppi_fusion_confirm/build_inputs.py`: step-0 label-free builds; `--structure-only` dev mode.
- `scripts/go_ppi_fusion_confirm/encode.py`: CPU fresh masked encodes.
- `scripts/go_ppi_fusion_confirm/make_pins.py`: pins revision 1 and `--revise`.
- `scripts/go_ppi_fusion_confirm/train.py`: 34-run ordered driver with gate 3 and ledger.
- `scripts/go_ppi_fusion_confirm/score.py`: gate 2a–2f, `--dev-local`, score export.
- `scripts/go_ppi_fusion_confirm/analyze.py`: estimates, decisions, labels, sensitivities, report.
- `go_ppi_domain/tests/test_fusion_confirm.py`: §5.1 unit tests (56 passed on the Mac).
- `.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_GO_PPI_FUSION.md`: runbook.
- Full diff: all new files are untracked on `orca-experiments-and-runner` (`git status`).

</details>
