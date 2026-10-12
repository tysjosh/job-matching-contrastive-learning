# Implementation Plan: GO protein-function study (gap items 2–11)

Sources, in precedence order: `findings_go_function.md` (spec), `protocol_go_function.md` v3 (wins on criteria), `design_go_function.md` v3 (owns code, files, interfaces), and the round-3 orchestrator decision in `design_review_go_function.md`, which overrides both documents on one point: **W2 and W1 are closed-world** (no loss mask on withheld proteins). Gap item 1 is done (`setup_baseline_go_function.md`).

`$R` = `/Users/olukotunjosh/Downloads/CDCL`. Every path in this plan is absolute. Every path that code writes into a config, manifest, pin, run spec or marker is repo-relative (computed from `ROOT = Path(__file__).resolve().parents[k]`). No `/Users/...` string may appear in any written artifact.

## Ground rules (apply to every item)

- **No git writes.** No commit, stage, stash, tag or branch. The user commits.
- **Where new code goes:** only `$R/go_function/` (library), `$R/go_function/tests/` (tests and fixtures), `$R/scripts/go_function/` (CLIs) and `$R/config/go_function/` (frozen configs). Data goes to `$R/dataset/go_function/` and `$R/preprocess/go_function/`, results to `$R/results/go_function/`. Task docs live in `$R/.agents/tasks/ontology-withholding-feasibility/`.
- **Protected, never modified:** `contrastive_learning/`, `go_ppi_domain/`, `scripts/go_withholding/`, `config/go_withholding/`, `trials_domain/`, `scripts/trials_fusion_confirm/` and `scripts/run_learning_curve_point.py`.
  - Code is reused by copy-with-attribution into `go_function/` (header: source path, function, source sha256).
  - Only `go_function/tests/test_vendored_parity.py` may import from `go_ppi_domain`/`scripts.go_withholding`.
- **Device:** local execution is CPU only. Every CLI that touches torch takes `--device auto|cuda|cpu`, resolved by `go_function.runtime.resolve_device`. `auto` gives `cuda` if `torch.cuda.is_available()`, else `cpu`. **MPS is never selected.** Every local command in this plan passes `--device cpu`.
- **Remote-only work, not run here:** the full grid, ESM-2 embeddings, MPNet term text, G4, the pilots (G7), G8, G9b, G10 and G11. They run later on a remote L40S that the user launches. Each item below says what runs locally.
- **Python and tests:** `$R/.venv/bin/python` (3.13.5; torch 2.8.0, numpy 2.3.2, scipy 1.16.1, sklearn 1.7.1, pandas 2.3.2, transformers 4.55.4, sentence-transformers 5.1.0).
  - Unit tests: `cd $R && .venv/bin/python -m pytest go_function/tests -q -p no:cacheprovider`.
  - Slow and real-data tests: add `GO_FUNCTION_SLOW=1`.
  - New dependencies: none.
- **Scripts:** start with `sys.path.insert(0, str(ROOT))` (the convention of `scripts/go_withholding/sweep.py`). Exit codes follow design §15: 0 ok, 2 fatal, 3 protocol stop, 4 refused by state.
- **Local machine limits:** Apple Silicon, 8 GB RAM, about 14 GB free disk.
  - Do not keep more than one BP dense label matrix in memory.
  - Do not allocate a mask tensor when a condition's mask is empty.
  - Check free disk before any large download (item 7).

## Decisions made in this plan (not in the design)

1. **Closed-world W2/W1** (round-3 HIGH, adopted by the orchestrator) is recorded as protocol D19.
   - D9 (Lost(p) masking) is withdrawn: Lost(p) entries become ordinary 0 labels.
   - The only masks left anywhere in training are WDGZrel/W2drop column masks over all train rows and the target-free validation column mask.
   - Item 8 updates the design and protocol before any withholding code is written.
2. **`go_function/runtime.py`** (new; not in design §16) holds `ROOT`, `resolve_device`, `seed_everything`, `setup_logging` and `repo_rel`. Every CLI needs these, and one owner keeps the no-MPS and repo-relative rules in one place.
3. **G9a local smoke runs in smoke mode.** The run spec field `max_train_batches` is accepted only when `mode = "smoke"`, set to 40 per epoch for 2 epochs on BP InterPro on CPU. Evaluation still uses the full evaluation set.
   - Why: a full BP epoch with the EL loss over about 110k axioms per batch is several CPU-hours on this machine, and G9a only tests the plumbing.
   - The full-epoch smoke is G9b on the L40S. This deviation is stated in the smoke report.
4. **Gate downloads and G2/G2b/G3 run locally**, as protocol §15 says. The about 5.2 GB of gate files (measured with Content-Length: checkpoints 1.53 GB, BP predictions 2.82 GB, MF predictions 0.80 GB) are fetched only if at least 8 GB stays free. Otherwise the coder signals a warning and stops; it does not delete anything.
5. **Pins** live in `results/go_function/protocol_pins.json`. This path is git-ignored, so `REMOTE_RUN_GO_FUNCTION.md` ships it with rsync. `config/go_function/input_hashes.json` is tracked and inside the code pin.
6. **The design §16 file list is binding**, plus `runtime.py` and `scripts/go_function/pin_inputs.py`, which writes `input_hashes.json` from `dataset/go_function/MANIFEST.json`.

## Plan

### FEAT-001: foundation, typed ontology, normal forms, labels (gaps 2–3, gate G1)

- [ ] 1. **Scaffold the package and the shared helpers.**
  - Create:
    - `go_function/__init__.py`;
    - `constants.py`: FLOOR 0.01, ALPHA 0.05, UNSTABLE_SHARE 0.05, ANALYSIS_SEED 20261011, RETENTION_SEED 20261011, RETENTION_SEED_2 20261012, SEEDS (13, 21, 42, 87, 123), KS (0, 1, 5, 10, 25, "all"), DGZ `EXP_CODES`, the relation-IRI map of design §6.4, exit codes, ARMS, RULES and REGIMES;
    - `runtime.py`: `resolve_device` raises on "mps"; `repo_rel` raises on paths outside ROOT;
    - `pins.py`: `sha256_file`, `csr_sha256` over (indptr, indices, data, shape), canonical JSON bytes (sorted keys, `\n`-terminated), `write_or_verify(out_dir, files)` copied with attribution from `go_ppi_domain/withholding._write_or_verify`, and `record_pin`/`verify_pin` on `results/go_function/protocol_pins.json` with gate keys from protocol §13;
    - `vendored.py`: `jaccard`; `_counts`, `two_way_draws`, `_summarise`, `two_way_bootstrap`, `hierarchical_bootstrap`, `holm`, `classify_contrast`, `ci_excludes_0`, with FLOOR/ALPHA/UNSTABLE_SHARE/seed as required keyword arguments with no defaults (design §13.2);
    - `go_function/tests/conftest.py`: a `slow` marker, skipped unless `GO_FUNCTION_SLOW=1`, and a `needs_dgz` skip when `dataset/go_function/dgz/go.obo` is absent;
    - fixtures `go_function/tests/fixtures/mini.obo` (12 terms: is_a, part_of within and across namespaces, regulates, positively_regulates, has_part, alt_id, obsolete, intersection_of) and `mini.norm` (every NF form, one role axiom, one `SubClassOf Nothing`).
  - Tests:
    - `test_runtime.py`: auto never gives mps, even with `torch.backends.mps.is_available()` patched to True; `repo_rel`; canonical JSON has no absolute path;
    - `test_pins.py`: write_or_verify identical → "unchanged", differing → exit-2 error;
    - `test_vendored_parity.py`: imports the originals and asserts identical outputs on fixed random inputs, with FLOOR 0.01, ALPHA 0.05, UNSTABLE_SHARE 0.05 and the original seed.
  - Files: the files above.
  - Verify: `cd /Users/olukotunjosh/Downloads/CDCL && .venv/bin/python -m pytest go_function/tests -q -p no:cacheprovider`. All pass.

- [ ] 2. **Typed ontology module** `go_function/ontology.py`, exactly as design §1.
  - `Term`, `Ontology`, `parse_obo` (all relations typed, `intersection_of`, alt/obsolete, `OntologyError` with a line number, `parse_report`).
  - `ancestors`/`descendants`/`parents` with `rels=TRUE_PATH` and `same_namespace=True`; a cycle raises.
  - `isa_siblings`, `two_hop`, `logical_definition`.
  - `DgzRule`: a port of DGZ `utils.Ontology(with_rels=True)` (every relationship becomes is_a, alt ids are aliases, so closure(alt) lacks the primary). Port it from the public DGZ source `https://raw.githubusercontent.com/bio-ontology-research-group/deepgozero/main/utils.py`, read with web_fetch, and cite it in the header.
  - `go_ppi_domain/go_ontology.py` is neither imported nor touched.
  - Tests in `test_ontology.py` on `mini.obo`: regulates never propagated; cross-namespace part_of not propagated under TP but propagated by DgzRule; alt and obsolete handling; siblings exclude anc/desc; two_hop; logical definition; dangling edge → OntologyError; cycle → OntologyError.
  - Files: `go_function/ontology.py`, `go_function/tests/test_ontology.py`.
  - Verify: the unit-test command passes.

- [ ] 3. **go.norm loader** `go_function/normforms.py` (design §6.4).
  - `load_normal_forms(path, terms_dict, dgz_compat=True)`: DGZ's parser order, vocabulary columns first, `Nothing` → `owl:Nothing`.
  - Fatal checks: unparsed must be 9; `owl:Nothing` must be in exactly 30 lines, all NF2 right-hand sides; an unknown relation IRI is fatal; a non-GO token is fatal.
  - `select(nf, arm)` builds the V1/V2/V3/V3c/V4/V6 ladder with per-arm re-indexing. V3c has one shared relation vector over part_of plus the regulates family. V1–V3c contain no `owl:Nothing`.
  - Tests in `test_normforms.py` on `mini.norm`: every form; the role-axiom count; `owl:Nothing` index position; ladder nesting asserted.
  - A `slow` + `needs_dgz` test on the real `dataset/go_function/dgz/go.norm` asserts NF1 80,942, NF2 11,841, NF3 11,810, NF4 19,594, 9 unparsed and 30 Nothing.
  - Files: `go_function/normforms.py`, `go_function/tests/test_normforms.py`.
  - Verify: `GO_FUNCTION_SLOW=1 .venv/bin/python -m pytest go_function/tests/test_normforms.py -q -p no:cacheprovider`. All pass, including the real-file counts.

- [ ] 4. **Label builder and G1.**
  - `go_function/labels.py` (design §2):
    - vocabulary from `terms.pkl` (in order), `primary()` and duplicate columns;
    - the A_TP / A_TPnoIPI / A_DGZ direct sets;
    - CSR uint8 `train`/`valid`/`test`, `direct_<split>`, `direct_extra_<split>`, `proteins_<split>`, `vocab`;
    - the MF protein-binding-only evaluation set (TP and TPnoIPI);
    - `_bfs_closure_tp`, an independent BFS sharing no helper with `ontology.ancestors`;
    - checks 1, 2, 3a–3d and 5 of design §2.4.
  - `scripts/go_function/pin_inputs.py`: writes `config/go_function/input_hashes.json` (full sha256 values from `dataset/go_function/MANIFEST.json`, repo-relative paths) and verifies every file against it.
  - `scripts/go_function/build_labels.py --aspect bp|mf --regime TP|DGZ|TPnoIPI|all [--verify-pins]`:
    - verifies input hashes first (exit 2 on mismatch);
    - writes `preprocess/go_function/<aspect>/labels_<regime>.npz` plus a `.json` manifest (counts, zero-positive columns per regime and split (D16), alt/obsolete counts, matrix hashes);
    - records pin `G1-labels`, and `G1-norm` from item 3;
    - TPnoIPI applies to MF only.
  - Tests in `test_labels.py` on mini fixtures:
    - duplicate columns;
    - protein-binding-only exclusion;
    - the noIPI evaluation set with an IPI-only protein;
    - check 3c trips on a planted regulates-propagated label and passes on an alt-annotated protein;
    - check 3d trips on a planted non-alt TP-only label.
  - A `slow` test runs DgzRule against `prop_annotations` on 200 fixed proteins per aspect.
  - Files: `go_function/labels.py`, `scripts/go_function/pin_inputs.py`, `scripts/go_function/build_labels.py`, `config/go_function/input_hashes.json`, `go_function/tests/test_labels.py`.
  - Verify:
    - unit and slow tests pass;
    - then run `.venv/bin/python scripts/go_function/pin_inputs.py`, `.venv/bin/python scripts/go_function/build_labels.py --aspect mf --regime all` and `... --aspect bp --regime all` (in the background, with a timeout ≥ 30 min). All exit 0;
    - every §2.4 check passes on every protein of every split (DGZ = `prop_annotations ∩ V`, BP and MF);
    - reported counts: alt-id/TP-only labels (about 778 BP / 59 MF on train) and the DGZ excess;
    - re-running is byte-identical ("unchanged").

### FEAT-002: evaluator, features, DGZ evaluation gate (gaps 4–5, gates G2 and G5-local)

- [ ] 5. **Evaluator** `go_function/evaluate.py` (design §8).
  - `class_metrics(logits_col, y_col)`: raw logits through `roc_auc_score`/`average_precision_score`, n_pos stored; asserts 0 < n_pos < n.
  - Protein-centric metrics over a term set T, with TP max-propagation: Fmax over thresholds 0.00–1.00 step 0.01, Smin with IC from train+valid TP labels, and micro-AUPR. Computed in torch, chunked by threshold, on the run's device.
    - Port Fmax/Smin/AUPR and the "0 < pos < n" term average from DGZ `evaluate.py` and `evaluate_terms.py` (raw.githubusercontent, deepgozero main branch, read with web_fetch). Cite the source and record any formula found that differs from design §8.2 in a comment and in FEAT findings.
  - `eval_label_source(regime)` is the single regime → (label file, evaluation set) mapping: `labels_TPnoIPI.npz` iff `regime == "TPnoIPI"`, else `labels_TP.npz`.
  - `collateral_terms(...)`: primary non-duplicate columns with ≥ 1 train positive, minus ∪D*_TP(pack), root excluded.
  - DGZ-faithful mode: DGZ-rule labels, `terms.pkl` columns, predictions as given.
  - Tests in `test_evaluate.py`: class AUC/AUPR equal sklearn; Fmax/Smin on hand toy cases; max-propagation; the regime mapping, including that a noIPI regime can never get TP labels.
  - Files: `go_function/evaluate.py`, `go_function/tests/test_evaluate.py`.
  - Verify: the unit-test command passes.

- [ ] 6. **Features** (design §3).
  - `go_function/features.py`: InterPro CSR from the pickle `interpros` column over the 26,406 vocabulary (unknown ids dropped and counted); ESM matrix views by protein-name lookup (a missing protein is fatal); a term-text loader.
  - `scripts/go_function/build_features.py --aspect bp|mf` writes `preprocess/go_function/<aspect>/features_interpro.npz` plus a manifest and records pin `G5-interpro`.
  - `scripts/go_function/embed_esm.py --device auto|cuda|cpu [--limit N]` (remote):
    - HF `EsmModel` `facebook/esm2_t33_650M_UR50D` at a `--revision` pin;
    - union of BP and MF sequences, deduplicated by name (a differing sequence is fatal);
    - truncation to 1,022 residues; mean over residues excluding BOS/EOS/pad; fp16 on CUDA;
    - length-sorted batches of 16,384 tokens; on OOM, halve the budget down to 2,048;
    - the 64-protein fp32 batch-1 cross-check (cosine ≥ 0.999);
    - outputs `preprocess/go_function/esm2_650m/{embeddings.npy, proteins.txt, manifest.json}`;
    - **refuses `--device cpu` without `--limit`** (exit 4).
  - `scripts/go_function/embed_terms.py --device auto|cuda|cpu` (remote): all-mpnet-base-v2 at revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, `normalize_embeddings=True`, over V ∪ go.norm GO classes; skips `owl:Nothing`; a GO class missing from go.obo → exit 2.
  - `scripts/go_function/requirements-remote.txt`: a copy of `scripts/trials_fusion_confirm/requirements-remote.txt` (same pins), with its own header.
  - Tests in `test_features.py`:
    - InterPro on a fixture DataFrame;
    - embed_esm pooling, truncation and BOS/EOS exclusion with an injected tiny random `EsmModel(EsmConfig(...))` plus a tokenizer built from a temporary vocab file (no download);
    - CPU without `--limit` refused;
    - embed_terms on the fixture with an injected fake encoder skips `owl:Nothing` and fails on a missing GO class.
  - Files: `go_function/features.py`, `scripts/go_function/{build_features,embed_esm,embed_terms}.py`, `scripts/go_function/requirements-remote.txt`, `go_function/tests/test_features.py`.
  - Verify:
    - unit tests pass;
    - `.venv/bin/python scripts/go_function/build_features.py --aspect bp` and `--aspect mf` exit 0, with BP/MF zero-InterPro shares about 2.5%/1.5% of train;
    - re-runs are byte-identical.

- [ ] 7. **DGZ evaluation gate G2.**
  - `scripts/go_function/download_dgz_gate.py`:
    - same never-overwrite and Content-Length rules as `download_dgz.py`;
    - files: per bp/mf, `mlp.th`, `deepgozero.th`, `deepgozero_zero.th`, `predictions_{mlp,deepgozero,deepgozero_zero}.pkl`, `zero_gos.pkl`, `zero_term_auc{,_all,_train}.pkl`, and `zero_<GO>_auc{,_all,_train}.pkl` for the withheld classes;
    - `--only checkpoints|predictions|zero|all`;
    - refuses (exit 4) if free disk after the download would drop below 8 GB;
    - writes `dataset/go_function/MANIFEST_gate.json`; `pin_inputs.py --gate` adds those hashes to `input_hashes.json`.
  - `config/go_function/dgz_published.json`: DGZ Table 3 values for MF and BP (MLP, DeepGOZero: class-centric AUC, Fmax, Smin, AUPR), transcribed from PMC9235501 (web_fetch `https://pmc.ncbi.nlm.nih.gov/articles/PMC9235501/`), each value with its table/row/column cell.
  - `scripts/go_function/gate_dgz_eval.py --aspect bp|mf --device cpu`: DGZ-faithful mode on `predictions_mlp.pkl` and `predictions_deepgozero.pkl`; tolerance ±0.005 on every metric; a miss → exit 3; writes `results/go_function/gates/gate_dgz_eval.json` and pin `G2`.
  - Tests in `test_gates_eval.py` on a synthetic predictions DataFrame with known metrics: pass, and a planted miss → exit 3.
  - Files: `scripts/go_function/download_dgz_gate.py`, `scripts/go_function/gate_dgz_eval.py`, `config/go_function/dgz_published.json`, `go_function/tests/test_gates_eval.py`.
  - Verify:
    - unit tests pass;
    - `df -h /Users/olukotunjosh/Downloads/CDCL` shows ≥ 13.5 GB free; otherwise send a warning and stop;
    - `.venv/bin/python scripts/go_function/download_dgz_gate.py --only all` exits 0;
    - `.venv/bin/python scripts/go_function/gate_dgz_eval.py --aspect mf --device cpu`, then `--aspect bp`. Both exit 0 (G2 pass).
    - **A G2 miss is a protocol stop:** send a warning with the per-metric numbers and do not continue to G3.

### FEAT-003: targets and closed-world withholding (gaps 7–8, gate G6)

- [ ] 8. **Record the round-3 fix in the documents before writing any code.**
  - `design_go_function.md`: §5.2 W2/W1 closed-world, using the reviewer's text block; §5.5 assertion 1 ("y = 0 and not masked"); assertion 7 (W2drop mask = {(p, c): p ∈ train, c ∈ T_cols(t)}, since W2 now has no mask); new assertion 10 (W1/W2 train mask empty; in WDGZrel/W2drop every masked column is masked on all train rows; no mask is row-selective on a target-related column); §6.5 note; a "Round 3" response table in §18.
  - `protocol_go_function.md`: §5 W2/W1; D9 withdrawn and replaced by new D19 (closed-world; a deviation from spec §5 "Masking"; withheld positives become unannotated closed-world negatives; for user/supervisor sign-off before G8); risk 6 rewritten (W2 trains column t on all-negatives at k = 0 and can depress it; C3_col measures it); §16 v4 changes.
  - NITs:
    - assertion 8 reworded to "(k = 0 only)";
    - LD filler n_train(F) = |{p ∈ train : F ∈ closure_TP(p)}| whether or not F ∈ V (design §4 step 3, protocol §4);
    - C3_leak_desc is also reported stratified by |Desc(t)| (≤ 5 vs > 5), and channel reads are compared by CI sign only (protocol §1);
    - protocol §8 "; Its" → "; its";
    - alt-id wording unified to "about 14 BP and 5 MF before the n_test filter".
  - Version both documents as v4. Run count unchanged at 5,939.
  - Files: `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/design_go_function.md`, `.../protocol_go_function.md`.
  - Verify: read both documents back. No remaining text says W2 masks D*_ρ(t) or Lost(p). D19 is present. Assertions 1, 7, 8 and 10 match the reviewer's wording.

- [ ] 9. **Target table** `go_function/targets.py` and `scripts/go_function/build_targets.py --aspect bp|mf` (design §4, protocol §4).
  - Base filter, including the alt-id exclusion (count reported); S_sib; the LD flag with filler support by closure (item 8 NIT); tertile cuts; bands B1/B2/B3 with quotas BP 4 and MF 8.
  - `accept_round_robin(slots, quotas, P_train, ancestors, max_jaccard)`, using vendored `jaccard`; borrowing B1→B2→B3→B1; shortfall → exit 3.
  - SMD balance report.
  - Packing into 8 × 6: two-hop exclusion; D*^GO_TP and D*^GO_DGZ disjointness on term-level sets; pack union ≤ 10% N_train; the DGZ union reported.
  - Retention orders for draw 1 and draw 2, plus the noIPI order (MF), with seed_t = int(sha256(f"{seed}:{t}")[:16], 16).
  - Channel sets per target: X(t), X_reg(t), Desc(t), Pa(t), the is_a siblings, and their sizes.
  - Output `preprocess/go_function/<aspect>/targets/target_table.json` (schema 1, canonical bytes). A second run must be byte-identical, else exit 2. Pin `G6-targets`. **Do not print or inspect target names** beyond counts.
  - Tests in `test_targets.py`: tertiles; LD flag; quotas; round-robin visit order and one acceptance per visit; borrowing; ancestor and Jaccard rejection; packing with a planted non-column term implying two candidates (rejected); determinism; the shortfall exit; alt-id exclusion.
  - Files: `go_function/targets.py`, `scripts/go_function/build_targets.py`, `go_function/tests/test_targets.py`.
  - Verify:
    - unit tests pass;
    - `.venv/bin/python scripts/go_function/build_targets.py --aspect bp`, then `--aspect mf`. Each exits 0 with 48 targets and 8 packs. An exit 3 is a protocol stop: send a warning with counts only;
    - re-runs are byte-identical.

- [ ] 10. **Withholding generator** `go_function/withholding.py` and `scripts/go_function/make_conditions.py --aspect bp|mf [--block NAME|all]` (design §5 as amended by item 8).
  - Term-level D*^GO_ρ and column-level D*_ρ.
  - W2 (closed-world): erase D*^GO_ρ(t) from A_ρ(p) for p ∈ W, then re-propagate. **No mask.**
  - W1 (closed-world): W2, then strict ancestors of t set positive. **No mask.**
  - WDGZrel: base DGZ labels unchanged; T_cols(t) masked on every train row; descriptive per-column `prop_annotations` vs `zero_annotations` disagreement counts.
  - W2drop: W2/DGZ/k0 labels, plus T_cols masked on all rows.
  - Draw 2 (`_d2`); noIPI conditions; one `all` per regime; multi-target erasure before one re-propagation.
  - Target-free validation column masks per (aspect, regime).
  - Files per design §5.4, via `write_or_verify`.
  - Assertions 1–10, as functions reusable by `run.py` preflight.
  - Blocks follow protocol §6.2, so every condition needed by the grid is generated:
    - TP/W2 k ∈ {0, 1, 5, 10, 25, all};
    - DGZ/W2 the same k;
    - TP/W1 k ∈ {0, 1, 5};
    - DGZ/WDGZrel k0; DGZ/W2drop k0;
    - TP/W2 d2 k ∈ {1, 5};
    - MF TPnoIPI/W2 k ∈ {0, 1, 5, all}, with the noIPI target drop rule reported.
  - Pin `G6-conditions` and `G6-validation`.
  - Tests in `test_withholding.py` on a synthetic 30-protein dataset with a "positive regulation of X" target:
    - every assertion holds for W2/W1/WDGZrel/W2drop;
    - under WDGZrel the X column keeps every t-positive, while under DGZ/W2 those labels are erased by re-propagation;
    - a deliberately leaked condition trips assertion 1;
    - a planted row-selective mask trips assertion 10;
    - the cross-namespace part_of fixture: an aspect-filtered erasure trips assertion 1 or 2;
    - the alt-id fixture: WDGZrel with only column t masked on an alt-column target trips assertion 8;
    - a non-column direct annotation implying t is erased;
    - draw 2 differs from draw 1 and is nested;
    - **closed-world test:** the W2 condition's train mask is empty and the labels differ from base only on erased (p, c) entries of withheld proteins.
  - Files: `go_function/withholding.py`, `scripts/go_function/make_conditions.py`, `go_function/tests/test_withholding.py`.
  - Verify:
    - unit tests pass;
    - `.venv/bin/python scripts/go_function/make_conditions.py --aspect bp --block all`, then `--aspect mf`. Both exit 0, all assertions pass, and re-runs report "unchanged";
    - `du -sh /Users/olukotunjosh/Downloads/CDCL/preprocess/go_function` is recorded in findings.

### FEAT-004: model, trainer, run CLI, checkpoint gates (gap 6, gates G3 and G2b)

- [ ] 11. **Model** `go_function/model.py` (design §6).
  - Protein tower: `MLPBlock` and a residual block, with norm `layernorm|batchnorm1d` as a constructor argument.
  - Term towers:
    - V0: free `Linear` rows;
    - EL arms: `Embedding(|V|+|Z|, d)` plus radius, `rel_embed` with a hasFunc row, `go_norm` BatchNorm1d inside EL only, uniform init ±√(1/d);
    - V5: `Linear(768, d, bias=False)` over frozen text plus a free bias;
    - V6: text centres plus a free `owl:Nothing` centre and radius.
  - EL losses: a direct port of DGZ `deepgozero.py` nf1–nf4 (web_fetch raw source; cite it), margin 0.1, `nf4_radius="dgz"|"class"`.
  - Masked BCE: mean over unmasked entries; mask `None` takes the unmasked path.
  - `DgzCompatModel.load(path, strict=True)` with key mapping; `CheckpointMismatch` lists missing/unexpected keys and shapes.
  - `hidden_dim` is configurable, so tests use d = 8.
  - Tests in `test_model.py`: forward shapes for every arm; masked BCE equals plain BCE with an empty mask; EL equals hand-computed 3-class examples in both NF4 modes; V3c shares one vector; key mapping on a synthetic state dict; mismatch raises.
  - Files: `go_function/model.py`, `go_function/tests/test_model.py`.
  - Verify: the unit-test command passes.

- [ ] 12. **Trainer and run CLI**: `go_function/train.py`, `scripts/go_function/run.py --spec PATH --device auto|cuda|cpu`, and `config/go_function/study.json` (design §7, §12.1).
  - `study.json`: batch 37, lr 5e-4, milestones [5, 20], gamma 0.1, dropout 0.1, dim 1024, margin 0.1, `norm: null` until G3, seeds, ks, retention and analysis seeds, `epochs_star: {}`, λ = 1.
  - Spec schema validation per design §7.1; unknown fields → exit 2. `mode` is `study|gate|pilot|smoke`; `max_train_batches` only in smoke.
  - Preflight:
    - pins: inputs, labels, condition, target table, validation mask, normal forms, term text;
    - §5.5 assertions 1–10 re-run on the condition;
    - code-pin import coverage, fatal if a module comes from a protected directory (skipped with a warning when `code_pin.json` is absent and mode ≠ study);
    - study-field equality.
  - Loop: Adam/MultiStepLR; seeded `torch.Generator`; deterministic flags plus `CUBLAS_WORKSPACE_CONFIG`; dense bool labels; InterPro CSR batches densified per batch.
  - Per-epoch `epochs.jsonl`: train BCE, EL per form, target-free validation BCE, lr, wall time. Every 50th batch: `torch.autograd.grad` norms of BCE and λ·EL **separately on the protein tower and the term tower**, plus their ratio (reimplementing `withholding_diagnostics._grad_norm`, with attribution).
  - A non-finite loss raises; checkpoints are kept only for `all`/seed 13 in study mode, and in gate/pilot mode.
  - After training: `eval_inputs.npz`, then `eval.json` (class metrics per target; collateral Fmax/Smin/micro-AUPR), then `scores.npz` (t, Pa, siblings; X/Desc only for V4 k = 0), then `run.json` (spec, hashes, environment), then `DONE.json`. Delete `eval_inputs.npz` after success. Any exception writes `FAILED.json` with `stage`.
  - Every path written is repo-relative.
  - Tests in `test_train.py`: a tiny synthetic dataset, end to end on CPU, writes all outputs; a schema rejection; a non-finite loss → FAILED stage train; seed determinism on CPU.
    - **Closed-world mask test (orchestrator requirement):** the per-row loss mask the trainer builds for a W2 condition is identical to the base-data mask (empty). The only differences are the erased labels, so no row-level signal identifies withheld proteins.
  - Files: `go_function/train.py`, `scripts/go_function/run.py`, `config/go_function/study.json`, `go_function/tests/test_train.py`.
  - Verify: the unit-test command passes.

- [ ] 13. **Checkpoint gates.**
  - `scripts/go_function/gate_checkpoints.py --aspect bp|mf --device cpu` (G3): loads `mlp.th` and `deepgozero.th` strictly, determines the norm type from the keys, runs forward plus DGZ-rule score propagation against the released predictions (max abs diff ≤ 1e-4), writes `gates/gate_checkpoints.json` and the `norm` field of `study.json`, and records pin `G3`.
  - `scripts/go_function/gate_dgz_zero.py --aspect bp|mf --device cpu` (G2b):
    - L1: strict load of `deepgozero_zero.th` with `terms_zero.pkl`;
    - L2: per withheld class, AUC vs `zero_<GO>_auc*.pkl` within 0.01 (BP ≥ 6/7, MF ≥ 4/5);
    - L3 (BP): the X→t AUC rule;
    - status confirmed / not confirmed / not checkable; a load failure writes `"loaded": false`;
    - also scores `predictions_deepgozero_zero.pkl`;
    - writes `gates/gate_dgz_zero.json` and pin `G2b`.
  - `scripts/go_function/gate_trainer.py` (G4; remote): compares four gate `run.py` outputs with `dgz_published.json` at ±0.01, with the 3-seed fallback rule. It also writes the four gate specs (InterPro, DGZ labels, `selection: best_valid_loss`, ≤ 256 epochs, seed 42) into `results/go_function/specs/gates/`.
  - Tests in `test_gates_model.py` on synthetic checkpoints and predictions: pass/miss paths, the status logic, and norm detection.
  - Files: `scripts/go_function/{gate_checkpoints,gate_dgz_zero,gate_trainer}.py`, `go_function/tests/test_gates_model.py`.
  - Verify:
    - unit tests pass;
    - `.venv/bin/python scripts/go_function/gate_checkpoints.py --aspect mf --device cpu`, then `--aspect bp`. Both exit 0 and `study.json` gets `norm`;
    - `.venv/bin/python scripts/go_function/gate_dgz_zero.py --aspect bp --device cpu` and `--aspect mf` exit 0, with the status recorded. G2b does not block, but a G3 miss is a stop: send a warning;
    - `gate_trainer.py --write-specs` exits 0 (the run itself is remote).

### FEAT-005: pilot, sweep, analysis, smoke, runbook (gaps 9–11, gate G9a)

- [ ] 14. **Pilot** `scripts/go_function/pilot.py --aspect A --feature F --device auto|cuda|cpu [--stage estar|lambda|text|all]` (design §9; remote).
  - E* from V0/V4 × seeds 13/21/42, ≤ 64 epochs, using the formula clip(ceil(median(i_best + 1)), 8, 64).
  - The λ record: V3/V4 × seeds 13/21 × λ ∈ {0.1, 0.3, 1, 3} at E*, with validation BCE and the grad-norm ratio per tower.
  - The text-arm record: V5/V6 at seed 13, ESM only.
  - Writes `results/go_function/pilots/<aspect>_<feature>/choice.json`, `epochs_star` into `study.json`, and pin `G7`.
  - An import probe asserts `go_function.evaluate` is never imported (run in a subprocess).
  - Tests in `test_pilot.py`: the E* formula on fabricated `epochs.jsonl` files; the probe fails if the evaluator is imported; λ never changes `study.json`'s λ.
  - Files: `scripts/go_function/pilot.py`, `go_function/tests/test_pilot.py`.
  - Verify: the unit-test command passes.

- [ ] 15. **Materializer and sweep driver.**
  - `scripts/go_function/materialize.py --aspect A --feature F`:
    - one spec per protocol §6.2 grid cell into `results/go_function/specs/<aspect>/<feature>/`;
    - the single-factor audit table of design §12.2, where any undeclared difference is fatal;
    - counts must equal BP 2,782 and MF 3,157 in total over features (asserted per block);
    - refuses (exit 4) without E* in `study.json`.
  - `scripts/go_function/sweep.py` with `plan|run|status|pin|rescore`, modelled on (not importing) `scripts/go_withholding/sweep.py`:
    - run dir `results/go_function/runs/<aspect>/<feature>/<variant>__<regime>__<rule>__<pack|all>__k<k>[__d2]__s<seed>/`;
    - seed-major order;
    - `--shard i/N` by run index over the full ordered grid;
    - `--workers N` (default 4, separate `run.py` subprocesses on one GPU) and a required `--max-runs M`;
    - `--device auto|cuda|cpu` forwarded;
    - an `O_EXCL` `CLAIM` file with pid and host, and `status --release-stale`;
    - resume: `DONE.json` → skip, never re-run;
    - attempt rename `<dir>.attempt<n>`: two failures leave `FAILED.json`, and an OOM retry runs with workers − 1;
    - `rescore` for kept checkpoints or `eval_inputs.npz`;
    - `pin` hashes every `.py` under `go_function/` and `scripts/go_function/` and every `.json` under `config/go_function/` into `results/go_function/code_pin.json`, after an explicit `--accept-content-pin` flag or the user's commit;
    - all paths repo-relative;
    - `--smoke` builds the G9a cell list under `results/go_function/smoke/`.
  - Tests in `test_materialize.py` and `test_sweep.py`:
    - spec counts per block;
    - the audit trips on a planted extra difference;
    - shards are disjoint and exhaustive;
    - claim and stale-claim release;
    - resume skips DONE runs;
    - two failures lead to FAILED;
    - no written file contains `/Users/` (scan all outputs of a fixture run);
    - `--device` is forwarded.
  - Files: `scripts/go_function/materialize.py`, `scripts/go_function/sweep.py`, `go_function/tests/test_materialize.py`, `go_function/tests/test_sweep.py`.
  - Verify: the unit-test command passes.

- [ ] 16. **Analysis**: `go_function/fusion.py`, `go_function/baselines.py`, `go_function/stats.py`, `scripts/go_function/analyze.py --runs DIR --out DIR --seed 20261011 --B 10000 [--smoke]` (design §10–12, protocol §1, §8, §9).
  - F: weight-free z-score fusion from V0 `scores.npz`.
  - DS: DiamondScore from `test_diamond.res` with the condition's train labels.
  - Long table keyed by (aspect, feature, variant, regime, rule, draw, pack, target, k, seed), with class_auc, class_aupr, fmax, smin and micro_aupr.
  - Paired deltas and Δ^small.
  - C1–C7, with Vbest by the D11 rule and descriptive C6/C7 for every arm.
  - Two-way bootstrap and S1 hierarchical (vendored), Holm per aspect family, classification.
  - C3 on its own cells, plus the four-run decomposition with the identity asserted to 1e-12 and both C3 values printed when cell sets differ.
  - Channel reads C3_leak_X/desc/anc, with empty-set counts and C3_leak_desc stratified by |Desc| ≤ 5 vs > 5.
  - Descriptive reads: W1 − W2, F, DS, InterPro, noIPI, re-draw and collateral harm.
  - The >10% missing rule (exploratory label); `DONE.json`/`run.json` hash disagreement → exit 2.
  - Outputs `long_table.csv`, `contrasts.json` and `decisions.json`.
  - Tests in `test_analysis.py`: a synthetic long table with a planted +0.05 effect recovered with covering CI; null → "practically null"; the decomposition identity; channel reads against hand AUCs; empty X excluded and counted; fusion and DS hand examples.
  - Files: `go_function/{fusion,baselines,stats}.py`, `scripts/go_function/analyze.py`, `go_function/tests/test_analysis.py`.
  - Verify: the unit-test command passes.

- [ ] 17. **Local G9a smoke, G0 regression, and the remote runbook.**
  - Smoke: `.venv/bin/python scripts/go_function/sweep.py run --smoke --device cpu --workers 1 --max-runs 6` covers BP p01 × k ∈ {0, 5} × {V0, V3, V3c} × seed 13, InterPro, TP/W2, 2 epochs, `max_train_batches` 40 (plan decision 3), with outputs in `results/go_function/smoke/`. Then `.venv/bin/python scripts/go_function/analyze.py --runs results/go_function/smoke/runs --out results/go_function/smoke/analysis --smoke`.
  - Expected: every run has `DONE.json`, `eval.json`, `scores.npz` and `epochs.jsonl` with grad-norm records; F and DS files exist; no `/Users/` in any written file. Record peak RSS and seconds per run.
  - G0:
    - `.venv/bin/python -m pytest go_ppi_domain/tests -q -p no:cacheprovider` → 334 passed, 3 skipped, 1 known failure (`test_real_protocol_amendments_table_parses`);
    - `trials_domain/tests` → 108 passed, 1 skipped;
    - the four career/CVE files → 8 passed;
    - `git status --short` shows changes only under the allowed directories and the task folder.
  - Write `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/REMOTE_RUN_GO_FUNCTION.md`, following the structure of `REMOTE_RUN_GO_PPI.md` and protocol §15:
    - prerequisites (the user commits and pushes; D19 sign-off);
    - clone; the rsync of `results/go_function/protocol_pins.json` and gates; never ship `/300` or `reports/`;
    - env (cu128/cu126 torch, `requirements-remote.txt`, HF downloads at pinned revisions, `HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8`);
    - re-download plus `pin_inputs.py` verify; rebuild labels/features/targets/conditions with `--verify-pins`;
    - `embed_esm.py --device cuda`; `embed_terms.py --device cuda`; pin G5;
    - G4 via `gate_trainer.py`; G7 pilots; G8 `materialize.py` plus `sweep.py pin`;
    - G9b `sweep.py run --smoke --device cuda` with the 10-day projection stop;
    - G10 in tmux with `--workers N --max-runs M`, shards for multiple processes, `status`/`--release-stale`/`rescore`;
    - G11 `analyze.py`; the return tarball list;
    - a time estimate labelled as an estimate.
  - Every command in the runbook is repo-relative from the clone root.
  - Files: `REMOTE_RUN_GO_FUNCTION.md`; outputs under `results/go_function/smoke/`.
  - Verify: the smoke and analysis commands exit 0 with every listed output present; the three G0 suites give exactly the baseline counts; the full `go_function/tests` suite with `GO_FUNCTION_SLOW=1` passes.

## Gaps and assumptions

- **DGZ source files** (`utils.py`, `deepgozero.py`, `evaluate.py`, `evaluate_terms.py`) are not in the repo. Coders read them from the public GitHub repo with web_fetch and port them. G2, G3 and G2b catch port errors.
- **The Table 3 numbers** are transcribed at item 7. Nothing in the plan assumes them beyond the MF class AUCs in the spec (0.882 MLP, 0.903 DGZ).
- **G9a reduced batches** (decision 3) and **D19** need user/supervisor sign-off before G8. Both are stated in the docs and the runbook.
- **G4, G5-ESM/MPNet, G7, G8, G9b, G10 and G11 are not run in this workflow.**

## Verification log

- **FEAT-005 G0 deviation (2026-10-12; accepted by the orchestrator, option c).**
  - `go_ppi_domain/tests` gave 326 passed, 3 skipped, 3 failed, 6 errors. The baseline is 334 passed, 3 skipped, 1 known failure (`test_real_protocol_amendments_table_parses`, which still fails).
  - The 8 extra non-passes (`test_withholding_generator.py::test_make_conditions_counts[24-121]` and `[20-101]`, plus 6 errors in `test_regenerate_conditions.py`) all stop at the free-disk guard of `scripts/go_withholding/make_conditions.py`: "only 4.05 GB free (< 5 GB)". Free disk fell from 5.7 GiB to 3.7 GiB because macOS swap grew to 6 GiB during the G9a smoke. The study outputs total about 7 MB.
  - Evidence that the cause is environmental: `git diff --stat` and `git status --short --untracked-files=all` against HEAD 6a8eda3 show nothing for `go_ppi_domain/`, `scripts/go_withholding/`, `config/go_withholding/`, `contrastive_learning/`, `trials_domain/`, `scripts/trials_fusion_confirm/` and `scripts/run_learning_curve_point.py` (0 lines). No protected file was modified.
  - The other G0 suites match the baseline: `trials_domain/tests` 108 passed, 1 skipped; the career/CVE files 8 passed.
  - **Open item (validation-smoke step and final report):** re-run `.venv/bin/python -m pytest go_ppi_domain/tests -q -p no:cacheprovider` once disk is freed. The validation-smoke step runs it first if at least 5.5 GB is free at that point (`df -k .`), and it must give 334 passed, 3 skipped, 1 known failure. Otherwise the step reports the item as still open.
- **G0 open item closed (2026-10-12, integration step).** At 5.01 GB free, `go_ppi_domain/tests` gave 334 passed, 3 skipped, 1 known failure, which is the baseline. Every other integration check also passed: the slow go_function suite (227 passed), all generator re-runs ('unchanged'), and a fresh G9a smoke plus `analyze.py --smoke` (exit 0, byte-identical to the previous smoke). There are no '/Users/' strings in any output and no changes in protected paths. Evidence is in FEAT-005 findings.
