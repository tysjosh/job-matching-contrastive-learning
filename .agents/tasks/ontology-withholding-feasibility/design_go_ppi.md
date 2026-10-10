# GO/PPI per-target withholding study: technical design

Status: **design v6**. v2 revised v1 (3 HIGH / 10 MEDIUM / 9 NIT, §17). v3 revised v2 after the second review (0 HIGH / 5 MEDIUM / 10 NIT, §18). v4 revised v3 after the third review (0 HIGH / 2 MEDIUM / 7 NIT, §19). v5 revised v4 after the fourth review (0 HIGH / 1 MEDIUM / 3 NIT, §20). v6 revises v5 after the fifth review (`design_review.json`, CHANGES_REQUESTED, 0 HIGH / 1 MEDIUM / 4 NIT); every finding is addressed, see §21. Covers gap-list items 2–11 of `findings_go_ppi.md` (iii). Gap item 1 (code freeze) is handled by a content pin, not by committing (§13). Companion protocol: `/Users/olukotunjosh/Downloads/CDCL/.agents/tasks/ontology-withholding-feasibility/protocol_go_ppi.md`. **Amended by AM1 (2026-10-10, before G2; protocol §13 row AM1, this design §22): one target per condition (singleton packs).**

Path convention: `$R` is the literal absolute path `/Users/olukotunjosh/Downloads/CDCL`. Every path in this document is `$R/...`, i.e. absolute. Code never uses relative paths: each new script defines `ROOT = Path(__file__).resolve().parents[2]` (scripts) or `parents[1]` (modules), asserts `ROOT == Path("/Users/olukotunjosh/Downloads/CDCL")`, inserts it into `sys.path` (the existing `run_learning_curve_point.py` pattern), and resolves every default path against it. Scripts are invoked by absolute file path (`"$PY" "$R/scripts/go_withholding/<name>.py"`), not `-m`, because `scripts/` has no `__init__.py`.

## 0. Overview

The study trains the GO/PPI projection head under four cells (baseline, loss_only, neg_only, both) on train sets from which the GO-BP-annotated train-partition proteins of selected target terms are withheld, down to k ∈ {0, 1, 5, 10, 25} retained proteins per target, plus one shared no-withholding `all` condition. It measures per-target hard AUC (grade 2 vs grade 1) on a frozen, protein-disjoint test set with an unchanged candidate pool. Everything new is additive and default-off:

- leak-pruned features: masked partner symbols/aliases in `encoder_view`, a GAF with/from filter, a rebuilt pruned GO index; produced by rewriting `proteins.jsonl` next to a verbatim copy of the reference `pairs.jsonl`, then re-running the deterministic splitter;
- a frozen target table with any-role target membership (`T_train`) separated from retention membership (`P_train`);
- a withholding generator;
- runner hardening: index-hash preflight, a strict epoch guard that turns swallowed batch failures into a fatal error, a run-time negative-selection recorder, condition assertions, a content-hash code pin;
- one selector flag (`go_ppi_unannotated_anchor_window`, shared by all four cells) so leak-pruned anchors with no BP terms get the control's random window in every cell (§5.6);
- one new loss term (`go_ontology_loss_weight`) backed by a loss-only `GoRelationTable` attached to the loss engine only, plus a pure `go_relation_loss` function;
- per-epoch loss and gradient diagnostics through an epoch-end callback;
- an offline negative audit;
- a cached-embedding evaluator with per-target metrics and a relation probe;
- a text-similarity covariate;
- a two-way (pack × seed) cluster bootstrap;
- a resumable sweep driver that never runs the full grid unasked.

**Technology stack (locked).** Python 3 in `$R/.venv` (`$R/.venv/bin/python`), torch 2.8.0, numpy 2.3.2, scipy 1.16.1, scikit-learn 1.7.1 (tests only), the existing `sentence-transformers` `all-mpnet-base-v2` (frozen), pytest. No new dependencies; `statsmodels` is not installed and must not be added. CPU only (Apple M2, 8 GB). Logic is ported from `/tmp/goinv/{a_split,b_gaf,c_targets,d_refine,e_negaudit,f_rawdeg,g_relation}.py` and `/tmp/v7inv/probe.py`; nothing imports from `/tmp`.

### 0.1 Inconsistencies between the fixed decisions and the code, and their resolution

1. **"Rebuild converted data with the same split seed" cannot be done by re-running the converter.** `data_converter.build` samples grade-0 pairs from `list(view_ids - forbidden)` over a `set`, so its output depends on `PYTHONHASHSEED` (review re-run: grade-0 overlap 802/281,640; sha differs even with `PYTHONHASHSEED=0`). Its `GoIndex.build_or_load` can also write into `$R/preprocess/go_ppi/`.
   - **Resolution:** the converter is not called and not modified. `$R/preprocess/go_ppi/pairs.jsonl` is copied byte-for-byte; only `proteins.jsonl` is regenerated (same ids, same order, masked `encoder_view`, pruned `go_uris`/`coarse_uris`/`n_terms`). `data_splitter.build_splits` is deterministic in (proteins ids, pairs, seed) — the review reproduced all four reference split files byte-identically — so the partition and every (anchor, partner, grade) tuple are preserved by construction and checked by structural hashes (§1.4).
   - Pruning removes all BP annotation from 1,187 of 15,155 proteins [v1 measurement]; they stay in the dataset with `go_uris=[]`. The reference data has none [review v3, ran: 0 of 15,155], so this is a new case for the selector. Its two roles behave differently:
     - **As pool members**, `_go_ordered` sorts them last in GO-tiered cells (unchanged behaviour).
     - **As anchors**, the current `select_batch_negatives` skips both GO branches when `anchor_uris` is empty and falls to the final `else`: a uniform draw from the **whole** tier pools, with no 34% window. In neg_only/both that would change window narrowing as well as GO ordering for these anchors, which is exactly what `randwin` holds fixed (findings D). The review estimates ~115 of 4,547 train-record anchors and ~140 of 4,948 train pool anchors are affected [review v3, approximate].
     - **Resolution (§5.6):** a new additive selector flag `go_ppi_unannotated_anchor_window` (default `False`, today's behaviour) routes empty-annotation anchors in GO-tiered cells through the `random_window` path. It is set to `true` in **all four** study configs, so it is a shared field and not part of either factor. Under it an unannotated anchor gets byte-identical negatives in all four cells, and the neg factor acts only on annotated anchors. Per-condition counts are recorded (§3.3).
   - Unannotated proteins are excluded from L_GO pairs (`annotated=False`).
2. **"Test hash unchanged" cannot mean `c65a1d2c…`.** Masking changes `encoder_view` and pruning changes `skill_uris` inside records, so the new test file has a new sha256. Resolution: the frozen evaluation file is `$R/preprocess/go_withholding/splits/test.jsonl`; its sha256 is recorded at G1 in `frozen_hashes.json` and `protocol_pins.json` and asserted by every run. Its **structural hash** (§1.4) must equal that of the reference `c65a1d2c…` file, which proves the candidate pool is unchanged.
3. **"Withheld proteins = P(t)" misses most target-annotated train proteins.** The train partition has 10,608 proteins; all 10,608 are members of some train pool, 4,948 are pool anchors and 4,709 have positive records. So 5,899 have no positive record: 5,498 are pool members only and 401 are pool anchors (and members) without positives [review v2, ran]. Resolution: membership is split (§2.1). `T_train(t)` (any role) defines what is withheld; `P_train(t)` (≥1 positive record) defines n(t), removal fraction and retention eligibility, as preregistered. W(pack,k) = ∪ (T_train(t) \ R_k(t)).
4. **"Drop withheld proteins from validation files" is vacuous** (train and validation share 0 proteins). Resolution: one shared target-free validation set removes every validation protein whose pruned BP closure contains any selected target, in any role (§3.2).
5. **The trainer swallows loss errors.** `/Users/olukotunjosh/Downloads/CDCL/contrastive_learning/trainer.py` `_train_batch` catches every exception and returns `None`; `_train_epoch` counts it as a failed batch and continues. A broken loss-on cell would train on a subset of batches (or none) and still save `checkpoint_epoch_{N-1}.pt`. Resolution: the trainer's epoch end passes `epoch_stats` to callbacks; the runner installs a non-catching `StrictEpochGuard` (§4.5) that raises outside the per-batch `try`, so `train()` saves an emergency checkpoint and re-raises, and the runner exits 2.
6. **The batch processor swallows selector errors.** `/Users/olukotunjosh/Downloads/CDCL/contrastive_learning/batch_processor.py` `_select_with_injected_selector` catches any selector exception and falls back to OSCAR/dummy negatives; a `None` return also falls back. Resolution: the run-time `SelectionRecorder` (§4.6) counts calls, `None` returns and exceptions; the sweep requires both counters to be 0.
7. **`GoIndex.build_or_load` silently rebuilds an unpruned index and overwrites the cache** if the pruned pickle is missing, unreadable or for the wrong aspect, and neg_only/both reach it through `run_config.build_go_matcher`. Resolution: the runner and every study script assert `sha256(config.go_index_cache) == frozen_hashes["go_index_P_sha256"]` and the pruning meta before any matcher is built; G1 makes the pruned pickle and all frozen split files read-only (`0o444`), so a rebuild attempt raises `PermissionError` instead of overwriting.
8. **Checkpoint selection is variant-dependent** if `best_checkpoint.pt` is used. Resolution: the selected checkpoint is the final epoch, `checkpoint_epoch_{epochs-1}.pt` (14 in the grid). Study runs never set `validation_path`.
9. **L_GO's pair set would couple the factors if it included negatives.** Resolution: L_GO uses only batch anchor and positive proteins, which are identical across cells for a given (condition, seed).
10. **Per-seed retention draws would need 5× the condition directories** (~7.5 GB; 9.6 GiB free on 2025-10-09). Resolution: one retention permutation per target (`retention_seed=20251009`), shared by all seeds and cells; stated as a limitation in the protocol.
11. **Packs vs per-target caps.** Three targets at ≤5% each can exceed 8%; and a pack-mate can make a target's retention set infeasible. Both are explicit packing constraints (§2.6). Under AM1 (§22) packs are singletons, so both reduce to the per-target checks.
12. **"with/from names a protein" is ambiguous.** Resolution: drop an annotation when any with/from token is a `UniProtKB:` accession (isoform suffix stripped) resolving to a **human** gene symbol other than the annotated gene. v1 measured 12,871 of 138,976 kept BP rows dropped (IBA 11,267, IGI 1,323, ISS 205, IMP 76) — recomputed and recorded at G1, not assumed. ComplexPortal/ENSEMBL/HGNC tokens are reported as an ungated residual (§1.5).
13. **Dropping `GO:0005515` descendants is a no-op for BP** (MF terms). Implemented generically (`excluded_term_roots`); the BP drop count (expected 0) is recorded.
14. **Projection parameters.** There is no `trainer.model.projection_head` attribute guarantee; the MPNet encoder is a separate, frozen `trainer.text_encoder`. Resolution: "projection params" = `[p for p in trainer.model.parameters() if p.requires_grad]`, with a check that none of them is a `trainer.text_encoder` parameter; names and numel are recorded.

## 1. Leak-pruned features (gap 2)

### 1.1 New module `$R/go_ppi_domain/leak_pruning.py`

Pure functions, unit-testable without data:

```python
ALIAS_SOURCES = frozenset({"Ensembl_HGNC_symbol", "Ensembl_HGNC_alias_symbol",
    "Ensembl_HGNC_prev_symbol", "UniProt_GN_Name", "UniProt_GN_Synonyms", "BioMart_HUGO"})
MASK_TOKEN = "[GENE]"
VIEW_SEP = " — "          # data_converter writes f"{g} — {text}"

@dataclass(frozen=True)
class MaskLexicon:
    owner: dict[str, frozenset[str]]   # entry -> symbols that own it (self-exemption)
    alnum: frozenset[str]              # purely [A-Za-z0-9] entries (dict lookup pass)
    special: tuple[str, ...]           # entries with - . / : , longest first (regex pass)
    params: dict

def load_symbol_aliases(aliases_gz: Path, info_gz: Path, symbols: set[str]) -> dict[str, set[str]]
    # ENSP -> symbol from protein.info; alias rows whose source field shares a token with
    # ALIAS_SOURCES; each symbol always includes itself.
def build_mask_lexicon(texts: dict[str, str], aliases: dict[str, set[str]],
                       min_alias_len: int = 3, min_symbol_len: int = 2,
                       max_doc_freq: float = 0.01) -> MaskLexicon
    # Primary symbols with len >= 2 are always eligible. Aliases need len >= 3, >= 1
    # uppercase letter or digit, no whitespace, and doc frequency (share of protein texts
    # containing the token whole-word) <= 1%. Excluded entries are listed in the manifest.
def mask_view_text(self_symbol: str, text: str, lexicon: MaskLexicon) -> tuple[str, int]
    # Replace whole-word occurrences of every entry NOT owned by self_symbol with MASK_TOKEN.
    # Whole-word = (?<![A-Za-z0-9])X(?![A-Za-z0-9]), case-sensitive (the leak-audit
    # semantics). Pass (a): [A-Za-z0-9]+ runs looked up in `alnum`; pass (b): one compiled
    # alternation over `special`. Returns (masked_text, n_replacements).
def human_accession_map(gaf_gz: Path, aliases_gz: Path, info_gz: Path) -> dict[str, str]
    # UniProt accession -> human symbol (GAF col2->col3, plus STRING UniProt_AC aliases).
def other_db_symbol_map(aliases_gz: Path, info_gz: Path) -> dict[str, str]
    # "HGNC:<n>" and "ENSEMBL_GeneID:<ENSG>" -> symbol, from STRING alias sources containing
    # "HGNC_ID" / "Ensembl_gene". Used only by the ungated residual metric (§1.5).
def rewrite_proteins(reference_proteins: Path, out_proteins: Path,
                     annotation_index: GoIndex,
                     view_text_fn: Callable[[str, str], tuple[str, int]]) -> dict
    """Reference proteins.jsonl -> same ids, same order; only encoder_view, go_uris,
    coarse_uris, n_terms change.
      prefix = f"{g}{VIEW_SEP}"; assert row["encoder_view"].startswith(prefix)  (fatal)
      raw = row["encoder_view"][len(prefix):]
      encoder_view = prefix + view_text_fn(g, raw)[0]
      go_uris = sorted(annotation_index.terms_for(g)); coarse_uris = sorted(annotation_index.coarse_for(g))
      n_terms = len(go_uris); protein_id and gene unchanged; key order unchanged.
    Written with json.dumps(row, ensure_ascii=False) + "\\n" (the converter's _write_jsonl format).
    Returns {"n": .., "ids_sha256": sha256("\\n".join(ids in order)), "replacements": {quantiles},
             "empty_go_uris": .., "token_mask_rate": ..}."""
```

The self prefix `"{SYMBOL} — "` is kept verbatim; masking applies only to the annotation text. v1 prototype token-mask rate: 3.1% of tokens.

### 1.2 Additive changes to `$R/go_ppi_domain/go_ontology.py`

- `parse_gaf(..., with_from_symbol_map: Optional[Dict[str,str]] = None, drop_with_from_human_protein: bool = False, excluded_term_roots: Iterable[str] = (), descendants_of: Optional[Callable[[str], FrozenSet[str]]] = None)`.
  - With `drop_with_from_human_protein`, a row is dropped (`stats["drop_with_from_human_protein"]`) if any `|`/`,`-separated col-8 token starting with `UniProtKB:` (isoform suffix `-n` stripped) maps to a symbol ≠ col 3. Self-references are kept.
  - A row is dropped (`stats["drop_excluded_root_descendant"]`) if `go_id` is a descendant of any `excluded_term_roots` (is_a + part_of over the full OBO).
  - Defaults reproduce current behaviour exactly (fixture test).
- `GoIndex.build` gains two keyword-only parameters, appended after the existing ones. This is the single signature used everywhere in this design (§1.4 step 3):

  ```python
  GoIndex.build(obo_path, gaf_path, aspect="P", excluded_evidence=..., excluded_terms=...,
                coarse_max_depth=..., *, leak_pruning: Optional[dict] = None,
                with_from_symbol_map: Optional[Dict[str, str]] = None)
  ```

  - `leak_pruning` keys: `"with_from_human_protein": bool`, `"excluded_term_roots": list[str]`.
  - `leak_pruning["with_from_human_protein"]` true and `with_from_symbol_map is None` → `ValueError("leak_pruning requires with_from_symbol_map")`.
  - The descendant set for `excluded_term_roots` is computed from the full, all-namespace `parse_obo` parents (is_a + part_of), not from the aspect-restricted ancestors.
  - Stores `meta["leak_pruning"] = {"with_from_human_protein": bool, "excluded_term_roots": [...], "accession_map_size": int, "gaf_stats": {...}}`. With `leak_pruning=None` the key is absent; existing pickles and meta are unchanged.
- `build_or_load` is not changed. Study code never calls it directly; the only indirect caller (`build_go_matcher`, neg_only/both) is protected by the preflight of §4.2 and the read-only bit.

`$R/go_ppi_domain/data_converter.py` is **not modified** (v1 §1.3 dropped).

### 1.3 Constants module `$R/go_ppi_domain/withholding_constants.py`

```python
ROOT = Path("/Users/olukotunjosh/Downloads/CDCL")
RAW_SHA256 = {...}            # the 8 raw files of protocol §11, from setup_baseline.md
REFERENCE_CONVERTED = ROOT / "preprocess/go_ppi"
REFERENCE_SPLITS = ROOT / "preprocess/go_ppi_splits"
REFERENCE_TEST_SHA256 = "c65a1d2ca0e5c16a1c69ecd70d921e670531d325c52625aa15dba88c2e2481e4"
STUDY_PRE = ROOT / "preprocess/go_withholding"
CONVERTED = STUDY_PRE / "converted"; SPLITS = STUDY_PRE / "splits"
VALIDATION_TF = STUDY_PRE / "validation_targetfree"; TARGETS = STUDY_PRE / "targets"
CONDITIONS = STUDY_PRE / "conditions"
STUDY_RES = ROOT / "results/go_withholding"
PINS = STUDY_RES / "protocol_pins.json"; CODE_PIN = STUDY_RES / "code_pin.json"
CONFIG_DIR = ROOT / "config/go_withholding"
EMBED_CACHE = ROOT / "embedding_cache/go_withholding_text_embeddings.pt"
PROTECTED = (REFERENCE_CONVERTED, REFERENCE_SPLITS, ROOT / "preprocess/go_ppi_lc", ROOT / "dataset/go_ppi")
RETENTION_SEED = 20251009; ANALYSIS_SEED = 20251009   # PACK_SEARCH_SEED removed by AM1 (§22)
KS = (0, 1, 5, 10, 25); SEEDS = (13, 21, 42, 87, 123); CELLS = ("baseline", "loss_only", "neg_only", "both")
GRID_EPOCHS = 15

def assert_writable_target(path: Path) -> None   # fatal if path is inside any PROTECTED dir
def sha256_file(path: Path) -> str
def structural_sha256(records_path: Path) -> str
    # sha256("\n".join(sorted(f"{r['metadata']['protein_id']}\t{r['metadata']['partner_id']}\t{r['metadata']['go_grade']}" for r))) 
CODE_PIN_PATHS = ("contrastive_learning", "go_ppi_domain", "scripts", "trials_domain",
                  "run_phase1_embedding_evaluation.py")
def code_sha256() -> tuple[str, list[str]]   # §13; (hash, sorted pinned .py files)
def imported_repo_modules() -> list[str]    # §13; repo .py files in sys.modules, .venv excluded
def append_pin(gate: str, values: dict) -> None   # appends {"gate", "time", **values} to PINS
RUN_DIR_IGNORE = (".attempt", ".groupfailed")     # renamed dirs skipped by sweep and analysis (§11, §12.3)
def group_ok_valid(run_dirs: list[Path]) -> list[bool]
    # per dir (cell order): GROUP_OK.json exists, its run_ids == [d.name for d in run_dirs],
    # and its lc_manifest_sha256[d.name] == sha256(d/"lc_manifest.json") == DONE.json["lc_manifest_sha256"].
    # The single predicate used by the sweep's eligibility rule and the analysis preflight.
PROTOCOL = ROOT / ".agents/tasks/ontology-withholding-feasibility/protocol_go_ppi.md"
AMENDMENT_ID_RE = r"^AM[1-9][0-9]*$"
AMENDMENT_COLUMNS = ("id", "date", "gate", "groups cleared", "cause", "fix", "re-pin")
def parse_amendment_rows(protocol_path: Path = PROTOCOL) -> list[dict[str, str]]
    # §12.3 (review v5 finding 1). Takes the text strictly between the line starting "## 13." and
    # the line starting "## 14."; finds the markdown table whose header cells, stripped and
    # lower-cased, equal AMENDMENT_COLUMNS; returns one dict per data row (separator row skipped),
    # keyed by AMENDMENT_COLUMNS, each cell split on "|" and stripped of whitespace. Raises
    # ValueError if the section, the table or its header is missing, or a row has != 7 cells.
```

### 1.4 Build script `$R/scripts/go_withholding/build_pruned_features.py`

CLI (all defaults come from `withholding_constants`, shown here):

```
--bulk-dir            $R/dataset/go_ppi/bulk
--reference-converted $R/preprocess/go_ppi            (read-only)
--reference-splits    $R/preprocess/go_ppi_splits     (read-only)
--out-converted       $R/preprocess/go_withholding/converted   (must not exist, or be empty)
--out-splits          $R/preprocess/go_withholding/splits      (must not exist, or be empty)
```

Steps (any failure: message, exit 2, partial output left in place but never reused because step 1 refuses non-empty dirs):

1. `assert_writable_target` for both outputs; refuse if either contains files. No `--force`.
2. Verify the raw sha256s against `RAW_SHA256`.
3. Build the pruned BP index **in memory** with `GoIndex.build(obo, gaf, "P", leak_pruning={"with_from_human_protein": True, "excluded_term_roots": ["GO:0005515"]}, with_from_symbol_map=human_accession_map(...))` and `save()` it to `out-converted/go_index_P.pkl`. `build_or_load` is never called.
4. Build the alias table and mask lexicon from the reference `proteins.jsonl` texts (prefix stripped); write `out-converted/mask_manifest.json` = `{lexicon_size, symbols_unmaskable, aliases_dropped_docfreq, alias_sources, params}`.
5. `shutil.copyfile($R/preprocess/go_ppi/pairs.jsonl, out-converted/pairs.jsonl)`; then `rewrite_proteins($R/preprocess/go_ppi/proteins.jsonl, out-converted/proteins.jsonl, pruned, lambda g, t: mask_view_text(g, t, lexicon))`. Write `out-converted/convert_manifest.json` = the reference manifest dict plus `{"annotation_index": pruned.meta, "view_masking": true, "mask_manifest_sha256": ..., "pairs_copied_from": "$R/preprocess/go_ppi/pairs.jsonl", "pairs_sha256": ..., "rewrite_stats": ...}`.
6. Assert `sha256(out pairs) == sha256(reference pairs)` (expected prefix `f789f237`) and that the protein-id list **in order** equals the reference list.
7. `data_splitter.build_splits(out-converted, out-splits, seed=42, val_no_interaction_cap=20, test_no_interaction_cap=0)` (the reference parameters; review reproduced reference files with them). Assert:
   - entity overlap 0/0/0 (the splitter asserts it; re-checked here);
   - for `train.jsonl`, `validation.jsonl`, `test.jsonl`: `structural_sha256(new) == structural_sha256(reference)`;
   - `negative_pools.jsonl`: the parsed rows (`split, anchor_id, weak_evidence, no_interaction`) equal the reference rows exactly (they contain ids only);
   - record counts 15,708 / 5,587 / 22,341 and 7,041 pool rows.
8. Write `out-splits/frozen_hashes.json`:
   ```json
   {"schema_version": 2,
    "test_sha256": "…", "test_structural_sha256": "…", "reference_test_sha256": "c65a1d2c…",
    "train_sha256": "…", "validation_sha256": "…", "negative_pools_sha256": "…",
    "pairs_sha256": "…", "proteins_sha256": "…", "go_index_P_sha256": "…",
    "mask_manifest_sha256": "…", "raw": {...}, "code_sha256": "…"}
   ```
9. `os.chmod(path, 0o444)` for `go_index_P.pkl`, `proteins.jsonl`, `pairs.jsonl` and the four split files. `append_pin("G1", frozen_hashes)`.

### 1.5 Leak re-measurement `$R/scripts/go_withholding/measure_leakage.py`

```
--splits $R/preprocess/go_withholding/splits  --converted $R/preprocess/go_withholding/converted
--reference-splits $R/preprocess/go_ppi_splits  --reference-converted $R/preprocess/go_ppi
--out $R/preprocess/go_withholding/leak_audit.json
```

For train and test, per grade, for reference and new data:

- (a) **symbol mention:** the partner's symbol appears whole-word in the other side's annotation text, either direction (port of `/tmp/goinv/a_split.py`);
- (b) **symbol + lexicon alias mention;**
- (c) **with/from hit (UniProtKB):** a *kept* annotation of one side names the other in col 8 via `human_accession_map`, computed against the index each side was built with (reference: default guards; new: pruned) (port of `b_gaf.py`). For the new data (c) is ≈0 by construction; it verifies the implementation.
- (d) **ungated residual:** kept BP rows whose with/from has `HGNC:`/`ENSEMBL_GeneID:` tokens resolving (via `other_db_symbol_map`) to the partner; and the count of positives where either side has a kept row with a `ComplexPortal:` token (not resolvable to one partner). Reported, never gated. Plus a count of descriptive complex-name mentions ("component of the … complex").

Reference values to reproduce (sanity, not gate): test (a) 11.07% / 0.17% / 0%; (c) 5.92% / 0.41% / 0%.

**Acceptance (fatal; exit 3; the study does not start):** on the new test split, (a), (b) and (c) at grade 2 are each ≤ 0.5%, and each grade-2 rate exceeds the grade-1 rate by ≤ 0.5 percentage points. The JSON records `{dataset: {split: {grade: {n, a, b, c, d_other_db, d_complexportal}}}}`, `"pass"`, `"failures"`, and is pinned (`append_pin("G1-leak", {...})`).

## 2. Frozen target table (gap 3)

### 2.1 Module `$R/go_ppi_domain/withholding_targets.py`

```python
def bp_graph(obo: Path, index: GoIndex) -> BpGraph      # parents/children restricted to index.ancestors keys,
                                                         # via go_ontology.parse_obo (is_a + part_of; regulates
                                                         # and obsolete terms never enter)
def descendants(graph) -> dict[str, frozenset[str]]
def n_sib(t, graph) -> frozenset[str]   # ∪ children(p) for p in parents(t), minus {t} ∪ ancestors(t) ∪ descendants(t)
def n_2(t, graph) -> frozenset[str]     # terms ≤2 undirected hops, same exclusion
def partition_roles(train_records, pool_rows) -> dict[str, set[str]]
    # {"positive": proteins in train records (either slot), "pool_anchor": .., "pool_member": ..,
    #  "any": union}
def term_proteins(index: GoIndex, proteins: Iterable[str], terms) -> dict[str, set[str]]
def undirected_positive_pairs(train_records) -> set[tuple[str, str]]
def degree(pairs) -> Counter
def support_S(t, nbr_terms, P_train, deg) -> dict   # {"S": Σ deg(p) for p ∈ NP(t), "proteins": |NP(t)|}
def retention_eligible(t, P_train, W, train_records, pools, views) -> set[str]   # §2.2 rule 5 / §2.6
def filter_candidates(...) -> list[dict]
def greedy_non_nested(rows, P_train, ancestors, max_jaccard=0.2) -> list[dict]
def match_strata(low, high, n_pairs=12) -> list[dict]
def singleton_packs(targets, pack_ok) -> list[list[str]]   # AM1 (§22); replaces pack_targets/pack_search
```

All inputs are post-fix data (`$R/preprocess/go_withholding/{converted,splits}`). Membership uses the **pruned** index propagated closure (`index.closure_for`). Definitions:

```text
train partition   = partition_roles(train.jsonl, negative_pools rows with split=="train")["any"]
T_train(t) = { p in train partition (record either slot, pool anchor, pool member) : t in closure(p) }
P_train(t) = { p in T_train(t) : p has >= 1 positive train record }       # n(t) = |P_train(t)|
P_test(t)  = { test anchors a : t in closure(a), a has >=1 grade-2 and >=1 grade-1 test record }
P_val(t)   = { validation proteins in any role : t in closure(p) }
NP_train(t; N) = ∪_{t' in N(t)} P_train(t') \ T_train(t)
NP_test(t; N)  = { test proteins p (any role) : closure(p) ∩ N(t) ≠ ∅ and t ∉ closure(p) }
W_t(0) = T_train(t)
```

### 2.2 Candidate rule (fixed)

1. `n(t) = |P_train(t)| ≥ 40`.
2. `|P_test(t)| ≥ 8`.
3. `removal_frac(t)` = undirected train positive pairs with an endpoint in `T_train(t)` (equivalently `P_train(t)`) / all undirected train positive pairs ≤ 0.05.
4. `|P_train(t)| ≤ 0.10 ×` train proteins with positives.
5. `|retention_eligible(t, W=T_train(t))| ≥ 25`, where p ∈ P_train(t) is eligible iff there is a train positive record r with endpoints {p, q}, q ∉ W, and the pool row of r's anchor has ≥1 member ∉ W that is in the condition's views. (Guarantees the record survives withholding and the empty-pool rule at every k.)
6. Non-nested with protein-Jaccard (over P_train) ≤ 0.2. Greedy order: |P_test(t)| descending, then GO id ascending. Reject if ancestor/descendant of an accepted term or Jaccard > 0.2 with one (port of `c_targets.py`).

### 2.3 Strata

S_sib on the non-nested set. `low` = S_sib ≤ 33rd percentile, `high` = S_sib ≥ 67th (numpy `percentile`, linear); middle excluded. S_2 stored for sensitivity only.

### 2.4 Matched selection

`scipy.optimize.linear_sum_assignment` on low × high; cost = Euclidean distance on z-scored (log n(t), log |P_test(t)|), z over low ∪ high. Sort pairs by (cost, low GO id, high GO id). Accept greedily while each new target's protein-Jaccard with every accepted target ≤ 0.2, until 12 pairs (24 targets). The accepted pairs are numbered `m01, m02, …` in acceptance order; pairs that passed the Jaccard check but were not needed form the ordered **replacement queue** (`replacement_queue` in the table). Size rule, with no other sizes permitted:

- ≥ 12 pairs accepted: the first 12 are used (24 targets), `"shortfall": false`.
- 10 or 11 pairs accepted: keep **exactly the first 10** in acceptance order (20 targets), `"shortfall": true`. An 11th pair is never used; it goes to the front of the replacement queue.
- < 10 pairs: fatal (exit 3), stop and ask the user.

Balance (SMD of log n and log support) is reported, not gated.

### 2.5 Neighbour-protein sets

For each selected t: `NP_train_sib`, `NP_test_sib`, `NP_train_2`, `NP_test_2`.

### 2.6 Packing

**Amended by AM1 (§22): one target per pack.** `singleton_packs(targets, pack_ok)` returns `[[t] for t in targets]` in target order (pair number, low before high), so pack `p01` holds the first target, `p02` the second, and so on; `n_packs = n_targets`. `pack_ok({t})` checks, with `W = T_train(t)`, the k = 0 removal fraction ≤ `pack_removal_cap` (0.08) and `|retention_eligible(t, W)| ≥ 25`. Both are implied by the candidate rule (§2.2 rules 3 and 5), so a failure is a protocol stop (exit 3). There are no pack-mate constraints, no randomized search, no slot cycling and no 24 → 20 packing fallback; the §2.4 size rule (24 or 20 targets) is unchanged. The table has no `pack_search_log`. The rest of this section is the superseded pre-AM1 procedure, kept for the record.

*Superseded (pre-AM1):* Partition the 24 targets into 8 packs of 3. Let `W_pack(0) = ∪_{t in pack} T_train(t)`. Constraints for every pack:

- pairwise `P_train(a) ∩ P_train(b) = ∅` (implies `P_train(a) ∩ T_train(b) = ∅`, so retained proteins are never withheld by a pack-mate);
- b ∉ N_2(a) ∪ ancestors(a) ∪ descendants(a), and vice versa;
- `|T_train(b) ∩ NP_train_sib(a)| / max(1, |NP_train_sib(a)|) ≤ 0.05` (a pack-mate does not erase the neighbour support being tested);
- ≥1 low and ≥1 high target;
- pack removal fraction (undirected train positive pairs with an endpoint in `W_pack(0)`) ≤ 0.08;
- **retention feasibility (review finding 5):** for every t in the pack, `|retention_eligible(t, W=W_pack(0))| ≥ 25`.

Pack shape: 24 targets → 8 packs of 3; 20 targets → 6 packs of 3 plus 1 pack of 2 (the 2-pack still needs one low and one high target). Same constraints in both shapes.

Search, complete and deterministic:

1. Seeded randomized greedy over the current target set (`random.Random(0)` re-created per attempt, up to 20,000 restarts); the first feasible partition in restart order wins.
2. If none: replace the highest-cost pair in the current set (ties: higher pair id first) with the next pair from the replacement queue, subject to the §2.4 Jaccard check against the remaining targets (a queued pair that fails it is skipped and consumed). Repeat step 1. Each replacement is logged in `pack_search_log` as `{"phase": 24 | 20, "removed": "mXX", "added": "mYY", "restarts_tried": 20000}` (a queued pair skipped by the Jaccard check is logged as `{"phase", "skipped": "mYY", "reason": "jaccard"}`).
   - **Slot cycling (stated explicitly).** Step 2 replaces only the slot of the highest-cost pair *currently* in the set. Queued pairs have cost ≥ every accepted pair (they come later in the cost-sorted order), so after the first replacement the added pair is the highest-cost pair and is the one replaced next. In effect, the last slot cycles through the queue while the other pairs (m01–m11 in the 24-target phase) are never revisited. Implementations must recompute "highest-cost pair currently in the set" each time; they must **not** walk down the original pairs (m12, then m11, …).
   - Example log (`restarts_tried` omitted; 24-target phase, queue `[m13, m14]`, both infeasible): `[{"phase": 24, "removed": "m12", "added": "m13"}, {"phase": 24, "removed": "m13", "added": "m14"}]`.
3. If the queue is exhausted while still at 24 targets: drop to the 20-target set = the **first 10 pairs of the original acceptance order** (m01–m10; replacements discarded), set `"shortfall": true`, and repeat steps 1–2 with the 6×3 + 1×2 shape. The queue for this phase is **m11, m12 (in that order), followed by the original replacement queue from its start**. Every queued pair is again subject to the §2.4 Jaccard check against the current set. Step 2's slot cycling applies unchanged: the first replacement removes m10.
   - Example log continuation (20-target phase, original queue `[m13, m14]`, all infeasible): `[{"phase": 20, "removed": "m10", "added": "m11"}, {"phase": 20, "removed": "m11", "added": "m12"}, {"phase": 20, "removed": "m12", "added": "m13"}, {"phase": 20, "removed": "m13", "added": "m14"}]`; then step 4. Every log entry carries `"phase": 24 | 20`.
4. If the queue is exhausted at 20 targets: fatal (exit 3), stop and ask the user.

After success the table records `n_targets`, `n_packs`, `n_conditions = 5·n_packs + 1` and `n_runs = 4·5·n_conditions`. Under AM1: 121 / 2,420 at 24 targets; 101 / 2,020 at 20 (pre-AM1: 41 / 820 and 36 / 720). Every downstream count (G3, sweep preflight, completeness rule, bootstrap) reads these fields; none hard-codes a count.

### 2.7 Retention order

For each target, after packing:

```python
eligible = retention_eligible(t, P_train, T_train(t), train_records, pools, views)   # >= 25 by §2.6 (AM1: W_pack(0) = T_train(t))
order = sorted(eligible)
random.Random(f"20251009|{go_id}").shuffle(order)
R_k(t) = order[:k]          # nested by construction
```

### 2.8 Script and output

`$R/scripts/go_withholding/build_target_table.py --converted $R/preprocess/go_withholding/converted --splits $R/preprocess/go_withholding/splits --obo $R/dataset/go_ppi/bulk/go-basic.obo --out $R/preprocess/go_withholding/targets/target_table.json`. Preflight: `frozen_hashes.json` hashes match the files and `protocol_pins.json` has G1. Refuses to overwrite. Writes `target_table.json`, `target_table.json.sha256`, `candidates.json` (all §2.2 rows) and `append_pin("G2-targets", {...})`.

Schema `target_table.json` (schema_version 2):

```json
{
  "schema_version": 2,
  "inputs": {"train_sha256": "…", "test_sha256": "…", "validation_sha256": "…",
             "negative_pools_sha256": "…", "go_index_P_sha256": "…", "obo_sha256": "…",
             "code_sha256": "…"},
  "params": {"min_train_proteins": 40, "min_test_support": 8, "max_removal_frac": 0.05,
             "max_protein_frac": 0.10, "min_retention_eligible": 25, "max_jaccard": 0.2,
             "strata_quantiles": [33, 67], "n_pairs": 12, "min_pairs": 10, "pack_size": 1,
             "pack_removal_cap": 0.08, "retention_seed": 20251009},
  "train_totals": {"partition_proteins": 0, "proteins_with_positives": 0,
                   "pool_only_proteins": 0, "undirected_positive_pairs": 0},
  "strata_cuts": {"low_max_S_sib": 0, "high_min_S_sib": 0},
  "n_candidates": 0, "n_non_nested": 0, "shortfall": false,
  "n_targets": 24, "n_packs": 24, "n_conditions": 121, "n_runs": 2420,
  "replacement_queue": ["m13"],
  "targets": [{
    "go_id": "GO:…", "name": "…", "depth": 0, "stratum": "low|high", "pair_id": "m01", "pack_id": "p01",
    "n_train_proteins": 0, "train_proteins": ["…"],
    "train_annotated_proteins": ["…"], "n_train_annotated": 0,
    "retention_eligible": ["…"], "retention_order": ["…"],
    "test_anchors": ["…"], "test_support": 0, "test_pos_records": 0, "test_weak_records": 0,
    "validation_proteins": ["…"],
    "removal_frac": 0.0,
    "S_sib": 0, "S_sib_proteins": 0, "S_2": 0, "S_2_proteins": 0,
    "N_sib": ["GO:…"], "N_2_size": 0,
    "nbr_train_proteins_sib": ["…"], "nbr_test_proteins_sib": ["…"],
    "nbr_train_proteins_2": ["…"], "nbr_test_proteins_2": ["…"]
  }],
  "pairs": [{"pair_id": "m01", "low": "GO:…", "high": "GO:…", "cost": 0.0}],
  "packs": [{"pack_id": "p01", "targets": ["GO:…"], "removal_frac_k0": 0.0,
             "withheld_k0": 0, "min_retention_eligible": 0}],
  "balance": {"smd_log_n": 0.0, "smd_log_support": 0.0}
}
```

## 3. Withholding generator (gap 4)

### 3.1 Module `$R/go_ppi_domain/withholding.py`

```python
def withheld_set(table, pack_id, k: int | Literal["all"]) -> dict
    # k == "all": {"W": set(), "R": {}}   (only for cond "all")
    # else: {"W": ∪_{t in pack} T_train(t) \ R_k(t), "R": {t: set(retention_order[:k])}}
def filter_train_records(records, W) -> (kept, dropped)       # drop if anchor ∈ W OR partner ∈ W
def filter_pools(pool_rows, W) -> (rows, stats)              # rows with anchor ∈ W removed; W removed from
                                                             # weak_evidence and no_interaction lists
def drop_empty_pool_anchors(records, pools, views) -> (records, dropped_anchors)
def build_targetfree_validation(table, base_splits, out_dir) -> manifest
def build_condition(table, cond_id, base_splits, targetfree_val, out_dir) -> manifest
def assert_condition(trainer, config, manifest_path, train_file, cell, frozen) -> dict   # §4.3
```

### 3.2 Target-free validation (shared, built once at G2)

`make_conditions.py --mode validation` writes `$R/preprocess/go_withholding/validation_targetfree/`, with V = ∪_t P_val(t) over all selected targets:

- `validation.jsonl`: graded records with anchor ∉ V and partner ∉ V;
- `validation_positive.jsonl`: the grade-2 subset, same order;
- `negative_pools_validation.jsonl`: `split=="validation"` rows, anchors in V dropped, V removed from member lists;
- `manifest.json`: `{V_size, records_kept, records_dropped, positive_records, pool_rows, sha256s}`.

Used only by the λ pilot and the optional fusion weight. Pinned (`G2-validation`).

### 3.3 Conditions (G3)

`make_conditions.py --mode conditions` writes `$R/preprocess/go_withholding/conditions/<cond_id>/` for `cond_id ∈ {all} ∪ {<pack>_k<k> : pack ∈ table packs, k ∈ KS}`, i.e. `target_table["n_conditions"]` dirs (121 at 24 targets, 101 under the shortfall; AM1). The script asserts the written count equals `n_conditions`. Each holds `train.jsonl`, `negative_pools.jsonl`, `withholding_manifest.json`.

- `negative_pools.jsonl` = the condition's filtered `split=="train"` rows, then the shared target-free validation rows (exact bytes of `negative_pools_validation.jsonl`). No test rows.
- Train records keep base order (minus dropped lines).
- `all` = full post-fix train, W = ∅.
- **Empty-pool rule:** after filtering, any train anchor with no pool row or with both lists empty (after intersecting with views) has its train records dropped and is recorded in `anchors_dropped_empty_pool`. Applied to `all` too. Without this the selector returns `None` and the batch processor trains on its dummy negative.
- Disk: ~36 MB × 121 ≈ 4.4 GB under AM1 (pre-AM1 41 ≈ 1.5 GB); the script requires ≥ 5 GB free (9.6 GiB free on 2025-10-09). Post-AM1 implementation note (review R3, 2026-10-10): the guard was 3 GB, sized for the pre-AM1 1.5 GB; it was raised to 5 GB because 4.4 GB > 3 GB. Operational only: no condition, count or gate changes.

Schema `withholding_manifest.json` (schema_version 2):

```json
{"schema_version": 2, "cond_id": "p01_k5", "pack_id": "p01", "k": 5,
 "_k_type": "int | \"all\"",
 "target_table_sha256": "…", "base_train_sha256": "…", "base_pools_sha256": "…",
 "targetfree_validation_pools_sha256": "…", "retention_seed": 20251009,
 "targets": [{"go_id": "GO:…", "retained": ["…"], "retained_count": 5,
              "withheld_positive": 0, "withheld_pool_only": 0,
              "retained_positive_pairs_undirected": 0}],
 "withheld_proteins": ["…"], "withheld_count": 0,
 "removal": {"train_records_dropped": 0, "train_records_kept": 0,
             "undirected_pairs_removed": 0, "removal_frac": 0.0,
             "pool_anchor_rows_dropped": 0, "pool_member_entries_removed": 0,
             "pool_member_share_removed": 0.0, "anchors_dropped_empty_pool": ["…"]},
 "unannotated_anchors": {"pool_anchors": 0, "record_anchors": 0, "records": 0, "ids": ["…"]},
 "files": {"train.jsonl": {"sha256": "…", "lines": 0},
           "negative_pools.jsonl": {"sha256": "…", "lines": 0}}}
```

For `all`: `"pack_id": "all", "k": "all", "targets": [], "withheld_proteins": []`.

`pool_member_share_removed` at k=0 is reported per pack (findings E: 0.6–2.3% per target), not gated.

`unannotated_anchors` (review v3 finding 1) counts, in the **written** condition files, the train pool anchors and the distinct train-record anchors whose pruned `go_uris` in `$R/preprocess/go_withholding/converted/proteins.jsonl` is empty, plus the number of train records with such an anchor. `ids` is the sorted list of those anchors. These are the anchors routed through the random window in neg_only/both by §5.6. The counts are descriptive (reported per condition by the analysis), not gated. The same list is passed to the `SelectionRecorder` (§4.6).

### 3.4 Generator validation rules (fatal, exit 2)

- `k ∈ {0, 1, 5, 10, 25}` or the literal `"all"` (only with `cond_id == "all"`); pack id exists.
- `len(retention_order) ≥ k` for every target (guaranteed by §2.6; asserted).
- k=0 `removal_frac ≤ 0.08`.
- W ∩ ∪R = ∅.
- **Exactly k:** for every target and numeric k, `|proteins(written train records) ∩ P_train(t)| == k`, and that set equals `R_k(t)`.
- **Residual scan after writing:** no W protein in `train.jsonl` (either slot), in any train pool row (anchor or member); no V protein in any validation pool row.
- Idempotent regeneration: if the output dir has files, regenerate in memory; identical bytes → left alone; any difference → fatal.
- **Retained proteins stay in the pools.** At numeric k ≥ 1, R_k(t) proteins are not in W, so they keep their pool-anchor rows and stay as pool members (they can be sampled as negatives). Only W is removed. Asserted after writing: for every target, R_k(t) ∩ (pool anchors ∪ pool members of the original train pools) ⊆ (pool anchors ∪ pool members of the written pools).
- `append_pin("G3", {"n_conditions": n, "manifests": {cond_id: manifest sha256}})` once all `n_conditions` succeed.

## 4. Runner (gap 5)

Additive edits to `$R/scripts/run_learning_curve_point.py`. With `--withholding-manifest` absent, behaviour is exactly as today (all new code is under that branch, except the `EXPECTED_FACTOR_FIELDS` addition, which only affects the audit printout).

### 4.1 CLI and fields

New args:

- `--go-ppi-split-dir PATH`: overrides `config.go_ppi_split_dir` before trainer construction.
- `--withholding-manifest PATH`: enables the study path (requires `--domain go_ppi`, `--study-cell`, `--go-diagnostics`; forbids `--validation-file`).
- `--study-cell {baseline,loss_only,neg_only,both}`.
- `--go-diagnostics PATH`: diagnostics JSONL output (§6).
- `--pilot` (`store_true`): marks a λ-pilot run. Requires `--withholding-manifest`; otherwise `argparse` error (exit 2). Skips only §4.2 step 4 (λ equality against `lambda_choice.json`, which does not exist yet). Writes `"pilot": true` into `lc_manifest.json` (`false` otherwise). `sweep.py` refuses, as a fatal preflight error, any run dir whose `lc_manifest.json` has `"pilot": true`, and `check-group` refuses them too.

`EXPECTED_FACTOR_FIELDS["go_ppi"]` gains `"go_ontology_loss_weight"`.

### 4.2 Preflight (before the trainer is constructed)

Executed inside one `try` whose `except BaseException` writes `<output-dir>/run_error.json` (`{"stage", "error", "traceback", "cell", "cond_id", "seed"}`), logs at ERROR, and returns exit code 2.

1. Absolute paths. Scope: the keys present in the **JSON config file** (`json.load(args.config)`, `_`-prefixed keys skipped) plus the CLI path overrides (`--go-ppi-split-dir`, `--train-file`, `--output-dir`, `--withholding-manifest`, `--go-diagnostics`). Every non-null string value in that scope whose key ends in `_path`, `_dir` or `_cache` must satisfy `Path(v).is_absolute()`. `TrainingConfig` defaults that the JSON file does not set (e.g. `skill_distance_cache_path`, `mesh_descriptor_path`, `mesh_index_cache`, `trials_split_dir`, `trials_converted_dir`) are out of scope; they are unused on the go_ppi path. This is the same scope `materialize_configs.py` asserts (§12.1).
2. `sha256(config.go_index_cache) == frozen_hashes["go_index_P_sha256"]`; `GoIndex.load(config.go_index_cache).meta["leak_pruning"]["with_from_human_protein"] is True` and `meta["aspect"] == "P"`. This runs before `attach_go_ppi`, so `build_or_load` inside `build_go_matcher` takes the load branch.
3. `config.validation_path` is None and `--validation-file` was not given.
4. `config.go_ontology_loss_weight == lambda_choice["lambda"]` for loss_only/both and `== 0.0` otherwise (the pin file `$R/results/go_withholding/lambda_pilot/lambda_choice.json`; skipped only under `--pilot`, §4.1, which the pilot passes with `--study-cell loss_only` and a template config, §12.2).
5. `code_sha256()` is computed and stored for the manifest (equality with the pin is enforced by the sweep, §12.3).

### 4.3 Wiring and condition assertions

Order in `main` on the study path:

```python
manifest = json.loads(Path(args.withholding_manifest).read_text())
trainer = ContrastiveLearningTrainer(config=config, output_dir=str(args.output_dir))
attach = attach_go_ppi(trainer, config)                       # unchanged function
attach["go_relation_loss"] = attach_go_relation_loss(trainer, config, args.train_file)
recorder = SelectionRecorder(trainer.batch_processor.domain_negative_selector,
                             unannotated_anchors=manifest["unannotated_anchors"]["ids"])   # §4.6
attach["preencoded"] = preencode_pool(...); attach["cache_check"] = assert_cache_valid(trainer)
diag = EpochDiagnostics(args.go_diagnostics, args.train_file, recorder=recorder, ...)   # §6
guard = StrictEpochGuard(cell=args.study_cell)                # §4.5
trainer.epoch_end_callbacks = [recorder.on_epoch_end, diag, guard]
checks = assert_condition(trainer, config, args.withholding_manifest, args.train_file,
                          args.study_cell, frozen)
result = trainer.train(str(args.train_file))
post = assert_post_training(trainer, config, manifest, recorder, guard)
```

`attach_go_relation_loss(trainer, config, train_file) -> dict` (new, in the runner):

- weight == 0: returns `{"weight": 0.0, "table": None}` and attaches nothing.
- weight > 0: `domain != "go_ppi"` → `ValueError`; `table = GoRelationTable.from_train_file(Path(config.go_index_cache), Path(train_file))`; `trainer.loss_engine.set_go_relation_table(table)`. It **never** calls `batch_processor.set_ontology_matcher` and never touches the selector. Returns `{"weight", "table_proteins", "annotated_proteins", "table_sha256", "index_leak_pruning"}`.

`go_ppi_domain.withholding.assert_condition(...)` — every failure raises `RuntimeError` with the first 5 offending ids (fatal):

1. `sha256(train_file)` and `sha256(<split_dir>/negative_pools.jsonl)` equal the manifest; `config.go_ppi_split_dir` resolves to the manifest's directory; `config.go_ppi_converted_dir` resolves to `$R/preprocess/go_withholding/converted`.
2. No W protein in `resume.protein_id` / `job.partner_id` of the train file.
3. Selector: `set(selector.pools) ∩ W = ∅`; union of pool members ∩ W = ∅; `set(selector.views) ∩ W = ∅`.
4. Selector validation pools equal (per anchor, as sets) the target-free validation pool file; anchors and members ∩ V = ∅.
5. Numeric k: per target, `|proteins(train records) ∩ P_train(t)| == k` and equals `R_k(t)`. k == "all": skipped.
6. `sha256($R/preprocess/go_withholding/splits/test.jsonl) == frozen_hashes["test_sha256"]`.
7. Cell wiring:

   | cell | `go_ppi_go_tiered_negatives` | `go_ppi_tier_sampling` | `go_ontology_loss_weight` | `selector.unannotated_anchor_window` | `selector.matcher` | `batch_processor.ontology_matcher_override` | `loss_engine.go_relation_table` |
   |---|---|---|---|---|---|---|---|
   | baseline | False | random_window | `== 0.0` | True | None | None | None |
   | loss_only | False | random_window | `> 0` | True | None | None | not None |
   | neg_only | True | stochastic | `== 0.0` | True | not None | not None | None |
   | both | True | stochastic | `> 0` | True | not None | not None | not None |

   Check 7 tests only the sign of the weight (`> 0` for loss-on cells, `== 0.0` for loss-off cells), so it is well defined under `--pilot` as well. Equality with `lambda_choice.json` is enforced separately by §4.2 step 4, which is the only step `--pilot` skips (review v3 finding 7). `config.go_ppi_unannotated_anchor_window is True` is also asserted in every cell (§5.6).

   `loss_engine.go_relation_table` must not expose `ontology_set_similarity`, `terms_for_gene` or `branch_distance` (`hasattr` checks; this, not `attach_go_ppi`, is the real guard against misuse as a matcher).
8. Every GoIndex in use (`selector.matcher.index` when present, and the table's source index) has `meta["leak_pruning"]["with_from_human_protein"] is True`.
9. Relation table protein set ⊆ train-record proteins and ∩ W = ∅.
10. `trainer.epoch_end_callbacks` contains the recorder, diagnostics and guard, in that order.

### 4.4 After training (`assert_post_training`, fatal → `run_error.json`, exit 2)

- `loss_engine.go_relation_seen_ids ∩ W = ∅`; `go_relation_missing_ids == 0` (no validation runs, so every training id must be in the table).
- `recorder.none_returns == 0` and `recorder.exceptions == 0`.
- Import coverage: `imported = withholding_constants.imported_repo_modules()` (§13) must be a subset of the code pin's file list (`code_sha256()[1]`); otherwise fatal with the uncovered files listed. Stored in `lc_manifest["code"]["imported"]`.
- `args.output_dir / f"checkpoint_epoch_{config.num_epochs-1}.pt"` exists, `torch.load(..., map_location="cpu", weights_only=False)["epoch"] == config.num_epochs-1`; no `*_emergency.pt` exists.

### 4.5 `StrictEpochGuard` (in `$R/go_ppi_domain/withholding_diagnostics.py`)

```python
class StrictEpochGuard:
    def __init__(self, cell: str, min_applied_share: float = 0.9): self.epochs = []
    def __call__(self, trainer, epoch: int, epoch_stats: dict) -> None:
        # epoch_stats = {"epoch", "failed_batches", "ok_batches", "avg_loss"} (trainer §5.5)
        # Raise RuntimeError (NOT caught here) if failed_batches > 0, ok_batches == 0,
        # or avg_loss is not finite.
        # Loss-on cells: applied = Δ loss_engine.go_relation_applied_batches over the epoch;
        # raise if applied < min_applied_share * ok_batches (the term silently not applying).
        # Off cells: raise if loss_engine.go_relation_applied_batches != 0.
        # Appends {"epoch", "failed_batches", "ok_batches", "avg_loss", "go_relation_applied"} to self.epochs.
```

Because the callback loop runs after `_train_epoch`'s batch loop and outside its per-batch `try`, the exception propagates to `train()`'s handler (emergency checkpoint, re-raise) and then to the runner's `run_error.json` / exit 2.

### 4.6 `SelectionRecorder` (in `$R/go_ppi_domain/withholding_diagnostics.py`)

Wraps the real selector the trainer consumes:

```python
class SelectionRecorder:
    def __init__(self, selector, unannotated_anchors: Iterable[str] = ()):
        self._inner = selector.select_batch_negatives
        selector.select_batch_negatives = self._wrapped      # instance attribute; BatchProcessor
        self.paused = False                                  # resolves it via getattr at call time
        self.calls = self.none_returns = self.exceptions = 0
        self._unannotated = frozenset(unannotated_anchors)
        self._lines: list[str] = []; self.by_epoch: dict[str, str] = {}; self.calls_by_epoch = {}
        self._n_unannotated = 0; self.unannotated_anchor_calls_by_epoch: dict[str, int] = {}
    def _wrapped(self, anchor_sample, candidate_negatives, max_negatives, epoch=None):
        try: out = self._inner(anchor_sample, candidate_negatives, max_negatives, epoch)
        except Exception: self.exceptions += 1; raise    # counted even while paused: intentional
        if self.paused: return out
        self.calls += 1
        if out is None: self.none_returns += 1; return out
        anchor = str(anchor_sample.metadata.get("resume_id") or anchor_sample.resume.get("protein_id"))
        pos = str(anchor_sample.job.get("partner_id"))
        negs = ",".join(f"{v['partner_id']}:{v['grade']}" for v in out[0])
        self._lines.append(f"{epoch}\t{anchor}\t{pos}\t{negs}")
        if anchor in self._unannotated: self._n_unannotated += 1
        return out
    @contextmanager
    def pause(self): ...                                     # used by diagnostics and the pilot
    def on_epoch_end(self, trainer, epoch, epoch_stats):
        self.by_epoch[str(epoch)] = sha256("\n".join(sorted(self._lines)))
        self.calls_by_epoch[str(epoch)] = len(self._lines); self._lines = []
        self.unannotated_anchor_calls_by_epoch[str(epoch)] = self._n_unannotated; self._n_unannotated = 0
```

Attribute names match the `lc_manifest.json` keys of §4.7 one to one (review v4 finding 4): `by_epoch` → `selection_sha256_by_epoch`, `calls_by_epoch` → `calls_by_epoch`, `unannotated_anchor_calls_by_epoch` → `unannotated_anchor_calls_by_epoch`, `none_returns` → `selector_none_returns`, `exceptions` → `selector_exceptions`.

**Selector exceptions while paused are intentionally fatal** (review v3 finding 8). The diagnostics pass (§6) draws the same anchors at the same epoch as training. A selector exception there means the same selector fails on a training anchor. The batch processor would swallow that failure and fall back to dummy negatives. So the exception is counted in `exceptions` whether or not the recorder is paused, and `assert_post_training` (§4.4) fails the run. The batch processor still swallows the exception inside `process_batch`, so the diagnostics callback itself continues: the epoch's row is written and training goes on until the post-training check. This is the one diagnostics-pass event that is fatal. Every other diagnostics failure stays non-fatal (§6, §15).

Lines are sorted before hashing, so the hash is order-independent; the selector's draws depend only on (pools, views, matcher, seed, epoch, anchor), so baseline and loss_only must agree, and neg_only and both must agree, epoch by epoch.

### 4.7 `lc_manifest.json` additions

```json
"withholding": {"cond_id": "…", "pack_id": "…", "k": 0, "cell": "…",
                "manifest_sha256": "…", "frozen_hashes_sha256": "…",
                "assertions": {"<check>": "ok", "counts": {}},
                "selected_checkpoint": "checkpoint_epoch_14.pt",
                "selected_checkpoint_sha256": "…"},
"epochs": 15,
"epoch_stats": [{"epoch": 0, "failed_batches": 0, "ok_batches": 246, "avg_loss": 0.0, "go_relation_applied": 0}],
"negative_selection": {"selection_sha256_by_epoch": {"0": "…"}, "calls_by_epoch": {"0": 0},
                       "unannotated_anchor_calls_by_epoch": {"0": 0},
                       "unannotated_anchor_window": true,
                       "selector_none_returns": 0, "selector_exceptions": 0},
"go_relation_loss": {"weight": 0.0, "table_proteins": 0, "table_sha256": null, "seen_ids": 0,
                     "missing_ids": 0, "skipped_batches": 0, "applied_batches": 0},
"projection_params": {"names": ["…"], "numel": 0},
"code": {"git_head": "…", "code_sha256": "…", "files": 0, "imported": ["contrastive_learning/trainer.py", "…"]},
"pilot": false
```

Source of each `negative_selection` value, written by the runner after `trainer.train` returns: `selection_sha256_by_epoch` = `recorder.by_epoch`; `calls_by_epoch` = `recorder.calls_by_epoch`; `unannotated_anchor_calls_by_epoch` = `recorder.unannotated_anchor_calls_by_epoch`; `unannotated_anchor_window` = `trainer.batch_processor.domain_negative_selector.unannotated_anchor_window` (the selector attribute of §5.6, not the config field, so the value the trainer actually used is recorded); `selector_none_returns` = `recorder.none_returns`; `selector_exceptions` = `recorder.exceptions`.

## 5. GO loss term and loss-only relation table (gap 6)

### 5.1 `TrainingConfig` (`$R/contrastive_learning/data_structures.py`)

Add next to the `go_ordinal_*` fields:

```python
# GO relation-preservation term (withholding study). 0.0 = off, byte-identical to every
# existing config. When > 0 requires domain_adapter="go_ppi" and loss_type in
# {"go_ordinal","infonce"}. The GoRelationTable is attached to the LOSS ENGINE only, never
# to the batch processor, so it cannot open GO-tiered negative selection.
go_ontology_loss_weight: float = 0.0
```

Add next to `go_ppi_tier_window_frac` (the second and last `TrainingConfig` field this design adds):

```python
# Withholding study (review v3 finding 1). True: anchors with no annotation in a
# GO-tiered cell use the random_window path (same draws as the control) instead of
# the uniform full-pool fallback. False (default) = today's behaviour, byte-identical
# for every existing config. Set True in all four study configs (a shared field).
go_ppi_unannotated_anchor_window: bool = False
```

### 5.2 New `$R/go_ppi_domain/go_relation.py`

```python
class GoRelationTable:
    """Dense simGIC_BP over a fixed protein set; loss-only, not a matcher."""
    ids: list[str]; row: dict[str, int]; annotated: np.ndarray  # bool
    sim: np.ndarray                                            # float32 [n, n]
    source_meta: dict                                          # index.meta of the source GoIndex
    @classmethod
    def from_train_file(cls, index_path: Path, train_file: Path) -> "GoRelationTable"
        # ids = sorted proteins in train records (either slot); index via GoIndex.load ONLY
    @classmethod
    def from_index(cls, index: GoIndex, proteins: Iterable[str]) -> "GoRelationTable"
    def submatrix(self, ids: list[str]) -> np.ndarray
    def sha256(self) -> str                                    # over "\n".join(ids) + sim.tobytes()

def go_relation_loss(table: GoRelationTable, triplets, embeddings: Dict[str, torch.Tensor],
                     key_fn: Callable[[dict], str], device=None
                     ) -> Tuple[Optional[torch.Tensor], List[str], int]:
    """Unweighted L_GO, or None if < 2 annotated table ids; ids used; n_missing. Pure, no state.
      1. For each triplet take (triplet.anchor['protein_id'], key_fn(triplet.anchor)) and
         (triplet.positive['partner_id'], key_fn(triplet.positive)). Negatives are excluded.
      2. Deduplicate by protein id keeping the first key; ids not in table -> n_missing += 1, skipped.
      3. Keep ids with annotated[row]; if < 2 remain return (None, ids, n_missing).
      4. E = F.normalize(stack(emb[key]), dim=-1); C = E @ E.T;
         S = torch.from_numpy(table.submatrix(ids)).to(C); mask = triu(ones, diagonal=1).bool()
         return (((C - S) ** 2)[mask].mean(), ids, n_missing)."""
```

Construction is vectorised: sparse protein × term binary matrix X over closures, IC vector w; `inter = X·diag(w)·Xᵀ`, `s_i = Σ_{t∈closure(i)} w_t`, `union = s_i + s_j − inter`, `sim = clip(inter/union, 0, 1)`, `sim = 0` where `union = 0`, `annotated[i] = closure non-empty`. ≈4.7k train proteins → ≈89 MB float32, seconds. It deliberately has no `ontology_set_similarity`, `terms_for_gene` or `branch_distance`.

### 5.3 Additive changes to `$R/contrastive_learning/loss_engine.py`

The uncommitted edits are untouched; `_compute_go_ordinal_loss` is **not edited**. New code goes after the `go_ordinal_*` init block and in `compute_loss`:

```python
self.go_ontology_loss_weight = float(getattr(config, 'go_ontology_loss_weight', 0.0))
if not (math.isfinite(self.go_ontology_loss_weight) and self.go_ontology_loss_weight >= 0.0):
    raise ValueError('go_ontology_loss_weight must be finite and nonnegative')
if self.go_ontology_loss_weight > 0:
    if getattr(config, 'domain_adapter', None) != 'go_ppi':
        raise ValueError('go_ontology_loss_weight requires domain_adapter="go_ppi"')
    if self.loss_type not in ('go_ordinal', 'infonce'):
        raise ValueError('go_ontology_loss_weight supports loss_type go_ordinal or infonce')
self.go_relation_table = None
self.go_relation_seen_ids: set = set()
self.go_relation_missing_ids = 0
self.go_relation_skipped_batches = 0
self.go_relation_applied_batches = 0
self._diagnostic_mode = False

def set_go_relation_table(self, table) -> None
def _go_relation_loss(self, triplets, embeddings) -> Optional[torch.Tensor]:
    # calls go_relation.go_relation_loss(self.go_relation_table, triplets, embeddings,
    #                                    self._get_content_key, self.device)
    # counters updated ONLY when torch.is_grad_enabled() and not self._diagnostic_mode:
    #   seen_ids.update(ids found), missing_ids += n_missing, skipped (None) / applied (+1)
def _with_go_relation(self, total, triplets, embeddings) -> torch.Tensor:
    # weight == 0 -> returns `total` (the same object). weight > 0 and table None ->
    # RuntimeError("go_ontology_loss_weight>0 but no GoRelationTable attached").
    # term None -> total; else total + weight * term.
def go_ordinal_components(self, triplets, embeddings, relation_table=None) -> Dict[str, Optional[torch.Tensor]]:
    # Re-implements the _compute_go_ordinal_loss loop, returning
    # {"infonce": mean base, "ranking": mean ranking, "base_total": mean(base+ranking),
    #  "go_relation": unweighted L_GO on `relation_table` (or None)}.
    # Never touches counters. A unit test and a per-batch runtime check (§6) assert
    # |base_total - _compute_go_ordinal_loss(...)| <= 1e-6.
```

Integration in `compute_loss` only:

- `if self.loss_type == 'go_ordinal': return self._with_go_relation(self._compute_go_ordinal_loss(triplets, embeddings), triplets, embeddings)`.
- InfoNCE tail: `return self._with_go_relation(losses[0], ...)` and `return self._with_go_relation(torch.stack(losses).mean(), ...)`. The zero-tensor early return is left unwrapped (no valid triplets; the guard's applied-share check catches systematic occurrence).
- Other loss types untouched. With weight 0, `_with_go_relation` returns its argument; a test asserts `torch.equal` and identical gradients.

Errors raised here inside training are caught per batch by `_train_batch`; they become fatal through `StrictEpochGuard` (§4.5) — the loss engine itself does not need to be fatal-safe.

Clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)` acts on the total gradient. How often it binds is reported (§6) as an eval-mode diagnostic.

### 5.4 Loss-only isolation proof

- Unit `test_loss_only_selects_identical_negatives`: synthetic split dir (60 anchors, mixed-grade pools), configs differing only in `go_ontology_loss_weight` (0 vs 0.3), stub trainer with a real `ContrastiveLossEngine` and a recording batch-processor stub. Run `attach_go_ppi` + `attach_go_relation_loss` for both. Assert `set_ontology_matcher` never called, `selector.matcher is None`, and for seed 13 and epochs 0..14 `select_batch_negatives` returns identical partner ids and grades for every anchor. Same identity for neg_only vs both.
- Slow integration test (real `ContrastiveLearningTrainer`, 1 epoch, 256-record slice of a synthetic-but-real-format condition): baseline vs loss_only `selection_sha256_by_epoch` equal, neg_only vs both equal, baseline vs neg_only differ, all `none_returns == exceptions == 0`.
- Real grid: the sweep asserts the same per (cond, seed) from `lc_manifest.json` (§12.3).

### 5.5 Trainer hook (`$R/contrastive_learning/trainer.py`, additive)

Immediately before `return avg_epoch_loss` in `_train_epoch`:

```python
epoch_stats = {"epoch": self.current_epoch, "failed_batches": failed_batches,
               "ok_batches": len(epoch_losses), "avg_loss": avg_epoch_loss}
for callback in getattr(self, 'epoch_end_callbacks', ()):
    callback(self, self.current_epoch, epoch_stats)
```

No `__init__` change; the attribute is absent unless the runner sets it, so every existing run is unaffected. This is the only trainer edit.

### 5.6 Unannotated-anchor window (`$R/go_ppi_domain/negative_selector.py`, additive)

This change resolves review v3 finding 1 (§0.1-1). It touches three places, all additive:

- **`__init__`** gains a trailing keyword `unannotated_anchor_window: bool = False`, stored as `self.unannotated_anchor_window = bool(...)`. It is also added to the `logger.info` line.
- **`from_config`** passes `unannotated_anchor_window=bool(getattr(config, "go_ppi_unannotated_anchor_window", False))`.
- **`select_batch_negatives`** computes one flag after `anchor_uris` is resolved (and after the `if self.matcher:` facet block, which is unchanged):

  ```python
  use_random_window = (self.tier_sampling == "random_window"
                       or (self.unannotated_anchor_window and self.go_tiered
                           and not anchor_uris))
  ```

  The two existing `if self.tier_sampling == "random_window":` tests are replaced by `if use_random_window:`. Their bodies are unchanged:
  - the fixed window: `wrng = _seeded_rng(self.training_seed, -1, anchor_id)`, then `_fixed_random_window` on the hard pool, then on the easy pool;
  - the sampling: `rng.sample` on the hard pool, then on the easy pool.

  The `elif self.go_tiered and … and anchor_uris` branches and the final `else` are untouched.

Why this gives identical draws. An unannotated anchor in a GO-tiered cell now consumes the same RNG sequence as in the control cells:

- `wrng`: hard window, then easy window;
- `rng`: hard sample, easy sample, `shuffle`.

The pools are the same, because `hard_pool`/`easy_pool` are filtered by `self.views`, whose key set is the same in every cell, and `tier_window_frac` is 0.34 everywhere. So the partner ids and grades are identical to baseline and loss_only. The `distances` are `NEUTRAL_DISTANCE` in every cell, because `_ontology_distance` returns neutral for empty `anchor_uris`.

Default `False` reproduces today's branch structure exactly: `use_random_window` reduces to the old test.

How "unannotated" is decided:

- In neg_only/both, `anchor_uris` comes from `matcher.terms_for_gene(anchor_id)` on the pruned index.
- `rewrite_proteins` sets `go_uris = sorted(index.terms_for(g))` from the same index.
- So "unannotated" here is the same set as `withholding_manifest["unannotated_anchors"]["ids"]` (§3.3).

`make_conditions.py` asserts the agreement once per condition by loading the pruned index: `{a : not index.terms_for(a)} == set(ids)` over the train pool anchors (fatal, exit 2).

Career safety:

- The selector is GO/PPI-only, and the flag defaults to `False`, so no existing config changes.
- The existing 20 `go_ppi_domain` tests must still pass (G0).

Tests are listed in §14 (`test_go_withholding_negatives.py`).

## 6. Diagnostics: loss and gradient scale (gap 7a)

`$R/go_ppi_domain/withholding_diagnostics.py`:

```python
@contextmanager
def preserved_rng(): ...   # saves/restores random, numpy and torch CPU RNG states
def projection_params(trainer) -> list[torch.nn.Parameter]
    # [p for p in trainer.model.parameters() if p.requires_grad]; assert none is a
    # trainer.text_encoder parameter; fatal (RuntimeError) if empty
class EpochDiagnostics:
    def __init__(self, out_path: Path, train_file: Path, recorder: SelectionRecorder,
                 n_batches: int = 8, batch_size: int = 64,
                 relation_table: Optional[GoRelationTable] = None, weight: float = 0.0)
    def __call__(self, trainer, epoch: int, epoch_stats: dict) -> None
```

- Diagnostic sample: the first `n_batches × batch_size` lines of the condition's train file, built into `TrainingSample`s via `GoPpiRecordAdapter`, fixed order.
- A relation table over the condition's train proteins is built for the callback in **every** cell (λ = 0 too), so relation fit is logged everywhere; it is held by the callback only, never by the loss engine.
- Per call, inside `preserved_rng()`, `recorder.pause()`, `torch.enable_grad()`, with `trainer.loss_engine._diagnostic_mode = True` and `trainer.model.eval()`, all restored in `finally`, for each batch:
  1. `triplets = trainer.batch_processor.process_batch(batch, None, None)` (the selector's epoch is the current epoch, so draws equal training's draws for these anchors);
  2. `emb = trainer._generate_embeddings(triplets)`;
  3. `comps = trainer.loss_engine.go_ordinal_components(triplets, emb, relation_table=self.table)`; `drift = |comps["base_total"] − _compute_go_ordinal_loss(triplets, emb)|`;
  4. `torch.autograd.grad(x, params, retain_graph=True, allow_unused=True)` for x ∈ {infonce, ranking, go_relation}, `params = projection_params(trainer)`; L2 norm over the concatenation; plus the norm of the gradient of `base_total + weight·go_relation` (pre-clip).
- No `.grad` is accumulated; the optimizer is untouched.

JSONL row per epoch (`<run_dir>/go_diagnostics.jsonl`):

```json
{"epoch": 0, "n_batches": 8, "n_triplets": 512, "lambda": 0.0,
 "infonce_mean": 0.0, "ranking_mean": 0.0, "go_relation_mean": 0.0, "max_component_drift": 0.0,
 "grad_norm_infonce": {"median": 0.0, "mean": 0.0},
 "grad_norm_ranking": {"median": 0.0, "mean": 0.0},
 "grad_norm_go_relation_unweighted": {"median": 0.0, "mean": 0.0},
 "ratio_weighted_go_to_infonce": {"median": 0.0, "per_batch": [0.0]},
 "ratio_ranking_to_infonce": {"median": 0.0},
 "grad_norm_total_preclip_eval_diag": {"median": 0.0, "share_above_clip_eval_diag": 0.0},
 "relation_skipped_batches": 0, "epoch_stats": {}, "elapsed_s": 0.0, "error": null}
```

`share_above_clip_eval_diag` is measured in eval mode (dropout off) on the 8 diagnostic batches; it is labelled as such and is not the training-time share.

Failure handling: any exception inside the diagnostics callback is caught, logged at WARNING and written as a row with `"error": "<repr>"`; training continues (secondary outcome). **One deliberate exception:** a selector exception raised during the diagnostics pass is counted by the recorder even though it is paused. That makes the run fail at `assert_post_training` (§4.4, §4.6). The diagnostics draw the same anchors at the same epoch as training, so such an exception exposes a training-time selector failure that the batch processor would otherwise hide. `max_component_drift > 1e-5` is logged at ERROR and flagged in the row (`"drift_flag": true`); the sweep reports flagged runs. RNG, recorder pause, `_diagnostic_mode` and `model.train()` are restored in `finally`.

Non-interference: slow integration test — 1 epoch on a 256-record slice, with and without the diagnostics callback, gives identical epoch loss, identical `state_dict`, and identical selection hash.

## 7. Negative audit (gap 7b)

`$R/scripts/go_withholding/audit_negatives.py --cond <cond_id> --cells baseline loss_only neg_only both --seed 13 --epochs 0 7 14 [--anchors all|N]`.

Offline reconstruction is a descriptive audit; the binding identity check is the run-time recorder (§4.6). Preflight: same index sha/meta check as §4.2 step 2 (exit 2 on failure). For each cell it loads the cell config, sets `cfg.go_ppi_split_dir = <cond dir>` and **`cfg.training_seed = args.seed`** (the base configs carry 42, and `from_config` reads `config.training_seed`), builds the production selector via `GoPpiNegativeSelector.from_config(cfg, "train", matcher=build_go_matcher(cfg) if tiered else None)`, and asserts `selector.training_seed == args.seed` (fatal, exit 2) and calls `select_batch_negatives` on a `SimpleNamespace` sample per anchor (port of `e_negaudit.py`). A separate measurement matcher `GoMatcher(GoIndex.load(pruned), similarity_mode="simgic")` scores simGIC. HuRI and Lit-BM are mapped ENSG → symbol via STRING `Ensembl_gene` aliases.

Output `$R/results/go_withholding/audits/neg__<cond>__<cell>__s<seed>.json`:

```json
{"cond_id": "…", "cell": "…", "seed": 13, "anchors": 0,
 "selection_sha256_offline": "…",
 "by_epoch": {"0": {"n": 0, "grade_mix": {"1": 0.0, "0": 0.0},
                    "simgic_mean": {"1": 0.0, "0": 0.0, "all": 0.0},
                    "indep_fn_rate": {"1": 0.0, "0": 0.0},
                    "share_target_neighbour_negatives": 0.0,
                    "share_target_neighbour_anchors": 0.0,
                    "share_withheld_negatives": 0.0,
                    "share_target_annotated_negatives": 0.0,
                    "share_retained_negatives": 0.0}}}
```

- `share_target_neighbour_negatives` = fraction of sampled negatives in ∪_{t∈pack} NP_train_sib(t); `_anchors` likewise; for `all`, the union over all packs.
- `share_withheld_negatives` = fraction of sampled negatives in W(cond) (from the condition's `withholding_manifest.json`). **Gate:** must be exactly 0 for every pack condition at every k (fatal, exit 2). For `all`, W = ∅ and the value is 0 trivially.
- `share_target_annotated_negatives` = fraction in ∪_{t∈pack} T_train(t). Descriptive, not gated: 0 at k = 0, and at k ≥ 1 it equals `share_retained_negatives`, because retained proteins remain pool members (§3.4) and may be drawn as negatives. For `all`, the union over all packs.
- `share_retained_negatives` = fraction in ∪_{t∈pack} R_k(t). Descriptive.
- Invariant asserted per epoch (fatal): `share_target_annotated_negatives == share_withheld_negatives + share_retained_negatives` within 1e-12 for pack conditions.
- The audit asserts offline identity baseline == loss_only and neg_only == both (exit 2 on mismatch).

## 8. Variant-independent checkpoint selection (gap 8)

The selected checkpoint is the final epoch, `checkpoint_epoch_{epochs-1}.pt` (`checkpoint_epoch_14.pt` in the grid, `checkpoint_epoch_1.pt` in the 2-epoch G6 smoke). Study configs set `checkpoint_frequency: 1000000000`, omit `validation_path`; `_should_save_checkpoint` then saves only at `current_epoch == num_epochs-1`. The runner verifies it (§4.4) and records its name and sha in `lc_manifest.json["withholding"]`. No metric influences which weights are evaluated, in any cell.

## 9. Evaluator (gap 9)

`$R/scripts/go_withholding/evaluate_withholding.py` with subcommands `build-cache` and `score`. Imports: `content_to_text` and `CareerAwareContrastiveModel` from the top-level `$R/run_phase1_embedding_evaluation.py`; `rank_auc`, `z_stats`, `z_apply`, `ranking_metrics` from `$R/scripts/evaluate_publication_fusion.py` (both importable after `sys.path.insert(0, ROOT)` and `sys.path.insert(0, ROOT/"scripts")`, as `evaluate_publication_fusion` itself does).

### 9.1 `build-cache`

```
--test $R/preprocess/go_withholding/splits/test.jsonl
--frozen-hashes $R/preprocess/go_withholding/splits/frozen_hashes.json
--out $R/results/go_withholding/eval_cache/
```

1. Assert the test sha.
2. Unique proteins in record order and their texts via `content_to_text` (strips whitespace; token-identical to the trainer path, checked at G6 by §9.2 `--encoder-crosscheck`). The trainer's `BatchEfficientEncoder._content_to_text` returns `encoder_view` verbatim, so the strings can differ by trailing whitespace: 3,494 of 15,155 reference `encoder_view`s end in one space, and the MPNet `input_ids` are identical in all 3,494 [review v4, ran].
3. Encode with `encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")` and **`encoder.max_seq_length = 512`**, the value `trainer.py:129` sets on `trainer.text_encoder` (the model default is 384). Batch 64, float32.
4. **Token-length assertion:** `max_tok = max(len(encoder.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts)`; require `max_tok < 512`, otherwise fatal (exit 2: "text would be truncated"). `max_tok` is recorded in the meta (≈274 expected [review v3, ran on the reference proteins]). Because no text reaches either cap, the 384-vs-512 setting cannot change an embedding. The setting is pinned anyway so that this holds by construction.
   The v3 cross-check against `$R/embedding_cache/go_withholding_text_embeddings.pt` is removed from `build-cache`: that file does not exist at G2. The trainer-encoder cross-check now runs at G6 (§9.2 `--encoder-crosscheck`).
5. Write `test_<sha16>.npz` (`protein_ids`, `emb[N,768]`, `rec_anchor`, `rec_partner`, `grade`, `query`) and `test_<sha16>.meta.json` (`{test_sha256, n_records, n_proteins, model, max_seq_length: 512, max_tok, created, frozen_global_spearman}`); `append_pin("G2-cache", {...})`.

### 9.2 `score`

```
--run-dir $R/results/go_withholding/runs/<run_id> --cache <npz> --target-table <json>
[--relation] [--fusion-weight-file <json>] [--encoder-crosscheck --config <cell config json>]
```

`--encoder-crosscheck` (used only at G6; review v3 finding 4). It runs before step 1:

1. Build `ContrastiveLearningTrainer(config=TrainingConfig.from_dict(json.load(config)), output_dir=<tmp under run_dir/eval>)` and take `trainer.text_encoder`. Assert `trainer.text_encoder.max_seq_length == 512`.
2. Draw 50 test proteins with `random.Random(0).sample(sorted(cache protein_ids), 50)`. For each, read `encoder_view` from `$R/preprocess/go_withholding/converted/proteins.jsonl`, form `view_slot = {"encoder_view": encoder_view}`, and build the re-encoding text with the **trainer's** text function: `trainer_text = trainer.batch_encoder._content_to_text(view_slot, "resume")` (review v4 finding 3). The cache side stays as built by `build-cache`, i.e. from `content_to_text`.
3. **Text-function token check** over **all** cache proteins (not only the 50): count the proteins where `trainer.text_encoder.tokenizer(trainer_text)["input_ids"] != trainer.text_encoder.tokenizer(content_to_text(view_slot, "resume"))["input_ids"]`. Require `n_token_mismatch == 0`; otherwise fatal (exit 2, `eval/eval_error.json` with the first 5 ids). This detects a divergence between the two text functions that a same-function comparison could not.
4. Re-encode the 50 `trainer_text`s with `trainer.text_encoder.encode(texts, batch_size=50, convert_to_numpy=True)`.
5. Require cos ≥ 0.9999 against the cached `emb` rows for every protein. Otherwise fatal: exit 2 and `eval/eval_error.json`.
6. Write `eval/encoder_crosscheck.json` = `{n: 50, min_cos, max_seq_length: 512, ids_sha256, n_token_checked, n_token_mismatch: 0, n_text_differs}`. `n_text_differs` counts proteins where the two strings differ at all (whitespace only, expected ≈ 3.5k); it is descriptive.

The check always re-encodes through the trainer's encoder and never looks up the trainer's embedding cache. The study test proteins are protein-disjoint from train, so that cache normally holds none of them, and a lookup branch would almost never run. The sweep never passes this flag.

1. Read `lc_manifest.json`: `epochs`, `withholding.selected_checkpoint`, `withholding.selected_checkpoint_sha256`. Load that file; assert `checkpoint["epoch"] == epochs-1` and the sha matches.
2. `CareerAwareContrastiveModel(input_dim=768, projection_dim=128, dropout=0.3, use_structured_features=False)`, `load_state_dict(strict=True)`, eval mode (pattern of `evaluate_publication_fusion.text_scores`).
3. Project the cache (L2-normalised by `forward`); `score = Σ left*right` per record.

Outputs in `<run_dir>/eval/`: `scores.npy` (float32, test record order, 22,341), `metrics.json`, `relation.json` (with `--relation`).

```json
{"run_id": "…", "test_sha256": "…", "checkpoint_sha256": "…", "scores_sha256": "…", "epochs": 15,
 "global": {"hard_auc": 0.0, "easy_auc": 0.0, "pooled_auc": 0.0,
            "macro_query_hard_auc": 0.0, "macro_mrr": 0.0, "n_queries": 0},
 "per_target": {"GO:…": {"queries": 0, "n_pos": 0, "n_weak": 0, "hard_auc": 0.0,
                         "macro_query_hard_auc": 0.0, "macro_mrr": 0.0}},
 "collateral": {"p01": {"queries": 0, "hard_auc": 0.0, "macro_query_hard_auc": 0.0}}}
```

- **Primary per-target `hard_auc`:** tie-correct `rank_auc` of all grade-2 vs all grade-1 records of queries in `P_test(t)`, pooled within the target.
- **Secondary:** macro per-query hard AUC over queries with both grades; macro MRR with grade 2 relevant (`ranking_metrics` restricted to the target's queries).
- `per_target` covers all selected targets in every run; the analysis picks the relevant (target, run) pairs.
- **Collateral:** queries whose anchor ∉ ∪_{t∈pack} (P_test(t) ∪ NP_test_sib(t)); pack runs report their own pack id; `all` runs report every pack id.
- The sweep's grid runs additionally assert `epochs == 15` (§12.3).

### 9.3 `relation.json` (gap 7c)

- **Global:** Spearman(cos, simGIC_BP) over 20,000 test-protein pairs drawn once with `random.Random(0)` from the cache protein list (identical across runs); simGIC from the measurement matcher.
- **Target × neighbour, per target:** pairs (i ∈ P_test(t), j ∈ NP_test_sib(t)), capped at 5,000 (`Random(0)`): Spearman(cos, simGIC), `target_nbr_mean_cos`, and `random_ref_mean_cos` = mean cos from the same i to 5,000 fixed random test proteins outside NP_test_sib(t) ∪ P_test(t). `excess = target_nbr_mean_cos − random_ref_mean_cos` feeds the mechanism rule.

### 9.4 Fusion reference (optional, exploratory)

z(text) + w·z(simGIC), w ∈ {0, 0.1, …, 1.0}, chosen once on target-free validation hard AUC from the `all`-baseline seed-13 run, stored in `$R/results/go_withholding/fusion_weight.json`, applied unchanged to all runs. Reported only.

### 9.5 Error handling

| condition | outcome |
|---|---|
| missing checkpoint, epoch or sha mismatch vs `lc_manifest.json` | fatal for that run; `eval/eval_error.json`; exit 2 |
| a target with no grade-2 or no grade-1 record | impossible on the frozen test (§2.2 rule 2); asserted, fatal |
| NaN/inf scores | fatal, exit 2 |

## 10. Text-similarity covariate (gap 10)

`$R/scripts/go_withholding/text_similarity_covariate.py --target-table … --cache <eval npz> --train-embeddings $R/embedding_cache/go_withholding_text_embeddings.pt --out $R/preprocess/go_withholding/targets/text_covariate.json`.

If the train-embeddings file is absent (the normal case at G2; it is first written in G4) or lacks entries, the missing train texts (via `content_to_text`) are encoded with the same frozen MPNet; `n_encoded` and `n_from_cache` are recorded. Per target:

- **Primary `tsim_excess(t)`:** mean over i ∈ P_test(t) of [mean_j cos(e_i, e_j), j ∈ NP_train_sib(t)] minus the same with j from a fixed reference of 2,000 train proteins outside T_train(t) ∪ NP_train_sib(t) (`Random(0)`).
- **Secondary `tsim_train(t)`:** mean pairwise cos between P_train(t) and NP_train_sib(t).
- N_2 analogues for sensitivity.

Computed once, before training; output sha pinned (`G2-covariate`).

## 11. Statistics (gap 11)

`$R/scripts/go_withholding/analyze_withholding.py --runs $R/results/go_withholding/runs --target-table … --covariate … --out $R/results/go_withholding/analysis/ --seed 20251009 --B 10000`.

1. **Long table** `per_target.csv`, one row per (target, k ∈ {0,1,5,10,25,all}, cell, seed): `hard_auc`, `macro_query_hard_auc`, `macro_mrr`, `collateral_hard_auc`, `stratum`, `pack_id`, `S_sib`, `S_2`, `tsim_excess`. Numeric k rows come from run `<pack(t)>__k<k>__<cell>__s<seed>`; k = all rows from `all__kall__<cell>__s<seed>`.
2. **Paired differences and definedness** (no partial averaging, no imputation):
   - `Δ_c(t,k,s) = AUC_c(t,k,s) − AUC_baseline(t,k,s)`, defined iff the (t,k,s) tuple is present (step 5).
   - `Δ_c,small(t,s) = mean_{k∈{0,1,5}} Δ_c(t,k,s)`, defined **only if all three** k ∈ {0,1,5} tuples are present; otherwise (t,s) is missing for H1 and H2.
   - `D_c(t,s) = Δ_c,small(t,s) − Δ_c(t,all,s)`, defined only if `Δ_c,small(t,s)` and `Δ_c(t,all,s)` are both defined.
   - `Δ^coll_c(p,k,s)` = collateral hard AUC of cell c minus baseline for pack p's run at (k, s) (§9.2 `collateral[p]`), defined iff the pack-run tuple is present.
   - `E_c(t,s) = Δ_c(t,0,s) − Δ^coll_c(pack(t),0,s)` (target-specific excess at k = 0, protocol §8 A1), defined iff both are defined.
3. **Primary: two-way cluster bootstrap** (packs × seeds), B = 10,000, `numpy.random.default_rng(20251009)`. In each replicate:
   1. draw `n_packs` pack indices with replacement (`target_table["n_packs"]`: 24, or 20 under the shortfall; one target per pack under AM1);
   2. draw one multiset of 5 seeds with replacement from {13,21,42,87,123}, shared by every pack in the replicate;
   3. statistic = mean of X(t,s) over the **defined** (t,s) cells among (targets of the drawn packs, with multiplicity) × (drawn seeds, with multiplicity), each cell weighted equally.

   Contrasts: per cell and k (X = Δ_c(t,k,s)); `Δ_c,small`; `D_c`; `E_c` (family `descriptive`); S moderation `M_c` = mean over the defined high cells of Δ_c,small minus mean over the defined low cells, both within the drawn packs × drawn seeds.

   A replicate with zero defined cells for a contrast (for `M_c`: zero defined high cells or zero defined low cells) is discarded **for that contrast only**; `contrasts.json` reports `n_discarded_replicates` per contrast, and the CI and p use the retained replicates. If more than 5% of replicates are discarded for a contrast, it is flagged `"unstable": true` and classified "inconclusive" by rule 5 regardless of its CI. The point estimate uses the same definedness rule on the full data.

   95% percentile CI; two-sided bootstrap p = `min(1, 2·min(mean(T* ≤ 0), mean(T* ≥ 0)))`; Holm within the protocol's families.
4. **Sensitivity analyses** (labelled, never change verdicts):
   - (S1) the v1 hierarchical scheme: resample targets, then seeds within target;
   - (S2) N_2 strata (S_2 re-tertiled over the same selected targets; targets falling in the middle tertile dropped);
   - (S3) the per-query metric (macro per-query hard AUC);
   - (S4) `tsim_excess` adjustment: OLS via `numpy.linalg.lstsq` of per-target mean Δ_c,small on [1, high, tsim_excess], `high` coefficient CI from the same two-way bootstrap;
   - (S5) seed spread: sd across seeds of the per-seed mean Δ.
5. **Group-check preflight (fatal, exit 2; review v3 finding 2).** Run dirs are the direct children of `--runs` whose name is exactly a `run_id` (`<pack>__k<k>__<cell>__s<seed>`); dirs whose name contains `.groupfailed` or `.attempt` (renamed by the failure and clearing rules of §12.3) are ignored entirely (review v4 finding 1). Before anything is computed, every (cond, seed) group is classified from its four run dirs:
   - **checked:** all four dirs carry `DONE.json` and a **valid** `GROUP_OK.json`. Valid (`group_ok_valid(group)`, one function in `withholding_constants.py` shared with the sweep's eligibility rule, §12.3): each `GROUP_OK.json` lists the same four `run_ids` (the group's, in cell order), and its `lc_manifest_sha256` for each dir equals the current `sha256(<dir>/lc_manifest.json)` and that dir's `DONE.json["lc_manifest_sha256"]`.
   - **failed-member:** at least one dir has `FAILED.json`. The group is not checked and the tuple is missing (below). A dir repaired by `sweep.py rescore` carries `FAILED.repaired.json` instead, which is not `FAILED.json`; such a group is classified by the other rules like any other.
   - **unchecked:** all four dirs have `DONE.json` and at least one lacks a valid `GROUP_OK.json`. This is **fatal** (exit 2: "run `sweep.py status` to check groups"). An unchecked group is not counted as missing.
   - **group-failed:** any dir has `GROUP_FAILED.json`. This is **fatal** (exit 2). Protocol §7 says a grid stopped by an identity mismatch is not analysed until an amendment fixes the cause. v3 counted this case as missing, which contradicted the protocol. After the §12.3 clearing step (amendment, rename to `.groupfailed<n>`) the group's dirs are ignored, so it reads as incomplete until it is retrained.
   - **incomplete:** fewer than four `DONE.json` and no `FAILED.json`, i.e. not yet run. The tuple is missing.

   `--allow-incomplete` does **not** relax the unchecked or group-failed cases. Under the v5 sweep rules an unchecked group should not survive a `status` call (§12.3), so this branch is a fail-closed backstop.
   **Completeness rule:** a (t, k, s) tuple is present only if its group is **checked** (k = all uses the `all` run's four cells). The denominator is all target-condition-seed tuples, `n_targets × 6 × 5` (6 = the five numeric k plus `all`; 720 at 24 targets, 600 at 20, unchanged by AM1), read from the table. Missing tuples are listed in `missing.json`, with their reason (`failed_member`, `incomplete`). More than 10% missing → exit 3 "incomplete grid", unless `--allow-incomplete`, which labels all outputs `"exploratory": true`. Missingness feeds step 2's definedness rules; nothing is imputed.
6. **Decisions** (`decisions.json`): the protocol §8 ordered first-match rules evaluated mechanically, per hypothesis and cell, plus the mechanism rule and the raw findings-(iv) flag `ci_excludes_0`.
7. scipy is used for `spearmanr` and `linear_sum_assignment` only; the bootstrap is numpy (`scipy.stats.bootstrap` is not clustered).

Outputs: `per_target.csv`; `contrasts.json` (`{family: {contrast: {estimate, ci_low, ci_high, p_boot, p_holm, n_defined_cells, n_targets, n_packs, n_seeds, n_discarded_replicates, unstable, scheme}}}`; families `H1`, `H2`, `H3`, `descriptive` (per-k Δ, `E_c`, neg_only k=0 Δ, collateral, mechanism (a))); `missing.json`; `sensitivity.json`; `decisions.json`; `diagnostics_summary.json` (gradient ratios, audit and relation-probe means per cell, drift flags).

## 12. Configs, λ pilot, sweep driver

### 12.1 Config materialiser `$R/scripts/go_withholding/materialize_configs.py --lambda <float|none>`

Writes `$R/config/go_withholding/{baseline,loss_only,neg_only,both}.json`. Baseline and loss_only start from `$R/config/go_decomposition_go_ordinal/randwin.json`; neg_only and both from `$R/config/go_decomposition_go_ordinal/go_P_simgic.json` (they differ only in `go_ppi_go_tiered_negatives`, `go_ppi_tier_sampling`, `go_similarity_mode`, `go_aspect_weights` [review, ran]). Overrides applied to **all four** (absolute values):

```
go_ppi_split_dir      = /Users/olukotunjosh/Downloads/CDCL/preprocess/go_withholding/conditions/all
go_ppi_converted_dir  = /Users/olukotunjosh/Downloads/CDCL/preprocess/go_withholding/converted
go_index_cache        = /Users/olukotunjosh/Downloads/CDCL/preprocess/go_withholding/converted/go_index_P.pkl
go_obo_path           = /Users/olukotunjosh/Downloads/CDCL/dataset/go_ppi/bulk/go-basic.obo
go_annotation_path    = /Users/olukotunjosh/Downloads/CDCL/dataset/go_ppi/bulk/goa_human.gaf.gz
embedding_cache_path  = /Users/olukotunjosh/Downloads/CDCL/embedding_cache/go_withholding_text_embeddings.pt
checkpoint_frequency  = 1000000000
go_similarity_mode    = "simgic"
go_aspect_weights     = {"P": 1.0, "F": 1.0, "C": 1.0}
go_ppi_unannotated_anchor_window = true      # shared field, all four cells (§5.6)
_study = "go_withholding_v4", _cell = <cell>
```

- `go_ontology_loss_weight` = 0.0 for baseline and neg_only, λ for loss_only and both. `validation_path` is removed if present.
- Every non-null string value **in the written JSON** whose key ends in `_path`, `_dir` or `_cache` must be absolute (assert; fatal). `TrainingConfig` defaults not written to the JSON are out of scope (same scope as §4.2 step 1).
- `--lambda none` writes only baseline, neg_only and `loss_pilot_template.json` (the loss_only config with `go_ontology_loss_weight` = 0.0). `lambda_pilot.py` copies the template to `<pilot out>/l<λ>/config.json` with the pilot λ set; those copies are never used by the grid.
- **Factor audit with exact diff sets** (review v3 finding 3). For each pair, `rep = report_factor("go_ppi", a, b)`. Both conditions below must hold, otherwise the script refuses to write anything (fatal, exit 2, both sets printed):
  - `rep["unexpected_fields"] == []`;
  - `set(rep["diff"]) == EXPECTED_PAIR_DIFF[(a, b)]`, where:

  ```python
  EXPECTED_PAIR_DIFF = {
      ("baseline", "loss_only"): {"go_ontology_loss_weight"},
      ("neg_only", "both"):      {"go_ontology_loss_weight"},
      ("baseline", "neg_only"):  {"go_ppi_go_tiered_negatives", "go_ppi_tier_sampling"},
      ("loss_only", "both"):     {"go_ppi_go_tiered_negatives", "go_ppi_tier_sampling"},
  }
  ```

  `config_diff` ignores `_`-prefixed keys, so `_cell` never appears. `go_ppi_unannotated_anchor_window`, `go_similarity_mode` and `go_aspect_weights` are equal in all four configs and must not appear either. The audit runs in memory on the four candidate dicts (written to a temp dir under `$R/config/go_withholding/.staging/`, deleted after) before the real files are written. Under `--lambda none` only the pairs that can be formed from the files it writes are audited: baseline↔neg_only, plus baseline↔`loss_pilot_template`, whose expected diff is `∅` because the template has weight 0.0.
- **Pins** (review v3 finding 6). `--lambda none` writes **no** pin and no `configs.sha256.json`. Only `--lambda <float>` writes `configs.sha256.json` = `{"baseline.json": sha, "loss_only.json": sha, "neg_only.json": sha, "both.json": sha}` (exactly these four keys) and calls `append_pin("G4-configs", {"configs_sha256_json": sha256(configs.sha256.json), "files": {...}, "lambda": λ})`. If `--lambda <float>` is re-run (e.g. after an amendment), a second `G4-configs` entry is appended. The sweep preflight uses the **last** `G4-configs` entry and asserts that its `files` keys are exactly the four cell files and that each hash matches the file on disk.

### 12.2 λ pilot `$R/scripts/go_withholding/lambda_pilot.py --out $R/results/go_withholding/lambda_pilot`

1. **Pilot run:** condition `all`, the loss_only template with λ₀ = 0.1 (written to `<out>/l0.1/config.json`), seed 42, 3 epochs, via the runner subprocess (`--withholding-manifest …/conditions/all/withholding_manifest.json --study-cell loss_only --pilot`; `--pilot` skips only §4.2 step 4). Output `<out>/l0.1/`.
2. **Target-free validation measurement** (in-process): build a trainer from the same config with `config.training_seed = 42`, `attach_go_ppi` (the selector already holds the target-free validation pools), load `checkpoint_epoch_2.pt` (strict). Then set the selector epoch to the checkpoint's epoch, so the validation triplets use the same grade-1 ramp position the checkpoint was trained at: `e = epochs - 1` (= 2); `trainer.batch_processor.set_epoch(e)`; `trainer.loss_engine.set_epoch(e)`; assert `trainer.batch_processor.current_epoch == e`. A freshly built trainer would otherwise sit at epoch 0. `selector_epoch = e` is recorded. Records = all grade-2 records of `$R/preprocess/go_withholding/validation_targetfree/validation_positive.jsonl`, in file order, in fixed batches of 32, last partial batch dropped. Require `n_batches ≥ 4`; otherwise exit 3 with "too few target-free validation batches" (stop and ask the user). Relation table = `GoRelationTable.from_index(pruned, validation proteins of those records)`. Per batch: the §6 gradient recipe with `relation_table=` that table (pure function; engine counters untouched). `r₀` = median over batches of `λ₀·‖∇L_GO‖ / ‖∇L_InfoNCE‖` (projection params).
3. **Choice:** `λ* = λ₀ · 0.20 / r₀`, rounded to the nearest value (in log space) of {1, 2, 3, 5} × 10ⁿ.
4. **Confirmation run** at λ* (same settings, `<out>/l<λ*>/`): validation ratio `r*` must be in [0.10, 0.30]. If not, `λ** = λ* · 0.20 / r*` (rounded), confirm once more. If that fails: exit 3, stop and ask the user.
5. **Output** `<out>/lambda_choice.json`: `{lambda, lambda0, r0, r_confirm, ratios_val_per_batch, ratios_train_diag_by_epoch, n_records, n_batches, batch_size: 32, seed: 42, epochs: 3, selector_epoch: 2, pilot_run_dirs, created, code_sha256}`; `append_pin("G4-lambda", …)`. Then `materialize_configs.py --lambda <λ>`.

The pilot trains on `all` (contains target proteins); acceptable because it reads only gradient magnitudes on target-free validation, never test or target metrics.

### 12.3 Sweep driver `$R/scripts/go_withholding/sweep.py`

```
sweep.py pin
sweep.py plan   [--seeds 13 21 42 87 123] [--packs p01 …] [--ks 0 1 5 10 25 all] [--cells …]
sweep.py run    --max-runs N   (required, 1 ≤ N ≤ 100)  [same filters] [--dry-run]
sweep.py status
sweep.py check-group --run-dirs <baseline> <loss_only> <neg_only> <both>   # group checks only; used by G6
sweep.py clear-group --cond <cond_id> --seed <s> --amendment <id>           # amendment-only rename of a GROUP_FAILED group
sweep.py rescore --run-id <run_id>                                           # repair an eval-stage FAILED.json member (no retraining)
```

- **`check-group`:** runs the group checks below on four given run dirs (cell order as listed), without the `epochs == 15` requirement; exit 4 on mismatch; exit 2 if any dir has `"pilot": true`.
- **`pin`:** writes `$R/results/go_withholding/code_pin.json` = `{git_head, code_sha256, files, created}` and `append_pin("G5", …)`. Refuses if `code_pin.json` exists (re-pinning is an amendment: the user deletes it by hand and records why).
- **Enumeration:** seed-major (13, 21, 42, 87, 123), then cond (`all` first, then the table's packs in `pack_id` order, p01…p24 or p01…p20 under AM1, × k ascending), then cell (baseline, loss_only, neg_only, both). `run_id = f"{pack}__k{k}__{cell}__s{seed}"` (pack = k = `all` for the shared run). Run dir `$R/results/go_withholding/runs/<run_id>/`.
- **Preflight (refuses to start, exit 2):** free disk ≥ 3 GB; `protocol_pins.json` contains G1, G1-leak (pass), G2-targets, G2-covariate, G2-cache, G2-validation, G3, G4-lambda, G4-configs, G5; every pinned file hash still matches its file (frozen_hashes files, target table, covariate, cache, the `target_table["n_conditions"]` condition manifests listed in pin G3 — the count in G3 must equal the table's — the four cell configs listed in the **last** `G4-configs` entry, which must list exactly `baseline.json`, `loss_only.json`, `neg_only.json`, `both.json` (§12.1), and lambda_choice); `code_sha256()` equals `code_pin.json`; no existing run dir under `runs/` has `"pilot": true`.
- **Stopped-grid preflight (review v4 finding 1).** Checked after the pin preflight, before `check_pending_groups()` and before any training subprocess:
  - `run`: if any `runs/<run_id>/GROUP_FAILED.json` exists (dirs named `*.groupfailed*` or `*.attempt*` are not scanned), exit 4 with `"grid stopped: group <cond>,s<seed> failed the identity check; see <path>/GROUP_FAILED.json; fix under an amendment (protocol §13)"`, listing every such group. Nothing is trained or checked.
  - `plan`: the same scan, printed as a WARNING with the same text; `plan` still prints the plan and exits 0.
  - `status`: unchanged; it never refuses on an existing `GROUP_FAILED.json`. It reports `n_groups_group_failed` and lists the groups, and still runs `check_pending_groups()` for other groups (a newly detected mismatch there exits 4, as below).
  - `check-group`: not affected (it works on four explicit dirs, used at G6).
- **Clearing a failed group (amendment only, by hand; never automatic).** (1) Diagnose and fix the cause; (2) add a row to the amendments table in protocol §13 (columns `id | date | gate | groups cleared | cause | fix | re-pin`; the "groups cleared" cell lists every group the amendment authorises as `<cond_id>,s<seed>` entries separated by `;`; a code fix also requires a new `sweep.py pin` under the existing re-pin rule, recorded in the "re-pin" cell); (3) for each affected group rename each of its existing run dirs `<run_dir>` to `<run_dir>.groupfailed<n>`, where `<n>` is the smallest integer ≥ 1 not already used for that dir (never delete). The group then has no dirs under its `run_id`s, so it is **incomplete** and the next `run` retrains it in normal enumeration order. Renamed dirs are ignored by enumeration, completion, preflight scans, `check_pending_groups()` and the analysis (§11 step 5).
  - Step (3) is done by `sweep.py clear-group --cond C --seed S --amendment ID`, which the user runs by hand; the sweep never calls it. It skips the pin preflight (a cleared group may follow a re-pin). It refuses, exit 2, nothing renamed, with an ERROR naming the failed rule, unless **all** of the following hold (review v5 findings 1 and 2):
    - `ID` matches `AMENDMENT_ID_RE` (`^AM[1-9][0-9]*$`);
    - `parse_amendment_rows(PROTOCOL)` (§1.3) succeeds and returns **exactly one** row whose `id` cell equals `ID` (zero or two or more → refuse);
    - that row's `date` cell parses with `datetime.date.fromisoformat` as `YYYY-MM-DD` (exactly 10 characters), and its `cause` and `fix` cells are non-empty;
    - that row's `groups cleared` cell, split on `;`, each entry stripped of whitespace and surrounding backticks, contains the exact string `f"{C},s{S}"`;
    - `sweep_log.jsonl` has no earlier `group_cleared` line with the same `(amendment, cond_id, seed)`, so one amendment row clears a given group once; a second failure of the same group needs a new amendment (an addition beyond the review);
    - **at least one** of the group's existing run dirs (any of the four `run_id`s, unrenamed) carries `GROUP_FAILED.json`. Dirs that do not exist are skipped. This covers an interrupt between the four `GROUP_FAILED.json` writes below.

    On success it renames **every existing** dir of the group (1–4 of them, whether or not each carries `GROUP_FAILED.json`) and appends one `{"event": "group_cleared", "cond_id", "seed", "renamed_to": [...], "amendment": ID, "amendment_row": {<the parsed row, verbatim cells>}, "time"}` line to `sweep_log.jsonl`. It is the only supported clearing path, so every clearing leaves a log line that names a recorded amendment row listing that group. Example ids are never written in protocol §13 outside a real row, and command examples use the placeholder `AM<n>`, which fails `AMENDMENT_ID_RE`.
- `plan` and `status` report totals as `target_table["n_runs"]` (2,420, or 2,020 under the shortfall; AM1); nothing in the driver hard-codes a condition or run count.
- **Per run:** `subprocess.run([sys.executable, str(ROOT/"scripts/run_learning_curve_point.py"), "--domain", "go_ppi", "--config", str(cfg), "--train-file", str(cond/"train.jsonl"), "--go-ppi-split-dir", str(cond), "--withholding-manifest", str(cond/"withholding_manifest.json"), "--study-cell", cell, "--go-diagnostics", str(run/"go_diagnostics.jsonl"), "--output-dir", str(run), "--seed", str(seed), "--compare-config", str(baseline_cfg)], cwd=ROOT, check=False)` — argument list, never a shell string; then `[sys.executable, str(ROOT/"scripts/go_withholding/evaluate_withholding.py"), "score", "--run-dir", …, "--relation"]`. The run asserts `lc_manifest["epochs"] == 15`.
- **Completion:** `<run>/DONE.json` = `{lc_manifest_sha256, metrics_sha256, code_sha256, finished}`. Complete runs are skipped, so the driver is resumable. A complete run is *not* a checked group: group status is tracked only by `GROUP_OK.json` (below).
- **Failures**, distinguished by which subprocess failed:
  - **Training failure** (runner exit ≠ 0): the run dir is renamed to `<run_dir>.attempt1` (never deleted) and the run is retrained once from scratch. A second training failure writes `FAILED.json` (`{"stage": "train", "attempts": 2, ...}`) and the driver continues.
  - **Evaluation-only failure** (runner exit 0, all §4.4 post-training checks recorded ok in `lc_manifest.json`, but `evaluate_withholding.py score` exit ≠ 0): the `eval/` subdir is renamed to `eval.attempt1` and scoring is re-run once on the **same run dir without retraining**. A second scoring failure writes `FAILED.json` (`{"stage": "eval", "attempts": 2, ...}`); the checkpoint is kept so scoring can be repaired later without retraining, only through `sweep.py rescore` (below).
- **`rescore --run-id <run_id>`** (review v5 finding 4), run by hand after the evaluation cause is fixed; the sweep never calls it. Refuses, exit 2, nothing changed, unless: `runs/<run_id>/FAILED.json` exists with `"stage": "eval"` (a `"train"` failure is never rescored; it stays permanently missing); `lc_manifest.json` exists, records every §4.4 post-training check ok and `epochs == 15`, and the checkpoint it names exists; the run dir has no `DONE.json`. It then runs the full pin preflight (exit 2 on mismatch; an evaluator code fix therefore needs a re-pin under an amendment first) and the stopped-grid preflight (exit 4 if any `GROUP_FAILED.json` exists). On passing: renames `eval/` to `eval.attempt<n>` (smallest unused `n`; never deleted), runs the same `evaluate_withholding.py score … --relation` subprocess as the per-run step, and:
  - **score exit 0:** writes `DONE.json` with the same schema and checks as a normal completion (`lc_manifest_sha256`, `metrics_sha256`, `code_sha256` equal to the pin), renames `FAILED.json` to `FAILED.repaired.json`, appends `{"event": "eval_repaired", "run_id", "attempt": n, "time"}` to `sweep_log.jsonl` (INFO), exits 0. The group then becomes eligible and is checked by the next `run`/`status` scan; `rescore` does not check groups itself.
  - **score exit ≠ 0:** leaves `FAILED.json` in place, updates its `attempts` count, logs `{"event": "eval_repair_failed", "run_id", "attempt", "exit_code", "time"}` (WARNING), exits 3.

  A repair does not change any protocol criterion; the protocol §7 missing-data rules apply to whatever is still `FAILED.json` when the analysis runs.
  - Both cases append a line to `sweep_log.jsonl` with `{"run_id", "stage", "attempt", "exit_code", "time"}`.
- **Group checks**, computed from the four `lc_manifest.json`:
  - `selection_sha256_by_epoch(baseline) == (loss_only)` and `(neg_only) == (both)` for every epoch;
  - `(baseline) != (neg_only)` in ≥ 1 epoch (proves the tiering actually changed negatives);
  - `unannotated_anchor_calls_by_epoch` equal across all four, and `unannotated_anchor_window is true` in all four (§5.6);
  - `selector_none_returns == selector_exceptions == 0` in all four;
  - `ok_batches` per epoch and `train_records` equal across all four;
  - `go_relation_loss.applied_batches > 0` exactly for loss_only and both.
- **Group-check sweep.** The check is triggered by a scan for groups that need it, not by "the 4th cell just finished", so neither an interrupt nor a `--cells` filter can skip it (review v3 finding 2). `check_pending_groups()` runs:
  - at the start of `run` (after preflight, before the first training subprocess);
  - after every completed run inside `run`;
  - at the end of `run`, including after `--max-runs` is reached;
  - at the start of `status`.

  It enumerates every (cond, seed) group under `runs/` (dirs named `*.groupfailed*` or `*.attempt*` are skipped). Eligibility is defined by validity, not by marker presence (review v4 finding 2). A group is **eligible** if:
  - all four cell dirs have `DONE.json`;
  - none has `GROUP_FAILED.json` or `FAILED.json`;
  - at least one dir lacks a **valid** `GROUP_OK.json`, using `group_ok_valid` from `withholding_constants.py`, the same predicate as §11 step 5 (same four `run_ids` in cell order; per-dir `lc_manifest_sha256` equal to the current `sha256(lc_manifest.json)` and to `DONE.json["lc_manifest_sha256"]`).

  So a group left with `GROUP_OK.json` in only some dirs (interrupt between the four writes), or with a stale `GROUP_OK.json` whose hashes no longer match, is re-checked.

  **Manifest-drift pre-check (review v5 finding 3).** For each eligible group, before any group check, every dir's `sha256(lc_manifest.json)` is compared with its `DONE.json["lc_manifest_sha256"]`. If any differ, the group is **not checked**: `GROUP_FAILED.json` is written into all four dirs with `"failures": ["lc_manifest_changed_after_done: <run_id>", …]` (one entry per drifted dir) and the other keys of the mismatch case below, any `GROUP_OK.json` is renamed to `GROUP_OK.superseded.json`, ERROR is logged, and the command exits 4. The group then follows the normal stopped-grid, amendment and `clear-group` path, and is retrained. Without this rule `group_ok_valid` could never become true for such a group, so it would be rewritten on every scan and `n_groups_unchecked` would never reach 0.

  For each eligible group that passes the pre-check, the checks are recomputed from the four `lc_manifest.json`, never copied from an existing marker:
  - **Pass:** write `GROUP_OK.json` into **all four** dirs, overwriting any existing one. It contains `{"run_ids": [baseline, loss_only, neg_only, both], "lc_manifest_sha256": {run_id: sha}, "selection_sha256_by_epoch": {"baseline_loss_only": {...}, "neg_only_both": {...}}, "checked": <ISO time>, "code_sha256": …}`. Each file is written atomically (a temp file in the same dir, then `os.replace`). The four writes together are not atomic, but rewriting is idempotent: an interrupt leaves the group eligible, and the next scan completes it. After an interrupted write the manifests are unchanged, so the recomputed check passes again and the new files differ from the earlier ones only in `checked`; a partial write alone can never turn a group into a failure.
  - **Mismatch:** write `GROUP_FAILED.json` (same keys plus `"failures": [...]`) into all four dirs, and rename any `GROUP_OK.json` in them to `GROUP_OK.superseded.json` (a stale pass marker must not coexist with a failure; nothing is deleted), log ERROR, and exit 4. Under `status` the outcome is identical: status is not read-only for this one action. From then on every `run` refuses at the stopped-grid preflight until the group is cleared.

  A group with a `FAILED.json` member is never checked, and the analysis counts the tuple as missing. The only way such a member changes state is `sweep.py rescore` (eval-stage failures only, above): on success the dir has `DONE.json` and `FAILED.repaired.json` and no `FAILED.json`, so the group becomes eligible and is checked by the next `run`/`status`. A `"stage": "train"` member is permanently missing.

  The interrupt reasoning for `GROUP_OK.json` does not carry over to `GROUP_FAILED.json`: an interrupt after 1–3 of the four writes leaves a stopped grid with the marker in only some dirs. That state is fail-closed (the stopped-grid preflight fires on any one marker) and is cleared by the normal path, because `clear-group` needs only one marked dir and renames all existing ones (review v5 finding 2).
- **`status`** additionally reports `n_groups_ok` (valid `GROUP_OK.json` in all four dirs), `n_groups_unchecked` (should be 0 after the initial scan), `n_groups_failed_member`, `n_groups_group_failed` (with the group list) and `n_groups_cleared` (from `group_cleared` log lines). It exits 0 when existing `GROUP_FAILED.json` markers are its only finding; it exits 4 only when its own scan finds a new mismatch.
- **`check-group --run-dirs …`** applies the same checks to four explicit dirs (the manifest-drift pre-check only to dirs that have a `DONE.json`; the G6 smoke dirs have none). It writes `GROUP_OK.json` or `GROUP_FAILED.json` into them with the same schema, so the G6 smoke dirs end up marked the same way.
- **Logging:** each attempt appends a JSON line to `$R/results/go_withholding/sweep_log.jsonl`.

There is no mode that runs all `n_runs` (2,420, or 2,020 under the shortfall; AM1) in one invocation; `--max-runs` is mandatory and capped at 100, so the full grid (≈ 136–257 h CPU under AM1, pre-AM1 ≈ 46–87 h; the existing GO f100 run took ≈ 278 s for 15 epochs) is an explicit, repeated user decision. Disk per run ≈ 3 MB (final checkpoint) + 90 KB scores + logs; 2,420 runs ≈ 7.7 GB. Post-AM1 implementation note (review R3, 2026-10-10): the 3 GB sweep preflight is kept because it is checked on every invocation, and one invocation writes at most 100 runs (≈ 0.32 GB). It does not bound the whole grid; the runbook asks for ≈ 12 GB free before regenerating the conditions.

## 13. Code freeze and career safety

- **Uncommitted edits.** Nothing is committed, stashed, reset or checked out. The uncommitted edits in `/Users/olukotunjosh/Downloads/CDCL/contrastive_learning/{batch_processor,loss_engine,trainer}.py` remain byte-for-byte; new edits are pure insertions:
  - `loss_engine.py`: init block, five methods, three wrapped return expressions in `compute_loss`;
  - `trainer.py`: the 4-line epoch_stats + callback loop (§5.5);
  - `data_structures.py`: two fields (`go_ontology_loss_weight`, `go_ppi_unannotated_anchor_window`);
  - `go_ppi_domain/negative_selector.py` (not a user-edited file): one constructor keyword, one `from_config` argument, one `use_random_window` flag replacing two identical tests (§5.6);
  - `batch_processor.py`: **not edited**.
- **Career/ordinal behaviour** is unchanged. Every new branch is gated on one of three things, none of which any existing config or code path sets:
  - `go_ontology_loss_weight > 0`;
  - `go_ppi_unannotated_anchor_window` (default `False`, read only by the GO/PPI selector);
  - `epoch_end_callbacks`, an attribute only the study runner sets.
- **Regression gate (G0):** `"$PY" -m pytest "$R/go_ppi_domain/tests" -q` (existing 20 + new) and `"$PY" -m pytest "$R/tests/test_career_ordinal_candidates.py" "$R/tests/test_mesh_soft_sampling.py" "$R/tests/test_career_adapter_equivalence.py" "$R/tests/test_career_golden_regression.py" -q` (28) both pass. Before implementation, the implementer records `git diff --stat` and `sha256` of the three user-edited files' current diff hunks (read-only `git diff HEAD -- <file>`), and re-checks after implementation that each original hunk is still present verbatim.
- **Code pin** (`withholding_constants.code_sha256`). `trials_domain` is included because the study runner executes `preencode_pool` and `assert_cache_valid` from `/Users/olukotunjosh/Downloads/CDCL/trials_domain/label_budget_runner.py`, which fill the embedding cache every run trains from:

  ```python
  CODE_PIN_PATHS = ("contrastive_learning", "go_ppi_domain", "scripts", "trials_domain",
                    "run_phase1_embedding_evaluation.py")
  out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "--", *CODE_PIN_PATHS],
                       cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
  files = sorted(p for p in out if p.endswith(".py"))
  code_sha256 = sha256("\n".join(f"{p}\t{sha256_file(ROOT / p)}" for p in files))
  ```

  Tracked **and untracked** sources are included (`__pycache__`, `.venv`, `results/`, `preprocess/`, `embedding_cache/` are gitignored). The `git ls-files` arguments are relative because the command runs with `cwd=ROOT`.

- **Import coverage** (`withholding_constants.imported_repo_modules() -> list[str]`), so the pin's coverage is self-checking:

  ```python
  VENV = ROOT / ".venv"
  out = set()
  for m in list(sys.modules.values()):
      f = getattr(m, "__file__", None)
      if not f or not os.path.isabs(f):        # torch registers pseudo-modules with relative
          continue                              # __file__ ("_classes.py", "_ops.py") [ran]
      p = Path(f).resolve()
      if p.suffix != ".py" or not p.is_file():
          continue
      if p.is_relative_to(ROOT) and not p.is_relative_to(VENV):   # .venv lives inside ROOT
          out.add(str(p.relative_to(ROOT)))
  return sorted(out)
  ```

  Called at the end of the runner (§4.4) and of `evaluate_withholding.py score`; each asserts `set(imported) <= set(code_sha256()[1])` (fatal, exit 2). An import probe run on 2025-10-09 found the runner's go_ppi path loads files only from `contrastive_learning/`, `go_ppi_domain/` and `trials_domain/` [ran]. `sweep.py pin` (G5) repeats this probe in a subprocess that imports the runner and evaluator modules and refuses to pin if the probe's set is not covered, so coverage gaps surface before G5, not at G6. Stored with `git_head` in `code_pin.json` and every `lc_manifest.json`; the sweep refuses to run on mismatch. The protocol asks the user to commit or tag before G5 (gap item 1); the content pin enforces constancy either way.

## 14. Testability

**Unit tests** (fast, synthetic fixtures, no MPNet), new files under `$R/go_ppi_domain/tests/`:

- `test_go_withholding_loss.py`: weight 0 → `torch.equal` loss and identical grads for go_ordinal and InfoNCE, with and without a table; L_GO equals a hand computation for 3 proteins; negatives receive zero gradient from L_GO; unannotated proteins excluded; `RuntimeError` when weight > 0 and no table; `ValueError` for negative/inf weight, wrong domain, `loss_type="ordinal"`; `go_ordinal_components["base_total"]` equals `_compute_go_ordinal_loss` (atol 1e-6); counters update only with grad enabled and `_diagnostic_mode=False`; `go_ordinal_components(..., relation_table=t2)` leaves all counters unchanged.
- `test_go_relation_table.py`: dense table equals `GoMatcher.ontology_set_similarity` on closures for all pairs of a synthetic index (atol 1e-6); protein set equals train-file proteins; no matcher API attributes; `go_relation_loss` is pure (same inputs → same outputs, no mutation).
- `test_go_withholding_negatives.py`: the §5.4 identity tests, plus the §5.6 cases (review v3 finding 1). All of them use a synthetic index in which anchor `U` has no terms and anchor `A` is annotated.
  - **Flag true, anchor `U`:** for seeds 13 and 42 and epochs 0..14, the neg_only selector (`go_tiered=True`, `tier_sampling="stochastic"`, matcher set) returns the same `(partner_id, grade)` list, in the same order, as the baseline selector (`tier_sampling="random_window"`, no matcher). Its `distances` are all `NEUTRAL_DISTANCE`.
  - **Flag true, anchor `A`:** neg_only output differs from baseline in ≥ 1 epoch.
  - **Flag false (default):** neg_only on `U` reproduces today's output, i.e. the full-pool `else` draws, checked against a frozen expected list generated from the unmodified selector before the edit.
  - **Defaults:** `from_config` on a config without the field gives `unannotated_anchor_window is False`.
  - **Recorder count:** `SelectionRecorder(..., unannotated_anchors={"U"})` counts `U`'s calls per epoch.
- `test_materialize_configs.py` (review v3 findings 3, 6):
  - **Exact diff sets:** a doctored loss_only that also sets `go_ppi_go_tiered_negatives=true` is refused, and so is a neg_only that differs in `go_ppi_unannotated_anchor_window`. The correct four configs pass. All four have `go_ppi_unannotated_anchor_window: true`.
  - **Pins:** `--lambda none` appends no pin and writes no `configs.sha256.json`. `--lambda 0.3` appends exactly one `G4-configs` entry, listing exactly the four cell files.
- `test_withholding_generator.py`: both directions dropped; pool anchors and members removed; **a pool-only protein annotated with t is removed as anchor and as member at k=0 and at k=25**; retention nested (R₁ ⊂ R₅ ⊂ R₁₀ ⊂ R₂₅); **exactly k** retained; **at k = 5 every R_5(t) protein that was a pool anchor/member in the base pools is still a pool anchor/member in the written pools, while every W protein is gone** (review v2 finding 1); the condition count equals `n_conditions` for a 24-target (121 conditions) and a 20-target (101 conditions) singleton-pack fixture table (AM1; pre-AM1 an 8-pack and a 7-pack table); empty-pool anchor rule; removal fractions; `k="all"` manifest typing; idempotent regeneration; mismatch raises.
- `test_target_table.py`: N_sib excludes ancestors/descendants; a `regulates` line in a fixture OBO is ignored; T_train vs P_train roles; S arithmetic; matching determinism; the size rule (13 → first 12, 11 → first 10, 10 → 10, 9 → exit 3); `n_conditions`/`n_runs` fields; `retention_order` is a list (the v1 `shuffle` bug) and deterministic; build twice → identical sha. Under AM1 (§22): `singleton_packs` gives one target per pack in target order and only ever checks single-target packs; a target failing the per-target removal cap stops with exit 3 and a counts-only message; `DEFAULT_PARAMS` has `pack_size` 1 and no `pack_nbr_overlap_cap`/`pack_search_seed`; the end-to-end table has `n_packs == n_targets`, `n_conditions = 5·n_targets + 1` and no `pack_search_log`. (The pre-AM1 packing-constraint, fallback-order and exact `pack_search_log` tests were removed with the machinery.)
- `test_leak_pruning.py`: self symbol kept, partner masked; boundaries ("TP53-binding" masked, "TP530" not); doc-frequency filter; `parse_gaf` with/from filter incl. self-reference kept; `parse_gaf` defaults identical to current output on a fixture GAF; **`rewrite_proteins`** preserves ids and order, keeps the prefix, recomputes `go_uris`/`coarse_uris`/`n_terms` from the given index, raises if the prefix is missing.
- `test_runner_withholding_assertions.py`: each §4.2–§4.4 check raises when violated (doctored manifests, stub trainers); relative `_path` value rejected; index sha mismatch rejected before any matcher is built.
- `test_epoch_guard_and_recorder.py`: guard raises on `failed_batches>0`, non-finite loss, loss-on cell with applied share < 0.9, off cell with applied > 0; a fake trainer whose `_train_batch` raises inside the batch `try` still ends with `RuntimeError` from `train()` (real `_train_epoch` path via a minimal trainer subclass); recorder hash is order-independent, counts `None` and exceptions (and re-raises), ignores paused calls.
- `test_withholding_stats.py`: deterministic for a fixed seed; two-way bootstrap CI covers a known mean on synthetic pack/seed-correlated data and is wider than the target-then-seed CI there; Holm ordering; ordered decision rules (each of the five branches and the boundary cases from review finding 8); completeness rule with the `n_targets × 6 × 5` denominator; definedness: Δ_small undefined when one of k ∈ {0,1,5} is missing, D_c undefined when the `all` tuple is missing, replicates with zero defined cells discarded and counted, `unstable` flag above 5%; `E_c` equals Δ_c(t,0,s) − Δ^coll_c on a hand-built fixture.
- `test_withholding_eval_metrics.py`: per-target AUC equals `sklearn.metrics.roc_auc_score`; collateral query selection; checkpoint resolved from `lc_manifest.json` (2-epoch fixture → `checkpoint_epoch_1.pt`).
- `test_code_pin.py`: an untracked `.py` file under `go_ppi_domain/` changes `code_sha256`; a file under `trials_domain/` is in the pin list; `imported_repo_modules()` skips modules with relative `__file__` and anything under `.venv`, and includes a freshly imported repo module.
- `test_runner_withholding_assertions.py` additionally: `--pilot` without `--withholding-manifest` is an argparse error; `--pilot` skips only the λ check and writes `"pilot": true`; a relative `_path` key in the JSON file is rejected while relative `TrainingConfig` defaults absent from the JSON are ignored.
- `test_negative_audit.py`: audit with `--seed 13` yields `selector.training_seed == 13` even though the config carries 42; `share_withheld_negatives` gate trips on a doctored pool containing a W protein, but a retained protein drawn at k = 5 only raises the descriptive shares.
- `test_sweep_driver.py`, existing cases: eval-only failure re-scores without retraining (the runner subprocess is not called again); training failure renames to `.attempt1` and retrains; pilot run dirs are refused.
- `test_sweep_driver.py`, group-check cases (review v3 finding 2). Subprocesses are stubbed and the `lc_manifest.json` files are synthetic.
  - **Interrupt case:** four `DONE.json` dirs of one group without `GROUP_OK.json`, as left by an interrupted driver. They are checked, and `GROUP_OK.json` is written to all four, by the next `run --max-runs 1` before its first training subprocess, and also by `status` alone.
  - **`--cells` case:** cells completed by four separate `run --cells <c>` invocations are checked after the 4th.
  - **Mismatch:** a doctored loss_only selection hash gives `GROUP_FAILED.json` in all four dirs and exit 4 from both `run` and `status`.
  - **Stopped grid (review v4 finding 1):** after the mismatch case has written `GROUP_FAILED.json`, a second `run --max-runs 1` exits 4 before any training subprocess (the stubbed runner records zero calls) and before `check_pending_groups()`; `plan` exits 0 and prints the WARNING; `status` exits 0 and reports `n_groups_group_failed = 1`.
  - **Clearing:** with a fixture protocol whose §13 has a valid row for the group, `clear-group` renames the four dirs to `<run_dir>.groupfailed1` (a pre-existing `.groupfailed1` yields `.groupfailed2`), appends one `group_cleared` log line containing the parsed `amendment_row`, and the next `run --max-runs 4` passes the stopped-grid preflight and retrains the group's four cells. `analyze_withholding.py` ignores the `.groupfailed*` and `.attempt*` dirs (a fixture with only renamed dirs classifies the group as incomplete).
  - **Amendment rows (review v5 finding 1).** Each refusal is exit 2 with nothing renamed:
    - (a) the **real** v6 `protocol_go_ppi.md` (empty amendments table, no rows) refuses `--amendment AM1`;
    - (b) a fixture row `| AM1 | 2025-11-01 | G7 | p03_k5,s21 | … | … | no |` allows clearing `p03_k5 s21` and refuses `p02_k0 s13`;
    - (c) the same row with an empty `fix` cell refuses; so do an empty `cause` cell and a `date` of `2025-13-01` or `01/11/2025`;
    - (d) `AM1` written only in prose in §13, not as a row, refuses; two rows with id `AM1` refuse; `--amendment AM0`, `AM01` and `A1` fail `AMENDMENT_ID_RE`;
    - (e) a row for `AM1` placed in §12 or §14 (outside the `## 13.`–`## 14.` span) is not found;
    - (f) a second `clear-group` for the same group with the same `AM1`, after the group failed again, refuses because of the earlier `group_cleared` line;
    - (g) `parse_amendment_rows` ignores the §13 pins table (different header), and a groups cell written with backticks (`` `p03_k5,s21`; `all,s13` ``) authorises both groups.
  - **Partial `GROUP_FAILED.json` (review v5 finding 2):** `GROUP_FAILED.json` in 2 of 4 dirs (interrupted write). `run` exits 4 at the stopped-grid preflight; `clear-group` with a valid row renames all four dirs. A group with 3 existing dirs (one never created), one of them marked, renames those 3. A group with no marked dir refuses.
  - **Manifest drift (review v5 finding 3):** four `DONE.json`, no `GROUP_OK.json`, one `lc_manifest.json` edited after `DONE.json`. `status` writes `GROUP_FAILED.json` into all four with `failures == ["lc_manifest_changed_after_done: <that run_id>"]`, runs no group check (a spy on the check function records zero calls), and exits 4; the same fixture with a pre-existing valid-looking `GROUP_OK.json` has it renamed to `GROUP_OK.superseded.json`.
  - **`rescore` (review v5 finding 4):** with a stubbed `score` subprocess:
    - an eval-stage `FAILED.json` member with exit-0 scoring gets `DONE.json`, `FAILED.repaired.json`, no `FAILED.json`, `eval.attempt<n>` kept, one `eval_repaired` log line, and the next `status` checks the group;
    - refuses (exit 2) on a `"stage": "train"` member, on a dir with no `FAILED.json`, on a dir that already has `DONE.json`, on a missing checkpoint, and on a pin mismatch;
    - exits 4 while any `GROUP_FAILED.json` exists;
    - a non-zero scoring exit leaves `FAILED.json`, increments `attempts`, logs `eval_repair_failed` and exits 3.
  - **Partial `GROUP_OK.json` (review v4 finding 2):** two of four dirs carry a valid `GROUP_OK.json`, two have none. `status` treats the group as eligible, recomputes the checks and writes `GROUP_OK.json` into the other two; all four files then agree on `run_ids` and `lc_manifest_sha256`. A stale `GROUP_OK.json` (`lc_manifest_sha256` not matching the file) also makes the group eligible and is rewritten. Mismatch on re-check renames any `GROUP_OK.json` to `GROUP_OK.superseded.json`.
  - **`FAILED.json` member:** the group is never checked.
  - **Analysis preflight** (`analyze_withholding.py`, same fixture):
    - exit 2 on an unchecked group, even with `--allow-incomplete`;
    - exit 2 on a `GROUP_FAILED.json` group;
    - exit 2 on a `GROUP_OK.json` whose `lc_manifest_sha256` does not match the file;
    - the tuple is present once checked;
    - a failed-member group is counted missing.
- `test_withholding_eval_metrics.py` additionally (review v3 finding 4): `build-cache` sets `max_seq_length == 512` and exits 2 on a fixture text whose token length is ≥ 512 (stub tokenizer).
- `test_withholding_eval_metrics.py`, encoder cross-check (review v4 finding 3), with a stub trainer whose `batch_encoder._content_to_text` is the real `BatchEfficientEncoder._content_to_text` and a stub tokenizer: a view with a trailing space counts in `n_text_differs` but not in `n_token_mismatch`; a stub tokenizer that maps trailing whitespace to an extra token makes `n_token_mismatch > 0` and the check exit 2.
- `test_epoch_guard_and_recorder.py` additionally (review v4 finding 4): the recorder exposes `unannotated_anchor_calls_by_epoch` (no `unannotated_calls_by_epoch` attribute), and the runner's `negative_selection` block built from a stub recorder and selector has exactly the §4.7 keys with the §4.7 sources.

**Integration tests** (`@pytest.mark.slow`, real configs and MPNet, run manually): diagnostics non-interference (§6); real-trainer selection identity (§5.4); a forced L_GO failure (table detached mid-run) aborts the run with exit 2 and `run_error.json`. The 2-epoch smoke on `p01_k0` × 4 cells × seed 13 is protocol gate G6; it includes one `score --encoder-crosscheck` call (§9.2) and `check-group`, which writes `GROUP_OK.json` into the smoke dirs.

## 15. Error handling summary

| operation | failure | class | caller receives | logged |
|---|---|---|---|---|
| build_pruned_features | raw sha mismatch, non-empty/protected out dir, pairs sha or id order mismatch, structural mismatch | fatal | exit 2 | ERROR |
| measure_leakage | acceptance fails | fatal for the study | `leak_audit.json` pass=false; exit 3 | ERROR |
| build_target_table | < 10 matched pairs; a selected target fails the single-target removal cap or retention feasibility (AM1; pre-AM1: packing infeasible at 20) | fatal; ask the user | exit 3 | ERROR |
| make_conditions | invalid k/pack, cap exceeded, not exactly k, residual W/V, disk < 5 GB (3 GB pre-AM1), non-idempotent | fatal | exit 2 | ERROR |
| runner preflight / condition assertions / post-training checks | any check (incl. relative path, index sha, wiring) | fatal | `run_error.json`; exit 2 | ERROR |
| loss engine | weight > 0 without table; bad weight; any L_GO exception | raised; swallowed per batch by the trainer, then **fatal via StrictEpochGuard** | `run_error.json`; exit 2 | ERROR (trainer batch log + guard) |
| loss engine | < 2 annotated proteins in a batch | recoverable | term skipped; `skipped_batches += 1` | DEBUG; counts in manifest; guard enforces applied share ≥ 0.9 |
| selector inside batch processor | exception or `None` during training | swallowed by batch processor (fallback); counted by recorder | post-training check fails → exit 2 | WARNING (bp) + ERROR (runner) |
| selector during the diagnostics pass (recorder paused) | exception | **intentionally fatal** (§4.6): counted in `recorder.exceptions` | diagnostics row written, training continues; post-training check fails → exit 2 | WARNING (bp) + ERROR (runner) |
| diagnostics callback | any other exception | recoverable | row with `error`; training continues | WARNING |
| make_conditions | `unannotated_anchors.ids` ≠ anchors with empty pruned `terms_for` | fatal | exit 2 | ERROR |
| materialize_configs | pair diff set ≠ `EXPECTED_PAIR_DIFF`, or any `unexpected_fields` | fatal; nothing written | exit 2 | ERROR |
| evaluator `build-cache` | a text's token length ≥ 512 | fatal | exit 2 | ERROR |
| evaluator `score --encoder-crosscheck` (G6) | cos < 0.9999 or `max_seq_length` ≠ 512 | fatal (G6 fails) | `eval_error.json`; exit 2 | ERROR |
| sweep `run`/`status` group scan | group identity mismatch on a newly eligible group | fatal for the sweep | `GROUP_FAILED.json`; exit 4 | ERROR |
| analysis preflight | group with four `DONE.json` but no valid `GROUP_OK.json`, or any `GROUP_FAILED.json` | fatal (not relaxed by `--allow-incomplete`) | exit 2 | ERROR |
| diagnostics | component drift > 1e-5 | recoverable, flagged | `drift_flag` in row; listed by analysis | ERROR |
| λ pilot | < 4 validation batches; ratio not confirmed after one re-estimate | fatal; ask the user | exit 3 | ERROR |
| negative audit | offline identity mismatch; withheld (W) protein among negatives in a pack condition; selector seed ≠ `--seed`; share invariant broken | fatal for the grid | exit 2 | ERROR |
| runner / evaluator | imported repo module outside the code pin | fatal | `run_error.json` / `eval_error.json`; exit 2 | ERROR |
| sweep | eval-only failure | recoverable: re-score once, no retraining | `eval.attempt1`; then `FAILED.json` (`stage: eval`) | WARNING + `sweep_log` |
| sweep | pilot run dir in `runs/` or passed to `check-group` | fatal | exit 2 | ERROR |
| analysis | contrast with > 5% discarded replicates | recoverable | `unstable: true`; classified inconclusive | WARNING |
| evaluator | checkpoint/epoch/hash mismatch, NaN | fatal for that run | `eval_error.json`; exit 2 | ERROR |
| sweep | preflight pin mismatch | fatal | exit 2 | ERROR |
| sweep `run` preflight | any existing `runs/<run_id>/GROUP_FAILED.json` (grid stopped) | fatal until cleared under an amendment (§12.3); `plan` warns and exits 0; `status` reports and exits 0 | exit 4, nothing trained | ERROR (`run`), WARNING (`plan`) |
| sweep `clear-group` | id fails `^AM[1-9][0-9]*$`; §13 amendments table missing or malformed; not exactly one row with that id; bad date, empty cause or fix; group not listed in the row's "groups cleared" cell; group already cleared under that amendment; no existing dir of the group carries `GROUP_FAILED.json` | fatal, nothing renamed | exit 2, message names the failed rule | ERROR |
| sweep `run`/`status` group scan | `sha256(lc_manifest.json) != DONE.json["lc_manifest_sha256"]` in an eligible group | fatal for the sweep; group not checked | `GROUP_FAILED.json` (`lc_manifest_changed_after_done: <run_id>`); exit 4 | ERROR |
| sweep `rescore` | no `FAILED.json`, stage ≠ `eval`, `DONE.json` already present, post-training checks not ok, checkpoint missing, pin mismatch | fatal, nothing changed | exit 2 | ERROR |
| sweep `rescore` | any `GROUP_FAILED.json` exists | fatal (grid stopped) | exit 4 | ERROR |
| sweep `rescore` | scoring exits ≠ 0 | recoverable (member stays missing) | `FAILED.json` kept, `attempts` incremented; exit 3 | WARNING + `eval_repair_failed` log line |
| sweep `rescore` | scoring succeeds | — | `DONE.json`, `FAILED.repaired.json`; exit 0 | INFO + `eval_repaired` log line |
| sweep group scan | `GROUP_OK.json` in only some dirs, or stale hashes | recoverable: group eligible, checks recomputed, marker rewritten in all four | normal | INFO |
| evaluator `score --encoder-crosscheck` (G6) | trainer and evaluator text functions give different token ids for any cache protein | fatal (G6 fails) | `eval_error.json` with first 5 ids; exit 2 | ERROR |
| sweep | training fails twice | recoverable for the sweep | `.attempt1` kept; `FAILED.json` (`stage: train`); continue | WARNING + `sweep_log` |
| sweep | group identity / ok_batches mismatch | fatal for the sweep | `GROUP_FAILED.json`; exit 4 | ERROR |
| analysis | > 10% tuples missing | fatal unless `--allow-incomplete` | exit 3 | ERROR |

## 16. Files

New code:

- `/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/{leak_pruning.py, withholding_constants.py, withholding_targets.py, withholding.py, go_relation.py, withholding_diagnostics.py}`
- `/Users/olukotunjosh/Downloads/CDCL/scripts/go_withholding/{build_pruned_features.py, measure_leakage.py, build_target_table.py, make_conditions.py, materialize_configs.py, lambda_pilot.py, audit_negatives.py, evaluate_withholding.py, text_similarity_covariate.py, analyze_withholding.py, sweep.py}` (no `__init__.py`; run by absolute path)
- the tests of §14 under `/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/`.

Modified (additive only):

- `/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/go_ontology.py`
- `/Users/olukotunjosh/Downloads/CDCL/contrastive_learning/data_structures.py`
- `/Users/olukotunjosh/Downloads/CDCL/contrastive_learning/loss_engine.py`
- `/Users/olukotunjosh/Downloads/CDCL/contrastive_learning/trainer.py`
- `/Users/olukotunjosh/Downloads/CDCL/scripts/run_learning_curve_point.py`
- `/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/negative_selector.py` (§5.6; moved from "not modified" in v4)

Not modified:

- `/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/data_converter.py`
- `/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/data_splitter.py`
- `/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/run_config.py`
- `/Users/olukotunjosh/Downloads/CDCL/contrastive_learning/batch_processor.py`
- `/Users/olukotunjosh/Downloads/CDCL/trials_domain/label_budget_runner.py` (imported, pinned, not edited)

New data/result locations:

- `/Users/olukotunjosh/Downloads/CDCL/preprocess/go_withholding/{converted, splits, validation_targetfree, targets, conditions/<cond_id>}` and `.../leak_audit.json`
- `/Users/olukotunjosh/Downloads/CDCL/config/go_withholding/`
- `/Users/olukotunjosh/Downloads/CDCL/embedding_cache/go_withholding_text_embeddings.pt`
- `/Users/olukotunjosh/Downloads/CDCL/results/go_withholding/{lambda_pilot, smoke, runs/<run_id>, eval_cache, audits, analysis}` (`runs/` may also hold `<run_id>.attempt<n>` and `<run_id>.groupfailed<n>` dirs, kept for audit and ignored by the sweep and analysis) plus `fusion_weight.json`, `code_pin.json`, `protocol_pins.json`, `sweep_log.jsonl`

Read-only, never written: `/Users/olukotunjosh/Downloads/CDCL/preprocess/go_ppi/`, `/Users/olukotunjosh/Downloads/CDCL/preprocess/go_ppi_splits/`, `/Users/olukotunjosh/Downloads/CDCL/preprocess/go_ppi_lc/`, `/Users/olukotunjosh/Downloads/CDCL/dataset/go_ppi/` (enforced by `assert_writable_target`).

## 17. Responses to design review v1

| # | sev | finding | response |
|---|---|---|---|
| 1 | HIGH | pairs byte-identity infeasible; converter hash-seed dependent | **Addressed.** Converter not called or modified; pairs copied verbatim; `rewrite_proteins` regenerates proteins; deterministic splitter + structural hashes (§0.1-1, §1.1, §1.4). `test_converter_defaults.py` replaced by a `rewrite_proteins` test. |
| 2 | HIGH | W misses pool-only target-annotated proteins | **Addressed.** `T_train` (any role) vs `P_train` (positives); W = ∪(T_train \ R_k); `train_annotated_proteins` stored; filters, residual scans, runner checks and audit (`share_target_annotated_negatives` = 0) use it; generator test added (§2.1, §3, §4.3, §7). |
| 3 | HIGH | loss errors swallowed per batch | **Addressed.** `epoch_stats` passed to callbacks; non-catching `StrictEpochGuard` (also checks the term actually applied); `ok_batches` recorded and compared across cells (§4.5, §5.5, §12.3). |
| 4 | MED | `build_or_load` can overwrite pruned index | **Addressed.** Index sha + meta preflight before any matcher, in runner and audit; `0o444` on the pickle and frozen files (§0.1-7, §1.4 step 9, §4.2). |
| 5 | MED | k=25 can become infeasible after packing | **Addressed.** Retention feasibility against `W_pack(0)` is a packing constraint, and eligibility also requires the record to survive the empty-pool rule (§2.2 rule 5, §2.6, §2.7). |
| 6 | MED | evaluator hard-codes epoch 14 | **Addressed.** Checkpoint and epochs read from `lc_manifest.json`; sweep asserts epochs == 15 (§9.2, §12.3). |
| 7 | MED | λ pilot batches infeasible | **Addressed.** All target-free grade-2 validation records, batches of 32, median, ≥ 4 batches else stop and ask; counts recorded (§12.2). |
| 8 | MED | interpretation rows overlap; mechanism rule vague | **Addressed** in protocol §8 (ordered first-match rules; operational mechanism rule); evaluated by `decisions.json` (§11 step 6). |
| 9 | MED | bootstrap ignores shared runs | **Addressed.** Primary = two-way pack × seed cluster bootstrap; v1 scheme kept as S1 (§11). |
| 10 | MED | code pin misses untracked files | **Addressed.** `git ls-files -co --exclude-standard` content hash over `.py` files (§13). Restricting to `.py` avoids doc edits breaking the pin. |
| 11 | MED | relative paths | **Addressed.** `$R` = literal absolute path; configs written absolute and asserted; scripts run by absolute path with `sys.executable` and `cwd=ROOT`; protocol commands start with `R=…; PY=…; cd "$R"` (§header, §4.2, §12.1, §12.3). |
| 12 | MED | negative identity not proven on the real trainer | **Addressed.** Run-time `SelectionRecorder` hashes per epoch; sweep group checks; real-trainer slow test (§4.6, §5.4, §12.3). Added the converse check baseline ≠ neg_only. |
| 13 | MED | no reusable relation-loss function; counters polluted | **Addressed.** Pure `go_relation.go_relation_loss`; counters gated on grad + `_diagnostic_mode`; `go_ordinal_components(relation_table=)` never touches counters (§5.2, §5.3). |
| 14 | NIT | `shuffle(sorted())` returns None | **Addressed** (§2.7). |
| 15 | NIT | `from_train_file` syntax | **Addressed** (`from_train_file(Path(config.go_index_cache), Path(train_file))`, §4.3). |
| 16 | NIT | evaluator imports unnamed | **Addressed** (§9). |
| 17 | NIT | ≤ k should be == k | **Addressed** in generator and runner (§3.4, §4.3 check 5). |
| 18 | NIT | with/from metric (c) circular | **Addressed.** (c) kept as an implementation check; ungated residual (d) for HGNC/ENSEMBL/ComplexPortal (§1.5). |
| 19 | NIT | clip share measured in eval mode | **Addressed** by relabelling `share_above_clip_eval_diag` (§6). Counting in-training clipping would need an edit inside `_train_batch` (user-edited file); not worth the risk for a descriptive metric. |
| 20 | NIT | covariate needs embeddings absent at G2 | **Addressed.** Encode missing texts; `n_encoded` recorded (§10). |
| 21 | NIT | `k` type for `all` | **Addressed** (`int | "all"`; check 5 skipped for all) (§3.3, §4.3). |
| 22 | NIT | preflight reads pin from markdown | **Addressed.** `protocol_pins.json` written at each gate; preflight reads it; protocol §13 mirrors it (§1.3, §12.3). |

Additional changes found while revising (not in the review): the batch processor also swallows selector exceptions (now counted, §0.1-6); "projection params" defined without assuming a `projection_head` attribute (§0.1-14); `_compute_go_ordinal_loss` is no longer edited — L_GO is applied in `compute_loss` (§5.3), which keeps the user-edited function untouched; a component-drift runtime check guards the duplicated components loop (§6).

## 18. Responses to design review v2

| # | sev | finding | response |
|---|---|---|---|
| 1 | MED | audit gate "zero target-annotated negatives at every k" contradicts retention | **Addressed.** Gate is `share_withheld_negatives == 0` (W only); `share_target_annotated_negatives` and new `share_retained_negatives` are descriptive, with the identity target-annotated = withheld + retained asserted (§7). Generator asserts R_k stays in pools and a k=5 test was added (§3.4, §14). Protocol §3 now says "zero withheld proteins". |
| 2 | MED | code pin misses `trials_domain/` | **Addressed.** `trials_domain` added to `CODE_PIN_PATHS`; `imported_repo_modules()` asserted ⊆ pin in runner and evaluator, and probed at `sweep.py pin` (§13, §4.4). The review's snippet was adjusted: `.venv` lives inside ROOT and torch registers `_classes.py`/`_ops.py` with relative `__file__`, so both are filtered [ran an import probe]. |
| 3 | MED | missing-tuple handling unspecified | **Addressed** exactly as proposed (§11 steps 2, 3, 5; protocol §7): Δ_small needs all of k∈{0,1,5}; D_c needs Δ_small and Δ_all; replicate mean over defined cells; empty replicates discarded and counted; denominator `n_targets×6×5` (720). One addition: > 5% discarded replicates → `unstable`, classified inconclusive, so a sparse contrast cannot be called supported. |
| 4 | MED | shortfall/packing fallbacks incomplete; 41/820 hard-coded | **Addressed.** 10–11 pairs → exactly the first 10; queue exhaustion at 24 → original first 10 pairs packed 6×3+1×2; exhaustion at 20 → exit 3 (§2.4, §2.6). `n_targets/n_packs/n_conditions/n_runs` stored in the table and read by G3, sweep preflight, plan/status and the analysis (§2.8, §3.3, §11, §12.3). |
| 5 | MED | A1 subtracts neg_only's k=0 Δ from loss_only's | **Addressed.** `E_c(t,s) = Δ_c(t,0,s) − Δ^coll_c(pack,0,s)` from the same runs, two-way bootstrap, `descriptive` family (§11). neg_only k=0 keeps the findings-(iv) reading, reported alongside, not subtracted (protocol §8 A1). |
| 6 | NIT | `--pilot` undeclared | **Addressed** (§4.1): store_true, requires manifest, skips only step 4, `pilot: true` in manifest, sweep and check-group refuse. |
| 7 | NIT | `GoIndex.build` signature inconsistent | **Addressed** (§1.2): one keyword-only signature with `leak_pruning` and `with_from_symbol_map`; `ValueError` if pruning without map; descendants from full `parse_obo` parents. Verified against the current signature at `go_ontology.py:260`. |
| 8 | NIT | absolute-path check scope | **Addressed** (§4.2 step 1, §12.1): JSON-file keys plus CLI overrides only. |
| 9 | NIT | audit selector seed implicit | **Addressed** (§7): `cfg.training_seed = args.seed`, assert `selector.training_seed`. |
| 10 | NIT | mechanism (a) is a point estimate | **Addressed** in protocol §8 (two-way bootstrap ci_low > 0); computed in the `descriptive` family (§11). |
| 11 | NIT | simGIC IC is corpus-wide | **Addressed** in protocol §5. |
| 12 | NIT | relative prose paths | **Addressed** (§0.1 items 5–6, §13, §16). `git ls-files` arguments stay relative (`cwd=ROOT`). |
| 13 | NIT | role counts and free disk | **Addressed** (§0.1 items 3 and 10, §3.3): 4,709 with positives, 4,948 anchors, 10,608 members; 5,498 member-only + 401 anchor-only = 5,899 without positives; 9.6 GiB free [ran `df`]. |
| 14 | NIT | eval vs training retry semantics | **Addressed** (§12.3): eval-only failure re-scores once without retraining; training failure renames to `.attempt1` and retrains; both logged. |
| 15 | NIT | selector epoch during λ validation | **Addressed** (§12.2 step 2): `set_epoch(epochs-1)` on batch processor and loss engine (both methods exist: `batch_processor.py:1035`, `loss_engine.py:540`), `selector_epoch` recorded. |

## 19. Responses to design review v3

| # | sev | finding | response |
|---|---|---|---|
| 1 | MED | Pruned (BP-empty) anchors take the selector's uniform full-pool `else` branch in neg_only/both: a window-narrowing confound | **Addressed** as proposed. <ul><li>New `TrainingConfig.go_ppi_unannotated_anchor_window` (default `False`) and selector keyword `unannotated_anchor_window`. One `use_random_window` flag routes empty-annotation anchors in GO-tiered cells through the unchanged `random_window` window and sampling code (§5.6, §5.1).</li><li>Set `true` in all four study configs, so it is a shared field (§12.1).</li><li>Asserted in check 7 and in the group check (§4.3, §12.3).</li><li>§0.1-1 prose corrected, and `negative_selector.py` moved to "Modified (additive)" (§16, §13).</li><li>Unit tests: neg_only == baseline draws for an unannotated anchor, and the default reproduces today's output (§14).</li><li>Per-condition counts in `withholding_manifest.json["unannotated_anchors"]`, cross-checked against the pruned index (§3.3). Per-epoch recorder counts in `lc_manifest.json` (§4.6, §4.7).</li><li>Protocol §5 updated.</li></ul> |
| 2 | MED | The negative-identity group check can be skipped on resume or with `--cells`, and the analysis admits unchecked tuples | **Addressed** as proposed. <ul><li>`check_pending_groups()` scans for eligible groups (four `DONE.json`, no marker) at the start of `run`, after every run, at the end of `run`, and at the start of `status`.</li><li>On pass it writes `GROUP_OK.json` atomically into all four dirs, including the `lc_manifest_sha256`s. On mismatch it writes `GROUP_FAILED.json` and exits 4. `check-group` writes the same markers (§12.3).</li><li>The analysis requires a valid `GROUP_OK.json` in all four dirs. An unchecked or group-failed group is a fatal preflight error (exit 2), not relaxed by `--allow-incomplete` (§11 step 5).</li><li>One alignment beyond the review: v3 counted `GROUP_FAILED` tuples as missing, while protocol §7 said such a grid is not analysed. v4 makes it fatal in the analysis too.</li><li>Sweep and analysis tests added (§14). Protocol §7 updated.</li></ul> |
| 3 | NIT | `report_factor`'s union expected-field set lets a loss_only config with tiering pass | **Addressed** (§12.1). `EXPECTED_PAIR_DIFF` asserts the exact diff set per pair; the materialiser writes nothing on mismatch; test added (§14). |
| 4 | NIT | Eval encoder not pinned to `max_seq_length=512`; the cross-check is a no-op at G2 | **Addressed.** <ul><li>`build-cache` sets 512 and asserts that the maximum token length is below it; `max_tok` is recorded (§9.1).</li><li>The no-op G2 cross-check is removed. `score --encoder-crosscheck` at G6 always re-encodes 50 fixed test texts through `trainer.text_encoder` and requires cos ≥ 0.9999 (§9.2).</li><li>The review's "trainer cache entries if present" branch is dropped: test proteins are disjoint from train, so that cache normally holds none of them.</li></ul> |
| 5 | NIT | Packing replacement only cycles the highest-cost slot; the 20-target queue is unstated | **Addressed** (§2.6). <ul><li>Slot cycling is stated explicitly, with a warning against walking down the original pairs.</li><li>The 20-target queue is m11, m12, then the original queue.</li><li>Log entries carry `phase`, and Jaccard skips are logged.</li><li>The fixture test asserts the exact expected replacement log (§14).</li></ul> |
| 6 | NIT | `G4-configs` pinned twice with different contents | **Addressed** (§12.1, §12.3). `--lambda none` writes no pin and no `configs.sha256.json`. `--lambda <float>` writes the four-file `configs.sha256.json` and `G4-configs`. The preflight uses the last entry and asserts exactly the four cell files. Protocol §9/§10 updated. |
| 7 | NIT | The wiring table's "λ" is undefined under `--pilot` | **Addressed** (§4.3 check 7). The check tests `> 0` for loss-on cells and `== 0.0` for loss-off cells. λ equality stays in §4.2 step 4. |
| 8 | NIT | Selector exceptions during paused diagnostics are fatal, contradicting "diagnostics non-fatal" | **Addressed** by documenting the fatal behaviour as intentional, the review's recommended option (§4.6, §6, §15). The diagnostics draw the same anchors and epoch as training, so the exception exposes a training-time failure the batch processor would hide. All other diagnostics failures stay non-fatal. |
| 9 | NIT | Relative path `go_ppi_domain/tests/` in protocol §9 G0 | **Addressed** in protocol §9: `/Users/olukotunjosh/Downloads/CDCL/go_ppi_domain/tests/`. |

No finding was backlogged or ignored. The v3 review's unverified items are unchanged and are resolved at their gates rather than in the design: the exact pruned counts at G1, the leak gates at G1, pair and packing feasibility at G2, validation size and pilot batches at G4, and non-interference and wall time at G0/G6. The unannotated-anchor counts become measured values at G3 (§3.3).

## 20. Responses to design review v4

The v4 review confirmed all nine v3 findings resolved in the text (its "Prior findings" table) and raised the four below.

| # | sev | finding | response |
|---|---|---|---|
| 1 | MED | A `GROUP_FAILED.json` stops only the invocation that finds it; the next `sweep.py run` keeps training | **Addressed** as proposed (§12.3). <ul><li>New stopped-grid preflight: `run` exits 4 before any training subprocess or group scan if any `runs/<run_id>/GROUP_FAILED.json` exists; `plan` warns and exits 0; `status` reports `n_groups_group_failed` and exits 0 on that alone.</li><li>Defined clearing path, amendment only: fix, record the amendment in protocol §13, then `sweep.py clear-group` renames the four dirs to `<run_dir>.groupfailed<n>` (never deletes) and logs a `group_cleared` line. The group becomes incomplete and is retrained by the next `run`.</li><li>Sweep enumeration, scans and the analysis ignore `*.groupfailed*` and `*.attempt*` dirs (`RUN_DIR_IGNORE`, §1.3, §11 step 5).</li><li>On a mismatch any existing `GROUP_OK.json` is renamed to `GROUP_OK.superseded.json`, so a pass marker never coexists with a failure.</li><li>Tests: second `run --max-runs 1` exits 4 with zero runner calls; `status` exits 0 with `n_groups_group_failed = 1`; clearing and retraining; analysis ignores renamed dirs (§14). §15 rows added. Protocol §3, §7, §10, §13 updated.</li></ul> One addition beyond the review: the `clear-group` helper, so that every clearing is tied to an amendment id that exists in protocol §13 and leaves a log line. It renames only; it never runs training or checks. (v5 wording. The v5 whole-word check did not achieve this, because the example ids in protocol §13 matched it and it ignored the group. v6 replaces it with a row match, §21 row 1.) |
| 2 | NIT | An interrupt between the four `GROUP_OK.json` writes leaves a group that is never re-checked | **Addressed** as proposed (§12.3). Eligibility = four `DONE.json`, no `GROUP_FAILED.json`/`FAILED.json`, and at least one dir without a **valid** `GROUP_OK.json`. Validity is `group_ok_valid` (§1.3), the same function the analysis preflight uses (§11 step 5). Checks are recomputed from the manifests and the marker is rewritten into all four dirs, which is idempotent. Stale markers are covered by the same rule. Test: 2-of-4 dirs marked, `status` completes the other two (§14). |
| 3 | NIT | Evaluator `content_to_text` strips whitespace and the trainer's `_content_to_text` does not; the G6 cross-check could not detect a divergence | **Addressed** as proposed (§9.1 step 2, §9.2). The cross-check builds the re-encoded texts with `trainer.batch_encoder._content_to_text({"encoder_view": …}, "resume")` and compares them with the cache built from `content_to_text`. It also counts, over all cache proteins, texts whose token ids differ between the two functions, and requires 0. §9.1 step 2 reworded to "strips whitespace; token-identical to the trainer path, checked at G6". Test with a whitespace-sensitive stub tokenizer (§14); §15 row added. |
| 4 | NIT | Recorder attribute `unannotated_calls_by_epoch` vs manifest key `unannotated_anchor_calls_by_epoch` | **Addressed** with both of the review's options: the attribute is renamed to `unannotated_anchor_calls_by_epoch` (§4.6), and §4.7 states the source of every `negative_selection` key, including `unannotated_anchor_window` = the selector attribute. Test added (§14). |

No finding was backlogged or ignored. The review's deferred items are unchanged and still resolve at their gates: pruned and unannotated-anchor counts (G1/G3), the leak gates (G1), pair and packing feasibility (G2), target-free validation size and pilot batches (G4), diagnostics non-interference and wall time (G0/G6).

## 21. Responses to design review v5

The v5 review confirmed all four v4 findings resolved in the text and raised the five below. Its prior-findings table also confirms that the nine v3 findings (§19) remain resolved.

| # | sev | finding | response |
|---|---|---|---|
| 1 | MED | The `clear-group` amendment check is satisfied by the example ids printed in protocol §13 and is not tied to the group | **Addressed** as proposed (§1.3, §12.3). <ul><li>Protocol §13 amendments are now a markdown table `id \| date \| gate \| groups cleared \| cause \| fix \| re-pin`, with no rows and no literal example ids.</li><li>`clear-group` requires: `ID` matching `^AM[1-9][0-9]*$`; exactly one row in the `## 13.`–`## 14.` span whose first cell equals `ID` (`parse_amendment_rows`); a valid `YYYY-MM-DD` date; non-empty cause and fix; and `C,s<S>` among the `;`-separated groups.</li><li>The parsed row is copied verbatim into the `group_cleared` log line as `amendment_row`.</li><li>Protocol §10 uses the placeholder `AM<n>`, which fails the regex.</li><li>Tests (a)–(c) as proposed, plus prose-only id, duplicate rows, out-of-section rows and bad formats (§14). §15 row extended. The §20 row-1 claim is annotated.</li><li>One addition beyond the review: a row clears a given group once. A second `clear-group` for the same `(amendment, cond, seed)` refuses, so a group that fails again after retraining needs a new amendment.</li></ul> |
| 2 | NIT | An interrupted `GROUP_FAILED.json` write deadlocks the grid | **Addressed** as proposed (§12.3). `clear-group` needs **at least one** existing dir of the group to carry `GROUP_FAILED.json`, and renames **every existing** dir (1–4). Test: 2-of-4 marked, all four renamed; 3 existing dirs, 3 renamed; none marked, refused (§14). |
| 3 | NIT | A `DONE.json`/`lc_manifest.json` sha mismatch makes a group eligible forever | **Addressed** as proposed (§12.3 "Manifest-drift pre-check"). Before the group checks, any dir whose `sha256(lc_manifest.json)` differs from `DONE.json["lc_manifest_sha256"]` makes the scan write `GROUP_FAILED.json` (`lc_manifest_changed_after_done: <run_id>`), supersede any `GROUP_OK.json`, and exit 4. The group then follows the normal amendment and clearing path. `check-group` applies the pre-check only to dirs with `DONE.json`. Test and §15 row added. |
| 4 | NIT | The `FAILED.json` → `FAILED.repaired.json` repair has no command or checks | **Addressed** with option (a) (§12.3 `rescore`). The checkpoint is already kept for this purpose, and an evaluator bug should not force retraining. <ul><li>`sweep.py rescore --run-id` requires an eval-stage `FAILED.json`, ok post-training checks, `epochs == 15`, an existing checkpoint, no `DONE.json`, the pin preflight and the stopped-grid preflight.</li><li>It renames `eval/` to `eval.attempt<n>` and runs `score`.</li><li>On exit 0 it writes `DONE.json`, renames `FAILED.json` to `FAILED.repaired.json` and logs `eval_repaired`. On failure it keeps `FAILED.json` and exits 3.</li><li>Train-stage failures are permanently missing.</li><li>The analysis treats `FAILED.repaired.json` as not failed (§11 step 5). Tests and §15 rows added. Protocol §7 and §10 updated.</li></ul> |
| 5 | NIT | Broken code fence in protocol §10 | **Addressed.** The closing fence is on its own line, followed by the prose. |

No finding was backlogged or ignored. The review's deferred items are unchanged and resolve at their gates, as listed under §20. One of them is token identity of the two text functions on the pruned views, which is checked at G6 by §9.2 step 3.

## 22. Amendment AM1 (2026-10-10, before G2): one target per condition

Mirrors protocol §13 row AM1 and protocol §14 "Amendment AM1".

**Cause.** The plan item-12 scratch run of `build_target_table.py` on the pruned data exited 3: "packing infeasible at 20 targets; queue exhausted". An exact solver confirmed that this is not a search weakness. At 24 targets only 5 valid 3-packs exist and 8 are needed; none of the 41 slot-cycled 20-target sets packs as 6 × 3 + 1 × 2. Among the 24 accepted targets, 204 of 276 target pairs fail P_train disjointness, 90 the 5% neighbour-overlap cap and 1 the N_2 rule; 86 of 276 exceed the 8% pack removal cap. No parameter was changed before the amendment.

**Decision.** Adopt option (c): one target per condition, singleton packs.

**What changes (implementation).**

- `withholding_targets.DEFAULT_PARAMS`: `pack_size` 1 (the only shape); `pack_nbr_overlap_cap` and `pack_search_seed` removed; `pack_removal_cap` 0.08 kept and applied to each single target. `withholding_constants.PACK_SEARCH_SEED` removed.
- `pack_shape`, `pack_targets`, `pack_search` and `DEFAULT_RESTARTS` are removed. `singleton_packs(targets, pack_ok)` (§2.6) replaces them. `TargetTableStop` no longer carries a log.
- Removed constraints: pack-mate P_train disjointness, the N_2/ancestor/descendant exclusion between mates, the 5% neighbour-overlap cap, ≥ 1 low and ≥ 1 high per pack, the slot cycling and the 24 → 20 packing fallback. Kept per target: k = 0 removal ≤ 8% and ≥ 25 retention-eligible proteins with `W = T_train(t)`.
- The §2.4 size rule is unchanged (≥ 12 pairs → 24 targets; 10–11 → exactly 20, `"shortfall": true`; < 10 → exit 3). `replacement_queue` is the queue left after the size rule.
- Table schema 2 is kept. `packs[*]` drops `max_nbr_overlap`; the top-level `pack_search_log` is removed. Pack ids `p01…` follow the target order (pair number, low before high), so `targets[i]["pack_id"] == f"p{i+1:02d}"`.
- `withholding.py` and `make_conditions.py` are unchanged in logic: they are generic over the table's packs and read `n_conditions` from the table.

**Counts.** `n_packs = n_targets`. 24 targets → `n_conditions` = 24 × 5 + 1 = 121 (one shared `all`), `n_runs` = 2,420; 20 targets → 101 / 2,020. The bootstrap resamples `n_packs` = 24 (or 20) single-target packs. The completeness denominator `n_targets × 6 × 5` counts the six k levels (five numeric plus `all`), not pack size, so it stays 720 (600 at 20). Collateral queries (§9.2) exclude one target's proteins per pack run. Conditions need ≈ 4.4 GB and the 2,420 runs ≈ 7.7 GB of disk.

**Consequence.** S is uncontaminated by pack-mates: the S reduction by mates is 0 by construction. On the scratch data the 24 targets are 12 low-S / 12 high-S with SMD log n −0.048 and log support −0.036.
