# GO protein-function prediction under per-term label scarcity: technical design

Version 4 (round-3 `design_review_go_function.md`: closed-world W2/W1, protocol D19, and the 4 NITs; round-1 to round-3 findings are answered in §18). Authoritative spec: `findings_go_function.md` (cited "spec §n"). Preregistered protocol: `protocol_go_function.md` (cited "protocol §n"). Where the two documents differ on a criterion, threshold or contrast, the protocol wins; this document owns the code, files, interfaces and error handling.

`$R` = `/Users/olukotunjosh/Downloads/CDCL` locally. Every path written into a config, manifest, pin or run spec is repo-relative, so the same files work on the remote L40S host.

Evidence tags as in the spec: [read], [ran], [inferred]. Facts marked [ran] here were checked during this design step against the pinned files in `dataset/go_function/dgz/`.

## 0. Overview

The study trains one bilinear multi-label model, s(p,t) = σ(⟨f(x_p), g(t)⟩ + b_t), on DeepGOZero's (DGZ) released SwissProt 2021_04 split, with BCE over DGZ's term vocabulary. Ontology factors are switches on the term tower g and on the axiom loss, so every arm shares one code path, one split, one feature matrix, one vocabulary and one evaluator. Scarcity is induced per target term t by label withholding (W2 primary, W1 secondary) at retention k ∈ {0, 1, 5, 10, 25, all}. Six targets are withheld per run (a "pack"), and class-centric AUC on t over sequence-disjoint test proteins is the primary metric. A DGZ reproduction gate comes first: our evaluator must reproduce DGZ's published numbers from their released predictions, and our trainer must reproduce their MLP and DGZ models within ±0.01 before any study run.

All new code lives in `go_function/` (library and tests), `scripts/go_function/` (CLIs) and `config/go_function/` (frozen configs). Data goes to `dataset/go_function/` (raw, pinned) and `preprocess/go_function/` (derived), and results to `results/go_function/`. Nothing in `go_ppi_domain/`, `scripts/go_withholding/`, `contrastive_learning/`, `trials_domain/` or any other code-pinned study is modified. Small pure helpers from the PPI study are vendored (copied with attribution and a parity test), not imported, so the code pin of this study covers only its own three directories (§13).

### 0.1 Technology stack (locked when this design is approved)

| layer | choice |
|---|---|
| language | Python 3.13 (local `.venv` is 3.13.5) |
| tensors / training | PyTorch 2.8.0 (local CPU/MPS for tests and smoke; remote CUDA 12.8 wheel on the L40S) |
| arrays / tables | numpy 2.3.2, pandas 2.3.2 (only to read DGZ pickles), scipy 1.16.1 (sparse matrices, `spearmanr`) |
| metrics | scikit-learn 1.7.1 `roc_auc_score`, `average_precision_score` for per-term metrics; Fmax/Smin/micro-AUPR ported from DGZ in torch/numpy |
| protein features | HF `transformers` 4.55.4 `EsmModel` with `facebook/esm2_t33_650M_UR50D` at a revision pinned at download (fair-esm is not installed and is not added); InterPro binary vectors from the DGZ pickles |
| term text | `sentence-transformers` 5.1.0, `all-mpnet-base-v2` at revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` (already cached locally and used by the earlier studies) |
| tests | pytest 8.4.2 |
| new dependencies | none. The remote host installs the same versions as `scripts/trials_fusion_confirm/requirements-remote.txt`; a copy is written to `scripts/go_function/requirements-remote.txt` |

DGZ's own stack (py3.9, torch 1.9, dgl 0.7.1, JPype) is not installed anywhere. Their code is read as a reference and ported; their checkpoints are loaded into our port (§6.6).

### 0.2 Facts about the pinned data that shape the design [ran]

- DGZ `go.obo` is `releases/2021-11-16`: 47,235 `[Term]` stanzas, 23,524 `intersection_of` lines; `relationship:` counts part_of 8,126, regulates 3,552, positively_regulates 3,062, negatively_regulates 3,089, has_part 803, occurs_in 192, happens_during 13, ends_during 1.
- `go.norm`: 124,196 lines. NF1 80,942; NF2 11,841; NF3 11,810; NF4 19,594; 9 lines match no form. Relation IRIs in NF3/NF4: BFO_0000050 part_of 10,470; RO_0002211 regulates 7,178; RO_0002212 negatively_regulates 6,211; RO_0002213 positively_regulates 6,180; BFO_0000051 has_part 889; BFO_0000066 occurs_in 449; RO_0002092 happens_during 26; RO_0002093 ends_during 1. Every class token is a GO id except the literal `Nothing` (owl:Nothing), which appears in exactly 30 lines, all NF2 disjointness axioms of the form `GO_A and GO_B SubClassOf Nothing`. The 9 unparsed lines are role axioms (transitivity, and `RO_0002212`/`RO_0002213 SubPropertyOf RO_0002211`).
- DGZ pickles: BP train/valid/test 47,733 / 5,552 / 5,444 proteins; MF 34,716 / 3,851 / 4,712. Columns: `proteins, accessions, genes, sequences, annotations, string_ids, orgs, interpros, exp_annotations, prop_annotations, cafa_target, zero_annotations`. InterPro vocabulary 26,406 (identical file for all aspects).
- Vocabulary `terms.pkl`: BP 21,381 terms, all `biological_process`, root not included, 25 are alt ids; MF 6,868, all `molecular_function`, root not included, 17 alt ids. `terms_zero.pkl` is `terms.pkl` minus the withheld classes (7 BP, 5 MF).
- **DGZ's propagation rule reproduces their `prop_annotations` exactly**: running DGZ's `Ontology(go.obo, with_rels=True).get_prop_terms(exp_annotations)` on 3,000 random train proteins per aspect matched `prop_annotations` for 3,000/3,000 (BP and MF).
- `exp_annotations` equals the set of GO ids in `annotations` whose evidence code is in DGZ's `EXP_CODES`, for 47,733/47,733 BP and 34,716/34,716 MF train proteins. So the IPI-excluded MF labels can be built from `annotations` (§2.4). `exp_annotations` contains all three namespaces and must be filtered by aspect.
- GO:0005515 (protein binding) never occurs in MF `exp_annotations` or `annotations` of the pinned release (0 train, 0 valid, 0 test; re-checked 2026-10-12). It reaches 12,355 train / 1,577 test proteins only through propagation from its descendants, so no test protein is protein-binding-only and the §2.5 exclusion removes nobody (`n_protein_binding_only` = 0 in `labels_TP.json` and `labels_TPnoIPI.json`). The earlier "1,024 MF train proteins have only GO:0005515" did not hold for this release. The rule stays in place.
- 4,896 BP and 3,373 MF train sequences are longer than 1,022 residues.
- In the train pickles, every protein carrying a withheld zero-shot class in `prop_annotations` also carries it in `zero_annotations` (6 of 7 BP classes, e.g. GO:0000381 90/90, GO:0032755 106/106; the exception is GO:0051897, 0/124; all 5 MF classes). BP `zero_annotations` holds 3,499,881 labels vs 3,502,405 in `prop_annotations`. So DGZ's released zero-shot labels are essentially **not re-propagated**: the withheld class was dropped from the vocabulary, but its positives keep every implied label. That includes three channels: the descendants of t, the true-path ancestors of t, and the DGZ-only fillers X reached through non-is_a relations such as `positively_regulates`. This protocol channel is what the study's DGZ-protocol arm must reproduce (§5.2 WDGZrel). Only the filler channel is the typed-relation (conflation) leak. It is measured separately (§7.3, protocol §1, C3_leak_X). Which column and vocabulary trained `deepgozero_zero.th` is still unverified (spec (vi)); G2b checks it.
- **Alt ids differ between the regimes** (checked in v3 on every train protein). DGZ's closure of an alt id c′ of term d contains c′ and the ancestors of d's parents, but not d itself. TP maps c′ to d. Column-level differences:
  - cols_TP(p) \ cols_DGZ(p) (column sets, §2.4) contains only (i) alt-id columns and (ii) primaries of p's alt-id direct annotations. This holds for every train protein: 0 BP and 0 MF exceptions.
  - Channel (ii) gives 778 BP and 59 MF primary-column labels that are TP-only. The reviewer's examples include GO:0009410 (409 proteins) and GO:0071466 (150).
  - Among candidates with 30 ≤ n_train ≤ 5% N, about 14 BP and 5 MF terms (before the n_test filter) have an alt-id column or an alt-id direct annotation. All of them have an alt-id column. The §4.1 base filter excludes them.
- **Cross-namespace edges.** 7,802 BP train proteins (9,492 annotations) carry a non-BP direct annotation with a DGZ-rule edge into BP; for MF the figures are 3,286 proteins and 4,214 annotations (round-2 review [ran]). Under ρ = DGZ, erasure must therefore operate on unfiltered `exp_annotations` (§2.2, §5.2).
- The DGZ release directory lists, per aspect, `deepgozero.th`, `mlp.th`, `deepgozero_zero.th`, `deepgozero_zero_10.th`, `predictions_{mlp,deepgozero,deepgozero_zero}.pkl` (MF also `predictions_deepgozero_zero_10.pkl`), `zero_gos.pkl`, `zero_term_auc{,_all,_train}.pkl` and `zero_<GO>_auc{,_all,_train}.pkl`. No 30%-identity split file is listed in `data/` (§17).

## 1. Typed ontology module (gap 2) — `go_function/ontology.py`

Pure Python, no torch. `go_ppi_domain/go_ontology.py` is not touched and not imported.

### 1.1 Parser

```python
@dataclass(frozen=True)
class Term:
    id: str; name: str; namespace: str; definition: str     # definition: text inside the first "..." of def:
    is_obsolete: bool; alt_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]                       # (rel, parent_id); rel in {"is_a", <relationship type>}
    intersection: tuple[tuple[str | None, str], ...]         # (rel or None for the genus, class id)

@dataclass(frozen=True)
class Ontology:
    terms: Mapping[str, Term]          # live terms only, keyed by primary id
    alt_to_primary: Mapping[str, str]
    obsolete: frozenset[str]
    def primary(self, go_id) -> str | None   # alt -> primary; obsolete or unknown -> None

def parse_obo(path: Path) -> Ontology
```

- Reads `[Term]` stanzas only; `[Typedef]` stanzas end the current term and are skipped.
- `is_a: X ! name` → ("is_a", X). `relationship: REL X ! name` → (REL, X) for every REL, kept typed.
- `intersection_of: X` → (None, X); `intersection_of: REL X` → (REL, X).
- Obsolete terms are excluded from `terms`; their ids are in `obsolete`. Alt ids map to the primary.
- Failure: a stanza without `id` or `namespace`, or an edge to an id that is neither live, alt nor obsolete → `OntologyError` naming the line number (fatal, exit 2 in every CLI). Edges to obsolete parents are dropped and counted in `parse_report` (returned alongside, written into every manifest that uses the ontology).

### 1.2 Closures and neighbourhoods

```python
TRUE_PATH = ("is_a", "part_of")
def ancestors(ont, rels=TRUE_PATH, same_namespace=True) -> dict[str, frozenset[str]]   # reflexive
def descendants(anc) -> dict[str, frozenset[str]]                                       # inverse, reflexive
def parents(ont, rels=TRUE_PATH, same_namespace=True) -> dict[str, frozenset[str]]
def isa_siblings(ont, t, anc, desc) -> frozenset[str]   # share an is_a parent; minus t, anc(t), desc(t)
def two_hop(ont, t, anc, desc, rels=TRUE_PATH) -> frozenset[str]   # <= 2 undirected hops; same exclusion
def logical_definition(ont, t) -> tuple[str | None, tuple[tuple[str, str], ...]]  # (genus, differentia)
```

- Closures are computed once by memoised DFS over the chosen relation types, restricted to parents in the same namespace when `same_namespace=True`. A cycle raises `OntologyError` (fatal; GO is acyclic over is_a+part_of, so a cycle means a parser bug).
- `regulates`, `positively_regulates`, `negatively_regulates`, `has_part`, `occurs_in`, `happens_during`, `ends_during` are never in `TRUE_PATH`. Per-relation closures (`rels=("is_a","regulates")`, etc.) exist for diagnostics only.

### 1.3 DGZ-rule replica

```python
class DgzRule:   # byte-for-byte port of DGZ utils.Ontology(with_rels=True) semantics
    def __init__(self, obo_path: Path)
    def closure(self, go_id) -> frozenset[str]      # get_anchestors
    def prop(self, go_ids) -> frozenset[str]         # get_prop_terms
```

Port rules, kept exactly as DGZ (§0.2 shows they reproduce `prop_annotations`): every `relationship:` target is appended to `is_a` regardless of type and namespace; alt ids are aliases to the same term object. A closure started from an alt id c′ of d therefore contains the string c′ and every ancestor of d's parents, but **not d** (the source of the TP/DGZ alt-id asymmetry, §0.2). Obsolete terms are deleted after alias registration; ids not in the dict give an empty closure. This class is used only to build the DGZ-rule label regime and for the DGZ reproduction gates.

## 2. Labels (gap 3) — `go_function/labels.py`, `scripts/go_function/build_labels.py`

### 2.1 Vocabulary

V_aspect = DGZ `terms.pkl` in file order (BP 21,381; MF 6,868), identical for every arm and both label regimes, so column j means the same term everywhere. Column ids are DGZ's strings. `primary(c)` maps alt-id columns to their primary; an alt-id column whose primary is also a column is a "duplicate column": trained like any other (DGZ comparability), never a target, and excluded from every evaluation term set. Counts are in the label manifest.

*Decision (protocol D16).* The spec ((i)) defines the vocabulary as terms with ≥ 1 train annotation after relabelling. Using DGZ's `terms.pkl` unchanged instead keeps one column set across the TP and DGZ regimes, which the regime and leak contrasts (C3 and its components) need for pairing, and keeps DGZ checkpoints loadable. Consequence: some columns may have zero TP train positives (positives only under DGZ's rule). The label manifest reports, per regime and split, the number of columns with zero train positives; columns with zero TP train positives are excluded from the collateral term set (§8.2) and can never be targets (n_train ≥ 30 already excludes them). They stay in training (all-negative columns under TP).

### 2.2 Regimes

Each regime ρ has its own direct-annotation set A_ρ(p). Its closure, its labels, the W2 erasure (§5.2) and the tests all use that set:

```text
A_TP(p)      = exp_annotations(p) mapped by primary(), obsolete/unknown dropped (counted), filtered to the aspect namespace
A_TPnoIPI(p) = same, built from annotations(p) entries with code in EXP_CODES \ {IPI}
A_DGZ(p)     = exp_annotations(p) as stored: all namespaces, raw id strings (alt ids kept as alt ids), no mapping
```

- **TP (true path, primary).** closure_TP(p) = ∪_{a∈A_TP(p)} ancestors_TP(a), within namespace; root removed. Label y_TP[p, c] = 1 iff primary(c) ∈ closure_TP(p).
- **DGZ (control).** closure_DGZ(p) = `DgzRule.prop(A_DGZ(p))`, a set of raw id strings. y_DGZ[p, c] = 1 iff c ∈ closure_DGZ(p) (string match on DGZ's column id). This reproduces `prop_annotations` restricted to V (asserted on every protein at G1).
- **TP-noIPI (MF sensitivity).** As TP, with A_TPnoIPI.

`direct_<split>` in the label files (§2.3) stores A_ρ for its regime: column indices where an annotation is a column, plus a per-row list of non-column ids (for DGZ, raw strings of every namespace). The withholding generator reads only this stored A_ρ.

Evaluation labels (protocol §3):

- Every block except MF noIPI evaluates on **TP test labels** and the §2.5 evaluation set. This includes the DGZ-regime runs (C3 and its components): training labels change, test labels do not.
- The **MF noIPI block trains and evaluates on TPnoIPI labels.** Its evaluation set is the MF test proteins with a non-empty TPnoIPI MF closure, minus the protein-binding-only proteins under TPnoIPI labels (§2.5). Its n_train and n_test for the target drop rule (protocol §6.2) are computed on TPnoIPI train labels and that evaluation set.
- DGZ-rule test labels are computed for the descriptive sensitivity read and for the reproduction gates (G2, G2b L2, G4: DGZ-faithful mode, §8.3).

The evaluator selects the test-label file from the run spec's `regime`: `labels_TPnoIPI.npz` if and only if `regime = TPnoIPI`, otherwise `labels_TP.npz`; DGZ-rule test labels are loaded only by the gate scripts and by the descriptive sensitivity read (invariant table, §15).

### 2.3 Storage

`preprocess/go_function/<aspect>/labels_<regime>.npz`: scipy CSR uint8 matrices `train`, `valid`, `test` (rows in pickle order), plus `direct_<split>` CSR (the regime's A_ρ(p) restricted to columns) and `direct_extra_<split>` (object array, one tuple per row, of the A_ρ(p) ids that are not columns; for DGZ these are raw strings of any namespace), both used by the withholding generator, `proteins_<split>` (DGZ protein names), `vocab`. A JSON manifest beside it records input sha256s, row/column counts, per-split label counts, obsolete and alt-id counts, the duplicate columns and the sha256 of each matrix (`table_sha256` over `indptr`, `indices`, `data`, shape).

Direct annotations of a protein that map to no column are still used for closure; only columns are restricted to V.

### 2.4 Checks (G1; any failure is fatal, exit 2)

1. y_DGZ equals `prop_annotations ∩ V` for every protein of every split (BP, MF).
2. `exp_annotations` equals the EXP-coded set of `annotations` for every protein.
3. TP invariants on every protein:
   - (a) for every positive column c and every TP parent q of primary(c) that is a column, y[p, q] = 1 (descendants imply ancestors);
   - (b) an independent closure, `labels._bfs_closure_tp`, written separately from `ontology.ancestors` (iterative BFS over the raw `Term.edges` filtered to `is_a` and same-namespace `part_of`, no memoisation, no shared helper), gives exactly closure_TP(p) for every protein;
   - (c) no TP label is reachable only through a non-TP relation. The check is at **column level**, through `primary`, never on raw strings. Define
     - cols_DGZ(p) = {c ∈ V : c ∈ closure_DGZ(p)};
     - cols_BFS(p) = {c ∈ V : primary(c) ∈ BFS-closure(p)}.

     The check is that the stored matrix row y_TP[p] has no positive in cols_DGZ(p) \ cols_BFS(p). So none of the extra labels DGZ's rule adds (through the regulates family, has_part, occurs_in, cross-namespace part_of, ...) is positive under TP. The count Σ_p |cols_DGZ(p) \ cols_BFS(p)| is reported; it equals the DGZ excess of check 5.
   - (d) the regimes differ in the TP direction only through alt ids. For every protein, cols_BFS(p) \ cols_DGZ(p) ⊆ AltCols ∪ PrimAlt(p), where:
     - AltCols = the alt-id columns of V;
     - PrimAlt(p) = {primary(a) : a ∈ A_DGZ(p) is an alt id}.

     Verified at design time on every train protein (0 exceptions, §0.2). The TP-only label count of each channel is reported (v3: BP 778 and MF 59 in PrimAlt on train).
4. Unit-level checks in tests (§14): regulates is never propagated; a part_of edge across namespaces is not propagated; alt and obsolete handling.
5. Counts reported, not gated: total labels per regime and the DGZ excess over TP (spec reports +66% BP, +31% MF on current GO; with the pinned 2021 GO the residual of spec (vi) is resolved by this number).

### 2.5 MF evaluation set

MF test proteins whose TP MF closure is exactly {GO:0005515 and its ancestors} ("protein-binding-only") are excluded from every MF evaluation (class-centric and protein-centric); they stay in training. Count in the manifest. The TPnoIPI evaluation set (MF noIPI block only, §2.2) is defined on TPnoIPI test labels: proteins with a non-empty TPnoIPI MF closure that is not protein-binding-only. Its size is in the TPnoIPI manifest.

## 3. Features (gap 5) — `go_function/features.py`, `scripts/go_function/build_features.py`, `scripts/go_function/embed_esm.py`, `scripts/go_function/embed_terms.py`

### 3.1 InterPro (secondary; all gates and local smoke)

Binary CSR matrix over the 26,406-entry `interpros.pkl` vocabulary, rows in pickle order per split, `preprocess/go_function/<aspect>/features_interpro.npz`, sha256 in its manifest. Unknown InterPro ids in a row are dropped and counted. Proteins with no InterPro entry get an all-zero row (2.5% BP, 1.5% MF train per spec), as in DGZ.

### 3.2 ESM-2 650M (primary; remote)

- Model: `EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", revision=<pinned sha>)`, fp16 on CUDA, `eval()`, `torch.no_grad()`.
- Input: the `sequences` column of the pickles (union over BP and MF, deduplicated by protein name; the same protein must have the same sequence in every pickle, else fatal). Sequences longer than 1,022 residues are truncated to the first 1,022 (count recorded). Non-standard residues are passed to the ESM tokenizer unchanged (it maps them to X/unk).
- Output: mean over residue positions of the last hidden layer, excluding BOS/EOS and padding; 1,280-d; stored float32 in `preprocess/go_function/esm2_650m/embeddings.npy` with `proteins.txt` (row order) and a manifest (revision, dtype, truncation count, sha256, wall time).
- Batching: sorted by length, token budget 16,384 tokens per batch.
- Cross-check (fatal if failed): 64 fixed proteins (first 64 by name hash) are recomputed in fp32 batch size 1; cosine with the stored fp16 vectors must be ≥ 0.999 for all.
- Per-aspect feature matrices are views built at run start by protein-name lookup; a missing protein is fatal.

### 3.3 Term text (V5, V6)

For every GO class that can appear in a term tower (V plus every GO class in `go.norm`), the string `"<name>. <definition>"` from the pinned 2021 `go.obo` (definition = text in the first quoted span of `def:`, references dropped). Encoded by frozen all-mpnet-base-v2 (pinned revision, `normalize_embeddings=True`, batch 256, CPU or GPU), stored float32 at `preprocess/go_function/term_text_mpnet.npy` with `term_ids.txt` and manifest. Text is required for GO classes only: the reserved class `owl:Nothing` (§6.4) has no text and is skipped by `embed_terms.py`. A **GO** class with no live term in `go.obo` is fatal (exit 2); any non-GO token other than `owl:Nothing` already fails in `normforms.py`. Text is used only on the term side; no protein text is read anywhere in this study (spec §2).
- *Obsolete classes (protocol A3).* go.norm names 728 GO classes that are obsolete in the pinned go.obo. Each takes its own obsolete stanza's text: the name with one leading "obsolete " removed, then the definition as written (`features.obsolete_term_text`; alt ids of an obsolete stanza take the same text). No free vector is used. A class with no stanza or an empty text is fatal (exit 2).

## 4. Targets and packs (gap 7) — `go_function/targets.py`, `scripts/go_function/build_targets.py`

Inputs: TP labels, the ontology, the vocabulary. Rules (fixed by protocol §4; this section gives the algorithm).

1. **Base filter** per aspect: t is a primary (non-duplicate) column, not the root, not GO:0005515 or GO:0005488; n_train(t) = |P_train(t)| ≥ 30; n_test(t) ≥ 10 (on the evaluation set of §2.5); n_train(t) ≤ 0.05 · N_train.
   - **Alt-id exclusion:** t is excluded if any vocabulary column is an alt id of t, or if any train, valid or test protein has an alt id of t in `exp_annotations`. The count excluded is reported per aspect. At design time it removes about 14 BP and 5 MF of the terms with 30 ≤ n_train ≤ 5% N_train, before the n_test filter; every one of them has an alt-id column.
   - Why: under DGZ's rule an alt-annotated protein is not a t-positive, while under TP it is (§0.2). Excluding these terms makes P_TP(t) ⊆ P_DGZ(t) hold for every target. It also keeps a duplicate column of t out of every DGZ-regime condition, and keeps WDGZrel equal to DGZ's labels.
2. **Support.** N_sib(t) = `isa_siblings ∩ V`; S_sib(t) = |∪_{u∈N_sib(t)} P_train(u) \ P_train(t)|.
3. **Logical definition (BP only).** LD = yes if t has an `intersection_of` genus and at least one differentia filler F that is a GO class with n_train(F) ≥ 30 under TP labels; LD = no if t has no `intersection_of`; otherwise t is excluded. Here n_train(F) = |{p ∈ train : F ∈ closure_TP(p)}|, computed from the TP closure whether or not F ∈ V (a filler of another namespace is in no TP closure of this aspect, so its n_train is 0).
4. **Strata.** Tertile cuts of S_sib over the base-filtered candidates (after step 3 for BP): low = S_sib ≤ 33.33rd percentile, high = S_sib ≥ 66.67th; the middle is excluded. Cells: BP {low, high} × {LD yes, LD no} (4 cells × 12); MF {low, high} (2 cells × 24).
5. **n bands.** B1 = 30–99, B2 = 100–249, B3 = 250+ train proteins. Quota per cell per band: BP 4, MF 8.
6. **Greedy acceptance.** Slots are visited round-robin in the fixed order (cell order as listed, then B1, B2, B3), one acceptance per visit, until all quotas are full. Within a slot, candidates are ordered by (−n_test, GO id). A candidate is accepted if, against every accepted target u: neither is an ancestor of the other (TP), and Jaccard(P_train(t), P_train(u)) ≤ 0.2. This is a new function, `accept_round_robin(slots, quotas, P_train, ancestors, max_jaccard) -> list[str]`, in `targets.py`, with its own unit tests (§14). Only `jaccard` is vendored from `go_ppi_domain/withholding_targets.py` (§13.2); the PPI `greedy_non_nested` has no slots, quotas or round-robin and is not used.
7. **Shortfall.** If a slot is exhausted, its quota moves to the next band of the same cell (B1→B2→B3→B1). If a cell still cannot reach its total (12 BP / 24 MF), the script exits 3 with a count-only message and the study stops for a user decision (protocol §4).
8. **Balance report.** Standardised mean differences of log n_train and log n_test between high and low (and BP: LD yes vs no). Reported, not gated; band quotas make the n-match by construction.
9. **Packing** (placement order amended 2026-10-11, protocol §14 A1). 8 packs × 6 targets. Targets are processed by decreasing n_train(t), ties broken by GO id; candidate packs are sorted by (number of targets of the same cell already in the pack, |∪ P_TP over the pack| after adding t, pack id); the first pack satisfying all constraints is used: size < 6; t ∉ two_hop(u) and u ∉ two_hop(t) for every pack-mate u; D*^GO_TP(t) ∩ D*^GO_TP(u) = ∅ and D*^GO_DGZ(t) ∩ D*^GO_DGZ(u) = ∅, using the term-level sets of §5.1 (all GO terms, not only columns, so a non-column direct annotation that implies two pack-mates is caught); |∪ P_TP over the pack| ≤ 0.10 · N_train (the DGZ-regime union ∪ P_DGZ is reported per pack, not gated). If no pack fits, exit 3 (study stop). *Amendment A1 (2026-10-11, before any G6 pin or result):* the v3 rule (acceptance order; packs by (same-cell count, pack size, pack id)) placed only 44/48 BP and 47/48 MF targets, although a feasible packing exists under the unchanged constraints. Only the placement order changed: the selected 48-target set, the quotas and every constraint above are unchanged. `build_targets.py` asserts this and records the sha256 of the sorted 48-target set before and after packing (they must be equal). The rule uses train counts only, never test outcomes, and is deterministic.
10. **Retention order.** For each target, the list P_train(t) sorted by protein name, permuted by `numpy.random.default_rng(seed_t)` with seed_t = int(sha256(f"{RETENTION_SEED}:{t}")[:16], 16). R_k = first k. Nested by construction; shared by all arms, regimes and seeds. A second order (draw 2) uses `RETENTION_SEED_2 = 20261012` the same way; it is used only by the descriptive retention re-draw block (protocol §6.2) at k ∈ {1, 5}. For the noIPI block the permutation is applied to P_train(t) under TPnoIPI labels (same seed formula), so R_k ⊆ P_TPnoIPI(t). Both orders are stored in the target table.

Output `preprocess/go_function/<aspect>/targets/target_table.json` (schema 1): parameters, input hashes, cut points, each target (id, name, cell, band, n_train, n_test, S_sib, LD flag and filler, pack, retention order as protein row indices), packs, balance report, candidate counts per cell/band. Written once; a second run must produce byte-identical JSON or exit 2. Target names are not read by any human before the protocol pins this hash (G6).

## 5. Withholding generator (gap 8) — `go_function/withholding.py`, `scripts/go_function/make_conditions.py`

### 5.1 Definitions (per regime ρ ∈ {TP, DGZ, TPnoIPI})

TPnoIPI uses the TP closure on the IPI-excluded direct annotations (§2.2); everything below that is stated for TP holds for it, with its own P(t) and retention order.

- closure_ρ as in §2.2.
- **Term-level** set D*^GO_ρ(t): the set the erasure tests against (a direct annotation a ∈ A_ρ(p) is erased iff a ∈ D*^GO_ρ(t)), and the set used for the pack-disjointness constraint (§4.9). Defined per regime on the regime's id space:
  - TP, TPnoIPI: D*^GO_TP(t) = {live primary GO ids a, aspect namespace : t ∈ ancestors_TP(a)}, including t.
  - DGZ: D*^GO_DGZ(t) = {raw id strings a appearing in any A_DGZ(p) of any split, or as an `id`/`alt_id` in go.obo : t ∈ DgzRule.closure(a)}, including t. This covers alt ids of descendants of t, whose DGZ closure contains t through the descendant's parents. It also covers terms of **other namespaces** (e.g. an MF term `part_of` a BP target). Pack disjointness under DGZ is tested on these raw-string sets.
- **Column-level** set D*_ρ(t): the set used for the assertions and the target-free validation mask (no train mask uses it; §5.2).
  - TP: {c ∈ V : primary(c) ∈ D*^GO_TP(t)}.
  - DGZ: {c ∈ V : c ∈ D*^GO_DGZ(t)} ∪ T_cols(t), where T_cols(t) = {c ∈ V : primary(c) = t}. After the §4.1 alt-id exclusion, T_cols(t) = {t} for every target; the union is kept as a defence.
- P_ρ(t) = train proteins with y_ρ[p, t] = 1. R_k ⊆ P_TP(t) ⊆ P_DGZ(t) is asserted for every target. This holds **for targets with no alt-id column and no alt-id annotation** (the §4.1 exclusion), because then the only TP-not-DGZ labels (§2.4 check 3d) never touch column t. It is not a general "TP closure ⊆ DGZ closure" property. For TPnoIPI, R_k ⊆ P_TPnoIPI(t) (own order, §4.10).
- W_ρ(t, k) = P_ρ(t) \ R_k (k numeric); W = ∅ at k = all.

### 5.2 Rules

For a pack (t₁…t₆), retention k and rule:

W2 and W1 are **closed-world** (round-3 review, protocol D19). The rule text, as adopted:

```text
W2 (closed-world). For each pack target t and p ∈ W_ρ(t):
  A_ρ(p) ← A_ρ(p) \ D*^GO_ρ(t); labels ← closure_ρ(A_ρ(p)) ∩ V
  No mask is added for p. Erased entries (D*_ρ(t) and Lost(p)) are ordinary 0 labels in the BCE,
  exactly as for a protein never annotated in t's subtree.
W1 (closed-world). As W2, then y[p, c] = 1 for c ∈ (ancestors_ρ(t) \ {t}) ∩ V. No mask.
WDGZrel: unchanged (T_cols(t) masked on every train row, which is information-free).
W2drop: W2 (closed-world) labels + T_cols(t) masked on every train row.
```

- **W2 (primary, closed-world).** For each target and each p ∈ W_ρ(t): A_ρ(p) ← A_ρ(p) \ D*^GO_ρ(t). This covers columns and non-column ids alike (the test is t ∈ closure_ρ(a)). Under ρ = DGZ it also covers raw alt-id strings and annotations of other namespaces; for example, an MF annotation `part_of` a BP target is erased. Re-propagate: labels ← closure_ρ(A_ρ(p)) ∩ V. **No mask.** Lost(p) = columns positive before and negative after re-propagation (ancestors that were implied only through the erased annotations) is counted in the manifest but not masked: Lost(p) and D*_ρ(t) entries are ordinary 0 labels, exactly as for a protein never annotated in t's subtree. *Why:* a row-selective mask on target-related columns marks exactly W = P(t) \ R_k, so the mask pattern is the withheld label, and k = 0 would not be zero-shot (round-3 Finding 1). In real annotation scarcity the unannotated true positives are unknown and closed-world BCE (DGZ's own loss) trains on them as negatives.
- **W1 (secondary, closed-world).** As W2, then set y[p, c] = 1 for every c ∈ (ancestors_ρ(t) \ {t}) ∩ V (strict ancestors of t). **No mask.** W1 − W2 reads "parent-level annotation vs no annotation in the branch".
- **WDGZrel ("DGZ as released"; DGZ regime, k = 0 only).** Reproduces the labels DGZ's zero model saw (§0.2): train labels = the base DGZ matrix y_DGZ (= `prop_annotations ∩ V`, asserted at G1) **unchanged**; no erasure, no re-propagation. Mask = the columns T_cols(t) = {c : primary(c) = t} of every pack target for **every** train row. This is equivalent to dropping t from the vocabulary, as DGZ did. After the §4.1 exclusion T_cols(t) = {t}; the set form is a defence against a duplicate column carrying t. The filler X of a "regulation of X" target keeps all its t-derived positives, and descendants of t keep their labels: the propagation channel X→t of spec §4 is intact. *Decision:* the base matrix is `prop_annotations ∩ V`, not `zero_annotations`; `zero_annotations` exists only for DGZ's own withheld classes and differs from `prop_annotations` by about 2.5k BP labels (where those differences fall is unverified). `make_conditions.py` reports, per target column, the number of train proteins where the two columns disagree (descriptive, not gated). G2b's finding on which column trained `deepgozero_zero.th` is recorded but does not change this rule.
- **W2drop (descriptive; DGZ regime, k = 0 only).** The closed-world W2 condition of the same pack under ρ = DGZ at k = 0 (labels identical, asserted by hash), with the mask {(p, c) : p ∈ train, c ∈ T_cols(t)} of every pack target (W2 itself has no mask). It differs from WDGZrel only in the erasure + re-propagation (both drop column t from the BCE), and from W2 only in dropping column t (W2 trains column t on negatives only at k = 0). It splits C3 into its parts (§12.2, protocol §1).
- *Dropped:* the v1 re-propagating "W_DGZ" (remove t's direct annotations, re-propagate with DgzRule, mask t). It was neither DGZ's released protocol nor leak-free (review finding 1) and is not built.
- Retained proteins R_k keep all labels. Proteins not in P_ρ(t) are untouched (negatives for t at every k).
- When a protein is withheld for two pack targets, both erasures are applied before one re-propagation (and, in W1, both targets' strict ancestors are set after it). WDGZrel/W2drop masks are the union of the targets' column masks.
- The only masks anywhere in training are the WDGZrel/W2drop column masks over all train rows and the target-free validation column mask (§5.3). No mask is ever row-selective on a target-related column (assertion 10).

### 5.3 Target-free validation

One per (aspect, regime): validation labels unchanged, and a column mask excluding ∪_{all 48 targets} D*_ρ(t) from every validation loss. Used only by pilots (§9) and logged per epoch in study runs. It never selects a checkpoint in study runs.

### 5.4 Files

`preprocess/go_function/<aspect>/conditions/<regime>/<rule>/<pack>_k<k>[_d2]/` (suffix `_d2` only for retention draw 2): `train_labels.npz` (CSR uint8), `train_mask.npz` (int32 pairs row, col of masked entries; zero pairs for W1/W2, and the trainer allocates no mask tensor then), `withholding_manifest.json` (target ids, |W|, |R_k|, retained row ids, mask count, hashes of labels/mask, base label hash, test label hash, validation mask hash). The `all` condition (no withholding) is one directory per regime. The generator never writes into an existing directory with different content (`_write_or_verify` pattern: identical bytes → ok; different → exit 2).

### 5.5 Generator assertions (fatal, exit 2; re-run by `run.py` preflight)

1. For every target t and p ∈ W: after re-propagation y[p, c] = 0 and (p, c) is not masked, for all c ∈ D*_ρ(t) (W1, W2, W2drop; under W2drop the columns T_cols(t) are masked on every train row by assertion 7, so for them only y = 0 is asserted).
2. |{p ∈ train : y[p, t] = 1}| = k exactly (numeric k, W1/W2/W2drop), and those proteins equal R_k of the condition's draw.
3. W1 only: every strict ancestor column of t is positive for every p ∈ W.
4. The test and validation label matrices and their hashes equal the base label manifest (withholding never touches them).
5. No protein outside ∪ P_ρ(t) changed any label.
6. WDGZrel: the train label matrix hash equals the base DGZ matrix hash. The mask is exactly {(p, c) : p ∈ train, c ∈ T_cols(t), t ∈ pack targets}: no erasure and no other masked entry. Assertions 1–3 do not apply.
7. W2drop: the train label hash equals the W2/DGZ/k = 0 condition of the same pack. The mask equals {(p, c) : p ∈ train, c ∈ T_cols(t), t ∈ pack targets} (W2 has no mask).
8. Every rule and regime, for every pack target t: no column c with primary(c) = t has an unmasked train positive outside R_k (at k = all, R_k is all of P_ρ(t)). Under WDGZrel/W2drop (k = 0 only), no column with primary(c) = t has any unmasked train entry. Also, every target has T_cols(t) = {t} (the §4.1 exclusion held).
9. Every target: R_k ⊆ P_TP(t) ⊆ P_DGZ(t) (§5.1; for TPnoIPI, R_k ⊆ P_TPnoIPI(t)).
10. Closed world: in W1/W2 (and `all`) the train mask is empty. In WDGZrel/W2drop every masked column is masked on all train rows. No mask is row-selective on a target-related column (any column of ∪_t D*_ρ(t) ∪ T_cols(t) ∪ ancestors_ρ(t)).

## 6. Model (gap 6) — `go_function/model.py`

### 6.1 Protein tower f

DGZ's encoder: `MLPBlock(d_in, 1024)` then `Residual(MLPBlock(1024, 1024))`; `MLPBlock` = Linear → ReLU → norm → Dropout(0.1). DGZ's published `deepgozero.py` writes `nn.BatchNorm(out_features)`, which does not exist in torch [read]; the released checkpoints decide the norm type. G3 loads `mlp.th` and `deepgozero.th` with `strict=True`; the norm is LayerNorm if the checkpoint keys/shapes match it, otherwise BatchNorm1d. The choice is fixed in `config/go_function/study.json` after G3 and asserted by every run. d_in = 26,406 (InterPro) or 1,280 (ESM).

### 6.2 Term towers

| arm | g(t) | b_t | axiom loss |
|---|---|---|---|
| V0 | free row of a `Linear(1024, |V|)` weight (= DGZ MLP) | Linear bias | none |
| V1, V2, V3, V3c, V4 | e_t + r_hasFunc, e_t from `Embedding(|V|+|Z|, 1024)` | \|ρ_t\| from `Embedding(·, 1)` | EL over the arm's axiom subset (§6.4) |
| V5 | P·m_t, m_t frozen MPNet text vector, P = `Linear(768, 1024, bias=False)` | free per-term bias | none |
| V6 | P·m_t + r_hasFunc | \|ρ_t\| | EL over the V4 axiom set, with ball centres go_norm(P·m_t) |

Z = classes appearing in the arm's axiom subset but not in V ("zero classes", DGZ). In V4 and V6, Z contains the reserved class `owl:Nothing` (§6.4); V1–V3 and V3c never contain it (no NF2). In V6, every GO class (columns and zero classes) gets its ball centre from P·m_c; `owl:Nothing` has no text and gets a free trainable centre (`Embedding(1, 1024)`, same uniform init as DGZ) and a free radius (`Embedding(1, 1)`), the only non-text class vector in V6. V1–V4 and V3c are DGZ's `DGELModel` structure exactly: `go_norm = BatchNorm1d(1024)` applied to ball centres inside the EL loss only, uniform init ±√(1/1024), `rel_embed` with one extra row for hasFunction, margin 0.1.

*Decision (V6 axiom set).* "Text + best typed EL" (spec (i)) is fixed to the V4 axiom set before any result, because choosing the best ladder arm on test data would be circular and no target-free criterion measures withheld-term performance. V4 is the complete typed axiom set (every relation has its own vector in DGZ's loss; the conflation is in DGZ's labels, which V6 does not use). Recorded as protocol decision D8; amendable before the freeze.

### 6.3 Score

logit(p, t) = ⟨f(x_p), g(t)⟩ + b_t; probability σ(logit). Class-centric AUC depends only on ⟨f, g(t)⟩ ordering, so b_t does not affect the primary metric.

### 6.4 go.norm loader and the axiom ladder — `go_function/normforms.py`

```python
@dataclass(frozen=True)
class NormalForms:
    nf1: np.ndarray  # (n,2) class ids
    nf2: np.ndarray  # (n,3)
    nf3: np.ndarray  # (n,3) rel, c, d    R some C SubClassOf D
    nf4: np.ndarray  # (n,3) c, rel, d    C SubClassOf R some D
    class_index: dict[str, int]; rel_index: dict[str, int]; unparsed: int
def load_normal_forms(path, terms_dict, dgz_compat=True) -> NormalForms
def select(nf: NormalForms, arm: str) -> NormalForms
```

- `dgz_compat=True` is DGZ's parser exactly (§0.2 source): `replace('_', ':')` per line; NF1 when both sides have length 10; NF2 if the left side contains " and "; NF3 if the left contains " some "; NF4 if the right contains " some "; ids assigned in order of first appearance, vocabulary columns first. This order is required to load DGZ checkpoints. `unparsed` must equal 9 (pinned at G1; a different count is fatal).
- Class tokens: after `replace('_', ':')` a token must be a GO id (`GO:` + 7 digits) or the literal `Nothing`, which is mapped to the reserved class id `owl:Nothing` and indexed in its first-appearance position like any zero class (DGZ's parser keeps it as a class, so V4 indices still match the checkpoint). `owl:Nothing` must occur in exactly 30 lines, all NF2 right-hand sides (pinned at G1; any other count or position is fatal, exit 2). Any other non-GO token is fatal.
- Relation names: BFO:0000050 part_of, BFO:0000051 has_part, BFO:0000066 occurs_in, RO:0002211 regulates, RO:0002212 negatively_regulates, RO:0002213 positively_regulates, RO:0002092 happens_during, RO:0002093 ends_during. An unknown IRI is fatal.
- Ladder (nested by construction, asserted):

| arm | NF1 | NF3/NF4 relations | NF2 | relation vectors |
|---|---|---|---|---|
| V1 | all | none | no | hasFunc only |
| V2 | all | part_of | no | part_of, hasFunc |
| V3 | all | part_of, regulates, positively_, negatively_regulates | no | 4 + hasFunc |
| V3c (untyped-relation control) | all | same as V3 | no | **one shared vector** for those 4 relations (part_of included), + hasFunc |
| V4 | all | all 8 | yes | 8 + hasFunc (= DGZ) |
| V6 | as V4 | | | |

- V3c is a shared-relation-vector control, not a reproduction of DGZ's conflation (that is in DGZ's labels, tested by C3). Its shared vector is still distinct from is_a (NF1 axioms are unchanged), and it collapses part_of together with the regulates family, so C1 = V3 − V3c mixes part_of-vs-regulates typing with typing within the regulates family. Only the relation vector is shared: V3c keeps V3's NF4 radius slots (§6.5, protocol A5). The spec's "conflation control" label is replaced by "untyped-relation control" in both documents.
- Zero classes and relation ids are re-indexed per arm in first-appearance order within the selected subset; only V4 must match DGZ's order (checkpoint loading), which holds because V4 selects everything.

### 6.5 Losses

- BCE: mean over unmasked entries of the batch, `F.binary_cross_entropy_with_logits(logit, y, reduction="none")` multiplied by the 0/1 mask, summed and divided by the mask sum. With no mask this equals DGZ's mean BCE. *Closed world (D19):* W1/W2 conditions have an empty train mask, so they take the unmasked path, and the per-row loss mask is identical to the base data's; the only masks in training are the WDGZrel/W2drop column masks over all train rows (and the target-free validation column mask on validation).
- EL: DGZ's `nf1_loss + nf2_loss + nf3_loss + nf4_loss` over all axioms of the arm, every step (DGZ computes it over the full set each batch [read]). An empty form contributes 0. Total = BCE + λ·EL with **λ = 1 fixed a priori** (DGZ's value) for every EL arm (V1–V4, V3c, V6), every regime and both features, in the study and in the gates (protocol D12). The λ sweep of §9.2 is a descriptive record and never selects λ.
- DGZ's `nf4_loss` takes the radius of `data[:, 1]` (the relation id, used as a class index) as rc [read]; this looks like a DGZ bug. *Decision:* the study keeps DGZ's formula unchanged in every EL arm, because the spec defines V4 as DGZ and the same formula applies to V2–V4, V3c and V6 alike. *V3c (protocol A5):* re-indexing V3c's four relations to one shared id would make every V3c NF4 axiom read class 0's radius, while V3 reads classes 0–3. `normforms.select` therefore carries a separate NF4 radius-index column `nf4_rad`: V3's untied relation ids in V3c, the relation ids themselves in every other arm (V4 still reproduces DGZ, asserted in `assert_ladder`). V3 and V3c read the same radius slots, so C1 differs only in relation-vector typing. A corrected variant (`nf4_radius="class"`) exists only as a G4 diagnostic: V4 full-data class AUC with and without the correction is reported (protocol D10). It is never used in the grid.

### 6.6 DGZ checkpoint compatibility

`DgzCompatModel` maps DGZ state-dict keys (`net.*`, `go_embed.weight`, `go_rad.weight`, `rel_embed.weight`, `go_norm.*`) and DGZ MLP keys onto our modules; `load(path, strict=True)` must succeed with the vocabulary of the checkpoint (`terms.pkl` for `deepgozero.th`/`mlp.th`, `terms_zero.pkl` for `deepgozero_zero.th`). A shape or key mismatch raises `CheckpointMismatch` (fatal for G3/G2b; the error lists missing/unexpected keys and shapes).

## 7. Training — `go_function/train.py`, `scripts/go_function/run.py`

### 7.1 Run spec (one JSON per run; validated before anything loads)

| field | type / allowed | rule on failure |
|---|---|---|
| `aspect` | "bp" \| "mf" | exit 2 |
| `feature` | "esm2_650m" \| "interpro" | exit 2 |
| `variant` | V0, V1, V2, V3, V3c, V4, V5, V6 | exit 2 |
| `regime` | "TP" \| "DGZ" \| "TPnoIPI" (MF only) | exit 2 |
| `rule` | "W2" \| "W1" \| "WDGZrel" \| "W2drop" (the last two: DGZ regime, k = 0 only) | exit 2 |
| `retention_draw` | 1 (default) \| 2 (only rule W2, regime TP, k ∈ {1, 5}) | exit 2 |
| `pack` | "p01".."p08" or null for `all` | must exist in target table, exit 2 |
| `k` | 0, 1, 5, 10, 25 or "all" | exit 2 |
| `seed` | 13, 21, 42, 87, 123 (study); any int (pilots/gates) | exit 2 in study mode |
| `epochs` | int 1–256 | must equal the pinned E* in study mode, exit 2 |
| `lambda_el` | float > 0 or null (no EL arms) | must equal 1.0 for EL arms and null otherwise in study mode, exit 2 |
| `batch_size`, `lr`, `milestones`, `gamma` | int, float, list[int], float | must equal `study.json`, exit 2 |
| `selection` | "final" (study) \| "best_valid_loss" (gates only) | exit 2 |
| `out_dir` | repo-relative path under `results/go_function/` | exit 2 |

Unknown fields are rejected. The single-factor audit (§12.2) is applied to the materialised specs, not at run time.

### 7.2 Loop

- Optimiser Adam(lr 5e-4); MultiStepLR(milestones [5, 20], gamma 0.1); batch 37; shuffle with a `torch.Generator` seeded by `seed`; `torch.manual_seed`, `numpy`, `random` seeded; `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Bitwise reproducibility on GPU is not claimed; seeds, not bit-equality, carry run-to-run variance.
- Labels: the condition CSR is densified to a bool tensor on device (BP 47,733 × 21,381 ≈ 1.0 GB); a non-empty mask (WDGZrel/W2drop only) is a bool tensor of the same shape built from the pair list (or a sparse index set when the device is CPU); an empty mask (W1/W2, `all`) allocates no tensor. Features: dense float32 ESM, or InterPro CSR converted per batch.
- `epochs` = E* (study, final-epoch checkpoint) or up to 256 with best target-free validation loss (gates).
- Diagnostics every epoch (`epochs.jsonl`): train BCE, EL per form, target-free validation BCE, lr, wall time; every 50th batch the gradient norms of BCE and λ·EL on the protein tower and the term tower separately, and their ratio (pattern of `withholding_diagnostics.projection_params/_grad_norm`, reimplemented). A non-finite loss aborts the run (§15).
- No checkpoint is kept by study runs except condition `all`, seed 13, every arm (audit). Gates and pilots keep their checkpoints.

### 7.3 After training (in-process evaluation, §8)

`eval.json`, then `scores.npz` with the evaluation-set test logits of, for each pack target t:

- t itself;
- every column used by fusion F: TP parents Pa(t) and is_a siblings of t;
- the **channel columns** of the C3_leak channel reads (protocol §1), all primary non-duplicate columns ≠ t, fixed in the target table at G6:
  - X(t), the DGZ-only ancestors: {c ∈ V : c ∈ closure_DGZ(t), primary(c) ∉ ancestors_TP(t)}. These are the fillers reached through regulates-family, has_part, occurs_in or cross-namespace edges, i.e. the relation-conflation channel. X_reg(t) ⊆ X(t) is the subset reached through a regulates-family edge on the first hop or named as a regulates-family differentia in `logical_definition(t)`; it is recorded for reporting;
  - Desc(t), the strict descendants: the columns of D*_TP(t) \ {t};
  - Pa(t), already stored for F, as the ancestor channel.

Stored as `logits` (float32, n_test × n_cols) with a `columns` index array. X(t) and Desc(t) are written only by V4 runs at k = 0, all regimes and rules and both features: 360 runs for BP and 400 for MF (MF adds the noIPI block), covering every run the reads use. This bounds disk use, because Desc(t) can hold hundreds of BP columns. All other runs store t, Pa(t) and the siblings only. The target table records |X(t)|, |X_reg(t)|, |Desc(t)| and |Pa(t)|. No read needs an extra run. Next come `run.json` (spec, input hashes, code pin hash, environment: torch/CUDA/GPU/driver, wall times), then `DONE.json`. Any exception after training writes `FAILED.json` with `stage` ∈ {preflight, train, eval} and the traceback summary.

### 7.4 Preflight (fatal, before the model is built)

Code-pin coverage of imported repo modules (§13); hashes of features, labels, condition, target table, validation mask, normal forms and term text against `protocol_pins.json`; re-run of the §5.5 assertions on the condition files; E*, λ = 1 for EL arms, `study.json` fields equal the pins (study mode).

## 8. Evaluator (gap 4) — `go_function/evaluate.py`, `scripts/go_function/gate_dgz_eval.py`

### 8.1 Class-centric (primary)

For target t: positives = evaluation-set test proteins with y_TP[p, t] = 1; negatives = all other evaluation-set test proteins. AUC = `roc_auc_score` on raw logits of column t (no score propagation); AUPR = `average_precision_score`. Also reported under DGZ-rule test labels (sensitivity). n_pos is stored. A target with n_pos = 0 or all positives would be a protocol stop; it cannot happen after the n_test ≥ 10 filter, and is asserted.

### 8.2 Protein-centric (collateral and gates)

Over a term set T (collateral: primary non-duplicate columns of V with ≥ 1 TP train positive in the base labels (D16), minus ∪_{pack} D*_TP(t), root excluded; for the noIPI block, TPnoIPI replaces TP throughout):

- Scores are max-propagated over TP ancestors within T (pred(c) = max over columns d ∈ T with c ∈ ancestors_TP(d) of σ(logit(d))).
- Fmax: thresholds 0.00…1.00 step 0.01; precision averaged over proteins with ≥ 1 predicted term, recall over all evaluated proteins (DGZ `compute_fmax` [read]).
- Smin: IC(c) = log2(min over parents' annotation counts / count(c)) from train+valid TP labels (DGZ `Ontology.calculate_ic` [read]); remaining uncertainty and misinformation summed per protein, averaged, minimum over the same thresholds. The exact Smin and micro-AUPR formulas are ported from DGZ's `evaluate.py` at implementation (that file was not read during this design; noted in §17); G2 validates the port.
- micro AUPR over flattened (protein, term) pairs.

Computed in torch on the run's device (BP: 5,444 × ~21k × 101 thresholds, chunked by threshold).

### 8.3 DGZ-faithful mode (gates only)

Uses DGZ-rule labels, `terms.pkl` columns as-is, their predictions as-is (already propagated by them), and the "average AUC over terms with 0 < pos < n" of `evaluate_terms.py` [read]. Published values are transcribed at implementation from DGZ Table 3 (Bioinformatics 2022, PMC9235501) into `config/go_function/dgz_published.json` with the table cell they came from; nothing in this design invents them (MF class AUC MLP 0.882, DGZ 0.903 are in the spec).

## 9. Pilots (gap 9) — `scripts/go_function/pilot.py`

Run on the remote host with the study feature (ESM) per aspect, after G6 and before the code pin; InterPro values for the InterPro sensitivity grid come from a second pilot on InterPro.

### 9.1 Epoch budget E*

Condition `all`, TP regime, V0 and V4 (λ = 1), seeds 13, 21, 42, up to 64 epochs with the standard schedule. For each run, i_best = the 0-based index of the epoch with minimum target-free validation BCE. The single formula is

E* = clip(ceil(median over the six runs of (i_best + 1)), 8, 64).

One E* per (aspect, feature), shared by every arm, k, rule and regime.

**Text-arm epoch record (descriptive).** V5 and V6 train a 768→1024 projection rather than free term vectors, so they may converge at a different epoch. The ESM pilot therefore also runs V5 and V6 at `all`, TP, seed 13 (the InterPro grid has no text arms), for up to 64 epochs, and records (i_best + 1) and the validation BCE at E* in `choice.json`. These two runs never enter the E* formula and never change E*.

### 9.2 Axiom weight (descriptive record only)

λ is fixed at 1 (§6.5, protocol D12): a criterion based on validation BCE of fully supervised non-target columns measures exactly what the EL term competes with and would favour small λ, shrinking every EL arm toward V0 as a tuning artefact. The pilot still runs V3 and V4 at `all`, TP, seeds 13 and 21, λ ∈ {0.1, 0.3, 1, 3}, E* epochs, and records the final target-free validation BCE and the λ·EL/BCE gradient-norm ratio per tower. The record is reported and never used to select anything.

The E* choice and the λ record are written to `results/go_function/pilots/<aspect>_<feature>/choice.json` and pinned. Test data and target columns are never read by the pilot (the evaluator is not imported; asserted by an import probe).

## 10. Fusion F (no training) — `go_function/fusion.py`

From a V0 run's `scores.npz`: z(·) = standardisation of a logit column over the evaluation-set test proteins. For target t: Pa(t) = TP parents of t that are primary columns (root excluded); Sib(t) = is_a siblings that are primary columns with ≥ 10 positives in that condition's train labels. s_F(p, t) = z(ℓ_t) + z(mean_{π∈Pa(t)} ℓ_π) − z(mean_{σ∈Sib(t)} ℓ_σ); an empty set drops its term (recorded). No weight is tuned. F is computed for V0 under W2 at every k and under W1 at k ∈ {0, 1, 5} (the only W1 conditions); F − V0 is descriptive (protocol §6).

## 11. Baselines without training

- **Naive:** term frequency in the condition's train labels (constant per term; class AUC 0.5 by definition, reported only for protein-centric context in G4).
- **DiamondScore (DS):** for test protein p and term c, Σ_{h ∈ hits(p) ∩ train} bitscore(p, h)·y_cond[h, c] / Σ_{h} bitscore(p, h), from `test_diamond.res`, with y_cond the condition's train labels (masked entries count as 0). Computed per condition, no seed. Descriptive reference curve.

## 12. Configs, grid and sweep — `config/go_function/`, `scripts/go_function/materialize.py`, `scripts/go_function/sweep.py`

### 12.1 `config/go_function/study.json`

Batch 37, lr 5e-4, milestones [5, 20], gamma 0.1, dropout 0.1, hidden/embed 1024, margin 0.1, norm type (after G3), seeds, k levels, retention seeds (draw 1 20261011, draw 2 20261012) and analysis seed, E* per (aspect, feature) after pilots. λ = 1 for EL arms. Edited only by `materialize.py` and pinned.

### 12.2 Materialiser and single-factor audit

`materialize.py --aspect A --feature F` writes one run spec per grid cell (protocol §6.2 table) into `results/go_function/specs/`. It asserts, for every preregistered contrast and every descriptive component, that the paired specs differ in exactly the declared field set; any other difference is fatal:

| contrast | arm A spec | arm B spec | declared differing fields |
|---|---|---|---|
| C1 | V3 | V3c | {variant} |
| C2, C4, C5, C6, C7 | V4 / V5 / V6 / Vbest | V0 / V0 / V5 / V0 | {variant} |
| **C3** (primary) | V4, DGZ, WDGZrel, k = 0 | V4, TP, W2, k = 0 | **{regime, rule}** (declared two-factor, protocol D17) |
| C3_leak (descr.) | V4, DGZ, WDGZrel | V4, DGZ, W2drop | {rule} |
| C3_col (descr.) | V4, DGZ, W2drop | V4, DGZ, W2 | {rule} |
| C3_reg (descr.) | V4, DGZ, W2 | V4, TP, W2 | {regime} |
| W1 − W2 | W1 | W2 | {rule} |
| re-draw | draw 2 | draw 1 | {retention_draw} |

`analyze.py` asserts the identity C3 = C3_leak + C3_col + C3_reg to 1e-12 on every (t, s) cell where all four runs are complete. The decomposition table, the three components and C3 itself are estimated on that common cell set, with its cell count. The Holm-tested C3 is estimated on its own two-run cell set; when the two cell sets differ, both C3 values are printed (protocol §8).

**Channel reads (descriptive).** They are computed by `analyze.py` from `scores.npz` of the V4 WDGZrel and W2drop runs, at k = 0, against TP test labels for t:

- C3_leak_X: the class AUC for t of max_{c∈X(t)} logit(c), WDGZrel − W2drop;
- C3_leak_desc: the same with Desc(t);
- C3_leak_anc: the same with Pa(t).

Each read is defined only on targets whose channel set is non-empty; the count of such targets is reported. Each uses the same bootstrap draws as C3. The channel AUCs are not additive and are not asserted to sum to C3_leak. C3_leak_desc is also reported stratified by |Desc(t)| (≤ 5 vs > 5 columns), because a max over many weak heads is noisier and biased upward. The channel reads are compared with each other by CI sign only, never by size.

### 12.3 Sweep driver

A new driver modelled on `scripts/go_withholding/sweep.py` (read, not imported): `plan`, `run --workers N --max-runs M [--filters]`, `status`, `pin`, `rescore`. Run dir `results/go_function/runs/<aspect>/<feature>/<variant>__<regime>__<rule>__<pack|all>__k<k>[__d2]__s<seed>/` (`__d2` only for retention draw 2). Claim by atomic `os.open(O_CREAT|O_EXCL)` of `CLAIM` with pid and host; stale claims (dead pid on the same host) are released by `status --release-stale`. A failed attempt is renamed `<dir>.attempt<n>`; two failed attempts leave `FAILED.json` and the run is missing for the analysis. `rescore` re-runs only evaluation for runs that kept a checkpoint (condition `all`, seed 13) or whose `FAILED.json` has stage `eval` and still holds the in-memory scores dump `eval_inputs.npz` (written before evaluation starts, deleted after success). Order: seed-major (all cells of seed 13 first), so an interrupted grid still has complete seeds. Workers are separate processes on the one GPU; default N = 4, set after the smoke measures memory.

## 13. Code pin and protection

### 13.1 Pin

`sweep.py pin` hashes every `.py` under `go_function/` and `scripts/go_function/` and every `.json` under `config/go_function/` into `results/go_function/code_pin.json`, after the user commits/tags or explicitly accepts a content pin (no git writes by any step). `run.py` and `analyze.py` check at start that every imported module whose file lies inside `$R` is covered by the pin, and that none comes from `go_ppi_domain/`, `scripts/go_withholding/`, `contrastive_learning/` or `trials_domain/` (fatal).

### 13.2 Vendored helpers

Copied into `go_function/vendored.py` with a header citing source file, function and the source file's sha256 at copy time: `jaccard` (from `go_ppi_domain/withholding_targets.py`; `greedy_non_nested` is not vendored, §4.6); `_counts`, `two_way_draws`, `_summarise`, `two_way_bootstrap`, `hierarchical_bootstrap`, `holm`, `classify_contrast`, `ci_excludes_0` (from `scripts/go_withholding/analyze_withholding.py`). The originals read module constants `ANALYSIS_SEED`, `FLOOR = 0.01`, `ALPHA = 0.05`, `UNSTABLE_SHARE = 0.05`; the vendored copies take them as keyword arguments with no defaults, and `go_function/constants.py` defines the study values (FLOOR 0.01, ALPHA 0.05, UNSTABLE_SHARE 0.05, ANALYSIS_SEED 20261011). A parity test imports the originals (tests are not under the pin's import restriction) and asserts identical outputs on fixed random inputs, passing exactly FLOOR = 0.01, ALPHA = 0.05, UNSTABLE_SHARE = 0.05 and the original's seed.

### 13.3 Protection checks (G0)

`git diff --stat` must show no change under any path outside `go_function/`, `scripts/go_function/`, `config/go_function/`, `dataset/go_function/`, `preprocess/go_function/`, `results/go_function/` and the task folder, compared with the pre-study status recorded in `setup_baseline_go_function.md`. The regression suites of that file must give the same results (334 passed / 3 skipped / 1 pre-existing failure in `go_ppi_domain/tests`; 108/1 in `trials_domain/tests`; 8 career/CVE tests).

## 14. Testability

Unit tests (`go_function/tests/`, CPU, seconds, run at G0 and in CI-like local runs):

- ontology: a 12-term synthetic OBO covering is_a, part_of across and within namespaces, regulates, has_part, alt_id, obsolete and intersection_of; closures, siblings, two_hop, logical definition; regulates never propagated; cross-namespace part_of not propagated under TP but propagated by `DgzRule`.
- `DgzRule` on the real pinned go.obo against `prop_annotations` of 200 fixed proteins per aspect (integration, ~20 s; marked `slow`, run at G0).
- normforms: a synthetic go.norm with every line form, the "unparsed" role-axiom case and an NF2 `SubClassOf Nothing` line; `owl:Nothing` mapping and index position; a non-GO token other than `Nothing` is fatal; DGZ-compatible id order; ladder nesting; V3c shared relation; V1–V3c contain no `owl:Nothing`.
- term text: `embed_terms.py` on the fixture skips `owl:Nothing`, fails on a GO class missing from the OBO; V6 builds with one free vector for `owl:Nothing`.
- labels: matrices on the synthetic ontology; duplicate columns; MF protein-binding-only exclusion; noIPI.
- withholding, alt ids and namespaces:
  - **Cross-namespace part_of fixture.** An MF term m with `m part_of t` for a BP target t, annotated on a withheld protein. Under ρ = DGZ, W2 erases m from A_DGZ(p) and t is not re-implied (assertions 1–2 pass). A planted variant that erases only from the aspect-filtered set trips assertion 1 or 2. Under TP, m is not in A_TP(p) and is irrelevant.
  - **Alt-id fixture.** A term d with alt id d′ that is a column and is annotated on some proteins. Checks:
    - `build_targets` excludes d and reports it;
    - DgzRule.closure(d′) lacks d but contains d's parents, and §2.4 check 3d accepts this;
    - when t is an ancestor of d, d′ is in D*^GO_DGZ(t) and is erased under DGZ W2;
    - a planted target with an alt column trips assertion 8 under WDGZrel when the mask covers only column t.
- withholding: on a synthetic 30-protein dataset with a "positive regulation of X" target, W2/W1/WDGZrel/W2drop satisfy every §5.5 assertion; under WDGZrel the X column keeps every t-positive and under W2 (DGZ regime) those entries are erased by re-propagation, as unmasked 0 labels (the leak channel is present in one and absent in the other); a deliberately leaked condition (one erased annotation left) trips assertion 1; a planted row-selective mask trips assertion 10; Lost(p) entries with multiple inheritance are 0 and unmasked; a non-column direct annotation implying t is erased; draw-2 retention differs from draw 1 and is nested; **closed-world test:** the W2 condition's train mask is empty and its labels differ from the base only on erased (p, c) entries of withheld proteins.
- labels:
  - the independent BFS closure (§2.4 check 3b) on the fixture;
  - check 3c (column level) trips on a planted regulates-propagated TP label and passes on an alt-annotated protein;
  - check 3d trips on a planted TP-only primary label that does not come from an alt id;
  - the TPnoIPI evaluation set on a fixture with an IPI-only MF protein.
- channel reads: on a planted `scores.npz` pair, C3_leak_X, C3_leak_desc and C3_leak_anc match hand-computed AUC differences, and a target with empty X(t) is excluded and counted.
- targets: tertiles, LD flag, band quotas, `accept_round_robin` (visit order, one acceptance per visit, quota borrowing B1→B2→B3→B1, ancestor and Jaccard rejection), packing constraints including a planted non-column term implying two candidates (must be rejected via D*^GO), determinism (two builds byte-identical), shortfall exit.
- model: forward shapes for every arm; masked BCE equals unmasked BCE when the mask is empty; EL loss equals DGZ's formulas on hand-computed 3-class examples (both NF4 modes); checkpoint key mapping on a synthetic state dict.
- evaluate: class AUC/AUPR against sklearn; Fmax/Smin on DGZ's toy cases; max-propagation.
- fusion, DS baseline: hand-computed examples.
- vendored parity (§13.2); analysis contrasts on a synthetic long table with known effects (planted +0.05 recovered, CI covers it; null gives "practically null").

Integration (local, InterPro, CPU): the protocol G9a smoke is the end-to-end test: build → condition → train 2 epochs → eval → analysis on 1 pack.

GPU-only (remote): ESM embedding cross-check; G4 reproduction; timing.

## 15. Error handling

Exit codes for every CLI: 0 success; 2 fatal input/invariant failure (nothing written or a `*.FAILED.json` written beside the intended output); 3 protocol stop (a preregistered rule says stop and ask the user); 4 refused because of the state of the grid (e.g., a pin is missing). Logging: `logging` at INFO to stderr and to `<out>/log.txt`; WARNING for recoverable conditions; ERROR before every non-zero exit.

| operation | failure condition | recoverable? | caller receives | logged |
|---|---|---|---|---|
| hash verification of any pinned input | sha256 differs or file missing | no | exit 2, message with path and both hashes | ERROR |
| OBO parse | malformed stanza, dangling edge, cycle | no | `OntologyError`, exit 2 | ERROR with line |
| go.norm parse | unparsed count ≠ 9, unknown relation IRI, non-GO token other than `Nothing`, `owl:Nothing` count ≠ 30 or outside an NF2 right-hand side | no | exit 2 | ERROR |
| term text build | a GO class (not `owl:Nothing`) has no live term in go.obo | no | exit 2, list of ids | ERROR |
| label build | any §2.4 check fails | no | exit 2, first 10 offending proteins | ERROR |
| ESM download/load | network error, revision mismatch | yes (retry ×3, 30 s backoff), then no | exit 2 | WARNING per retry, ERROR at end |
| ESM embedding | CUDA OOM | yes: halve the token budget down to 2,048, then fatal | exit 2 if still failing | WARNING per halving |
| ESM cross-check | cosine < 0.999 | no | exit 2 | ERROR with protein ids |
| target build | slot shortfall after borrowing, no feasible pack | protocol stop | exit 3, counts only | ERROR |
| target build | alt-id exclusion (§4.1) | expected | count in the target table | INFO |
| condition build | any §5.5 assertion (1–10); existing dir differs | no | exit 2 | ERROR |
| analysis | channel read with empty X(t), Desc(t) or Pa(t) for a target | yes (target excluded from that read) | count in the output | INFO |
| run spec | schema violation, pin mismatch | no | exit 2, `FAILED.json` stage preflight | ERROR |
| training | non-finite loss or gradient | no (no automatic retry inside the run) | `FAILED.json` stage train; sweep retries once in a fresh attempt dir | ERROR |
| training | CUDA OOM | no inside the run; sweep retries once with workers reduced by one for that run | as above | ERROR |
| evaluation | exception | yes via `sweep.py rescore` | `FAILED.json` stage eval, `eval_inputs.npz` kept | ERROR |
| sweep claim | claim exists, live pid | yes (skip run) | run skipped | INFO |
| checkpoint load (G2b/G3) | key/shape mismatch | no | `CheckpointMismatch`, exit 2; G2b writes `gate_dgz_zero.json` with `"loaded": false` instead (protocol rule) | ERROR |
| analysis | missing > 10% of tuples | protocol rule | outputs labelled exploratory | WARNING |
| analysis | run dir with `DONE.json` whose hashes disagree with `run.json` | no | exit 2 | ERROR |

Invariant ownership:

| invariant | enforced by | why there |
|---|---|---|
| evaluation labels are TP test labels, never touched by withholding | `evaluate.py` loads test labels from the base label file itself; `make_conditions.py` never writes test labels | the only code path to test labels has no access to condition files |
| evaluation label source per block | `evaluate.py`: allowed sources are `labels_TP.npz` (regime TP or DGZ) and `labels_TPnoIPI.npz` (only when `regime = TPnoIPI`), each with its own evaluation set; DGZ-rule test labels only in gate scripts and the descriptive sensitivity read | one function maps regime → label file and evaluation set, so a noIPI run cannot be scored on with-IPI labels or vice versa |
| λ = 1 for every EL arm | `run.py` preflight (spec field check) | the study has no λ selection step to drift |
| no D*(t) positive for W proteins; exactly k positives | `withholding.py` at generation; re-asserted in `run.py` preflight | generation owns correctness; preflight catches stale or edited files |
| closed world: no row-selective mask (W1/W2 mask empty; WDGZrel/W2drop masks are whole train columns) | `withholding.py` assertion 10 at generation and in `run.py` preflight; the trainer builds no mask tensor for an empty mask | a mask pattern that marks W would leak P(t) into training (round-3 Finding 1) |
| no target has an alt-id column or annotation; P_TP(t) ⊆ P_DGZ(t) | `targets.py` (exclusion at build); `withholding.py` assertions 8–9 | the target table is the single point where targets enter; generation re-checks it per condition |
| erasure operand per regime (A_TP, A_TPnoIPI, A_DGZ) | `labels.py` stores A_ρ in `direct_*`; `withholding.py` reads only that | one producer defines the id space per regime, so DGZ erasure cannot fall back to aspect-filtered ids |
| same vocabulary across all arms | `labels.py` writes vocab hash; `run.py` asserts it equals the model's column list | one producer, one consumer check |
| target-free validation and pilots | `make_conditions.py` masks; `pilot.py` import probe forbids the evaluator | prevents selection on targets by construction |
| single-factor contrasts | `materialize.py` diff audit | specs are the only source of run configuration |
| hyperparameters and E* fixed | `run.py` preflight vs pins | runtime cannot drift from the frozen config |
| code identity | `sweep.py pin` + per-run import coverage | catches edits after the pin |

## 16. Files

New (all under the allowed directories):

```
go_function/__init__.py  constants.py  ontology.py  normforms.py  labels.py  features.py
go_function/targets.py  withholding.py  model.py  train.py  evaluate.py  fusion.py
go_function/baselines.py  stats.py  vendored.py  pins.py
go_function/tests/test_*.py  go_function/tests/fixtures/{mini.obo, mini.norm}
scripts/go_function/download_dgz.py (exists)  download_dgz_gate.py  build_labels.py
scripts/go_function/gate_dgz_eval.py  gate_dgz_zero.py  gate_checkpoints.py  gate_trainer.py  build_features.py  embed_esm.py  embed_terms.py
scripts/go_function/build_targets.py  make_conditions.py  pilot.py  materialize.py  run.py
scripts/go_function/sweep.py  analyze.py  requirements-remote.txt
config/go_function/study.json  dgz_published.json
```

Data and outputs:

```
dataset/go_function/dgz/...               (pinned; MANIFEST.json, MANIFEST_gate.json)
preprocess/go_function/{bp,mf}/labels_{TP,DGZ,TPnoIPI}.npz  features_interpro.npz  targets/  conditions/
preprocess/go_function/esm2_650m/  term_text_mpnet.npy
results/go_function/gates/  pilots/  specs/  runs/  analysis/  protocol_pins.json  code_pin.json
```

`download_dgz_gate.py` fetches, per aspect bp/mf: `mlp.th`, `deepgozero.th`, `deepgozero_zero.th`, `predictions_mlp.pkl`, `predictions_deepgozero.pkl`, `predictions_deepgozero_zero.pkl`, `zero_gos.pkl`, `zero_term_auc{,_all,_train}.pkl`, and `zero_<GO>_auc{,_all,_train}.pkl` for the withheld classes, into `dataset/go_function/dgz/<aspect>/` with its own `MANIFEST_gate.json` (same never-overwrite and Content-Length rules as `download_dgz.py`). About 6–9 GB; run where G2/G2b run.

Modified: none outside the allowed directories.

## 17. Not verified at design time

- DGZ `evaluate.py` (Smin, micro-AUPR) and the DGZ MLP model file were not read; the port is validated by G2.
- The norm layer type of the released checkpoints (§6.1) and whether `deepgozero_zero.th` loads with `terms_zero.pkl` (§6.6).
- The DGZ Table 3 values other than MF class AUC.
- Per-run GPU time; the grid time estimate in protocol §11 is an estimate until the smoke measures it.
- A 30%-identity split is not present in the release's `data/` listing [ran]; the spec's 30% sensitivity is deferred (protocol D13).
- [read]-pending, taken from DGZ's paper and code as cited in the spec but not re-read against DGZ's source in this design step (DGZ source is not in the repo): hyperparameters (batch 37, lr 5e-4, milestones [5, 20], margin 0.1, dropout 0.1); `nn.BatchNorm` in `deepgozero.py`; the NF4 radius indexing; EL over the full axiom set every batch; `evaluate.py` Smin/micro-AUPR; that released predictions are already DGZ-rule propagated (G3 tolerance 1e-4). Backstops: G2, G3, G4 stop the study on any mismatch.
- Whether `deepgozero_zero.th` was trained on `prop_annotations` or `zero_annotations` (G2b records it; WDGZrel is fixed to `prop_annotations ∩ V`, §5.2).
- The §0.2 [ran] facts were checked during design v1; G1 re-asserts each on every protein. The `owl:Nothing` count (30, all NF2) was re-checked in v2 [ran].
- The v3 alt-id facts were checked on train proteins only, with a scratch replica of DgzRule and a TP closure written for the check (not the study code):
  - §2.4 check 3d has 0 exceptions;
  - 778 BP and 59 MF TP-only primary labels;
  - about 14 BP and 5 MF terms excluded by alt id, before the n_test filter.

  G1 re-runs check 3d on every split, and G6 reports the final exclusion count. The cross-namespace counts are the reviewer's [ran] and were not re-run here.
- Whether DGZ's regulates-family fillers X(t) are trained columns for every BP target with a regulation logical definition. Targets with empty X(t) drop out of C3_leak_X, and the count is reported.

## 18. Responses to the design reviews

### Round 1 (v1 → v2)

Review: `design_review_go_function.md` (CHANGES_REQUESTED; 1 HIGH, 3 MEDIUM, 11 NIT). Every finding is addressed; one fix differs from the reviewer's wording, with the reason given.

| # | sev | response | where |
|---|---|---|---|
| 1 | HIGH | **Addressed, with one deliberate difference.** Added `WDGZrel` (DGZ labels unchanged, column t masked for all rows, k = 0) exactly as proposed, and `W2drop`. The re-propagating W_DGZ is dropped. *Difference:* the reviewer redefines C3 as WDGZrel − W2 within the DGZ regime and demotes the TP-vs-DGZ comparison to descriptive. The user's decision 7 and the spec ((i), (ii)) define the primary contrast as "V4 under DGZ-rule labels vs true-path labels at k = 0". Primary C3 is therefore AUC_{V4,DGZ,WDGZrel} − AUC_{V4,TP,W2}: DGZ-rule labels as DGZ trained on them vs true-path labels under the leak-free protocol. This keeps the user's contrast and makes it sensitive to the leak. It is a declared two-factor contrast. Its exact additive decomposition into three single-factor descriptive parts (C3_leak = WDGZrel − W2drop, the propagation channel; C3_col = W2drop − W2, the negatives-only-column confound of the review's point 6; C3_reg = DGZ-W2 − TP-W2, the reviewer's C3r) is computed on the same runs. The central-result reading needs C3_leak > 0 as well (protocol §9.3). H3 and §9.3 are reworded. | §0.2, §5.2, §5.5, §7.1, §12.2; protocol §1, D7, D17, §5, §6.2, §9 |
| 2 | MED | Addressed with option (a): λ = 1 fixed a priori for every EL arm; the λ sweep is a descriptive record. D12 now covers E* only. | §6.5, §7.1, §9.2; protocol D12, §6.4 |
| 3 | MED | Addressed: `Nothing` → reserved `owl:Nothing`, kept in DGZ index order, count 30 (all NF2) pinned at G1; text is required for GO classes only; V6 gives `owl:Nothing` a free centre and radius; §0.2 corrected. | §0.2, §3.3, §6.2, §6.4, §14, §15 |
| 4 | MED | Addressed: the noIPI block trains and evaluates on TPnoIPI labels with its own evaluation set and drop rule; all other blocks use TP test labels; there is a new invariant-ownership row. | §2.2, §2.5, §8.2, §15; protocol §3, §6.2 |
| 5 | NIT | Addressed: decision D16; zero-positive column counts reported; such columns are excluded from the collateral set. | §2.1, §8.2 |
| 6 | NIT | Addressed: new `accept_round_robin` with its own tests; only `jaccard` is vendored. | §4.6, §13.2, §14 |
| 7 | NIT | Addressed: term-level D*^GO for erasure and pack disjointness; column D* for masks. | §4.9, §5.1, §5.2 |
| 8 | NIT | Addressed: independent BFS closure check plus a "no DGZ-only label positive under TP" check. | §2.4 |
| 9 | NIT | Addressed: renamed "untyped-relation control"; the part_of collapse is stated; H1 is reworded. | §6.4; protocol §1, §6.1 |
| 10 | NIT | Addressed: F under W1 at k ∈ {0, 1, 5}. | §10 |
| 11 | NIT | Addressed in the protocol: G2b L2 and G4 use DGZ-rule labels and DGZ-faithful mode. | protocol §10 |
| 12 | NIT | Addressed in the protocol: near-orthogonality is stated; C6/C7 are reported for every arm descriptively. | protocol §1 |
| 13 | NIT | Addressed: added as a risk, and a descriptive retention re-draw block was added (draw 2, V0/V4, TP, W2, k ∈ {1, 5}, seed 13; 32 runs per aspect). | §4.10, §5.4, §7.1; protocol §6.2, §11 |
| 14 | NIT | Addressed: the constants are explicit arguments; the parity test passes FLOOR 0.01, ALPHA 0.05, UNSTABLE_SHARE 0.05. | §13.2 |
| 15 | NIT | Addressed: cross-references fixed (§12.2 → protocol §6.2, §14 → protocol G9a, §17 → D13); TPnoIPI added to §5.1; one E* formula. | §5.1, §9.1, §12.2, §14, §17; protocol §6.4 |

Grid change from v1: +40 W2drop runs per aspect per feature (ESM and InterPro) and +32 re-draw runs per aspect. BP 2,670 → 2,782, MF 3,045 → 3,157, total 5,715 → 5,939 (protocol §6.2).

### Round 2 (v2 → v3)

Review: `design_review_go_function.md` round 2 (CHANGES_REQUESTED; 1 HIGH, 2 MEDIUM, 4 NIT). Every finding is addressed; none is backlogged.

| # | sev | response | where |
|---|---|---|---|
| 1 | HIGH | **Addressed, all three parts.** (a) Base filter excludes any t with an alt-id column or an alt-id annotation in any split; the count is reported. A scratch check on train labels gives 14 BP and 5 MF exclusions among 4,638 BP and 790 MF terms with 30 ≤ n_train ≤ 5% N, so the strata quotas are not at risk. (b) DGZ-regime masks (W2, WDGZrel, W2drop) use T_cols(t) = {c : primary(c) = t}. New assertion 8 (no unmasked positive on any column of t outside R_k) and assertion 9 (R_k ⊆ P_TP(t) ⊆ P_DGZ(t)). (c) Check 3c is restated at column level via `primary`. New check 3d: TP-only labels come only from alt columns or alt-annotation primaries; it has 0 exceptions on train. The §5.1 rationale now says the subset holds for targets without alt ids, not "TP closure ⊆ DGZ closure". | §0.2, §1.3, §2.4, §4.1, §5.1, §5.2, §5.5, §14, §15; protocol §4, §5 |
| 2 | MED | **Addressed.** Per-regime direct sets A_TP, A_TPnoIPI and A_DGZ (raw, all namespaces), stored in `direct_*`/`direct_extra_*`. D*^GO_DGZ is defined on raw strings via `DgzRule.closure`, and W2 erases from A_ρ. Added a cross-namespace part_of fixture test (with a planted aspect-filtered erasure that must fail) and an invariant-ownership row. | §0.2, §2.2, §2.3, §5.1, §5.2, §14, §15; protocol §5 |
| 3 | MED | **Addressed.** `scores.npz` of V4 k = 0 runs stores X(t) (DGZ-only ancestors; X_reg recorded), Desc(t) and Pa(t). The descriptive reads are C3_leak_X and C3_leak_desc, as requested, plus C3_leak_anc for the third channel. The readings now call C3_leak "the protocol's propagation channel (descendant, ancestor and non-is_a filler labels)". The typed-relation (filler) channel is claimed only if C3_leak_X's CI excludes 0 **and** G2b L3 is confirmed; otherwise the inflation is attributed to subtree non-erasure in general. | §0.2, §7.3, §12.2, §14; protocol §1, §9.2, §9.3 |
| 4 | NIT | Addressed: MF L2 criterion is ≥ 4 of 5 withheld classes within 0.01. | protocol §10 |
| 5 | NIT | Addressed: C3 and its components are estimated on the common four-run cell set, with the count; the Holm-tested C3 uses its own two-run cells, and both values are printed if they differ. | §12.2; protocol §8 |
| 6 | NIT | Addressed: added as risk 14; the ESM pilot records the V5/V6 best-validation epoch at `all`, seed 13 (2 pilot runs per aspect, not in E*). | §9.1; protocol §6.4, §11 |
| 7 | NIT | Addressed: WDGZrel wording now says it matches `prop_annotations ∩ V`, which equals `zero_annotations` on the withheld column for 6 of 7 BP classes. | protocol §5 |

The grid is unchanged (5,939 runs). The pilot adds 2 descriptive runs per aspect.

### Round 3 (v3 → v4)

Review: `design_review_go_function.md` round 3 (CHANGES_REQUESTED; 1 HIGH, 0 MEDIUM, 4 NIT), with the orchestrator decision adopting the HIGH fix as specified. Every finding is addressed.

| # | sev | response | where |
|---|---|---|---|
| 1 | HIGH | **Addressed as specified (closed world).** W2 and W1 add no mask: erased D*_ρ(t) and Lost(p) entries are ordinary 0 labels, exactly as for a protein never annotated in t's subtree, so no row-level signal identifies W. WDGZrel is unchanged; W2drop = closed-world W2 labels + T_cols(t) masked on every train row. Assertion 1 now reads "y = 0 and not masked"; assertion 7 is the W2drop column mask alone; new assertion 10 (W1/W2 train mask empty; WDGZrel/W2drop masks cover whole train columns; no row-selective mask on a target-related column). The trainer builds no mask tensor for an empty mask, and a trainer unit test checks that a W2 condition's per-row loss mask equals the base data's. Protocol D9 is withdrawn and replaced by D19 (a deviation from spec §5 "Masking", for user/supervisor sign-off before G8); risk 6 is rewritten. The masked rule is not built. | §5.2, §5.4, §5.5, §6.5, §7.2, §14, §15; protocol §2 (D4, D9, D19), §5, §11 risk 6, §16 |
| 2 | NIT | Addressed: assertion 8 says "Under WDGZrel/W2drop (k = 0 only), no column with primary(c) = t has any unmasked train entry". | §5.5 |
| 3 | NIT | Addressed: n_train(F) = \|{p ∈ train : F ∈ closure_TP(p)}\|, from the closure whether or not F ∈ V. | §4 step 3; protocol §4 |
| 4 | NIT | Addressed: C3_leak_desc is also reported stratified by \|Desc(t)\| ≤ 5 vs > 5; channel reads are compared by CI sign only. | §12.2; protocol §1 |
| 5 | NIT | Addressed: protocol §8 "; its"; the alt-id count reads "about 14 BP and 5 MF before the n_test filter" in both documents. | §0.2, §4.1, §17; protocol §4, §8, §11 |

The grid is unchanged (5,939 runs): the fix changes labels and masks only.
