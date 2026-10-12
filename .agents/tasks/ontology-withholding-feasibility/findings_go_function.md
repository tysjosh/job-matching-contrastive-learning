# GO protein-function prediction under label scarcity: feasibility findings

This is a read-only investigation. Nothing in the repo was modified and nothing was trained. Scratch scripts and outputs are in `/tmp/gofunc/`:

| script | what it does |
|---|---|
| `a_obo.py` | relation counts |
| `b_gaf.py` | GAF statistics |
| `c_leak.py` | lexical leak |
| `f_zeroshot.py` | untrained zero-shot AUC |
| `d_targets.py` | targets on a human random split |
| `e_relations.py` | label growth per relation |
| `g_dgz_prop.py` | DeepGOZero propagation rule on human labels |
| `h_dgz_targets.py` | targets on DeepGOZero's split |
| `i_decomp.py` | DeepGOZero label excess, decomposed |

Downloaded to `/tmp/gofunc` only:
- `go.obo` (37 MB, releases/2026-07-26);
- the DeepGOZero BP and MF `train_data.pkl`/`test_data.pkl` and `terms*.pkl` (about 165 MB in total).

The local `go-basic.obo` is also data-version releases/2026-07-26. The local `goa_human.gaf.gz` was generated 2026-05-21.

Evidence tags:
- **[read]**: from code or docs.
- **[ran]**: computed here.
- **[web]**: from a URL.
- **[inferred]**: reasoned, not verified.

## Summary answer

1. **Go, but only as a re-scoped study.** The ontology question should be "does correctly typed GO structure help scarce-term prediction?", measured with retention curves and CIs, with DeepGOZero (DGZ) as the comparator, on DGZ's own released split.
   - "Ontology helps scarce terms" alone is not new.
   - The repo's PPI dual-encoder trainer is not the right vehicle.
2. **The repo's protein text cannot be the protein feature.** STRING `protein.info` text is close to circular for GO.
   - An *untrained* frozen all-mpnet-base-v2, scoring proteins by cosine between their STRING text and the GO term's name plus definition, gets a mean class-centric AUC of **0.781 (BP), 0.827 (MF) and 0.805 (CC)** over 200 random terms per aspect. TF-IDF gets 0.69 / 0.76 / 0.74 [ran].
   - That is already above DGZ's reported zero-shot mean of 0.745.
   - Protein features must be sequence-derived. Recommended: ESM-2 650M embeddings as primary, with DGZ's InterPro vectors for an exact like-for-like comparison.
3. **Typed relations matter in practice, and DGZ conflates them.**
   - DGZ's label propagation (`utils.Ontology(with_rels=True)`) treats *every* `relationship:` line as `is_a`, including regulates, positively/negatively_regulates, has_part and occurs_in, across namespaces [read, web].
   - On human curated labels, that rule adds **+35% BP, +22% MF and +11% CC labels** over the true-path rule (is_a+part_of) [ran].
   - In DGZ's released BP training data, every protein labelled with each of their 7 withheld zero-shot BP classes (all "positive regulation of X") is also labelled with the filler class X. **37–84% of X's training positives carry the withheld class** [ran]. So the "zero-shot" BP classes were predicted from a filler class trained largely on their own positives.
   - This gives a concrete, testable contribution: correct typed propagation plus a leak-free zero-shot/withholding protocol.
4. **The data supply is ample on DGZ's split** (SwissProt 2021_04, 50% sequence-identity groups, InterPro and sequences included, files individually downloadable).
   - Terms with ≥30 train and ≥10 test proteins, covering ≤5% of train: **1,934 BP (673 with a go.obo logical definition) and 408 MF** [ran].
   - The human-only random split gives 616 BP / 171 MF / 164 CC, but has no sequence grouping.
5. **Recommended target:** class-centric binary prediction of a held-out term t on sequence-disjoint test proteins (formulation (a)). Use a bilinear "dual-encoder" scorer, which nests the MLP, DGZ and text-term-encoder variants in one code path.
   - **Scarcity intervention:** per-term *label* withholding. Proteins stay in training; the t-subtree labels are erased for withheld proteins and their labels re-propagated. Retention k ∈ {0, 1, 5, 10, 25, all}.
   - **Stratify targets** by n(t) × structural support (is_a-sibling support; logical definition with a well-annotated filler).

---

## 1. Prediction target

| | (a) multi-label protein → term (CAFA / DGZ) | (b) contrastive protein ↔ term-text retrieval | (c) PPI with a GO regulariser (prior repo study) |
|---|---|---|---|
| Query / candidate | protein / every term of a sub-ontology (one sigmoid per term) | protein / term embedding g(t) = proj(text-encoder(name + def)); InfoNCE over sampled terms | protein / protein |
| Ontology entry points | label propagation; output consistency; axiom loss on term embeddings (DGZ NF1–NF4) | as (a), plus hierarchy-aware negative terms (siblings), plus text neighbours | GO only as a regulariser or negatives; the supervisor's point (2) applies |
| Zero-shot possible? | only with a term embedding from axioms (DGZ) or text | yes, by construction (BioTranslator / ProTranslator / STAR-GO style) | n/a |
| Fit to repo trainer | none: the repo model is one *shared* frozen text encoder + one shared projection head for both slots (`contrastive_learning/trainer.py` L1483–1537) [read] | partial in spirit; needs two towers (ESM vs MPNet) and a term-vocabulary softmax, which the trainer lacks | direct (already built) |
| Cleanest scarcity test | withhold t's labels; class-centric AUC/AUPR on t over test proteins vs k | same; k=0 is natural zero-shot | per-target withholding (built, not run) |

**Recommendation: (a) as the target and evaluation, scored by a bilinear head s(p, t) = ⟨f(x_p), g(t)⟩ + b_t.** [inferred]
- g(t) can be:
  - a free vector, which reproduces the DGZ "MLP" baseline;
  - a DGZ-style n-ball centre constrained by EL axioms;
  - a projection of frozen MPNet(name + definition), which is (b)'s term tower;
  - text plus an axiom loss.
- Training uses BCE over the term vocabulary, as DGZ does, so the only thing that differs across arms is the ontology factor. Contrastive InfoNCE with sampled negative terms is kept as one optional arm for hierarchy-aware negatives.
- This avoids a "use X or Y" choice: one evaluation target, one code path, and ontology factors as switches.

**How this differs from DGZ (what is new):**
1. **Typed relations, done correctly.**
   - True-path propagation (is_a+part_of only, no regulates) for labels.
   - Separate relation embeddings in the axiom loss.
   - An ablation ladder (is_a → +part_of → +regulates family → full go.norm axioms) plus an "untyped-collapsed" control that reproduces DGZ's conflation.
2. **A leak-free scarcity protocol.** Erase the t-subtree and re-propagate (DGZ removed only direct annotations before propagation and did not remove descendants: `zero_data.py` [read]). Use controlled retention curves k = 0…25 instead of a single zero-shot point on 16 classes.
3. **Uncertainty.** Five seeds, plus a two-way bootstrap over targets × seeds. DGZ reports none.
4. **Moderators.** Neighbourhood support S(t) and logical-definition support are pre-registered moderators. DGZ only notes that classes with definition axioms do better.
5. **Mechanism separation.** Train-time axiom loss vs text-encoded term embeddings (zero-shot without axioms) vs inference-time structure fusion.

## 2. Protein features and leakage

**Lexical leak [ran, `c_leak.py`].**
- Setup: 1,500 proteins per aspect, direct curated annotations (evidence EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC, HTP, HDA, HMP, HGI, HEP; `GO:0005515` and `GO:0005488` excluded).
- Comparison: frequency-matched random terms from outside the protein's closure.
- "Key-token coverage" = the share of the term name's content tokens (≥4 characters, stop-words removed) found in the STRING text.

| aspect | exact name, true / random | name or exact/narrow synonym, true / random | key-token coverage ≥0.5, true / random | coverage = 1, true / random | proteins with any direct term fully covered |
|---|---|---|---|---|---|
| BP | 2.1% / 0.01% | 3.3% / 0.03% | **34.9% / 4.1%** | 10.7% / 0.7% | 31.5% |
| MF | 1.8% / 0.04% | 3.3% / 0.04% | **47.3% / 9.4%** | 21.5% / 1.5% | 41.8% |
| CC | 5.3% / 0.12% | 6.9% / 0.21% | **20.7% / 3.7%** | 9.4% / 0.4% | 23.9% |

**Untrained zero-shot ranking [ran, `f_zeroshot.py`].** 200 random terms per aspect, each with 25 to 5%-of-proteins propagated members. The score is cosine(protein text, term text).

| aspect | proteins | TF-IDF name | TF-IDF name + def | frozen MPNet name + def | MPNet share of terms with AUC > 0.8 |
|---|---|---|---|---|---|
| BP | 11,912 | 0.657 | 0.686 | **0.781** | 43.5% |
| MF | 11,254 | 0.729 | 0.762 | **0.827** | 66% |
| CC | 14,354 | 0.685 | 0.741 | **0.805** | 53% |

- With no learning, the STRING text plus the GO definition already beats DGZ's zero-shot (0.745) and approaches trained class-centric AUCs. DGZ MF averages are 0.882 (MLP) and 0.903 (DGZ) [web: PMC9235501 Table 3].
- STRING annotation text is UniProt function text, which curators write from the same literature as GO annotations [inferred]. **Excluded as a protein feature. Text is allowed only on the term side.**

**Leak-free options:**

| feature | source / size / licence | cost on L40S | DGZ-comparable? |
|---|---|---|---|
| InterPro binary vectors | already inside DGZ's `train/valid/test_data.pkl` (column `interpros`; 97.5% BP and 98.5% MF train proteins have ≥1) plus `interpros.pkl` vocabulary (318 KB) [ran, web]. Fresh human: UniProt cross-references or InterProScan (CPU). Freely available [web: EBI terms; not re-checked] | zero (precomputed) | **exact** |
| ESM-2 650M mean-pooled (1280-d) | fair-esm / HF `facebook/esm2_t33_650M_UR50D`, MIT licence; about 2.5 GB weights. Sequences: DGZ `swissprot_exp.fa` (41.9 MB) or the `sequences` column [web, ran] | about 77.6k sequences, truncated at 1,022: about 20–40 min fp16; 3B about 2–3 h. Store about 400 MB float32 [inferred] | same split; DeepGO-SE already uses ESM-2, so comparable to DeepGO-SE |
| ProtT5-XL-U50 (1024-d) | Rostlab HF, about 11 GB fp32 (half-precision encoder about 2.4 GB) [inferred] | about 1–2 h | no direct DGZ number |
| Raw sequence (CNN/LM fine-tune) | UniProt (CC BY 4.0) | high | DeepGOCNN baseline only |

**Recommendation:**
- ESM-2 650M embeddings as the primary feature (strong, cheap, leak-free).
- InterPro vectors as a mandatory secondary, so DGZ and MLP run on their exact published inputs.
- Note: InterPro and ESM can still carry homology to training proteins. That is why the sequence-grouped split matters (§5).

## 3. Annotations (`goa_human.gaf.gz`)

[ran, `b_gaf.py`]

**Volume:**
- 906,445 lines: UniProtKB 858,661, RNAcentral 38,226, ComplexPortal 9,558. Use UniProtKB only.
- NOT qualifiers: 1,501 lines, to be dropped.
- Proteins with any annotation: P 31,606 / F 27,079 / C 32,428.

| evidence group | P | F | C |
|---|---|---|---|
| experimental (EXP, IDA, IPI, IMP, IGI, IEP) | 67,210 | 329,522 (IPI 288,813) | 60,410 |
| high-throughput (HTP, HDA, HMP, HGI, HEP) | 280 | 2,027 | 8,843 |
| TAS / IC | 12,559 | 8,691 | 90,409 |
| IEA | 65,031 | 51,663 | 78,233 |
| ISS, IBA, ISO, ISA, … | 45,982 | 25,021 | 34,306 |
| NAS | 13,755 | 4,160 | 8,043 |

**Proteins with ≥1 label and propagated term frequency** (is_a+part_of, root excluded):

| label set | aspect | proteins | terms with ≥1 / ≥10 / ≥25 / ≥50 / ≥100 / ≥500 proteins |
|---|---|---|---|
| EXP | P | 11,641 | 10,828 / 3,845 / 2,092 / 1,240 / 690 / 156 |
| EXP | F | 16,692 (5,547 have only protein binding) | 4,596 / 1,026 / 520 / 323 / 194 / 47 |
| EXP | C | 12,704 | 1,680 / 613 / 370 / 247 / 164 / 54 |
| EXP + HTP + TAS + IC (DGZ set) | P | 13,162 | 11,105 / 4,143 / 2,295 / 1,380 / 773 / 179 |
| same | F | 17,086 | 4,724 / 1,097 / 570 / 357 / 216 / 55 |
| same | C | 15,470 | 1,774 / 680 / 424 / 304 / 196 / 70 |
| DGZ set without IPI | F | 11,840 | 4,595 / 1,012 / 528 / 331 / 201 / 49 |

**Circularity and label choices:**
- IEA, ISS, IBA and similar codes are themselves predictions (often sequence- or InterPro2GO-based). **Exclude them from labels.** This matters especially with InterPro features, because InterPro2GO IEA would make labels a function of the features.
- IPI dominates MF: 88% of experimental MF lines. It is mostly "protein binding" and its children, and it is interaction-derived. Follow DGZ and keep IPI for comparability, but report MF without IPI as a sensitivity analysis, and drop `GO:0005515`-only proteins from MF evaluation (the CAFA convention used in DGZ's NetGO split [web]).
- TAS and IC are curator statements; DGZ includes them.

**Human-only vs DGZ SwissProt.**
- Human-only gives about 12k BP and 11k MF proteins, and has no sequences or InterPro locally.
- DGZ's dataset has 77,647 proteins; BP split train 47,733 (human 9,393) / test 5,444 (human 1,069); MF train 34,716 / test 4,712 [ran, web].
- **Use DGZ's split.** It is sequence-grouped, published and directly comparable.

**What the DGZ repo provides [web].**
- The tarball `https://bio2vec.net/data/deepgo/deepgozero/data.tar.gz` (8.32 GB) and `data-netgo.tar.gz` (2.54 GB). The README's KAUST URL now returns 404.
- An *individually browsable* `data/` directory containing:
  - `go.obo` (2021-11), `go.owl`, `go.norm` (normalised axioms, 4.8 MB), `definitions_go.txt`, `swissprot_exp.{pkl,fa,dmnd,sim}`;
  - per-ontology `train/valid/test_data.pkl` + `.fa`, `interpros.pkl`, `terms.pkl`, `terms_zero.pkl`, `terms_zero_10.pkl`, trained `.th` checkpoints, `predictions_*.pkl` (≤1.4 GB each) and `test_diamond.res`.
- Repo code was last committed 2022-09-19 and pins py3.9, torch 1.9 + cu111, dgl-cu111 0.7.1 and JPype (`requirements.txt`) [web].
- DeepGO-SE (successor, ESM-2 based) training data: `https://bio2vec.net/data/deepgo/deepgo2/training-data.tar.gz` (7.06 GB) [web].

## 4. Relationships

**Counts [ran, `a_obo.py`].** Live terms: BP 23,974, MF 10,041, CC 4,077.

| relation | go-basic BP | go-basic MF | go-basic CC | go.obo BP | go.obo MF | go.obo CC |
|---|---|---|---|---|---|---|
| is_a | 40,486 | 12,274 | 4,696 | 40,486 | 12,274 | 4,696 |
| part_of | 4,461 | 9 | 1,800 | 4,461 | 795 (786 → BP) | 1,802 |
| regulates | 2,820 | 0 | 0 | 2,890 (70 → MF) | 49 | 0 |
| positively_regulates | 2,475 | 0 | 0 | 2,563 (88 → MF) | 64 | 0 |
| negatively_regulates | 2,475 | 0 | 0 | 2,548 (73 → MF) | 76 | 0 |
| has_part | 0 | 0 | 0 | 328 (175 → MF) | 234 | 181 |
| occurs_in | 0 | 0 | 0 | 137 (→ CC) | 36 (→ CC) | 0 |
| terms with intersection_of (logical definition) | **0** | **0** | **0** | **8,820** | 114 | 343 |

- BP terms with any regulates edge: 7,766 in go-basic and 7,997 in go.obo.
- go.obo intersection_of differentia: regulates 2,905, positively_regulates 2,577, negatively_regulates 2,553, part_of 1,088, occurs_in 177, has_part 53. All fillers are GO classes.
- **go-basic** strips logical definitions, has_part, occurs_in and cross-namespace edges.
- **go.obo** has GO-internal logical definitions.
- **go-plus** adds external-ontology axioms (CHEBI, Uberon, CL); DGZ normalises go.owl to `go.norm`, and DeepGO-SE also uses `go-plus.norm` [web].
- **A DGZ-style axiom loss needs go.obo/go.owl, not go-basic.** The repo only has go-basic.

**Label propagation per relation** [ran, `e_relations.py`]. Human curated labels, propagated term–protein pairs:

| | is_a only | + part_of | + regulates family (invalid) |
|---|---|---|---|
| BP | 449,847 | 502,718 (+11.8%) | 618,446 (+23.0%; 6,252 of 12,171 proteins affected) |
| MF | 139,396 | 139,467 (+0.05%) | +0 |
| CC | 177,783 | 257,705 (+45.0%) | +0 |

**Valid for propagation (true path rule).**
- is_a and part_of are valid within a namespace.
- MF part_of BP (cross-aspect) is used by GO's own inference, but that is a separate decision. Keep it off for per-aspect labels.
- regulates does NOT imply the regulated process. has_part points the wrong direction (X has_part Y does not make X's annotatees members of Y in general). occurs_in is cross-aspect context.

**DGZ's rule** (`utils.py Ontology.load`, `with_rels=True` → `obj['is_a'].append(it[1])` for every relationship; used in `uni2pandas.py`, `deepgozero_data.py` and `zero_data.py`) [read, web].
- On human curated labels it adds +35.1% BP, +22.1% MF and +10.5% CC labels [ran, `g_dgz_prop.py`].
- On DGZ's own BP train set, their `prop_annotations` hold 2.51M term labels against 1.51M under the true path (+66%), and MF +31% [ran, `h_dgz_targets.py`].
- Decomposed with the current go.obo: regulates accounts for +289k BP labels, has_part/occurs_in for +118k, and cross-namespace edges for +40k. A residual of about 550k is unexplained, probably GO 2021-vs-2026 drift [ran, `i_decomp.py`; residual inferred].

**Required fix in `go_ppi_domain/go_ontology.py`.**
- `parse_obo` (L108–163) adds `is_a:` and `relationship: part_of` to one untyped `parents` set (L156–160) and ignores the rest.
- `withholding_targets.bp_graph` (L70–74) then computes siblings over that mixed graph.
- Fix: add a typed parser returning `{term: [(rel, parent, parent_ns)]}` for all relations, plus `intersection_of`. Then:
  - closure(rels=…, same_namespace=True);
  - is_a-only siblings;
  - keep the current function byte-identical for the PPI study.

**Typed treatments and ablation (proposed).**
- **Labels:** always true-path (is_a+part_of, within namespace) in every arm, so the evaluation target never changes.
- **Axiom loss** over term embeddings, with a separate relation vector per relation type (ELEmbeddings NF3/NF4 use a relation vector r):
  - L1: is_a only (NF1);
  - L2: + part_of (NF3 with r_part_of);
  - L3: + regulates / positively / negatively as three relation vectors;
  - L4: full go.norm (adds NF2 intersections and NF4) = the DGZ loss;
  - L3-collapsed: the L3 edges with a *single shared* relation vector (the conflation control).
- **Propagation control:** DGZ labels (all relations as is_a) vs true-path labels, run on DGZ and on the MLP.

## 5. Scarcity intervention

**Unit.** A target GO term t in one aspect. D*(t) = t plus all is_a/part_of descendants. P_train(t) = train proteins whose true-path closure contains t.

**Withholding rule (W2, primary).**
- For each p ∈ W = P_train(t) \ R_k, delete p's direct annotations in D*(t) and re-propagate (closure of what remains).
- p stays in training with all other labels, and it is a genuine negative only if the remaining annotations do not imply t. They cannot, by construction.
- Without re-propagation, a retained child annotation would re-imply t under the true path rule. DGZ's `zero_data.py` removes only the class's own direct annotations, so descendants still propagate to it [read].
- **Masking:** for p ∈ W, the D*(t) outputs are excluded from the loss (unknown, not negative). Non-P(t) proteins remain negatives at every k, so the curve changes exactly one thing: the number of positives.
- **W1 (secondary, realistic "coarse annotation"):** W proteins keep the ancestors of t that their original labels implied, as if curated at the parent level. W1 − W2 measures how much parent labels alone help.
- **Validation:** D*(t) is removed from validation labels too (target-free validation). Use a fixed epoch budget or a validation metric over non-target terms, never t (the lesson in `findings.md` on variant-specific checkpoint selection).
- **Retention:** R_k is nested and seeded within P_train(t), k ∈ {0, 1, 5, 10, 25, all}. Proteins are sampled from those with a *direct* annotation in D*(t).

**Target counts.**
- **On DGZ's 50% split, true-path labels** (n_train ≥ 30, n_test ≥ 10, n_train ≤ 5% of train) [ran, `h_dgz_targets.py`]:
  - BP: 1,934 terms (30–49: 50; 50–99: 300; 100–249: 813; 250+: 771), 673 with a logical definition.
  - MF: 408 terms (30 / 69 / 162 / 147), only 14 with a logical definition.
  - Naturally rare test terms (1–29 train, ≥3 test): BP 1,390, MF 424.
- **On a human random 80/10/10 split** [ran, `d_targets.py`]: BP 616 / MF 171 / CC 164.
  - BP S_sib quartiles (train proteins on is_a siblings, outside P(t)): 0 / 125 / 431 / 1,265 / 9,666. Spearman(n, S_sib) = 0.27 for BP and 0.03 for MF.
  - 215 BP targets have a logical definition whose filler has ≥30 train proteins; these are mostly regulation terms (243 of 616).

**Strata (pre-registered).** n(t) band × support:
- S_sib tertiles;
- for BP, a logical-definition flag with filler support. This is the DGZ mechanism; under true-path labels it is now leak-free.

**Packing.** Several targets can be withheld in one run if they are mutually non-nested, outside each other's 2-hop neighbourhood, and have protein Jaccard ≤ 0.2. This reuses `greedy_non_nested`/`jaccard` (§7). With about 1,900 BP candidates, 6–8 targets per pack is feasible [inferred].

**Sequence-similarity leakage.**
- A random protein split leaks through homologs. Human paralogue families share both InterPro and ESM features and GO terms.
- DGZ already groups at 50% identity with Diamond (and provides a 30% split); `swissprot_exp.sim` is provided.
- Reusing their split costs nothing. If we ever re-split (for example, human + new 2026 annotations), MMseqs2 `easy-cluster --min-seq-id 0.3/0.5` on about 80k sequences runs in minutes on CPU [inferred].
- Report the k-curves on the 50% split (primary) and the 30% split (sensitivity).

## 6. Baselines and comparators

**Required:**
- **Naive:** term frequency (DGZ "Naive").
- **DiamondScore:** use DGZ's `test_diamond.res` and the normalised bitscore transfer [web].
- **MLP on the same features:** InterPro and ESM.
- **DGZ:**
  - reproduce *their* numbers first, from released predictions and checkpoints (no training needed to verify the evaluation code);
  - then re-run inside our trainer with an EL-loss reimplementation validated against theirs on the full split (target: class-centric AUC within ±0.01 of 0.903 MF; Fmax within ±0.01);
  - then run DGZ under both label regimes (their propagation vs true path).
  - Running their py3.9/torch 1.9/dgl stack on an L40S (CUDA 12) is likely painful. Treat it as a reproduction check, not the experimental harness [inferred].
- **Our no-ontology model:** bilinear head with free term vectors (= MLP) and with a text term encoder (no axioms).

**Contextual** (cite, and optionally run one):
- DeepGO-SE (ESM-2 + axioms; Nature Mach. Intell. 2024, https://www.nature.com/articles/s42256-024-00795-w; https://github.com/bio-ontology-research-group/deepgo2).
- Text-description zero-shot: ProTranslator (https://arxiv.org/abs/2204.10286), BioTranslator (Nat. Commun. 2023, https://www.nature.com/articles/s41467-023-36476-2), STAR-GO (text + structure, zero-shot; arXiv 2512.05245, https://arxiv.org/abs/2512.05245).
- TALE (label embedding with hierarchy; https://academic.oup.com/bioinformatics/article/37/18/2825/6182677).
- TransFew (https://pubmed.ncbi.nlm.nih.gov/39233898/).
- **TransformerGO** (Ieremie, Ewing and Niranjan, Bioinformatics 2022, doi 10.1093/bioinformatics/btac104). A PPI comparator only, not a function baseline. It embeds GO terms with node2vec on an *undirected, untyped is_a + part_of* graph, a further example of relation conflation. It keeps IPI and uses pair-random rather than protein-disjoint splits (details as relayed by the parent session; not re-read here).

**A convincing comparison needs, at minimum:**
1. identical split, features, label set and evaluation code for every arm;
2. DGZ reproduced on its own split before modification;
3. per-target retention curves with seed × target CIs;
4. the typed-relation ladder plus the collapsed control;
5. the leak-free W2 protocol applied to DGZ too, with DGZ's own zero-shot protocol reported as a sensitivity.

## 7. Reuse vs new

**Reusable as-is or near as-is [read]:**
- `scripts/go_withholding/analyze_withholding.py`:
  - `paired_deltas`, `two_way_bootstrap` (packs × seeds), `hierarchical_bootstrap`, `holm`, `classify_contrast`, `decide`;
  - needs a long table keyed (target, k, variant, seed) with `class_auc`/`aupr` instead of `hard_auc`.
- `scripts/go_withholding/sweep.py`:
  - grid, sharding, resume, claim/release, pins, code hash and `FAILED.json` handling;
  - replace `runner_cmd`/`eval_cmd` (L564–586) and the PPI-specific `pin_preflight` checks.
- `go_ppi_domain/withholding_targets.py`:
  - `greedy_non_nested`, `jaccard`, `assign_strata`, `match_strata`, `_smd`, `apply_size_rule`, `singleton_packs`, `n_sib`/`n_2`;
  - feed them a typed is_a graph instead of `bp_graph`.
- `go_ppi_domain/withholding.py` patterns: deterministic manifests with sha256 (`_write_or_verify`, `table_sha256`) and assertion style (`assert_condition`), rewritten for label matrices.
- `go_ppi_domain/withholding_diagnostics.py`: `projection_params`, `_grad_norm` and `EpochDiagnostics`, for per-loss-component gradient-norm logging. They need a new trainer hook.
- `scripts/go_withholding/lambda_pilot.py`: the λ-pilot pattern for setting the axiom-loss weight on target-free validation.

**Not reusable:**
- data_converter, data_splitter, negative_selector, record_adapter, leak_pruning: these are PPI and text-specific. Leak pruning is moot once text is not a protein feature.
- `evaluate_withholding.py` metrics (graded PPI AUCs).
- `contrastive_learning/trainer.py`: a shared single-tower projection with PPI records.

**New:**
1. Typed OBO parser and closure (go.obo and go.norm loader).
2. DGZ data loader with true-path relabelling.
3. Feature builders (InterPro from pkl; ESM-2 embedding script).
4. Bilinear multi-label trainer with switchable term tower (free / EL n-balls / text), typed EL loss (NF1–NF4 with relation vectors) and masking.
5. Withholding generator for label matrices (W2/W1, nested R_k).
6. Class-centric evaluator: per-term AUC/AUPR, Fmax/Smin/AUPR via the CAFA formulas DGZ uses (`evaluate.py`, `evaluate_terms.py`), and non-target collateral.
7. DGZ reproduction harness from released predictions.

## 8. Compute and data size (remote L40S, 48 GB)

[inferred unless noted]
- **Data:**
  - DGZ per-ontology pickles 86 + 10 + 10 MB (BP) and 66 + 9 + 9 MB (MF) [ran]; fasta about 40 MB; go.norm 4.8 MB; go.owl 132 MB.
  - Skip the multi-GB `predictions_*.pkl` except for one reproduction check (BP about 0.94 GB each).
- **ESM-2 650M embeddings** for about 77.6k proteins: about 0.5 h; store about 0.4 GB.
- **Training one model:**
  - inputs 47.7k × 1280 (or 26k sparse InterPro); outputs 21.4k BP terms.
  - The bilinear head is 21k × 1024 parameters. DGZ checkpoints are about 298 MB.
  - About 0.5–2 s per epoch on GPU, so roughly 2–5 min per run including the EL loss over about 50–100k normal-form axioms.
- **Grid:**
  - per aspect, 48 targets (2 support strata × 2 logical-definition strata × 12, n-matched) in 8 packs of 6;
  - 6 k values (the "all" model is shared), 8 variants, 5 seeds;
  - (8 × 5 + 1) × 8 × 5 = 1,640 runs, about 55–140 GPU-h for BP, and less for MF.
  - Several runs fit concurrently in 48 GB, so wall time is about 1–2 days.
- **Evaluation** on 5.4k test proteins is seconds per run.

---

## Conclusions and recommendations

### (i) Recommended study design (one page)

- **Question:** under per-term label scarcity, does *correctly typed* GO structure improve prediction of the scarce term on sequence-novel proteins? Which mechanism helps: axiom loss, text term embedding or inference fusion? Is the effect moderated by structural support?
- **Data and split:**
  - DGZ UniProtKB-SwissProt 2021_04 experimental set, using their 50%-identity split (primary) and 30% split (sensitivity).
  - Ontology: their go.obo / go.owl / go.norm (2021-11), so DGZ is reproducible.
  - Optional external test (open decision): GOA 2026 annotations newly added to DGZ test proteins.
- **Aspects:**
  - BP is primary: the only aspect with is_a, part_of and regulates and 673 logically defined targets.
  - MF is secondary: the cleanest "function", near-pure is_a. Report it without IPI as a sensitivity.
  - CC is optional: the part_of-heavy case.
- **Labels:**
  - Evidence: EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC, HTP, HDA, HMP, HGI, HEP (DGZ set), NOT dropped.
  - Propagation: true path, is_a + part_of, within namespace.
  - Term vocabulary: terms with ≥1 train annotation, as in DGZ, after relabelling.
  - MF evaluation drops proteins annotated only to protein binding.
- **Features:** ESM-2 650M mean-pooled (primary); InterPro binary (DGZ-exact secondary). No STRING or UniProt text on the protein side.
- **Model:** s(p, t) = σ(⟨f(x_p), g(t)⟩ + b_t), with f a 2-layer MLP and residual (DGZ MLPBlock) and BCE over the vocabulary.
- **Variants:**

| arm | term tower g(t) | ontology use |
|---|---|---|
| V0 | free | none (= MLP) |
| V1 | EL n-balls | is_a NF1 |
| V2 | EL n-balls | + part_of (typed r) |
| V3 | EL n-balls | + regulates family (3 typed r) |
| V4 | EL n-balls | full go.norm (= DGZ) |
| V3c | EL n-balls | V3 edges, one shared r (conflation control) |
| V5 | text: frozen MPNet(name + def) → projection | none (zero-shot without axioms) |
| V6 | text | + best typed EL loss |
| F | V0 at inference | k=0/low-k score from parent/sibling heads (inference fusion; no retraining) |

- **Propagation control:** V0 and V4 trained on DGZ-rule labels vs true-path labels.
- **Scarcity:** W2 withholding (D*(t) erased, re-propagated, masked) with nested R_k, k ∈ {0, 1, 5, 10, 25, all}. W1 as secondary. Validation is target-free; use a fixed epoch budget or non-target validation loss.
- **Targets:** per aspect, 48 n-matched targets in strata of S_sib (low/high) × logical definition with filler support (yes/no) for BP; S_sib only for MF. Packed 6 per run (non-nested, non-neighbouring, protein Jaccard ≤ 0.2).
- **Metrics:**
  - primary: class-centric AUC of t on test proteins vs k;
  - secondary: class AUPR (imbalance-robust);
  - collateral: protein-centric Fmax, Smin and AUPR over non-target terms;
  - mechanism: ΔAUC(Vx − V0) at each k, and the S(t) moderation slope.
- **Uncertainty:** 5 seeds; paired deltas per (target, k, seed); two-way bootstrap over packs × seeds (`two_way_bootstrap`); Holm across pre-registered contrasts. Pre-registered primary contrasts:
  - V3 − V3c at k ≤ 5 (typed vs untyped);
  - V4 − V0 at k = 0 under W2 (does DGZ zero-shot survive the leak fix?);
  - V5 − V0 and V6 − V5 at k ≤ 5;
  - support × (Vbest − V0).

### (ii) Positioning against DeepGOZero (what is genuinely new)

- **DGZ established that** EL axioms help specific (low-annotation) classes and allow zero-shot. Its evidence:
  - 16 withheld classes with ≥100 annotations: mean zero-shot AUC 0.745 vs 0.898 trained;
  - classes with fewer than 10 annotations: 0.804 / 0.737 / 0.819 (MF / BP / CC);
  - no seeds or CIs.
- **New here:**
  1. **Typed-relation correctness as a measured factor.** DGZ propagates labels through regulates, has_part and occurs_in as if they were is_a (+66% BP labels on their own split [ran]). Its 7 BP zero-shot classes are all "positive regulation of X", whose filler X was trained on labels that include 100% of the withheld class's positives (X's positives are 37–84% withheld-class proteins) [ran].
     - DGZ's BP zero-shot mean (about 0.80) is far higher than its MF mean (about 0.61) [web, Table 8]. That pattern is consistent with this leak, but not proof of it [inferred].
     - We test it directly: V4 under DGZ labels vs true-path labels at k = 0.
  2. **Controlled retention curves** (k = 0…25) on hundreds of stratified targets with uncertainty, instead of one zero-shot point and an observational annotation-count plot (DGZ Fig. 3/4).
  3. **Mechanism separation:** axiom loss vs text-defined term embeddings (the BioTranslator/STAR-GO family, never compared to DGZ under a controlled withholding protocol [inferred from abstracts]) vs inference-only fusion.
  4. **Pre-registered structural-support moderation:** S_sib and logical-definition filler support.
  5. **A documented leak** of protein function text, which bears on text-based protein encoders.
- **What is not new, and should not be claimed:** the architectures (EL n-balls, text term encoders, ESM features) and "ontology helps rare terms" in general.

### (iii) Go / no-go and main risks

**GO, conditional on** the DGZ split and data being usable as downloaded (individual files are reachable today [ran]) and on dropping protein text.

Main risks:
1. **The leak finding is partly an artefact of GO-version drift.** It is verified on DGZ's own released pickles (regulates propagation is present in `prop_annotations`). Whether their zero model was trained on `prop_annotations` with `terms_zero.pkl` is inferred from code (`deepgozero.py load_data/get_data` uses `prop_annotations`), not from a run log.
2. **The typed-relation effect may be small after true-path relabelling.** That is still a publishable negative only with CIs. The ladder must be powered: 48 targets × 5 seeds.
3. **The EL-loss reimplementation may not match DGZ.** Mitigate by evaluating their released predictions with our evaluator first.
4. **ESM features may make scarcity effects vanish** (strong homology transfer). The 30% split and the stratification by S(t) address this.
5. **Scope creep across 3 aspects and 9 arms.** Fix BP as primary and MF as secondary, and pre-register.
6. **Novelty overlap with STAR-GO / DeepGO-SE.** Positioning must rest on the protocol and typed relations, not the model.

### (iv) Ordered implementation gap list

1. Pin inputs: download DGZ `data/` (go.obo, go.owl, go.norm, definitions, swissprot_exp.fa) and the bp/mf/cc `{train,valid,test}_data.pkl`, `interpros.pkl`, `terms*.pkl`, `test_diamond.res`. Record sha256 values. Commit or tag the code.
2. Typed ontology module: parse go.obo with all relations + intersection_of; per-relation closures; is_a-sibling and 2-hop neighbourhoods; a go.norm loader (NF1–NF4 with relation ids). Leave `go_ppi_domain/go_ontology.parse_obo` byte-identical.
3. Label builder: true-path relabelling from `exp_annotations`, with a DGZ-rule option. Unit tests: regulates is never propagated; descendants imply ancestors; the DGZ rule reproduces their `prop_annotations` on a sample.
4. Evaluator: class-centric AUC/AUPR, Fmax/Smin/AUPR, ported from DGZ `evaluate.py`/`evaluate_terms.py`. **Gate:** reproduce DGZ Table 3 numbers from their released `predictions_*.pkl` (MF and BP).
5. Features: an InterPro matrix from the pkl; an ESM-2 650M embedding script (L40S, fp16, cached .npy with hashes).
6. Trainer: bilinear multi-label model with term towers (free / EL / text), typed EL loss, loss masking, fixed-epoch or non-target validation, and per-component grad-norm logging. **Gate:** the V0 = MLP and V4 = DGZ full-data numbers are within ±0.01 of DGZ's (InterPro features, their labels).
7. Target table: candidates, non-nesting, strata, n-matching, packs. Reuse `withholding_targets` helpers on the typed graph. Freeze it with a hash.
8. Withholding generator (W2/W1, nested R_k, target-free validation) with manifest assertions: no D*(t) positive for W proteins after re-propagation; |R_k| = k; test labels unchanged.
9. λ pilot for the axiom-loss weight on target-free validation (pattern from `lambda_pilot.py`).
10. Sweep adaptation (`runner_cmd`/`eval_cmd` and pins), then a smoke run: 1 pack × k ∈ {0, 5} × V0/V3/V3c × 1 seed.
11. Analysis: adapt the `analyze_withholding` long table and contrasts. Pre-register the protocol before the grid.

### (v) Open decisions for user and supervisor

1. **Primary aspect:** BP (where typed relations bite; recommended) or MF (the cleaner "function"; little typed structure).
2. **Features:** ESM-2 primary with InterPro secondary (recommended), or InterPro only (pure DGZ comparability, cheaper).
3. **Keep IPI in MF labels** (DGZ comparability), or exclude it (interaction-derived; supervisor's point 2).
4. **Withholding:** W2 primary with W1 secondary (recommended), or W1 primary (the realistic "annotated at parent level" case).
5. **Include the text term tower (V5/V6)?** Recommended, because it is the zero-shot-without-axioms mechanism and it links to the repo's text-encoder work.
6. **External time split:** whether to add a time-split external test (GOA 2026 annotations on DGZ test proteins and new terms).
7. **How hard to push the DGZ leak finding.** Report it as a central result, which needs a direct check with the authors' zero model (we can score their released `deepgozero_zero.th` checkpoint), or as a secondary methodological note.

### (vi) What was not verified

- **DGZ zero-model training data.** Which data their `deepgozero_zero.th` was actually trained with: inferred from code paths, not from logs. Their `zero_data.py` has an early `return` before writing outputs, so the published code does not itself produce `zero_annotations`.
- **The ~550k residual** of DGZ label excess not explained by relation type: attributed to GO 2021 vs 2026 drift without checking against their 2021 go.obo, which is downloadable (33.8 MB) and is a cheap next check.
- **Compute and licence figures.** ESM-2/ProtT5 throughput and per-run training times on an L40S are estimates. The InterPro/UniProt licence terms were not re-read.
- **Prior-work details.** STAR-GO, BioTranslator, ProTranslator, TransFew and DeepGO-SE were reviewed from abstracts and READMEs only. TransformerGO details are as relayed by the parent session.
- **Lexical and zero-shot leak scope.** These were measured on the human GOA + STRING text; they were not measured on the SwissProt multi-species set. The zero-shot AUC used 200 sampled terms per aspect and one random seed.
- **Target viability.** The human-split counts use a random protein split with no sequence grouping. The DGZ-split counts use current-GO true-path labels on their 2021 annotations.
