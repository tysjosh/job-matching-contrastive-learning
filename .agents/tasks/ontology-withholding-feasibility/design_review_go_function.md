# Design review (round 3): GO protein-function prediction under per-term label scarcity

Reviewed: `design_go_function.md` (v3) and `protocol_go_function.md` (v3 DRAFT), against the spec `findings_go_function.md` ((i) design, (iii) risks, (iv) gaps, (v) decisions, (vi) not verified) and the round-2 review. Supporting reads: `setup_baseline_go_function.md` (pinned hashes), `scripts/go_withholding/analyze_withholding.py` and `go_ppi_domain/withholding_targets.py` (vendored helpers). Nothing was run, built or modified.

Verdict: **CHANGES_REQUESTED** (1 HIGH, 0 MEDIUM, 4 NIT).

v3 closes all three round-2 blocking findings:

- The alt-id exclusion plus T_cols(t) masks and assertions 8–9 make P_TP(t) ⊆ P_DGZ(t) hold for every target and close the duplicate-column leak.
- Per-regime direct sets (A_DGZ raw, all namespaces) fix cross-namespace re-implication.
- The C3_leak channel reads plus the two-level attribution keep descendant-head effects from being reported as typed-relation evidence.

The remaining blocking issue was present in every version and is a leak of the W2 rule itself. The loss mask that marks withheld positives as "unknown" is target information, so k = 0 under W2 is not zero-shot.

## Findings

### 1. HIGH: the W2 mask leaks P(t) into training, so k = 0 is not zero-shot and arm contrasts are biased

Where: design §5.2 (W2: "Mask, for p, the columns D*_ρ(t) ∪ Lost(p)"), §5.5 assertion 1, §6.5 (masked BCE); protocol §5 W2 step 3, D9, risk 6, C1/C2/C4/C5 (all include k = 0), C3 (TP/W2 side) and C3_col; spec §5 ("Masking").

The problem: for column t at numeric k, the unmasked rows are R_k (positives) and every non-P(t) protein (negatives). The masked rows are exactly W = P(t) \ R_k. So the mask pattern is itself the withheld label, and the model is told which rows *not* to push down.

- With a jointly trained encoder f and a per-term vector g(t), BCE on negatives only pushes ⟨f(x), g(t)⟩ down on every unmasked row. Those gradients also shape f. The W rows get no such push, so after training the region of feature space around W scores relatively high.
- Test t-positives resemble W (same function, related sequences). They inherit that relatively high score.
- This is the one-class / PU "hole in the negatives" effect. It can give AUC > 0.5 at k = 0 with no ontology and no text at all. The same holds for the D*(t) descendant columns, and more weakly for Lost(p) parent columns, which feed fusion F, the EL ball-containment path and C3_leak_desc/anc.

Consequences:

1. **Zero-shot claims are not zero-shot.** C2 ("DGZ-style axioms give a zero-shot benefit that survives the leak-free protocol") is measured at a k = 0 where column t was trained against an oracle-defined negative set. DGZ's own protocol, and WDGZrel and W2drop here, drop the column on every row, which carries no such information.
2. **Contrasts are biased, not just shifted.** The arms differ in how freely g(t) can use the mask:
   - V0 and the EL arms have a free per-term vector e_t;
   - V5's g(t) = P·m_t goes through a projection shared by all 21k terms.
   So C4 = V5 − V0, C5 and C2 inherit an arm-dependent bias of unknown sign. The pairing does not cancel it.
3. **C3_col changes meaning, and risk 6 is mis-stated.** Risk 6 says training t's column on negatives "can depress zero-shot scores". Under masking it can also inflate them, so the sign of C3_col (W2drop − W2) is not interpretable as "drop vs negatives-only".
4. **The realism argument for D9 runs the other way.** In real annotation scarcity, the unannotated true positives are *not* known. Standard closed-world BCE, which DGZ also uses, trains on them as negatives. Masking uses knowledge that no real scarce-term setting has.

Fix (no new runs; the grid count is unchanged). Adopt closed-world withholding as the primary W2, and record it as decision D19, a deviation from spec §5 "Masking", for user/supervisor sign-off before G8:

```text
W2 (closed-world). For each pack target t and p ∈ W_ρ(t):
  A_ρ(p) ← A_ρ(p) \ D*^GO_ρ(t); labels ← closure_ρ(A_ρ(p)) ∩ V
  No mask is added for p. Erased entries (D*_ρ(t) and Lost(p)) are ordinary 0 labels in the BCE,
  exactly as for a protein never annotated in t's subtree.
W1 (closed-world). As W2, then y[p, c] = 1 for c ∈ (ancestors_ρ(t) \ {t}) ∩ V. No mask.
WDGZrel: unchanged (T_cols(t) masked on every train row, which is information-free).
W2drop: W2 (closed-world) labels + T_cols(t) masked on every train row.
```

Then update the following:

- §5.5 assertion 1: "y[p, c] = 0 and (p, c) is not masked for all c ∈ D*_ρ(t)".
- Assertion 7: the W2drop mask equals {(p, c) : p ∈ train, c ∈ T_cols(t)}.
- Add assertion 10: in W1/W2, the train mask is empty, and in WDGZrel/W2drop every masked column is masked on all train rows. A mask may never be row-selective on a target-related column.
- Rewrite D9: W1 − W2 now reads "parent-level annotation vs no annotation in the branch".
- Rewrite risk 6: W2 trains column t on all-negatives at k = 0, which can depress it, and C3_col measures it.
- Rewrite spec §5's "the curve changes exactly one thing" as "the number of annotated positives; withheld positives become unannotated (closed-world negatives)".

If the user keeps the masked rule instead, it must be renamed W2mask and declared an oracle "known-unknown" setting in §1, §9.2 and the report. C2, C4 and C5 may then not be worded as zero-shot. A closed-world W2 must also be added as a descriptive control for V0, V4 and V5 at k ∈ {0, 1, 5} (360 runs per aspect). The first option is recommended.

### 2. NIT: assertion 8 has a moot clause

Design §5.5 assertion 8 says "under WDGZrel/W2drop no unmasked positive is allowed at all" inside a sentence about k = all, but both rules exist only at k = 0. Fix: "Under WDGZrel/W2drop (k = 0 only), no column with primary(c) = t has any unmasked train entry."

### 3. NIT: LD filler support is undefined when the filler is not a column

Design §4 step 3 and protocol §4 require "n_train(F) ≥ 30 under TP labels". n_train is defined only for vocabulary columns. Fix: "n_train(F) = |{p ∈ train : F ∈ closure_TP(p)}|, computed from the closure whether or not F ∈ V." Alternatively: "F ∉ V counts as n_train(F) = 0."

### 4. NIT: max-over-columns channel reads scale with channel size

C3_leak_desc takes the max over Desc(t), which can be hundreds of BP columns, while X(t) and Pa(t) are small. A max over many weak heads is noisier and biased upward relative to small sets, so the three reads are not comparable in magnitude. Fix: also report |Desc(t)|-stratified (≤ 5 vs > 5 columns) means of C3_leak_desc. Add a sentence that the channel reads are compared by CI sign only, never by size.

### 5. NIT: wording

Protocol §8 has "…of the same (pack, k = 0, seed); Its components…" with a stray capital. Design §0.2 says the alt-id exclusion removes "14 BP and 5 MF". Protocol §4 says "about 14 BP and 5 MF" (before the n_test filter). Use the same wording in both.

## Verified assumptions

- Pinned raw-input hash prefixes in protocol §12 match `setup_baseline_go_function.md` for every listed file [read].
- The vendored helpers exist with the stated names: `jaccard` in `go_ppi_domain/withholding_targets.py`, and `_counts`, `two_way_draws`, `_summarise`, `two_way_bootstrap`, `hierarchical_bootstrap`, `holm`, `classify_contrast` and `ci_excludes_0` in `scripts/go_withholding/analyze_withholding.py` [read]. The originals read `wc.ANALYSIS_SEED` as a default, consistent with design §13.2 making the constants explicit.
- Round-2 Finding 1 (alt ids) is resolved:
  - the exclusion removes every target with an alt-id column or alt-id annotation;
  - check 3d (TP-only columns ⊆ AltCols ∪ PrimAlt(p)) then implies t ∈ cols_TP(p) ⇒ t ∈ cols_DGZ(p), so P_TP(t) ⊆ P_DGZ(t) holds for targets;
  - assertions 8–9 enforce it per condition.
- Round-2 Finding 2 (cross-namespace) is resolved. A_DGZ is the raw `exp_annotations`, D*^GO_DGZ is defined on raw strings through `DgzRule.closure`, and W2 erases from A_ρ. A fixture test with a planted aspect-filtered erasure must fail.
- Round-2 Finding 3 is resolved. X(t), Desc(t) and Pa(t) are stored by every V4 k = 0 run, and the counts check out: BP 200 ESM + 160 InterPro = 360; MF adds noIPI V4 k = 0 (40) = 400. The two-level attribution in §9.2 and the split §9.3 table are consistent.
- Pack-level disjointness of D*^GO under both regimes stops a "regulation of u" target t and its filler u from sharing a pack, because t ∈ D*^GO_DGZ(u) ∩ D*^GO_DGZ(t). Two-hop over TRUE_PATH alone would not.
- With pack-mates withheld together, erasing D*^GO(t₂) from a protein retained for t₁ cannot remove t₁'s implying annotations, because the D*^GO sets are disjoint. So assertion 2 (exactly k positives) holds.
- The C3 telescoping identity is exact, and the four-run and two-run cell-set rules are stated (protocol §8, design §12.2).
- Target-free validation masks ∪ D*_ρ(t) of all 48 targets. Validation is used only by the pilots (condition `all`) and G4. Study runs use the final epoch at E*, and λ = 1 is fixed. No checkpoint or hyperparameter is selected on a target.
- Holm families are BP {C1…C7} and MF {C1…C6}. Classification order and constants match the vendored code. Vbest excludes V3c and F, and the per-arm C6/C7 descriptive report addresses selection bias.
- V3c is correctly described as a shared-relation-vector control (part_of collapsed with the regulates family). DGZ's own conflation is tested through labels (C3), not V3c.
- The DGZ reproduction gates are ordered, and each stops the study on a miss:
  - G2: ±0.005 on released predictions;
  - G3: strict load, ≤ 1e-4;
  - G4: ±0.01, with a 3-seed fallback.
  G2b (BP L1 ∧ L2 ∧ L3; MF criterion ≥ 4 of 5) decides only §9.3.
- Text is used only on the term side. Protein inputs are ESM-2 sequence embeddings or InterPro vectors, and DS uses Diamond on sequences. No protein text column is read, and `owl:Nothing` has no text and gets a free vector.
- The grid arithmetic is unchanged at 5,939 runs. The Finding-1 fix changes labels and masks only, not the run count.

## Unverified or wrong assumptions

- **Wrong:** that the masked W2 rule changes "exactly one thing: the number of positives" and that k = 0 under W2 is zero-shot (spec §5, protocol §5/§9.2 C2, risk 6). The row-selective mask on target-related columns encodes P(t) (Finding 1).
- **Wrong, as stated:** risk 6, "can depress zero-shot scores". Under masking the effect can go either way (Finding 1).
- **Unverified:** the size of the mask leak. It was not measured; Finding 1 rests on the training dynamics of BCE with a row-selective mask. The closed-world fix removes the question rather than needing a measurement.
- **Unverified (DGZ source not local), unchanged and correctly listed in design §17:**
  - DGZ hyperparameters, norm type, NF4 indexing and the `evaluate.py` Smin/micro-AUPR formulas;
  - which column trained `deepgozero_zero.th`;
  - whether the released predictions are already propagated.
  G2, G2b, G3 and G4 are the backstops.
- **Unverified here:** the v3 [ran] counts (14 BP / 5 MF alt-id exclusions; 778 BP / 59 MF PrimAlt labels; 0 exceptions to check 3d). They were not re-run in this review. G1 and G6 re-assert them on every split.


---

## Orchestrator decision: design frozen at v3 with the round-3 HIGH fix adopted

The design loop is closed after round 3 (1 HIGH, 0 MEDIUM, 4 NIT; every round-1 and round-2 finding resolved).

**Round-3 HIGH, adopted as specified by the reviewer.** Under W2, withheld proteins must NOT be loss-masked for the target subtree D*(t). They are treated as ordinary negatives for every term in D*(t), exactly like any unannotated protein. The masked rows would otherwise reveal which proteins carry t, so k = 0 would not be true zero-shot.
- The planner must reflect this in the plan.
- The implementer must apply it in the withholding generator and the trainer, and update `design_go_function.md` and `protocol_go_function.md` (the W2 definition, the related assertions and a response-table row).
- Add a unit test: the trainer's per-row loss mask is identical, apart from the erased labels, between a W2 condition and the base data. No row-level signal identifies the withheld proteins.
- The run count is unchanged.

**The 4 NITs** are implementation or documentation items. They go to the implementer and code reviewer.

Protocol v3, with this fix, is the design to plan and build against.
