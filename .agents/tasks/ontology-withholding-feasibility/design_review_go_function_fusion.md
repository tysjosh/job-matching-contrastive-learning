# Design review: `protocol_go_function_fusion.md` (v3 DRAFT, 2026-10-14)

Reviewer: independent design-review subagent, round 3. Spec: `findings_go_function_fusion.md` (Summary, §2 pilot, §§3–5, conclusions (i)–(vi)) and the user decision table (protocol §2, U1–U7). Severity rule: HIGH/MEDIUM only for scientific validity, leakage, selection on test, or metric/decision-rule correctness. Wording, paths and docs polish are NIT.

**Verdict: APPROVED** (0 HIGH, 0 MEDIUM, 7 NIT).

Both round-2 blockers are resolved correctly:
- **Round-2 #1 (which control scoring a reading uses).** §3.5 now has a "Which scoring a reading uses" block. It defines D_struct_b and D_typed_b on the frozen-parameter scorings (draw sets U and R). §6.2 and §12 read them by ci_low > 0, and the re-selected scorings are descriptive only. D_typed ≡ 0 with the "null selected on validation" label when w_R* = 0. The vacuous-check path is closed.
- **Round-2 #2 (co-annotation built from TP-closed labels).** The confirmatory CoA now excludes cols(t), the root, duplicates, cols(anc_TP(t)) and cols(desc_TP(t)), and requires J > 0. The unrestricted construction is kept as descriptive CoA⁺, with a per-band overlap share. The §12 reading uses CoA at frozen (c*, τ*) only. §16 adds a limitation: CoA is "no direct ontology edges", not fully ontology-free. That is the correct, honest scoping.

Round-2 NITs 3–7 are all addressed:
- Sib is defined via `isa_siblings`.
- Holm is written out.
- PC's parameters are written (c_PC*, τ_PC*, α*).
- S2's R block is X(t) ∩ cols(Reg(t)).
- The w_H transfer-source note is in place.

Against the brief:
- **Primary and secondary fusions match spec (i)/(iii) exactly:**
  - AP-T: w_t = c·τ/(τ + n_train(t)), Pa_TP = is_a + part_of, same namespace, root removed, float64 per-term z with population sd, c ∈ {1, 2, 3}, τ ∈ {10, 30, 100}, ties → smaller c, larger τ.
  - Fixed-w variant: as specified.
  - AP-T+H: w_H ∈ {0, 0.25, 0.5, 1}, tie → 0.
  - PC: σ(ℓ + w_t·z-prior), convex with DS, selected on validation Fmax.
  - Typed contrast: TP bases only, never a winner candidate.
- **Controls:**
  - Every control has an expected direction and is in the form of the fusion it controls (AP-T, AP-T+H, PC or AP-T+R).
  - Every control is scored with frozen and re-selected parameters.
- **Pilot disclosure:** explicit (§10.1–10.2). The headline table matches the pinned pilot JSONs (re-checked below).
- **Selection:** validation only, with finalists frozen at G-F/G-F2. Test is scored once at G-T, behind a guard and with `test_read.json` written first.
- **Gates:** G-R, G-V, G-F, G-D, G-TP, G-V2/G-F2 and G-T are concrete, with fatal conditions and tolerances.
- **S2:** a separate preregistered descriptive add-on. It uses S1-frozen Tier A MLP parameters, and remote parameters only under the mechanical snapshot rule.
- **§11.6:** a single dated row (X1) that changes no GF decision.
- **User decisions:** U1–U7 are recorded.
- **Statistics:** the bootstrap p, the percentile CI, the Holm step-down and the classification rules are correct.

The remaining findings are polish or very-low-probability corner cases. None changes a confirmatory decision under any realistic outcome.

---

## Findings

### 1. NIT: Holm handling of a contrast that is "not computed" for an empty band (§8)

§8 says a band with 0 test terms yields no draw block, so a contrast that needs it is "not computed". §8 Multiplicity specifies two cases: a missing *base* (its contrasts are dropped and m shrinks) and a null-selected contrast (slot kept, p = 1). It does not cover a not-computed Sc0/Sc1. On DGZ's test split this cannot realistically occur (validation has 366 BP and 220 MF band-0 terms), so it is a NIT.

Fix: add one sentence to §8: "A contrast not computed because a required band is empty keeps its Holm slot with p = 1 (family size fixed) and is labelled 'not computed'."

### 2. NIT: GF amendment A6 is not reflected (§ header, §10.4, §11.2)

`protocol_go_function.md` §14 now has A6 (2026-10-14). The GF G4 V0 gate cells use DGZ's `deepgopro.py` optimiser settings (lr 1e-3, MultiStepLR [1, 3], dropout 0.3). Every study-mode V0, including the Tier B V0 `all` checkpoint, keeps study.json settings. A6 also records that two V0 gate test results were seen. Three places are affected:
- The header cites "GF v4, A1–A5".
- §10.4 lists GF test-side contact (G2, smoke), but not the G4 gate results.
- §11.2 justifies S2 parameters by "the released MLP has V0's architecture". The architecture matches, but the training settings do not.

None of these affects a fusion selection or test read, since they are base-only exposures.

Fix:
- Header: "(v4, A1–A6, D19 approved)".
- §10.4: add "GF G4 gate runs (A6) scored V0/V4 gate replications on test (base metrics only, no fusion)".
- §11.2: "the released MLP has V0's architecture (GF §6.1) but DGZ `deepgopro.py` training settings, whereas grid V0 uses study.json (GF A6); S2 parameter transfer carries this difference".
- §16: one matching line.

### 3. NIT: protein-centric column set and alt-id duplicates (§5)

§5 defines confirmatory class-centric terms as primary columns that are "not an alt-id column of another column". The protein-centric set is "all primary columns except root", which leaves open whether `vocab.duplicate_cols` are included. Duplicates double-count a term in precision, recall and Smin. Every PC contrast is paired on the same set, so no decision changes.

Fix: in §5, "Protein-centric: … all primary, non-duplicate columns except root (the same column universe as the class-centric set, before the 0 < n_pos < n filter)". Alternatively, state that duplicates are kept for GF G2 parity. Pick one.

### 4. NIT: duplicated G-0 test item (§9 G-0)

"Holm with a p = 1 slot" appears twice in the G-0 list. Delete the second occurrence.

### 5. NIT: tie wording (§3.1)

"Ties (equal to 1e-12)" should read "Ties (|Δ| ≤ 1e-12)". Apply the same wording to every tie rule (§3.2, §3.3, §3.4, §3.5, §6.3).

### 6. NIT: draw set for the descriptive "matches/exceeds DGZ" rule (§12)

"Matches" is defined as a CI within ±0.01 per band, and "exceeds" as ci_low > 0. The rule does not name the bootstrap. Fix: "per-band CIs from draw set S (§8), terms evaluated for both scores".

### 7. NIT: S2 z protein set differs from S1 (§11.3)

S2 standardises over GF `eval_rows`. S1 standardises over test proteins with ≥ 1 TP label in the aspect. If GF `eval_rows` is a different set, the frozen (c*, τ*) is applied on a different z scale. S2 is descriptive, so this is a NIT. Fix: one line in §11.3 stating that `eval_rows` is GF's evaluation set, and that any difference from the S1 protein definition is reported as a count per run.

---

## Verified assumptions

- **Pilot snapshot:** the sha256 of all 7 pilot JSONs, `c_struct.py`, `b_valid_pilot.py` and `dgz_evaluate.py` in `/tmp/gofunc_fusion/` match §13. The `swissprot_exp.sim` hash was not recomputed this round; it was checked in round 2.
- **§10.2 pilot table:** every cell matches the `struct_*.json` files:
  - `own`, `pa_tp_adapt_tau30`, `pa_tp_shuffled` and `F_protocol`, for the all / band 0 / band 1–10 columns, for BP MLP, BP DGZ, MF MLP and MF DGZ.
  - Example: BP DGZ AP-T 0.7926 / 0.7331 / 0.7901; MF DGZ shuffled 0.8544.
  - Band counts: BP 366/3255/2865/2568/1760/329, MF 220/860/545/421/306/55.
- **DGZ comparator:** `dgz_evaluate.py` L62 sets `alpha = 0.5`, and L66 has `preds = row.blast_preds * alpha + row.preds * (1 - alpha)`, as §6.3 states.
- **`go_function/pins.py`:**
  - `PROTECTED` = contrastive_learning, go_ppi_domain, scripts/go_withholding, config/go_withholding, trials_domain, scripts/trials_fusion_confirm, scripts/run_learning_curve_point.py.
  - `assert_writable_target` checks only `PROTECTED`, so `results/go_function_fusion/tp_bases/` is writable.
  - `canonical_json` refuses absolute paths.
- **`go_function/train.py`:**
  - `run(spec_raw, device, *, log, loader, study, out_dir)` accepts the override (docstring: "tests only") and calls `assert_writable_target(out_dir)`.
  - `_finish` writes eval.json → scores.npz → run.json → DONE.json. `run.json` has `wall_seconds`, not a start time.
  - `RESULTS_PREFIX = "results/go_function/"`.
- **`go_function.ontology`:**
  - `TRUE_PATH = ("is_a", "part_of")`.
  - `ancestors` is reflexive. `descendants` is its reflexive inverse.
  - `isa_siblings` removes t, anc(t) and desc(t), matching §3.
- **`go_function.targets.channel_sets`:**
  - Pa = TP parents minus root, over primary non-duplicate columns minus t's columns. This matches S1 confirmatory Pa_TP.
  - X excludes TP ancestors.
  - `REG_FAMILY` matches §3.
- **`go_function.fusion.zscore`:** float64, population sd, zero vector for a constant or non-finite sd. `MIN_SIB_POS = 10`.
- **`go_function.baselines.diamond_matrix`:** row-normalises bitscores over train hits and ignores non-train hits. DS is therefore a convex combination of TP-closed rows and TP-consistent, as §3.3 claims, so no propagation is needed.
- **`go_function.evaluate`:** `tp_ancestor_pairs`, `propagate_max`, `ic_from_labels` and `protein_centric` exist.
- **`trials_domain.fusion_confirm`:**
  - `boot_p` is exactly the §8 formula.
  - `draw_indices`, `boot_stats` and `summarize` exist with a (models, patients) layout.
  - `holm` is two-contrast only, so the local Holm is justified. `go_function.vendored.holm` is a general step-down too. Its name-order tie break is irrelevant, because tied p receive equal adjusted values under the running max.
- **Holm formula in §8:** p_adj(i) = min(1, max_{j≤i}(m − j + 1)·p(j)) is the correct step-down.
- **Classification rules 1–5:** consistent with GF §9.1. Rule 2 uses p_adj.
- **GF state:** the pins are G1-labels, G1-norm, G2, G2b, G3, G5 and G6, with no G7 or G8. U5's "before G8" is therefore still achievable.
- **Disk:** 6.3 GiB free at review time.

## Unverified or wrong assumptions

- **Wrong (docs only):** "GF v4, A1–A5". GF now has A6 (finding 2).
- **Unverified (disclosed):** that DGZ early-stopped its released models on `valid_data.pkl`.
- **Unverified (deferred to G-R):** the DGZ Tables 3–4 DS cells.
- **Unverified (handled by the G-TP source rule):** the GF grid's remote status, and therefore whether Tier B uses source (a) or (b).
- **Unverified:** whether GF `eval_rows` equals S1's "≥ 1 TP label in aspect" test protein set (finding 7, descriptive S2 only).
