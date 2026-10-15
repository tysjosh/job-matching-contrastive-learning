# Design review: `protocol_go_ppi_fusion.md` (DRAFT v3, round 2)

Reviewed against `protocol_trials_fusion.md` (approved template) and `findings_go_ppi_fusion.md` (F). This is a design review only. No experiment was run and no number was recomputed. Code claims were spot-checked by reading source and hashing three pinned files.

**Verdict: CHANGES_REQUESTED** (0 HIGH, 2 MEDIUM, 2 NIT).

Round-1 findings are all resolved as specified:
- C2 fail branch 2 now requires LB95(Δ_S) > 0.
- Both round-1 examples are added and unit-tested.
- Suffix reach is A 3-7 / B 6-7.
- HuRI z-stat and w sources are explicit for S and C5.
- The `Bclean` ≥ 2,400 assert is in place.
- Version strings are bumped.

What remains is one structural gap shared by C2 and C5. GO removed the trials rule "a confident control effect is a fail", which was correct. But no branch replaced it for the case where the control carries a confident share of the gain while the residual is *not* confidently positive. That case currently falls through to *inconclusive*, and *inconclusive* never blocks label 7 ("replicated").

## Requirement checklist

| requirement | where | status |
|---|---|---|
| Co-primaries + correct Holm | §1.1, §4.3 | OK. H1/H2/H3; 0.05/3 → 0.05/2 → 0.05 step-down; tie order H1 < H2 < H3; positive iff rejected and point > 0; per cohort, Bclean for B |
| Shuffled-GO derangement (no own or partner set) | §2.3 | OK. Rules (a)–(e); decile strata; empty-set anchors excluded; built pre-pins; gate 2e checks the S path |
| Exact vs simGIC (+ is_a-only typed) | C1, C1a, C1t, §2.2.1 | OK. T is self-validating (gate 2c) |
| With/from-pruned primary + unpruned reported | P, U, C3, A-pub | OK |
| HuRI cohort/stratum + study-bias covariate | §3d, C4/C5, §4.5 | OK in construction. **C5 state: finding 2** |
| Clean held-out replication + exposure of the 1,396 disclosed | §0.2–0.3, §3c | OK. `Bclean` primary; `all_B` sensitivity with sign warning |
| One primary weight rule, grid, tie rule; other as sensitivity | §2.6 | OK. nDCG@10 over {0, …, 1.0}, ties → smaller w; global pooled AUC as sensitivity |
| Zero-shot frozen-MPNet baseline + fusion | §3a, H1z–H3z | OK |
| DE fraction × Δ trend, secondary | §1.3, §4.4 | OK |
| Two-level bootstrap, B and seeds fixed | §4.1 | OK. B = 2000; rng [31415, 1..4]; common random numbers |
| TOST bounds, success/replication criteria | §4.2, §4.3 | Structure OK. **C2 state: finding 1** |
| Reproduction gate with tiered tolerances | §5.2 | OK. exact / rank_per_query / rank / continuous; 3 near-tie cells |
| Exposure disclosure | §0.3, §8 | OK |
| Remote-host policy | header, §6 | OK. One L40S; pins there; train CUDA, score CPU; nothing confirmatory on the Mac |

## Findings

### 1. MEDIUM: C2 labels "confident density component, no confident pair-specific component" *inconclusive*, so label 7 is reachable (§4.3 C2 state; §1.5; §8)
Take this trained cohort-A row:
- Δ_P = +0.020
- Δ_S = +0.018 [+0.014, +0.022]
- P − S = +0.002 [−0.003, +0.007]

Walk the C2 states:
- Fail branch 1 does not fire: point(Δ_P) > point(Δ_S).
- Fail branch 2 does not fire: UB95(P − S) = 0.007 ≥ 0.005.
- Pass does not fire: UB95(Δ_S) ≥ 0.005.
- Partial does not fire: LB95(P − S) ≤ 0.
- So the state is **inconclusive**.

Inconclusive only adds a suffix. If the C5 and B rows also clear, H gets label 7, "replicated (shuffled control inconclusive: A)".

That contradicts the protocol in two places:
- §1.5 defines *fail* as "no confident pair-specific component".
- §8 says label 7 means "GO adds … beyond the anchor-specific-annotation … controls".

Here 90% of the gain survives the shuffle with a confident Δ_S, and the pair-specific part is not distinguishable from 0. The trials template labels this case *fail* through its `LB95(Δ_S) > 0` branch. GO dropped that branch on purpose and replaced it only for the small-UB case.

The region is plausible, not an edge case. It is the expected pattern under the study-bias hypothesis F§6 R3 names, given P − S half-widths of about 0.005. Its label is wrong, so this is a decision-rule correctness problem.

**Fix.** Extend fail branch 1 with the absorption condition:
```
C2 state (GO), first match wins:
  1. fail:     [LB95(Δ_P − Δ_S) ≤ 0 and (point(Δ_P) ≤ point(Δ_S) or LB95(Δ_S) > 0)]
               or [UB95(Δ_P − Δ_S) < +0.005 and LB95(Δ_S) > 0]
  2. pass:     UB95(Δ_S) < +0.005 and LB95(Δ_P − Δ_S) > 0
  3. partial:  LB95(Δ_S) > 0 and LB95(Δ_P − Δ_S) > 0
  4. inconclusive   (now only: Δ_S not confidently > 0 and P − S not confidently > 0, point P > point S,
                     or UB95(Δ_S) ≥ 0.005 with LB95(Δ_S) ≤ 0 and LB95(P − S) > 0)
```
All six existing worked examples keep their states. Example 4 has LB95(Δ_S) = −0.001, so it stays *inconclusive*. Add one unit-tested example: Δ_P = +0.020, Δ_S = +0.018 [+0.014, +0.022], P − S = +0.002 [−0.003, +0.007] → *fail*.

`c2_trials_state` is unchanged. Bump `c2_rule` to `go_pair_specific_v4`. Update §1.5, §5.1, §10 and §11.

### 2. MEDIUM: C5 has the same gap, "Bz absorbed a confident share, residual not confidently > 0" → *inconclusive* (§4.3 C5 state)
Take Δ_P = +0.020 with LB95(Δ_P − Δ_C5) > 0, and Δ_C5 = +0.002 [−0.002, +0.006]. Walk the states:
- Fail branch 1 does not fire: point(Δ_C5) > 0.
- Fail branch 2 does not fire: UB95 = 0.006 ≥ 0.005.
- Pass does not fire: LB95 ≤ 0.
- So the state is **inconclusive**.

H can then reach label 7, "replicated (study-bias control inconclusive: A)", although Bz carried 90% of the gain and GO's residual beyond study bias is not distinguishable from 0. §8 defines label 7 as "beyond the … partner-study-bias controls", so the label is wrong in a plausible region. C5 was made a label gate precisely because study bias is the main named threat (§10).

**Fix.**
```
C5 state, first match wins:
  1. fail:  [LB95(Δ_C5) ≤ 0 and (point(Δ_C5) ≤ 0 or LB95(Δ_P − Δ_C5) > 0)]
            or [UB95(Δ_C5) < +0.005 and LB95(Δ_P − Δ_C5) > 0]
  2. pass:  LB95(Δ_C5) > 0
  3. inconclusive   (now only: LB95(Δ_C5) ≤ 0, point > 0, and Bz absorbed no confident share)
```
The existing examples keep their states once example 4 is made explicit (finding 3). Add one unit-tested example: Δ_P = +0.020, Δ_C5 = +0.002 [−0.002, +0.006], LB95(Δ_P − Δ_C5) > 0 → *fail*. Bump `c5_rule` to `absorbed_share_v3` and `final_label_rule` to `ordered_7_label_v4`.

### 3. NIT: the C5 *inconclusive* example is underspecified (§4.3 C5 examples)
"Δ_C5 = +0.001 [−0.002, +0.004] → *inconclusive*" has UB95 < 0.005. If LB95(Δ_P − Δ_C5) > 0, fail branch 2 fires, so the example is only inconclusive when LB95(Δ_P − Δ_C5) ≤ 0, and the example does not say so. Fix: write it as "Δ_C5 = +0.001 [−0.002, +0.004] with Δ_P = +0.003 and LB95(Δ_P − Δ_C5) ≤ 0 → *inconclusive*". With finding 2 applied, that is also the only way it stays inconclusive.

### 4. NIT: interpretive text and the §11 closing line need to follow findings 1–2
- Update §1.5's C2 sentence to: fail = "no confident pair-specific component while the shuffle carries a confident share, or the point estimate favours S, or a confidently small one".
- Add a round-2 row to §11.
- Say that round 2 changes only fail-branch conditions (no hypothesis, cohort or threshold value).

## Verified assumptions (read in source or data this round)
- `scripts/evaluate_publication_fusion.py` defines `rank_auc` (96), `z_stats` (122), `z_apply` (128), `prepare_split` (140), `text_scores` (295) and `ranking_metrics` (361).
- `GoMatcher` accepts `similarity_mode` ∈ {`exact`, `ancestor`, `simgic`} (548–553). `exact` skips the closure (641) and only `simgic` uses IC (647). `GoIndex.load` exists (490), and so does `ontology_set_similarity` (619).
- `go_independent_signal_diagnosis.load_ensembl_gene_map` (51) and `decile_bins` (79) exist.
- `data_splitter._slot` (94) and `_build_record` (112) exist, as does `leak_pruning.entries_in_text` (199). `run_learning_curve_point.report_factor` exists (92).
- These sha256 values match §7 on the Mac:
  - `go_ppi_domain/go_ontology.py` (`e370785f…`)
  - `scripts/evaluate_publication_fusion.py` (`606793ef…`)
  - `preprocess/go_withholding/converted/go_index_P.pkl` (`a0c8b0b4…`)
- The trials template's C2 (§1 lines 54–55) treats LB95(Δ_S) > 0 as *fail*, and *inconclusive* there only adds a suffix (line 305). That confirms the GO rule's *inconclusive* inherits the non-blocking role but now covers a broader region (findings 1–2).
- Holm step-down thresholds, tie order and the "rejected and point > 0" rule are correct (§4.3).
- Round-1 fixes are present in §3c, §3d, §4.3, §5.1, §7 and §10–11.
- Earlier-round verifications (pruned record URIs ⊂ pruned index keys; device-record fields; F counts 2,694 / 769 / 2,860 / 13-20-38) were not re-run. The cited files' hashes are unchanged where checked.

## Unverified / wrong assumptions
- Wrong as a decision rule: that a *non-fail* C2 or C5 state supports the §8 meaning of label 7. With the current rules, a confident control share plus a non-confident residual is *inconclusive* and does not block "replicated" (findings 1–2).
- Not verified (each is deferred by the protocol to a step-0 assert with a pre-pins amendment):
  - feasibility of the stratified derangement under pruned sets and rule (d);
  - the 14 groups / 31 proteins view-group count;
  - HuRI anchors 769 ± 5 and the grade-1 counts;
  - hard-defined `Bclean` ≥ 2,400 under fold seed 2027;
  - the `report_factor` unexpected-field set.
- Not verified: that `epf.text_scores` works on a ported `prepared` dict without an encoder. The `/tmp` port may be gone, and the protocol states that it is then the spec.
- Not re-hashed: the other §3b and §7 sha256 values.


---

## Orchestrator decision: protocol frozen at DRAFT v3, with the round-2 fixes adopted exactly as specified

The preregistration loop is closed after round 2 (0 HIGH, 2 MEDIUM, 2 NIT). Every round-0 and round-1 finding is resolved. The two remaining MEDIUM findings are decision-rule gaps, and each has an exact, reviewer-specified fix. They are adopted as follows, and the implementer applies them to `protocol_go_ppi_fusion.md` (as DRAFT v4) before any code depends on them:

1. **C2 (shuffled GO).** Fail branch 1 becomes: LB95(Δ_P − Δ_S) ≤ 0 **and** (point(Δ_P) ≤ point(Δ_S) **or** LB95(Δ_S) > 0). Branch 2 is kept. Rule id `c2_rule = go_pair_specific_v4`. Add a unit test: Δ_P = +0.020, Δ_S = +0.018 [+0.014, +0.022], P − S = +0.002 [−0.003, +0.007] → **fail**.
2. **C5 (study bias).** Fail branch 1 becomes: LB95(Δ_C5) ≤ 0 **and** (point(Δ_C5) ≤ 0 **or** LB95(Δ_P − Δ_C5) > 0). Rule id `c5_rule = absorbed_share_v3`; `final_label_rule = ordered_7_label_v4`. Add a unit test: Δ_P = +0.020, Δ_C5 = +0.002 [−0.002, +0.006], LB95(Δ_P − Δ_C5) > 0 → **fail**.
3. **The NITs.** The C5 inconclusive example states Δ_P = +0.003 and LB95(Δ_P − Δ_C5) ≤ 0. Update the C2 fail wording in §1.5, and add a round-2 row to §11.

With these fixes, label 7 ("replicated") is not reachable when a control accounts for a confident share of the gain. That restores the protocol's own definition of "replicated" in §8. No hypothesis, cohort, φ variant, weight rule or threshold value changes. The code reviewer checks both rules and both unit tests.

---

# Round 3: design review of `protocol_go_ppi_fusion.md` (DRAFT v4)

I reviewed the document on its own. I checked it against the v3 copy (`/tmp/goppi_fusion_best/protocol_go_ppi_fusion.v3.md`, sha256 `29f4617e…`, which matches the header), `results/go_ppi_fusion_best/test_results.json` (sha256 `4cb540ab…`), `results_go_ppi_fusion_best.md` and the frozen `go_ppi_fusion/` source. This is a document review only. Nothing was built, tested or scored. Numbers below were read from existing JSON or recomputed by hand from those values.

**Verdict: APPROVED** (0 HIGH, 0 MEDIUM, 10 NIT).

## Checks requested for this round

| check | result |
|---|---|
| §12 vs the v3 → v4 diff | The change list is correct where it speaks. It is incomplete in places (NIT 3). No undeclared change touches a hypothesis, estimand, cohort, threshold, RNG, label order or training design. |
| Round-2 fix C2 = `go_pair_specific_v4` | Present and correct (§4.3, §7, §1.5, §10, §11). Fail branch 1 = LB95(Δ_X − Δ_S) ≤ 0 and [point(Δ_X) ≤ point(Δ_S) or LB95(Δ_S) > 0]; branch 2 kept. I hand-walked all 8 example rows under the ordered rule and every stated state is correct, including the round-2 row (only the new clause fires) and the exploratory w = 0 row (pass). |
| Round-2 fix C5 = `absorbed_share_v3` | Present and correct. I hand-walked all 5 rows; every state is correct. The inconclusive row now states Δ_X = +0.003 and LB95(Δ_X − Δ_C5) ≤ 0 (round-2 NIT 3). |
| `final_label_rule = ordered_7_label_v4` | Present (§7). The 7-label order, suffix reach (A 3–7, B 6–7) and headline examples are consistent with the v4 C2/C5 states. |
| Gate 2f | Feasible against the reference. `variants.F3_P` and `variants.PUB_ndcg_P` exist. Each has `theta` (15 selected w) and `delta.{auc_hard, auc_pooled, auc_easy, average_precision, ndcg@10}.per_checkpoint` (15 values). The specs match §2.6: F3_P = `anchor_z` / `exact_P_pruned` / `global_pooled_auc` / step 0.1; PUB_ndcg_P = `linear` / `simgic_P_pruned` / `missing: zero` (= `epf` global z) / `ndcg@10` / step 0.1. Near-tie handling is underspecified (NIT 1). |
| Every AZ-X control in AZ-X form | Yes. C1 (P − X), C1a (N), C1t (T), C2 (S, exact on the deranged map), C3 (UX − X), C4 (text + Bz), C5 (2-D text + X + Bz vs text + Bz) and ZS all use within-anchor z with their own validation-selected w by global pooled AUC (§1.5 table header, §2.3 last bullet, §2.6, §3d, §4.5). The C1 fixed-weight form is AZ at w = 0.4, which is AZ-X's modal selected w (12/15, verified). Sensitivities in another form (GZ-X; AZ-X by nDCG@10) are labelled as sensitivities, not controls. C4 and the C5 comparator inherit the rule only through the generic "AZ forms" sentence (NIT 10). |
| Every GZ-P control in GZ-P form | Yes. S_P = simGIC on the same derangement, global z from the **deranged** validation file, w by nDCG@10. C3 = U − P, U at its own GZ w. On HuRI, both use cohort-A GZ stats and w. All are descriptive and never enter a label (§1.5, §2.3, §2.5, §3d, §4.3 notation, §4.5). |

## Findings

### 1. NIT: gate 2f near-tie exception has no reference set and no stated fallback comparison (§5.2 2f)
2a derives its near-tie cells from the published validation curves and pins them as `gate2_near_tie_cells`. Then, if the other weight is chosen, it compares against a re-score at the published weight. 2f only says "as in 2a". But `test_results.json` stores no validation curves or top-two gaps for F3_P or PUB_ndcg_P. That means the code under test would be the only source of the near-tie set.

**Fix.** Add to 2f:
> "Near-tie cells for 2f are computed from the scorer's own validation curve, written to the gate output, and pinned as `analysis.gate2f_near_tie_cells`. In such a cell, either of the two top weights passes. If the non-reference weight is chosen, the cell's per-checkpoint Δ is compared at the reference `theta` by re-scoring at that w. Validation file for 2f: `preprocess/go_ppi_splits/validation.jsonl`, published text, with pruned URIs from the aligned `preprocess/go_withholding/splits/validation.jsonl`."

The Mac dev smoke (§6.5) can compute and record this set, because it is already-exposed data.

### 2. NIT: "Δ = 0 for empty-φ anchors" is false at w = 1.0 (§2.7)
At w = 1.0 under AZ, fused = z_a(X) ≡ 0, so every record ties. AUC is 0.5, and nDCG falls back to file order, so Δ ≠ 0. GZ behaves the same way. The grid includes 1.0. In the exploration it is never selected, so this changes nothing in practice.

**Fix.** Append to that §2.7 bullet: "(for w < 1; at w = 1.0 the fused score is constant within these anchors and their metrics are the all-ties values, which the code computes as is)".

### 3. NIT: §12 omits several v3 → v4 edits
These all appear in the diff but not in §12:
- §1.1 half-widths (now AZ-X 0.007 / 0.014 / 0.010);
- §1.4 cross-reference fixed (§4.4 → §4.2) and the "no φ-only HuRI AUC for exact φ" note;
- §1.5 C2 prediction softened ("Δ_S > 0 is the expected outcome" → "is possible"), plus the exploratory w = 0 note;
- §2.3 feasibility evidence added (680 / 1,354 units, accepted-draw ranges);
- §3d HuRI scoring rewritten per form (AZ needs no stats);
- §4.6 power numbers;
- §5.1 new unit tests (within-anchor z, global pooled AUC);
- §5.2 2e extended to UX and S_P;
- §6.1 φ list and `go_ppi_fusion` imports;
- headline wording (names AZ-X).

**Fix.** Add one §12 row: "Editorial / consequential: §1.1, §1.4, §1.5, §2.3, §3d, §4.6, §5.1, §5.2 2e, §6.1, headline text; follow from the rows above; no rule or threshold change."

### 4. NIT: §0.3 item 9 lists variants that are not in §2.2 and overstates what was scored
Resnik and MF/CC are not §2.2 variants. `search_log.jsonl` (208 rows) contains no Bz / study-bias variant, so C4/C5 were never scored, not even on the published checkpoints.

**Fix.** Reword the item to read "(P, U, X, UX, N, is_a-only simGIC; also Resnik, IC-direct and MF/CC variants outside §2.2)", and add "No Bz / C4 / C5 arm was scored."

### 5. NIT: reuse source for the derangement and nDCG@10 is not named consistently (§6.1, §2.3, §2.6)
I verified that `go_ppi_fusion.final.derange_anchors` implements §2.3 exactly:
- (size, id) sort;
- `np.array_split` into 10 strata;
- one `default_rng(11)` per file, consumed in stratum order;
- receiver `order[i]` ← donor `order[(i+1) % n]`;
- rules (a)–(e) with simGIC ≥ 0.5;
- 10,000-draw cap.

§6.1 does not name it. §6.1 also says macro nDCG@10 is "imported from `go_ppi_fusion`", while §2.6 says `epf.ranking_metrics`.

**Fix.** In §6.1, write: "derangement: port of `go_ppi_fusion.final.derange_anchors`; selection nDCG@10: `epf.ranking_metrics` (`go_ppi_fusion.metrics.per_query` is numerically equal and may be used in vectorised paths)". Add a label-free 2d/2e check: the cohort-A validation and test maps reproduce the exploratory accepted-draw counts (7, 2, 9, 4, 1, 4, 4, 1, 1, 5 and 69, 1, 2, 9, 1, 4, 3, 1, 1, 1) and the 680/17 and 1,354/42 unit counts.

### 6. NIT: the fixed-w shuffle diagnostic from the exploration is not carried forward (§1.5, §4.5)
The exploration reported `shuffled_fixed_w`, the shuffle at AZ-X's own w. Its hard AUC is −0.0050 [−0.0105, +0.0011] and its AP is −0.0369. It complements the own-w Δ_S, which is ≡ 0 when validation picks w = 0. When that happens, the C2 state reduces to LB95(Δ_X) > 0.

**Fix.** Add a descriptive arm to §4.5: "S at the AZ-X selected w of the same model (`S_fixed_w`), four-state label, not in any label."

### 7. NIT: `go_ppi_fusion/` is untracked in git, and `results/go_ppi_fusion_best/` is git-ignored (§6.2 step −1, §7)
Both are hashed or pinned, and transfer is by rsync, so the design holds. A git-only checkout on the remote would still miss them.

**Fix.** In §6.2, list `go_ppi_fusion/` and `results/go_ppi_fusion_best/{test_results,finalists,test_plan_amendment}.json` explicitly among the rsync'd paths. Have `verify_transfer.py` check them.

### 8. NIT: duplicated table header row in §2.2
**Fix.** Delete the second `| id | index … |` / `|---|…` pair.

### 9. NIT: the §1.2b head-to-head prediction cites numbers for a different comparator
§0.1b's paired numbers (+0.0065 / +0.0135 / −0.0104) are AZ-X − PUB_ndcg_**U**. The head-to-head is AZ-X − GZ-P (pruned). The point differences from `test_results.json` are +0.0061 (hard), +0.0083 (AP) and −0.0089 (nDCG@10). The directions agree.

**Fix.** In §1.2b, cite those three point differences and say the paired CIs were not computed for GZ-P.

### 10. NIT: the selection rule for C4 and for the C5 comparator is only implicit (§2.6)
**Fix.** In the C5 bullet, write: "The comparator text + Bz (AZ) and C4 use 1-D w_b selected by validation global pooled AUC, ties → smaller w_b."

## Verified assumptions (read in source or data this round)
- v3 copy sha256 = `29f4617eba8c…`, which matches the header. Rule ids in v3: `go_pair_specific_v3`, `absorbed_share_v2`, `ordered_7_label_v3`. In v4: `go_pair_specific_v4`, `absorbed_share_v3`, `ordered_7_label_v4`.
- sha256 of `results/go_ppi_fusion_best/{test_results,finalists,test_plan_amendment}.json` = `4cb540ab…`, `49943320…`, `efe265f5…`, which matches §0.1b and §7. Also re-hashed and matching §7:
  - both published learning-curve JSONs;
  - `config/lc_go_ppi_baseline.json`;
  - `scripts/evaluate_publication_fusion.py`;
  - `go_ppi_domain/go_ontology.py`.
- §0.1b numbers match `test_results.json`:
  - AZ-X hard +0.0287 [+0.0194, +0.0384], pooled +0.0271, AP +0.0390, nDCG@10 +0.0155, 15/15;
  - GZ-P +0.0226 / +0.0249 / +0.0307 / +0.0244;
  - AZ-X − PUB_ndcg_U +0.0065 / +0.0135 / −0.0104;
  - simGIC arm − AZ-X: hard −0.0084, nDCG +0.0038; pooled −0.0065 and AP −0.0115 are also negative, consistent with the C1 prediction;
  - F3_U − F3_P within ±0.0025;
  - shuffled-arm θ = 0 for 15/15;
  - F3_P θ: 12 × 0.4, 2 × 0.3, 1 × 0.5;
  - hard Δ by fraction +0.034 … +0.024.
  
  The DE slope from those five points is ≈ −0.0097 per decade (≈ −0.010 as stated). The half-widths are 0.007 / 0.014 / 0.010.
- `go_ppi_fusion.fusion.within_query_z`: per anchor, `nanmean` and `nanstd` (ddof 0), SD 0 → 1, so a constant score gives z ≡ 0 (§2.5).
- `go_ppi_fusion.metrics.global_auc` is a tie-aware rank AUC of grade 2 vs {0, 1} over all records. It equals `epf.validation_selection_value(..., "pooled_auc")` (= `rank_auc(grades == 2, grades != 2)`). `search._argmax_first` and `epf` `max()` both send ties to the smaller w.
- `epf.z_stats` and `go_ppi_fusion.fusion.z_fit` are identical (mean, `np.std` ddof 0, SD 0 → 1). `linear` with `missing: zero` is the published GZ form.
- `go_ppi_fusion.final.derange_anchors` implements §2.3 rules (a)–(e), strata, RNG consumption and the cap (finding 5). Feasibility counts in §2.3 match `results_go_ppi_fusion_best.md` §4 and `test_results.json` `diagnostics.shuffle_manifest`.
- `test_plan_amendment.json` declares F3_P as the headline before test scoring (`before_test_scoring: true`), as §0.1b says.
- Hand-walk of all C2 (8) and C5 (5) example rows under the v4 rules: every stated state is correct.

## Unverified / wrong assumptions
- **Wrong (minor):**
  - §2.7 "Δ = 0" for empty-φ anchors does not hold at w = 1.0 (finding 2).
  - §0.3 item 9 implies a wider scored set than the search log shows (finding 4).
  - §1.2b cites AZ-X − U numbers for an AZ-X − GZ-P prediction (finding 9).
- **Not verifiable from the reference:** the near-tie status of F3_P / PUB_ndcg_P validation selection. No validation curves are stored (finding 1).
- **Not verified, deferred by the protocol to step-0 asserts with pre-pins amendments:**
  - derangement feasibility on cohort-B folds and the HuRI file;
  - the 14 / 31 view groups;
  - HuRI 769 ± 5 and the grade-1 counts;
  - hard-defined `Bclean` ≥ 2,400 under seed 2027;
  - the `report_factor` unexpected-field set;
  - the typed-parser check (2c);
  - pruned record URIs = pruned index `direct`.
- **Not re-hashed this round:** the remaining §3b checkpoint and §7 input hashes (earlier rounds or the protocol's own writing).
- **Not verified:** that `epf.text_scores` works on the ported read-only `prepared` dict. The `/tmp/goppi_fusion` port may be gone, and the protocol is then the spec.
