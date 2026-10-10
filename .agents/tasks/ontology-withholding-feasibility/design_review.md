# Design review v6: GO/PPI per-target withholding (design_go_ppi.md v6 + protocol_go_ppi.md v6)

Verdict: **APPROVED** (0 HIGH, 0 MEDIUM, 3 NIT).

`R` = `/Users/olukotunjosh/Downloads/CDCL`. The step prompt calls the documents "v4" and the prior counts "2 MEDIUM, 7 NIT". On disk both documents are now **v6** (design header and §21, protocol header and §14 "From v5 to v6"). The most recent prior review is v5 (1 MEDIUM, 4 NIT). The "2 MEDIUM, 7 NIT" counts are the v3 review's; the v4 review confirmed them resolved, the v5 review re-confirmed that, and the v6 text for those sections is unchanged. This review checks v6 against the v5 findings and reviews the new v6 text.

Read: `findings_go_ppi.md` (headings and gap list), `design_go_ppi.md` v6 (§0–§2.6, §1.3, §11 step 5, §12.1–§12.3, §14–§16, §19–§21), `protocol_go_ppi.md` v6 (§3, §7, §10, §11–§14), and the prior `design_review.md` (v5). Code facts verified in earlier rounds (`trainer.py:256`, `embedding_cache.py:509-524`, `run_phase1_embedding_evaluation.py:222-238`) are cited by text that did not change in v6, so they were not re-opened. Nothing was run, no test suite was run, and nothing was written except this file and `design_review.json`.

All five v5 findings are resolved in the design and protocol text itself, not only in the §21 table. The new v6 text (amendment-row parser, partial-marker clearing, manifest-drift pre-check, `rescore`) is consistent across design §1.3, §11 step 5, §12.3, §14, §15 and protocol §3, §7, §10, §13. The three findings below are all NIT.

---

## Prior findings (v5): resolution check

| v5 # | sev | resolved? | evidence in v6 text |
|---|---|---|---|
| 1 | MED | **Yes** | Protocol §13 amendments are now a 7-column table (`id \| date \| gate \| groups cleared \| cause \| fix \| re-pin`) with no rows. The literal `AM1`/`AM2` examples are gone: "Example ids are never written in this section outside a real row." Protocol §10 uses `AM<n>` and says it is not a valid id. Design §1.3 adds `AMENDMENT_ID_RE`, `AMENDMENT_COLUMNS` and `parse_amendment_rows`, which looks only between the `## 13.` and `## 14.` lines and matches the table by header. Design §12.3 `clear-group` requires: a regex match; exactly one row with that id; an ISO date of exactly 10 characters; non-empty cause and fix; and `f"{C},s{S}"` among the `;`-separated, backtick-stripped groups. The row is logged verbatim as `amendment_row`. Added beyond the review: one clearing per `(amendment, cond, seed)`. Tests (a)–(g) in §14 cover all of this, including (a) the real protocol refusing `AM1`, (b) the row naming one group refusing a different group, and (g) the pins table being ignored. The §15 row is extended and the §20 row-1 claim is annotated. I checked the protocol's own structure: §13 is "## 13. Pins and amendments", the next heading is "## 14. Changes", and the pins table header (`pin key \| value \| date`) differs from `AMENDMENT_COLUMNS`. |
| 2 | NIT | **Yes** | §12.3: `clear-group` needs "**at least one** of the group's existing run dirs" to carry `GROUP_FAILED.json`, skips dirs that do not exist, and renames "**every existing** dir of the group (1–4)". The closing paragraph of §12.3 explains why the `GROUP_OK` interrupt reasoning does not carry over. §14 has 2-of-4, 3-existing and none-marked cases. Protocol §3 matches. |
| 3 | NIT | **Yes** | §12.3 "Manifest-drift pre-check": an eligible group with any dir whose `sha256(lc_manifest.json) != DONE.json["lc_manifest_sha256"]` is not checked. It gets `GROUP_FAILED.json` with `lc_manifest_changed_after_done: <run_id>`, any `GROUP_OK.json` is superseded, and the command exits 4. Such a group is always eligible, because `group_ok_valid` is false for a drifted dir, so the pre-check is always reached. `check-group` applies it only to dirs with `DONE.json`. A §14 case (spy shows zero group-check calls) and a §15 row are present. Protocol §3 matches. |
| 4 | NIT | **Yes** | §12.3 adds `sweep.py rescore --run-id`. Preconditions: eval-stage `FAILED.json`, post-training checks ok, `epochs == 15`, the checkpoint exists, no `DONE.json`, the pin preflight and the stopped-grid preflight. Actions: rename `eval/` to `eval.attempt<n>`, then run the same `score` subprocess. Exit 0 → `DONE.json`, `FAILED.repaired.json`, `eval_repaired` log line. Failure → `FAILED.json` kept, `attempts` incremented, exit 3. §11 step 5 classifies `FAILED.repaired.json` as not failed. Covered by §14 cases and §15 rows (exit 2/3/4/0). Protocol §7 and §10 updated. |
| 5 | NIT | **Yes** | Protocol §10: the closing fence is on its own line, followed by the prose paragraph. |

v3 findings (the "2 MEDIUM, 7 NIT" in the step prompt): §19 claims and v4/v5 review confirmations are unchanged; the cited v6 text (§5.6 flag, §12.1 `EXPECTED_PAIR_DIFF`, §9.1/§9.2 encoder checks, §2.6 packing log, §4.3 check 7, §4.6 fatal-diagnostics note, protocol §9 absolute path) is unchanged in substance.

---

## Findings

### 1. NIT: test (a) for amendment rows reads the live protocol, so it breaks as soon as a real `AM1` is recorded

**Where:** design §14, `test_sweep_driver.py` "Amendment rows", case (a): "the **real** v6 `protocol_go_ppi.md` (empty amendments table, no rows) refuses `--amendment AM1`".

**Problem.** The protocol is a living document: §13 rows are added at G7 by design. Once the first real amendment `AM1` lists a group, case (a) fails, or passes only by accident, depending on the group the test uses. Any later regression run, for example after a code fix that needs a re-pin, would then report a spurious failure in exactly the situation where the test suite matters most.

**Fix.** Freeze the input. Replace case (a) with: "a fixture file containing a verbatim copy of the v6 protocol §13 section (table header and separator only, no rows), placed between `## 13.` and `## 14.` headings, refuses `--amendment AM1`." Add a second assertion that is not tied to the row count: "`parse_amendment_rows(PROTOCOL)` on the real protocol succeeds (header found, every row has 7 cells)." That keeps the header-format check on the real file without depending on which rows exist.

### 2. NIT: a `|` inside a cause or fix cell makes the whole amendments table unparseable

**Where:** design §1.3 `parse_amendment_rows`: "each cell split on `|` … Raises ValueError if … a row has != 7 cells"; protocol §13 rules for `cause`/`fix`.

**Problem.** Cause and fix are free text, often copied from a `GROUP_FAILED.json` failure string or a shell command, and they can contain `|`. A row like that has more than 7 cells. The function then raises for the **whole table**, so every `clear-group` is refused, including clearings for other, well-formed rows. The design's own §21 table escapes pipes as `\|`, but the parser does not say whether `\|` is honoured. The outcome is fail-closed, hence NIT.

**Fix.** Pick one rule and state it in both documents. Recommended: in `parse_amendment_rows`, split on the regex `(?<!\\)\|` and then replace `\|` with `|` in each cell. In protocol §13, add: "a literal `|` inside a cell is written `\|`." Add a §14 case: a row whose `cause` contains `\|` parses to 7 cells and the cell text holds a plain `|`.

### 3. NIT: `rescore` bakes the current `lc_manifest.json` into `DONE.json` without checking that it is unchanged since training

**Where:** design §12.3 `rescore`, "writes `DONE.json` with the same schema … (`lc_manifest_sha256`, …)", and the manifest-drift pre-check, which compares only against `DONE.json`.

**Problem.** A normal run writes `DONE.json` right after training and scoring, so the drift pre-check catches any later edit to `lc_manifest.json`. An eval-stage `FAILED.json` member can wait an arbitrary time before `rescore`, and `rescore` has no reference hash to compare against. An `lc_manifest.json` edited in that window (by hand, or by a "repair") would be recorded as the trusted hash, and the drift check would never fire. The window is narrow, and the group identity checks still run on the edited manifest, hence NIT.

**Fix.** Two changes in §12.3. (1) In the evaluation-only failure branch, write `FAILED.json` with `"lc_manifest_sha256": sha256(lc_manifest.json)`. (2) Add a `rescore` refusal (exit 2): "`sha256(lc_manifest.json) != FAILED.json["lc_manifest_sha256"]` → refuse with `lc_manifest_changed_after_failure`". Add the §15 cell and a §14 case: edit `lc_manifest.json` after `FAILED.json` was written → `rescore` exits 2.

---

## Verified assumptions

- **Amendment parsing works on the actual protocol.** Protocol §13 is headed `## 13. Pins and amendments` and is followed by `## 14. Changes`. Its amendments table header matches `AMENDMENT_COLUMNS` exactly, it has no data rows, and the pins table in the same section has a different header. No `AM<digit>` id appears in §13. Protocol §10 uses only `AM<n>`.
- **Drift groups always reach the pre-check.** `group_ok_valid` (§1.3) requires `sha256(lc_manifest.json) == DONE.json["lc_manifest_sha256"]`. A drifted dir is therefore never valid, so its group meets the "at least one dir lacks a valid `GROUP_OK.json`" eligibility rule and the pre-check runs before any group check.
- **`FAILED.repaired.json` handling is consistent.** It is named in §11 step 5 (failed-member rule), §12.3 (the `rescore` success path and the `FAILED.json`-member paragraph), §14, §15 and protocol §10. The analysis does not treat it as `FAILED.json`.
- **Partial `GROUP_FAILED.json` is recoverable.** The stopped-grid preflight fires on any single marker, and `clear-group` needs only one marked dir and renames all existing ones, so no deadlock state remains.
- **Exit codes are consistent.** `clear-group` refusals → 2; `rescore` precondition and pin failures → 2, stopped grid → 4, scoring failure → 3, success → 0; drift and mismatch scans → 4. Design §12.3, §15 and protocol §10 agree.
- **Unchanged since v5.** Everything the v5 review listed as verified (trainer and evaluator text functions, the single validity predicate, recorder ↔ manifest names, the stopped-grid behaviour, and the earlier v3/v4 items) is unchanged in the v6 text.

## Unverified or wrong assumptions

- **Wrong in a corner case:** "every clearing leaves a log line that names a recorded amendment row listing that group" (§12.3) holds, but the row's validity depends on a parser whose behaviour with `|` inside cells is unspecified (finding 2).
- **Wrong in a corner case:** the claim that `DONE.json` pins the manifest as written at training time holds for normal completions but not for `rescore` (finding 3).
- **Deferred to their gates (unverified, unchanged):** pruned and unannotated-anchor counts (G1/G3); the leak gates (G1); pair and packing feasibility (G2); target-free validation size and pilot batch count (G4); diagnostics non-interference and wall time (G0/G6); token identity of the two text functions on the pruned views (G6, §9.2 step 3).
