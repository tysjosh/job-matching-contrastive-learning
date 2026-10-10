# Frozen copy of protocol_go_ppi.md v6, section 13 (plan decision 4a)

Test fixture for `test_sweep_driver.py` amendment case (a): the v6 section 13 with an
empty amendments table (no rows), i.e. before amendment AM1 and before the plan
decision 4b sentence. Only the `## 13.` to `## 14.` span is read.

## 13. Pins and amendments

**Pins** (mirrored from `protocol_pins.json`; filled at each gate):

| pin key | value | date |
|---|---|---|
| G1 | | |
| G1-leak | | |
| G2-targets | | |
| G2-cache | | |
| G2-covariate | | |
| G2-validation | | |
| G3 | | |
| G4-lambda | | |
| G4-configs | | |
| G5 | | |

**Amendments.**

| id | date | gate | groups cleared | cause | fix | re-pin |
|---|---|---|---|---|---|---|

(No rows yet.)

Rules for rows:

- **id.** Ids are "AM" followed by a positive integer without leading zeros, assigned in order. They are distinct from the auxiliary reading A1. Example ids are never written in this section outside a real row.
- **date.** `YYYY-MM-DD`.
- **gate.** The gate at which the amendment is made.
- **groups cleared.** The (condition, seed) groups a stopped-grid amendment authorises `sweep.py clear-group` to rename (§3). They are written as `<cond_id>,s<seed>` entries separated by `;`, or `—` for an amendment that clears nothing.
- **cause.** The cause found in `GROUP_FAILED.json`, or for other amendments, the field changed and why.
- **fix.** The fix applied.
- **re-pin.** Whether `sweep.py pin` was re-run (`yes`/`no`).

Every row needs a non-empty cause and fix. `sweep.py clear-group` reads only this table, between the `## 13.` and `## 14.` headings. It refuses unless exactly one row carries its `--amendment` id, that row has a valid date and a non-empty cause and fix, and its "groups cleared" cell lists the group. It also refuses a group already cleared under that row (design §12.3). The full row is copied into `sweep_log.jsonl`.

## 14. Changes

(Not part of the fixture.)
