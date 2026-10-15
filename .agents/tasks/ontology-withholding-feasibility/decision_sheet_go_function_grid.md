# Decision sheet: GO function grid size and the E* rule (2026-10-15)

**Decision (user, 2026-10-15): Core A, shared E* — recorded as protocol amendment A7.**

For the user and supervisor, before G8. Nothing here changes the protocol; any option other than "full grid, shared E*" needs a §14 amendment (A7) before G8. No grid run exists, so an amendment is still allowed.

## E* (G7 E* stage, complete: 24 runs, validation BCE only, no test data)

| pilot | V0 best epoch + 1 (s13, s21, s42) | V4 best epoch + 1 | E* = clip(ceil(median), 8, 64) |
|---|---|---|---|
| BP ESM-2 (primary) | 6, 7, 6 | 62, 56, 62 | 32 |
| BP InterPro | 2, 1, 1 | 19, 19, 21 | 11 |
| MF ESM-2 | 6, 7, 6 | 64, 64, 64 (cap) | 36 |
| MF InterPro | 1, 1, 2 | 64, 64, 64 (cap) | 33 |

V0 peaks early and V4 late, so the shared E* sits between them. Cost to each arm (validation BCE vs its own best): V0-ESM a few percent at 32–36, V0-InterPro about 20% at 33 (BP InterPro at 11 is close to fair); V4 stops short of its best. The bias favours the axiom arms in V4 − V0, mostly in the InterPro replicate.

## GPU cost (estimate)

Assumptions: measured solo seconds per epoch on the L40S (V4: BP 95–100, MF 69–73; V0: 6–12); other axiom arms scaled by their axiom counts (V1 0.70, V2 0.75, V3/V3c 0.80 (V3 measured), V6 1.05 × V4); 90 s per run for preflight and evaluation; parallel runs give no net GPU gain for axiom arms (measured: 2 × BP V4 at 217 s/epoch each vs 100 s alone). A 10-hour reservation gives about 9 GPU-hours. G9b replaces these with a measured projection.

| design | runs | GPU-hours | 10-h reservations (1 machine) |
|---|---|---|---|
| Full grid, shared E* (as preregistered) | 5,939 | 2,800 | 311 |
| Full grid, E* per arm (V0/V5 at 8; axiom arms at 19–64) | 5,939 | 4,850 | 539 |
| Protocol §11 cuts (no MF noIPI, W1 V0 only, MF main 5 arms) | 4,469 | 2,130 | 236 |
| … + 3 seeds | 2,707 | 1,290 | 143 |
| BP only, all blocks | 2,782 | 1,430 | 159 |
| BP only, 3 seeds, W1 V0 only | 1,538 | 800 | 88 |
| **Core A**: BP ESM; V0, V3, V3c, V4, V5, V6; k 0, 1, 5 (+ all); 3 seeds; 8 packs; C3 cells (V4 DGZ/W2, WDGZrel, W2drop at k 0) | 522 | 320 | 35 |
| Core B: as A with 4 packs (24 targets) | 270 | 165 | 18 |
| Core C: V0, V3, V3c, V4; k 0, 5; 3 seeds; 8 packs; C3 cells (C1, C2, C3 only) | 276 | 185 | 20 |

Two machines halve the reservation count. An E* per arm costs about 70% more, because the axiom arms dominate the cost and their own best epoch is later.

## What each core design keeps

- **Core A** keeps every BP primary contrast: C1 (V3 − V3c, k ≤ 5), C2 (V4 − V0, k = 0), C3 and its decomposition, C4/C5 (text arms) and Δ^small (k ∈ {0, 1, 5}), on all 48 targets. It drops k = 10/25 (the upper retention curve), the V1/V2 ladder, W1, the re-draw, MF, and the InterPro replicate.
- **Core B** halves the targets: CIs widen by about √2.
- **Core C** drops the text arms (C4/C5) and k = 1.

Optional cheap add-ons to Core A (estimate): InterPro replicate on BP at E* = 11 (V0, V4; k 0 + all; 3 seeds) about 10 GPU-hours; V1/V2 ladder about 95 GPU-hours; k = 25 about 95 GPU-hours.

## Recommendation

1. **E* rule: keep the preregistered shared median** (no amendment for E*), and report the V0/V4 epoch asymmetry. An E* per arm is fairer to V0 but costs about 70% more and changes a frozen rule.
2. **Grid: Core A** (about 35 reservations, about 18 with two machines), recorded as amendment A7 before G8, with MF, InterPro, W1, re-draw and the upper retention curve explicitly descoped (or kept as the cheap add-ons above).
3. **Seeds 5 → 3**: the V0 seeds agree to about 3 significant digits at full data, so targets, not seeds, carry the variance. This is at full data only; at small k the seed variance is unmeasured.
4. The six S1 Tier B cells (V0-InterPro, V4-InterPro, V0-ESM, BP and MF, `all`, seed 13) are kept in every option; they are about 2 GPU-hours at the current E* values.
