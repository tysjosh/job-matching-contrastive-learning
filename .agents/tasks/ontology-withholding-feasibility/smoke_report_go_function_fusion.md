# Mac dev run: GO function fusion study (S1 Tier A + S2 dry-run)

Protocol: `protocol_go_function_fusion.md` (v3, sha256 `53325461…a9b8`). Code: `go_function_fusion/`, `scripts/go_function_fusion/` (24 hashed files, `code_pin.json` sha256 `c915d8c8…65d2`). Outputs: `results/go_function_fusion_dev/` only (dev run, not gate outputs of `results/go_function_fusion/`). Mac, CPU only (`CUDA_VISIBLE_DEVICES=""`, `--device cpu`), no MPS, no training, no Diamond. Date 2026-10-14.

**Status.** G-0, G-R(b), G-R(c), G-V, G-F and G-S2dry pass. **G-R(a) is deferred to the remote** by user decision: it reads test labels, and this step's constraints forbid any test read on the Mac (see Deviations, D1). No test label was read and no test split was scored by any fusion rule.

## 1. Gate results

| gate | result | detail |
|---|---|---|
| G-0 unit tests | PASS | `pytest go_function_fusion/tests/ -q`: 41 passed (4.9 s) |
| G-R(a) GF G2 repro | DEFERRED TO REMOTE (user decision) | needs `--allow-g2-test-read` (reads released test predictions and test labels). Runbook §2 places it on the remote. |
| G-R(b) pilot repro, pilot-compat mode | PASS, 0 misses | struct JSONs: 174 fields each for bp/mlp, bp/dgz, mf/mlp, mf/dgz. Pilot JSONs: 138 fields each for bp/mlp, mf/mlp, mf/dgz (no BP DGZ pilot file, as specified). Counts exact; means, band means and Fmax within 5e-5 and equal at 4 decimals. Pilot snapshot (7 JSON, 7 npz, 2 scripts) copied and hash-checked against §13; `dgz_evaluate_ref.py.txt` hash OK. |
| G-R(c) confirmatory mode on validation | reported (not gated) | see table below |
| G-V validation search | PASS | 4 (aspect, base) pairs, search mode, 0 guard violations; `search_log.jsonl`: 4 start, 60 selection, 4 done events |
| G-F freeze | PASS | `finalists_tierA.json` sha256 `b9a791af8bb41601a29c466ee031097aa427374e1b60d966cdc3ea7ac7dea570`; `code_pin.json` `c915d8c84f72584f15a83ecbdce3e8cefbd015f3d8579756a38c157b76e765d2`; pin G-F UTC 2026-10-14T08:09:00Z |
| G-S2dry | PASS | 6 BP smoke runs, 36 rows, schema complete; F parity 12/12 targets vs `fusion_F.csv` (label-free fields + `auc_V0` from eval.json), score max diff vs independent `go_function.fusion.fuse` = 0.0; 12 V0 rows with AP-T_S2; 18 Reg (t, column) pairs skipped (not stored in X(t)). Uses frozen S2 params (c, τ) = (3, 10), not the placeholder. Remote params ineligible (`no_GF2_snapshot`). Smoke, 2 epochs, non-informative. |

G-R(c), confirmatory vs pilot-compat (validation, mean class AUC; diff vs pilot-compat in brackets):

| pair | n_rows | n_terms | alt/unprimary cols removed | changed sets (pa_tp / reg / sib / desc) | own | pa_tp | AP-T τ=30 c=2 | shuffled | protocol F |
|---|---|---|---|---|---|---|---|---|---|
| BP MLP | 5389 | 11118 | 25 | 68 / 19 / 437 / 159 | 0.7746 (+0.0001) | 0.7936 (+0.0000) | 0.7970 (+0.0000) | 0.7349 (+0.0000) | 0.7768 (+0.0001) |
| BP DGZ | 5389 | 11118 | 25 | 68 / 19 / 437 / 159 | 0.7774 (+0.0001) | 0.7894 (+0.0000) | 0.7926 (−0.0000) | 0.7250 (−0.0001) | 0.7855 (+0.0000) |
| MF MLP | 3755 | 2398 | 9 | 13 / 0 / 88 / 44 | 0.8750 (+0.0003) | 0.9115 (−0.0002) | 0.9153 (−0.0000) | 0.8461 (+0.0002) | 0.8934 (−0.0002) |
| MF DGZ | 3755 | 2398 | 9 | 13 / 0 / 88 / 44 | 0.8954 (+0.0005) | 0.9143 (+0.0002) | 0.9161 (+0.0003) | 0.8551 (+0.0007) | 0.9008 (+0.0002) |

## 2. Validation table (G-V; validation only, Tier A released checkpoints; never a result)

Mean class AUC over the confirmatory term set, by TP n_train band. "re-sel." = control re-selected over the AP-T grid plus the null cell (descriptive only); the frozen-parameter control scoring is listed under each table. Tier A typed reads are confounded (DGZ-rule labels). H_sim reads are exploratory only.

**BP MLP** (5389 proteins, 11118 terms, reg-subset 3515; band n: 0 366, 1–10 3250, 11–30 2860, 31–100 2562, 101–1000 1752, >1000 328)

| variant | cell | all | 0 | 1–10 | 11–30 | 31–100 | 101–1000 | >1000 | reg-subset | share improved |
|---|---|---|---|---|---|---|---|---|---|---|
| base | – | 0.7746 | 0.5896 | 0.7488 | 0.7922 | 0.8034 | 0.7922 | 0.7625 | 0.7497 | – |
| AP-T (selected) | c=3, τ=10 | 0.7978 | 0.7508 | 0.7969 | 0.8035 | 0.8070 | 0.7932 | 0.7625 | 0.7693 | 0.592 |
| fixed-w (selected) | c=2 | 0.7947 | 0.7426 | 0.7958 | 0.8025 | 0.8032 | 0.7860 | 0.7537 | 0.7666 | 0.547 |
| parent-only | – | 0.7936 | 0.7187 | 0.7893 | 0.8032 | 0.8060 | 0.7901 | 0.7582 | 0.7656 | 0.575 |
| GF arm F (w ≡ 1) | – | 0.7768 | 0.6220 | 0.7675 | 0.7901 | 0.7976 | 0.7808 | 0.7408 | 0.7494 | 0.457 |
| protocol F (re-sel.) | c=1, τ=30 | 0.7816 | 0.6220 | 0.7674 | 0.7937 | 0.8040 | 0.7924 | 0.7625 | 0.7535 | 0.527 |
| any-relation (re-sel.) | c=3, τ=30 | 0.7998 | 0.7539 | 0.8002 | 0.8061 | 0.8078 | 0.7936 | 0.7626 | 0.7759 | 0.592 |
| CoA (re-sel.) | c=1, τ=10 | 0.7779 | 0.5896 | 0.7568 | 0.7946 | 0.8047 | 0.7926 | 0.7625 | 0.7540 | 0.508 |
| CoA⁺ (re-sel.) | c=2, τ=10 | 0.7804 | 0.5896 | 0.7634 | 0.7964 | 0.8055 | 0.7927 | 0.7625 | 0.7554 | 0.527 |
| Desc-max (re-sel.) | null | 0.7746 | = base | | | | | | | |
| shuffled graph (re-sel.) | null | 0.7746 | = base | | | | | | | |

Frozen (c*, τ*) controls, all terms: shuffled graph 0.7366, CoA 0.7755, CoA⁺ 0.7803, any-relation 0.7995, Desc-max 0.7709, protocol F 0.7789; pure prop_TP(σ(ℓ)) 0.7736. AP-T+R (reg-subset): w_R* = 1 (0.7693 / 0.7728 / 0.7749 / 0.7770 for w_R 0 / 0.25 / 0.5 / 1); shuffled Reg at w_R* 0.7495, re-selected w_R = 0. H_sim: own+H 0.7764 (w 0.5), AP-T+H_sim 0.7996 (w 0.5). AP-T grid (c, τ): (1,100) 0.7943, (1,30) 0.7938, (1,10) 0.7922, (2,100) 0.7965, (2,30) 0.7970, (2,10) 0.7964, (3,100) 0.7965, (3,30) 0.7976, (3,10) 0.7978.

**BP DGZ** (same proteins, terms and bands as BP MLP)

| variant | cell | all | 0 | 1–10 | 11–30 | 31–100 | 101–1000 | >1000 | reg-subset | share improved |
|---|---|---|---|---|---|---|---|---|---|---|
| base | – | 0.7774 | 0.5419 | 0.7624 | 0.7930 | 0.8030 | 0.7942 | 0.7613 | 0.7499 | – |
| AP-T (selected) | c=3, τ=10 | 0.7936 | 0.7451 | 0.7913 | 0.7974 | 0.8030 | 0.7941 | 0.7613 | 0.7651 | 0.543 |
| fixed-w (selected) | c=2 | 0.7898 | 0.7334 | 0.7906 | 0.7959 | 0.7984 | 0.7848 | 0.7516 | 0.7616 | 0.508 |
| parent-only | – | 0.7894 | 0.7000 | 0.7867 | 0.7974 | 0.8012 | 0.7891 | 0.7559 | 0.7605 | 0.524 |
| GF arm F (w ≡ 1) | – | 0.7855 | 0.5934 | 0.7774 | 0.8003 | 0.8067 | 0.7930 | 0.7455 | 0.7593 | 0.545 |
| protocol F (re-sel.) | c=2, τ=30 | 0.7873 | 0.6048 | 0.7785 | 0.8002 | 0.8073 | 0.7963 | 0.7615 | 0.7609 | 0.572 |
| any-relation (re-sel.) | c=3, τ=10 | 0.7947 | 0.7397 | 0.7930 | 0.7995 | 0.8039 | 0.7943 | 0.7613 | 0.7685 | 0.557 |
| CoA (re-sel.) | null | 0.7774 | = base | | | | | | | |
| CoA⁺ (re-sel.) | c=1, τ=10 | 0.7788 | 0.5419 | 0.7670 | 0.7934 | 0.8030 | 0.7942 | 0.7613 | 0.7502 | 0.455 |
| Desc-max (re-sel.) | null | 0.7774 | = base | | | | | | | |
| shuffled graph (re-sel.) | null | 0.7774 | = base | | | | | | | |

Frozen controls: shuffled graph 0.7273, CoA 0.7729, CoA⁺ 0.7778, any-relation 0.7947, Desc-max 0.7720, protocol F 0.7870; prop-only 0.7770. AP-T+R: w_R* = 1 (0.7651 / 0.7679 / 0.7696 / 0.7713); shuffled Reg at w_R* 0.7377, re-selected 0. H_sim: own+H 0.7809, AP-T+H_sim 0.7971. AP-T grid: (1,100) 0.7905, (1,30) 0.7907, (1,10) 0.7902, (2,100) 0.7917, (2,30) 0.7926, (2,10) 0.7929, (3,100) 0.7915, (3,30) 0.7927, (3,10) 0.7936.

**MF MLP** (3755 proteins, 2398 terms, reg-subset 82; band n: 0 220, 1–10 857, 11–30 544, 31–100 418, 101–1000 305, >1000 54)

| variant | cell | all | 0 | 1–10 | 11–30 | 31–100 | 101–1000 | >1000 | reg-subset | share improved |
|---|---|---|---|---|---|---|---|---|---|---|
| base | – | 0.8750 | 0.6305 | 0.8804 | 0.9051 | 0.9159 | 0.9244 | 0.8850 | 0.7586 | – |
| AP-T (selected) | c=3, τ=10 | 0.9162 | 0.8794 | 0.9254 | 0.9120 | 0.9192 | 0.9255 | 0.8849 | 0.7972 | 0.499 |
| fixed-w (selected) | c=2 | 0.9130 | 0.8727 | 0.9254 | 0.9094 | 0.9159 | 0.9190 | 0.8602 | 0.7994 | 0.450 |
| parent-only | – | 0.9115 | 0.8558 | 0.9204 | 0.9120 | 0.9183 | 0.9235 | 0.8720 | 0.7861 | 0.478 |
| GF arm F (w ≡ 1) | – | 0.8934 | 0.7779 | 0.8995 | 0.9085 | 0.9121 | 0.9141 | 0.8522 | 0.7592 | 0.413 |
| protocol F (re-sel.) | c=2, τ=10 | 0.8977 | 0.7907 | 0.8988 | 0.9101 | 0.9175 | 0.9247 | 0.8850 | 0.7662 | 0.456 |
| any-relation (re-sel.) | c=3, τ=10 | 0.9170 | 0.8835 | 0.9260 | 0.9130 | 0.9193 | 0.9255 | 0.8849 | 0.8124 | 0.504 |
| CoA (re-sel.) | c=1, τ=10 | 0.8775 | 0.6305 | 0.8893 | 0.9026 | 0.9153 | 0.9246 | 0.8850 | 0.7565 | 0.332 |
| CoA⁺ (re-sel.) | c=2, τ=10 | 0.8823 | 0.6305 | 0.9008 | 0.9052 | 0.9159 | 0.9246 | 0.8849 | 0.7640 | 0.375 |
| Desc-max (re-sel.) | null | 0.8750 | = base | | | | | | | |
| shuffled graph (re-sel.) | null | 0.8750 | = base | | | | | | | |

Frozen controls: shuffled graph 0.8294, CoA 0.8729, CoA⁺ 0.8814, any-relation 0.9170, Desc-max 0.8719, protocol F 0.8950; prop-only 0.8737. AP-T+R: w_R* = 1 (0.7972 / 0.8025 / 0.8058 / 0.8105); shuffled Reg at w_R* 0.7745, re-selected 0. H_sim: own+H 0.8764 (w 0.25), AP-T+H_sim 0.9175 (w 0.5). AP-T grid: (1,100) 0.9122, (1,30) 0.9117, (1,10) 0.9105, (2,100) 0.9148, (2,30) 0.9153, (2,10) 0.9149, (3,100) 0.9148, (3,30) 0.9158, (3,10) 0.9162.

**MF DGZ** (same proteins, terms and bands as MF MLP)

| variant | cell | all | 0 | 1–10 | 11–30 | 31–100 | 101–1000 | >1000 | reg-subset | share improved |
|---|---|---|---|---|---|---|---|---|---|---|
| base | – | 0.8954 | 0.7193 | 0.9150 | 0.9095 | 0.9140 | 0.9202 | 0.8751 | 0.8096 | – |
| AP-T (selected) | c=2, τ=10 | 0.9169 | 0.8873 | 0.9296 | 0.9127 | 0.9146 | 0.9205 | 0.8751 | 0.8184 | 0.449 |
| fixed-w (selected) | c=1 | 0.9143 | 0.8744 | 0.9294 | 0.9118 | 0.9122 | 0.9167 | 0.8639 | 0.8190 | 0.426 |
| parent-only | – | 0.9143 | 0.8744 | 0.9294 | 0.9118 | 0.9122 | 0.9167 | 0.8639 | 0.8190 | 0.426 |
| GF arm F (w ≡ 1) | – | 0.9008 | 0.7963 | 0.9205 | 0.9056 | 0.9106 | 0.9087 | 0.8458 | 0.8059 | 0.418 |
| protocol F (re-sel.) | c=1, τ=10 | 0.9051 | 0.7963 | 0.9217 | 0.9099 | 0.9147 | 0.9203 | 0.8751 | 0.8122 | 0.447 |
| any-relation (re-sel.) | c=2, τ=10 | 0.9175 | 0.8886 | 0.9307 | 0.9130 | 0.9146 | 0.9205 | 0.8751 | 0.8315 | 0.453 |
| CoA (re-sel.) | c=1, τ=10 | 0.8955 | 0.7193 | 0.9157 | 0.9089 | 0.9137 | 0.9202 | 0.8751 | 0.8123 | 0.312 |
| CoA⁺ (re-sel.) | c=1, τ=10 | 0.8973 | 0.7193 | 0.9204 | 0.9093 | 0.9143 | 0.9202 | 0.8751 | 0.8153 | 0.319 |
| Desc-max (re-sel.) | null | 0.8954 | = base | | | | | | | |
| shuffled graph (re-sel.) | null | 0.8954 | = base | | | | | | | |

Frozen controls: shuffled graph 0.8575, CoA 0.8927, CoA⁺ 0.8959, any-relation 0.9175, Desc-max 0.8945, protocol F 0.9037; prop-only 0.8957. AP-T+R: w_R* = 1 (0.8184 / 0.8229 / 0.8252 / 0.8268); shuffled Reg at w_R* 0.7937, re-selected 0. H_sim: own+H 0.8963 (w 0.25), AP-T+H_sim 0.9180 (w 0.5). AP-T grid: (1,100) 0.9154, (1,30) 0.9157, (1,10) 0.9156, (2,100) 0.9149, (2,30) 0.9161, (2,10) 0.9169, (3,100) 0.9135, (3,30) 0.9150, (3,10) 0.9165.

Neighbour-set counts (confirmatory): empty Pa_TP BP 21 / MF 19, empty Reg BP 7603 / MF 2316, empty CoA BP 420 / MF 330, empty CoA⁺ BP 366 / MF 220, empty Sib BP 1300 / MF 429. Shuffled-graph permutation report: BP 21,634 (t, π) pairs, 77 perm[π] alt-id, 4 perm[π] = t, 0 root; MF 3,081 pairs, 25 alt-id, 2 = t, 0 root.

Observations (validation, descriptive): AP-T's selected cell sits on the grid corner (c = 3, τ = 10) for BP MLP, BP DGZ and MF MLP, and w_R* = 1 is the grid edge on every pair. The grids are frozen by the protocol, so this is recorded, not acted on. Every shuffled-graph, Desc-max and shuffled-Reg re-selection picked the null cell, as §3.5 expects. CoA re-selected a non-null cell on 3 of 4 pairs (null on BP DGZ), with small gains; §3.5 says such a control is reported as such. Any-relation ≥ AP-T on all Tier A pairs (confounded, as expected under DGZ-rule labels).

## 3. Frozen finalists (`finalists_tierA.json`, G-F)

| aspect/base | AP-T (c*, τ*) | fixed-w c* | AP-T+R w_R* (confounded) | control re-selections (shuf_graph / CoA / CoA⁺ / any-rel / Desc-max / protocol F / shuf Reg) |
|---|---|---|---|---|
| bp/mlp | (3, 10) | 2 | 1 | null / (1,10) / (2,10) / (3,30) / null / (1,30) / w_R 0 |
| bp/deepgozero | (3, 10) | 2 | 1 | null / null / (1,10) / (3,10) / null / (2,30) / w_R 0 |
| mf/mlp | (3, 10) | 2 | 1 | null / (1,10) / (2,10) / (3,10) / null / (2,10) / w_R 0 |
| mf/deepgozero | (2, 10) | 1 | 1 | null / (1,10) / (1,10) / (2,10) / null / (1,10) / w_R 0 |

Frozen control cells equal each base's AP-T (c*, τ*). S2 parameters (Tier A released MLP): BP (c, τ) = (3, 10), MF (3, 10). Input hashes recorded: `preprocess/go_function/bp/labels_TP.json` 63e1bdeb…db4a, `bp/labels_TP.npz` 96c1e5fd…af67, `mf/labels_TP.json` 3eba3e96…fcfd, `mf/labels_TP.npz` da2efa13…a626. The remote w_H, PC and Tier B parameters are frozen later at G-F2.

## 4. Wall times (Mac CPU, 8 GB RAM)

| step | wall |
|---|---|
| G-0 tests | 6.0 s |
| G-R(b) mf/mlp, mf/dgz, bp/mlp, bp/dgz | 69.8 s, 68.4 s, 352.6 s, 89.3 s |
| G-R(c) mf/mlp, mf/dgz, bp/mlp, bp/dgz | 8.0 s, 8.4 s, 35.6 s, 40.4 s |
| G-V mf/mlp, mf/dgz, bp/mlp, bp/dgz | 64 s, 63 s, 439 s, 436 s (peak RSS 1.5, 1.5, 2.4, 3.0 GB) |
| G-F | 0.6 s |
| G-S2dry | 2.1 s |

## 5. Files created under `results/go_function_fusion_dev/` (13 MB total)

- `protocol_pins.json` (pins G-R b/c × 4 pairs, G-F), `code_pin.json`, `finalists_tierA.json`
- `gates/gate_repro.json` (sections b/{bp,mf}/{mlp,deepgozero}, c/…, snapshot)
- `pilot_snapshot/`: 7 pilot JSONs, 7 per-term npz, `b_valid_pilot.py`, `c_struct.py`
- `search/{bp,mf}_{mlp,deepgozero}/`: `selections.json` + 16 per-term AUC tables each (`auc_{base,apt,fixed,parent_only,F_weightfree,shuf_graph,coa,coap,pa_any,desc_max,protocol_F,prop_only,apt_r,apt_r_shuf,own_hsim,apt_hsim}.npz`, float32, ≤ 0.41 MB each; no dense score matrix)
- `search_log.jsonl`
- `s2_dryrun/{s2_rows.csv, s2_summary.json, f_parity_detail.json}` (labelled smoke, non-informative)

Outside that directory: `config/go_function_fusion/dgz_published_ds.json` (new, see D2).

## 6. Deviations

- **D1. G-R(a) not run.** It reads the released test predictions and test labels (GF G2 exposure only, no fusion). This step forbids test reads on the Mac. The protocol puts G-R(a) on the Mac (§9) but the runbook already allows it on the remote. **User decision (2026-10-14): run G-R(a) on the remote** (runbook §2, `gate_repro.py --part a --aspect {bp,mf} --allow-g2-test-read --results results/go_function_fusion`), before G-V2. The Mac run ends here.
- **D2. `config/go_function_fusion/dgz_published_ds.json` written** (DGZ Tables 3–4, MLP + DiamondScore and DeepGOZero + DiamondScore rows, with cell references, from PMC9235501). `freeze_finalists.py` refuses G-F without it, and §6.3 requires it before G-F. It is a fusion-owned input, outside the dev results directory, and it is used only by the descriptive G-T DGZ-faithful check. **User decision (2026-10-14): keep it.**
- **D3. File name.** G-F wrote `finalists_tierA.json` (the protocol and script name), not `finalists.json`.
- **D4. Fresh dev directory.** An earlier partial dev run (mf/mlp only, 03:29–03:33) was moved to `/tmp/gofunc_fusion_devprev/` before this run, so pins and logs come from one consistent run.
- **D5. Smoke S2 inputs.** The S2 dry-run reads smoke `scores.npz` (logits on GF smoke test rows) and copies `auc_base` from each smoke run's existing `eval.json`. It computes fusion scores on those logits but no fusion AUC and reads no label (§11.5).
- **D6. Disk/swap.** During G-R(b) BP, macOS swap grew to 6.3 GB and free disk briefly fell to about 275 MB. G-V ran under a watchdog that kills the job below 150–200 MB free; it never fired (min free 277 MB MF, 552 MB BP). Free disk at the end: 2.6 GB.

Protected paths: `git status --porcelain` and `git diff --stat` are identical before and after the run. The pre-existing modifications under `go_function/` and `scripts/go_function/` predate this run (same diff hash), and no protected file has a newer mtime. In `protocol_go_function.md` the only fusion edit is the X1 row; the A6 row in the same diff is a GF amendment (DGZ MLP optimiser settings, user decision, tied to the earlier `go_function/train.py` change) and is not a fusion edit.

## 8. Confirmatory Mac gates (runbook §0 steps 2, 3, 5, 6, 7; 2026-10-14, user decision A)

Run into `results/go_function_fusion/` with the post-review code (CPU, no test read, no Diamond, no training). All 18 commands exited 0: G-R(b) × 4 pass, G-R(c) × 4 reported, G-V × 4 (0 guard violations, 4 done events), G-F, G-S2dry (F parity 12/12). The 256 per-term AUC arrays are byte-equal to the dev run, and every selection is unchanged; the only differences are code hashes (review fixes) and the `pass2_star` field now present on all four pairs. Frozen finalists are those in §3. Pins: `finalists_tierA.json` 5342268747cc…77f1, `code_pin.json` c212317fe73c…0140659 (equals the current code), `gates/gate_repro.json` 1e99cc3272b1…0aca; mirrored in protocol §13. G-R(a) stays on the remote (D1). Wall about 27 min; 13 MB, 92 files.

## 7. Code-review fixes (round 1, 2026-10-14)

- Guard: every file under `results/go_function_fusion*/tp_bases/` is blocked in search mode except `checkpoint.pt` and `tp_base_provenance.json`; `search_valid.py` registers only `results/go_function/` checkpoints with `allow_checkpoints` (source (b) no longer crashes).
- `search_valid.py --diamond-valid` (Tier A only) builds pass 2 at the G-F frozen (c*, τ*) from the pinned `finalists_tierA.json` and writes to `search/<aspect>_<base>_dv/`; `freeze_finalists.py --remote` refuses unless every `_dv` `pass2_star` equals the G-F AP-T cell.
- G-TP is now `verify_tp_bases.py` (replaces `make_pins.py tp-base`): spec checks, hashes, source (b) provenance vs GF `code_pin.json`, §10.4 exposure facts, strict load and two-call determinism in search mode.
- Re-run on the Mac after the fixes: G-V mf/mlp only (to exercise the edited code path). `selections.json` gains `pass2_star`; every other selection value and all 16 `auc_*.npz` are byte-equal to the earlier run. The dev `code_pin.json` / G-F pin predate the fixes (dev only; the confirmatory G-F is written on the Mac per runbook §0 with the current code).
- PC shuffled-hits control: re-selected over α only, at the frozen (c_PC*, τ_PC*) (descriptive control).
