# Mac dev smoke: GO/PPI fusion confirmation (protocol v4 §6.5)

2026-10-13, Mac (Darwin, CPU only), repo `R` = `/Users/olukotunjosh/Downloads/CDCL`, branch `orca-experiments-and-runner` at HEAD `e69d26f`, with the step-2 code untracked. **Non-confirmatory.** Mac runs do not count for gate 1 or gate 2 (§5.1, §6.5). Nothing here is read by a confirmatory script.

Environment for every command: `CUDA_VISIBLE_DEVICES=""`, `PYTHONHASHSEED=0`, `HF_HUB_OFFLINE=1`, `--device cpu`, `.venv/bin/python`. No MPS. No SentenceTransformer was constructed, because every text came from the frozen cache or the published chunks. Free disk was 2.8 GiB before and after the smoke.

## Results

| check | command | result | wall |
|---|---|---|---|
| unit tests | `pytest go_ppi_domain/tests/test_fusion_confirm.py -q` | **56 passed** (rerun after the protocol edit: 56 passed) | 1.9 s |
| gate 2a + 2f (dev) | `score.py --gate-reproduce --dev-local --parts 2a,2f --device cpu --out results/go_ppi_fusion_confirm_dev/gate_reproduction_dev.json` | **PASSED** `{2a: true, 2f: true}` | 138.6 s |
| cohort-B fold structure | `build_inputs.py --structure-only --device cpu --out results/go_ppi_fusion_confirm_dev/fold_structure_dev.json` | **all asserts passed** | 12.7 s (peak RSS 0.84 GB) |
| `trials_domain/tests` | `pytest trials_domain/tests -q` | **108 passed, 1 skipped** (matches the baseline) | 6.1 s |
| `go_ppi_domain/tests` (full) | not run | **environmental**: 2.8 GiB free < the 5 GB the suite needs (`make_conditions.py` disk guard) | — |

### Gate 2a: published reproduction (U, grid 0–1.0, both rules)
- Embedding sources: validation 1,457 texts (1,432 frozen cache, 25 published chunks); test 3,014 texts (all from published chunks). Both chunk manifest digests matched, and all 15 checkpoint sha256 matched the §3b table.
- 15/15 runs passed under each rule. Each run compared 14,068 leaves against `results/publication_phase1_fixedval/go_ppi_fusion_learning_curve{_ndcg_selected,}.json`, with 30 threshold leaves skipped and 0 missing.

  | rule | selected w (f10…f100 × s42,13,21) | max abs by class |
  |---|---|---|
  | `ndcg@10` | 0.5 0.5 0.4 · 0.5 0.6 0.6 · 0.5 0.5 0.4 · 0.6 0.5 0.4 · 0.6 0.6 0.5 | exact 0; rank_per_query 0; rank 7.7e-8; continuous 3.1e-7 |
  | `pooled_auc` | 0.3 0.3 0.3 · 0.3 0.4 0.3 · 0.3 0.3 0.3 · 0.3 0.2 0.2 · 0.3 0.3 0.2 | exact 0; rank_per_query 0; rank 1.2e-6; continuous 3.1e-7 |

  Every value is inside its tier (1e-3 / 1e-4 / 1e-5).
- Every `selected_weight` equals the published one, so no near-tie exception was used. The derived near-tie cells equal the §5.2a set: ndcg f10_s42 (gap 7.19e-6), pooled f10_s42 (3.61e-5) and pooled f75_s21 (6.39e-5).
- Zero-shot on the published test file: hard 0.78624, pooled 0.83782, nDCG@10 0.77224. To 3 dp this equals 0.786 / 0.838 / 0.772.

### Gate 2f: best-fusion reproduction vs `results/go_ppi_fusion_best/test_results.json`
The reference file's sha256 is `4cb540ab…`, unchanged.

| arm | reference | selected w (15) | mean Δ vs text, hard / pooled / easy / AP / nDCG@10 | max \|Δ − ref\| per checkpoint |
|---|---|---|---|---|
| AZ-X | `variants.F3_P` | 0.4 0.5 0.4 · 0.4 0.4 0.4 · 0.4 0.4 0.4 · 0.4 0.3 0.3 · 0.4 0.4 0.4 | +0.02872 / +0.02708 / +0.02512 / +0.03903 / +0.01551 | 0.0 |
| GZ-P | `variants.PUB_ndcg_P` | 0.5 0.5 0.4 · 0.4 0.4 0.6 · 0.5 0.5 0.4 · 0.7 0.5 0.4 · 0.6 0.6 0.5 | +0.02258 / +0.02490 / +0.02669 / +0.03075 / +0.02440 | 1.4e-17 |

- All 15 selected w are exact for both arms, with no exception used.
- The rank tolerance is 1e-4. The means match §0.1b: AZ-X +0.0287 / +0.0271 / +0.0390 / +0.0155, and GZ-P +0.0226 / +0.0249 / +0.0307 / +0.0244.
- Two 2f cells have a validation top-two gap < 1e-4, so on the remote they are eligible for the logged near-tie exception: AZ-X f10_s13 (3.25e-5) and GZ-P f75_s13 (9.81e-5). On the Mac both chose the reference weight.

### Cohort-B fold structure (fold seed 2027, nothing written but counts, nothing scored)
- Identical-view groups: 14 groups / 31 proteins / max size 4, as expected.
- Fold assignment is deterministic. Fold sizes are 3,031 × 5.
- Two independent builds gave identical sha256 for every fold file in all 5 folds. The files were built in memory only.
- Per outer fold, the partitions are train 9,093, validation 3,031 and test 3,031 proteins. Grade-2 train records are 10,991 / 11,203 / 11,811 / 11,719 / 10,802 (protocol estimate ≈ 11,505). Cross-partition pairs dropped are 318k–323k per fold.
- Exposure sets: `exposed` 4,471 (= 1,457 + 3,014) and `exposed_closed` 4,481.
- `Bclean` has 4,942 test anchors, **2,669 hard-defined (≥ 2,400 holds)**; the protocol estimate under seed 2026 was ≈ 2,694. List sha256 `e891158129ea13775ea0eb20b2493ece34734377a528858746995848b302fa56`. `all_B` has 7,041 test anchors, 3,834 hard-defined.
- `bclean_records_with_exposed_partner` = 23,758 records in 4,891 anchors.
- The `Bclean` count and the runbook's "Mac fold-structure count under seed 2027 was 2,669" (written in step 2) agree.

## Observations (no action taken)
1. **Validation grade-0 cap never binds in cohort B.** In every fold, the validation file's grade counts equal the test file's for the same protein partition. For example, fold 0 test and fold 4 validation are both 1,359 / 11,008 / 11,927. Validation is capped at 20 grade-0 per anchor and test is uncapped, so in practice no anchor has more than 20 within-partition grade-0 partners: ≈ 8 per anchor survive the protein-disjoint filter. This matches §3c as written ("when > 20"). It does mean cohort-B validation files are about 4× larger per anchor than cohort A's (≈ 16.5 vs 8 records per anchor). Noted for the code review; it is not a deviation.
2. **`Bclean` hard-defined is 2,669, not ≈ 2,694.** The protocol figure is the seed-2026 estimate, and §3c already says it is recounted at build. The 2,400 floor holds.

## Deviations from the step brief / §6.5
- The full `go_ppi_domain/tests` suite was **not run**. This is environmental: 2.8 GiB free, below the 5 GB its `make_conditions.py` guard needs. At 3.9 GB that guard is known to produce 8 guard-only non-passes (`smoke_report_go_function.md`). To run it, free ≥ 5.5 GB, then run `.venv/bin/python -m pytest go_ppi_domain/tests -q -p no:cacheprovider`.
- No other deviation. Only the §6.5-allowed quantities were computed: U under both rules, AZ-X, GZ-P and zero-shot on the published files, plus the fold structure. Nothing was scored on cohort B, HuRI or any retrained model, and there was no training or confirmatory scoring.

## Exposure created by this smoke (disclosed as trials §9 A0)
**No new fusion-outcome exposure.** Added as protocol §0.3 item 10. This is the only protocol edit; it adds no table row and changes no rule. The protocol sha256 is now `9dc31d7a…`, and the `verify_transfer` parser still finds 15 checkpoint rows and 51 input rows.
- **Gate 2a/2f.** It recomputed only quantities already reported in the published fusion JSONs and `test_results.json` (§0.3 items 1, 3 and 9), on already-exposed cohort-A anchors. No new φ variant, control or rule was scored.
- **Fold structure.** It produced label-level counts (records by grade, hard-defined anchors, `Bclean` size and list hash) under the confirmatory seed 2027. These are of the item-6 kind and contain no score. They are deterministic and identical to what `build_inputs.py` computes and pins in step 0. `Bclean` stays unscored.

## Files created
- `results/go_ppi_fusion_confirm_dev/gate_reproduction_dev.json` (114 KB, `"confirmatory": false`).
- `results/go_ppi_fusion_confirm_dev/fold_structure_dev.json` (8.8 KB, `"confirmatory": false`, counts only).
- `/tmp/goppi_confirm_dev_gate.log` and `/tmp/goppi_confirm_dev_folds.log` (stdout logs).
- This report, plus the §0.3 item-10 edit in `protocol_go_ppi_fusion.md`.
- `results/go_ppi_fusion_confirm/` and `preprocess/go_ppi_fusion_confirm/` were **not** created.

## Protected paths and inputs
- `git status --porcelain` over every protected path plus `go_ppi_domain/` shows only the two new step-2 files: `?? go_ppi_domain/fusion_confirm.py` and `?? go_ppi_domain/tests/test_fusion_confirm.py`. `go_ppi_fusion/` and `results/go_ppi_fusion_best/` are untracked or ignored as before.
- `git diff --stat HEAD` over the same paths is empty.
- No file under `go_ppi_fusion/`, `results/go_ppi_fusion_best/`, `results/publication_phase1_fixedval/`, `embedding_cache/` or `preprocess/`, and no `__pycache__` in a protected package, was modified after the smoke started (checked with `find -newer`).
- sha256 after the smoke, all equal to the protocol values:

  | file | sha256 prefix |
  |---|---|
  | `test_results.json` | `4cb540ab` |
  | `finalists.json` | `49943320` |
  | `test_plan_amendment.json` | `efe265f5` |
  | `embedding_cache/go_withholding_text_embeddings.pt` | `3aed5148` |
  | `embedding_cache/go_ppi_text_embeddings.pt` | `923412df` |

  The scorer also checked both cache hashes before and after gate 2a/2f.
- No git commit, stash, reset, checkout, revert or clean was run.
