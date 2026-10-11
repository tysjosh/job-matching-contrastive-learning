# Design review, round 7: `protocol_trials_fusion.md` (DRAFT v7)

**Verdict: CHANGES_REQUESTED** (0 HIGH, 1 MEDIUM, 1 NIT).

All 5 round-6 findings are resolved as §15 says:
- tiered gate-2 tolerance (exact / 1e-6 continuous / 1e-4 rank), pinned, with the "stop and amend through §7" path;
- a GPU-agnostic gate 3 (`cuda_device_name == pins.environment.gpu_name`), with a unit test;
- `export PYTHONHASHSEED=0` / `HF_HUB_OFFLINE=1` in every block, plus a script guard;
- the `verify_transfer.py` protocol parser;
- the MPNet snapshot rule.

One new problem remains. The gate-2 leaf classifier lists its patterns as relative key paths (`ranking.*`, `global_auc.*`) but never says how they match the real nested paths in the published JSON. Read as prefix matches, every test-side rank leaf falls into the 1e-6 "continuous" class, and the round-6 stall comes back (finding 1).

## Findings

### 1. MEDIUM: gate-2 leaf-class patterns have undefined matching semantics; a prefix reading reintroduces the round-6 stall (§5.2, §5.1, §7 `gate2_leaf_classes`)
I read the published JSON [ran]. Its real leaf paths are nested under `runs[i]`:
- `test.fusion.ranking.macro.ndcg@10`, `test.fusion.ranking.n_queries`, `test.fusion.ranking.per_query[j].average_precision`;
- `test.fusion.contrasts.global_auc.pooled`, `test.fusion.contrasts.macro_query_auc.hard`, `test.fusion.contrasts.queries_with_both_classes.pooled`;
- `test.fusion.ordinal.{kendall_tau_b, spearman_rho, cohens_d.*, tier_stats.{0,1,2}.{mean,sd,median,n}}`;
- `test.delta.global_auc.*`, `test.delta.ranking_macro.*`;
- `validation_selection_curve.<"0.0".."1.0">`, `validation_stats.{text,ontology}.{mean,sd}`.

The protocol's patterns (`"ranking.*"`, `"global_auc.*"`, `"macro_query_auc.*"`, `"queries_with_both_classes.*"`, `"tier_stats.*.n"`) match none of these as prefixes of the full path. Only `validation_selection_curve.*` and `validation_stats.*` sit at the top of a run. Three problems follow:
- **Rank leaves fall to 1e-6.** Under `fnmatch` on the full path, or a prefix match, every test-side AUC, AP and nDCG falls to "every other computed leaf" → continuous → 1e-6. That is the exact failure round 6 fixed (measured up to 8.8e-6 from rounding alone). The unit test (`ranking.macro.ndcg@10` passes at 5e-5) would still pass, because it feeds a relative path, so it would not catch this.
- **Dotted keys.** Curve keys such as `"0.1"` contain a dot, so splitting a string path on "." is ambiguous.
- **`per_query` pairing.** `per_query` is a list. The protocol does not say whether rows are paired by index or by their `query` field. `epf` orders them by `np.unique`, and a port could differ.

The gate also checks only "every numeric leaf … that the new scorer computes". A scorer that silently omits a subtree would pass it.

**Fix (replace the leaf-class bullet in §5.2 and `pins.analysis.gate2_leaf_classes`):**
```
Leaf paths are tuples of keys (never dot-joined strings), relative to one runs[i] object,
with list elements of test.{text,fusion}.ranking.per_query paired by their "query" field
(not by index; a query present on one side only fails the gate).
Class of a leaf with key tuple P (first match wins):
  exact      : P[-1] in {"selected_weight","n_queries"} or "queries_with_both_classes" in P
               or "confusion_matrix" in P or (P[-1] == "n" and "tier_stats" in P)
  rank       : any of {"validation_selection_curve","ranking","ranking_macro","per_query",
               "global_auc","macro_query_auc"} in P, or P[-1] in {"kendall_tau_b","spearman_rho"}
  continuous : everything else ("validation_stats", "tier_stats" mean/sd/median, "cohens_d")
  skipped    : any leaf with "binary_classification" or "ordinal_classification" in P
Coverage: every numeric leaf of the published runs[i] that is not "skipped" must be computed;
a missing non-skipped leaf fails the gate. String leaves ("checkpoint", per_query "query") are
compared exactly where they identify rows.
```
Extend the §5.1 classifier test to feed full key tuples taken from the real file. For example, `("test","fusion","contrasts","global_auc","pooled")` → rank; `("test","delta","ranking_macro","ndcg@10")` → rank; `("test","fusion","ranking","n_queries")` → exact; `("test","text","ordinal","tier_stats","2","n")` → exact; `("test","text","ordinal","tier_stats","2","median")` → continuous; `("validation_selection_curve","0.1")` → rank. Add a coverage test: drop one non-skipped leaf from a toy scorer output, and the gate must fail.

### 2. NIT: the §7 inputs-table row pattern in `verify_transfer.py` is not anchored (§6.2 step −1)
`| <path> | <64-hex> |` also matches the tail of every §3b checkpoint row (`| 441 | <hex> |`) unless it is anchored to a line with exactly two cells. Today this would make the parser treat `441` as a path and abort, which is safe but confusing. It would also count those rows toward the "≥ 18" threshold.

**Fix.** "An inputs row is a whole line matching `^\|\s*(\S+)[^|]*\|\s*([0-9a-f]{64})\s*\|\s*$` (exactly two cells); a checkpoint row is a whole line with exactly four cells whose last is 64-hex. The parser must find exactly 15 checkpoint rows and exactly 18 inputs rows in this revision." Add a unit test that the two sets are disjoint.

## Checklist

| check | status |
|---|---|
| (1) co-primary H1/H2, Holm | Present: per cohort and per model type, α = 0.05, p ordering with H1-first ties, 0.025/0.05 steps, positive-point requirement, harm label, ordered 5-label rule with tests (§1, §4.3, §5.1). |
| (2) fusion rule and z-scoring vs `epf` | Matches `epf.main` [read]: (1 − w)·z(text) + w·z(φ), with z fit on the whole validation file (population SD, SD 0 → 1) and applied unchanged to test. The protocol says this is validation-global z, not within-patient z, and justifies the divergence from the brief. Within-patient z (SD 0 → 0) is a sensitivity. |
| (3) grid, tie rule, no test leakage | W = {0, …, 0.5}. Ties go to the smaller w (`max` over the ascending grid). Selection uses macro nDCG@10 on the 2021 validation file (cohort 2022) or the inner-validation fold (2021). No test data is used. |
| (4) 2021 CV patient-disjoint, paraphrases co-located | Groups are connected components of the fixed list ∪ frozen-cache cosine ≥ 0.80 edges, which is label-free. Each whole group goes to one fold, asserted along with topic-disjointness. The qrels-Jaccard origin of list (1) is disclosed and can only make the split more conservative. |
| (5) inner validation for w in 2021 | Fold (k+1) mod 5 serves `best_checkpoint`, z-stats and w. The test fold is not read before scoring. |
| (6) seeds, population, B, criteria, commands, outputs, hashes, limitations | All present: the 15 checkpoint hashes, new seeds 87/123 at F 25/100, 2021 seeds 13/21/42, B = 2000, RNG `[31415, year]`, shuffle seed 11, fold seed 2021, the success/replication rule, exact commands and paths, input hashes, §8. Gate-2 leaf classification needs the fix in finding 1. |
| (7) new files only | Yes. None of the new paths exist [ran]. The cache is redirected to a `cp -n` copy, the published prepared cache is read-only, and there is a no-overwrite rule with ledger and rename exceptions. |

## Verified assumptions
- **Hashes** [ran]. All 18 §7 inputs-table sha256 values match the local files. All 15 §3b `best_checkpoint.pt` sha256 values match.
- **Chunk manifest** [ran]. The test chunk directory holds 36 chunks, and its manifest digest is `3d015e60…3f11`, as stated. The validation prepared directory exists.
- **Repo state** [ran]. HEAD is `b524f2a`. The only changes are under `.agents/` (plus the untracked `300` and `reports/`). No `results/trials_fusion_confirm`, `preprocess/trials_fusion_confirm`, `scripts/trials_fusion_confirm` or `trials_domain/fusion_confirm.py` exists yet.
- **Published JSON structure** [ran]. Its leaf set is the one listed in finding 1. `per_query` rows carry a `query` key. `test.delta` holds `global_auc` and `ranking_macro` only. `validation_selection_curve` has 11 keys "0.0"–"1.0".
- **`epf`** [read]. `ranking_metrics` returns `macro` plus a `per_query` row list, ordered by `np.unique(queries)`. `delta.ranking_macro` = fused − text macro. Fusion, `z_stats`/`z_apply` and the first-maximiser `max` were checked in earlier rounds, and the file hash is unchanged.
- **Zero-shot gate value** [read]. Macro pooled AUC 0.795 matches `findings_trials_where_ontology_helps.md` (L106, L171).
- **Ground truth** [read, carried]. §0 matches findings §B1/§C, and the paraphrase list matches `findings_trials.md` §A. Earlier rounds verified the 5-topic component, the 23-topic `exposed_closed` and the derangement's feasibility, and none of their inputs has changed.

## Unverified / wrong assumptions
- **Wrong as written:** that the listed gate-2 leaf patterns classify the real published leaves. They only do under an unstated any-segment matching rule (finding 1).
- **Unverified:** the remote-vs-Mac projection differences. The 1e-4 rank tolerance rests on round-6 perturbation measurements, not on a remote run.
- **Unverified:** GPU wall time for the 19 training runs, and CPU time for fresh encodes of the uncapped 2021 test folds.
- **Unverified:** derangement draw counts on the real fold files. They are known only at run time, and they are logged.


---

## Orchestrator decision: protocol frozen at DRAFT v7

The orchestrator closed the preregistration loop after round 7 (0 HIGH, 1 MEDIUM, 1 NIT). Protocol v7 is the preregistered protocol. The two remaining findings are implementation details and do not change any hypothesis, metric, control or decision rule. The implementer must resolve them in code and document them in `verify_trials_fusion.md`:

- **MEDIUM:** define how the gate-2 leaf-class patterns match the real nested JSON paths (exact path-matching semantics, for example dotted paths with wildcards). Unit-test them against the real published fusion JSON structure.
- **NIT:** anchor the row regex in the verify_transfer parser.

The code reviewer checks both.
