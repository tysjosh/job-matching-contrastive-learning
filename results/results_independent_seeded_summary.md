# Independent Seeded Results (Deleted Runs)

These results were from independently-sampled splits with training seed=42.
The result directories were deleted but the numbers are preserved here.

## Configuration
- Training seed: 42 (set via `_set_seed(42)` in trainer)
- Data splits: independent stratified sampling at each fraction (not nested)
- Configs: Vanilla = `phase1_infonce_baseline_config.json`, Pre-ISCO = `phase1_infonce_ontology_config.json`
- Both use old KG (`esco_kg.gexf`), no reuse weighting

## AUC-ROC (good_vs_rest) on test set

| Fraction | Samples | Vanilla | Pre-ISCO | Δ | Winner |
|---|---|---|---|---|---|
| 20% | 833 | 0.648 | **0.688** | +0.040 | Pre-ISCO |
| 40% | 1,666 | 0.703 | **0.714** | +0.011 | Pre-ISCO |
| 50% | 2,082 | **0.713** | 0.675 | −0.038 | Vanilla |
| 60% | 2,499 | 0.738 | **0.744** | +0.006 | Pre-ISCO |
| 80% | 3,332 | 0.704 | **0.725** | +0.021 | Pre-ISCO |
| 100% | 4,167 | 0.726 | 0.725 | −0.001 | Tie |

## Full metrics at each fraction

### 20%
| Metric | Vanilla | Pre-ISCO |
|---|---|---|
| AUC-ROC | 0.648 | **0.688** |
| Spearman | 0.286 | **0.323** |
| d(g,p) | 0.266 | **0.447** |
| d(g,n) | 0.728 | **0.797** |
| Triplet | 0.294 | **0.323** |
| NDCG | **0.837** | 0.817 |
| MAP | 0.351 | **0.379** |

### 40%
| Metric | Vanilla | Pre-ISCO |
|---|---|---|
| AUC-ROC | 0.703 | **0.714** |
| Spearman | 0.377 | **0.377** |
| d(g,p) | 0.419 | **0.516** |
| d(g,n) | **0.929** | 0.939 |
| Triplet | **0.343** | 0.338 |
| NDCG | 0.872 | **0.872** |
| MAP | **0.443** | 0.434 |

### 50%
| Metric | Vanilla | Pre-ISCO |
|---|---|---|
| AUC-ROC | **0.713** | 0.675 |
| Spearman | **0.396** | 0.327 |
| d(g,p) | **0.439** | 0.384 |
| d(g,n) | **0.957** | 0.844 |
| Triplet | **0.355** | 0.316 |
| NDCG | 0.850 | **0.851** |
| MAP | **0.409** | 0.385 |

### 60%
| Metric | Vanilla | Pre-ISCO |
|---|---|---|
| AUC-ROC | 0.738 | **0.744** |
| Spearman | 0.413 | **0.420** |
| d(g,p) | 0.611 | **0.612** |
| d(g,n) | **1.005** | **1.034** |
| Triplet | **0.377** | 0.365 |
| NDCG | 0.852 | **0.866** |
| MAP | 0.438 | **0.447** |

### 80%
| Metric | Vanilla | Pre-ISCO |
|---|---|---|
| AUC-ROC | 0.704 | **0.725** |
| Spearman | 0.352 | **0.389** |
| d(g,p) | 0.533 | **0.568** |
| d(g,n) | 0.911 | **0.983** |
| Triplet | 0.330 | **0.355** |
| NDCG | 0.874 | **0.878** |
| MAP | 0.435 | **0.440** |

## Notes
- The 50% anomaly persisted on this seeded run (same as unseeded)
- The 80% dip (both methods dropped from 60%) was caused by independent sampling — the 80% split was not a superset of the 60% split
- Pre-ISCO won at 20%, 40%, 60%, 80% — only lost at 50%
- These results were deleted to free disk space before running nested splits

### 100%
| Metric | Vanilla | Pre-ISCO |
|---|---|---|
| AUC-ROC | **0.726** | 0.725 |
| Spearman | **0.398** | 0.358 |
| d(g,p) | 0.565 | **0.714** |
| d(g,n) | **0.991** | 0.922 |
| Triplet | **0.354** | 0.342 |
| NDCG | 0.847 | **0.847** |
| MAP | **0.424** | 0.424 |
