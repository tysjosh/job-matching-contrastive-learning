#!/usr/bin/env python3
"""Aggregate ORCA + OSCAR/InfoNCE baselines on the Table-4 metric set.

Reads the ordinal evaluation artifact (``phase1_evaluation/ordinal_evaluation_results.json``
-> ``ordinal_v3``) that ``run_ordinal_evaluation.py`` writes, and reports the same
five columns as the OSCAR paper's Table 4 (mean ± std across seeds):

    AUC-ROC   = binary_aucs.good_vs_rest
    Spearman  = spearmans_rho.rho
    Cohen's d = separations.good_vs_no.cohens_d
    NDCG@10   = ranking.ndcg@10
    MAP strict= ranking.map_strict

This is the apples-to-apples comparison: the ORCA variants must have been
evaluated with run_ordinal_evaluation.py on the SAME graded test set the
baselines used (metadata.original_label in data_splits_v7/test.jsonl).
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "research_runs"
DATASET = "cnamuangtoun"

# Row order: baselines first, then the ORCA family, then the unfrozen ordinal probe.
ORDER = [
    "E4-InfoNCE", "E4-OSCAR-Skill", "E4-OSCAR-Hybrid", "E4-OSCAR-ISCO",
    "ER-DEN", "ER-EXT", "ER-NOONT", "ER-NOALIGN", "ER-FULL",
    # Ontology-source ablations: NOONTW removes ontology from the weak-target
    # supervision only; NOONTALL removes it from both the supervision and the
    # ReliabilityMLP inputs. Both are matched to ER-DEN on every other factor.
    "ER-NOONTW", "ER-NOONTALL",
    # Frozen-encoder ordinal ablation family (EO): A=base, B=+OSCAR, C=no-curriculum,
    # D=fixed-margin, E=no-grouping, OntNeg=skill-level ontology negatives (decoupled
    # from sample weighting), ISCONeg=occupation-level ISCO hard negatives,
    # RandNeg=random (non-ontology) negatives.
    "EO-A", "EO-B", "EO-C", "EO-D", "EO-E", "EO-OntNeg", "EO-ISCONeg", "EO-RandNeg",
    # Ontology-negative-base knob ablations (EO-ON-*): same knobs as B/C/D/E above,
    # but layered on the EO-OntNeg (skill-level ontology negative) base instead of
    # EO-A's random-negative base.
    "EO-ON-B", "EO-ON-C", "EO-ON-D", "EO-ON-E",
    # Unfrozen-encoder ordinal probe + its ablations (compare ONLY within this block).
    "UF-InfoNCE", "UF-Ordinal", "UF-Ordinal-RandNeg", "UF-Ordinal-FixedMargin",
    "UF-Ordinal-NoGrouping", "UF-Ordinal-NoCurriculum",
]

COLS = [
    ("AUC-ROC", lambda r: r.get("binary_aucs", {}).get("good_vs_rest")),
    ("Spearman", lambda r: r.get("spearmans_rho", {}).get("rho")),
    ("Cohen_d", lambda r: r.get("separations", {}).get("good_vs_no", {}).get("cohens_d")),
    ("NDCG@10", lambda r: r.get("ranking", {}).get("ndcg@10")),
    ("MAP_strict", lambda r: r.get("ranking", {}).get("map_strict")),
]


def collect(variant):
    """Per-seed metric dict for a variant, read from its ordinal_v3 JSON."""
    pattern = str(RUNS / f"{variant}__{DATASET}__s*" /
                  "phase1_evaluation" / "ordinal_evaluation_results.json")
    per_seed = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"__s(\d+)", path)
        seed = m.group(1) if m else path
        with open(path) as f:
            r = json.load(f).get("ordinal_v3", {})
        per_seed[seed] = {name: fn(r) for name, fn in COLS}
    return per_seed


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "   n/a   "
    return f"{np.mean(vals):.3f}±{np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f}"


def main():
    header = f"{'variant':<16}" + "".join(f"{c:>14}" for c, _ in COLS) + "   n"
    print("Table-4 metrics (ordinal_v3), mean ± std across seeds — dataset:", DATASET)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for variant in ORDER:
        per_seed = collect(variant)
        if not per_seed:
            continue
        cells = []
        for name, _ in COLS:
            cells.append(fmt([per_seed[s][name] for s in per_seed]))
        row = f"{variant:<16}" + "".join(f"{c:>14}" for c in cells) + f"   [{len(per_seed)}]"
        print(row)


if __name__ == "__main__":
    main()
