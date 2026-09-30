#!/usr/bin/env python3
"""Verify the career single-factor ontology arm actually differs from its baseline.

Motivation: ``ontology_guided_negatives`` builds the ESCO skill matcher, but the
negative-selection routing in ``BatchProcessor._select_negatives`` was gated on
``use_pathway_negatives`` alone. With that gate in place, an "ontology arm" config
that leaves ``use_pathway_negatives=False`` trains *identically to the baseline*
while every config field says otherwise. That failure mode is silent — no
exception, no warning, just a null result attributed to the ontology.

So the factor is checked empirically rather than trusted: build both configs
against the same records and the same seed, and confirm the selected negatives
differ. If they do not, the arm is inert and must not be run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(path: Path, limit: int):
    from contrastive_learning.data_loader import DataLoader  # noqa: F401  (side effects)
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _build(config_path: Path):
    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.batch_processor import BatchProcessor

    config = TrainingConfig.from_json(str(config_path))
    bp = BatchProcessor(config=config, esco_graph_path=config.esco_graph_path)
    return config, bp


def _select(bp, samples, seed):
    """Run in-batch negative selection over ``samples`` with a pinned RNG.

    ``_select_negatives(anchor_sample, batch, global_job_pool=None)`` takes the
    full batch of ``TrainingSample`` objects, not job dicts, and reads the cap
    from ``self.max_negatives_per_anchor``. The RNG is reseeded identically per
    anchor in both arms so any difference in the selected set comes from the
    ontology and not from RNG drift.
    """
    out = []
    for i, sample in enumerate(samples):
        random.seed(seed * 1000 + i)
        try:
            negs, dists = bp._select_negatives(sample, samples)
        except Exception as exc:  # noqa: BLE001
            print(f"  selection failed on sample {i}: {type(exc).__name__}: {exc}")
            continue
        ids = [n.get("job_id") or n.get("title", "") for n in negs]
        out.append((ids, [round(float(d), 4) for d in dists]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", type=Path,
                    default=ROOT / "config" / "lc_career_baseline.json")
    ap.add_argument("--ontology", type=Path,
                    default=ROOT / "config" / "lc_career_ontneg_only.json")
    ap.add_argument("--train-file", type=Path,
                    default=ROOT / "preprocess" / "learning_curve_v7" / "frac_10" / "train.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from contrastive_learning.data_loader import DataLoader
    from contrastive_learning.data_structures import TrainingConfig

    print(f"Loading up to {args.n} records from {args.train_file}")
    # Load through the baseline config so record parsing is identical for both
    # arms; only negative SELECTION is allowed to differ.
    loader = DataLoader(TrainingConfig.from_json(str(args.baseline)))
    samples = []
    for batch in loader.load_batches(args.train_file):
        samples.extend(batch)
        if len(samples) >= args.n:
            break
    samples = samples[: args.n]
    print(f"  {len(samples)} training samples")

    with_uris = sum(1 for s in samples if s.resume.get("skill_uris"))
    print(f"  samples carrying skill_uris: {with_uris}/{len(samples)}")
    if not with_uris:
        print("\nFAIL: no sample carries skill_uris, so the ontology path can "
              "never be reached regardless of the gate.")
        return 1

    results = {}
    for name, path in (("baseline", args.baseline), ("ontology", args.ontology)):
        print(f"\n--- {name}: {path.name} ---")
        config, bp = _build(path)
        print(f"  use_pathway_negatives      = {bp.use_pathway_negatives}")
        print(f"  ontology_guided_negatives  = "
              f"{getattr(bp, 'ontology_guided_negatives', None)}")
        print(f"  skill_matcher built        = {bp.skill_matcher is not None}")
        print(f"  rank tiers                 = "
              f"{getattr(config, 'ontology_negative_rank_tiers', False)}")
        results[name] = _select(bp, samples, args.seed)
        dists = [d for _, ds in results[name] for d in ds]
        nonzero = [d for d in dists if d != 0.0]
        print(f"  anchors selected           = {len(results[name])}")
        print(f"  distances non-zero         = {len(nonzero)}/{len(dists)}")
        if nonzero:
            print(f"  distance range             = "
                  f"{min(nonzero):.3f} .. {max(nonzero):.3f}")

    a, b = results["baseline"], results["ontology"]
    n = min(len(a), len(b))
    if n == 0:
        print("\nFAIL: no anchors produced negatives in either arm.")
        return 1

    differing = sum(1 for i in range(n) if a[i][0] != b[i][0])
    overlap = []
    for i in range(n):
        sa, sb = set(a[i][0]), set(b[i][0])
        if sa or sb:
            overlap.append(len(sa & sb) / max(len(sa | sb), 1))
    jaccard = sum(overlap) / len(overlap) if overlap else 0.0

    print("\n" + "=" * 70)
    print("SINGLE-FACTOR CHECK")
    print("=" * 70)
    print(f"  anchors compared            : {n}")
    print(f"  anchors with different negs : {differing} ({differing / n:.0%})")
    print(f"  mean Jaccard overlap        : {jaccard:.3f}")

    if differing == 0:
        print("\nFAIL: the two arms select IDENTICAL negatives. The ontology arm "
              "is inert — do not run it as an ontology condition.")
        return 1

    print("\nPASS: the arms select different negatives, so the config field "
          "changes training behaviour and the contrast measures something.")
    print("      (This checks the factor is LIVE, not that it helps.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
