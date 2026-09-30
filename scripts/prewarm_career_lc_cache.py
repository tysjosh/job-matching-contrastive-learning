#!/usr/bin/env python3
"""Encode the career learning-curve corpus once, so the 24 training runs are cache hits.

Why this is worth a separate pass
--------------------------------
The frozen text encoder's output is the same for every arm, seed and fraction, so
it should be paid for exactly once. Without it, each run encodes its texts lazily
inside the training loop — measured at ~120 s/batch with a 0.00% cache hit rate,
i.e. ~20 min per epoch on the smallest fraction alone.

Only ``frac_100/train.jsonl`` plus ``validation.jsonl`` are encoded, because the
fractions are verified nested subsets of frac_100's train file (640/1600/3200/4800
lines all present in it) and every fraction ships a byte-identical validation
file. Encoding those two therefore covers all 5 fractions x 2 arms x 3 seeds.

The cache is keyed by content hash and stores pre-projection embeddings from the
frozen encoder, so it carries nothing arm-specific and nothing learned. Sharing it
across arms cannot leak information between them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path,
                    default=ROOT / "config" / "lc_career_baseline.json")
    ap.add_argument("--lc-dir", type=Path,
                    default=ROOT / "preprocess" / "learning_curve_v7")
    ap.add_argument("--files", nargs="+",
                    default=["frac_100/train.jsonl", "frac_100/validation.jsonl"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.trainer import ContrastiveLearningTrainer

    config = TrainingConfig.from_json(str(args.config))
    config.orca_enabled = False
    cache_path = config.embedding_cache_path
    if cache_path.endswith("embedding_cache/text_embeddings.pt"):
        raise SystemExit(
            "Refusing to warm the historical shared cache "
            f"({cache_path}). Point embedding_cache_path at a dedicated file: "
            "that one predates v7, covers none of these content hashes, and is "
            "relied on by earlier experiments.")

    print(f"cache target : {cache_path}")
    print(f"cache size   : {config.embedding_cache_size}")

    trainer = ContrastiveLearningTrainer(config=config,
                                         output_dir="/tmp/prewarm_career_lc")

    for rel in args.files:
        path = args.lc_dir / rel
        if not path.exists():
            print(f"[skip] {rel}: not found")
            continue
        print(f"\n=== {rel} ===")
        started = time.time()
        before = len(trainer.embedding_cache.cache)
        trainer.preload_dataset_embeddings(path)
        after = len(trainer.embedding_cache.cache)
        print(f"  cache {before} -> {after} (+{after - before}) "
              f"in {time.time() - started:.0f}s")

    # Persist explicitly. preload_dataset_embeddings saves only when it encoded
    # something, so an already-covered file would otherwise leave nothing written.
    trainer.embedding_cache.save_to_disk(cache_path)

    entries = len(trainer.embedding_cache.cache)
    dims = {tuple(v.shape) for v in trainer.embedding_cache.cache.values()
            if hasattr(v, "shape")}
    print(f"\nwrote {entries} entries to {cache_path}")
    print(f"dims present : {dims}")
    if len(dims) > 1:
        raise SystemExit(
            f"Cache holds mixed dimensions {dims}; refusing to declare success. "
            "Projected vectors were probably written into a text-embedding cache.")
    if entries >= config.embedding_cache_size:
        raise SystemExit(
            f"Cache is at its {config.embedding_cache_size} entry limit, so "
            "entries were evicted and training will still miss. Raise "
            "embedding_cache_size.")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
