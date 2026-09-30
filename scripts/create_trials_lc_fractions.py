#!/usr/bin/env python3
"""Create nested Trials learning-curve training fractions.

The canonical Trials training split is shuffled reproducibly by the topic-
disjoint splitter (seed 42) and contains grade-2/eligible pairs only.  Nested
prefixes therefore vary the amount of positive Phase-1 supervision while keeping
the graded negative pool, validation topics, and test year fixed.

The subset is deliberately independent of the optimization seed.  Seeds 42, 13,
and 21 must see byte-identical data at a given fraction; otherwise variation due
to data selection would be mislabeled as model-seed variation.

Validation is symlinked to the canonical validation split.  The test split is
neither read nor linked by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "preprocess" / "trec_ct_splits"
OUTPUT_DIR = ROOT / "preprocess" / "trec_ct_lc"
DEFAULT_FRACTIONS = (10, 25, 50, 75, 100)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_count(total: int, pct: int) -> int:
    return int(total * pct / 100.0 + 0.5)


def replace_symlink(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        if link.is_file() and link.read_bytes() == target.read_bytes():
            return
        raise FileExistsError(f"refusing to replace existing {link}")
    os.symlink(os.path.relpath(target, link.parent), link)


def build(fractions: Iterable[int], dry_run: bool = False) -> dict:
    source_train = SOURCE_DIR / "train.jsonl"
    source_val = SOURCE_DIR / "validation.jsonl"
    lines = source_train.read_bytes().splitlines(keepends=True)
    validation_lines = source_val.read_bytes().splitlines(keepends=True)
    positive_validation_lines = [
        line for line in validation_lines
        if json.loads(line).get("label") in (1, "1", True)
    ]
    if not lines:
        raise RuntimeError(f"empty training split: {source_train}")

    fractions = tuple(sorted(set(fractions)))
    if not fractions or fractions[0] <= 0 or fractions[-1] > 100:
        raise ValueError("fractions must be unique integers in [1, 100]")

    counts = {pct: target_count(len(lines), pct) for pct in fractions}
    manifest = {
        "source_train": str(source_train.relative_to(ROOT)),
        "source_validation": str(source_val.relative_to(ROOT)),
        "source_train_records": len(lines),
        "source_train_sha256": sha256(source_train),
        "source_validation_sha256": sha256(source_val),
        "source_validation_records": len(validation_lines),
        "training_objective_validation_records": len(positive_validation_lines),
        "training_objective_validation":
            "grade-2 positives only; graded negatives come from the validation selector",
        "split_seed": 42,
        "sampling": "nested prefixes of the canonical seed-42-shuffled training split",
        "fraction_unit": "positive training pairs",
        "graded_negative_pool": "fixed canonical per-topic pool",
        "optimization_seed_affects_subset": False,
        "test_split_touched": False,
        "fractions": {str(pct): counts[pct] for pct in fractions},
    }

    if dry_run:
        return manifest

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation_positive.jsonl").write_bytes(
        b"".join(positive_validation_lines))
    for pct in fractions:
        fraction_dir = OUTPUT_DIR / f"frac_{pct}"
        fraction_dir.mkdir(parents=True, exist_ok=True)
        train_path = fraction_dir / "train.jsonl"
        train_path.write_bytes(b"".join(lines[:counts[pct]]))
        replace_symlink(fraction_dir / "validation.jsonl", source_val)

    (OUTPUT_DIR / "fraction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fractions", nargs="+", type=int,
                        default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = build(args.fractions, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
