#!/usr/bin/env python3
"""Create nested GO/PPI learning-curve training fractions.

The canonical GO/PPI training split is already shuffled reproducibly by the
protein-disjoint splitter (seed 42), and contains only grade-2 positives.  A
nested prefix therefore preserves the single training label while ensuring
that every smaller fraction is a strict subset of every larger fraction.

Validation is symlinked to the canonical validation split so checkpoint
selection is identical at every fraction.  The untouched test split is not
read or linked by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "preprocess" / "go_ppi_splits"
OUTPUT_DIR = ROOT / "preprocess" / "go_ppi_lc"
DEFAULT_FRACTIONS = (5, 10, 25, 50, 75, 100)


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
        "test_split_touched": False,
        "fractions": {str(pct): counts[pct] for pct in fractions},
    }

    print(f"source train: {len(lines)} records")
    for pct in fractions:
        print(f"  frac_{pct}: {counts[pct]} records")
        if dry_run:
            continue
        out = OUTPUT_DIR / f"frac_{pct}"
        out.mkdir(parents=True, exist_ok=True)
        train = out / "train.jsonl"
        expected = b"".join(lines[:counts[pct]])
        if train.exists():
            if train.read_bytes() != expected:
                raise FileExistsError(f"refusing to replace non-matching {train}")
        else:
            train.write_bytes(expected)
        replace_symlink(out / "validation.jsonl", source_val)

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "validation_positive.jsonl").write_bytes(
            b"".join(positive_validation_lines))
        (OUTPUT_DIR / "fraction_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fractions", nargs="+", type=int,
                        default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args.fractions, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
