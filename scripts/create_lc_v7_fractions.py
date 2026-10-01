#!/usr/bin/env python3
"""Create additional nested stratified learning-curve fractions for career v7.

Extends the existing chain to 5 / 15 / 20 percent, densifying the low-data region
where the ontology effect is largest and where the two earlier run series disagree
most sharply.

Protocol — matched to ``scripts/create_10pct_split_v7.py``
---------------------------------------------------------
* **Nested.** ``frac_5 ⊂ frac_10 ⊂ frac_15 ⊂ frac_20 ⊂ frac_25``. Nesting is what
  makes a learning curve low-variance: consecutive points share their data, so a
  change between fractions reflects the added records rather than an independent
  resample of the whole set. The existing ``frac_10`` is treated as fixed and is
  never rewritten, so every number already computed against it stays valid.
* **Stratified** on ``metadata.original_label`` (good_fit / potential_fit /
  no_fit), holding the grade mix of ``frac_25`` (50% no_fit, 25% each of the
  others) at every size. Without this the low fractions would drift in class
  balance and the curve would confound "less data" with "different label mix".
* **Seed 42**, same as the original builder, so the chain is reproducible.
* ``validation.jsonl`` / ``test.jsonl`` are symlinked to ``data_splits_v7``,
  identical across all fractions — the curve must be measured against one fixed
  held-out set. Verified byte-identical across the existing fractions.

Identity is a **hash of the whole record**. These records carry no usable id: there
is no ``sample_id``, ``metadata.resume_id`` / ``job_id`` / ``pair_id`` are absent,
and the top-level ``job_applicant_id`` is ``None`` on all 1,600 records. Keying set
algebra on it silently made every record look identical, so the nested top-up found
an empty remainder and returned the input unchanged — frac_15 and frac_20 both came
out at 640 while reporting nesting as satisfied. The record hash is verified unique
(1,600/1,600 and 640/640) and confirms frac_10 is a subset of frac_25.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

SEED = 42
ROOT = Path("preprocess/learning_curve_v7")
SPLITS_V7 = Path("preprocess/data_splits_v7")

#: Full-data train size (frac_100), the denominator for every percentage.
FULL_N = 6400

#: Percent -> record count. Built in ascending order so each grows the previous.
TARGETS = {5: 320, 15: 960, 20: 1280}

#: The chain each new fraction grows from / is drawn out of.
BASE_25 = ROOT / "frac_25" / "train.jsonl"
BASE_10 = ROOT / "frac_10" / "train.jsonl"

LABEL_KEY = "original_label"


def load(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_of(rec: Dict) -> str:
    return rec.get("metadata", {}).get(LABEL_KEY, "unknown")


def ident(rec: Dict) -> str:
    """Stable per-record identity: a hash of the canonicalized record."""
    return hashlib.sha256(
        json.dumps(rec, sort_keys=True).encode("utf-8")
    ).hexdigest()


def stratified_targets(pool: List[Dict], total: int) -> Dict[str, int]:
    """Per-label counts holding ``pool``'s proportions, summing to ``total``."""
    counts = defaultdict(int)
    for rec in pool:
        counts[label_of(rec)] += 1
    n = len(pool)
    raw = {lab: total * c / n for lab, c in counts.items()}
    out = {lab: int(v) for lab, v in raw.items()}
    # Distribute the rounding remainder to the largest fractional parts so the
    # counts sum exactly to `total`.
    deficit = total - sum(out.values())
    for lab in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:deficit]:
        out[lab] += 1
    return out


def shrink(pool: List[Dict], total: int, rng: random.Random) -> List[Dict]:
    """A stratified subsample of ``pool`` of size ``total``."""
    want = stratified_targets(pool, total)
    by_label = defaultdict(list)
    for rec in pool:
        by_label[label_of(rec)].append(rec)
    picked: List[Dict] = []
    for lab in sorted(by_label):
        k = min(want.get(lab, 0), len(by_label[lab]))
        picked.extend(rng.sample(by_label[lab], k))
    rng.shuffle(picked)
    return picked


def grow(keep: List[Dict], universe: List[Dict], total: int,
         rng: random.Random) -> List[Dict]:
    """``keep`` plus a stratified top-up from ``universe``, to size ``total``.

    Guarantees ``keep`` is a subset of the result, which is what makes the chain
    nested.
    """
    want = stratified_targets(universe, total)
    kept_ids = {ident(r) for r in keep}
    have = defaultdict(int)
    for rec in keep:
        have[label_of(rec)] += 1

    remainder = defaultdict(list)
    for rec in universe:
        if ident(rec) not in kept_ids:
            remainder[label_of(rec)].append(rec)

    out = list(keep)
    for lab in sorted(want):
        need = want[lab] - have.get(lab, 0)
        if need > 0:
            pool = remainder.get(lab, [])
            out.extend(rng.sample(pool, min(need, len(pool))))

    # If stratified top-up fell short (a label was exhausted), fill from whatever
    # remains so the requested size is still met.
    if len(out) < total:
        present = {ident(r) for r in out}
        spare = [r for r in universe if ident(r) not in present]
        rng.shuffle(spare)
        out.extend(spare[: total - len(out)])

    rng.shuffle(out)
    return out


def link_eval_splits(out_dir: Path) -> None:
    """Symlink validation/test to the canonical v7 splits."""
    for name in ("validation.jsonl", "test.jsonl"):
        src = SPLITS_V7 / name
        dst = out_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.relpath(src, out_dir), dst)


def write(out_dir: Path, records: List[Dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    link_eval_splits(out_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without writing")
    args = ap.parse_args(argv)

    rng = random.Random(SEED)
    universe = load(BASE_25)
    frac_10 = load(BASE_10)
    print(f"frac_25 (universe): {len(universe)} records")
    print(f"frac_10 (fixed):    {len(frac_10)} records")

    # 5% shrinks out of the existing frac_10 so frac_5 subset frac_10.
    built: Dict[int, List[Dict]] = {}
    built[5] = shrink(frac_10, TARGETS[5], rng)
    # 15% and 20% grow upward, each containing the previous.
    built[15] = grow(frac_10, universe, TARGETS[15], rng)
    built[20] = grow(built[15], universe, TARGETS[20], rng)

    chain = {5: built[5], 10: frac_10, 15: built[15], 20: built[20], 25: universe}
    print("\nnesting + stratification check")
    print("-" * 62)
    order = sorted(chain)
    ok = True
    for pct in order:
        recs = chain[pct]
        counts = defaultdict(int)
        for r in recs:
            counts[label_of(r)] += 1
        share = {k: f"{v / len(recs):.0%}" for k, v in sorted(counts.items())}
        print(f"  {pct:3d}%  n={len(recs):5d}  target={int(FULL_N*pct/100):5d}  {share}")
    for a, b in zip(order, order[1:]):
        sub = {ident(r) for r in chain[a]} <= {ident(r) for r in chain[b]}
        ok &= sub
        print(f"  frac_{a} subset of frac_{b}: {sub}")

    # Size check, separate from nesting. Nesting alone is not enough: a broken
    # identity key made the top-up a no-op, which left every fraction at 640 while
    # reporting nesting as satisfied (a subset relation trivially holds when the
    # sets are equal). Both must pass.
    for pct, recs in built.items():
        if len(recs) != TARGETS[pct]:
            print(f"\nSIZE MISMATCH at frac_{pct}: got {len(recs)}, "
                  f"want {TARGETS[pct]} — refusing to write.")
            ok = False

    if not ok:
        print("\nCHECKS FAILED — refusing to write.")
        return 1

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    for pct in TARGETS:
        out = ROOT / f"frac_{pct}"
        write(out, built[pct])
        print(f"wrote {out}/train.jsonl ({len(built[pct])} records) + val/test symlinks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
