"""Synthetic GO/PPI fixtures for the withholding-study tests (no real data, no MPNet).

Two fixtures:

* ``write_small_fixture`` -- a tiny pickled ``GoIndex`` plus a split dir and a
  converted dir with two train anchors: ``U`` (no GO annotation) and ``A``
  (annotated). Its neg_only selector output with the flag off was frozen into
  ``fixtures/selector_flag_false_expected.json`` from the UNMODIFIED selector,
  before ``unannotated_anchor_window`` existed. Do not change this builder: the
  frozen file would no longer describe it. Never regenerate the JSON after the
  selector edit.
* ``write_identity_fixture`` -- 60 train anchors with mixed-grade pools (some
  anchors unannotated), plus validation pools and train records, for the
  loss-only == baseline negatives identity test (design section 5.4).

Run ``python _withholding_fixtures.py`` only to create the frozen JSON; it refuses
to overwrite an existing file.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go_ppi_domain.go_ontology import GoIndex, GoMatcher  # noqa: E402
from go_ppi_domain.negative_selector import GoPpiNegativeSelector  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FLAG_FALSE_EXPECTED = FIXTURE_DIR / "selector_flag_false_expected.json"

SMALL_SEEDS = (13, 42)
SMALL_EPOCHS = tuple(range(15))
SMALL_ANCHORS = ("U", "A")
SMALL_MAX_NEGATIVES = 6
SMALL_TOTAL_EPOCHS = 15

# ----------------------------------------------------------------- GO index
#: Terms: R root; T1..T4 children of R; T5 child of T1; T6 child of T2.
_PARENTS = {"R": (), "T1": ("R",), "T2": ("R",), "T3": ("R",), "T4": ("R",),
            "T5": ("T1",), "T6": ("T2",)}


def _ancestors() -> Dict[str, frozenset]:
    out = {}
    for t in _PARENTS:
        acc, stack = {t}, [t]
        while stack:
            for p in _PARENTS[stack.pop()]:
                if p not in acc:
                    acc.add(p)
                    stack.append(p)
        out[t] = frozenset(acc)
    return out


def build_index(direct: Dict[str, Iterable[str]], aspect: str = "P",
                meta_extra: Dict[str, Any] | None = None) -> GoIndex:
    """A GoIndex over the fixed fixture DAG with IC computed from ``direct``."""
    import math

    anc = _ancestors()
    depth = {"R": 0, "T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 2, "T6": 2}
    d, closure, coarse = {}, {}, {}
    for gene, terms in direct.items():
        terms = frozenset(terms)
        if not terms:
            continue
        acc = frozenset().union(*(anc[t] for t in terms))
        d[gene] = terms
        closure[gene] = acc
        coarse[gene] = frozenset(x for x in acc if depth[x] <= 1)
    freq: Dict[str, int] = {}
    for acc in closure.values():
        for t in acc:
            freq[t] = freq.get(t, 0) + 1
    total = len(closure)
    ic = {t: -math.log(n / total) for t, n in freq.items()}
    meta = {"aspect": aspect, "genes_annotated": total}
    meta.update(meta_extra or {})
    return GoIndex(anc, depth, d, closure, coarse, ic, meta)


# ------------------------------------------------------------ small fixture
def _small_direct() -> Dict[str, List[str]]:
    terms = ["T5", "T1", "T2", "T6", "T3", "T4"]
    direct = {"A": ["T5"]}
    for i in range(20):
        pid = f"P{i:02d}"
        if i % 7 == 3:          # a few unannotated pool members
            continue
        direct[pid] = sorted({terms[i % 6], terms[(i * 5) % 6]})
    return direct


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _protein_rows(ids: Iterable[str], index: GoIndex) -> List[Dict[str, Any]]:
    return [{"protein_id": pid, "gene": pid,
             "encoder_view": f"{pid} — synthetic protein {pid}",
             "go_uris": sorted(index.terms_for(pid)),
             "coarse_uris": sorted(index.coarse_for(pid))} for pid in ids]


def write_small_fixture(root: Path) -> Dict[str, Path]:
    """Write the small U/A fixture under ``root``; returns its paths."""
    root = Path(root)
    index = build_index(_small_direct())
    index_path = root / "go_index_P.pkl"
    index.save(index_path)
    members = [f"P{i:02d}" for i in range(20)]
    pools = [
        {"split": "train", "anchor_id": "U",
         "weak_evidence": members[:8], "no_interaction": members[8:]},
        {"split": "train", "anchor_id": "A",
         "weak_evidence": members[4:12], "no_interaction": members[12:] + members[:4]},
    ]
    split_dir, converted_dir = root / "splits", root / "converted"
    _write_jsonl(split_dir / "negative_pools.jsonl", pools)
    _write_jsonl(converted_dir / "proteins.jsonl",
                 _protein_rows(["U", "A", *members], index))
    return {"index": index_path, "split_dir": split_dir,
            "converted_dir": converted_dir}


def small_selector(paths: Dict[str, Path], cell: str, seed: int,
                   **extra) -> GoPpiNegativeSelector:
    """``cell`` in {"baseline", "neg_only"}; the production selector for that cell."""
    tiered = cell == "neg_only"
    matcher = (GoMatcher(GoIndex.load(paths["index"]), similarity_mode="simgic")
               if tiered else None)
    return GoPpiNegativeSelector.from_split_dir(
        paths["split_dir"], paths["converted_dir"], "train", matcher=matcher,
        total_epochs=SMALL_TOTAL_EPOCHS, training_seed=seed, go_tiered=tiered,
        tier_sampling="stochastic" if tiered else "random_window",
        tier_window_frac=0.34, **extra)


def make_sample(anchor: str, positive: str = "POS") -> SimpleNamespace:
    """A minimal TrainingSample stand-in with the attributes the selector reads."""
    return SimpleNamespace(
        metadata={"resume_id": anchor},
        resume={"protein_id": anchor, "skill_uris": [], "coarse_uris": []},
        job={"partner_id": positive, "skill_uris": [], "coarse_uris": []},
    )


def select(selector, anchor: str, epoch: int,
           max_negatives: int = SMALL_MAX_NEGATIVES) -> Dict[str, list]:
    out = selector.select_batch_negatives(make_sample(anchor), [], max_negatives, epoch)
    if out is None:
        return {"partner_ids": None, "grades": None, "distances": None}
    negatives, distances = out
    return {"partner_ids": [n["partner_id"] for n in negatives],
            "grades": [n["grade"] for n in negatives],
            "distances": [float(x) for x in distances]}


def record_neg_only_outputs(paths: Dict[str, Path], **extra) -> Dict[str, Any]:
    """{anchor: {seed: {epoch: select(...)}}} for the neg_only selector."""
    out: Dict[str, Any] = {}
    for anchor in SMALL_ANCHORS:
        out[anchor] = {}
        for seed in SMALL_SEEDS:
            selector = small_selector(paths, "neg_only", seed, **extra)
            out[anchor][str(seed)] = {
                str(e): select(selector, anchor, e) for e in SMALL_EPOCHS}
    return out


# --------------------------------------------------------- identity fixture
IDENTITY_N_ANCHORS = 60


def write_identity_fixture(root: Path) -> Dict[str, Any]:
    """60 train anchors with mixed-grade pools, validation pools, train records.

    Every 7th anchor is unannotated. Returns paths plus the anchor list.
    """
    root = Path(root)
    rnd = random.Random(20251009)
    terms = ["T5", "T1", "T2", "T6", "T3", "T4"]
    anchors = [f"a{i:02d}" for i in range(IDENTITY_N_ANCHORS)]
    partners = [f"q{i:02d}" for i in range(IDENTITY_N_ANCHORS)]
    members = [f"m{i:03d}" for i in range(120)]
    val_anchors = [f"v{i:02d}" for i in range(6)]
    val_members = [f"w{i:02d}" for i in range(20)]
    direct: Dict[str, List[str]] = {}
    for pid in anchors + partners + members + val_anchors + val_members:
        if rnd.random() < 0.15 or (pid in anchors and anchors.index(pid) % 7 == 0):
            continue
        direct[pid] = sorted(set(rnd.sample(terms, rnd.randint(1, 3))))
    index = build_index(direct)
    index_path = root / "go_index_P.pkl"
    index.save(index_path)

    pools = []
    for a in anchors:
        hard = rnd.sample(members, rnd.randint(3, 15))
        easy = rnd.sample([m for m in members if m not in hard], rnd.randint(5, 30))
        pools.append({"split": "train", "anchor_id": a,
                      "weak_evidence": hard, "no_interaction": easy})
    for a in val_anchors:
        pools.append({"split": "validation", "anchor_id": a,
                      "weak_evidence": rnd.sample(val_members, 4),
                      "no_interaction": rnd.sample(val_members, 8)})
    split_dir, converted_dir = root / "splits", root / "converted"
    _write_jsonl(split_dir / "negative_pools.jsonl", pools)
    _write_jsonl(converted_dir / "proteins.jsonl", _protein_rows(
        anchors + partners + members + val_anchors + val_members, index))

    def slot(pid: str) -> Dict[str, Any]:
        return {"protein_id": pid, "encoder_view": f"{pid} — synthetic protein {pid}",
                "skill_uris": sorted(index.terms_for(pid)),
                "coarse_uris": sorted(index.coarse_for(pid))}

    train_rows = []
    for a, p in zip(anchors, partners):
        resume = slot(a)
        job = slot(p)
        job["partner_id"] = job.pop("protein_id")
        job.update({"title": p, "grade": 2, "original_label": "good_fit"})
        train_rows.append({"resume": resume, "job": job, "label": 1,
                           "metadata": {"resume_id": a, "protein_id": a,
                                        "partner_id": p, "original_label": "good_fit",
                                        "go_grade": 2}})
    train_file = root / "train.jsonl"
    _write_jsonl(train_file, train_rows)
    return {"index": index_path, "split_dir": split_dir,
            "converted_dir": converted_dir, "train_file": train_file,
            "anchors": anchors, "partners": dict(zip(anchors, partners)),
            "unannotated": sorted(a for a in anchors if not index.terms_for(a))}


def main() -> int:
    import tempfile

    if FLAG_FALSE_EXPECTED.exists():
        print(f"refusing to overwrite {FLAG_FALSE_EXPECTED}")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        outputs = record_neg_only_outputs(write_small_fixture(Path(tmp)))
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    FLAG_FALSE_EXPECTED.write_text(json.dumps({
        "_note": ("neg_only selector output (go_tiered=True, stochastic, simGIC "
                  "matcher) on write_small_fixture, recorded from the UNMODIFIED "
                  "go_ppi_domain/negative_selector.py before the "
                  "unannotated_anchor_window edit. Never regenerate."),
        "max_negatives": SMALL_MAX_NEGATIVES,
        "outputs": outputs}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {FLAG_FALSE_EXPECTED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
