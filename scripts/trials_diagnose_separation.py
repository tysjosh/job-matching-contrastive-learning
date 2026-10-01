#!/usr/bin/env python3
"""Diagnose whether trials embeddings separate the TREC grades at all.

Answers two questions that the aggregate binary AUC cannot:

1. **What does chance look like here?** Compares the trained projection head
   against an *untrained* head with the same architecture and seed, and against
   the raw frozen-encoder embeddings with no projection. Without this reference
   an AUC of 0.515 is uninterpretable — it could be a broken pipeline or it could
   be the ceiling of the representation.

2. **Is a low binary AUC a mixing artifact?** The binary label collapses grade 1
   (ineligible: the patient has the condition but fails eligibility) and grade 0
   (not relevant) into a single negative class. MeSH separates grade 2 from
   grade 0 well (Cohen's d = 0.91 measured on the ontology alone) but is nearly
   blind to grade 2 vs grade 1 (d = 0.21). If the model recovers that ordering,
   a poor pooled AUC is expected rather than a failure, and the per-pair
   breakdown will show it.

Reads the split's cached text embeddings so no re-encoding is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contrastive_learning.data_structures import TrainingConfig  # noqa: E402
# Registers the "trials" domain adapter as an import-time side effect. Required
# before a trainer is constructed, because its DataLoader resolves
# config.domain_adapter immediately and raises KeyError for an unknown name.
import trials_domain.record_adapter  # noqa: F401,E402


def _auc(pos: List[float], neg: List[float]) -> Optional[float]:
    """Rank-based AUC (Mann-Whitney U), tie-corrected. ``None`` if a side is empty."""
    if not pos or not neg:
        return None
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks: Dict[int, float] = {}
    i = 0
    idx = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_v, lab) in enumerate(merged) if lab == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _cohens_d(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    return (ma - mb) / (pooled ** 0.5) if pooled > 0 else None


def _build_model(config: TrainingConfig, state_dict=None):
    """A projection head with this config's architecture, optionally loaded."""
    from contrastive_learning.trainer import ContrastiveLearningTrainer

    torch.manual_seed(int(getattr(config, "training_seed", 42)))
    trainer = ContrastiveLearningTrainer(config=config, output_dir="/tmp/_diag_trainer")
    model = trainer.model
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    return model, trainer


def _project(model, emb: torch.Tensor) -> torch.Tensor:
    """Run one text embedding through the projection head, duck-typed."""
    with torch.no_grad():
        for name in ("forward_one", "project", "encode"):
            fn = getattr(model, name, None)
            if callable(fn):
                try:
                    return fn(emb.unsqueeze(0)).squeeze(0)
                except Exception:
                    continue
        out = model(emb.unsqueeze(0))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.squeeze(0)


def _report(title: str, sims_by_grade: Dict[int, List[float]]) -> None:
    g2 = sims_by_grade.get(2, [])
    g1 = sims_by_grade.get(1, [])
    g0 = sims_by_grade.get(0, [])
    print(f"\n  {title}")
    for grade, name in ((2, "eligible"), (1, "ineligible"), (0, "not_relevant")):
        v = sims_by_grade.get(grade, [])
        if v:
            mean = sum(v) / len(v)
            sd = (sum((x - mean) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
            print(f"    grade {grade} {name:14s} n={len(v):6d} sim={mean:.4f}±{sd:.4f}")

    def fmt(x):
        return "  n/a " if x is None else f"{x:+.4f}"

    print(f"    AUC  g2 vs g0 (easy)      : {fmt(_auc(g2, g0))}"
          f"    d={fmt(_cohens_d(g2, g0))}")
    print(f"    AUC  g2 vs g1 (eligibility): {fmt(_auc(g2, g1))}"
          f"    d={fmt(_cohens_d(g2, g1))}")
    print(f"    AUC  g2 vs {{g1,g0}} pooled : {fmt(_auc(g2, g1 + g0))}"
          f"    d={fmt(_cohens_d(g2, g1 + g0))}")
    print(f"    AUC  g1 vs g0 (ambiguity)  : {fmt(_auc(g1, g0))}"
          f"    d={fmt(_cohens_d(g1, g0))}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="preprocess/trec_ct_splits/validation.jsonl")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-per-grade", type=int, default=200,
                    help="subsample this many records per TREC grade "
                         "(0 = use all; encoding is the bottleneck)")
    args = ap.parse_args(argv)

    config = TrainingConfig.from_json(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    trained, trainer = _build_model(config, ckpt["model_state_dict"])
    untrained, _ = _build_model(config, None)

    cache = trainer.embedding_cache
    key_fn = cache.get_content_key
    # Populate from the on-disk cache written by the earlier evaluation run, so
    # this diagnostic never re-encodes.
    try:
        cache.load_from_disk(getattr(config, "embedding_cache_path",
                                     "embedding_cache/text_embeddings.pt"))
    except Exception as exc:
        print(f"WARNING: could not load the disk cache ({exc}); "
              f"only in-memory entries will be available.")
    print(f"Embedding cache holds {len(cache.cache)} entries")

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    print(f"Loaded {len(rows)} records from {args.dataset}")

    # Subsample per grade. Encoding is the bottleneck (~5 texts/sec on CPU) and a
    # separation estimate does not need the full split: a few hundred per grade
    # pins the mean to well inside the effect sizes we are trying to detect.
    if args.max_per_grade:
        import random as _random

        buckets: Dict[int, List[dict]] = defaultdict(list)
        for row in rows:
            buckets[row["metadata"]["trec_grade"]].append(row)
        rng = _random.Random(int(getattr(config, "training_seed", 42)))
        rows = []
        for grade in sorted(buckets):
            pool = buckets[grade]
            rows.extend(pool if len(pool) <= args.max_per_grade
                        else rng.sample(pool, args.max_per_grade))
        print(f"Subsampled to {len(rows)} records "
              f"(<= {args.max_per_grade} per grade)")

    raw: Dict[int, List[float]] = defaultdict(list)
    tr: Dict[int, List[float]] = defaultdict(list)
    un: Dict[int, List[float]] = defaultdict(list)
    missing = 0

    emb_cache: Dict[str, torch.Tensor] = {}

    encoded = [0]

    def get(slot, content_type) -> Optional[torch.Tensor]:
        """Cached text embedding, encoding on miss.

        The evaluation script does not persist to the shared cache file, so a
        validation/test slot is usually absent even after an eval run. Encoding on
        miss keeps the diagnostic self-sufficient.
        """
        key = key_fn(slot)
        if key in emb_cache:
            return emb_cache[key]
        e = cache.cache.get(key)
        if e is None:
            enc = getattr(trainer, "_encode_content_to_text_embedding", None)
            if not callable(enc):
                return None
            try:
                with torch.no_grad():
                    e = enc(slot, content_type)
                encoded[0] += 1
                if encoded[0] % 100 == 0:
                    print(f"    encoded {encoded[0]} texts...")
            except Exception as exc:
                print(f"    encode failed for {key}: {exc}")
                return None
        e = e.detach().float().reshape(-1)
        emb_cache[key] = e
        return e

    proj_cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def proj(model, tag, slot, emb):
        key = (key_fn(slot), tag)
        if key not in proj_cache:
            proj_cache[key] = _project(model, emb)
        return proj_cache[key]

    for row in rows:
        grade = row["metadata"]["trec_grade"]
        a_raw = get(row["resume"], "resume")
        b_raw = get(row["job"], "job")
        if a_raw is None or b_raw is None:
            missing += 1
            continue
        raw[grade].append(float(F.cosine_similarity(a_raw, b_raw, dim=0)))
        for model, tag, store in ((trained, "tr", tr), (untrained, "un", un)):
            za = proj(model, tag, row["resume"], a_raw)
            zb = proj(model, tag, row["job"], b_raw)
            store[grade].append(float(F.cosine_similarity(za, zb, dim=0)))

    if missing:
        print(f"WARNING: {missing} records had no cached embedding and were skipped. "
              f"Run the evaluation once first so the cache is populated.")
    if not raw:
        print("No cached embeddings found — nothing to diagnose.")
        return 1

    print("\n" + "=" * 74)
    print("GRADE SEPARATION DIAGNOSTIC")
    print("=" * 74)
    _report("RAW frozen encoder (no projection) — representation ceiling", raw)
    _report("UNTRAINED projection head (same seed) — chance reference", un)
    _report("TRAINED projection head — the checkpoint under test", tr)

    print("\n" + "-" * 74)
    print("How to read this:")
    print("  * TRAINED ~ UNTRAINED  -> training moved nothing; suspect the")
    print("    pipeline or an undertrained head, not the task.")
    print("  * g2-vs-g0 strong but g2-vs-g1 flat -> the model learned topical")
    print("    relevance but not eligibility. A weak POOLED AUC is then a mixing")
    print("    artifact, since grade 0 outnumbers grade 1 in the negative class.")
    print("  * RAW already strong -> the frozen encoder carries the signal and the")
    print("    projection head is discarding it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
