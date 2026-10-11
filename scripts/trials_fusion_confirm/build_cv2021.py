#!/usr/bin/env python3
"""Build the paraphrase-aware 5-fold 2021 CV files (protocol_trials_fusion.md §3), label-free.

Groups = connected components of the fixed paraphrase list ∪ frozen-cache cosine >= 0.80
edges. Asserts the expected added edges, the 5-topic component, max component size,
the 23-topic ``exposed_closed`` set, topic-disjointness and co-located groups. Writes
``fold_{k}/{train,validation,validation_positive,test,negative_pools}.jsonl`` plus
``fold_manifest.json`` per fold and ``cv2021_manifest.json`` (the §7 ``cohort2021`` block).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trials_domain import fusion_confirm as fc  # noqa: E402


def main(argv=None) -> int:
    fc.require_hashseed()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--converted-dir", type=Path, default=Path("preprocess/trec_ct"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold-seed", type=int, default=2021)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cosine-threshold", type=float, default=0.80)
    ap.add_argument("--max-component", type=int, default=6)
    ap.add_argument("--device", choices=["cpu"], default="cpu")
    ap.add_argument("--frozen-cache", type=Path, default=Path(fc.ORIGINAL_CACHE_REL))
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    fc.require_cpu_scoring(args.device)
    out = fc.refuse_existing(args.out)

    import torch
    from contrastive_learning.embedding_cache import EmbeddingCache
    from trials_domain.data_splitter import _ontology_slot

    conv = fc.absp(args.converted_dir)
    topics = {r["topic_id"]: r for r in fc.read_jsonl(conv / "topics.jsonl") if r["year"] == "2021"}
    if len(topics) != 75:
        raise fc.ProtocolError(f"expected 75 2021 topics, found {len(topics)}")
    trials = {r["nct_id"]: r for r in fc.read_jsonl(conv / "trials.jsonl")}
    graded = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in fc.read_jsonl(conv / "qrels.jsonl"):
        if row["topic_id"] in topics:
            graded[row["topic_id"]][int(row["grade"])].append(row["nct_id"])

    # --- paraphrase cosine (label-free): frozen cache, fresh CPU encode only if missing
    cache_sha_before = fc.file_sha256(args.frozen_cache)
    frozen = fc.load_frozen_cache(args.frozen_cache)
    keyer = EmbeddingCache(max_cache_size=0, device=torch.device("cpu"), enable_stats=False)
    vectors, fresh = {}, []
    for tid in fc.sort_topics(topics):
        slot = _ontology_slot(topics[tid], "topic_id", tid)
        v = frozen.get(keyer.get_content_key(slot))
        if v is None:
            fresh.append(tid)
            continue
        vectors[tid] = (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
    if fresh:
        fc.require_hf_offline()
        from sentence_transformers import SentenceTransformer
        _, path = fc.resolve_mpnet_snapshot("e8c3b32edf5434bc2275fc9bab85f82640a19130")
        enc = SentenceTransformer(path, device="cpu")
        from run_phase1_embedding_evaluation import content_to_text
        texts = [content_to_text(_ontology_slot(topics[t], "topic_id", t), "resume") for t in fresh]
        for t, v in zip(fresh, enc.encode(texts, batch_size=64, convert_to_numpy=True)):
            vectors[t] = v
        print(f"freshly encoded topics (not in cache): {fresh}")
    if fc.file_sha256(args.frozen_cache) != cache_sha_before:
        raise fc.ProtocolError("frozen cache changed while reading it")

    cos = fc.cosine_edges(vectors, args.cosine_threshold)
    if cos["near"]:
        raise fc.ProtocolError(f"cosine within 1e-6 of {args.cosine_threshold}: {cos['near']}; amend before building folds")
    fixed = {frozenset(e) for e in fc.FIXED_PARAPHRASE_EDGES}
    added = [e for e in cos["edges"] if frozenset(e[:2]) not in fixed]
    if {frozenset(e[:2]) for e in added} != set(fc.EXPECTED_ADDED_EDGES):
        raise fc.ProtocolError(f"cosine-added edges {added} differ from the expected set; stop and amend")
    edges = [list(e) for e in fc.FIXED_PARAPHRASE_EDGES] + [e[:2] for e in added]
    components = fc.connected_components(fc.sort_topics(topics), edges)
    multi = [c for c in components if len(c) > 1]
    if max(len(c) for c in components) > args.max_component:
        raise fc.ProtocolError(f"component larger than {args.max_component}: {max(components, key=len)}")
    if tuple(max(components, key=len)) != fc.EXPECTED_LARGEST_COMPONENT:
        raise fc.ProtocolError(f"largest component {max(components, key=len)} != expected")
    closed = fc.exposure_closure(fc.EXPOSED_TOPICS, components)
    if tuple(closed) != fc.EXPOSED_TOPICS_CLOSED:
        raise fc.ProtocolError(f"exposed_closed {closed} != expected 23-topic set; stop and amend")

    folds = fc.assign_folds(components, args.n_folds, args.fold_seed)
    manifests = {}
    out.mkdir(parents=True)
    for k in range(args.n_folds):
        manifests[k] = fc.build_fold_files(topics, trials, graded, folds, k, out / f"fold_{k}",
                                           components=multi)
    fold_files = {f"fold_{k}/{name}": h for k, m in manifests.items() for name, h in m["sha256"].items()}
    for k in range(args.n_folds):
        fold_files[f"fold_{k}/fold_manifest.json"] = fc.file_sha256(out / f"fold_{k}" / "fold_manifest.json")
    block = {
        "fold_seed": args.fold_seed, "n_folds": args.n_folds, "cosine_threshold": args.cosine_threshold,
        "max_component": args.max_component, "fold_dir": fc.rel(out),
        "exposed_topics": list(fc.EXPOSED_TOPICS),
        "exposed_topics_source": "preprocess/trec_ct_splits/validation.jsonl (metadata.resume_id)",
        "exposed_topics_closed": closed,
        "exposed_topics_closed_rule": "union of paraphrase components containing an exposed_topics member",
        "cosine_source": "frozen cache, _ontology_slot(topic,'topic_id',id), L2 float64",
        "freshly_encoded_topics": fresh,
        "paraphrase_edges_fixed": [list(e) for e in fc.FIXED_PARAPHRASE_EDGES],
        "paraphrase_edges_cosine": cos["edges"],
        "expected_added_edges": [sorted(e, key=fc.topic_key) for e in fc.EXPECTED_ADDED_EDGES],
        "added_edges": added,
        "pairs_cosine_0_70_to_0_80": cos["audit"],
        "components": multi,
        "folds": {str(k): f for k, f in enumerate(folds)},
        "outer": [fc.outer_split(k, args.n_folds) for k in range(args.n_folds)],
        "fold_files": fold_files,
        "records": {f"fold_{k}": m["records"] for k, m in manifests.items()},
        "skipped_missing_trials": {f"fold_{k}": m["skipped_missing_trials"] for k, m in manifests.items()},
        "seeds": [13, 21, 42],
        "run_env": fc.run_env_record(),
    }
    (out / "cv2021_manifest.json").write_text(json.dumps(block, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"folds": {k: len(f) for k, f in enumerate(folds)}, "components": multi,
                      "added_edges": added, "records": block["records"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
