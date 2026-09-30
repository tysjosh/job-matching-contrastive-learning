"""Attach one label-budget arm to a trainer, and run the arm end to end.

Wires whichever negative source the arm requires, through the domain-selector slot
so the ORCA orchestrator cannot clear it. The three arms differ in exactly one
factor each:

  ``full``          graded negatives from qrels        (all judgments)
  ``low_random``    uniform from the unjudged corpus   (few judgments, no knowledge)
  ``low_ontology``  MeSH-tiered from unjudged corpus   (few judgments, + ontology)

Kept deliberately independent of ``OrcaPhaseOrchestrator``: this study trains
single-phase so the schedule cannot confound the data-efficiency contrast, which
is the mistake the ER-SCHED result exposed in the earlier comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set

logger = logging.getLogger(__name__)

ARM_SELECTOR = {
    "full": "graded",
    "low_random": "corpus_random",
    "low_ontology": "corpus_mesh",
}


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _view(trial: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "nct_id": trial["nct_id"],
        "encoder_view": trial["encoder_view"],
        "title": trial.get("title", ""),
        "skill_uris": trial.get("mesh_uris", []),
        "coarse_uris": trial.get("condition_uris", []),
    }


def _judged_by_topic(converted_dir: Path) -> Dict[str, Set[str]]:
    """``{topic_id: judged nct_ids}`` — excluded from corpus negatives.

    Without this exclusion a corpus arm could hand a topic one of its own judged
    trials, leaking label information into an arm whose whole premise is that no
    judgments are available. That would void the comparison.
    """
    out: Dict[str, Set[str]] = {}
    for row in _read_jsonl(converted_dir / "qrels.jsonl"):
        out.setdefault(str(row["topic_id"]), set()).add(row["nct_id"])
    return out


def build_selector(arm: str, config, matcher=None):
    """Construct the negative selector for ``arm``."""
    if arm not in ARM_SELECTOR:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_SELECTOR)}")

    split_dir = Path(getattr(config, "trials_split_dir", "preprocess/trec_ct_splits"))
    converted_dir = Path(getattr(config, "trials_converted_dir", "preprocess/trec_ct"))
    total_epochs = int(getattr(config, "num_epochs", 10))
    seed = int(getattr(config, "training_seed", 42))

    if arm == "full":
        from trials_domain.negative_selector import TrialsNegativeSelector

        return TrialsNegativeSelector.from_split_dir(
            split_dir, converted_dir, "train", matcher=matcher,
            total_epochs=total_epochs, training_seed=seed,
            start_hard_ratio=float(getattr(config, "trials_start_hard_ratio", 0.2)),
            end_hard_ratio=float(getattr(config, "trials_end_hard_ratio", 0.6)),
        )

    pool_path = converted_dir / "unjudged_pool.jsonl"
    if not pool_path.exists():
        raise FileNotFoundError(
            f"{pool_path} is required for arm {arm!r}. Regenerate it with "
            f"'python -m trials_domain.data_converter --include-unjudged N'."
        )

    # Cap the in-memory pool. Two hard constraints, both hit at 12,000:
    #   * memory — the view dicts carry full encoder_view text and the process was
    #     OOM-killed alongside the MeSH index, the encoder, and a growing cache;
    #   * encoding cost — every unseen negative is a cache miss, and at ~5 texts/s
    #     on CPU a 12,000 pool is ~40 minutes of encoding per arm before the cache
    #     warms, which showed up as 44.7 s/batch.
    # A smaller pool costs little discriminative power: the MeSH selector already
    # scores only ``score_cap`` (400) candidates per anchor, so anything beyond a
    # few thousand is never compared against anyway.
    pool_size = int(getattr(config, "trials_corpus_pool_size", 2000))
    views: Dict[str, Dict[str, Any]] = {}
    for trial in _read_jsonl(pool_path):
        if pool_size and len(views) >= pool_size:
            break
        views[trial["nct_id"]] = _view(trial)
    logger.info("Corpus negative pool: %d trials (cap=%d)", len(views), pool_size)
    judged = _judged_by_topic(converted_dir)

    if arm == "low_random":
        from trials_domain.corpus_negative_selector import CorpusRandomNegativeSelector

        return CorpusRandomNegativeSelector(
            views, judged=judged, total_epochs=total_epochs, training_seed=seed)

    from trials_domain.corpus_negative_selector import CorpusMeshNegativeSelector

    if matcher is None:
        raise RuntimeError(
            "arm 'low_ontology' requires a MeSH matcher; without it the arm is "
            "identical to low_random and the contrast measures nothing."
        )
    # score_cap bounds how many pool candidates are MeSH-scored per anchor per
    # epoch. It is the dominant cost: at 400 it produced ~25,600 set-similarity
    # computations per batch and ~37 s/batch. 100 still leaves enough candidates
    # to form meaningful terciles while cutting that cost 4x.
    return CorpusMeshNegativeSelector(
        views, matcher, judged=judged, total_epochs=total_epochs,
        training_seed=seed,
        start_hard_ratio=float(getattr(config, "trials_start_hard_ratio", 0.2)),
        end_hard_ratio=float(getattr(config, "trials_end_hard_ratio", 0.6)),
        score_cap=int(getattr(config, "trials_mesh_score_cap", 100)),
    )


def attach_arm(trainer, config, arm: str) -> Dict[str, Any]:
    """Attach the arm's ontology matcher (if any) and negative selector."""
    import trials_domain.record_adapter  # noqa: F401  registers "trials"

    batch_processor = getattr(trainer, "batch_processor", None)
    if batch_processor is None:
        raise RuntimeError("attach_arm requires a trainer exposing .batch_processor")

    # The ontology matcher is attached for the graded and ontology arms only.
    # low_random must have NO matcher, so its career_distances and any ORCA
    # feature capture carry no ontology signal — otherwise the arms would differ
    # in more than the negative source.
    # Only the ontology arm gets a matcher, so each arm has exactly ONE
    # supervision source: full = judgments, low_random = none, low_ontology =
    # ontology. ``full`` previously received one too. It was provably inert there
    # (the graded selector picks by grade, and the matcher only filled
    # ``career_distances``, whose *values* the loss engine never reads — it uses
    # the list only for its length), but that inertness was incidental rather than
    # enforced: it depended on pathway_weight being 0 and ORCA being off. Leaving
    # it attached also made the arms asymmetric, since low_random had none.
    matcher = None
    if arm == "low_ontology":
        from trials_domain.run_config import build_mesh_matcher

        matcher = build_mesh_matcher(config)
        setter = getattr(batch_processor, "set_ontology_matcher", None)
        if not callable(setter):
            raise RuntimeError(
                "BatchProcessor does not expose set_ontology_matcher; cannot "
                "supply MeSH features for arm %r." % arm)
        setter(matcher, coarse_distance_fn=matcher.branch_distance)

    selector = build_selector(arm, config, matcher=matcher)

    setter = getattr(batch_processor, "set_domain_negative_selector", None)
    if not callable(setter):
        raise RuntimeError(
            "BatchProcessor does not expose set_domain_negative_selector; the "
            "arm's negatives cannot be supplied.")
    setter(selector)

    summary = {
        "arm": arm,
        "selector": type(selector).__name__,
        "matcher": type(matcher).__name__ if matcher is not None else None,
        "pool": selector.pool_summary(),
    }
    logger.info("Label-budget arm attached: %s", summary)
    return summary


def preencode_pool(trainer, selector, batch_size: int = 64) -> int:
    """Encode every candidate in the selector's pool once, so the cost is explicit.

    Lazily encoding the pool as negatives are drawn makes the first epochs
    dominated by cache misses (measured at 44.7 s/batch) and hides the cost inside
    training time.

    IMPORTANT — what gets cached
    ---------------------------
    This must cache **pre-projection text embeddings** (768-d for mpnet), never the
    projection head's output. ``BatchEfficientEncoder.encode_batch`` runs
    ``self.model(text_embeddings)`` and returns 128-d *projected* vectors, so using
    it here is doubly wrong:

      * it mixes 128-d entries into a cache the training path fills with 768-d
        text embeddings, which fails at ``torch.stack`` with "expected each tensor
        to be equal size, but got [768] ... and [128]";
      * projected vectors are a function of the weights being trained, so they are
        not a cacheable artifact at all — they go stale after every optimizer step.

    The text encoder is frozen (``freeze_text_encoder: true``), which is what makes
    the *text* embeddings safe to cache and reuse.
    """
    views = getattr(selector, "views", None)
    if not views:
        return 0

    cache = trainer.embedding_cache

    # Load the on-disk cache BEFORE deciding what needs encoding. ``trainer.train``
    # only loads it later, inside ``preload_dataset_embeddings``, so without this
    # every arm re-encodes a pool a previous arm already paid for — measured at
    # ~5 min for a 2,000-trial pool and ~50 min for the full arm's 18,604 views.
    cache_path = getattr(trainer.config, "embedding_cache_path", None)
    if cache_path and Path(cache_path).exists():
        try:
            cache.load_from_disk(cache_path)
            logger.info("Loaded %d cached embeddings from %s before pre-encoding.",
                        len(cache.cache), cache_path)
        except Exception as exc:
            logger.warning("Could not load %s: %s", cache_path, exc)

    encoder = getattr(trainer, "_encode_content_to_text_embedding", None)
    if not callable(encoder):
        logger.warning(
            "Trainer exposes no _encode_content_to_text_embedding; skipping "
            "pre-encoding (the pool will be encoded lazily during training).")
        return 0

    pending = [v for v in views.values() if cache.get_content_key(v) not in cache.cache]
    if not pending:
        logger.info("Corpus pool already fully cached (%d trials).", len(views))
        return 0

    logger.info("Pre-encoding %d/%d corpus-pool trials (text embeddings)...",
                len(pending), len(views))
    import torch

    encoded = 0
    expected_dim: Optional[int] = None
    for view in pending:
        try:
            with torch.no_grad():
                emb = encoder(view, "job")
        except Exception as exc:
            # Shared by the trials and go_ppi domains, so identify the view by
            # whichever id key it carries rather than assuming nct_id.
            view_id = (view.get("nct_id") or view.get("partner_id")
                       or view.get("protein_id") or "<unknown>")
            logger.warning("Pre-encode failed for %s: %s", view_id, exc)
            continue

        emb = emb.detach().reshape(-1)
        if expected_dim is None:
            expected_dim = emb.shape[0]
            logger.info("  text embedding dim = %d", expected_dim)
        elif emb.shape[0] != expected_dim:
            # Refuse to write a ragged cache rather than failing later inside a
            # torch.stack during training.
            raise RuntimeError(
                f"Inconsistent embedding dim while pre-encoding: got "
                f"{emb.shape[0]}, expected {expected_dim}. Refusing to write a "
                f"mixed-dimension cache.")

        cache._add_to_cache(cache.get_content_key(view), emb, (view, "job"))
        encoded += 1
        if encoded % (batch_size * 5) == 0:
            logger.info("  pre-encoded %d/%d", encoded, len(pending))

    logger.info("Pre-encoded %d trials at dim %s.", encoded, expected_dim)

    # Persist so the second corpus arm and later seeds reuse the pool for free.
    # Safe now that the cached vectors are the same pre-projection text embeddings
    # the training path caches; the dim check above guarantees homogeneity.
    path = getattr(trainer.config, "embedding_cache_path", None)
    if path and encoded:
        try:
            trainer.embedding_cache.save_to_disk(path)
            logger.info("Persisted embedding cache to %s", path)
        except Exception as exc:
            logger.warning("Could not persist the embedding cache: %s", exc)
    return encoded


def expected_text_dim(trainer) -> Optional[int]:
    """The frozen text encoder's output dimension, or ``None`` if undiscoverable."""
    encoder = getattr(trainer, "text_encoder", None)
    getter = getattr(encoder, "get_sentence_embedding_dimension", None)
    if callable(getter):
        try:
            dim = getter()
            if dim:
                return int(dim)
        except Exception:
            pass
    for attr in ("text_encoder_dim", "text_embed_dim"):
        dim = getattr(trainer, attr, None) or getattr(trainer.config, attr, None)
        if dim:
            return int(dim)
    return None


def assert_cache_valid(trainer) -> Dict[str, Any]:
    """Fail fast if the embedding cache does not hold pre-projection text embeddings.

    Two distinct failures, and checking only the first is not enough:

    1. **Mixed dimensions.** Surfaces late, inside a ``torch.stack`` during batch
       assembly, as "expected each tensor to be equal size, but got [768] ... and
       [128]" — after epochs have already run on partially wrong inputs.

    2. **Uniformly wrong dimensions.** Strictly worse, and the reason a
       homogeneity-only check is inadequate: if *every* cached vector is the
       projection head's 128-d output, nothing ever mismatches, no exception is
       raised, and training proceeds silently on the wrong inputs. The corrupted
       cache this study actually produced was uniformly 128-d — it only errored
       because the trainer later added 768-d entries alongside. Had the pollution
       been complete, the run would have looked healthy and produced numbers.

    So the dimension is checked against the frozen encoder's own output size, not
    merely for internal consistency.
    """
    import collections

    cache = getattr(trainer, "embedding_cache", None)
    store = getattr(cache, "cache", None) or {}
    dims = collections.Counter(
        tuple(v.shape) for v in store.values() if hasattr(v, "shape")
    )
    cache_path = getattr(trainer.config, "embedding_cache_path", "<cache>")

    if len(dims) > 1:
        raise RuntimeError(
            f"Embedding cache holds mixed dimensions {dict(dims)}. Projected "
            f"(post-projection-head) vectors were probably written into a "
            f"text-embedding cache. Delete {cache_path} and rerun."
        )

    expected = expected_text_dim(trainer)
    if dims and expected is not None:
        (shape,) = dims
        if shape != (expected,):
            projection_dim = getattr(trainer.config, "projection_dim", None)
            hint = (
                " That matches projection_dim, so the cache holds the projection "
                "head's output rather than text embeddings — those are a function "
                "of the weights being trained and must never be cached."
                if projection_dim and shape == (projection_dim,) else ""
            )
            raise RuntimeError(
                f"Embedding cache holds {shape} vectors but the frozen text "
                f"encoder emits {expected}-d.{hint} Delete {cache_path} and rerun."
            )

    return {
        "dims": {str(k): v for k, v in dims.items()},
        "expected_text_dim": expected,
        "entries": sum(dims.values()),
    }


#: Retained under the old name so existing callers keep working.
assert_cache_homogeneous = assert_cache_valid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARM_SELECTOR))
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override num_epochs (single-phase training)")
    ap.add_argument("--pool-size", type=int, default=None,
                    help="cap the corpus negative pool (default 2000; 12000 "
                         "OOM-killed the process and ran at 44.7 s/batch)")
    ap.add_argument("--no-preencode", action="store_true",
                    help="skip up-front pool encoding (encodes lazily instead)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import trials_domain.record_adapter  # noqa: F401
    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.trainer import ContrastiveLearningTrainer

    config = TrainingConfig.from_json(args.config)
    config.training_seed = args.seed
    if args.epochs is not None:
        config.num_epochs = args.epochs
    # Single-phase: ORCA off so the staged schedule cannot confound the
    # data-efficiency contrast.
    config.orca_enabled = False

    if args.pool_size is not None:
        config.trials_corpus_pool_size = args.pool_size

    trainer = ContrastiveLearningTrainer(config=config, output_dir=args.output_dir)
    summary = attach_arm(trainer, config, args.arm)

    # Pay the candidate-pool encoding cost up front rather than inside epoch 1.
    # Applies to EVERY arm, ``full`` included: its graded selector indexes 18,604
    # trial views, and encoding those lazily during training is what produced the
    # ~37 s/batch behaviour earlier. Pre-encoding makes the cost explicit and
    # keeps the arms on the same code path.
    if not args.no_preencode:
        selector = trainer.batch_processor.domain_negative_selector
        summary["preencoded"] = preencode_pool(trainer, selector)

    # Guard before training rather than discovering a bad cache mid-run.
    summary["cache_check"] = assert_cache_valid(trainer)

    result = trainer.train(args.train_file)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "arm_summary.json").write_text(
        json.dumps({"arm": args.arm, "seed": args.seed,
                    "epochs": config.num_epochs,
                    "attach": summary,
                    "train_result": _jsonable(result)}, indent=2),
        encoding="utf-8")
    logger.info("Arm %s complete -> %s", args.arm, out)
    return 0


def _jsonable(obj):
    """Best-effort conversion of a trainer result to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    sys.exit(main())
