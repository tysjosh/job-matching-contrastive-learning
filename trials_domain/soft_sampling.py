"""Full-pool, grade-preserving sampling for the second MeSH study.

Scores are frozen before training. Equal scores have equal probability, missing
scores get the mean observed score, and a constant/unavailable signal is uniform.
Each draw mixes uniform and guided probabilities, without replacement.
"""
from __future__ import annotations

import math
import random


def distribution(scores, temperature):
    if temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    if not scores:
        return []
    known = [s for s in scores if s is not None]
    if any(not math.isfinite(s) for s in known):
        raise ValueError("non-finite mining score")
    if not known or max(known) == min(known):
        return [1 / len(scores)] * len(scores)
    mean = sum(known) / len(known)
    filled = [mean if s is None else s for s in scores]
    top = max(filled)
    weights = [math.exp((s - top) / temperature) for s in filled]
    total = sum(weights)
    return [w / total for w in weights]


def sample(pool, count, rng, signals, mix, temperature):
    """Recompute the mixture over remaining candidates after every draw."""
    if not 0 <= mix < 1:
        raise ValueError("mix must be in [0, 1) to retain full-pool support")
    remaining = sorted(set(pool))
    selected = []
    for _ in range(min(count, len(remaining))):
        probs = [distribution([signal.get(n) for n in remaining], temperature)
                 for signal in signals]
        uniform = 1 / len(remaining)
        weights = [(1 - mix) * uniform + mix * (
            sum(p[i] for p in probs) / len(probs) if probs else uniform)
            for i in range(len(remaining))]
        index = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        selected.append(remaining.pop(index))
    return selected


def prepare_text_scores(trainer, selector, converted_dir):
    """Reuse detached encoder caches; never mine from test topics or labels."""
    import json
    from pathlib import Path
    import torch
    import torch.nn.functional as F

    if selector.soft_guidance not in {"text", "hybrid"}:
        return 0
    if not trainer.freeze_text_encoder:
        raise ValueError("V2 mining requires the shared frozen text encoder")
    cache = trainer.embedding_cache

    def vector(view, role):
        key = cache.get_content_key(view)
        value = cache.cache.get(key)
        if value is None:
            with torch.no_grad():
                value = trainer._encode_content_to_text_embedding(view, role)
        if value is None or not torch.isfinite(value).all():
            raise ValueError("invalid frozen mining embedding")
        return F.normalize(value.detach().float().cpu().reshape(-1), dim=0)

    train_topics = set(selector.pools) - selector.common_validation_topics
    topics = {}
    with (Path(converted_dir) / "topics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["topic_id"] in train_topics:
                topics[row["topic_id"]] = row
    if set(topics) != train_topics:
        raise ValueError("missing training topics for frozen text mining")
    trial_vectors = {}
    for topic, row in topics.items():
        anchor = vector({"encoder_view": row["encoder_view"]}, "resume")
        candidates = sorted(set(selector.pools[topic]["not_relevant"]))
        values = {}
        for n in candidates:
            if n not in selector.views:
                raise ValueError(f"missing trial view: {n}")
            if n not in trial_vectors:
                trial_vectors[n] = vector(selector.views[n], "job")
            values[n] = float(anchor @ trial_vectors[n])
        selector.text_scores[topic] = values
    return sum(map(len, selector.text_scores.values()))
