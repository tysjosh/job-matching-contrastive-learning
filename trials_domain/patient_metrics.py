"""Patient-macro metrics: preserve pooled metrics but avoid cross-patient pairs."""
from collections import defaultdict
import math
import statistics

from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


def patient_report(topic_ids, scores, grades):
    if not (len(topic_ids) == len(scores) == len(grades)):
        raise ValueError("patient metric arrays differ in length")
    if not any(topic_ids):
        return None
    groups = defaultdict(list)
    for topic, score, grade in zip(topic_ids, scores, grades):
        if not topic or grade not in (0, 1, 2) or not math.isfinite(float(score)):
            raise ValueError("missing patient identity, grade or finite score")
        groups[str(topic)].append((int(grade), float(score)))
    per_topic = {}
    for topic, rows in sorted(groups.items()):
        g, s = zip(*rows)
        eligible = [int(x == 2) for x in g]
        metrics = {}
        if any(eligible):
            metrics["eligible_map"] = float(average_precision_score(eligible, s))
            # Explicit gains: eligible=3, ineligible=1, not-relevant=0.
        gains = [{0: 0, 1: 1, 2: 3}[x] for x in g]
        if sum(gains) and len(rows) > 1:
            metrics["graded_ndcg_at_10"] = float(ndcg_score([gains], [s], k=10))
        if len(set(eligible)) == 2:
            metrics["eligible_auc"] = float(roc_auc_score(eligible, s))
        for high, low, name in ((2, 0, "eligible_vs_not_relevant"),
                                (2, 1, "eligible_vs_ineligible"),
                                (1, 0, "ineligible_vs_not_relevant")):
            pairs = [(int(a == high), b) for a, b in rows if a in (high, low)]
            if len({a for a, _ in pairs}) == 2:
                metrics[name] = float(roc_auc_score(
                    [a for a, _ in pairs], [b for _, b in pairs]))
        per_topic[topic] = metrics
    keys = sorted({k for row in per_topic.values() for k in row})
    return {"topics": len(groups), "ndcg_gains": {"0": 0, "1": 1, "2": 3},
            "per_topic": per_topic,
            "macro": {k: statistics.mean(row[k] for row in per_topic.values() if k in row)
                      for k in keys},
            "n_topics": {k: sum(k in row for row in per_topic.values()) for k in keys}}
