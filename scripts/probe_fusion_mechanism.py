#!/usr/bin/env python3
"""Which mechanism can actually deliver the ontology's signal? Tested, not argued.

Ontology-guided NEGATIVE SELECTION is null everywhere: career (4 arms, 96 runs),
trials, and GO/PPI (+0.0011 on the hard contrast, n=5 seeds, once the
negative-diversity confound is controlled). Yet the ontology demonstrably holds
signal the encoder's text does not:

    trials  text 0.4938 -> MeSH 0.6031   increment +0.1223  [+0.041, +0.174]
    go_ppi  text 0.7790 -> GO   0.7735   increment +0.0173  [+0.007, +0.038]

The reason selection cannot deliver it is structural: it only decides WHICH pairs
appear in a batch. The model never receives an ontology value, at training or at
inference, so the signal can only arrive indirectly through which contrasts the
encoder happens to see.

The obvious alternative is to put the ontology INTO THE SCORE. This script tests
that on checkpoints that already exist, so no training is required and the answer
does not depend on a new training run behaving well:

    fused = (1 - w) * z(cosine similarity) + w * z(ontology similarity)

z(.) uses mean and standard deviation fitted on VALIDATION and applies those
statistics unchanged to TEST. Rank-AUC is invariant to a monotone transform of
either feature alone, but not to how the two features are weighted against each
other, so fitting the scale on test would be mildly transductive.

WHY THE WEIGHT IS CHOSEN ON VALIDATION
--------------------------------------
Sweeping w and reporting the best test AUC would be selection on the test set --
the same optimism this project already measured once, where picking a checkpoint
by validation loss and then scoring on that same validation split inflated two
career arms by +0.0122 and +0.0208 AUC and turned both from null into
"significant". So w is chosen on validation and the reported number is the test
AUC at that w. The full test sweep is printed too, but only so the shape of the
curve is visible; the headline is the validation-selected point.

WHAT TO EXPECT, AND WHY THE TWO DOMAINS DIFFER
----------------------------------------------
    domain  contrast  text    onto    both     who wins
    trials  hard      0.4938  0.6031  0.6029   ontology, decisively
    trials  easy      0.8028  0.7012  0.8215   text, decisively
    go_ppi  hard      0.7790  0.7735  0.7944   both (genuine complementarity)
    go_ppi  easy      0.8853  0.8865  0.9130   both

On go_ppi the two features are complementary WITHIN each contrast, so a single
fused score should beat either. On trials they are complementary ACROSS
contrasts: weighting toward the ontology should help the hard contrast and hurt
the easy one. A single global w must therefore trade them off, and the sweep makes
that trade-off explicit rather than hiding it inside one pooled number. If the
curve shows that, the implication is that trials needs a contrast-aware or
two-stage design, not a single blended score.

Usage
    .venv/bin/python3 scripts/probe_fusion_mechanism.py --domain go_ppi
    .venv/bin/python3 scripts/probe_fusion_mechanism.py --domain trials
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRADE_OF_LABEL = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}

DOMAINS: Dict[str, Dict[str, Any]] = {
    "go_ppi": {
        "config": "config/lc_go_ppi_ontneg_stoch.json",
        "val": "preprocess/go_ppi_splits/validation.jsonl",
        "test": "preprocess/go_ppi_splits/test.jsonl",
        "ontology": "go",
        "checkpoints": [
            "results/lc_single_factor/go_ppi_baseline_f100_s42/best_checkpoint.pt",
            "results/lc_single_factor/go_ppi_baseline_f100_s13/best_checkpoint.pt",
            "results/lc_single_factor/go_ppi_baseline_f100_s21/best_checkpoint.pt",
        ],
    },
    "trials": {
        "config": "config/lc_trials_ontneg_only.json",
        "val": "preprocess/trec_ct_splits/validation.jsonl",
        "test": "preprocess/trec_ct_splits/test.jsonl",
        "ontology": "mesh",
        # All six available trials checkpoints, not just the two `full` arms. The
        # three arms differ in how the BASE model was trained (full labels,
        # low-budget with ontology negatives, low-budget with random negatives), so
        # showing fusion helps across all of them is a stronger claim than showing
        # it on one training recipe: the gain is a property of adding the ontology
        # to the SCORE, not an interaction with a particular training setup.
        "checkpoints": [
            "results/label_budget/full_s42/best_checkpoint.pt",
            "results/label_budget/full_s13/best_checkpoint.pt",
            "results/label_budget/low_ontology_s42/best_checkpoint.pt",
            "results/label_budget/low_ontology_s13/best_checkpoint.pt",
            "results/label_budget/low_random_s42/best_checkpoint.pt",
            "results/label_budget/low_random_s13/best_checkpoint.pt",
        ],
    },
}


def rank_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    n = len(merged)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rsum = sum(r for r, (_v, lab) in zip(ranks, merged) if lab == 1)
    np_, nn = len(pos), len(neg)
    return (rsum - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def build_matcher(kind: str, config):
    if kind == "go":
        from go_ppi_domain.run_config import build_go_matcher
        return build_go_matcher(config)
    if kind == "mesh":
        from trials_domain.run_config import build_mesh_matcher
        return build_mesh_matcher(config)
    raise SystemExit(f"unknown ontology {kind!r}")


#: Cache of the EXPENSIVE, checkpoint-independent work, keyed by split path.
#: The frozen SentenceTransformer embeddings and the ontology similarities do not
#: depend on which checkpoint is being scored -- only the small projection head
#: does. Re-encoding per checkpoint made the trials run (6 checkpoints x 2 splits,
#: texts up to 11.6k chars) take hours doing the same work six times.
_SPLIT_CACHE: Dict[Any, Any] = {}


def _prepare_split(split_path: Path, config, matcher, max_records: int, seed: int):
    """Encode texts and score the ontology ONCE per split, independent of checkpoint."""
    key = (str(split_path), max_records, seed, id(matcher))
    hit = _SPLIT_CACHE.get(key)
    if hit is not None:
        return hit

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from run_phase1_embedding_evaluation import content_to_text

    records = []
    with open(split_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    if max_records and len(records) > max_records:
        import random
        records = random.Random(seed).sample(records, max_records)

    texts_a, texts_b, onto, grades = [], [], [], []
    for r in records:
        a, b = r.get("resume"), r.get("job")
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        g = GRADE_OF_LABEL.get((r.get("metadata") or {}).get("original_label"))
        if g is None:
            continue
        ta, tb = content_to_text(a, "resume"), content_to_text(b, "job")
        if not ta.strip() or not tb.strip():
            continue
        ua, ub = a.get("skill_uris") or [], b.get("skill_uris") or []
        try:
            s = float(matcher.ontology_set_similarity(ua, ub)) if (ua and ub) else 0.0
        except Exception:
            s = 0.0
        texts_a.append(ta)
        texts_b.append(tb)
        onto.append(s)
        grades.append(g)

    enc = SentenceTransformer(config.text_encoder_model)
    uniq = sorted(set(texts_a) | set(texts_b))
    print(f"      encoding {len(uniq)} unique texts for {split_path.name} "
          f"(once, reused across checkpoints) ...", flush=True)
    with torch.no_grad():
        base = enc.encode(uniq, batch_size=64, convert_to_numpy=True,
                          show_progress_bar=False)
    idx = {t: i for i, t in enumerate(uniq)}
    ia = np.array([idx[t] for t in texts_a])
    ib = np.array([idx[t] for t in texts_b])
    out = (base, ia, ib, np.asarray(onto, dtype=float),
           np.asarray(grades, dtype=int), enc.get_sentence_embedding_dimension())
    _SPLIT_CACHE[key] = out
    return out


def score_split(split_path: Path, checkpoint: Path, config, matcher,
                max_records: int = 0, seed: int = 42):
    """Return (cos_sim, onto_sim, grades) for one split under one checkpoint.

    Only the projection head is applied per checkpoint; the frozen text embeddings
    and the ontology similarities come from the shared per-split cache.
    """
    import numpy as np
    import torch
    from run_phase1_embedding_evaluation import CareerAwareContrastiveModel

    base, ia, ib, onto, grades, dim = _prepare_split(
        split_path, config, matcher, max_records, seed)

    device = torch.device("cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    use_sf = any("structured_encoder" in k for k in state.keys())

    model = CareerAwareContrastiveModel(
        input_dim=dim,
        projection_dim=getattr(config, "projection_dim", 128),
        dropout=getattr(config, "projection_dropout", 0.1),
        use_structured_features=use_sf,
        structured_feature_dim=ckpt.get("config", {}).get("structured_feature_dim", 32),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        proj = model(torch.tensor(base, dtype=torch.float32)).cpu().numpy()
    # the model L2-normalises its output, so a dot product is the cosine
    cos = np.sum(proj[ia] * proj[ib], axis=1)
    return cos, onto, grades


def z_stats(x):
    """Mean and sd of a feature, to be FITTED ON VALIDATION and reused on test.

    Fitting these on the split being scored is a real (if small) leak. Rank-AUC is
    invariant to a monotone transform of one feature alone, so normalising a single
    feature in isolation would be harmless -- but the fused score weights the two
    features against each other, and that weighting depends on each feature's sd.
    Taking the sd from the test split therefore lets test statistics influence the
    combination, which is transductive: it is not a result you could reproduce on a
    query you had never seen.
    """
    import numpy as np
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return mu, (sd if sd > 0 else 1.0)


def z_apply(x, stats):
    mu, sd = stats
    return (x - mu) / sd


def contrast_aucs(score, grades) -> Dict[str, float]:
    out = {}
    for name, hi, lo in (("hard", 2, 1), ("easy", 2, 0)):
        m = (grades == hi) | (grades == lo)
        if m.sum() < 30:
            out[name] = float("nan")
            continue
        y = grades[m] == hi
        s = score[m]
        out[name] = rank_auc(s[y].tolist(), s[~y].tolist())
    # pooled: grade 2 vs everything else, the shape the headline AUC-ROC uses
    y = grades == 2
    out["pooled"] = rank_auc(score[y].tolist(), score[~y].tolist())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="go_ppi", choices=sorted(DOMAINS))
    ap.add_argument("--max-records", type=int, default=6000)
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--select-on", default="hard", choices=["hard", "easy", "pooled"],
                    help="which contrast the validation-chosen weight optimises")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import numpy as np
    from contrastive_learning.data_structures import TrainingConfig

    spec = DOMAINS[args.domain]
    config = TrainingConfig.from_json(str(ROOT / spec["config"]))
    matcher = build_matcher(spec["ontology"], config)

    ckpts = [ROOT / c for c in spec["checkpoints"]]
    ckpts = [c for c in ckpts if c.exists()]
    if not ckpts:
        sys.exit(f"no checkpoints found for {args.domain}")

    print("=" * 100)
    print(f"{args.domain}: can SCORE FUSION deliver what negative selection could not?")
    print("=" * 100)
    print(f"fused = (1-w)*z(cosine) + w*z(ontology similarity)")
    print(f"weight chosen on VALIDATION (optimising the {args.select_on} contrast), "
          f"reported on TEST")
    print(f"checkpoints: {len(ckpts)} (all configured training arms)")
    print()

    results: Dict[str, Any] = {
        "domain": args.domain,
        "select_on": args.select_on,
        "validation_split": spec["val"],
        "test_split": spec["test"],
        "max_records_per_split": args.max_records,
        "sampling_seed": 42,
        "weights": args.weights,
        "normalization": "fit mean/sd on validation; apply unchanged to test",
        "per_checkpoint": [],
    }
    test_at_selected, test_at_zero = [], []

    for ck in ckpts:
        tag = ck.parent.name
        print(f"--- {tag}")
        cv, ov, gv = score_split(ROOT / spec["val"], ck, config, matcher,
                                 args.max_records)
        ct, ot, gt = score_split(ROOT / spec["test"], ck, config, matcher,
                                 args.max_records)
        # Normalisation statistics come from VALIDATION and are applied unchanged to
        # test, so no test-split statistic touches the feature weighting.
        cs, os_ = z_stats(cv), z_stats(ov)
        zcv, zov = z_apply(cv, cs), z_apply(ov, os_)
        zct, zot = z_apply(ct, cs), z_apply(ot, os_)

        val_curve, test_curve = {}, {}
        for w in args.weights:
            val_curve[w] = contrast_aucs((1 - w) * zcv + w * zov, gv)
            test_curve[w] = contrast_aucs((1 - w) * zct + w * zot, gt)

        best_w = max(args.weights,
                     key=lambda w: (val_curve[w][args.select_on]
                                    if not math.isnan(val_curve[w][args.select_on])
                                    else -1))
        base = test_curve[0.0]
        sel = test_curve[best_w]
        print(f"    validation picked w={best_w:.1f}")
        print(f"    TEST  w=0.0 (text only)   hard {base['hard']:.4f}  "
              f"easy {base['easy']:.4f}  pooled {base['pooled']:.4f}")
        print(f"    TEST  w={best_w:.1f} (fused)      hard {sel['hard']:.4f}  "
              f"easy {sel['easy']:.4f}  pooled {sel['pooled']:.4f}")
        print(f"    delta                     hard {sel['hard']-base['hard']:+.4f}  "
              f"easy {sel['easy']-base['easy']:+.4f}  "
              f"pooled {sel['pooled']-base['pooled']:+.4f}")
        test_at_selected.append(sel)
        test_at_zero.append(base)
        results["per_checkpoint"].append({
            "checkpoint": tag, "selected_w": best_w,
            "val_curve": {str(k): v for k, v in val_curve.items()},
            "test_curve": {str(k): v for k, v in test_curve.items()},
        })
        print()

    # ---- pooled across checkpoints
    print("=" * 100)
    print("POOLED ACROSS CHECKPOINTS (validation-selected weight, test AUC)")
    print("=" * 100)
    for cname in ("hard", "easy", "pooled"):
        b = [d[cname] for d in test_at_zero if not math.isnan(d[cname])]
        s = [d[cname] for d in test_at_selected if not math.isnan(d[cname])]
        if not b or not s:
            continue
        mb, ms = sum(b) / len(b), sum(s) / len(s)
        print(f"  {cname:<8} text-only {mb:.4f} -> fused {ms:.4f}   {ms-mb:+.4f}  "
              f"({sum(1 for x, y in zip(s, b) if y < x)}/{len(s)} checkpoints up)")
        results.setdefault("pooled", {})[cname] = {
            "text_only": mb, "fused": ms, "delta": ms - mb, "n": len(s)}

    # ---- full test sweep, averaged, so the trade-off is visible
    print()
    print("  full TEST sweep, averaged over checkpoints (shape only -- the headline")
    print("  above is the validation-selected point, not the best cell here):")
    print(f"    {'w':>5} {'hard':>9} {'easy':>9} {'pooled':>9}")
    for w in args.weights:
        rows = [pc["test_curve"][str(w)] for pc in results["per_checkpoint"]]
        def m(k):
            v = [r[k] for r in rows if not math.isnan(r[k])]
            return sum(v) / len(v) if v else float("nan")
        print(f"    {w:>5.1f} {m('hard'):>9.4f} {m('easy'):>9.4f} {m('pooled'):>9.4f}")

    out = args.out or (ROOT / "results" / "ontology_ceiling" /
                       f"fusion_probe_{args.domain}.json")
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
