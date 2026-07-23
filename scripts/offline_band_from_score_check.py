#!/usr/bin/env python3
"""Offline check: does bucketing the predicted priority_score beat the collapsed
softmax band head?

Loads a completed experiment's saved Stage 1 projection + Stage 2 heads, reloads
the frozen mpnet encoder, and runs the REAL ``CVEStage2Trainer.predict_records``
over the test split. Then it compares three band predictors against ground truth:

  1. argmax        -- the trained multiclass band head (what the run reported)
  2. score_bucket  -- bucket the model's PREDICTED priority_score at 45/70/85
  3. oracle_bucket -- bucket the GROUND-TRUTH score (sanity upper bound = 1.0)

No retraining. On a GPU box this is a few minutes; on CPU it's slower.

Layout assumed under --results-root:
    <root>/<EXP>/stage1/best_checkpoint.pt
    <root>/<EXP>/stage2/stage2_best_checkpoint.pt
    <root>/<EXP>/stage2_config.resolved.json
    <root>/_shared/split_<split>_seed<seed>/test.jsonl

Examples
--------
GPU box (native run artifacts)::

    .venv/bin/python scripts/offline_band_from_score_check.py \
        --results-root cve_domain/runs/experiments --experiment E1

Local (files pulled from the HF results repo into hf_cve_results/cve)::

    .venv/bin/python scripts/offline_band_from_score_check.py \
        --results-root hf_cve_results/cve --experiment E1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Band thresholds recovered from the data: priority_band is a deterministic
# bucketing of priority_score with cut points at 45 / 70 / 85 (zero overlap).
BANDS = ["watch", "medium", "high", "critical"]  # ascending severity


def bucket_score(score: float) -> str:
    if score < 45.0:
        return "watch"
    if score < 70.0:
        return "medium"
    if score < 85.0:
        return "high"
    return "critical"


def macro_f1(y_true, y_pred, labels):
    f1s, per_class = [], {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": sum(1 for t in y_true if t == c)}
        f1s.append(f1)
    return sum(f1s) / len(f1s), per_class


def accuracy(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def adjacent_accuracy(y_true, y_pred, order):
    idx = {b: i for i, b in enumerate(order)}
    return sum(1 for t, p in zip(y_true, y_pred) if abs(idx[t] - idx[p]) <= 1) / len(y_true)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="cve_domain/runs/experiments",
                    help="Root holding <EXP>/ and _shared/ (default: the GPU run layout)")
    ap.add_argument("--experiment", default="E1", help="Experiment id, e.g. E1")
    ap.add_argument("--split", default="stratified", help="Split strategy name")
    ap.add_argument("--seed", type=int, default=42, help="Split seed")
    ap.add_argument("--device", default=None, help="Force device (cpu/cuda/mps); auto if unset")
    ap.add_argument("--output", default=None, help="Optional path to write a JSON summary")
    args = ap.parse_args()

    root = Path(args.results_root)
    exp = root / args.experiment
    stage1_ckpt = exp / "stage1" / "best_checkpoint.pt"
    stage2_ckpt = exp / "stage2" / "stage2_best_checkpoint.pt"
    stage2_cfg = exp / "stage2_config.resolved.json"
    test = root / "_shared" / f"split_{args.split}_seed{args.seed}" / "test.jsonl"

    for p in (stage1_ckpt, stage2_ckpt, stage2_cfg, test):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    import torch
    from cve_domain.run_config import CVERunConfig
    from cve_domain.stage2 import CVEStage2Trainer

    cfg = json.loads(stage2_cfg.read_text())
    cfg["pretrained_model_path"] = str(stage1_ckpt)
    if args.device:
        cfg["text_encoder_device"] = args.device
    elif torch.cuda.is_available():
        cfg["text_encoder_device"] = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        cfg["text_encoder_device"] = "mps"
    else:
        cfg["text_encoder_device"] = "cpu"
    print(f"experiment: {args.experiment}  device: {cfg['text_encoder_device']}", flush=True)

    config = CVERunConfig.from_dict(cfg)
    trainer = CVEStage2Trainer(
        config=config,
        output_dir=str(exp / "_offline_tmp"),
        stage1_checkpoint_path=str(stage1_ckpt),
        base_embeddings=False,
    )
    print("loading encoder + projection + heads from checkpoints ...", flush=True)
    trainer.load_heads_from_checkpoint(stage2_ckpt)

    records = [json.loads(l) for l in test.read_text().splitlines() if l.strip()]
    print(f"test records: {len(records)}  (encoding through mpnet) ...", flush=True)

    predictions = trainer.predict_records(records)

    y_true, y_argmax, y_scorebucket, y_oracle = [], [], [], []
    missing = 0
    for rec in records:
        cve = str(rec.get("cve", "")).strip()
        labels = rec.get("cve_labels") or {}
        gt_band = labels.get("priority_band")
        gt_score = labels.get("priority_score")
        pred = predictions.get(cve)
        if not pred or gt_band is None or gt_score is None or "ranking_score" not in pred:
            missing += 1
            continue
        y_true.append(gt_band)
        y_argmax.append(pred.get("priority_band", "watch"))
        y_scorebucket.append(bucket_score(float(pred["ranking_score"])))
        y_oracle.append(bucket_score(float(gt_score)))

    print(f"evaluated: {len(y_true)}  (missing/skipped: {missing})", flush=True)
    print(f"ground-truth band counts: {dict(Counter(y_true))}\n", flush=True)

    summary = {"experiment": args.experiment, "evaluated": len(y_true),
               "gt_band_counts": dict(Counter(y_true)), "variants": {}}
    for name, y_pred in [("argmax_reported_head", y_argmax),
                         ("score_bucket_predicted", y_scorebucket),
                         ("oracle_bucket_gt_score", y_oracle)]:
        mf1, per = macro_f1(y_true, y_pred, BANDS)
        acc = accuracy(y_true, y_pred)
        adj = adjacent_accuracy(y_true, y_pred, BANDS)
        summary["variants"][name] = {"macro_f1": mf1, "accuracy": acc,
                                     "adjacent_accuracy": adj, "per_class": per,
                                     "pred_counts": dict(Counter(y_pred))}
        print(f"=== {name} ===")
        print(f"  macro_f1={mf1:.4f}  accuracy={acc:.4f}  adjacent_acc={adj:.4f}")
        for c in BANDS:
            pc = per[c]
            print(f"    {c:<9} P={pc['precision']:.3f} R={pc['recall']:.3f} "
                  f"F1={pc['f1']:.3f} (support={pc['support']})")
        print(f"  predicted band counts: {dict(Counter(y_pred))}\n")

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"wrote summary -> {args.output}")


if __name__ == "__main__":
    main()
