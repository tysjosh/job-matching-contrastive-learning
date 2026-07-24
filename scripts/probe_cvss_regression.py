#!/usr/bin/env python3
"""Decisive probe: can a STANDALONE regressor predict CVSS from the description?

Isolates "is the signal there?" from "is the Stage-2 multi-head trainer broken?".

The Stage-2 CVSS regression head collapsed to a low near-constant (MSE ~0.20 vs
~0.03 for a mean-predictor) across all experiments including the no-Stage-1 E0.
This probe trains ONLY a priority_score regressor (MSE, no other heads, no focal,
no summed-loss checkpoint) on the frozen base sentence-transformer embeddings of
the same CVSS split, and compares against the trivial mean-predictor baseline.

Interpretation:
  * probe MSE << mean-baseline MSE and Spearman clearly > 0
        -> the signal IS there; the Stage-2 multi-head training is the bug
           (best-checkpoint-by-summed-loss / head interaction). FIXABLE.
  * probe MSE ~= mean-baseline and Spearman ~ 0
        -> frozen embeddings of the description carry little CVSS signal.
           GENUINE negative result.

Runs on the box (needs the encoder + the CVSS shared split). Subsamples for speed.

    python scripts/probe_cvss_regression.py \
        --split-dir cve_domain/runs/experiments_cvss/_shared/split_stratified_seed42 \
        --train-n 20000 --test-n 5000 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

SCALE = 100.0  # priority_score stored as cvss*10 in [0,100]; normalize to [0,1].


def _read_jsonl(path: Path, limit: int, seed: int = 42) -> List[dict]:
    """Read all rows, then take a RANDOM sample of ``limit`` (representative).

    Head-of-file slicing is unsafe here: the split files are stratified/ordered,
    so the first N rows can be a skewed, low-variance slice. Random sampling with
    a fixed seed keeps the probe representative and reproducible.
    """
    import random
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit and len(rows) > limit:
        rng = random.Random(seed)
        rows = rng.sample(rows, limit)
    return rows


def _xy(records) -> Tuple[List[str], np.ndarray, List[str]]:
    texts, ys, bands = [], [], []
    for r in records:
        labs = r.get("cve_labels", {})
        score = labs.get("priority_score")
        if score is None:
            continue
        texts.append(r["encoder_view"])
        ys.append(float(score) / SCALE)
        bands.append(str(labs.get("priority_band", "")))
    return texts, np.asarray(ys, dtype=np.float32), bands


def _spearman(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="cve_domain/runs/experiments_cvss/_shared/split_stratified_seed42")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    sd = Path(args.split_dir)
    tr = _read_jsonl(sd / "train.jsonl", args.train_n)
    te = _read_jsonl(sd / "test.jsonl", args.test_n)
    tr_txt, tr_y, _ = _xy(tr)
    te_txt, te_y, te_band = _xy(te)
    print(f"train={len(tr_txt)} test={len(te_txt)}  target mean(norm)={tr_y.mean():.3f} std={tr_y.std():.3f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(args.model, device=dev)
    print("encoding train/test (frozen base embeddings)...")
    Xtr = enc.encode(tr_txt, batch_size=128, convert_to_numpy=True, show_progress_bar=False)
    Xte = enc.encode(te_txt, batch_size=128, convert_to_numpy=True, show_progress_bar=False)

    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(tr_y, device=dev)
    Xte_t = torch.tensor(Xte, device=dev)

    reg = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(reg.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    reg.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        pred = reg(Xtr_t).squeeze(-1)
        loss = lossf(pred, ytr_t)
        loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:>3} train_mse={loss.item():.4f}")

    reg.eval()
    with torch.no_grad():
        pte = reg(Xte_t).squeeze(-1).cpu().numpy()

    probe_mse = float(np.mean((pte - te_y) ** 2))
    mean_pred = float(tr_y.mean())
    mean_mse = float(np.mean((mean_pred - te_y) ** 2))
    rho = _spearman(pte, te_y)

    print("\n================ RESULT ================")
    print(f"  mean-predictor test MSE : {mean_mse:.4f}   (predict constant {mean_pred:.3f})")
    print(f"  probe        test MSE : {probe_mse:.4f}")
    print(f"  Spearman(probe, true) : {rho:.4f}")
    print(f"  predicted (norm) mean={pte.mean():.3f} std={pte.std():.3f} min={pte.min():.3f} max={pte.max():.3f}")
    if probe_mse < 0.8 * mean_mse and rho > 0.15:
        print("  VERDICT: signal IS present -> Stage-2 multi-head training is the bug (fixable).")
    elif probe_mse >= 0.95 * mean_mse and abs(rho) < 0.1:
        print("  VERDICT: no usable CVSS signal in frozen description embeddings (genuine negative).")
    else:
        print("  VERDICT: weak/partial signal -> borderline; inspect std and Spearman.")


if __name__ == "__main__":
    main()
