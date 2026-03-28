#!/usr/bin/env python3
"""
Experiment 1: Error analysis by φ quartile.

For each of OG-OCL and Fixed Margin, loads the test set, encodes through
the saved checkpoint, bins samples by φ into quartiles, and computes
d(g,p) and triplet accuracy per quartile.

Usage:
    python scripts/phi_quartile_analysis.py
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

# ── Paths ──
TEST_PATH = "preprocess/data_splits_v6/test.jsonl"
STRATEGIES = {
    "OG-OCL": {
        "checkpoint": "results_ordinal_v6_phi_corrected/phase1_pretraining/best_checkpoint.pt",
        "config": "CDCL/config/phase1_ordinal_config.json",
    },
    "Fixed Margin": {
        "checkpoint": "results_ordinal_v6_fixed_margin/phase1_pretraining/best_checkpoint.pt",
        "config": "CDCL/config/phase1_ordinal_fixed_margin_config.json",
    },
}
OUTPUT_PATH = "results_phi_quartile_analysis.json"

# φ quartile boundaries
QUARTILE_BOUNDS = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
QUARTILE_NAMES = ["Q1 [0, 0.25)", "Q2 [0.25, 0.50)", "Q3 [0.50, 0.75)", "Q4 [0.75, 1.0]"]


# ── Model ──
class CareerAwareContrastiveModel(torch.nn.Module):
    def __init__(self, input_dim=768, projection_dim=128, dropout=0.1):
        super().__init__()
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, projection_dim),
        )

    def forward(self, x):
        projected = self.projection_head(x)
        return torch.nn.functional.normalize(projected, p=2, dim=-1)


def load_model(checkpoint_path, config_path, device):
    with open(config_path) as f:
        cfg = json.load(f)
    model = CareerAwareContrastiveModel(
        input_dim=768,
        projection_dim=cfg.get("projection_dim", 128),
        dropout=cfg.get("projection_dropout", 0.1),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def content_to_text(content, content_type):
    """Convert resume/job dict to text string."""
    if isinstance(content, str):
        return content
    parts = []
    if content_type == "resume":
        if content.get("title"):
            parts.append(f"Title: {content['title']}")
        if content.get("skills"):
            skills = content["skills"] if isinstance(content["skills"], list) else [content["skills"]]
            parts.append(f"Skills: {', '.join(str(s) for s in skills)}")
        if content.get("experience"):
            exp = content["experience"]
            if isinstance(exp, list):
                parts.append(f"Experience: {' '.join(str(e) for e in exp)}")
            else:
                parts.append(f"Experience: {exp}")
    else:
        if content.get("title"):
            parts.append(f"Title: {content['title']}")
        if content.get("required_skills"):
            skills = content["required_skills"] if isinstance(content["required_skills"], list) else [content["required_skills"]]
            parts.append(f"Required Skills: {', '.join(str(s) for s in skills)}")
        if content.get("description"):
            parts.append(f"Description: {content['description']}")
    return " | ".join(parts) if parts else str(content)


# ── Dataset ──
class OrdinalDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    label = item.get("metadata", {}).get("original_label", "unknown")
                    if label in ("good_fit", "potential_fit", "no_fit"):
                        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "resume": item["resume"],
            "job": item["job"],
            "original_label": item["metadata"]["original_label"],
            "phi": item["metadata"].get("phi"),
        }


def collate_fn(batch):
    return {
        "resume": [b["resume"] for b in batch],
        "job": [b["job"] for b in batch],
        "original_label": [b["original_label"] for b in batch],
        "phi": [b["phi"] for b in batch],
    }


def encode_test_set(model, text_encoder, dataset, device):
    """Return arrays: similarities, labels, phis."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    sims, labels, phis = [], [], []
    with torch.no_grad():
        for batch in loader:
            r_texts = [content_to_text(r, "resume") for r in batch["resume"]]
            j_texts = [content_to_text(j, "job") for j in batch["job"]]
            r_emb = text_encoder.encode(r_texts, convert_to_tensor=True).to(device)
            j_emb = text_encoder.encode(j_texts, convert_to_tensor=True).to(device)
            r_proj = model(r_emb)
            j_proj = model(j_emb)
            s = torch.sum(r_proj * j_proj, dim=1).cpu().numpy()
            sims.extend(s)
            labels.extend(batch["original_label"])
            phis.extend([p if p is not None else 0.5 for p in batch["phi"]])
    return np.array(sims), labels, np.array(phis)


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return float("nan")
    return (np.mean(g1) - np.mean(g2)) / pooled


def triplet_accuracy(sims, labels):
    """Fraction of (g, p, n) triples where sim(g) > sim(p) > sim(n)."""
    g = [s for s, l in zip(sims, labels) if l == "good_fit"]
    p = [s for s, l in zip(sims, labels) if l == "potential_fit"]
    n = [s for s, l in zip(sims, labels) if l == "no_fit"]
    if not g or not p or not n:
        return float("nan"), 0
    correct, total = 0, 0
    for sg in g:
        for sp in p:
            for sn in n:
                total += 1
                if sg > sp > sn:
                    correct += 1
    return correct / total if total > 0 else float("nan"), total


def analyze_quartiles(sims, labels, phis):
    """Compute d(g,p) and triplet accuracy per φ quartile."""
    results = {}
    for qname, (lo, hi) in zip(QUARTILE_NAMES, QUARTILE_BOUNDS):
        mask = (phis >= lo) & (phis < hi)
        q_sims = sims[mask]
        q_labels = [l for l, m in zip(labels, mask) if m]
        q_phis = phis[mask]

        g_sims = [s for s, l in zip(q_sims, q_labels) if l == "good_fit"]
        p_sims = [s for s, l in zip(q_sims, q_labels) if l == "potential_fit"]
        n_sims = [s for s, l in zip(q_sims, q_labels) if l == "no_fit"]

        d_gp = cohens_d(g_sims, p_sims) if g_sims and p_sims else float("nan")
        trip, trip_total = triplet_accuracy(q_sims, q_labels)

        results[qname] = {
            "n_total": int(mask.sum()),
            "n_good": len(g_sims),
            "n_potential": len(p_sims),
            "n_no": len(n_sims),
            "phi_range": f"[{lo}, {hi})",
            "phi_mean": float(np.mean(q_phis)) if mask.sum() > 0 else None,
            "d_g_p": round(d_gp, 4) if not np.isnan(d_gp) else None,
            "triplet_accuracy": round(trip, 4) if not np.isnan(trip) else None,
            "triplet_total": trip_total,
            "mean_sim_good": round(float(np.mean(g_sims)), 4) if g_sims else None,
            "mean_sim_potential": round(float(np.mean(p_sims)), 4) if p_sims else None,
            "mean_sim_no": round(float(np.mean(n_sims)), 4) if n_sims else None,
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = OrdinalDataset(TEST_PATH)
    print(f"Test samples: {len(dataset)}")

    # Load shared text encoder
    print("Loading text encoder: sentence-transformers/all-mpnet-base-v2")
    text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)

    all_results = {}

    for strategy_name, paths in STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"  {strategy_name}")
        print(f"{'='*60}")

        model = load_model(paths["checkpoint"], paths["config"], device)
        sims, labels, phis = encode_test_set(model, text_encoder, dataset, device)

        # Overall stats
        print(f"  φ distribution: min={phis.min():.3f}, mean={phis.mean():.3f}, "
              f"max={phis.max():.3f}, std={phis.std():.3f}")

        quartile_results = analyze_quartiles(sims, labels, phis)
        all_results[strategy_name] = quartile_results

        # Print table
        print(f"\n  {'Quartile':<20} {'N':>5} {'g/p/n':>12} {'d(g,p)':>8} {'Trip.Acc':>10} {'φ̄':>6}")
        print(f"  {'-'*65}")
        for qname in QUARTILE_NAMES:
            r = quartile_results[qname]
            gpn = f"{r['n_good']}/{r['n_potential']}/{r['n_no']}"
            d_str = f"{r['d_g_p']:.3f}" if r['d_g_p'] is not None else "N/A"
            t_str = f"{r['triplet_accuracy']:.3f}" if r['triplet_accuracy'] is not None else "N/A"
            phi_str = f"{r['phi_mean']:.3f}" if r['phi_mean'] is not None else "N/A"
            print(f"  {qname:<20} {r['n_total']:>5} {gpn:>12} {d_str:>8} {t_str:>10} {phi_str:>6}")

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON: OG-OCL vs Fixed Margin by φ quartile")
    print(f"{'='*70}")
    print(f"\n  {'Quartile':<20} {'d(g,p) FM':>10} {'d(g,p) OG':>10} {'Δd':>8} {'Trip FM':>9} {'Trip OG':>9} {'ΔTrip':>8}")
    print(f"  {'-'*78}")
    for qname in QUARTILE_NAMES:
        fm = all_results["Fixed Margin"][qname]
        og = all_results["OG-OCL"][qname]
        d_fm = fm["d_g_p"]
        d_og = og["d_g_p"]
        t_fm = fm["triplet_accuracy"]
        t_og = og["triplet_accuracy"]
        dd = (d_og - d_fm) if (d_og is not None and d_fm is not None) else None
        dt = (t_og - t_fm) if (t_og is not None and t_fm is not None) else None
        d_fm_s = f"{d_fm:.3f}" if d_fm is not None else "N/A"
        d_og_s = f"{d_og:.3f}" if d_og is not None else "N/A"
        dd_s = f"{dd:+.3f}" if dd is not None else "N/A"
        t_fm_s = f"{t_fm:.3f}" if t_fm is not None else "N/A"
        t_og_s = f"{t_og:.3f}" if t_og is not None else "N/A"
        dt_s = f"{dt:+.3f}" if dt is not None else "N/A"
        print(f"  {qname:<20} {d_fm_s:>10} {d_og_s:>10} {dd_s:>8} {t_fm_s:>9} {t_og_s:>9} {dt_s:>8}")

    # Save (convert numpy types for JSON)
    def to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    with open(OUTPUT_PATH, "w") as f:
        json.dump(to_native(all_results), f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
