#!/usr/bin/env python3
"""Analyze which test samples flip between vanilla and ontology models at 50%.

Computes per-sample cosine similarity for both models, then identifies samples
where vanilla ranks correctly but ontology doesn't (and vice versa).
"""
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict


def load_model(checkpoint_path, text_encoder_name="sentence-transformers/all-mpnet-base-v2"):
    """Load text encoder + projection head from checkpoint."""
    text_encoder = SentenceTransformer(text_encoder_name, device="cpu")
    text_encoder.eval()
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Infer dims from state dict keys
    first_key = [k for k in state_dict if "weight" in k][0]
    prefix = first_key.rsplit(".", 2)[0]
    input_dim = state_dict[f"{prefix}.0.weight"].shape[1]
    hidden_dim = state_dict[f"{prefix}.0.weight"].shape[0]
    proj_dim = state_dict[f"{prefix}.3.weight"].shape[0] if f"{prefix}.3.weight" in state_dict else hidden_dim
    
    # Build model inline to match checkpoint keys
    class ProjectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection_head = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, proj_dim),
            )
        def forward(self, x):
            x = self.projection_head(x)
            return torch.nn.functional.normalize(x, p=2, dim=-1)
    
    model = ProjectionModel()
    model.load_state_dict(state_dict)
    model.eval()
    
    return text_encoder, model


def encode_text(text_encoder, model, texts, batch_size=64):
    """Encode texts through frozen encoder + projection head."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            text_embs = text_encoder.encode(batch, convert_to_tensor=True, device="cpu", show_progress_bar=False)
            proj_embs = model(text_embs)
        all_embs.append(proj_embs.cpu())
    return torch.cat(all_embs, dim=0)


def content_to_text(content, content_type):
    """Convert resume/job dict to text string."""
    if content_type == "resume":
        parts = []
        if content.get("name"): parts.append(str(content["name"]))
        if content.get("summary"): parts.append(str(content["summary"]))
        if content.get("experience"):
            for exp in content["experience"][:3]:
                if isinstance(exp, dict):
                    parts.append(f"{exp.get('title','')} at {exp.get('company','')}: {exp.get('description','')}")
                else:
                    parts.append(str(exp))
        if content.get("skills"):
            skills = content["skills"]
            if isinstance(skills, list): parts.append("Skills: " + ", ".join(str(s) for s in skills[:20]))
        return " ".join(str(p) for p in parts)[:1000]
    else:
        parts = []
        if content.get("title"): parts.append(str(content["title"]))
        if content.get("description"):
            desc = content["description"]
            if isinstance(desc, dict):
                parts.append(" ".join(str(v) for v in desc.values()))
            elif isinstance(desc, list):
                parts.append(" ".join(str(d) for d in desc))
            else:
                parts.append(str(desc))
        if content.get("requirements"):
            reqs = content["requirements"]
            if isinstance(reqs, list): parts.append("Requirements: " + ", ".join(str(r) for r in reqs[:10]))
        return " ".join(str(p) for p in parts)[:1000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="preprocess/data_splits_v6/test.jsonl")
    parser.add_argument("--vanilla-ckpt", required=True)
    parser.add_argument("--ontology-ckpt", required=True)
    parser.add_argument("--output", default="analysis_sample_flips.json")
    args = parser.parse_args()

    # Load test data
    samples = []
    with open(args.test_data) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} test samples")

    # Prepare texts
    resume_texts = [content_to_text(s["resume"], "resume") for s in samples]
    job_texts = [content_to_text(s["job"], "job") for s in samples]
    labels = [s.get("metadata", {}).get("original_label", "unknown") for s in samples]
    ont_sims = [s.get("metadata", {}).get("ontology_similarity") for s in samples]
    ot_dists = [s.get("metadata", {}).get("ot_distance") for s in samples]
    tiers = [s.get("metadata", {}).get("quality_tier", "F") for s in samples]

    # Load and encode with vanilla model
    print("Loading vanilla model...")
    v_enc, v_model = load_model(args.vanilla_ckpt)
    print("Encoding with vanilla...")
    v_resume_embs = encode_text(v_enc, v_model, resume_texts)
    v_job_embs = encode_text(v_enc, v_model, job_texts)
    v_sims = torch.nn.functional.cosine_similarity(v_resume_embs, v_job_embs).numpy()

    # Load and encode with ontology model
    print("Loading ontology model...")
    o_enc, o_model = load_model(args.ontology_ckpt)
    print("Encoding with ontology...")
    o_resume_embs = encode_text(o_enc, o_model, resume_texts)
    o_job_embs = encode_text(o_enc, o_model, job_texts)
    o_sims = torch.nn.functional.cosine_similarity(o_resume_embs, o_job_embs).numpy()

    # Analyze per-label similarity distributions
    label_order = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}
    
    print("\n=== Per-label similarity distributions ===")
    for label in ["good_fit", "potential_fit", "no_fit"]:
        idx = [i for i, l in enumerate(labels) if l == label]
        vs = v_sims[idx]
        os_ = o_sims[idx]
        print(f"{label} (n={len(idx)}):")
        print(f"  Vanilla:  mean={vs.mean():.4f}, std={vs.std():.4f}")
        print(f"  Ontology: mean={os_.mean():.4f}, std={os_.std():.4f}")
        print(f"  Diff:     {(os_.mean() - vs.mean()):.4f}")

    # Find samples where ranking flips
    # For each good_fit sample, check if it scores above each no_fit sample
    good_idx = [i for i, l in enumerate(labels) if l == "good_fit"]
    nofit_idx = [i for i, l in enumerate(labels) if l == "no_fit"]
    pot_idx = [i for i, l in enumerate(labels) if l == "potential_fit"]

    # Pairwise: good_fit vs no_fit
    v_correct_gn = 0
    o_correct_gn = 0
    v_only_gn = 0  # vanilla correct, ontology wrong
    o_only_gn = 0  # ontology correct, vanilla wrong
    total_gn = 0

    flip_samples_v_wins = []  # vanilla correct, ontology wrong
    flip_samples_o_wins = []  # ontology correct, vanilla wrong

    for gi in good_idx:
        for ni in nofit_idx:
            total_gn += 1
            v_ok = v_sims[gi] > v_sims[ni]
            o_ok = o_sims[gi] > o_sims[ni]
            if v_ok: v_correct_gn += 1
            if o_ok: o_correct_gn += 1
            if v_ok and not o_ok:
                v_only_gn += 1
                if len(flip_samples_v_wins) < 20:
                    flip_samples_v_wins.append({
                        "good_idx": int(gi), "nofit_idx": int(ni),
                        "v_good_sim": float(v_sims[gi]), "v_nofit_sim": float(v_sims[ni]),
                        "o_good_sim": float(o_sims[gi]), "o_nofit_sim": float(o_sims[ni]),
                        "good_ont_sim": ont_sims[gi], "nofit_ont_sim": ont_sims[ni],
                        "good_tier": tiers[gi], "nofit_tier": tiers[ni],
                    })
            if o_ok and not v_ok:
                o_only_gn += 1
                if len(flip_samples_o_wins) < 20:
                    flip_samples_o_wins.append({
                        "good_idx": int(gi), "nofit_idx": int(ni),
                        "v_good_sim": float(v_sims[gi]), "v_nofit_sim": float(v_sims[ni]),
                        "o_good_sim": float(o_sims[gi]), "o_nofit_sim": float(o_sims[ni]),
                        "good_ont_sim": ont_sims[gi], "nofit_ont_sim": ont_sims[ni],
                        "good_tier": tiers[gi], "nofit_tier": tiers[ni],
                    })

    print(f"\n=== Good vs No_fit pairwise ({total_gn} pairs) ===")
    print(f"  Vanilla correct: {v_correct_gn} ({v_correct_gn/total_gn*100:.1f}%)")
    print(f"  Ontology correct: {o_correct_gn} ({o_correct_gn/total_gn*100:.1f}%)")
    print(f"  Vanilla-only wins: {v_only_gn} ({v_only_gn/total_gn*100:.1f}%)")
    print(f"  Ontology-only wins: {o_only_gn} ({o_only_gn/total_gn*100:.1f}%)")

    # Analyze flip characteristics
    print(f"\n=== Vanilla-wins flip characteristics (n={len(flip_samples_v_wins)}) ===")
    if flip_samples_v_wins:
        good_onts = [f["good_ont_sim"] for f in flip_samples_v_wins if f["good_ont_sim"] is not None]
        nofit_onts = [f["nofit_ont_sim"] for f in flip_samples_v_wins if f["nofit_ont_sim"] is not None]
        if good_onts: print(f"  Good sample ont_sim: mean={np.mean(good_onts):.4f}")
        if nofit_onts: print(f"  Nofit sample ont_sim: mean={np.mean(nofit_onts):.4f}")
        good_tiers = [f["good_tier"] for f in flip_samples_v_wins]
        print(f"  Good tiers: {dict(defaultdict(int, **{t: good_tiers.count(t) for t in set(good_tiers)}))}")

    # Per-sample analysis: which samples does ontology hurt most?
    print(f"\n=== Samples where ontology hurts most ===")
    diffs = o_sims - v_sims  # positive = ontology gives higher sim
    
    # For good_fit: ontology should give HIGH sim. If diff is very negative, ontology hurts.
    good_diffs = [(i, diffs[i]) for i in good_idx]
    good_diffs.sort(key=lambda x: x[1])  # most negative first
    print("Good_fit samples most hurt by ontology (sim dropped):")
    for idx, diff in good_diffs[:10]:
        print(f"  idx={idx}: v_sim={v_sims[idx]:.4f}, o_sim={o_sims[idx]:.4f}, diff={diff:.4f}, "
              f"ont_sim={ont_sims[idx]}, tier={tiers[idx]}")

    # For no_fit: ontology should give LOW sim. If diff is very positive, ontology hurts.
    nofit_diffs = [(i, diffs[i]) for i in nofit_idx]
    nofit_diffs.sort(key=lambda x: -x[1])  # most positive first
    print("\nNo_fit samples most hurt by ontology (sim increased when it shouldn't):")
    for idx, diff in nofit_diffs[:10]:
        print(f"  idx={idx}: v_sim={v_sims[idx]:.4f}, o_sim={o_sims[idx]:.4f}, diff={diff:.4f}, "
              f"ont_sim={ont_sims[idx]}, tier={tiers[idx]}")

    # Save results
    results = {
        "vanilla_good_mean": float(v_sims[good_idx].mean()),
        "ontology_good_mean": float(o_sims[good_idx].mean()),
        "vanilla_nofit_mean": float(v_sims[nofit_idx].mean()),
        "ontology_nofit_mean": float(o_sims[nofit_idx].mean()),
        "vanilla_pairwise_accuracy": v_correct_gn / total_gn,
        "ontology_pairwise_accuracy": o_correct_gn / total_gn,
        "vanilla_only_wins": v_only_gn,
        "ontology_only_wins": o_only_gn,
        "flip_samples_v_wins": flip_samples_v_wins[:10],
        "flip_samples_o_wins": flip_samples_o_wins[:10],
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
