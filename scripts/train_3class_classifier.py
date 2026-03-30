#!/usr/bin/env python3
"""
Phase 2: Three-class classifier on top of frozen phase 1 contrastive embeddings.

Loads a phase 1 checkpoint, encodes all samples, then trains a small MLP:
  [resume_emb ‖ job_emb] (256-d) → 128 → ReLU → Dropout → 3 (softmax)

Labels: good_fit=2, potential_fit=1, no_fit=0

Usage:
    python3 scripts/train_3class_classifier.py \
        --checkpoint results_ordinal_v6_enhanced/phase1_pretraining/best_checkpoint.pt \
        --train preprocess/data_splits_v6/train.jsonl \
        --val preprocess/data_splits_v6/validation.jsonl \
        --test preprocess/data_splits_v6/test.jsonl \
        --output-dir results_3class_enhanced
"""
import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}
LABEL_NAMES = ["no_fit", "potential_fit", "good_fit"]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, projection_dim=128, dropout=0.3):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, projection_dim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.projection_head(x), p=2, dim=-1)


class ThreeClassHead(nn.Module):
    def __init__(self, input_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.classifier(x)


def content_to_text(content, content_type):
    if isinstance(content, str):
        return content
    parts = []
    if content_type == "resume":
        if content.get("title"): parts.append(f"Title: {content['title']}")
        if content.get("role"): parts.append(f"Role: {content['role']}")
        if content.get("skills"):
            skills = content["skills"] if isinstance(content["skills"], list) else [content["skills"]]
            parts.append(f"Skills: {', '.join(str(s) for s in skills)}")
        if content.get("experience"):
            exp = content["experience"]
            if isinstance(exp, list): parts.append(f"Experience: {' '.join(str(e) for e in exp)}")
            else: parts.append(f"Experience: {exp}")
    else:
        if content.get("title"): parts.append(f"Title: {content['title']}")
        if content.get("required_skills") or content.get("skills"):
            skills = content.get("required_skills") or content.get("skills")
            if isinstance(skills, list): parts.append(f"Skills: {', '.join(str(s) for s in skills)}")
        if content.get("description"): parts.append(f"Description: {content['description']}")
    return " | ".join(parts) if parts else str(content)


def load_and_encode(jsonl_path, text_encoder, proj_head, device, batch_size=64):
    """Load JSONL, encode through text encoder + projection head, return (features, labels)."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                label_str = d.get("metadata", {}).get("original_label")
                if label_str in LABEL_MAP:
                    samples.append((d["resume"], d["job"], LABEL_MAP[label_str]))

    logger.info(f"Loaded {len(samples)} samples from {jsonl_path}")

    all_features, all_labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        r_texts = [content_to_text(s[0], "resume") for s in batch]
        j_texts = [content_to_text(s[1], "job") for s in batch]
        labels = [s[2] for s in batch]

        with torch.no_grad():
            r_emb = text_encoder.encode(r_texts, convert_to_tensor=True).to(device)
            j_emb = text_encoder.encode(j_texts, convert_to_tensor=True).to(device)
            r_proj = proj_head(r_emb)
            j_proj = proj_head(j_emb)
            features = torch.cat([r_proj, j_proj], dim=-1).cpu()

        all_features.append(features)
        all_labels.extend(labels)

    return torch.cat(all_features, dim=0), torch.tensor(all_labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", default="results_3class")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load text encoder
    logger.info("Loading text encoder...")
    text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)

    # Load phase 1 projection head
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    proj_head = ProjectionHead(input_dim=768, projection_dim=128, dropout=0.3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    proj_head.load_state_dict(state, strict=False)
    proj_head.eval()
    for p in proj_head.parameters():
        p.requires_grad = False

    # Encode all splits
    logger.info("Encoding training set...")
    X_train, y_train = load_and_encode(args.train, text_encoder, proj_head, device)
    logger.info("Encoding validation set...")
    X_val, y_val = load_and_encode(args.val, text_encoder, proj_head, device)
    logger.info("Encoding test set...")
    X_test, y_test = load_and_encode(args.test, text_encoder, proj_head, device)

    # Class weights for imbalanced data
    counts = torch.bincount(y_train, minlength=3).float()
    weights = (1.0 / counts) * counts.sum() / 3.0
    logger.info(f"Class counts: {counts.tolist()}, weights: {weights.tolist()}")

    # Train classifier
    classifier = ThreeClassHead(input_dim=256, dropout=args.dropout).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_val_f1, best_epoch = 0.0, 0
    best_state = None

    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_val.to(device))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val.numpy(), val_preds, average="macro")

        avg_loss = total_loss / len(train_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_macro_f1={val_f1:.4f} (best={best_val_f1:.4f} @{best_epoch+1})")

    # Evaluate on test with best model
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(X_test.to(device))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

    y_test_np = y_test.numpy()
    report = classification_report(y_test_np, test_preds, target_names=LABEL_NAMES, output_dict=True)
    cm = confusion_matrix(y_test_np, test_preds)

    print(f"\nBest model from epoch {best_epoch+1} (val macro F1={best_val_f1:.4f})")
    print(classification_report(y_test_np, test_preds, target_names=LABEL_NAMES))
    print("Confusion matrix:")
    print(cm)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "best_epoch": best_epoch + 1,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_classification_report": report,
        "test_confusion_matrix": cm.tolist(),
        "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
        "test_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "test_accuracy": round(report["accuracy"], 4),
    }
    with open(out_dir / "3class_results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save({"model_state_dict": best_state, "epoch": best_epoch + 1}, out_dir / "3class_classifier.pt")
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
