#!/usr/bin/env python3
"""
Three-Class Ordinal Evaluation for Phase 1 Embeddings.

Evaluates whether the contrastive model learned the ordinal ordering:
    good_fit > potential_fit > no_fit

Metrics:
  - Per-tier similarity distributions (mean, std, separation)
  - Pairwise ordering accuracy: P(sim(good) > sim(potential) > sim(no))
  - Kendall's tau-b between predicted similarity rank and true ordinal rank
  - Cohen's d effect sizes between adjacent tiers
  - Three-class AUC (one-vs-rest)
  - Ordinal accuracy: fraction of triplets where full ordering is correct

Runs on both the ordinal checkpoint and the baseline checkpoint for comparison.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sentence_transformers import SentenceTransformer
from scipy import stats


class CareerAwareContrastiveModel(nn.Module):
    """Must match trainer.py architecture."""
    def __init__(self, input_dim: int = 384, projection_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection_head(x)
        return F.normalize(projected, p=2, dim=-1)


class OrdinalDataset(Dataset):
    """Loads JSONL with three-class original_label."""
    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    label = item.get('metadata', {}).get('original_label', 'unknown')
                    if label in ('good_fit', 'potential_fit', 'no_fit'):
                        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'resume': item['resume'],
            'job': item['job'],
            'original_label': item['metadata']['original_label'],
            'phi': item['metadata'].get('phi'),
            'quality_tier': item['metadata'].get('quality_tier', 'F'),
        }


def content_to_text(content: Dict, content_type: str) -> str:
    """Convert structured content to text for embedding."""
    if content_type == 'resume':
        parts = []
        if 'skills' in content and content['skills']:
            if isinstance(content['skills'], list):
                names = [s.get('name', '') if isinstance(s, dict) else s for s in content['skills']]
                if names:
                    parts.append("Skills: " + ", ".join(filter(None, names)))
            elif isinstance(content['skills'], str):
                parts.append("Skills: " + content['skills'])
        if 'experience' in content and content['experience']:
            if isinstance(content['experience'], list):
                exps = []
                for exp in content['experience']:
                    if isinstance(exp, dict):
                        t = f"{exp.get('title', '')} at {exp.get('company', '')}"
                        if exp.get('description'):
                            t += f": {exp['description']}"
                        exps.append(t)
                    elif isinstance(exp, str):
                        exps.append(exp)
                parts.append("Experience: " + ". ".join(exps))
            elif isinstance(content['experience'], str):
                parts.append("Experience: " + content['experience'])
        if 'education' in content and content['education']:
            if isinstance(content['education'], list):
                edus = []
                for edu in content['education']:
                    if isinstance(edu, dict):
                        edus.append(f"{edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}")
                    elif isinstance(edu, str):
                        edus.append(edu)
                parts.append("Education: " + ". ".join(edus))
            elif isinstance(content['education'], str):
                parts.append("Education: " + content['education'])
        return " | ".join(parts) if parts else "No resume information"
    elif content_type == 'job':
        parts = []
        if content.get('title'):
            parts.append(f"Job Title: {content['title']}")
        if content.get('company'):
            parts.append(f"Company: {content['company']}")
        if content.get('description'):
            parts.append(f"Description: {content['description']}")
        if 'required_skills' in content and content['required_skills']:
            if isinstance(content['required_skills'], list):
                names = [s.get('name', '') if isinstance(s, dict) else s for s in content['required_skills']]
                if names:
                    parts.append("Required Skills: " + ", ".join(filter(None, names)))
        return " | ".join(parts) if parts else "No job information"
    return ""


def compute_embeddings(model, text_encoder, data_loader, device):
    """Compute cosine similarities and collect labels."""
    model.eval()
    similarities = []
    labels = []
    phis = []

    with torch.no_grad():
        for batch in data_loader:
            resume_texts = [content_to_text(r, 'resume') for r in batch['resume']]
            job_texts = [content_to_text(j, 'job') for j in batch['job']]

            resume_emb = text_encoder.encode(resume_texts, convert_to_tensor=True).to(device)
            job_emb = text_encoder.encode(job_texts, convert_to_tensor=True).to(device)

            resume_proj = model(resume_emb)
            job_proj = model(job_emb)

            sims = torch.sum(resume_proj * job_proj, dim=1).cpu().numpy()
            similarities.extend(sims)
            labels.extend(batch['original_label'])
            phis.extend([p if p is not None else 0.5 for p in batch['phi']])

    return np.array(similarities), labels, np.array(phis)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def ordinal_metrics(similarities, labels):
    """Compute all ordinal evaluation metrics."""
    sims_by_label = defaultdict(list)
    for sim, label in zip(similarities, labels):
        sims_by_label[label].append(sim)

    good = np.array(sims_by_label['good_fit'])
    potential = np.array(sims_by_label['potential_fit'])
    no = np.array(sims_by_label['no_fit'])

    results = {}

    # ── 1. Per-tier similarity distributions ──
    results['tier_stats'] = {}
    for name, arr in [('good_fit', good), ('potential_fit', potential), ('no_fit', no)]:
        results['tier_stats'][name] = {
            'count': len(arr),
            'mean': float(np.mean(arr)) if len(arr) > 0 else None,
            'std': float(np.std(arr)) if len(arr) > 0 else None,
            'median': float(np.median(arr)) if len(arr) > 0 else None,
            'min': float(np.min(arr)) if len(arr) > 0 else None,
            'max': float(np.max(arr)) if len(arr) > 0 else None,
        }

    # ── 2. Pairwise separation ──
    results['separations'] = {}
    if len(good) > 0 and len(potential) > 0:
        results['separations']['good_vs_potential'] = {
            'mean_diff': float(np.mean(good) - np.mean(potential)),
            'cohens_d': float(cohens_d(good, potential)),
            'mannwhitney_p': float(stats.mannwhitneyu(good, potential, alternative='greater').pvalue),
        }
    if len(potential) > 0 and len(no) > 0:
        results['separations']['potential_vs_no'] = {
            'mean_diff': float(np.mean(potential) - np.mean(no)),
            'cohens_d': float(cohens_d(potential, no)),
            'mannwhitney_p': float(stats.mannwhitneyu(potential, no, alternative='greater').pvalue),
        }
    if len(good) > 0 and len(no) > 0:
        results['separations']['good_vs_no'] = {
            'mean_diff': float(np.mean(good) - np.mean(no)),
            'cohens_d': float(cohens_d(good, no)),
            'mannwhitney_p': float(stats.mannwhitneyu(good, no, alternative='greater').pvalue),
        }

    # ── 3. Pairwise ordering accuracy ──
    # P(sim(good) > sim(potential)) across all pairs
    results['pairwise_ordering'] = {}
    if len(good) > 0 and len(potential) > 0:
        correct = sum(1 for g in good for p in potential if g > p)
        total = len(good) * len(potential)
        results['pairwise_ordering']['good_above_potential'] = float(correct / total)
    if len(potential) > 0 and len(no) > 0:
        correct = sum(1 for p in potential for n in no if p > n)
        total = len(potential) * len(no)
        results['pairwise_ordering']['potential_above_no'] = float(correct / total)
    if len(good) > 0 and len(no) > 0:
        correct = sum(1 for g in good for n in no if g > n)
        total = len(good) * len(no)
        results['pairwise_ordering']['good_above_no'] = float(correct / total)

    # ── 4. Full ordinal triplet accuracy ──
    # Sample triplets (one from each tier) and check if ordering is correct
    if len(good) > 0 and len(potential) > 0 and len(no) > 0:
        np.random.seed(42)
        n_triplets = min(10000, len(good) * len(potential) * len(no))
        correct_triplets = 0
        for _ in range(n_triplets):
            g = np.random.choice(good)
            p = np.random.choice(potential)
            n = np.random.choice(no)
            if g > p > n:
                correct_triplets += 1
        results['ordinal_triplet_accuracy'] = float(correct_triplets / n_triplets)
    else:
        results['ordinal_triplet_accuracy'] = None

    # ── 5. Kendall's tau-b ──
    # Map labels to ordinal ranks: good_fit=2, potential_fit=1, no_fit=0
    rank_map = {'good_fit': 2, 'potential_fit': 1, 'no_fit': 0}
    true_ranks = np.array([rank_map[l] for l in labels])
    tau, tau_p = stats.kendalltau(similarities, true_ranks)
    results['kendalls_tau'] = {'tau': float(tau), 'p_value': float(tau_p)}

    # ── 6. Spearman's rho ──
    rho, rho_p = stats.spearmanr(similarities, true_ranks)
    results['spearmans_rho'] = {'rho': float(rho), 'p_value': float(rho_p)}

    # ── 7. Three-class AUC (one-vs-rest) ──
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    classes = ['no_fit', 'potential_fit', 'good_fit']
    y_true = label_binarize([l for l in labels], classes=classes)
    # Use similarity as score — higher sim should predict good_fit
    y_score = np.column_stack([
        1 - similarities,  # no_fit: low similarity
        0.5 * np.ones_like(similarities),  # potential_fit: middle (uniform prior)
        similarities,  # good_fit: high similarity
    ])
    try:
        auc_ovr = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        results['three_class_auc_ovr'] = float(auc_ovr)
    except Exception:
        results['three_class_auc_ovr'] = None

    # Better approach: use similarity directly for binary AUCs
    results['binary_aucs'] = {}
    # good_fit vs rest
    y_good = np.array([1 if l == 'good_fit' else 0 for l in labels])
    results['binary_aucs']['good_vs_rest'] = float(roc_auc_score(y_good, similarities))
    # potential_fit vs no_fit (excluding good_fit)
    mask_no_good = np.array([l != 'good_fit' for l in labels])
    if mask_no_good.sum() > 0:
        y_pot = np.array([1 if l == 'potential_fit' else 0 for l in labels])[mask_no_good]
        s_pot = similarities[mask_no_good]
        if len(np.unique(y_pot)) > 1:
            results['binary_aucs']['potential_vs_no'] = float(roc_auc_score(y_pot, s_pot))
    # good_fit vs potential_fit (excluding no_fit)
    mask_no_no = np.array([l != 'no_fit' for l in labels])
    if mask_no_no.sum() > 0:
        y_gp = np.array([1 if l == 'good_fit' else 0 for l in labels])[mask_no_no]
        s_gp = similarities[mask_no_no]
        if len(np.unique(y_gp)) > 1:
            results['binary_aucs']['good_vs_potential'] = float(roc_auc_score(y_gp, s_gp))

    # ── 8. Three-class classification metrics ──
    results['classification'] = _three_class_classification(similarities, labels, good, potential, no)

    # ── 9. Graduated ranking metrics ──
    results['ranking'] = _graduated_ranking(similarities, labels)

    return results


def _three_class_classification(similarities, labels, good, potential, no):
    """
    Three-class classification using optimal similarity thresholds.

    Finds two thresholds (t_high, t_low) that maximize three-class accuracy:
      sim >= t_high  → good_fit
      t_low <= sim < t_high → potential_fit
      sim < t_low → no_fit
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    true_labels = np.array(labels)
    best_acc = 0
    best_thresholds = (0.5, 0.3)

    # Grid search over threshold pairs
    for t_high in np.arange(0.3, 0.9, 0.02):
        for t_low in np.arange(0.0, t_high, 0.02):
            preds = []
            for s in similarities:
                if s >= t_high:
                    preds.append('good_fit')
                elif s >= t_low:
                    preds.append('potential_fit')
                else:
                    preds.append('no_fit')
            acc = accuracy_score(true_labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresholds = (t_high, t_low)

    # Compute metrics at optimal thresholds
    t_high, t_low = best_thresholds
    preds = []
    for s in similarities:
        if s >= t_high:
            preds.append('good_fit')
        elif s >= t_low:
            preds.append('potential_fit')
        else:
            preds.append('no_fit')
    preds = np.array(preds)

    # Per-class metrics
    class_names = ['good_fit', 'potential_fit', 'no_fit']
    report = classification_report(true_labels, preds, labels=class_names, output_dict=True, zero_division=0)

    # Ordinal accuracy: prediction is "adjacent correct" (off by at most 1 tier)
    rank_map = {'good_fit': 2, 'potential_fit': 1, 'no_fit': 0}
    true_ranks = np.array([rank_map[l] for l in true_labels])
    pred_ranks = np.array([rank_map[p] for p in preds])
    exact_correct = np.sum(true_ranks == pred_ranks)
    adjacent_correct = np.sum(np.abs(true_ranks - pred_ranks) <= 1)

    # Mean Absolute Error on ordinal ranks
    mae = float(np.mean(np.abs(true_ranks - pred_ranks)))

    return {
        'optimal_thresholds': {'t_high': float(t_high), 't_low': float(t_low)},
        'accuracy': float(best_acc),
        'macro_f1': float(f1_score(true_labels, preds, labels=class_names, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(true_labels, preds, labels=class_names, average='weighted', zero_division=0)),
        'per_class': {
            name: {
                'precision': report[name]['precision'],
                'recall': report[name]['recall'],
                'f1': report[name]['f1-score'],
                'support': report[name]['support'],
            }
            for name in class_names
        },
        'exact_accuracy': float(exact_correct / len(true_labels)),
        'adjacent_accuracy': float(adjacent_correct / len(true_labels)),
        'ordinal_mae': mae,
        'confusion_matrix': _confusion_matrix_3class(true_labels, preds, class_names),
    }


def _confusion_matrix_3class(true_labels, pred_labels, class_names):
    """Build a 3x3 confusion matrix as a dict."""
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(true_labels, pred_labels, labels=class_names)
    result = {}
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            result[f'true_{true_name}_pred_{pred_name}'] = int(cm[i][j])
    return result


def _graduated_ranking(similarities, labels):
    """
    Graduated ranking metrics using three-tier relevance scores.

    Relevance: good_fit=2, potential_fit=1, no_fit=0
    This gives partial credit to potential_fit ranked highly (unlike binary).

    Groups samples by job and ranks resumes for each job.
    """
    # Group by job (use job title + description hash as key)
    from collections import defaultdict
    import hashlib

    # We need to work with the raw data — but we only have similarities and labels here.
    # Instead, compute global ranking metrics: for each sample, its similarity is the score
    # and its relevance is the ordinal label.

    relevance_map = {'good_fit': 2, 'potential_fit': 1, 'no_fit': 0}
    relevances = np.array([relevance_map[l] for l in labels])

    # Sort by similarity (descending) — this is the model's ranking
    sorted_indices = np.argsort(-similarities)
    sorted_relevances = relevances[sorted_indices]
    sorted_sims = similarities[sorted_indices]

    # ── Graduated NDCG (using 2^rel - 1 gains) ──
    def dcg(rels, k=None):
        if k is not None:
            rels = rels[:k]
        gains = (2.0 ** rels) - 1.0
        discounts = np.log2(np.arange(1, len(rels) + 1) + 1)
        return float(np.sum(gains / discounts))

    # Ideal ranking: sort by relevance descending
    ideal_rels = np.sort(relevances)[::-1]

    ndcg_full = dcg(sorted_relevances) / dcg(ideal_rels) if dcg(ideal_rels) > 0 else 0
    ndcg_10 = dcg(sorted_relevances, 10) / dcg(ideal_rels, 10) if dcg(ideal_rels, 10) > 0 else 0
    ndcg_20 = dcg(sorted_relevances, 20) / dcg(ideal_rels, 20) if dcg(ideal_rels, 20) > 0 else 0
    ndcg_50 = dcg(sorted_relevances, 50) / dcg(ideal_rels, 50) if dcg(ideal_rels, 50) > 0 else 0

    # ── Graduated MAP (treat good_fit=relevant, potential_fit=partially relevant) ──
    # MAP with relevance >= 1 (good_fit and potential_fit are "relevant")
    relevant_mask_broad = sorted_relevances >= 1  # good_fit + potential_fit
    relevant_mask_strict = sorted_relevances >= 2  # good_fit only

    def average_precision(relevant_mask):
        if not np.any(relevant_mask):
            return 0.0
        precisions = []
        relevant_count = 0
        for i, is_rel in enumerate(relevant_mask):
            if is_rel:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        return float(np.mean(precisions))

    map_broad = average_precision(relevant_mask_broad)
    map_strict = average_precision(relevant_mask_strict)

    # ── Weighted MAP: good_fit contributes 1.0, potential_fit contributes 0.5 ──
    def weighted_average_precision(sorted_rels):
        weights = {2: 1.0, 1: 0.5, 0: 0.0}
        total_weight = sum(weights[int(r)] for r in sorted_rels if weights.get(int(r), 0) > 0)
        if total_weight == 0:
            return 0.0
        weighted_prec_sum = 0.0
        relevant_count = 0
        for i, rel in enumerate(sorted_rels):
            w = weights.get(int(rel), 0)
            if w > 0:
                relevant_count += 1
                weighted_prec_sum += w * (relevant_count / (i + 1))
        return float(weighted_prec_sum / total_weight)

    wmap = weighted_average_precision(sorted_relevances)

    # ── Precision@k for different k values ──
    precisions_at_k = {}
    for k in [5, 10, 20, 50]:
        if k <= len(sorted_relevances):
            top_k = sorted_relevances[:k]
            # Broad: good_fit + potential_fit
            precisions_at_k[f'p@{k}_broad'] = float(np.sum(top_k >= 1) / k)
            # Strict: good_fit only
            precisions_at_k[f'p@{k}_strict'] = float(np.sum(top_k >= 2) / k)
            # Weighted: good_fit=1.0, potential_fit=0.5
            precisions_at_k[f'p@{k}_weighted'] = float((np.sum(top_k == 2) + 0.5 * np.sum(top_k == 1)) / k)

    return {
        'ndcg_full': float(ndcg_full),
        'ndcg@10': float(ndcg_10),
        'ndcg@20': float(ndcg_20),
        'ndcg@50': float(ndcg_50),
        'map_broad': float(map_broad),
        'map_strict': float(map_strict),
        'weighted_map': float(wmap),
        'precisions_at_k': precisions_at_k,
    }


def load_model(checkpoint_path, config_path, device):
    """Load a model from checkpoint."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect dimensions from checkpoint
    model_state = checkpoint.get('model_state_dict', checkpoint)
    input_dim = model_state['projection_head.0.weight'].shape[1]
    projection_dim = model_state['projection_head.3.weight'].shape[0]

    model = CareerAwareContrastiveModel(
        input_dim=input_dim,
        projection_dim=projection_dim,
        dropout=0.1
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config_dict


def print_results(name, results):
    """Pretty-print ordinal evaluation results."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    # Tier stats
    print(f"\n  Similarity Distributions:")
    print(f"  {'Tier':<15} {'Count':>6} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print(f"  {'-'*50}")
    for tier in ['good_fit', 'potential_fit', 'no_fit']:
        s = results['tier_stats'][tier]
        print(f"  {tier:<15} {s['count']:>6} {s['mean']:>8.4f} {s['std']:>8.4f} {s['median']:>8.4f}")

    # Separations
    print(f"\n  Pairwise Separations:")
    print(f"  {'Pair':<25} {'Mean Diff':>10} {'Cohen d':>10} {'p-value':>10}")
    print(f"  {'-'*58}")
    for pair, s in results.get('separations', {}).items():
        sig = '***' if s['mannwhitney_p'] < 0.001 else '**' if s['mannwhitney_p'] < 0.01 else '*' if s['mannwhitney_p'] < 0.05 else 'ns'
        print(f"  {pair:<25} {s['mean_diff']:>10.4f} {s['cohens_d']:>10.4f} {s['mannwhitney_p']:>9.4f} {sig}")

    # Ordering accuracy
    print(f"\n  Pairwise Ordering Accuracy:")
    for pair, acc in results.get('pairwise_ordering', {}).items():
        print(f"    {pair}: {acc:.4f} ({acc*100:.1f}%)")

    if results.get('ordinal_triplet_accuracy') is not None:
        print(f"    Full triplet (g > p > n): {results['ordinal_triplet_accuracy']:.4f} ({results['ordinal_triplet_accuracy']*100:.1f}%)")

    # Rank correlations
    print(f"\n  Rank Correlations:")
    tau = results['kendalls_tau']
    rho = results['spearmans_rho']
    print(f"    Kendall's tau-b: {tau['tau']:.4f} (p={tau['p_value']:.4e})")
    print(f"    Spearman's rho: {rho['rho']:.4f} (p={rho['p_value']:.4e})")

    # AUCs
    print(f"\n  Binary AUCs:")
    for pair, auc in results.get('binary_aucs', {}).items():
        print(f"    {pair}: {auc:.4f}")

    if results.get('three_class_auc_ovr') is not None:
        print(f"    Three-class OVR macro: {results['three_class_auc_ovr']:.4f}")

    # Classification metrics
    clf = results.get('classification', {})
    if clf:
        t_h = clf['optimal_thresholds']['t_high']
        t_l = clf['optimal_thresholds']['t_low']
        print(f"\n  Three-Class Classification (thresholds: t_high={t_h:.2f}, t_low={t_l:.2f}):")
        print(f"    Accuracy:           {clf['accuracy']:.4f} ({clf['accuracy']*100:.1f}%)")
        print(f"    Macro F1:           {clf['macro_f1']:.4f}")
        print(f"    Weighted F1:        {clf['weighted_f1']:.4f}")
        print(f"    Adjacent Accuracy:  {clf['adjacent_accuracy']:.4f} ({clf['adjacent_accuracy']*100:.1f}%)")
        print(f"    Ordinal MAE:        {clf['ordinal_mae']:.4f}")
        print(f"\n    Per-Class:")
        print(f"    {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        print(f"    {'-'*55}")
        for name in ['good_fit', 'potential_fit', 'no_fit']:
            c = clf['per_class'][name]
            print(f"    {name:<15} {c['precision']:>10.4f} {c['recall']:>10.4f} {c['f1']:>10.4f} {c['support']:>8}")

        print(f"\n    Confusion Matrix (rows=true, cols=predicted):")
        cm = clf['confusion_matrix']
        classes = ['good_fit', 'potential_fit', 'no_fit']
        print(f"    {'':>15} {'good_fit':>10} {'potential':>10} {'no_fit':>10}")
        for true_cls in classes:
            row = [cm.get(f'true_{true_cls}_pred_{pred_cls}', 0) for pred_cls in classes]
            print(f"    {true_cls:>15} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    # Ranking metrics
    rnk = results.get('ranking', {})
    if rnk:
        print(f"\n  Graduated Ranking Metrics (relevance: good=2, potential=1, no=0):")
        print(f"    NDCG (full):    {rnk['ndcg_full']:.4f}")
        print(f"    NDCG@10:        {rnk['ndcg@10']:.4f}")
        print(f"    NDCG@20:        {rnk['ndcg@20']:.4f}")
        print(f"    NDCG@50:        {rnk['ndcg@50']:.4f}")
        print(f"    MAP (broad):    {rnk['map_broad']:.4f}  (good_fit + potential_fit = relevant)")
        print(f"    MAP (strict):   {rnk['map_strict']:.4f}  (good_fit only = relevant)")
        print(f"    Weighted MAP:   {rnk['weighted_map']:.4f}  (good=1.0, potential=0.5)")
        pak = rnk.get('precisions_at_k', {})
        if pak:
            print(f"\n    Precision@k:")
            print(f"    {'k':>5} {'Broad':>10} {'Strict':>10} {'Weighted':>10}")
            print(f"    {'-'*38}")
            for k in [5, 10, 20, 50]:
                b = pak.get(f'p@{k}_broad', 0)
                s = pak.get(f'p@{k}_strict', 0)
                w = pak.get(f'p@{k}_weighted', 0)
                print(f"    {k:>5} {b:>10.4f} {s:>10.4f} {w:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Three-Class Ordinal Evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Validation JSONL with original_label")
    parser.add_argument("--ordinal-checkpoint", type=str, required=True, help="Ordinal v3 checkpoint")
    parser.add_argument("--ordinal-config", type=str, required=True, help="Ordinal config JSON")
    parser.add_argument("--baseline-checkpoint", type=str, default=None, help="Baseline InfoNCE checkpoint (optional)")
    parser.add_argument("--baseline-config", type=str, default=None, help="Baseline config JSON (optional)")
    parser.add_argument("--output-dir", type=str, default="ordinal_evaluation", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = OrdinalDataset(args.dataset)
    data_loader = TorchDataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=lambda x: {
            'resume': [item['resume'] for item in x],
            'job': [item['job'] for item in x],
            'original_label': [item['original_label'] for item in x],
            'phi': [item['phi'] for item in x],
            'quality_tier': [item['quality_tier'] for item in x],
        }
    )
    print(f"Dataset: {len(dataset)} samples")

    # Load text encoder (shared)
    with open(args.ordinal_config, 'r') as f:
        cfg = json.load(f)
    encoder_name = cfg.get('text_encoder_model', 'sentence-transformers/all-mpnet-base-v2')
    print(f"Loading text encoder: {encoder_name}")
    text_encoder = SentenceTransformer(encoder_name).to(device)

    all_results = {}

    # ── Evaluate ordinal model ──
    print(f"\nLoading ordinal model: {args.ordinal_checkpoint}")
    ordinal_model, _ = load_model(args.ordinal_checkpoint, args.ordinal_config, device)
    sims_ord, labels_ord, phis_ord = compute_embeddings(ordinal_model, text_encoder, data_loader, device)
    results_ord = ordinal_metrics(sims_ord, labels_ord)
    all_results['ordinal_v3'] = results_ord
    print_results("ORDINAL v3 (InfoNCE + L₂ + L₃)", results_ord)

    # ── Evaluate baseline model (if provided) ──
    if args.baseline_checkpoint and args.baseline_config:
        print(f"\nLoading baseline model: {args.baseline_checkpoint}")
        baseline_model, _ = load_model(args.baseline_checkpoint, args.baseline_config, device)
        sims_base, labels_base, _ = compute_embeddings(baseline_model, text_encoder, data_loader, device)
        results_base = ordinal_metrics(sims_base, labels_base)
        all_results['baseline_infonce'] = results_base
        print_results("BASELINE (InfoNCE only)", results_base)

        # ── Comparison table ──
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON: Ordinal v3 vs Baseline")
        print(f"{'=' * 70}")
        print(f"\n  {'Metric':<35} {'Baseline':>10} {'Ordinal':>10} {'Delta':>10}")
        print(f"  {'-'*68}")

        comparisons = [
            ("good_fit mean sim", results_base['tier_stats']['good_fit']['mean'], results_ord['tier_stats']['good_fit']['mean']),
            ("potential_fit mean sim", results_base['tier_stats']['potential_fit']['mean'], results_ord['tier_stats']['potential_fit']['mean']),
            ("no_fit mean sim", results_base['tier_stats']['no_fit']['mean'], results_ord['tier_stats']['no_fit']['mean']),
        ]

        for pair in ['good_vs_potential', 'potential_vs_no', 'good_vs_no']:
            if pair in results_base.get('separations', {}) and pair in results_ord.get('separations', {}):
                comparisons.append((
                    f"Cohen's d ({pair})",
                    results_base['separations'][pair]['cohens_d'],
                    results_ord['separations'][pair]['cohens_d'],
                ))

        for pair in ['good_above_potential', 'potential_above_no', 'good_above_no']:
            if pair in results_base.get('pairwise_ordering', {}) and pair in results_ord.get('pairwise_ordering', {}):
                comparisons.append((
                    f"P({pair})",
                    results_base['pairwise_ordering'][pair],
                    results_ord['pairwise_ordering'][pair],
                ))

        if results_base.get('ordinal_triplet_accuracy') is not None and results_ord.get('ordinal_triplet_accuracy') is not None:
            comparisons.append(("Triplet accuracy (g>p>n)", results_base['ordinal_triplet_accuracy'], results_ord['ordinal_triplet_accuracy']))

        comparisons.append(("Kendall's tau", results_base['kendalls_tau']['tau'], results_ord['kendalls_tau']['tau']))
        comparisons.append(("Spearman's rho", results_base['spearmans_rho']['rho'], results_ord['spearmans_rho']['rho']))

        for pair in ['good_vs_rest', 'potential_vs_no', 'good_vs_potential']:
            if pair in results_base.get('binary_aucs', {}) and pair in results_ord.get('binary_aucs', {}):
                comparisons.append((f"AUC ({pair})", results_base['binary_aucs'][pair], results_ord['binary_aucs'][pair]))

        # Classification metrics
        clf_b = results_base.get('classification', {})
        clf_o = results_ord.get('classification', {})
        if clf_b and clf_o:
            comparisons.append(("3-class accuracy", clf_b['accuracy'], clf_o['accuracy']))
            comparisons.append(("3-class macro F1", clf_b['macro_f1'], clf_o['macro_f1']))
            comparisons.append(("Adjacent accuracy", clf_b['adjacent_accuracy'], clf_o['adjacent_accuracy']))
            comparisons.append(("Ordinal MAE", clf_b['ordinal_mae'], clf_o['ordinal_mae']))

        # Ranking metrics
        rnk_b = results_base.get('ranking', {})
        rnk_o = results_ord.get('ranking', {})
        if rnk_b and rnk_o:
            comparisons.append(("NDCG (graduated)", rnk_b['ndcg_full'], rnk_o['ndcg_full']))
            comparisons.append(("NDCG@10", rnk_b['ndcg@10'], rnk_o['ndcg@10']))
            comparisons.append(("MAP (broad)", rnk_b['map_broad'], rnk_o['map_broad']))
            comparisons.append(("MAP (strict)", rnk_b['map_strict'], rnk_o['map_strict']))
            comparisons.append(("Weighted MAP", rnk_b['weighted_map'], rnk_o['weighted_map']))

        for name, base_val, ord_val in comparisons:
            if base_val is not None and ord_val is not None:
                delta = ord_val - base_val
                marker = "▲" if delta > 0 else "▼" if delta < 0 else "="
                print(f"  {name:<35} {base_val:>10.4f} {ord_val:>10.4f} {marker}{abs(delta):>9.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "ordinal_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
