#!/usr/bin/env python3
"""
Phase 1 Embedding Evaluation Script

Evaluates the quality of contrastive embeddings learned in Phase 1.
Uses cosine similarity between resume and job embeddings for classification.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.structured_features import (
    StructuredFeatureExtractor, 
    StructuredFeatureEncoder,
    EXPERIENCE_LEVELS
)


class CareerAwareContrastiveModel(nn.Module):
    """
    Minimal contrastive model for career-aware resume-job matching.
    Must match the architecture used in trainer.py
    """
    def __init__(self, input_dim: int = 384, projection_dim: int = 128, dropout: float = 0.1,
                 use_structured_features: bool = False, structured_feature_dim: int = 32):
        super().__init__()
        self.use_structured_features = use_structured_features
        self.structured_feature_dim = structured_feature_dim if use_structured_features else 0
        
        # Structured feature encoder (if enabled)
        if use_structured_features:
            self.structured_encoder = StructuredFeatureEncoder(
                num_experience_levels=10,
                experience_embed_dim=16,
                numerical_features=3,
                output_dim=structured_feature_dim
            )
        else:
            self.structured_encoder = None
        
        # Combined input dimension
        combined_dim = input_dim + self.structured_feature_dim
        
        self.projection_head = nn.Sequential(
            nn.Linear(combined_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim)
        )

    def forward(self, x: torch.Tensor, 
                experience_level_idx: torch.Tensor = None,
                numerical_features: torch.Tensor = None) -> torch.Tensor:
        if self.use_structured_features and experience_level_idx is not None and numerical_features is not None:
            # Encode structured features
            structured_encoded = self.structured_encoder(experience_level_idx, numerical_features)
            # Concatenate text and structured features
            combined = torch.cat([x, structured_encoded], dim=-1)
        else:
            combined = x
        
        projected = self.projection_head(combined)
        return F.normalize(projected, p=2, dim=-1)


class JSONLDataset(Dataset):
    """Dataset for loading JSONL evaluation data"""
    
    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract label
        if 'metadata' in item and 'label' in item['metadata']:
            label = int(item['metadata']['label'])
        elif 'label' in item:
            if isinstance(item['label'], str):
                label_map = {'positive': 1, 'negative': 0}
                label = label_map.get(item['label'].lower(), 1)
            else:
                label = int(item['label'])
        else:
            label = 1
        
        return {
            'resume': item['resume'],
            'job': item['job'],
            'label': label,
            'sample_id': item.get('sample_id', f'sample_{idx}')
        }


#: Process-local cache of frozen text-encoder outputs, keyed by the exact text.
#: The encoder is frozen during Phase-1 evaluation, so its output for a given
#: string is fixed and safe to reuse. Without this the script re-encodes every
#: record on every invocation — 6,216 texts per run on the trials validation
#: split, ~20 minutes each, repeated for every arm and seed.
#:
#: Keyed by text rather than by record so the many (topic, trial) pairs that share
#: a topic narrative encode it once.
_TEXT_EMB_CACHE: Dict[str, "torch.Tensor"] = {}


def _batch_grades(jobs, n: int) -> List:
    """Per-record graded relevance, or ``None`` where unavailable.

    Reads the ``grade`` the trials domain attaches to each candidate slot. Career
    and CVE records carry none, so this yields ``None`` and the per-grade section
    is simply omitted from the report.

    The DataLoader may hand ``jobs`` over as a dict of collated lists rather than a
    list of dicts, so both shapes are handled.
    """
    def one(i):
        try:
            if isinstance(jobs, dict):
                value = jobs.get('grade')
                if value is None:
                    return None
                item = value[i] if hasattr(value, '__getitem__') else value
            else:
                item = jobs[i].get('grade') if isinstance(jobs[i], dict) else None
            if item is None:
                return None
            return int(item.item() if hasattr(item, 'item') else item)
        except Exception:
            return None

    return [one(i) for i in range(n)]


def per_grade_report(similarities, grades) -> Optional[Dict]:
    """Pairwise AUC between graded relevance levels.

    Reports each contrast separately: eligible-vs-irrelevant measures topical
    relevance, whereas eligible-vs-ineligible measures the eligibility judgement
    that is the actual task. These can diverge sharply — a model may be strong on
    the first and at chance on the second — and a pooled figure hides that.
    """
    pairs = [(g, s) for g, s in zip(grades, similarities) if g is not None]
    if len(pairs) < 2:
        return None

    by_grade: Dict[int, List[float]] = {}
    for g, s in pairs:
        by_grade.setdefault(g, []).append(float(s))
    if len(by_grade) < 2:
        return None

    def auc(pos: List[float], neg: List[float]) -> Optional[float]:
        if not pos or not neg:
            return None
        merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
        ranks: Dict[int, float] = {}
        i = 0
        while i < len(merged):
            j = i
            while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[k] = avg
            i = j + 1
        rank_sum = sum(ranks[k] for k, (_v, lab) in enumerate(merged) if lab == 1)
        n_pos, n_neg = len(pos), len(neg)
        return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    NAMES = {2: 'eligible', 1: 'ineligible', 0: 'not_relevant'}
    report: Dict = {
        'counts': {NAMES.get(g, str(g)): len(v) for g, v in sorted(by_grade.items())},
        'mean_similarity': {
            NAMES.get(g, str(g)): sum(v) / len(v) for g, v in sorted(by_grade.items())
        },
        'pairwise_auc': {},
    }
    for hi, lo, label in ((2, 0, 'eligible_vs_not_relevant'),
                          (2, 1, 'eligible_vs_ineligible'),
                          (1, 0, 'ineligible_vs_not_relevant')):
        value = auc(by_grade.get(hi, []), by_grade.get(lo, []))
        if value is not None:
            report['pairwise_auc'][label] = value
    return report


def _encode_cached(text_encoder, texts: List[str], device) -> "torch.Tensor":
    """Encode ``texts``, reusing previously computed embeddings.

    Only genuinely new strings are sent to the encoder; results are stacked back
    into the caller's original order.
    """
    missing = [t for t in dict.fromkeys(texts) if t not in _TEXT_EMB_CACHE]
    if missing:
        fresh = text_encoder.encode(missing, convert_to_tensor=True)
        for text, emb in zip(missing, fresh):
            _TEXT_EMB_CACHE[text] = emb.detach().cpu()
    return torch.stack([_TEXT_EMB_CACHE[t] for t in texts]).to(device)


def content_to_text(content: Dict, content_type: str) -> str:
    """Convert structured content to text for embedding.

    A pre-serialized ``encoder_view`` takes precedence over the career-schema
    branches below. Domains whose records are not resume/job shaped (the CVE and
    TREC clinical-trials domains) carry their text there, and the training path
    already honours it — ``trainer._encode_content_to_text_embedding`` and
    ``BatchEfficientEncoder._content_to_text`` both check for it first.

    Without this check the career branches find none of their expected keys and
    return a near-empty string, so evaluation silently scores the model on blank
    text. That produced AUC 0.5050 for a trials checkpoint that actually scores
    0.6876 when its real text is encoded — a null result manufactured entirely by
    the metric.
    """
    view = content.get('encoder_view')
    if isinstance(view, str) and view.strip():
        return view.strip()

    if content_type == 'resume':
        parts = []
        
        # Skills
        if 'skills' in content and content['skills']:
            if isinstance(content['skills'], list):
                skill_names = []
                for skill in content['skills']:
                    if isinstance(skill, dict):
                        skill_names.append(skill.get('name', ''))
                    elif isinstance(skill, str):
                        skill_names.append(skill)
                if skill_names:
                    parts.append("Skills: " + ", ".join(filter(None, skill_names)))
            elif isinstance(content['skills'], str):
                parts.append("Skills: " + content['skills'])
        
        # Experience
        if 'experience' in content and content['experience']:
            if isinstance(content['experience'], list):
                exp_texts = []
                for exp in content['experience']:
                    if isinstance(exp, dict):
                        exp_text = f"{exp.get('title', '')} at {exp.get('company', '')}"
                        if exp.get('description'):
                            exp_text += f": {exp['description']}"
                        exp_texts.append(exp_text)
                    elif isinstance(exp, str):
                        exp_texts.append(exp)
                parts.append("Experience: " + ". ".join(exp_texts))
            elif isinstance(content['experience'], str):
                parts.append("Experience: " + content['experience'])
        
        # Education
        if 'education' in content and content['education']:
            if isinstance(content['education'], list):
                edu_texts = []
                for edu in content['education']:
                    if isinstance(edu, dict):
                        edu_text = f"{edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}"
                        edu_texts.append(edu_text)
                    elif isinstance(edu, str):
                        edu_texts.append(edu)
                parts.append("Education: " + ". ".join(edu_texts))
            elif isinstance(content['education'], str):
                parts.append("Education: " + content['education'])
        
        return " | ".join(parts) if parts else "No resume information"
    
    elif content_type == 'job':
        parts = []
        
        if 'title' in content and content['title']:
            parts.append(f"Job Title: {content['title']}")
        
        if 'company' in content and content['company']:
            parts.append(f"Company: {content['company']}")
        
        if 'description' in content and content['description']:
            parts.append(f"Description: {content['description']}")
        
        if 'required_skills' in content and content['required_skills']:
            if isinstance(content['required_skills'], list):
                skill_names = [s.get('name', '') if isinstance(s, dict) else s 
                              for s in content['required_skills']]
                if skill_names:
                    parts.append("Required Skills: " + ", ".join(filter(None, skill_names)))
        
        return " | ".join(parts) if parts else "No job information"
    
    return ""


def evaluate_phase1_embeddings(model: nn.Module, 
                               text_encoder: SentenceTransformer,
                               data_loader: TorchDataLoader,
                               device: torch.device,
                               use_structured_features: bool = False,
                               feature_extractor: StructuredFeatureExtractor = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate using cosine similarity between embeddings.
    
    The model uses a shared projection head for both resumes and jobs,
    projecting them into the same embedding space for comparison.
    
    Returns:
        predictions: Binary predictions (0/1)
        similarities: Cosine similarity scores
        true_labels: Ground truth labels
    """
    model.eval()
    
    all_similarities = []
    all_labels = []
    # Graded relevance, when the dataset carries it. Pooled binary AUC collapses
    # several distinct contrasts into one number whose value depends on the
    # negative class's grade mixture, which is a property of how the split was
    # built rather than of the model. Collected alongside — never instead of — the
    # binary labels, so the pooled metric is unchanged.
    all_grades = []
    all_topic_ids = []

    with torch.no_grad():
        for batch in data_loader:
            resumes = batch['resume']
            all_topic_ids.extend(r.get('topic_id') for r in resumes)
            jobs = batch['job']
            labels = batch['label'].numpy()
            all_grades.extend(_batch_grades(jobs, len(labels)))
            
            # Convert to text and get base embeddings from SentenceTransformer
            resume_texts = [content_to_text(r, 'resume') for r in resumes]
            job_texts = [content_to_text(j, 'job') for j in jobs]

            # Refuse to score blank text. Encoding empty strings yields a valid
            # tensor and a plausible-looking AUC near chance, so this failure is
            # invisible without an explicit check — it is exactly how a trials
            # checkpoint scoring 0.6876 was reported as 0.5050.
            for name, texts in (('resume', resume_texts), ('job', job_texts)):
                blank = sum(1 for t in texts if not t or not t.strip())
                if blank:
                    raise ValueError(
                        f"{blank}/{len(texts)} {name} records serialized to empty "
                        f"text. content_to_text found none of its expected fields. "
                        f"For a non-career domain, ensure each record carries a "
                        f"non-empty 'encoder_view'."
                    )

            resume_base = _encode_cached(text_encoder, resume_texts, device)
            job_base = _encode_cached(text_encoder, job_texts, device)
            
            if use_structured_features and feature_extractor is not None:
                # Extract structured features for resumes
                resume_exp_levels = []
                resume_numerical = []
                for resume in resumes:
                    features = feature_extractor.extract_features(resume, 'resume')
                    exp_idx = features[:10].argmax().item()
                    numerical = features[10:]
                    resume_exp_levels.append(exp_idx)
                    resume_numerical.append(numerical)
                
                resume_exp_tensor = torch.tensor(resume_exp_levels, dtype=torch.long, device=device)
                resume_num_tensor = torch.stack(resume_numerical).to(device)
                
                # Extract structured features for jobs
                job_exp_levels = []
                job_numerical = []
                for job in jobs:
                    features = feature_extractor.extract_features(job, 'job')
                    exp_idx = features[:10].argmax().item()
                    numerical = features[10:]
                    job_exp_levels.append(exp_idx)
                    job_numerical.append(numerical)
                
                job_exp_tensor = torch.tensor(job_exp_levels, dtype=torch.long, device=device)
                job_num_tensor = torch.stack(job_numerical).to(device)
                
                # Pass through contrastive projection model with structured features
                resume_projected = model(resume_base, resume_exp_tensor, resume_num_tensor)
                job_projected = model(job_base, job_exp_tensor, job_num_tensor)
            else:
                # Pass through contrastive projection model (shared for both)
                resume_projected = model(resume_base)
                job_projected = model(job_base)
            
            # Compute cosine similarity (embeddings are already normalized by model)
            similarities = torch.sum(resume_projected * job_projected, dim=1)
            
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels)
    
    similarities = np.array(all_similarities)
    true_labels = np.array(all_labels)
    graded = per_grade_report(all_similarities, all_grades)
    
    # Convert similarities to probabilities (map from [-1, 1] to [0, 1])
    probabilities = (similarities + 1) / 2
    
    # Default threshold of 0.5 similarity (0.0 in cosine space)
    predictions = (similarities > 0.0).astype(int)

    # Stash the graded breakdown on the function so main() can attach it to the
    # results file without changing this function's return signature, which other
    # callers depend on.
    evaluate_phase1_embeddings.last_per_grade = graded
    from trials_domain.patient_metrics import patient_report
    evaluate_phase1_embeddings.last_per_patient = patient_report(
        all_topic_ids, all_similarities, all_grades)

    return predictions, probabilities, true_labels


def find_optimal_threshold(probabilities: np.ndarray, 
                           true_labels: np.ndarray) -> Tuple[float, float]:
    """Find optimal similarity threshold based on F1 score"""
    thresholds = np.linspace(-1.0, 1.0, 101)
    best_threshold = 0.0
    best_f1 = 0.0
    
    for threshold in thresholds:
        preds = (probabilities * 2 - 1 > threshold).astype(int)
        f1 = f1_score(true_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 Contrastive Embeddings")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Phase 1 checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to training config JSON")
    parser.add_argument("--output-dir", type=str, default="phase1_evaluation", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 1 EMBEDDING EVALUATION")
    print("=" * 80)
    print("\nEvaluating contrastive embeddings (no classification head):")
    print("  - Uses cosine similarity between resume/job embeddings")
    print("  - Measures representation quality from Phase 1 pretraining")
    print("  - Baseline for comparing Phase 2 improvements\n")
    
    # Load config.
    #
    # Uses TrainingConfig.from_json (which routes through from_dict and drops keys
    # that are not dataclass fields) rather than TrainingConfig(**config_dict).
    # The raw-kwargs form raises TypeError on any ``_``-prefixed documentation key,
    # which every non-career arm config carries (``_description``, ``_paired_with``,
    # ``_comment_the_factor``, ...). Those keys are deliberate provenance: they
    # record which single factor an arm varies. Loading the same way the trainer
    # does also removes a real hazard — the two paths could otherwise disagree
    # about a config, so a checkpoint would be evaluated under settings it was not
    # trained with.
    print(f"Loading config from: {args.config}")
    config = TrainingConfig.from_json(args.config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load checkpoint first to detect structured features
    print(f"Loading Phase 1 model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Detect if checkpoint uses structured features
    checkpoint_config = checkpoint.get('config', {})
    use_structured_features = checkpoint_config.get('use_structured_features', False)
    structured_feature_dim = checkpoint_config.get('structured_feature_dim', 32)
    
    # Also check model state dict for structured encoder keys
    model_state = checkpoint.get('model_state_dict', checkpoint)
    has_structured_keys = any('structured_encoder' in k for k in model_state.keys())
    
    if has_structured_keys and not use_structured_features:
        print("  Detected structured encoder in checkpoint, enabling structured features")
        use_structured_features = True
    
    print(f"  Structured features: {'enabled' if use_structured_features else 'disabled'}")
    if use_structured_features:
        print(f"  Structured feature dim: {structured_feature_dim}")
    
    # Load text encoder
    print(f"\nLoading text encoder: {config.text_encoder_model}")
    text_encoder = SentenceTransformer(config.text_encoder_model).to(device)
    text_encoder_dim = text_encoder.get_sentence_embedding_dimension()
    
    # Create model with same architecture as trainer
    projection_dim = getattr(config, 'projection_dim', 128)
    projection_dropout = getattr(config, 'projection_dropout', 0.1)
    
    print(f"Creating model: input_dim={text_encoder_dim}, projection_dim={projection_dim}")
    model = CareerAwareContrastiveModel(
        input_dim=text_encoder_dim,
        projection_dim=projection_dim,
        dropout=projection_dropout,
        use_structured_features=use_structured_features,
        structured_feature_dim=structured_feature_dim
    ).to(device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Model loaded successfully")
    
    # Initialize feature extractor if using structured features
    feature_extractor = None
    if use_structured_features:
        feature_extractor = StructuredFeatureExtractor()
        print(f"  Feature extractor initialized with {feature_extractor.feature_dim} features")
    
    # Print checkpoint info if available
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'loss' in checkpoint:
        print(f"  Training loss: {checkpoint['loss']:.6f}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    dataset = JSONLDataset(args.dataset)
    data_loader = TorchDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            'resume': [item['resume'] for item in x],
            'job': [item['job'] for item in x],
            'label': torch.tensor([item['label'] for item in x])
        }
    )
    
    print(f"Dataset samples: {len(dataset)}")
    print(f"Dataset batches: {len(data_loader)}")
    
    # Run evaluation
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    
    predictions, probabilities, true_labels = evaluate_phase1_embeddings(
        model, text_encoder, data_loader, device,
        use_structured_features=use_structured_features,
        feature_extractor=feature_extractor
    )
    
    # Find optimal threshold
    best_threshold, best_f1 = find_optimal_threshold(probabilities, true_labels)
    predictions_optimal = ((probabilities * 2 - 1) > best_threshold).astype(int)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("PHASE 1 EMBEDDING RESULTS")
    print("=" * 80)
    
    print(f"\n🔍 Threshold Analysis (cosine similarity):")
    print(f"  Default threshold (0.0): F1 = {f1_score(true_labels, predictions):.4f}")
    print(f"  Optimal threshold: {best_threshold:.4f}, F1 = {best_f1:.4f}")
    
    accuracy = accuracy_score(true_labels, predictions_optimal)
    precision = precision_score(true_labels, predictions_optimal, zero_division=0)
    recall = recall_score(true_labels, predictions_optimal, zero_division=0)
    f1 = f1_score(true_labels, predictions_optimal, zero_division=0)
    
    # Handle AUC-ROC calculation
    try:
        auc_roc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        auc_roc = 0.5  # Default if only one class present
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions_optimal)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, len(true_labels)
    
    print(f"\n🎯 Classification Metrics (Optimal Threshold = {best_threshold:.4f}):")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC-ROC:   {auc_roc:.4f} ({auc_roc*100:.2f}%)")
    
    print(f"\n🎲 Confusion Matrix:")
    print(f"  True Positives:    {tp}")
    print(f"  True Negatives:    {tn}")
    print(f"  False Positives:   {fp}")
    print(f"  False Negatives:   {fn}")
    
    print(f"\n📊 Additional Metrics:")
    print(f"  True Positive Rate (Recall):  {recall:.4f}")
    print(f"  True Negative Rate:           {tn/(tn+fp) if (tn+fp) > 0 else 0:.4f}")
    print(f"  False Positive Rate:          {fp/(fp+tn) if (fp+tn) > 0 else 0:.4f}")
    print(f"  False Negative Rate:          {fn/(fn+tp) if (fn+tp) > 0 else 0:.4f}")
    
    # Probability statistics
    pos_probs = probabilities[true_labels == 1]
    neg_probs = probabilities[true_labels == 0]
    
    print(f"\n📈 Similarity Statistics:")
    if len(pos_probs) > 0:
        print(f"  Positive samples avg similarity: {np.mean(pos_probs):.4f} ± {np.std(pos_probs):.4f}")
    if len(neg_probs) > 0:
        print(f"  Negative samples avg similarity: {np.mean(neg_probs):.4f} ± {np.std(neg_probs):.4f}")
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        print(f"  Separation:                      {abs(np.mean(pos_probs) - np.mean(neg_probs)):.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'phase': 'phase1_contrastive',
        'evaluation_method': 'cosine_similarity',
        'optimal_threshold': float(best_threshold),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc),
            # Kept as-is. The graded breakdown below is additive context, not a
            # replacement: pooled auc_roc remains the headline binary metric.
        },
        'confusion_matrix': {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        },
        'similarity_stats': {
            'positive_mean': float(np.mean(pos_probs)) if len(pos_probs) > 0 else None,
            'positive_std': float(np.std(pos_probs)) if len(pos_probs) > 0 else None,
            'negative_mean': float(np.mean(neg_probs)) if len(neg_probs) > 0 else None,
            'negative_std': float(np.std(neg_probs)) if len(neg_probs) > 0 else None,
            'separation': float(abs(np.mean(pos_probs) - np.mean(neg_probs))) if len(pos_probs) > 0 and len(neg_probs) > 0 else None
        },
        'dataset_size': len(dataset),
        'checkpoint': args.checkpoint,
        'dataset_path': args.dataset
    }

    # Additive: graded relevance breakdown, present only for datasets that carry
    # per-candidate grades. Pooled auc_roc above is untouched.
    per_patient = getattr(evaluate_phase1_embeddings, 'last_per_patient', None)
    if per_patient:
        results['per_patient'] = per_patient
    per_grade = getattr(evaluate_phase1_embeddings, 'last_per_grade', None)
    if per_grade:
        results['per_grade'] = per_grade
        print(f"\n📐 Graded relevance breakdown (additive to AUC-ROC above):")
        for name, count in per_grade['counts'].items():
            mean = per_grade['mean_similarity'][name]
            print(f"  {name:16s} n={count:6d}  mean_sim={mean:.4f}")
        print("  pairwise AUC:")
        for label, value in per_grade['pairwise_auc'].items():
            print(f"    {label:32s} {value:.4f}")
        elig = per_grade['pairwise_auc'].get('eligible_vs_ineligible')
        topical = per_grade['pairwise_auc'].get('eligible_vs_not_relevant')
        if elig is not None and topical is not None:
            print(f"  -> topical relevance {topical:.4f} vs eligibility {elig:.4f}; "
                  f"the pooled AUC-ROC mixes these two in a ratio set by the "
                  f"split's grade composition.")

    results_path = output_dir / "phase1_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 1 EVALUATION COMPLETE")
    print("=" * 80)
    print("\n💡 Interpretation:")
    print(f"  Phase 1 embeddings achieve {accuracy*100:.1f}% accuracy")
    print("  This is the baseline before Phase 2 fine-tuning")
    print("  Compare this with Phase 2 results to measure fine-tuning impact")


if __name__ == "__main__":
    main()
