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
from typing import Dict, Tuple

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


def content_to_text(content: Dict, content_type: str) -> str:
    """Convert structured content to text for embedding"""
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
    
    with torch.no_grad():
        for batch in data_loader:
            resumes = batch['resume']
            jobs = batch['job']
            labels = batch['label'].numpy()
            
            # Convert to text and get base embeddings from SentenceTransformer
            resume_texts = [content_to_text(r, 'resume') for r in resumes]
            job_texts = [content_to_text(j, 'job') for j in jobs]
            
            resume_base = text_encoder.encode(resume_texts, convert_to_tensor=True).to(device)
            job_base = text_encoder.encode(job_texts, convert_to_tensor=True).to(device)
            
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
    
    # Convert similarities to probabilities (map from [-1, 1] to [0, 1])
    probabilities = (similarities + 1) / 2
    
    # Default threshold of 0.5 similarity (0.0 in cosine space)
    predictions = (similarities > 0.0).astype(int)
    
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
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = TrainingConfig(**config_dict)
    
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
            'auc_roc': float(auc_roc)
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
