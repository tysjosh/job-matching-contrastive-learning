#!/usr/bin/env python3
"""
Phase 2 Classification Model Evaluation
Evaluates the trained Phase 2 classification model with proper metrics.
"""

import json
import torch
import numpy as np
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from collections import defaultdict

from contrastive_learning.contrastive_classification_model import ContrastiveClassificationModel
from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.structured_features import StructuredFeatureExtractor


class JSONLDataset(Dataset):
    """Simple dataset for loading JSONL files"""

    def __init__(self, file_path):
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    resume_text = self._extract_resume_text(data['resume'])
                    job_text = self._extract_job_text(data['job'])

                    # Get label
                    if 'metadata' in data and 'label' in data['metadata']:
                        label = int(data['metadata']['label'])
                    elif 'label' in data:
                        if isinstance(data['label'], str):
                            label_map = {'positive': 1, 'negative': 0}
                            label = label_map.get(data['label'].lower(), 1)
                        else:
                            label = int(data['label'])
                    else:
                        label = 1

                    self.samples.append({
                        'resume': resume_text,
                        'job': job_text,
                        'resume_raw': data['resume'],  # Keep raw data for structured features
                        'job_raw': data['job'],
                        'label': label
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    continue

    def _extract_resume_text(self, resume):
        """Extract text from resume dict"""
        parts = []
        if isinstance(resume, dict):
            if 'role' in resume:
                parts.append(str(resume['role']))
            if 'experience' in resume:
                if isinstance(resume['experience'], list):
                    for exp in resume['experience']:
                        if isinstance(exp, dict) and 'description' in exp:
                            parts.append(str(exp['description']))
                elif isinstance(resume['experience'], str):
                    parts.append(resume['experience'])
            if 'skills' in resume:
                skills = resume['skills']
                if isinstance(skills, list):
                    skill_names = [s.get('name', str(s)) if isinstance(
                        s, dict) else str(s) for s in skills]
                    parts.append(', '.join(skill_names))
                elif isinstance(skills, str):
                    parts.append(skills)
        return ' '.join(parts)

    def _extract_job_text(self, job):
        """Extract text from job dict"""
        parts = []
        if isinstance(job, dict):
            if 'title' in job:
                parts.append(job['title'])
            if 'description' in job:
                desc = job['description']
                if isinstance(desc, dict) and 'original' in desc:
                    parts.append(desc['original'])
                elif isinstance(desc, str):
                    parts.append(desc)
        return ' '.join(parts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate_classification_model(model, text_encoder, data_loader, device, feature_extractor=None):
    """
    Evaluate the classification model using its actual outputs.

    Returns predictions and true labels.
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    use_structured_features = model.uses_structured_features() and feature_extractor is not None

    with torch.no_grad():
        for batch in data_loader:
            # Get texts
            resume_texts = batch['resume']
            job_texts = batch['job']
            labels = batch['label'].numpy()

            # Encode texts - clone to avoid inference tensor issues
            resume_embeddings = text_encoder.encode(
                resume_texts, convert_to_tensor=True).clone()
            job_embeddings = text_encoder.encode(
                job_texts, convert_to_tensor=True).clone()

            # Move to device
            resume_embeddings = resume_embeddings.to(device)
            job_embeddings = job_embeddings.to(device)

            # Extract structured features if needed
            if use_structured_features:
                resume_exp_indices = []
                resume_num_features = []
                job_exp_indices = []
                job_num_features = []
                
                for i in range(len(resume_texts)):
                    resume_raw = batch['resume_raw'][i] if isinstance(batch['resume_raw'], list) else batch['resume_raw']
                    job_raw = batch['job_raw'][i] if isinstance(batch['job_raw'], list) else batch['job_raw']
                    
                    # Handle dict conversion if needed
                    if isinstance(resume_raw, str):
                        resume_raw = json.loads(resume_raw) if resume_raw.startswith('{') else {'text': resume_raw}
                    if isinstance(job_raw, str):
                        job_raw = json.loads(job_raw) if job_raw.startswith('{') else {'text': job_raw}
                    
                    r_features = feature_extractor.extract_features(resume_raw, 'resume')
                    j_features = feature_extractor.extract_features(job_raw, 'job')
                    
                    # Split into experience level (one-hot) and numerical
                    r_exp_onehot = r_features[:10]
                    r_numerical = r_features[10:]
                    j_exp_onehot = j_features[:10]
                    j_numerical = j_features[10:]
                    
                    resume_exp_indices.append(r_exp_onehot.argmax().item())
                    resume_num_features.append(r_numerical)
                    job_exp_indices.append(j_exp_onehot.argmax().item())
                    job_num_features.append(j_numerical)
                
                resume_exp_batch = torch.tensor(resume_exp_indices, dtype=torch.long, device=device)
                resume_num_batch = torch.stack(resume_num_features).to(device)
                job_exp_batch = torch.tensor(job_exp_indices, dtype=torch.long, device=device)
                job_num_batch = torch.stack(job_num_features).to(device)
                
                # Get classification probabilities with structured features
                logits = model(resume_embeddings, job_embeddings,
                              resume_exp_batch, resume_num_batch,
                              job_exp_batch, job_num_batch)
            else:
                # Get classification probabilities without structured features
                logits = model(resume_embeddings, job_embeddings)
            
            # Model already applies sigmoid, so logits are probabilities
            probabilities = logits.cpu().numpy()

            # Convert to binary predictions (threshold = 0.5)
            predictions = (probabilities > 0.5).astype(int)

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels)

    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)


def calculate_dcg(relevance_scores, k=None):
    """Calculate Discounted Cumulative Gain"""
    if k is not None:
        relevance_scores = relevance_scores[:k]

    if len(relevance_scores) == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i+1)) for i from 1 to k
    dcg = relevance_scores[0]  # First position doesn't get discounted
    for i in range(1, len(relevance_scores)):
        # i+2 because we're 0-indexed
        dcg += relevance_scores[i] / np.log2(i + 2)
    return dcg


def calculate_ndcg(relevance_scores, k=None):
    """Calculate Normalized Discounted Cumulative Gain"""
    dcg = calculate_dcg(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(np.array(ideal_scores), k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_average_precision(relevance_scores):
    """Calculate Average Precision for a single query"""
    if len(relevance_scores) == 0 or relevance_scores.sum() == 0:
        return 0.0

    # AP = sum(P(k) * rel(k)) / num_relevant
    num_relevant = relevance_scores.sum()
    precisions = []

    for k in range(len(relevance_scores)):
        if relevance_scores[k] == 1:
            # Precision at k = number of relevant items in top k / k
            precision_at_k = relevance_scores[:k+1].sum() / (k + 1)
            precisions.append(precision_at_k)

    if len(precisions) == 0:
        return 0.0

    return sum(precisions) / num_relevant


def evaluate_job_ranking_global_pool(model, text_encoder, dataset, device, feature_extractor=None, max_candidates=50):
    """
    Evaluate job ranking using a global job pool approach.
    
    For each resume, ranks all unique jobs in the dataset as candidates.
    The matching job gets label=1, all others get label=0.
    
    Args:
        model: The classification model
        text_encoder: Text encoder for embeddings
        dataset: Dataset with samples
        device: Computation device
        feature_extractor: Optional structured feature extractor
        max_candidates: Maximum number of candidate jobs per query (for efficiency)
    """
    import random
    
    model.eval()
    
    # Collect all unique jobs
    all_jobs = {}
    for sample in dataset.samples:
        job_text = sample['job']
        if job_text not in all_jobs:
            all_jobs[job_text] = {
                'job': job_text,
                'job_raw': sample.get('job_raw', {})
            }
    
    unique_jobs = list(all_jobs.values())
    print(f"Total unique jobs in dataset: {len(unique_jobs)}")
    
    # Collect unique resume queries with their matching jobs
    resume_queries = {}
    for sample in dataset.samples:
        resume_text = sample['resume']
        if resume_text not in resume_queries:
            resume_queries[resume_text] = {
                'resume': resume_text,
                'resume_raw': sample.get('resume_raw', {}),
                'matching_job': sample['job'],
                'label': sample['label']
            }
    
    # Only evaluate resumes that have positive matches (label=1)
    positive_queries = {k: v for k, v in resume_queries.items() if v['label'] == 1}
    print(f"Resumes with positive matches: {len(positive_queries)}")
    
    if len(positive_queries) == 0:
        print("⚠️  No positive resume-job pairs found!")
        return None
    
    use_structured_features = model.uses_structured_features() and feature_extractor is not None
    
    all_aps = []
    all_ndcgs = []
    all_ndcgs_at_10 = []
    all_mrrs = []
    
    with torch.no_grad():
        for query_idx, (resume_text, query_info) in enumerate(positive_queries.items()):
            matching_job = query_info['matching_job']
            
            # Build candidate pool: matching job + sample of other jobs
            other_jobs = [j for j in unique_jobs if j['job'] != matching_job]
            
            # Sample negative candidates if we have too many
            if len(other_jobs) > max_candidates - 1:
                negative_candidates = random.sample(other_jobs, max_candidates - 1)
            else:
                negative_candidates = other_jobs
            
            # Include the matching job
            matching_job_info = all_jobs[matching_job]
            candidates = [matching_job_info] + negative_candidates
            
            # Shuffle to avoid position bias
            random.shuffle(candidates)
            
            # Create labels (1 for matching job, 0 for others)
            labels = [1 if c['job'] == matching_job else 0 for c in candidates]
            
            # Get resume embedding
            query_embedding = text_encoder.encode(
                [resume_text], convert_to_tensor=True).clone().to(device)
            
            # Score each candidate job
            scores = []
            for i, candidate in enumerate(candidates):
                job_embedding = text_encoder.encode(
                    [candidate['job']], convert_to_tensor=True).clone().to(device)
                
                if use_structured_features:
                    # Extract structured features for resume
                    r_raw = query_info.get('resume_raw', {})
                    if isinstance(r_raw, str):
                        r_raw = json.loads(r_raw) if r_raw.startswith('{') else {}
                    r_features = feature_extractor.extract_features(r_raw, 'resume')
                    r_exp_idx = torch.tensor([r_features[:10].argmax().item()], dtype=torch.long, device=device)
                    r_num = r_features[10:].unsqueeze(0).to(device)
                    
                    # Extract structured features for job
                    j_raw = candidate.get('job_raw', {})
                    if isinstance(j_raw, str):
                        j_raw = json.loads(j_raw) if j_raw.startswith('{') else {}
                    j_features = feature_extractor.extract_features(j_raw, 'job')
                    j_exp_idx = torch.tensor([j_features[:10].argmax().item()], dtype=torch.long, device=device)
                    j_num = j_features[10:].unsqueeze(0).to(device)
                    
                    logit = model(query_embedding, job_embedding,
                                 r_exp_idx, r_num, j_exp_idx, j_num)
                else:
                    logit = model(query_embedding, job_embedding)
                
                score = logit.item()
                scores.append(score)
            
            # Sort by scores (descending)
            true_labels = np.array(labels)
            scores = np.array(scores)
            sorted_indices = np.argsort(-scores)
            sorted_labels = true_labels[sorted_indices]
            
            # Calculate metrics
            ap = calculate_average_precision(sorted_labels)
            ndcg = calculate_ndcg(sorted_labels)
            ndcg_at_10 = calculate_ndcg(sorted_labels, k=10)
            
            # Calculate MRR (Mean Reciprocal Rank)
            # Find the rank of the first relevant item
            rank_of_positive = np.where(sorted_labels == 1)[0]
            if len(rank_of_positive) > 0:
                mrr = 1.0 / (rank_of_positive[0] + 1)
            else:
                mrr = 0.0
            
            all_aps.append(ap)
            all_ndcgs.append(ndcg)
            all_ndcgs_at_10.append(ndcg_at_10)
            all_mrrs.append(mrr)
    
    # Calculate mean metrics
    map_score = np.mean(all_aps)
    mean_ndcg = np.mean(all_ndcgs)
    mean_ndcg_at_10 = np.mean(all_ndcgs_at_10)
    mean_mrr = np.mean(all_mrrs)
    
    print(f"\n🎯 Job Ranking Metrics (Global Pool):")
    print(f"  MAP (Mean Average Precision): {map_score:.4f}")
    print(f"  MRR (Mean Reciprocal Rank):   {mean_mrr:.4f}")
    print(f"  NDCG:                         {mean_ndcg:.4f}")
    print(f"  NDCG@10:                      {mean_ndcg_at_10:.4f}")
    print(f"  Number of queries evaluated:  {len(all_aps)}")
    print(f"  Avg candidates per query:     {max_candidates}")

    return {
        'map': float(map_score),
        'mrr': float(mean_mrr),
        'ndcg': float(mean_ndcg),
        'ndcg_at_10': float(mean_ndcg_at_10),
        'num_queries': len(all_aps),
        'avg_candidates_per_query': max_candidates,
        'evaluation_method': 'global_job_pool',
        'ap_scores': [float(ap) for ap in all_aps],
        'mrr_scores': [float(mrr) for mrr in all_mrrs],
        'ndcg_scores': [float(ndcg) for ndcg in all_ndcgs],
        'ndcg_at_10_scores': [float(ndcg) for ndcg in all_ndcgs_at_10]
    }


def evaluate_ranking(model, text_encoder, dataset, device, mode='job', feature_extractor=None):
    """
    Evaluate ranking performance.

    Args:
        mode: 'job' for job ranking (given resume, rank jobs)
              'resume' for resume ranking (given job, rank resumes)
        feature_extractor: Optional structured feature extractor
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {'JOB' if mode == 'job' else 'RESUME'} RANKING")
    print(f"{'='*80}")

    model.eval()

    # Group samples by query (resume for job ranking, job for resume ranking)
    query_groups = defaultdict(list)

    for idx, sample in enumerate(dataset.samples):
        if mode == 'job':
            # Group by resume (each resume is a query, jobs are candidates)
            query_key = sample['resume']
        else:
            # Group by job (each job is a query, resumes are candidates)
            query_key = sample['job']

        query_groups[query_key].append({
            'index': idx,
            'resume': sample['resume'],
            'job': sample['job'],
            'resume_raw': sample.get('resume_raw', {}),
            'job_raw': sample.get('job_raw', {}),
            'label': sample['label']
        })

    # Filter groups that have both positive and negative examples
    valid_queries = {k: v for k, v in query_groups.items()
                     if any(s['label'] == 1 for s in v) and any(s['label'] == 0 for s in v)}

    print(f"Total unique queries: {len(query_groups)}")
    print(
        f"Valid queries (with both positive and negative): {len(valid_queries)}")

    if len(valid_queries) == 0:
        print("⚠️  No valid queries found for standard ranking evaluation!")
        
        # For job ranking, try global job pool approach
        if mode == 'job':
            print("Attempting global job pool ranking evaluation...")
            return evaluate_job_ranking_global_pool(
                model, text_encoder, dataset, device, feature_extractor
            )
        return None
    
    use_structured_features = model.uses_structured_features() and feature_extractor is not None

    # Calculate ranking metrics for each query
    all_aps = []
    all_ndcgs = []
    all_ndcgs_at_10 = []

    with torch.no_grad():
        for query_idx, (query_text, candidates) in enumerate(valid_queries.items()):
            # Get all candidate texts and labels
            if mode == 'job':
                # Query is resume, candidates are jobs
                query_embedding = text_encoder.encode(
                    [query_text], convert_to_tensor=True).clone().to(device)
                candidate_texts = [c['job'] for c in candidates]
                candidate_embeddings = text_encoder.encode(
                    candidate_texts, convert_to_tensor=True).clone().to(device)

                # Get scores
                scores = []
                for i, cand_emb in enumerate(candidate_embeddings):
                    if use_structured_features:
                        # Extract structured features for query (resume)
                        query_raw = candidates[0].get('resume_raw', {})
                        if isinstance(query_raw, str):
                            query_raw = json.loads(query_raw) if query_raw.startswith('{') else {}
                        r_features = feature_extractor.extract_features(query_raw, 'resume')
                        r_exp_idx = torch.tensor([r_features[:10].argmax().item()], dtype=torch.long, device=device)
                        r_num = r_features[10:].unsqueeze(0).to(device)
                        
                        # Extract structured features for candidate (job)
                        cand_raw = candidates[i].get('job_raw', {})
                        if isinstance(cand_raw, str):
                            cand_raw = json.loads(cand_raw) if cand_raw.startswith('{') else {}
                        j_features = feature_extractor.extract_features(cand_raw, 'job')
                        j_exp_idx = torch.tensor([j_features[:10].argmax().item()], dtype=torch.long, device=device)
                        j_num = j_features[10:].unsqueeze(0).to(device)
                        
                        logit = model(query_embedding, cand_emb.unsqueeze(0),
                                     r_exp_idx, r_num, j_exp_idx, j_num)
                    else:
                        logit = model(query_embedding, cand_emb.unsqueeze(0))
                    score = logit.item()  # Model already applies sigmoid
                    scores.append(score)
            else:
                # Query is job, candidates are resumes
                query_embedding = text_encoder.encode(
                    [query_text], convert_to_tensor=True).clone().to(device)
                candidate_texts = [c['resume'] for c in candidates]
                candidate_embeddings = text_encoder.encode(
                    candidate_texts, convert_to_tensor=True).clone().to(device)

                # Get scores
                scores = []
                for i, cand_emb in enumerate(candidate_embeddings):
                    if use_structured_features:
                        # Extract structured features for candidate (resume)
                        cand_raw = candidates[i].get('resume_raw', {})
                        if isinstance(cand_raw, str):
                            cand_raw = json.loads(cand_raw) if cand_raw.startswith('{') else {}
                        r_features = feature_extractor.extract_features(cand_raw, 'resume')
                        r_exp_idx = torch.tensor([r_features[:10].argmax().item()], dtype=torch.long, device=device)
                        r_num = r_features[10:].unsqueeze(0).to(device)
                        
                        # Extract structured features for query (job)
                        query_raw = candidates[0].get('job_raw', {})
                        if isinstance(query_raw, str):
                            query_raw = json.loads(query_raw) if query_raw.startswith('{') else {}
                        j_features = feature_extractor.extract_features(query_raw, 'job')
                        j_exp_idx = torch.tensor([j_features[:10].argmax().item()], dtype=torch.long, device=device)
                        j_num = j_features[10:].unsqueeze(0).to(device)
                        
                        logit = model(cand_emb.unsqueeze(0), query_embedding,
                                     r_exp_idx, r_num, j_exp_idx, j_num)
                    else:
                        logit = model(cand_emb.unsqueeze(0), query_embedding)
                    score = logit.item()  # Model already applies sigmoid
                    scores.append(score)

            # Get true labels
            true_labels = np.array([c['label'] for c in candidates])
            scores = np.array(scores)

            # Sort by scores (descending)
            sorted_indices = np.argsort(-scores)
            sorted_labels = true_labels[sorted_indices]

            # Calculate metrics
            ap = calculate_average_precision(sorted_labels)
            ndcg = calculate_ndcg(sorted_labels)
            ndcg_at_10 = calculate_ndcg(sorted_labels, k=10)

            all_aps.append(ap)
            all_ndcgs.append(ndcg)
            all_ndcgs_at_10.append(ndcg_at_10)

    # Calculate mean metrics
    map_score = np.mean(all_aps)
    mean_ndcg = np.mean(all_ndcgs)
    mean_ndcg_at_10 = np.mean(all_ndcgs_at_10)

    print(f"\n🎯 Ranking Metrics:")
    print(f"  MAP (Mean Average Precision): {map_score:.4f}")
    print(f"  NDCG:                         {mean_ndcg:.4f}")
    print(f"  NDCG@10:                      {mean_ndcg_at_10:.4f}")
    print(f"  Number of queries evaluated:  {len(all_aps)}")

    return {
        'map': float(map_score),
        'ndcg': float(mean_ndcg),
        'ndcg_at_10': float(mean_ndcg_at_10),
        'num_queries': len(all_aps),
        'ap_scores': [float(ap) for ap in all_aps],
        'ndcg_scores': [float(ndcg) for ndcg in all_ndcgs],
        'ndcg_at_10_scores': [float(ndcg) for ndcg in all_ndcgs_at_10]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 Classification Model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to test dataset (JSONL)")
    parser.add_argument("--validation-dataset", type=str, default=None, help="Path to validation dataset for threshold tuning (JSONL)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Phase 2 model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to training config JSON")
    parser.add_argument("--output-dir", type=str, default="phase2_classification_evaluation", help="Output directory for results")
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 2 CLASSIFICATION MODEL EVALUATION")
    print("=" * 80)
    print("\nEvaluating the full Phase 2 classification model:")
    print("  - Uses classification head outputs (not just embeddings)")
    print("  - Calculates Accuracy, Precision, Recall, F1, AUC-ROC")
    print("  - Confusion Matrix and detailed metrics")
    if args.validation_dataset:
        print("  - Tunes threshold on VALIDATION set")
        print("  - Reports metrics on TEST set\n")
    else:
        print("  - ⚠️  No validation set - tuning threshold on test set (data leakage!)\n")

    # Paths
    dataset_path = args.dataset
    validation_dataset_path = args.validation_dataset
    checkpoint_path = args.checkpoint
    config_path = args.config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check files exist
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    # Load config
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    training_config = TrainingConfig(**config_dict)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load text encoder
    print(f"\nLoading text encoder: {training_config.text_encoder_model}")
    text_encoder = SentenceTransformer(training_config.text_encoder_model)
    text_encoder = text_encoder.to(device)

    # Load Phase 2 classification model
    print(f"Loading Phase 2 model from: {checkpoint_path}")
    model = ContrastiveClassificationModel(config=training_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")
    
    # Initialize structured feature extractor if model uses it
    feature_extractor = None
    if model.uses_structured_features():
        feature_extractor = StructuredFeatureExtractor()
        print(f"✓ Structured feature extractor initialized")

    # Load test dataset
    print(f"\nLoading test dataset from: {dataset_path}")
    dataset = JSONLDataset(dataset_path)

    data_loader = TorchDataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: {
            'resume': [item['resume'] for item in x],
            'job': [item['job'] for item in x],
            'resume_raw': [item['resume_raw'] for item in x],
            'job_raw': [item['job_raw'] for item in x],
            'label': torch.tensor([item['label'] for item in x])
        }
    )

    print(f"Dataset samples: {len(dataset)}")
    print(f"Dataset batches: {len(data_loader)}")

    # Run evaluation
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    predictions, probabilities, true_labels = evaluate_classification_model(
        model, text_encoder, data_loader, device, feature_extractor
    )

    # Calculate metrics
    print("\n" + "=" * 80)
    print("PHASE 2 CLASSIFICATION MODEL RESULTS")
    print("=" * 80)

    # Find optimal threshold on validation set (if provided) or test set
    if validation_dataset_path and Path(validation_dataset_path).exists():
        print(f"\n🔍 Finding optimal threshold on VALIDATION set...")
        val_dataset = JSONLDataset(validation_dataset_path)
        val_data_loader = TorchDataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                'resume': [item['resume'] for item in x],
                'job': [item['job'] for item in x],
                'resume_raw': [item['resume_raw'] for item in x],
                'job_raw': [item['job_raw'] for item in x],
                'label': torch.tensor([item['label'] for item in x])
            }
        )
        
        _, val_probabilities, val_true_labels = evaluate_classification_model(
            model, text_encoder, val_data_loader, device, feature_extractor
        )
        
        thresholds = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in thresholds:
            preds_at_threshold = (val_probabilities > threshold).astype(int)
            f1_at_threshold = f1_score(
                val_true_labels, preds_at_threshold, zero_division=0)
            if f1_at_threshold > best_f1:
                best_f1 = f1_at_threshold
                best_threshold = threshold
        
        print(f"  Validation set size: {len(val_dataset)} samples")
        print(f"  Optimal threshold from validation: {best_threshold:.4f} (F1={best_f1:.4f})")
        print(f"\n📊 Applying threshold to TEST set...")
    else:
        print(f"\n⚠️  No validation set provided - finding threshold on TEST set (data leakage!)")
        thresholds = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in thresholds:
            preds_at_threshold = (probabilities > threshold).astype(int)
            f1_at_threshold = f1_score(
                true_labels, preds_at_threshold, zero_division=0)
            if f1_at_threshold > best_f1:
                best_f1 = f1_at_threshold
                best_threshold = threshold

    print(f"\n🔍 Threshold Analysis:")
    print(
        f"  Default threshold (0.5): F1 = {f1_score(true_labels, (probabilities > 0.5).astype(int)):.4f}")
    print(f"  Optimal threshold: {best_threshold:.4f}, F1 = {best_f1:.4f}")

    # Use optimal threshold
    predictions_optimal = (probabilities > best_threshold).astype(int)

    accuracy = accuracy_score(true_labels, predictions_optimal)
    precision = precision_score(
        true_labels, predictions_optimal, zero_division=0)
    recall = recall_score(true_labels, predictions_optimal, zero_division=0)
    f1 = f1_score(true_labels, predictions_optimal, zero_division=0)

    # Also calculate metrics with default 0.5 threshold
    accuracy_default = accuracy_score(true_labels, predictions)
    precision_default = precision_score(
        true_labels, predictions, zero_division=0)
    recall_default = recall_score(true_labels, predictions, zero_division=0)
    f1_default = f1_score(true_labels, predictions, zero_division=0)

    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        auc_roc = 0.0

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions_optimal)
    tn, fp, fn, tp = cm.ravel()

    print("\n🎯 Classification Metrics (Optimal Threshold = {:.4f}):".format(
        best_threshold))
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC-ROC:   {auc_roc:.4f} ({auc_roc*100:.2f}%)")

    print("\n📋 With Default Threshold (0.5):")
    cm_default = confusion_matrix(true_labels, predictions)
    tn_d, fp_d, fn_d, tp_d = cm_default.ravel()
    print(f"  Accuracy:  {accuracy_default:.4f} ({accuracy_default*100:.2f}%)")
    print(
        f"  Precision: {precision_default:.4f} ({precision_default*100:.2f}%)")
    print(f"  Recall:    {recall_default:.4f} ({recall_default*100:.2f}%)")
    print(f"  F1 Score:  {f1_default:.4f} ({f1_default*100:.2f}%)")
    print(f"  [TP={tp_d}, TN={tn_d}, FP={fp_d}, FN={fn_d}]")

    print("\n🎲 Confusion Matrix:")
    print(f"  True Positives:  {tp:6d}")
    print(f"  True Negatives:  {tn:6d}")
    print(f"  False Positives: {fp:6d}")
    print(f"  False Negatives: {fn:6d}")

    print("\n📊 Additional Metrics:")
    print(f"  True Positive Rate (Recall):  {recall:.4f}")
    print(f"  True Negative Rate:           {tn/(tn+fp):.4f}")
    print(f"  False Positive Rate:          {fp/(fp+tn):.4f}")
    print(f"  False Negative Rate:          {fn/(fn+tp):.4f}")

    # Probability statistics
    pos_probs = probabilities[true_labels == 1]
    neg_probs = probabilities[true_labels == 0]

    print("\n📈 Probability Statistics:")
    print(
        f"  Positive samples avg prob: {pos_probs.mean():.4f} ± {pos_probs.std():.4f}")
    print(
        f"  Negative samples avg prob: {neg_probs.mean():.4f} ± {neg_probs.std():.4f}")
    print(
        f"  Separation:                {pos_probs.mean() - neg_probs.mean():.4f}")

    # Save results
    results = {
        'metrics_optimal_threshold': {
            'threshold': float(best_threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc)
        },
        'metrics_default_threshold': {
            'threshold': 0.5,
            'accuracy': float(accuracy_default),
            'precision': float(precision_default),
            'recall': float(recall_default),
            'f1_score': float(f1_default),
            'auc_roc': float(auc_roc)
        },
        'confusion_matrix_optimal': {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        },
        'confusion_matrix_default': {
            'true_positives': int(tp_d),
            'true_negatives': int(tn_d),
            'false_positives': int(fp_d),
            'false_negatives': int(fn_d)
        },
        'probability_stats': {
            'positive_mean': float(pos_probs.mean()),
            'positive_std': float(pos_probs.std()),
            'negative_mean': float(neg_probs.mean()),
            'negative_std': float(neg_probs.std()),
            'separation': float(pos_probs.mean() - neg_probs.mean())
        },
        'config': {
            'checkpoint': checkpoint_path,
            'dataset': dataset_path,
            'num_samples': len(dataset),
            'batch_size': training_config.batch_size
        }
    }

    results_path = output_dir / "classification_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # ==========================================
    # RANKING EVALUATION
    # ==========================================
    print("\n" + "=" * 80)
    print("RUNNING RANKING EVALUATION")
    print("=" * 80)

    # Job Ranking (given resume, rank relevant jobs)
    job_ranking_results = evaluate_ranking(
        model, text_encoder, dataset, device, mode='job', feature_extractor=feature_extractor
    )

    # Resume Ranking (given job, rank relevant resumes)
    resume_ranking_results = evaluate_ranking(
        model, text_encoder, dataset, device, mode='resume', feature_extractor=feature_extractor
    )

    # Add ranking results to output
    if job_ranking_results:
        results['job_ranking'] = job_ranking_results
    if resume_ranking_results:
        results['resume_ranking'] = resume_ranking_results

    # Save updated results with ranking metrics
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results with ranking metrics saved to: {results_path}")

    # Comparison with baseline and training accuracy
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # Load training results
    training_results_path = Path("phase2_finetuning/fine_tuning_results.json")
    if training_results_path.exists():
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        training_acc = training_results.get('final_accuracy', 0)
        print(
            f"\n  Training Accuracy (epoch 10): {training_acc:.4f} ({training_acc*100:.2f}%)")
        print(
            f"  Evaluation Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(
            f"  Difference:                   {abs(training_acc - accuracy):.4f} ({abs(training_acc - accuracy)*100:.2f}pp)")

    # Load baseline
    baseline_path = Path("baseline_evaluation/evaluation_results.json")
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        baseline_acc = baseline_data['metrics'].get('contrastive_accuracy', 0)

        print(
            f"\n  Baseline Accuracy:  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        print(f"  Phase 2 Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(
            f"  Improvement:        +{(accuracy - baseline_acc):.4f} (+{(accuracy - baseline_acc)*100:.2f}pp)")
        print(
            f"  Relative Gain:      {((accuracy / baseline_acc - 1) * 100):.1f}%")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
