#!/usr/bin/env python3
"""
Baseline Evaluation: Text Encoder Only (No Contrastive Model)
Evaluates raw SentenceTransformer embeddings to establish baseline performance.
"""

import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from contrastive_learning.evaluator import ContrastiveEvaluator, EvaluationConfig
from contrastive_learning.data_structures import TrainingConfig


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
                    # Extract text content from resume and job
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


def main():
    print("=" * 80)
    print("BASELINE EVALUATION: TEXT ENCODER ONLY")
    print("=" * 80)
    print("\nThis evaluates the raw SentenceTransformer without contrastive training")
    print("to establish a baseline for comparison.\n")

    # Use the augmented training dataset (same as Phase 2 used)
    dataset_path = "preprocess/augmented_enriched_data_training_updated_with_uri.jsonl"
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    # Load config for parameters
    config_path = "config/phase2_finetuning_config.json"
    print(f"Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    training_config = TrainingConfig(**config_dict)

    # Load text encoder
    print(f"\nLoading text encoder: {training_config.text_encoder_model}")
    text_encoder = SentenceTransformer(training_config.text_encoder_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device)
    print(f"Device: {device}")

    # Load test data
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = JSONLDataset(dataset_path)

    data_loader = TorchDataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: {
            'resume': [item['resume'] for item in x],
            'job': [item['job'] for item in x],
            'label': torch.tensor([item['label'] for item in x])
        }
    )

    print(f"Dataset samples: {len(dataset)}")
    print(f"Dataset batches: {len(data_loader)}")

    # Create evaluation config with baseline mode
    eval_config = EvaluationConfig(
        metrics=["contrastive_accuracy", "precision_at_k", "recall_at_k"],
        k_values=[1, 5, 10],
        batch_size=training_config.batch_size,
        device=str(device),
        temperature=training_config.temperature,
        similarity_threshold=None,  # Auto-calibrate
        use_text_encoder_baseline=True  # KEY: Use baseline mode
    )

    print("\n" + "=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Mode: TEXT ENCODER BASELINE (no contrastive model)")
    print(f"Temperature: {eval_config.temperature}")
    print(f"Similarity Threshold: Auto-calibrated")
    print(f"Batch Size: {eval_config.batch_size}")

    # Create evaluator
    evaluator = ContrastiveEvaluator(
        config=eval_config,
        text_encoder=text_encoder
    )

    # Run evaluation (model=None since we're using baseline)
    print("\n" + "=" * 80)
    print("RUNNING BASELINE EVALUATION")
    print("=" * 80)

    output_dir = "baseline_evaluation"
    Path(output_dir).mkdir(exist_ok=True)

    results = evaluator.evaluate_model(
        model=None,  # Not used in baseline mode
        data_loader=data_loader,
        output_dir=output_dir
    )

    # Display results
    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)

    if results and results.metrics:
        print("\nKey Metrics:")
        for metric, value in sorted(results.metrics.items()):
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

        # Save results
        results_path = Path(output_dir) / "baseline_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'metrics': results.metrics,
                'config': {
                    'text_encoder': training_config.text_encoder_model,
                    'temperature': eval_config.temperature,
                    'use_baseline': True
                }
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    else:
        print("\nERROR: No results returned from evaluation")

    print("\n" + "=" * 80)
    print("COMPARISON NOTE")
    print("=" * 80)
    print("\nTo compare with trained model performance:")
    print("  - Baseline (this run): Raw text encoder embeddings")
    print("  - Phase 2 trained: 54.74% accuracy")
    print("\nThe difference shows the value added by contrastive learning.")
    print("=" * 80)


if __name__ == "__main__":
    main()
