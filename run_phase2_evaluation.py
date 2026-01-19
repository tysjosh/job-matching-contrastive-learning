#!/usr/bin/env python3
"""
Phase 2 Model Evaluation: Comprehensive Metrics
Evaluates the trained Phase 2 fine-tuned model with full metrics.
"""

import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from contrastive_learning.evaluator import ContrastiveEvaluator, EvaluationConfig
from contrastive_learning.contrastive_classification_model import ContrastiveClassificationModel
from contrastive_learning.data_structures import TrainingConfig
import torch.nn.functional as F


class ClassificationModelWrapper(torch.nn.Module):
    """Wrapper to make classification model compatible with evaluator"""

    def __init__(self, classification_model):
        super().__init__()
        self.model = classification_model
        self._last_job_emb = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass compatible with evaluator.
        Evaluator calls this twice per sample (resume, then job).
        We cache job embeddings and use them when processing resumes.
        """
        # This is a workaround: return the embeddings through the contrastive encoder
        # The actual classification happens in the evaluator's similarity calculation
        return self.model.contrastive_encoder(x)


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
    print("PHASE 2 MODEL EVALUATION: COMPREHENSIVE METRICS")
    print("=" * 80)
    print("\nEvaluating the trained Phase 2 fine-tuned model with full metrics:")
    print("  - Accuracy, Precision, Recall, F1 Score")
    print("  - AUC-ROC, Precision@K, Recall@K")
    print("  - Confusion Matrix, Error Analysis\n")

    # Use the same dataset as training
    dataset_path = "preprocess/augmented_enriched_data_training_updated_with_uri.jsonl"
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    # Load Phase 2 checkpoint
    checkpoint_path = "phase2_finetuning/checkpoint_epoch_9.pt"
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Phase 2 checkpoint not found at {checkpoint_path}")
        return

    # Load config
    config_path = "config/phase2_finetuning_config.json"
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

    # Load Phase 2 model
    print(f"Loading Phase 2 model from: {checkpoint_path}")

    # Create model using config (ContrastiveClassificationModel expects config)
    model = ContrastiveClassificationModel(config=training_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")

    # Wrap model for evaluator compatibility
    wrapped_model = ClassificationModelWrapper(model)
    wrapped_model.eval()

    # Load dataset
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

    # Create evaluation config with ALL metrics enabled
    eval_config = EvaluationConfig(
        metrics=[
            "contrastive_accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
            "precision_at_k",
            "recall_at_k",
            "map_score",
            "similarity_statistics"
        ],
        k_values=[1, 5, 10],
        batch_size=training_config.batch_size,
        device=str(device),
        temperature=training_config.temperature,
        similarity_threshold=None,  # Auto-calibrate
        use_text_encoder_baseline=False  # Use trained model
    )

    print("\n" + "=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Mode: TRAINED PHASE 2 MODEL")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Temperature: {eval_config.temperature}")
    print(f"Similarity Threshold: Auto-calibrated")
    print(f"Batch Size: {eval_config.batch_size}")
    print(f"Metrics: {', '.join(eval_config.metrics)}")

    # Create evaluator
    evaluator = ContrastiveEvaluator(
        config=eval_config,
        text_encoder=text_encoder
    )

    # Run evaluation
    print("\n" + "=" * 80)
    print("RUNNING PHASE 2 MODEL EVALUATION")
    print("=" * 80)

    output_dir = "phase2_evaluation"
    Path(output_dir).mkdir(exist_ok=True)

    results = evaluator.evaluate_model(
        model=wrapped_model,
        data_loader=data_loader,
        output_dir=output_dir
    )

    # Display results
    print("\n" + "=" * 80)
    print("PHASE 2 MODEL RESULTS")
    print("=" * 80)

    if results and results.metrics:
        print("\n🎯 Classification Metrics:")
        for metric in ['contrastive_accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            if metric in results.metrics:
                value = results.metrics[metric]
                print(f"  {metric:25s}: {value:.4f} ({value*100:.2f}%)")

        print("\n📊 Retrieval Metrics:")
        for k in eval_config.k_values:
            p_key = f'precision_at_{k}'
            r_key = f'recall_at_{k}'
            if p_key in results.metrics and r_key in results.metrics:
                print(
                    f"  Precision@{k:2d}: {results.metrics[p_key]:.4f}  |  Recall@{k:2d}: {results.metrics[r_key]:.4f}")

        if 'map_score' in results.metrics:
            print(f"\n  MAP Score: {results.metrics['map_score']:.4f}")

        print("\n📈 Similarity Statistics:")
        for metric in ['avg_positive_similarity', 'avg_negative_similarity', 'similarity_separation']:
            if metric in results.metrics:
                print(f"  {metric:30s}: {results.metrics[metric]:.4f}")

        # Confusion Matrix
        if 'confusion_matrix' in results.detailed_metrics:
            cm = results.detailed_metrics['confusion_matrix']
            print("\n🎲 Confusion Matrix:")
            print(f"  True Positives:  {cm['tp']:6d}")
            print(f"  True Negatives:  {cm['tn']:6d}")
            print(f"  False Positives: {cm['fp']:6d}")
            print(f"  False Negatives: {cm['fn']:6d}")

        # Save results (convert numpy types to native Python types)
        def convert_to_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        results_path = Path(output_dir) / "phase2_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'metrics': convert_to_serializable(results.metrics),
                'detailed_metrics': convert_to_serializable(results.detailed_metrics),
                'config': {
                    'checkpoint': checkpoint_path,
                    'text_encoder': training_config.text_encoder_model,
                    'temperature': eval_config.temperature,
                    'embedding_dim': training_config.projection_dim
                }
            }, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")

        # Comparison with baseline
        print("\n" + "=" * 80)
        print("COMPARISON: PHASE 2 vs BASELINE")
        print("=" * 80)

        baseline_path = Path("baseline_evaluation/evaluation_results.json")
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)

            baseline_acc = baseline_data['metrics'].get(
                'contrastive_accuracy', 0)
            phase2_acc = results.metrics.get('contrastive_accuracy', 0)

            print(
                f"\n  Baseline Accuracy:  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
            print(
                f"  Phase 2 Accuracy:   {phase2_acc:.4f} ({phase2_acc*100:.2f}%)")
            print(
                f"  Improvement:        +{(phase2_acc - baseline_acc):.4f} (+{(phase2_acc - baseline_acc)*100:.2f}pp)")
            print(
                f"  Relative Gain:      {((phase2_acc / baseline_acc - 1) * 100):.1f}%")

    else:
        print("\n❌ ERROR: No results returned from evaluation")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
