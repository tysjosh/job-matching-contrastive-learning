#!/usr/bin/env python3
"""
Comprehensive evaluation framework for contrastive learning models
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    metrics: List[str] = None
    k_values: List[int] = None
    save_embeddings: bool = True
    save_predictions: bool = True
    generate_visualizations: bool = True
    batch_size: int = 32
    device: str = "cpu"
    temperature: float = 0.2  # Match training temperature for scaled similarity
    # Optional fixed threshold; when None, auto-calibrate from evaluation data.
    similarity_threshold: Optional[float] = None
    # If True, evaluate using raw text encoder embeddings (baseline) instead of the contrastive model.
    use_text_encoder_baseline: bool = False

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "contrastive_accuracy", "triplet_accuracy", "precision_at_k",
                "recall_at_k", "map_score", "embedding_quality"
            ]
        if self.k_values is None:
            self.k_values = [1, 5, 10, 20]


@dataclass
class EvaluationResults:
    """Results from model evaluation"""
    metrics: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    predictions: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[np.ndarray] = None
    visualization_paths: Optional[List[str]] = None


class ContrastiveEvaluator:
    """Comprehensive evaluation for contrastive learning models"""

    def __init__(self, config: EvaluationConfig, text_encoder=None):
        self.config = config
        self.device = torch.device(config.device)
        self.text_encoder = text_encoder

    def evaluate_model(self, model, data_loader, output_dir: str) -> EvaluationResults:
        """Comprehensive model evaluation"""

        logger.info("Starting comprehensive model evaluation")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Check if model is valid unless running text-encoder baseline
        if not self.config.use_text_encoder_baseline:
            if model is None:
                logger.error("Model is None - cannot evaluate")
                return EvaluationResults(
                    metrics={"error": "Model is None"},
                    detailed_metrics={},
                    predictions=[],
                    embeddings=None,
                    visualization_paths=[]
                )

            model.eval()
            # Ensure model is on the correct device
            model = model.to(self.device)

        # Collect predictions and embeddings
        all_predictions = []
        all_embeddings = []
        all_labels = []
        all_similarities = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Get embeddings using the same pattern as trainer
                if self.text_encoder is None:
                    raise ValueError("Text encoder not provided to evaluator")

                # Handle resume texts (could be list or individual strings)
                resume_texts = batch['resume']
                if isinstance(resume_texts, str):
                    resume_texts = [resume_texts]
                elif torch.is_tensor(resume_texts):
                    resume_texts = resume_texts.tolist()
                elif not isinstance(resume_texts, list):
                    resume_texts = list(resume_texts)

                # Handle job texts (could be list or individual strings)
                job_texts = batch['job']
                if isinstance(job_texts, str):
                    job_texts = [job_texts]
                elif torch.is_tensor(job_texts):
                    job_texts = job_texts.tolist()
                elif not isinstance(job_texts, list):
                    job_texts = list(job_texts)

                # Encode to SentenceTransformer embeddings
                try:
                    resume_text_embeddings = self.text_encoder.encode(
                        resume_texts, convert_to_tensor=True)
                    job_text_embeddings = self.text_encoder.encode(
                        job_texts, convert_to_tensor=True)

                    # Explicitly move to device
                    resume_text_embeddings = resume_text_embeddings.to(
                        self.device)
                    job_text_embeddings = job_text_embeddings.to(self.device)

                except Exception as e:
                    logger.error(f"Error encoding texts: {e}")
                    continue

                if self.config.use_text_encoder_baseline:
                    # Use raw text encoder embeddings as a cosine-similarity baseline
                    resume_embeddings = resume_text_embeddings
                    job_embeddings = job_text_embeddings
                else:
                    # Pass through contrastive model to get final embeddings
                    try:
                        resume_embeddings = model(resume_text_embeddings)
                        job_embeddings = model(job_text_embeddings)
                    except Exception as e:
                        logger.error(f"Error in model forward pass: {e}")
                        continue

                # Calculate similarities
                similarities = F.cosine_similarity(
                    resume_embeddings, job_embeddings, dim=1)

                # Store results
                all_embeddings.append(
                    torch.cat([resume_embeddings, job_embeddings], dim=0))
                all_similarities.extend(similarities.cpu().numpy())

                # Handle batch labels properly (could be tensor, list, or single values)
                batch_labels = batch['label']
                if torch.is_tensor(batch_labels):
                    batch_labels = batch_labels.cpu().numpy().tolist()
                elif not isinstance(batch_labels, list):
                    batch_labels = [batch_labels] if not hasattr(
                        batch_labels, '__iter__') else list(batch_labels)

                # Convert labels to strings if they're numeric
                batch_labels = [
                    str(label) if label is not None else 'unknown' for label in batch_labels]

                # Ensure batch size consistency with actual tensor sizes
                actual_batch_size = resume_embeddings.shape[0]

                # Adjust labels to match actual processed batch size
                if len(batch_labels) != actual_batch_size:
                    if len(batch_labels) > actual_batch_size:
                        batch_labels = batch_labels[:actual_batch_size]
                    else:
                        # Pad with the last label or 'unknown'
                        last_label = batch_labels[-1] if batch_labels else 'unknown'
                        batch_labels.extend(
                            [last_label] * (actual_batch_size - len(batch_labels)))

                all_labels.extend(batch_labels)

                # Store detailed predictions
                batch_size = len(batch_labels)
                resume_ids = batch.get('resume_id', [None] * batch_size)
                job_ids = batch.get('job_id', [None] * batch_size)

                # Ensure ID lists are the right length and type
                if not isinstance(resume_ids, list):
                    resume_ids = [
                        resume_ids] if resume_ids is not None else [None]
                if not isinstance(job_ids, list):
                    job_ids = [job_ids] if job_ids is not None else [None]

                if len(resume_ids) != batch_size:
                    resume_ids = [None] * batch_size
                if len(job_ids) != batch_size:
                    job_ids = [None] * batch_size

                # Ensure similarities match batch size
                similarities_length = min(len(similarities), batch_size)

                for i in range(similarities_length):
                    try:
                        raw_similarity = float(similarities[i].item())
                        # Apply temperature scaling to match training objective
                        scaled_similarity = raw_similarity / self.config.temperature

                        prediction = {
                            'similarity': raw_similarity,
                            'scaled_similarity': scaled_similarity,
                            'label': batch_labels[i],
                            'resume_id': resume_ids[i],
                            'job_id': job_ids[i]
                        }
                        all_predictions.append(prediction)
                    except Exception as e:
                        logger.warning(
                            f"Error creating prediction for sample {i}: {e}")
                        continue

        # Combine embeddings
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        else:
            embeddings = np.array([])

        # Check if we have any data
        if not all_predictions:
            logger.warning("No predictions generated during evaluation")
            return EvaluationResults(
                metrics={"error": "no_predictions"},
                detailed_metrics={
                    "error": "No valid predictions were generated"},
                predictions=[],
                embeddings=embeddings if self.config.save_embeddings else None,
                visualization_paths=[]
            )

        similarity_threshold = self._determine_similarity_threshold(
            all_similarities, all_labels)

        for prediction in all_predictions:
            prediction['predicted_label'] = (
                'positive' if prediction['similarity'] > similarity_threshold else 'negative'
            )

        # Calculate metrics
        metrics = self._calculate_all_metrics(
            all_predictions, all_similarities, all_labels, similarity_threshold)

        # Generate detailed analysis
        detailed_metrics = self._calculate_detailed_metrics(
            all_predictions, embeddings, all_similarities, all_labels
        )

        # Save results
        results_path = output_path / "evaluation_results.json"
        self._save_metrics(metrics, detailed_metrics, results_path)

        # Save predictions if requested
        predictions_path = None
        if self.config.save_predictions:
            predictions_path = output_path / "predictions.json"
            self._save_predictions(all_predictions, predictions_path)

        # Save embeddings if requested
        embeddings_path = None
        if self.config.save_embeddings:
            embeddings_path = output_path / "embeddings.npy"
            np.save(embeddings_path, embeddings)

        # Generate visualizations
        visualization_paths = []
        if self.config.generate_visualizations:
            visualization_paths = self._generate_visualizations(
                all_predictions, embeddings, all_similarities, all_labels, output_path
            )

        logger.info(f"Evaluation complete. Results saved to {output_path}")

        return EvaluationResults(
            metrics=metrics,
            detailed_metrics=detailed_metrics,
            predictions=all_predictions if self.config.save_predictions else None,
            embeddings=embeddings if self.config.save_embeddings else None,
            visualization_paths=visualization_paths
        )

    def _calculate_all_metrics(self, predictions: List[Dict], similarities: List[float],
                               labels: List[str], similarity_threshold: float) -> Dict[str, float]:
        """Calculate all requested metrics"""

        metrics = {}
        metrics["similarity_threshold"] = similarity_threshold

        # Convert labels to binary - handle both string and numeric labels
        def label_to_binary(label):
            if isinstance(label, str):
                label_lower = label.lower()
                if label_lower in ['positive', 'pos', '1', 'true']:
                    return 1
                elif label_lower in ['negative', 'neg', '0', 'false']:
                    return 0
                else:
                    # Default: treat numeric 1 as positive, everything else as negative
                    try:
                        return 1 if float(label) > 0.5 else 0
                    except (ValueError, TypeError):
                        return 0
            else:
                # Numeric label
                try:
                    return 1 if float(label) > 0.5 else 0
                except (ValueError, TypeError):
                    return 0

        binary_labels = [label_to_binary(label) for label in labels]
        predicted_labels = [1 if pred['predicted_label']
                            == 'positive' else 0 for pred in predictions]

        # Basic classification metrics
        if "contrastive_accuracy" in self.config.metrics:
            if len(binary_labels) > 0:
                metrics["contrastive_accuracy"] = np.mean(
                    [p == l for p, l in zip(predicted_labels, binary_labels)])
            else:
                metrics["contrastive_accuracy"] = 0.0

        if "precision" in self.config.metrics:
            if len(binary_labels) > 0 and len(predicted_labels) > 0:
                metrics["precision"] = precision_score(
                    binary_labels, predicted_labels, zero_division=0)
            else:
                metrics["precision"] = 0.0

        if "recall" in self.config.metrics:
            if len(binary_labels) > 0 and len(predicted_labels) > 0:
                metrics["recall"] = recall_score(
                    binary_labels, predicted_labels, zero_division=0)
            else:
                metrics["recall"] = 0.0

        if "f1_score" in self.config.metrics:
            if len(binary_labels) > 0 and len(predicted_labels) > 0:
                metrics["f1_score"] = f1_score(
                    binary_labels, predicted_labels, zero_division=0)
            else:
                metrics["f1_score"] = 0.0

        if "auc_roc" in self.config.metrics:
            # Need both classes for ROC
            if len(set(binary_labels)) > 1 and len(similarities) == len(binary_labels):
                try:
                    metrics["auc_roc"] = roc_auc_score(
                        binary_labels, similarities)
                except ValueError as e:
                    logger.warning(f"AUC ROC calculation failed: {e}")
                    metrics["auc_roc"] = 0.0
            else:
                metrics["auc_roc"] = 0.0

        # Retrieval metrics
        if "precision_at_k" in self.config.metrics:
            for k in self.config.k_values:
                metrics[f"precision_at_{k}"] = self._precision_at_k(
                    predictions, k)

        if "recall_at_k" in self.config.metrics:
            for k in self.config.k_values:
                metrics[f"recall_at_{k}"] = self._recall_at_k(predictions, k)

        if "map_score" in self.config.metrics:
            metrics["map_score"] = self._mean_average_precision(predictions)

        if "ndcg_score" in self.config.metrics:
            for k in self.config.k_values:
                metrics[f"ndcg_at_{k}"] = self._ndcg_at_k(predictions, k)

        # Similarity distribution metrics
        if "similarity_statistics" in self.config.metrics:
            pos_similarities = [s for s, l in zip(
                similarities, binary_labels) if l == 1]
            neg_similarities = [s for s, l in zip(
                similarities, binary_labels) if l == 0]

            if pos_similarities:
                metrics["avg_positive_similarity"] = np.mean(pos_similarities)
                metrics["std_positive_similarity"] = np.std(pos_similarities)

            if neg_similarities:
                metrics["avg_negative_similarity"] = np.mean(neg_similarities)
                metrics["std_negative_similarity"] = np.std(neg_similarities)

            if pos_similarities and neg_similarities:
                metrics["similarity_separation"] = np.mean(
                    pos_similarities) - np.mean(neg_similarities)

        return metrics

    def _determine_similarity_threshold(self, similarities: List[float],
                                        labels: List[str]) -> float:
        """Determine similarity threshold from data, falling back to config if needed."""
        if self.config.similarity_threshold is not None:
            return self.config.similarity_threshold

        if not similarities or not labels or len(similarities) != len(labels):
            return 0.0

        def label_to_binary(label):
            if isinstance(label, str):
                label_lower = label.lower()
                if label_lower in ['positive', 'pos', '1', 'true']:
                    return 1
                if label_lower in ['negative', 'neg', '0', 'false']:
                    return 0
                try:
                    return 1 if float(label) > 0.5 else 0
                except (ValueError, TypeError):
                    return 0
            try:
                return 1 if float(label) > 0.5 else 0
            except (ValueError, TypeError):
                return 0

        binary_labels = np.array([label_to_binary(label) for label in labels])

        if len(np.unique(binary_labels)) < 2:
            return 0.0

        similarities_array = np.array(similarities)
        thresholds = np.unique(similarities_array)
        best_threshold = float(thresholds[0])
        best_f1 = -1.0

        for threshold in thresholds:
            predicted = (similarities_array > threshold).astype(int)
            score = f1_score(binary_labels, predicted, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        return best_threshold

    def _calculate_detailed_metrics(self, predictions: List[Dict], embeddings: np.ndarray,
                                    similarities: List[float], labels: List[str]) -> Dict[str, Any]:
        """Calculate detailed metrics for analysis"""

        detailed = {}

        # Confusion matrix data
        def label_to_binary(label):
            if isinstance(label, str):
                label_lower = label.lower()
                if label_lower in ['positive', 'pos', '1', 'true']:
                    return 1
                if label_lower in ['negative', 'neg', '0', 'false']:
                    return 0
                try:
                    return 1 if float(label) > 0.5 else 0
                except (ValueError, TypeError):
                    return 0
            try:
                return 1 if float(label) > 0.5 else 0
            except (ValueError, TypeError):
                return 0

        binary_labels = [label_to_binary(label) for label in labels]
        predicted_labels = [1 if pred['predicted_label']
                            == 'positive' else 0 for pred in predictions]

        tp = sum(1 for p, l in zip(predicted_labels,
                 binary_labels) if p == 1 and l == 1)
        tn = sum(1 for p, l in zip(predicted_labels,
                 binary_labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(predicted_labels,
                 binary_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predicted_labels,
                 binary_labels) if p == 0 and l == 1)

        detailed["confusion_matrix"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

        # Embedding quality metrics
        if embeddings is not None:
            detailed["embedding_stats"] = {
                "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
                "dimension": embeddings.shape[1],
                "num_samples": embeddings.shape[0]
            }

        # Similarity distribution analysis
        pos_similarities = [s for s, l in zip(
            similarities, binary_labels) if l == 1]
        neg_similarities = [s for s, l in zip(
            similarities, binary_labels) if l == 0]

        detailed["similarity_distributions"] = {
            "positive": {
                "count": len(pos_similarities),
                "mean": float(np.mean(pos_similarities)) if pos_similarities else 0.0,
                "std": float(np.std(pos_similarities)) if pos_similarities else 0.0,
                "min": float(np.min(pos_similarities)) if pos_similarities else 0.0,
                "max": float(np.max(pos_similarities)) if pos_similarities else 0.0,
                "percentiles": {
                    "25": float(np.percentile(pos_similarities, 25)) if pos_similarities else 0.0,
                    "50": float(np.percentile(pos_similarities, 50)) if pos_similarities else 0.0,
                    "75": float(np.percentile(pos_similarities, 75)) if pos_similarities else 0.0
                }
            },
            "negative": {
                "count": len(neg_similarities),
                "mean": float(np.mean(neg_similarities)) if neg_similarities else 0.0,
                "std": float(np.std(neg_similarities)) if neg_similarities else 0.0,
                "min": float(np.min(neg_similarities)) if neg_similarities else 0.0,
                "max": float(np.max(neg_similarities)) if neg_similarities else 0.0,
                "percentiles": {
                    "25": float(np.percentile(neg_similarities, 25)) if neg_similarities else 0.0,
                    "50": float(np.percentile(neg_similarities, 50)) if neg_similarities else 0.0,
                    "75": float(np.percentile(neg_similarities, 75)) if neg_similarities else 0.0
                }
            }
        }

        # Error analysis
        errors = []
        for pred in predictions:
            if pred['label'] != pred['predicted_label']:
                errors.append({
                    'type': 'false_positive' if pred['predicted_label'] == 'positive' else 'false_negative',
                    'similarity': pred['similarity'],
                    'resume_id': pred['resume_id'],
                    'job_id': pred['job_id']
                })

        detailed["error_analysis"] = {
            "total_errors": len(errors),
            "false_positives": len([e for e in errors if e['type'] == 'false_positive']),
            "false_negatives": len([e for e in errors if e['type'] == 'false_negative']),
            "error_rate": len(errors) / len(predictions) if predictions else 0.0
        }

        return detailed

    def _precision_at_k(self, predictions: List[Dict], k: int) -> float:
        """Calculate Precision@K"""
        # Sort by similarity score
        sorted_preds = sorted(
            predictions, key=lambda x: x['similarity'], reverse=True)
        top_k = sorted_preds[:k]

        if not top_k:
            return 0.0

        relevant = sum(1 for pred in top_k if pred['label'] == 'positive')
        return relevant / len(top_k)

    def _recall_at_k(self, predictions: List[Dict], k: int) -> float:
        """Calculate Recall@K"""
        total_relevant = sum(
            1 for pred in predictions if pred['label'] == 'positive')

        if total_relevant == 0:
            return 0.0

        # Sort by similarity score
        sorted_preds = sorted(
            predictions, key=lambda x: x['similarity'], reverse=True)
        top_k = sorted_preds[:k]

        relevant_in_top_k = sum(
            1 for pred in top_k if pred['label'] == 'positive')
        return relevant_in_top_k / total_relevant

    def _mean_average_precision(self, predictions: List[Dict]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        # Sort by similarity score
        sorted_preds = sorted(
            predictions, key=lambda x: x['similarity'], reverse=True)

        average_precisions = []
        relevant_count = 0

        for i, pred in enumerate(sorted_preds):
            if pred['label'] == 'positive':
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                average_precisions.append(precision_at_i)

        return np.mean(average_precisions) if average_precisions else 0.0

    def _ndcg_at_k(self, predictions: List[Dict], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        # Sort by similarity score
        sorted_preds = sorted(
            predictions, key=lambda x: x['similarity'], reverse=True)
        top_k = sorted_preds[:k]

        # Calculate DCG
        dcg = 0.0
        for i, pred in enumerate(top_k):
            relevance = 1 if pred['label'] == 'positive' else 0
            dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0

        # Calculate IDCG (ideal DCG)
        relevant_count = sum(
            1 for pred in predictions if pred['label'] == 'positive')
        ideal_relevances = [
            1] * min(k, relevant_count) + [0] * max(0, k - relevant_count)

        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += relevance / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _generate_visualizations(self, predictions: List[Dict], embeddings: np.ndarray,
                                 similarities: List[float], labels: List[str],
                                 output_path: Path) -> List[str]:
        """Generate evaluation visualizations"""

        viz_paths = []

        # Skip visualizations if no data
        if not predictions or len(similarities) == 0 or len(labels) == 0:
            logger.warning("No data available for visualization generation")
            return viz_paths

        # Ensure all lists have consistent lengths
        min_length = min(len(predictions), len(similarities), len(labels))
        if min_length == 0:
            logger.warning("Empty data arrays for visualization generation")
            return viz_paths

        # Truncate to consistent length if needed
        predictions = predictions[:min_length]
        similarities = similarities[:min_length]
        labels = labels[:min_length]

        try:
            # 1. Similarity distribution plot
            plt.figure(figsize=(12, 8))

            # Convert labels to binary using the same logic as metrics
            def label_to_binary(label):
                if isinstance(label, str):
                    label_lower = label.lower()
                    if label_lower in ['positive', 'pos', '1', 'true']:
                        return 1
                    else:
                        try:
                            return 1 if float(label) > 0.5 else 0
                        except (ValueError, TypeError):
                            return 0
                else:
                    try:
                        return 1 if float(label) > 0.5 else 0
                    except (ValueError, TypeError):
                        return 0

            binary_labels = [label_to_binary(label) for label in labels]
            pos_similarities = [s for s, l in zip(
                similarities, binary_labels) if l == 1]
            neg_similarities = [s for s, l in zip(
                similarities, binary_labels) if l == 0]

            plt.subplot(2, 2, 1)
            if pos_similarities:
                plt.hist(pos_similarities, bins=30, alpha=0.7,
                         label='Positive', color='green')
            if neg_similarities:
                plt.hist(neg_similarities, bins=30, alpha=0.7,
                         label='Negative', color='red')
            plt.xlabel('Similarity Score')
            plt.ylabel('Count')
            plt.title('Similarity Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 2. Precision-Recall curve
            plt.subplot(2, 2, 2)
            thresholds = np.linspace(0, 1, 100)
            precisions = []
            recalls = []

            for threshold in thresholds:
                pred_labels = [
                    1 if s >= threshold else 0 for s in similarities]
                if sum(pred_labels) > 0:
                    precision = precision_score(
                        binary_labels, pred_labels, zero_division=0)
                    recall = recall_score(
                        binary_labels, pred_labels, zero_division=0)
                else:
                    precision = recall = 0.0
                precisions.append(precision)
                recalls.append(recall)

            plt.plot(recalls, precisions, 'b-', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True, alpha=0.3)

            # 3. ROC Curve
            plt.subplot(2, 2, 3)
            if len(set(binary_labels)) > 1:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(binary_labels, similarities)
                plt.plot(fpr, tpr, 'b-', linewidth=2)
                plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.grid(True, alpha=0.3)

            # 4. Confusion Matrix
            plt.subplot(2, 2, 4)
            predicted_labels = [1 if pred['predicted_label']
                                == 'positive' else 0 for pred in predictions]
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(binary_labels, predicted_labels)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            plt.tight_layout()
            viz_path = output_path / "evaluation_metrics.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(viz_path))

            # 5. Embedding visualization (t-SNE)
            if embeddings is not None and embeddings.shape[0] > 50:
                plt.figure(figsize=(10, 8))

                # Sample embeddings if too many
                max_samples = 1000
                if embeddings.shape[0] > max_samples:
                    indices = np.random.choice(
                        embeddings.shape[0], max_samples, replace=False)
                    sample_embeddings = embeddings[indices]
                    # Ensure we don't access out of bounds labels
                    sample_labels = []
                    for i in indices:
                        if i < len(labels):
                            sample_labels.append(labels[i])
                        else:
                            # Use a default label if index is out of bounds
                            sample_labels.append('unknown')
                else:
                    sample_embeddings = embeddings
                    sample_labels = labels

                # Apply t-SNE
                tsne = TSNE(n_components=2, random_state=42,
                            perplexity=min(30, len(sample_embeddings)-1))
                embeddings_2d = tsne.fit_transform(sample_embeddings)

                # Plot
                colors = ['red' if label ==
                          'negative' else 'green' for label in sample_labels]
                plt.scatter(
                    embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6, s=20)
                plt.title('Embedding Space Visualization (t-SNE)')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')

                # Add legend
                import matplotlib.patches as mpatches
                red_patch = mpatches.Patch(color='red', label='Negative')
                green_patch = mpatches.Patch(color='green', label='Positive')
                plt.legend(handles=[red_patch, green_patch])

                plt.tight_layout()
                viz_path = output_path / "embedding_visualization.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths.append(str(viz_path))

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            # Continue without visualizations

        return viz_paths

    def _save_metrics(self, metrics: Dict[str, float], detailed_metrics: Dict[str, Any],
                      output_path: Path):
        """Save metrics to file"""
        results = {
            "metrics": metrics,
            "detailed_metrics": detailed_metrics,
            "evaluation_config": {
                "metrics": self.config.metrics,
                "k_values": self.config.k_values,
                "device": self.config.device
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Metrics saved to {output_path}")

    def _save_predictions(self, predictions: List[Dict], output_path: Path):
        """Save predictions to file"""
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Predictions saved to {output_path}")


def main():
    """CLI interface for model evaluation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate contrastive learning model")
    parser.add_argument("model", help="Path to trained model checkpoint")
    parser.add_argument("dataset", help="Path to evaluation dataset (JSONL)")
    parser.add_argument(
        "--output-dir", default="evaluation_output", help="Output directory")
    parser.add_argument("--metrics", nargs='+', default=None,
                        help="Metrics to calculate")
    parser.add_argument("--k-values", nargs='+', type=int,
                        default=[1, 5, 10], help="K values for ranking metrics")
    parser.add_argument("--batch-size", type=int,
                        default=32, help="Evaluation batch size")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--no-visualizations", action="store_true",
                        help="Skip visualization generation")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    # Create evaluation config
    config = EvaluationConfig(
        metrics=args.metrics,
        k_values=args.k_values,
        batch_size=args.batch_size,
        device=args.device,
        generate_visualizations=not args.no_visualizations
    )

    # TODO: Load model and create data loader
    print("Model evaluation CLI - implementation needed for model loading")


if __name__ == "__main__":
    main()
