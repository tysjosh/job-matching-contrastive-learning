"""
FineTuningTrainer for supervised learning on pre-trained contrastive models.

This module implements a trainer for fine-tuning pre-trained contrastive encoders
on labeled data for downstream tasks like binary classification.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Iterator, Tuple
from dataclasses import asdict, dataclass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .data_structures import TrainingConfig, TrainingResults
from .contrastive_classification_model import ContrastiveClassificationModel
from .logging_utils import setup_training_logger, MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class LabeledSample:
    """Represents a labeled sample for fine-tuning."""
    resume: Dict[str, Any]
    job: Dict[str, Any]
    label: int  # 0 or 1 for binary classification
    sample_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        if self.label not in [0, 1]:
            raise ValueError(f"label must be 0 or 1, got: {self.label}")


class FineTuningTrainer:
    """
    Trainer for fine-tuning pre-trained contrastive models on labeled data.
    
    This trainer loads a pre-trained contrastive encoder and adds a classification head
    for supervised learning on downstream tasks like binary classification.
    """
    
    def __init__(self, config: TrainingConfig, output_dir: str = "fine_tuning_output"):
        """
        Initialize the FineTuningTrainer.
        
        Args:
            config: Training configuration with fine-tuning parameters
            output_dir: Directory for saving checkpoints and logs
            
        Raises:
            ImportError: If PyTorch is not available
            ValueError: If configuration is invalid for fine-tuning
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for FineTuningTrainer. "
                "Install with: pip install torch"
            )
        
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration for fine-tuning
        self._validate_config()
        
        # Initialize structured logging
        self.structured_logger = setup_training_logger(
            name="fine_tuning",
            log_dir=self.output_dir / "logs",
            log_level=logging.INFO
        )
        self.memory_monitor = MemoryMonitor()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize text encoder (same as used in pre-training)
        self.text_encoder = SentenceTransformer(config.text_encoder_model)
        self.text_encoder.to(self.device)
        
        # Freeze text encoder to maintain consistency with pre-training
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Initialize ContrastiveClassificationModel with pre-trained encoder
        self.model = ContrastiveClassificationModel(config)
        
        # Setup optimizer to train only unfrozen parameters
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = optim.Adam(trainable_params, lr=config.learning_rate)
        
        # Configure binary cross-entropy loss for binary classification
        self.criterion = nn.BCELoss()
        
        # Training state
        self.current_epoch = 0
        self.total_batches_processed = 0
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.training_metrics = {
            'total_samples_processed': 0,
            'training_start_time': None,
            'training_end_time': None
        }
        
        # Log initialization
        model_info = self.model.get_model_info()
        self.structured_logger.logger.info(f"FineTuningTrainer initialized")
        self.structured_logger.logger.info(f"Device: {self.device}")
        self.structured_logger.logger.info(f"Pre-trained model: {config.pretrained_model_path}")
        self.structured_logger.logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        self.structured_logger.logger.info(f"Frozen parameters: {model_info['frozen_parameters']:,}")
        self.structured_logger.logger.info(f"Classification dropout: {config.classification_dropout}")
        
        # Log initial memory usage
        memory_info = self.memory_monitor.get_memory_usage()
        self.structured_logger.log_memory_usage(
            "initialization",
            memory_info["system_memory_mb"] or 0,
            memory_info["gpu_memory_mb"]
        )
    
    def _validate_config(self):
        """Validate the configuration for fine-tuning."""
        if self.config.training_phase != "fine_tuning":
            raise ValueError(
                f"FineTuningTrainer requires training_phase='fine_tuning', "
                f"got: {self.config.training_phase}"
            )
        
        if not self.config.pretrained_model_path:
            raise ValueError("pretrained_model_path is required for fine-tuning")
        
        pretrained_path = Path(self.config.pretrained_model_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pre-trained model not found: {self.config.pretrained_model_path}"
            )
        
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.config.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
    
    def _create_labeled_dataloader(self, dataset_path: Union[str, Path]) -> Iterator[List[LabeledSample]]:
        """
        Create a data loader for labeled binary classification data.
        
        Args:
            dataset_path: Path to labeled dataset (JSONL format)
            
        Yields:
            Batches of LabeledSample objects
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If data format is invalid
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"Loading labeled data from: {dataset_path}")
        
        batch = []
        total_samples = 0
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Validate required fields
                        required_fields = ['resume', 'job', 'sample_id']
                        for field in required_fields:
                            if field not in data:
                                raise ValueError(f"Missing required field: {field}")
                        
                        # Extract label (try metadata.label first, then fall back to label field)
                        label_value = None
                        if 'metadata' in data and 'label' in data['metadata']:
                            label_value = int(data['metadata']['label'])
                        elif 'label' in data:
                            # Try to convert string label or use integer label
                            if isinstance(data['label'], str):
                                label_map = {'positive': 1, 'negative': 0}
                                label_value = label_map.get(data['label'].lower())
                                if label_value is None:
                                    raise ValueError(f"Invalid string label: {data['label']}")
                            else:
                                label_value = int(data['label'])
                        else:
                            raise ValueError("Missing label field in data and metadata")
                        
                        # Create labeled sample
                        sample = LabeledSample(
                            resume=data['resume'],
                            job=data['job'],
                            label=label_value,
                            sample_id=data['sample_id'],
                            metadata=data.get('metadata', {})
                        )
                        
                        batch.append(sample)
                        total_samples += 1
                        
                        # Yield batch when full
                        if len(batch) >= self.config.batch_size:
                            yield batch
                            batch = []
                    
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(f"Skipping invalid line {line_num}: {e}")
                        continue
            
            # Yield remaining samples
            if batch:
                yield batch
            
            logger.info(f"Loaded {total_samples} labeled samples from dataset")
        
        except Exception as e:
            logger.error(f"Error loading labeled dataset: {e}")
            raise
    
    def _encode_content_to_text_embedding(self, content: Dict[str, Any], content_type: str) -> torch.Tensor:
        """
        Convert structured resume/job data to text embeddings using frozen SentenceTransformer.
        
        Args:
            content: Content dictionary (resume or job)
            content_type: 'resume' or 'job'
            
        Returns:
            Text embedding tensor from SentenceTransformer
        """
        if content_type == 'resume':
            # Extract text from resume data
            text_parts = []
            
            # Add skills if available
            if 'skills' in content and content['skills']:
                if isinstance(content['skills'], list):
                    skill_names = []
                    for skill in content['skills']:
                        if isinstance(skill, dict):
                            skill_names.append(skill.get('name', ''))
                        elif isinstance(skill, str):
                            skill_names.append(skill)
                    if skill_names:
                        text_parts.append("Skills: " + ", ".join(filter(None, skill_names)))
                elif isinstance(content['skills'], str):
                    text_parts.append("Skills: " + content['skills'])
            
            # Add experience if available
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
                    text_parts.append("Experience: " + ". ".join(exp_texts))
                elif isinstance(content['experience'], str):
                    text_parts.append("Experience: " + content['experience'])
            
            # Add education if available
            if 'education' in content and content['education']:
                if isinstance(content['education'], list):
                    edu_texts = []
                    for edu in content['education']:
                        if isinstance(edu, dict):
                            edu_text = f"{edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}"
                            edu_texts.append(edu_text)
                        elif isinstance(edu, str):
                            edu_texts.append(edu)
                    text_parts.append("Education: " + ". ".join(edu_texts))
                elif isinstance(content['education'], str):
                    text_parts.append("Education: " + content['education'])
            
            # Fallback to any text field
            if not text_parts and 'text' in content:
                text_parts.append(content['text'])
            
            text = " | ".join(text_parts) if text_parts else "No resume information available"
        
        elif content_type == 'job':
            # Extract text from job data
            text_parts = []
            
            # Add job title
            if 'title' in content and content['title']:
                text_parts.append(f"Job Title: {content['title']}")
            
            # Add company
            if 'company' in content and content['company']:
                text_parts.append(f"Company: {content['company']}")
            
            # Add description
            if 'description' in content and content['description']:
                text_parts.append(f"Description: {content['description']}")
            
            # Add required skills
            if 'required_skills' in content and content['required_skills']:
                if isinstance(content['required_skills'], list):
                    skill_names = []
                    for skill in content['required_skills']:
                        if isinstance(skill, dict):
                            skill_names.append(skill.get('name', ''))
                        elif isinstance(skill, str):
                            skill_names.append(skill)
                    if skill_names:
                        text_parts.append("Required Skills: " + ", ".join(filter(None, skill_names)))
                elif isinstance(content['required_skills'], str):
                    text_parts.append("Required Skills: " + content['required_skills'])
            
            # Also check for 'skills' field (alternate format)
            if 'skills' in content and content['skills']:
                if isinstance(content['skills'], list):
                    skill_names = []
                    for skill in content['skills']:
                        if isinstance(skill, dict):
                            skill_names.append(skill.get('name', ''))
                        elif isinstance(skill, str):
                            skill_names.append(skill)
                    if skill_names:
                        text_parts.append("Skills: " + ", ".join(filter(None, skill_names)))
                elif isinstance(content['skills'], str):
                    text_parts.append("Skills: " + content['skills'])
            
            # Add requirements
            if 'requirements' in content and content['requirements']:
                text_parts.append(f"Requirements: {content['requirements']}")
            
            # Fallback to any text field
            if not text_parts and 'text' in content:
                text_parts.append(content['text'])
            
            text = " | ".join(text_parts) if text_parts else "No job information available"
        
        else:
            raise ValueError(f"Unknown content_type: {content_type}")
        
        # Generate embedding using frozen SentenceTransformer
        with torch.no_grad():
            embedding = self.text_encoder.encode(
                text, 
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=False
            )
        
        return embedding
    
    def _train_batch(self, batch: List[LabeledSample]) -> Tuple[float, Dict[str, Any]]:
        """
        Train on a single batch of labeled data.
        
        Args:
            batch: List of labeled samples
            
        Returns:
            Tuple of (batch_loss, batch_statistics)
        """
        batch_stats = {
            'samples_processed': len(batch),
            'correct_predictions': 0,
            'processing_time': 0
        }
        
        batch_start_time = time.time()
        
        try:
            # Prepare batch data
            resume_embeddings = []
            job_embeddings = []
            labels = []
            
            for sample in batch:
                # Generate text embeddings using frozen SentenceTransformer
                resume_text_emb = self._encode_content_to_text_embedding(sample.resume, 'resume')
                job_text_emb = self._encode_content_to_text_embedding(sample.job, 'job')
                
                resume_embeddings.append(resume_text_emb)
                job_embeddings.append(job_text_emb)
                labels.append(float(sample.label))
            
            # Stack embeddings into batch tensors
            resume_batch = torch.stack(resume_embeddings)
            job_batch = torch.stack(job_embeddings)
            label_batch = torch.tensor(labels, dtype=torch.float32, device=self.device)
            
            # Forward pass through classification model
            self.model.train()
            predictions = self.model(resume_batch, job_batch)
            
            # Compute loss
            loss = self.criterion(predictions, label_batch)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predicted_labels = (predictions > 0.5).float()
                correct = (predicted_labels == label_batch).sum().item()
                batch_stats['correct_predictions'] = correct
            
            batch_stats['processing_time'] = time.time() - batch_start_time
            
            return loss.item(), batch_stats
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            batch_stats['processing_time'] = time.time() - batch_start_time
            raise
    
    def _train_epoch(self, dataset_path: Path) -> Tuple[float, float]:
        """
        Train for one epoch on labeled data.
        
        Args:
            dataset_path: Path to labeled dataset
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.structured_logger.log_epoch_start(self.current_epoch, self.config.num_epochs)
        
        epoch_start_time = time.time()
        epoch_losses = []
        total_samples = 0
        total_correct = 0
        failed_batches = 0
        
        self.model.train()
        
        try:
            # Process batches
            for batch_idx, batch in enumerate(self._create_labeled_dataloader(dataset_path)):
                self.structured_logger.log_batch_start(batch_idx, self.current_epoch, len(batch))
                
                try:
                    batch_loss, batch_stats = self._train_batch(batch)
                    
                    epoch_losses.append(batch_loss)
                    total_samples += batch_stats['samples_processed']
                    total_correct += batch_stats['correct_predictions']
                    
                    # Log batch progress
                    if (batch_idx + 1) % self.config.log_frequency == 0:
                        batch_accuracy = batch_stats['correct_predictions'] / batch_stats['samples_processed']
                        logger.info(
                            f"Epoch {self.current_epoch}, Batch {batch_idx + 1}: "
                            f"Loss = {batch_loss:.4f}, Accuracy = {batch_accuracy:.4f}"
                        )
                
                except Exception as batch_error:
                    failed_batches += 1
                    logger.error(f"Batch {batch_idx} failed: {batch_error}")
                
                self.total_batches_processed += 1
        
        except Exception as e:
            logger.error(f"Error during epoch {self.current_epoch}: {e}")
            raise
        
        # Calculate epoch metrics
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        
        # Update training metrics
        self.training_metrics['total_samples_processed'] += total_samples
        
        # Log epoch completion
        logger.info(
            f"Epoch {self.current_epoch} completed: "
            f"Loss = {avg_epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}, "
            f"Time = {epoch_time:.2f}s, Samples = {total_samples}, Failed batches = {failed_batches}"
        )
        
        return avg_epoch_loss, epoch_accuracy
    
    def train(self, labeled_dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run the complete fine-tuning training process.
        
        Args:
            labeled_dataset_path: Path to labeled dataset for fine-tuning
            
        Returns:
            Dictionary containing training results and metrics
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
        dataset_path = Path(labeled_dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Start training session
        session_id = self.structured_logger.start_training_session({
            "config": asdict(self.config),
            "dataset_path": str(dataset_path),
            "model_info": self.model.get_model_info(),
            "device": str(self.device)
        })
        
        self.training_metrics['training_start_time'] = time.time()
        
        try:
            # Save initial configuration
            self._save_config()
            
            # Run training epochs
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                epoch_loss, epoch_accuracy = self._train_epoch(dataset_path)
                
                self.epoch_losses.append(epoch_loss)
                self.epoch_accuracies.append(epoch_accuracy)
                
                # Update best metrics
                if epoch_accuracy > self.best_accuracy:
                    self.best_accuracy = epoch_accuracy
                
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                
                # Save checkpoint if needed
                if self._should_save_checkpoint():
                    checkpoint_path = self._save_checkpoint(epoch, epoch_loss, epoch_accuracy)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Training completed successfully
            self.training_metrics['training_end_time'] = time.time()
            results = self._create_training_results()
            
            # Save final results
            self._save_training_results(results)
            
            self.structured_logger.end_training_session({
                "final_loss": results['final_loss'],
                "final_accuracy": results['final_accuracy'],
                "best_accuracy": self.best_accuracy,
                "total_samples": results['total_samples']
            })
            
            logger.info("Fine-tuning completed successfully")
            return results
        
        except Exception as e:
            logger.error(f"Fine-tuning failed with error: {e}")
            
            # Save emergency checkpoint
            try:
                checkpoint_path = self._save_checkpoint(
                    self.current_epoch, float('inf'), 0.0, emergency=True
                )
                logger.info(f"Emergency checkpoint saved: {checkpoint_path}")
            except Exception as checkpoint_error:
                logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
            
            # End session with error
            self.structured_logger.end_training_session({
                "error": str(e),
                "final_loss": None,
                "final_accuracy": None
            })
            raise
    
    def _should_save_checkpoint(self) -> bool:
        """Determine if a checkpoint should be saved."""
        return (self.current_epoch + 1) % max(1, self.config.num_epochs // 5) == 0
    
    def _save_checkpoint(self, epoch: int, loss: float, accuracy: float, emergency: bool = False) -> str:
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_epoch_{epoch}"
        if emergency:
            checkpoint_name += "_emergency"
        checkpoint_name += ".pt"
        
        checkpoint_path = self.output_dir / checkpoint_name
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'config': asdict(self.config),
            'training_metrics': self.training_metrics
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        return str(checkpoint_path)
    
    def _save_config(self):
        """Save training configuration."""
        config_path = self.output_dir / "fine_tuning_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def _create_training_results(self) -> Dict[str, Any]:
        """Create training results dictionary."""
        training_time = self.training_metrics['training_end_time'] - self.training_metrics['training_start_time']
        
        return {
            'final_loss': self.epoch_losses[-1] if self.epoch_losses else float('inf'),
            'final_accuracy': self.epoch_accuracies[-1] if self.epoch_accuracies else 0.0,
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'epoch_losses': self.epoch_losses,
            'epoch_accuracies': self.epoch_accuracies,
            'training_time': training_time,
            'total_batches': self.total_batches_processed,
            'total_samples': self.training_metrics['total_samples_processed'],
            'model_info': self.model.get_model_info(),
            'config': asdict(self.config)
        }
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file."""
        results_path = self.output_dir / "fine_tuning_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to: {results_path}")