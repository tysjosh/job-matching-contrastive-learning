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
from .structured_features import StructuredFeatureExtractor

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
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize text encoder (same as used in pre-training)
        self.text_encoder = SentenceTransformer(config.text_encoder_model)
        self.text_encoder.max_seq_length = 512
        self.text_encoder.to(self.device)

        # Freeze text encoder to maintain consistency with pre-training
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Initialize ContrastiveClassificationModel with pre-trained encoder
        self.model = ContrastiveClassificationModel(config)

        # Initialize structured feature extractor if model uses structured features
        self.use_structured_features = self.model.uses_structured_features()
        if self.use_structured_features:
            self.feature_extractor = StructuredFeatureExtractor()
            logger.info(
                "Structured feature extraction enabled for fine-tuning")
        else:
            self.feature_extractor = None
            logger.info(
                "Structured feature extraction disabled (model doesn't use it)")

        # Setup optimizer to train only unfrozen parameters
        trainable_params = self.model.get_trainable_parameters()
        weight_decay = getattr(config, 'weight_decay', 0.0)
        self.optimizer = optim.Adam(
            trainable_params, lr=config.learning_rate, weight_decay=weight_decay)
        logger.info(
            f"Optimizer: Adam with lr={config.learning_rate}, weight_decay={weight_decay}")

        # Configure binary cross-entropy loss with class weighting
        # pos_class_weight > 1.0 upweights positive samples to handle imbalance
        # (72% negative / 28% positive → weight ≈ 2.5 recommended)
        self.pos_class_weight = getattr(config, 'pos_class_weight', 0.0)
        if self.pos_class_weight > 0:
            self.criterion = nn.BCELoss(reduction='none')  # per-sample loss for weighting
            logger.info(f"Using BCELoss with pos_class_weight={self.pos_class_weight}")
        else:
            self.criterion = nn.BCELoss()
            logger.info("Using standard BCELoss (no class weighting)")

        # Training state
        self.current_epoch = 0
        self.total_batches_processed = 0
        self.best_accuracy = 0.0
        self.best_loss = float('inf')

        # Metrics tracking
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.checkpoint_paths = []
        self.training_metrics = {
            'total_samples_processed': 0,
            'training_start_time': None,
            'training_end_time': None
        }

        # Log initialization
        model_info = self.model.get_model_info()
        self.structured_logger.logger.info(f"FineTuningTrainer initialized")
        self.structured_logger.logger.info(f"Device: {self.device}")
        self.structured_logger.logger.info(
            f"Pre-trained model: {config.pretrained_model_path}")
        self.structured_logger.logger.info(
            f"Trainable parameters: {model_info['trainable_parameters']:,}")
        self.structured_logger.logger.info(
            f"Frozen parameters: {model_info['frozen_parameters']:,}")
        self.structured_logger.logger.info(
            f"Classification dropout: {config.classification_dropout}")
        self.structured_logger.logger.info(
            f"Uses structured features: {self.use_structured_features}")

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
            raise ValueError(
                "pretrained_model_path is required for fine-tuning")

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

                        # Validate required fields (sample_id can be job_applicant_id)
                        if 'resume' not in data:
                            raise ValueError("Missing required field: resume")
                        if 'job' not in data:
                            raise ValueError("Missing required field: job")

                        # Get sample_id from either sample_id or job_applicant_id
                        sample_id = data.get('sample_id') or data.get(
                            'job_applicant_id')
                        if not sample_id:
                            raise ValueError(
                                "Missing required field: sample_id or job_applicant_id")

                        # Extract label (try metadata.label first, then fall back to label field)
                        label_value = None
                        if 'metadata' in data and 'label' in data['metadata']:
                            label_value = int(data['metadata']['label'])
                        elif 'label' in data:
                            # Try to convert string label or use integer label
                            if isinstance(data['label'], str):
                                label_map = {'positive': 1, 'negative': 0}
                                label_value = label_map.get(
                                    data['label'].lower())
                                if label_value is None:
                                    raise ValueError(
                                        f"Invalid string label: {data['label']}")
                            else:
                                label_value = int(data['label'])
                        else:
                            raise ValueError(
                                "Missing label field in data and metadata")

                        # Create labeled sample
                        sample = LabeledSample(
                            resume=data['resume'],
                            job=data['job'],
                            label=label_value,
                            sample_id=sample_id,
                            metadata=data.get('metadata', {})
                        )

                        batch.append(sample)
                        total_samples += 1

                        # Yield batch when full
                        if len(batch) >= self.config.batch_size:
                            yield batch
                            batch = []

                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(
                            f"Skipping invalid line {line_num}: {e}")
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

        Prioritizes the most discriminative fields first to fit within
        the SentenceTransformer's 256-token window.

        Args:
            content: Content dictionary (resume or job)
            content_type: 'resume' or 'job'

        Returns:
            Text embedding tensor from SentenceTransformer
        """
        if content_type == 'resume':
            text_parts = []

            # 1. Role and experience level FIRST (most discriminative metadata)
            role = content.get('role', '')
            exp_level = content.get('experience_level', '')
            if role and exp_level:
                text_parts.append(f"Position: {exp_level} {role}")
            elif role:
                text_parts.append(f"Position: {role}")

            # 2. Skills SECOND (key signal for matching, must always be included)
            if 'skills' in content and content['skills']:
                skill_names = []
                if isinstance(content['skills'], list):
                    for skill in content['skills']:
                        if isinstance(skill, dict):
                            skill_name = skill.get('name', '')
                            if skill_name:
                                skill_names.append(skill_name)
                        elif isinstance(skill, str):
                            skill_names.append(skill)
                elif isinstance(content['skills'], str):
                    skill_names.append(content['skills'])
                if skill_names:
                    text_parts.append(f"Skills: {', '.join(skill_names)}")

            # 3. Experience text LAST (truncated, fills remaining token budget)
            if 'experience' in content and content['experience']:
                experience_text = ''
                if isinstance(content['experience'], list) and len(content['experience']) > 0:
                    exp_entry = content['experience'][0]
                    if isinstance(exp_entry, dict):
                        desc_field = exp_entry.get('description', '')
                        if isinstance(desc_field, list) and len(desc_field) > 0:
                            if isinstance(desc_field[0], dict):
                                experience_text = desc_field[0].get('description', '')
                            else:
                                experience_text = str(desc_field[0])
                        elif isinstance(desc_field, str):
                            experience_text = desc_field
                        else:
                            experience_text = str(desc_field) if desc_field else ''
                    elif isinstance(exp_entry, str):
                        experience_text = exp_entry
                elif isinstance(content['experience'], str):
                    experience_text = content['experience']

                if experience_text:
                    if len(experience_text) > 800:
                        experience_text = experience_text[:800] + "..."
                    text_parts.append(f"Profile: {experience_text}")

            if not text_parts and 'text' in content:
                text_parts.append(content['text'])

            text = " [SEP] ".join(text_parts) if text_parts else "No resume information available"

        elif content_type == 'job':
            text_parts = []

            # 1. Title FIRST
            title = content.get('title', content.get('jobtitle', ''))
            if title:
                text_parts.append(f"Position: {title}")

            # 2. Job skills SECOND (key signal for matching, must always be included)
            skills = content.get('required_skills', content.get('skills', []))
            if skills:
                skill_strings = []
                if isinstance(skills, list):
                    for skill in skills:
                        if isinstance(skill, dict):
                            skill_name = skill.get('name', skill.get('skill', ''))
                            if skill_name:
                                skill_strings.append(skill_name)
                        elif isinstance(skill, str):
                            # Skip malformed skills (sentence fragments)
                            if len(skill) <= 40 and '.' not in skill:
                                skill_strings.append(skill)
                            elif len(skill) <= 20:
                                skill_strings.append(skill)
                elif isinstance(skills, str):
                    skill_strings.append(skills)
                if skill_strings:
                    text_parts.append(f"Required Skills: {', '.join(skill_strings)}")

            # 3. Job description LAST (fills remaining token budget)
            description = content.get('description', content.get('jobdescription', ''))
            if description:
                if isinstance(description, dict):
                    desc_text = description.get('original', '')
                elif isinstance(description, str):
                    desc_text = description
                else:
                    desc_text = str(description)

                if desc_text:
                    if len(desc_text) > 800:
                        desc_text = desc_text[:800] + "..."
                    text_parts.append(f"Description: {desc_text}")

            if not text_parts and 'text' in content:
                text_parts.append(content['text'])

            text = " [SEP] ".join(text_parts) if text_parts else "No job information available"

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

    def _extract_structured_features(self, content: Dict[str, Any], content_type: str) -> Tuple[int, torch.Tensor]:
        """
        Extract structured features from content.

        Args:
            content: Resume or job content dictionary
            content_type: 'resume' or 'job'

        Returns:
            Tuple of (experience_level_idx, numerical_features)
        """
        features = self.feature_extractor.extract_features(
            content, content_type)

        # Split into experience level (one-hot) and numerical
        exp_onehot = features[:10]  # First 10 are one-hot
        numerical = features[10:]   # Rest are numerical

        # Convert one-hot to index
        exp_idx = exp_onehot.argmax().item()

        return exp_idx, numerical

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

            # Structured features (if enabled)
            resume_exp_indices = []
            resume_num_features = []
            job_exp_indices = []
            job_num_features = []

            for sample in batch:
                # Generate text embeddings using frozen SentenceTransformer
                resume_text_emb = self._encode_content_to_text_embedding(
                    sample.resume, 'resume')
                job_text_emb = self._encode_content_to_text_embedding(
                    sample.job, 'job')

                resume_embeddings.append(resume_text_emb)
                job_embeddings.append(job_text_emb)
                labels.append(float(sample.label))

                # Extract structured features if enabled
                if self.use_structured_features:
                    r_exp_idx, r_num = self._extract_structured_features(
                        sample.resume, 'resume')
                    j_exp_idx, j_num = self._extract_structured_features(
                        sample.job, 'job')

                    resume_exp_indices.append(r_exp_idx)
                    resume_num_features.append(r_num)
                    job_exp_indices.append(j_exp_idx)
                    job_num_features.append(j_num)

            # Stack embeddings into batch tensors
            resume_batch = torch.stack(resume_embeddings)
            job_batch = torch.stack(job_embeddings)
            label_batch = torch.tensor(
                labels, dtype=torch.float32, device=self.device)

            # Prepare structured feature tensors if enabled
            if self.use_structured_features:
                resume_exp_batch = torch.tensor(
                    resume_exp_indices, dtype=torch.long, device=self.device)
                resume_num_batch = torch.stack(
                    resume_num_features).to(self.device)
                job_exp_batch = torch.tensor(
                    job_exp_indices, dtype=torch.long, device=self.device)
                job_num_batch = torch.stack(job_num_features).to(self.device)
            else:
                resume_exp_batch = None
                resume_num_batch = None
                job_exp_batch = None
                job_num_batch = None

            # Forward pass through classification model
            self.model.train()
            predictions = self.model(
                resume_batch, job_batch,
                resume_exp_batch, resume_num_batch,
                job_exp_batch, job_num_batch
            )

            # Compute loss (with optional class weighting)
            if self.pos_class_weight > 0:
                # Per-sample loss with class weighting
                per_sample_loss = self.criterion(predictions, label_batch)
                weights = torch.where(
                    label_batch == 1.0,
                    torch.tensor(self.pos_class_weight, device=self.device),
                    torch.tensor(1.0, device=self.device)
                )
                loss = (per_sample_loss * weights).mean()
            else:
                loss = self.criterion(predictions, label_batch)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(), max_norm=1.0)

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
        self.structured_logger.log_epoch_start(
            self.current_epoch, self.config.num_epochs)

        epoch_start_time = time.time()
        epoch_losses = []
        total_samples = 0
        total_correct = 0
        failed_batches = 0

        self.model.train()

        try:
            # Process batches
            for batch_idx, batch in enumerate(self._create_labeled_dataloader(dataset_path)):
                self.structured_logger.log_batch_start(
                    batch_idx, self.current_epoch, len(batch))

                try:
                    batch_loss, batch_stats = self._train_batch(batch)

                    epoch_losses.append(batch_loss)
                    total_samples += batch_stats['samples_processed']
                    total_correct += batch_stats['correct_predictions']

                    # Log batch progress
                    if (batch_idx + 1) % self.config.log_frequency == 0:
                        batch_accuracy = batch_stats['correct_predictions'] / \
                            batch_stats['samples_processed']
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
        avg_epoch_loss = sum(epoch_losses) / \
            len(epoch_losses) if epoch_losses else float('inf')
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

    def _validate_epoch(self, validation_dataset_path: Path) -> Tuple[float, float]:
        """
        Validate on validation set (no gradient updates).

        Args:
            validation_dataset_path: Path to validation dataset

        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.model.eval()

        val_losses = []
        total_samples = 0
        total_correct = 0

        with torch.no_grad():
            for batch in self._create_labeled_dataloader(validation_dataset_path):
                try:
                    # Encode resume and job using frozen text encoder
                    resume_texts = []
                    job_texts = []

                    # Structured features (if enabled)
                    resume_exp_indices = []
                    resume_num_features = []
                    job_exp_indices = []
                    job_num_features = []

                    for sample in batch:
                        resume_text = self._encode_content_to_text_embedding(
                            sample.resume, 'resume')
                        job_text = self._encode_content_to_text_embedding(
                            sample.job, 'job')
                        resume_texts.append(resume_text)
                        job_texts.append(job_text)

                        # Extract structured features if enabled
                        if self.use_structured_features:
                            r_exp_idx, r_num = self._extract_structured_features(
                                sample.resume, 'resume')
                            j_exp_idx, j_num = self._extract_structured_features(
                                sample.job, 'job')

                            resume_exp_indices.append(r_exp_idx)
                            resume_num_features.append(r_num)
                            job_exp_indices.append(j_exp_idx)
                            job_num_features.append(j_num)

                    # Get embeddings from text encoder
                    resume_embeddings = torch.stack(
                        resume_texts).to(self.device)
                    job_embeddings = torch.stack(job_texts).to(self.device)

                    # Prepare structured feature tensors if enabled
                    if self.use_structured_features:
                        resume_exp_batch = torch.tensor(
                            resume_exp_indices, dtype=torch.long, device=self.device)
                        resume_num_batch = torch.stack(
                            resume_num_features).to(self.device)
                        job_exp_batch = torch.tensor(
                            job_exp_indices, dtype=torch.long, device=self.device)
                        job_num_batch = torch.stack(
                            job_num_features).to(self.device)
                    else:
                        resume_exp_batch = None
                        resume_num_batch = None
                        job_exp_batch = None
                        job_num_batch = None

                    # Get labels
                    labels = torch.tensor([sample.label for sample in batch],
                                          dtype=torch.float32).to(self.device)

                    # Forward pass through classification head
                    outputs = self.model(
                        resume_embeddings, job_embeddings,
                        resume_exp_batch, resume_num_batch,
                        job_exp_batch, job_num_batch
                    )
                    predictions = outputs.squeeze()

                    # Calculate loss (with optional class weighting)
                    if self.pos_class_weight > 0:
                        per_sample_loss = self.criterion(predictions, labels)
                        weights = torch.where(
                            labels == 1.0,
                            torch.tensor(self.pos_class_weight, device=self.device),
                            torch.tensor(1.0, device=self.device)
                        )
                        loss = (per_sample_loss * weights).mean()
                    else:
                        loss = self.criterion(predictions, labels)
                    val_losses.append(loss.item())

                    # Calculate accuracy
                    predicted_classes = (predictions > 0.5).int()
                    correct = (predicted_classes == labels.int()).sum().item()
                    total_correct += correct
                    total_samples += len(batch)

                except Exception as e:
                    logger.warning(f"Validation batch failed: {e}")
                    continue

        avg_val_loss = sum(val_losses) / \
            len(val_losses) if val_losses else float('inf')
        val_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        self.model.train()  # Switch back to training mode

        return avg_val_loss, val_accuracy

    def train(self, labeled_dataset_path: Union[str, Path],
              validation_data_path: Union[str, Path, None] = None) -> Dict[str, Any]:
        """
        Run the complete fine-tuning training process.

        Args:
            labeled_dataset_path: Path to labeled dataset for fine-tuning
            validation_data_path: Optional path to validation dataset for monitoring
                                  (if None, uses config.validation_path)

        Returns:
            Dictionary containing training results and metrics

        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
        dataset_path = Path(labeled_dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Use validation_path from config if not provided as parameter
        if validation_data_path is None and hasattr(self.config, 'validation_path') and self.config.validation_path:
            validation_data_path = self.config.validation_path

        # Validate validation dataset if provided
        val_dataset_path = None
        if validation_data_path:
            val_dataset_path = Path(validation_data_path)
            if not val_dataset_path.exists():
                logger.warning(
                    f"Validation dataset not found: {val_dataset_path}, skipping validation")
                val_dataset_path = None
            else:
                logger.info(f"Using validation dataset: {val_dataset_path}")

        # Track best validation loss for checkpoint selection
        best_val_loss = float('inf')


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

                # Compute validation metrics if validation set provided
                val_loss = None
                val_accuracy = None
                if val_dataset_path:
                    val_loss, val_accuracy = self._validate_epoch(
                        val_dataset_path)
                    self.validation_losses.append(val_loss)
                    self.validation_accuracies.append(val_accuracy)
                    logger.info(
                        f"Epoch {epoch} - Train: Loss={epoch_loss:.4f}, Acc={epoch_accuracy:.4f} | "
                        f"Val: Loss={val_loss:.4f}, Acc={val_accuracy:.4f}"
                    )

                    # Save best checkpoint based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_best_checkpoint(
                            epoch, val_loss, val_accuracy)
                        logger.info(
                            f"New best validation loss: {val_loss:.4f} - saved best_checkpoint.pt")
                else:
                    logger.info(
                        f"Epoch {epoch} - Train: Loss={epoch_loss:.4f}, Acc={epoch_accuracy:.4f}"
                    )

                # Update best metrics (based on training accuracy)
                if epoch_accuracy > self.best_accuracy:
                    self.best_accuracy = epoch_accuracy

                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss

                # Save checkpoint if needed
                if self._should_save_checkpoint():
                    checkpoint_path = self._save_checkpoint(
                        epoch, epoch_loss, epoch_accuracy)
                    self.checkpoint_paths.append(checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Add best checkpoint to paths if it exists
            best_checkpoint = self.output_dir / "best_checkpoint.pt"
            if best_checkpoint.exists() and str(best_checkpoint) not in self.checkpoint_paths:
                self.checkpoint_paths.insert(0, str(best_checkpoint))

            # Training completed successfully
            self.training_metrics['training_end_time'] = time.time()
            results = self._create_training_results()

            # Save final results
            self._save_training_results(results)

            self.structured_logger.end_training_session({
                "final_loss": results['final_loss'],
                "final_accuracy": results['final_accuracy'],
                "best_accuracy": self.best_accuracy,
                "best_val_loss": best_val_loss if val_dataset_path else None,
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
                logger.error(
                    f"Failed to save emergency checkpoint: {checkpoint_error}")

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

    def _save_best_checkpoint(self, epoch: int, val_loss: float, val_accuracy: float) -> str:
        """
        Save best model checkpoint based on validation loss.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss value
            val_accuracy: Validation accuracy value

        Returns:
            Path to saved best checkpoint
        """
        best_checkpoint_path = self.output_dir / "best_checkpoint.pt"

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Training loss
            'loss': self.epoch_losses[-1] if self.epoch_losses else float('inf'),
            # Training accuracy
            'accuracy': self.epoch_accuracies[-1] if self.epoch_accuracies else 0.0,
            # Validation loss (metric used for selection)
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,  # Validation accuracy
            'config': asdict(self.config),
            'training_metrics': self.training_metrics
        }

        torch.save(checkpoint_data, best_checkpoint_path)
        return str(best_checkpoint_path)

    def _save_config(self):
        """Save training configuration."""
        config_path = self.output_dir / "fine_tuning_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

    def _create_training_results(self) -> Dict[str, Any]:
        """Create training results dictionary."""
        training_time = self.training_metrics['training_end_time'] - \
            self.training_metrics['training_start_time']

        results = {
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
            'config': asdict(self.config),
            'checkpoint_paths': self.checkpoint_paths
        }

        # Add validation metrics if available
        if self.validation_losses:
            results['validation_losses'] = self.validation_losses
            results['validation_accuracies'] = self.validation_accuracies
            results['best_validation_loss'] = min(self.validation_losses)
            results['best_validation_accuracy'] = max(
                self.validation_accuracies)
            results['final_validation_loss'] = self.validation_losses[-1]
            results['final_validation_accuracy'] = self.validation_accuracies[-1]

        return results

    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file."""
        results_path = self.output_dir / "fine_tuning_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training results saved to: {results_path}")
