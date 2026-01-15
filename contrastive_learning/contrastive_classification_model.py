"""
ContrastiveClassificationModel for fine-tuning pre-trained contrastive encoders.

This module implements a classification model that loads a pre-trained contrastive encoder
and adds a classification head for downstream tasks like interview prediction.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .data_structures import TrainingConfig

logger = logging.getLogger(__name__)


class ContrastiveClassificationModel(nn.Module):
    """
    Classification model that uses a pre-trained contrastive encoder with a classification head.

    This model loads a pre-trained contrastive encoder from a checkpoint and adds a 
    classification head for supervised fine-tuning on downstream tasks.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the ContrastiveClassificationModel.

        Args:
            config: Training configuration containing model parameters

        Raises:
            ImportError: If PyTorch is not available
            ValueError: If required configuration parameters are missing
            FileNotFoundError: If pretrained model path doesn't exist
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ContrastiveClassificationModel. "
                "Install with: pip install torch"
            )

        super().__init__()

        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Validate configuration
        self._validate_config()

        # Load pre-trained contrastive encoder
        self.contrastive_encoder = self._load_pretrained_encoder()

        # Get embedding dimension from the contrastive encoder
        self.embedding_dim = self._get_encoder_embedding_dim()

        # Create classification head
        self.classification_head = self._create_classification_head()

        # Apply parameter freezing if configured
        if config.freeze_contrastive_layers:
            self._freeze_contrastive_layers()

        # Move model to device
        self.to(self.device)

        logger.info(
            f"ContrastiveClassificationModel initialized with embedding_dim={self.embedding_dim}")
        logger.info(
            f"Contrastive layers frozen: {config.freeze_contrastive_layers}")
        logger.info(f"Classification dropout: {config.classification_dropout}")

    def _validate_config(self):
        """Validate the configuration parameters."""
        if self.config.training_phase != "fine_tuning":
            logger.warning(
                f"ContrastiveClassificationModel typically used with training_phase='fine_tuning', "
                f"but got '{self.config.training_phase}'"
            )

        if not self.config.pretrained_model_path:
            raise ValueError(
                "pretrained_model_path is required for ContrastiveClassificationModel")

        pretrained_path = Path(self.config.pretrained_model_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found: {self.config.pretrained_model_path}")

        if not 0.0 <= self.config.classification_dropout <= 1.0:
            raise ValueError(
                f"classification_dropout must be between 0.0 and 1.0, got: {self.config.classification_dropout}")

    def _load_pretrained_encoder(self) -> nn.Module:
        """
        Load the pre-trained contrastive encoder from checkpoint.

        Returns:
            Pre-trained contrastive encoder model

        Raises:
            RuntimeError: If checkpoint loading fails
        """
        try:
            checkpoint_path = Path(self.config.pretrained_model_path)
            logger.info(
                f"Loading pre-trained contrastive encoder from: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model_state_dict = checkpoint['model']
            else:
                # Assume the checkpoint is the model state dict itself
                model_state_dict = checkpoint

            # Create a new contrastive model with the same architecture
            # We need to infer the architecture from the state dict
            encoder = self._create_contrastive_encoder_from_state_dict(
                model_state_dict)

            # Load the state dict
            encoder.load_state_dict(model_state_dict, strict=True)

            logger.info("Pre-trained contrastive encoder loaded successfully")
            return encoder

        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained encoder: {e}")

    def _create_contrastive_encoder_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Create a contrastive encoder model based on the state dict structure.

        Args:
            state_dict: Model state dictionary

        Returns:
            Contrastive encoder model with matching architecture
        """
        # Analyze state dict to determine architecture
        projection_layers = [
            key for key in state_dict.keys() if 'projection_head' in key]

        if not projection_layers:
            raise ValueError(
                "Could not find projection_head layers in pretrained model")

        # Find input and output dimensions from the projection head
        input_dim = None
        output_dim = None

        for key in projection_layers:
            if 'weight' in key and key.endswith('0.weight'):  # First layer
                input_dim = state_dict[key].shape[1]
            elif 'weight' in key and not any(f'{i}.' in key for i in range(10) if f'{i}.' != key.split('.')[-2] + '.'):
                # Last layer (heuristic to find the final layer)
                output_dim = state_dict[key].shape[0]

        # If we can't determine dimensions, use defaults from config
        if input_dim is None:
            input_dim = 384  # Default SentenceTransformer dimension
            logger.warning(
                f"Could not determine input dimension, using default: {input_dim}")

        if output_dim is None:
            output_dim = getattr(self.config, 'projection_dim', 128)
            logger.warning(
                f"Could not determine output dimension, using config default: {output_dim}")

        # Create the contrastive encoder (reuse the existing architecture)
        class CareerAwareContrastiveModel(nn.Module):
            def __init__(self, input_dim: int, projection_dim: int, dropout: float = 0.1):
                super().__init__()
                self.projection_head = nn.Sequential(
                    nn.Linear(input_dim, projection_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(projection_dim * 2, projection_dim)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                projected = self.projection_head(x)
                return projected  # Remove double normalization - SentenceTransformer already normalizes

            def get_embedding_dim(self) -> int:
                for layer in reversed(self.projection_head):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
                return output_dim

        dropout = getattr(self.config, 'projection_dropout', 0.1)
        encoder = CareerAwareContrastiveModel(input_dim, output_dim, dropout)

        logger.info(
            f"Created contrastive encoder with input_dim={input_dim}, output_dim={output_dim}")
        return encoder

    def _get_encoder_embedding_dim(self) -> int:
        """
        Get the embedding dimension from the contrastive encoder.

        Returns:
            Embedding dimension of the contrastive encoder
        """
        if hasattr(self.contrastive_encoder, 'get_embedding_dim'):
            return self.contrastive_encoder.get_embedding_dim()

        # Fallback: try to infer from the last linear layer
        for module in reversed(list(self.contrastive_encoder.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features

        # Default fallback
        return getattr(self.config, 'projection_dim', 128)

    def _create_classification_head(self) -> nn.Module:
        """
        Create the classification head that takes concatenated embeddings.

        Returns:
            Classification head module
        """
        # Input dimension is 2 * embedding_dim (concatenated resume and job embeddings)
        input_dim = 2 * self.embedding_dim

        # Create a simple but effective classification head
        classification_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.classification_dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.config.classification_dropout),
            nn.Linear(input_dim // 4, 1),  # Binary classification
            nn.Sigmoid()  # Output probability for binary classification
        )

        # Initialize weights
        self._initialize_classification_weights(classification_head)

        logger.info(f"Created classification head with input_dim={input_dim}")
        return classification_head

    def _initialize_classification_weights(self, module: nn.Module):
        """Initialize weights for the classification head."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_contrastive_layers(self):
        """Freeze all parameters in the contrastive encoder."""
        for param in self.contrastive_encoder.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel()
                            for p in self.contrastive_encoder.parameters())
        trainable_params = sum(
            p.numel() for p in self.classification_head.parameters() if p.requires_grad)

        logger.info(
            f"Frozen contrastive encoder parameters: {frozen_params:,}")
        logger.info(
            f"Trainable classification head parameters: {trainable_params:,}")

    def _generate_embeddings(self, resume_text_emb: torch.Tensor, job_text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate embeddings using the frozen pre-trained contrastive encoder.

        Args:
            resume_text_emb: Text embeddings for resume (from SentenceTransformer)
            job_text_emb: Text embeddings for job (from SentenceTransformer)

        Returns:
            Tuple of (resume_embedding, job_embedding) from contrastive encoder
        """
        # Ensure inputs are on the correct device
        resume_text_emb = resume_text_emb.to(self.device)
        job_text_emb = job_text_emb.to(self.device)

        # Generate embeddings using the contrastive encoder
        with torch.set_grad_enabled(not self.config.freeze_contrastive_layers):
            resume_emb = self.contrastive_encoder(resume_text_emb)
            job_emb = self.contrastive_encoder(job_text_emb)

        return resume_emb, job_emb

    def forward(self, resume_text_emb: torch.Tensor, job_text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            resume_text_emb: Text embeddings for resume (from SentenceTransformer)
            job_text_emb: Text embeddings for job (from SentenceTransformer)

        Returns:
            Classification probability (0-1) for interview prediction
        """
        # Generate embeddings using the pre-trained contrastive encoder
        resume_emb, job_emb = self._generate_embeddings(
            resume_text_emb, job_text_emb)

        # Concatenate embeddings
        concatenated_emb = torch.cat([resume_emb, job_emb], dim=-1)

        # Pass through classification head
        classification_output = self.classification_head(concatenated_emb)

        # Remove last dimension for binary classification
        return classification_output.squeeze(-1)

    def get_contrastive_embeddings(self, resume_text_emb: torch.Tensor, job_text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get contrastive embeddings without classification (useful for analysis).

        Args:
            resume_text_emb: Text embeddings for resume (from SentenceTransformer)
            job_text_emb: Text embeddings for job (from SentenceTransformer)

        Returns:
            Tuple of (resume_embedding, job_embedding) from contrastive encoder
        """
        return self._generate_embeddings(resume_text_emb, job_text_emb)

    def get_trainable_parameters(self):
        """Get only the trainable parameters (useful for optimizer setup)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture and parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            'embedding_dim': self.embedding_dim,
            'classification_dropout': self.config.classification_dropout,
            'freeze_contrastive_layers': self.config.freeze_contrastive_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'device': str(self.device),
            'pretrained_model_path': self.config.pretrained_model_path
        }
