"""
Configuration system for Sentence Transformers matching infrastructure.
Supports model selection, caching, and performance parameters.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SentenceTransformerConfig:
    """Configuration class for Sentence Transformer matching system."""

    # Model configuration
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "auto"  # "auto", "cpu", "cuda"
    normalize_embeddings: bool = True

    # Caching configuration
    enable_cache: bool = True
    cache_dir: str = ".sentence_transformer_cache"
    cache_max_size: int = 10000  # Maximum number of cached embeddings
    persist_cache: bool = True

    # Performance configuration
    batch_size: int = 32
    max_seq_length: int = 512
    show_progress_bar: bool = True

    # Matching configuration
    text_similarity_weight: float = 0.7
    skills_overlap_weight: float = 0.3
    similarity_threshold: float = 0.5

    # Dual mode configuration (for comparison with TF-IDF)
    enable_dual_mode: bool = False
    tfidf_comparison: bool = False

    # Error handling
    fallback_to_cpu: bool = True
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.text_similarity_weight + self.skills_overlap_weight != 1.0:
            raise ValueError(
                "Text similarity weight and skills overlap weight must sum to 1.0")

        if self.similarity_threshold < 0.0 or self.similarity_threshold > 1.0:
            raise ValueError(
                "Similarity threshold must be between 0.0 and 1.0")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.max_seq_length <= 0:
            raise ValueError("Max sequence length must be positive")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SentenceTransformerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'normalize_embeddings': self.normalize_embeddings,
            'enable_cache': self.enable_cache,
            'cache_dir': self.cache_dir,
            'cache_max_size': self.cache_max_size,
            'persist_cache': self.persist_cache,
            'batch_size': self.batch_size,
            'max_seq_length': self.max_seq_length,
            'show_progress_bar': self.show_progress_bar,
            'text_similarity_weight': self.text_similarity_weight,
            'skills_overlap_weight': self.skills_overlap_weight,
            'similarity_threshold': self.similarity_threshold,
            'enable_dual_mode': self.enable_dual_mode,
            'tfidf_comparison': self.tfidf_comparison,
            'fallback_to_cpu': self.fallback_to_cpu,
            'max_retries': self.max_retries
        }

    def get_device(self) -> str:
        """Get the appropriate device for model loading."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def validate_model_name(self) -> bool:
        """Validate that the specified model is available."""
        try:
            from sentence_transformers import SentenceTransformer
            # Try to load model info without downloading
            model = SentenceTransformer(self.model_name)
            return True
        except Exception:
            return False


# Default configurations for different use cases
DEFAULT_CONFIG = SentenceTransformerConfig()

FAST_CONFIG = SentenceTransformerConfig(
    model_name="all-MiniLM-L6-v2",
    batch_size=64,
    max_seq_length=256,
    enable_cache=True,
    cache_max_size=5000
)

ACCURATE_CONFIG = SentenceTransformerConfig(
    model_name="all-mpnet-base-v2",
    batch_size=16,
    max_seq_length=512,
    enable_cache=True,
    cache_max_size=15000
)

COMPARISON_CONFIG = SentenceTransformerConfig(
    model_name="all-MiniLM-L6-v2",
    enable_dual_mode=True,
    tfidf_comparison=True,
    batch_size=32
)


def load_config_from_file(config_path: str) -> SentenceTransformerConfig:
    """Load configuration from JSON file."""
    import json

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return SentenceTransformerConfig.from_dict(config_dict)


def save_config_to_file(config: SentenceTransformerConfig, config_path: str) -> None:
    """Save configuration to JSON file."""
    import json

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
