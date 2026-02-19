"""
Core data structures for contrastive learning training system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class TrainingSample:
    """Represents a single training sample with resume-job pair and label."""
    resume: Dict[str, Any]
    job: Dict[str, Any]
    label: str  # 'positive' or 'negative'
    sample_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the training sample after initialization."""
        if self.label not in ['positive', 'negative']:
            raise ValueError(
                f"Label must be 'positive' or 'negative', got: {self.label}")

        if not isinstance(self.resume, dict):
            raise ValueError("Resume must be a dictionary")

        if not isinstance(self.job, dict):
            raise ValueError("Job must be a dictionary")

        if not self.sample_id:
            raise ValueError("Sample ID cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'resume': self.resume,
            'job': self.job,
            'label': self.label,
            'sample_id': self.sample_id,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSample':
        """Create TrainingSample from dictionary."""
        return cls(
            resume=data['resume'],
            job=data['job'],
            label=data['label'],
            sample_id=data['sample_id'],
            metadata=data.get('metadata', {})
        )


@dataclass
class ContrastiveTriplet:
    """Represents a contrastive learning triplet with anchor, positive, and negatives."""
    anchor: Dict[str, Any]  # Resume
    positive: Dict[str, Any]  # Matching job
    negatives: List[Dict[str, Any]]  # Non-matching jobs
    career_distances: List[float]  # Distance scores for pathway weighting
    view_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the contrastive triplet after initialization."""
        if not isinstance(self.anchor, dict):
            raise ValueError("Anchor must be a dictionary")

        if not isinstance(self.positive, dict):
            raise ValueError("Positive must be a dictionary")

        if not isinstance(self.negatives, list) or not self.negatives:
            raise ValueError("Negatives must be a non-empty list")

        if len(self.career_distances) != len(self.negatives):
            raise ValueError("Career distances must match number of negatives")

        for distance in self.career_distances:
            if not isinstance(distance, (int, float)) or distance < 0:
                raise ValueError(
                    "Career distances must be non-negative numbers")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'anchor': self.anchor,
            'positive': self.positive,
            'negatives': self.negatives,
            'career_distances': self.career_distances,
            'view_metadata': self.view_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContrastiveTriplet':
        """Create ContrastiveTriplet from dictionary."""
        return cls(
            anchor=data['anchor'],
            positive=data['positive'],
            negatives=data['negatives'],
            career_distances=data['career_distances'],
            view_metadata=data.get('view_metadata', {})
        )


@dataclass
class TrainingConfig:
    """Configuration parameters for contrastive learning training."""
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 10
    temperature: float = 0.1
    # Deprecated: mixed sampling is not used; select one strategy via use_pathway_negatives.
    # Kept for backward compatibility with existing configs.
    negative_sampling_ratio: float = 0.7
    pathway_weight: float = 2.0
    use_pathway_negatives: bool = True
    use_view_augmentation: bool = True
    checkpoint_frequency: int = 1000  # batches
    log_frequency: int = 100  # batches
    shuffle_data: bool = True
    # Text encoder configuration
    text_encoder_model: str = 'all-MiniLM-L6-v2'  # SentenceTransformer model name
    text_encoder_device: Optional[str] = None  # Auto-detect if None
    
    # Embedding cache configuration
    embedding_cache_size: int = 10000  # Maximum number of embeddings to cache
    enable_embedding_preload: bool = True  # Whether to preload embeddings before training
    clear_cache_between_epochs: bool = True  # Clear cache between epochs to prevent memory leaks
    
    # View augmentation specific settings
    max_resume_views: int = 5  # Maximum number of resume views to generate
    max_job_views: int = 10  # Maximum number of job views to generate
    # Use original data if augmentation fails
    fallback_on_augmentation_failure: bool = True
    # Distance threshold settings for pathway-aware negatives
    hard_negative_max_distance: float = 2.0  # Maximum distance for hard negatives
    # Maximum distance for medium negatives
    medium_negative_max_distance: float = 4.0
    # Maximum negatives sampled per anchor
    max_negatives_per_anchor: int = 20
    # ESCO graph configuration
    # Path to ESCO graph file (.gexf format) — used for career graph / pathway negatives
    esco_graph_path: Optional[str] = None
    # Path to full ESCO knowledge graph (.gexf) — used for skill-level ontology matching
    # Falls back to esco_graph_path if not set
    esco_kg_path: Optional[str] = None
    
    # Global negative sampling configuration
    global_negative_sampling: bool = False  # Enable global negative sampling
    global_negative_pool_size: int = 1000   # Max jobs to keep in memory
    
    # Research-grade model configuration
    freeze_text_encoder: bool = True        # Freeze SentenceTransformer to avoid catastrophic forgetting
    projection_dim: int = 128              # Smaller projection to reduce overfitting (was 256)
    projection_dropout: float = 0.1        # Dropout for regularization
    weight_decay: float = 0.0              # L2 regularization for optimizer (0.001 recommended)
    
    # NEW: Structured features configuration
    use_structured_features: bool = False   # Enable explicit level encoding and structured features
    structured_feature_dim: int = 32        # Dimension of encoded structured features
    
    # NEW: 2-Phase Training Configuration
    training_phase: str = "supervised"  # "self_supervised" | "supervised" | "fine_tuning"
    
    # NEW: Self-supervised training settings
    use_augmentation_labels_only: bool = False  # Use only augmentation-generated positive pairs
    augmentation_positive_ratio: float = 1.0    # Percentage of augmented positives to use (0.0-1.0)
    
    # NEW: Fine-tuning configuration
    pretrained_model_path: Optional[str] = None  # Path to pre-trained contrastive model
    freeze_contrastive_layers: bool = True       # Freeze contrastive encoder during fine-tuning
    classification_dropout: float = 0.1          # Dropout for classification head
    
    # NEW: Enhanced augmentation configuration
    augmentation_config_path: Optional[str] = None  # Path to augmentation configuration file
    augmentation_quality_profile: str = "balanced"  # Quality profile: fast, balanced, high_quality
    enhanced_augmentation_validation: bool = True   # Enable enhanced validation
    augmentation_diversity_monitoring: bool = True  # Enable diversity monitoring
    augmentation_metadata_sync: bool = True         # Enable metadata synchronization
    
    # NEW: Augmentation quality gates (optional detailed configuration)
    augmentation_quality_gates: Optional[Dict[str, float]] = None
    augmentation_similarity_thresholds: Optional[Dict[str, float]] = None
    augmentation_fallback_config: Optional[Dict[str, Any]] = None
    
    # Validation configuration
    validation_path: Optional[str] = None  # Path to validation dataset (JSONL format)
    validate_every_n_epochs: int = 1       # Run validation every N epochs

    # Ontology-aware loss weighting (uses precomputed ESCO enrichment scores)
    ontology_weight: float = 0.0           # 0.0 = disabled, 0.3 = moderate, 0.5 = strong
    ot_distance_scale: float = 10.0        # Normalization scale for OT distance

    # Phase 2 class imbalance handling
    pos_class_weight: float = 0.0          # 0.0 = disabled, 2.5 = recommended for 28% positive ratio

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")

        if not 0 <= self.negative_sampling_ratio <= 1:
            raise ValueError("Negative sampling ratio must be between 0 and 1")

        if self.pathway_weight < 0:
            raise ValueError("Pathway weight must be non-negative")

        if self.checkpoint_frequency <= 0:
            raise ValueError("Checkpoint frequency must be positive")

        if self.log_frequency <= 0:
            raise ValueError("Log frequency must be positive")

        if self.max_resume_views <= 0:
            raise ValueError("Max resume views must be positive")

        if self.max_job_views <= 0:
            raise ValueError("Max job views must be positive")

        if not self.text_encoder_model or not isinstance(self.text_encoder_model, str):
            raise ValueError("Text encoder model must be a non-empty string")

        if self.hard_negative_max_distance <= 0:
            raise ValueError("Hard negative max distance must be positive")

        if self.medium_negative_max_distance <= self.hard_negative_max_distance:
            raise ValueError(
                "Medium negative max distance must be greater than hard negative max distance")

        if self.max_negatives_per_anchor <= 0:
            raise ValueError("Max negatives per anchor must be positive")
        
        # NEW: Validate 2-phase training configuration
        self._validate_training_phase_config()

    def _validate_training_phase_config(self):
        """Validate 2-phase training specific configuration parameters."""
        # Validate training_phase
        valid_phases = ["self_supervised", "supervised", "fine_tuning"]
        if self.training_phase not in valid_phases:
            raise ValueError(
                f"training_phase must be one of {valid_phases}, got: {self.training_phase}")
        
        # Validate augmentation_positive_ratio
        if not 0.0 <= self.augmentation_positive_ratio <= 1.0:
            raise ValueError(
                f"augmentation_positive_ratio must be between 0.0 and 1.0, got: {self.augmentation_positive_ratio}")
        
        # Validate classification_dropout
        if not 0.0 <= self.classification_dropout <= 1.0:
            raise ValueError(
                f"classification_dropout must be between 0.0 and 1.0, got: {self.classification_dropout}")
        
        # Phase-specific validation
        if self.training_phase == "fine_tuning":
            if self.pretrained_model_path is None:
                raise ValueError(
                    "pretrained_model_path is required when training_phase is 'fine_tuning'")
            
            if not isinstance(self.pretrained_model_path, str) or not self.pretrained_model_path.strip():
                raise ValueError(
                    "pretrained_model_path must be a non-empty string when training_phase is 'fine_tuning'")
        
        if self.training_phase == "self_supervised":
            if self.use_augmentation_labels_only and self.augmentation_positive_ratio == 0.0:
                raise ValueError(
                    "augmentation_positive_ratio cannot be 0.0 when use_augmentation_labels_only is True")
        
        # Backward compatibility check - ensure existing configurations work
        if self.training_phase == "supervised":
            # In supervised mode, these parameters should not conflict with existing behavior
            if self.use_augmentation_labels_only:
                # This could potentially break existing workflows, so warn but allow
                pass
            
            if self.pretrained_model_path is not None:
                # This is fine - user might want to load a pre-trained model for regular training
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'temperature': self.temperature,
            'negative_sampling_ratio': self.negative_sampling_ratio,
            'pathway_weight': self.pathway_weight,
            'use_pathway_negatives': self.use_pathway_negatives,
            'use_view_augmentation': self.use_view_augmentation,
            'checkpoint_frequency': self.checkpoint_frequency,
            'log_frequency': self.log_frequency,
            'shuffle_data': self.shuffle_data,
            'text_encoder_model': self.text_encoder_model,
            'text_encoder_device': self.text_encoder_device,
            'max_resume_views': self.max_resume_views,
            'max_job_views': self.max_job_views,
            'fallback_on_augmentation_failure': self.fallback_on_augmentation_failure,
            'hard_negative_max_distance': self.hard_negative_max_distance,
            'medium_negative_max_distance': self.medium_negative_max_distance,
            'esco_graph_path': self.esco_graph_path,
            'esco_kg_path': self.esco_kg_path,
            'global_negative_sampling': self.global_negative_sampling,
            'global_negative_pool_size': self.global_negative_pool_size,
            'freeze_text_encoder': self.freeze_text_encoder,
            'projection_dim': self.projection_dim,
            'projection_dropout': self.projection_dropout,
            'weight_decay': self.weight_decay,
            # Structured features configuration
            'use_structured_features': self.use_structured_features,
            'structured_feature_dim': self.structured_feature_dim,
            # NEW: 2-phase training fields
            'training_phase': self.training_phase,
            'use_augmentation_labels_only': self.use_augmentation_labels_only,
            'augmentation_positive_ratio': self.augmentation_positive_ratio,
            'pretrained_model_path': self.pretrained_model_path,
            'freeze_contrastive_layers': self.freeze_contrastive_layers,
            'classification_dropout': self.classification_dropout,
            # Enhanced augmentation configuration
            'augmentation_config_path': self.augmentation_config_path,
            'augmentation_quality_profile': self.augmentation_quality_profile,
            'enhanced_augmentation_validation': self.enhanced_augmentation_validation,
            'augmentation_diversity_monitoring': self.augmentation_diversity_monitoring,
            'augmentation_metadata_sync': self.augmentation_metadata_sync,
            'augmentation_quality_gates': self.augmentation_quality_gates,
            'augmentation_similarity_thresholds': self.augmentation_similarity_thresholds,
            'augmentation_fallback_config': self.augmentation_fallback_config,
            'validation_path': self.validation_path,
            'validate_every_n_epochs': self.validate_every_n_epochs,
            'ontology_weight': self.ontology_weight,
            'ot_distance_scale': self.ot_distance_scale,
            'pos_class_weight': self.pos_class_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary, ignoring unknown keys."""
        import dataclasses as _dc
        valid_keys = {f.name for f in _dc.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install PyYAML")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {file_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, file_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {file_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install PyYAML")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingResults:
    """Results and metrics from contrastive learning training."""
    final_loss: float
    epoch_losses: List[float]
    training_time: float
    total_batches: int
    total_samples: int
    checkpoint_paths: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_losses: List[float] = field(default_factory=list)  # Validation loss per epoch

    def __post_init__(self):
        """Validate training results."""
        if self.final_loss < 0:
            raise ValueError("Final loss cannot be negative")

        if any(loss < 0 for loss in self.epoch_losses):
            raise ValueError("Epoch losses cannot be negative")

        if self.training_time < 0:
            raise ValueError("Training time cannot be negative")

        if self.total_batches < 0:
            raise ValueError("Total batches cannot be negative")

        if self.total_samples < 0:
            raise ValueError("Total samples cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'final_loss': self.final_loss,
            'epoch_losses': self.epoch_losses,
            'validation_losses': self.validation_losses,
            'training_time': self.training_time,
            'total_batches': self.total_batches,
            'total_samples': self.total_samples,
            'checkpoint_paths': self.checkpoint_paths,
            'metrics': self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingResults':
        """Create TrainingResults from dictionary."""
        return cls(
            final_loss=data['final_loss'],
            epoch_losses=data['epoch_losses'],
            training_time=data['training_time'],
            total_batches=data['total_batches'],
            total_samples=data['total_samples'],
            checkpoint_paths=data['checkpoint_paths'],
            metrics=data.get('metrics', {}),
            validation_losses=data.get('validation_losses', [])
        )

    def save_json(self, file_path: str) -> None:
        """Save results to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, file_path: str) -> 'TrainingResults':
        """Load results from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)
