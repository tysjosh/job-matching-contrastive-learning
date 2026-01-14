"""
Contrastive Learning Training System

A pathway-aware contrastive learning system for resume-job matching.
"""

from .data_structures import (
    TrainingSample,
    ContrastiveTriplet,
    TrainingConfig,
    TrainingResults
)
from .data_loader import DataLoader, DataLoaderStats
from .data_adapter import DataAdapter, DataAdapterConfig
from .batch_processor import BatchProcessor
from .career_graph import CareerGraph
from .loss_engine import ContrastiveLossEngine
from .trainer import ContrastiveLearningTrainer

# Two-phase training components
from .two_phase_trainer import TwoPhaseTrainer, TwoPhaseResults
from .fine_tuning_trainer import FineTuningTrainer
from .contrastive_classification_model import ContrastiveClassificationModel
from .two_phase_metrics import TwoPhaseMetricsTracker
from .embedding_cache import EmbeddingCache, BatchEfficientEncoder
from .training_mode_detector import TrainingModeDetector, TrainingMode

# Training strategy pattern
from .training_strategy import (
    TrainingStrategy,
    TrainingStrategyResult,
    SinglePhaseStrategy,
    TwoPhaseStrategy
)

__all__ = [
    # Core data structures
    'TrainingSample',
    'ContrastiveTriplet',
    'TrainingConfig',
    'TrainingResults',
    
    # Data processing
    'DataLoader',
    'DataLoaderStats',
    'DataAdapter',
    'DataAdapterConfig',
    'BatchProcessor',
    
    # Training components
    'CareerGraph',
    'ContrastiveLossEngine',
    'ContrastiveLearningTrainer',
    
    # Two-phase training
    'TwoPhaseTrainer',
    'TwoPhaseResults',
    'FineTuningTrainer',
    'ContrastiveClassificationModel',
    'TwoPhaseMetricsTracker',
    'TrainingModeDetector',
    'TrainingMode',
    
    # Training strategy pattern
    'TrainingStrategy',
    'TrainingStrategyResult',
    'SinglePhaseStrategy',
    'TwoPhaseStrategy',
    
    # Caching and efficiency
    'EmbeddingCache',
    'BatchEfficientEncoder'
]
