"""
Career-Aware Data Augmentation Module

This module implements the "Career Time Machine" approach for creating
career-progression-aware training data that helps models understand
how professionals describe work at different seniority levels.
"""

from .career_aware_augmenter import CareerAwareAugmenter
from .upward_transformer import UpwardTransformer
from .downward_transformer import DownwardTransformer
from .semantic_validator import SemanticValidator
from .progression_constraints import ProgressionConstraints
from .transformation_config_loader import (
    TransformationConfigLoader,
    get_config_loader,
    reset_config_loader
)

__all__ = [
    'CareerAwareAugmenter',
    'UpwardTransformer', 
    'DownwardTransformer',
    'SemanticValidator',
    'ProgressionConstraints',
    'TransformationConfigLoader',
    'get_config_loader',
    'reset_config_loader'
]
